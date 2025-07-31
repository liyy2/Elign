import torch
import torch.nn.functional as F
import numpy as np
import logging
from ase import Atoms
from ase.constraints import FixAtoms
from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
from equivariant_diffusion import utils as diffusion_utils
from equivariant_diffusion.en_diffusion import EnVariationalDiffusion

# Set up logger for MLFF guidance debugging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] MLFF: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Wandb import with error handling for logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not available for MLFF logging")


class MLFFGuidedDiffusion(EnVariationalDiffusion):
    """
    Enhanced diffusion model with MLFF force field guidance.
    
    This class extends the EnVariationalDiffusion to incorporate guidance from
    a pretrained Machine Learning Force Field (MLFF) during the sampling process.
    """
    
    def __init__(self, *args, mlff_predictor=None, guidance_scale=1.0, 
                 dataset_info=None, guidance_iterations=1, noise_threshold=0.8, force_clip_threshold=None, **kwargs):
        """
        Initialize MLFF-guided diffusion model.
        
        Args:
            *args: Arguments for parent EnVariationalDiffusion class
            mlff_predictor: Pretrained MLFF predictor from fairchem
            guidance_scale: Scale factor for MLFF force guidance
            dataset_info: Dataset information with atom decoders
            guidance_iterations: Number of iterative force field evaluations per diffusion step
            noise_threshold: Skip guidance when noise level exceeds this threshold (0.8 = skip first ~20% of steps)
            force_clip_threshold: Maximum force magnitude allowed (None = no clipping)
            **kwargs: Keyword arguments for parent class
        """
        super().__init__(*args, **kwargs)
        self.mlff_predictor = mlff_predictor
        self.guidance_scale = guidance_scale
        self.dataset_info = dataset_info
        self.guidance_iterations = guidance_iterations
        self.noise_threshold = noise_threshold
        self.force_clip_threshold = force_clip_threshold
        
        # Default molecule cell size for molecular systems
        self.molecule_cell_size = 20.0
        
        # Default task name for MLFF predictor
        self.task_name = "omol"  # For molecular systems
        
    def diffusion_to_atomic_data(self, z, node_mask, dataset_info):
        """
        Convert diffusion model data to AtomicData format required by MLFF predictor.
        
        Args:
            z: Diffusion state tensor [batch_size, n_nodes, n_dims + n_features]
            node_mask: Node mask tensor [batch_size, n_nodes, 1]
            dataset_info: Dataset information with atom decoders
            
        Returns:
            List of AtomicData objects for MLFF predictor
        """
        batch_size = z.shape[0]
        atomic_data_list = []
        
        for batch_idx in range(batch_size):
            # Extract positions and node features
            positions = z[batch_idx, :, :self.n_dims]  # [n_nodes, 3]
            node_features = z[batch_idx, :, self.n_dims:]  # [n_nodes, n_features]
            mask = node_mask[batch_idx, :, 0]  # [n_nodes]
            
            # Filter out masked nodes
            valid_nodes = mask.bool()
            positions = positions[valid_nodes]  # [n_valid_nodes, 3]
            node_features = node_features[valid_nodes]  # [n_valid_nodes, n_features]
            
            if positions.shape[0] == 0:
                continue
                
            # Extract atom types from node features
            if self.include_charges:
                categorical_features = node_features[:, :-1]  # Remove charge dimension
            else:
                categorical_features = node_features
            
            # Convert categorical features to atom types
            atom_types = torch.argmax(categorical_features, dim=1)  # [n_valid_nodes]
            
            # Map atom types to atomic numbers using dataset info
            # Convert to CPU for indexing operations to avoid device mismatch
            atom_types_cpu = atom_types.cpu()
            if 'atomic_nb' in dataset_info:
                # For GEOM dataset
                atomic_numbers = torch.tensor([dataset_info['atomic_nb'][idx] for idx in atom_types_cpu])
            else:
                # For QM9 dataset - map through decoder
                atom_decoder = dataset_info['atom_decoder']
                atomic_number_map = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 
                                   'B': 5, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 
                                   'Cl': 17, 'As': 33, 'Br': 35, 'I': 53, 'Hg': 80, 'Bi': 83}
                atomic_numbers = torch.tensor([atomic_number_map[atom_decoder[idx]] for idx in atom_types_cpu])
            
            # Create ASE atoms object
            atoms = Atoms(
                numbers=atomic_numbers.detach().cpu().numpy(),
                positions=positions.detach().cpu().numpy(),
                cell=np.eye(3) * self.molecule_cell_size,
                pbc=False
            )
            
            # Set charge and spin for molecular systems
            atoms.info['charge'] = 0  # Default charge
            atoms.info['spin'] = 1    # Default spin multiplicity
            
            # Convert to AtomicData
            try:
                atomic_data = AtomicData.from_ase(
                    atoms,
                    r_edges=True,
                    radius=6.0,
                    max_neigh=50,
                    task_name=self.task_name,
                    r_data_keys=["charge", "spin"]
                )
                atomic_data_list.append(atomic_data)
            except Exception as e:
                print(f"Warning: Could not convert batch {batch_idx} to AtomicData: {e}")
                continue
                
        return atomic_data_list
    
    def check_system_feasibility(self, z, node_mask, max_distance=6.0):
        """
        Check if the system is feasible for MLFF prediction.
        A system is considered feasible if it has multi-atom molecules where 
        at least some atoms are within reasonable distance of each other.
        
        Args:
            z: Diffusion state tensor [batch_size, n_nodes, n_dims + n_features]
            node_mask: Node mask tensor [batch_size, n_nodes, 1]
            max_distance: Maximum distance threshold in Angstroms
            
        Returns:
            Boolean indicating if system is feasible for MLFF
        """
        positions = z[:, :, :self.n_dims]  # [batch_size, n_nodes, 3]
        
        for batch_idx in range(positions.shape[0]):
            # Get valid positions for this batch
            mask = node_mask[batch_idx, :, 0].bool()
            valid_positions = positions[batch_idx, mask]  # [n_valid_nodes, 3]
            
            # Skip single atoms or empty molecules
            if valid_positions.shape[0] <= 1:
                continue
            
            # Calculate pairwise distances
            distances = torch.cdist(valid_positions, valid_positions)  # [n_valid, n_valid]
            
            # Remove diagonal (self-distances) by setting to infinity
            distances = distances + torch.eye(distances.shape[0], device=distances.device) * float('inf')
            
            # Find minimum distance to any other atom for each atom
            min_distances = torch.min(distances, dim=1)[0]
            
            # Check if any atom has a neighbor within max_distance
            # If so, this batch has a feasible molecular structure
            if torch.any(min_distances <= max_distance):
                return True
        
        # No batch had atoms within max_distance of each other
        return False
    
    def check_numerical_stability(self, tensor, name="tensor"):
        """
        Check for NaN or infinity values in tensor and log warnings.
        
        Args:
            tensor: Tensor to check
            name: Name for logging purposes
            
        Returns:
            Boolean indicating if tensor is numerically stable
        """
        if torch.isnan(tensor).any():
            logger.warning(f"NaN detected in {name}: {torch.isnan(tensor).sum().item()} values")
            return False
        if torch.isinf(tensor).any():
            logger.warning(f"Infinity detected in {name}: {torch.isinf(tensor).sum().item()} values")
            return False
        return True
    
    def get_mlff_forces(self, z, node_mask, dataset_info):
        """
        Get forces from MLFF predictor for current diffusion state.
        
        Args:
            z: Diffusion state tensor [batch_size, n_nodes, n_dims + n_features]
            node_mask: Node mask tensor [batch_size, n_nodes, 1]
            dataset_info: Dataset information
            
        Returns:
            Forces tensor [batch_size, n_nodes, 3] (zero for masked nodes)
        """
        if self.mlff_predictor is None:
            return torch.zeros_like(z[:, :, :self.n_dims])
        
        # Check input stability
        if not self.check_numerical_stability(z, "input z"):
            logger.warning("Unstable input detected, returning zero forces")
            return torch.zeros_like(z[:, :, :self.n_dims])
        
        # # Check if system is feasible for MLFF prediction
        # if not self.check_system_feasibility(z, node_mask, max_distance=6.0):
        #     return torch.zeros_like(z[:, :, :self.n_dims])
            
        # Convert to AtomicData format
        atomic_data_list = self.diffusion_to_atomic_data(z, node_mask, dataset_info)
        
        if not atomic_data_list:
            return torch.zeros_like(z[:, :, :self.n_dims])
        
        # Batch the atomic data using recommended UMA approach
        try:
            batch = atomicdata_list_to_batch(atomic_data_list)
        except Exception as e:
            print(f"Warning: Could not batch atomic data: {e}")
            return torch.zeros_like(z[:, :, :self.n_dims])
        
        # Get predictions from MLFF
        try:
            with torch.no_grad():
                predictions = self.mlff_predictor.predict(batch)
                
            if 'forces' not in predictions:
                logger.warning("No forces in MLFF predictions")
                return torch.zeros_like(z[:, :, :self.n_dims])
                
            forces = predictions['forces']  # [total_atoms, 3]
            
            # Check forces numerical stability
            if not self.check_numerical_stability(forces, "MLFF forces"):
                logger.warning("Unstable MLFF forces detected, returning zero forces")
                return torch.zeros_like(z[:, :, :self.n_dims])
            
            # Log force statistics
            force_magnitudes = torch.norm(forces, dim=-1)
            force_stats = {
                'mean_magnitude': force_magnitudes.mean().item(),
                'max_magnitude': force_magnitudes.max().item(),
                'min_magnitude': force_magnitudes.min().item(),
                'std_magnitude': force_magnitudes.std().item()
            }
            
            logger.info(f"MLFF forces - mean magnitude: {force_stats['mean_magnitude']:.6f}, "
                       f"max magnitude: {force_stats['max_magnitude']:.6f}, "
                       f"min magnitude: {force_stats['min_magnitude']:.6f}")
            
            # Log to wandb if available
            if WANDB_AVAILABLE and wandb.run is not None:
                wandb.log({
                    'mlff/forces_mean_magnitude': force_stats['mean_magnitude'],
                    'mlff/forces_max_magnitude': force_stats['max_magnitude'],
                    'mlff/forces_min_magnitude': force_stats['min_magnitude'],
                    'mlff/forces_std_magnitude': force_stats['std_magnitude']
                })
            
        except Exception as e:
            print(f"Warning: MLFF prediction failed: {e}")
            return torch.zeros_like(z[:, :, :self.n_dims])
        
        # Reshape forces back to original batch format using UMA batch approach
        forces_batched = torch.zeros_like(z[:, :, :self.n_dims])
        
        # Extract forces for each system in the batch using batch.batch attribute
        for batch_idx, atomic_data in enumerate(atomic_data_list):
            # Get forces for this system using batch.batch indexing
            system_forces = forces[batch.batch == batch_idx]  # [n_atoms_in_system, 3]
            
            # Get valid node indices for this batch
            mask = node_mask[batch_idx, :, 0]
            valid_indices = torch.where(mask)[0]
            
            if len(valid_indices) == len(system_forces):
                # Ensure system_forces is on the same device as forces_batched
                system_forces = system_forces.to(forces_batched.device)
                forces_batched[batch_idx, valid_indices] = system_forces
            
        return forces_batched
    
    def apply_mlff_guidance(self, z, node_mask, dataset_info, sigma):
        """
        Apply MLFF force guidance to the diffusion process with iterative refinement.
        
        Args:
            z: Current diffusion state [batch_size, n_nodes, n_dims + n_features]
            node_mask: Node mask [batch_size, n_nodes, 1]
            dataset_info: Dataset information
            sigma: Current noise level
            
        Returns:
            Modified z with MLFF guidance applied to positions
        """
        if self.mlff_predictor is None or self.guidance_scale == 0:
            return z
        
        # Note: Noise threshold check is already done in DPM-Solver++ loop
        # This function is only called when guidance should be applied
        
        # Log guidance application
        noise_level = sigma.mean().item() if sigma.numel() > 0 else 1.0
        logger.info(f"Applying MLFF guidance: noise_level={noise_level:.6f}, "
                   f"guidance_scale={self.guidance_scale:.6f}")
        
        # Log to wandb if available
        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.log({
                'mlff/noise_level': noise_level,
                'mlff/guidance_scale': self.guidance_scale,
                'mlff/guidance_iterations': self.guidance_iterations,
                'mlff/guidance_applied': True
            })
        
        # Additional feasibility check for moderately noisy systems
        max_distance = 6.0 if noise_level < 0.5 else 10.0  # More lenient for high noise
        
        # if not self.check_system_feasibility(z, node_mask, max_distance=max_distance):
        #     return z
        
        z_guided = z.clone()
        
        # Iterative force field guidance
        for iteration in range(self.guidance_iterations):
            # Get MLFF forces for current state
            forces = self.get_mlff_forces(z_guided, node_mask, dataset_info)
            
            # Check if forces are meaningful (not all zeros)
            if torch.all(forces == 0):
                logger.info(f"All forces are zero at iteration {iteration + 1}, breaking")
                break
            
            # Check forces numerical stability
            if not self.check_numerical_stability(forces, f"forces iteration {iteration + 1}"):
                logger.warning(f"Unstable forces at iteration {iteration + 1}, breaking")
                break
            
            # Log force statistics for this iteration
            force_magnitudes = torch.norm(forces, dim=-1)
            iteration_force_stats = {
                'mean': force_magnitudes.mean().item(),
                'max': force_magnitudes.max().item(),
                'std': force_magnitudes.std().item()
            }
            
            logger.info(f"Iteration {iteration + 1}: force magnitudes - mean: {iteration_force_stats['mean']:.6f}, "
                       f"max: {iteration_force_stats['max']:.6f}")
            
            # Log iteration-specific metrics to wandb
            if WANDB_AVAILABLE and wandb.run is not None:
                wandb.log({
                    f'mlff/iteration_{iteration + 1}_force_mean': iteration_force_stats['mean'],
                    f'mlff/iteration_{iteration + 1}_force_max': iteration_force_stats['max'],
                    f'mlff/iteration_{iteration + 1}_force_std': iteration_force_stats['std']
                })
            
            # Apply force clipping if threshold is specified
            if self.force_clip_threshold is not None:
                # Compute force magnitudes for each atom
                force_magnitudes = torch.norm(forces, dim=-1, keepdim=True)  # [batch_size, n_nodes, 1]
                
                # Create mask for forces that exceed the threshold
                force_clip_mask = force_magnitudes > self.force_clip_threshold
                
                # Clip forces while preserving direction
                forces = torch.where(
                    force_clip_mask,
                    forces * (self.force_clip_threshold / force_magnitudes),
                    forces
                )
            
            # Scale forces by guidance scale and noise level
            # Divide by guidance_iterations to avoid too large updates per iteration
            # Ensure sigma has the correct shape for broadcasting with forces
            if sigma.dim() == 3:  # [batch_size, n_nodes, 1]
                sigma_for_forces = sigma.squeeze(-1)  # [batch_size, n_nodes]
            else:  # [batch_size, n_nodes]
                sigma_for_forces = sigma
            
            # Match the shape of forces tensor
            batch_size, n_nodes_forces, _ = forces.shape
            if sigma_for_forces.shape[1] != n_nodes_forces:
                # Take only the valid nodes that match forces
                sigma_for_forces = sigma_for_forces[:, :n_nodes_forces]
            
            # Scale forces by guidance scale, noise level, and number of iterations
            # Check for potential division issues
            if self.guidance_iterations == 0:
                logger.warning("guidance_iterations is 0, skipping guidance")
                break
                
            base_scale = self.guidance_scale / self.guidance_iterations
            force_scale = base_scale * sigma_for_forces.unsqueeze(-1)  # [batch_size, n_nodes_forces, 1]
            
            # Check force_scale stability
            if not self.check_numerical_stability(force_scale, "force_scale"):
                logger.warning(f"Unstable force_scale at iteration {iteration + 1}, breaking")
                break
            
            position_guidance = forces * force_scale
            
            # Log guidance statistics
            guidance_magnitudes = torch.norm(position_guidance, dim=-1)
            guidance_stats = {
                'base_scale': base_scale,
                'effective_noise_level': sigma_for_forces.mean().item(),
                'guidance_mean': guidance_magnitudes.mean().item(),
                'guidance_max': guidance_magnitudes.max().item(),
                'guidance_std': guidance_magnitudes.std().item(),
                'force_to_guidance_ratio': guidance_magnitudes.mean().item() / iteration_force_stats['mean'] if iteration_force_stats['mean'] > 0 else 0
            }
            
            logger.info(f"Iteration {iteration + 1}: guidance scale={guidance_stats['base_scale']:.6f}, "
                       f"noise_level={guidance_stats['effective_noise_level']:.6f}, "
                       f"guidance magnitudes - mean: {guidance_stats['guidance_mean']:.6f}, "
                       f"max: {guidance_stats['guidance_max']:.6f}")
            
            # Log detailed guidance metrics to wandb - this is crucial for diagnosing small scale issues
            if WANDB_AVAILABLE and wandb.run is not None:
                wandb.log({
                    f'mlff/iteration_{iteration + 1}_base_scale': guidance_stats['base_scale'],
                    f'mlff/iteration_{iteration + 1}_effective_noise_level': guidance_stats['effective_noise_level'],
                    f'mlff/iteration_{iteration + 1}_guidance_mean': guidance_stats['guidance_mean'],
                    f'mlff/iteration_{iteration + 1}_guidance_max': guidance_stats['guidance_max'],
                    f'mlff/iteration_{iteration + 1}_guidance_std': guidance_stats['guidance_std'],
                    f'mlff/iteration_{iteration + 1}_force_to_guidance_ratio': guidance_stats['force_to_guidance_ratio'],
                    # Track the mathematical operations that might cause issues with small scales
                    f'mlff/iteration_{iteration + 1}_raw_guidance_scale': self.guidance_scale,
                    f'mlff/iteration_{iteration + 1}_guidance_iterations': self.guidance_iterations,
                    f'mlff/iteration_{iteration + 1}_scale_division_result': self.guidance_scale / self.guidance_iterations,
                })
            
            # Check position guidance stability
            if not self.check_numerical_stability(position_guidance, "position_guidance"):
                logger.warning(f"Unstable position_guidance at iteration {iteration + 1}, breaking")
                break
            
            # Apply guidance to positions (first n_dims dimensions)
            # Ensure position guidance matches the z tensor shape
            if position_guidance.shape[1] <= z_guided.shape[1]:
                z_guided[:, :position_guidance.shape[1], :self.n_dims] += position_guidance
            else:
                z_guided[:, :, :self.n_dims] += position_guidance[:, :z_guided.shape[1], :]
            
            # Ensure mean-zero constraint is maintained after each iteration
            z_guided[:, :, :self.n_dims] = diffusion_utils.remove_mean_with_mask(
                z_guided[:, :, :self.n_dims], node_mask
            )
        
        # Log final guidance summary to wandb
        if WANDB_AVAILABLE and wandb.run is not None:
            # Calculate total position change from guidance
            position_change = torch.norm(z_guided[:, :, :self.n_dims] - z[:, :, :self.n_dims], dim=-1)
            
            wandb.log({
                'mlff/total_iterations_completed': iteration + 1,
                'mlff/mean_position_change': position_change.mean().item(),
                'mlff/max_position_change': position_change.max().item(),
                'mlff/guidance_applied': True,
                'mlff/final_noise_level': noise_level
            })
        
        return z_guided
    
    def sample_p_zs_given_zt_guided(self, s, t, zt, node_mask, edge_mask, context, dataset_info, fix_noise=False):
        """
        Enhanced sampling step with MLFF guidance.
        
        This method extends the original sampling to include force field guidance.
        """
        # Get noise parameters
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)
        
        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = \
            self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, zt)
        
        sigma_s = self.sigma(gamma_s, target_tensor=zt)
        sigma_t = self.sigma(gamma_t, target_tensor=zt)
        
        # Neural net prediction
        eps_t = self.phi(zt, t, node_mask, edge_mask, context)
        
        # Apply MLFF guidance before computing mu
        zt_guided = self.apply_mlff_guidance(zt, node_mask, dataset_info, sigma_t)
        
        # Compute mu for p(zs | zt) using guided state
        diffusion_utils.assert_mean_zero_with_mask(zt_guided[:, :, :self.n_dims], node_mask)
        diffusion_utils.assert_mean_zero_with_mask(eps_t[:, :, :self.n_dims], node_mask)
        mu = zt_guided / alpha_t_given_s - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_t
        
        # Compute sigma for p(zs | zt)
        sigma = sigma_t_given_s * sigma_s / sigma_t
        
        # Sample zs given the parameters derived from zt
        zs = self.sample_normal(mu, sigma, node_mask, fix_noise)
        
        # Project down to avoid numerical runaway of the center of gravity
        zs = torch.cat([
            diffusion_utils.remove_mean_with_mask(zs[:, :, :self.n_dims], node_mask),
            zs[:, :, self.n_dims:]
        ], dim=2)
        
        return zs
    
    @torch.no_grad()
    def sample(self, n_samples, n_nodes, node_mask, edge_mask, context, fix_noise=False, **kwargs):
        """
        Override the main sample method to handle both DDPM and DPM-Solver++ with MLFF guidance.
        """
        # Extract dataset_info from kwargs if available (needed for MLFF guidance)
        dataset_info = kwargs.get('dataset_info', None)
        
        if self.sampling_method == 'dpm_solver++':
            return self.sample_dpm_solver_with_mlff_guidance(
                n_samples, n_nodes, node_mask, edge_mask, context, dataset_info, fix_noise
            )
        else:
            # Use existing DDPM sampling with MLFF guidance
            return self.sample_with_mlff_guidance(
                n_samples, n_nodes, node_mask, edge_mask, context, dataset_info, fix_noise
            )
    
    @torch.no_grad()
    def sample_dpm_solver_with_mlff_guidance(self, n_samples, n_nodes, node_mask, edge_mask, context, dataset_info, fix_noise=False):
        """
        DPM-Solver++ sampling with MLFF guidance applied at each solver step.
        """
        if dataset_info is None:
            logger.warning("No dataset_info provided, falling back to unguided DPM-Solver++ sampling")
            return super().sample(n_samples, n_nodes, node_mask, edge_mask, context, fix_noise)
        
        # Initialize noise
        if fix_noise:
            z = self.sample_combined_position_feature_noise(1, n_nodes, node_mask)
        else:
            z = self.sample_combined_position_feature_noise(n_samples, n_nodes, node_mask)

        diffusion_utils.assert_mean_zero_with_mask(z[:, :, :self.n_dims], node_mask)

        # Initialize DPM-Solver++ lazily if needed
        if self.dpm_solver is None:
            from equivariant_diffusion.dpm_solver import DPMSolverPlusPlus
            self.dpm_solver = DPMSolverPlusPlus(
                model_fn=self.phi,
                noise_schedule_fn=self.gamma,
                order=self.dpm_solver_order,
                timesteps=self.T  # Pass number of training timesteps for DPM linear schedule
            )
        
        # Generate timesteps for DPM-Solver++
        from equivariant_diffusion.dpm_solver import get_time_steps_for_dpm_solver
        timesteps = get_time_steps_for_dpm_solver(self.dpm_solver_steps, z.device)
        
        logger.info(f"Starting DPM-Solver++ with MLFF guidance: {len(timesteps)} timesteps, "
                   f"guidance_scale={self.guidance_scale}, iterations={self.guidance_iterations}")
        
        # Clear DPM-Solver++ cache
        self.dpm_solver.model_prev_list = []
        self.dpm_solver.t_prev_list = []
        
        n_dims = z.size(2) if len(z.shape) > 2 else 3  # Assume first 3 dims are positions
        
        # DPM-Solver++ sampling loop with MLFF guidance
        for i in range(len(timesteps) - 1):
            t_prev = timesteps[i].item()
            t = timesteps[i + 1].item()
            
            # Compute noise level for guidance threshold
            noise_level = t  # t goes from 1.0 to 0.0
            
            
            # Perform one DPM-Solver++ step
            if self.dpm_solver.order == 2:
                z_before_guidance, model_current = self.dpm_solver.multistep_dpm_solver_second_update(
                    z, t_prev, t, node_mask, edge_mask, context)
            elif self.dpm_solver.order == 3:
                z_before_guidance, model_current = self.dpm_solver.multistep_dpm_solver_third_update(
                    z, t_prev, t, node_mask, edge_mask, context)
            else:
                raise ValueError(f"DPM-Solver++ order {self.dpm_solver.order} not supported")
            
            # Apply MLFF guidance if noise level is below threshold
            # Use simple timestep-based threshold to avoid memory issues
            if t <= self.noise_threshold:
                # Convert current noise level to sigma for guidance scaling
                t_tensor = torch.tensor([t], device=z.device)  # Make it 1D instead of scalar
                gamma_t = self.gamma(t_tensor)
                sigma_t = self.sigma(gamma_t, target_tensor=z_before_guidance)
                
                # Apply MLFF guidance
                z = self.apply_mlff_guidance(z_before_guidance, node_mask, dataset_info, sigma_t)
                
                logger.info(f"DPM-Solver++ step {i+1}/{len(timesteps)-1}: t={t:.3f}, "
                           f"noise_level={noise_level:.3f}, MLFF guidance applied")
            else:
                z = z_before_guidance
                logger.info(f"DPM-Solver++ step {i+1}/{len(timesteps)-1}: t={t:.3f}, "
                           f"noise_level={noise_level:.3f}, guidance skipped (above threshold)")
            
            # Preserve molecular constraints - remove center of mass drift
            if len(z.shape) > 2:  # Handle molecular data
                x_pos = z[:, :, :n_dims]
                x_features = z[:, :, n_dims:]
                x_pos = diffusion_utils.remove_mean_with_mask(x_pos, node_mask)
                z = torch.cat([x_pos, x_features], dim=-1)
            
            # Update DPM-Solver++ cache for next iteration - reuse computed model prediction
            if i < len(timesteps) - 2:  # Don't store on the last iteration
                if len(self.dpm_solver.model_prev_list) >= self.dpm_solver.order - 1:
                    self.dpm_solver.model_prev_list.pop(0)
                    self.dpm_solver.t_prev_list.pop(0)
                
                # Store the already computed prediction (no redundant computation!)
                self.dpm_solver.model_prev_list.append(model_current)
                self.dpm_solver.t_prev_list.append(t)

        # Finally sample p(x, h | z_0)
        x, h = self.sample_p_xh_given_z0(z, node_mask, edge_mask, context, fix_noise=fix_noise)

        diffusion_utils.assert_mean_zero_with_mask(x, node_mask)

        # Check for center of mass drift
        max_cog = torch.sum(x, dim=1, keepdim=True).abs().max().item()
        if max_cog > 5e-2:
            logger.warning(f'DPM-Solver++ + MLFF: Center of mass drift {max_cog:.3f}. Projecting positions.')
            x = diffusion_utils.remove_mean_with_mask(x, node_mask)

        logger.info("DPM-Solver++ with MLFF guidance completed successfully")
        return x, h
    
    @torch.no_grad()
    def sample_with_mlff_guidance(self, n_samples, n_nodes, node_mask, edge_mask, context, dataset_info, fix_noise=False):
        """
        Enhanced sampling method with MLFF guidance.
        
        Args:
            n_samples: Number of samples to generate
            n_nodes: Number of nodes per sample
            node_mask: Node mask tensor
            edge_mask: Edge mask tensor
            context: Context tensor (optional)
            dataset_info: Dataset information with atom decoders
            fix_noise: Whether to fix noise for reproducibility
            
        Returns:
            x: Generated positions
            h: Generated node features (atom types and charges)
        """
        if fix_noise:
            # Noise is broadcasted over the batch axis, useful for visualizations
            z = self.sample_combined_position_feature_noise(1, n_nodes, node_mask)
        else:
            z = self.sample_combined_position_feature_noise(n_samples, n_nodes, node_mask)
        
        diffusion_utils.assert_mean_zero_with_mask(z[:, :, :self.n_dims], node_mask)
        
        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1
        for s in reversed(range(0, self.T)):
            s_array = torch.full((n_samples, 1), fill_value=s, device=z.device)
            t_array = s_array + 1
            s_array = s_array / self.T
            t_array = t_array / self.T
            
            # Use guided sampling
            z = self.sample_p_zs_given_zt_guided(
                s_array, t_array, z, node_mask, edge_mask, context, dataset_info, fix_noise=fix_noise
            )
        
        # Finally sample p(x, h | z_0)
        x, h = self.sample_p_xh_given_z0(z, node_mask, edge_mask, context, fix_noise=fix_noise)
        
        diffusion_utils.assert_mean_zero_with_mask(x, node_mask)
        
        max_cog = torch.sum(x, dim=1, keepdim=True).abs().max().item()
        if max_cog > 5e-2:
            print(f'Warning cog drift with error {max_cog:.3f}. Projecting the positions down.')
            x = diffusion_utils.remove_mean_with_mask(x, node_mask)
        
        return x, h


def create_mlff_guided_model(original_model, mlff_predictor, guidance_scale=1.0, dataset_info=None, guidance_iterations=1, noise_threshold=0.8, force_clip_threshold=None):
    """
    Create an MLFF-guided version of an existing diffusion model.
    
    Args:
        original_model: Existing EnVariationalDiffusion model
        mlff_predictor: Pretrained MLFF predictor
        guidance_scale: Scale factor for force guidance
        dataset_info: Dataset information with atom decoders
        guidance_iterations: Number of iterative force field evaluations per diffusion step
        noise_threshold: Skip guidance when noise level exceeds this threshold (0.8 = skip first ~20% of steps)
        force_clip_threshold: Maximum force magnitude allowed (None = no clipping)
        
    Returns:
        MLFFGuidedDiffusion model
    """
    # Create guided model with same parameters as original
    # Use dummy noise schedule since we'll copy the exact gamma later
    guided_model = MLFFGuidedDiffusion(
        dynamics=original_model.dynamics,
        in_node_nf=original_model.in_node_nf,
        n_dims=original_model.n_dims,
        timesteps=original_model.T,
        parametrization='eps',
        noise_schedule='polynomial_2',  # Dummy - will be replaced
        loss_type=original_model.loss_type,
        norm_values=original_model.norm_values,
        norm_biases=original_model.norm_biases,
        include_charges=original_model.include_charges,
        mlff_predictor=mlff_predictor,
        guidance_scale=guidance_scale,
        dataset_info=dataset_info,
        guidance_iterations=guidance_iterations,
        noise_threshold=noise_threshold,
        force_clip_threshold=force_clip_threshold
    )
    
    # Move guided model to same device as original model
    device = next(original_model.parameters()).device
    guided_model = guided_model.to(device)
    
    # Copy ALL trained parameters including the exact gamma schedule
    guided_model.load_state_dict(original_model.state_dict(), strict=False)
    
    # Directly copy the gamma schedule from the original model
    # This ensures exactly the same noise schedule is used
    guided_model.gamma = original_model.gamma
    
    # Copy sampling method and DPM solver configuration
    guided_model.sampling_method = original_model.sampling_method
    guided_model.dpm_solver_order = getattr(original_model, 'dpm_solver_order', 2)
    guided_model.dpm_solver_steps = getattr(original_model, 'dpm_solver_steps', 20)
    guided_model.dpm_solver = getattr(original_model, 'dpm_solver', None)
    
    return guided_model


def enhanced_sampling_with_mlff(model, mlff_predictor, n_samples, n_nodes, node_mask, edge_mask, 
                              context, dataset_info, guidance_scale=1.0, guidance_iterations=1, noise_threshold=0.8, force_clip_threshold=None, fix_noise=False):
    """
    Convenience function for enhanced sampling with MLFF guidance.
    
    Args:
        model: Trained diffusion model
        mlff_predictor: Pretrained MLFF predictor
        n_samples: Number of samples to generate
        n_nodes: Number of nodes per sample
        node_mask: Node mask tensor
        edge_mask: Edge mask tensor
        context: Context tensor (optional)
        dataset_info: Dataset information
        guidance_scale: Scale factor for force guidance
        guidance_iterations: Number of iterative force field evaluations per diffusion step
        noise_threshold: Skip guidance when noise level exceeds this threshold (0.8 = skip first ~20% of steps)
        force_clip_threshold: Maximum force magnitude allowed (None = no clipping)
        fix_noise: Whether to fix noise for reproducibility
        
    Returns:
        x: Generated positions
        h: Generated node features
    """
    # Create guided model
    guided_model = create_mlff_guided_model(
        model, mlff_predictor, guidance_scale, dataset_info, guidance_iterations, noise_threshold, force_clip_threshold
    )
    
    # Generate samples with MLFF guidance (automatically routes to DPM-Solver++ or DDPM)
    return guided_model.sample(
        n_samples, n_nodes, node_mask, edge_mask, context, fix_noise, dataset_info=dataset_info
    ) 