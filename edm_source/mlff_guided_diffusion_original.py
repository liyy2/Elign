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


class MLFFLogger:
    """Centralized logging for MLFF guidance with optional wandb integration."""
    
    def __init__(self, use_wandb=True):
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        
    def log_input_statistics(self, z, node_mask, position_scale):
        """Log statistics about input to force computation."""
        positions = z[:, :, :3]
        pos_magnitudes = torch.norm(positions, dim=-1)
        valid_atoms_per_batch = node_mask.sum(dim=1).squeeze(-1)
        
        logger.info(f"Input positions - mean mag: {pos_magnitudes.mean().item():.6f}, "
                   f"max mag: {pos_magnitudes.max().item():.6f}, "
                   f"min mag: {pos_magnitudes.min().item():.6f}")
        logger.info(f"Valid atoms per batch: {valid_atoms_per_batch.tolist()}")
        logger.info(f"Position scale factor: {position_scale}")
        
        # Physical positions after scaling
        phys_pos_magnitudes = pos_magnitudes * position_scale
        logger.info(f"Physical positions - mean mag: {phys_pos_magnitudes.mean().item():.6f}, "
                   f"max mag: {phys_pos_magnitudes.max().item():.6f}")
    
    def log_batch_properties(self, batch, atomic_data_list):
        """Log properties of the atomic batch before MLFF prediction."""
        logger.info(f"Created {len(atomic_data_list)} atomic data objects")
        logger.info(f"Batch info - num atoms: {batch.pos.shape[0]}, "
                   f"num systems: {batch.ptr.shape[0] - 1 if hasattr(batch, 'ptr') else len(atomic_data_list)}")
        
        if hasattr(batch, 'atomic_numbers'):
            unique_atoms = torch.unique(batch.atomic_numbers)
            logger.info(f"Unique atomic numbers in batch: {unique_atoms.tolist()}")
        
        # Check inter-atomic distances
        if batch.pos.shape[0] > 1:
            distances = torch.cdist(batch.pos, batch.pos)
            mask = ~torch.eye(distances.shape[0], dtype=bool, device=distances.device)
            if mask.any():
                min_dist = distances[mask].min().item()
                max_dist = distances[mask].max().item()
                logger.info(f"Inter-atomic distances - min: {min_dist:.6f}, max: {max_dist:.6f}")
    
    def log_mlff_predictions(self, predictions):
        """Log MLFF prediction results."""
        logger.info(f"MLFF prediction keys: {list(predictions.keys())}")
        
        if 'energy' in predictions:
            energy = predictions['energy']
            logger.info(f"MLFF energy: {energy.item() if energy.numel() == 1 else energy.mean().item():.6f} eV")
    
    def log_force_statistics(self, forces):
        """Log detailed force statistics."""
        force_magnitudes = torch.norm(forces, dim=-1)
        zero_force_count = (force_magnitudes == 0).sum().item()
        nonzero_force_count = (force_magnitudes > 0).sum().item()
        
        stats = {
            'mean_magnitude': force_magnitudes.mean().item(),
            'max_magnitude': force_magnitudes.max().item(),
            'min_magnitude': force_magnitudes.min().item(),
            'std_magnitude': force_magnitudes.std().item(),
            'zero_forces': zero_force_count,
            'nonzero_forces': nonzero_force_count
        }
        
        logger.info(f"MLFF forces - mean magnitude: {stats['mean_magnitude']:.6f}, "
                   f"max magnitude: {stats['max_magnitude']:.6f}, "
                   f"min magnitude: {stats['min_magnitude']:.6f}, "
                   f"zero/nonzero: {zero_force_count}/{nonzero_force_count}")
        
        if self.use_wandb and wandb.run is not None:
            wandb.log({
                'mlff/forces_mean_magnitude': stats['mean_magnitude'],
                'mlff/forces_max_magnitude': stats['max_magnitude'],
                'mlff/forces_min_magnitude': stats['min_magnitude'],
                'mlff/forces_std_magnitude': stats['std_magnitude'],
                'mlff/forces_zero_count': zero_force_count,
                'mlff/forces_nonzero_count': nonzero_force_count
            })
        
        return stats
    
    def log_zero_forces_warning(self, batch):
        """Log detailed warning when MLFF returns all zero forces."""
        logger.warning("MLFF returned all zero forces!")
        logger.warning(f"This might indicate: 1) Energy minimum reached, 2) UMA model issue, "
                      f"3) Invalid molecular configuration")
        logger.warning(f"Batch positions range: [{batch.pos.min().item():.6f}, {batch.pos.max().item():.6f}]")
        if hasattr(batch, 'atomic_numbers'):
            logger.warning(f"Atomic numbers (first 10): {batch.atomic_numbers.tolist()[:10]}...")
    
    def log_guidance_iteration(self, iteration, force_stats, guidance_stats):
        """Log statistics for a guidance iteration."""
        logger.info(f"Iteration {iteration + 1}: force magnitudes - mean: {force_stats['mean']:.6f}, "
                   f"max: {force_stats['max']:.6f}")
        logger.info(f"Iteration {iteration + 1}: guidance scale={guidance_stats['base_scale']:.6f}, "
                   f"noise_level={guidance_stats['effective_noise_level']:.6f}, "
                   f"guidance magnitudes - mean: {guidance_stats['guidance_mean']:.6f}, "
                   f"max: {guidance_stats['guidance_max']:.6f}")
        
        if self.use_wandb and wandb.run is not None:
            wandb.log({
                f'mlff/iteration_{iteration + 1}_force_mean': force_stats['mean'],
                f'mlff/iteration_{iteration + 1}_force_max': force_stats['max'],
                f'mlff/iteration_{iteration + 1}_force_std': force_stats['std'],
                f'mlff/iteration_{iteration + 1}_base_scale': guidance_stats['base_scale'],
                f'mlff/iteration_{iteration + 1}_effective_noise_level': guidance_stats['effective_noise_level'],
                f'mlff/iteration_{iteration + 1}_guidance_mean': guidance_stats['guidance_mean'],
                f'mlff/iteration_{iteration + 1}_guidance_max': guidance_stats['guidance_max'],
                f'mlff/iteration_{iteration + 1}_guidance_std': guidance_stats['guidance_std'],
            })
    
    def log_guidance_summary(self, z_original, z_guided, iterations_completed, noise_level):
        """Log summary after guidance application."""
        position_change = torch.norm(z_guided[:, :, :3] - z_original[:, :, :3], dim=-1)
        
        if self.use_wandb and wandb.run is not None:
            wandb.log({
                'mlff/total_iterations_completed': iterations_completed,
                'mlff/mean_position_change': position_change.mean().item(),
                'mlff/max_position_change': position_change.max().item(),
                'mlff/guidance_applied': True,
                'mlff/final_noise_level': noise_level
            })


class MLFFForceComputer:
    """Handles MLFF force computation with proper error handling."""
    
    def __init__(self, mlff_predictor, n_dims, norm_values, include_charges, molecule_cell_size=20.0):
        self.mlff_predictor = mlff_predictor
        self.n_dims = n_dims
        self.norm_values = norm_values
        self.include_charges = include_charges
        self.molecule_cell_size = molecule_cell_size
        self.task_name = "omol"  # For molecular systems
        self.logger = MLFFLogger()
    
    def compute_forces(self, z, node_mask, dataset_info):
        """
        Compute MLFF forces for the given molecular configuration.
        
        Returns:
            Forces tensor [batch_size, n_nodes, 3] (zero for masked nodes)
        """
        if self.mlff_predictor is None:
            return torch.zeros_like(z[:, :, :self.n_dims])
        
        
        # Get position scale
        position_scale = self._get_position_scale()
        
        # Log input statistics
        self.logger.log_input_statistics(z, node_mask, position_scale)
        
        # Convert to physical coordinates
        z_physical = self._to_physical_coordinates(z, position_scale)
        
        # Convert to atomic data
        atomic_data_list = self._diffusion_to_atomic_data(z_physical, node_mask, dataset_info)
        if not atomic_data_list:
            logger.warning("No atomic data created from diffusion state")
            return torch.zeros_like(z[:, :, :self.n_dims])
        
        # Batch atomic data
        try:
            batch = atomicdata_list_to_batch(atomic_data_list)
        except Exception as e:
            logger.warning(f"Could not batch atomic data: {e}")
            return torch.zeros_like(z[:, :, :self.n_dims])
        
        # Log batch properties
        self.logger.log_batch_properties(batch, atomic_data_list)
        
        # Get MLFF predictions
        try:
            forces = self._get_mlff_predictions(batch)
            if forces is None:
                return torch.zeros_like(z[:, :, :self.n_dims])
        except Exception as e:
            logger.warning(f"MLFF prediction failed: {e}")
            return torch.zeros_like(z[:, :, :self.n_dims])
        
        # Convert forces back to normalized coordinates
        forces_batched = self._reshape_forces_to_batch(
            forces, z, node_mask, batch, atomic_data_list, position_scale
        )
        
        return forces_batched
    
    def _check_numerical_stability(self, tensor, name="tensor"):
        """Check for NaN or infinity values in tensor."""
        if torch.isnan(tensor).any():
            logger.warning(f"NaN detected in {name}: {torch.isnan(tensor).sum().item()} values")
            return False
        if torch.isinf(tensor).any():
            logger.warning(f"Infinity detected in {name}: {torch.isinf(tensor).sum().item()} values")
            return False
        return True
    
    def _get_position_scale(self):
        """Get the position scaling factor."""
        try:
            if isinstance(self.norm_values, (list, tuple)):
                return float(self.norm_values[0])
            return float(self.norm_values)
        except Exception:
            return 1.0
    
    def _to_physical_coordinates(self, z, position_scale):
        """Convert normalized positions to physical units."""
        z_physical = z.clone()
        z_physical[:, :, :self.n_dims] = z[:, :, :self.n_dims] * position_scale
        return z_physical
    
    def _diffusion_to_atomic_data(self, z, node_mask, dataset_info):
        """Convert diffusion state to AtomicData format."""
        batch_size = z.shape[0]
        atomic_data_list = []
        
        for batch_idx in range(batch_size):
            atomic_data = self._create_atomic_data_for_batch(
                z[batch_idx], node_mask[batch_idx], dataset_info
            )
            if atomic_data is not None:
                atomic_data_list.append(atomic_data)
        
        return atomic_data_list
    
    def _create_atomic_data_for_batch(self, z_single, mask_single, dataset_info):
        """Create AtomicData for a single batch element."""
        positions = z_single[:, :self.n_dims]
        node_features = z_single[:, self.n_dims:]
        mask = mask_single[:, 0]
        
        # Filter valid nodes
        valid_nodes = mask.bool()
        positions = positions[valid_nodes]
        node_features = node_features[valid_nodes]
        
        if positions.shape[0] == 0:
            return None
        
        # Extract atom types
        if self.include_charges:
            categorical_features = node_features[:, :-1]
        else:
            categorical_features = node_features
        
        atom_types = torch.argmax(categorical_features, dim=1)
        
        # Map to atomic numbers
        atomic_numbers = self._map_to_atomic_numbers(atom_types, dataset_info)
        
        # Create ASE atoms
        atoms = Atoms(
            numbers=atomic_numbers.detach().cpu().numpy(),
            positions=positions.detach().cpu().numpy(),
            cell=np.eye(3) * self.molecule_cell_size,
            pbc=False
        )
        atoms.info['charge'] = 0
        atoms.info['spin'] = 1
        
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
            return atomic_data
        except Exception as e:
            logger.debug(f"Could not convert to AtomicData: {e}")
            return None
    
    def _map_to_atomic_numbers(self, atom_types, dataset_info):
        """Map atom types to atomic numbers."""
        atom_types_cpu = atom_types.cpu()
        
        if 'atomic_nb' in dataset_info:
            # GEOM dataset
            return torch.tensor([dataset_info['atomic_nb'][idx] for idx in atom_types_cpu])
        else:
            # QM9 dataset
            atom_decoder = dataset_info['atom_decoder']
            atomic_number_map = {
                'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9,
                'B': 5, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16,
                'Cl': 17, 'As': 33, 'Br': 35, 'I': 53, 'Hg': 80, 'Bi': 83
            }
            return torch.tensor([atomic_number_map[atom_decoder[idx]] for idx in atom_types_cpu])
    
    def _get_mlff_predictions(self, batch):
        """Get force predictions from MLFF model."""
        with torch.no_grad():
            predictions = self.mlff_predictor.predict(batch)
        
        self.logger.log_mlff_predictions(predictions)
        
        if 'forces' not in predictions:
            logger.warning("No forces in MLFF predictions")
            return None
        
        forces = predictions['forces']
        
        # Check for all-zero forces
        if torch.all(forces == 0):
            self.logger.log_zero_forces_warning(batch)
            return None
        
        # Check numerical stability
        if not self._check_numerical_stability(forces, "MLFF forces"):
            logger.warning("Unstable MLFF forces detected")
            return None
        
        # Log force statistics
        self.logger.log_force_statistics(forces)
        
        return forces
    
    def _reshape_forces_to_batch(self, forces, z, node_mask, batch, atomic_data_list, position_scale):
        """Reshape forces back to original batch format."""
        forces_batched = torch.zeros_like(z[:, :, :self.n_dims])
        
        for batch_idx, atomic_data in enumerate(atomic_data_list):
            system_forces = forces[batch.batch == batch_idx]
            mask = node_mask[batch_idx, :, 0]
            valid_indices = torch.where(mask)[0]
            
            if len(valid_indices) == len(system_forces):
                system_forces = system_forces.to(forces_batched.device)
                # Convert physical forces to normalized coordinate forces
                system_forces_normalized = system_forces * position_scale
                forces_batched[batch_idx, valid_indices] = system_forces_normalized
        
        return forces_batched


class MLFFGuidedDiffusion(EnVariationalDiffusion):
    """
    Enhanced diffusion model with MLFF force field guidance.
    
    This class extends the EnVariationalDiffusion to incorporate guidance from
    a pretrained Machine Learning Force Field (MLFF) during the sampling process.
    """
    
    def __init__(self, *args, mlff_predictor=None, guidance_scale=1.0, 
                 dataset_info=None, guidance_iterations=1, noise_threshold=0.8, 
                 force_clip_threshold=None, displacement_clip=None, **kwargs):
        """
        Initialize MLFF-guided diffusion model.
        
        Args:
            *args: Arguments for parent EnVariationalDiffusion class
            mlff_predictor: Pretrained MLFF predictor from fairchem
            guidance_scale: Scale factor for MLFF force guidance
            dataset_info: Dataset information with atom decoders
            guidance_iterations: Number of iterative force field evaluations per diffusion step
            noise_threshold: Skip guidance when noise level exceeds this threshold
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
        # Optional per-step clamp on total position update magnitude (in normalized units).
        # If None, no clamping is applied.
        self.displacement_clip = displacement_clip
        
        # Initialize force computer and logger
        self.force_computer = MLFFForceComputer(
            mlff_predictor, self.n_dims, self.norm_values, 
            self.include_charges
        )
        self.mlff_logger = MLFFLogger()
    
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
        return self.force_computer.compute_forces(z, node_mask, dataset_info)
    
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
        
        noise_level = sigma.mean().item() if sigma.numel() > 0 else 1.0
        logger.info(f"Applying MLFF guidance: noise_level={noise_level:.6f}, "
                   f"guidance_scale={self.guidance_scale:.6f}")
        
        # Skip guidance entirely on high-noise steps
        noise_level = sigma.mean().item() if sigma.numel() > 0 else 1.0
        if noise_level > self.noise_threshold:
            logger.info(
                f"Skipping MLFF guidance: noise_level={noise_level:.6f} exceeds threshold={self.noise_threshold:.6f}"
            )
            return z

        z_guided = z.clone()
        iterations_completed = 0
        
        # Iterative force field guidance
        for iteration in range(self.guidance_iterations):
            # Get MLFF forces
            forces = self.get_mlff_forces(z_guided, node_mask, dataset_info)
            
            # Check if forces are meaningful
            if torch.all(forces == 0):
                logger.info(f"All forces are zero at iteration {iteration + 1}, breaking")
                break
            
            # Apply force clipping if specified
            if self.force_clip_threshold is not None:
                forces = self._clip_forces(forces, self.force_clip_threshold)
            
            # Calculate guidance scaling
            sigma_for_forces = self._prepare_sigma_for_forces(sigma, forces.shape)
            
            if self.guidance_iterations == 0:
                logger.warning("guidance_iterations is 0, skipping guidance")
                break
            
            base_scale = self.guidance_scale / self.guidance_iterations
            force_scale = base_scale * sigma_for_forces.unsqueeze(-1)
            
            # Check numerical stability
            if not self._check_numerical_stability(force_scale, "force_scale"):
                logger.warning(f"Unstable force_scale at iteration {iteration + 1}, breaking")
                break
            
            # Apply scaled forces
            position_guidance = forces * force_scale

            # Optionally clamp displacement magnitude to prevent large jumps
            if self.displacement_clip is not None and self.displacement_clip > 0:
                disp_mag = torch.norm(position_guidance, dim=-1, keepdim=True)  # [B, N, 1]
                exceed = disp_mag > self.displacement_clip
                # Avoid division by zero
                safe_den = torch.where(disp_mag > 0, disp_mag, torch.ones_like(disp_mag))
                scale = torch.where(
                    exceed,
                    (self.displacement_clip / safe_den),
                    torch.ones_like(disp_mag)
                )
                clamped_before = exceed.sum().item()
                position_guidance = position_guidance * scale
                if clamped_before > 0:
                    logger.info(
                        f"Clamped {clamped_before} guidance vectors to max displacement {self.displacement_clip:.4f}"
                    )
            
            # Log iteration statistics
            self._log_iteration_stats(iteration, forces, position_guidance, sigma_for_forces, base_scale)
            
            # Apply guidance to positions
            z_guided = self._apply_position_update(z_guided, position_guidance, node_mask)
            iterations_completed += 1
        
        # Log final summary
        self.mlff_logger.log_guidance_summary(z, z_guided, iterations_completed, noise_level)
        
        return z_guided
    
    def _check_numerical_stability(self, tensor, name="tensor"):
        """Check for NaN or infinity values."""
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            return False
        return True
    
    def _clip_forces(self, forces, threshold):
        """Clip force magnitudes while preserving direction."""
        force_magnitudes = torch.norm(forces, dim=-1, keepdim=True)
        force_clip_mask = force_magnitudes > threshold
        return torch.where(
            force_clip_mask,
            forces * (threshold / force_magnitudes),
            forces
        )
    
    def _prepare_sigma_for_forces(self, sigma, forces_shape):
        """Prepare sigma tensor for broadcasting with forces."""
        if sigma.dim() == 3:
            sigma_for_forces = sigma.squeeze(-1)
        else:
            sigma_for_forces = sigma
        
        batch_size, n_nodes_forces, _ = forces_shape
        if sigma_for_forces.shape[1] != n_nodes_forces:
            sigma_for_forces = sigma_for_forces[:, :n_nodes_forces]
        
        return sigma_for_forces
    
    def _log_iteration_stats(self, iteration, forces, position_guidance, sigma_for_forces, base_scale):
        """Log statistics for current iteration."""
        force_magnitudes = torch.norm(forces, dim=-1)
        force_stats = {
            'mean': force_magnitudes.mean().item(),
            'max': force_magnitudes.max().item(),
            'std': force_magnitudes.std().item()
        }
        
        guidance_magnitudes = torch.norm(position_guidance, dim=-1)
        guidance_stats = {
            'base_scale': base_scale,
            'effective_noise_level': sigma_for_forces.mean().item(),
            'guidance_mean': guidance_magnitudes.mean().item(),
            'guidance_max': guidance_magnitudes.max().item(),
            'guidance_std': guidance_magnitudes.std().item()
        }
        
        self.mlff_logger.log_guidance_iteration(iteration, force_stats, guidance_stats)
    
    def _apply_position_update(self, z_guided, position_guidance, node_mask):
        """Apply position guidance and maintain constraints."""
        if position_guidance.shape[1] <= z_guided.shape[1]:
            z_guided[:, :position_guidance.shape[1], :self.n_dims] += position_guidance
        else:
            z_guided[:, :, :self.n_dims] += position_guidance[:, :z_guided.shape[1], :]
        
        # Maintain mean-zero constraint
        z_guided[:, :, :self.n_dims] = diffusion_utils.remove_mean_with_mask(
            z_guided[:, :, :self.n_dims], node_mask
        )
        
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
        dataset_info = kwargs.get('dataset_info', None)
        
        if self.sampling_method == 'dpm_solver++':
            return self.sample_dpm_solver_with_mlff_guidance(
                n_samples, n_nodes, node_mask, edge_mask, context, dataset_info, fix_noise
            )
        else:
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
        
        # Create guided model function
        def guided_model_fn(x, t, node_mask, edge_mask, context):
            """Guided model function that applies MLFF guidance to epsilon prediction."""
            eps_pred = self.phi(x, t, node_mask, edge_mask, context)
            
            if (self.mlff_predictor is not None and 
                self.guidance_scale > 0 and 
                dataset_info is not None):
                
                t_value = t.mean().item() if t.numel() > 0 else 1.0
                
                if t_value <= self.noise_threshold:
                    forces = self.get_mlff_forces(x, node_mask, dataset_info)
                    
                    if not torch.all(forces == 0):
                        # Get sigma_t for current timestep
                        if t.dim() == 0:
                            t_tensor = t.unsqueeze(0)
                        elif t.dim() == 1 and t.size(0) == 1:
                            t_tensor = t
                        else:
                            t_tensor = t[:1]
                        
                        gamma_t = self.gamma(t_tensor)
                        sigma_t = self.sigma(gamma_t, target_tensor=x)
                        
                        while sigma_t.dim() < forces.dim():
                            sigma_t = sigma_t.unsqueeze(-1)
                        
                        # Apply guidance formula
                        if forces.shape[1] <= eps_pred.shape[1]:
                            eps_pred[:, :forces.shape[1], :self.n_dims] -= (
                                self.guidance_scale * sigma_t * forces
                            )
                        
                        logger.info(f"MLFF guidance applied: t={t_value:.3f}, "
                                   f"sigma_t={sigma_t.mean().item():.6f}")
            
            return eps_pred
        
        # Initialize DPM-Solver++
        if self.dpm_solver is None:
            from equivariant_diffusion.dpm_solver import DPMSolverPlusPlus
            self.dpm_solver = DPMSolverPlusPlus(
                model_fn=guided_model_fn,
                noise_schedule_fn=self.gamma,
                order=self.dpm_solver_order,
                timesteps=self.T
            )
        
        # Generate timesteps
        from equivariant_diffusion.dpm_solver import get_time_steps_for_dpm_solver
        timesteps = get_time_steps_for_dpm_solver(self.dpm_solver_steps, z.device)
        
        logger.info(f"Starting DPM-Solver++ with MLFF guidance: {len(timesteps)} timesteps")
        
        # Clear cache
        self.dpm_solver.model_prev_list = []
        self.dpm_solver.t_prev_list = []
        
        n_dims = z.size(2) if len(z.shape) > 2 else 3
        
        # DPM-Solver++ sampling loop
        for i in range(len(timesteps) - 1):
            t_prev = timesteps[i].item()
            t = timesteps[i + 1].item()
            
            # Perform one DPM-Solver++ step
            if self.dpm_solver.order == 2:
                z, model_current = self.dpm_solver.multistep_dpm_solver_second_update(
                    z, t_prev, t, node_mask, edge_mask, context)
            elif self.dpm_solver.order == 3:
                z, model_current = self.dpm_solver.multistep_dpm_solver_third_update(
                    z, t_prev, t, node_mask, edge_mask, context)
            else:
                raise ValueError(f"DPM-Solver++ order {self.dpm_solver.order} not supported")
            
            logger.info(f"DPM-Solver++ step {i+1}/{len(timesteps)-1}: t={t:.3f}")
            
            # Preserve molecular constraints
            if len(z.shape) > 2:
                x_pos = z[:, :, :n_dims]
                x_features = z[:, :, n_dims:]
                x_pos = diffusion_utils.remove_mean_with_mask(x_pos, node_mask)
                z = torch.cat([x_pos, x_features], dim=-1)
            
            # Update cache for next iteration
            if i < len(timesteps) - 2:
                if len(self.dpm_solver.model_prev_list) >= self.dpm_solver.order - 1:
                    self.dpm_solver.model_prev_list.pop(0)
                    self.dpm_solver.t_prev_list.pop(0)
                
                self.dpm_solver.model_prev_list.append(model_current)
                self.dpm_solver.t_prev_list.append(t)
        
        # Final sampling step
        x, h = self.sample_p_xh_given_z0(z, node_mask, edge_mask, context, fix_noise=fix_noise)
        
        diffusion_utils.assert_mean_zero_with_mask(x, node_mask)
        
        # Check for center of mass drift
        max_cog = torch.sum(x, dim=1, keepdim=True).abs().max().item()
        if max_cog > 5e-2:
            logger.warning(f'Center of mass drift {max_cog:.3f}. Projecting positions.')
            x = diffusion_utils.remove_mean_with_mask(x, node_mask)
        
        logger.info("DPM-Solver++ with MLFF guidance completed successfully")
        return x, h
    
    @torch.no_grad()
    def sample_with_mlff_guidance(self, n_samples, n_nodes, node_mask, edge_mask, context, dataset_info, fix_noise=False):
        """
        Enhanced sampling method with MLFF guidance for DDPM.
        """
        if fix_noise:
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
            logger.warning(f'Center of mass drift {max_cog:.3f}. Projecting positions.')
            x = diffusion_utils.remove_mean_with_mask(x, node_mask)
        
        return x, h


def create_mlff_guided_model(original_model, mlff_predictor, guidance_scale=1.0, 
                           dataset_info=None, guidance_iterations=1, 
                           noise_threshold=0.8, force_clip_threshold=None):
    """
    Create an MLFF-guided version of an existing diffusion model.
    
    Args:
        original_model: Existing EnVariationalDiffusion model
        mlff_predictor: Pretrained MLFF predictor
        guidance_scale: Scale factor for force guidance
        dataset_info: Dataset information with atom decoders
        guidance_iterations: Number of iterative force field evaluations per diffusion step
        noise_threshold: Skip guidance when noise level exceeds this threshold
        force_clip_threshold: Maximum force magnitude allowed (None = no clipping)
        
    Returns:
        MLFFGuidedDiffusion model
    """
    # Create guided model with same parameters as original
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
    
    # Move to same device as original
    device = next(original_model.parameters()).device
    guided_model = guided_model.to(device)
    
    # Copy trained parameters
    guided_model.load_state_dict(original_model.state_dict(), strict=False)
    
    # Copy gamma schedule
    guided_model.gamma = original_model.gamma
    
    # Copy sampling configuration
    guided_model.sampling_method = original_model.sampling_method
    guided_model.dpm_solver_order = getattr(original_model, 'dpm_solver_order', 2)
    guided_model.dpm_solver_steps = getattr(original_model, 'dpm_solver_steps', 20)
    guided_model.dpm_solver = getattr(original_model, 'dpm_solver', None)
    
    return guided_model


def enhanced_sampling_with_mlff(model, mlff_predictor, n_samples, n_nodes, node_mask, edge_mask, 
                              context, dataset_info, guidance_scale=1.0, guidance_iterations=1, 
                              noise_threshold=0.8, force_clip_threshold=None, fix_noise=False):
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
        noise_threshold: Skip guidance when noise level exceeds this threshold
        force_clip_threshold: Maximum force magnitude allowed (None = no clipping)
        fix_noise: Whether to fix noise for reproducibility
        
    Returns:
        x: Generated positions
        h: Generated node features
    """
    # Create guided model
    guided_model = create_mlff_guided_model(
        model, mlff_predictor, guidance_scale, dataset_info, 
        guidance_iterations, noise_threshold, force_clip_threshold
    )
    
    # Generate samples with MLFF guidance
    return guided_model.sample(
        n_samples, n_nodes, node_mask, edge_mask, context, fix_noise, dataset_info=dataset_info
    )
