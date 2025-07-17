import torch
import torch.nn.functional as F
import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.datasets import data_list_collater
from equivariant_diffusion import utils as diffusion_utils
from equivariant_diffusion.en_diffusion import EnVariationalDiffusion


class MLFFGuidedDiffusion(EnVariationalDiffusion):
    """
    Enhanced diffusion model with MLFF force field guidance.
    
    This class extends the EnVariationalDiffusion to incorporate guidance from
    a pretrained Machine Learning Force Field (MLFF) during the sampling process.
    """
    
    def __init__(self, *args, mlff_predictor=None, guidance_scale=1.0, 
                 dataset_info=None, **kwargs):
        """
        Initialize MLFF-guided diffusion model.
        
        Args:
            *args: Arguments for parent EnVariationalDiffusion class
            mlff_predictor: Pretrained MLFF predictor from fairchem
            guidance_scale: Scale factor for MLFF force guidance
            dataset_info: Dataset information with atom decoders
            **kwargs: Keyword arguments for parent class
        """
        super().__init__(*args, **kwargs)
        self.mlff_predictor = mlff_predictor
        self.guidance_scale = guidance_scale
        self.dataset_info = dataset_info
        
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
            
            if valid_positions.shape[0] <= 1:
                continue  # Skip single atoms
            
            # Calculate pairwise distances
            distances = torch.cdist(valid_positions, valid_positions)  # [n_valid, n_valid]
            
            # Remove diagonal (self-distances)
            distances = distances + torch.eye(distances.shape[0], device=distances.device) * float('inf')
            
            # Find minimum distance to any other atom
            min_distances = torch.min(distances, dim=1)[0]
            
            # Check if any atom has a neighbor within max_distance
            if torch.any(min_distances <= max_distance):
                return True
        
        return False
    
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
        
        # Check if system is feasible for MLFF prediction
        if not self.check_system_feasibility(z, node_mask, max_distance=6.0):
            return torch.zeros_like(z[:, :, :self.n_dims])
            
        # Convert to AtomicData format
        atomic_data_list = self.diffusion_to_atomic_data(z, node_mask, dataset_info)
        
        if not atomic_data_list:
            return torch.zeros_like(z[:, :, :self.n_dims])
        
        # Batch the atomic data
        try:
            batch = data_list_collater(atomic_data_list, otf_graph=True)
        except Exception as e:
            print(f"Warning: Could not batch atomic data: {e}")
            return torch.zeros_like(z[:, :, :self.n_dims])
        
        # Get predictions from MLFF
        try:
            with torch.no_grad():
                predictions = self.mlff_predictor.predict(batch)
                
            if 'forces' not in predictions:
                return torch.zeros_like(z[:, :, :self.n_dims])
                
            forces = predictions['forces']  # [total_atoms, 3]
            
        except Exception as e:
            print(f"Warning: MLFF prediction failed: {e}")
            return torch.zeros_like(z[:, :, :self.n_dims])
        
        # Reshape forces back to original batch format
        forces_batched = torch.zeros_like(z[:, :, :self.n_dims])
        
        atom_idx = 0
        for batch_idx, atomic_data in enumerate(atomic_data_list):
            n_atoms = atomic_data.natoms.item()
            batch_forces = forces[atom_idx:atom_idx + n_atoms]  # [n_atoms, 3]
            
            # Get valid node indices for this batch
            mask = node_mask[batch_idx, :, 0]
            valid_indices = torch.where(mask)[0]
            
            if len(valid_indices) == n_atoms:
                # Ensure batch_forces is on the same device as forces_batched
                batch_forces = batch_forces.to(forces_batched.device)
                forces_batched[batch_idx, valid_indices] = batch_forces
                
            atom_idx += n_atoms
            
        return forces_batched
    
    def apply_mlff_guidance(self, z, node_mask, dataset_info, sigma):
        """
        Apply MLFF force guidance to the diffusion process.
        
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
        
        # Skip guidance if system is too noisy/dispersed
        # Use adaptive threshold based on noise level
        noise_level = sigma.mean().item() if sigma.numel() > 0 else 1.0
        max_distance = 6.0 if noise_level < 0.5 else 10.0  # More lenient for high noise
        
        if not self.check_system_feasibility(z, node_mask, max_distance=max_distance):
            return z
            
        # Get MLFF forces
        forces = self.get_mlff_forces(z, node_mask, dataset_info)
        
        # Check if forces are meaningful (not all zeros)
        if torch.all(forces == 0):
            return z
        
        # Apply force guidance to positions only
        z_guided = z.clone()
        
        # Scale forces by guidance scale and noise level
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
        
        force_scale = self.guidance_scale * sigma_for_forces.unsqueeze(-1)  # [batch_size, n_nodes_forces, 1]
        position_guidance = forces * force_scale
        
        # Apply guidance to positions (first n_dims dimensions)
        # Ensure position guidance matches the z tensor shape
        if position_guidance.shape[1] <= z_guided.shape[1]:
            z_guided[:, :position_guidance.shape[1], :self.n_dims] += position_guidance
        else:
            z_guided[:, :, :self.n_dims] += position_guidance[:, :z_guided.shape[1], :]
        
        # Ensure mean-zero constraint is maintained
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


def create_mlff_guided_model(original_model, mlff_predictor, guidance_scale=1.0, dataset_info=None):
    """
    Create an MLFF-guided version of an existing diffusion model.
    
    Args:
        original_model: Existing EnVariationalDiffusion model
        mlff_predictor: Pretrained MLFF predictor
        guidance_scale: Scale factor for force guidance
        dataset_info: Dataset information with atom decoders
        
    Returns:
        MLFFGuidedDiffusion model
    """
    # Detect noise schedule and loss type from original model
    # Check if gamma is a GammaNetwork (learned) or PredefinedNoiseSchedule (fixed)
    if hasattr(original_model.gamma, 'l1'):  # GammaNetwork has l1, l2, l3 layers
        noise_schedule = 'learned'
        loss_type = 'vlb'  # Learned schedules require vlb loss
    else:  # PredefinedNoiseSchedule
        noise_schedule = 'cosine'  # Default to cosine for safety
        loss_type = original_model.loss_type
    
    # Create guided model with same parameters as original
    guided_model = MLFFGuidedDiffusion(
        dynamics=original_model.dynamics,
        in_node_nf=original_model.in_node_nf,
        n_dims=original_model.n_dims,
        timesteps=original_model.T,
        parametrization='eps',
        noise_schedule=noise_schedule,
        loss_type=loss_type,
        norm_values=original_model.norm_values,
        norm_biases=original_model.norm_biases,
        include_charges=original_model.include_charges,
        mlff_predictor=mlff_predictor,
        guidance_scale=guidance_scale,
        dataset_info=dataset_info
    )
    
    # Move guided model to same device as original model
    device = next(original_model.parameters()).device
    guided_model = guided_model.to(device)
    
    # Copy trained parameters
    guided_model.load_state_dict(original_model.state_dict(), strict=False)
    
    return guided_model


def enhanced_sampling_with_mlff(model, mlff_predictor, n_samples, n_nodes, node_mask, edge_mask, 
                              context, dataset_info, guidance_scale=1.0, fix_noise=False):
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
        fix_noise: Whether to fix noise for reproducibility
        
    Returns:
        x: Generated positions
        h: Generated node features
    """
    # Create guided model
    guided_model = create_mlff_guided_model(
        model, mlff_predictor, guidance_scale, dataset_info
    )
    
    # Generate samples with MLFF guidance
    return guided_model.sample_with_mlff_guidance(
        n_samples, n_nodes, node_mask, edge_mask, context, dataset_info, fix_noise
    ) 