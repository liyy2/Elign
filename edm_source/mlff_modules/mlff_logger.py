"""
MLFF Logger Module
Centralized logging for MLFF guidance with optional wandb integration.
"""

import torch
import logging

# Set up logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] MLFF: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Wandb import with error handling
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
    
    def log_force_statistics(self, forces, node_mask=None):
        """Log detailed force statistics.
        
        Args:
            forces: Force tensor [batch_size, max_n_nodes, 3]
            node_mask: Valid node mask [batch_size, max_n_nodes, 1] or None
        """
        force_magnitudes = torch.norm(forces, dim=-1)
        
        if node_mask is not None:
            # Only count forces for valid atoms
            mask = node_mask.squeeze(-1).bool() if node_mask.dim() == 3 else node_mask.bool()
            valid_force_magnitudes = force_magnitudes[mask]
            zero_force_count = (valid_force_magnitudes == 0).sum().item()
            nonzero_force_count = (valid_force_magnitudes > 0).sum().item()
            # Use only valid forces for statistics
            force_magnitudes_for_stats = valid_force_magnitudes
        else:
            # Fallback to old behavior if no mask provided
            zero_force_count = (force_magnitudes == 0).sum().item()
            nonzero_force_count = (force_magnitudes > 0).sum().item()
            force_magnitudes_for_stats = force_magnitudes.flatten()
        
        # Compute statistics only on valid forces
        if force_magnitudes_for_stats.numel() > 0:
            stats = {
                'mean_magnitude': force_magnitudes_for_stats.mean().item(),
                'max_magnitude': force_magnitudes_for_stats.max().item(),
                'min_magnitude': force_magnitudes_for_stats.min().item(),
                'std_magnitude': force_magnitudes_for_stats.std().item() if force_magnitudes_for_stats.numel() > 1 else 0.0,
                'zero_forces': zero_force_count,
                'nonzero_forces': nonzero_force_count
            }
        else:
            # No valid forces
            stats = {
                'mean_magnitude': 0.0,
                'max_magnitude': 0.0,
                'min_magnitude': 0.0,
                'std_magnitude': 0.0,
                'zero_forces': 0,
                'nonzero_forces': 0
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
                      f"3) Invalid molecular configuration, 4) Atoms too far apart (>6Å)")
        
        # Get position range without printing the full tensor
        pos_min = batch.pos.min().item()
        pos_max = batch.pos.max().item()
        logger.warning(f"Batch positions range: [{pos_min:.6f}, {pos_max:.6f}]")
        
        if hasattr(batch, 'atomic_numbers'):
            # Only show first 10 atomic numbers to avoid excessive output
            atom_nums = batch.atomic_numbers.tolist()
            if len(atom_nums) > 10:
                logger.warning(f"Atomic numbers (first 10): {atom_nums[:10]}...")
            else:
                logger.warning(f"Atomic numbers: {atom_nums}")
    
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