"""
MLFF Guided Diffusion Core Module
Main class implementing MLFF-guided diffusion sampling.
"""

import torch
import logging
from typing import Optional, Dict

# Import from sibling modules
from .mlff_logger import MLFFLogger
from .mlff_force_computer import MLFFForceComputer
from .mlff_utils import get_mlff_predictor, remove_mean_with_constraint, apply_force_clipping

# Set up logger
logger = logging.getLogger(__name__)


class MLFFGuidedDiffusion:
    """
    MLFF-guided diffusion sampler that extends base diffusion with physics-based guidance.
    This class wraps around the base diffusion model and adds MLFF force guidance.
    """
    
    def __init__(
        self,
        base_diffusion,
        mlff_model='uma-s-1p1',
        guidance_scale=1.0,
        guidance_iterations=1,
        noise_threshold=0.8,
        force_clip_threshold=None,
        position_scale=None,
        use_wandb=False,
        device='cuda'
    ):
        """
        Initialize MLFF-guided diffusion sampler.
        
        Args:
            base_diffusion: Base EnVariationalDiffusion model
            mlff_model: Name of the MLFF model to use (e.g., 'uma-s-1p1')
            guidance_scale: Scale factor for MLFF guidance (0.0 = no guidance)
            guidance_iterations: Number of guidance iterations per timestep
            noise_threshold: Skip guidance when noise level > threshold (for efficiency)
            force_clip_threshold: Maximum force magnitude allowed (None = no clipping)
            position_scale: Scale factor to convert from normalized to physical positions
            use_wandb: Whether to use Weights & Biases for logging
            device: Device to run computations on
        """
        self.base_diffusion = base_diffusion
        self.guidance_scale = guidance_scale
        self.guidance_iterations = guidance_iterations
        self.noise_threshold = noise_threshold
        self.force_clip_threshold = force_clip_threshold
        
        # Extract position scale from model's normalize_factors if not provided
        if position_scale is None:
            if hasattr(base_diffusion, 'norm_values'):
                position_scale = float(base_diffusion.norm_values[0])
            else:
                position_scale = 1.0  # Default to 1.0 if not found
                logger.info(f"Could not find norm_values in model, using position_scale={position_scale}")
        
        self.position_scale = position_scale
        self.device = device
        
        # Initialize components
        self.logger = MLFFLogger(use_wandb=use_wandb)
        self.mlff_predictor = get_mlff_predictor(mlff_model, device) if guidance_scale > 0 else None
        
        if self.mlff_predictor is not None:
            self.force_computer = MLFFForceComputer(
                mlff_predictor=self.mlff_predictor,
                position_scale=position_scale,
                device=device
            )
        else:
            self.force_computer = None
            if guidance_scale > 0:
                logger.warning("MLFF predictor not loaded, running without guidance")
    
    def apply_mlff_guidance(
        self,
        z: torch.Tensor,
        node_mask: torch.Tensor,
        noise_level: float,
        dataset_info: Dict,
        t: Optional[int] = None
    ) -> torch.Tensor:
        """
        Apply MLFF force guidance to molecular positions.
        
        Args:
            z: Current molecular state [batch_size, max_n_nodes, n_dims + n_features]
            node_mask: Valid node mask [batch_size, max_n_nodes, 1]
            noise_level: Current noise level in the diffusion process
            dataset_info: Dataset information including atom decoder
            t: Current timestep (optional, for logging)
            
        Returns:
            Guided molecular state
        """
        # Skip guidance if disabled or noise level too high
        if self.force_computer is None or self.guidance_scale == 0:
            return z
        
        if noise_level > self.noise_threshold:
            logger.debug(f"Skipping guidance at t={t}, noise_level={noise_level:.4f} > threshold={self.noise_threshold}")
            return z
        
        z_original = z.clone()
        
        # Apply guidance iterations
        for iteration in range(self.guidance_iterations):
            # Compute MLFF forces (just returns forces tensor)
            forces = self.force_computer.compute_mlff_forces(z, node_mask, dataset_info)
            
            # Logger computes statistics and logs them (only for valid atoms)
            force_stats = self.logger.log_force_statistics(forces, node_mask)
            
            # Skip if all forces are zero
            if force_stats['nonzero_forces'] == 0:
                logger.warning(f"All forces are zero at t={t}, iteration {iteration + 1}")
                break
            
            # Apply force clipping if enabled
            if self.force_clip_threshold is not None:
                forces, _ = apply_force_clipping(forces, self.force_clip_threshold, node_mask)
            
            # Scale forces based on guidance scale and noise level
            base_scale = self.guidance_scale / self.guidance_iterations
            
            # Adaptive scaling based on noise level (less guidance when noisier)
            effective_noise_level = 1.0 - noise_level  # Convert to clarity level
            force_scale = base_scale * effective_noise_level
            
            # Apply guidance to positions
            position_guidance = forces * force_scale
            z[:, :, :3] = z[:, :, :3] + position_guidance  # Move in direction of forces
            
            # Maintain center of mass constraint
            z[:, :, :3] = remove_mean_with_constraint(z[:, :, :3], node_mask)
            
            # Log iteration statistics
            guidance_magnitudes = torch.norm(position_guidance, dim=-1)
            guidance_stats = {
                'base_scale': base_scale,
                'effective_noise_level': effective_noise_level,
                'guidance_mean': guidance_magnitudes.mean().item(),
                'guidance_max': guidance_magnitudes.max().item(),
                'guidance_std': guidance_magnitudes.std().item()
            }
            
            force_stats_summary = {
                'mean': force_stats['mean_magnitude'],
                'max': force_stats['max_magnitude'],
                'std': force_stats['std_magnitude']
            }
            
            self.logger.log_guidance_iteration(iteration, force_stats_summary, guidance_stats)
        
        # Log summary
        self.logger.log_guidance_summary(z_original, z, self.guidance_iterations, noise_level)
        
        return z
    
    def sample(self, *args, **kwargs):
        """
        Sample molecules with MLFF guidance.
        Wraps the base diffusion sampler and adds guidance.
        """
        # Check if we need to add guidance
        if self.force_computer is None or self.guidance_scale == 0:
            # No guidance, just use base sampler (remove dataset_info if present)
            kwargs.pop('dataset_info', None)
            return self.base_diffusion.sample(*args, **kwargs)
        
        # Extract dataset_info and remove from kwargs since base sampler doesn't accept it
        dataset_info = kwargs.pop('dataset_info', None)
        if dataset_info is None:
            logger.warning("No dataset_info provided, running without MLFF guidance")
            return self.base_diffusion.sample(*args, **kwargs)
        
        # Override the sample function to add guidance
        original_sample_fn = self.base_diffusion.sample_p_zs_given_zt
        
        # Use closure to capture dataset_info for the guided function
        def guided_sample_fn(s, t, zt, node_mask, edge_mask, context, **inner_kwargs):
            # Get base sample - pass through all kwargs
            zs = original_sample_fn(s, t, zt, node_mask, edge_mask, context, **inner_kwargs)
            
            # Calculate noise level based on actual noise schedule (sigma_t)
            # Get gamma_t from the noise schedule
            gamma_t = self.base_diffusion.gamma(t)
            # Compute sigma_t which represents the actual noise level
            # Note: sigma_t = sqrt(sigmoid(gamma_t)) represents the noise standard deviation
            sigma_t = torch.sqrt(torch.sigmoid(gamma_t))
            # Take mean if batched, otherwise get single value
            noise_level = sigma_t.mean().item() if sigma_t.numel() > 1 else sigma_t.item()
            
            # Apply MLFF guidance (dataset_info is captured from outer scope)
            zs = self.apply_mlff_guidance(zs, node_mask, noise_level, dataset_info, t.item() if t.numel() == 1 else None)
            
            return zs
        
        # Temporarily replace the sampling function
        self.base_diffusion.sample_p_zs_given_zt = guided_sample_fn
        
        try:
            # Run sampling with guidance (dataset_info removed from kwargs)
            result = self.base_diffusion.sample(*args, **kwargs)
        finally:
            # Restore original function
            self.base_diffusion.sample_p_zs_given_zt = original_sample_fn
        
        return result
    
    def sample_chain(self, *args, **kwargs):
        """
        Sample molecule chain (with intermediate states) with MLFF guidance.
        """
        # Check if we need to add guidance
        if self.force_computer is None or self.guidance_scale == 0:
            # No guidance, just use base sampler (remove dataset_info if present)
            kwargs.pop('dataset_info', None)
            return self.base_diffusion.sample_chain(*args, **kwargs)
        
        # Extract dataset_info and remove from kwargs since base sampler doesn't accept it
        dataset_info = kwargs.pop('dataset_info', None)
        if dataset_info is None:
            logger.warning("No dataset_info provided, running without MLFF guidance")
            return self.base_diffusion.sample_chain(*args, **kwargs)
        
        # Override the sample function to add guidance
        original_sample_fn = self.base_diffusion.sample_p_zs_given_zt
        
        # Use closure to capture dataset_info for the guided function
        def guided_sample_fn(s, t, zt, node_mask, edge_mask, context, **inner_kwargs):
            # Get base sample - pass through all kwargs
            zs = original_sample_fn(s, t, zt, node_mask, edge_mask, context, **inner_kwargs)
            
            # Calculate noise level based on actual noise schedule (sigma_t)
            # Get gamma_t from the noise schedule
            gamma_t = self.base_diffusion.gamma(t)
            # Compute sigma_t which represents the actual noise level
            # Note: sigma_t = sqrt(sigmoid(gamma_t)) represents the noise standard deviation
            sigma_t = torch.sqrt(torch.sigmoid(gamma_t))
            # Take mean if batched, otherwise get single value
            noise_level = sigma_t.mean().item() if sigma_t.numel() > 1 else sigma_t.item()
            
            # Apply MLFF guidance (dataset_info is captured from outer scope)
            zs = self.apply_mlff_guidance(zs, node_mask, noise_level, dataset_info, t.item() if t.numel() == 1 else None)
            
            return zs
        
        # Temporarily replace the sampling function
        self.base_diffusion.sample_p_zs_given_zt = guided_sample_fn
        
        try:
            # Run sampling with guidance (dataset_info removed from kwargs)
            result = self.base_diffusion.sample_chain(*args, **kwargs)
        finally:
            # Restore original function
            self.base_diffusion.sample_p_zs_given_zt = original_sample_fn
        
        return result