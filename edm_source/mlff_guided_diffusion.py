"""
MLFF-Guided Diffusion
Compatibility wrapper that provides the old API using the new modular implementation.
"""

import torch
import logging
from typing import Optional, Dict, Any

# Import all components from the mlff_modules package
from mlff_modules import (
    MLFFLogger,
    MLFFForceComputer,
    MLFFGuidedDiffusion,
    get_mlff_predictor,
    remove_mean_with_constraint,
    apply_force_clipping
)

# Set up logger
logger = logging.getLogger(__name__)

# Re-export main classes for backward compatibility
__all__ = [
    'MLFFGuidedDiffusion',
    'MLFFLogger',
    'MLFFForceComputer',
    'get_mlff_predictor',
    'remove_mean_with_constraint',
    'apply_force_clipping',
    'create_mlff_guided_model',
    'enhanced_sampling_with_mlff'
]


def create_mlff_guided_model(
    base_diffusion,
    mlff_predictor,
    guidance_scale=1.0,
    dataset_info=None,
    guidance_iterations=1,
    noise_threshold=0.8,
    force_clip_threshold=None,
    displacement_clip=None,
    position_scale=None,
    use_wandb=False,
    device='cuda'
):
    """
    Create an MLFF-guided diffusion model (compatibility function).
    
    This wraps the base diffusion model with MLFF guidance capabilities.
    position_scale: If None, will be extracted from base_diffusion.norm_values[0]
    """
    # Extract position scale from model's normalize_factors if not provided
    if position_scale is None:
        if hasattr(base_diffusion, 'norm_values'):
            position_scale = float(base_diffusion.norm_values[0])
        else:
            position_scale = 1.0  # Default to 1.0 if not found
            print(f"Warning: Could not find norm_values in model, using position_scale={position_scale}")
    
    # Create the guided diffusion instance
    guided_model = MLFFGuidedDiffusion(
        base_diffusion=base_diffusion,
        mlff_model=None,  # We already have the predictor
        mlff_predictor=mlff_predictor,
        guidance_scale=guidance_scale,
        guidance_iterations=guidance_iterations,
        noise_threshold=noise_threshold,
        force_clip_threshold=force_clip_threshold,
        displacement_clip=displacement_clip,
        position_scale=position_scale,
        use_wandb=use_wandb,
        device=device
    )
    
    
    # Store dataset_info for use during sampling
    guided_model.dataset_info = dataset_info
    
    return guided_model


def enhanced_sampling_with_mlff(
    flow,
    mlff_predictor,
    batch_size,
    max_n_nodes,
    node_mask,
    edge_mask,
    context,
    dataset_info,
    guidance_scale=1.0,
    guidance_iterations=1,
    noise_threshold=0.8,
    force_clip_threshold=None,
    displacement_clip=None,
    fix_noise=False,
    position_scale=None,
    use_wandb=False
):
    """
    Enhanced sampling with MLFF guidance (compatibility function).
    
    This function samples molecules using the base diffusion model with optional MLFF guidance.
    """
    device = node_mask.device
    
    # Check if sampling_method attribute exists, if not default to ddpm
    if not hasattr(flow, 'sampling_method'):
        flow.sampling_method = 'ddpm'
    
    # Extract position scale from model if not provided
    if position_scale is None:
        if hasattr(flow, 'norm_values'):
            position_scale = float(flow.norm_values[0])
        else:
            position_scale = 1.0
    
    # Create guided model if predictor is available and guidance is enabled
    if mlff_predictor is not None and guidance_scale > 0:
        guided_model = create_mlff_guided_model(
            flow,
            mlff_predictor,
            guidance_scale=guidance_scale,
            dataset_info=dataset_info,
            guidance_iterations=guidance_iterations,
            noise_threshold=noise_threshold,
            force_clip_threshold=force_clip_threshold,
            displacement_clip=displacement_clip,
            position_scale=position_scale,
            use_wandb=use_wandb,
            device=device
        )
        
        # Use the guided model's sample method
        if flow.sampling_method == 'dpm_solver++':
            # DPM-Solver++ sampling with MLFF guidance by providing a guided
            # epsilon-prediction function to the solver.
            from equivariant_diffusion.dpm_solver import DPMSolverPlusPlus

            base = guided_model.base_diffusion

            # Build guided model function from the class for reuse and clarity
            guided_model_fn = guided_model.build_dpm_guided_model_fn(dataset_info)

            # Install guided solver on the base diffusion model and sample
            base.dpm_solver = DPMSolverPlusPlus(
                model_fn=guided_model_fn,
                noise_schedule_fn=base.gamma,
                order=base.dpm_solver_order,
                timesteps=base.T
            )

            # Call the standard sample method (solver is used internally)
            x, h = base.sample(
                batch_size,
                max_n_nodes,
                node_mask=node_mask,
                edge_mask=edge_mask,
                context=context,
                fix_noise=fix_noise,
            )
        else:
            # Standard DDPM sampling with guidance - use the wrapped sample method
            x, h = guided_model.sample(
                batch_size,
                max_n_nodes,
                node_mask=node_mask,
                edge_mask=edge_mask,
                context=context,
                fix_noise=fix_noise,
                dataset_info=dataset_info
            )
    else:
        # No guidance, use base sampler
        if flow.sampling_method == 'dpm_solver++':
            # Use the base model's DPM-Solver++ path
            x, h = flow.sample(
                batch_size,
                max_n_nodes,
                node_mask=node_mask,
                edge_mask=edge_mask,
                context=context,
                fix_noise=fix_noise,
            )
        else:
            x, h = flow.sample(
                batch_size,
                max_n_nodes,
                node_mask=node_mask,
                edge_mask=edge_mask,
                context=context,
                fix_noise=fix_noise
            )
    
    return x, h
