"""
MLFF Utility Functions
Helper functions for MLFF-guided diffusion.
"""

import torch
import logging
from typing import Optional, Tuple

# Set up logger
logger = logging.getLogger(__name__)


def get_mlff_predictor(mlff_model: str = 'uma-s-1p1', device: str = 'cuda'):
    """
    Load and initialize the MLFF predictor.
    
    Args:
        mlff_model: Name of the MLFF model to use (without 'p' suffix)
        device: Device to load the model on
        
    Returns:
        Initialized MLFF predictor
    """
    try:
        from fairchem.core import pretrained_mlip
        
        # Map model names to the new API format
        if mlff_model == 'uma-s-1p1':
            model_name = 'uma-s-1'
        else:
            model_name = mlff_model
        
        # Use the new pretrained_mlip API
        mlff_predictor = pretrained_mlip.get_predict_unit(model_name, device=device)
        logger.info(f"Successfully loaded MLFF predictor: {model_name}")
        
        return mlff_predictor
        
    except Exception as e:
        logger.error(f"Failed to load MLFF predictor: {e}")
        logger.warning("Continuing without MLFF guidance")
        return None


def remove_mean_with_constraint(x: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    """
    Remove center of mass from positions while respecting node mask.
    
    Args:
        x: Positions tensor [batch_size, n_nodes, 3]
        node_mask: Valid node mask [batch_size, n_nodes, 1]
        
    Returns:
        Centered positions tensor
    """
    # Compute center of mass for each molecule
    node_mask_expanded = node_mask.expand_as(x)
    masked_x = x * node_mask_expanded
    
    # Sum positions for valid atoms
    sum_x = masked_x.sum(dim=1, keepdim=True)  # [batch_size, 1, 3]
    
    # Count valid atoms
    n_valid = node_mask.sum(dim=1, keepdim=True)  # [batch_size, 1, 1]
    n_valid = n_valid.clamp(min=1)  # Avoid division by zero
    
    # Compute center of mass
    com = sum_x / n_valid  # [batch_size, 1, 3]
    
    # Remove center of mass
    x_centered = x - com
    
    # Apply mask to ensure invalid atoms remain at zero
    x_centered = x_centered * node_mask_expanded
    
    return x_centered


def apply_force_clipping(forces: torch.Tensor, threshold: float, 
                        node_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, int]:
    """
    Clip force magnitudes that exceed threshold.
    
    Args:
        forces: Force tensor [batch_size, n_nodes, 3]
        threshold: Maximum allowed force magnitude
        node_mask: Optional mask for valid nodes
        
    Returns:
        Tuple of (clipped forces, number of clipped forces)
    """
    force_magnitudes = torch.norm(forces, dim=-1, keepdim=True)  # [batch_size, n_nodes, 1]
    
    # Find forces that exceed threshold
    exceed_mask = force_magnitudes > threshold
    
    if node_mask is not None:
        # Only count valid atoms
        exceed_mask = exceed_mask & (node_mask > 0)
    
    n_clipped = exceed_mask.sum().item()
    
    if n_clipped > 0:
        # Scale down forces that exceed threshold
        scale_factors = torch.where(
            exceed_mask,
            threshold / (force_magnitudes + 1e-10),
            torch.ones_like(force_magnitudes)
        )
        forces_clipped = forces * scale_factors
        
        logger.info(f"Clipped {n_clipped} forces exceeding threshold {threshold}")
        return forces_clipped, n_clipped
    
    return forces, 0