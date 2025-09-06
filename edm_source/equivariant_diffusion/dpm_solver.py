"""
DPM-Solver++ implementation for fast sampling of diffusion models.

Based on "DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models"
https://arxiv.org/abs/2211.01095

This implementation provides a fast ODE solver for diffusion model sampling
that can achieve high-quality results in 10-20 steps instead of 1000 steps.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Union, List, Optional, Callable
from equivariant_diffusion import utils as diffusion_utils


class DPMSolverPlusPlus:
    """
    DPM-Solver++ for fast sampling of diffusion models.
    
    This solver can generate high-quality samples in 10-20 steps compared to 
    1000 steps required by DDPM sampling, while maintaining compatibility
    with molecular generation constraints.
    """
    
    def __init__(self, 
                 model_fn: Callable,
                 noise_schedule_fn: Callable,
                 order: int = 2,
                 timesteps: int = 1000):
        """
        Initialize DPM-Solver++.
        
        Args:
            model_fn: The neural network model (epsilon prediction)
            noise_schedule_fn: Original noise schedule function used during training
            order: Order of the solver (2 for guided, 3 for unconditional)
            timesteps: Number of timesteps for creating DPM-compatible linear schedule
        """
        self.model_fn = model_fn
        self.original_noise_schedule_fn = noise_schedule_fn
        self.order = order
        
        # Create DPM-compatible linear schedule for sampling
        self._create_dpm_linear_schedule(timesteps)
        
        # Cache for intermediate results
        self.model_prev_list = []
        self.t_prev_list = []
    
    def _create_dpm_linear_schedule(self, timesteps: int):
        """Create DPM-compatible linear noise schedule following official DPM-Solver implementation."""
        import numpy as np
        
        # Use official DPM-Solver linear schedule parameters
        beta_0 = 0.1  # continuous_beta_0 from official implementation
        beta_1 = 20.0  # continuous_beta_1 from official implementation
        
        # Create time grid
        t = np.linspace(1e-3, 1.0, timesteps)  # t_0 = 1e-3, T = 1.0 as in official implementation
        
        # Compute log_alpha_t using official formula
        log_alpha_t = -0.25 * t**2 * (beta_1 - beta_0) - 0.5 * t * beta_0
        
        # Compute sigma_t = sqrt(1 - alpha_t^2)
        alpha_t_squared = np.exp(2 * log_alpha_t)
        sigma_t_squared = 1.0 - alpha_t_squared
        
        # Ensure numerical stability
        sigma_t_squared = np.maximum(sigma_t_squared, 1e-8)
        log_sigma_t = 0.5 * np.log(sigma_t_squared)
        
        # Compute lambda = log(alpha_t / sigma_t)
        lambda_t = log_alpha_t - log_sigma_t

        # Convert to base-model gamma parameterization: gamma_pre = -2 * lambda
        # This matches EnVariationalDiffusion where:
        #   alpha = sqrt(sigmoid(-gamma)), sigma = sqrt(sigmoid(gamma))
        gamma_pre = -2.0 * lambda_t

        # Create gamma lookup table for DPM sampling (in base-model gamma parameterization)
        self.dpm_gamma_schedule = torch.from_numpy(gamma_pre).float()
        self.timesteps = timesteps
        
        print(f"DPM-Solver++: Using official linear schedule (β₀={beta_0}, β₁={beta_1})")
    
    def noise_schedule_fn(self, t: torch.Tensor) -> torch.Tensor:
        """DPM-compatible linear noise schedule in base-model gamma parameterization."""
        t_int = torch.round(t * self.timesteps).long()
        t_int = torch.clamp(t_int, 0, self.timesteps - 1)
        
        # Move gamma schedule to same device as t
        if self.dpm_gamma_schedule.device != t.device:
            self.dpm_gamma_schedule = self.dpm_gamma_schedule.to(t.device)
        
        return self.dpm_gamma_schedule[t_int]
    
    def marginal_log_mean_coeff(self, t: torch.Tensor) -> torch.Tensor:
        """Compute log mean coefficient alpha_t using DPM linear schedule (base gamma)."""
        gamma_t = self.noise_schedule_fn(t)
        return -0.5 * F.softplus(gamma_t)  # log(alpha_t) = -0.5 * softplus(gamma_t)
    
    def marginal_std(self, t: torch.Tensor) -> torch.Tensor:
        """Compute standard deviation sigma_t using DPM linear schedule (base gamma)."""
        gamma_t = self.noise_schedule_fn(t)
        return torch.sqrt(torch.sigmoid(gamma_t))  # sigma_t = sqrt(sigmoid(gamma_t))
    
    def marginal_lambda(self, t: torch.Tensor) -> torch.Tensor:
        """Compute lambda_t = log(alpha_t/sigma_t) using DPM linear schedule.
        With base gamma parameterization: lambda = -0.5 * gamma.
        """
        gamma_t = self.noise_schedule_fn(t)
        return -0.5 * gamma_t
    
    def data_prediction_fn(self, x: torch.Tensor, t: torch.Tensor,
                          node_mask: torch.Tensor, edge_mask: torch.Tensor,
                          context: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Convert epsilon prediction to data prediction.
        
        For DPM-Solver++, we use data prediction model instead of noise prediction.
        """
        # Ensure t is properly shaped for the model
        if t.dim() == 0:  # scalar
            t = t.unsqueeze(0).repeat(x.size(0), 1)
        elif t.dim() == 1 and t.size(0) == 1:  # single timestep
            t = t.repeat(x.size(0), 1)
        
        # Get noise prediction from model
        eps_pred = self.model_fn(x, t, node_mask, edge_mask, context)
        
        # Convert to data prediction: x_0 = (x_t - sigma_t * eps_pred) / alpha_t
        log_alpha_t = self.marginal_log_mean_coeff(t.squeeze())
        alpha_t = torch.exp(log_alpha_t)
        sigma_t = self.marginal_std(t.squeeze())
        
        # Handle scalar vs batch dimension
        if alpha_t.dim() == 0:
            alpha_t = alpha_t.unsqueeze(0)
            sigma_t = sigma_t.unsqueeze(0)
        
        # Reshape for broadcasting to match x dimensions
        while alpha_t.dim() < x.dim():
            alpha_t = alpha_t.unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1)
        
        x_0_pred = (x - sigma_t * eps_pred) / alpha_t
        
        # Ensure COM constraint is satisfied for molecular data
        if x_0_pred.dim() > 2:  # Handle molecular data
            n_dims = 3  # First 3 dims are positions
            x_pos = x_0_pred[:, :, :n_dims]
            x_features = x_0_pred[:, :, n_dims:]
            
            # Remove COM drift from predicted clean positions
            x_pos = diffusion_utils.remove_mean_with_mask(x_pos, node_mask)
            x_0_pred = torch.cat([x_pos, x_features], dim=-1)
        
        return x_0_pred
    
    def _compute_step_coefficients(self, t_prev: float, t: float, device: torch.device):
        """Compute common coefficients for DPM-Solver++ steps."""
        lambda_prev = self.marginal_lambda(torch.tensor([t_prev], device=device))
        lambda_t = self.marginal_lambda(torch.tensor([t], device=device))
        
        log_alpha_prev = self.marginal_log_mean_coeff(torch.tensor([t_prev], device=device))
        log_alpha_t = self.marginal_log_mean_coeff(torch.tensor([t], device=device))
        
        alpha_prev = torch.exp(log_alpha_prev)
        alpha_t = torch.exp(log_alpha_t)
        h = lambda_t - lambda_prev
        
        return alpha_prev, alpha_t, h
    
    def multistep_dpm_solver_second_update(self, x: torch.Tensor, t_prev: float, t: float,
                                         node_mask: torch.Tensor, edge_mask: torch.Tensor,
                                         context: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Second order multistep update."""
        alpha_prev, alpha_t, h = self._compute_step_coefficients(t_prev, t, x.device)
        
        # Get current model prediction
        model_t = self.data_prediction_fn(x, torch.tensor([t], device=x.device), 
                                        node_mask, edge_mask, context)
        
        if len(self.model_prev_list) == 0:
            # First step
            x_t = (alpha_t / alpha_prev) * x - (alpha_t * torch.expm1(h)) * model_t
        else:
            # Second order update
            model_prev = self.model_prev_list[-1]
            D1_t = 0.5 * (model_t - model_prev)
            x_t = (alpha_t / alpha_prev) * x - (alpha_t * torch.expm1(h)) * model_t - \
                  (alpha_t * torch.expm1(h) * h) * D1_t
        
        return x_t, model_t
    
    def multistep_dpm_solver_third_update(self, x: torch.Tensor, t_prev: float, t: float,
                                        node_mask: torch.Tensor, edge_mask: torch.Tensor,
                                        context: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Third order multistep update."""
        alpha_prev, alpha_t, h = self._compute_step_coefficients(t_prev, t, x.device)
        
        # Get current model prediction
        model_t = self.data_prediction_fn(x, torch.tensor([t], device=x.device),
                                        node_mask, edge_mask, context)
        
        if len(self.model_prev_list) == 0:
            # First step
            x_t = (alpha_t / alpha_prev) * x - (alpha_t * torch.expm1(h)) * model_t
        elif len(self.model_prev_list) == 1:
            # Second step
            model_prev = self.model_prev_list[-1]
            D1_t = 0.5 * (model_t - model_prev)
            x_t = (alpha_t / alpha_prev) * x - (alpha_t * torch.expm1(h)) * model_t - \
                  (alpha_t * torch.expm1(h) * h) * D1_t
        else:
            # Third order update
            model_prev_1 = self.model_prev_list[-1]
            model_prev_2 = self.model_prev_list[-2]
            
            # Compute h_1 for second-order differences
            t_prev_1 = self.t_prev_list[-1]
            t_prev_2 = self.t_prev_list[-2]
            lambda_prev_1 = self.marginal_lambda(torch.tensor([t_prev_1], device=x.device))
            lambda_prev_2 = self.marginal_lambda(torch.tensor([t_prev_2], device=x.device))
            h_1 = lambda_prev_1 - lambda_prev_2
            
            D1_t = 0.5 * (model_t - model_prev_1)
            D1_t_1 = 0.5 * (model_prev_1 - model_prev_2)
            D2_t = (D1_t - D1_t_1) / h_1
            
            expm1_h = torch.expm1(h)
            x_t = (alpha_t / alpha_prev) * x - (alpha_t * expm1_h) * model_t - \
                  (alpha_t * expm1_h * h) * D1_t - \
                  (alpha_t * expm1_h * h**2 / 2.) * D2_t
        
        return x_t, model_t
    
    def sample(self, 
               x: torch.Tensor,
               timesteps: torch.Tensor,
               node_mask: torch.Tensor,
               edge_mask: torch.Tensor, 
               context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Sample using DPM-Solver++.
        
        Args:
            x: Initial noise tensor
            timesteps: Timesteps for sampling
            node_mask: Mask for valid nodes/atoms
            edge_mask: Mask for edges
            context: Optional context for conditional generation
            
        Returns:
            Denoised sample
        """
        # Clear cache
        self.model_prev_list = []
        self.t_prev_list = []
        
        n_dims = x.size(2) if len(x.shape) > 2 else 3  # Assume first 3 dims are positions
        
        for i in range(len(timesteps) - 1):
            t_prev = timesteps[i].item()
            t = timesteps[i + 1].item()
            
            if self.order == 2:
                x, model_current = self.multistep_dpm_solver_second_update(x, t_prev, t, 
                                                                         node_mask, edge_mask, context)
            elif self.order == 3:
                x, model_current = self.multistep_dpm_solver_third_update(x, t_prev, t,
                                                                        node_mask, edge_mask, context)
            else:
                raise ValueError(f"Order {self.order} not supported")
            
            # Preserve molecular constraints - remove center of mass drift
            if len(x.shape) > 2:  # Handle molecular data
                x_pos = x[:, :, :n_dims]
                x_features = x[:, :, n_dims:]
                x_pos = diffusion_utils.remove_mean_with_mask(x_pos, node_mask)
                x = torch.cat([x_pos, x_features], dim=-1)
            
            # Store for next iteration - reuse the already computed model prediction
            if i < len(timesteps) - 2:  # Don't store on the last iteration
                if len(self.model_prev_list) >= self.order - 1:
                    self.model_prev_list.pop(0)
                    self.t_prev_list.pop(0)
                
                # Store the already computed prediction (no redundant computation!)
                self.model_prev_list.append(model_current)
                self.t_prev_list.append(t)
        
        return x


def get_time_steps_for_dpm_solver(num_steps: int, device: torch.device) -> torch.Tensor:
    """
    Generate uniform time steps for DPM-Solver++.
    
    Args:
        num_steps: Number of sampling steps
        device: Device to create tensor on
        
    Returns:
        Time steps tensor from 1.0 to 0.0
    """
    return torch.linspace(1.0, 0.0, num_steps + 1, device=device)