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
        # No explicit position clamping; rely on correct integrator behavior.
        
        # Create DPM-compatible linear schedule for sampling (kept for fallback)
        # Note: By default we will use the original training noise schedule function
        # provided via `noise_schedule_fn` to stay consistent with the model.
        # The linear schedule is still created for debugging or fallback usage.
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
        
        print(f"DPM-Solver++: Initialized linear schedule (β₀={beta_0}, β₁={beta_1}); using model schedule for sampling")
    
    def noise_schedule_fn(self, t: torch.Tensor) -> torch.Tensor:
        """Noise schedule (gamma) used by the solver.

        For stability and consistency with the trained diffusion model, we
        defer to the original noise schedule function passed at construction
        time (i.e., the model's training schedule). This avoids mismatch
        between the solver schedule and the model, which can otherwise cause
        instabilities or divergence.
        """
        return self.original_noise_schedule_fn(t)
    
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
        # Ensure t is [B,1] for model and schedule
        B = x.size(0)
        t = self._as_column(t, B)
        
        # Project input to CoM-free subspace before model evaluation
        x = self._project_com(x, node_mask)

        # Get noise prediction from model at the provided time
        eps_pred = self.model_fn(x, t, node_mask, edge_mask, context)
        
        # Convert to data prediction: x_0 = (x_t - sigma_t * eps_pred) / alpha_t
        log_alpha_t = self.marginal_log_mean_coeff(t)
        alpha_t = torch.exp(log_alpha_t).to(x.dtype)
        sigma_t = self.marginal_std(t).to(x.dtype)
        
        # Handle scalar vs batch dimension
        if alpha_t.dim() == 0:
            alpha_t = alpha_t.unsqueeze(0)
            sigma_t = sigma_t.unsqueeze(0)
        
        # Reshape for broadcasting to match x dimensions
        while alpha_t.dim() < x.dim():
            alpha_t = alpha_t.unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1)
        
        x_0_pred = (x - sigma_t * eps_pred) / alpha_t
        
        # Project output to CoM-free subspace
        x_0_pred = self._project_com(x_0_pred, node_mask)

        return x_0_pred

    def _project_com(self, x: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        """Project to center-of-mass-free subspace on position channels.

        Expects first 3 feature channels to be positions; leaves features intact.
        """
        if x.dim() <= 2:
            return x
        n_dims = 3
        x_pos = x[:, :, :n_dims]
        x_feat = x[:, :, n_dims:]
        x_pos = diffusion_utils.remove_mean_with_mask(x_pos, node_mask)
        # Mask out padded nodes' features to prevent leakage
        if x_feat.numel() > 0:
            x_feat = x_feat * node_mask
        return torch.cat([x_pos, x_feat], dim=-1)

    @staticmethod
    def _phi_2(h: torch.Tensor) -> torch.Tensor:
        """Stable computation of phi2(h) = (e^h - 1 - h) / h^2."""
        eps = 1e-6
        return torch.where(
            h.abs() < eps,
            0.5 + h / 6.0 + (h * h) / 24.0,
            (torch.expm1(h) - h) / (h * h)
        )

    @staticmethod
    def _phi1_minus(h: torch.Tensor) -> torch.Tensor:
        """phi1(-h) = expm1(-h) (used by dpmsolver++ data-pred)."""
        return torch.expm1(-h)

    @staticmethod
    def _as_column(t: torch.Tensor, B: int) -> torch.Tensor:
        """Ensure t has shape [B,1]."""
        if t.dim() == 0:
            t = t.expand(B)
        if t.dim() == 1:
            t = t.unsqueeze(1)
        if t.size(0) != B:
            t = t.expand(B, 1)
        return t
    
    def _compute_step_coefficients(self, t_prev: float, t: float, device: torch.device):
        """Compute common coefficients for DPM-Solver++ steps.

        Returns:
            alpha_prev, alpha_t, sigma_prev, sigma_t, h
        """
        t_prev_t = torch.tensor([t_prev], device=device)
        t_t = torch.tensor([t], device=device)

        lambda_prev = self.marginal_lambda(t_prev_t)
        lambda_t = self.marginal_lambda(t_t)

        log_alpha_prev = self.marginal_log_mean_coeff(t_prev_t)
        log_alpha_t = self.marginal_log_mean_coeff(t_t)

        alpha_prev = torch.exp(log_alpha_prev)
        alpha_t = torch.exp(log_alpha_t)
        sigma_prev = self.marginal_std(t_prev_t)
        sigma_t = self.marginal_std(t_t)

        h = lambda_t - lambda_prev

        return alpha_prev, alpha_t, sigma_prev, sigma_t, h
    
    def multistep_dpm_solver_second_update(self, x: torch.Tensor, t_prev: float, t: float,
                                         node_mask: torch.Tensor, edge_mask: torch.Tensor,
                                         context: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Second-order DPM-Solver++ (2M) update in data-prediction form.

        Official form:
            x_t = (σ_t/σ_s) x_s - α_t φ1(-h) x0_s - 1/2 α_t φ1(-h) D1_0,
            with D1_0 = (1/r0)(x0_s - x0_{s-1}), r0 = h0/h
        """
        alpha_prev, alpha_t, sigma_prev, sigma_t, h = self._compute_step_coefficients(t_prev, t, x.device)
        lambda_prev = self.marginal_lambda(torch.tensor([t_prev], device=x.device))

        # Evaluate model at current state/time (t_prev)
        x0_s = self.data_prediction_fn(x, torch.tensor([t_prev], device=x.device),
                                       node_mask, edge_mask, context)

        # Ensure dtype matches x
        alpha_t = alpha_t.to(x.dtype)
        sigma_prev = sigma_prev.to(x.dtype)
        sigma_t = sigma_t.to(x.dtype)
        h = h.to(x.dtype)

        phi1 = DPMSolverPlusPlus._phi1_minus(h)

        if len(self.model_prev_list) == 0:
            # First step: dpmsolver++ 1st order
            x_t = (sigma_t / sigma_prev) * x - alpha_t * phi1 * x0_s
        else:
            # Second-order dpmsolver++ 2M
            x0_s_minus = self.model_prev_list[-1]
            t_s_minus = self.t_prev_list[-1]
            lambda_s_minus = self.marginal_lambda(torch.tensor([t_s_minus], device=x.device))
            h0 = lambda_prev - lambda_s_minus

            # If the previous step in lambda is tiny, degrade to first-order for this step
            if torch.any(h0.abs() < 1e-8):
                x_t = (sigma_t / sigma_prev) * x - alpha_t * phi1 * x0_s
            else:
                r0 = h0 / h
                D1_0 = (1.0 / r0) * (x0_s - x0_s_minus)
                x_t = (sigma_t / sigma_prev) * x \
                      - alpha_t * phi1 * x0_s \
                      - 0.5 * alpha_t * phi1 * D1_0

        # Project and mask after update to avoid drift/leak
        x_t = self._project_com(x_t, node_mask)

        return x_t, x0_s
    
    def multistep_dpm_solver_third_update(self, x: torch.Tensor, t_prev: float, t: float,
                                        node_mask: torch.Tensor, edge_mask: torch.Tensor,
                                        context: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Third-order dpmsolver++ is disabled for guided setups.

        Rationale: 3rd order in dpmsolver++ (data-pred) requires the exact
        official formulation with D1/D2 in λ-space. To avoid future footguns
        and because order=2 is recommended for guided/CoM-constrained models,
        we disable it here. Use order=2.
        """
        raise NotImplementedError("dpmsolver++ order=3 is disabled. Use order=2.")
    
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
        
        # Assume first 3 dims are positions for molecular data
        n_dims = 3 if len(x.shape) > 2 else 3

        # Ensure initial state is CoM-free
        x = self._project_com(x, node_mask)
        
        n_steps = len(timesteps) - 1
        for i in range(n_steps):
            t_prev = timesteps[i].item()
            t = timesteps[i + 1].item()
            
            # lower_order_final: degrade to 1st-order on the last step
            use_first_order = (self.order >= 2 and i == n_steps - 1)

            if self.order == 2 and use_first_order:
                # Force 1st order by clearing history for this step
                cache_backup = (list(self.model_prev_list), list(self.t_prev_list))
                self.model_prev_list = []
                self.t_prev_list = []
                x, x0_s = self.multistep_dpm_solver_second_update(
                    x, t_prev, t, node_mask, edge_mask, context)
                # Restore cache for completeness (not used further)
                self.model_prev_list, self.t_prev_list = cache_backup
            elif self.order == 2:
                x, x0_s = self.multistep_dpm_solver_second_update(
                    x, t_prev, t, node_mask, edge_mask, context)
            elif self.order == 3:
                x, x0_s = self.multistep_dpm_solver_third_update(
                    x, t_prev, t, node_mask, edge_mask, context)
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
                if len(self.model_prev_list) >= 1:
                    self.model_prev_list.pop(0)
                    self.t_prev_list.pop(0)
                # Cache x0 at current step time (t_prev) for multistep use
                self.model_prev_list.append(x0_s)
                self.t_prev_list.append(t_prev)

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


@torch.no_grad()
def inverse_lambda_gamma(lambda_targets: torch.Tensor, gamma_fn: Callable, iters: int = 40, device: Optional[torch.device] = None) -> torch.Tensor:
    """Invert lambda(t) = -0.5 * gamma(t) via bisection to solve for t.

    Args:
        lambda_targets: Tensor of target λ values
        gamma_fn: schedule function γ(t) expecting shape [B,1]
        iters: bisection iterations
        device: device for tensors
    Returns:
        t in [0,1] with same shape as lambda_targets
    """
    if device is None:
        device = lambda_targets.device
    t_lo = torch.zeros_like(lambda_targets, device=device)
    t_hi = torch.ones_like(lambda_targets, device=device)
    for _ in range(iters):
        t_mid = 0.5 * (t_lo + t_hi)
        lam_mid = -0.5 * gamma_fn(t_mid.unsqueeze(1)).squeeze(1)
        left = lam_mid > lambda_targets  # lam decreases with t when gamma increases
        t_lo = torch.where(left, t_mid, t_lo)
        t_hi = torch.where(left, t_hi, t_mid)
    return 0.5 * (t_lo + t_hi)


@torch.no_grad()
def get_time_steps_via_lambda(gamma_fn: Callable, num_steps: int, device: torch.device, grid: int = 4097) -> torch.Tensor:
    """
    Generate time steps by uniform spacing in lambda = log(alpha/sigma) = -0.5 * gamma(t).

    Uses an exact inverse of λ via bisection for stability.
    Returns t values descending from 1.0 to 0.0 with length num_steps + 1.
    """
    # Determine λ endpoints
    lam_1 = (-0.5 * gamma_fn(torch.tensor([[1.0]], device=device))).squeeze(1)[0]
    lam_0 = (-0.5 * gamma_fn(torch.tensor([[0.0]], device=device))).squeeze(1)[0]

    lam_targets = torch.linspace(lam_1, lam_0, num_steps + 1, device=device)
    t_vals = inverse_lambda_gamma(lam_targets, gamma_fn, device=device)
    return t_vals
