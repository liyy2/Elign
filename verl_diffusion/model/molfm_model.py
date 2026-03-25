from __future__ import annotations

import os
import sys
from typing import Dict, Optional

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EDM_SOURCE_ROOT = os.path.join(REPO_ROOT, "edm_source")
if EDM_SOURCE_ROOT not in sys.path:
    sys.path.insert(0, EDM_SOURCE_ROOT)

from equivariant_diffusion import utils as diffusion_utils

from .edm_model import EDMModel


class MolFMModel(EDMModel):
    """Fixed-step MolFM wrapper that exposes PPO-compatible transition log-probs."""

    def __init__(self, model, dequantizer, config, policy_config: Optional[Dict] = None):
        policy_config = dict(policy_config or {})
        super().__init__(model, config, backend="molfm")
        self.dequantizer = dequantizer
        self.policy_config = policy_config

        self.T = int(self.policy_config.get("time_step", getattr(config, "diffusion_steps", 1000)))
        self.policy_start_idx = int(
            self.policy_config.get(
                "policy_start_idx",
                self.policy_config.get("skip_prefix", 0) or 0,
            )
        )
        self.sde_noise_scale = float(self.policy_config.get("sde_noise_scale", 0.0) or 0.0)
        coord_scale = self.policy_config.get("sde_coordinate_noise_scale")
        feat_scale = self.policy_config.get("sde_feature_noise_scale")
        if coord_scale is None:
            coord_scale = self.sde_noise_scale
        if feat_scale is None:
            feat_scale = self.sde_noise_scale
        self.sde_coordinate_noise_scale = float(coord_scale or 0.0)
        self.sde_feature_noise_scale = float(feat_scale or 0.0)
        self.sde_min_sigma = float(self.policy_config.get("sde_min_sigma", 1e-6) or 1e-6)
        self.expose_z0_preds = False

    def _flatten_step(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            return t.view(1)
        if t.dim() == 1:
            return t
        return t.reshape(t.shape[0], -1)[:, 0]

    def _step_index_from_s(self, s: torch.Tensor) -> torch.Tensor:
        s_flat = self._flatten_step(s)
        s_step = torch.round(s_flat * float(self.T)).long()
        return (self.T - 1 - s_step).clamp(min=0, max=max(self.T - 1, 0))

    def _stochastic_step_mask(self, s: torch.Tensor) -> torch.Tensor:
        if self.sde_coordinate_noise_scale <= 0.0 and self.sde_feature_noise_scale <= 0.0:
            return torch.zeros_like(self._flatten_step(s), dtype=torch.bool)
        if self.policy_start_idx <= 0:
            return torch.ones_like(self._flatten_step(s), dtype=torch.bool)
        return self._step_index_from_s(s) >= self.policy_start_idx

    def _build_sigma(self, zt: torch.Tensor, dt: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
        batch = zt.size(0)
        sigma = zt.new_zeros(batch, 1, zt.size(-1))
        if batch == 0:
            return sigma

        sqrt_dt = torch.sqrt(torch.clamp(-dt.reshape(batch, 1, 1), min=0.0))
        if self.sde_coordinate_noise_scale > 0.0:
            sigma[:, :, : self.n_dims] = sqrt_dt * self.sde_coordinate_noise_scale
        if zt.size(-1) > self.n_dims and self.sde_feature_noise_scale > 0.0:
            sigma[:, :, self.n_dims :] = sqrt_dt * self.sde_feature_noise_scale

        sigma = sigma * active_mask.view(batch, 1, 1).to(dtype=zt.dtype, device=zt.device)
        return sigma

    def sample_p_xh_given_z0(
        self,
        z0,
        node_mask,
        edge_mask,
        context=None,
        fix_noise=False,
        prev_sample=None,
    ):
        del edge_mask, context, fix_noise, prev_sample

        x, h = self.model.sample_p_xh_given_z0(self.dequantizer, z0, node_mask)
        xh = torch.cat([x, h["categorical"].float(), h["integer"].float()], dim=2)
        mu = xh
        sigma = torch.zeros((z0.size(0), 1), device=z0.device, dtype=z0.dtype)
        log_p = torch.zeros(z0.size(0), device=z0.device, dtype=z0.dtype)
        zeros = torch.zeros((z0.size(0), 1), device=z0.device, dtype=z0.dtype)
        return x, h, mu, sigma, log_p, zeros, xh

    def sample_p_zs_given_zt(
        self,
        s,
        t,
        zt,
        node_mask,
        edge_mask,
        context=None,
        fix_noise=False,
        prev_sample=None,
    ):
        del fix_noise

        drift = self.model.flow_drift(t, zt, node_mask, edge_mask, context)
        dt = self._flatten_step(s - t)
        mu = zt + drift * dt.view(-1, 1, 1)
        active_mask = self._stochastic_step_mask(s)
        sigma = self._build_sigma(zt, dt, active_mask)

        if prev_sample is None:
            noise = self.sample_combined_position_feature_noise(
                zt.size(0),
                zt.size(1),
                node_mask,
            )
            zs = mu + sigma * noise
        else:
            zs = prev_sample

        zs = torch.cat(
            [
                diffusion_utils.remove_mean_with_mask(zs[:, :, : self.n_dims], node_mask),
                zs[:, :, self.n_dims :],
            ],
            dim=2,
        )

        log_p = torch.zeros(zt.size(0), device=zt.device, dtype=zt.dtype)
        if torch.any(active_mask):
            sigma_for_logp = torch.clamp(sigma[active_mask], min=self.sde_min_sigma)
            log_p_active = self.compute_log_p_zs_given_zt(
                zs[active_mask],
                mu[active_mask],
                sigma_for_logp,
                node_mask=node_mask[active_mask],
            )
            log_p[active_mask] = log_p_active

        z0_pred = mu
        return zs, log_p, mu, sigma, z0_pred
