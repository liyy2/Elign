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
from verl_diffusion.utils.math import policy_step_logprob


class MolFMModel(EDMModel):
    """Fixed-step MolFM wrapper that exposes PPO-compatible transition log-probs."""

    def __init__(self, model, dequantizer, config, policy_config: Optional[Dict] = None):
        policy_config = dict(policy_config or {})
        super().__init__(model, config, backend="molfm")
        self.dequantizer = dequantizer
        self.policy_config = policy_config

        self.T = int(self.policy_config.get("time_step", getattr(config, "diffusion_steps", 1000)))
        self.sde_mode = str(self.policy_config.get("sde_mode", "constant") or "constant").strip().lower()
        self.policy_start_idx = int(
            self.policy_config.get(
                "policy_start_idx",
                self.policy_config.get("skip_prefix", 0) or 0,
            )
        )
        window_size = self.policy_config.get("sde_window_size")
        if window_size is None:
            window_size = max(self.T - self.policy_start_idx, 0)
        self.sde_window_size = int(max(window_size, 0))
        self.sde_noise_scale = float(self.policy_config.get("sde_noise_scale", 0.0) or 0.0)
        coord_scale = self.policy_config.get("sde_coordinate_noise_scale")
        feat_scale = self.policy_config.get("sde_feature_noise_scale")
        if coord_scale is None:
            coord_scale = self.sde_noise_scale
        if feat_scale is None:
            feat_scale = self.sde_noise_scale
        self.sde_coordinate_noise_scale = float(coord_scale or 0.0)
        self.sde_feature_noise_scale = float(feat_scale or 0.0)
        if self.sde_mode == "sigma_corrected_coord_only":
            self.sde_feature_noise_scale = 0.0
        self.sde_min_sigma = float(self.policy_config.get("sde_min_sigma", 1e-6) or 1e-6)
        self.expose_z0_preds = False

    def _uses_sigma_corrected_coords(self) -> bool:
        return self.sde_mode in {"sigma_corrected_coord_only", "sigma_corrected_hb"}

    def _uses_sigma_corrected_features(self) -> bool:
        return self.sde_mode == "sigma_corrected_hb"

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
        if self.sde_window_size <= 0:
            return torch.zeros_like(self._flatten_step(s), dtype=torch.bool)

        step_index = self._step_index_from_s(s)
        window_start = max(self.policy_start_idx, 0)
        window_end = min(window_start + self.sde_window_size, self.T)
        if window_start <= 0 and window_end >= self.T:
            return torch.ones_like(step_index, dtype=torch.bool)
        return (step_index >= window_start) & (step_index < window_end)

    def _flow_path_sigma(self, t: torch.Tensor) -> torch.Tensor:
        sigma_t = self._flatten_step(t).reshape(-1, 1, 1)
        return torch.clamp(sigma_t, min=self.sde_min_sigma, max=1.0 - self.sde_min_sigma)

    def _build_sigma(
        self,
        zt: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch = zt.size(0)
        sigma = zt.new_zeros(batch, 1, zt.size(-1))
        if batch == 0:
            return sigma

        sqrt_dt = torch.sqrt(torch.clamp(-dt.reshape(batch, 1, 1), min=0.0))
        if self._uses_sigma_corrected_coords():
            if self.sde_coordinate_noise_scale > 0.0:
                sigma_t = self._flow_path_sigma(t)
                denom = torch.clamp(1.0 - sigma_t, min=self.sde_min_sigma)
                coord_std = self.sde_coordinate_noise_scale * torch.sqrt(sigma_t / denom)
                sigma[:, :, : self.n_dims] = sqrt_dt * coord_std
            if zt.size(-1) > self.n_dims and self._uses_sigma_corrected_features() and self.sde_feature_noise_scale > 0.0:
                sigma_t = self._flow_path_sigma(t)
                denom = torch.clamp(1.0 - sigma_t, min=self.sde_min_sigma)
                feat_std = self.sde_feature_noise_scale * torch.sqrt(sigma_t / denom)
                sigma[:, :, self.n_dims :] = sqrt_dt * feat_std
        else:
            if self.sde_coordinate_noise_scale > 0.0:
                sigma[:, :, : self.n_dims] = sqrt_dt * self.sde_coordinate_noise_scale
            if zt.size(-1) > self.n_dims and self.sde_feature_noise_scale > 0.0:
                sigma[:, :, self.n_dims :] = sqrt_dt * self.sde_feature_noise_scale

        sigma = sigma * active_mask.view(batch, 1, 1).to(dtype=zt.dtype, device=zt.device)
        return sigma

    def _build_transition_mean(
        self,
        zt: torch.Tensor,
        raw_velocity: torch.Tensor,
        drift: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dt_view = dt.view(-1, 1, 1)
        mu = zt + drift * dt_view
        z0_pred = mu

        if not (self._uses_sigma_corrected_coords() or self._uses_sigma_corrected_features()) or not torch.any(active_mask):
            return mu, z0_pred

        sigma_t = self._flow_path_sigma(t)
        active_view = active_mask.view(-1, 1, 1)
        denom = torch.clamp(1.0 - sigma_t, min=self.sde_min_sigma)

        if self._uses_sigma_corrected_coords():
            coord_t = zt[:, :, : self.n_dims]
            coord_velocity = raw_velocity[:, :, : self.n_dims]
            coord_drift = drift[:, :, : self.n_dims]
            x0_hat = coord_t - sigma_t * coord_velocity
            x1_hat = coord_t + (1.0 - sigma_t) * coord_velocity

            correction = 0.5 * (self.sde_coordinate_noise_scale ** 2) / denom
            corrected_coord_mu = coord_t + (coord_drift + correction * x1_hat) * dt_view
            mu = torch.cat(
                [
                    torch.where(active_view, corrected_coord_mu, mu[:, :, : self.n_dims]),
                    mu[:, :, self.n_dims :],
                ],
                dim=2,
            )
            z0_pred = torch.cat(
                [
                    torch.where(active_view, x0_hat, z0_pred[:, :, : self.n_dims]),
                    z0_pred[:, :, self.n_dims :],
                ],
                dim=2,
            )

        if zt.size(-1) > self.n_dims and self._uses_sigma_corrected_features() and self.sde_feature_noise_scale > 0.0:
            feat_t = zt[:, :, self.n_dims :]
            feat_velocity = raw_velocity[:, :, self.n_dims :]
            feat_drift = drift[:, :, self.n_dims :]
            h0_hat = feat_t - sigma_t * feat_velocity
            h1_hat = feat_t + (1.0 - sigma_t) * feat_velocity

            feature_correction = 0.5 * (self.sde_feature_noise_scale ** 2) / denom
            corrected_feat_mu = feat_t + (feat_drift + feature_correction * h1_hat) * dt_view
            mu = torch.cat(
                [
                    mu[:, :, : self.n_dims],
                    torch.where(active_view, corrected_feat_mu, mu[:, :, self.n_dims :]),
                ],
                dim=2,
            )
            z0_pred = torch.cat(
                [
                    z0_pred[:, :, : self.n_dims],
                    torch.where(active_view, h0_hat, z0_pred[:, :, self.n_dims :]),
                ],
                dim=2,
            )
        return mu, z0_pred

    def _transition_logprob_mask(
        self,
        zt: torch.Tensor,
        sigma: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        base_mask = node_mask.to(dtype=zt.dtype, device=zt.device)
        if base_mask.dim() < zt.dim():
            base_mask = base_mask.expand(-1, zt.size(1), zt.size(2))
        stochastic_dims = (sigma > 0).to(dtype=zt.dtype, device=zt.device)
        if stochastic_dims.size(1) == 1 and zt.size(1) != 1:
            stochastic_dims = stochastic_dims.expand(-1, zt.size(1), -1)
        return base_mask * stochastic_dims

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

        if hasattr(self.model, "raw_flow_velocity"):
            raw_velocity = self.model.raw_flow_velocity(t, zt, node_mask, edge_mask, context)
        else:
            raw_velocity = self.model.flow_drift(t, zt, node_mask, edge_mask, context)
        drift = self.model.flow_drift(t, zt, node_mask, edge_mask, context)
        dt = self._flatten_step(s - t)
        active_mask = self._stochastic_step_mask(s)
        mu, z0_pred = self._build_transition_mean(zt, raw_velocity, drift, t, dt, active_mask)
        sigma = self._build_sigma(zt, t, dt, active_mask)

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
            logprob_mask = self._transition_logprob_mask(
                zt[active_mask],
                sigma[active_mask],
                node_mask[active_mask],
            )
            log_p_active = policy_step_logprob(
                zs[active_mask],
                mu[active_mask],
                sigma_for_logp,
                logprob_mask,
            )
            log_p[active_mask] = log_p_active

        return zs, log_p, mu, sigma, z0_pred
