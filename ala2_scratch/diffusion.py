from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np
import torch
from torch.nn import functional as F

from .model import Ala2EGNNDenoiser


def _remove_mean_with_mask(x: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    """Enforce zero center-of-mass over valid nodes."""
    denom = node_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    mean = (x * node_mask).sum(dim=1, keepdim=True) / denom
    return (x - mean) * node_mask


def _centered_gaussian_like(x: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    """Sample Gaussian noise in the zero center-of-mass subspace."""
    noise = torch.randn_like(x)
    return _remove_mean_with_mask(noise, node_mask)


def _centered_gaussian_log_prob(
    sample: torch.Tensor,
    mean: torch.Tensor,
    log_var: torch.Tensor,
    node_mask: torch.Tensor,
) -> torch.Tensor:
    """Log-density of the centered isotropic Gaussian used by the reverse sampler."""
    diff = _remove_mean_with_mask(sample - mean, node_mask)
    var = torch.exp(log_var).reshape(sample.shape[0])
    sq_mahal = (diff.square() * node_mask).sum(dim=(1, 2)) / var
    valid_nodes = node_mask.squeeze(-1).sum(dim=1).to(sample.dtype)
    dof = (valid_nodes - 1.0).clamp(min=0.0) * sample.shape[-1]
    return -0.5 * (sq_mahal + dof * (log_var.reshape(sample.shape[0]) + math.log(2.0 * math.pi)))


def _first_tensor(mapping: Mapping[str, torch.Tensor], keys: tuple[str, ...]) -> Optional[torch.Tensor]:
    for key in keys:
        value = mapping.get(key)
        if torch.is_tensor(value):
            return value
    return None


def clip_noise_schedule(alphas2: np.ndarray, clip_value: float = 0.001) -> np.ndarray:
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)
    alphas_step = np.clip(alphas2[1:] / alphas2[:-1], a_min=clip_value, a_max=1.0)
    return np.cumprod(alphas_step, axis=0)


def polynomial_schedule(timesteps: int, s: float = 1e-4, power: float = 3.0) -> np.ndarray:
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas2 = (1 - np.power(x / steps, power)) ** 2
    alphas2 = clip_noise_schedule(alphas2, clip_value=0.001)
    precision = 1 - 2 * s
    return precision * alphas2 + s


def cosine_alpha_schedule(timesteps: int, s: float = 0.008, raise_to_power: float = 1.0) -> np.ndarray:
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = np.clip(1 - (alphas_cumprod[1:] / alphas_cumprod[:-1]), a_min=0.0, a_max=0.999)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    if raise_to_power != 1.0:
        alphas_cumprod = np.power(alphas_cumprod, raise_to_power)
    return alphas_cumprod


class PredefinedNoiseSchedule(torch.nn.Module):
    """Lookup-based gamma schedule copied from the original EDM implementation."""

    def __init__(self, noise_schedule: str, timesteps: int, precision: float) -> None:
        super().__init__()
        self.timesteps = int(timesteps)
        if noise_schedule == "cosine":
            alphas2 = cosine_alpha_schedule(timesteps)
        elif str(noise_schedule).startswith("polynomial"):
            splits = str(noise_schedule).split("_")
            if len(splits) != 2:
                raise ValueError(f"Invalid polynomial noise schedule '{noise_schedule}'")
            alphas2 = polynomial_schedule(timesteps, s=precision, power=float(splits[1]))
        else:
            raise ValueError(f"Unsupported noise schedule '{noise_schedule}'")

        sigmas2 = 1.0 - alphas2
        log_alphas2 = np.log(alphas2)
        log_sigmas2 = np.log(sigmas2)
        gamma = - (log_alphas2 - log_sigmas2)
        self.register_buffer("gamma", torch.from_numpy(gamma).float(), persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_int = torch.round(t * self.timesteps).long().clamp_(min=0, max=self.timesteps)
        gamma = self.gamma
        if gamma.device != t_int.device:
            gamma = gamma.to(device=t_int.device)
        return gamma[t_int]


@dataclass
class DiffusionSample:
    """Outputs from reverse diffusion sampling."""

    x0: torch.Tensor
    x_t: Optional[torch.Tensor]
    timesteps: torch.Tensor
    log_prob_sum: Optional[torch.Tensor] = None
    log_probs: Optional[torch.Tensor] = None
    chain: Optional[torch.Tensor] = None
    x0_preds: Optional[torch.Tensor] = None


@dataclass
class DiffusionRollout:
    """Detached reverse-diffusion trajectory for replay-style RL updates."""

    positions: torch.Tensor
    z_chain: torch.Tensor
    timesteps: torch.Tensor


class CoordinateDiffusion:
    """Coordinate-only diffusion wrapper aligned with the original EDM formulas."""

    def __init__(
        self,
        *,
        model: Optional[Ala2EGNNDenoiser] = None,
        denoiser: Optional[Ala2EGNNDenoiser] = None,
        n_nodes: int = 22,
        num_nodes: Optional[int] = None,
        timesteps: int = 100,
        num_steps: Optional[int] = None,
        num_timesteps: Optional[int] = None,
        noise_schedule: str = "cosine",
        beta_schedule: Optional[str] = None,
        noise_precision: float = 1e-4,
        coord_scale: float = 1.0,
        norm_value: Optional[float] = None,
        device: Optional[str] = None,
        **_: Any,
    ) -> None:
        self.model = model or denoiser
        self.device = device
        if num_nodes is not None:
            n_nodes = num_nodes
        if num_steps is not None:
            timesteps = num_steps
        if num_timesteps is not None:
            timesteps = num_timesteps
        if beta_schedule is not None:
            noise_schedule = beta_schedule
        if norm_value is not None:
            coord_scale = norm_value
        self.n_nodes = int(n_nodes)
        self.timesteps = int(timesteps)
        self.coord_scale = float(coord_scale)
        if self.coord_scale <= 0:
            raise ValueError("coord_scale must be > 0")
        self.gamma = PredefinedNoiseSchedule(
            noise_schedule=str(noise_schedule),
            timesteps=self.timesteps,
            precision=float(noise_precision),
        )

    def _full_node_mask(self, batch_size: int, x: torch.Tensor | None = None, device=None, dtype=None) -> torch.Tensor:
        if x is not None:
            device = x.device
            dtype = x.dtype
        if device is None or dtype is None:
            raise ValueError("Provide either `x` or explicit `device` and `dtype`.")
        return torch.ones(batch_size, self.n_nodes, 1, device=device, dtype=dtype)

    def _inflate_batch_array(self, array: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_shape = (array.size(0),) + (1,) * (target.dim() - 1)
        return array.view(target_shape)

    def sigma(self, gamma: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self._inflate_batch_array(torch.sqrt(torch.sigmoid(gamma)), target)

    def alpha(self, gamma: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self._inflate_batch_array(torch.sqrt(torch.sigmoid(-gamma)), target)

    def snr(self, gamma: torch.Tensor) -> torch.Tensor:
        return torch.exp(-gamma)

    def normalize_positions(self, x: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        return _remove_mean_with_mask(x / self.coord_scale, node_mask)

    def unnormalize_positions(self, z: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        return _remove_mean_with_mask(z * self.coord_scale, node_mask)

    def sample_position_noise(self, batch_size: int, device: torch.device, node_mask: torch.Tensor) -> torch.Tensor:
        reference = torch.zeros(batch_size, self.n_nodes, 3, device=device, dtype=node_mask.dtype)
        return _centered_gaussian_like(reference, node_mask)

    def q_sample(
        self,
        z0: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        noise: Optional[torch.Tensor] = None,
        node_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if z0.dim() != 3 or z0.shape[1:] != (self.n_nodes, 3):
            raise ValueError(
                f"Expected z0 with shape [batch, {self.n_nodes}, 3], got {tuple(z0.shape)}"
            )
        batch_size = z0.shape[0]
        if node_mask is None:
            node_mask = self._full_node_mask(batch_size, x=z0)
        if noise is None:
            noise = self.sample_position_noise(batch_size, z0.device, node_mask)
        t = timesteps.to(device=z0.device, dtype=torch.float32).view(batch_size, 1) / self.timesteps
        gamma_t = self._inflate_batch_array(self.gamma(t), z0)
        alpha_t = self.alpha(gamma_t, z0)
        sigma_t = self.sigma(gamma_t, z0)
        z_t = alpha_t * z0 + sigma_t * noise
        z_t = _remove_mean_with_mask(z_t, node_mask)
        return z_t, noise

    def sigma_and_alpha_t_given_s(
        self,
        gamma_t: torch.Tensor,
        gamma_s: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sigma2_t_given_s = self._inflate_batch_array(
            -torch.expm1(F.softplus(gamma_s) - F.softplus(gamma_t)),
            target,
        )
        log_alpha2_t = F.logsigmoid(-gamma_t)
        log_alpha2_s = F.logsigmoid(-gamma_s)
        alpha_t_given_s = torch.exp(0.5 * (log_alpha2_t - log_alpha2_s))
        alpha_t_given_s = self._inflate_batch_array(alpha_t_given_s, target)
        sigma_t_given_s = torch.sqrt(sigma2_t_given_s)
        return sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s

    def compute_x_pred(
        self,
        eps: torch.Tensor,
        z_t: torch.Tensor,
        gamma_t: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        sigma_t = self.sigma(gamma_t, z_t)
        alpha_t = self.alpha(gamma_t, z_t)
        return _remove_mean_with_mask((z_t - sigma_t * eps) / alpha_t, node_mask)

    def training_loss(
        self,
        model: Optional[Ala2EGNNDenoiser] = None,
        x0: Optional[torch.Tensor] = None,
        node_features: Optional[torch.Tensor] = None,
        *,
        batch: Optional[Mapping[str, torch.Tensor]] = None,
        coordinates: Optional[torch.Tensor] = None,
        static_features: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        node_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        model = model or self.model
        if model is None:
            raise ValueError("CoordinateDiffusion.training_loss requires a model.")
        if batch is not None:
            x0 = _first_tensor(batch, ("positions", "coordinates", "x"))
            node_features = _first_tensor(
                batch,
                ("node_features", "static_features", "features", "atom_features"),
            )
            node_mask = _first_tensor(batch, ("node_mask", "mask", "atom_mask"))
        elif coordinates is not None:
            x0 = coordinates
            node_features = static_features if static_features is not None else node_features
        if x0 is None or node_features is None:
            raise ValueError("CoordinateDiffusion.training_loss requires coordinates and node features.")

        batch_size = x0.shape[0]
        if node_mask is None:
            node_mask = self._full_node_mask(batch_size, x=x0)
        elif node_mask.dim() == 2:
            node_mask = node_mask.unsqueeze(-1)
        node_mask = node_mask.to(device=x0.device, dtype=x0.dtype)

        z0 = self.normalize_positions(x0, node_mask)
        if timesteps is None:
            timesteps = torch.randint(
                low=0,
                high=self.timesteps + 1,
                size=(batch_size,),
                device=x0.device,
            )
        z_t, eps = self.q_sample(z0, timesteps, noise=noise, node_mask=node_mask)
        t = timesteps.to(device=x0.device, dtype=torch.float32) / self.timesteps
        eps_pred = model(z_t, t, node_features, node_mask=node_mask)
        coord_count = (node_mask.sum() * z0.shape[-1]).clamp(min=1.0)
        mse = (((eps_pred - eps) ** 2) * node_mask).sum() / coord_count
        return {"loss": 0.5 * mse, "eps_mse": mse.detach(), "t_mean": t.mean().detach()}

    def p_mean_variance(
        self,
        model: Ala2EGNNDenoiser,
        z_t: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor,
        node_features: torch.Tensor,
        *,
        node_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if node_mask is None:
            node_mask = self._full_node_mask(z_t.shape[0], x=z_t)
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)
        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = self.sigma_and_alpha_t_given_s(
            gamma_t,
            gamma_s,
            z_t,
        )
        sigma_s = self.sigma(gamma_s, z_t)
        sigma_t = self.sigma(gamma_t, z_t)
        eps_t = model(z_t, t.view(z_t.shape[0]), node_features, node_mask=node_mask)
        mu = z_t / alpha_t_given_s - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_t
        mu = _remove_mean_with_mask(mu, node_mask)
        sigma = sigma_t_given_s * sigma_s / sigma_t
        x0_pred = self.compute_x_pred(eps_t, z_t, self._inflate_batch_array(gamma_t, z_t), node_mask)
        return mu, sigma, x0_pred, eps_t

    def sample_p_x_given_z0(
        self,
        model: Ala2EGNNDenoiser,
        z0: torch.Tensor,
        node_features: torch.Tensor,
        *,
        node_mask: torch.Tensor,
        fix_noise: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu_x, log_var = self.p_x_given_z0_params(
            model,
            z0,
            node_features,
            node_mask=node_mask,
        )
        sigma_x = torch.exp(0.5 * log_var).view(z0.size(0), 1, 1)
        noise_batch = 1 if fix_noise else z0.shape[0]
        noise_mask = node_mask[:1] if fix_noise else node_mask
        noise = self.sample_position_noise(noise_batch, z0.device, noise_mask)
        if fix_noise:
            noise = noise.expand(z0.shape[0], -1, -1)
        x = mu_x + sigma_x * noise
        x = self.unnormalize_positions(x, node_mask)
        return x, mu_x, log_var

    def p_x_given_z0_params(
        self,
        model: Ala2EGNNDenoiser,
        z0: torch.Tensor,
        node_features: torch.Tensor,
        *,
        node_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        zeros = torch.zeros(size=(z0.size(0), 1), device=z0.device, dtype=z0.dtype)
        gamma_0 = self.gamma(zeros)
        sigma_x = self.snr(-0.5 * gamma_0).unsqueeze(1)
        eps_0 = model(z0, zeros.view(z0.shape[0]), node_features, node_mask=node_mask)
        mu_x = self.compute_x_pred(eps_0, z0, self._inflate_batch_array(gamma_0, z0), node_mask)
        log_var = torch.log((sigma_x.squeeze(1) ** 2).clamp(min=1e-20))
        return mu_x, log_var

    def sample_rollout(
        self,
        model: Optional[Ala2EGNNDenoiser] = None,
        *,
        batch_size: int,
        node_features: Optional[torch.Tensor] = None,
        static_features: Optional[torch.Tensor] = None,
        device: Optional[torch.device | str] = None,
        num_nodes: Optional[int] = None,
        n_nodes: Optional[int] = None,
        node_mask: Optional[torch.Tensor] = None,
        initial_noise: Optional[torch.Tensor] = None,
        move_to_cpu: bool = False,
    ) -> DiffusionRollout:
        model = model or self.model
        if model is None:
            raise ValueError("CoordinateDiffusion.sample_rollout requires a model.")
        if node_features is None:
            node_features = static_features
        if node_features is None:
            raise ValueError("CoordinateDiffusion.sample_rollout requires node_features/static_features.")
        if num_nodes is not None and int(num_nodes) != self.n_nodes:
            raise ValueError(f"Expected num_nodes={self.n_nodes}, got {num_nodes}")
        if n_nodes is not None and int(n_nodes) != self.n_nodes:
            raise ValueError(f"Expected n_nodes={self.n_nodes}, got {n_nodes}")

        if device is None:
            if torch.is_tensor(node_features):
                device = node_features.device
            elif self.device is not None:
                device = self.device
            else:
                device = "cpu"
        device = torch.device(device)
        if node_mask is None:
            reference = torch.zeros(batch_size, self.n_nodes, 3, device=device)
            node_mask = self._full_node_mask(batch_size, x=reference)
        else:
            if node_mask.dim() == 2:
                node_mask = node_mask.unsqueeze(-1)
            node_mask = node_mask.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            if initial_noise is None:
                z_t = self.sample_position_noise(batch_size, device, node_mask)
            else:
                z_t = initial_noise.to(device=device, dtype=torch.float32)
                if z_t.shape != (batch_size, self.n_nodes, 3):
                    raise ValueError(
                        f"Expected initial_noise shape {(batch_size, self.n_nodes, 3)}, got {tuple(z_t.shape)}"
                    )
                z_t = _remove_mean_with_mask(z_t, node_mask)

            z_history = [z_t.detach()]
            for s_int in reversed(range(0, self.timesteps)):
                s = torch.full((batch_size, 1), fill_value=s_int, device=device, dtype=torch.float32) / self.timesteps
                t = s + (1.0 / self.timesteps)
                mu, sigma, _x0_pred, _eps_t = self.p_mean_variance(
                    model,
                    z_t,
                    s,
                    t,
                    node_features,
                    node_mask=node_mask,
                )
                noise = self.sample_position_noise(batch_size, device, node_mask)
                z_prev = mu + sigma * noise
                z_prev = _remove_mean_with_mask(z_prev, node_mask)
                z_history.append(z_prev.detach())
                z_t = z_prev

            positions, _mu_x, _log_var_x = self.sample_p_x_given_z0(
                model,
                z_t,
                node_features,
                node_mask=node_mask,
            )

        z_chain = torch.stack(z_history, dim=1)
        timestep_tensor = torch.arange(
            self.timesteps,
            -1,
            -1,
            device=device,
            dtype=torch.long,
        ).view(1, self.timesteps + 1).expand(batch_size, -1)
        if move_to_cpu:
            z_chain = z_chain.cpu()
            positions = positions.detach().cpu()
            timestep_tensor = timestep_tensor.cpu()
        else:
            positions = positions.detach()
        return DiffusionRollout(
            positions=positions,
            z_chain=z_chain,
            timesteps=timestep_tensor,
        )

    def replay_log_probs(
        self,
        model: Optional[Ala2EGNNDenoiser] = None,
        *,
        rollout: Optional[DiffusionRollout] = None,
        positions: Optional[torch.Tensor] = None,
        z_chain: Optional[torch.Tensor] = None,
        node_features: Optional[torch.Tensor] = None,
        static_features: Optional[torch.Tensor] = None,
        node_mask: Optional[torch.Tensor] = None,
        device: Optional[torch.device | str] = None,
        skip_prefix: int = 0,
        tail_steps: Optional[int] = None,
    ) -> dict[str, torch.Tensor]:
        model = model or self.model
        if model is None:
            raise ValueError("CoordinateDiffusion.replay_log_probs requires a model.")
        if rollout is not None:
            positions = rollout.positions
            z_chain = rollout.z_chain
        if node_features is None:
            node_features = static_features
        if positions is None or z_chain is None or node_features is None:
            raise ValueError("Replay requires rollout/positions, z_chain, and node_features.")

        if device is None:
            if torch.is_tensor(node_features):
                device = node_features.device
            elif self.device is not None:
                device = self.device
            else:
                device = "cpu"
        device = torch.device(device)
        positions = positions.to(device=device, dtype=torch.float32)
        z_chain = z_chain.to(device=device, dtype=torch.float32)
        batch_size = int(z_chain.shape[0])
        if z_chain.shape[1] != self.timesteps + 1:
            raise ValueError(
                f"Expected z_chain length {self.timesteps + 1}, got {tuple(z_chain.shape)}"
            )
        if node_mask is None:
            node_mask = self._full_node_mask(batch_size, x=positions)
        elif node_mask.dim() == 2:
            node_mask = node_mask.unsqueeze(-1)
        node_mask = node_mask.to(device=device, dtype=positions.dtype)

        log_prob_steps = []
        for idx in range(self.timesteps):
            t_int = self.timesteps - idx
            s_int = t_int - 1
            s = torch.full((batch_size, 1), fill_value=s_int, device=device, dtype=torch.float32) / self.timesteps
            t = torch.full((batch_size, 1), fill_value=t_int, device=device, dtype=torch.float32) / self.timesteps
            z_t = z_chain[:, idx]
            z_prev = z_chain[:, idx + 1]
            mu, sigma, _x0_pred, _eps_t = self.p_mean_variance(
                model,
                z_t,
                s,
                t,
                node_features,
                node_mask=node_mask,
            )
            log_var = torch.log((sigma.squeeze(1) ** 2).clamp(min=1e-20))
            log_prob_steps.append(
                _centered_gaussian_log_prob(
                    z_prev.detach(),
                    mu,
                    log_var,
                    node_mask,
                )
            )

        z0 = z_chain[:, -1]
        mu_x, log_var_x = self.p_x_given_z0_params(
            model,
            z0,
            node_features,
            node_mask=node_mask,
        )
        x_normalized = self.normalize_positions(positions, node_mask)
        log_prob_steps.append(
            _centered_gaussian_log_prob(
                x_normalized.detach(),
                mu_x,
                log_var_x,
                node_mask,
            )
        )

        trajectory_log_probs = torch.stack(log_prob_steps, dim=1)
        if skip_prefix > 0:
            trajectory_log_probs = trajectory_log_probs[:, min(skip_prefix, trajectory_log_probs.shape[1]) :]
        if tail_steps is not None:
            tail_steps = max(1, int(tail_steps))
            trajectory_log_probs = trajectory_log_probs[:, -tail_steps:]
        return {
            "trajectory_log_probs": trajectory_log_probs,
            "log_prob_mean": trajectory_log_probs.mean(dim=1),
            "log_prob_sum": trajectory_log_probs.sum(dim=1),
        }


    def sample(
        self,
        model: Optional[Ala2EGNNDenoiser] = None,
        *,
        batch_size: int,
        node_features: Optional[torch.Tensor] = None,
        static_features: Optional[torch.Tensor] = None,
        device: Optional[torch.device | str] = None,
        num_nodes: Optional[int] = None,
        n_nodes: Optional[int] = None,
        node_mask: Optional[torch.Tensor] = None,
        initial_noise: Optional[torch.Tensor] = None,
        return_chain: bool = False,
        return_log_probs: bool = False,
        return_logprobs: Optional[bool] = None,
        return_intermediates: Optional[bool] = None,
        return_x0_preds: bool = False,
    ) -> DiffusionSample:
        model = model or self.model
        if model is None:
            raise ValueError("CoordinateDiffusion.sample requires a model.")
        if node_features is None:
            node_features = static_features
        if node_features is None:
            raise ValueError("CoordinateDiffusion.sample requires node_features/static_features.")
        if return_logprobs is not None:
            return_log_probs = bool(return_logprobs)
        if return_intermediates:
            return_chain = True
            return_x0_preds = True
        if num_nodes is not None and int(num_nodes) != self.n_nodes:
            raise ValueError(f"Expected num_nodes={self.n_nodes}, got {num_nodes}")
        if n_nodes is not None and int(n_nodes) != self.n_nodes:
            raise ValueError(f"Expected n_nodes={self.n_nodes}, got {n_nodes}")

        if device is None:
            if torch.is_tensor(node_features):
                device = node_features.device
            elif self.device is not None:
                device = self.device
            else:
                device = "cpu"
        device = torch.device(device)
        if node_mask is None:
            reference = torch.zeros(batch_size, self.n_nodes, 3, device=device)
            node_mask = self._full_node_mask(batch_size, x=reference)
        else:
            if node_mask.dim() == 2:
                node_mask = node_mask.unsqueeze(-1)
            node_mask = node_mask.to(device=device, dtype=torch.float32)

        if initial_noise is None:
            z_t = self.sample_position_noise(batch_size, device, node_mask)
        else:
            z_t = initial_noise.to(device=device, dtype=torch.float32)
            if z_t.shape != (batch_size, self.n_nodes, 3):
                raise ValueError(
                    f"Expected initial_noise shape {(batch_size, self.n_nodes, 3)}, got {tuple(z_t.shape)}"
                )
            z_t = _remove_mean_with_mask(z_t, node_mask)
        z_init = z_t.clone()

        chain_steps = [self.unnormalize_positions(z_t, node_mask).clone()] if return_chain else None
        x0_preds = [] if return_x0_preds else None
        log_probs = [] if return_log_probs else None
        timestep_records = []

        for s_int in reversed(range(0, self.timesteps)):
            s = torch.full((batch_size, 1), fill_value=s_int, device=device, dtype=torch.float32) / self.timesteps
            t = s + (1.0 / self.timesteps)
            timestep_records.append((t.view(batch_size) * self.timesteps).round().long())

            mu, sigma, x0_pred, _eps_t = self.p_mean_variance(
                model,
                z_t,
                s,
                t,
                node_features,
                node_mask=node_mask,
            )
            if return_x0_preds:
                x0_preds.append(self.unnormalize_positions(x0_pred, node_mask).clone())

            noise = self.sample_position_noise(batch_size, device, node_mask)
            z_prev = mu + sigma * noise
            z_prev = _remove_mean_with_mask(z_prev, node_mask)

            if return_log_probs:
                log_var = torch.log((sigma.squeeze(1) ** 2).clamp(min=1e-20))
                log_prob = _centered_gaussian_log_prob(
                    z_prev.detach(),
                    mu,
                    log_var,
                    node_mask,
                )
                log_probs.append(log_prob)

            z_t = z_prev
            if return_chain:
                chain_steps.append(self.unnormalize_positions(z_t, node_mask).clone())

        x, mu_x, log_var_x = self.sample_p_x_given_z0(
            model,
            z_t,
            node_features,
            node_mask=node_mask,
        )
        if return_log_probs:
            x_normalized = self.normalize_positions(x, node_mask)
            log_probs.append(
                _centered_gaussian_log_prob(
                    x_normalized.detach(),
                    mu_x,
                    log_var_x,
                    node_mask,
                )
            )
        if return_chain:
            chain_steps.append(x.clone())
        if return_x0_preds:
            x0_preds.append(x.clone())
        timestep_records.append(torch.zeros(batch_size, device=device, dtype=torch.long))

        log_prob_tensor = None
        log_prob_sum = None
        if return_log_probs:
            log_prob_tensor = torch.stack(log_probs, dim=1)
            log_prob_sum = log_prob_tensor.sum(dim=1)

        chain_tensor = None
        if return_chain:
            chain_tensor = torch.stack(chain_steps, dim=1)

        x0_pred_tensor = None
        if return_x0_preds:
            x0_pred_tensor = torch.stack(x0_preds, dim=1)

        timestep_tensor = torch.stack(timestep_records, dim=1)
        return DiffusionSample(
            x0=x,
            x_t=self.unnormalize_positions(z_init, node_mask),
            timesteps=timestep_tensor,
            log_prob_sum=log_prob_sum,
            log_probs=log_prob_tensor,
            chain=chain_tensor,
            x0_preds=x0_pred_tensor,
        )

    def sample_with_logprobs(
        self,
        *,
        batch_size: int,
        node_features: Optional[torch.Tensor] = None,
        static_features: Optional[torch.Tensor] = None,
        num_nodes: Optional[int] = None,
        n_nodes: Optional[int] = None,
        device: Optional[torch.device | str] = None,
        model: Optional[Ala2EGNNDenoiser] = None,
        return_intermediates: bool = False,
        return_logprobs: bool = True,
    ) -> dict[str, torch.Tensor]:
        sample = self.sample(
            model=model,
            batch_size=batch_size,
            node_features=node_features,
            static_features=static_features,
            num_nodes=num_nodes,
            n_nodes=n_nodes,
            device=device,
            return_chain=bool(return_intermediates),
            return_log_probs=bool(return_logprobs),
            return_x0_preds=bool(return_intermediates),
        )
        return {
            "positions": sample.x0,
            "log_probs": sample.log_prob_sum
            if sample.log_prob_sum is not None
            else torch.zeros(batch_size, device=sample.x0.device, dtype=sample.x0.dtype),
            "trajectory_log_probs": sample.log_probs,
            "chain": sample.chain,
            "timesteps": sample.timesteps,
        }


def build_diffusion(
    config: Mapping[str, Any],
    *,
    metadata: Optional[Mapping[str, Any]] = None,
    example_batch: Optional[Mapping[str, torch.Tensor]] = None,
    model: Optional[Ala2EGNNDenoiser] = None,
    denoiser: Optional[Ala2EGNNDenoiser] = None,
) -> CoordinateDiffusion:
    diff_cfg = dict(config.get("diffusion", {}))
    n_nodes = diff_cfg.get("n_nodes") or diff_cfg.get("num_nodes")
    if n_nodes is None and example_batch is not None:
        for key in ("positions", "coordinates", "x"):
            value = example_batch.get(key)
            if torch.is_tensor(value):
                n_nodes = int(value.shape[-2])
                break
    if n_nodes is None and metadata is not None and metadata.get("atomic_numbers") is not None:
        n_nodes = len(metadata["atomic_numbers"])

    return CoordinateDiffusion(
        model=model or denoiser,
        denoiser=denoiser,
        n_nodes=int(n_nodes or 22),
        num_steps=int(diff_cfg.get("num_steps", diff_cfg.get("num_timesteps", 100))),
        noise_schedule=str(diff_cfg.get("noise_schedule", diff_cfg.get("beta_schedule", "cosine"))),
        noise_precision=float(diff_cfg.get("noise_precision", 1e-4)),
        coord_scale=float(diff_cfg.get("coord_scale", diff_cfg.get("norm_value", 1.0))),
        device=config.get("device"),
    )


create_diffusion = build_diffusion
