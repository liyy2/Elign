from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class RewardSchedule:
    """Indices and interval metadata for reward shaping evaluation."""

    indices: torch.Tensor
    timesteps: torch.Tensor
    intervals: torch.Tensor


class RewardScheduler:
    """Select diffusion steps for reward computation using uniform or adaptive stride."""

    # The scheduler encapsulates stride logic so reward modules can switch between
    # skip-only, fixed-stride, or adaptive policies without reimplementing book-keeping.

    def __init__(
        self,
        mode: str = "uniform",
        skip_prefix: int = 0,
        uniform_stride: int = 1,
        adaptive_config: Optional[Dict] = None,
        include_terminal: bool = True,
    ):
        self.mode = mode.lower()
        if self.mode not in {"uniform", "adaptive"}:
            raise ValueError(f"Unsupported reward scheduler mode: {mode}")

        self.skip_prefix = max(0, int(skip_prefix))
        self.uniform_stride = max(1, int(uniform_stride))
        self.include_terminal = include_terminal

        adaptive_config = adaptive_config or {}
        self.coarse_stride = max(1, int(adaptive_config.get("coarse_stride", 10)))
        self.fine_stride = max(1, int(adaptive_config.get("fine_stride", 2)))
        self.threshold_fraction = float(adaptive_config.get("threshold_fraction", 0.25))
        self.threshold_timestep = adaptive_config.get("threshold_timestep")

    def build(
        self,
        timesteps: torch.Tensor,
        start_idx: int = 0,
        end_idx: Optional[int] = None,
    ) -> RewardSchedule:
        """
        Build reward schedule given diffusion timesteps.

        Args:
            timesteps: 1D tensor of diffusion times ordered as in the rollout.
            start_idx: Earliest index to consider (e.g., horizon truncation).
            end_idx: Last index to consider (default: final diffusion step).

        Returns:
            RewardSchedule containing selected indices, their raw diffusion times,
            and the number of original steps skipped between entries. The intervals
            make it possible to weight shaped rewards when stride > 1.
        """
        if timesteps.dim() != 1:
            raise ValueError("RewardScheduler expects a 1D timesteps tensor.")

        total_steps = timesteps.size(0)
        start_idx = max(0, int(start_idx))
        end_idx = total_steps - 1 if end_idx is None else min(int(end_idx), total_steps - 1)
        if start_idx > end_idx:
            start_idx = end_idx

        effective_start = min(max(start_idx, self.skip_prefix), end_idx)

        indices = []
        current_idx = effective_start
        T_max = torch.max(timesteps).item() if total_steps > 0 else 0.0

        while current_idx <= end_idx:
            indices.append(current_idx)
            stride = self._stride_for_timestep(timesteps[current_idx].item(), T_max)
            current_idx += stride

        if not indices:
            indices = [end_idx]
        elif self.include_terminal and indices[-1] != end_idx:
            indices.append(end_idx)

        indices = torch.tensor(sorted(set(indices)), dtype=torch.long)
        timesteps_sel = timesteps[indices]

        if indices.numel() > 1:
            deltas = indices[1:] - indices[:-1]
            deltas = torch.cat([deltas, deltas.new_ones(1)], dim=0)
        else:
            deltas = indices.new_ones(1)

        return RewardSchedule(
            indices=indices,
            timesteps=timesteps_sel,
            intervals=deltas,
        )

    def _stride_for_timestep(self, timestep: float, t_max: float) -> int:
        """Return stride length for a given diffusion time under the active policy."""
        if self.mode == "uniform":
            return self.uniform_stride

        threshold = None
        if self.threshold_timestep is not None:
            threshold = float(self.threshold_timestep)
        elif t_max > 0:
            threshold = t_max * self.threshold_fraction

        if threshold is None:
            return self.fine_stride

        return self.coarse_stride if timestep > threshold else self.fine_stride
