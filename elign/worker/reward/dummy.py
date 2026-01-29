from __future__ import annotations

from typing import Optional

import torch

from verl_diffusion.protocol import DataProto, TensorDict
from verl_diffusion.worker.reward.base import BaseReward


class DummyReward(BaseReward):
    """Lightweight rewarder for smoke tests and offline runs.

    Produces constant rewards and (optionally) constant stability flags. This is
    useful for validating the RL plumbing without requiring UMA/MLFF models.
    """

    def __init__(
        self,
        reward_value: float = 0.0,
        stability_value: float = 0.0,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.reward_value = float(reward_value)
        self.stability_value = float(stability_value)
        self.device = device or torch.device("cpu")

    def calculate_rewards(self, data: DataProto) -> DataProto:
        batch_size = int(data.batch.batch_size[0])
        device = self.device

        rewards = torch.full((batch_size,), self.reward_value, dtype=torch.float32, device=device)
        stability = torch.full((batch_size,), self.stability_value, dtype=torch.float32, device=device)

        result = {
            "rewards": rewards,
            "force_rewards": rewards.clone(),
            "energy_rewards": torch.zeros_like(rewards),
            "weighted_force_rewards": rewards.clone(),
            "weighted_energy_rewards": torch.zeros_like(rewards),
            "stability": stability,
            "stability_rewards": torch.zeros_like(rewards),
        }
        return DataProto(batch=TensorDict(result, batch_size=[batch_size]), meta_info=data.meta_info.copy())

