import unittest

import torch

from elign.protocol import DataProto, TensorDict
from elign.worker.filter.filter import Filter


class TestFilterPenaltyInjection(unittest.TestCase):
    def test_add_terminal_penalty_updates_all_reward_channels(self):
        batch_size = 2
        horizon = 3
        base_scalar = torch.tensor([1.0, 2.0])
        base_ts = torch.tensor([[0.0, 0.5, 1.0], [0.0, 1.5, 2.0]])
        penalty = torch.tensor([-0.25, -1.0])

        batch = TensorDict(
            {
                "rewards": base_scalar.clone(),
                "force_rewards": base_scalar.clone(),
                "energy_rewards": base_scalar.clone(),
                "rewards_ts": base_ts.clone(),
                "force_rewards_ts": base_ts.clone(),
                "energy_rewards_ts": base_ts.clone(),
            },
            batch_size=[batch_size],
        )
        data = DataProto(batch=batch, meta_info={})

        Filter._add_terminal_penalty(data, penalty)

        for key in ("rewards", "force_rewards", "energy_rewards"):
            self.assertTrue(torch.allclose(data.batch[key], base_scalar + penalty))

        for key in ("rewards_ts", "force_rewards_ts", "energy_rewards_ts"):
            expected = base_ts.clone()
            expected[:, -1] = expected[:, -1] + penalty
            self.assertTrue(torch.allclose(data.batch[key], expected))


if __name__ == "__main__":
    unittest.main()
