import unittest

import torch

from verl_diffusion.protocol import DataProto, TensorDict
from verl_diffusion.worker.actor.edm_actor import EDMActor


class _DummyActorModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

    def get_mask(self, nodesxsample, batch_size, max_n_nodes):
        del nodesxsample
        node_mask = torch.ones(batch_size, max_n_nodes, 1)
        edge_mask = torch.ones(batch_size * max_n_nodes * max_n_nodes, 1)
        return node_mask, edge_mask

    def sample_p_zs_given_zt(self, *args, **kwargs):
        raise AssertionError("train_batched_samples is patched in this test; sampling should not run")


class TestActorPolicyStartIdx(unittest.TestCase):
    def test_update_policy_respects_policy_start_idx_without_shared_prefix(self):
        model = _DummyActorModel()
        config = {
            "model": {
                "time_step": 4,
                "share_initial_noise": False,
            },
            "train": {
                "learning_rate": 1e-4,
                "adam_beta1": 0.9,
                "adam_beta2": 0.999,
                "adam_weight_decay": 0.0,
                "adam_epsilon": 1e-8,
                "train_micro_batch_size": 8,
                "max_grad_norm": 1.0,
                "clip_range": 0.2,
                "gradient_accumulation_steps": 1,
                "epoch_per_rollout": 1,
                "kl_penalty_weight": 0.0,
            },
        }
        actor = EDMActor(model, config)
        captured = {}

        def fake_train(samples_batched):
            captured["samples_batched"] = samples_batched
            return {"ClipFrac": 0.0, "Loss": 0.0, "lr": actor.optimizer.param_groups[0]["lr"]}

        actor.train_batched_samples = fake_train

        batch_size = 2
        latents = torch.arange(batch_size * 6, dtype=torch.float32).view(batch_size, 6, 1, 1, 1)
        logps = torch.arange(batch_size * 5, dtype=torch.float32).view(batch_size, 5)
        timesteps = torch.tensor([[3, 2, 1, 0, 0], [3, 2, 1, 0, 0]], dtype=torch.long)
        advantages = torch.ones(batch_size, dtype=torch.float32)
        nodesxsample = torch.ones(batch_size, dtype=torch.long)

        data = DataProto(
            batch=TensorDict(
                {
                    "latents": latents,
                    "logps": logps,
                    "timesteps": timesteps,
                    "advantages": advantages,
                    "nodesxsample": nodesxsample,
                },
                batch_size=[batch_size],
            ),
            meta_info={
                "condition": False,
                "max_n_nodes": 1,
                "policy_start_idx": 2,
                "share_initial_noise": False,
            },
        )

        actor.update_policy(data)
        self.assertIn("samples_batched", captured)
        samples = captured["samples_batched"][0]

        self.assertEqual(samples["latents"].shape[1], 3)
        self.assertEqual(samples["next_latents"].shape[1], 2)
        self.assertEqual(samples["timesteps"].shape[1], 2)
        self.assertEqual(samples["logps"].shape[1], 2)
        self.assertTrue(torch.equal(samples["timesteps"], torch.tensor([[1, 0], [1, 0]])))


if __name__ == "__main__":
    unittest.main()
