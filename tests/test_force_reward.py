import unittest
from unittest import mock

import torch
from typing import Optional, Dict

from verl_diffusion.protocol import DataProto, TensorDict
from verl_diffusion.worker.reward.force import UMAForceReward, _resolve_mlff_device


class _StubForceComputer:
    def __init__(self, forces: torch.Tensor, energies: Optional[torch.Tensor] = None):
        self._forces = forces
        self._energies = energies

    def compute_mlff_forces(self, z: torch.Tensor, node_mask: torch.Tensor, dataset_info: dict):
        forces = self._forces.to(device=z.device, dtype=z.dtype)
        if self._energies is None:
            return forces
        energies = self._energies.to(device=z.device, dtype=z.dtype)
        return forces, energies


def _make_minimal_qm9_like_dataset_info():
    return {
        "name": "qm9",
        "atom_decoder": ["H", "C", "N", "O", "F"],
        "normalize_factors": [1.0],
    }


def _make_single_sample_dataproto(
    positions: torch.Tensor,
    categorical: torch.Tensor,
    nodesxsample: torch.Tensor,
    meta_info: Optional[Dict] = None,
) -> DataProto:
    batch = TensorDict(
        {
            "x": positions,
            "categorical": categorical,
            "nodesxsample": nodesxsample,
        },
        batch_size=[positions.shape[0]],
    )
    return DataProto(batch=batch, meta_info=meta_info or {})


class TestResolveMlffDevice(unittest.TestCase):
    def test_resolves_torch_device(self):
        self.assertEqual(_resolve_mlff_device(torch.device("cpu"), "cpu"), "cpu")
        self.assertEqual(_resolve_mlff_device(torch.device("cuda", 0), "cpu"), "cuda:0")
        self.assertEqual(_resolve_mlff_device(torch.device("cuda"), "cpu"), "cuda")

    def test_resolves_strings(self):
        self.assertEqual(_resolve_mlff_device("cpu", "cuda"), "cpu")
        self.assertEqual(_resolve_mlff_device("CUDA", "cpu"), "cuda")
        self.assertEqual(_resolve_mlff_device("CUDA:1", "cpu"), "cuda:1")
        self.assertEqual(_resolve_mlff_device(None, "cpu"), "cpu")

    def test_rejects_invalid(self):
        with self.assertRaises(ValueError):
            _resolve_mlff_device("cuda:abc", "cpu")
        with self.assertRaises(ValueError):
            _resolve_mlff_device("tpu", "cpu")


class TestUMAForceReward(unittest.TestCase):
    def test_aggregate_force_metric_rms(self):
        dataset_info = _make_minimal_qm9_like_dataset_info()
        rewarder = UMAForceReward(
            dataset_info=dataset_info,
            device="cpu",
            mlff_device="cpu",
            force_computer=_StubForceComputer(torch.zeros(1, 1, 3)),
            force_aggregation="rms",
            shaping={"enabled": False, "terminal_weight": 1.0},
        )

        forces = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]]])
        node_mask = torch.tensor([[[1.0], [1.0], [0.0]]])

        aggregated = rewarder._aggregate_force_metric(forces, node_mask)
        expected = torch.sqrt(torch.tensor(2.5))  # sqrt((1^2 + 2^2) / 2)
        self.assertTrue(torch.allclose(aggregated, expected, atol=1e-6))

    def test_aggregate_force_metric_max(self):
        dataset_info = _make_minimal_qm9_like_dataset_info()
        rewarder = UMAForceReward(
            dataset_info=dataset_info,
            device="cpu",
            mlff_device="cpu",
            force_computer=_StubForceComputer(torch.zeros(1, 1, 3)),
            force_aggregation="max",
            shaping={"enabled": False, "terminal_weight": 1.0},
        )

        forces = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [3.0, 0.0, 0.0]]])
        node_mask = torch.tensor([[[1.0], [1.0], [0.0]]])

        aggregated = rewarder._aggregate_force_metric(forces, node_mask)
        expected = torch.tensor([2.0])
        self.assertTrue(torch.allclose(aggregated, expected, atol=1e-6))

    def test_calculate_rewards_terminal_force_only(self):
        dataset_info = _make_minimal_qm9_like_dataset_info()
        max_n_nodes = 4
        positions = torch.zeros(1, max_n_nodes, 3)
        categorical = torch.zeros(1, max_n_nodes, len(dataset_info["atom_decoder"]))
        categorical[0, 0, 0] = 1.0
        categorical[0, 1, 1] = 1.0
        nodesxsample = torch.tensor([2], dtype=torch.long)

        forces = torch.zeros(1, max_n_nodes, 3)
        forces[0, 0] = torch.tensor([1.0, 0.0, 0.0])
        forces[0, 1] = torch.tensor([0.0, 2.0, 0.0])

        rewarder = UMAForceReward(
            dataset_info=dataset_info,
            device="cpu",
            mlff_device="cpu",
            force_computer=_StubForceComputer(forces),
            use_energy=False,
            stability_weight=0.0,
            force_aggregation="rms",
            shaping={"enabled": False, "terminal_weight": 1.0},
        )

        sample = _make_single_sample_dataproto(positions, categorical, nodesxsample)
        with mock.patch("verl_diffusion.worker.reward.force.check_stability", return_value=(1.0,)):
            out = rewarder.calculate_rewards(sample)

        expected_force_metric = torch.sqrt(torch.tensor(2.5))
        expected_reward = -expected_force_metric
        self.assertTrue(torch.allclose(out.batch["rewards"], torch.tensor([expected_reward]), atol=1e-6))
        self.assertTrue(torch.allclose(out.batch["force_rewards"], torch.tensor([expected_reward]), atol=1e-6))
        self.assertTrue(torch.allclose(out.batch["weighted_force_rewards"], torch.tensor([expected_reward]), atol=1e-6))

    def test_force_reward_gated_to_stable(self):
        dataset_info = _make_minimal_qm9_like_dataset_info()
        max_n_nodes = 4
        positions = torch.zeros(1, max_n_nodes, 3)
        categorical = torch.zeros(1, max_n_nodes, len(dataset_info["atom_decoder"]))
        categorical[0, 0, 0] = 1.0
        categorical[0, 1, 1] = 1.0
        nodesxsample = torch.tensor([2], dtype=torch.long)

        forces = torch.zeros(1, max_n_nodes, 3)
        forces[0, 0] = torch.tensor([1.0, 0.0, 0.0])
        forces[0, 1] = torch.tensor([0.0, 2.0, 0.0])

        rewarder = UMAForceReward(
            dataset_info=dataset_info,
            device="cpu",
            mlff_device="cpu",
            force_computer=_StubForceComputer(forces),
            use_energy=False,
            stability_weight=0.0,
            force_only_if_stable=True,
            force_aggregation="rms",
            shaping={"enabled": False, "terminal_weight": 1.0},
        )

        sample = _make_single_sample_dataproto(positions, categorical, nodesxsample)
        with mock.patch("verl_diffusion.worker.reward.force.check_stability", return_value=(0.0,)):
            out = rewarder.calculate_rewards(sample)

        self.assertTrue(torch.allclose(out.batch["stability"], torch.tensor([0.0]), atol=1e-6))
        self.assertTrue(torch.allclose(out.batch["rewards"], torch.tensor([0.0]), atol=1e-6))
        self.assertTrue(torch.allclose(out.batch["force_rewards"], torch.tensor([0.0]), atol=1e-6))
        self.assertTrue(torch.allclose(out.batch["weighted_force_rewards"], torch.tensor([0.0]), atol=1e-6))

    def test_calculate_rewards_terminal_with_energy_and_stability_bonus(self):
        dataset_info = _make_minimal_qm9_like_dataset_info()
        max_n_nodes = 3
        positions = torch.zeros(1, max_n_nodes, 3)
        categorical = torch.zeros(1, max_n_nodes, len(dataset_info["atom_decoder"]))
        categorical[0, 0, 0] = 1.0
        categorical[0, 1, 1] = 1.0
        nodesxsample = torch.tensor([2], dtype=torch.long)

        forces = torch.zeros(1, max_n_nodes, 3)
        forces[0, 0] = torch.tensor([2.0, 0.0, 0.0])
        forces[0, 1] = torch.tensor([0.0, 1.0, 0.0])
        energies = torch.tensor([4.0])

        rewarder = UMAForceReward(
            dataset_info=dataset_info,
            device="cpu",
            mlff_device="cpu",
            force_computer=_StubForceComputer(forces, energies=energies),
            use_energy=True,
            force_weight=1.0,
            energy_weight=2.0,
            stability_weight=0.5,
            force_aggregation="max",
            energy_transform_offset=0.0,
            energy_transform_scale=1.0,
            shaping={"enabled": False, "terminal_weight": 1.0},
        )

        sample = _make_single_sample_dataproto(positions, categorical, nodesxsample)
        with mock.patch("verl_diffusion.worker.reward.force.check_stability", return_value=(1.0,)):
            out = rewarder.calculate_rewards(sample)

        # Force metric = max(norm([2,0,0]), norm([0,1,0])) = 2
        # Force reward = -2 + stability_bonus(=+0.5) = -1.5
        # Energy reward = -((4 + 0) / 1) = -4, weighted by energy_weight=2 => -8
        # Total = -1.5 + -8 = -9.5
        self.assertTrue(torch.allclose(out.batch["stability"], torch.tensor([1.0]), atol=1e-6))
        self.assertTrue(torch.allclose(out.batch["stability_rewards"], torch.tensor([0.5]), atol=1e-6))
        self.assertTrue(torch.allclose(out.batch["force_rewards"], torch.tensor([-1.5]), atol=1e-6))
        self.assertTrue(torch.allclose(out.batch["energy_rewards"], torch.tensor([-4.0]), atol=1e-6))
        self.assertTrue(torch.allclose(out.batch["weighted_energy_rewards"], torch.tensor([-8.0]), atol=1e-6))
        self.assertTrue(torch.allclose(out.batch["rewards"], torch.tensor([-9.5]), atol=1e-6))

    def test_valence_underbond_penalty_applied(self):
        dataset_info = _make_minimal_qm9_like_dataset_info()
        max_n_nodes = 2
        # C-H single bond at 1.0 Å -> carbon has bond order 1, missing 3 to reach valence 4.
        positions = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
        categorical = torch.zeros(1, max_n_nodes, len(dataset_info["atom_decoder"]))
        categorical[0, 0, 1] = 1.0  # C
        categorical[0, 1, 0] = 1.0  # H
        nodesxsample = torch.tensor([2], dtype=torch.long)

        forces = torch.zeros(1, max_n_nodes, 3)
        rewarder = UMAForceReward(
            dataset_info=dataset_info,
            device="cpu",
            mlff_device="cpu",
            force_computer=_StubForceComputer(forces),
            use_energy=False,
            stability_weight=0.0,
            atom_stability_weight=0.0,
            valence_underbond_weight=0.5,
            valence_overbond_weight=0.0,
            force_aggregation="rms",
            shaping={"enabled": False, "terminal_weight": 1.0},
        )

        sample = _make_single_sample_dataproto(positions, categorical, nodesxsample)
        with mock.patch("verl_diffusion.worker.reward.force.check_stability", return_value=(1.0, 2, 2)):
            out = rewarder.calculate_rewards(sample)

        self.assertTrue(torch.allclose(out.batch["valence_underbond"], torch.tensor([3.0]), atol=1e-6))
        self.assertTrue(torch.allclose(out.batch["valence_overbond"], torch.tensor([0.0]), atol=1e-6))
        self.assertTrue(torch.allclose(out.batch["valence_underbond_rewards"], torch.tensor([-1.5]), atol=1e-6))
        self.assertTrue(torch.allclose(out.batch["valence_overbond_rewards"], torch.tensor([0.0]), atol=1e-6))
        self.assertTrue(torch.allclose(out.batch["rewards"], torch.tensor([-1.5]), atol=1e-6))

    def test_valence_overbond_penalty_applied(self):
        dataset_info = _make_minimal_qm9_like_dataset_info()
        max_n_nodes = 3
        # Central carbon has a triple bond to carbon (1.2 Å) and a double bond to oxygen (1.2 Å).
        # Bond order sum for the first carbon: 3 + 2 = 5 -> overbond by 1.
        # Second carbon has bond order 3 -> underbond by 1.
        positions = torch.tensor([[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.0, 1.2, 0.0]]])
        categorical = torch.zeros(1, max_n_nodes, len(dataset_info["atom_decoder"]))
        categorical[0, 0, 1] = 1.0  # C
        categorical[0, 1, 1] = 1.0  # C
        categorical[0, 2, 3] = 1.0  # O
        nodesxsample = torch.tensor([3], dtype=torch.long)

        forces = torch.zeros(1, max_n_nodes, 3)
        rewarder = UMAForceReward(
            dataset_info=dataset_info,
            device="cpu",
            mlff_device="cpu",
            force_computer=_StubForceComputer(forces),
            use_energy=False,
            stability_weight=0.0,
            atom_stability_weight=0.0,
            valence_underbond_weight=1.0,
            valence_overbond_weight=2.0,
            force_aggregation="rms",
            shaping={"enabled": False, "terminal_weight": 1.0},
        )

        sample = _make_single_sample_dataproto(positions, categorical, nodesxsample)
        with mock.patch("verl_diffusion.worker.reward.force.check_stability", return_value=(0.0, 0, 3)):
            out = rewarder.calculate_rewards(sample)

        self.assertTrue(torch.allclose(out.batch["valence_underbond"], torch.tensor([1.0]), atol=1e-6))
        self.assertTrue(torch.allclose(out.batch["valence_overbond"], torch.tensor([1.0]), atol=1e-6))
        self.assertTrue(torch.allclose(out.batch["valence_underbond_rewards"], torch.tensor([-1.0]), atol=1e-6))
        self.assertTrue(torch.allclose(out.batch["valence_overbond_rewards"], torch.tensor([-2.0]), atol=1e-6))
        self.assertTrue(torch.allclose(out.batch["rewards"], torch.tensor([-3.0]), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
