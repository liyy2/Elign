import math
import sys
import types
import unittest

import torch

from verl_diffusion.protocol import DataProto, TensorDict
from verl_diffusion.worker.filter.filter import Filter


class TestFilterHistoryPenalty(unittest.TestCase):
    def _make_data(self, dataset_info):
        batch_size = 1
        x = torch.zeros(batch_size, dataset_info["max_n_nodes"], 3)
        categorical = torch.zeros(batch_size, dataset_info["max_n_nodes"], 1)
        categorical[:, 0, 0] = 1.0
        nodesxsample = torch.tensor([1], dtype=torch.long)
        return DataProto(
            batch=TensorDict(
                {
                    "x": x,
                    "categorical": categorical,
                    "nodesxsample": nodesxsample,
                    "rewards": torch.zeros(batch_size),
                    "force_rewards": torch.zeros(batch_size),
                    "energy_rewards": torch.zeros(batch_size),
                },
                batch_size=[batch_size],
            ),
            meta_info={},
        )

    def test_history_penalty_log_scales_with_count(self):
        from rdkit import Chem

        stub_mod = types.ModuleType("edm_source.qm9.rdkit_functions")

        def build_molecule(*_args, **_kwargs):
            return Chem.MolFromSmiles("CC")

        def mol2smiles(mol):
            return Chem.MolToSmiles(mol)

        stub_mod.build_molecule = build_molecule
        stub_mod.mol2smiles = mol2smiles

        original_mod = sys.modules.get("edm_source.qm9.rdkit_functions")
        sys.modules["edm_source.qm9.rdkit_functions"] = stub_mod
        try:
            dataset_info = {"max_n_nodes": 4}

            filt = Filter(
                dataset_info=dataset_info,
                file_name=None,
                condition=False,
                enable_filtering=False,
                enable_penalty=False,
                penalty_scale=0.0,
                invalid_penalty_scale=0.0,
                duplicate_penalty_scale=0.0,
                history_size=8,
                history_penalty_scale=1.0,
                history_penalty_mode="log",
            )

            data1 = self._make_data(dataset_info)
            out1, *_rest1 = filt.filter(data1)
            self.assertAlmostEqual(out1.batch["rewards"].item(), 0.0)

            data2 = self._make_data(dataset_info)
            out2, *_rest2 = filt.filter(data2)
            self.assertAlmostEqual(out2.batch["rewards"].item(), -1.0)
            self.assertTrue(torch.allclose(out2.batch["force_rewards"], out2.batch["rewards"]))
            self.assertTrue(torch.allclose(out2.batch["energy_rewards"], out2.batch["rewards"]))

            expected_third = -(math.log1p(2.0) / math.log(2.0))
            data3 = self._make_data(dataset_info)
            out3, *_rest3 = filt.filter(data3)
            self.assertLess(out3.batch["rewards"].item(), out2.batch["rewards"].item())
            self.assertAlmostEqual(out3.batch["rewards"].item(), expected_third, places=5)
        finally:
            if original_mod is not None:
                sys.modules["edm_source.qm9.rdkit_functions"] = original_mod
            else:
                sys.modules.pop("edm_source.qm9.rdkit_functions", None)

    def test_history_penalty_respects_max_multiplier(self):
        from rdkit import Chem

        stub_mod = types.ModuleType("edm_source.qm9.rdkit_functions")

        def build_molecule(*_args, **_kwargs):
            return Chem.MolFromSmiles("CC")

        def mol2smiles(mol):
            return Chem.MolToSmiles(mol)

        stub_mod.build_molecule = build_molecule
        stub_mod.mol2smiles = mol2smiles

        original_mod = sys.modules.get("edm_source.qm9.rdkit_functions")
        sys.modules["edm_source.qm9.rdkit_functions"] = stub_mod
        try:
            dataset_info = {"max_n_nodes": 4}

            filt = Filter(
                dataset_info=dataset_info,
                file_name=None,
                condition=False,
                enable_filtering=False,
                enable_penalty=False,
                penalty_scale=0.0,
                invalid_penalty_scale=0.0,
                duplicate_penalty_scale=0.0,
                history_size=8,
                history_penalty_scale=1.0,
                history_penalty_mode="log",
                history_penalty_max_multiplier=1.0,
            )

            _out1, *_rest1 = filt.filter(self._make_data(dataset_info))
            out2, *_rest2 = filt.filter(self._make_data(dataset_info))
            out3, *_rest3 = filt.filter(self._make_data(dataset_info))

            self.assertAlmostEqual(out2.batch["rewards"].item(), -1.0)
            self.assertAlmostEqual(out3.batch["rewards"].item(), -1.0)
        finally:
            if original_mod is not None:
                sys.modules["edm_source.qm9.rdkit_functions"] = original_mod
            else:
                sys.modules.pop("edm_source.qm9.rdkit_functions", None)


if __name__ == "__main__":
    unittest.main()

