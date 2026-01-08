import random
import sys
import types
import unittest

import torch

from verl_diffusion.protocol import DataProto, TensorDict
from verl_diffusion.worker.filter.filter import Filter


class TestFilterDuplicatePenalty(unittest.TestCase):
    def test_duplicate_penalty_hits_kept_sample_even_with_filtering(self):
        from rdkit import Chem

        mols = iter([Chem.MolFromSmiles("CC.C"), Chem.MolFromSmiles("CC.O")])

        stub_mod = types.ModuleType("edm_source.qm9.rdkit_functions")

        def build_molecule(*_args, **_kwargs):
            return next(mols)

        def mol2smiles(mol):
            return Chem.MolToSmiles(mol)

        stub_mod.build_molecule = build_molecule
        stub_mod.mol2smiles = mol2smiles

        original_mod = sys.modules.get("edm_source.qm9.rdkit_functions")
        sys.modules["edm_source.qm9.rdkit_functions"] = stub_mod
        try:
            dataset_info = {"max_n_nodes": 4}
            batch_size = 2
            x = torch.zeros(batch_size, dataset_info["max_n_nodes"], 3)
            categorical = torch.zeros(batch_size, dataset_info["max_n_nodes"], 1)
            categorical[:, 0, 0] = 1.0
            nodesxsample = torch.tensor([1, 1], dtype=torch.long)

            data = DataProto(
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

            filt = Filter(
                dataset_info=dataset_info,
                file_name=None,
                condition=False,
                enable_filtering=True,
                enable_penalty=False,
                penalty_scale=0.0,
                invalid_penalty_scale=0.0,
                duplicate_penalty_scale=1.0,
            )

            random.seed(0)
            filtered, _filter_ratio, _novelty_ratio, validity, uniqueness = filt.filter(data)

            # Both samples map to the same largest-fragment SMILES ("CC"), i.e. a duplicate batch.
            self.assertEqual(validity, 1.0)
            self.assertEqual(uniqueness, 0.5)
            self.assertEqual(len(filtered), 1)

            # With enable_filtering=true, we keep a single representative for the duplicate SMILES,
            # so we assign the full duplicate cost to the kept sample: -(count-1) * scale.
            expected = torch.tensor([-1.0], dtype=filtered.batch["rewards"].dtype)
            self.assertTrue(torch.allclose(filtered.batch["rewards"], expected))
            self.assertTrue(torch.allclose(filtered.batch["force_rewards"], expected))
            self.assertTrue(torch.allclose(filtered.batch["energy_rewards"], expected))
        finally:
            if original_mod is not None:
                sys.modules["edm_source.qm9.rdkit_functions"] = original_mod
            else:
                sys.modules.pop("edm_source.qm9.rdkit_functions", None)


if __name__ == "__main__":
    unittest.main()
