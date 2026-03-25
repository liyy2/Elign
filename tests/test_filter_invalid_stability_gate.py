import sys
import types
import unittest

import torch

from verl_diffusion.protocol import DataProto, TensorDict
from verl_diffusion.worker.filter.filter import Filter


class TestFilterInvalidStabilityGate(unittest.TestCase):
    def test_invalid_samples_do_not_receive_positive_stability_reward(self):
        from rdkit import Chem

        mols = iter([Chem.MolFromSmiles("CC")])

        stub_mod = types.ModuleType("edm_source.qm9.rdkit_functions")

        def build_molecule(*_args, **_kwargs):
            try:
                return next(mols)
            except StopIteration:
                return None

        def mol2smiles(mol):
            if mol is None:
                return None
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
                        "force_rewards": torch.tensor([2.0, 2.0]),
                        "energy_rewards": torch.zeros(batch_size),
                        "weighted_force_rewards": torch.tensor([4.0, 4.0]),
                        "weighted_energy_rewards": torch.zeros(batch_size),
                        # Positive stability shaping that should not be credited to RDKit-invalid samples.
                        "stability_rewards": torch.tensor([5.0, 5.0]),
                    },
                    batch_size=[batch_size],
                ),
                meta_info={},
            )

            filt = Filter(
                dataset_info=dataset_info,
                file_name=None,
                condition=False,
                enable_filtering=False,
                enable_penalty=False,
                penalty_scale=0.0,
                invalid_penalty_scale=1.0,
                duplicate_penalty_scale=0.0,
            )

            filtered, *_rest = filt.filter(data)

            # First sample is RDKit-valid; second is invalid.
            self.assertTrue(torch.allclose(filtered.batch["force_rewards"], torch.tensor([2.0, -1.0])))
            self.assertTrue(torch.allclose(filtered.batch["weighted_force_rewards"], torch.tensor([4.0, 0.0])))
        finally:
            if original_mod is not None:
                sys.modules["edm_source.qm9.rdkit_functions"] = original_mod
            else:
                sys.modules.pop("edm_source.qm9.rdkit_functions", None)


if __name__ == "__main__":
    unittest.main()
