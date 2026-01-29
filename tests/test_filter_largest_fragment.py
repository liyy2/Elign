import random
import sys
import types
import unittest

import torch

from elign.protocol import DataProto, TensorDict
from elign.worker.filter.filter import Filter


class TestFilterLargestFragment(unittest.TestCase):
    def test_filter_uses_largest_fragment_smiles(self):
        # Build two RDKit mols whose *full* SMILES differ by a small extra fragment,
        # but whose largest fragment is identical ("CC").
        from rdkit import Chem

        mols = iter([Chem.MolFromSmiles("CC.C"), Chem.MolFromSmiles("CC.O")])

        stub_mod = types.ModuleType("edm_source.qm9.rdkit_functions")

        def build_molecule(*_args, **_kwargs):
            return next(mols)

        def mol2smiles(mol):
            return Chem.MolToSmiles(mol)

        stub_mod.build_molecule = build_molecule
        stub_mod.mol2smiles = mol2smiles

        # Inject stub so Filter.filter() can run without importing the real QM9
        # rdkit_functions (which depends on edm_source sys.path setup).
        sys.modules["edm_source.qm9.rdkit_functions"] = stub_mod

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
        )

        random.seed(0)
        filtered, _filter_ratio, _novelty_ratio, validity, uniqueness = filt.filter(data)

        self.assertEqual(validity, 1.0)
        # With largest-fragment canonicalization both samples map to "CC" => 1 unique out of 2.
        self.assertEqual(uniqueness, 0.5)
        self.assertEqual(len(filtered), 1)


if __name__ == "__main__":
    unittest.main()
