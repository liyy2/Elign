import os
import tempfile
import unittest
from types import SimpleNamespace

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


def _write_geom_stub(path: str, num_mols: int = 10, atoms_per_mol: int = 3) -> None:
    if np is None:
        raise RuntimeError("numpy is required to write the GEOM stub dataset")
    rng = np.random.RandomState(0)
    atomic_numbers = [1, 6, 7, 8]  # H, C, N, O
    rows = []
    for mol_id in range(num_mols):
        for atom_idx in range(atoms_per_mol):
            z = float(atomic_numbers[(mol_id + atom_idx) % len(atomic_numbers)])
            xyz = rng.randn(3).astype(np.float32)
            rows.append([float(mol_id), z, float(xyz[0]), float(xyz[1]), float(xyz[2])])
    np.save(path, np.asarray(rows, dtype=np.float32))


class TestGeomDatasetLoading(unittest.TestCase):
    @unittest.skipUnless(np is not None, "numpy is required for GEOM dataset tests")
    def test_load_split_data_creates_permutation(self):
        from edm_source import build_geom_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "geom_drugs_30.npy")
            _write_geom_stub(data_path, num_mols=10, atoms_per_mol=3)

            split = build_geom_dataset.load_split_data(
                data_path, val_proportion=0.1, test_proportion=0.1
            )

            self.assertTrue(os.path.exists(os.path.join(tmpdir, "geom_permutation.npy")))
            train_idx, val_idx, test_idx = split["splits"]
            self.assertEqual(len(train_idx) + len(val_idx) + len(test_idx), 10)

    @unittest.skipUnless(
        np is not None and torch is not None,
        "numpy and torch are required for dataloader smoke tests",
    )
    def test_retrieve_dataloaders_uses_geom_data_file(self):
        from edm_source.configs.datasets_config import get_dataset_info
        from edm_source.qm9.dataset import retrieve_dataloaders

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "geom_drugs_30.npy")
            _write_geom_stub(data_path, num_mols=10, atoms_per_mol=3)

            cfg = SimpleNamespace(
                dataset="geom",
                remove_h=False,
                filter_molecule_size=None,
                include_charges=False,
                device=torch.device("cpu"),
                sequential=False,
                batch_size=2,
                geom_data_file=data_path,
            )

            dataloaders, _ = retrieve_dataloaders(cfg)
            batch = next(iter(dataloaders["train"]))

            dataset_info = get_dataset_info("geom", remove_h=False)
            self.assertIn("positions", batch)
            self.assertIn("one_hot", batch)
            self.assertIn("atom_mask", batch)
            self.assertIn("edge_mask", batch)

            self.assertEqual(batch["positions"].shape[-1], 3)
            self.assertEqual(batch["one_hot"].shape[-1], len(dataset_info["atom_decoder"]))


if __name__ == "__main__":
    unittest.main()
