import json
import os
import tempfile
import unittest

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None


@unittest.skipUnless(np is not None, "numpy is required for Ala2 scratch tests")
class TestAla2ScratchEval(unittest.TestCase):
    def test_histogram_metrics_are_zero_for_identical_inputs(self):
        from ala2_scratch import eval as ala2_eval

        phi = np.linspace(-np.pi, np.pi, 64, endpoint=False)
        psi = np.linspace(np.pi, -np.pi, 64, endpoint=False)
        hist_a, _, _ = ala2_eval.normalized_hist2d(phi, psi, bins=24)
        hist_b, _, _ = ala2_eval.normalized_hist2d(phi, psi, bins=24)

        self.assertAlmostEqual(ala2_eval.js_divergence(hist_a, hist_b), 0.0, places=8)
        self.assertAlmostEqual(ala2_eval.free_energy_rmse(hist_a, hist_b), 0.0, places=8)

    def test_smoke_test_processed_data_loads_splits_and_computes_metric(self):
        from ala2_scratch import eval as ala2_eval

        rng = np.random.RandomState(0)
        coords = rng.randn(24, 22, 3).astype(np.float64)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "coords.npz")
            np.savez(path, coords=coords)

            report = ala2_eval.smoke_test_processed_data(
                path,
                phi_indices=(0, 1, 2, 3),
                psi_indices=(1, 2, 3, 4),
                batch_size=6,
                block_size=4,
            )

            self.assertEqual(report["num_frames"], 24)
            self.assertEqual(report["num_atoms"], 22)
            self.assertEqual(sum((report["train_size"], report["val_size"], report["test_size"])), 24)
            self.assertEqual(report["batch_shape"], [6, 22, 3])
            self.assertTrue(np.isfinite(report["batch_rms"]))
            self.assertIn("phi_psi_hist_entropy", report)
            self.assertTrue(np.isfinite(report["phi_psi_hist_entropy"]))

    def test_normalized_hist2d_raises_on_invalid_torsions(self):
        from ala2_scratch import eval as ala2_eval

        phi = np.asarray([np.nan, np.nan])
        psi = np.asarray([np.nan, np.nan])

        with self.assertRaises(ValueError):
            ala2_eval.normalized_hist2d(phi, psi, bins=24)

    def test_cli_smoke_mode_writes_json(self):
        from ala2_scratch import eval as ala2_eval

        rng = np.random.RandomState(1)
        coords = rng.randn(16, 22, 3).astype(np.float64)

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "coords.npz")
            out_path = os.path.join(tmpdir, "smoke.json")
            np.savez(data_path, positions=coords)

            rc = ala2_eval.main(
                [
                    "--smoke-check-data",
                    "--processed-data",
                    data_path,
                    "--phi-indices",
                    "0,1,2,3",
                    "--psi-indices",
                    "1,2,3,4",
                    "--output-json",
                    out_path,
                ]
            )
            self.assertEqual(rc, 0)
            self.assertTrue(os.path.exists(out_path))
            with open(out_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(payload["mode"], "data_smoke")
            self.assertEqual(payload["num_atoms"], 22)

    def test_prepare_data_reports_successful_loading_and_processing(self):
        from ala2_scratch.prepare_data import prepare_ala2_dataset

        pdb_text = """\
ATOM      1  C   ACE A   1       0.000   0.000   0.000  1.00  0.00           C
ATOM      2  N   ALA A   2       1.200   0.100   0.000  1.00  0.00           N
ATOM      3  CA  ALA A   2       2.100   0.500   0.100  1.00  0.00           C
ATOM      4  C   ALA A   2       3.200   0.200   0.000  1.00  0.00           C
ATOM      5  N   NME A   3       4.100   0.600   0.000  1.00  0.00           N
END
"""
        frames = np.stack(
            [
                np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [1.2, 0.1, 0.0],
                        [2.1, 0.5, 0.1],
                        [3.2, 0.2, 0.0],
                        [4.1, 0.6, 0.0],
                    ],
                    dtype=np.float32,
                )
                + 0.01 * i
                for i in range(6)
            ],
            axis=0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = os.path.join(tmpdir, "ala2.pdb")
            traj_path = os.path.join(tmpdir, "ala2_traj.npz")
            dataset_path = os.path.join(tmpdir, "ala2_processed.npz")
            topology_path = os.path.join(tmpdir, "ala2_topology.json")
            split_path = os.path.join(tmpdir, "ala2_splits.npz")

            with open(pdb_path, "w", encoding="utf-8") as handle:
                handle.write(pdb_text)
            np.savez(traj_path, positions=frames)

            summary = prepare_ala2_dataset(
                trajectory_path=traj_path,
                pdb_path=pdb_path,
                dataset_output_path=dataset_path,
                topology_output_path=topology_path,
                split_output_path=split_path,
                block_size=2,
                seed=0,
            )

            success = summary["success_criteria"]
            self.assertTrue(all(bool(value) for value in success.values()))
            self.assertTrue(os.path.exists(dataset_path))
            self.assertTrue(os.path.exists(topology_path))
            self.assertTrue(os.path.exists(split_path))
            self.assertEqual(summary["num_frames"], 6)
            self.assertEqual(summary["num_atoms"], 5)
            self.assertEqual(tuple(summary["phi_indices"]), (0, 1, 2, 3))
            self.assertEqual(tuple(summary["psi_indices"]), (1, 2, 3, 4))
            self.assertAlmostEqual(float(summary["position_scale"]), 1.0, places=6)

    def test_prepare_data_auto_scales_nm_trajectory_to_angstroms(self):
        from ala2_scratch.data import load_processed_dataset
        from ala2_scratch.prepare_data import prepare_ala2_dataset

        pdb_text = """\
ATOM      1  C   ACE A   1       0.000   0.000   0.000  1.00  0.00           C
ATOM      2  N   ALA A   2       1.200   0.000   0.000  1.00  0.00           N
ATOM      3  CA  ALA A   2       2.400   0.000   0.000  1.00  0.00           C
ATOM      4  C   ALA A   2       3.600   0.000   0.000  1.00  0.00           C
ATOM      5  N   NME A   3       4.800   0.000   0.000  1.00  0.00           N
END
"""
        frames_angstrom = np.stack(
            [
                np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [1.2, 0.0, 0.0],
                        [2.4, 0.0, 0.0],
                        [3.6, 0.0, 0.0],
                        [4.8, 0.0, 0.0],
                    ],
                    dtype=np.float32,
                )
                + 0.01 * i
                for i in range(4)
            ],
            axis=0,
        )
        frames_nm = frames_angstrom / 10.0

        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = os.path.join(tmpdir, "ala2_nm.pdb")
            traj_path = os.path.join(tmpdir, "ala2_nm_traj.npz")
            dataset_path = os.path.join(tmpdir, "ala2_nm_processed.npz")
            topology_path = os.path.join(tmpdir, "ala2_nm_topology.json")

            with open(pdb_path, "w", encoding="utf-8") as handle:
                handle.write(pdb_text)
            np.savez(traj_path, positions=frames_nm)

            summary = prepare_ala2_dataset(
                trajectory_path=traj_path,
                pdb_path=pdb_path,
                dataset_output_path=dataset_path,
                topology_output_path=topology_path,
                block_size=2,
                seed=0,
            )
            processed = load_processed_dataset(dataset_path, topology_path)

            self.assertAlmostEqual(float(summary["position_scale"]), 10.0, places=5)
            self.assertAlmostEqual(float(processed.position_scale), 10.0, places=5)
            self.assertGreater(float(np.abs(processed.positions).mean()), 0.1)


if __name__ == "__main__":
    unittest.main()
