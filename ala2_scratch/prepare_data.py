from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .data import (
    align_frames,
    blocked_split_indices,
    infer_position_scale,
    load_npz_frames,
    save_processed_dataset,
    save_split_indices,
    smoke_load_processed_dataset,
)
from .topology import parse_pdb, residue_summary, topology_to_json


def prepare_ala2_dataset(
    *,
    trajectory_path: str | Path,
    pdb_path: str | Path,
    dataset_output_path: str | Path,
    topology_output_path: str | Path,
    split_output_path: Optional[str | Path] = None,
    test_trajectory_path: Optional[str | Path] = None,
    test_pdb_path: Optional[str | Path] = None,
    stride: int = 1,
    max_frames: Optional[int] = None,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    block_size: int = 100,
    seed: int = 0,
    align: bool = True,
    position_scale: Optional[float] = None,
) -> Dict[str, object]:
    """Prepare a fixed-topology Ala2 dataset bundle from PDB + trajectory NPZ."""

    trajectory_path = Path(trajectory_path)
    pdb_path = Path(pdb_path)
    dataset_output_path = Path(dataset_output_path)
    topology_output_path = Path(topology_output_path)
    if split_output_path is None:
        split_output_path = dataset_output_path.with_name(f"{dataset_output_path.stem}_splits.npz")
    split_output_path = Path(split_output_path)
    test_trajectory_path = None if test_trajectory_path is None else Path(test_trajectory_path)
    test_pdb_path = None if test_pdb_path is None else Path(test_pdb_path)

    topology, pdb_positions = parse_pdb(pdb_path)
    frames, positions_key = load_npz_frames(trajectory_path, expected_n_atoms=len(topology.atoms))
    inferred_position_scale = (
        infer_position_scale(frames, pdb_positions) if position_scale is None else float(position_scale)
    )
    frames = (frames * inferred_position_scale).astype(np.float32, copy=False)

    if stride <= 0:
        raise ValueError("stride must be positive.")
    frames = frames[::stride]
    if max_frames is not None:
        frames = frames[: int(max_frames)]
    if frames.shape[0] == 0:
        raise ValueError("No frames left after applying stride/max_frames.")

    if align:
        positions = align_frames(frames, pdb_positions)
        reference_positions = align_frames(pdb_positions[None, :, :], pdb_positions)[0]
    else:
        positions = frames.astype(np.float32, copy=False)
        reference_positions = pdb_positions.astype(np.float32, copy=False)

    split_indices = blocked_split_indices(
        int(positions.shape[0]),
        val_fraction=val_fraction,
        test_fraction=test_fraction if test_trajectory_path is None else 0.0,
        block_size=block_size,
        seed=seed,
    )

    if test_trajectory_path is not None:
        test_topology, test_pdb_positions = parse_pdb(test_pdb_path or pdb_path)
        if test_topology.atomic_numbers != topology.atomic_numbers:
            raise ValueError("Held-out test topology must match the training topology exactly.")
        test_frames, test_positions_key = load_npz_frames(
            test_trajectory_path,
            expected_n_atoms=len(test_topology.atoms),
        )
        test_frames = (test_frames * inferred_position_scale).astype(np.float32, copy=False)
        test_frames = test_frames[::stride]
        if max_frames is not None:
            test_frames = test_frames[: int(max_frames)]
        if test_frames.shape[0] == 0:
            raise ValueError("No held-out test frames left after applying stride/max_frames.")
        if align:
            test_positions = align_frames(test_frames, test_pdb_positions)
        else:
            test_positions = test_frames.astype(np.float32, copy=False)
        train_frame_count = int(positions.shape[0])
        positions = np.concatenate([positions, test_positions], axis=0)
        split_indices = {
            "train": split_indices["train"],
            "val": split_indices["val"],
            "test": np.arange(train_frame_count, train_frame_count + test_positions.shape[0], dtype=np.int64),
        }
    else:
        test_positions_key = None

    save_processed_dataset(
        dataset_output_path,
        positions=positions,
        reference_positions=reference_positions,
        split_indices=split_indices,
        position_scale=inferred_position_scale,
    )
    topology_to_json(topology, topology_output_path)
    save_split_indices(split_output_path, split_indices)

    smoke = smoke_load_processed_dataset(
        dataset_output_path,
        topology_output_path,
        split_path=split_output_path,
    )
    success_criteria = {
        "source_pdb_parsed": True,
        "source_traj_loaded": True,
        "metadata_inferred": True,
        "processed_dataset_saved": dataset_output_path.exists(),
        "topology_json_saved": topology_output_path.exists(),
        "split_file_saved": split_output_path.exists(),
        "smoke_load_passed": True,
    }

    return {
        "num_frames": int(positions.shape[0]),
        "num_atoms": int(positions.shape[1]),
        "trajectory_key": positions_key,
        "test_trajectory_key": test_positions_key,
        "position_scale": float(inferred_position_scale),
        "dataset_output_path": str(dataset_output_path),
        "topology_output_path": str(topology_output_path),
        "split_output_path": str(split_output_path),
        "split_sizes": {key: int(value.shape[0]) for key, value in split_indices.items()},
        "phi_indices": tuple(int(v) for v in topology.phi_indices),
        "psi_indices": tuple(int(v) for v in topology.psi_indices),
        "residues": residue_summary(topology),
        "smoke_load": smoke,
        "success_criteria": success_criteria,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a fixed-topology Ala2 trajectory dataset.")
    parser.add_argument("--trajectory", type=str, required=True, help="Path to the trajectory NPZ file.")
    parser.add_argument("--pdb", type=str, required=True, help="Path to the matching PDB topology file.")
    parser.add_argument("--out", type=str, required=True, help="Path to the processed dataset NPZ.")
    parser.add_argument("--topology-out", type=str, required=True, help="Path to the topology JSON output.")
    parser.add_argument("--split-out", type=str, default=None, help="Optional path to save split indices separately.")
    parser.add_argument(
        "--test-trajectory",
        type=str,
        default=None,
        help="Optional held-out trajectory NPZ. When provided, it becomes the full test split.",
    )
    parser.add_argument(
        "--test-pdb",
        type=str,
        default=None,
        help="Optional PDB path matching the held-out test trajectory.",
    )
    parser.add_argument("--stride", type=int, default=1, help="Frame subsampling stride.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap on the number of frames.")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Validation fraction by blocks.")
    parser.add_argument("--test-fraction", type=float, default=0.1, help="Test fraction by blocks.")
    parser.add_argument("--block-size", type=int, default=1000, help="Contiguous block size for split assignment.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for block-level split shuffling.")
    parser.add_argument(
        "--position-scale",
        type=float,
        default=None,
        help="Optional scalar to convert raw trajectory positions into Angstroms. Defaults to auto-inference.",
    )
    parser.add_argument(
        "--no-align",
        action="store_true",
        help="Disable centering + Kabsch alignment and keep raw trajectory coordinates.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    summary = prepare_ala2_dataset(
        trajectory_path=args.trajectory,
        pdb_path=args.pdb,
        dataset_output_path=args.out,
        topology_output_path=args.topology_out,
        split_output_path=args.split_out,
        test_trajectory_path=args.test_trajectory,
        test_pdb_path=args.test_pdb,
        stride=args.stride,
        max_frames=args.max_frames,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        block_size=args.block_size,
        seed=args.seed,
        align=not args.no_align,
        position_scale=args.position_scale,
    )
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
