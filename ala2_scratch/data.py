from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .topology import TopologyInfo, build_one_hot_features, topology_from_json


_PREFERRED_POSITION_KEYS = (
    "positions",
    "coords",
    "coordinates",
    "trajectory",
    "traj",
    "frames",
    "xyz",
    "R",
)

_AUTO_POSITION_SCALE_CANDIDATES = (1.0, 10.0)


@dataclass
class ProcessedDataset:
    positions: np.ndarray
    topology: TopologyInfo
    reference_positions: np.ndarray
    split_indices: Dict[str, np.ndarray]
    position_scale: float = 1.0


class Ala2TrajectoryDataset(Dataset):
    """Fixed-topology Ala2 trajectory dataset for coordinate diffusion."""

    def __init__(
        self,
        positions: np.ndarray,
        atomic_numbers: Sequence[int],
        split_indices: Sequence[int],
        *,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> None:
        if positions.ndim != 3 or positions.shape[-1] != 3:
            raise ValueError(f"Expected positions with shape [frames, atoms, 3], got {positions.shape}.")
        self.positions = torch.as_tensor(positions[np.asarray(split_indices)], dtype=torch.float32)
        self.atomic_numbers = torch.as_tensor(np.asarray(atomic_numbers), dtype=torch.long)
        node_features, feature_atomic_numbers = build_one_hot_features(atomic_numbers)
        self.node_features = torch.as_tensor(node_features, dtype=torch.float32)
        self.atom_mask = torch.ones(self.positions.shape[1], dtype=torch.float32)
        self.split_indices = torch.as_tensor(np.asarray(split_indices), dtype=torch.long)
        self.metadata = dict(metadata or {})
        self.metadata["feature_atomic_numbers"] = feature_atomic_numbers

    def __len__(self) -> int:
        return int(self.positions.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "positions": self.positions[idx],
            "node_features": self.node_features,
            "atom_mask": self.atom_mask,
            "atomic_numbers": self.atomic_numbers,
            "frame_index": self.split_indices[idx],
        }


def center_positions(positions: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Center coordinates to zero center-of-mass under an optional mask."""

    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"Expected positions with shape [atoms, 3], got {positions.shape}.")
    if mask is None:
        mean = positions.mean(axis=0, keepdims=True)
    else:
        weights = np.asarray(mask, dtype=np.float64).reshape(-1, 1)
        denom = float(weights.sum())
        if denom <= 0:
            raise ValueError("Mask must contain at least one positive entry.")
        mean = (positions * weights).sum(axis=0, keepdims=True) / denom
    return (positions - mean).astype(np.float32, copy=False)


def kabsch_align(mobile: np.ndarray, reference: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Align a coordinate set to a reference with the Kabsch algorithm."""

    mobile_centered = center_positions(mobile, mask=mask).astype(np.float64, copy=False)
    reference_centered = center_positions(reference, mask=mask).astype(np.float64, copy=False)

    if mask is not None:
        weights = np.asarray(mask, dtype=np.float64).reshape(-1, 1)
        mobile_used = mobile_centered * weights
        reference_used = reference_centered * weights
    else:
        mobile_used = mobile_centered
        reference_used = reference_centered

    covariance = mobile_used.T @ reference_used
    u, _, vt = np.linalg.svd(covariance, full_matrices=False)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    aligned = mobile_centered @ rotation
    return aligned.astype(np.float32, copy=False)


def align_frames(frames: np.ndarray, reference: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Center and Kabsch-align a stack of frames against a reference."""

    frames = np.asarray(frames, dtype=np.float32)
    aligned = np.empty_like(frames, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    for idx in range(frames.shape[0]):
        aligned[idx] = kabsch_align(frames[idx], reference, mask=mask)
    return aligned


def infer_position_scale(
    frames: np.ndarray,
    reference: np.ndarray,
    *,
    candidate_scales: Sequence[float] = _AUTO_POSITION_SCALE_CANDIDATES,
    sample_frames: int = 32,
    mask: Optional[np.ndarray] = None,
) -> float:
    """Infer a scalar position conversion factor by matching frames to a reference geometry."""

    frames = np.asarray(frames, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    if frames.ndim != 3 or frames.shape[-1] != 3:
        raise ValueError(f"Expected frames with shape [frames, atoms, 3], got {frames.shape}.")
    if reference.shape != (frames.shape[1], 3):
        raise ValueError(
            "Reference geometry shape must match a single frame: "
            f"expected {(frames.shape[1], 3)}, got {reference.shape}."
        )

    sample_count = min(int(sample_frames), int(frames.shape[0]))
    if sample_count <= 0:
        raise ValueError("Need at least one frame to infer a position scale.")

    reference_centered = center_positions(reference, mask=mask).astype(np.float64, copy=False)
    best_scale = float(candidate_scales[0])
    best_error = math.inf
    for candidate in candidate_scales:
        scale = float(candidate)
        if scale <= 0.0:
            raise ValueError("candidate_scales must contain only positive values.")
        errors = []
        for frame in frames[:sample_count]:
            aligned = kabsch_align(frame * scale, reference, mask=mask).astype(np.float64, copy=False)
            diff = aligned - reference_centered
            errors.append(float(np.sqrt(np.mean(diff ** 2))))
        median_error = float(np.median(errors))
        if median_error < best_error:
            best_error = median_error
            best_scale = scale
    return best_scale


def blocked_split_indices(
    num_frames: int,
    *,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    block_size: int = 100,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """Create train/val/test splits by whole contiguous blocks."""

    if num_frames <= 0:
        raise ValueError("num_frames must be positive.")
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    if val_fraction < 0 or test_fraction < 0 or (val_fraction + test_fraction) >= 1.0:
        raise ValueError("val_fraction and test_fraction must be non-negative and sum to less than 1.")

    n_blocks = int(math.ceil(num_frames / float(block_size)))
    block_ids = np.arange(n_blocks, dtype=np.int64)
    rng = np.random.RandomState(seed)
    permuted = rng.permutation(block_ids)

    n_val_blocks = int(round(val_fraction * n_blocks))
    n_test_blocks = int(round(test_fraction * n_blocks))
    if n_val_blocks + n_test_blocks >= n_blocks:
        n_test_blocks = max(0, min(n_test_blocks, n_blocks - 2))
        n_val_blocks = max(0, min(n_val_blocks, n_blocks - 1 - n_test_blocks))

    val_blocks = np.sort(permuted[:n_val_blocks])
    test_blocks = np.sort(permuted[n_val_blocks : n_val_blocks + n_test_blocks])
    train_blocks = np.sort(permuted[n_val_blocks + n_test_blocks :])
    if train_blocks.size == 0:
        raise ValueError("Blocked split produced no training blocks.")

    def _expand(selected_blocks: np.ndarray) -> np.ndarray:
        indices: List[int] = []
        for block_idx in selected_blocks.tolist():
            start = block_idx * block_size
            end = min(num_frames, start + block_size)
            indices.extend(range(start, end))
        return np.asarray(indices, dtype=np.int64)

    return {
        "train": _expand(train_blocks),
        "val": _expand(val_blocks),
        "test": _expand(test_blocks),
    }


def _normalize_frame_array(array: np.ndarray, expected_n_atoms: Optional[int]) -> Optional[np.ndarray]:
    array = np.asarray(array)
    if array.size == 0:
        return None

    if array.ndim == 2 and expected_n_atoms is not None and array.shape[1] == expected_n_atoms * 3:
        return array.reshape(array.shape[0], expected_n_atoms, 3)

    if array.ndim == 4 and array.shape[-1] == 3:
        leading = int(np.prod(array.shape[:-2]))
        return array.reshape(leading, array.shape[-2], 3)

    if array.ndim != 3 or array.shape[-1] != 3:
        return None

    if expected_n_atoms is None:
        if array.shape[0] < array.shape[1]:
            return np.swapaxes(array, 0, 1)
        return array

    if array.shape[1] == expected_n_atoms:
        return array
    if array.shape[0] == expected_n_atoms:
        return np.swapaxes(array, 0, 1)
    return None


def load_npz_frames(npz_path: str | Path, *, expected_n_atoms: Optional[int] = None) -> Tuple[np.ndarray, str]:
    """Load trajectory frames from an NPZ file using flexible key detection."""

    path = Path(npz_path)
    with np.load(path, allow_pickle=True) as payload:
        arrays = {key: payload[key] for key in payload.files}

    for key in _PREFERRED_POSITION_KEYS:
        if key not in arrays:
            continue
        normalized = _normalize_frame_array(arrays[key], expected_n_atoms)
        if normalized is not None:
            return normalized.astype(np.float32, copy=False), key

    for key, value in arrays.items():
        normalized = _normalize_frame_array(value, expected_n_atoms)
        if normalized is not None:
            return normalized.astype(np.float32, copy=False), key

    keys = ", ".join(arrays.keys())
    raise ValueError(f"Could not find a positions array in '{path}'. Available keys: {keys}")


def load_split_indices(path: str | Path) -> Dict[str, np.ndarray]:
    """Load split indices from a standalone NPZ file."""

    split_path = Path(path)
    with np.load(split_path, allow_pickle=False) as payload:
        required = ("train_idx", "val_idx", "test_idx")
        missing = [key for key in required if key not in payload]
        if missing:
            raise KeyError(f"Missing split arrays in '{split_path}': {missing}")
        return {
            "train": payload["train_idx"].astype(np.int64, copy=False),
            "val": payload["val_idx"].astype(np.int64, copy=False),
            "test": payload["test_idx"].astype(np.int64, copy=False),
        }


def save_processed_dataset(
    dataset_path: str | Path,
    *,
    positions: np.ndarray,
    reference_positions: np.ndarray,
    split_indices: Mapping[str, np.ndarray],
    position_scale: float = 1.0,
) -> None:
    """Write aligned coordinates and split indices to a compressed NPZ."""

    output_path = Path(dataset_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        positions=np.asarray(positions, dtype=np.float32),
        reference_positions=np.asarray(reference_positions, dtype=np.float32),
        train_idx=np.asarray(split_indices["train"], dtype=np.int64),
        val_idx=np.asarray(split_indices["val"], dtype=np.int64),
        test_idx=np.asarray(split_indices["test"], dtype=np.int64),
        position_scale=np.asarray([float(position_scale)], dtype=np.float32),
    )


def save_split_indices(path: str | Path, split_indices: Mapping[str, np.ndarray]) -> None:
    """Write split indices to a standalone NPZ for quick validation/debugging."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        train_idx=np.asarray(split_indices["train"], dtype=np.int64),
        val_idx=np.asarray(split_indices["val"], dtype=np.int64),
        test_idx=np.asarray(split_indices["test"], dtype=np.int64),
    )


def load_processed_dataset(
    dataset_path: str | Path,
    topology_path: str | Path,
    *,
    split_path: Optional[str | Path] = None,
) -> ProcessedDataset:
    """Load a processed Ala2 dataset bundle and matching topology metadata."""

    dataset_path = Path(dataset_path)
    topology_path = Path(topology_path)
    with np.load(dataset_path, allow_pickle=False) as payload:
        positions = payload["positions"].astype(np.float32, copy=False)
        reference_positions = payload["reference_positions"].astype(np.float32, copy=False)
        embedded_split_indices = {
            "train": payload["train_idx"].astype(np.int64, copy=False),
            "val": payload["val_idx"].astype(np.int64, copy=False),
            "test": payload["test_idx"].astype(np.int64, copy=False),
        }
        position_scale = float(payload["position_scale"][0]) if "position_scale" in payload else 1.0
    split_indices = embedded_split_indices
    if split_path is not None:
        external_split_indices = load_split_indices(split_path)
        for split_name, embedded in embedded_split_indices.items():
            external = external_split_indices[split_name]
            if not np.array_equal(embedded, external):
                raise ValueError(
                    f"Embedded split '{split_name}' in '{dataset_path}' does not match external split file '{split_path}'."
                )
        split_indices = external_split_indices
    topology = topology_from_json(topology_path)
    return ProcessedDataset(
        positions=positions,
        topology=topology,
        reference_positions=reference_positions,
        split_indices=split_indices,
        position_scale=position_scale,
    )


def smoke_load_processed_dataset(
    dataset_path: str | Path,
    topology_path: str | Path,
    *,
    split_path: Optional[str | Path] = None,
) -> Dict[str, object]:
    """Load a processed bundle and return shape/consistency checks."""

    processed = load_processed_dataset(dataset_path, topology_path, split_path=split_path)
    positions = processed.positions
    reference_positions = processed.reference_positions
    atomic_numbers = processed.topology.atomic_numbers
    split_indices = processed.split_indices

    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError(f"Processed positions must have shape [frames, atoms, 3], got {positions.shape}.")
    if reference_positions.shape != (positions.shape[1], 3):
        raise ValueError(
            "Reference positions must have shape [atoms, 3], "
            f"got {reference_positions.shape} for {positions.shape[1]} atoms."
        )
    if len(atomic_numbers) != positions.shape[1]:
        raise ValueError(
            "Topology atom count does not match processed positions: "
            f"{len(atomic_numbers)} vs {positions.shape[1]}."
        )

    seen = np.concatenate([split_indices["train"], split_indices["val"], split_indices["test"]], axis=0)
    if seen.size != positions.shape[0]:
        raise ValueError(
            "Split indices do not cover every frame exactly once: "
            f"covered {seen.size} vs {positions.shape[0]}."
        )
    if np.unique(seen).size != seen.size:
        raise ValueError("Split indices contain duplicate frame assignments.")

    return {
        "num_frames": int(positions.shape[0]),
        "num_atoms": int(positions.shape[1]),
        "train_size": int(split_indices["train"].shape[0]),
        "val_size": int(split_indices["val"].shape[0]),
        "test_size": int(split_indices["test"].shape[0]),
        "phi_indices": tuple(int(v) for v in processed.topology.phi_indices),
        "psi_indices": tuple(int(v) for v in processed.topology.psi_indices),
        "position_scale": float(processed.position_scale),
    }


def create_dataloaders(
    dataset_path: str | Path,
    topology_path: str | Path,
    *,
    split_path: Optional[str | Path] = None,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last_train: bool = False,
) -> Tuple[Dict[str, DataLoader], Dict[str, object]]:
    """Create train/val/test data loaders and return shared topology metadata."""

    processed = load_processed_dataset(dataset_path, topology_path, split_path=split_path)
    metadata: Dict[str, object] = {
        "atomic_numbers": processed.topology.atomic_numbers,
        "atom_names": processed.topology.atom_names,
        "residue_names": processed.topology.residue_names,
        "residue_ids": processed.topology.residue_ids,
        "phi_indices": processed.topology.phi_indices,
        "psi_indices": processed.topology.psi_indices,
        "reference_positions": processed.reference_positions,
        "position_scale": float(processed.position_scale),
    }

    loaders: Dict[str, DataLoader] = {}
    for split_name, shuffle in (("train", True), ("val", False), ("test", False)):
        dataset = Ala2TrajectoryDataset(
            positions=processed.positions,
            atomic_numbers=processed.topology.atomic_numbers,
            split_indices=processed.split_indices[split_name],
            metadata=metadata,
        )
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=(split_name == "train" and drop_last_train),
        )
    return loaders, metadata


def _nested_get(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def build_dataloaders(config: Mapping[str, Any]) -> Dict[str, object]:
    """Config-driven dataloader entrypoint used by the standalone trainers."""

    data_cfg = dict(config.get("data", {}))
    dataset_path = data_cfg.get("dataset_path") or data_cfg.get("processed_data_path")
    topology_path = (
        data_cfg.get("topology_path")
        or data_cfg.get("metadata_path")
        or data_cfg.get("metadata_json")
    )
    split_path = data_cfg.get("split_path") or data_cfg.get("split_npz")
    if not dataset_path:
        raise ValueError("Config must provide data.dataset_path for Ala2 pretraining.")
    if not topology_path:
        raise ValueError(
            "Config must provide data.topology_path or data.metadata_path for Ala2 pretraining."
        )

    loaders, metadata = create_dataloaders(
        dataset_path=dataset_path,
        topology_path=topology_path,
        split_path=split_path,
        batch_size=int(data_cfg.get("batch_size", 64)),
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=bool(data_cfg.get("pin_memory", False)),
        drop_last_train=bool(data_cfg.get("drop_last_train", False)),
    )
    metadata = dict(metadata)
    metadata["dataset_path"] = str(dataset_path)
    metadata["topology_path"] = str(topology_path)
    metadata["split_path"] = None if split_path is None else str(split_path)
    metadata["batch_size"] = int(data_cfg.get("batch_size", 64))
    metadata["split_hint"] = {
        "train_fraction": _nested_get(data_cfg, "split", "train_fraction", default=None),
        "val_fraction": _nested_get(data_cfg, "split", "val_fraction", default=None),
        "test_fraction": _nested_get(data_cfg, "split", "test_fraction", default=None),
    }
    return {
        "train": loaders["train"],
        "val": loaders["val"],
        "test": loaders["test"],
        "metadata": metadata,
    }


create_dataloaders_from_config = build_dataloaders
