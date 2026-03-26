from __future__ import annotations

import argparse
import json
import math
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
    np = None
    _NUMPY_IMPORT_ERROR = exc
else:  # pragma: no cover - environment dependent
    _NUMPY_IMPORT_ERROR = None

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
    torch = None
    _TORCH_IMPORT_ERROR = exc
else:  # pragma: no cover - environment dependent
    _TORCH_IMPORT_ERROR = None

try:  # pragma: no cover - environment dependent
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    matplotlib = None
    plt = None

try:  # pragma: no cover - environment dependent
    import wandb
except ModuleNotFoundError:
    wandb = None

from .data import blocked_split_indices as data_blocked_split_indices
from .data import load_processed_dataset
from .topology import build_one_hot_features


DEFAULT_BASINS_DEG: Dict[str, Dict[str, Tuple[float, float]]] = {
    "alpha": {"phi": (-140.0, -20.0), "psi": (-90.0, 60.0)},
    "beta": {"phi": (-180.0, -40.0), "psi": (60.0, 180.0)},
    "left": {"phi": (20.0, 180.0), "psi": (-60.0, 180.0)},
}


@dataclass
class EvaluationResult:
    phi_psi_jsd: float
    free_energy_rmse: float
    basin_pop_l1: float
    reference_basin_populations: Dict[str, float]
    generated_basin_populations: Dict[str, float]
    reference_energy_mean: Optional[float] = None
    generated_energy_mean: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "phi_psi_jsd": float(self.phi_psi_jsd),
            "free_energy_rmse": float(self.free_energy_rmse),
            "basin_pop_l1": float(self.basin_pop_l1),
            "reference_basin_populations": dict(self.reference_basin_populations),
            "generated_basin_populations": dict(self.generated_basin_populations),
        }
        if self.reference_energy_mean is not None:
            payload["reference_energy_mean"] = float(self.reference_energy_mean)
        if self.generated_energy_mean is not None:
            payload["generated_energy_mean"] = float(self.generated_energy_mean)
        return payload


def _require_dependencies() -> None:
    if np is None:
        raise RuntimeError("numpy is required for ala2_scratch.eval") from _NUMPY_IMPORT_ERROR
    if torch is None:
        raise RuntimeError("torch is required for ala2_scratch.eval") from _TORCH_IMPORT_ERROR


def parse_index_spec(spec: Optional[str]) -> Optional[Tuple[int, ...]]:
    if spec is None:
        return None
    values = [part.strip() for part in str(spec).split(",") if part.strip()]
    if not values:
        return None
    out = tuple(int(value) for value in values)
    if len(out) != 4:
        raise ValueError(f"Expected four torsion indices, got {out}")
    return out


def _to_numpy_coords(value: Any) -> np.ndarray:
    _require_dependencies()
    if torch is not None and isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
    elif np is not None and isinstance(value, np.ndarray):
        arr = value
    elif isinstance(value, (list, tuple)):
        arr = np.asarray(value)
    else:
        raise TypeError(f"Unsupported coordinate container: {type(value)!r}")
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected coordinates with shape [frames, atoms, 3], got {arr.shape}")
    return arr.astype(np.float64, copy=False)


def _maybe_extract_mapping_value(mapping: Mapping[str, Any]) -> Optional[np.ndarray]:
    preferred = (
        "coords",
        "coordinates",
        "positions",
        "x",
        "samples",
        "generated",
        "reference",
    )
    for key in preferred:
        if key in mapping:
            try:
                return _to_numpy_coords(mapping[key])
            except Exception:
                continue
    for value in mapping.values():
        if isinstance(value, Mapping):
            found = _maybe_extract_mapping_value(value)
            if found is not None:
                return found
        else:
            try:
                return _to_numpy_coords(value)
            except Exception:
                continue
    return None


def load_coordinate_array(path: str) -> np.ndarray:
    _require_dependencies()
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"Coordinate file not found: {src}")
    if src.suffix == ".npz":
        with np.load(src, allow_pickle=True) as payload:
            array = _maybe_extract_mapping_value({key: payload[key] for key in payload.files})
    elif src.suffix == ".npy":
        array = _to_numpy_coords(np.load(src))
    elif src.suffix in {".pt", ".pth"}:
        payload = torch.load(src, map_location="cpu")
        if isinstance(payload, Mapping):
            array = _maybe_extract_mapping_value(payload)
        else:
            array = _to_numpy_coords(payload)
    else:
        raise ValueError(f"Unsupported coordinate file extension: {src.suffix}")
    if array is None:
        raise ValueError(f"Could not find coordinate tensor in {src}")
    return array


def blocked_split_indices(
    num_frames: int,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    *,
    block_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    _require_dependencies()
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    block_starts = list(range(0, num_frames, block_size))
    n_blocks = len(block_starts)
    if n_blocks < 3:
        return (
            np.arange(num_frames, dtype=np.int64),
            np.asarray([], dtype=np.int64),
            np.asarray([], dtype=np.int64),
        )
    n_val = max(1, int(round(n_blocks * val_fraction)))
    n_test = max(1, int(round(n_blocks * test_fraction)))
    n_val = min(n_val, max(1, n_blocks - 2))
    n_test = min(n_test, max(1, n_blocks - n_val - 1))
    val_blocks = set(range(0, n_val))
    test_blocks = set(range(n_val, n_val + n_test))

    train: List[int] = []
    val: List[int] = []
    test: List[int] = []
    for block_idx, start in enumerate(block_starts):
        end = min(start + block_size, num_frames)
        target = train
        if block_idx in val_blocks:
            target = val
        elif block_idx in test_blocks:
            target = test
        target.extend(range(start, end))
    return (
        np.asarray(train, dtype=np.int64),
        np.asarray(val, dtype=np.int64),
        np.asarray(test, dtype=np.int64),
    )


def iter_coordinate_batches(coords: np.ndarray, indices: Sequence[int], batch_size: int) -> Iterable[np.ndarray]:
    _require_dependencies()
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    idx_array = np.asarray(indices, dtype=np.int64)
    for start in range(0, len(idx_array), batch_size):
        batch_indices = idx_array[start : start + batch_size]
        if batch_indices.size == 0:
            continue
        yield coords[batch_indices]


def dihedral_angles(coords: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    _require_dependencies()
    idx = tuple(int(v) for v in indices)
    if len(idx) != 4:
        raise ValueError(f"Expected 4 indices for a dihedral, got {idx}")
    p0, p1, p2, p3 = [coords[:, i, :] for i in idx]
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2

    b1_norm = b1 / np.clip(np.linalg.norm(b1, axis=1, keepdims=True), 1e-12, None)
    v = b0 - (b0 * b1_norm).sum(axis=1, keepdims=True) * b1_norm
    w = b2 - (b2 * b1_norm).sum(axis=1, keepdims=True) * b1_norm

    x = (v * w).sum(axis=1)
    y = (np.cross(b1_norm, v) * w).sum(axis=1)
    return np.arctan2(y, x)


def compute_phi_psi(coords: np.ndarray, phi_indices: Sequence[int], psi_indices: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    phi = dihedral_angles(coords, phi_indices)
    psi = dihedral_angles(coords, psi_indices)
    return phi, psi


def normalized_hist2d(
    phi: np.ndarray,
    psi: np.ndarray,
    *,
    bins: int = 72,
    range_deg: Tuple[float, float] = (-180.0, 180.0),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    _require_dependencies()
    phi_deg = np.degrees(phi)
    psi_deg = np.degrees(psi)
    finite_mask = np.isfinite(phi_deg) & np.isfinite(psi_deg)
    phi_deg = phi_deg[finite_mask]
    psi_deg = psi_deg[finite_mask]
    xedges = np.linspace(range_deg[0], range_deg[1], bins + 1, dtype=np.float64)
    yedges = np.linspace(range_deg[0], range_deg[1], bins + 1, dtype=np.float64)
    if phi_deg.size == 0 or psi_deg.size == 0:
        raise ValueError("No finite phi/psi torsions were available to build a histogram.")
    hist, xedges, yedges = np.histogram2d(
        phi_deg,
        psi_deg,
        bins=bins,
        range=[[range_deg[0], range_deg[1]], [range_deg[0], range_deg[1]]],
    )
    hist = hist.astype(np.float64, copy=False)
    total = hist.sum()
    if total <= 0:
        raise ValueError("Phi/psi histogram is empty after binning.")
    hist /= total
    return hist, xedges, yedges


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    _require_dependencies()
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / np.clip(p.sum(), eps, None)
    q = q / np.clip(q.sum(), eps, None)
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > eps
        return float(np.sum(a[mask] * (np.log(a[mask] + eps) - np.log(b[mask] + eps))))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def free_energy_surface(hist: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    _require_dependencies()
    energy = -np.log(np.clip(hist, eps, None))
    energy -= float(np.nanmin(energy))
    return energy


def free_energy_rmse(reference_hist: np.ndarray, generated_hist: np.ndarray, eps: float = 1e-12) -> float:
    ref_energy = free_energy_surface(reference_hist, eps=eps)
    gen_energy = free_energy_surface(generated_hist, eps=eps)
    mask = (reference_hist > eps) | (generated_hist > eps)
    if not np.any(mask):
        return 0.0
    diff = ref_energy[mask] - gen_energy[mask]
    return float(np.sqrt(np.mean(diff ** 2)))


def basin_populations(
    phi: np.ndarray,
    psi: np.ndarray,
    basins: Optional[Mapping[str, Mapping[str, Sequence[float]]]] = None,
) -> Dict[str, float]:
    _require_dependencies()
    spec = basins or DEFAULT_BASINS_DEG
    phi_deg = np.degrees(phi)
    psi_deg = np.degrees(psi)
    finite_mask = np.isfinite(phi_deg) & np.isfinite(psi_deg)
    phi_deg = phi_deg[finite_mask]
    psi_deg = psi_deg[finite_mask]
    pops: Dict[str, float] = {}
    total = max(1, phi_deg.shape[0])
    assigned = np.zeros(phi_deg.shape[0], dtype=bool)
    for name, bounds in spec.items():
        phi_min, phi_max = bounds["phi"]
        psi_min, psi_max = bounds["psi"]
        mask = (
            (phi_deg >= float(phi_min))
            & (phi_deg < float(phi_max))
            & (psi_deg >= float(psi_min))
            & (psi_deg < float(psi_max))
        )
        pops[name] = float(mask.sum() / total)
        assigned |= mask
    pops["other"] = float((~assigned).sum() / total)
    return pops


def basin_population_l1(reference: Mapping[str, float], generated: Mapping[str, float]) -> float:
    keys = sorted(set(reference) | set(generated))
    return float(sum(abs(float(reference.get(key, 0.0)) - float(generated.get(key, 0.0))) for key in keys))


def coordinate_batch_rms(coords: np.ndarray) -> float:
    _require_dependencies()
    return float(np.sqrt(np.mean(coords ** 2)))


def smoke_test_processed_data(
    processed_data: str,
    *,
    topology_path: Optional[str] = None,
    phi_indices: Optional[Sequence[int]] = None,
    psi_indices: Optional[Sequence[int]] = None,
    batch_size: int = 8,
    block_size: int = 32,
) -> Dict[str, Any]:
    src = Path(processed_data)
    coords = None
    train_idx = val_idx = test_idx = None
    if src.suffix == ".npz":
        with np.load(src, allow_pickle=False) as payload:
            if {"positions", "train_idx", "val_idx", "test_idx"}.issubset(payload.files):
                coords = np.asarray(payload["positions"], dtype=np.float64)
                train_idx = payload["train_idx"].astype(np.int64, copy=False)
                val_idx = payload["val_idx"].astype(np.int64, copy=False)
                test_idx = payload["test_idx"].astype(np.int64, copy=False)
    if coords is None:
        coords = load_coordinate_array(processed_data)
        fallback_splits = data_blocked_split_indices(coords.shape[0], block_size=block_size)
        train_idx = fallback_splits["train"]
        val_idx = fallback_splits["val"]
        test_idx = fallback_splits["test"]
    batch = next(iter(iter_coordinate_batches(coords, train_idx, batch_size=batch_size)))
    result: Dict[str, Any] = {
        "num_frames": int(coords.shape[0]),
        "num_atoms": int(coords.shape[1]),
        "train_size": int(train_idx.shape[0]),
        "val_size": int(val_idx.shape[0]),
        "test_size": int(test_idx.shape[0]),
        "batch_shape": [int(v) for v in batch.shape],
        "batch_rms": coordinate_batch_rms(batch),
    }
    if phi_indices is not None and psi_indices is not None:
        phi, psi = compute_phi_psi(batch, phi_indices, psi_indices)
        hist, _, _ = normalized_hist2d(phi, psi)
        result["phi_mean_deg"] = float(np.degrees(phi).mean())
        result["psi_mean_deg"] = float(np.degrees(psi).mean())
        result["phi_psi_hist_entropy"] = float(-np.sum(hist * np.log(np.clip(hist, 1e-12, None))))
    return result


def _select_reference_subset(
    reference_coords: np.ndarray,
    sample_count: int,
    *,
    seed: int,
) -> np.ndarray:
    if reference_coords.shape[0] == 0:
        raise ValueError("Reference coordinate array is empty.")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive.")
    rng = np.random.RandomState(seed)
    replace = reference_coords.shape[0] < sample_count
    indices = rng.choice(reference_coords.shape[0], size=sample_count, replace=replace)
    return reference_coords[indices]


def evaluate_reference_vs_generated(
    reference_coords: np.ndarray,
    generated_coords: np.ndarray,
    *,
    phi_indices: Sequence[int],
    psi_indices: Sequence[int],
    bins: int = 72,
    basins: Optional[Mapping[str, Mapping[str, Sequence[float]]]] = None,
    reference_energies: Optional[np.ndarray] = None,
    generated_energies: Optional[np.ndarray] = None,
) -> EvaluationResult:
    ref_phi, ref_psi = compute_phi_psi(reference_coords, phi_indices, psi_indices)
    gen_phi, gen_psi = compute_phi_psi(generated_coords, phi_indices, psi_indices)
    ref_hist, _, _ = normalized_hist2d(ref_phi, ref_psi, bins=bins)
    gen_hist, _, _ = normalized_hist2d(gen_phi, gen_psi, bins=bins)

    ref_basin = basin_populations(ref_phi, ref_psi, basins=basins)
    gen_basin = basin_populations(gen_phi, gen_psi, basins=basins)

    return EvaluationResult(
        phi_psi_jsd=js_divergence(ref_hist, gen_hist),
        free_energy_rmse=free_energy_rmse(ref_hist, gen_hist),
        basin_pop_l1=basin_population_l1(ref_basin, gen_basin),
        reference_basin_populations=ref_basin,
        generated_basin_populations=gen_basin,
        reference_energy_mean=None if reference_energies is None else float(np.mean(reference_energies)),
        generated_energy_mean=None if generated_energies is None else float(np.mean(generated_energies)),
    )


def save_ramachandran_plot(
    reference_coords: np.ndarray,
    generated_coords: np.ndarray,
    *,
    phi_indices: Sequence[int],
    psi_indices: Sequence[int],
    output_path: str,
    bins: int = 72,
) -> Optional[str]:
    if plt is None:
        return None
    ref_phi, ref_psi = compute_phi_psi(reference_coords, phi_indices, psi_indices)
    gen_phi, gen_psi = compute_phi_psi(generated_coords, phi_indices, psi_indices)
    ref_hist, _, _ = normalized_hist2d(ref_phi, ref_psi, bins=bins)
    gen_hist, _, _ = normalized_hist2d(gen_phi, gen_psi, bins=bins)
    diff = gen_hist - ref_hist

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    common_vmax = float(max(ref_hist.max(), gen_hist.max(), 1e-12))
    diff_abs_max = float(max(np.abs(diff).max(), 1e-12))
    panels = (
        (ref_hist, "Reference"),
        (gen_hist, "Generated"),
        (diff, "Generated - Reference"),
    )
    for ax, (panel, title) in zip(axes, panels):
        image = ax.imshow(
            panel.T,
            origin="lower",
            extent=[-180, 180, -180, 180],
            aspect="auto",
            cmap="viridis" if title != "Generated - Reference" else "coolwarm",
            vmin=None if title == "Generated - Reference" else 0.0,
            vmax=None if title == "Generated - Reference" else common_vmax,
        )
        if title == "Generated - Reference":
            image.set_clim(-diff_abs_max, diff_abs_max)
        ax.set_title(title)
        ax.set_xlabel("phi (deg)")
        ax.set_ylabel("psi (deg)")
        fig.colorbar(image, ax=ax, shrink=0.8)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return str(out_path)


def _maybe_load_energies(path: Optional[str]) -> Optional[np.ndarray]:
    if path is None:
        return None
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"Energy file not found: {src}")
    if src.suffix == ".npz":
        with np.load(src, allow_pickle=True) as payload:
            for key in ("energy", "energies", "E"):
                if key in payload:
                    return np.asarray(payload[key], dtype=np.float64)
    elif src.suffix == ".npy":
        return np.asarray(np.load(src), dtype=np.float64)
    elif src.suffix in {".pt", ".pth"} and torch is not None:
        value = torch.load(src, map_location="cpu")
        if isinstance(value, Mapping):
            for key in ("energy", "energies", "E"):
                if key in value:
                    return np.asarray(value[key], dtype=np.float64)
        return np.asarray(value, dtype=np.float64)
    raise ValueError(f"Unsupported energy file format: {src.suffix}")


def _resolve_data_paths(cfg: Mapping[str, Any]) -> Tuple[str, str]:
    data_cfg = cfg.get("data", {})
    dataset_path = data_cfg.get("dataset_path") or data_cfg.get("processed_data_path")
    topology_path = (
        data_cfg.get("topology_path")
        or data_cfg.get("metadata_path")
        or data_cfg.get("metadata_json")
    )
    if not dataset_path or not topology_path:
        raise ValueError(
            "Evaluation requires data.dataset_path and data.topology_path/metadata_path."
        )
    return str(dataset_path), str(topology_path)


def _torch_device(device: Optional[str]) -> "torch.device":
    _require_dependencies()
    requested = str(device or "cpu")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        requested = "cpu"
    return torch.device(requested)


def _node_features_from_atomic_numbers(
    atomic_numbers: Sequence[int],
    *,
    device: "torch.device",
) -> "torch.Tensor":
    _require_dependencies()
    one_hot, _ = build_one_hot_features(atomic_numbers)
    return torch.as_tensor(one_hot, dtype=torch.float32, device=device)


def _extract_sample_positions(sample_output: Any) -> np.ndarray:
    if hasattr(sample_output, "x0"):
        return _to_numpy_coords(getattr(sample_output, "x0"))
    if isinstance(sample_output, Mapping):
        for key in ("positions", "samples", "x0", "coordinates", "x"):
            if key in sample_output:
                return _to_numpy_coords(sample_output[key])
    if isinstance(sample_output, (tuple, list)) and sample_output:
        return _to_numpy_coords(sample_output[0])
    return _to_numpy_coords(sample_output)


def _sample_generated_coords(
    *,
    model: Any,
    diffusion: Any,
    atomic_numbers: Sequence[int],
    sample_count: int,
    batch_size: int,
    device: "torch.device",
) -> np.ndarray:
    _require_dependencies()
    node_features = _node_features_from_atomic_numbers(atomic_numbers, device=device)
    chunks: List[np.ndarray] = []
    remaining = int(sample_count)
    while remaining > 0:
        current_batch = min(batch_size, remaining)
        with torch.no_grad():
            sample_output = diffusion.sample(
                model=model,
                batch_size=current_batch,
                node_features=node_features,
                device=device,
            )
        chunks.append(_extract_sample_positions(sample_output))
        remaining -= current_batch
    return np.concatenate(chunks, axis=0)


def _maybe_compute_mlff_energy(
    cfg: Mapping[str, Any],
    coords: np.ndarray,
    atomic_numbers: Sequence[int],
) -> Optional[np.ndarray]:
    mlff_cfg = cfg.get("mlff")
    if not mlff_cfg:
        return None
    try:
        from .mlff_energy import MLFFEnergyConfig, MLFFEnergyOracle
    except Exception as exc:
        raise RuntimeError("Failed to import MLFF energy dependencies for evaluation.") from exc
    try:
        device = str(mlff_cfg.get("device") or cfg.get("device") or "cpu")
        oracle = MLFFEnergyOracle(atomic_numbers, MLFFEnergyConfig(**mlff_cfg))
        coord_tensor = torch.as_tensor(coords, dtype=torch.float32, device=device)
        energies = oracle(coord_tensor)
        return energies.detach().cpu().numpy().astype(np.float64, copy=False)
    except Exception as exc:
        raise RuntimeError("MLFF energy evaluation failed.") from exc


def evaluate_runtime(
    *,
    cfg: Mapping[str, Any],
    model: Any,
    diffusion: Any,
    output_dir: str | Path,
) -> Dict[str, Any]:
    _require_dependencies()
    dataset_path, topology_path = _resolve_data_paths(cfg)
    processed = load_processed_dataset(dataset_path, topology_path)
    reference_indices = processed.split_indices["test"]
    reference_coords = processed.positions[reference_indices]
    atomic_numbers = processed.topology.atomic_numbers

    eval_cfg = cfg.get("eval", {})
    sample_count = int(
        eval_cfg.get(
            "sample_count",
            cfg.get("train", {}).get("num_eval_samples", cfg.get("num_eval_samples", 256)),
        )
    )
    batch_size = int(eval_cfg.get("sample_batch_size", min(sample_count, 64)))
    device = _torch_device(cfg.get("device"))
    if hasattr(model, "eval"):
        model.eval()
    if hasattr(diffusion, "eval"):
        diffusion.eval()

    generated_coords = _sample_generated_coords(
        model=model,
        diffusion=diffusion,
        atomic_numbers=atomic_numbers,
        sample_count=sample_count,
        batch_size=max(1, batch_size),
        device=device,
    )
    reference_subset = _select_reference_subset(
        reference_coords,
        generated_coords.shape[0],
        seed=int(eval_cfg.get("reference_seed", cfg.get("seed", 0))),
    )

    result = evaluate_reference_vs_generated(
        reference_subset,
        generated_coords,
        phi_indices=processed.topology.phi_indices,
        psi_indices=processed.topology.psi_indices,
        bins=int(eval_cfg.get("histogram_bins", eval_cfg.get("bins", 72))),
        reference_energies=_maybe_compute_mlff_energy(cfg, reference_subset, atomic_numbers),
        generated_energies=_maybe_compute_mlff_energy(cfg, generated_coords, atomic_numbers),
    )
    metrics = result.to_dict()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    generated_path = output_path / "generated_samples.npz"
    np.savez_compressed(generated_path, positions=generated_coords)
    metrics["generated_path"] = str(generated_path)

    image_path = save_ramachandran_plot(
        reference_subset,
        generated_coords,
        phi_indices=processed.topology.phi_indices,
        psi_indices=processed.topology.psi_indices,
        output_path=str(output_path / "ramachandran.png"),
        bins=int(eval_cfg.get("histogram_bins", eval_cfg.get("bins", 72))),
    )
    if image_path is not None:
        metrics["ramachandran_plot"] = image_path
    return metrics


def evaluate_model(
    *,
    cfg: Mapping[str, Any],
    model: Any,
    diffusion: Any,
    output_dir: str | Path,
) -> Dict[str, Any]:
    return evaluate_runtime(cfg=cfg, model=model, diffusion=diffusion, output_dir=output_dir)


def evaluate_checkpoint(
    checkpoint_path: str,
    config: Optional[Mapping[str, Any]] = None,
    output_dir: Optional[str] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    _require_dependencies()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    runtime_cfg = dict(config or checkpoint.get("config") or {})
    if "config" in checkpoint and isinstance(checkpoint["config"], Mapping):
        merged_cfg = dict(checkpoint["config"])
        for key, value in runtime_cfg.items():
            if isinstance(value, Mapping) and isinstance(merged_cfg.get(key), Mapping):
                merged_cfg[key] = {**merged_cfg[key], **value}
            else:
                merged_cfg[key] = value
        runtime_cfg = merged_cfg
    if device is not None:
        runtime_cfg["device"] = device

    dataset_path, topology_path = _resolve_data_paths(runtime_cfg)
    processed = load_processed_dataset(dataset_path, topology_path)
    metadata = {
        "atomic_numbers": processed.topology.atomic_numbers,
        "phi_indices": processed.topology.phi_indices,
        "psi_indices": processed.topology.psi_indices,
    }
    example_batch = {
        "positions": torch.as_tensor(processed.positions[:1], dtype=torch.float32),
        "node_features": torch.as_tensor(
            build_one_hot_features(processed.topology.atomic_numbers)[0],
            dtype=torch.float32,
        ),
    }

    from .diffusion import build_diffusion
    from .model import build_model

    runtime_device = _torch_device(runtime_cfg.get("device"))
    model = build_model(config=runtime_cfg, metadata=metadata, example_batch=example_batch)
    model_state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint/model mismatch for '{checkpoint_path}'. Missing={missing}, unexpected={unexpected}"
        )
    model.to(runtime_device)
    diffusion = build_diffusion(
        runtime_cfg,
        metadata=metadata,
        example_batch=example_batch,
        model=model,
    )
    if isinstance(diffusion, torch.nn.Module) and checkpoint.get("diffusion_state_dict"):
        diff_missing, diff_unexpected = diffusion.load_state_dict(
            checkpoint["diffusion_state_dict"],
            strict=False,
        )
        if diff_missing or diff_unexpected:
            raise RuntimeError(
                f"Checkpoint/diffusion mismatch for '{checkpoint_path}'. "
                f"Missing={diff_missing}, unexpected={diff_unexpected}"
            )
        diffusion.to(runtime_device)

    return evaluate_runtime(
        cfg=runtime_cfg,
        model=model,
        diffusion=diffusion,
        output_dir=output_dir or tempfile.mkdtemp(prefix="ala2_eval_"),
    )


def _maybe_init_wandb(args: argparse.Namespace, config_payload: Dict[str, Any]):
    if not getattr(args, "wandb_enabled", False) or wandb is None:
        return None
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        name=args.wandb_name,
        tags=[tag for tag in (args.wandb_tags or "").split(",") if tag],
        mode=args.wandb_mode,
        config=config_payload,
        job_type="eval" if not args.smoke_check_data else "data_smoke",
    )


def _dump_json(payload: Dict[str, Any], output_json: Optional[str]) -> None:
    if output_json is None:
        return
    path = Path(output_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone Ala2 evaluation and data smoke checks.")
    parser.add_argument("--reference", type=str, default=None, help="Reference trajectory coordinates (.npz/.npy/.pt).")
    parser.add_argument("--generated", type=str, default=None, help="Generated sample coordinates (.npz/.npy/.pt).")
    parser.add_argument("--reference-energies", type=str, default=None, help="Optional reference energy file.")
    parser.add_argument("--generated-energies", type=str, default=None, help="Optional generated energy file.")
    parser.add_argument("--processed-data", type=str, default=None, help="Processed trajectory file for smoke checks.")
    parser.add_argument("--topology-path", type=str, default=None, help="Optional topology path for processed-data smoke checks.")
    parser.add_argument("--smoke-check-data", action="store_true", help="Validate processed data load/split/batch/metric path.")
    parser.add_argument("--phi-indices", type=str, default=None, help="Comma-separated phi torsion indices.")
    parser.add_argument("--psi-indices", type=str, default=None, help="Comma-separated psi torsion indices.")
    parser.add_argument("--bins", type=int, default=72, help="Number of histogram bins for phi/psi.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for processed-data smoke checks.")
    parser.add_argument("--block-size", type=int, default=32, help="Blocked split size for smoke checks.")
    parser.add_argument("--output-json", type=str, default=None, help="Optional JSON output path.")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional directory for plot outputs.")
    parser.add_argument("--wandb-enabled", action="store_true", help="Enable W&B logging.")
    parser.add_argument("--wandb-project", type=str, default="ala2-diffusion", help="W&B project name.")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity name.")
    parser.add_argument("--wandb-group", type=str, default="ala2-300k", help="W&B group name.")
    parser.add_argument("--wandb-name", type=str, default=None, help="W&B run name.")
    parser.add_argument("--wandb-tags", type=str, default="", help="Comma-separated W&B tags.")
    parser.add_argument("--wandb-mode", type=str, default=os.environ.get("WANDB_MODE", "offline"))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    phi_indices = parse_index_spec(args.phi_indices)
    psi_indices = parse_index_spec(args.psi_indices)
    config_payload = {key: getattr(args, key) for key in vars(args)}
    run = _maybe_init_wandb(args, config_payload)

    try:
        if args.smoke_check_data:
            if args.processed_data is None:
                raise ValueError("--processed-data is required when --smoke-check-data is set")
            result = smoke_test_processed_data(
                args.processed_data,
                topology_path=args.topology_path,
                phi_indices=phi_indices,
                psi_indices=psi_indices,
                batch_size=args.batch_size,
                block_size=args.block_size,
            )
            payload = {"mode": "data_smoke", **result}
            print("SMOKE_TEST_OK", json.dumps(payload, sort_keys=True))
            if run is not None:
                run.log({f"smoke/{key}": value for key, value in payload.items() if key != "mode"})
            _dump_json(payload, args.output_json)
            return 0

        if args.reference is None or args.generated is None:
            raise ValueError("--reference and --generated are required for evaluation mode")
        if phi_indices is None or psi_indices is None:
            raise ValueError("--phi-indices and --psi-indices are required for evaluation mode")

        reference_coords = load_coordinate_array(args.reference)
        generated_coords = load_coordinate_array(args.generated)
        result = evaluate_reference_vs_generated(
            reference_coords,
            generated_coords,
            phi_indices=phi_indices,
            psi_indices=psi_indices,
            bins=args.bins,
            reference_energies=_maybe_load_energies(args.reference_energies),
            generated_energies=_maybe_load_energies(args.generated_energies),
        )
        payload = result.to_dict()

        image_path = None
        if args.output_dir is not None:
            image_path = save_ramachandran_plot(
                reference_coords,
                generated_coords,
                phi_indices=phi_indices,
                psi_indices=psi_indices,
                output_path=str(Path(args.output_dir) / "ramachandran.png"),
                bins=args.bins,
            )
        if image_path is not None:
            payload["ramachandran_plot"] = image_path

        print(json.dumps(payload, indent=2, sort_keys=True))
        _dump_json(payload, args.output_json)

        if run is not None:
            run.log({f"eval/{key}": value for key, value in payload.items() if key != "ramachandran_plot"})
            if image_path is not None:
                run.log({"eval/ramachandran": wandb.Image(image_path)})
        return 0
    finally:
        if run is not None:
            run.finish()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
