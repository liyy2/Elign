from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

try:
    from edm_source.qm9.rdkit_functions import BasicMolecularMetrics
except ImportError:  # pragma: no cover - RDKit is an external optional dependency
    BasicMolecularMetrics = None  # type: ignore[assignment]


RDKitMetrics = Dict[str, Any]


def _to_cpu_tensor(value: Any, dtype: torch.dtype) -> torch.Tensor:
    """Normalize heterogeneous inputs (numpy arrays, tensors, lists) to CPU tensors."""
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    return tensor.detach().to(dtype=dtype, device=torch.device("cpu"))


def _sample_to_graph(sample: Dict[str, Any]) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Convert a stored rollout sample into the (positions, atom_types) pair RDKit expects."""
    if "positions" not in sample or "atom_types" not in sample:
        return None

    positions = _to_cpu_tensor(sample["positions"], torch.float32)
    atom_types = _to_cpu_tensor(sample["atom_types"], torch.long)

    if positions.ndim != 2 or atom_types.ndim != 1:
        return None

    max_atoms = min(positions.shape[0], atom_types.shape[0])
    num_atoms = int(sample.get("num_atoms", max_atoms))
    num_atoms = max(0, min(num_atoms, max_atoms))

    if num_atoms == 0:
        return None

    return positions[:num_atoms], atom_types[:num_atoms]


def compute_rdkit_metrics(
    samples: List[Dict[str, Any]],
    dataset_info: Dict[str, Any],
) -> RDKitMetrics:
    """Compute RDKit validity/uniqueness statistics for rollout samples.

    Args:
        samples: List of rollout sample dictionaries produced by eval_verl_rollout.py.
        dataset_info: Dataset metadata used by BasicMolecularMetrics.

    Returns:
        Dictionary containing aggregate counts/ratios and an optional ``error`` message if
        RDKit-based metrics could not be computed (e.g., RDKit not installed).
    """

    totals: RDKitMetrics = {
        "num_total": len(samples),
        "num_valid": 0,
        "num_unique": 0,
        "validity": 0.0,
        "uniqueness": 0.0,
    }

    if not samples:
        return totals

    if BasicMolecularMetrics is None:
        totals["error"] = "RDKit / BasicMolecularMetrics is unavailable. Please install RDKit."
        return totals

    graphs: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for sample in samples:
        graph = _sample_to_graph(sample)
        if graph is not None:
            graphs.append(graph)

    if not graphs:
        totals["error"] = "No valid samples contained positions + atom types for RDKit evaluation."
        return totals

    metrics_helper = BasicMolecularMetrics(dataset_info)
    valid_smiles, validity = metrics_helper.compute_validity(graphs)
    unique_smiles: List[str] = []
    uniqueness = 0.0
    if valid_smiles:
        unique_smiles, uniqueness = metrics_helper.compute_uniqueness(valid_smiles)

    totals.update(
        {
            "num_valid": len(valid_smiles),
            "num_unique": len(unique_smiles),
            "validity": validity,
            "uniqueness": uniqueness if valid_smiles else 0.0,
        }
    )
    return totals
