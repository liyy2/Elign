from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

try:
    # RDKit can emit very noisy stderr logs when sanitization fails
    # (e.g., "Explicit valence for atom ... is greater than permitted").
    # These failures are expected when evaluating invalid generations, so silence them.
    from rdkit import RDLogger  # type: ignore

    RDLogger.DisableLog("rdApp.error")
    RDLogger.DisableLog("rdApp.warning")

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


def graph_largest_fragment_smiles(
    positions: torch.Tensor,
    atom_types: torch.Tensor,
    dataset_info: Dict[str, Any],
) -> Optional[str]:
    """Return the canonical SMILES of the largest fragment for a generated graph.

    This matches `BasicMolecularMetrics.compute_validity` behavior: build the full molecule from
    predicted bonds, then (if valid) re-canonicalize on the *largest connected fragment*.

    Returns None when RDKit sanitization fails.
    """
    try:
        from edm_source.qm9.rdkit_functions import build_molecule, mol2smiles
        from rdkit import Chem
    except ImportError:  # pragma: no cover - RDKit is an external optional dependency
        return None

    mol = build_molecule(positions, atom_types, dataset_info)
    smiles = mol2smiles(mol)
    if smiles is None:
        return None

    try:
        mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
        largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
        return mol2smiles(largest_mol)
    except Exception:
        # Fall back to the original SMILES if fragment extraction fails.
        return smiles


def sample_largest_fragment_smiles(sample: Dict[str, Any], dataset_info: Dict[str, Any]) -> Optional[str]:
    """Convenience wrapper around :func:`graph_largest_fragment_smiles` for stored rollout samples."""
    graph = _sample_to_graph(sample)
    if graph is None:
        return None
    positions, atom_types = graph
    return graph_largest_fragment_smiles(positions, atom_types, dataset_info)


def compute_rdkit_metrics(
    samples: List[Dict[str, Any]],
    dataset_info: Dict[str, Any],
) -> RDKitMetrics:
    """Compute RDKit validity/uniqueness (+ novelty) statistics for rollout samples.

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
        "num_novel": 0,
        "validity": 0.0,
        "uniqueness": 0.0,
        "novelty_frac": 0.0,
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
    novelty_frac = 0.0
    num_novel = 0
    if valid_smiles:
        unique_smiles, uniqueness = metrics_helper.compute_uniqueness(valid_smiles)
        if unique_smiles and getattr(metrics_helper, "dataset_smiles_list", None) is not None:
            # BasicMolecularMetrics.compute_novelty checks membership against a list, which is
            # quadratic for large datasets. Convert once to a set for faster evaluation.
            dataset_smiles = set(metrics_helper.dataset_smiles_list)
            num_novel = sum(1 for smiles in unique_smiles if smiles not in dataset_smiles)
            novelty_frac = num_novel / len(unique_smiles) if unique_smiles else 0.0

    totals.update(
        {
            "num_valid": len(valid_smiles),
            "num_unique": len(unique_smiles),
            "num_novel": num_novel,
            "validity": validity,
            "uniqueness": uniqueness if valid_smiles else 0.0,
            "novelty_frac": novelty_frac,
        }
    )
    return totals
