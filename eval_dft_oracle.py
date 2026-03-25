import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent
EDM_SOURCE_ROOT = REPO_ROOT / "edm_source"
for path in (REPO_ROOT, EDM_SOURCE_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from edm_source.configs.datasets_config import get_dataset_info

HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANGSTROM = 0.529177210903
HARTREE_PER_BOHR_TO_EV_PER_A = HARTREE_TO_EV / BOHR_TO_ANGSTROM

DEFAULT_ATOMIC_SPINS = {
    # PySCF uses spin = 2S (Nalpha - Nbeta) = number of unpaired electrons.
    # These are common neutral ground-state spins for GEOM/QM9 elements.
    "H": 1,
    "C": 2,
    "N": 3,
    "O": 2,
    "F": 1,
}


@dataclass(frozen=True)
class DFTConfig:
    xc: str
    basis: str
    ecp: Optional[str]
    charge: int
    spin: Optional[int]
    max_cycle: int
    conv_tol: float
    grids_level: int
    density_fit: bool
    retry_newton: bool
    accept_unconverged: bool
    min_interatomic_distance: float


def _load_samples(path: Path) -> List[Dict[str, Any]]:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "samples" in payload:
        samples = payload["samples"]
    else:
        samples = payload
    if not isinstance(samples, list):
        raise TypeError(f"Unsupported samples payload at {path} (expected list or dict with 'samples').")
    return samples


def _to_numpy_positions(positions: Any, num_atoms: int) -> np.ndarray:
    if isinstance(positions, torch.Tensor):
        positions = positions.detach().cpu().numpy()
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must have shape (N, 3); got {positions.shape}")
    if positions.shape[0] < num_atoms:
        raise ValueError(f"positions has {positions.shape[0]} rows but num_atoms={num_atoms}")
    return positions[:num_atoms].copy()


def _to_numpy_atom_types(atom_types: Any, num_atoms: int) -> np.ndarray:
    if isinstance(atom_types, torch.Tensor):
        atom_types = atom_types.detach().cpu().numpy()
    atom_types = np.asarray(atom_types, dtype=np.int64).reshape(-1)
    if atom_types.shape[0] < num_atoms:
        raise ValueError(f"atom_types has {atom_types.shape[0]} entries but num_atoms={num_atoms}")
    return atom_types[:num_atoms].copy()


def _atom_symbols_from_types(
    atom_types: np.ndarray, dataset: str, remove_h: bool = False
) -> List[str]:
    dataset_info = get_dataset_info(dataset, remove_h)
    decoder = dataset_info.get("atom_decoder")
    if not isinstance(decoder, list):
        raise ValueError(f"Dataset info for '{dataset}' is missing atom_decoder.")
    symbols: List[str] = []
    for idx in atom_types.tolist():
        if idx < 0 or idx >= len(decoder):
            raise ValueError(f"Atom type index {idx} out of range for dataset '{dataset}'.")
        symbols.append(str(decoder[idx]))
    return symbols


def _count_heavy_atoms(symbols: Sequence[str]) -> int:
    return int(sum(1 for s in symbols if s.upper() != "H"))


def _min_pairwise_distance(positions: np.ndarray) -> float:
    if positions.shape[0] <= 1:
        return float("inf")
    diff = positions[:, None, :] - positions[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist, np.inf)
    return float(dist.min())


def _infer_spin(symbols: Sequence[str], charge: int) -> int:
    try:
        from pyscf.data.elements import charge as element_charge
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "PySCF is required (install into the `edm` conda env: "
            "`conda install -n edm -c conda-forge pyscf`)."
        ) from exc

    nelec = int(sum(int(element_charge(sym)) for sym in symbols) - int(charge))
    if nelec < 0:
        raise ValueError(f"Invalid electron count inferred (nelec={nelec}). Check charge/symbols.")
    return int(nelec % 2)


def _default_atomic_spin(symbol: str) -> Optional[int]:
    return DEFAULT_ATOMIC_SPINS.get(str(symbol).strip().upper())


def _build_mol(
    symbols: Sequence[str],
    positions_angstrom: np.ndarray,
    dft_cfg: DFTConfig,
) -> Any:
    try:
        from pyscf import gto
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "PySCF is required (install into the `edm` conda env: "
            "`conda install -n edm -c conda-forge pyscf`)."
        ) from exc

    spin = dft_cfg.spin
    if spin is None:
        spin = _infer_spin(symbols, dft_cfg.charge)

    mol = gto.Mole()
    mol.atom = [(str(sym), tuple(map(float, xyz))) for sym, xyz in zip(symbols, positions_angstrom)]
    mol.unit = "Angstrom"
    mol.basis = dft_cfg.basis
    if dft_cfg.ecp:
        mol.ecp = dft_cfg.ecp
    mol.charge = int(dft_cfg.charge)
    mol.spin = int(spin)
    mol.verbose = 0
    mol.build()
    return mol


def _run_single_point(
    mol: Any,
    dft_cfg: DFTConfig,
) -> Tuple[float, np.ndarray, bool]:
    """Return (energy_hartree, gradient_hartree_per_bohr, converged)."""
    try:
        from pyscf import dft
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "PySCF is required (install into the `edm` conda env: "
            "`conda install -n edm -c conda-forge pyscf`)."
        ) from exc

    is_closed_shell = int(getattr(mol, "spin", 0)) == 0
    if is_closed_shell:
        mf = dft.RKS(mol)
    else:
        mf = dft.UKS(mol)

    mf.xc = str(dft_cfg.xc)
    mf.conv_tol = float(dft_cfg.conv_tol)
    mf.max_cycle = int(dft_cfg.max_cycle)
    mf.verbose = 0
    try:
        mf.grids.level = int(dft_cfg.grids_level)
    except Exception:
        pass

    if dft_cfg.density_fit:
        try:
            mf = mf.density_fit()
        except Exception:
            pass

    energy = float(mf.kernel())
    converged = bool(getattr(mf, "converged", False))

    if not converged and dft_cfg.retry_newton:
        try:
            newton_mf = mf.newton()
            newton_mf.conv_tol = float(dft_cfg.conv_tol)
            newton_mf.max_cycle = int(dft_cfg.max_cycle)
            newton_mf.verbose = 0
            energy = float(newton_mf.kernel())
            converged = bool(getattr(newton_mf, "converged", False))
            mf = newton_mf
        except Exception:
            pass

    grad_obj = mf.nuc_grad_method()
    gradient = np.asarray(grad_obj.kernel(), dtype=np.float64)

    return energy, gradient, converged


def _evaluate_one(
    job: Dict[str, Any],
    dft_cfg: DFTConfig,
    atom_ref_energies_hartree: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    mol_id = int(job["mol_id"])
    symbols = list(job["symbols"])
    positions = np.asarray(job["positions_angstrom"], dtype=np.float64)

    num_atoms = int(len(symbols))
    heavy_atoms = _count_heavy_atoms(symbols)

    min_dist = _min_pairwise_distance(positions)
    if min_dist < float(dft_cfg.min_interatomic_distance):
        return {
            "mol_id": mol_id,
            "ok": False,
            "error": f"min_interatomic_distance={min_dist:.3f} Å < {dft_cfg.min_interatomic_distance:.3f} Å",
            "num_atoms": num_atoms,
            "num_heavy_atoms": heavy_atoms,
            "min_interatomic_distance_angstrom": min_dist,
        }

    try:
        mol = _build_mol(symbols, positions, dft_cfg)
        e_h, grad_h_per_bohr, converged = _run_single_point(mol, dft_cfg)
    except Exception as exc:
        return {
            "mol_id": mol_id,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "num_atoms": num_atoms,
            "num_heavy_atoms": heavy_atoms,
            "min_interatomic_distance_angstrom": min_dist,
        }

    forces_ev_per_a = (-grad_h_per_bohr) * HARTREE_PER_BOHR_TO_EV_PER_A
    per_atom_norm = np.linalg.norm(forces_ev_per_a, axis=-1)
    force_rms = float(math.sqrt(float(np.mean(per_atom_norm**2))))
    force_max = float(np.max(per_atom_norm)) if per_atom_norm.size else 0.0
    force_mean = float(np.mean(per_atom_norm)) if per_atom_norm.size else 0.0
    heavy_mask = np.asarray([sym.upper() != "H" for sym in symbols], dtype=bool)
    if per_atom_norm.size and heavy_mask.any():
        heavy_norm = per_atom_norm[heavy_mask]
        force_rms_heavy = float(math.sqrt(float(np.mean(heavy_norm**2))))
        force_max_heavy = float(np.max(heavy_norm))
        force_mean_heavy = float(np.mean(heavy_norm))
    else:
        force_rms_heavy = 0.0
        force_max_heavy = 0.0
        force_mean_heavy = 0.0

    ok = bool(converged or dft_cfg.accept_unconverged)
    result: Dict[str, Any] = {
        "mol_id": mol_id,
        "ok": ok,
        "converged": bool(converged),
        "num_atoms": num_atoms,
        "num_heavy_atoms": heavy_atoms,
        "min_interatomic_distance_angstrom": min_dist,
        "energy_hartree": float(e_h),
        "energy_ev": float(e_h * HARTREE_TO_EV),
        "energy_ev_per_atom": float(e_h * HARTREE_TO_EV / max(num_atoms, 1)),
        "energy_ev_per_heavy_atom": float(e_h * HARTREE_TO_EV / max(heavy_atoms, 1)),
        "force_rms_ev_per_a": force_rms,
        "force_max_ev_per_a": force_max,
        "force_mean_ev_per_a": force_mean,
        "force_rms_ev_per_a_heavy": force_rms_heavy,
        "force_max_ev_per_a_heavy": force_max_heavy,
        "force_mean_ev_per_a_heavy": force_mean_heavy,
    }

    if atom_ref_energies_hartree:
        missing = [sym for sym in symbols if sym not in atom_ref_energies_hartree]
        if not missing:
            e_atoms = float(sum(float(atom_ref_energies_hartree[sym]) for sym in symbols))
            atomization_ev = float((e_atoms - float(e_h)) * HARTREE_TO_EV)
            result.update(
                {
                    "atomic_reference_energy_hartree": e_atoms,
                    "atomization_energy_ev": atomization_ev,
                    "atomization_energy_ev_per_atom": atomization_ev / max(num_atoms, 1),
                    "atomization_energy_ev_per_heavy_atom": atomization_ev / max(heavy_atoms, 1),
                }
            )
    return result


def _summarize(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    ok_results = [r for r in results if r.get("ok")]
    ok = len(ok_results)
    failed = total - ok

    num_atoms = [int(r.get("num_atoms", 0)) for r in results]
    num_heavy = [int(r.get("num_heavy_atoms", 0)) for r in results]
    num_atoms_ok = [int(r.get("num_atoms", 0)) for r in ok_results]
    num_heavy_ok = [int(r.get("num_heavy_atoms", 0)) for r in ok_results]

    def _mean(xs: Sequence[float]) -> Optional[float]:
        if not xs:
            return None
        return float(np.mean(np.asarray(xs, dtype=np.float64)))

    def _median(xs: Sequence[float]) -> Optional[float]:
        if not xs:
            return None
        return float(np.median(np.asarray(xs, dtype=np.float64)))

    energies_per_atom = [float(r["energy_ev_per_atom"]) for r in ok_results if "energy_ev_per_atom" in r]
    energies_per_heavy = [
        float(r["energy_ev_per_heavy_atom"]) for r in ok_results if "energy_ev_per_heavy_atom" in r
    ]
    atomization_per_atom = [
        float(r["atomization_energy_ev_per_atom"])
        for r in ok_results
        if "atomization_energy_ev_per_atom" in r
    ]
    atomization_per_heavy = [
        float(r["atomization_energy_ev_per_heavy_atom"])
        for r in ok_results
        if "atomization_energy_ev_per_heavy_atom" in r
    ]
    force_rms = [float(r["force_rms_ev_per_a"]) for r in ok_results if "force_rms_ev_per_a" in r]
    force_max = [float(r["force_max_ev_per_a"]) for r in ok_results if "force_max_ev_per_a" in r]
    force_rms_heavy = [
        float(r["force_rms_ev_per_a_heavy"]) for r in ok_results if "force_rms_ev_per_a_heavy" in r
    ]
    force_max_heavy = [
        float(r["force_max_ev_per_a_heavy"]) for r in ok_results if "force_max_ev_per_a_heavy" in r
    ]

    return {
        "num_total": total,
        "num_ok": ok,
        "num_failed": failed,
        "avg_num_atoms": _mean([float(x) for x in num_atoms]),
        "avg_num_heavy_atoms": _mean([float(x) for x in num_heavy]),
        "avg_num_atoms_ok": _mean([float(x) for x in num_atoms_ok]),
        "avg_num_heavy_atoms_ok": _mean([float(x) for x in num_heavy_ok]),
        "energy_ev_per_atom_mean": _mean(energies_per_atom),
        "energy_ev_per_atom_median": _median(energies_per_atom),
        "energy_ev_per_heavy_atom_mean": _mean(energies_per_heavy),
        "energy_ev_per_heavy_atom_median": _median(energies_per_heavy),
        "atomization_energy_ev_per_atom_mean": _mean(atomization_per_atom),
        "atomization_energy_ev_per_atom_median": _median(atomization_per_atom),
        "atomization_energy_ev_per_heavy_atom_mean": _mean(atomization_per_heavy),
        "atomization_energy_ev_per_heavy_atom_median": _median(atomization_per_heavy),
        "force_rms_ev_per_a_mean": _mean(force_rms),
        "force_rms_ev_per_a_median": _median(force_rms),
        "force_max_ev_per_a_mean": _mean(force_max),
        "force_max_ev_per_a_median": _median(force_max),
        "force_rms_ev_per_a_heavy_mean": _mean(force_rms_heavy),
        "force_rms_ev_per_a_heavy_median": _median(force_rms_heavy),
        "force_max_ev_per_a_heavy_mean": _mean(force_max_heavy),
        "force_max_ev_per_a_heavy_median": _median(force_max_heavy),
    }


def _compare(baseline_summary: Dict[str, Any], rl_summary: Dict[str, Any]) -> Dict[str, Any]:
    def _delta(metric: str) -> Optional[float]:
        b = baseline_summary.get(metric)
        r = rl_summary.get(metric)
        if b is None or r is None:
            return None
        return float(b) - float(r)

    def _delta_rl_minus_baseline(metric: str) -> Optional[float]:
        b = baseline_summary.get(metric)
        r = rl_summary.get(metric)
        if b is None or r is None:
            return None
        return float(r) - float(b)

    return {
        "delta_energy_ev_per_atom_mean": _delta("energy_ev_per_atom_mean"),
        "delta_energy_ev_per_heavy_atom_mean": _delta("energy_ev_per_heavy_atom_mean"),
        "delta_atomization_energy_ev_per_atom_mean": _delta_rl_minus_baseline(
            "atomization_energy_ev_per_atom_mean"
        ),
        "delta_atomization_energy_ev_per_heavy_atom_mean": _delta_rl_minus_baseline(
            "atomization_energy_ev_per_heavy_atom_mean"
        ),
        "delta_force_rms_ev_per_a_mean": _delta("force_rms_ev_per_a_mean"),
        "delta_force_max_ev_per_a_mean": _delta("force_max_ev_per_a_mean"),
        "delta_force_rms_ev_per_a_heavy_mean": _delta("force_rms_ev_per_a_heavy_mean"),
        "delta_force_max_ev_per_a_heavy_mean": _delta("force_max_ev_per_a_heavy_mean"),
        "delta_avg_num_atoms": _delta("avg_num_atoms"),
        "delta_avg_num_heavy_atoms": _delta("avg_num_heavy_atoms"),
        "note": (
            "Energy/force deltas are baseline - RL (positive => RL lower/better). "
            "Atomization-energy deltas are RL - baseline (positive => RL higher/better)."
        ),
    }


def _size_stats(samples: Sequence[Dict[str, Any]], dataset: str, remove_h: bool) -> Dict[str, Any]:
    counts: List[int] = []
    heavy_counts: List[int] = []
    for sample in samples:
        num_atoms = int(sample.get("num_atoms") or sample.get("n_atoms") or sample.get("num_nodes") or 0)
        if num_atoms <= 0:
            positions = sample.get("positions")
            if positions is None:
                continue
            if isinstance(positions, torch.Tensor):
                num_atoms = int(positions.shape[0])
            else:
                num_atoms = int(np.asarray(positions).shape[0])

        counts.append(num_atoms)
        try:
            atom_types = _to_numpy_atom_types(sample["atom_types"], num_atoms)
            symbols = _atom_symbols_from_types(atom_types, dataset=dataset, remove_h=remove_h)
            heavy_counts.append(_count_heavy_atoms(symbols))
        except Exception:
            continue

    def _mean_int(xs: Sequence[int]) -> Optional[float]:
        if not xs:
            return None
        return float(np.mean(np.asarray(xs, dtype=np.float64)))

    return {
        "num_total": len(samples),
        "avg_num_atoms": _mean_int(counts),
        "avg_num_heavy_atoms": _mean_int(heavy_counts) if heavy_counts else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate generated molecules with a PySCF DFT oracle. "
            "Loads two rollout files (baseline vs RL), runs single-point DFT energies + gradients, "
            "and reports energy/force statistics plus average molecule size."
        )
    )
    parser.add_argument("--baseline-samples", type=str, required=True, help="torch.load()-able file with samples list.")
    parser.add_argument("--rl-samples", type=str, required=True, help="torch.load()-able file with samples list.")
    parser.add_argument("--dataset", type=str, default="geom", choices=["geom", "qm9"], help="Atom-type decoder.")
    parser.add_argument("--remove-h", action="store_true", help="Use the remove_h dataset decoder (QM9 only).")
    parser.add_argument("--num-eval", type=int, default=50, help="Number of molecules to evaluate per model.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for subsampling molecules from the rollouts.")
    parser.add_argument("--xc", type=str, default="PBE", help="DFT functional string for PySCF (e.g., PBE, B3LYP).")
    parser.add_argument("--basis", type=str, default="sto-3g", help="Basis set (e.g., sto-3g, 6-31g*, def2-svp).")
    parser.add_argument("--ecp", type=str, default=None, help="Optional ECP name (e.g., def2-svp, lanl2dz).")
    parser.add_argument("--charge", type=int, default=0, help="Total molecular charge.")
    parser.add_argument("--spin", type=int, default=None, help="2S (Nalpha-Nbeta). Default inferred from electrons.")
    parser.add_argument("--max-cycle", type=int, default=50, help="SCF max cycles.")
    parser.add_argument("--conv-tol", type=float, default=1e-7, help="SCF convergence tolerance.")
    parser.add_argument("--grids-level", type=int, default=3, help="Numerical integration grid level.")
    parser.add_argument("--density-fit", action="store_true", help="Enable density fitting when available.")
    parser.add_argument("--no-retry-newton", dest="retry_newton", action="store_false", help="Disable Newton retry.")
    parser.set_defaults(retry_newton=True)
    parser.add_argument(
        "--accept-unconverged",
        action="store_true",
        help="Record SCF results even if mf.converged is False (still flagged).",
    )
    parser.add_argument(
        "--min-interatomic-distance",
        type=float,
        default=0.55,
        help="Skip molecules with any interatomic distance below this threshold (Å).",
    )
    parser.add_argument("--num-workers", type=int, default=1, help="Parallel worker processes for DFT.")
    parser.add_argument(
        "--min-atoms",
        type=int,
        default=None,
        help="Optional filter: only evaluate molecules with >= this many atoms.",
    )
    parser.add_argument(
        "--max-atoms",
        type=int,
        default=None,
        help="Optional filter: only evaluate molecules with <= this many atoms.",
    )
    parser.add_argument(
        "--min-heavy-atoms",
        type=int,
        default=None,
        help="Optional filter: only evaluate molecules with >= this many heavy atoms.",
    )
    parser.add_argument(
        "--max-heavy-atoms",
        type=int,
        default=None,
        help="Optional filter: only evaluate molecules with <= this many heavy atoms.",
    )
    parser.add_argument(
        "--match-on",
        nargs="*",
        default=[],
        choices=["atoms", "heavy"],
        help=(
            "Optional: size-match baseline and RL *before* subsampling. "
            "Pass one or both of {atoms, heavy}. Example: `--match-on heavy` forces identical "
            "heavy-atom-count distributions (up to availability)."
        ),
    )
    parser.add_argument(
        "--match-atom-bin",
        type=int,
        default=1,
        help="Bin size (atoms) used when --match-on includes 'atoms'. 1 means exact matching.",
    )
    parser.add_argument(
        "--match-heavy-bin",
        type=int,
        default=1,
        help="Bin size (heavy atoms) used when --match-on includes 'heavy'. 1 means exact matching.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write a JSON report (includes per-molecule results).",
    )
    return parser.parse_args()


def _prepare_jobs(
    samples: Sequence[Dict[str, Any]],
    dataset: str,
    remove_h: bool,
    num_eval: int,
    seed: int,
    min_atoms: Optional[int] = None,
    max_atoms: Optional[int] = None,
    min_heavy_atoms: Optional[int] = None,
    max_heavy_atoms: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[int]]:
    rng = np.random.default_rng(int(seed))
    n = len(samples)
    if n == 0:
        raise ValueError("No samples found.")

    min_atoms_int = None
    if min_atoms is not None:
        min_atoms_int = int(min_atoms)
        if min_atoms_int <= 0:
            min_atoms_int = None

    max_atoms_int = None
    if max_atoms is not None:
        max_atoms_int = int(max_atoms)
        if max_atoms_int <= 0:
            max_atoms_int = None

    min_heavy_int = None
    if min_heavy_atoms is not None:
        min_heavy_int = int(min_heavy_atoms)
        if min_heavy_int <= 0:
            min_heavy_int = None

    max_heavy_int = None
    if max_heavy_atoms is not None:
        max_heavy_int = int(max_heavy_atoms)
        if max_heavy_int <= 0:
            max_heavy_int = None

    candidate_indices: List[int] = []
    for idx, sample in enumerate(samples):
        num_atoms_i = int(sample.get("num_atoms") or sample.get("n_atoms") or sample.get("num_nodes") or 0)
        if num_atoms_i <= 0:
            positions = sample.get("positions")
            if positions is None:
                continue
            if isinstance(positions, torch.Tensor):
                num_atoms_i = int(positions.shape[0])
            else:
                num_atoms_i = int(np.asarray(positions).shape[0])
        if min_atoms_int is not None and num_atoms_i < min_atoms_int:
            continue
        if max_atoms_int is not None and num_atoms_i > max_atoms_int:
            continue

        if min_heavy_int is not None or max_heavy_int is not None:
            try:
                atom_types = _to_numpy_atom_types(sample["atom_types"], num_atoms_i)
                symbols = _atom_symbols_from_types(atom_types, dataset=dataset, remove_h=remove_h)
                heavy_atoms_i = _count_heavy_atoms(symbols)
            except Exception:
                continue
            if min_heavy_int is not None and heavy_atoms_i < min_heavy_int:
                continue
            if max_heavy_int is not None and heavy_atoms_i > max_heavy_int:
                continue

        candidate_indices.append(int(idx))

    if not candidate_indices:
        raise ValueError("No samples remain after applying --max-atoms/--max-heavy-atoms filters.")

    k = min(int(num_eval), len(candidate_indices))
    indices = rng.choice(candidate_indices, size=k, replace=False).tolist()

    jobs: List[Dict[str, Any]] = []
    for local_id, sample_idx in enumerate(indices):
        sample = samples[sample_idx]
        num_atoms = int(sample.get("num_atoms") or sample.get("n_atoms") or sample.get("num_nodes") or 0)
        if num_atoms <= 0:
            positions = sample.get("positions")
            if positions is None:
                raise ValueError(f"Sample {sample_idx} missing num_atoms and positions.")
            if isinstance(positions, torch.Tensor):
                num_atoms = int(positions.shape[0])
            else:
                num_atoms = int(np.asarray(positions).shape[0])

        positions = _to_numpy_positions(sample["positions"], num_atoms)
        atom_types = _to_numpy_atom_types(sample["atom_types"], num_atoms)
        symbols = _atom_symbols_from_types(atom_types, dataset=dataset, remove_h=remove_h)

        jobs.append(
            {
                "mol_id": local_id,
                "sample_idx": int(sample_idx),
                "symbols": symbols,
                "positions_angstrom": positions,
            }
        )
    return jobs, indices


def _jobs_from_indices(
    samples: Sequence[Dict[str, Any]],
    indices: Sequence[int],
    dataset: str,
    remove_h: bool,
) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    for local_id, sample_idx in enumerate(indices):
        sample = samples[int(sample_idx)]
        num_atoms = int(sample.get("num_atoms") or sample.get("n_atoms") or sample.get("num_nodes") or 0)
        if num_atoms <= 0:
            positions = sample.get("positions")
            if positions is None:
                raise ValueError(f"Sample {sample_idx} missing num_atoms and positions.")
            if isinstance(positions, torch.Tensor):
                num_atoms = int(positions.shape[0])
            else:
                num_atoms = int(np.asarray(positions).shape[0])

        positions = _to_numpy_positions(sample["positions"], num_atoms)
        atom_types = _to_numpy_atom_types(sample["atom_types"], num_atoms)
        symbols = _atom_symbols_from_types(atom_types, dataset=dataset, remove_h=remove_h)

        jobs.append(
            {
                "mol_id": int(local_id),
                "sample_idx": int(sample_idx),
                "symbols": symbols,
                "positions_angstrom": positions,
            }
        )
    return jobs


def _extract_num_atoms(sample: Dict[str, Any]) -> Optional[int]:
    num_atoms = int(sample.get("num_atoms") or sample.get("n_atoms") or sample.get("num_nodes") or 0)
    if num_atoms > 0:
        return int(num_atoms)
    positions = sample.get("positions")
    if positions is None:
        return None
    if isinstance(positions, torch.Tensor):
        return int(positions.shape[0])
    return int(np.asarray(positions).shape[0])


def _extract_num_heavy_atoms(
    sample: Dict[str, Any],
    dataset: str,
    remove_h: bool,
    num_atoms: int,
) -> Optional[int]:
    try:
        atom_types = _to_numpy_atom_types(sample["atom_types"], num_atoms)
        symbols = _atom_symbols_from_types(atom_types, dataset=dataset, remove_h=remove_h)
        return int(_count_heavy_atoms(symbols))
    except Exception:
        return None


def _size_key(
    num_atoms: int,
    num_heavy_atoms: Optional[int],
    match_on: Sequence[str],
    atom_bin: int,
    heavy_bin: int,
) -> Tuple[int, ...]:
    key_parts: List[int] = []
    if "atoms" in match_on:
        atom_bin = max(int(atom_bin), 1)
        key_parts.append(int((num_atoms // atom_bin) * atom_bin))
    if "heavy" in match_on:
        if num_heavy_atoms is None:
            raise ValueError("num_heavy_atoms is required for match_on=heavy")
        heavy_bin = max(int(heavy_bin), 1)
        key_parts.append(int((int(num_heavy_atoms) // heavy_bin) * heavy_bin))
    return tuple(key_parts)


def _collect_candidate_bins(
    samples: Sequence[Dict[str, Any]],
    dataset: str,
    remove_h: bool,
    min_atoms: Optional[int],
    max_atoms: Optional[int],
    min_heavy_atoms: Optional[int],
    max_heavy_atoms: Optional[int],
    match_on: Sequence[str],
    match_atom_bin: int,
    match_heavy_bin: int,
) -> Dict[Tuple[int, ...], List[int]]:
    min_atoms_int = int(min_atoms) if min_atoms is not None and int(min_atoms) > 0 else None
    max_atoms_int = int(max_atoms) if max_atoms is not None and int(max_atoms) > 0 else None
    min_heavy_int = int(min_heavy_atoms) if min_heavy_atoms is not None and int(min_heavy_atoms) > 0 else None
    max_heavy_int = int(max_heavy_atoms) if max_heavy_atoms is not None and int(max_heavy_atoms) > 0 else None

    need_heavy = bool(min_heavy_int is not None or max_heavy_int is not None or "heavy" in match_on)
    bins: Dict[Tuple[int, ...], List[int]] = {}

    for idx, sample in enumerate(samples):
        num_atoms_i = _extract_num_atoms(sample)
        if num_atoms_i is None or int(num_atoms_i) <= 0:
            continue
        if min_atoms_int is not None and num_atoms_i < min_atoms_int:
            continue
        if max_atoms_int is not None and num_atoms_i > max_atoms_int:
            continue

        num_heavy_i: Optional[int] = None
        if need_heavy:
            num_heavy_i = _extract_num_heavy_atoms(sample, dataset=dataset, remove_h=remove_h, num_atoms=num_atoms_i)
            if num_heavy_i is None:
                continue
            if min_heavy_int is not None and num_heavy_i < min_heavy_int:
                continue
            if max_heavy_int is not None and num_heavy_i > max_heavy_int:
                continue

        key = _size_key(
            num_atoms=num_atoms_i,
            num_heavy_atoms=num_heavy_i,
            match_on=match_on,
            atom_bin=match_atom_bin,
            heavy_bin=match_heavy_bin,
        )
        bins.setdefault(key, []).append(int(idx))

    return bins


def _match_size_bins(
    baseline_bins: Dict[Tuple[int, ...], List[int]],
    rl_bins: Dict[Tuple[int, ...], List[int]],
    num_eval: int,
    seed: int,
) -> Tuple[List[int], List[int], Dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    baseline_bins = {k: list(v) for k, v in baseline_bins.items()}
    rl_bins = {k: list(v) for k, v in rl_bins.items()}
    for key in baseline_bins:
        rng.shuffle(baseline_bins[key])
    for key in rl_bins:
        rng.shuffle(rl_bins[key])

    common_keys = sorted(set(baseline_bins.keys()) & set(rl_bins.keys()))
    slots: List[Tuple[int, ...]] = []
    per_key_capacity: Dict[Tuple[int, ...], int] = {}
    for key in common_keys:
        cap = min(len(baseline_bins[key]), len(rl_bins[key]))
        if cap <= 0:
            continue
        per_key_capacity[key] = int(cap)
        slots.extend([key] * int(cap))

    max_matched = len(slots)
    if max_matched <= 0:
        raise ValueError("No size-matched samples available between baseline and RL (after filters).")
    if int(num_eval) > max_matched:
        raise ValueError(
            f"Requested --num-eval {num_eval} but only {max_matched} size-matched molecules are available. "
            "Reduce --num-eval, relax size filters, or increase bin sizes (e.g., --match-atom-bin 5)."
        )

    rng.shuffle(slots)
    slots = slots[: int(num_eval)]

    baseline_indices: List[int] = []
    rl_indices: List[int] = []
    picked_hist: Dict[Tuple[int, ...], int] = {}
    for key in slots:
        baseline_indices.append(int(baseline_bins[key].pop()))
        rl_indices.append(int(rl_bins[key].pop()))
        picked_hist[key] = int(picked_hist.get(key, 0) + 1)

    meta = {
        "max_size_matched": int(max_matched),
        "num_eval": int(num_eval),
        "num_common_bins": int(len(per_key_capacity)),
        "picked_bin_counts": {str(k): int(v) for k, v in sorted(picked_hist.items(), key=lambda kv: kv[0])},
        "bin_capacities": {str(k): int(v) for k, v in sorted(per_key_capacity.items(), key=lambda kv: kv[0])},
    }
    return baseline_indices, rl_indices, meta


def _evaluate_jobs(
    jobs: Sequence[Dict[str, Any]],
    dft_cfg: DFTConfig,
    num_workers: int,
    atom_ref_energies_hartree: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    if num_workers <= 1:
        return [_evaluate_one(job, dft_cfg, atom_ref_energies_hartree) for job in jobs]

    results: List[Optional[Dict[str, Any]]] = [None for _ in range(len(jobs))]
    with ProcessPoolExecutor(max_workers=int(num_workers)) as ex:
        future_to_id = {
            ex.submit(_evaluate_one, job, dft_cfg, atom_ref_energies_hartree): int(job["mol_id"])
            for job in jobs
        }
        for fut in as_completed(future_to_id):
            mol_id = future_to_id[fut]
            try:
                results[mol_id] = fut.result()
            except Exception as exc:  # pragma: no cover
                results[mol_id] = {
                    "mol_id": mol_id,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
    return [
        r if r is not None else {"mol_id": i, "ok": False, "error": "missing result"}
        for i, r in enumerate(results)
    ]


def _compute_atomic_reference_energies(symbols: Sequence[str], dft_cfg: DFTConfig) -> Dict[str, float]:
    """Compute single-point energies for isolated atoms at the same DFT level.

    Absolute DFT energies are dominated by atomic core contributions; comparing molecules with
    different compositions via E/atom can be misleading. Atomization energy (sum(E_atom) - E_mol)
    provides a composition-normalized metric that is more comparable across samples.
    """
    unique_symbols = sorted({str(sym).strip() for sym in symbols if str(sym).strip()})
    ref: Dict[str, float] = {}
    for sym in unique_symbols:
        atomic_spin = dft_cfg.spin
        if atomic_spin is None and int(dft_cfg.charge) == 0:
            atomic_spin = _default_atomic_spin(sym)
        atomic_cfg = replace(dft_cfg, spin=atomic_spin)
        mol = _build_mol([sym], np.zeros((1, 3), dtype=np.float64), atomic_cfg)
        e_h, _, converged = _run_single_point(mol, atomic_cfg)
        if not converged and not dft_cfg.accept_unconverged:
            raise RuntimeError(f"Atomic SCF did not converge for {sym}. Use --accept-unconverged to proceed.")
        ref[sym] = float(e_h)
    return ref


def main() -> None:
    args = parse_args()

    baseline_path = Path(args.baseline_samples).expanduser().resolve()
    rl_path = Path(args.rl_samples).expanduser().resolve()
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline samples file not found: {baseline_path}")
    if not rl_path.exists():
        raise FileNotFoundError(f"RL samples file not found: {rl_path}")

    dft_cfg = DFTConfig(
        xc=str(args.xc),
        basis=str(args.basis),
        ecp=str(args.ecp) if args.ecp else None,
        charge=int(args.charge),
        spin=int(args.spin) if args.spin is not None else None,
        max_cycle=int(args.max_cycle),
        conv_tol=float(args.conv_tol),
        grids_level=int(args.grids_level),
        density_fit=bool(args.density_fit),
        retry_newton=bool(args.retry_newton),
        accept_unconverged=bool(args.accept_unconverged),
        min_interatomic_distance=float(args.min_interatomic_distance),
    )

    baseline_samples = _load_samples(baseline_path)
    rl_samples = _load_samples(rl_path)

    baseline_size_all = _size_stats(baseline_samples, dataset=args.dataset, remove_h=bool(args.remove_h))
    rl_size_all = _size_stats(rl_samples, dataset=args.dataset, remove_h=bool(args.remove_h))

    match_on = list(args.match_on or [])
    match_atom_bin = max(int(args.match_atom_bin), 1)
    match_heavy_bin = max(int(args.match_heavy_bin), 1)
    selection_meta: Dict[str, Any] = {
        "match_on": match_on,
        "match_atom_bin": int(match_atom_bin),
        "match_heavy_bin": int(match_heavy_bin),
    }

    if match_on:
        baseline_bins = _collect_candidate_bins(
            baseline_samples,
            dataset=args.dataset,
            remove_h=bool(args.remove_h),
            min_atoms=args.min_atoms,
            max_atoms=args.max_atoms,
            min_heavy_atoms=args.min_heavy_atoms,
            max_heavy_atoms=args.max_heavy_atoms,
            match_on=match_on,
            match_atom_bin=match_atom_bin,
            match_heavy_bin=match_heavy_bin,
        )
        rl_bins = _collect_candidate_bins(
            rl_samples,
            dataset=args.dataset,
            remove_h=bool(args.remove_h),
            min_atoms=args.min_atoms,
            max_atoms=args.max_atoms,
            min_heavy_atoms=args.min_heavy_atoms,
            max_heavy_atoms=args.max_heavy_atoms,
            match_on=match_on,
            match_atom_bin=match_atom_bin,
            match_heavy_bin=match_heavy_bin,
        )
        baseline_indices, rl_indices, match_meta = _match_size_bins(
            baseline_bins,
            rl_bins,
            num_eval=int(args.num_eval),
            seed=int(args.seed),
        )
        selection_meta["matching"] = match_meta
        baseline_jobs = _jobs_from_indices(
            baseline_samples, baseline_indices, dataset=args.dataset, remove_h=bool(args.remove_h)
        )
        rl_jobs = _jobs_from_indices(rl_samples, rl_indices, dataset=args.dataset, remove_h=bool(args.remove_h))
    else:
        baseline_jobs, baseline_indices = _prepare_jobs(
            baseline_samples,
            dataset=args.dataset,
            remove_h=bool(args.remove_h),
            num_eval=args.num_eval,
            seed=args.seed,
            min_atoms=args.min_atoms,
            max_atoms=args.max_atoms,
            min_heavy_atoms=args.min_heavy_atoms,
            max_heavy_atoms=args.max_heavy_atoms,
        )
        rl_jobs, rl_indices = _prepare_jobs(
            rl_samples,
            dataset=args.dataset,
            remove_h=bool(args.remove_h),
            num_eval=args.num_eval,
            seed=args.seed,
            min_atoms=args.min_atoms,
            max_atoms=args.max_atoms,
            min_heavy_atoms=args.min_heavy_atoms,
            max_heavy_atoms=args.max_heavy_atoms,
        )

    atom_ref_symbols: List[str] = []
    for job in baseline_jobs:
        atom_ref_symbols.extend(job.get("symbols", []))
    for job in rl_jobs:
        atom_ref_symbols.extend(job.get("symbols", []))
    atom_ref_energies_h = _compute_atomic_reference_energies(atom_ref_symbols, dft_cfg)

    print(
        f"DFT config: xc={dft_cfg.xc} basis={dft_cfg.basis}"
        + (f" ecp={dft_cfg.ecp}" if dft_cfg.ecp else "")
        + f" charge={dft_cfg.charge} spin={dft_cfg.spin if dft_cfg.spin is not None else 'infer'}"
        + f" max_cycle={dft_cfg.max_cycle} conv_tol={dft_cfg.conv_tol:g} grids={dft_cfg.grids_level}"
        + f" density_fit={dft_cfg.density_fit} workers={args.num_workers}"
    )

    print(f"Evaluating baseline: {baseline_path.name} (n={len(baseline_jobs)}/{len(baseline_samples)})")
    baseline_results = _evaluate_jobs(
        baseline_jobs,
        dft_cfg,
        num_workers=int(args.num_workers),
        atom_ref_energies_hartree=atom_ref_energies_h,
    )
    baseline_summary = _summarize(baseline_results)

    print(f"Evaluating RL: {rl_path.name} (n={len(rl_jobs)}/{len(rl_samples)})")
    rl_results = _evaluate_jobs(
        rl_jobs,
        dft_cfg,
        num_workers=int(args.num_workers),
        atom_ref_energies_hartree=atom_ref_energies_h,
    )
    rl_summary = _summarize(rl_results)

    comparison = _compare(baseline_summary, rl_summary)

    report = {
        "dft_config": {
            "xc": dft_cfg.xc,
            "basis": dft_cfg.basis,
            "ecp": dft_cfg.ecp,
            "charge": dft_cfg.charge,
            "spin": dft_cfg.spin,
            "max_cycle": dft_cfg.max_cycle,
            "conv_tol": dft_cfg.conv_tol,
            "grids_level": dft_cfg.grids_level,
            "density_fit": dft_cfg.density_fit,
            "retry_newton": dft_cfg.retry_newton,
            "accept_unconverged": dft_cfg.accept_unconverged,
            "min_interatomic_distance": dft_cfg.min_interatomic_distance,
        },
        "selection": selection_meta,
        "atom_reference_energies_hartree": atom_ref_energies_h,
        "baseline": {
            "path": str(baseline_path),
            "size_all_samples": baseline_size_all,
            "picked_indices": baseline_indices,
            "summary": baseline_summary,
            "results": baseline_results,
        },
        "rl": {
            "path": str(rl_path),
            "size_all_samples": rl_size_all,
            "picked_indices": rl_indices,
            "summary": rl_summary,
            "results": rl_results,
        },
        "comparison": comparison,
    }

    def _fmt(x: Optional[float]) -> str:
        if x is None:
            return "n/a"
        return f"{x:.4f}"

    print("---- Summary (means over successful molecules) ----")
    print(
        "Energy eV/atom: "
        f"baseline={_fmt(baseline_summary.get('energy_ev_per_atom_mean'))} "
        f"rl={_fmt(rl_summary.get('energy_ev_per_atom_mean'))} "
        f"delta(b-r)={_fmt(comparison.get('delta_energy_ev_per_atom_mean'))}"
    )
    print(
        "Energy eV/heavy atom: "
        f"baseline={_fmt(baseline_summary.get('energy_ev_per_heavy_atom_mean'))} "
        f"rl={_fmt(rl_summary.get('energy_ev_per_heavy_atom_mean'))} "
        f"delta(b-r)={_fmt(comparison.get('delta_energy_ev_per_heavy_atom_mean'))}"
    )
    if baseline_summary.get("atomization_energy_ev_per_atom_mean") is not None and rl_summary.get(
        "atomization_energy_ev_per_atom_mean"
    ) is not None:
        print(
            "Atomization eV/atom: "
            f"baseline={_fmt(baseline_summary.get('atomization_energy_ev_per_atom_mean'))} "
            f"rl={_fmt(rl_summary.get('atomization_energy_ev_per_atom_mean'))} "
            f"delta(r-b)={_fmt(comparison.get('delta_atomization_energy_ev_per_atom_mean'))}"
        )
        print(
            "Atomization eV/heavy atom: "
            f"baseline={_fmt(baseline_summary.get('atomization_energy_ev_per_heavy_atom_mean'))} "
            f"rl={_fmt(rl_summary.get('atomization_energy_ev_per_heavy_atom_mean'))} "
            f"delta(r-b)={_fmt(comparison.get('delta_atomization_energy_ev_per_heavy_atom_mean'))}"
        )
    print(
        "Force RMS eV/Å: "
        f"baseline={_fmt(baseline_summary.get('force_rms_ev_per_a_mean'))} "
        f"rl={_fmt(rl_summary.get('force_rms_ev_per_a_mean'))} "
        f"delta(b-r)={_fmt(comparison.get('delta_force_rms_ev_per_a_mean'))}"
    )
    print(
        "Force RMS (heavy) eV/Å: "
        f"baseline={_fmt(baseline_summary.get('force_rms_ev_per_a_heavy_mean'))} "
        f"rl={_fmt(rl_summary.get('force_rms_ev_per_a_heavy_mean'))} "
        f"delta(b-r)={_fmt(comparison.get('delta_force_rms_ev_per_a_heavy_mean'))}"
    )
    print(
        "Avg #atoms: "
        f"baseline={_fmt(baseline_summary.get('avg_num_atoms'))} "
        f"(all={_fmt(baseline_size_all.get('avg_num_atoms'))}) "
        f"rl={_fmt(rl_summary.get('avg_num_atoms'))} "
        f"(all={_fmt(rl_size_all.get('avg_num_atoms'))}) "
        f"delta(b-r)={_fmt(comparison.get('delta_avg_num_atoms'))}"
    )
    print(
        "Avg #heavy atoms: "
        f"baseline={_fmt(baseline_summary.get('avg_num_heavy_atoms'))} "
        f"(all={_fmt(baseline_size_all.get('avg_num_heavy_atoms'))}) "
        f"rl={_fmt(rl_summary.get('avg_num_heavy_atoms'))} "
        f"(all={_fmt(rl_size_all.get('avg_num_heavy_atoms'))}) "
        f"delta(b-r)={_fmt(comparison.get('delta_avg_num_heavy_atoms'))}"
    )
    print(
        "Success rate: "
        f"baseline={baseline_summary.get('num_ok', 0)}/{baseline_summary.get('num_total', 0)} "
        f"rl={rl_summary.get('num_ok', 0)}/{rl_summary.get('num_total', 0)}"
    )

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2, sort_keys=True)
        print(f"Wrote report to {out_path}")


if __name__ == "__main__":
    main()
