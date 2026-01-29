from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANGSTROM = 0.52917721092
HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM = HARTREE_TO_EV / BOHR_TO_ANGSTROM

# Avoid oversubscribing CPU threads when using multiprocessing with PySCF/BLAS.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate energy and force of QM9 samples using the available oracle.\n"
            "Supported oracles:\n"
            "  - uma: UMA ML force field (fast; DFT-trained proxy)\n"
            "  - pyscf: DFT single-point energy + forces via PySCF (slower; CPU by default)\n"
        )
    )
    parser.add_argument(
        "--oracle",
        type=str,
        choices=["uma", "pyscf"],
        default="uma",
        help="Oracle backend to use: uma or pyscf (default: uma).",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        help="Path to baseline samples (e.g. outputs/.../eval_rollouts.pt).",
    )
    parser.add_argument(
        "--candidate",
        type=str,
        required=True,
        help="Path to candidate samples (e.g. outputs/.../eval_rollouts_512_seed123.pt).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=128,
        help="Number of samples to evaluate from each set (default: 128).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="RNG seed used to subsample molecules (default: 123).",
    )
    parser.add_argument(
        "--stable-only",
        action="store_true",
        help="Evaluate only QM9-stable molecules (as defined by check_stability).",
    )
    parser.add_argument(
        "--pvalues",
        action="store_true",
        help=(
            "Compute p-values comparing candidate vs baseline for force RMS and formation energy/atom "
            "(Welch t-test + Mann–Whitney U). Requires scipy."
        ),
    )
    parser.add_argument(
        "--dump-values",
        action="store_true",
        help="Include per-molecule scalar values in JSON output (force_rms_values, formation_energy_per_atom_values).",
    )
    parser.add_argument(
        "--mlff-model",
        type=str,
        default="uma-s-1p1",
        help="UMA model name (default: uma-s-1p1). Ignored when --oracle=pyscf.",
    )
    parser.add_argument(
        "--xc",
        type=str,
        default="B3LYP",
        help="PySCF XC functional (default: B3LYP). Only used when --oracle=pyscf.",
    )
    parser.add_argument(
        "--basis",
        type=str,
        default="6-31G(d)",
        help="PySCF basis (default: 6-31G(d)). Only used when --oracle=pyscf.",
    )
    parser.add_argument(
        "--grid-level",
        type=int,
        default=3,
        help="PySCF integration grid level (default: 3). Only used when --oracle=pyscf.",
    )
    parser.add_argument(
        "--density-fit",
        action="store_true",
        help="Use density fitting (RI) for PySCF DFT calculations to reduce memory (default: off).",
    )
    parser.add_argument(
        "--max-memory-mb",
        type=int,
        default=None,
        help="Set PySCF mol.max_memory in MB to cap per-process memory (default: None).",
    )
    parser.add_argument(
        "--scf-max-cycle",
        type=int,
        default=200,
        help="PySCF SCF max cycles (default: 200). Only used when --oracle=pyscf.",
    )
    parser.add_argument(
        "--scf-conv-tol",
        type=float,
        default=1e-9,
        help="PySCF SCF convergence tolerance (default: 1e-9). Only used when --oracle=pyscf.",
    )
    parser.add_argument(
        "--n-procs",
        type=int,
        default=1,
        help="Number of parallel PySCF worker processes (default: 1). Only used when --oracle=pyscf.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for UMA inference (cpu/cuda/cuda:0...). Default: cpu. Ignored when --oracle=pyscf.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional path to write a JSON summary.",
    )
    return parser.parse_args()


def load_samples(path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    import torch

    payload = torch.load(path, map_location="cpu")
    metadata: Dict[str, Any] = {}
    if isinstance(payload, dict) and "samples" in payload:
        samples = payload["samples"]
        metadata = {k: v for k, v in payload.items() if k != "samples"}
    elif isinstance(payload, list):
        samples = payload
    else:
        raise ValueError(f"Unrecognized sample payload structure in {path}")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"No samples found in {path}")
    return samples, metadata


def stable_filter(samples: List[Dict[str, Any]], dataset_info: Dict[str, Any]) -> List[Dict[str, Any]]:
    from edm_source.qm9.analyze import check_stability

    stable: List[Dict[str, Any]] = []
    for sample in samples:
        pos = sample["positions"].cpu().numpy()
        types = sample["atom_types"].cpu().numpy()
        is_stable, _, _ = check_stability(pos, types, dataset_info)
        if is_stable:
            stable.append(sample)
    return stable


def subsample(samples: List[Dict[str, Any]], k: int, seed: int) -> List[Dict[str, Any]]:
    if k <= 0:
        raise ValueError("--k must be positive")
    if k >= len(samples):
        return list(samples)
    rng = random.Random(seed)
    idxs = rng.sample(range(len(samples)), k=k)
    return [samples[i] for i in idxs]


def batch_from_samples(
    samples: List[Dict[str, Any]],
    num_classes: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    import torch
    import torch.nn.functional as F

    max_n = max(int(sample["num_atoms"]) for sample in samples)
    batch_size = len(samples)

    positions = torch.zeros((batch_size, max_n, 3), dtype=torch.float32)
    one_hot = torch.zeros((batch_size, max_n, num_classes), dtype=torch.float32)
    node_mask = torch.zeros((batch_size, max_n, 1), dtype=torch.float32)
    atom_types_padded = torch.full((batch_size, max_n), fill_value=-1, dtype=torch.long)

    for i, sample in enumerate(samples):
        n = int(sample["num_atoms"])
        types = sample["atom_types"].to(torch.long)
        positions[i, :n] = sample["positions"].to(torch.float32)
        one_hot[i, :n] = F.one_hot(types, num_classes=num_classes).to(torch.float32)
        node_mask[i, :n, 0] = 1.0
        atom_types_padded[i, :n] = types

    z = torch.cat([positions, one_hot], dim=-1)
    return z, node_mask, atom_types_padded


def force_rms(forces: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    import torch

    device = forces.device
    mask = (node_mask[..., 0] > 0.5).to(device=device)
    counts = mask.sum(dim=1).clamp(min=1).to(dtype=forces.dtype)
    norms = torch.norm(forces, dim=-1)
    return torch.sqrt((norms.pow(2) * mask.to(dtype=norms.dtype)).sum(dim=1) / counts)


def summarize(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "p10": float("nan"),
            "p90": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    vals = sorted(float(v) for v in values)

    def quantile(p: float) -> float:
        idx = int(round((len(vals) - 1) * p))
        return vals[idx]

    return {
        "count": len(vals),
        "mean": statistics.fmean(vals),
        "median": statistics.median(vals),
        "p10": quantile(0.10),
        "p90": quantile(0.90),
        "min": vals[0],
        "max": vals[-1],
    }


def formation_energy_per_atom(
    energies: torch.Tensor,
    atom_types_padded: torch.Tensor,
    node_mask: torch.Tensor,
    dataset_info: Dict[str, Any],
    atom_refs: Any,
) -> torch.Tensor:
    import torch

    decoder = dataset_info["atom_decoder"]
    symbol_to_z = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}
    mask = node_mask[..., 0] > 0.5
    counts = mask.sum(dim=1).clamp(min=1).to(torch.float32)

    ref_sums: List[float] = []
    for i in range(atom_types_padded.shape[0]):
        ref_sum = 0.0
        for t in atom_types_padded[i][mask[i]]:
            znum = symbol_to_z.get(decoder[int(t)], 1)
            ref_sum += float(atom_refs[int(znum)][0])  # charge state 0
        ref_sums.append(ref_sum)

    ref_sums_t = torch.tensor(ref_sums, dtype=energies.dtype)
    return (energies.detach().cpu() - ref_sums_t) / counts.detach().cpu()


def _qm9_atomic_symbols(dataset_info: Dict[str, Any], atom_types: torch.Tensor) -> List[str]:
    decoder = dataset_info["atom_decoder"]
    return [decoder[int(t)] for t in atom_types.view(-1).tolist()]


def _guess_spin_from_symbols(symbols: List[str], charge: int = 0) -> int:
    symbol_to_z = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}
    n_electrons = sum(symbol_to_z.get(sym, 0) for sym in symbols) - int(charge)
    return int(n_electrons % 2)


def _pyscf_atomic_reference_energies(
    xc: str,
    basis: str,
    grid_level: int,
    density_fit: bool,
    max_memory_mb: Optional[int],
    scf_max_cycle: int,
    scf_conv_tol: float,
) -> Dict[str, float]:
    from pyscf import dft, gto

    # Ground-state spin (2S) guesses for isolated atoms.
    spins = {"H": 1, "C": 2, "N": 3, "O": 2, "F": 1}
    refs: Dict[str, float] = {}
    for sym, spin in spins.items():
        mol = gto.M(
            atom=f"{sym} 0 0 0",
            basis=basis,
            charge=0,
            spin=spin,
            unit="Angstrom",
            verbose=0,
        )
        if max_memory_mb is not None:
            mol.max_memory = float(max_memory_mb)
        mf = dft.UKS(mol)
        mf.xc = xc
        mf.grids.level = int(grid_level)
        mf.max_cycle = int(scf_max_cycle)
        mf.conv_tol = float(scf_conv_tol)
        if density_fit:
            mf = mf.density_fit()
        energy = float(mf.kernel())
        if not mf.converged:
            mf_newton = mf.newton()
            mf_newton.max_cycle = int(scf_max_cycle)
            mf_newton.conv_tol = float(scf_conv_tol)
            energy = float(mf_newton.kernel())
            if not mf_newton.converged:
                raise RuntimeError(f"PySCF failed to converge atomic reference for {sym}")
        refs[sym] = energy
    return refs


def _pyscf_single_point_energy_forces(
    symbols: List[str],
    coords_angstrom: List[List[float]],
    xc: str,
    basis: str,
    grid_level: int,
    density_fit: bool,
    max_memory_mb: Optional[int],
    scf_max_cycle: int,
    scf_conv_tol: float,
    charge: int = 0,
    spin: Optional[int] = None,
) -> Tuple[float, List[List[float]]]:
    from pyscf import dft, gto

    if spin is None:
        spin = _guess_spin_from_symbols(symbols, charge=charge)

    atom = [(sym, tuple(map(float, xyz))) for sym, xyz in zip(symbols, coords_angstrom)]
    mol = gto.M(
        atom=atom,
        basis=basis,
        charge=int(charge),
        spin=int(spin),
        unit="Angstrom",
        verbose=0,
    )
    if max_memory_mb is not None:
        mol.max_memory = float(max_memory_mb)

    if spin == 0:
        mf = dft.RKS(mol)
    else:
        mf = dft.UKS(mol)
    mf.xc = xc
    mf.grids.level = int(grid_level)
    mf.max_cycle = int(scf_max_cycle)
    mf.conv_tol = float(scf_conv_tol)
    if density_fit:
        mf = mf.density_fit()

    energy = float(mf.kernel())
    if not mf.converged:
        mf_newton = mf.newton()
        mf_newton.max_cycle = int(scf_max_cycle)
        mf_newton.conv_tol = float(scf_conv_tol)
        energy = float(mf_newton.kernel())
        mf = mf_newton

    if not mf.converged:
        raise RuntimeError("PySCF SCF did not converge")

    grad = mf.nuc_grad_method().kernel()  # dE/dR in Hartree/Bohr
    forces = (-grad * HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM).tolist()
    return energy, forces


def _pyscf_worker_eval(payload: Dict[str, Any]) -> Dict[str, Any]:
    # Ensure PySCF uses a single thread per worker.
    try:
        from pyscf import lib as pyscf_lib

        pyscf_lib.num_threads(1)
    except Exception:
        pass

    symbols = payload["symbols"]
    coords = payload["coords"]
    dft_cfg = payload["dft_cfg"]
    atomic_refs = payload["atomic_refs"]

    spin = _guess_spin_from_symbols(symbols, charge=0)
    energy_ha, forces_eva = _pyscf_single_point_energy_forces(
        symbols,
        coords,
        xc=dft_cfg["xc"],
        basis=dft_cfg["basis"],
        grid_level=dft_cfg["grid_level"],
        density_fit=bool(dft_cfg.get("density_fit", False)),
        max_memory_mb=dft_cfg.get("max_memory_mb"),
        scf_max_cycle=dft_cfg["scf_max_cycle"],
        scf_conv_tol=dft_cfg["scf_conv_tol"],
        charge=0,
        spin=spin,
    )

    n_atoms = len(symbols)
    ref_sum_ha = sum(float(atomic_refs[sym]) for sym in symbols)
    form_energy_ev_per_atom = (energy_ha - ref_sum_ha) * HARTREE_TO_EV / max(1, n_atoms)

    force_sq_sum = 0.0
    for fx, fy, fz in forces_eva:
        force_sq_sum += float(fx) ** 2 + float(fy) ** 2 + float(fz) ** 2
    rms = math.sqrt(force_sq_sum / max(1, n_atoms))

    return {
        "force_rms": float(rms),
        "formation_energy_per_atom": float(form_energy_ev_per_atom),
    }


def eval_set(
    name: str,
    samples_path: str,
    dataset_info: Dict[str, Any],
    oracle: str,
    oracle_state: Dict[str, Any],
    n_procs: int,
    k: int,
    seed: int,
    stable_only: bool,
    return_values: bool,
) -> Dict[str, Any]:
    samples, metadata = load_samples(samples_path)
    stable_rate = None
    if stable_only:
        stable_samples = stable_filter(samples, dataset_info)
        stable_rate = len(stable_samples) / len(samples)
        samples = stable_samples

    if not samples:
        raise ValueError(f"{name}: no samples available after filtering")

    samples = subsample(samples, k=k, seed=seed)

    out: Dict[str, Any] = {
        "name": name,
        "samples_path": str(Path(samples_path)),
        "k": len(samples),
        "stable_only": stable_only,
        "stable_rate_all": stable_rate,
    }
    if metadata.get("rdkit_metrics") is not None:
        out["rdkit_metrics"] = metadata["rdkit_metrics"]

    if oracle == "uma":
        import torch

        force_computer = oracle_state["force_computer"]
        atom_refs = oracle_state["atom_refs"]
        num_classes = len(dataset_info["atom_decoder"])
        z, node_mask, atom_types_padded = batch_from_samples(samples, num_classes=num_classes)

        forces, energies = force_computer.compute_mlff_forces(z, node_mask, dataset_info)
        rms = force_rms(forces, node_mask).detach().cpu().tolist()
        form_e = formation_energy_per_atom(energies, atom_types_padded, node_mask, dataset_info, atom_refs).tolist()

        out["oracle"] = "uma"
        out["force_rms"] = summarize(rms)
        out["formation_energy_per_atom"] = summarize(form_e)
        if return_values:
            out["force_rms_values"] = list(map(float, rms))
            out["formation_energy_per_atom_values"] = list(map(float, form_e))
        return out

    if oracle != "pyscf":
        raise ValueError(f"Unsupported oracle: {oracle}")

    dft_cfg = oracle_state["dft_cfg"]
    atomic_refs: Dict[str, float] = oracle_state["atomic_refs"]

    force_rms_vals: List[float] = []
    form_energy_vals: List[float] = []
    failures = 0
    failure_types: Dict[str, int] = {}
    failure_examples: List[str] = []

    decoder = dataset_info["atom_decoder"]
    tasks: List[Dict[str, Any]] = []
    for sample in samples:
        atom_types = sample["atom_types"]
        symbols = [decoder[int(t)] for t in atom_types.tolist()]
        coords = sample["positions"].tolist()
        tasks.append(
            {
                "symbols": symbols,
                "coords": coords,
                "dft_cfg": dft_cfg,
                "atomic_refs": atomic_refs,
            }
        )

    if n_procs < 1:
        raise ValueError("--n-procs must be >= 1")

    if n_procs == 1:
        from tqdm import tqdm

        for task in tqdm(tasks, desc=f"PySCF {name}", unit="mol"):
            try:
                out_task = _pyscf_worker_eval(task)
            except Exception as exc:
                failures += 1
                exc_name = type(exc).__name__
                failure_types[exc_name] = failure_types.get(exc_name, 0) + 1
                if len(failure_examples) < 5:
                    failure_examples.append(f"{exc_name}: {exc}")
                continue
            force_rms_vals.append(out_task["force_rms"])
            form_energy_vals.append(out_task["formation_energy_per_atom"])
    else:
        import concurrent.futures as cf
        import multiprocessing as mp
        from tqdm import tqdm

        mp_ctx = mp.get_context("spawn")
        with cf.ProcessPoolExecutor(max_workers=n_procs, mp_context=mp_ctx) as executor:
            futures = [executor.submit(_pyscf_worker_eval, task) for task in tasks]
            for fut in tqdm(cf.as_completed(futures), total=len(futures), desc=f"PySCF {name}", unit="mol"):
                try:
                    out_task = fut.result()
                except Exception as exc:
                    failures += 1
                    exc_name = type(exc).__name__
                    failure_types[exc_name] = failure_types.get(exc_name, 0) + 1
                    if len(failure_examples) < 5:
                        failure_examples.append(f"{exc_name}: {exc}")
                    continue
                force_rms_vals.append(out_task["force_rms"])
                form_energy_vals.append(out_task["formation_energy_per_atom"])

    out["oracle"] = "pyscf"
    out["dft_cfg"] = dict(dft_cfg)
    out["units"] = {"force_rms": "eV/Å", "formation_energy_per_atom": "eV/atom"}
    out["success"] = len(force_rms_vals)
    out["failures"] = failures
    out["failure_types"] = dict(sorted(failure_types.items(), key=lambda kv: (-kv[1], kv[0])))
    out["failure_examples"] = list(failure_examples)
    out["force_rms"] = summarize(force_rms_vals)
    out["formation_energy_per_atom"] = summarize(form_energy_vals)
    if return_values:
        out["force_rms_values"] = list(map(float, force_rms_vals))
        out["formation_energy_per_atom_values"] = list(map(float, form_energy_vals))
    return out


def _compute_p_values(candidate_vals: List[float], baseline_vals: List[float]) -> Dict[str, Any]:
    try:
        from scipy import stats
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("scipy is required for --pvalues") from exc

    def _finite(xs: List[float]) -> List[float]:
        out: List[float] = []
        for x in xs:
            xf = float(x)
            if math.isfinite(xf):
                out.append(xf)
        return out

    cand = _finite(candidate_vals)
    base = _finite(baseline_vals)
    if not cand or not base:
        raise ValueError("Cannot compute p-values with empty/invalid value lists")

    t_less = stats.ttest_ind(cand, base, equal_var=False, alternative="less", nan_policy="omit")
    t_two = stats.ttest_ind(cand, base, equal_var=False, alternative="two-sided", nan_policy="omit")
    u_less = stats.mannwhitneyu(cand, base, alternative="less", nan_policy="omit")
    u_two = stats.mannwhitneyu(cand, base, alternative="two-sided", nan_policy="omit")

    return {
        "n_candidate": len(cand),
        "n_baseline": len(base),
        "welch_t": {
            "statistic": float(t_less.statistic),
            "p_one_sided_less": float(t_less.pvalue),
            "p_two_sided": float(t_two.pvalue),
        },
        "mannwhitneyu": {
            "statistic": float(u_less.statistic),
            "p_one_sided_less": float(u_less.pvalue),
            "p_two_sided": float(u_two.pvalue),
        },
    }


def main() -> None:
    args = parse_args()
    for path in [args.baseline, args.candidate]:
        if not Path(path).exists():
            raise FileNotFoundError(path)

    from edm_source.configs.datasets_config import get_dataset_info

    dataset_info = get_dataset_info("qm9", remove_h=False)

    oracle_state: Dict[str, Any] = {}
    if args.oracle == "uma":
        from edm_source.mlff_modules.mlff_force_computer import MLFFForceComputer
        from edm_source.mlff_modules.mlff_utils import get_mlff_predictor

        predictor = get_mlff_predictor(args.mlff_model, device=args.device)
        if predictor is None:
            raise RuntimeError("Failed to load UMA predictor; check your environment and model cache.")
        atom_refs = predictor.atom_refs.get("omol")
        if atom_refs is None:
            raise RuntimeError("UMA predictor is missing 'omol' atom references; cannot compute formation energy.")
        oracle_state["force_computer"] = MLFFForceComputer(
            predictor,
            position_scale=1.0,
            device=args.device,
            compute_energy=True,
        )
        oracle_state["atom_refs"] = atom_refs
    else:
        dft_cfg = {
            "xc": args.xc,
            "basis": args.basis,
            "grid_level": int(args.grid_level),
            "density_fit": bool(args.density_fit),
            "max_memory_mb": args.max_memory_mb,
            "scf_max_cycle": int(args.scf_max_cycle),
            "scf_conv_tol": float(args.scf_conv_tol),
        }
        oracle_state["dft_cfg"] = dft_cfg
        oracle_state["atomic_refs"] = _pyscf_atomic_reference_energies(**dft_cfg)

    needs_values = bool(args.pvalues or args.dump_values)
    baseline = eval_set(
        "baseline",
        args.baseline,
        dataset_info,
        oracle=args.oracle,
        oracle_state=oracle_state,
        n_procs=int(args.n_procs),
        k=args.k,
        seed=args.seed,
        stable_only=args.stable_only,
        return_values=needs_values,
    )
    candidate = eval_set(
        "candidate",
        args.candidate,
        dataset_info,
        oracle=args.oracle,
        oracle_state=oracle_state,
        n_procs=int(args.n_procs),
        k=args.k,
        seed=args.seed,
        stable_only=args.stable_only,
        return_values=needs_values,
    )

    delta = {
        "force_rms_mean": candidate["force_rms"]["mean"] - baseline["force_rms"]["mean"],
        "formation_energy_per_atom_mean": candidate["formation_energy_per_atom"]["mean"]
        - baseline["formation_energy_per_atom"]["mean"],
    }

    result: Dict[str, Any] = {"baseline": baseline, "candidate": candidate, "delta": delta}
    if args.pvalues:
        result["p_values"] = {
            "force_rms": _compute_p_values(
                candidate["force_rms_values"],
                baseline["force_rms_values"],
            ),
            "formation_energy_per_atom": _compute_p_values(
                candidate["formation_energy_per_atom_values"],
                baseline["formation_energy_per_atom_values"],
            ),
        }

    if needs_values and not args.dump_values:
        for block in (baseline, candidate):
            block.pop("force_rms_values", None)
            block.pop("formation_energy_per_atom_values", None)

    print(json.dumps(result, indent=2))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
