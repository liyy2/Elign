from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

# Avoid oversubscribing CPU threads when using multiprocessing with PySCF/BLAS.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from eval_qm9_oracle_energy_force import (
    load_samples,
    stable_filter,
    subsample,
    summarize,
    _guess_spin_from_symbols,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run PySCF geometry optimization on QM9 samples and compare "
            "the number of steps required to relax."
        )
    )
    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        help="Path to baseline samples (eval_rollouts*.pt).",
    )
    parser.add_argument(
        "--candidate",
        type=str,
        required=True,
        help="Path to candidate samples (eval_rollouts*.pt).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=50,
        help="Number of samples to evaluate from each set (default: 50).",
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
        "--xc",
        type=str,
        default="B3LYP",
        help="PySCF XC functional (default: B3LYP).",
    )
    parser.add_argument(
        "--basis",
        type=str,
        default="6-31G(d)",
        help="PySCF basis (default: 6-31G(d)).",
    )
    parser.add_argument(
        "--grid-level",
        type=int,
        default=3,
        help="PySCF integration grid level (default: 3).",
    )
    parser.add_argument(
        "--density-fit",
        action="store_true",
        help="Use density fitting (RI) for PySCF DFT calculations (default: off).",
    )
    parser.add_argument(
        "--max-memory-mb",
        type=int,
        default=None,
        help="Set PySCF mol.max_memory in MB per worker (default: None).",
    )
    parser.add_argument(
        "--scf-max-cycle",
        type=int,
        default=200,
        help="PySCF SCF max cycles (default: 200).",
    )
    parser.add_argument(
        "--scf-conv-tol",
        type=float,
        default=1e-9,
        help="PySCF SCF convergence tolerance (default: 1e-9).",
    )
    parser.add_argument(
        "--geom-max-steps",
        type=int,
        default=100,
        help="Maximum geometry optimization steps (default: 100).",
    )
    parser.add_argument(
        "--n-procs",
        type=int,
        default=1,
        help="Number of parallel PySCF worker processes (default: 1).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional path to write a JSON summary.",
    )
    return parser.parse_args()


def _geomopt_single(
    symbols: List[str],
    coords: List[List[float]],
    dft_cfg: Dict[str, Any],
    geom_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    from pyscf import dft, gto
    from pyscf.geomopt import geometric_solver
    from pyscf import lib as pyscf_lib

    pyscf_lib.num_threads(1)

    spin = _guess_spin_from_symbols(symbols, charge=0)
    atom = [(sym, tuple(map(float, xyz))) for sym, xyz in zip(symbols, coords)]
    mol = gto.M(
        atom=atom,
        basis=dft_cfg["basis"],
        charge=0,
        spin=int(spin),
        unit="Angstrom",
        verbose=0,
    )
    if dft_cfg.get("max_memory_mb") is not None:
        mol.max_memory = float(dft_cfg["max_memory_mb"])

    if spin == 0:
        mf = dft.RKS(mol)
    else:
        mf = dft.UKS(mol)
    mf.xc = dft_cfg["xc"]
    mf.grids.level = int(dft_cfg["grid_level"])
    mf.max_cycle = int(dft_cfg["scf_max_cycle"])
    mf.conv_tol = float(dft_cfg["scf_conv_tol"])
    if dft_cfg.get("density_fit", False):
        mf = mf.density_fit()

    opt = geometric_solver.GeometryOptimizer(mf)
    opt.max_cycle = int(geom_cfg["max_steps"])

    step_counter = {"steps": 0}

    def _callback(env: Dict[str, Any]) -> None:
        try:
            step_counter["steps"] = int(env["self"].cycle)
        except Exception:
            pass

    opt.callback = _callback
    opt.kernel()

    return {"steps": step_counter["steps"], "converged": bool(opt.converged)}


def eval_set(
    name: str,
    samples_path: str,
    dataset_info: Dict[str, Any],
    dft_cfg: Dict[str, Any],
    geom_cfg: Dict[str, Any],
    n_procs: int,
    k: int,
    seed: int,
    stable_only: bool,
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
            }
        )

    steps: List[int] = []
    converged: List[bool] = []
    failures = 0
    failure_types: Dict[str, int] = {}
    failure_examples: List[str] = []

    if n_procs < 1:
        raise ValueError("--n-procs must be >= 1")

    if n_procs == 1:
        from tqdm import tqdm

        for task in tqdm(tasks, desc=f"GeomOpt {name}", unit="mol"):
            try:
                result = _geomopt_single(task["symbols"], task["coords"], dft_cfg, geom_cfg)
            except Exception as exc:
                failures += 1
                exc_name = type(exc).__name__
                failure_types[exc_name] = failure_types.get(exc_name, 0) + 1
                if len(failure_examples) < 5:
                    failure_examples.append(f"{exc_name}: {exc}")
                continue
            steps.append(int(result["steps"]))
            converged.append(bool(result["converged"]))
    else:
        import concurrent.futures as cf
        import multiprocessing as mp
        from tqdm import tqdm

        mp_ctx = mp.get_context("spawn")
        with cf.ProcessPoolExecutor(max_workers=n_procs, mp_context=mp_ctx) as executor:
            futures = [
                executor.submit(_geomopt_single, task["symbols"], task["coords"], dft_cfg, geom_cfg)
                for task in tasks
            ]
            for fut in tqdm(cf.as_completed(futures), total=len(futures), desc=f"GeomOpt {name}", unit="mol"):
                try:
                    result = fut.result()
                except Exception as exc:
                    failures += 1
                    exc_name = type(exc).__name__
                    failure_types[exc_name] = failure_types.get(exc_name, 0) + 1
                    if len(failure_examples) < 5:
                        failure_examples.append(f"{exc_name}: {exc}")
                    continue
                steps.append(int(result["steps"]))
                converged.append(bool(result["converged"]))

    success = len(steps)
    out["dft_cfg"] = dict(dft_cfg)
    out["geom_cfg"] = dict(geom_cfg)
    out["success"] = success
    out["failures"] = failures
    out["failure_types"] = dict(sorted(failure_types.items(), key=lambda kv: (-kv[1], kv[0])))
    out["failure_examples"] = list(failure_examples)
    out["converged"] = int(sum(1 for c in converged if c))
    out["converged_rate"] = (sum(1 for c in converged if c) / success) if success else 0.0
    out["steps"] = summarize(steps)
    out["units"] = {"geom_steps": "steps"}
    return out


def main() -> None:
    args = parse_args()
    for path in [args.baseline, args.candidate]:
        if not Path(path).exists():
            raise FileNotFoundError(path)

    from edm_source.configs.datasets_config import get_dataset_info

    dataset_info = get_dataset_info("qm9", remove_h=False)

    dft_cfg = {
        "xc": args.xc,
        "basis": args.basis,
        "grid_level": int(args.grid_level),
        "density_fit": bool(args.density_fit),
        "max_memory_mb": args.max_memory_mb,
        "scf_max_cycle": int(args.scf_max_cycle),
        "scf_conv_tol": float(args.scf_conv_tol),
    }
    geom_cfg = {"max_steps": int(args.geom_max_steps)}

    baseline = eval_set(
        "baseline",
        args.baseline,
        dataset_info,
        dft_cfg,
        geom_cfg,
        n_procs=int(args.n_procs),
        k=args.k,
        seed=args.seed,
        stable_only=args.stable_only,
    )
    candidate = eval_set(
        "candidate",
        args.candidate,
        dataset_info,
        dft_cfg,
        geom_cfg,
        n_procs=int(args.n_procs),
        k=args.k,
        seed=args.seed,
        stable_only=args.stable_only,
    )

    delta = {
        "steps_mean": candidate["steps"]["mean"] - baseline["steps"]["mean"],
        "converged_rate": candidate["converged_rate"] - baseline["converged_rate"],
    }

    result = {"baseline": baseline, "candidate": candidate, "delta": delta}
    print(json.dumps(result, indent=2))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
