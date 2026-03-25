#!/usr/bin/env python3
"""
Offline DFT (PySCF) comparison for GEOM samples.

Pipeline:
1) Load two eval_verl_rollout.py sample files (baseline + post-trained).
2) Run a PySCF DFT single-point + nuclear gradient for each molecule.
3) Save per-sample metrics to CSV.
4) Produce a Nature-style PDF plot (TrueType fonts; pdf.fonttype=42).

Notes
-----
- This is CPU-bound and can be slow for large molecules. Use `--workers` to parallelize.
- The default XC/basis are chosen for robustness across GEOM elements (including heavier atoms).
  Adjust to your preferred "oracle" settings as needed.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
EDM_SOURCE_ROOT = REPO_ROOT / "edm_source"
import sys

for path in (REPO_ROOT, EDM_SOURCE_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from edm_source.configs.datasets_config import get_dataset_info

HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANGSTROM = 0.529177210903
HARTREE_PER_BOHR_TO_EV_PER_A = HARTREE_TO_EV / BOHR_TO_ANGSTROM

# GEOM elements covered by the dataset config.
ATOMIC_NUMBER_MAP: Dict[str, int] = {
    "H": 1,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "AL": 13,
    "SI": 14,
    "P": 15,
    "S": 16,
    "CL": 17,
    "AS": 33,
    "BR": 35,
    "I": 53,
    "HG": 80,
    "BI": 83,
}


@dataclass(frozen=True)
class DFTSettings:
    xc: str
    basis: str
    charge: int
    spin: Optional[int]
    max_cycle: int
    conv_tol: float
    grids_level: int
    density_fit: bool
    accept_unconverged: bool
    min_interatomic_distance: float


def _set_thread_env() -> None:
    # Avoid massive oversubscription when using multiprocessing.
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(key, "1")


def _min_pairwise_distance(positions: np.ndarray) -> float:
    if positions.shape[0] <= 1:
        return float("inf")
    diff = positions[:, None, :] - positions[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist, np.inf)
    return float(dist.min())


def _infer_spin(atomic_numbers: np.ndarray, charge: int) -> int:
    nelec = int(np.sum(atomic_numbers)) - int(charge)
    if nelec < 0:
        return 0
    # Minimal parity-based guess: even -> singlet, odd -> doublet.
    return int(nelec % 2)


def _symbols_and_atomic_numbers(atom_type_indices: np.ndarray, decoder: Sequence[str]) -> Tuple[List[str], np.ndarray]:
    symbols: List[str] = [str(decoder[int(idx)]).strip() for idx in atom_type_indices.tolist()]
    numbers: List[int] = []
    for sym in symbols:
        key = sym.upper()
        z = ATOMIC_NUMBER_MAP.get(key)
        if z is None:
            raise ValueError(f"Unsupported element '{sym}' for DFT oracle")
        numbers.append(int(z))
    return symbols, np.asarray(numbers, dtype=np.int32)


def _load_samples(samples_path: Path) -> List[Dict[str, Any]]:
    payload = torch.load(samples_path, map_location="cpu")
    if isinstance(payload, dict) and "samples" in payload:
        samples = payload["samples"]
    else:
        samples = payload
    if not isinstance(samples, list):
        raise ValueError(f"Unrecognized samples format in {samples_path}")
    return samples


def _prepare_tasks(samples: List[Dict[str, Any]]) -> List[Tuple[int, int, np.ndarray, np.ndarray]]:
    tasks: List[Tuple[int, int, np.ndarray, np.ndarray]] = []
    for idx, s in enumerate(samples):
        num_atoms = int(s["num_atoms"])
        atom_types = s["atom_types"]
        positions = s["positions"]
        if isinstance(atom_types, torch.Tensor):
            atom_types_np = atom_types.detach().cpu().numpy().astype(np.int32)
        else:
            atom_types_np = np.asarray(atom_types, dtype=np.int32)
        if isinstance(positions, torch.Tensor):
            positions_np = positions.detach().cpu().numpy().astype(np.float64)
        else:
            positions_np = np.asarray(positions, dtype=np.float64)
        tasks.append((idx, num_atoms, atom_types_np, positions_np))
    return tasks


def _eval_one_dft(
    task: Tuple[int, int, np.ndarray, np.ndarray],
    decoder: Sequence[str],
    settings: DFTSettings,
) -> Dict[str, Any]:
    idx, num_atoms, atom_types, positions_ang = task
    start = time.time()

    # Local import keeps the parent process light and avoids fork-after-import issues.
    from pyscf import dft, gto, lib  # noqa: WPS433

    try:
        lib.num_threads(1)
    except Exception:
        pass

    result: Dict[str, Any] = {
        "idx": int(idx),
        "num_atoms": int(num_atoms),
        "success": False,
        "converged": False,
        "energy_hartree": float("nan"),
        "force_rms_hartree_per_bohr": float("nan"),
        "runtime_s": float("nan"),
        "error": "",
    }

    try:
        if num_atoms <= 0:
            result["error"] = "empty_molecule"
            return result

        if positions_ang.shape[0] < num_atoms:
            raise ValueError("positions shorter than num_atoms")
        if atom_types.shape[0] < num_atoms:
            raise ValueError("atom_types shorter than num_atoms")

        pos = positions_ang[:num_atoms, :3].astype(np.float64, copy=False)
        if _min_pairwise_distance(pos) < float(settings.min_interatomic_distance):
            result["error"] = "min_interatomic_distance"
            return result

        symbols, atomic_numbers = _symbols_and_atomic_numbers(atom_types[:num_atoms], decoder)

        spin = settings.spin
        if spin is None:
            spin = _infer_spin(atomic_numbers, settings.charge)

        mol = gto.Mole()
        mol.atom = [(sym, (float(x), float(y), float(z))) for sym, (x, y, z) in zip(symbols, pos)]
        mol.unit = "Angstrom"
        mol.basis = settings.basis
        mol.charge = int(settings.charge)
        mol.spin = int(spin)
        mol.verbose = 0
        mol.build()

        mf = dft.RKS(mol) if int(spin) == 0 else dft.UKS(mol)
        mf.xc = str(settings.xc)
        mf.conv_tol = float(settings.conv_tol)
        mf.max_cycle = int(settings.max_cycle)
        mf.verbose = 0
        try:
            mf.grids.level = int(settings.grids_level)
        except Exception:
            pass
        if bool(settings.density_fit):
            try:
                mf = mf.density_fit()
            except Exception:
                pass

        energy = float(mf.kernel())
        converged = bool(getattr(mf, "converged", False))
        if not converged and not bool(settings.accept_unconverged):
            result["error"] = "scf_unconverged"
            result["converged"] = False
            result["energy_hartree"] = energy
            return result

        grad = mf.nuc_grad_method().kernel()  # dE/dR in Hartree/Bohr
        forces = -np.asarray(grad, dtype=np.float64)
        rms = float(np.sqrt(np.mean(np.sum(forces * forces, axis=1))))

        result["success"] = True
        result["converged"] = converged
        result["energy_hartree"] = energy
        result["force_rms_hartree_per_bohr"] = rms
        return result
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result
    finally:
        result["runtime_s"] = float(time.time() - start)


def _write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "idx",
        "num_atoms",
        "success",
        "converged",
        "energy_hartree",
        "energy_ev",
        "energy_ev_per_atom",
        "force_rms_hartree_per_bohr",
        "force_rms_ev_per_a",
        "runtime_s",
        "error",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            energy_h = float(row.get("energy_hartree", float("nan")))
            force_rms = float(row.get("force_rms_hartree_per_bohr", float("nan")))
            n = int(row.get("num_atoms", 0) or 0)
            energy_ev = energy_h * HARTREE_TO_EV if math.isfinite(energy_h) else float("nan")
            force_ev_per_a = (
                force_rms * HARTREE_PER_BOHR_TO_EV_PER_A if math.isfinite(force_rms) else float("nan")
            )
            row_out = dict(row)
            row_out["energy_ev"] = energy_ev
            row_out["energy_ev_per_atom"] = energy_ev / float(n) if n > 0 and math.isfinite(energy_ev) else float("nan")
            row_out["force_rms_ev_per_a"] = force_ev_per_a
            writer.writerow({k: row_out.get(k, "") for k in fieldnames})


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _to_float(rows: List[Dict[str, Any]], key: str) -> np.ndarray:
    values: List[float] = []
    for r in rows:
        try:
            values.append(float(r.get(key, "nan")))
        except Exception:
            values.append(float("nan"))
    return np.asarray(values, dtype=np.float64)


def _plot(
    base_rows: List[Dict[str, Any]],
    post_rows: List[Dict[str, Any]],
    output_pdf: Path,
    title: str,
) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.2,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.fontsize": 8.0,
            # Prefer Helvetica-like sans fonts; fall back to DejaVu Sans (bundled).
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        }
    )

    # Filter to successful DFT calls.
    def ok(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for r in rows:
            s = str(r.get("success", "")).lower()
            if s in {"true", "1", "yes"}:
                out.append(r)
        return out

    base_ok = ok(base_rows)
    post_ok = ok(post_rows)

    base_e = _to_float(base_ok, "energy_ev_per_atom")
    post_e = _to_float(post_ok, "energy_ev_per_atom")
    base_f = _to_float(base_ok, "force_rms_ev_per_a")
    post_f = _to_float(post_ok, "force_rms_ev_per_a")

    # Drop NaNs (should be rare when success=True).
    base_e = base_e[np.isfinite(base_e)]
    post_e = post_e[np.isfinite(post_e)]
    base_f = base_f[np.isfinite(base_f)]
    post_f = post_f[np.isfinite(post_f)]

    labels = ["Pretrained", "Post-trained"]
    colors = ["#4D4D4D", "#0072B2"]

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0), constrained_layout=True)
    fig.suptitle(title)

    def violin(ax: Any, data_a: np.ndarray, data_b: np.ndarray, ylabel: str, panel: str) -> None:
        parts = ax.violinplot([data_a, data_b], positions=[0, 1], widths=0.75, showmeans=False, showmedians=False)
        for i, body in enumerate(parts["bodies"]):
            body.set_facecolor(colors[i])
            body.set_edgecolor(colors[i])
            body.set_alpha(0.25)
            body.set_linewidth(1.0)
        for key in ("cbars", "cmins", "cmaxes"):
            if key in parts:
                parts[key].set_color("#333333")
                parts[key].set_linewidth(0.8)

        # Box (median + IQR) overlay.
        box = ax.boxplot(
            [data_a, data_b],
            positions=[0, 1],
            widths=0.28,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#111111", "linewidth": 1.2},
            boxprops={"linewidth": 0.9, "edgecolor": "#111111"},
            whiskerprops={"linewidth": 0.9, "color": "#111111"},
            capprops={"linewidth": 0.9, "color": "#111111"},
        )
        for i, patch in enumerate(box["boxes"]):
            patch.set_facecolor("white")
            patch.set_alpha(0.9)

        # Individual points (jittered), light for Nature-style readability.
        rng = np.random.default_rng(0)
        for i, vals in enumerate([data_a, data_b]):
            x = rng.normal(loc=float(i), scale=0.06, size=vals.shape[0])
            ax.scatter(
                x,
                vals,
                s=8,
                alpha=0.35,
                linewidths=0.0,
                color=colors[i],
                rasterized=True,
            )

        ax.set_xticks([0, 1], labels)
        ax.set_ylabel(ylabel)
        ax.text(
            -0.18,
            1.04,
            panel,
            transform=ax.transAxes,
            fontsize=9.5,
            fontweight="bold",
            va="top",
            ha="left",
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    violin(axes[0], base_e, post_e, "DFT energy (eV/atom)", "a")
    violin(axes[1], base_f, post_f, "DFT RMS force (eV/A)", "b")

    # Add concise sample-count footer.
    fig.text(
        0.01,
        0.01,
        f"DFT-success: pretrained n={len(base_e)}, post-trained n={len(post_e)} (NaNs dropped)",
        ha="left",
        va="bottom",
        fontsize=7.0,
        color="#333333",
    )

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, dpi=300, transparent=True)
    plt.close(fig)


def _default_workers() -> int:
    try:
        count = os.cpu_count() or 1
    except Exception:
        count = 1
    return max(1, min(4, int(count)))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples-base", type=str, required=True, help="Path to baseline samples (torch.save).")
    p.add_argument("--samples-post", type=str, required=True, help="Path to post-trained samples (torch.save).")
    p.add_argument("--out-dir", type=str, required=True, help="Output directory for CSVs and plot.")

    p.add_argument("--dataset", type=str, default="geom", help="Dataset name (default: geom).")
    p.add_argument("--remove-h", action="store_true", help="Use the no-H dataset config (not supported for geom).")

    p.add_argument("--xc", type=str, default="pbe", help="DFT XC functional (PySCF string).")
    p.add_argument(
        "--basis",
        type=str,
        default="minao",
        help="DFT basis set. Default 'minao' is robust for GEOM elements incl. Hg/Bi.",
    )
    p.add_argument("--charge", type=int, default=0, help="Total charge (default: 0).")
    p.add_argument("--spin", type=int, default=None, help="Spin (2S) for PySCF; default infers from parity.")
    p.add_argument("--max-cycle", type=int, default=30, help="SCF max cycles (default: 30).")
    p.add_argument("--conv-tol", type=float, default=1e-6, help="SCF convergence tolerance (default: 1e-6).")
    p.add_argument("--grids-level", type=int, default=1, help="DFT grids level (default: 1).")
    p.add_argument("--no-density-fit", action="store_true", help="Disable density fitting (slower).")
    p.add_argument("--reject-unconverged", action="store_true", help="Drop unconverged SCF results.")
    p.add_argument(
        "--min-interatomic-distance",
        type=float,
        default=0.6,
        help="Skip molecules with any pair closer than this threshold in Angstrom (default: 0.6).",
    )
    p.add_argument("--workers", type=int, default=None, help="Parallel worker processes (default: min(4, cpu_count)).")
    p.add_argument("--force", action="store_true", help="Recompute even if CSVs already exist.")
    p.add_argument(
        "--title",
        type=str,
        default="GEOM DFT oracle comparison",
        help="Figure title.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _set_thread_env()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_info = get_dataset_info(str(args.dataset), bool(args.remove_h))
    decoder = dataset_info.get("atom_decoder")
    if not isinstance(decoder, list):
        raise ValueError("dataset_info missing atom_decoder")

    settings = DFTSettings(
        xc=str(args.xc),
        basis=str(args.basis),
        charge=int(args.charge),
        spin=None if args.spin is None else int(args.spin),
        max_cycle=int(args.max_cycle),
        conv_tol=float(args.conv_tol),
        grids_level=int(args.grids_level),
        density_fit=not bool(args.no_density_fit),
        accept_unconverged=not bool(args.reject_unconverged),
        min_interatomic_distance=float(args.min_interatomic_distance),
    )

    base_samples_path = Path(args.samples_base).resolve()
    post_samples_path = Path(args.samples_post).resolve()
    if not base_samples_path.exists():
        raise FileNotFoundError(base_samples_path)
    if not post_samples_path.exists():
        raise FileNotFoundError(post_samples_path)

    base_csv = out_dir / "dft_base.csv"
    post_csv = out_dir / "dft_post.csv"

    workers = int(args.workers) if args.workers is not None else _default_workers()
    workers = max(1, workers)

    def eval_file(samples_path: Path, out_csv: Path) -> List[Dict[str, Any]]:
        if out_csv.exists() and not bool(args.force):
            return _read_csv(out_csv)

        samples = _load_samples(samples_path)
        tasks = _prepare_tasks(samples)

        # Store settings snapshot next to the CSV for provenance.
        (out_csv.parent / (out_csv.stem + "_settings.json")).write_text(
            json.dumps({**asdict(settings), "samples_path": str(samples_path)}, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        rows: List[Dict[str, Any]] = [None] * len(tasks)  # type: ignore[list-item]

        if workers == 1:
            for task in tqdm(tasks, desc=f"DFT {out_csv.stem}", unit="mol"):
                res = _eval_one_dft(task, decoder, settings)
                rows[int(res["idx"])] = res
        else:
            import multiprocessing as mp

            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=workers) as pool:
                it = pool.imap_unordered(
                    _worker_entry,
                    [(task, decoder, settings) for task in tasks],
                    chunksize=1,
                )
                for res in tqdm(it, total=len(tasks), desc=f"DFT {out_csv.stem}", unit="mol"):
                    rows[int(res["idx"])] = res

        # Defensive: replace any None rows (shouldn't happen) with failures.
        for i, row in enumerate(rows):
            if row is None:
                rows[i] = {
                    "idx": i,
                    "num_atoms": int(tasks[i][1]),
                    "success": False,
                    "converged": False,
                    "energy_hartree": float("nan"),
                    "force_rms_hartree_per_bohr": float("nan"),
                    "runtime_s": float("nan"),
                    "error": "missing_result",
                }

        _write_csv(rows, out_csv)
        return rows

    base_rows = eval_file(base_samples_path, base_csv)
    post_rows = eval_file(post_samples_path, post_csv)

    plot_pdf = out_dir / "dft_energy_force_compare.pdf"
    _plot(base_rows, post_rows, plot_pdf, title=str(args.title))

    print(f"Wrote: {base_csv}")
    print(f"Wrote: {post_csv}")
    print(f"Wrote: {plot_pdf}")


def _worker_entry(payload: Tuple[Tuple[int, int, np.ndarray, np.ndarray], Sequence[str], DFTSettings]) -> Dict[str, Any]:
    task, decoder, settings = payload
    return _eval_one_dft(task, decoder, settings)


if __name__ == "__main__":
    main()
