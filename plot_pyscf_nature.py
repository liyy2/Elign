from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def _nature_rcparams() -> Dict[str, Any]:
    # Nature-like aesthetics: clean spines, compact typography, no grid by default.
    return {
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.size": 7,
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "legend.fontsize": 6.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        # Force TrueType fonts in PDF/PS outputs (Type 42)
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        # Prefer common sans-serif fonts; fall back if missing.
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Nature-style PDF figures from eval_qm9_oracle_energy_force.py JSON outputs."
    )
    parser.add_argument("--input", type=str, required=True, help="Path to *_dump.json file.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory. Defaults to <input-parent>/plots_nature_pdf.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="pyscf",
        help="Filename prefix for outputs (default: pyscf).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    obj = json.loads(in_path.read_text())

    out_dir = Path(args.out_dir) if args.out_dir else (in_path.parent / "plots_nature_pdf")
    out_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update(_nature_rcparams())

    base_force = np.array(obj["baseline"]["force_rms_values"], dtype=float)
    cand_force = np.array(obj["candidate"]["force_rms_values"], dtype=float)
    base_e = np.array(obj["baseline"]["formation_energy_per_atom_values"], dtype=float)
    cand_e = np.array(obj["candidate"]["formation_energy_per_atom_values"], dtype=float)

    # Use a square 2×2 layout so each panel can be square.
    # (Matplotlib accounts for labels/legends via tight_layout; box_aspect enforces square axes boxes.)
    fig_size_in = 175.0 / 25.4  # ~2-column width, but square overall
    fig, axes = plt.subplots(2, 2, figsize=(fig_size_in, fig_size_in))

    # Force RMS histogram (linear) with percentile-based cap for readability.
    ax = axes[0, 0]
    ax.set_box_aspect(1)
    cap = float(max(np.percentile(base_force, 99.0), np.percentile(cand_force, 99.0)))
    bins = np.linspace(0.0, max(cap, 1e-6), 50)
    ax.hist(base_force, bins=bins, alpha=0.55, label="Baseline", density=True, color="#4C72B0")
    ax.hist(cand_force, bins=bins, alpha=0.55, label="Post-trained", density=True, color="#DD8452")
    ax.axvline(base_force.mean(), color="#4C72B0", linestyle="--", linewidth=1.0)
    ax.axvline(cand_force.mean(), color="#DD8452", linestyle="--", linewidth=1.0)
    ax.set_title("Force RMS (DFT)")
    ax.set_xlabel("eV/Å")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)

    # Force RMS histogram (log-x), showing tails.
    ax = axes[0, 1]
    ax.set_box_aspect(1)
    base_pos = base_force[base_force > 0]
    cand_pos = cand_force[cand_force > 0]
    log_bins = np.logspace(
        np.log10(min(base_pos.min(), cand_pos.min())),
        np.log10(max(base_pos.max(), cand_pos.max())),
        60,
    )
    ax.hist(base_pos, bins=log_bins, alpha=0.55, label="Baseline", density=True, color="#4C72B0")
    ax.hist(cand_pos, bins=log_bins, alpha=0.55, label="Post-trained", density=True, color="#DD8452")
    ax.set_xscale("log")
    ax.set_title("Force RMS (log scale)")
    ax.set_xlabel("eV/Å")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)

    # Energy/atom histogram.
    ax = axes[1, 0]
    ax.set_box_aspect(1)
    e_min = float(min(np.percentile(base_e, 0.5), np.percentile(cand_e, 0.5)))
    e_max = float(max(np.percentile(base_e, 99.5), np.percentile(cand_e, 99.5)))
    e_bins = np.linspace(e_min, e_max, 60)
    ax.hist(base_e, bins=e_bins, alpha=0.55, label="Baseline", density=True, color="#4C72B0")
    ax.hist(cand_e, bins=e_bins, alpha=0.55, label="Post-trained", density=True, color="#DD8452")
    ax.axvline(base_e.mean(), color="#4C72B0", linestyle="--", linewidth=1.0)
    ax.axvline(cand_e.mean(), color="#DD8452", linestyle="--", linewidth=1.0)
    ax.set_title("Formation energy / atom (DFT)")
    ax.set_xlabel("eV/atom")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)

    # ECDFs for force and energy.
    ax = axes[1, 1]
    ax.set_box_aspect(1)

    def ecdf(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.sort(x)
        y = np.arange(1, len(x) + 1, dtype=float) / float(len(x))
        return x, y

    x0, y0 = ecdf(base_force)
    x1, y1 = ecdf(cand_force)
    ax.plot(x0, y0, label="Force baseline", color="#4C72B0", linewidth=1.2)
    ax.plot(x1, y1, label="Force post-trained", color="#DD8452", linewidth=1.2)
    ax.set_xlabel("Force RMS (eV/Å)")
    ax.set_ylabel("ECDF")
    ax.set_title("Empirical CDFs")

    ax2 = ax.twiny()
    x0e, y0e = ecdf(base_e)
    x1e, y1e = ecdf(cand_e)
    ax2.plot(x0e, y0e, label="Energy baseline", color="#4C72B0", linestyle="--", linewidth=1.2)
    ax2.plot(x1e, y1e, label="Energy post-trained", color="#DD8452", linestyle="--", linewidth=1.2)
    ax2.set_xlabel("Formation energy / atom (eV/atom)")

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, frameon=False, fontsize=6, loc="lower right")

    fig.tight_layout()
    fig_path = out_dir / f"{args.prefix}_comparison.pdf"
    fig.savefig(fig_path)
    plt.close(fig)

    # Force boxplot (log y) for tail visibility.
    fig, ax = plt.subplots(figsize=(85.0 / 25.4, 85.0 / 25.4))  # square, ~1-column width
    ax.boxplot([base_force, cand_force], tick_labels=["Baseline", "Post-trained"], showfliers=True)
    ax.set_box_aspect(1)
    ax.set_yscale("log")
    ax.set_ylabel("Force RMS (eV/Å)")
    ax.set_title("Force RMS (log scale)")
    fig.tight_layout()
    fig_path2 = out_dir / f"{args.prefix}_force_box_log.pdf"
    fig.savefig(fig_path2)
    plt.close(fig)

    print(f"Wrote: {fig_path}")
    print(f"Wrote: {fig_path2}")


if __name__ == "__main__":
    main()
