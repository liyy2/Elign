#!/usr/bin/env python
import argparse
import csv
from collections import defaultdict
import os
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import LogLocator, NullFormatter


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _to_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _to_int(value: str, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def _group_xy(rows: List[Dict[str, str]], group_key: str, x_key: str, y_key: str) -> Dict[str, List[Tuple[int, float]]]:
    grouped: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    for row in rows:
        group = row.get(group_key, "unknown")
        x = _to_int(row.get(x_key, "0"))
        y = _to_float(row.get(y_key, "0"))
        grouped[group].append((x, y))
    for k in list(grouped.keys()):
        grouped[k] = sorted(grouped[k], key=lambda t: t[0])
    return grouped


def _pretty_label(s: str) -> str:
    s_low = s.lower()
    if s_low == "posttrained":
        return "Post-trained"
    if s_low == "guided":
        return "Baseline"
    if s_low == "mlff":
        return "MLFF"
    if s_low == "xtb":
        return "xTB"
    if s_low == "dft":
        return "DFT"
    return s


_MODE_STYLE = {
    "guided": {"color": "#0072B2", "marker": "s"},  # blue (Baseline)
    "posttrained": {"color": "#E69F00", "marker": "o"},  # orange
}

_REWARD_STYLE = {
    "dft": {"color": "black", "marker": "o"},
    "mlff": {"color": "#E69F00", "marker": "s"},  # orange
    "xtb": {"color": "#0072B2", "marker": "^"},  # blue
}


def _register_conda_fonts() -> None:
    # If running inside a conda env, packages like `mscorefonts` drop fonts into $CONDA_PREFIX/fonts.
    # Matplotlib doesn't always scan that directory automatically, so we register them explicitly.
    prefix = Path(os.environ.get("CONDA_PREFIX", sys.prefix))
    font_dir = prefix / "fonts"
    if not font_dir.is_dir():
        return
    for ext in ("*.ttf", "*.otf"):
        for p in sorted(font_dir.glob(ext)):
            try:
                font_manager.fontManager.addfont(str(p))
            except Exception:
                # Best-effort: if one font fails, continue.
                continue


def _apply_paper_style(paper_style: str) -> None:
    _register_conda_fonts()

    # Journal-friendly vector exports: embed TrueType fonts (Type 42 / CIDFontType2) so text stays editable.
    common = {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "pdf.use14corefonts": False,
        "text.usetex": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "legend.frameon": False,
    }

    if paper_style == "nature":
        # Nature/Science: Arial-ish, open axes, no grid.
        matplotlib.rcParams.update(
            {
                **common,
                "font.family": "sans-serif",
                "font.sans-serif": ["Arial", "Nimbus Sans", "DejaVu Sans"],
                "mathtext.fontset": "dejavusans",
                "font.size": 8,
                "figure.figsize": (3.3, 2.5),
                "axes.labelsize": 8,
                "axes.titlesize": 11,
                "axes.labelsize": 10,
                "axes.linewidth": 1.0,
                "axes.spines.top": True,
                "axes.spines.right": True,
                "axes.grid": True,
                "grid.color": "0.85",
                "grid.linestyle": "--",
                "grid.linewidth": 0.6,
                "grid.alpha": 0.8,
                "xtick.labelsize": 8,
                "ytick.labelsize": 8,
                "xtick.direction": "out",
                "ytick.direction": "out",
                "xtick.major.size": 3.5,
                "ytick.major.size": 3.5,
                "xtick.minor.size": 2.0,
                "ytick.minor.size": 2.0,
                "xtick.major.width": 1.0,
                "ytick.major.width": 1.0,
                "xtick.minor.width": 0.8,
                "ytick.minor.width": 0.8,
                "lines.linewidth": 2.0,
                "lines.markersize": 5.5,
                "legend.fontsize": 8,
                "legend.frameon": True,
                "legend.framealpha": 0.95,
                "legend.facecolor": "white",
                "legend.edgecolor": "0.85",
                "legend.fancybox": False,
            }
        )
        return

    # "open" style: clean, modern axes (Nature-like but without boxed spines).
    matplotlib.rcParams.update(
        {
            **common,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Nimbus Sans", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "font.size": 8,
            "figure.figsize": (3.3, 2.5),
            "axes.labelsize": 8,
            "axes.linewidth": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.minor.size": 1.6,
            "ytick.minor.size": 1.6,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "lines.linewidth": 1.5,
            "lines.markersize": 4.5,
        }
    )


def _format_axes(ax, *, logy: bool = False, boxed: bool = False) -> None:
    ax.minorticks_on()
    ax.set_axisbelow(True)
    ax.tick_params(
        axis="both",
        which="both",
        top=boxed,
        right=boxed,
        direction="in" if boxed else "out",
        pad=1.5,
    )

    if logy:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(0.2, 0.4, 0.6, 0.8)))
        ax.yaxis.set_minor_formatter(NullFormatter())
        # No gridlines by default (Nature/Science often omit); minor ticks give scale cues.


def _set_xticks_from_data(ax, xs: List[int]) -> None:
    uniq = sorted({int(x) for x in xs})
    ax.set_xticks(uniq)
    ax.set_xlim(min(uniq) - 0.02 * (max(uniq) - min(uniq)), max(uniq) + 0.02 * (max(uniq) - min(uniq)))


def _save(fig, out_base: Path) -> None:
    fig.savefig(out_base.with_suffix(".png"), dpi=300)
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot scaling curves from benchmark CSVs.")
    parser.add_argument("--inference-csv", type=str, default="benchmarks/results/inference_scaling.csv")
    parser.add_argument("--training-csv", type=str, default="benchmarks/results/training_scaling.csv")
    parser.add_argument("--out-dir", type=str, default="benchmarks/plots")
    parser.add_argument("--paper-style", type=str, choices=["nature", "open"], default="nature")
    args = parser.parse_args()

    _apply_paper_style(args.paper_style)
    boxed = False

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Inference: seconds/molecule vs n_nodes (preferred; more interpretable than throughput)
    inf_rows = _read_csv(Path(args.inference_csv))
    if inf_rows and "sec_per_molecule" not in inf_rows[0]:
        # Back-compat: derive from elapsed/batch when older CSVs are used.
        for row in inf_rows:
            bs = _to_float(row.get("batch_size", "0"), default=0.0)
            elapsed = _to_float(row.get("elapsed_sec", "0"), default=0.0)
            row["sec_per_molecule"] = str(elapsed / bs) if bs > 0 else "0"
    inf_sec = _group_xy(inf_rows, "mode", "n_nodes", "sec_per_molecule")

    fig, ax = plt.subplots()
    for mode in ["posttrained", "guided"]:
        if mode not in inf_sec:
            continue
        points = inf_sec[mode]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        style = _MODE_STYLE.get(mode, {})
        color = style.get("color", None)
        marker = style.get("marker", "o")
        ax.plot(
            xs,
            ys,
            label=_pretty_label(mode),
            color=color,
            marker=marker,
            linestyle="--",
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.7,
        )
    ax.set_xlabel("Number of atoms")
    ax.set_ylabel("Time per molecule (s)")
    _format_axes(ax, boxed=boxed)
    _set_xticks_from_data(ax, xs)
    ax.legend(loc="upper left", handlelength=1.2, handletextpad=0.4, borderaxespad=0.2)
    fig.tight_layout(pad=0.25)
    out_base = out_dir / "inference_sec_per_molecule"
    _save(fig, out_base)

    # Inference (secondary): steps/s vs n_nodes
    inf_steps = _group_xy(inf_rows, "mode", "n_nodes", "steps_per_sec")
    fig, ax = plt.subplots()
    for mode in ["posttrained", "guided"]:
        if mode not in inf_steps:
            continue
        points = inf_steps[mode]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        style = _MODE_STYLE.get(mode, {})
        color = style.get("color", None)
        marker = style.get("marker", "o")
        ax.plot(
            xs,
            ys,
            label=_pretty_label(mode),
            color=color,
            marker=marker,
            linestyle="--",
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.7,
        )
    ax.set_xlabel("Number of atoms")
    ax.set_ylabel("Diffusion steps per second")
    _format_axes(ax, boxed=boxed)
    _set_xticks_from_data(ax, xs)
    ax.legend(loc="upper left", handlelength=1.2, handletextpad=0.4, borderaxespad=0.2)
    fig.tight_layout(pad=0.25)
    out_base = out_dir / "inference_steps_per_sec"
    _save(fig, out_base)

    # Training: epoch wall time vs n_nodes
    train_rows = _read_csv(Path(args.training_csv))
    time_key = "timing/epoch_wall_sec"
    if train_rows and time_key not in train_rows[0]:
        # Fall back to a common alternative key name.
        time_key = "timing/epoch_wall_seconds"
    tr = _group_xy(train_rows, "reward_type", "n_nodes", time_key)

    fig, ax = plt.subplots()
    for reward_type in ["dft", "mlff", "xtb"]:
        if reward_type not in tr:
            continue
        points = tr[reward_type]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        style = _REWARD_STYLE.get(reward_type, {})
        color = style.get("color", None)
        marker = style.get("marker", "o")
        ax.plot(
            xs,
            ys,
            label=_pretty_label(reward_type),
            color=color,
            marker=marker,
            linestyle="--",
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.7,
        )
    ax.set_xlabel("Number of atoms")
    ax.set_ylabel("Epoch time (s)")
    _format_axes(ax, boxed=boxed)
    _set_xticks_from_data(ax, xs)
    ax.legend(loc="upper left", ncol=1, handlelength=1.2, handletextpad=0.4, borderaxespad=0.2)
    fig.tight_layout(pad=0.25)
    out_base = out_dir / "training_epoch_wall_sec"
    _save(fig, out_base)

    # Same plot on a log scale for readability when DFT dominates.
    fig, ax = plt.subplots()
    for reward_type in ["dft", "mlff", "xtb"]:
        if reward_type not in tr:
            continue
        points = tr[reward_type]
        xs = [p[0] for p in points]
        ys = [max(p[1], 1e-6) for p in points]
        style = _REWARD_STYLE.get(reward_type, {})
        color = style.get("color", None)
        marker = style.get("marker", "o")
        ax.plot(
            xs,
            ys,
            label=_pretty_label(reward_type),
            color=color,
            marker=marker,
            linestyle="--",
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.7,
        )
    ax.set_xlabel("Number of atoms")
    ax.set_ylabel("Epoch time (s)")
    _format_axes(ax, logy=True, boxed=boxed)
    _set_xticks_from_data(ax, xs)
    ax.legend(loc="upper left", ncol=1, handlelength=1.2, handletextpad=0.4, borderaxespad=0.2)
    fig.tight_layout(pad=0.25)
    out_base = out_dir / "training_epoch_wall_sec_log"
    _save(fig, out_base)

    # Training (component): reward compute time vs n_nodes (helps explain MLFF vs xTB similarity)
    reward_key = "timing/reward_compute_sec"
    if train_rows and reward_key in train_rows[0]:
        tr_reward = _group_xy(train_rows, "reward_type", "n_nodes", reward_key)
        fig, ax = plt.subplots()
        for reward_type in ["dft", "mlff", "xtb"]:
            if reward_type not in tr_reward:
                continue
            points = tr_reward[reward_type]
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            style = _REWARD_STYLE.get(reward_type, {})
            color = style.get("color", None)
            marker = style.get("marker", "o")
            ax.plot(
                xs,
                ys,
                label=_pretty_label(reward_type),
                color=color,
                marker=marker,
                linestyle="--",
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.7,
            )
        ax.set_xlabel("Number of atoms")
        ax.set_ylabel("Reward compute time (s)")
        _format_axes(ax, boxed=boxed)
        _set_xticks_from_data(ax, xs)
        ax.legend(loc="upper left", ncol=1, handlelength=1.2, handletextpad=0.4, borderaxespad=0.2)
        fig.tight_layout(pad=0.25)
        out_base = out_dir / "training_reward_compute_sec"
        _save(fig, out_base)

        fig, ax = plt.subplots()
        for reward_type in ["dft", "mlff", "xtb"]:
            if reward_type not in tr_reward:
                continue
            points = tr_reward[reward_type]
            xs = [p[0] for p in points]
            ys = [max(p[1], 1e-6) for p in points]
            style = _REWARD_STYLE.get(reward_type, {})
            color = style.get("color", None)
            marker = style.get("marker", "o")
            ax.plot(
                xs,
                ys,
                label=_pretty_label(reward_type),
                color=color,
                marker=marker,
                linestyle="--",
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.7,
            )
        ax.set_xlabel("Number of atoms")
        ax.set_ylabel("Reward compute time (s)")
        _format_axes(ax, logy=True, boxed=boxed)
        _set_xticks_from_data(ax, xs)
        ax.legend(loc="upper left", ncol=1, handlelength=1.2, handletextpad=0.4, borderaxespad=0.2)
        fig.tight_layout(pad=0.25)
        out_base = out_dir / "training_reward_compute_sec_log"
        _save(fig, out_base)

    print(f"Wrote plots to {out_dir}")


if __name__ == "__main__":
    main()
