import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot post-hoc force-alignment diagnostics produced by analyze_force_alignment_posthoc.py."
    )
    parser.add_argument("--json", type=str, default=None, help="Path to force_alignment_posthoc*.json.")
    parser.add_argument("--npz", type=str, default=None, help="Path to pairs npz produced with --save-npz.")
    parser.add_argument("--out-dir", type=str, required=True, help="Directory to write PNG plots.")
    parser.add_argument("--prefix", type=str, default="force_alignment", help="Filename prefix for plots.")
    parser.add_argument("--max-scatter", type=int, default=50000, help="Max points to draw in scatter plots.")
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png"],
        help="One or more output formats to save (e.g. png pdf).",
    )
    parser.add_argument(
        "--style",
        choices=["default", "nature"],
        default="default",
        help="Plot styling preset. 'nature' aims for a clean publication look.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    import json

    return json.loads(path.read_text())


def _ensure_out_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def _apply_style(style: str) -> None:
    if style != "nature":
        return
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "legend.frameon": False,
            "axes.linewidth": 0.8,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "lines.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _beautify_ax(ax, style: str) -> None:
    if style != "nature":
        return
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")


def _savefig(fig, base_path: Path, formats: list[str]) -> None:
    for fmt in formats:
        fmt = fmt.lower().lstrip(".")
        out_path = base_path.with_suffix(f".{fmt}")
        save_kwargs = {"bbox_inches": "tight"}
        if fmt in {"png", "jpg", "jpeg", "tif", "tiff"}:
            save_kwargs["dpi"] = 300
        fig.savefig(out_path, **save_kwargs)


def _maybe_downsample(x: np.ndarray, max_n: int, rng: np.random.Generator) -> np.ndarray:
    n = int(x.shape[0])
    if n <= max_n:
        return np.arange(n, dtype=np.int64)
    return rng.choice(n, size=max_n, replace=False)


def _plot_histograms(arr: Dict[str, np.ndarray], out_dir: Path, prefix: str, formats: list[str], style: str) -> None:
    import matplotlib.pyplot as plt

    bins = np.linspace(-1.0, 1.0, 81)

    fig = plt.figure(figsize=(3.5, 2.6) if style == "nature" else (7.5, 4.5))
    ax = plt.gca()
    ax.hist(
        arr["cos_delta"],
        bins=bins,
        density=True,
        alpha=0.75,
        label=r"$\cos(\Delta\mu_{\mathrm{pos}}, \mathbf{F})$",
        edgecolor="black" if style == "nature" else None,
        linewidth=0.5 if style == "nature" else None,
    )
    ax.axvline(np.nanmean(arr["cos_delta"]), color="k", linestyle="--", linewidth=0.8 if style == "nature" else 1)
    ax.set_xlabel("cosine")
    ax.set_ylabel("density")
    if style != "nature":
        ax.set_title("Cosine histogram: drift delta vs force")
        ax.legend()
    _beautify_ax(ax, style)
    _savefig(fig, out_dir / f"{prefix}_hist_cos_delta", formats)
    plt.close(fig)

    abs_bins = np.linspace(0.0, 1.0, 51)
    fig = plt.figure(figsize=(3.5, 2.6) if style == "nature" else (7.5, 4.5))
    ax = plt.gca()
    abs_cos = np.abs(arr["cos_delta"])
    ax.hist(
        abs_cos,
        bins=abs_bins,
        density=True,
        alpha=0.85,
        label=r"$|\cos(\Delta\mu_{\mathrm{pos}}, \mathbf{F})|$",
        edgecolor="black" if style == "nature" else None,
        linewidth=0.5 if style == "nature" else None,
    )
    ax.axvline(np.nanmean(abs_cos), color="k", linestyle="--", linewidth=0.8 if style == "nature" else 1)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel(r"$|\cos|$")
    ax.set_ylabel("density")
    if style != "nature":
        ax.set_title("Absolute cosine histogram: drift delta vs force")
        ax.legend()
    _beautify_ax(ax, style)
    _savefig(fig, out_dir / f"{prefix}_hist_abs_cos_delta", formats)
    plt.close(fig)

    fig = plt.figure(figsize=(3.5, 2.6) if style == "nature" else (7.5, 4.5))
    ax = plt.gca()
    ax.hist(
        arr["cos_step_pre"],
        bins=bins,
        density=True,
        alpha=0.5,
        label=r"$\cos(\mu_{\mathrm{pre}}-z_t,\mathbf{F})$",
        histtype="stepfilled" if style == "nature" else "bar",
    )
    ax.hist(
        arr["cos_step_post"],
        bins=bins,
        density=True,
        alpha=0.5,
        label=r"$\cos(\mu_{\mathrm{post}}-z_t,\mathbf{F})$",
        histtype="stepfilled" if style == "nature" else "bar",
    )
    ax.axvline(np.nanmean(arr["cos_step_pre"]), color="C0", linestyle="--", linewidth=0.8 if style == "nature" else 1)
    ax.axvline(np.nanmean(arr["cos_step_post"]), color="C1", linestyle="--", linewidth=0.8 if style == "nature" else 1)
    ax.set_xlabel("cosine")
    ax.set_ylabel("density")
    if style != "nature":
        ax.set_title("Cosine histogram: actual reverse-step vs force")
        ax.legend()
    _beautify_ax(ax, style)
    _savefig(fig, out_dir / f"{prefix}_hist_cos_step_pre_post", formats)
    plt.close(fig)


def _plot_per_timestep(arr: Dict[str, np.ndarray], out_dir: Path, prefix: str, formats: list[str], style: str) -> None:
    import matplotlib.pyplot as plt

    t = arr["timestep"].astype(np.int64)
    unique_t = np.unique(t)[::-1]
    means_delta = []
    means_pre = []
    means_post = []
    counts = []
    for ti in unique_t:
        mask = t == ti
        counts.append(int(mask.sum()))
        means_delta.append(float(np.nanmean(arr["cos_delta"][mask])))
        means_pre.append(float(np.nanmean(arr["cos_step_pre"][mask])))
        means_post.append(float(np.nanmean(arr["cos_step_post"][mask])))

    fig = plt.figure(figsize=(3.7, 2.6) if style == "nature" else (8.0, 4.8))
    ax = plt.gca()
    ax.plot(unique_t, means_delta, marker="o", markersize=3, label=r"$\cos(\Delta\mu_{\mathrm{pos}},\mathbf{F})$")
    ax.plot(unique_t, means_pre, marker="o", markersize=3, label=r"$\cos(\mu_{\mathrm{pre}}-z_t,\mathbf{F})$")
    ax.plot(unique_t, means_post, marker="o", markersize=3, label=r"$\cos(\mu_{\mathrm{post}}-z_t,\mathbf{F})$")
    ax.axhline(0.0, color="k", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("diffusion timestep $s$")
    ax.set_ylabel("mean cosine")
    if style != "nature":
        ax.set_title("Mean cosine vs diffusion timestep")
        ax.legend()
    _beautify_ax(ax, style)
    _savefig(fig, out_dir / f"{prefix}_per_timestep_mean_cos", formats)
    plt.close(fig)

    fig = plt.figure(figsize=(3.7, 1.9) if style == "nature" else (8.0, 2.2))
    ax = plt.gca()
    ax.bar(unique_t, counts)
    ax.set_xlabel("diffusion timestep $s$")
    ax.set_ylabel("#pairs")
    if style != "nature":
        ax.set_title("Stored pairs per timestep")
    _beautify_ax(ax, style)
    _savefig(fig, out_dir / f"{prefix}_per_timestep_counts", formats)
    plt.close(fig)


def _plot_scatter(
    arr: Dict[str, np.ndarray],
    out_dir: Path,
    prefix: str,
    max_scatter: int,
    formats: list[str],
    style: str,
    seed: int = 0,
) -> None:
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(int(seed))
    idx = _maybe_downsample(arr["force_norm"], max_scatter, rng)
    x = arr["force_norm"][idx].astype(np.float64)
    y = arr["proj_delta"][idx].astype(np.float64)
    c = arr["cos_delta"][idx].astype(np.float64)

    # Force norms can have heavy tails; clip for readability.
    x_clip = np.clip(x, 1e-6, np.percentile(x, 99.5))
    y_clip = np.clip(y, np.percentile(y, 0.5), np.percentile(y, 99.5))

    fig = plt.figure(figsize=(3.7, 2.8) if style == "nature" else (7.5, 4.8))
    ax = plt.gca()
    ax.scatter(x_clip, y_clip, s=4, alpha=0.18)
    ax.set_xscale("log")
    ax.set_xlabel(r"$||\mathbf{F}||$ (log)")
    ax.set_ylabel(r"$\Delta\mu_{\mathrm{pos}}\cdot \hat{\mathbf{F}}$")
    if style != "nature":
        ax.set_title("Projection of drift delta onto force direction")
    _beautify_ax(ax, style)
    _savefig(fig, out_dir / f"{prefix}_scatter_proj_delta_vs_force", formats)
    plt.close(fig)

    fig = plt.figure(figsize=(3.7, 2.8) if style == "nature" else (7.5, 4.8))
    ax = plt.gca()
    ax.scatter(x_clip, c, s=4, alpha=0.18)
    ax.set_xscale("log")
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel(r"$||\mathbf{F}||$ (log)")
    ax.set_ylabel(r"$\cos(\Delta\mu_{\mathrm{pos}}, \mathbf{F})$")
    if style != "nature":
        ax.set_title("Cosine vs force magnitude")
    _beautify_ax(ax, style)
    _savefig(fig, out_dir / f"{prefix}_scatter_cos_delta_vs_force", formats)
    plt.close(fig)


def _plot_abs_cos_bar(arr: Dict[str, np.ndarray], out_dir: Path, prefix: str, formats: list[str], style: str) -> None:
    import matplotlib.pyplot as plt

    mean_abs_delta = float(np.nanmean(np.abs(arr["cos_delta"])))
    mean_abs_pre = float(np.nanmean(np.abs(arr["cos_step_pre"])))
    mean_abs_post = float(np.nanmean(np.abs(arr["cos_step_post"])))

    labels = ["|cos(delta)|", "|cos(pre step)|", "|cos(post step)|"]
    values = [mean_abs_delta, mean_abs_pre, mean_abs_post]

    fig = plt.figure(figsize=(3.8, 2.6) if style == "nature" else (7.6, 4.2))
    ax = plt.gca()
    bars = ax.bar(labels, values)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("mean $|\\cos|$")
    if style != "nature":
        ax.set_title("Mean absolute cosine alignment (sign-agnostic)")
    _beautify_ax(ax, style)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            min(1.0, value + 0.02),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    _savefig(fig, out_dir / f"{prefix}_bar_mean_abs_cos", formats)
    plt.close(fig)


def _plot_from_json_only(data: Dict[str, Any], out_dir: Path, prefix: str, formats: list[str], style: str) -> None:
    per_t = data.get("alignment", {}).get("per_timestep")
    if not per_t:
        return
    import matplotlib.pyplot as plt

    t = np.array([d["timestep"] for d in per_t], dtype=np.int64)
    mean = np.array([d["mean_cosine"] for d in per_t], dtype=np.float64)
    std = np.array([d["std_cosine"] for d in per_t], dtype=np.float64)
    cnt = np.array([d["count"] for d in per_t], dtype=np.int64)

    fig = plt.figure(figsize=(3.7, 2.6) if style == "nature" else (8.0, 4.8))
    ax = plt.gca()
    ax.plot(t, mean, marker="o", markersize=3, label=r"mean $\cos(\Delta\mu_{\mathrm{pos}},\mathbf{F})$")
    ax.fill_between(t, mean - std, mean + std, alpha=0.2, label=r"$\pm$1 std")
    ax.axhline(0.0, color="k", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("diffusion timestep $s$")
    ax.set_ylabel("cosine")
    if style != "nature":
        ax.set_title("Per-timestep cosine from JSON summary")
        ax.legend()
    _beautify_ax(ax, style)
    _savefig(fig, out_dir / f"{prefix}_per_timestep_cos_from_json", formats)
    plt.close(fig)

    fig = plt.figure(figsize=(3.7, 1.9) if style == "nature" else (8.0, 2.2))
    ax = plt.gca()
    ax.bar(t, cnt)
    ax.set_xlabel("diffusion timestep $s$")
    ax.set_ylabel("#pairs")
    if style != "nature":
        ax.set_title("Pairs per timestep (JSON summary)")
    _beautify_ax(ax, style)
    _savefig(fig, out_dir / f"{prefix}_per_timestep_counts_from_json", formats)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    _ensure_out_dir(out_dir)

    formats = [str(f).lower().lstrip(".") for f in (args.formats or ["png"])]
    allowed = {"png", "pdf", "svg"}
    for fmt in formats:
        if fmt not in allowed:
            raise ValueError(f"Unsupported format '{fmt}'. Supported: {sorted(allowed)}")

    import matplotlib

    matplotlib.use("Agg")
    _apply_style(args.style)
    import matplotlib.pyplot as plt  # noqa: F401  (imported for side effects / backend)

    data_json: Optional[Dict[str, Any]] = None
    if args.json:
        json_path = Path(args.json).expanduser().resolve()
        data_json = _load_json(json_path)
        _plot_from_json_only(data_json, out_dir, args.prefix, formats=formats, style=args.style)

    if not args.npz:
        if data_json is None:
            raise ValueError("Provide at least one of --json or --npz.")
        print(f"Wrote plots to: {out_dir}")
        return

    npz_path = Path(args.npz).expanduser().resolve()
    npz = np.load(npz_path, allow_pickle=True)
    required = [
        "timestep",
        "force_norm",
        "delta_norm",
        "cos_delta",
        "proj_delta",
        "cos_step_pre",
        "cos_step_post",
        "proj_step_pre",
        "proj_step_post",
    ]
    for key in required:
        if key not in npz:
            raise KeyError(f"Missing '{key}' in {npz_path}")
    arr = {k: npz[k] for k in required}

    _plot_histograms(arr, out_dir, args.prefix, formats=formats, style=args.style)
    _plot_per_timestep(arr, out_dir, args.prefix, formats=formats, style=args.style)
    _plot_abs_cos_bar(arr, out_dir, args.prefix, formats=formats, style=args.style)
    _plot_scatter(arr, out_dir, args.prefix, int(args.max_scatter), formats=formats, style=args.style, seed=0)

    print(f"Wrote plots to: {out_dir}")


if __name__ == "__main__":
    main()
