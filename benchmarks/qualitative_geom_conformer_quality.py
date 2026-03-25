#!/usr/bin/env python
"""
Conformer quality panel (GEOM): baseline vs post-trained.

Outputs a publication-style 2xK grid with:
- 3D (projected) ball-and-stick renderings + faint 2D overlay
- Stability diagnostics (atom/molecule stability)
- MLFF (UMA) energy + force RMS computed on the largest connected component

All text uses Arial and PDFs embed Type42 fonts for Nature/Science editing.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
EDM_SOURCE_ROOT = REPO_ROOT / "edm_source"
for p in (REPO_ROOT, EDM_SOURCE_ROOT):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

from benchmarks.qualitative_geom_compare import (  # noqa: E402
    _BASELINE_COLOR,
    _INVALID_COLOR,
    _POSTTRAINED_COLOR,
    _draw_mol_projected,
    _extract_atoms,
    _filter_h,
    _graph_stats,
    _infer_bonds,
    _largest_component_indices,
    _make_masks,
    _project_3d_to_2d,
    _seed_all,
    _select_device,
    _selection_score,
    _set_plot_style,
    _smiles_from_atoms,
    _stable_pca_rotation,
)
from edm_source.configs.datasets_config import get_dataset_info  # noqa: E402
from edm_source.mlff_modules.mlff_force_computer import MLFFForceComputer  # noqa: E402
from edm_source.mlff_modules.mlff_utils import get_mlff_predictor  # noqa: E402
from edm_source.qm9.analyze import check_stability  # noqa: E402
from edm_source.qm9.dataset import retrieve_dataloaders  # noqa: E402
from edm_source.qm9.models import DistributionNodes, get_model  # noqa: E402
from eval_verl_rollout import load_model_weights  # noqa: E402
from verl_diffusion.model.edm_model import EDMModel  # noqa: E402


@dataclass
class Pair:
    index: int
    n_atoms: int
    base_smiles: str
    post_smiles: str
    base_coords: np.ndarray
    base_types: np.ndarray
    base_symbols: List[str]
    post_coords: np.ndarray
    post_types: np.ndarray
    post_symbols: List[str]


def _largest_component(
    coords: np.ndarray,
    atom_types: np.ndarray,
    symbols: List[str],
    bonds: List[Tuple[int, int, int]],
) -> Tuple[np.ndarray, np.ndarray, List[str], List[Tuple[int, int, int]]]:
    keep = _largest_component_indices(int(coords.shape[0]), bonds)
    if len(keep) == int(coords.shape[0]):
        return coords, atom_types, symbols, bonds
    keep_set = set(keep)
    remap = {old: new for new, old in enumerate(keep)}
    coords2 = coords[keep]
    types2 = atom_types[keep]
    symbols2 = [symbols[i] for i in keep]
    bonds2: List[Tuple[int, int, int]] = []
    for i, j, order in bonds:
        if i in keep_set and j in keep_set:
            bonds2.append((remap[i], remap[j], int(order)))
    return coords2, types2, symbols2, bonds2


def _compute_stability(coords: np.ndarray, atom_types: np.ndarray, dataset_info: Dict[str, Any]) -> Tuple[float, int]:
    if coords.shape[0] == 0:
        return 0.0, 0
    mol_stable, nr_stable, total = check_stability(coords, atom_types, dataset_info, debug=False)
    atom_frac = float(nr_stable) / float(max(1, total))
    return atom_frac, int(bool(mol_stable))


def _compute_mlff_metrics(
    coords: np.ndarray,
    atom_types: np.ndarray,
    *,
    dataset_info: Dict[str, Any],
    force_computer: MLFFForceComputer,
    device: torch.device,
) -> Tuple[Optional[float], Optional[float]]:
    """Return (energy_per_atom, force_rms). None if MLFF is unavailable."""
    if coords.shape[0] == 0 or force_computer is None:
        return None, None
    n_cat = len(dataset_info["atom_decoder"])
    pos = torch.tensor(coords, dtype=torch.float32, device=device).unsqueeze(0)
    onehot = torch.nn.functional.one_hot(torch.tensor(atom_types, device=device), num_classes=n_cat).float().unsqueeze(0)
    z = torch.cat([pos, onehot], dim=-1)
    node_mask = torch.ones((1, coords.shape[0], 1), dtype=torch.float32, device=device)
    out = force_computer.compute_mlff_forces(z, node_mask, dataset_info)
    if isinstance(out, tuple):
        forces, energies = out
        energy = float(energies[0].detach().cpu().item())
    else:
        forces = out
        energy = None
    forces = forces[0]
    norms = torch.norm(forces, dim=-1)
    force_rms = float(torch.sqrt(torch.mean(norms.pow(2))).detach().cpu().item()) if norms.numel() > 0 else 0.0
    if energy is None:
        return None, force_rms
    energy_per_atom = energy / float(max(1, coords.shape[0]))
    return float(energy_per_atom), float(force_rms)


def _annotate_metrics(
    ax,
    *,
    atom_stab: float,
    mol_stab: int,
    energy_per_atom: Optional[float],
    force_rms: Optional[float],
) -> None:
    lines = [f"AtomStab {atom_stab:.2f}", f"MolStab {mol_stab:d}"]
    if energy_per_atom is not None:
        lines.append(f"E {energy_per_atom:.3f} eV/atom")
    if force_rms is not None:
        lines.append(f"|F|rms {force_rms:.2f}")
    txt = "\n".join(lines)
    ax.text(
        0.03,
        0.97,
        txt,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        color="#2B2B2B",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.78),
        zorder=80,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Conformer quality panel (GEOM): baseline vs post-trained.")
    parser.add_argument("--args-pickle", type=str, default="pretrained/edm/edm_geom_drugs/args.pickle")
    parser.add_argument("--base-weights", type=str, default="pretrained/edm/edm_geom_drugs/generative_model_ema.npy")
    parser.add_argument(
        "--posttrained-weights",
        type=str,
        default="outputs/verl_geom_smoke_v5_20260120_104415/generative_model_ema.npy",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--mlff-model", type=str, default="uma-s-1p1")
    parser.add_argument("--mlff-device", type=str, default=None, help="Overrides MLFF device (e.g., 'cpu', 'cuda:0').")
    parser.add_argument("--time-step", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-show", type=int, default=5)
    parser.add_argument("--n-candidates", type=int, default=384)
    parser.add_argument("--nodes-min", type=int, default=20)
    parser.add_argument("--nodes-max", type=int, default=80)
    parser.add_argument("--min-connected-frac", type=float, default=0.65)
    parser.add_argument("--max-components", type=int, default=6)
    parser.add_argument("--heavy-min", type=int, default=15, help="Minimum heavy atoms (non-H) in the drawn LCC.")
    parser.add_argument("--heavy-max", type=int, default=80, help="Maximum heavy atoms (non-H) in the drawn LCC.")
    parser.add_argument(
        "--heavy-delta-max",
        type=int,
        default=6,
        help="Maximum |heavy_baseline - heavy_posttrained| allowed for selected examples.",
    )
    parser.add_argument(
        "--selection",
        choices=["balanced", "stability_improve", "force_improve", "energy_force_improve"],
        default="stability_improve",
        help="How to rank candidates (after size/connectivity filtering).",
    )
    parser.add_argument(
        "--mlff-top-k",
        type=int,
        default=48,
        help="(force_improve) compute MLFF metrics for the top-K candidates before selecting examples.",
    )
    parser.add_argument("--keep-h", action="store_true", help="Keep explicit H in drawings (default: remove H).")
    parser.add_argument("--out-dir", type=str, default="benchmarks/qualitative_geom")
    parser.add_argument("--out-name", type=str, default="geom_conformer_quality_3d_overlay")
    args = parser.parse_args()

    _set_plot_style()
    device = _select_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(Path(args.args_pickle), "rb") as f:
        edm_config = pickle.load(f)
    edm_config.cuda = device.type == "cuda"
    edm_config.device = device
    if hasattr(edm_config, "no_cuda"):
        edm_config.no_cuda = not edm_config.cuda

    dataset_info = get_dataset_info(edm_config.dataset, edm_config.remove_h)

    dataloaders, _ = retrieve_dataloaders(edm_config)
    flow, nodes_dist, _ = get_model(edm_config, device, dataset_info, dataloaders["train"])
    flow.to(device).eval()

    base = EDMModel(flow, edm_config).to(device)
    load_model_weights(base, Path(args.base_weights), device)
    base_flow = base.model
    setattr(base_flow, "T", int(args.time_step))

    flow_post, _, _ = get_model(edm_config, device, dataset_info, dataloaders["train"])
    flow_post.to(device).eval()
    post = EDMModel(flow_post, edm_config).to(device)
    load_model_weights(post, Path(args.posttrained_weights), device)
    post_flow = post.model
    setattr(post_flow, "T", int(args.time_step))

    # Sample node counts.
    nodes_dist = nodes_dist if nodes_dist is not None else DistributionNodes(dataset_info["n_nodes"])
    target = int(args.n_candidates)
    nodes: List[int] = []
    while len(nodes) < target:
        cand = nodes_dist.sample(n_samples=max(64, target * 2)).tolist()
        for n in cand:
            n = int(n)
            if int(args.nodes_min) <= n <= int(args.nodes_max):
                nodes.append(n)
                if len(nodes) >= target:
                    break
    nodes_per_sample = torch.tensor(nodes[:target], dtype=torch.long)
    max_n_nodes = int(nodes_per_sample.max().item())
    node_mask, edge_mask = _make_masks(nodes_per_sample, max_n_nodes, device)

    # Paired sampling with identical RNG.
    _seed_all(int(args.seed), device)
    x_base, h_base = base_flow.sample(
        n_samples=target,
        n_nodes=max_n_nodes,
        node_mask=node_mask,
        edge_mask=edge_mask,
        context=None,
        fix_noise=False,
    )
    _seed_all(int(args.seed), device)
    x_post, h_post = post_flow.sample(
        n_samples=target,
        n_nodes=max_n_nodes,
        node_mask=node_mask,
        edge_mask=edge_mask,
        context=None,
        fix_noise=False,
    )

    # Candidate pool: both valid + size-matched + reasonably connected.
    heavy_min = int(args.heavy_min)
    heavy_max = int(args.heavy_max)
    heavy_delta_max = int(args.heavy_delta_max)

    candidates: List[Dict[str, Any]] = []
    for i in range(target):
        nm = node_mask[i]
        n_atoms = int(nm.squeeze(-1).sum().item())
        coords_b, types_b, syms_b = _extract_atoms(x_base[i], h_base["categorical"][i], nm, dataset_info)
        coords_p, types_p, syms_p = _extract_atoms(x_post[i], h_post["categorical"][i], nm, dataset_info)
        smi_b = _smiles_from_atoms(coords_b, types_b, dataset_info)
        smi_p = _smiles_from_atoms(coords_p, types_p, dataset_info)
        if smi_b is None or smi_p is None:
            continue

        # Largest-component filter for metrics/drawing consistency (with explicit H).
        bonds_b_full = _infer_bonds(coords_b, types_b, dataset_info) if coords_b.shape[0] > 0 else []
        bonds_p_full = _infer_bonds(coords_p, types_p, dataset_info) if coords_p.shape[0] > 0 else []
        coords_b_lcc, types_b_lcc, syms_b_lcc, bonds_b_lcc = _largest_component(coords_b, types_b, syms_b, bonds_b_full)
        coords_p_lcc, types_p_lcc, syms_p_lcc, bonds_p_lcc = _largest_component(coords_p, types_p, syms_p, bonds_p_full)

        # Connectivity sanity: avoid extreme fragmentation (tiny panels).
        _, comp_frac_b, n_comp_b = _graph_stats(int(coords_b.shape[0]), bonds_b_full)
        _, comp_frac_p, n_comp_p = _graph_stats(int(coords_p.shape[0]), bonds_p_full)
        if comp_frac_b < float(args.min_connected_frac) or comp_frac_p < float(args.min_connected_frac):
            continue
        if n_comp_b > int(args.max_components) or n_comp_p > int(args.max_components):
            continue

        # Size matching is on the drawn (heavy-atom) LCC.
        heavy_b = int(sum(1 for s in syms_b_lcc if s != "H"))
        heavy_p = int(sum(1 for s in syms_p_lcc if s != "H"))
        if not (heavy_min <= heavy_b <= heavy_max and heavy_min <= heavy_p <= heavy_max):
            continue
        if abs(heavy_b - heavy_p) > heavy_delta_max:
            continue

        # Stability signal is cheap and correlates with the intended RL objectives.
        b_atom_stab, b_mol_stab = _compute_stability(coords_b_lcc, types_b_lcc, dataset_info)
        p_atom_stab, p_mol_stab = _compute_stability(coords_p_lcc, types_p_lcc, dataset_info)
        stab_delta = float(p_atom_stab - b_atom_stab) + 2.0 * float(p_mol_stab - b_mol_stab)

        # Aesthetics score (prefer connected/ring-rich examples).
        score_b = _selection_score(coords_b_lcc, syms_b_lcc, bonds_b_lcc, smi_b)
        score_p = _selection_score(coords_p_lcc, syms_p_lcc, bonds_p_lcc, smi_p)
        score = float(min(score_b, score_p))  # ensure both sides look non-trivial

        pair = Pair(
            index=i,
            n_atoms=n_atoms,
            base_smiles=smi_b,
            post_smiles=smi_p,
            base_coords=coords_b_lcc,
            base_types=types_b_lcc,
            base_symbols=syms_b_lcc,
            post_coords=coords_p_lcc,
            post_types=types_p_lcc,
            post_symbols=syms_p_lcc,
        )
        candidates.append(
            {
                "pair": pair,
                "heavy_b": heavy_b,
                "heavy_p": heavy_p,
                "score_b": float(score_b),
                "score_p": float(score_p),
                "score": float(score),
                "b_atom_stab": float(b_atom_stab),
                "b_mol_stab": int(b_mol_stab),
                "p_atom_stab": float(p_atom_stab),
                "p_mol_stab": int(p_mol_stab),
                "stab_delta": float(stab_delta),
            }
        )

    if not candidates:
        raise SystemExit("No valid candidates found; try increasing --n-candidates or relaxing filters.")

    # MLFF setup (optional).
    mlff_device = args.mlff_device or (str(device) if device.type == "cuda" else "cpu")
    mlff_predictor = get_mlff_predictor(args.mlff_model, mlff_device)
    force_computer = None
    if mlff_predictor is not None:
        # For GEOM config, norm_values[0] is typically 1 (already Angstrom). Keep it explicit.
        position_scale = float(getattr(base_flow, "norm_values", [1.0])[0])
        force_computer = MLFFForceComputer(
            mlff_predictor=mlff_predictor,
            position_scale=position_scale,
            device=str(mlff_device),
            compute_energy=True,
        )

    # Candidate selection.
    selection = str(args.selection)
    if selection == "balanced":
        candidates.sort(key=lambda d: (d["score"], d["stab_delta"]), reverse=True)
    elif selection == "stability_improve":
        # Prefer cases where post-trained improves stability, and keep examples visually rich.
        candidates.sort(key=lambda d: (d["stab_delta"], d["score"]), reverse=True)
    else:  # MLFF-based selections
        # We'll rank by MLFF force improvement after evaluating a small top-K pool.
        candidates.sort(key=lambda d: (d["stab_delta"], d["score"]), reverse=True)

    chosen_meta: List[Dict[str, Any]] = []
    if selection not in {"force_improve", "energy_force_improve"} or force_computer is None:
        # If MLFF isn't available, fall back to stability-based selection.
        chosen_meta = candidates[: int(args.n_show)]
    else:
        want = int(args.n_show)
        top_k = int(max(want, int(args.mlff_top_k)))

        # Compute MLFF metrics in chunks so we can optionally expand beyond top_k if we
        # don't find enough "energy+force improved" examples.
        evaluated: List[Dict[str, Any]] = []
        max_k = int(min(len(candidates), max(top_k, want * 10)))
        cursor = 0
        while cursor < max_k:
            chunk = candidates[cursor : min(cursor + 32, max_k)]
            for d in chunk:
                if "b_f" in d and "p_f" in d and "b_e" in d and "p_e" in d:
                    evaluated.append(d)
                    continue
                pair: Pair = d["pair"]
                b_e, b_f = _compute_mlff_metrics(
                    pair.base_coords, pair.base_types, dataset_info=dataset_info, force_computer=force_computer, device=device
                )
                p_e, p_f = _compute_mlff_metrics(
                    pair.post_coords, pair.post_types, dataset_info=dataset_info, force_computer=force_computer, device=device
                )
                d["b_e"] = b_e
                d["b_f"] = b_f
                d["p_e"] = p_e
                d["p_f"] = p_f
                d["force_delta"] = float(b_f - p_f) if (b_f is not None and p_f is not None) else -1e9
                d["energy_delta"] = float(b_e - p_e) if (b_e is not None and p_e is not None) else -1e9
                evaluated.append(d)

            cursor += len(chunk)

            if selection == "force_improve":
                # Force-only selection needs no extra constraint; top_k is enough.
                if cursor >= top_k:
                    break
            else:  # energy_force_improve
                improved = [d for d in evaluated if d.get("force_delta", -1e9) > 0 and d.get("energy_delta", -1e9) > 0]
                if cursor >= top_k and len(improved) >= want:
                    break

        if selection == "force_improve":
            evaluated.sort(key=lambda d: (d.get("force_delta", -1e9), d["stab_delta"], d["score"]), reverse=True)
            chosen_meta = evaluated[:want]
        else:
            improved = [d for d in evaluated if d.get("force_delta", -1e9) > 0 and d.get("energy_delta", -1e9) > 0]
            if not improved:
                # Fallback: still show force improvements (energy did not improve).
                evaluated.sort(key=lambda d: (d.get("force_delta", -1e9), d["stab_delta"], d["score"]), reverse=True)
                chosen_meta = evaluated[:want]
            else:
                improved.sort(
                    key=lambda d: (d.get("force_delta", -1e9), d.get("energy_delta", -1e9), d["stab_delta"], d["score"]),
                    reverse=True,
                )
                chosen_meta = improved[:want]

    chosen = [d["pair"] for d in chosen_meta]

    # Render grid: Baseline row over Post-trained row.
    n_cols = len(chosen)
    fig_w = 7.2
    fig_h = 3.6 if n_cols <= 5 else 4.0
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(2, n_cols, left=0.06, right=0.995, top=0.90, bottom=0.08, wspace=0.02, hspace=0.02)
    axes = [[fig.add_subplot(gs[r, c]) for c in range(n_cols)] for r in range(2)]

    fig.text(
        0.01,
        0.66,
        "Baseline",
        rotation=90,
        va="center",
        ha="left",
        fontsize=12,
        fontweight="bold",
        color=_BASELINE_COLOR,
    )
    fig.text(
        0.01,
        0.25,
        "Post-trained",
        rotation=90,
        va="center",
        ha="left",
        fontsize=12,
        fontweight="bold",
        color=_POSTTRAINED_COLOR,
    )

    # Compute metrics only for selected examples (and write them to JSON).
    metrics: List[Dict[str, Any]] = []
    for c, pair in enumerate(chosen):
        # Drawing: optionally drop H for readability.
        b_coords_draw, b_types_draw, b_syms_draw = _filter_h(pair.base_coords, pair.base_types, pair.base_symbols, keep_h=bool(args.keep_h))
        p_coords_draw, p_types_draw, p_syms_draw = _filter_h(pair.post_coords, pair.post_types, pair.post_symbols, keep_h=bool(args.keep_h))
        bonds_b = _infer_bonds(b_coords_draw, b_types_draw, dataset_info) if b_coords_draw.shape[0] > 0 else []
        bonds_p = _infer_bonds(p_coords_draw, p_types_draw, dataset_info) if p_coords_draw.shape[0] > 0 else []

        # Align view per column using baseline PCA.
        ref = b_coords_draw if b_coords_draw.shape[0] > 0 else p_coords_draw
        rot = _stable_pca_rotation(ref) if ref.shape[0] > 0 else np.eye(3)
        b_rot = (b_coords_draw - b_coords_draw.mean(axis=0, keepdims=True)) @ rot if b_coords_draw.shape[0] > 0 else b_coords_draw
        p_rot = (p_coords_draw - p_coords_draw.mean(axis=0, keepdims=True)) @ rot if p_coords_draw.shape[0] > 0 else p_coords_draw
        all_xy = []
        if b_rot.shape[0] > 0:
            xy, _ = _project_3d_to_2d(b_rot, elev=18.0, azim=35.0)
            all_xy.append(xy)
        if p_rot.shape[0] > 0:
            xy, _ = _project_3d_to_2d(p_rot, elev=18.0, azim=35.0)
            all_xy.append(xy)
        shared_bounds = None
        if all_xy:
            stack = np.concatenate(all_xy, axis=0)
            shared_bounds = (stack.min(axis=0), stack.max(axis=0))

        _draw_mol_projected(
            axes[0][c],
            b_rot,
            b_syms_draw,
            bonds_b,
            valid=True,
            border_color=_BASELINE_COLOR,
            overlay_smiles=pair.base_smiles,
            shared_xy_bounds=shared_bounds,
        )
        _draw_mol_projected(
            axes[1][c],
            p_rot,
            p_syms_draw,
            bonds_p,
            valid=True,
            border_color=_POSTTRAINED_COLOR,
            overlay_smiles=pair.post_smiles,
            shared_xy_bounds=shared_bounds,
        )

        # Metrics on LCC with explicit H retained (better chemistry + MLFF semantics).
        b_atom_stab, b_mol_stab = _compute_stability(pair.base_coords, pair.base_types, dataset_info)
        p_atom_stab, p_mol_stab = _compute_stability(pair.post_coords, pair.post_types, dataset_info)
        b_e, b_f = _compute_mlff_metrics(
            pair.base_coords, pair.base_types, dataset_info=dataset_info, force_computer=force_computer, device=device
        )
        p_e, p_f = _compute_mlff_metrics(
            pair.post_coords, pair.post_types, dataset_info=dataset_info, force_computer=force_computer, device=device
        )

        _annotate_metrics(axes[0][c], atom_stab=b_atom_stab, mol_stab=b_mol_stab, energy_per_atom=b_e, force_rms=b_f)
        _annotate_metrics(axes[1][c], atom_stab=p_atom_stab, mol_stab=p_mol_stab, energy_per_atom=p_e, force_rms=p_f)

        axes[0][c].set_title(f"N={pair.n_atoms}  (valid -> valid)", fontsize=9, pad=6)
        metrics.append(
            {
                "index": int(pair.index),
                "n_atoms": int(pair.n_atoms),
                "heavy_baseline": int(sum(1 for s in pair.base_symbols if s != "H")),
                "heavy_posttrained": int(sum(1 for s in pair.post_symbols if s != "H")),
                "baseline": {
                    "smiles": pair.base_smiles,
                    "atom_stability": b_atom_stab,
                    "mol_stability": b_mol_stab,
                    "energy_per_atom": b_e,
                    "force_rms": b_f,
                },
                "posttrained": {
                    "smiles": pair.post_smiles,
                    "atom_stability": p_atom_stab,
                    "mol_stability": p_mol_stab,
                    "energy_per_atom": p_e,
                    "force_rms": p_f,
                },
            }
        )

    fig.suptitle("Conformer quality (stability + MLFF energy/forces)", fontsize=11, fontweight="bold", y=0.985)
    out_base = out_dir / args.out_name
    fig.savefig(out_base.with_suffix(".png"), dpi=300)
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)

    meta = {
        "args": vars(args),
        "chosen_indices": [int(p.index) for p in chosen],
        "metrics": metrics,
        "notes": [
            "Stability computed by edm_source.qm9.analyze.check_stability on the largest connected component.",
            "MLFF metrics computed with UMA via MLFFForceComputer(compute_energy=True); force aggregation is RMS over atoms.",
            "3D panels use projected ball-and-stick with a faint 2D RDKit overlay for readability.",
        ],
    }
    out_base.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
