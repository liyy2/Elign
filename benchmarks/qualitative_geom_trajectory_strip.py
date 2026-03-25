#!/usr/bin/env python
"""
Matched-noise diffusion trajectory strip (GEOM).

This script samples a *single* molecule from the baseline and post-trained
diffusion checkpoints using the same RNG seed so both start from identical
stochastic noise. It then captures intermediate diffusion states via
`sample_chain` and renders a 2xK strip (Baseline vs Post-trained) that shows
how each model denoises over time.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    _graph_stats,
    _selection_score,
    _largest_component_indices,
    _infer_bonds,
    _project_3d_to_2d,
    _seed_all,
    _select_device,
    _set_plot_style,
    _smiles_from_atoms,
    _stable_pca_rotation,
)
from edm_source.configs.datasets_config import get_dataset_info  # noqa: E402
from edm_source.qm9.dataset import retrieve_dataloaders  # noqa: E402
from edm_source.qm9.models import DistributionNodes, get_model  # noqa: E402
from eval_verl_rollout import load_model_weights  # noqa: E402
from verl_diffusion.model.edm_model import EDMModel  # noqa: E402


def _parse_frame_steps(steps: Optional[str], *, T: int) -> Optional[List[int]]:
    if steps is None:
        return None
    out: List[int] = []
    for part in steps.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            v = int(part)
        except ValueError:
            continue
        out.append(int(np.clip(v, 0, T)))
    return out or None


def _make_single_masks(n_nodes: int, device: torch.device, *, n_samples: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (node_mask, edge_mask) for a fixed n_nodes (no padding).

    Shapes match edm_source.qm9.models._make_masks:
      - node_mask: [B, N, 1]
      - edge_mask: [B*N*N, 1]
    """
    n_samples = int(max(1, n_samples))
    node_mask = torch.ones(n_samples, n_nodes, 1, device=device, dtype=torch.float32)
    edge_mask_2d = node_mask.squeeze(-1).unsqueeze(1) * node_mask.squeeze(-1).unsqueeze(2)  # [B, N, N]
    eye = torch.eye(n_nodes, device=device, dtype=torch.bool).unsqueeze(0)  # [1, N, N]
    edge_mask_2d = edge_mask_2d.masked_fill(eye, 0.0)
    edge_mask = edge_mask_2d.reshape(n_samples * n_nodes * n_nodes, 1)
    return node_mask, edge_mask


def _sample_final(
    flow,
    *,
    n_nodes: int,
    node_mask: torch.Tensor,
    edge_mask: torch.Tensor,
    seed: int,
    device: torch.device,
    dataset_info: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, List[str], Optional[str]]:
    _seed_all(int(seed), device)
    with torch.no_grad():
        x, h = flow.sample(
            n_samples=1,
            n_nodes=n_nodes,
            node_mask=node_mask,
            edge_mask=edge_mask,
            context=None,
            fix_noise=False,
        )
    coords, types, syms = _extract_atoms(x[0], h["categorical"][0], node_mask[0], dataset_info)
    smiles = _smiles_from_atoms(coords, types, dataset_info)
    return coords, types, syms, smiles


def _sample_chain(
    flow,
    *,
    n_samples: int,
    n_nodes: int,
    node_mask: torch.Tensor,
    edge_mask: torch.Tensor,
    seed: int,
    device: torch.device,
    keep_frames: Optional[int],
) -> torch.Tensor:
    _seed_all(int(seed), device)
    with torch.no_grad():
        chain_flat = flow.sample_chain(
            n_samples=int(n_samples),
            n_nodes=n_nodes,
            node_mask=node_mask,
            edge_mask=edge_mask,
            context=None,
            keep_frames=keep_frames,
        )
    # chain_flat: [B*keep_frames, N, D]
    b = int(max(1, n_samples))
    n_chain = int(chain_flat.shape[0] // b)
    chain = chain_flat.view(n_chain, b, n_nodes, chain_flat.shape[-1])
    return chain[:, 0] if b == 1 else chain


def _frame_indices(
    n_chain: int,
    *,
    n_frames: int,
    frame_steps: Optional[Sequence[int]],
    T: int,
) -> Tuple[List[int], List[int]]:
    """
    Return (chain_indices, labeled_steps) in left->right order (noisy -> final).

    For DDPM chains with keep_frames == T, chain index k corresponds roughly to step s=k
    (with index 0 overwritten by the final x/h state). For other keep_frames values we
    map steps linearly to indices.
    """
    if frame_steps:
        idxs: List[int] = []
        labels: List[int] = []
        for s in frame_steps:
            s = int(np.clip(s, 0, T))
            # Map step s to chain index in [0, n_chain-1].
            if T <= 0:
                k = 0
            else:
                k = int(round((float(s) / float(T)) * float(max(1, n_chain - 1))))
            k = int(np.clip(k, 0, n_chain - 1))
            idxs.append(k)
            labels.append(s)
        # Ensure unique, keep order.
        uniq: List[int] = []
        uniq_labels: List[int] = []
        seen = set()
        for k, s in zip(idxs, labels):
            if k in seen:
                continue
            seen.add(k)
            uniq.append(k)
            uniq_labels.append(s)
        # Left->right: noisy -> final.
        order = np.argsort(uniq)[::-1].tolist()
        return [uniq[i] for i in order], [uniq_labels[i] for i in order]

    # Default: evenly spaced indices (noisy -> final).
    n_frames = int(max(2, n_frames))
    idxs = np.linspace(n_chain - 1, 0, n_frames)
    idxs = [int(np.clip(int(round(v)), 0, n_chain - 1)) for v in idxs]
    # De-dupe while preserving order.
    uniq = []
    for k in idxs:
        if not uniq or uniq[-1] != k:
            uniq.append(k)
    # Approximate step labels.
    labels = [int(round((float(k) / float(max(1, n_chain - 1))) * float(T))) for k in uniq]
    return uniq, labels


def _draw_lcc_size(
    coords: np.ndarray,
    atom_types: np.ndarray,
    symbols: List[str],
    *,
    dataset_info: Dict[str, Any],
    keep_h: bool,
) -> int:
    """Return the number of atoms that would be drawn (after H-filter + LCC)."""
    if coords.shape[0] == 0:
        return 0
    if keep_h:
        keep = list(range(int(coords.shape[0])))
    else:
        keep = [i for i, s in enumerate(symbols) if s != "H"]
    if not keep:
        return 0
    coords2 = coords[keep]
    types2 = atom_types[keep]
    bonds = _infer_bonds(coords2, types2, dataset_info) if coords2.shape[0] > 1 else []
    lcc = _largest_component_indices(int(coords2.shape[0]), bonds)
    return int(len(lcc))


def _lcc_view(
    coords: np.ndarray,
    atom_types: np.ndarray,
    symbols: List[str],
    *,
    dataset_info: Dict[str, Any],
    keep_h: bool,
    smiles: Optional[str],
) -> Tuple[np.ndarray, np.ndarray, List[str], List[Tuple[int, int, int]], float, int, float]:
    """Prepare a draw-ready (H-filtered, LCC) view + simple graph stats.

    Returns (coords_lcc, atom_types_lcc, symbols_lcc, bonds_lcc, comp_frac_vis, n_comp_vis, score_lcc).
    comp_frac_vis / n_comp_vis are computed on the full visible graph (before taking LCC),
    while score_lcc is a visual-quality heuristic on the LCC.
    """
    if coords.shape[0] == 0:
        return coords[:0], atom_types[:0], [], [], 0.0, 0, -1e9

    if keep_h:
        visible_idx = list(range(int(coords.shape[0])))
    else:
        visible_idx = [i for i, s in enumerate(symbols) if s != "H"]
    if not visible_idx:
        return coords[:0], atom_types[:0], [], [], 0.0, 0, -1e9

    coords_vis = coords[visible_idx]
    types_vis = atom_types[visible_idx]
    symbols_vis = [symbols[i] for i in visible_idx]
    bonds_vis = _infer_bonds(coords_vis, types_vis, dataset_info) if coords_vis.shape[0] > 1 else []
    _, comp_frac_vis, n_comp_vis = _graph_stats(int(coords_vis.shape[0]), bonds_vis)

    lcc_idx = _largest_component_indices(int(coords_vis.shape[0]), bonds_vis)
    if len(lcc_idx) == int(coords_vis.shape[0]):
        coords_lcc = coords_vis
        types_lcc = types_vis
        symbols_lcc = symbols_vis
        bonds_lcc = bonds_vis
    else:
        keep_set = set(lcc_idx)
        remap = {old: new for new, old in enumerate(lcc_idx)}
        coords_lcc = coords_vis[lcc_idx]
        types_lcc = types_vis[lcc_idx]
        symbols_lcc = [symbols_vis[i] for i in lcc_idx]
        bonds_lcc: List[Tuple[int, int, int]] = []
        for i, j, order in bonds_vis:
            if i in keep_set and j in keep_set:
                bonds_lcc.append((remap[i], remap[j], int(order)))

    # Score on LCC only (what we actually show).
    score_lcc = _selection_score(coords_lcc, symbols_lcc, bonds_lcc, smiles=smiles)
    return coords_lcc, types_lcc, symbols_lcc, bonds_lcc, float(comp_frac_vis), int(n_comp_vis), float(score_lcc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Matched-noise diffusion trajectory strip (GEOM).")
    parser.add_argument("--args-pickle", type=str, default="pretrained/edm/edm_geom_drugs/args.pickle")
    parser.add_argument("--base-weights", type=str, default="pretrained/edm/edm_geom_drugs/generative_model_ema.npy")
    parser.add_argument(
        "--posttrained-weights",
        type=str,
        default="outputs/verl_geom_smoke_v5_20260120_104415/generative_model_ema.npy",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--time-step", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-trials", type=int, default=40, help="Try seed, seed+1, ... until selection is satisfied.")
    parser.add_argument(
        "--pick-best",
        action="store_true",
        help="Search all trials and pick the best-looking/improving example (slower than stopping at first match).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="If >1, sample a batch of matched-noise trajectories with the same base seed and pick the best example.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=-1,
        help="(batch mode) Force a specific sample index [0..batch_size-1] instead of auto-picking.",
    )
    parser.add_argument("--n-nodes", type=int, default=0, help="Fix the number of nodes. If 0, sample from dataset distribution.")
    parser.add_argument("--nodes-min", type=int, default=40)
    parser.add_argument("--nodes-max", type=int, default=60)
    parser.add_argument("--selection", choices=["both_valid", "improvement", "any"], default="both_valid")
    parser.add_argument(
        "--baseline-comp-frac-max",
        type=float,
        default=1.0,
        help="Optional filter on baseline largest-component fraction (visible graph).",
    )
    parser.add_argument(
        "--post-comp-frac-min",
        type=float,
        default=0.0,
        help="Optional filter on post-trained largest-component fraction (visible graph).",
    )
    parser.add_argument(
        "--baseline-min-components",
        type=int,
        default=1,
        help="Optional filter: require at least this many connected components in baseline (visible graph).",
    )
    parser.add_argument(
        "--post-max-components",
        type=int,
        default=999,
        help="Optional filter: require at most this many connected components in post-trained (visible graph).",
    )
    parser.add_argument("--draw-min", type=int, default=15, help="Minimum drawn atoms in the LCC (after H filter).")
    parser.add_argument("--draw-max", type=int, default=80, help="Maximum drawn atoms in the LCC (after H filter).")
    parser.add_argument(
        "--draw-delta-max",
        type=int,
        default=6,
        help="Maximum |drawn_baseline - drawn_posttrained| allowed when searching seeds.",
    )
    parser.add_argument("--keep-frames", type=int, default=0, help="Number of stored chain frames. 0 = default (usually T).")
    parser.add_argument("--n-frames", type=int, default=6, help="Number of panels across the strip.")
    parser.add_argument(
        "--frame-steps",
        type=str,
        default=None,
        help="Comma-separated diffusion steps to show (e.g., '1000,800,600,400,200,0').",
    )
    parser.add_argument("--keep-h", action="store_true", help="Keep explicit H in drawings (default: remove H).")
    parser.add_argument("--out-dir", type=str, default="benchmarks/qualitative_geom")
    parser.add_argument("--out-name", type=str, default="geom_trajectory_strip_3d_overlay")
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
    flow_base, nodes_dist, _ = get_model(edm_config, device, dataset_info, dataloaders["train"])
    flow_base.to(device).eval()
    base = EDMModel(flow_base, edm_config).to(device)
    load_model_weights(base, Path(args.base_weights), device)
    base_flow = base.model
    setattr(base_flow, "T", int(args.time_step))

    flow_post, _, _ = get_model(edm_config, device, dataset_info, dataloaders["train"])
    flow_post.to(device).eval()
    post = EDMModel(flow_post, edm_config).to(device)
    load_model_weights(post, Path(args.posttrained_weights), device)
    post_flow = post.model
    setattr(post_flow, "T", int(args.time_step))

    # Choose N.
    T = int(args.time_step)
    n_nodes = int(args.n_nodes)
    if n_nodes <= 0:
        nodes_dist = nodes_dist if nodes_dist is not None else DistributionNodes(dataset_info["n_nodes"])
        while True:
            cand = int(nodes_dist.sample(n_samples=1).item())
            if int(args.nodes_min) <= cand <= int(args.nodes_max):
                n_nodes = cand
                break
    n_nodes = int(n_nodes)
    batch_size = int(max(1, args.batch_size))
    node_mask, edge_mask = _make_single_masks(n_nodes, device, n_samples=batch_size)

    keep_frames: Optional[int] = None if int(args.keep_frames) <= 0 else int(args.keep_frames)
    frame_steps = _parse_frame_steps(args.frame_steps, T=T)

    keep_h = bool(args.keep_h)
    draw_min = int(args.draw_min)
    draw_max = int(args.draw_max)
    draw_delta_max = int(args.draw_delta_max)
    # Pick an example (either via seed scanning or batch selection).
    chosen_seed = int(args.seed)
    chosen_index: Optional[int] = None
    picked_meta: Optional[Dict[str, Any]] = None

    def _rank_score(
        *,
        selection: str,
        b_comp_frac: float,
        b_n_comp: int,
        b_score: float,
        p_comp_frac: float,
        p_n_comp: int,
        p_score: float,
    ) -> float:
        if selection == "improvement":
            return float(p_score - b_score) + 25.0 * float(p_comp_frac - b_comp_frac) + 2.5 * float(b_n_comp - p_n_comp)
        if selection == "both_valid":
            return float(p_score - b_score) + 10.0 * float(p_comp_frac - b_comp_frac)
        return float(p_score)

    if batch_size > 1:
        # Sample matched-noise chains for a batch and pick the most convincing example.
        chain_b_full = _sample_chain(
            base_flow,
            n_samples=batch_size,
            n_nodes=n_nodes,
            node_mask=node_mask,
            edge_mask=edge_mask,
            seed=chosen_seed,
            device=device,
            keep_frames=keep_frames,
        )
        chain_p_full = _sample_chain(
            post_flow,
            n_samples=batch_size,
            n_nodes=n_nodes,
            node_mask=node_mask,
            edge_mask=edge_mask,
            seed=chosen_seed,
            device=device,
            keep_frames=keep_frames,
        )

        # Decode final x/h for each sample index; choose the best one.
        n_cat = len(dataset_info["atom_decoder"])
        forced = int(args.sample_index)
        if forced >= 0:
            idx = int(np.clip(forced, 0, batch_size - 1))
        else:
            cand_list: List[Dict[str, Any]] = []
            for i in range(batch_size):
                x_b0 = chain_b_full[0, i, :, :3]
                h_b0 = chain_b_full[0, i, :, 3 : 3 + n_cat]
                x_p0 = chain_p_full[0, i, :, :3]
                h_p0 = chain_p_full[0, i, :, 3 : 3 + n_cat]

                coords_b, types_b, syms_b = _extract_atoms(
                    x_b0.detach().cpu(), h_b0.detach().cpu(), node_mask[i].cpu(), dataset_info
                )
                coords_p, types_p, syms_p = _extract_atoms(
                    x_p0.detach().cpu(), h_p0.detach().cpu(), node_mask[i].cpu(), dataset_info
                )
                smi_b = _smiles_from_atoms(coords_b, types_b, dataset_info)
                smi_p = _smiles_from_atoms(coords_p, types_p, dataset_info)
                valid_b = smi_b is not None
                valid_p = smi_p is not None

                draw_b = _draw_lcc_size(coords_b, types_b, syms_b, dataset_info=dataset_info, keep_h=keep_h)
                draw_p = _draw_lcc_size(coords_p, types_p, syms_p, dataset_info=dataset_info, keep_h=keep_h)
                if not (draw_min <= draw_b <= draw_max and draw_min <= draw_p <= draw_max):
                    continue
                if abs(draw_b - draw_p) > draw_delta_max:
                    continue

                ok = False
                if args.selection == "both_valid":
                    ok = valid_b and valid_p
                elif args.selection == "improvement":
                    ok = (not valid_b) and valid_p
                else:
                    ok = valid_b or valid_p
                if not ok:
                    continue

                _, _, _, _, b_comp_frac, b_n_comp, b_score = _lcc_view(
                    coords_b, types_b, syms_b, dataset_info=dataset_info, keep_h=keep_h, smiles=smi_b
                )
                _, _, _, _, p_comp_frac, p_n_comp, p_score = _lcc_view(
                    coords_p, types_p, syms_p, dataset_info=dataset_info, keep_h=keep_h, smiles=smi_p
                )

                if b_comp_frac > float(args.baseline_comp_frac_max):
                    continue
                if p_comp_frac < float(args.post_comp_frac_min):
                    continue
                if int(b_n_comp) < int(args.baseline_min_components):
                    continue
                if int(p_n_comp) > int(args.post_max_components):
                    continue

                score = _rank_score(
                    selection=str(args.selection),
                    b_comp_frac=float(b_comp_frac),
                    b_n_comp=int(b_n_comp),
                    b_score=float(b_score),
                    p_comp_frac=float(p_comp_frac),
                    p_n_comp=int(p_n_comp),
                    p_score=float(p_score),
                )
                cand_list.append(
                    {
                        "index": int(i),
                        "baseline": (coords_b, types_b, syms_b, smi_b, valid_b, draw_b),
                        "posttrained": (coords_p, types_p, syms_p, smi_p, valid_p, draw_p),
                        "score": float(score),
                        "b_comp_frac": float(b_comp_frac),
                        "b_n_comp": int(b_n_comp),
                        "p_comp_frac": float(p_comp_frac),
                        "p_n_comp": int(p_n_comp),
                    }
                )

            if not cand_list:
                idx = 0
            else:
                cand_list.sort(key=lambda d: d.get("score", -1e9), reverse=True)
                best = cand_list[0]
                idx = int(best["index"])
                picked_meta = best

        idx = int(np.clip(idx, 0, batch_size - 1))
        chosen_index = int(idx)
        chain_b = chain_b_full[:, idx, :, :]
        chain_p = chain_p_full[:, idx, :, :]

        # Final stats for meta.
        x_b0 = chain_b[0, :, :3]
        h_b0 = chain_b[0, :, 3 : 3 + n_cat]
        x_p0 = chain_p[0, :, :3]
        h_p0 = chain_p[0, :, 3 : 3 + n_cat]
        coords_b, types_b, syms_b = _extract_atoms(x_b0.detach().cpu(), h_b0.detach().cpu(), node_mask[0].cpu(), dataset_info)
        coords_p, types_p, syms_p = _extract_atoms(x_p0.detach().cpu(), h_p0.detach().cpu(), node_mask[0].cpu(), dataset_info)
        smi_b = _smiles_from_atoms(coords_b, types_b, dataset_info)
        smi_p = _smiles_from_atoms(coords_p, types_p, dataset_info)
        base_final = (
            coords_b,
            types_b,
            syms_b,
            smi_b,
            smi_b is not None,
            _draw_lcc_size(coords_b, types_b, syms_b, dataset_info=dataset_info, keep_h=keep_h),
        )
        post_final = (
            coords_p,
            types_p,
            syms_p,
            smi_p,
            smi_p is not None,
            _draw_lcc_size(coords_p, types_p, syms_p, dataset_info=dataset_info, keep_h=keep_h),
        )
        if picked_meta is None:
            _, _, _, _, b_comp_frac, b_n_comp, b_score = _lcc_view(
                coords_b, types_b, syms_b, dataset_info=dataset_info, keep_h=keep_h, smiles=smi_b
            )
            _, _, _, _, p_comp_frac, p_n_comp, p_score = _lcc_view(
                coords_p, types_p, syms_p, dataset_info=dataset_info, keep_h=keep_h, smiles=smi_p
            )
            picked_meta = {
                "index": int(idx),
                "score": None,
                "b_comp_frac": float(b_comp_frac),
                "b_n_comp": int(b_n_comp),
                "p_comp_frac": float(p_comp_frac),
                "p_n_comp": int(p_n_comp),
            }
    else:
        # Seed scan mode (batch_size == 1): try seed, seed+1, ... until selection is satisfied.
        base_final = None
        post_final = None
        candidates: List[Dict[str, Any]] = []
        for trial in range(int(max(1, args.max_trials))):
            s = int(args.seed) + int(trial)
            coords_b, types_b, syms_b, smi_b = _sample_final(
                base_flow,
                n_nodes=n_nodes,
                node_mask=node_mask,
                edge_mask=edge_mask,
                seed=s,
                device=device,
                dataset_info=dataset_info,
            )
            coords_p, types_p, syms_p, smi_p = _sample_final(
                post_flow,
                n_nodes=n_nodes,
                node_mask=node_mask,
                edge_mask=edge_mask,
                seed=s,
                device=device,
                dataset_info=dataset_info,
            )
            valid_b = smi_b is not None
            valid_p = smi_p is not None

            draw_b = _draw_lcc_size(coords_b, types_b, syms_b, dataset_info=dataset_info, keep_h=keep_h)
            draw_p = _draw_lcc_size(coords_p, types_p, syms_p, dataset_info=dataset_info, keep_h=keep_h)
            if not (draw_min <= draw_b <= draw_max and draw_min <= draw_p <= draw_max):
                continue
            if abs(draw_b - draw_p) > draw_delta_max:
                continue

            ok = False
            if args.selection == "both_valid":
                ok = valid_b and valid_p
            elif args.selection == "improvement":
                ok = (not valid_b) and valid_p
            else:
                ok = valid_b or valid_p
            if not ok:
                continue

            _, _, _, _, b_comp_frac, b_n_comp, b_score = _lcc_view(
                coords_b, types_b, syms_b, dataset_info=dataset_info, keep_h=keep_h, smiles=smi_b
            )
            _, _, _, _, p_comp_frac, p_n_comp, p_score = _lcc_view(
                coords_p, types_p, syms_p, dataset_info=dataset_info, keep_h=keep_h, smiles=smi_p
            )
            if b_comp_frac > float(args.baseline_comp_frac_max):
                continue
            if p_comp_frac < float(args.post_comp_frac_min):
                continue
            if int(b_n_comp) < int(args.baseline_min_components):
                continue
            if int(p_n_comp) > int(args.post_max_components):
                continue

            score = _rank_score(
                selection=str(args.selection),
                b_comp_frac=float(b_comp_frac),
                b_n_comp=int(b_n_comp),
                b_score=float(b_score),
                p_comp_frac=float(p_comp_frac),
                p_n_comp=int(p_n_comp),
                p_score=float(p_score),
            )
            cand = {
                "seed": int(s),
                "baseline": (coords_b, types_b, syms_b, smi_b, valid_b, draw_b),
                "posttrained": (coords_p, types_p, syms_p, smi_p, valid_p, draw_p),
                "score": float(score),
                "b_comp_frac": float(b_comp_frac),
                "b_n_comp": int(b_n_comp),
                "p_comp_frac": float(p_comp_frac),
                "p_n_comp": int(p_n_comp),
            }
            candidates.append(cand)
            if not args.pick_best:
                chosen_seed = int(s)
                base_final = cand["baseline"]
                post_final = cand["posttrained"]
                picked_meta = cand
                break

        if args.pick_best and candidates:
            candidates.sort(key=lambda d: d.get("score", -1e9), reverse=True)
            best = candidates[0]
            chosen_seed = int(best["seed"])
            base_final = best["baseline"]
            post_final = best["posttrained"]
            picked_meta = best

        if base_final is None or post_final is None:
            # Fall back to the base seed.
            coords_b, types_b, syms_b, smi_b = _sample_final(
                base_flow,
                n_nodes=n_nodes,
                node_mask=node_mask,
                edge_mask=edge_mask,
                seed=chosen_seed,
                device=device,
                dataset_info=dataset_info,
            )
            coords_p, types_p, syms_p, smi_p = _sample_final(
                post_flow,
                n_nodes=n_nodes,
                node_mask=node_mask,
                edge_mask=edge_mask,
                seed=chosen_seed,
                device=device,
                dataset_info=dataset_info,
            )
            base_final = (
                coords_b,
                types_b,
                syms_b,
                smi_b,
                smi_b is not None,
                _draw_lcc_size(coords_b, types_b, syms_b, dataset_info=dataset_info, keep_h=keep_h),
            )
            post_final = (
                coords_p,
                types_p,
                syms_p,
                smi_p,
                smi_p is not None,
                _draw_lcc_size(coords_p, types_p, syms_p, dataset_info=dataset_info, keep_h=keep_h),
            )
            _, _, _, _, b_comp_frac, b_n_comp, b_score = _lcc_view(
                coords_b, types_b, syms_b, dataset_info=dataset_info, keep_h=keep_h, smiles=smi_b
            )
            _, _, _, _, p_comp_frac, p_n_comp, p_score = _lcc_view(
                coords_p, types_p, syms_p, dataset_info=dataset_info, keep_h=keep_h, smiles=smi_p
            )
            picked_meta = {
                "seed": int(chosen_seed),
                "score": None,
                "b_comp_frac": float(b_comp_frac),
                "b_n_comp": int(b_n_comp),
                "p_comp_frac": float(p_comp_frac),
                "p_n_comp": int(p_n_comp),
            }

        chain_b = _sample_chain(
            base_flow,
            n_samples=1,
            n_nodes=n_nodes,
            node_mask=node_mask,
            edge_mask=edge_mask,
            seed=chosen_seed,
            device=device,
            keep_frames=keep_frames,
        )
        chain_p = _sample_chain(
            post_flow,
            n_samples=1,
            n_nodes=n_nodes,
            node_mask=node_mask,
            edge_mask=edge_mask,
            seed=chosen_seed,
            device=device,
            keep_frames=keep_frames,
        )

    # Final decoded atom types for consistent coloring across frames.
    n_cat = len(dataset_info["atom_decoder"])
    x_b0 = chain_b[0, :, :3]
    h_b0 = chain_b[0, :, 3 : 3 + n_cat]
    x_p0 = chain_p[0, :, :3]
    h_p0 = chain_p[0, :, 3 : 3 + n_cat]
    coords_b0, types_b0, syms_b0 = _extract_atoms(
        x_b0.detach().cpu(), h_b0.detach().cpu(), node_mask[0].cpu(), dataset_info
    )
    coords_p0, types_p0, syms_p0 = _extract_atoms(
        x_p0.detach().cpu(), h_p0.detach().cpu(), node_mask[0].cpu(), dataset_info
    )

    def _prepare_draw_spec(
        coords: np.ndarray,
        atom_types: np.ndarray,
        symbols: List[str],
    ) -> Tuple[List[int], List[int], np.ndarray, List[str], List[Tuple[int, int, int]]]:
        """Return (visible_idx, lcc_idx, coords_lcc, symbols_lcc, bonds_lcc)."""
        if bool(args.keep_h):
            visible_idx = list(range(int(coords.shape[0])))
        else:
            visible_idx = [i for i, s in enumerate(symbols) if s != "H"]
        coords_vis = coords[visible_idx] if visible_idx else coords[:0]
        types_vis = atom_types[visible_idx] if visible_idx else atom_types[:0]
        symbols_vis = [symbols[i] for i in visible_idx]
        bonds = _infer_bonds(coords_vis, types_vis, dataset_info) if coords_vis.shape[0] > 0 else []
        lcc_idx = _largest_component_indices(int(coords_vis.shape[0]), bonds)
        if len(lcc_idx) != int(coords_vis.shape[0]):
            keep_set = set(lcc_idx)
            remap = {old: new for new, old in enumerate(lcc_idx)}
            coords_lcc = coords_vis[lcc_idx]
            symbols_lcc = [symbols_vis[i] for i in lcc_idx]
            bonds_lcc: List[Tuple[int, int, int]] = []
            for i, j, order in bonds:
                if i in keep_set and j in keep_set:
                    bonds_lcc.append((remap[i], remap[j], int(order)))
        else:
            coords_lcc = coords_vis
            symbols_lcc = symbols_vis
            bonds_lcc = bonds
        return visible_idx, lcc_idx, coords_lcc, symbols_lcc, bonds_lcc

    vis_b, lcc_b, coords_b0_draw, syms_b0_draw, bonds_b = _prepare_draw_spec(coords_b0, types_b0, syms_b0)
    vis_p, lcc_p, coords_p0_draw, syms_p0_draw, bonds_p = _prepare_draw_spec(coords_p0, types_p0, syms_p0)

    # Consistent view rotation and shared bounds across all panels (paper-friendly).
    ref = coords_b0_draw if coords_b0_draw.shape[0] > 0 else coords_p0_draw
    rot = _stable_pca_rotation(ref) if ref.shape[0] > 0 else np.eye(3)

    # Pick frames to show (indices in chain arrays).
    idxs, step_labels = _frame_indices(chain_b.shape[0], n_frames=int(args.n_frames), frame_steps=frame_steps, T=T)

    # Collect global 2D bounds so each panel has identical scaling.
    all_xy: List[np.ndarray] = []
    for k in idxs:
        xb = chain_b[k, :, :3].detach().cpu().numpy()
        xp = chain_p[k, :, :3].detach().cpu().numpy()

        xb_vis = xb[vis_b] if vis_b else xb[:0]
        xp_vis = xp[vis_p] if vis_p else xp[:0]
        if xb_vis.shape[0] > 0:
            xb_vis = (xb_vis - xb_vis.mean(axis=0, keepdims=True)) @ rot
        if xp_vis.shape[0] > 0:
            xp_vis = (xp_vis - xp_vis.mean(axis=0, keepdims=True)) @ rot
        xb_draw = xb_vis[lcc_b] if xb_vis.shape[0] > 0 else xb_vis
        xp_draw = xp_vis[lcc_p] if xp_vis.shape[0] > 0 else xp_vis
        if xb_draw.shape[0] > 0:
            xy, _ = _project_3d_to_2d(xb_draw, elev=18.0, azim=35.0)
            all_xy.append(xy)
        if xp_draw.shape[0] > 0:
            xy, _ = _project_3d_to_2d(xp_draw, elev=18.0, azim=35.0)
            all_xy.append(xy)
    shared_bounds = None
    if all_xy:
        stack = np.concatenate(all_xy, axis=0)
        shared_bounds = (stack.min(axis=0), stack.max(axis=0))

    # Render.
    n_cols = len(idxs)
    fig_w = 7.2
    fig_h = 2.6
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(2, n_cols, left=0.06, right=0.995, top=0.86, bottom=0.12, wspace=0.02, hspace=0.02)
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

    coords_b_final, _, _, smi_b_final, valid_b_final, draw_b_final = base_final
    coords_p_final, _, _, smi_p_final, valid_p_final, draw_p_final = post_final

    # Keep row colors consistent; highlight invalidity only on the final panel.
    color_b_row = _BASELINE_COLOR
    color_p_row = _POSTTRAINED_COLOR

    for c, (k, s_lbl) in enumerate(zip(idxs, step_labels)):
        xb = chain_b[k, :, :3].detach().cpu().numpy()
        xp = chain_p[k, :, :3].detach().cpu().numpy()
        xb_vis = xb[vis_b] if vis_b else xb[:0]
        xp_vis = xp[vis_p] if vis_p else xp[:0]
        if xb_vis.shape[0] > 0:
            xb_vis = (xb_vis - xb_vis.mean(axis=0, keepdims=True)) @ rot
        if xp_vis.shape[0] > 0:
            xp_vis = (xp_vis - xp_vis.mean(axis=0, keepdims=True)) @ rot
        xb_draw = xb_vis[lcc_b] if xb_vis.shape[0] > 0 else xb_vis
        xp_draw = xp_vis[lcc_p] if xp_vis.shape[0] > 0 else xp_vis

        # Only add 2D overlay on the final panel to keep the strip uncluttered.
        overlay_b = smi_b_final if c == n_cols - 1 else None
        overlay_p = smi_p_final if c == n_cols - 1 else None

        is_final = c == n_cols - 1
        b_border = _INVALID_COLOR if (is_final and (not valid_b_final)) else color_b_row
        p_border = _INVALID_COLOR if (is_final and (not valid_p_final)) else color_p_row
        b_valid_flag = bool(valid_b_final) if is_final else True
        p_valid_flag = bool(valid_p_final) if is_final else True

        # Early trajectory frames are very noisy: suppress "spiderweb" bonds that
        # stretch far beyond plausible chemistry, and relax the cutoff as we
        # approach t=0. This keeps the strip readable while retaining 3D cues.
        T = max(1, int(args.time_step))
        frac = float(np.clip(float(s_lbl) / float(T), 0.0, 1.0))  # 1=noisy, 0=final
        bond_max = 2.4 + 1.2 * (1.0 - frac)

        _draw_mol_projected(
            axes[0][c],
            xb_draw,
            syms_b0_draw,
            bonds_b,
            valid=b_valid_flag,
            border_color=b_border,
            overlay_smiles=overlay_b,
            shared_xy_bounds=shared_bounds,
            bond_max_3d_dist=bond_max,
        )
        _draw_mol_projected(
            axes[1][c],
            xp_draw,
            syms_p0_draw,
            bonds_p,
            valid=p_valid_flag,
            border_color=p_border,
            overlay_smiles=overlay_p,
            shared_xy_bounds=shared_bounds,
            bond_max_3d_dist=bond_max,
        )
        axes[0][c].set_title(f"t={int(s_lbl)}", fontsize=9, pad=6)

    fig.suptitle(
        (
            f"Matched-noise diffusion trajectory (N={n_nodes}, seed={chosen_seed}, idx={chosen_index})"
            if chosen_index is not None
            else f"Matched-noise diffusion trajectory (N={n_nodes}, seed={chosen_seed})"
        ),
        fontsize=11,
        fontweight="bold",
        y=0.985,
    )

    out_base = out_dir / args.out_name
    fig.savefig(out_base.with_suffix(".png"), dpi=300)
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)

    meta = {
        "args": vars(args),
        "picked": {
            "seed": int(chosen_seed),
            "batch_size": int(batch_size),
            "sample_index": int(chosen_index) if chosen_index is not None else None,
            "n_nodes": int(n_nodes),
            "baseline_smiles": base_final[3],
            "posttrained_smiles": post_final[3],
            "drawn_baseline": int(draw_b_final),
            "drawn_posttrained": int(draw_p_final),
            "selection": str(args.selection),
            "pick_best": bool(args.pick_best),
            "baseline_comp_frac": float(picked_meta.get("b_comp_frac", 0.0)) if picked_meta else None,
            "baseline_n_components": int(picked_meta.get("b_n_comp", 0)) if picked_meta else None,
            "post_comp_frac": float(picked_meta.get("p_comp_frac", 0.0)) if picked_meta else None,
            "post_n_components": int(picked_meta.get("p_n_comp", 0)) if picked_meta else None,
            "selection_score": float(picked_meta.get("score")) if (picked_meta and picked_meta.get("score") is not None) else None,
        },
        "notes": [
            "Trajectory frames come from flow.sample_chain (intermediate latents), colored by final decoded atom types.",
            "Panels share identical axis bounds for direct visual comparison across time and between models.",
        ],
    }
    out_base.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
