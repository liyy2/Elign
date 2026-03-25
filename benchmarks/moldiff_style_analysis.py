#!/usr/bin/env python
"""
MolDiff-style post-hoc analysis for GEOM generations.

Reference: "MolDiff: Addressing Atom-Bond Inconsistency in Molecular Diffusion Models"
https://arxiv.org/abs/2305.07508

This script compares a baseline (pretrained) diffusion model against a post-trained
checkpoint using *paired sampling* (same RNG seed + same sampled node counts).

Reported metrics (subset of MolDiff-style evaluation):
  - Validity (RDKit sanitization)
  - Connectivity (single connected component under distance-inferred bonds)
  - Success rate = Valid ∧ Connected
  - Transition counts: invalid→valid, valid→invalid
  - Geometry distributions from inferred bonds:
      * bond lengths
      * bond angles
      * dihedral angles
  - Ring statistics on valid molecules:
      * ring count, ring size distribution, macrocycle fraction

Outputs:
  - JSON summary with per-split counts + JS divergences (baseline vs post-trained).
  - Publication-friendly overlay plots (PDF + PNG).
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from rdkit import Chem
from rdkit.Chem import Crippen, Lipinski, QED

REPO_ROOT = Path(__file__).resolve().parents[1]
EDM_SOURCE_ROOT = REPO_ROOT / "edm_source"
for p in (REPO_ROOT, EDM_SOURCE_ROOT):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

from edm_source.configs.datasets_config import get_dataset_info  # noqa: E402
from edm_source.qm9.models import get_model  # noqa: E402
from edm_source.qm9.rdkit_functions import bond_dict, build_xae_molecule  # noqa: E402
from edm_source.qm9.analyze import check_stability  # noqa: E402
from eval_verl_rollout import load_model_weights  # noqa: E402
from verl_diffusion.model.edm_model import EDMModel  # noqa: E402


def _seed_all(seed: int, device: torch.device) -> None:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))


def _select_device(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _make_masks(nodes_per_sample: torch.Tensor, max_n_nodes: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    nodes_per_sample = nodes_per_sample.to(device=device)
    batch = int(nodes_per_sample.shape[0])
    arange = torch.arange(max_n_nodes, device=device).view(1, -1)
    node_mask = (arange < nodes_per_sample.view(-1, 1)).to(dtype=torch.float32).unsqueeze(-1)

    edge_mask_2d = node_mask.squeeze(-1).unsqueeze(1) * node_mask.squeeze(-1).unsqueeze(2)
    eye = torch.eye(max_n_nodes, device=device, dtype=torch.bool).unsqueeze(0)
    edge_mask_2d = edge_mask_2d.masked_fill(eye, 0.0)
    edge_mask = edge_mask_2d.reshape(batch * max_n_nodes * max_n_nodes, 1)
    return node_mask, edge_mask


def _reweight_nodes_dist(nodes_dist, *, focus_min: Optional[int], focus_max: Optional[int], focus_multiplier: float) -> None:
    if nodes_dist is None:
        return
    if focus_min is None or focus_max is None:
        return
    if focus_multiplier is None:
        return
    try:
        focus_multiplier = float(focus_multiplier)
    except (TypeError, ValueError):
        return
    if focus_multiplier == 1.0:
        return

    if not (hasattr(nodes_dist, "prob") and hasattr(nodes_dist, "n_nodes") and hasattr(nodes_dist, "m")):
        return

    prob = nodes_dist.prob.detach().clone().to(dtype=torch.float64)
    n_nodes = nodes_dist.n_nodes.detach().to(dtype=torch.long)
    mask = (n_nodes >= int(focus_min)) & (n_nodes <= int(focus_max))
    if not mask.any():
        return
    prob[mask] = prob[mask] * focus_multiplier
    prob = prob / prob.sum().clamp(min=1e-12)
    nodes_dist.prob = prob.to(dtype=torch.float32)
    nodes_dist.m = torch.distributions.Categorical(nodes_dist.prob)


def _collect_nodes(
    nodes_dist,
    *,
    n_samples: int,
    nodes_min: int,
    nodes_max: int,
) -> torch.Tensor:
    out: List[int] = []
    target = int(n_samples)
    while len(out) < target:
        cand = nodes_dist.sample(n_samples=max(64, target * 2)).detach().cpu().tolist()
        for n in cand:
            n = int(n)
            if int(nodes_min) <= n <= int(nodes_max):
                out.append(n)
                if len(out) >= target:
                    break
    return torch.tensor(out[:target], dtype=torch.long)


def _infer_bonds(coords: np.ndarray, atom_types: np.ndarray, dataset_info: Dict[str, Any]) -> List[Tuple[int, int, int]]:
    if coords.shape[0] == 0:
        return []
    pos = torch.as_tensor(coords, dtype=torch.float32)
    types = torch.as_tensor(atom_types, dtype=torch.long)
    _, adjacency, edge_types = build_xae_molecule(pos, types, dataset_info)
    rows, cols = torch.nonzero(adjacency, as_tuple=True)
    bonds: List[Tuple[int, int, int]] = []
    for i, j in zip(rows.tolist(), cols.tolist()):
        order = int(edge_types[i, j].item())
        bonds.append((int(i), int(j), order))
    return bonds


def _neighbors_from_bonds(n_atoms: int, bonds: Sequence[Tuple[int, int, int]]) -> List[List[int]]:
    neigh = [[] for _ in range(int(n_atoms))]
    for i, j, order in bonds:
        if order <= 0:
            continue
        if 0 <= i < n_atoms and 0 <= j < n_atoms:
            neigh[i].append(j)
            neigh[j].append(i)
    return neigh


def _component_sizes(neigh: Sequence[Sequence[int]]) -> List[int]:
    n = len(neigh)
    seen = [False] * n
    sizes: List[int] = []
    for start in range(n):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        size = 0
        while stack:
            u = stack.pop()
            size += 1
            for v in neigh[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        sizes.append(size)
    return sizes


def _components(neigh: Sequence[Sequence[int]]) -> List[List[int]]:
    n = len(neigh)
    seen = [False] * n
    comps: List[List[int]] = []
    for start in range(n):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        comp: List[int] = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in neigh[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        comps.append(comp)
    return comps


def _min_intercomponent_dist(coords: np.ndarray, comps: Sequence[Sequence[int]]) -> Optional[float]:
    if coords.shape[0] == 0:
        return None
    if len(comps) <= 1:
        return None
    best: Optional[float] = None
    for a_idx in range(len(comps)):
        a = np.asarray(comps[a_idx], dtype=np.int64)
        if a.size == 0:
            continue
        for b_idx in range(a_idx + 1, len(comps)):
            b = np.asarray(comps[b_idx], dtype=np.int64)
            if b.size == 0:
                continue
            diff = coords[a][:, None, :] - coords[b][None, :, :]
            dist = np.sqrt(np.sum(diff * diff, axis=-1))
            if dist.size == 0:
                continue
            cand = float(np.min(dist))
            if not math.isfinite(cand):
                continue
            if best is None or cand < best:
                best = cand
    return best


def _angle_deg(v1: np.ndarray, v2: np.ndarray, eps: float = 1e-12) -> float:
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 <= eps or n2 <= eps:
        return float("nan")
    cos = float(np.dot(v1, v2) / (n1 * n2))
    cos = float(np.clip(cos, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def _dihedral_deg(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, eps: float = 1e-12) -> float:
    """Return dihedral angle in degrees in [-180, 180]."""
    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2

    b1_norm = np.linalg.norm(b1)
    if b1_norm <= eps:
        return float("nan")
    b1u = b1 / b1_norm

    v = b0 - np.dot(b0, b1u) * b1u
    w = b2 - np.dot(b2, b1u) * b1u

    v_norm = np.linalg.norm(v)
    w_norm = np.linalg.norm(w)
    if v_norm <= eps or w_norm <= eps:
        return float("nan")
    v /= v_norm
    w /= w_norm

    x = float(np.dot(v, w))
    y = float(np.dot(np.cross(b1u, v), w))
    angle = float(np.degrees(np.arctan2(y, x)))
    return angle


def _build_rdkit_mol(atom_types: np.ndarray, dataset_info: Dict[str, Any], bonds: Sequence[Tuple[int, int, int]]) -> Chem.Mol:
    atom_decoder = dataset_info["atom_decoder"]
    mol = Chem.RWMol()
    for t in atom_types.tolist():
        mol.AddAtom(Chem.Atom(atom_decoder[int(t)]))
    for i, j, order in bonds:
        if order <= 0:
            continue
        try:
            mol.AddBond(int(i), int(j), bond_dict[int(order)])
        except Exception:
            continue
    return mol


def _sanitize_smiles(mol: Chem.Mol) -> Optional[str]:
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def _ring_stats(mol: Chem.Mol) -> Tuple[int, List[int], int]:
    """Return (num_rings, ring_sizes, num_intersections)."""
    try:
        rings = Chem.GetSymmSSSR(mol)
    except Exception:
        return 0, [], 0
    ring_sets = [set(list(r)) for r in rings]
    ring_sizes = [len(s) for s in ring_sets]
    inter = 0
    for i in range(len(ring_sets)):
        for j in range(i + 1, len(ring_sets)):
            if ring_sets[i].intersection(ring_sets[j]):
                inter += 1
    return int(len(ring_sets)), ring_sizes, int(inter)


def _safe_mean(values: Sequence[float]) -> Optional[float]:
    vals = [v for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _hist(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.zeros((len(bins) - 1,), dtype=np.float64)
    counts, _ = np.histogram(values, bins=bins)
    return counts.astype(np.float64)


def _js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> Optional[float]:
    if p.size == 0 or q.size == 0:
        return None
    p = p.astype(np.float64, copy=False)
    q = q.astype(np.float64, copy=False)
    if p.sum() <= 0 or q.sum() <= 0:
        return None
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log((p + eps) / (m + eps)))
    kl_qm = np.sum(q * np.log((q + eps) / (m + eps)))
    js = 0.5 * (kl_pm + kl_qm) / math.log(2.0)
    return float(js)


@dataclass
class PerMol:
    valid: bool
    connected: bool
    n_atoms: int
    n_components: int
    n_singletons: int
    lcc_frac: float
    n_bonds: int
    min_intercomponent_dist: Optional[float]
    atom_stability: float
    mol_stability: int
    smiles: Optional[str]
    qed: Optional[float]
    logp: Optional[float]
    h_donors: Optional[int]
    h_acceptors: Optional[int]
    num_rings: Optional[int]
    ring_sizes: Optional[List[int]]
    ring_intersections: Optional[int]


def _compute_per_mol(
    coords: np.ndarray,
    atom_types: np.ndarray,
    dataset_info: Dict[str, Any],
) -> Tuple[PerMol, Dict[str, np.ndarray]]:
    n_atoms = int(coords.shape[0])
    bonds = _infer_bonds(coords, atom_types, dataset_info)
    neigh = _neighbors_from_bonds(n_atoms, bonds)
    comps = _components(neigh) if n_atoms > 0 else []
    comp_sizes = [len(c) for c in comps] if comps else []
    n_components = int(len(comp_sizes)) if n_atoms > 0 else 0
    lcc = int(max(comp_sizes)) if comp_sizes else 0
    lcc_frac = float(lcc) / float(max(1, n_atoms))
    connected = bool(n_atoms > 0 and n_components == 1)
    n_singletons = int(sum(1 for s in comp_sizes if s == 1))

    bond_pairs = set()
    for i, j, order in bonds:
        if order <= 0:
            continue
        a, b = (int(i), int(j))
        if a == b:
            continue
        if a > b:
            a, b = b, a
        bond_pairs.add((a, b))
    n_bonds = int(len(bond_pairs))

    min_intercomponent_dist = _min_intercomponent_dist(coords, comps) if n_components > 1 else None

    atom_stability = 0.0
    mol_stability = 0
    if n_atoms > 0:
        try:
            mol_stable, nr_stable, total = check_stability(coords, atom_types, dataset_info, debug=False)
            atom_stability = float(nr_stable) / float(max(1, int(total)))
            mol_stability = int(bool(mol_stable))
        except Exception:
            atom_stability = 0.0
            mol_stability = 0

    mol = _build_rdkit_mol(atom_types, dataset_info, bonds)
    smiles = _sanitize_smiles(mol)
    valid = smiles is not None

    qed = None
    logp = None
    h_donors = None
    h_acceptors = None
    num_rings = None
    ring_sizes: Optional[List[int]] = None
    ring_intersections = None
    if valid:
        try:
            qed = float(QED.qed(mol))
        except Exception:
            qed = None
        try:
            logp = float(Crippen.MolLogP(mol))
        except Exception:
            logp = None
        try:
            h_donors = int(Lipinski.NumHDonors(mol))
            h_acceptors = int(Lipinski.NumHAcceptors(mol))
        except Exception:
            h_donors = None
            h_acceptors = None
        try:
            num_rings_i, ring_sizes_i, inter = _ring_stats(mol)
            num_rings = int(num_rings_i)
            ring_sizes = list(map(int, ring_sizes_i))
            ring_intersections = int(inter)
        except Exception:
            num_rings = None
            ring_sizes = None
            ring_intersections = None

    # Geometry distributions from inferred bonds.
    bond_lengths: List[float] = []
    for i, j, order in bonds:
        if order <= 0:
            continue
        bond_lengths.append(float(np.linalg.norm(coords[i] - coords[j])))

    angles: List[float] = []
    for center, nbrs in enumerate(neigh):
        if len(nbrs) < 2:
            continue
        for a_idx in range(len(nbrs)):
            for b_idx in range(a_idx + 1, len(nbrs)):
                i = nbrs[a_idx]
                k = nbrs[b_idx]
                ang = _angle_deg(coords[i] - coords[center], coords[k] - coords[center])
                if math.isfinite(ang):
                    angles.append(float(ang))

    dihedrals: List[float] = []
    # Enumerate dihedrals via bonds (j-k), only once for j<k to avoid duplicates.
    for j, nbrs_j in enumerate(neigh):
        for k in nbrs_j:
            if j >= k:
                continue
            nbrs_k = neigh[k]
            for i in nbrs_j:
                if i == k:
                    continue
                for l in nbrs_k:
                    if l == j or l == i:
                        continue
                    dih = _dihedral_deg(coords[i], coords[j], coords[k], coords[l])
                    if math.isfinite(dih):
                        dihedrals.append(float(dih))

    geo = {
        "bond_lengths": np.asarray(bond_lengths, dtype=np.float32),
        "bond_angles_deg": np.asarray(angles, dtype=np.float32),
        "dihedrals_deg": np.asarray(dihedrals, dtype=np.float32),
    }

    return (
        PerMol(
            valid=valid,
            connected=connected,
            n_atoms=n_atoms,
            n_components=n_components,
            n_singletons=n_singletons,
            lcc_frac=lcc_frac,
            n_bonds=n_bonds,
            min_intercomponent_dist=min_intercomponent_dist,
            atom_stability=atom_stability,
            mol_stability=mol_stability,
            smiles=smiles,
            qed=qed,
            logp=logp,
            h_donors=h_donors,
            h_acceptors=h_acceptors,
            num_rings=num_rings,
            ring_sizes=ring_sizes,
            ring_intersections=ring_intersections,
        ),
        geo,
    )


def _extract_atoms(x: torch.Tensor, h_cat: torch.Tensor, node_mask: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    mask = node_mask.squeeze(-1).bool()
    coords = x[mask].detach().cpu().numpy().astype(np.float32, copy=False)
    atom_types = torch.argmax(h_cat[mask], dim=-1).detach().cpu().numpy().astype(np.int64, copy=False)
    return coords, atom_types


def _plot_overlay_hist(
    *,
    baseline: np.ndarray,
    post: np.ndarray,
    bins: np.ndarray,
    xlabel: str,
    title: str,
    out_base: Path,
) -> Dict[str, Any]:
    base_counts = _hist(baseline, bins)
    post_counts = _hist(post, bins)
    js = _js_divergence(base_counts, post_counts)

    fig = plt.figure(figsize=(3.6, 2.6))
    ax = plt.gca()
    ax.hist(baseline, bins=bins, density=True, alpha=0.45, label="baseline", color="#4C72B0")
    ax.hist(post, bins=bins, density=True, alpha=0.45, label="post-trained", color="#55A868")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    return {"js_divergence": js, "baseline_n": int(baseline.size), "post_n": int(post.size)}


def _summary_stats(values: np.ndarray) -> Dict[str, Any]:
    if values.size == 0:
        return {"n": 0}
    values = values.astype(np.float64, copy=False)
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "p10": float(np.percentile(values, 10)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MolDiff-style analysis for baseline vs post-trained GEOM generations.")
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Optional run dir (config.yaml) to auto-fill args-pickle/base/post paths.",
    )
    parser.add_argument("--args-pickle", type=str, default=None, help="Path to pretrained EDM args.pickle.")
    parser.add_argument("--base-weights", type=str, default=None, help="Baseline weights (e.g., generative_model_ema.npy).")
    parser.add_argument("--posttrained-weights", type=str, default=None, help="Post-trained checkpoint (.pth/.npy).")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--time-step", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-samples", type=int, default=512)
    parser.add_argument("--nodes-min", type=int, default=20)
    parser.add_argument("--nodes-max", type=int, default=80)
    parser.add_argument("--out-dir", type=str, default="benchmarks/moldiff_style")
    parser.add_argument("--tag", type=str, default="geom_moldiff_style")
    return parser.parse_args()


def _load_run_config(run_dir: Path) -> Dict[str, Any]:
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        return {}
    import yaml

    return yaml.safe_load(cfg_path.read_text()) or {}


def _resolve_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path]:
    run_dir = Path(args.run_dir).resolve() if args.run_dir else None
    run_cfg: Dict[str, Any] = _load_run_config(run_dir) if run_dir else {}

    def _resolve(p: Optional[str]) -> Optional[Path]:
        if not p:
            return None
        path = Path(p)
        if path.is_absolute():
            return path
        return (Path.cwd() / path).resolve()

    args_pickle = _resolve(args.args_pickle)
    base = _resolve(args.base_weights)
    post = _resolve(args.posttrained_weights)

    if run_dir is not None:
        model_cfg = run_cfg.get("model", {}) if isinstance(run_cfg, dict) else {}
        if args_pickle is None and model_cfg.get("config"):
            args_pickle = Path(str(model_cfg["config"])).resolve()
        if base is None and model_cfg.get("model_path"):
            base = Path(str(model_cfg["model_path"])).resolve()
        if post is None:
            # Prefer latest, then best.
            for cand in ("checkpoint_latest.pth", "checkpoint_best.pth"):
                candidate = run_dir / cand
                if candidate.exists():
                    post = candidate.resolve()
                    break

    if args_pickle is None or not args_pickle.exists():
        raise FileNotFoundError(f"args.pickle not found (got {args_pickle})")
    if base is None or not base.exists():
        raise FileNotFoundError(f"base weights not found (got {base})")
    if post is None or not post.exists():
        raise FileNotFoundError(f"posttrained weights not found (got {post})")
    return args_pickle, base, post


def main() -> None:
    args = parse_args()
    device = _select_device(args.device)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    args_pickle_path, base_weights_path, post_weights_path = _resolve_paths(args)

    with open(args_pickle_path, "rb") as f:
        edm_config = pickle.load(f)
    if isinstance(edm_config, dict):
        edm_config = OmegaConf.create(edm_config)

    edm_config.cuda = device.type == "cuda"
    edm_config.device = device
    if hasattr(edm_config, "no_cuda"):
        edm_config.no_cuda = not edm_config.cuda

    dataset_info = get_dataset_info(edm_config.dataset, edm_config.remove_h)

    # Build flows (no training dataloader needed for unconditional models).
    flow_base, nodes_dist, _ = get_model(edm_config, device, dataset_info, dataloader_train=None)
    flow_post, _, _ = get_model(edm_config, device, dataset_info, dataloader_train=None)

    base = EDMModel(flow_base, edm_config).to(device)
    post = EDMModel(flow_post, edm_config).to(device)
    load_model_weights(base, base_weights_path, device)
    load_model_weights(post, post_weights_path, device)
    base.model.eval()
    post.model.eval()
    # Sampling uses EnVariationalDiffusion.sample, which reads `self.T`.
    setattr(base.model, "T", int(args.time_step))
    setattr(post.model, "T", int(args.time_step))

    # Optional node-prior reweighting from run config (if provided).
    if args.run_dir:
        run_cfg = _load_run_config(Path(args.run_dir).resolve())
        dl_cfg = run_cfg.get("dataloader", {}) if isinstance(run_cfg, dict) else {}
        if isinstance(dl_cfg, dict):
            _reweight_nodes_dist(
                nodes_dist,
                focus_min=dl_cfg.get("nodes_dist_focus_min"),
                focus_max=dl_cfg.get("nodes_dist_focus_max"),
                focus_multiplier=float(dl_cfg.get("nodes_dist_focus_multiplier", 1.0) or 1.0),
            )

    nodes_per_sample = _collect_nodes(
        nodes_dist,
        n_samples=int(args.n_samples),
        nodes_min=int(args.nodes_min),
        nodes_max=int(args.nodes_max),
    )
    max_n_nodes = int(nodes_per_sample.max().item())
    node_mask, edge_mask = _make_masks(nodes_per_sample, max_n_nodes, device)

    # Sample paired molecules.
    _seed_all(int(args.seed), device)
    with torch.no_grad():
        x_base, h_base = base.model.sample(
            n_samples=int(args.n_samples),
            n_nodes=max_n_nodes,
            node_mask=node_mask,
            edge_mask=edge_mask,
            context=None,
            fix_noise=False,
        )
    _seed_all(int(args.seed), device)
    with torch.no_grad():
        x_post, h_post = post.model.sample(
            n_samples=int(args.n_samples),
            n_nodes=max_n_nodes,
            node_mask=node_mask,
            edge_mask=edge_mask,
            context=None,
            fix_noise=False,
        )

    per_base: List[PerMol] = []
    per_post: List[PerMol] = []
    lengths_base: List[np.ndarray] = []
    lengths_post: List[np.ndarray] = []
    angles_base: List[np.ndarray] = []
    angles_post: List[np.ndarray] = []
    dihed_base: List[np.ndarray] = []
    dihed_post: List[np.ndarray] = []

    for i in range(int(args.n_samples)):
        coords_b, types_b = _extract_atoms(x_base[i], h_base["categorical"][i], node_mask[i])
        coords_p, types_p = _extract_atoms(x_post[i], h_post["categorical"][i], node_mask[i])

        b_meta, b_geo = _compute_per_mol(coords_b, types_b, dataset_info)
        p_meta, p_geo = _compute_per_mol(coords_p, types_p, dataset_info)
        per_base.append(b_meta)
        per_post.append(p_meta)

        lengths_base.append(b_geo["bond_lengths"])
        lengths_post.append(p_geo["bond_lengths"])
        angles_base.append(b_geo["bond_angles_deg"])
        angles_post.append(p_geo["bond_angles_deg"])
        dihed_base.append(b_geo["dihedrals_deg"])
        dihed_post.append(p_geo["dihedrals_deg"])

    def _count(pred: Iterable[bool]) -> int:
        return int(sum(1 for v in pred if bool(v)))

    base_valid = np.array([m.valid for m in per_base], dtype=bool)
    post_valid = np.array([m.valid for m in per_post], dtype=bool)
    base_conn = np.array([m.connected for m in per_base], dtype=bool)
    post_conn = np.array([m.connected for m in per_post], dtype=bool)
    base_success = base_valid & base_conn
    post_success = post_valid & post_conn

    base_atom_stab = np.array([m.atom_stability for m in per_base], dtype=np.float32)
    post_atom_stab = np.array([m.atom_stability for m in per_post], dtype=np.float32)
    base_mol_stab = np.array([m.mol_stability for m in per_base], dtype=np.float32)
    post_mol_stab = np.array([m.mol_stability for m in per_post], dtype=np.float32)

    base_n_components = np.array([m.n_components for m in per_base], dtype=np.int32)
    post_n_components = np.array([m.n_components for m in per_post], dtype=np.int32)
    base_lcc_frac = np.array([m.lcc_frac for m in per_base], dtype=np.float32)
    post_lcc_frac = np.array([m.lcc_frac for m in per_post], dtype=np.float32)
    base_n_bonds = np.array([m.n_bonds for m in per_base], dtype=np.int32)
    post_n_bonds = np.array([m.n_bonds for m in per_post], dtype=np.int32)
    base_n_singletons = np.array([m.n_singletons for m in per_base], dtype=np.int32)
    post_n_singletons = np.array([m.n_singletons for m in per_post], dtype=np.int32)

    base_min_inter = np.array(
        [float(m.min_intercomponent_dist) if m.min_intercomponent_dist is not None else float("nan") for m in per_base],
        dtype=np.float32,
    )
    post_min_inter = np.array(
        [float(m.min_intercomponent_dist) if m.min_intercomponent_dist is not None else float("nan") for m in per_post],
        dtype=np.float32,
    )

    invalid_to_valid = int(np.sum((~base_valid) & post_valid))
    valid_to_invalid = int(np.sum(base_valid & (~post_valid)))

    # Flatten geometry arrays.
    bond_len_base = np.concatenate(lengths_base, axis=0) if lengths_base else np.empty((0,), dtype=np.float32)
    bond_len_post = np.concatenate(lengths_post, axis=0) if lengths_post else np.empty((0,), dtype=np.float32)
    angle_base = np.concatenate(angles_base, axis=0) if angles_base else np.empty((0,), dtype=np.float32)
    angle_post = np.concatenate(angles_post, axis=0) if angles_post else np.empty((0,), dtype=np.float32)
    dihed_base_all = np.concatenate(dihed_base, axis=0) if dihed_base else np.empty((0,), dtype=np.float32)
    dihed_post_all = np.concatenate(dihed_post, axis=0) if dihed_post else np.empty((0,), dtype=np.float32)

    # Ring stats distributions (valid only).
    ring_counts_base = np.asarray([m.num_rings for m in per_base if m.valid and m.num_rings is not None], dtype=np.int32)
    ring_counts_post = np.asarray([m.num_rings for m in per_post if m.valid and m.num_rings is not None], dtype=np.int32)

    ring_sizes_base_list: List[int] = []
    ring_sizes_post_list: List[int] = []
    for m in per_base:
        if m.valid and m.ring_sizes:
            ring_sizes_base_list.extend(m.ring_sizes)
    for m in per_post:
        if m.valid and m.ring_sizes:
            ring_sizes_post_list.extend(m.ring_sizes)
    ring_sizes_base = np.asarray(ring_sizes_base_list, dtype=np.int32)
    ring_sizes_post = np.asarray(ring_sizes_post_list, dtype=np.int32)

    macro_base = int(np.sum([any((s >= 8) for s in (m.ring_sizes or [])) for m in per_base if m.valid]))
    macro_post = int(np.sum([any((s >= 8) for s in (m.ring_sizes or [])) for m in per_post if m.valid]))

    summary: Dict[str, Any] = {
        "reference": "arXiv:2305.07508 (MolDiff)",
        "n_samples": int(args.n_samples),
        "time_step": int(args.time_step),
        "paths": {
            "args_pickle": str(args_pickle_path),
            "baseline": str(base_weights_path),
            "posttrained": str(post_weights_path),
        },
        "baseline": {
            "validity": float(np.mean(base_valid)),
            "connectivity": float(np.mean(base_conn)),
            "success_rate": float(np.mean(base_success)),
            "atom_stability": float(np.mean(base_atom_stab)),
            "molecule_stability": float(np.mean(base_mol_stab)),
            "num_valid": int(np.sum(base_valid)),
            "num_connected": int(np.sum(base_conn)),
        },
        "posttrained": {
            "validity": float(np.mean(post_valid)),
            "connectivity": float(np.mean(post_conn)),
            "success_rate": float(np.mean(post_success)),
            "atom_stability": float(np.mean(post_atom_stab)),
            "molecule_stability": float(np.mean(post_mol_stab)),
            "num_valid": int(np.sum(post_valid)),
            "num_connected": int(np.sum(post_conn)),
        },
        "transitions": {
            "invalid_to_valid": int(invalid_to_valid),
            "valid_to_invalid": int(valid_to_invalid),
        },
        "ring_stats_valid_only": {
            "macrocycle_frac_baseline": (macro_base / float(max(1, int(np.sum(base_valid))))),
            "macrocycle_frac_post": (macro_post / float(max(1, int(np.sum(post_valid))))),
        },
        "fragmentation_all": {
            "frac_multi_component_baseline": float(np.mean(base_n_components > 1)),
            "frac_multi_component_post": float(np.mean(post_n_components > 1)),
            "n_components_baseline": _summary_stats(base_n_components.astype(np.float32)),
            "n_components_post": _summary_stats(post_n_components.astype(np.float32)),
            "lcc_frac_baseline": _summary_stats(base_lcc_frac),
            "lcc_frac_post": _summary_stats(post_lcc_frac),
            "n_bonds_baseline": _summary_stats(base_n_bonds.astype(np.float32)),
            "n_bonds_post": _summary_stats(post_n_bonds.astype(np.float32)),
            "n_singletons_baseline": _summary_stats(base_n_singletons.astype(np.float32)),
            "n_singletons_post": _summary_stats(post_n_singletons.astype(np.float32)),
        },
        "fragmentation_disconnected_only": {
            "baseline": {
                "n": int(np.sum(base_n_components > 1)),
                "lcc_frac": _summary_stats(base_lcc_frac[base_n_components > 1]),
                "n_bonds": _summary_stats(base_n_bonds[base_n_components > 1].astype(np.float32)),
                "n_singletons": _summary_stats(base_n_singletons[base_n_components > 1].astype(np.float32)),
                "min_intercomponent_dist": _summary_stats(base_min_inter[np.isfinite(base_min_inter)]),
            },
            "post": {
                "n": int(np.sum(post_n_components > 1)),
                "lcc_frac": _summary_stats(post_lcc_frac[post_n_components > 1]),
                "n_bonds": _summary_stats(post_n_bonds[post_n_components > 1].astype(np.float32)),
                "n_singletons": _summary_stats(post_n_singletons[post_n_components > 1].astype(np.float32)),
                "min_intercomponent_dist": _summary_stats(post_min_inter[np.isfinite(post_min_inter)]),
            },
        },
        "conditional": {
            "baseline": {
                "connected": {
                    "n": int(np.sum(base_conn)),
                    "atom_stability": _summary_stats(base_atom_stab[base_conn]),
                    "molecule_stability": _summary_stats(base_mol_stab[base_conn]),
                    "n_bonds": _summary_stats(base_n_bonds[base_conn].astype(np.float32)),
                },
                "disconnected": {
                    "n": int(np.sum(~base_conn)),
                    "atom_stability": _summary_stats(base_atom_stab[~base_conn]),
                    "molecule_stability": _summary_stats(base_mol_stab[~base_conn]),
                    "n_bonds": _summary_stats(base_n_bonds[~base_conn].astype(np.float32)),
                },
            },
            "posttrained": {
                "connected": {
                    "n": int(np.sum(post_conn)),
                    "atom_stability": _summary_stats(post_atom_stab[post_conn]),
                    "molecule_stability": _summary_stats(post_mol_stab[post_conn]),
                    "n_bonds": _summary_stats(post_n_bonds[post_conn].astype(np.float32)),
                },
                "disconnected": {
                    "n": int(np.sum(~post_conn)),
                    "atom_stability": _summary_stats(post_atom_stab[~post_conn]),
                    "molecule_stability": _summary_stats(post_mol_stab[~post_conn]),
                    "n_bonds": _summary_stats(post_n_bonds[~post_conn].astype(np.float32)),
                },
            },
        },
        "js_divergence_baseline_vs_post": {},
    }

    # Plots (overlay baseline vs post).
    tag = str(args.tag)
    js_out: Dict[str, Any] = {}
    js_out["bond_length"] = _plot_overlay_hist(
        baseline=bond_len_base,
        post=bond_len_post,
        bins=np.linspace(0.6, 2.8, 91),
        xlabel="bond length (Å)",
        title="Bond length (inferred bonds)",
        out_base=out_dir / f"{tag}_bond_length",
    )
    js_out["bond_angle"] = _plot_overlay_hist(
        baseline=angle_base,
        post=angle_post,
        bins=np.linspace(0.0, 180.0, 73),
        xlabel="bond angle (deg)",
        title="Bond angle (inferred bonds)",
        out_base=out_dir / f"{tag}_bond_angle",
    )
    js_out["dihedral"] = _plot_overlay_hist(
        baseline=dihed_base_all,
        post=dihed_post_all,
        bins=np.linspace(-180.0, 180.0, 73),
        xlabel="dihedral (deg)",
        title="Dihedral angle (inferred bonds)",
        out_base=out_dir / f"{tag}_dihedral",
    )

    # Ring count histogram (valid only).
    ring_bins = np.arange(-0.5, 20.5, 1.0)
    js_out["num_rings"] = _plot_overlay_hist(
        baseline=ring_counts_base.astype(np.float32),
        post=ring_counts_post.astype(np.float32),
        bins=ring_bins,
        xlabel="#rings",
        title="Ring count (valid only)",
        out_base=out_dir / f"{tag}_num_rings",
    )

    ring_size_bins = np.arange(2.5, 20.5, 1.0)
    js_out["ring_size"] = _plot_overlay_hist(
        baseline=ring_sizes_base.astype(np.float32),
        post=ring_sizes_post.astype(np.float32),
        bins=ring_size_bins,
        xlabel="ring size (#atoms)",
        title="Ring size distribution (valid only)",
        out_base=out_dir / f"{tag}_ring_size",
    )

    js_out["atom_stability"] = _plot_overlay_hist(
        baseline=base_atom_stab.astype(np.float32),
        post=post_atom_stab.astype(np.float32),
        bins=np.linspace(0.0, 1.0, 51),
        xlabel="atom stability fraction",
        title="Atom stability (distance-based valency)",
        out_base=out_dir / f"{tag}_atom_stability",
    )

    comp_plot_max = int(min(10, max(int(np.max(base_n_components)), int(np.max(post_n_components)))))
    base_comp_clip = np.minimum(base_n_components, comp_plot_max).astype(np.float32)
    post_comp_clip = np.minimum(post_n_components, comp_plot_max).astype(np.float32)
    js_out["n_components"] = _plot_overlay_hist(
        baseline=base_comp_clip,
        post=post_comp_clip,
        bins=np.arange(-0.5, comp_plot_max + 1.5, 1.0),
        xlabel="num components (clipped)",
        title="Connected components (inferred bonds)",
        out_base=out_dir / f"{tag}_n_components",
    )
    js_out["lcc_frac"] = _plot_overlay_hist(
        baseline=base_lcc_frac,
        post=post_lcc_frac,
        bins=np.linspace(0.0, 1.0, 51),
        xlabel="largest component fraction",
        title="Largest component fraction",
        out_base=out_dir / f"{tag}_lcc_frac",
    )

    bond_plot_max = int(min(200, max(int(np.max(base_n_bonds)), int(np.max(post_n_bonds)))))
    base_bond_clip = np.minimum(base_n_bonds, bond_plot_max).astype(np.float32)
    post_bond_clip = np.minimum(post_n_bonds, bond_plot_max).astype(np.float32)
    js_out["n_bonds"] = _plot_overlay_hist(
        baseline=base_bond_clip,
        post=post_bond_clip,
        bins=np.arange(-0.5, bond_plot_max + 1.5, 1.0),
        xlabel="num bonds (clipped)",
        title="Inferred bonds per molecule",
        out_base=out_dir / f"{tag}_n_bonds",
    )

    base_min = base_min_inter[np.isfinite(base_min_inter)]
    post_min = post_min_inter[np.isfinite(post_min_inter)]
    if base_min.size > 0 and post_min.size > 0:
        upper = float(min(10.0, max(float(np.max(base_min)), float(np.max(post_min)))))
        js_out["min_intercomponent_dist"] = _plot_overlay_hist(
            baseline=base_min.astype(np.float32),
            post=post_min.astype(np.float32),
            bins=np.linspace(0.0, upper, 41),
            xlabel="min inter-component dist (Å)",
            title="Fragment separation (disconnected only)",
            out_base=out_dir / f"{tag}_min_intercomponent_dist",
        )

    summary["js_divergence_baseline_vs_post"] = js_out

    out_json = out_dir / f"{tag}_summary.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote: {out_json}")
    print(f"Wrote plots to: {out_dir}")
    print("Baseline validity/connectivity/success:", summary["baseline"])
    print("Post validity/connectivity/success:", summary["posttrained"])
    print("Transitions:", summary["transitions"])


if __name__ == "__main__":
    main()
