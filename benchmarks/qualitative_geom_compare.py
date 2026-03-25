#!/usr/bin/env python
"""
Qualitative side-by-side visualization: GEOM baseline vs post-trained.

This script samples paired molecules from two checkpoints using identical RNG
seeds so both models start from the same stochastic noise. It then renders a
publication-style grid (Baseline vs Post-trained) for quick qualitative
comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager
from matplotlib.patches import Rectangle
from PIL import Image
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

REPO_ROOT = Path(__file__).resolve().parents[1]
EDM_SOURCE_ROOT = REPO_ROOT / "edm_source"
for p in (REPO_ROOT, EDM_SOURCE_ROOT):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

from edm_source.configs.datasets_config import get_dataset_info
from edm_source.qm9.dataset import retrieve_dataloaders
from edm_source.qm9.models import DistributionNodes, get_model
from edm_source.qm9.rdkit_functions import build_molecule, build_xae_molecule, mol2smiles
from eval_verl_rollout import load_model_weights
from verl_diffusion.model.edm_model import EDMModel


def _register_conda_fonts() -> None:
    """Register fonts installed into $CONDA_PREFIX/fonts (e.g., mscorefonts)."""
    prefix = Path(os.environ.get("CONDA_PREFIX", sys.prefix))
    font_dir = prefix / "fonts"
    if not font_dir.is_dir():
        return
    for ext in ("*.ttf", "*.otf"):
        for p in sorted(font_dir.glob(ext)):
            try:
                font_manager.fontManager.addfont(str(p))
            except Exception:
                continue


def _set_plot_style() -> None:
    _register_conda_fonts()
    matplotlib.rcParams.update(
        {
            # Editable vector text
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "text.usetex": False,
            # Publication-friendly sans font (prefer Arial if available).
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Nimbus Sans", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "figure.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            # Clean, publication-ish defaults.
            "axes.linewidth": 0.8,
            "lines.solid_capstyle": "round",
        }
    )


def _select_device(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _seed_all(seed: int, device: torch.device) -> None:
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))


def _make_masks(nodes_per_sample: torch.Tensor, max_n_nodes: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create node_mask [B,N,1] and edge_mask [B*N*N,1] for variable-sized batches."""
    nodes_per_sample = nodes_per_sample.to(device=device)
    batch = int(nodes_per_sample.shape[0])
    arange = torch.arange(max_n_nodes, device=device).view(1, -1)
    node_mask = (arange < nodes_per_sample.view(-1, 1)).to(dtype=torch.float32).unsqueeze(-1)

    # Edge mask is node_mask_i * node_mask_j with diagonal removed.
    edge_mask_2d = node_mask.squeeze(-1).unsqueeze(1) * node_mask.squeeze(-1).unsqueeze(2)
    eye = torch.eye(max_n_nodes, device=device, dtype=torch.bool).unsqueeze(0)
    edge_mask_2d = edge_mask_2d.masked_fill(eye, 0.0)
    edge_mask = edge_mask_2d.reshape(batch * max_n_nodes * max_n_nodes, 1)
    return node_mask, edge_mask


@dataclass
class MolRecord:
    index: int
    n_atoms: int
    baseline_smiles: Optional[str]
    posttrained_smiles: Optional[str]


def _graph_stats(n_atoms: int, bonds: List[Tuple[int, int, int]]) -> Tuple[int, float, int]:
    """Return (n_bonds, largest_component_frac, n_components) for an undirected graph."""
    if n_atoms <= 0:
        return 0, 0.0, 0
    adj: List[List[int]] = [[] for _ in range(n_atoms)]
    for i, j, _ in bonds:
        if i == j:
            continue
        if 0 <= i < n_atoms and 0 <= j < n_atoms:
            adj[i].append(j)
            adj[j].append(i)

    seen = [False] * n_atoms
    comp_sizes: List[int] = []
    for s in range(n_atoms):
        if seen[s]:
            continue
        stack = [s]
        seen[s] = True
        size = 0
        while stack:
            u = stack.pop()
            size += 1
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        comp_sizes.append(size)
    largest = max(comp_sizes) if comp_sizes else 0
    return int(len(bonds)), float(largest) / float(n_atoms), int(len(comp_sizes))


def _largest_component_indices(n_atoms: int, bonds: List[Tuple[int, int, int]]) -> List[int]:
    """Return atom indices of the largest connected component (undirected)."""
    if n_atoms <= 0:
        return []
    if not bonds:
        return list(range(n_atoms))
    adj: List[List[int]] = [[] for _ in range(n_atoms)]
    for i, j, _ in bonds:
        if i == j:
            continue
        if 0 <= i < n_atoms and 0 <= j < n_atoms:
            adj[i].append(j)
            adj[j].append(i)

    seen = [False] * n_atoms
    comps: List[List[int]] = []
    for s in range(n_atoms):
        if seen[s]:
            continue
        stack = [s]
        seen[s] = True
        comp: List[int] = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        comps.append(comp)
    if not comps:
        return list(range(n_atoms))
    comps.sort(key=lambda c: len(c), reverse=True)
    return sorted(comps[0])


def _keep_largest_component(
    coords: np.ndarray,
    atom_types: np.ndarray,
    symbols: List[str],
    bonds: List[Tuple[int, int, int]],
) -> Tuple[np.ndarray, np.ndarray, List[str], List[Tuple[int, int, int]]]:
    """Filter to largest connected component for cleaner visualization."""
    n = int(coords.shape[0])
    if n == 0:
        return coords, atom_types, symbols, bonds
    keep = _largest_component_indices(n, bonds)
    if len(keep) == n:
        return coords, atom_types, symbols, bonds
    keep_set = set(keep)
    remap = {old: new for new, old in enumerate(keep)}
    coords2 = coords[keep]
    atom_types2 = atom_types[keep]
    symbols2 = [symbols[i] for i in keep]
    bonds2: List[Tuple[int, int, int]] = []
    for i, j, order in bonds:
        if i in keep_set and j in keep_set:
            bonds2.append((remap[i], remap[j], int(order)))
    return coords2, atom_types2, symbols2, bonds2


def _rdkit_ring_count(smiles: Optional[str]) -> int:
    if not smiles:
        return 0
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0
        return int(mol.GetRingInfo().NumRings())
    except Exception:
        return 0


def _selection_score(
    coords: np.ndarray,
    symbols: List[str],
    bonds: List[Tuple[int, int, int]],
    smiles: Optional[str],
) -> float:
    """Heuristic: prefer connected, bond-rich, ring-containing examples for nicer figures."""
    n = int(coords.shape[0])
    if n == 0:
        return -1e9
    n_bonds, comp_frac, _ = _graph_stats(n, bonds)
    rings = _rdkit_ring_count(smiles)
    return float(n) + 0.6 * float(n_bonds) + 4.0 * float(rings) + 10.0 * float(comp_frac)


def _heavy_atom_count(symbols: List[str]) -> int:
    return int(sum(1 for s in symbols if s != "H"))


def _largest_fragment(mol: Chem.Mol) -> Chem.Mol:
    frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
    return max(frags, default=mol, key=lambda m: m.GetNumAtoms())


def _extract_atoms(
    x: torch.Tensor,
    h_cat: torch.Tensor,
    node_mask: torch.Tensor,
    dataset_info: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Return (coords[N,3], atom_type_idx[N], atom_symbols[N]) for masked atoms."""
    mask = node_mask.squeeze(-1).bool()
    coords = x[mask].detach().cpu().numpy()
    atom_types = torch.argmax(h_cat[mask], dim=-1).detach().cpu().numpy().astype(int)
    atom_decoder = dataset_info["atom_decoder"]
    symbols = [str(atom_decoder[int(t)]) for t in atom_types.tolist()]
    return coords, atom_types, symbols


def _filter_h(coords: np.ndarray, atom_types: np.ndarray, symbols: List[str], *, keep_h: bool) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if keep_h:
        return coords, atom_types, symbols
    keep = [i for i, s in enumerate(symbols) if s != "H"]
    if not keep:
        return coords[:0], atom_types[:0], []
    return coords[keep], atom_types[keep], [symbols[i] for i in keep]


def _smiles_from_atoms(coords: np.ndarray, atom_types: np.ndarray, dataset_info: Dict[str, Any]) -> Optional[str]:
    """Best-effort RDKit SMILES; returns None if sanitize fails."""
    if coords.shape[0] == 0:
        return None
    mol = build_molecule(torch.tensor(coords), torch.tensor(atom_types), dataset_info)
    try:
        mol = _largest_fragment(mol)
    except Exception:
        pass
    return mol2smiles(mol)


_ELEMENT_COLOR = {
    "H": "#CFCFCF",
    "B": "#FFB5B5",
    "C": "#3A3A3A",
    "N": "#3050F8",
    "O": "#FF0D0D",
    "F": "#90E050",
    "Al": "#BFA6A6",
    "Si": "#F0C8A0",
    "P": "#FF8000",
    "S": "#FFFF30",
    "Cl": "#1FF01F",
    "As": "#BD80E3",
    "Br": "#A62929",
    "I": "#940094",
    "Hg": "#B8B8D0",
    "Bi": "#9E4FB5",
}

_ELEMENT_RADIUS = {
    "H": 0.25,
    "B": 0.55,
    "C": 0.55,
    "N": 0.55,
    "O": 0.55,
    "F": 0.55,
    "Al": 0.65,
    "Si": 0.65,
    "P": 0.65,
    "S": 0.65,
    "Cl": 0.65,
    "As": 0.75,
    "Br": 0.75,
    "I": 0.85,
    "Hg": 0.85,
    "Bi": 0.85,
}

# Paper palette (user-specified).
_BASELINE_COLOR = "#9EB89F"
_POSTTRAINED_COLOR = "#D99DAE"
_INVALID_COLOR = "#D55E00"  # vermillion (Okabe-Ito)


def _stable_pca_rotation(coords: np.ndarray) -> np.ndarray:
    """Deterministic right-handed PCA basis for consistent view alignment."""
    if coords.shape[0] < 3:
        return np.eye(3)
    x = coords - coords.mean(axis=0, keepdims=True)
    # If degenerate (all points same), skip.
    if float(np.linalg.norm(x)) < 1e-9:
        return np.eye(3)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    r = vt.T
    # Ensure right-handed.
    if np.linalg.det(r) < 0:
        r[:, 2] *= -1.0
    # Fix sign ambiguity per-axis based on the point with max abs projection.
    proj = x @ r
    for k in range(3):
        idx = int(np.argmax(np.abs(proj[:, k])))
        if proj[idx, k] < 0:
            r[:, k] *= -1.0
    return r


def _infer_bonds(coords: np.ndarray, atom_types: np.ndarray, dataset_info: Dict[str, Any]) -> List[Tuple[int, int, int]]:
    """Infer bonds using the same distance heuristic as RDKit conversion."""
    if coords.shape[0] < 2:
        return []
    # build_xae_molecule expects torch tensors on CPU.
    _, a, e = build_xae_molecule(torch.tensor(coords), torch.tensor(atom_types), dataset_info)
    edges = []
    for i, j in torch.nonzero(a):
        order = int(e[i, j].item()) if e is not None else 1
        edges.append((int(i.item()), int(j.item()), int(order)))
    return edges


def _rot_x(deg: float) -> np.ndarray:
    rad = np.deg2rad(float(deg))
    c = float(np.cos(rad))
    s = float(np.sin(rad))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)


def _rot_z(deg: float) -> np.ndarray:
    rad = np.deg2rad(float(deg))
    c = float(np.cos(rad))
    s = float(np.sin(rad))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def _project_3d_to_2d(
    coords: np.ndarray,
    *,
    elev: float,
    azim: float,
    perspective: float = 0.10,
    camera_distance: float = 8.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project 3D coords -> 2D points + depth (z) after view rotation.

    We use a mild perspective projection so the result reads as 3D but stays
    stable for publication (no extreme distortion).
    """
    if coords.shape[0] == 0:
        return coords[:, :2], coords[:, 2]
    # Match mpl-style convention: azim rotates about z, elev about x.
    r_view = _rot_x(elev) @ _rot_z(azim)
    c = coords @ r_view.T
    if perspective <= 0.0:
        xy = c[:, :2]
    else:
        z = c[:, 2]
        denom = float(camera_distance) - float(perspective) * z
        denom = np.clip(denom, 1e-3, None)
        scale = float(camera_distance) / denom
        xy = c[:, :2] * scale[:, None]
    depth = c[:, 2]
    return xy, depth


def _hex_to_rgb01(hex_color: str) -> Tuple[float, float, float]:
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return (0.5, 0.5, 0.5)
    return (int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0)


def _rgb01_to_hex(rgb: Tuple[float, float, float]) -> str:
    r, g, b = [int(np.clip(x, 0.0, 1.0) * 255.0 + 0.5) for x in rgb]
    return f"#{r:02X}{g:02X}{b:02X}"


def _blend(hex_a: str, hex_b: str, t: float) -> str:
    """Blend hex_a -> hex_b by factor t in [0,1]."""
    ar, ag, ab = _hex_to_rgb01(hex_a)
    br, bg, bb = _hex_to_rgb01(hex_b)
    t = float(np.clip(t, 0.0, 1.0))
    return _rgb01_to_hex((ar + (br - ar) * t, ag + (bg - ag) * t, ab + (bb - ab) * t))


def _shade(hex_color: str, t: float) -> str:
    """Blend a color toward white by factor t in [0,1]."""
    r, g, b = _hex_to_rgb01(hex_color)
    r = r + (1.0 - r) * float(t)
    g = g + (1.0 - g) * float(t)
    b = b + (1.0 - b) * float(t)
    return _rgb01_to_hex((r, g, b))


def _rdkit_depiction_rgba(smiles: Optional[str], *, size: Tuple[int, int] = (900, 600)) -> Optional[Image.Image]:
    """Return a transparent RGBA depiction for overlay (background removed)."""
    if not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        rdDepictor.Compute2DCoords(mol)
        w, h = int(size[0]), int(size[1])
        drawer = rdMolDraw2D.MolDraw2DCairo(w, h)
        opts = drawer.drawOptions()
        # Simple, readable 2D overlay; main rendering carries color.
        opts.useBWAtomPalette()
        opts.bondLineWidth = 1.8
        opts.padding = 0.10
        opts.minFontSize = 6
        # Avoid a solid white box; we'll strip the remaining background anyway.
        try:
            opts.clearBackground = False
        except Exception:
            pass
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        png = drawer.GetDrawingText()
        img = Image.open(BytesIO(png)).convert("RGBA")
        arr = np.asarray(img).copy()
        # Make near-white pixels transparent (background).
        bg = (arr[:, :, 0] > 250) & (arr[:, :, 1] > 250) & (arr[:, :, 2] > 250)
        arr[bg, 3] = 0
        # Lighten remaining strokes/text for an unobtrusive overlay.
        mask = arr[:, :, 3] > 0
        arr[mask, 0:3] = 170
        arr[mask, 3] = np.minimum(arr[mask, 3], 180)
        return Image.fromarray(arr, mode="RGBA")
    except Exception:
        return None


def _draw_mol_projected(
    ax,
    coords: np.ndarray,
    symbols: List[str],
    bonds: List[Tuple[int, int, int]],
    *,
    valid: bool,
    border_color: str,
    overlay_smiles: Optional[str],
    shared_xy_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    elev: float = 18.0,
    azim: float = 35.0,
    atom_scale: float = 0.44,
    bond_lw: float = 1.45,
    bond_max_3d_dist: Optional[float] = None,
) -> None:
    ax.set_axis_off()
    ax.set_aspect("equal")

    # Panel border (thin, paper-friendly).
    ax.add_patch(
        Rectangle(
            (0.0, 0.0),
            1.0,
            1.0,
            transform=ax.transAxes,
            fill=False,
            lw=1.0,
            ec=border_color,
            zorder=50,
            clip_on=False,
        )
    )

    if coords.shape[0] == 0:
        if not valid:
            ax.text(
                0.5,
                0.5,
                "Invalid",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=11,
                color=border_color,
                fontweight="bold",
            )
        return

    xy, depth = _project_3d_to_2d(coords, elev=elev, azim=azim)
    depth_min = float(np.min(depth))
    depth_max = float(np.max(depth))
    denom = max(depth_max - depth_min, 1e-6)
    depth_norm = (depth - depth_min) / denom  # 0 = far, 1 = near

    # Draw bonds (trimmed to the sphere surfaces).
    base_radii = np.array([_ELEMENT_RADIUS.get(s, 0.6) for s in symbols], dtype=float) * float(atom_scale)
    # Mild depth scaling for a more 3D look.
    radii = base_radii * (0.88 + 0.22 * depth_norm)

    def _draw_bond_segment(p0: np.ndarray, p1: np.ndarray, *, order: int, z: float) -> None:
        d = p1 - p0
        dist = float(np.linalg.norm(d))
        if dist < 1e-6:
            return
        u = d / dist
        # Perpendicular in screen plane for multiple bonds.
        perp = np.array([-u[1], u[0]], dtype=float)
        sep = 0.055  # in data units; visually tuned
        offsets = [0.0]
        if order == 2:
            offsets = [-sep / 2.0, sep / 2.0]
        elif order >= 3:
            offsets = [-sep, 0.0, sep]
        # Cylinder-ish bonds: soft shadow + mid-tone tube + dark core + highlight.
        col_mid = _shade("#2B2B2B", t=0.35 * float(1.0 - z))
        col_core = _blend(col_mid, "#000000", 0.22)
        col_hi = _blend(col_mid, "#FFFFFF", 0.70)
        lw = float(bond_lw) * (0.85 + 0.30 * float(z))
        for off in offsets:
            o = perp * off
            # Soft bond shadow (adds depth without heavy outlines).
            ax.plot(
                [p0[0] + o[0] + 0.010, p1[0] + o[0] + 0.010],
                [p0[1] + o[1] - 0.010, p1[1] + o[1] - 0.010],
                color="#000000",
                lw=lw * 1.25,
                alpha=0.10,
                solid_capstyle="round",
                zorder=8 + z,
            )
            # Tube body.
            ax.plot(
                [p0[0] + o[0], p1[0] + o[0]],
                [p0[1] + o[1], p1[1] + o[1]],
                color=col_mid,
                lw=lw * 1.25,
                solid_capstyle="round",
                zorder=10 + z,
            )
            # Dark core.
            ax.plot(
                [p0[0] + o[0], p1[0] + o[0]],
                [p0[1] + o[1], p1[1] + o[1]],
                color=col_core,
                lw=lw * 0.95,
                solid_capstyle="round",
                zorder=10 + z,
            )
            # Highlight stroke.
            ax.plot(
                [p0[0] + o[0] - 0.012 * perp[0], p1[0] + o[0] - 0.012 * perp[0]],
                [p0[1] + o[1] - 0.012 * perp[1], p1[1] + o[1] - 0.012 * perp[1]],
                color=col_hi,
                lw=lw * 0.28,
                alpha=0.80,
                solid_capstyle="round",
                zorder=11 + z,
            )

    for i, j, order in bonds:
        if bond_max_3d_dist is not None:
            try:
                if float(np.linalg.norm(coords[int(i)] - coords[int(j)])) > float(bond_max_3d_dist):
                    continue
            except Exception:
                pass
        p0 = xy[i]
        p1 = xy[j]
        d = p1 - p0
        dist = float(np.linalg.norm(d))
        if dist < 1e-6:
            continue
        u = d / dist
        p0t = p0 + u * radii[i]
        p1t = p1 - u * radii[j]
        z = float(0.5 * (depth_norm[i] + depth_norm[j]))
        _draw_bond_segment(p0t, p1t, order=int(order), z=z)

    # Draw atoms back-to-front for proper occlusion.
    order = np.argsort(depth_norm)
    for idx in order:
        sym = symbols[int(idx)]
        base = _ELEMENT_COLOR.get(sym, "#808080")
        # Farther atoms slightly lighter (subtle depth cue).
        z = float(depth_norm[idx])
        face = _shade(base, t=0.18 * float(1.0 - z))
        x0 = float(xy[idx, 0])
        y0 = float(xy[idx, 1])
        r0 = float(radii[idx])
        zbase = 30.0 + z

        # Soft drop shadow to lift spheres off the page (small but effective in print).
        ax.add_patch(
            plt.Circle(
                (x0 + 0.10 * r0, y0 - 0.10 * r0),
                1.02 * r0,
                facecolor="#000000",
                edgecolor="none",
                alpha=0.10 * (0.55 + 0.45 * float(z)),
                zorder=zbase - 0.20,
            )
        )
        # Base "sphere".
        ax.add_patch(
            plt.Circle(
                (x0, y0),
                r0,
                facecolor=face,
                edgecolor=_blend(face, "#000000", 0.08),
                linewidth=0.6,
                alpha=0.98,
                zorder=zbase,
            )
        )
        # Specular highlight for a more 3D, publication-like look.
        ax.add_patch(
            plt.Circle(
                (x0 - 0.25 * r0, y0 + 0.25 * r0),
                0.42 * r0,
                facecolor=_blend(face, "#FFFFFF", 0.70),
                edgecolor="none",
                alpha=0.85,
                zorder=zbase + 0.02,
            )
        )
        # Subtle shadow on the lower-right to enhance roundness.
        ax.add_patch(
            plt.Circle(
                (x0 + 0.22 * r0, y0 - 0.22 * r0),
                0.78 * r0,
                facecolor="#000000",
                edgecolor="none",
                alpha=0.07 * (0.55 + 0.45 * float(z)),
                zorder=zbase + 0.01,
            )
        )

    # Tight limits (optionally shared across a column for comparability).
    if shared_xy_bounds is None:
        mins = xy.min(axis=0)
        maxs = xy.max(axis=0)
    else:
        mins, maxs = shared_xy_bounds
    center = 0.5 * (mins + maxs)
    span = float((maxs - mins).max())
    half = 0.5 * max(span, 1e-3)
    pad = 0.35 * half + 0.25
    ax.set_xlim(center[0] - half - pad, center[0] + half + pad)
    ax.set_ylim(center[1] - half - pad, center[1] + half + pad)

    if not valid:
        ax.text(
            0.5,
            0.5,
            "Invalid",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            color=border_color,
            fontweight="bold",
            zorder=60,
        )

    # Optional 2D overlay (transparent) for chemical graph readability.
    overlay = _rdkit_depiction_rgba(overlay_smiles)
    if overlay is not None:
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        w = float(xmax - xmin)
        h = float(ymax - ymin)
        # Bottom-left inset region; overlay background is transparent.
        ox0 = xmin + 0.04 * w
        oy0 = ymin + 0.04 * h
        ox1 = xmin + 0.58 * w
        oy1 = ymin + 0.32 * h
        ax.imshow(
            overlay,
            extent=[ox0, ox1, oy0, oy1],
            origin="upper",
            interpolation="bilinear",
            zorder=2,
            alpha=1.0,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Qualitative GEOM visualization: baseline vs post-trained.")
    parser.add_argument("--args-pickle", type=str, default="pretrained/edm/edm_geom_drugs/args.pickle")
    parser.add_argument("--base-weights", type=str, default="pretrained/edm/edm_geom_drugs/generative_model_ema.npy")
    parser.add_argument(
        "--posttrained-weights",
        type=str,
        default="outputs/verl_geom_smoke_v5_20260120_104415/generative_model_ema.npy",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--time-step", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-show", type=int, default=6, help="Number of molecule pairs to render.")
    parser.add_argument("--n-candidates", type=int, default=48, help="How many paired samples to generate before filtering.")
    parser.add_argument("--nodes-min", type=int, default=20)
    parser.add_argument("--nodes-max", type=int, default=60)
    parser.add_argument(
        "--heavy-min",
        type=int,
        default=20,
        help="Minimum number of heavy atoms (non-H) to keep a candidate for plotting.",
    )
    parser.add_argument(
        "--heavy-max",
        type=int,
        default=80,
        help="Maximum number of heavy atoms (non-H) to keep a candidate for plotting.",
    )
    parser.add_argument(
        "--heavy-delta-max",
        type=int,
        default=6,
        help="Maximum |heavy_baseline - heavy_posttrained| allowed when selecting candidates.",
    )
    parser.add_argument(
        "--min-connected-frac",
        type=float,
        default=0.70,
        help="For selected examples, require largest component fraction >= this.",
    )
    parser.add_argument(
        "--max-connected-frac-baseline-improve",
        type=float,
        default=1.00,
        help="For invalid->valid examples, prefer baseline with fragmentation: largest component fraction <= this.",
    )
    parser.add_argument(
        "--max-components",
        type=int,
        default=4,
        help="For improvement examples, require the post-trained number of components <= this.",
    )
    parser.add_argument("--keep-h", action="store_true", help="Keep explicit H in drawings (default: remove H).")
    parser.add_argument(
        "--selection",
        type=str,
        choices=["mixed", "both_valid", "improvement", "invalid_to_valid"],
        default="mixed",
        help="Which pairs to prioritize in the grid.",
    )
    parser.add_argument("--max-improvements", type=int, default=2, help="(mixed) cap baseline invalid -> post valid pairs.")
    parser.add_argument("--out-dir", type=str, default="benchmarks/qualitative_geom")
    parser.add_argument("--out-name", type=str, default="geom_baseline_vs_posttrained")
    args = parser.parse_args()

    _set_plot_style()

    device = _select_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    edm_config = None
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

    # Base model
    base = EDMModel(flow, edm_config).to(device)
    load_model_weights(base, Path(args.base_weights), device)
    base_flow = base.model
    setattr(base_flow, "T", int(args.time_step))

    # Post-trained model
    flow_post, _, _ = get_model(edm_config, device, dataset_info, dataloaders["train"])
    flow_post.to(device).eval()
    post = EDMModel(flow_post, edm_config).to(device)
    load_model_weights(post, Path(args.posttrained_weights), device)
    post_flow = post.model
    setattr(post_flow, "T", int(args.time_step))

    # Sample node counts (filtered for readability).
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

    keep_h = bool(args.keep_h)
    heavy_min = int(args.heavy_min)
    heavy_max = int(args.heavy_max)
    heavy_delta_max = int(args.heavy_delta_max)
    min_connected_frac = float(args.min_connected_frac)
    max_connected_frac_baseline_improve = float(args.max_connected_frac_baseline_improve)
    max_components = int(args.max_components)

    base_coords: List[np.ndarray] = []
    base_types: List[np.ndarray] = []
    base_symbols: List[List[str]] = []
    post_coords: List[np.ndarray] = []
    post_types: List[np.ndarray] = []
    post_symbols: List[List[str]] = []
    records: List[MolRecord] = []

    for i in range(target):
        nm = node_mask[i]
        n_atoms = int(nm.squeeze(-1).sum().item())

        coords_b, types_b, syms_b = _extract_atoms(x_base[i], h_base["categorical"][i], nm, dataset_info)
        coords_p, types_p, syms_p = _extract_atoms(x_post[i], h_post["categorical"][i], nm, dataset_info)

        base_coords.append(coords_b)
        base_types.append(types_b)
        base_symbols.append(syms_b)
        post_coords.append(coords_p)
        post_types.append(types_p)
        post_symbols.append(syms_p)

        smi_b = _smiles_from_atoms(coords_b, types_b, dataset_info)
        smi_p = _smiles_from_atoms(coords_p, types_p, dataset_info)

        records.append(MolRecord(index=i, n_atoms=n_atoms, baseline_smiles=smi_b, posttrained_smiles=smi_p))

    # Choose rows to show, prioritizing "interesting" pairs:
    # - baseline invalid -> post valid (and the post structure isn't just H)
    # - both valid but different
    improve_scored: List[Tuple[float, int]] = []
    improve_scored_relaxed: List[Tuple[float, int]] = []
    both_valid_scored: List[Tuple[float, int]] = []
    both_valid_scored_relaxed: List[Tuple[float, int]] = []
    any_valid_scored: List[Tuple[float, int]] = []

    for i, r in enumerate(records):
        coords_b, types_b, syms_b = _filter_h(base_coords[i], base_types[i], base_symbols[i], keep_h=keep_h)
        coords_p, types_p, syms_p = _filter_h(post_coords[i], post_types[i], post_symbols[i], keep_h=keep_h)
        if coords_b.shape[0] == 0 or coords_p.shape[0] == 0:
            continue

        bonds_b = _infer_bonds(coords_b, types_b, dataset_info) if coords_b.shape[0] > 0 else []
        bonds_p = _infer_bonds(coords_p, types_p, dataset_info) if coords_p.shape[0] > 0 else []

        # Work with the largest connected component to avoid selecting pairs where
        # one model emits many fragments (which would look tiny after LCC filtering).
        coords_b_lcc, types_b_lcc, syms_b_lcc, bonds_b_lcc = _keep_largest_component(coords_b, types_b, syms_b, bonds_b)
        coords_p_lcc, types_p_lcc, syms_p_lcc, bonds_p_lcc = _keep_largest_component(coords_p, types_p, syms_p, bonds_p)

        heavy_b = int(coords_b_lcc.shape[0])
        heavy_p = int(coords_p_lcc.shape[0])
        if not (heavy_min <= heavy_b <= heavy_max and heavy_min <= heavy_p <= heavy_max):
            continue
        if abs(heavy_b - heavy_p) > heavy_delta_max:
            continue

        score_b = _selection_score(coords_b_lcc, syms_b_lcc, bonds_b_lcc, r.baseline_smiles)
        score_p = _selection_score(coords_p_lcc, syms_p_lcc, bonds_p_lcc, r.posttrained_smiles)
        _, comp_frac_b, n_comp_b = _graph_stats(int(coords_b.shape[0]), bonds_b)
        _, comp_frac_p, n_comp_p = _graph_stats(int(coords_p.shape[0]), bonds_p)

        if r.baseline_smiles is None and r.posttrained_smiles is not None:
            improve_scored_relaxed.append((score_p, i))
            # Prefer post-trained samples that are mostly connected (prettier 3D panels).
            if (
                comp_frac_p >= min_connected_frac
                and n_comp_p <= max_components
                and comp_frac_b <= max_connected_frac_baseline_improve
            ):
                improve_scored.append((score_p, i))
        if r.baseline_smiles is not None and r.posttrained_smiles is not None and r.baseline_smiles != r.posttrained_smiles:
            both_valid_scored_relaxed.append((0.5 * (score_b + score_p), i))
            if comp_frac_b >= min_connected_frac and comp_frac_p >= min_connected_frac and n_comp_b <= max_components and n_comp_p <= max_components:
                both_valid_scored.append((0.5 * (score_b + score_p), i))
        if r.baseline_smiles is not None or r.posttrained_smiles is not None:
            if comp_frac_b >= min_connected_frac and comp_frac_p >= min_connected_frac and n_comp_b <= max_components and n_comp_p <= max_components:
                any_valid_scored.append((max(score_b, score_p), i))

    for lst in (improve_scored, improve_scored_relaxed, both_valid_scored, both_valid_scored_relaxed, any_valid_scored):
        lst.sort(key=lambda t: t[0], reverse=True)
    idx_improve = [i for _, i in improve_scored]
    idx_improve_relaxed = [i for _, i in improve_scored_relaxed]
    idx_both_valid = [i for _, i in both_valid_scored]
    idx_both_valid_relaxed = [i for _, i in both_valid_scored_relaxed]
    idx_any_valid = [i for _, i in any_valid_scored]

    selection = str(args.selection)
    chosen: List[int] = []
    if selection == "invalid_to_valid":
        for pool in (idx_improve, idx_improve_relaxed):
            for i in pool:
                if i not in chosen:
                    chosen.append(i)
                if len(chosen) >= int(args.n_show):
                    break
            if len(chosen) >= int(args.n_show):
                break
        if len(chosen) < int(args.n_show):
            raise SystemExit(
                f"Only found {len(chosen)} invalid->valid pairs (requested {int(args.n_show)}). "
                "Try increasing --n-candidates or relaxing size/connectivity filters."
            )
    elif selection == "improvement":
        for pool in (idx_improve, idx_improve_relaxed, idx_both_valid, idx_any_valid):
            for i in pool:
                if i not in chosen:
                    chosen.append(i)
                if len(chosen) >= int(args.n_show):
                    break
            if len(chosen) >= int(args.n_show):
                break
    elif selection == "both_valid":
        for pool in (idx_both_valid, idx_both_valid_relaxed, idx_any_valid):
            for i in pool:
                if i not in chosen:
                    chosen.append(i)
                if len(chosen) >= int(args.n_show):
                    break
            if len(chosen) >= int(args.n_show):
                break
    else:
        # mixed: a few improvements (if any), then fill with both-valid, then any-valid.
        for pool in (idx_improve, idx_improve_relaxed):
            for i in pool[: int(args.max_improvements)]:
                if i not in chosen:
                    chosen.append(i)
        for pool in (idx_both_valid, idx_both_valid_relaxed, idx_any_valid):
            for i in pool:
                if i not in chosen:
                    chosen.append(i)
                if len(chosen) >= int(args.n_show):
                    break
            if len(chosen) >= int(args.n_show):
                break

    if not chosen:
        chosen = list(range(min(int(args.n_show), target)))

    # Render grid (2 rows x N columns): Baseline row over Post-trained row.
    n_cols = len(chosen)
    fig_w = 7.2  # two-column figure width
    fig_h = 3.6 if n_cols <= 5 else 4.0
    fig = plt.figure(figsize=(fig_w, fig_h))
    if selection == "invalid_to_valid":
        fig.suptitle("Invalid → valid examples", fontsize=11, fontweight="bold", y=0.985)
    elif selection == "improvement":
        fig.suptitle("Improvement examples (invalid→valid preferred)", fontsize=11, fontweight="bold", y=0.985)
    gs = fig.add_gridspec(
        2,
        n_cols,
        left=0.06,
        right=0.995,
        top=0.90 if selection not in {"improvement", "invalid_to_valid"} else 0.93,
        bottom=0.08,
        wspace=0.02,
        hspace=0.02,
    )
    axes = [[fig.add_subplot(gs[r, c]) for c in range(n_cols)] for r in range(2)]

    # Row labels (outside axes).
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

    for c, idx in enumerate(chosen):
        rec = records[idx]
        valid_b = rec.baseline_smiles is not None
        valid_p = rec.posttrained_smiles is not None

        coords_b, types_b, syms_b = _filter_h(base_coords[idx], base_types[idx], base_symbols[idx], keep_h=keep_h)
        coords_p, types_p, syms_p = _filter_h(post_coords[idx], post_types[idx], post_symbols[idx], keep_h=keep_h)

        bonds_b = _infer_bonds(coords_b, types_b, dataset_info) if coords_b.shape[0] > 0 else []
        bonds_p = _infer_bonds(coords_p, types_p, dataset_info) if coords_p.shape[0] > 0 else []

        # For clearer qualitative figures, show only the largest connected component.
        coords_b, types_b, syms_b, bonds_b = _keep_largest_component(coords_b, types_b, syms_b, bonds_b)
        coords_p, types_p, syms_p, bonds_p = _keep_largest_component(coords_p, types_p, syms_p, bonds_p)

        # Align the view per column using a deterministic PCA basis from the baseline sample.
        ref = coords_b if coords_b.shape[0] > 0 else coords_p
        rot = _stable_pca_rotation(ref) if ref.shape[0] > 0 else np.eye(3)
        coords_b_rot = (coords_b - coords_b.mean(axis=0, keepdims=True)) @ rot if coords_b.shape[0] > 0 else coords_b
        coords_p_rot = (coords_p - coords_p.mean(axis=0, keepdims=True)) @ rot if coords_p.shape[0] > 0 else coords_p

        # Use shared 2D limits per column for apples-to-apples visual comparison.
        all_xy = []
        if coords_b_rot.shape[0] > 0:
            xy_b, _ = _project_3d_to_2d(coords_b_rot, elev=18.0, azim=35.0)
            all_xy.append(xy_b)
        if coords_p_rot.shape[0] > 0:
            xy_p, _ = _project_3d_to_2d(coords_p_rot, elev=18.0, azim=35.0)
            all_xy.append(xy_p)
        shared_bounds = None
        if all_xy:
            stack = np.concatenate(all_xy, axis=0)
            shared_bounds = (stack.min(axis=0), stack.max(axis=0))

        color_b = _INVALID_COLOR if not valid_b else _BASELINE_COLOR
        color_p = _INVALID_COLOR if not valid_p else _POSTTRAINED_COLOR
        _draw_mol_projected(
            axes[0][c],
            coords_b_rot,
            syms_b,
            bonds_b,
            valid=valid_b,
            border_color=color_b,
            overlay_smiles=rec.baseline_smiles,
            shared_xy_bounds=shared_bounds,
        )
        _draw_mol_projected(
            axes[1][c],
            coords_p_rot,
            syms_p,
            bonds_p,
            valid=valid_p,
            border_color=color_p,
            overlay_smiles=rec.posttrained_smiles,
            shared_xy_bounds=shared_bounds,
        )

        # Column label: size + validity transition.
        if selection == "improvement":
            label = f"N={rec.n_atoms}"
            axes[0][c].set_title(label, fontsize=10, pad=6)
        else:
            label = f"N={rec.n_atoms}  ({'valid' if valid_b else 'invalid'} -> {'valid' if valid_p else 'invalid'})"
            axes[0][c].set_title(label, fontsize=9, pad=6)

    out_base = out_dir / args.out_name
    fig.savefig(out_base.with_suffix(".png"), dpi=300)
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)

    # Metadata for paper captions / reproducibility.
    meta_path = out_base.with_suffix(".json")
    meta = {
        "args": vars(args),
        "chosen_indices": chosen,
        "records": [r.__dict__ for r in records],
        "notes": "Paired samples share RNG seed; baseline and post-trained were sampled separately with the same seed.",
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote figure to {out_base.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
