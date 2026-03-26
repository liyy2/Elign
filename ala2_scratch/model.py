from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn

try:
    from edm_source.egnn.egnn_new import EGNN
except ModuleNotFoundError:  # pragma: no cover
    from egnn.egnn_new import EGNN  # type: ignore[no-redef]


def _expand_node_features(
    node_features: torch.Tensor,
    batch_size: int,
    n_nodes: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Broadcast static node features to `[batch, n_nodes, feat_dim]`."""
    if node_features.dim() == 2:
        if node_features.shape[0] != n_nodes:
            raise ValueError(
                f"Expected node_features.shape[0] == n_nodes ({n_nodes}), got {node_features.shape[0]}"
            )
        node_features = node_features.unsqueeze(0).expand(batch_size, -1, -1)
    elif node_features.dim() == 3:
        if node_features.shape[:2] != (batch_size, n_nodes):
            raise ValueError(
                f"Expected node_features.shape[:2] == ({batch_size}, {n_nodes}), "
                f"got {tuple(node_features.shape[:2])}"
            )
    else:
        raise ValueError(
            f"node_features must have rank 2 or 3, got shape {tuple(node_features.shape)}"
        )
    return node_features.to(device=device, dtype=dtype)


def _first_tensor(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> Optional[torch.Tensor]:
    for key in keys:
        value = mapping.get(key)
        if torch.is_tensor(value):
            return value
    return None


def _normalize_timesteps(t: torch.Tensor | int | float, batch_size: int, device: torch.device) -> torch.Tensor:
    """Convert scalar or vector timesteps to shape `[batch]` on the target device."""
    if isinstance(t, (int, float)):
        return torch.full((batch_size,), float(t), device=device, dtype=torch.float32)
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"Unsupported timestep type: {type(t)!r}")
    t = t.to(device=device, dtype=torch.float32).view(-1)
    if t.numel() == 1:
        return t.expand(batch_size)
    if t.numel() != batch_size:
        raise ValueError(f"Expected {batch_size} timesteps, got {t.numel()}")
    return t


def _remove_mean_with_mask(x: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    """Enforce zero center-of-mass over valid nodes."""
    denom = node_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    mean = (x * node_mask).sum(dim=1, keepdim=True) / denom
    return (x - mean) * node_mask


class Ala2EGNNDenoiser(nn.Module):
    """Coordinate-only EGNN epsilon predictor aligned with the original QM9/GEOM wrapper."""

    def __init__(
        self,
        node_feature_dim: Optional[int] = None,
        *,
        in_node_nf: Optional[int] = None,
        n_nodes: int = 22,
        hidden_nf: int = 128,
        n_layers: int = 4,
        attention: bool = True,
        tanh: bool = False,
        norm_constant: float = 1.0,
        inv_sublayers: int = 2,
        sin_embedding: bool = False,
        normalization_factor: float = 1.0,
        aggregation_method: str = "sum",
        device: Optional[str] = None,
        dropout: float = 0.0,
        atom_embed_dim: Optional[int] = None,
        atom_type_embed_dim: Optional[int] = None,
        index_embed_dim: Optional[int] = None,
        atom_index_embed_dim: Optional[int] = None,
        time_embedding_dim: Optional[int] = None,
        time_embed_dim: Optional[int] = None,
        bond_pairs: Optional[Sequence[Sequence[int]]] = None,
        graph_type: str = "complete",
        graph_knn_k: int = 0,
        graph_knn_exclude_bonds: bool = True,
        **_: Any,
    ) -> None:
        super().__init__()
        if node_feature_dim is None:
            node_feature_dim = in_node_nf
        if node_feature_dim is None or node_feature_dim <= 0:
            raise ValueError("node_feature_dim must be > 0")
        if atom_index_embed_dim not in (None, 0) and index_embed_dim not in (None, 0):
            if int(atom_index_embed_dim) != int(index_embed_dim):
                raise ValueError(
                    "atom_index_embed_dim and index_embed_dim disagree. "
                    f"Got atom_index_embed_dim={atom_index_embed_dim}, index_embed_dim={index_embed_dim}."
                )
        resolved_index_embed_dim = int(atom_index_embed_dim or index_embed_dim or 0)
        unsupported = {
            "atom_embed_dim": atom_embed_dim,
            "atom_type_embed_dim": atom_type_embed_dim,
        }
        invalid = {key: value for key, value in unsupported.items() if value not in (None, 0)}
        if invalid:
            raise ValueError(
                "Ala2EGNNDenoiser does not implement auxiliary atom/type embeddings. "
                f"Remove unsupported config keys: {invalid}"
            )
        if any(value not in (None, 0, 32, 64) for value in (time_embedding_dim, time_embed_dim)):
            raise ValueError(
                "Ala2EGNNDenoiser uses the original scalar time conditioning; "
                "remove unsupported time embedding settings."
            )
        if float(dropout) != 0.0:
            raise ValueError(
                "Ala2EGNNDenoiser now mirrors the original EGNN wrapper and does not support dropout."
            )

        self.n_nodes = int(n_nodes)
        self.node_feature_dim = int(node_feature_dim)
        self.index_embed_dim = resolved_index_embed_dim
        self.graph_type = str(graph_type).lower()
        self.graph_knn_k = max(int(graph_knn_k), 0)
        self.graph_knn_exclude_bonds = bool(graph_knn_exclude_bonds)
        self._nan_warning_count = 0
        self.device = device or "cpu"
        if self.graph_type not in {"complete", "bonded", "knn", "hybrid"}:
            raise ValueError(f"Unsupported graph_type '{graph_type}'.")
        self.atom_index_embedding = (
            nn.Embedding(self.n_nodes, self.index_embed_dim) if self.index_embed_dim > 0 else None
        )
        bond_adjacency = torch.zeros(self.n_nodes, self.n_nodes, dtype=torch.bool)
        if bond_pairs is not None:
            for pair in bond_pairs:
                if len(pair) != 2:
                    raise ValueError(f"Bond pairs must have length 2, got {pair!r}.")
                i, j = int(pair[0]), int(pair[1])
                if i == j:
                    continue
                if not (0 <= i < self.n_nodes and 0 <= j < self.n_nodes):
                    raise ValueError(
                        f"Bond pair {(i, j)} is out of range for n_nodes={self.n_nodes}."
                    )
                bond_adjacency[i, j] = True
                bond_adjacency[j, i] = True
        self.register_buffer("bond_adjacency", bond_adjacency, persistent=False)
        effective_node_feature_dim = self.node_feature_dim + self.index_embed_dim
        self.egnn = EGNN(
            in_node_nf=effective_node_feature_dim + 1,
            in_edge_nf=1,
            hidden_nf=int(hidden_nf),
            out_node_nf=effective_node_feature_dim + 1,
            device=self.device,
            act_fn=nn.SiLU(),
            n_layers=int(n_layers),
            attention=bool(attention),
            tanh=bool(tanh),
            norm_constant=float(norm_constant),
            inv_sublayers=int(inv_sublayers),
            sin_embedding=bool(sin_embedding),
            normalization_factor=float(normalization_factor),
            aggregation_method=str(aggregation_method),
        )
        self._edge_cache: Dict[Tuple[torch.device, int], Tuple[torch.Tensor, torch.Tensor]] = {}

    def _sanitize_non_finite(self, vel: torch.Tensor) -> torch.Tensor:
        if torch.isfinite(vel).all():
            return vel
        if self._nan_warning_count < 5:
            print("Warning: detected non-finite EGNN output; sanitizing to zero.")
        elif self._nan_warning_count == 5:
            print("Warning: detected non-finite EGNN output; further warnings suppressed.")
        self._nan_warning_count += 1
        return torch.nan_to_num(vel, nan=0.0, posinf=0.0, neginf=0.0)

    def _get_edges(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (device, batch_size)
        cached = self._edge_cache.get(key)
        if cached is not None:
            return cached
        rows = []
        cols = []
        for batch_idx in range(batch_size):
            offset = batch_idx * self.n_nodes
            for i in range(self.n_nodes):
                for j in range(self.n_nodes):
                    rows.append(offset + i)
                    cols.append(offset + j)
        row_tensor = torch.tensor(rows, device=device, dtype=torch.long)
        col_tensor = torch.tensor(cols, device=device, dtype=torch.long)
        self._edge_cache[key] = (row_tensor, col_tensor)
        return row_tensor, col_tensor

    def _build_graph_adjacency(
        self,
        x_t: torch.Tensor,
        node_mask_3d: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, n_nodes, _ = x_t.shape
        valid_nodes = node_mask_3d.squeeze(-1) > 0.5
        adjacency = torch.zeros(batch_size, n_nodes, n_nodes, dtype=torch.bool, device=x_t.device)

        if self.graph_type in {"bonded", "hybrid"} and torch.any(self.bond_adjacency):
            adjacency |= self.bond_adjacency.unsqueeze(0)

        if self.graph_type in {"knn", "hybrid"} and self.graph_knn_k > 0:
            coords = _remove_mean_with_mask(x_t, node_mask_3d)
            distances = torch.cdist(coords, coords)
            pair_mask = valid_nodes.unsqueeze(1) & valid_nodes.unsqueeze(2)
            distances = distances.masked_fill(~pair_mask, float("inf"))
            diag = torch.eye(n_nodes, dtype=torch.bool, device=x_t.device).unsqueeze(0)
            distances = distances.masked_fill(diag, float("inf"))
            if self.graph_knn_exclude_bonds and torch.any(self.bond_adjacency):
                distances = distances.masked_fill(self.bond_adjacency.unsqueeze(0), float("inf"))
            max_neighbors = int((torch.isfinite(distances)).sum(dim=-1).max().item()) if distances.numel() > 0 else 0
            k = min(self.graph_knn_k, max_neighbors)
            if k > 0:
                knn_dist, knn_idx = torch.topk(distances, k=k, dim=-1, largest=False)
                valid_knn = torch.isfinite(knn_dist)
                batch_idx = torch.arange(batch_size, device=x_t.device).view(batch_size, 1, 1).expand_as(knn_idx)
                row_idx = torch.arange(n_nodes, device=x_t.device).view(1, n_nodes, 1).expand_as(knn_idx)
                adjacency[batch_idx[valid_knn], row_idx[valid_knn], knn_idx[valid_knn]] = True

        adjacency &= pair_mask
        adjacency &= ~torch.eye(n_nodes, dtype=torch.bool, device=x_t.device).unsqueeze(0)
        return adjacency

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor | int | float,
        node_features: torch.Tensor,
        *,
        node_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x_t.dim() != 3 or x_t.shape[-1] != 3:
            raise ValueError(f"Expected x_t with shape [batch, n_nodes, 3], got {tuple(x_t.shape)}")
        batch_size, n_nodes, _ = x_t.shape
        if n_nodes != self.n_nodes:
            raise ValueError(f"Expected n_nodes={self.n_nodes}, got {n_nodes}")

        if node_mask is None:
            node_mask_3d = torch.ones(batch_size, n_nodes, 1, device=x_t.device, dtype=x_t.dtype)
        else:
            if node_mask.dim() == 2:
                node_mask_3d = node_mask.unsqueeze(-1)
            elif node_mask.dim() == 3 and node_mask.shape[-1] == 1:
                node_mask_3d = node_mask
            else:
                raise ValueError(
                    f"Expected node_mask shape [batch, n_nodes] or [batch, n_nodes, 1], got {tuple(node_mask.shape)}"
                )
            node_mask_3d = node_mask_3d.to(device=x_t.device, dtype=x_t.dtype)

        node_features = _expand_node_features(
            node_features=node_features,
            batch_size=batch_size,
            n_nodes=n_nodes,
            device=x_t.device,
            dtype=x_t.dtype,
        )
        if self.atom_index_embedding is not None:
            atom_indices = torch.arange(n_nodes, device=x_t.device, dtype=torch.long)
            index_features = self.atom_index_embedding(atom_indices).to(dtype=x_t.dtype)
            index_features = index_features.unsqueeze(0).expand(batch_size, -1, -1)
            node_features = torch.cat([node_features, index_features], dim=-1)
        timesteps = _normalize_timesteps(t=t, batch_size=batch_size, device=x_t.device).view(batch_size, 1)
        x_in = _remove_mean_with_mask(x_t, node_mask_3d)
        node_mask_flat = node_mask_3d.reshape(batch_size * n_nodes, 1)
        if self.graph_type == "complete":
            edge_mask = (node_mask_3d.squeeze(-1).unsqueeze(1) * node_mask_3d.squeeze(-1).unsqueeze(2))
            diag_mask = ~torch.eye(n_nodes, dtype=torch.bool, device=x_t.device).unsqueeze(0)
            edge_mask = (edge_mask * diag_mask).reshape(batch_size * n_nodes * n_nodes, 1).to(dtype=x_t.dtype)
            edges = self._get_edges(batch_size=batch_size, device=x_t.device)
            keep = edge_mask.view(-1) > 0.5
            if torch.any(keep):
                edges = (edges[0][keep], edges[1][keep])
                edge_mask = edge_mask[keep]
            else:
                edges = (edges[0][:0], edges[1][:0])
                edge_mask = edge_mask[:0]
        else:
            adjacency = self._build_graph_adjacency(x_in, node_mask_3d)
            batch_idx, row_idx, col_idx = torch.where(adjacency)
            edges = (batch_idx * n_nodes + row_idx, batch_idx * n_nodes + col_idx)
            edge_mask = torch.ones((edges[0].shape[0], 1), device=x_t.device, dtype=x_t.dtype)

        xh = torch.cat([x_in, node_features], dim=-1).reshape(batch_size * n_nodes, -1) * node_mask_flat
        x = xh[:, :3].clone()
        h = xh[:, 3:].clone()
        h_time = timesteps.repeat(1, n_nodes).reshape(batch_size * n_nodes, 1)
        h = torch.cat([h, h_time], dim=1)

        _h_final, x_final = self.egnn(
            h,
            x,
            edges,
            node_mask=node_mask_flat,
            edge_mask=edge_mask,
        )
        vel = (x_final - x) * node_mask_flat
        vel = vel.reshape(batch_size, n_nodes, 3)
        vel = self._sanitize_non_finite(vel)
        return _remove_mean_with_mask(vel, node_mask_3d)


def _infer_node_feature_dim(
    metadata: Optional[Mapping[str, Any]],
    example_batch: Optional[Mapping[str, torch.Tensor]],
) -> int:
    if example_batch is not None:
        value = _first_tensor(example_batch, ("node_features", "static_features", "features", "atom_features"))
        if value is not None:
            return int(value.shape[-1])
    if metadata is not None:
        atomic_numbers = metadata.get("atomic_numbers")
        if atomic_numbers is not None:
            return len({int(z) for z in atomic_numbers})
    raise ValueError("Unable to infer node feature dimension for Ala2EGNNDenoiser.")


def build_model(
    *,
    config: Mapping[str, Any],
    metadata: Optional[Mapping[str, Any]] = None,
    example_batch: Optional[Mapping[str, torch.Tensor]] = None,
) -> Ala2EGNNDenoiser:
    model_cfg = dict(config.get("model", {}))
    node_feature_dim = model_cfg.get("node_feature_dim") or model_cfg.get("in_node_nf")
    if node_feature_dim is None:
        node_feature_dim = _infer_node_feature_dim(metadata, example_batch)

    n_nodes = model_cfg.get("n_nodes")
    if n_nodes is None and example_batch is not None:
        coordinates = _first_tensor(example_batch, ("positions", "coordinates", "x"))
        if coordinates is not None:
            n_nodes = int(coordinates.shape[-2])
    if n_nodes is None and metadata is not None:
        atomic_numbers = metadata.get("atomic_numbers")
        if atomic_numbers is not None:
            n_nodes = len(atomic_numbers)
    n_nodes = int(n_nodes or 22)

    return Ala2EGNNDenoiser(
        node_feature_dim=int(node_feature_dim),
        n_nodes=n_nodes,
        hidden_nf=int(model_cfg.get("hidden_nf", 128)),
        n_layers=int(model_cfg.get("n_layers", 4)),
        attention=bool(model_cfg.get("attention", True)),
        tanh=bool(model_cfg.get("tanh", False)),
        norm_constant=float(model_cfg.get("norm_constant", 1.0)),
        inv_sublayers=int(model_cfg.get("inv_sublayers", 2)),
        sin_embedding=bool(model_cfg.get("sin_embedding", False)),
        normalization_factor=float(model_cfg.get("normalization_factor", 1.0)),
        aggregation_method=str(model_cfg.get("aggregation_method", "sum")),
        device=model_cfg.get("device"),
        dropout=float(model_cfg.get("dropout", 0.0)),
        atom_embed_dim=model_cfg.get("atom_embed_dim"),
        atom_type_embed_dim=model_cfg.get("atom_type_embed_dim"),
        index_embed_dim=model_cfg.get("index_embed_dim"),
        atom_index_embed_dim=model_cfg.get("atom_index_embed_dim"),
        time_embedding_dim=model_cfg.get("time_embedding_dim"),
        time_embed_dim=model_cfg.get("time_embed_dim"),
        bond_pairs=None if metadata is None else metadata.get("bond_pairs"),
        graph_type=str(model_cfg.get("graph_type", "complete")),
        graph_knn_k=int(model_cfg.get("graph_knn_k", 0)),
        graph_knn_exclude_bonds=bool(model_cfg.get("graph_knn_exclude_bonds", True)),
    )


create_model = build_model
