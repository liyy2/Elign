"""MolFM QM9 sampling core adapted for fixed-step rollout integration.

This module intentionally vendors only the small subset of MolFM needed for
QM9 checkpoint loading and fixed-step sampling in this repository. The upstream
public repo currently ships a sampling bundle rather than a full training stack,
so we keep the implementation narrow and dependency-light here.
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EDM_SOURCE_ROOT = os.path.join(REPO_ROOT, "edm_source")
if EDM_SOURCE_ROOT not in sys.path:
    sys.path.insert(0, EDM_SOURCE_ROOT)

from egnn.models import EGNN_dynamics_QM9
from edm_source.qm9.models import DistributionNodes
from equivariant_diffusion import utils as diffusion_utils


def T(t: torch.Tensor) -> torch.Tensor:
    beta_min = 0.1
    beta_max = 20.0
    return 0.5 * (beta_max - beta_min) * t**2 + beta_min * t


def T_hat(t: torch.Tensor) -> torch.Tensor:
    beta_min = 0.1
    beta_max = 20.0
    return (beta_max - beta_min) * t + beta_min


class UniformDequantizer(nn.Module):
    """Minimal dequantizer used by the upstream MolFM QM9 sampler."""

    def forward(self, tensor, node_mask, edge_mask, context):
        del edge_mask, context
        category, integer = tensor["categorical"], tensor["integer"]
        zeros = torch.zeros(integer.size(0), device=integer.device)

        out_category = category + torch.rand_like(category) - 0.5
        out_integer = integer + torch.rand_like(integer) - 0.5

        if node_mask is not None:
            out_category = out_category * node_mask
            out_integer = out_integer * node_mask

        out = {"categorical": out_category, "integer": out_integer}
        return out, zeros

    def reverse(self, tensor):
        categorical, integer = tensor["categorical"], tensor["integer"]
        categorical = torch.round(categorical)
        integer = torch.round(integer)
        return {"categorical": categorical, "integer": integer}


class MolFMFlowCore(nn.Module):
    """Checkpoint-compatible MolFM QM9 flow model without `torchdiffeq`.

    The upstream public repo uses an adaptive ODE solver (`odeint`) for sampling.
    For RL post-training we need explicit, step-aligned transitions, so this core
    only retains the learned vector field and decoding helpers.
    """

    def __init__(
        self,
        dynamics: nn.Module,
        in_node_nf: int,
        n_dims: int,
        timesteps: int = 1000,
        parametrization: str = "eps",
        time_embed: bool = False,
        noise_schedule: str = "learned",
        noise_precision: float = 1e-4,
        loss_type: str = "ot",
        norm_values: Sequence[float] = (1.0, 1.0, 1.0),
        norm_biases: Tuple[Optional[float], float, float] = (None, 0.0, 0.0),
        include_charges: bool = True,
        discrete_path: str = "OT_path",
        cat_loss: str = "l2",
        cat_loss_step: int = -1,
        on_hold_batch: int = -1,
        sampling_method: str = "vanilla",
        weighted_methods: str = "jump",
        ode_method: str = "fixed_euler",
        without_cat_loss: bool = False,
        angle_penalty: bool = False,
    ):
        del noise_schedule, noise_precision
        super().__init__()

        self.loss_type = loss_type
        self.include_charges = bool(include_charges)
        self.discrete_path = discrete_path
        self.ode_method = ode_method
        self.cat_loss = cat_loss
        self.cat_loss_step = cat_loss_step
        self.on_hold_batch = on_hold_batch
        self.sampling_method = sampling_method
        self.weighted_methods = weighted_methods
        self.without_cat_loss = without_cat_loss
        self.angle_penalty = angle_penalty

        self.dynamics = dynamics
        self.in_node_nf = int(in_node_nf)
        self.n_dims = int(n_dims)
        self.num_classes = self.in_node_nf - int(self.include_charges)
        self.T = int(timesteps)
        self.parametrization = parametrization
        self.norm_values = tuple(norm_values)
        self.norm_biases = tuple(norm_biases)
        self.time_embed = bool(time_embed)
        self.register_buffer("buffer", torch.zeros(1))

    def phi(self, t, x, node_mask, edge_mask, context):
        return self.dynamics._forward(t, x, node_mask, edge_mask, context)

    def subspace_dimensionality(self, node_mask):
        number_of_nodes = torch.sum(node_mask.squeeze(2), dim=1)
        return (number_of_nodes - 1) * self.n_dims

    def normalize(self, x, h, node_mask):
        cat_bias = 0.0 if self.norm_biases[1] is None else self.norm_biases[1]
        int_bias = 0.0 if self.norm_biases[2] is None else self.norm_biases[2]
        x = x / self.norm_values[0]
        delta_log_px = -self.subspace_dimensionality(node_mask) * np.log(self.norm_values[0])
        h_cat = ((h["categorical"].float() - cat_bias) / self.norm_values[1]) * node_mask
        h_int = (h["integer"].float() - int_bias) / self.norm_values[2]

        if self.include_charges:
            h_int = h_int * node_mask

        h = {"categorical": h_cat, "integer": h_int}
        return x, h, delta_log_px

    def unnormalize(self, x, h_cat, h_int, node_mask):
        cat_bias = 0.0 if self.norm_biases[1] is None else self.norm_biases[1]
        int_bias = 0.0 if self.norm_biases[2] is None else self.norm_biases[2]
        x = x * self.norm_values[0]
        h_cat = h_cat * self.norm_values[1] + cat_bias
        h_cat = h_cat * node_mask
        h_int = h_int * self.norm_values[2] + int_bias

        if self.include_charges:
            h_int = h_int * node_mask

        return x, h_cat, h_int

    def sample_p_xh_given_z0(self, dequantizer, z0, node_mask):
        x = z0[:, :, : self.n_dims]
        if self.include_charges:
            h_int = z0[:, :, -1:]
        else:
            h_int = z0.new_zeros(z0.size(0), z0.size(1), 0)

        x, h_cat, h_int = self.unnormalize(
            x,
            z0[:, :, self.n_dims : self.n_dims + self.num_classes],
            h_int,
            node_mask,
        )
        tensor = dequantizer.reverse({"categorical": h_cat, "integer": h_int})
        h = {"integer": tensor["integer"], "categorical": tensor["categorical"]}
        return x, h

    def sample_combined_position_feature_noise(self, n_samples, n_nodes, node_mask):
        z_x = diffusion_utils.sample_center_gravity_zero_gaussian_with_mask(
            size=(n_samples, n_nodes, self.n_dims),
            device=node_mask.device,
            node_mask=node_mask,
        )
        z_h = diffusion_utils.sample_gaussian_with_mask(
            size=(n_samples, n_nodes, self.in_node_nf),
            device=node_mask.device,
            node_mask=node_mask,
        )
        return torch.cat([z_x, z_h], dim=2)

    def flow_drift(self, t, z, node_mask, edge_mask, context):
        dx = self.phi(t, z, node_mask, edge_mask, context)

        if self.cat_loss_step > 0:
            cat_slice = slice(self.n_dims, -1 if self.include_charges else None)
            if torch.is_tensor(t):
                t_scalar = t.reshape(t.shape[0], -1)[:, 0]
                rescale = torch.where(
                    t_scalar > float(self.cat_loss_step),
                    torch.zeros_like(t_scalar),
                    torch.full_like(t_scalar, 1.0 / float(self.cat_loss_step)),
                ).view(-1, 1, 1)
                dx[:, :, cat_slice] = dx[:, :, cat_slice] * rescale

        if self.discrete_path == "VP_path":
            scale = -0.5 * T_hat(t) / (1 - torch.exp(-T(t)) + 1e-5)
            dx = dx * scale.reshape(scale.shape[0], 1, 1)
        elif self.discrete_path == "HB_path":
            scale = -0.5 * T_hat(t) / (1 - torch.exp(-T(t)) + 1e-5)
            dx[:, :, self.n_dims :] = dx[:, :, self.n_dims :] * scale.reshape(scale.shape[0], 1, 1)

        return dx


def build_molfm_qm9_components(args, device, dataset_info, dataloader_train=None):
    del dataloader_train

    dataset_name = str(getattr(args, "dataset", "") or "").lower()
    if "qm9" not in dataset_name:
        raise NotImplementedError(
            "The public MolFM bundle currently only exposes a QM9 checkpoint/config; "
            f"got dataset={getattr(args, 'dataset', None)!r}."
        )

    histogram = dataset_info["n_nodes"]
    include_charges = bool(getattr(args, "include_charges", True))
    in_node_nf = len(dataset_info["atom_decoder"]) + int(include_charges)
    nodes_dist = DistributionNodes(histogram)

    condition_time = bool(getattr(args, "condition_time", True))
    if condition_time:
        dynamics_in_node_nf = in_node_nf + 1
    else:
        dynamics_in_node_nf = in_node_nf

    net_dynamics = EGNN_dynamics_QM9(
        in_node_nf=dynamics_in_node_nf,
        context_node_nf=int(getattr(args, "context_node_nf", 0) or 0),
        n_dims=3,
        device=device,
        hidden_nf=int(getattr(args, "nf", 256)),
        act_fn=torch.nn.SiLU(),
        n_layers=int(getattr(args, "n_layers", 9)),
        attention=bool(getattr(args, "attention", True)),
        tanh=bool(getattr(args, "tanh", True)),
        mode=str(getattr(args, "model", "egnn_dynamics")),
        norm_constant=float(getattr(args, "norm_constant", 1.0)),
        inv_sublayers=int(getattr(args, "inv_sublayers", 1)),
        sin_embedding=bool(getattr(args, "sin_embedding", False)),
        normalization_factor=float(getattr(args, "normalization_factor", 1.0)),
        aggregation_method=str(getattr(args, "aggregation_method", "sum")),
    )

    flow = MolFMFlowCore(
        dynamics=net_dynamics,
        in_node_nf=in_node_nf,
        n_dims=3,
        timesteps=int(getattr(args, "diffusion_steps", 1000)),
        noise_schedule=str(getattr(args, "diffusion_noise_schedule", "polynomial_2")),
        noise_precision=float(getattr(args, "diffusion_noise_precision", 1e-5)),
        loss_type=str(getattr(args, "diffusion_loss_type", "l2")),
        norm_values=tuple(getattr(args, "normalize_factors", [1.0, 1.0, 1.0])),
        include_charges=include_charges,
        discrete_path=str(getattr(args, "discrete_path", "HB_path")),
        cat_loss=str(getattr(args, "cat_loss", "l2")),
        cat_loss_step=int(getattr(args, "cat_loss_step", -1)),
        on_hold_batch=int(getattr(args, "on_hold_batch", -1)),
        sampling_method=str(getattr(args, "sampling_method", "vanilla")),
        weighted_methods=str(getattr(args, "weighted_methods", "jump")),
        ode_method=str(getattr(args, "ode_method", "fixed_euler")),
        without_cat_loss=bool(getattr(args, "without_cat_loss", False)),
        angle_penalty=bool(getattr(args, "angle_penalty", False)),
    )
    dequantizer = UniformDequantizer()
    return flow, nodes_dist, None, dequantizer
