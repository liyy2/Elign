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


# VP-SDE integrated noise schedule: T(t) = integral of beta(s) from 0 to t,
# where beta(s) = beta_min + (beta_max - beta_min)*s (linear schedule).
def T(t: torch.Tensor) -> torch.Tensor:
    beta_min = 0.1
    beta_max = 20.0
    return 0.5 * (beta_max - beta_min) * t**2 + beta_min * t


# Derivative of T(t), i.e. the instantaneous noise rate beta(t).
def T_hat(t: torch.Tensor) -> torch.Tensor:
    beta_min = 0.1
    beta_max = 20.0
    return (beta_max - beta_min) * t + beta_min


class UniformDequantizer(nn.Module):
    """Minimal dequantizer used by the upstream MolFM QM9 sampler.

    During training (forward): adds uniform noise in [-0.5, 0.5] to discrete
    node features (atom types and charges) so they become continuous.
    During sampling (reverse): rounds continuous values back to integers to
    recover discrete atom types and charges.
    """

    def forward(self, tensor, node_mask, edge_mask, context):
        del edge_mask, context
        category, integer = tensor["categorical"], tensor["integer"]
        zeros = torch.zeros(integer.size(0), device=integer.device)

        # Add uniform noise to make discrete features continuous (dequantization)
        out_category = category + torch.rand_like(category) - 0.5
        out_integer = integer + torch.rand_like(integer) - 0.5

        # Zero out padded nodes
        if node_mask is not None:
            out_category = out_category * node_mask
            out_integer = out_integer * node_mask

        out = {"categorical": out_category, "integer": out_integer}
        return out, zeros

    def reverse(self, tensor):
        """Discretize continuous features back to integers (decoding step)."""
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

        self.dynamics = dynamics  # EGNN backbone that predicts the vector field
        self.in_node_nf = int(in_node_nf)  # node feature dim (atom types + charges)
        self.n_dims = int(n_dims)  # spatial dims (3 for 3D molecules)
        self.num_classes = self.in_node_nf - int(self.include_charges)  # atom type count
        self.T = int(timesteps)
        self.parametrization = parametrization
        self.norm_values = tuple(norm_values)  # (x_scale, cat_scale, int_scale)
        self.norm_biases = tuple(norm_biases)  # (x_bias, cat_bias, int_bias)
        self.time_embed = bool(time_embed)
        # Dummy buffer to track device placement
        self.register_buffer("buffer", torch.zeros(1))

    def phi(self, t, x, node_mask, edge_mask, context):
        """Evaluate the learned vector field (EGNN dynamics) at time t."""
        return self.dynamics._forward(t, x, node_mask, edge_mask, context)

    def subspace_dimensionality(self, node_mask):
        """Effective dimensionality of the zero-CoM subspace (N-1)*d per sample."""
        number_of_nodes = torch.sum(node_mask.squeeze(2), dim=1)
        return (number_of_nodes - 1) * self.n_dims

    def normalize(self, x, h, node_mask):
        """Scale positions and features into the normalized space the model operates in.

        Returns (x_norm, h_norm, delta_log_px) where delta_log_px is the
        log-determinant of the Jacobian from the coordinate rescaling.
        """
        cat_bias = 0.0 if self.norm_biases[1] is None else self.norm_biases[1]
        int_bias = 0.0 if self.norm_biases[2] is None else self.norm_biases[2]
        x = x / self.norm_values[0]
        # Log-det correction for the change of variables on positions
        delta_log_px = -self.subspace_dimensionality(node_mask) * np.log(self.norm_values[0])
        h_cat = ((h["categorical"].float() - cat_bias) / self.norm_values[1]) * node_mask
        h_int = (h["integer"].float() - int_bias) / self.norm_values[2]

        if self.include_charges:
            h_int = h_int * node_mask

        h = {"categorical": h_cat, "integer": h_int}
        return x, h, delta_log_px

    def unnormalize(self, x, h_cat, h_int, node_mask):
        """Inverse of normalize: map back from model space to data space."""
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
        """Decode final latent z0 into molecule coordinates and discrete features.

        z0 layout along last dim: [positions (n_dims) | atom types (num_classes) | charges (1)]
        Steps: unnormalize continuous values, then round to recover discrete types/charges.
        """
        x = z0[:, :, : self.n_dims]
        if self.include_charges:
            h_int = z0[:, :, -1:]  # charges are the last channel
        else:
            h_int = z0.new_zeros(z0.size(0), z0.size(1), 0)

        # Map from normalized model space back to data space
        x, h_cat, h_int = self.unnormalize(
            x,
            z0[:, :, self.n_dims : self.n_dims + self.num_classes],
            h_int,
            node_mask,
        )
        # Round continuous features to discrete atom types and charges
        tensor = dequantizer.reverse({"categorical": h_cat, "integer": h_int})
        h = {"integer": tensor["integer"], "categorical": tensor["categorical"]}
        return x, h

    def sample_combined_position_feature_noise(self, n_samples, n_nodes, node_mask):
        """Sample initial noise z_T for the reverse ODE.

        Positions get zero-center-of-mass Gaussian noise (translation invariance),
        while node features get standard Gaussian noise.
        Returns concatenated tensor [z_x | z_h] along the feature dim.
        """
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
        """Compute the ODE drift for one Euler step: dz/dt = flow_drift(t, z, ...).

        This is called at each fixed step during sampling. The raw vector field
        from the EGNN is post-processed based on the discrete_path mode:
          - OT_path:  raw drift (optimal transport), no extra scaling
          - VP_path:  entire drift scaled by VP-SDE score-to-drift conversion
          - HB_path:  only feature channels scaled (hybrid: OT for positions, VP for features)
        """
        dx = self.raw_flow_velocity(t, z, node_mask, edge_mask, context)

        # Apply VP-SDE score-to-drift factor: -0.5 * beta(t) / (1 - exp(-T(t)))
        if self.discrete_path == "VP_path":
            scale = -0.5 * T_hat(t) / (1 - torch.exp(-T(t)) + 1e-5)
            dx = dx * scale.reshape(scale.shape[0], 1, 1)
        elif self.discrete_path == "HB_path":
            # Hybrid: only scale feature channels, leave positions as OT
            scale = -0.5 * T_hat(t) / (1 - torch.exp(-T(t)) + 1e-5)
            dx[:, :, self.n_dims :] = dx[:, :, self.n_dims :] * scale.reshape(scale.shape[0], 1, 1)

        return dx

    def raw_flow_velocity(self, t, z, node_mask, edge_mask, context):
        """Return the pre-path-scaled flow velocity with cat-loss gating applied."""
        dx = self.phi(t, z, node_mask, edge_mask, context)

        # Optional: zero out categorical drift after a certain time threshold,
        # ramping it by 1/cat_loss_step before that point.
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

        return dx


def build_molfm_qm9_components(args, device, dataset_info, dataloader_train=None):
    """Factory: build all components needed for MolFM QM9 sampling.

    Returns (flow, nodes_dist, None, dequantizer) where:
      - flow: MolFMFlowCore model with EGNN dynamics
      - nodes_dist: DistributionNodes for sampling molecule sizes
      - None: placeholder (upstream uses a property distribution here)
      - dequantizer: UniformDequantizer for encoding/decoding discrete features
    """
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

    # If conditioning on time, append a scalar time channel to node features
    condition_time = bool(getattr(args, "condition_time", True))
    if condition_time:
        dynamics_in_node_nf = in_node_nf + 1
    else:
        dynamics_in_node_nf = in_node_nf

    # Build the equivariant GNN backbone that predicts the vector field
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

    # Wrap the dynamics in the flow-matching core with all sampling config
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
