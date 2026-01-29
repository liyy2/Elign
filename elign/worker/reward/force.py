import logging
import os
from typing import Optional, Union

import numpy as np
import torch
from tqdm import tqdm as tq

from .base import BaseReward

os.environ.setdefault("OMP_NUM_THREADS", "18")

def _is_main_process() -> bool:
    """Return True if current worker is considered global rank zero."""
    for key in ("RANK", "WORLD_RANK", "SLURM_PROCID"):
        value = os.environ.get(key)
        if value is not None:
            try:
                return int(value) == 0
            except ValueError:
                return value == "0"
    return True


def _tqdm_enabled() -> bool:
    """Enable tqdm progress bars only when explicitly requested.

    Set `ELIGN_TQDM=1` to turn them on.
    """
    value = os.environ.get("ELIGN_TQDM", "").strip().lower()
    return value in {"1", "true", "yes", "on"}

from edm_source.qm9.analyze import check_stability
from edm_source.qm9 import bond_analyze as qm9_bond_analyze

from edm_source.mlff_modules.mlff_utils import get_mlff_predictor

from elign.protocol import DataProto, TensorDict
from .scheduler import RewardScheduler


logger = logging.getLogger(__name__)

_SYMBOL_TO_Z = {
    "H": 1,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "As": 33,
    "Br": 35,
    "I": 53,
    "Hg": 80,
    "Bi": 83,
}


def _resolve_mlff_device(device_like, default_device):
    """Normalize MLFF device hints to strings understood by the loader."""
    if device_like is None:
        candidate = default_device
    else:
        candidate = device_like
    if isinstance(candidate, torch.device):
        if candidate.type == "cuda":
            if candidate.index is not None:
                return f"cuda:{candidate.index}"
            return "cuda"
        if candidate.type == "cpu":
            return "cpu"
    else:
        candidate_str = str(candidate).lower()
        if candidate_str == "cuda":
            return "cuda"
        if candidate_str.startswith("cuda:"):
            try:
                index = int(candidate_str.split(":", 1)[1])
            except ValueError as exc:
                raise ValueError(f"Invalid CUDA device string: {candidate}") from exc
            return f"cuda:{index}"
        if candidate_str == "cpu":
            return "cpu"
    raise ValueError(f"Unsupported MLFF device specification: {device_like}")

class UMAForceReward(BaseReward):
    """Reward module for post-training that scores molecules using UMA MLFF forces (+ optional energies).

    The reward has three pieces:

    1) Force term (always): encourages locally relaxed structures.
       - Compute per-atom forces via UMA and aggregate to a scalar magnitude per molecule
         (RMS or max). Reward is the negative magnitude.

    2) Energy term (optional): encourages low energy *only* when the structure is physically reasonable.
       - Energy is transformed as: `(E + offset) / scale` with optional clipping.
       - Reward is the negative transformed energy (i.e. minimize energy).
       - Energy can be gated to stable terminal states to avoid out-of-distribution energy exploits.

    3) Stability bonus (optional): adds +w for stable molecules and -w for unstable molecules.
       - This is molecule-level on purpose: partially-invalid molecules should not get a positive bonus.

    If reward shaping is enabled and latents are provided, we also compute potential-based shaping deltas
    across a scheduled subset of diffusion steps; PPO then uses the per-timestep reward trace (`rewards_ts`).
    """

    def __init__(
        self,
        dataset_info: dict,
        condition: bool = False,
        mlff_model: str = "uma-s-1p1",
        mlff_predictor: Optional[object] = None,
        force_computer: Optional[object] = None,
        position_scale: Optional[float] = None,
        force_clip_threshold: Optional[float] = None,
        device: Optional[Union[str, torch.device]] = None,
        mlff_device: Optional[Union[str, torch.device]] = None,
        shaping: Optional[dict] = None,
        use_energy: bool = False,
        energy_only_if_stable: bool = False,
        force_only_if_stable: bool = False,
        force_weight: float = 1.0,
        energy_weight: float = 1.0,
        stability_weight: float = 0.0,
        atom_stability_weight: float = 0.0,
        valence_underbond_weight: float = 0.0,
        valence_overbond_weight: float = 0.0,
        valence_underbond_soft_weight: float = 0.0,
        valence_overbond_soft_weight: float = 0.0,
        valence_soft_temperature: float = 0.02,
        force_aggregation: str = "rms",
        energy_transform_offset: float = 10000.0,
        energy_transform_scale: float = 1000.0,
        energy_transform_clip: Optional[float] = None,
        energy_normalize_by_atoms: bool = False,
        energy_atom_refs: Optional[str] = None,
    ):
        super().__init__()
        self.is_main_process = _is_main_process()
        self.dataset_info = dataset_info
        self.condition = condition

        if device is None:
            # Prefer an explicit CUDA index so torch.cuda.set_device works reliably.
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, torch.device):
            if device.type == "cuda" and device.index is None:
                self.device = torch.device("cuda:0")
            else:
                self.device = device
        else:
            device_str = str(device)
            if device_str.lower() == "cuda":
                device_str = "cuda:0"
            self.device = torch.device(device_str)
        self.mlff_device = _resolve_mlff_device(mlff_device, "cuda" if self.device.type == "cuda" else "cpu")
        if position_scale is None:
            norm_values = self.dataset_info.get("normalize_factors")
            if isinstance(norm_values, (list, tuple)) and len(norm_values) > 0:
                position_scale = float(norm_values[0])
            else:
                position_scale = 1.0
        self.position_scale = position_scale
        self.force_clip_threshold = None if force_clip_threshold is None else float(force_clip_threshold)
        
        # Energy reward configuration
        self.use_energy = use_energy
        self.energy_only_if_stable = bool(energy_only_if_stable)
        # Optional gate: only grant the *force* objective to stable molecules.
        # This prevents a common exploit where an unstable structure has very low predicted forces
        # (e.g., stretched/missing bonds) and therefore receives a high force reward.
        self.force_only_if_stable = bool(force_only_if_stable)
        self.force_weight = force_weight
        self.energy_weight = energy_weight
        self.stability_weight = stability_weight
        self.atom_stability_weight = float(atom_stability_weight or 0.0)
        # Optional: penalize missing valence bonds (under-bonded atoms).
        # This targets the common \"RDKit-valid but QM9-unstable\" failure mode where
        # RDKit fills missing bonds with implicit H, but `check_stability` expects explicit valence.
        self.valence_underbond_weight = float(valence_underbond_weight or 0.0)
        # Optional: penalize over-bonded atoms (too many inferred bond orders for their valence).
        # This complements the under-bond penalty: over-bonded atoms often show up as RDKit
        # sanitization errors (e.g., carbon with valence 5).
        self.valence_overbond_weight = float(valence_overbond_weight or 0.0)
        # Optional: smooth (sigmoid) valence penalty that provides a *continuous* signal when
        # atoms are close to the bond-length thresholds but haven't crossed them yet. This is
        # especially helpful for the common QM9 failure mode where a single H is slightly too far
        # away to count as bonded, making the discrete under-bond penalty very sparse.
        self.valence_underbond_soft_weight = float(valence_underbond_soft_weight or 0.0)
        self.valence_overbond_soft_weight = float(valence_overbond_soft_weight or 0.0)
        self.valence_soft_temperature = float(valence_soft_temperature)
        if self.valence_soft_temperature <= 0.0:
            raise ValueError("valence_soft_temperature must be > 0")
        force_aggregation = (force_aggregation or "rms").lower()
        if force_aggregation not in {"rms", "max"}:
            raise ValueError(f"Unsupported force aggregation '{force_aggregation}'. Use 'rms' or 'max'.")
        self.force_aggregation = force_aggregation
        self.energy_transform_offset = energy_transform_offset
        self.energy_transform_scale = energy_transform_scale
        self.energy_transform_clip = None if energy_transform_clip is None else float(energy_transform_clip)
        self.energy_normalize_by_atoms = bool(energy_normalize_by_atoms)
        self.energy_atom_refs = str(energy_atom_refs) if energy_atom_refs else None
        self._atom_ref_by_type = None


        # Reward shaping config
        # Scheduler allows switching between uniform and adaptive sampling of diffusion steps.
        self.shaping_cfg = shaping or {}
        self.shaping_enabled = bool(self.shaping_cfg.get("enabled", False))
        self.shaping_gamma = float(self.shaping_cfg.get("gamma", 1.0))
        shaping_mode = str(self.shaping_cfg.get("mode", "delta") or "delta").lower()
        if shaping_mode in {"pbrs", "return_to_go", "pbrs_return_to_go", "paper"}:
            shaping_mode = "pbrs_return_to_go"
        if shaping_mode not in {"delta", "pbrs_return_to_go"}:
            raise ValueError(
                f"Unsupported shaping.mode '{self.shaping_cfg.get('mode')}'. "
                "Use 'delta' (legacy) or 'pbrs_return_to_go' (paper Eq. 4)."
            )
        self.shaping_mode = shaping_mode
        default_skip_prefix = int(self.shaping_cfg.get("skip_prefix", 0))
        # Option to shape only with energy while keeping force+energy terminals intact.
        self.shaping_energy_only = bool(self.shaping_cfg.get("only_energy_reshape", False))
        # Control number of diffusion steps per MLFF evaluation batch (0 => no chunking)
        self.mlff_batch_size = int(self.shaping_cfg.get("mlff_batch_size", 0) or 32)

        scheduler_cfg = {}
        if isinstance(self.shaping_cfg, dict):
            scheduler_cfg = self.shaping_cfg.get("scheduler", {}) or {}
        if not isinstance(scheduler_cfg, dict):
            scheduler_cfg = {}

        scheduler_mode = scheduler_cfg.get("mode", self.shaping_cfg.get("schedule_mode", "uniform"))
        scheduler_skip = int(scheduler_cfg.get("skip_prefix", default_skip_prefix))
        uniform_stride = scheduler_cfg.get("uniform_stride", scheduler_cfg.get("stride", self.shaping_cfg.get("stride", 1)))
        if uniform_stride is None:
            uniform_stride = 1
        uniform_stride = int(max(1, uniform_stride))
        include_terminal = bool(scheduler_cfg.get("include_terminal", True))
        adaptive_cfg = scheduler_cfg.get("adaptive", self.shaping_cfg.get("adaptive", {})) or {}

        self.reward_scheduler = RewardScheduler(
            mode=scheduler_mode,
            skip_prefix=scheduler_skip,
            uniform_stride=uniform_stride,
            adaptive_config=adaptive_cfg,
            include_terminal=include_terminal,
        )
        self.shaping_skip_prefix = self.reward_scheduler.skip_prefix

        # `terminal_weight` exists to re-emphasize the terminal reward relative to shaping deltas.
        # When shaping is disabled (terminal-only reward), keep this at 1.0 to avoid confusion
        # and unintended reward rescaling.
        if self.shaping_enabled:
            self.terminal_reward_weight = max(0.0, float(self.shaping_cfg.get("terminal_weight", 1.5)))
        else:
            self.terminal_reward_weight = 1.0


        if force_computer is not None:
            self.force_computer = force_computer
            self.mlff_predictor = getattr(force_computer, "mlff_predictor", mlff_predictor)
        else:
            if mlff_predictor is not None:
                self.mlff_predictor = mlff_predictor
            else:
                self.mlff_predictor = get_mlff_predictor(mlff_model, self.mlff_device)

            if self.mlff_predictor is not None:
                try:
                    from edm_source.mlff_modules.mlff_force_computer import MLFFForceComputer
                except ModuleNotFoundError as exc:
                    raise ModuleNotFoundError(
                        "UMAForceReward: MLFF force computation requires optional dependencies "
                        "(`fairchem` + `ase`). Install them, pass a custom `force_computer`, "
                        "or set `reward.type=dummy`."
                    ) from exc
                self.force_computer = MLFFForceComputer(
                    mlff_predictor=self.mlff_predictor,
                    position_scale=self.position_scale,
                    device=self.device,
                    compute_energy=self.use_energy,  # Enable energy computation based on config
                )
            else:
                self.force_computer = None
                logger.warning("UMAForceReward: MLFF predictor not loaded, rewards will default to zero.")

        if self.use_energy and self.energy_atom_refs:
            self._atom_ref_by_type = self._build_atom_ref_by_type(self.energy_atom_refs)

    def _build_atom_ref_by_type(self, ref_key: str) -> torch.Tensor:
        """Build a lookup table mapping dataset atom-type indices -> reference energies.

        The UMA predictor exposes per-element reference energies (used to compute formation energies)
        under `predictor.atom_refs[ref_key][Z][charge]`.
        """
        predictor = getattr(self, "mlff_predictor", None)
        if predictor is None:
            raise RuntimeError("energy_atom_refs requires an MLFF predictor, but none is loaded.")
        atom_refs = getattr(predictor, "atom_refs", None)
        if atom_refs is None:
            raise RuntimeError("MLFF predictor is missing atom_refs; cannot use energy_atom_refs.")
        try:
            refs_for_key = atom_refs.get(ref_key)
        except Exception:
            refs_for_key = None
        if refs_for_key is None:
            raise RuntimeError(f"MLFF predictor is missing atom_refs['{ref_key}']; cannot use energy_atom_refs.")

        decoder = self.dataset_info.get("atom_decoder")
        if not decoder:
            raise RuntimeError("dataset_info is missing atom_decoder; required for energy_atom_refs.")

        ref_values = []
        missing_symbols = []
        for symbol in decoder:
            znum = _SYMBOL_TO_Z.get(symbol)
            if znum is None:
                missing_symbols.append(symbol)
                znum = 1
            try:
                entry = refs_for_key.get(int(znum))
            except Exception:
                entry = None
            if entry is None:
                raise RuntimeError(f"MLFF atom_refs['{ref_key}'] missing atomic number {znum} (symbol={symbol}).")

            energy0 = None
            if hasattr(entry, "get"):
                energy0 = entry.get(0)
                if energy0 is None:
                    energy0 = entry.get("0")
            if energy0 is None:
                try:
                    energy0 = entry[0]
                except Exception as exc:
                    raise RuntimeError(
                        f"Invalid MLFF atom_ref entry for Z={znum} (symbol={symbol}): {entry}"
                    ) from exc
            ref_values.append(float(energy0))

        if missing_symbols and self.is_main_process:
            logger.warning(
                "UMAForceReward: missing atomic-number mapping for symbols %s; defaulting to H reference.",
                sorted(set(missing_symbols)),
            )

        return torch.tensor(ref_values, dtype=torch.float32, device=self.device)

    def process_data(self, samples: DataProto) -> dict:
        positions = samples.batch["x"].detach()
        categorical = samples.batch["categorical"].detach()
        nodesxsample = samples.batch["nodesxsample"].long().to(positions.device)
        batch_size, max_n_nodes, _ = positions.shape

        node_indices = torch.arange(max_n_nodes, device=positions.device).unsqueeze(0)
        node_mask = (node_indices < nodesxsample.unsqueeze(1)).unsqueeze(-1).float()

        z = torch.cat([positions, categorical], dim=-1)

        processed = {
            "z": z,
            "node_mask": node_mask,
            "positions": positions,
            "categorical": categorical,
            "nodesxsample": nodesxsample,
            "batch_size": batch_size,
        }

        if self.condition and "context" in samples.batch:
            processed["context"] = samples.batch["context"].detach()

        return processed

    def _aggregate_force_metric(self, forces: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        """
        Reduce per-atom force vectors to a single scalar per sample according to the configured aggregation.
        """
        if forces.dim() == 2:
            forces = forces.unsqueeze(0)
        if node_mask.dim() == 1:
            node_mask = node_mask.unsqueeze(0)

        # Optional force clipping.
        # This is mainly a "don't over-penalize tail outliers" knob that can help exploration/diversity.
        if self.force_clip_threshold is not None:
            magnitudes = torch.norm(forces, dim=-1, keepdim=True)
            scale = torch.clamp(self.force_clip_threshold / (magnitudes + 1e-12), max=1.0)
            forces = forces * scale

        if node_mask.dim() == 3:
            mask = node_mask[..., 0] > 0.5
        else:
            mask = node_mask > 0.5

        force_norms = torch.norm(forces, dim=-1)
        mask_float = mask.float()
        valid_counts = mask_float.sum(dim=-1)

        if self.force_aggregation == "rms":
            denom = valid_counts.clamp(min=1.0)
            squared = (force_norms.pow(2) * mask_float)
            aggregated = torch.sqrt(squared.sum(dim=-1) / denom)
        else:  # self.force_aggregation == "max"
            masked_norms = force_norms * mask_float
            aggregated = masked_norms.max(dim=-1).values

        aggregated = torch.where(valid_counts > 0, aggregated, torch.zeros_like(aggregated))
        return aggregated.view(forces.shape[0])

    def calculate_rewards(self, data: DataProto) -> DataProto:
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        meta_info = data.meta_info.copy()
        processed = self.process_data(data)

        z = processed["z"].to(self.device)
        node_mask = processed["node_mask"].to(self.device)
        positions = processed["positions"].to(self.device)
        categorical = processed["categorical"].to(self.device)
        batch_size = processed["batch_size"]

        # Decide shaping path: if enabled, compute terminal + shaped in one MLFF pass over latents
        shaping_active = self.shaping_enabled and ("latents" in data.batch.keys()) and (self.force_computer is not None)

        atom_ref_sums = None
        if self.use_energy and self._atom_ref_by_type is not None:
            atom_type_indices = categorical.argmax(dim=-1)
            valid_mask = node_mask[..., 0] > 0.5
            ref_by_type = self._atom_ref_by_type.to(device=categorical.device, dtype=torch.float32)
            ref_per_atom = ref_by_type[atom_type_indices]
            atom_ref_sums = (ref_per_atom * valid_mask.to(dtype=ref_per_atom.dtype)).sum(dim=1)

        force_rewards = torch.zeros(batch_size, device=self.device)
        energy_rewards = torch.zeros(batch_size, device=self.device)
        stability_flags = torch.zeros(batch_size, device=self.device)
        atom_stability = torch.zeros(batch_size, device=self.device)
        valence_underbond = torch.zeros(batch_size, device=self.device)
        valence_overbond = torch.zeros(batch_size, device=self.device)
        valence_underbond_soft = torch.zeros(batch_size, device=self.device)
        valence_overbond_soft = torch.zeros(batch_size, device=self.device)
        result = {}
        rewards_ts = None
        force_rewards_ts = None
        energy_rewards_ts = None
        terminal_force_metric = None
        terminal_energy_metric = None
        last_idx = None


        if shaping_active:
            if "z0_preds" in data.batch.keys():
                latents = data.batch["z0_preds"]
            else:
                latents = data.batch["latents"]

            if "timesteps" in data.batch.keys():
                T_steps = data.batch["timesteps"].shape[1]
                latents = latents[:, :T_steps]
            else:
                T_steps = latents.shape[1]

            B, S, N, D = latents.shape
            # B corresponds to the number of prompts/molecules in the current rollout batch.
            # It matches the sampler micro-batch size (after any chunking of the dataloader group).
            node_mask_b = processed["node_mask"].to(self.device)
            F_force = torch.zeros((B, S), device=self.device)
            F_energy = torch.zeros((B, S), device=self.device) if self.use_energy else None

            if "timesteps" in data.batch.keys():
                timesteps_schedule = data.batch["timesteps"][0, :S].detach().cpu()
            else:
                timesteps_schedule = torch.linspace(float(S - 1), 0.0, steps=S)

            # Select the diffusion steps that will receive MLFF calls and potential shaping.
            schedule = self.reward_scheduler.build(
                timesteps=timesteps_schedule,
                end_idx=S - 1,
            )
            schedule_indices = schedule.indices.to(latents.device)
            interval_steps = schedule.intervals.to(latents.device)
            meta_info["reward_schedule_indices"] = schedule_indices.detach().cpu().tolist()
            alignment_active = bool(data.meta_info.get("force_alignment_enabled", False))
            if alignment_active:
                fine_mask_selected = schedule.fine_mask.to(latents.device)
            else:
                fine_mask_selected = None
            selected_count = int(schedule_indices.numel())

            if selected_count == 0:
                schedule_indices = torch.tensor([S - 1], device=latents.device)
                interval_steps = torch.ones(1, device=latents.device, dtype=torch.long)
                if alignment_active:
                    fine_mask_selected = torch.ones(1, device=latents.device, dtype=torch.bool)
                selected_count = 1
                if self.is_main_process:
                    print('Warning: Selected count Zero')

            if alignment_active and fine_mask_selected is not None:
                fine_mask_selected = fine_mask_selected.float()
            latents_selected = torch.index_select(latents, 1, schedule_indices).contiguous()
            flat_total = B * selected_count
            # selected_count is the number of diffusion steps passing through MLFF;
            # the total samples evaluated is prompts (B) times those scheduled steps.
            if self.mlff_batch_size > 0:
                chunk_size = max(1, min(self.mlff_batch_size, flat_total))
            else:
                chunk_size = flat_total

            latents_flat = latents_selected.view(flat_total, N, D)
            node_mask_flat_all = node_mask_b.repeat_interleave(selected_count, dim=0)

            flat_force_metric = torch.zeros(flat_total, device=latents.device)
            flat_energy = torch.zeros(flat_total, device=latents.device) if self.use_energy else None
            if alignment_active:
                force_vectors_flat = torch.zeros(
                    (flat_total, N, 3), device=latents.device, dtype=latents.dtype
                )
            else:
                force_vectors_flat = None

            cur = 0
            progress_bar = tq(
                total=flat_total,
                desc="UMAForce MLFF",
                leave=False,
                disable=flat_total <= 1,
                dynamic_ncols=True,
                smoothing=0,
            )
            try:
                while cur < flat_total:
                    end_flat = min(flat_total, cur + chunk_size)
                    z_flat = latents_flat[cur:end_flat]
                    node_mask_flat = node_mask_flat_all[cur:end_flat]

                    if self.use_energy:
                        forces_flat, energies_flat = self.force_computer.compute_mlff_forces(
                            z_flat, node_mask_flat, self.dataset_info
                        )
                    else:
                        forces_flat = self.force_computer.compute_mlff_forces(
                            z_flat, node_mask_flat, self.dataset_info
                        )

                    metrics_flat = self._aggregate_force_metric(forces_flat, node_mask_flat)
                    flat_force_metric[cur:end_flat] = metrics_flat

                    if alignment_active and force_vectors_flat is not None:
                        force_vectors_flat[cur:end_flat] = forces_flat

                    if self.use_energy and flat_energy is not None:
                        flat_energy[cur:end_flat] = energies_flat.view(-1)

                    processed_count = end_flat - cur
                    cur = end_flat
                    if progress_bar is not None:
                        progress_bar.update(processed_count)
                        progress_bar.refresh()
            finally:
                if progress_bar is not None:
                    progress_bar.close()

            force_metric_bt = flat_force_metric.view(B, selected_count)
            F_force.index_copy_(1, schedule_indices, -force_metric_bt)

            if self.use_energy and flat_energy is not None:
                energies_bt = flat_energy.view(B, selected_count)
                if atom_ref_sums is not None:
                    energies_bt = energies_bt - atom_ref_sums.to(device=energies_bt.device, dtype=energies_bt.dtype).unsqueeze(1)
                if self.energy_normalize_by_atoms:
                    atom_counts = (node_mask_b[..., 0] > 0.5).sum(dim=1).clamp(min=1).to(energies_bt.dtype)
                    energies_bt = energies_bt / atom_counts.unsqueeze(1)
                transformed_energies = (energies_bt + self.energy_transform_offset) / self.energy_transform_scale
                if self.energy_transform_clip is not None:
                    transformed_energies = transformed_energies.clamp(
                        min=-self.energy_transform_clip, max=self.energy_transform_clip
                    )
                # Energy reward always minimizes energy (negative sign).
                F_energy.index_copy_(1, schedule_indices, -transformed_energies)

            if alignment_active and force_vectors_flat is not None:
                force_vectors_selected = force_vectors_flat.view(B, selected_count, N, 3)
            else:
                force_vectors_selected = None

            selected_count = int(schedule_indices.numel())
            if selected_count > 0:
                last_idx = schedule_indices[-1]
                terminal_force_metric = F_force[:, last_idx]
                if self.use_energy and F_energy is not None:
                    terminal_energy_metric = F_energy[:, last_idx]

            # Terminal rewards are always computed from the terminal z0 prediction (decode frame).
            # In PBRS mode, intermediate *energy potentials* are returned separately and the trainer
            # computes the return-to-go per Eq. (4) in the paper.
            if terminal_force_metric is not None:
                force_rewards = terminal_force_metric
            else:
                force_rewards = torch.zeros(batch_size, device=self.device)
            if self.use_energy and terminal_energy_metric is not None:
                energy_rewards = terminal_energy_metric
            else:
                energy_rewards = torch.zeros_like(force_rewards)

            if self.shaping_mode == "delta":
                shaped_force = torch.zeros_like(F_force)
                shaped_energy = torch.zeros_like(F_force)
                # Potential shaping on a scheduled subset of diffusion steps:
                #   delta(t_k) = gamma^Δt * F(t_{k+1}) - F(t_k)
                # We store these deltas in the per-timestep traces and separately overwrite the
                # terminal column with the terminal reward so PPO sees it exactly once.
                if selected_count > 1:
                    idx_current = schedule_indices[:-1]
                    idx_next = schedule_indices[1:]
                    intervals = interval_steps[:-1].clamp(min=1)
                    intervals_f = intervals.to(F_force.dtype)
                    gamma_factor = torch.pow(
                        torch.full_like(intervals_f, self.shaping_gamma),
                        intervals_f,
                    )

                    if not self.shaping_energy_only:
                        force_current = torch.index_select(F_force, 1, idx_current)
                        force_next = torch.index_select(F_force, 1, idx_next)
                        shaped_vals = gamma_factor.unsqueeze(0) * force_next - force_current
                        shaped_force.index_copy_(1, idx_current, shaped_vals)

                    if self.use_energy and F_energy is not None:
                        energy_current = torch.index_select(F_energy, 1, idx_current)
                        energy_next = torch.index_select(F_energy, 1, idx_next)
                        shaped_energy_vals = gamma_factor.unsqueeze(0) * energy_next - energy_current
                        shaped_energy.index_copy_(1, idx_current, shaped_energy_vals)

                weighted_shaped_force = self.force_weight * shaped_force
                weighted_shaped_energy = (
                    self.energy_weight * shaped_energy if self.use_energy else torch.zeros_like(shaped_force)
                )

                if last_idx is not None:
                    if terminal_force_metric is not None:
                        terminal_force_weighted = (
                            self.force_weight * self.terminal_reward_weight * terminal_force_metric
                        )
                    else:
                        terminal_force_weighted = None
                    if self.use_energy and terminal_energy_metric is not None:
                        terminal_energy_weighted = (
                            self.energy_weight * self.terminal_reward_weight * terminal_energy_metric
                        )
                    else:
                        terminal_energy_weighted = None

                rewards_ts = weighted_shaped_force + weighted_shaped_energy
                force_rewards_ts = shaped_force.clone()
                if self.use_energy:
                    energy_rewards_ts = shaped_energy.clone()

                if last_idx is not None:
                    if terminal_force_metric is not None:
                        rewards_ts[:, last_idx] = terminal_force_weighted
                        force_rewards_ts[:, last_idx] = self.terminal_reward_weight * terminal_force_metric
                    if terminal_energy_metric is not None and energy_rewards_ts is not None:
                        rewards_ts[:, last_idx] = rewards_ts[:, last_idx] + terminal_energy_weighted
                        energy_rewards_ts[:, last_idx] = self.terminal_reward_weight * terminal_energy_metric
            else:
                # Paper-aligned PBRS mode:
                # - Expose energy potentials Ψ_t = -E_ϕ(ẑ_{0|t}) over the scheduled indices (plus Ψ_0 at terminal).
                # - Keep force reward terminal-only (no force PBRS).
                #
                # The trainer converts these into per-step returns-to-go:
                #   G_t^(E) = γ^t Ψ_0 - Ψ_t   (Eq. 4 in the paper)
                # and broadcasts the terminal force advantage across diffusion steps (Alg. 1).
                rewards_ts = None
                if last_idx is not None:
                    force_rewards_ts = torch.zeros_like(F_force)
                    force_rewards_ts[:, last_idx] = terminal_force_metric
                else:
                    force_rewards_ts = None
                energy_rewards_ts = F_energy if self.use_energy else None

            if alignment_active and force_vectors_selected is not None and fine_mask_selected is not None:
                schedule_indices_cpu = (
                    schedule_indices.detach().cpu().unsqueeze(0).expand(B, -1).clone()
                )
                fine_mask_cpu = (
                    fine_mask_selected.detach().cpu().unsqueeze(0).expand(B, -1).clone()
                )
                result["force_vectors_schedule"] = force_vectors_selected.detach().cpu()
                result["force_schedule_indices"] = schedule_indices_cpu
                result["force_fine_mask"] = fine_mask_cpu
        else:
            # Final-state path only (single MLFF call)
            if self.force_computer is not None:
                # Chunk terminal MLFF evaluation to avoid OOM on large GEOM batches.
                total = int(z.shape[0])
                chunk = total
                if self.mlff_batch_size > 0:
                    chunk = max(1, min(int(self.mlff_batch_size), total))

                forces = torch.zeros((total, z.shape[1], 3), device=self.device, dtype=z.dtype)
                energies = torch.zeros(total, device=self.device, dtype=z.dtype)
                for start in range(0, total, chunk):
                    end = min(total, start + chunk)
                    z_chunk = z[start:end]
                    node_chunk = node_mask[start:end]
                    if self.use_energy:
                        forces_chunk, energies_chunk = self.force_computer.compute_mlff_forces(
                            z_chunk, node_chunk, self.dataset_info
                        )
                        forces[start:end] = forces_chunk
                        energies[start:end] = energies_chunk.view(-1)
                    else:
                        forces_chunk = self.force_computer.compute_mlff_forces(
                            z_chunk, node_chunk, self.dataset_info
                        )
                        forces[start:end] = forces_chunk
            else:
                forces = torch.zeros_like(z[:, :, :3], device=self.device)
                energies = torch.zeros(z.shape[0], device=self.device)

            tqdm_enabled = _tqdm_enabled()
            for batch_idx in tq(
                range(batch_size),
                desc="UMAForce terminal reward",
                leave=False,
                disable=(batch_size <= 1) or (not self.is_main_process) or (not tqdm_enabled),
                dynamic_ncols=True,
            ):
                valid_mask = node_mask[batch_idx, :, 0] > 0
                if not torch.any(valid_mask):
                    continue
                aggregated_force = self._aggregate_force_metric(
                    forces[batch_idx : batch_idx + 1],
                    node_mask[batch_idx : batch_idx + 1],
                ).squeeze(0)
                force_rewards[batch_idx] = -aggregated_force
                if self.use_energy:
                    energy = energies[batch_idx]
                    if atom_ref_sums is not None:
                        energy = energy - atom_ref_sums[batch_idx].to(dtype=energy.dtype)
                    if self.energy_normalize_by_atoms:
                        num_atoms = valid_mask.sum().clamp(min=1).to(dtype=energy.dtype)
                        energy = energy / num_atoms
                    transformed_energy = (energy + self.energy_transform_offset) / self.energy_transform_scale
                    if self.energy_transform_clip is not None:
                        transformed_energy = transformed_energy.clamp(
                            min=-self.energy_transform_clip, max=self.energy_transform_clip
                        )
                    # Energy reward always minimizes energy (negative sign).
                    energy_rewards[batch_idx] = -transformed_energy

        # Compute stability flags once
        tqdm_enabled = _tqdm_enabled()
        for batch_idx in tq(
            range(batch_size),
            desc="UMAForce stability",
            leave=False,
            disable=(batch_size <= 1) or (not self.is_main_process) or (not tqdm_enabled),
            dynamic_ncols=True,
        ):
            valid_mask = node_mask[batch_idx, :, 0] > 0
            if not torch.any(valid_mask):
                continue
            atom_type_indices = categorical[batch_idx].argmax(dim=1)[valid_mask]
            positions_valid = positions[batch_idx][valid_mask]
            try:
                validity_results = check_stability(
                    positions_valid.detach().cpu().numpy(),
                    atom_type_indices.detach().cpu().tolist(),
                    self.dataset_info,
                )
                is_stable = float(validity_results[0])
                stability_flags[batch_idx] = is_stable
                if isinstance(validity_results, (list, tuple)) and len(validity_results) >= 3:
                    stable_atoms = float(validity_results[1])
                    total_atoms = float(validity_results[2])
                    if total_atoms > 0:
                        atom_stability[batch_idx] = stable_atoms / total_atoms
                    else:
                        atom_stability[batch_idx] = is_stable
                else:
                    atom_stability[batch_idx] = is_stable
            except Exception:
                stability_flags[batch_idx] = 0.0
                atom_stability[batch_idx] = 0.0

            # Under-bonded valence penalty (optional).
            #
            # Why this exists:
            # - RDKit validity allows radicals by assigning implicit H when a heavy atom has fewer
            #   explicit bonds (e.g., C with 3 instead of 4). `check_stability` marks these as
            #   unstable because it expects explicit valence for all atoms in the sample.
            # - A per-atom graded penalty on *missing* bond order gives a stronger, more specific
            #   signal than a binary stable/unstable flag.
            if self.valence_underbond_weight != 0.0 or self.valence_overbond_weight != 0.0:
                try:
                    atom_types_list = atom_type_indices.detach().cpu().tolist()
                    pos_np = positions_valid.detach().cpu().numpy()
                    atom_decoder = self.dataset_info["atom_decoder"]
                    n_atoms = int(pos_np.shape[0])
                    nr_bonds = np.zeros(n_atoms, dtype="int")

                    for i in range(n_atoms):
                        for j in range(i + 1, n_atoms):
                            dist = float(np.linalg.norm(pos_np[i] - pos_np[j]))
                            atom1 = atom_decoder[atom_types_list[i]]
                            atom2 = atom_decoder[atom_types_list[j]]
                            if self.dataset_info["name"].startswith("qm9"):
                                order = qm9_bond_analyze.get_bond_order(atom1, atom2, dist)
                            elif self.dataset_info["name"] == "geom":
                                pair = sorted((atom_types_list[i], atom_types_list[j]))
                                order = qm9_bond_analyze.geom_predictor(
                                    (atom_decoder[pair[0]], atom_decoder[pair[1]]), dist
                                )
                            else:
                                order = 0
                            nr_bonds[i] += order
                            nr_bonds[j] += order

                    missing_bonds = 0
                    excess_bonds = 0
                    for atom_type_i, bonds_i in zip(atom_types_list, nr_bonds):
                        symbol = atom_decoder[atom_type_i]
                        allowed = qm9_bond_analyze.allowed_bonds.get(symbol)
                        if allowed is None:
                            continue
                        if isinstance(allowed, int):
                            missing_bonds += max(int(allowed) - int(bonds_i), 0)
                            excess_bonds += max(int(bonds_i) - int(allowed), 0)
                        else:
                            if bonds_i in allowed:
                                continue
                            allowed_sorted = sorted(int(x) for x in allowed)
                            lower = max((v for v in allowed_sorted if v < bonds_i), default=None)
                            upper = min((v for v in allowed_sorted if v > bonds_i), default=None)
                            if lower is None and upper is not None:
                                missing_bonds += max(int(upper) - int(bonds_i), 0)
                            elif upper is None and lower is not None:
                                excess_bonds += max(int(bonds_i) - int(lower), 0)
                            elif lower is not None and upper is not None:
                                # Between two allowed values (e.g., N can be 3 or 5).
                                #
                                # We want a graded penalty that nudges toward the *easier* fix.
                                # In particular, N=4 is common in unstable samples and can be
                                # fixed either by removing one bond (->3) or adding one (->5).
                                # Choose the direction that yields the smaller *weighted* penalty
                                # given the configured under/over weights.
                                missing_delta = int(upper) - int(bonds_i)  # >0
                                excess_delta = int(bonds_i) - int(lower)  # >0
                                under_w = abs(float(self.valence_underbond_weight))
                                over_w = abs(float(self.valence_overbond_weight))
                                if under_w == 0.0:
                                    excess_bonds += excess_delta
                                elif over_w == 0.0:
                                    missing_bonds += missing_delta
                                elif (excess_delta * over_w) <= (missing_delta * under_w):
                                    excess_bonds += excess_delta
                                else:
                                    missing_bonds += missing_delta

                    valence_underbond[batch_idx] = float(missing_bonds)
                    valence_overbond[batch_idx] = float(excess_bonds)
                except Exception:
                    valence_underbond[batch_idx] = 0.0
                    valence_overbond[batch_idx] = 0.0

            # Smooth valence penalty (optional).
            # Uses the same bond-length tables as `get_bond_order`, but replaces the hard thresholds
            # with sigmoids. This provides a graded signal even when a pair is just outside the
            # cutoff that would otherwise flip bond order from 1->0.
            if (
                (self.valence_underbond_soft_weight != 0.0 or self.valence_overbond_soft_weight != 0.0)
                and str(self.dataset_info.get("name", "")).startswith("qm9")
            ):
                try:
                    atom_types_list = atom_type_indices.detach().cpu().tolist()
                    pos_np = positions_valid.detach().cpu().numpy()
                    atom_decoder = self.dataset_info["atom_decoder"]
                    n_atoms = int(pos_np.shape[0])
                    soft_valence = np.zeros(n_atoms, dtype=np.float32)

                    temp = float(self.valence_soft_temperature)
                    bonds1 = qm9_bond_analyze.bonds1
                    bonds2 = qm9_bond_analyze.bonds2
                    bonds3 = qm9_bond_analyze.bonds3
                    thr1_margin = float(qm9_bond_analyze.margin1) / 100.0
                    thr2_margin = float(qm9_bond_analyze.margin2) / 100.0
                    thr3_margin = float(qm9_bond_analyze.margin3) / 100.0

                    def sigmoid(x: float) -> float:
                        x = float(np.clip(x, -60.0, 60.0))
                        return 1.0 / (1.0 + float(np.exp(-x)))

                    for i in range(n_atoms):
                        atom1 = atom_decoder[atom_types_list[i]]
                        for j in range(i + 1, n_atoms):
                            atom2 = atom_decoder[atom_types_list[j]]
                            if atom1 not in bonds1 or atom2 not in bonds1[atom1]:
                                continue
                            dist = float(np.linalg.norm(pos_np[i] - pos_np[j]))
                            thr1 = float(bonds1[atom1][atom2]) / 100.0 + thr1_margin
                            order = sigmoid((thr1 - dist) / temp)
                            if atom1 in bonds2 and atom2 in bonds2[atom1]:
                                thr2 = float(bonds2[atom1][atom2]) / 100.0 + thr2_margin
                                order += sigmoid((thr2 - dist) / temp)
                                if atom1 in bonds3 and atom2 in bonds3[atom1]:
                                    thr3 = float(bonds3[atom1][atom2]) / 100.0 + thr3_margin
                                    order += sigmoid((thr3 - dist) / temp)
                            soft_valence[i] += order
                            soft_valence[j] += order

                    soft_missing = 0.0
                    soft_excess = 0.0
                    for atom_type_i, bonds_i in zip(atom_types_list, soft_valence):
                        symbol = atom_decoder[atom_type_i]
                        allowed = qm9_bond_analyze.allowed_bonds.get(symbol)
                        if allowed is None:
                            continue
                        if isinstance(allowed, int):
                            soft_missing += max(float(allowed) - float(bonds_i), 0.0)
                            soft_excess += max(float(bonds_i) - float(allowed), 0.0)
                        else:
                            allowed_sorted = sorted(float(x) for x in allowed)
                            best_cost = None
                            best_missing = 0.0
                            best_excess = 0.0
                            for candidate in allowed_sorted:
                                missing = max(candidate - float(bonds_i), 0.0)
                                excess = max(float(bonds_i) - candidate, 0.0)
                                cost = missing + excess
                                if best_cost is None or cost < best_cost:
                                    best_cost = cost
                                    best_missing = missing
                                    best_excess = excess
                            soft_missing += best_missing
                            soft_excess += best_excess

                    valence_underbond_soft[batch_idx] = float(soft_missing)
                    valence_overbond_soft[batch_idx] = float(soft_excess)
                except Exception:
                    valence_underbond_soft[batch_idx] = 0.0
                    valence_overbond_soft[batch_idx] = 0.0

        if self.stability_weight != 0.0:
            # Molecule-level: +w if stable, -w if unstable.
            stability_signal = stability_flags.mul(2.0).sub(1.0).clamp(min=-1.0, max=1.0)
            stability_bonus_molecule = self.stability_weight * stability_signal
        else:
            stability_bonus_molecule = torch.zeros_like(stability_flags)

        # Atom-level shaping: penalize the fraction of atoms that violate the simple valence rules.
        #
        # Why this exists:
        # - `check_stability` is binary at the molecule level. Under FED-GRPO group normalization,
        #   if a whole prompt group is unstable, the binary term becomes a constant and provides
        #   no within-group learning signal.
        # - `atom_stability` provides a *graded* notion of how close a sample is to full stability,
        #   which helps the policy make incremental progress (e.g., fixing a few stray H atoms).
        if self.atom_stability_weight != 0.0:
            atom_instability = (1.0 - atom_stability).clamp(min=0.0, max=1.0)
            atom_stability_bonus = -self.atom_stability_weight * atom_instability
        else:
            atom_stability_bonus = torch.zeros_like(atom_stability)

        if self.valence_underbond_weight != 0.0:
            valence_underbond_bonus = -self.valence_underbond_weight * valence_underbond
        else:
            valence_underbond_bonus = torch.zeros_like(valence_underbond)

        if self.valence_overbond_weight != 0.0:
            valence_overbond_bonus = -self.valence_overbond_weight * valence_overbond
        else:
            valence_overbond_bonus = torch.zeros_like(valence_overbond)

        if self.valence_underbond_soft_weight != 0.0:
            valence_underbond_soft_bonus = -self.valence_underbond_soft_weight * valence_underbond_soft
        else:
            valence_underbond_soft_bonus = torch.zeros_like(valence_underbond_soft)

        if self.valence_overbond_soft_weight != 0.0:
            valence_overbond_soft_bonus = -self.valence_overbond_soft_weight * valence_overbond_soft
        else:
            valence_overbond_soft_bonus = torch.zeros_like(valence_overbond_soft)

        stability_bonus = (
            stability_bonus_molecule
            + atom_stability_bonus
            + valence_underbond_bonus
            + valence_overbond_bonus
            + valence_underbond_soft_bonus
            + valence_overbond_soft_bonus
        )

        if self.force_only_if_stable:
            gate = stability_flags.to(dtype=force_rewards.dtype)
            force_rewards = force_rewards * gate
            if force_rewards_ts is not None:
                force_rewards_ts = force_rewards_ts * gate.unsqueeze(1)
            if rewards_ts is not None and not self.use_energy:
                rewards_ts = rewards_ts * gate.unsqueeze(1)

        force_rewards = force_rewards + stability_bonus

        if self.use_energy:
            energy_gate = torch.ones_like(energy_rewards, device=self.device, dtype=energy_rewards.dtype)
            if self.energy_only_if_stable:
                energy_gate = energy_gate * stability_flags.to(dtype=energy_rewards.dtype)
            energy_rewards = energy_rewards * energy_gate
            if energy_rewards_ts is not None:
                energy_rewards_ts = energy_rewards_ts * energy_gate.unsqueeze(1)

        weighted_force_rewards = self.force_weight * self.terminal_reward_weight * force_rewards
        weighted_energy_rewards = self.energy_weight * self.terminal_reward_weight * energy_rewards
        rewards = weighted_force_rewards + weighted_energy_rewards

        # Inject stability bonus into per-step traces so PPO sees the same signal as scalar rewards.
        if rewards_ts is not None and last_idx is not None:
            stability_term = self.force_weight * self.terminal_reward_weight * stability_bonus
            rewards_ts[:, last_idx] = rewards_ts[:, last_idx] + stability_term
            if force_rewards_ts is not None:
                force_rewards_ts[:, last_idx] = force_rewards_ts[:, last_idx] + self.terminal_reward_weight * stability_bonus

        if rewards_ts is not None:
            if self.use_energy and energy_rewards_ts is not None and force_rewards_ts is not None:
                rewards_ts = self.force_weight * force_rewards_ts + self.energy_weight * energy_rewards_ts
            total_rewards = rewards_ts.sum(dim=1)
        else:
            total_rewards = rewards

        # Populate final fields
        if rewards_ts is not None:
            result["rewards_ts"] = rewards_ts.detach().cpu()
        if force_rewards_ts is not None:
            result["force_rewards_ts"] = force_rewards_ts.detach().cpu()
        if self.use_energy and energy_rewards_ts is not None:
            result["energy_rewards_ts"] = energy_rewards_ts.detach().cpu()
        if terminal_force_metric is not None:
            result["force_terminal_metric"] = terminal_force_metric.detach().cpu()
        if terminal_energy_metric is not None:
            result["energy_terminal_metric"] = terminal_energy_metric.detach().cpu()

        result["rewards"] = total_rewards.detach().cpu()
        result["force_rewards"] = force_rewards.detach().cpu()
        result["energy_rewards"] = energy_rewards.detach().cpu()
        result["weighted_force_rewards"] = weighted_force_rewards.detach().cpu()
        result["weighted_energy_rewards"] = weighted_energy_rewards.detach().cpu()
        result["stability"] = stability_flags.cpu()
        result["atom_stability"] = atom_stability.detach().cpu()
        result["stability_bonus_molecule"] = stability_bonus_molecule.detach().cpu()
        result["stability_bonus_atom"] = atom_stability_bonus.detach().cpu()
        result["valence_underbond"] = valence_underbond.detach().cpu()
        result["valence_underbond_rewards"] = valence_underbond_bonus.detach().cpu()
        result["valence_overbond"] = valence_overbond.detach().cpu()
        result["valence_overbond_rewards"] = valence_overbond_bonus.detach().cpu()
        result["valence_underbond_soft"] = valence_underbond_soft.detach().cpu()
        result["valence_underbond_soft_rewards"] = valence_underbond_soft_bonus.detach().cpu()
        result["valence_overbond_soft"] = valence_overbond_soft.detach().cpu()
        result["valence_overbond_soft_rewards"] = valence_overbond_soft_bonus.detach().cpu()
        result["stability_rewards"] = stability_bonus.detach().cpu()

        if self.is_main_process:
            print("Rewards calculated via UMAForceReward")
        return DataProto(batch=TensorDict(result, batch_size=[batch_size]), meta_info=meta_info)
