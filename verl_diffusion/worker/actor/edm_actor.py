from typing import Callable, Dict, Optional
from tqdm import tqdm as tq
from collections import defaultdict
from verl_diffusion.protocol import DataProto
from verl_diffusion.utils.math import kl_divergence_normal
from verl_diffusion.utils.torch_functional import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_wsd_schedule_with_warmup,
)
from .base import BaseActor
import torch
from torch.nn.utils import clip_grad_norm_
import numpy as np
import torch.distributed as dist
class EDMActor(BaseActor):
    def __init__(self, model, config):
        """
        Initialize the EDM rollout worker.
        
        Args:
            model: The EDM model to use for rollout
            config: Configuration parameters for the rollout
        """
        super().__init__()  
        self.model = model
        self.config = config
        dist_cfg = self.config.get("distributed") or {}
        self.is_main_process = bool(dist_cfg.get("is_main_process", True))
        optimizer_cls = torch.optim.AdamW
        self.optimizer = optimizer_cls(
            self.model.parameters(),
            lr=self.config["train"]["learning_rate"],
            betas=(self.config["train"]["adam_beta1"], self.config["train"]["adam_beta2"]),
            weight_decay=self.config["train"]["adam_weight_decay"],
            eps=self.config["train"]["adam_epsilon"],
        )
        scheduler_cfg = self.config["train"].get("scheduler", {})
        self.scheduler_config = scheduler_cfg if isinstance(scheduler_cfg, dict) else {}
        self.lr_scheduler = None
        self.lr_history = []
        self.train_micro_batch_size = self.config["train"]["train_micro_batch_size"]
        self.max_grad_norm = self.config["train"]["max_grad_norm"]
        self.clip_range = self.config["train"]["clip_range"]
        self.adv_clip_max = max(0.0, float(self.config["train"].get("adv_clip_max", 5.0) or 5.0))
        self.epoch_per_rollout = max(int(self.config["train"].get("epoch_per_rollout", 1)), 1)
        self.num_timesteps = self.config["model"]["time_step"]
        alignment_cfg = self.config.get("train", {})
        raw_alignment_weight = float(alignment_cfg.get("force_alignment_weight", 0.0))
        enabled_cfg = alignment_cfg.get("force_alignment_enabled")
        if enabled_cfg is None:
            self.force_alignment_enabled = raw_alignment_weight > 0.0
        else:
            self.force_alignment_enabled = bool(enabled_cfg)
        self.force_alignment_weight = raw_alignment_weight if self.force_alignment_enabled else 0.0
        self.force_alignment_min_force = float(alignment_cfg.get("force_alignment_min_force", 1e-4))
        self.force_alignment_min_delta = float(alignment_cfg.get("force_alignment_min_delta", 1e-4))
        self._force_alignment_eps = 1e-8
        model_cfg = self.config.get("model") or {}
        self.share_initial_noise = bool(model_cfg.get("share_initial_noise", False))
        reward_cfg = self.config.get("reward") or {}
        shaping_cfg = reward_cfg.get("shaping", {}) if isinstance(reward_cfg, dict) else {}
        scheduler_cfg = {}
        if isinstance(shaping_cfg, dict):
            scheduler_cfg = shaping_cfg.get("scheduler", {}) or {}
        # The reward shaper mirrors DanceGRPO's API: allow a global skip prefix
        # with optional per-scheduler overrides. Normalize everything to an int.
        skip_default = shaping_cfg.get("skip_prefix", 0) if isinstance(shaping_cfg, dict) else 0
        skip_value = scheduler_cfg.get("skip_prefix", skip_default)
        try:
            self.skip_prefix = max(0, int(skip_value))
        except Exception:
            self.skip_prefix = 0
        self.active_num_timesteps = self.num_timesteps

    def setup_scheduler(self, total_training_steps: Optional[int] = None) -> None:
        """Initialize the learning rate scheduler if configured."""
        if not self.scheduler_config:
            return

        name = str(self.scheduler_config.get("name", "")).lower()
        if name in {"", "none"}:
            return

        # Determine schedule parameters
        warmup_steps = int(self.scheduler_config.get("warmup_steps", 0))
        if total_training_steps is None:
            total_training_steps = self.scheduler_config.get("total_steps")
            if total_training_steps is not None:
                total_training_steps = int(total_training_steps)

        if total_training_steps is None or total_training_steps <= 0:
            raise ValueError(
                "total_training_steps must be provided for learning rate scheduling. "
                "Set train.scheduler.total_steps or ensure the trainer passes a valid value."
            )

        if name in {"cosine", "cosine_warmup"}:
            min_lr_ratio = float(self.scheduler_config.get("min_lr_ratio", 0.0))
            num_cycles = float(self.scheduler_config.get("num_cycles", 0.5))
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_training_steps,
                min_lr_ratio=min_lr_ratio,
                num_cycles=num_cycles,
            )
        elif name in {"constant", "constant_warmup", "linear_warmup"}:
            self.lr_scheduler = get_constant_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
            )
        elif name in {"wsd", "warmup_stable_decay", "warmup-stable-decay"}:
            min_lr_ratio = float(self.scheduler_config.get("min_lr_ratio", 0.0))
            num_cycles = float(self.scheduler_config.get("num_cycles", 0.5))
            stable_ratio = float(self.scheduler_config.get("stable_ratio", 0.9))
            self.lr_scheduler = get_wsd_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_training_steps,
                min_lr_ratio=min_lr_ratio,
                num_cycles=num_cycles,
                stable_ratio=stable_ratio,
            )
        else:
            raise ValueError(f"Unsupported scheduler name '{name}'.")

    def train_batched_samples(self, batched_samples):
        """
        Train on a batch of samples. Main training segment

        Args:
            global_step (int): The current global step
            batched_samples (list[dict[str, torch.Tensor]]): The batched samples to train on

        Side Effects:
            - Model weights are updated
            - Logs the statistics to the accelerator trackers.

        Returns:
            global_step (int): The updated global step
        """
        info = defaultdict(list)
        # Train only over latent transitions (exclude terminal x/h step)
        self.T = getattr(self, "active_num_timesteps", self.num_timesteps)
        if self.T <= 0:
            metric = {
                "ClipFrac": 0.0,
                "Loss": 0.0,
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            return metric
        alignment_active = self.force_alignment_enabled and self.force_alignment_weight > 0.0
        grad_accum_steps = int(self.config["train"].get("gradient_accumulation_steps", 1) or 1)
        if grad_accum_steps <= 0:
            grad_accum_steps = 1
        num_batches = len(batched_samples)
        self.optimizer.zero_grad()
        last_lr = None
        for _i, sample in tq(
            enumerate(batched_samples),
            desc="Training",
            unit="Batch",
            leave=False,
            disable=not self.is_main_process,
        ):
            mus = sample.get("mus") if alignment_active else None # number of total samples \times number of active steps \times number of nodes \times dim (9)
            force_vectors = sample.get("force_vectors") if alignment_active else None
            force_indices = sample.get("force_schedule_indices") if alignment_active else None
            force_fine_mask = sample.get("force_fine_mask") if alignment_active else None
            force_index_lookup = None
            if alignment_active and force_indices is not None:
                reference = force_indices[0]
                try:
                    repeated = reference.unsqueeze(0).expand_as(force_indices)
                    if torch.equal(force_indices, repeated):
                        force_index_lookup = {int(step): idx for idx, step in enumerate(reference.tolist())}
                except RuntimeError:
                    force_index_lookup = None

            for j in tq(
                range(self.T),
                desc="Training Batch",
                unit="timesteps",
                leave=False,
                disable=not self.is_main_process,
            ):
                context = sample["context"] if self.condition else None
                if "advantages_ts" in sample:
                    advantages_j = sample["advantages_ts"][:, j]
                else:
                    advantages_j = sample["advantages"]

                mu_old_step = mus[:, j] if mus is not None else None
                force_vectors_step = None
                fine_mask_step = None
                if force_vectors is not None and force_indices is not None:
                    slot = None
                    if force_index_lookup is not None:
                        slot = force_index_lookup.get(int(j))
                        if slot is not None:
                            force_vectors_step = force_vectors[:, slot] # number of total samples \times num nodes \times 3
                            if force_fine_mask is not None:
                                fine_mask_step = force_fine_mask[:, slot]
                    if slot is None:
                        force_mask = (force_indices == j)
                        if force_mask.any():
                            force_mask_f = force_mask.float()
                            force_vectors_step = (
                                force_vectors * force_mask_f.unsqueeze(-1).unsqueeze(-1)
                            ).sum(dim=1)
                            if force_fine_mask is not None:
                                fine_mask_step = (force_fine_mask * force_mask_f).sum(dim=1)

                loss, clipfrac, _adv_clipfrac, align_penalty, align_cosine = self.calculate_loss(
                        sample["latents"][:, j],
                        sample["timesteps"][:, j],
                        sample["next_latents"][:, j],
                        sample["logps"][:, j],
                        advantages_j,
                        sample["nodesxsample"],
                        context = context,
                        mu_old=mu_old_step,
                        force_vectors=force_vectors_step,
                        fine_stage_mask=fine_mask_step,
                    )
                info["clipfrac"].append(clipfrac.item())
                info["loss"].append(loss.detach().cpu().item())
                if align_penalty is not None:
                    info["force_alignment_penalty"].append(align_penalty.detach().cpu().item())
                if align_cosine is not None:
                    info["force_alignment_cosine"].append(align_cosine.detach().cpu().item())
                scaled_loss = loss / float(grad_accum_steps * self.T)
                scaled_loss.backward()

            step_condition = ((_i + 1) % grad_accum_steps == 0) or (_i + 1 == num_batches)
            if step_condition:
                current_lr = self.optimizer.param_groups[0]["lr"]
                if self.max_grad_norm is not None and self.max_grad_norm > 0:
                    # Clip gradients once per optimizer step, matching DanceGRPO's
                    # accumulate-then-clip pattern instead of clipping after every
                    # micro batch.
                    clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=float(self.max_grad_norm),
                    )
                self.optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                self.lr_history.append(current_lr)
                last_lr = current_lr
                self.optimizer.zero_grad()
        
            
        metric = {}
        metric["ClipFrac"] = float(np.mean(np.array(info["clipfrac"]))) if info["clipfrac"] else 0.0
        metric["Loss"] = float(np.mean(np.array(info["loss"]))) if info["loss"] else 0.0
        if last_lr is None:
            last_lr = self.optimizer.param_groups[0]["lr"]
        metric["lr"] = last_lr
        if info["force_alignment_penalty"]:
            metric["ForceAlignPenalty"] = np.mean(np.array(info["force_alignment_penalty"]))
        if info["force_alignment_cosine"]:
            metric["ForceAlignCosine"] = np.mean(np.array(info["force_alignment_cosine"]))
        return metric
    
    def calculate_loss(
        self,
        latents,
        timesteps,
        next_latents,
        log_prob_old,
        advantages,
        nodesxsample,
        context=None,
        mu_old=None,
        force_vectors=None,
        fine_stage_mask=None,
    ):
        """
        Calculate the loss for a batch of an unpacked sample

        Args:
            latents (torch.Tensor):
                The latents sampled from the diffusion model, shape: [batch_size, num_channels_latents, height, width]
            timesteps (torch.Tensor):
                The timesteps sampled from the diffusion model, shape: [batch_size]
            next_latents (torch.Tensor):
                The next latents sampled from the diffusion model, shape: [batch_size, num_channels_latents, height, width]
            log_prob (torch.Tensor):
                The log probabilities of the latents, shape: [batch_size]
            advantages (torch.Tensor):
                The advantages of the latents, shape: [batch_size]
            context (torch.Tensor):
                The embedding of context.
            mu_old (torch.Tensor, optional):
                Drift produced by the behavior policy for the current diffusion step.
            force_vectors (torch.Tensor, optional):
                Force direction estimates for the current diffusion step, shaped [batch_size, n_nodes, 3].
            fine_stage_mask (torch.Tensor, optional):
                Per-sample indicator (1 -> fine-stage timestep) used to gate force alignment.
        Returns:
            loss (torch.Tensor), clipfrac (torch.Tensor), adv_clipfrac (torch.Tensor or None), alignment_penalty (torch.Tensor or None), alignment_cosine (torch.Tensor or None)
        """
        s_array = timesteps # a broadcast vector of a scalar step
        t_array = s_array + 1
        s_array = s_array / self.num_timesteps
        t_array = t_array / self.num_timesteps
        ## need add
        # Use the underlying model (in case it is wrapped by DataParallel) to build diffusion masks.
        model_ref = self.model.module if hasattr(self.model, "module") else self.model
        # Generate the per-node and per-edge validity masks for this batch, given how many nodes each sample uses.
        node_mask, edge_mask = model_ref.get_mask(nodesxsample, latents.shape[0], self.max_n_nodes)
        # Move masks onto the same device as the latent tensors so later ops don't hit CPU/GPU mismatches.
        node_mask = node_mask.to(latents.device)
        edge_mask = edge_mask.to(latents.device)

        # Evaluate the current diffusion policy for the reverse step p(z_s | z_t) to get the
        # transition log-probability and drift (mu) that will be compared against the behavior policy.
        _, log_prob_current, mu_current, _ = model_ref.sample_p_zs_given_zt(
            s_array,
            t_array,
            latents,
            node_mask,
            edge_mask,
            prev_sample=next_latents,
            context=context,
        )
    
        ## log_prob_current is old latents in new policy
        
        # compute the log prob of next_latents given latents under the current model
        dif_logp = (log_prob_current - log_prob_old)
        ratio = torch.exp(dif_logp)
        adv_clipfrac = None
        if self.adv_clip_max > 0:
            clip_limit = float(self.adv_clip_max)
            adv_clipfrac = (advantages.abs() > clip_limit).float().mean()
            advantages = torch.clamp(advantages, -clip_limit, clip_limit)

        loss = self.loss(advantages, self.clip_range, ratio)
        clipfrac = torch.mean((torch.abs(ratio - 1.0) > self.clip_range).float())

        alignment_penalty = None
        alignment_cosine = None
        if not (self.force_alignment_enabled and self.force_alignment_weight > 0):
            return loss, clipfrac, adv_clipfrac, None, None
        if mu_old is None or force_vectors is None or fine_stage_mask is None:
            return loss, clipfrac, adv_clipfrac, None, None
        if torch.is_tensor(fine_stage_mask) and torch.all(fine_stage_mask == 0):
            return loss, clipfrac, adv_clipfrac, None, None

        mu_old = mu_old.to(latents.device)
        force_vectors = force_vectors.to(latents.device)
        fine_stage_mask = fine_stage_mask.to(latents.device)
        alignment_metrics = self._compute_force_alignment_penalty(
            mu_old=mu_old,
            mu_new=mu_current,
            force_vectors=force_vectors,
            node_mask=node_mask,
            fine_stage_mask=fine_stage_mask,
        )
        if alignment_metrics is not None:
            alignment_penalty, alignment_cosine = alignment_metrics
        if alignment_penalty is not None:
            loss = loss + alignment_penalty

        return loss, clipfrac, adv_clipfrac, alignment_penalty, alignment_cosine

    def _compute_force_alignment_penalty(self, mu_old, mu_new, force_vectors, node_mask, fine_stage_mask):
        """
        Encourage the updated drift to stay aligned with force directions during fine-stage steps.
        """
        if fine_stage_mask.dim() == 1:
            fine_stage_mask = fine_stage_mask.view(-1, 1, 1)
        else:
            fine_stage_mask = fine_stage_mask.view(mu_new.size(0), 1, 1)
        fine_stage_mask = fine_stage_mask.to(mu_new.dtype)

        mask = (node_mask * fine_stage_mask).to(mu_new.dtype)
        if mask.sum().item() <= 0:
            return None

        mu_old = mu_old.detach().to(mu_new.dtype)
        force_vectors = force_vectors.detach().to(mu_new.dtype)

        force_vectors = force_vectors * mask
        delta_mu = (mu_new[:, :, :3] - mu_old[:, :, :3]) * mask

        force_norm = force_vectors.norm(dim=-1, keepdim=True)
        delta_norm = delta_mu.norm(dim=-1, keepdim=True)

        valid_force = (force_norm > self.force_alignment_min_force).float()
        valid_delta = (delta_norm > self.force_alignment_min_delta).float()
        valid = valid_force * valid_delta * mask

        if valid.sum().item() <= 0:
            return None

        force_unit = force_vectors / (force_norm + self._force_alignment_eps)
        delta_unit = delta_mu / (delta_norm + self._force_alignment_eps)

        cosine = (force_unit * delta_unit).sum(dim=-1).clamp(-1.0, 1.0)
        weighted_area = valid.squeeze(-1)
        alignment_error = (1.0 - cosine) * weighted_area
        penalty_scalar = alignment_error.sum() / (weighted_area.sum() + self._force_alignment_eps)
        penalty = self.force_alignment_weight * penalty_scalar
        mean_cosine = (cosine * weighted_area).sum() / (weighted_area.sum() + self._force_alignment_eps)
        return penalty, mean_cosine
    
    def loss(
        self,
        advantages: torch.Tensor,
        clip_range: float,
        ratio: torch.Tensor,
    ):
        unclipped_loss =    -1.0 * advantages * ratio
        clipped_loss =   -1.0  * advantages * torch.clamp(
            ratio,
            1.0 - clip_range,
            1.0 + clip_range,
        )
        
        return torch.mean(torch.maximum(unclipped_loss, clipped_loss))
    
    def update_policy(
        self,
        data: DataProto,
        epoch_callback: Optional[Callable[[int, Dict[str, float]], None]] = None,
    ) -> Dict:
        samples = {}
        self.max_n_nodes = data.meta_info["max_n_nodes"]
        self.batch_size = data.batch.batch_size[0]
        self.condition = data.meta_info["condition"]
        # Optional per-step advantages from reward shaping
        if "advantages_ts" in data.batch.keys():
            samples["advantages_ts"] = data.batch["advantages_ts"]
        if "advantages" in data.batch.keys():
            samples["advantages"] = data.batch["advantages"]

        # Latent sequences contain z_T..z_0 and a final decoded x/h frame; exclude terminal for training
        latents_full = data.batch["latents"][:, : self.num_timesteps + 1].clone()
        next_latents_full = data.batch["latents"][:, 1 : self.num_timesteps + 1].clone()
        timesteps_full = data.batch["timesteps"][:, : self.num_timesteps].clone()
        logps_full = data.batch["logps"][:, : self.num_timesteps].clone()

        meta_skip_prefix = data.meta_info.get("skip_prefix", self.skip_prefix)
        try:
            meta_skip_prefix = int(meta_skip_prefix)
        except Exception:
            meta_skip_prefix = self.skip_prefix
        share_prefix = bool(data.meta_info.get("share_initial_noise", self.share_initial_noise))
        start_idx = min(meta_skip_prefix if share_prefix else 0, self.num_timesteps)

        if start_idx > 0:
            latents_full = latents_full[:, start_idx:]
            next_latents_full = next_latents_full[:, start_idx:]
            timesteps_full = timesteps_full[:, start_idx:]
            logps_full = logps_full[:, start_idx:]

        samples["latents"] = latents_full
        samples["next_latents"] = next_latents_full
        samples["timesteps"] = timesteps_full
        samples["logps"] = logps_full
        self.active_num_timesteps = logps_full.size(1)
        samples["nodesxsample"] = data.batch["nodesxsample"]
        if self.force_alignment_enabled and "mus" in data.batch.keys():
            mus_full = data.batch["mus"][:, : self.num_timesteps + 1].clone()
            if start_idx > 0:
                mus_full = mus_full[:, start_idx:]
            samples["mus"] = mus_full
        if self.force_alignment_enabled and "force_vectors_schedule" in data.batch.keys():
            force_vectors = data.batch["force_vectors_schedule"].clone()
            force_schedule_indices = data.batch["force_schedule_indices"].clone()
            keep_mask = None
            if start_idx > 0 and force_schedule_indices.numel() > 0:
                keep_mask = force_schedule_indices[0] >= start_idx
                force_vectors = force_vectors[:, keep_mask]
                force_schedule_indices = force_schedule_indices[:, keep_mask]
            force_schedule_indices = force_schedule_indices - start_idx
            samples["force_vectors"] = force_vectors
            samples["force_schedule_indices"] = force_schedule_indices
            if "force_fine_mask" in data.batch.keys():
                force_fine_mask = data.batch["force_fine_mask"].clone()
                if keep_mask is not None and force_fine_mask.numel() > 0:
                    force_fine_mask = force_fine_mask[:, keep_mask]
                samples["force_fine_mask"] = force_fine_mask
        if self.condition:
            samples["context"] = data.batch["context"]
        # Optional per-step advantages should align with latent steps only
        if "advantages_ts" in samples:
            adv_ts = samples["advantages_ts"]
            adv_ts = adv_ts[:, : self.num_timesteps]
            if start_idx > 0:
                adv_ts = adv_ts[:, start_idx:]
            samples["advantages_ts"] = adv_ts
            samples["advantages"] = adv_ts.sum(dim=1)

        original_keys = list(samples.keys())
        original_values = list(samples.values())
        # rebatch them as user defined train_batch_size is different from sample_batch_size
        samples_batched = []

        if self.train_micro_batch_size >= self.batch_size:
            # If train_micro_batch_size is larger than or equal to batch_size, just use the entire batch
            samples_batched.append({k: v for k, v in samples.items()})
        else:
            # Calculate number of complete batches and remaining samples
            n_complete_batches = self.batch_size // self.train_micro_batch_size
            remaining_samples = self.batch_size % self.train_micro_batch_size

            # Process complete batches
            if n_complete_batches > 0:
                reshaped_values = [v.reshape(-1, self.train_micro_batch_size, *v.shape[1:]) for v in original_values]
                transposed_values = zip(*reshaped_values)
                samples_batched.extend([dict(zip(original_keys, row_values)) for row_values in transposed_values])

            # Process remaining samples if any
            if remaining_samples > 0:
                # Keep batch dimension as [remaining, ...] without adding an extra leading axis
                remaining_values = [v[-remaining_samples:] for v in original_values]
                samples_batched.append(dict(zip(original_keys, remaining_values)))

        epochs = max(int(self.epoch_per_rollout), 1)
        metrics = []
        for epoch_idx in range(epochs):
            metric = self.train_batched_samples(samples_batched)
            metrics.append(metric)
            if epoch_callback is not None:
                safe_metric = {}
                for key, value in metric.items():
                    if isinstance(value, torch.Tensor):
                        safe_metric[key] = value.detach().cpu().item()
                    elif isinstance(value, np.generic):
                        safe_metric[key] = float(value)
                    else:
                        safe_metric[key] = value
                epoch_callback(epoch_idx, safe_metric)

        if epochs == 1:
            return metrics[0]

        aggregated = {}
        for metric in metrics:
            for key, value in metric.items():
                aggregated.setdefault(key, []).append(value)

        for key, values in aggregated.items():
            aggregated[key] = float(np.mean(values))

        last_lr = metrics[-1].get('lr')
        if last_lr is not None:
            aggregated['lr'] = last_lr
        aggregated['PolicyEpochs'] = epochs

        return aggregated
