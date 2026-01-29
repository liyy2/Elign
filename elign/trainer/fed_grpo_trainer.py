import torch
import torch.distributed as dist
import queue
import time
import threading
import ray
from tqdm import tqdm
import wandb
import os
import yaml
import numpy as np
from typing import Dict
from verl_diffusion.trainer.base import BaseTrainer
from verl_diffusion.protocol import DataProto, TensorDict


class DDPOTrainer(BaseTrainer):
    def __init__(self, config, model, dataset_info, device, dataloader, rollout, rewarder, actor, filters=None):
        """
        Initialize the DDPO Trainer
        
        Args:
            config: Configuration dictionary
            model: The EDM model to use for generation
            dataset_info: Dataset information
            device: The device to run the model on
            dataloader: DataLoader for generating prompts
            rollout: The EDMRollout instance
            rewarder: The ForceReward instance
        """
        # Call parent constructor with required arguments
        super().__init__(config=config, rollout=rollout, reward=rewarder)
        
        # Store additional parameters
        self.model = model
        self.dataset_info = dataset_info
        self.device = device
        self.dataloader = dataloader
        self.config = config
        dist_cfg = self.config.get("distributed") or {}
        self.rank = int(dist_cfg.get("rank", 0))
        self.world_size = int(dist_cfg.get("world_size", 1))
        self.is_main_process = bool(dist_cfg.get("is_main_process", self.rank == 0))
        self.distributed = dist.is_initialized() and self.world_size > 1

        save_path = config.get("save_path")
        if not save_path:
            wandb_cfg = config.get("wandb") or {}
            default_name = "edm-ddp-run"
            if isinstance(wandb_cfg, dict):
                default_name = wandb_cfg.get("wandb_name", default_name)
            save_path = os.path.join("./exp", default_name)
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        self.epoches = self.config["dataloader"]["epoches"]
        # Use provided rollout and rewarder instances
        self.rollout = rollout
        self.rewarder = rewarder
        
        # Create queue between rollout and reward calculation
        self.samples_queue = queue.Queue(maxsize=config.get("queue_size", 256))
        
        # List to store reward calculation results
        self.reward_results = []
        self.filters = filters
        self.actor = actor
        model_cfg = self.config.get("model") or {}
        try:
            self.num_timesteps = int(model_cfg.get("time_step", 0))
        except Exception:
            self.num_timesteps = 0
        self.best_checkpoint_metric = config.get("best_checkpoint_metric", "reward")
        self.best_checkpoint_mode = str(config.get("best_checkpoint_mode", "max")).lower()
        if self.best_checkpoint_mode not in {"max", "min"}:
            raise ValueError(f"Unsupported best_checkpoint_mode '{self.best_checkpoint_mode}'. Use 'max' or 'min'.")
        self.best_metric_value = float('-inf') if self.best_checkpoint_mode == "max" else float('inf')

        reward_cfg = self.config.get("reward", {})
        if not isinstance(reward_cfg, dict):
            reward_cfg = {}
        self.force_adv_weight = float(reward_cfg.get("force_adv_weight", reward_cfg.get("force_weight", 1.0)))
        self.energy_adv_weight = float(reward_cfg.get("energy_adv_weight", reward_cfg.get("energy_weight", 1.0)))
        shaping_cfg = reward_cfg.get("shaping", {}) if isinstance(reward_cfg, dict) else {}
        self.terminal_adv_weight = float(shaping_cfg.get("terminal_weight", 1.5))

        # Initialize Ray for parallel processing (optional).
        ray_cfg = self.config.get("ray", {})
        ray_enabled = False
        if isinstance(ray_cfg, dict):
            ray_enabled = bool(ray_cfg.get("enabled", False))
        elif ray_cfg:
            ray_enabled = True

        if ray_enabled and self.is_main_process and not ray.is_initialized():
            ray.init()

        # Initialize wandb if enabled in config
        wandb_cfg = config.get("wandb")
        project = "edm-ddp"
        name = "edm-ddp-run"
        wandb_enabled = False

        if isinstance(wandb_cfg, dict):
            project = wandb_cfg.get("wandb_project", project)
            name = wandb_cfg.get("wandb_name", name)
            wandb_enabled = wandb_cfg.get("enabled", True)
        elif wandb_cfg:
            wandb_enabled = True

        if wandb_enabled and self.is_main_process:
            wandb.init(
                project=project,
                name=name,
                config=config
            )
            self.wandb_enabled = True
        else:
            self.wandb_enabled = False
    
    def _reward_worker(self):
        """Reward calculation worker thread function"""
        while True:
            try:
                # Get sample from queue
                sample = self.samples_queue.get(timeout=1.0)
                
                # Check for termination signal
                if sample is None:
                    self.samples_queue.task_done()
                    break
                
                # Calculate rewards

                result = self.rewarder.calculate_rewards(sample)
            
                
                # Store results
                self.reward_results.append(result)
                
                
                # Mark task as done
                self.samples_queue.task_done()
                
            except queue.Empty:
                # Queue is empty, continue waiting
                continue
            except Exception as exc:
                if self.is_main_process:
                    print(f"Error in reward calculation: {exc}")
                try:
                    batch_size = int(sample.batch.batch_size[0]) if sample.batch is not None else 0
                except Exception:
                    batch_size = 0
                if batch_size > 0:
                    fallback = {
                        "rewards": torch.full((batch_size,), -5.0),
                        "stability": torch.zeros(batch_size),
                    }
                    if sample.batch is not None and "timesteps" in sample.batch.keys():
                        timesteps = sample.batch["timesteps"]
                        if isinstance(timesteps, torch.Tensor) and timesteps.ndim == 2:
                            rewards_ts = torch.zeros((batch_size, timesteps.size(1)))
                            rewards_ts[:, -1] = fallback["rewards"]
                            fallback["rewards_ts"] = rewards_ts
                    self.reward_results.append(
                        DataProto(batch=TensorDict(fallback, batch_size=[batch_size]), meta_info=sample.meta_info.copy())
                    )
                self.samples_queue.task_done()

    def _sync_metrics(self, metrics):
        if not self.distributed:
            return metrics

        synced = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float, np.floating)):
                tensor = torch.tensor(float(value), device=self.device, dtype=torch.float32)
                dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
                reduced = tensor.item()
                if isinstance(value, int):
                    synced[key] = int(round(reduced))
                else:
                    synced[key] = reduced
            elif isinstance(value, torch.Tensor) and value.numel() == 1:
                tensor = value.to(self.device, dtype=torch.float32)
                dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
                synced[key] = tensor.item()
            else:
                synced[key] = value
        return synced

    def process_batch(self, batch_idx, prompts):
        """
        Process a single batch from the dataloader
        
        Args:
            batch_idx: Batch index
            prompts: Batch data from dataloader
            
        Returns:
            List of reward results for this batch
        """
        
        # Start reward calculation thread
        self.reward_results = []  # Clear previous results
        reward_thread = threading.Thread(target=self._reward_worker, daemon=True)
        reward_thread.start()
        
        try:
            batch_size = prompts.batch.batch_size[0]
            # Use dataloader.micro_batch_size if provided; fallback to top-level for backward compatibility
            micro_cfg = (self.config.get("dataloader") or {}).get("micro_batch_size", self.config.get("micro_batch_size", batch_size))
            try:
                micro_bs = int(micro_cfg)
            except Exception:
                micro_bs = batch_size
            num_chunks = max(batch_size // micro_bs, 1)
            
            # Split into mini-batches
            batch_prompts = prompts.chunk(chunks=num_chunks)
            
            # Initialize a list to store all sample results as DataProto objects
            sample_results = []
            # Process each mini-batch
            for chunk_idx, batch in enumerate(batch_prompts):
            
                
                # Generate sample using rollout
                sample = self.rollout.generate_minibatch(batch)
                
                # Add batch identifier
                sample_results.append(sample) 
                
                # Queue sample for reward calculation
                # This will block if queue is full, until reward calculation catches up
                self.samples_queue.put(sample)
                
            
            # Wait for all samples in this batch to complete reward calculation
            self.samples_queue.join()
            
            # Processing complete, send termination signal to reward worker
            self.samples_queue.put(None)
            
            # Wait for reward thread to complete
            reward_thread.join(timeout=10)
            
            # Output statistics for this batch
            sample_results = DataProto.concat(sample_results)
            reward_results = DataProto.concat(self.reward_results)
            reward_results = reward_results.to(self.device)
            return sample_results.union(reward_results)
            
        except Exception as e:
            if self.is_main_process:
                print(f"Error processing batch: {e}")
            # Send termination signal (if not already sent)
            try:
                self.samples_queue.put_nowait(None)
            except queue.Full:
                pass
            
            # Wait for reward thread to complete
            if reward_thread.is_alive():
                reward_thread.join(timeout=5)
            
            return None
    def compute_advantage(self, samples):
        """Compute GRPO-style advantages.

        - Samples are grouped by `group_index` (same prompt, multiple rollouts).
        - If reward shaping is enabled, PPO should consume per-timestep rewards (`rewards_ts`) and we
          normalize/clip advantages per timestep.
        - When both `force_rewards_ts` and `energy_rewards_ts` exist, we normalize them *separately*
          within each group and then mix the normalized advantages via `force_adv_weight` and
          `energy_adv_weight`. This is the main knob to reduce force dominance without relying on
          raw reward scaling (which cancels under normalization).
        """
        group_index = samples.batch["group_index"]
        clip_value = self.config["train"].get(
            "clip_advantage_value",
            self.config["train"].get("adv_clip_max", 5.0),
        )
        force_adv_weight = getattr(self, "force_adv_weight", 1.0)
        energy_adv_weight = getattr(self, "energy_adv_weight", 1.0)
        terminal_weight_applied = False

        # Per-timestep rewards available (reward shaping)
        if "rewards_ts" in samples.batch.keys():
            rewards_ts = samples.batch["rewards_ts"].to(group_index.device)  # [B, S]
            # Align terminal column (decode step) to the last diffusion transition.
            target_T = getattr(self, "num_timesteps", 0)
            if target_T <= 0:
                target_T = rewards_ts.size(1)
            if rewards_ts.size(1) > target_T:
                extra = rewards_ts[:, target_T:]
                rewards_ts = rewards_ts[:, :target_T].clone()
                rewards_ts[:, -1] = rewards_ts[:, -1] + extra.sum(dim=1)
                samples.batch["rewards_ts"] = rewards_ts
            elif rewards_ts.size(1) < target_T:
                target_T = rewards_ts.size(1)

            unique_groups = torch.unique(group_index)
            advantages_ts = torch.zeros_like(rewards_ts)

            # Check if we have separate force and energy rewards for GRPO
            if "force_rewards_ts" in samples.batch.keys() and "energy_rewards_ts" in samples.batch.keys():
                force_rewards_ts = samples.batch["force_rewards_ts"].to(group_index.device)  # [B, S]
                energy_rewards_ts = samples.batch["energy_rewards_ts"].to(group_index.device)  # [B, S]
                if force_rewards_ts.size(1) > target_T:
                    extra_force = force_rewards_ts[:, target_T:]
                    force_rewards_ts = force_rewards_ts[:, :target_T].clone()
                    force_rewards_ts[:, -1] = force_rewards_ts[:, -1] + extra_force.sum(dim=1)
                    samples.batch["force_rewards_ts"] = force_rewards_ts
                if energy_rewards_ts.size(1) > target_T:
                    extra_energy = energy_rewards_ts[:, target_T:]
                    energy_rewards_ts = energy_rewards_ts[:, :target_T].clone()
                    energy_rewards_ts[:, -1] = energy_rewards_ts[:, -1] + extra_energy.sum(dim=1)
                    samples.batch["energy_rewards_ts"] = energy_rewards_ts

                force_advantages_ts = torch.zeros_like(force_rewards_ts)
                energy_advantages_ts = torch.zeros_like(energy_rewards_ts)

                # Normalize force and energy separately within each group (GRPO)
                for group_id in unique_groups:
                    gmask = group_index == group_id
                    if gmask.any():
                        # Normalize force rewards
                        grp_force = force_rewards_ts[gmask]  # [B_g, S]
                        mean_force = grp_force.mean(dim=0, keepdim=True)  # [1, S]
                        # NOTE: use population std to avoid NaNs when a group collapses to size 1
                        # (e.g., due to filtering/dedup). `torch.std` defaults to unbiased=True,
                        # which returns NaN for B_g==1.
                        std_force = grp_force.std(dim=0, keepdim=True, unbiased=False)
                        force_advantages_ts[gmask] = (grp_force - mean_force) / (std_force + 1e-8)

                        # Normalize energy rewards
                        grp_energy = energy_rewards_ts[gmask]  # [B_g, S]
                        mean_energy = grp_energy.mean(dim=0, keepdim=True)  # [1, S]
                        std_energy = grp_energy.std(dim=0, keepdim=True, unbiased=False)
                        energy_advantages_ts[gmask] = (grp_energy - mean_energy) / (std_energy + 1e-8)

                # Combine normalized force and energy advantages
                weighted_force_advantages_ts = force_adv_weight * force_advantages_ts
                weighted_energy_advantages_ts = energy_adv_weight * energy_advantages_ts
                advantages_ts = weighted_force_advantages_ts + weighted_energy_advantages_ts

                # Reapply terminal weighting so PPO sees the same emphasis as scalar rewards
                if advantages_ts.shape[1] > 0 and self.terminal_adv_weight != 1.0:
                    time_weights = torch.ones_like(advantages_ts)
                    time_weights[:, -1] = self.terminal_adv_weight
                    advantages_ts = advantages_ts * time_weights
                    weighted_force_advantages_ts = weighted_force_advantages_ts * time_weights
                    weighted_energy_advantages_ts = weighted_energy_advantages_ts * time_weights
                    terminal_weight_applied = True

                # Store separate advantages for logging/analysis if needed
                samples.batch["force_advantages_ts"] = torch.clamp(
                    weighted_force_advantages_ts, -clip_value, clip_value
                )
                samples.batch["energy_advantages_ts"] = torch.clamp(
                    weighted_energy_advantages_ts, -clip_value, clip_value
                )
            else:
                # Standard normalization for combined rewards
                for group_id in unique_groups:
                    gmask = group_index == group_id
                    if gmask.any():
                        grp = rewards_ts[gmask]  # [B_g, S]
                        mean = grp.mean(dim=0, keepdim=True)  # [1, S]
                        std = grp.std(dim=0, keepdim=True, unbiased=False)
                        advantages_ts[gmask] = (grp - mean) / (std + 1e-8)

            # Apply terminal weighting to per-step advantages before clipping/summing
            if advantages_ts.shape[1] > 0 and self.terminal_adv_weight != 1.0 and not terminal_weight_applied:
                time_weights = torch.ones_like(advantages_ts)
                time_weights[:, -1] = self.terminal_adv_weight
                advantages_ts = advantages_ts * time_weights

            # Clip per-step advantages
            advantages_ts = torch.clamp(advantages_ts, -clip_value, clip_value)
            samples.batch["advantages_ts"] = advantages_ts

            # Keep scalar advantages for logging/backward compat (sum over time)
            advantages = advantages_ts.sum(dim=1)
            samples.batch["advantages"] = torch.clamp(advantages, -clip_value, clip_value)
            return samples

        # Fallback: scalar rewards only
        rewards = samples.batch["rewards"]
        unique_groups = torch.unique(group_index)
        normalized_rewards = torch.zeros_like(rewards)

        # Check if we have separate force and energy rewards for scalar case
        if "force_rewards" in samples.batch.keys() and "energy_rewards" in samples.batch.keys():
            force_rewards = samples.batch["force_rewards"].to(group_index.device)
            energy_rewards = samples.batch["energy_rewards"].to(group_index.device)

            force_advantages = torch.zeros_like(force_rewards)
            energy_advantages = torch.zeros_like(energy_rewards)

            # Normalize force and energy separately within each group (GRPO)
            for group_id in unique_groups:
                group_mask = group_index == group_id
                if group_mask.any():
                    # Normalize force rewards
                    group_force = force_rewards[group_mask]
                    force_mean = group_force.mean()
                    force_std = group_force.std(unbiased=False)
                    force_advantages[group_mask] = (group_force - force_mean) / (force_std + 1e-8)

                    # Normalize energy rewards
                    group_energy = energy_rewards[group_mask]
                    energy_mean = group_energy.mean()
                    energy_std = group_energy.std(unbiased=False)
                    energy_advantages[group_mask] = (group_energy - energy_mean) / (energy_std + 1e-8)

            # Combine normalized advantages
            weighted_force_advantages = force_adv_weight * force_advantages
            weighted_energy_advantages = energy_adv_weight * energy_advantages
            normalized_rewards = weighted_force_advantages + weighted_energy_advantages

            # Store separate advantages for logging/analysis
            samples.batch["force_advantages"] = torch.clamp(weighted_force_advantages, -clip_value, clip_value)
            samples.batch["energy_advantages"] = torch.clamp(weighted_energy_advantages, -clip_value, clip_value)
        else:
            # Standard normalization
            for group_id in unique_groups:
                group_mask = group_index == group_id
                group_rewards = rewards[group_mask]
                group_mean = group_rewards.mean()
                group_std = group_rewards.std(unbiased=False)
                normalized_rewards[group_mask] = (group_rewards - group_mean) / (group_std + 1e-8)

        samples.batch["advantages"] = torch.clamp(normalized_rewards, -clip_value, clip_value)
        return samples

    def save_checkpoint(self, epoch, metrics=None):
        """Save model checkpoint and config"""
        if not self.is_main_process:
            return

        def atomic_torch_save(obj, path: str) -> None:
            tmp_path = f"{path}.tmp"
            torch.save(obj, tmp_path)
            os.replace(tmp_path, path)

        def atomic_write_yaml(obj, path: str) -> None:
            tmp_path = f"{path}.tmp"
            if isinstance(obj, dict):
                with open(tmp_path, "w") as f:
                    yaml.safe_dump(obj, f)
                    f.flush()
                    os.fsync(f.fileno())
            else:
                obj.to_yaml(tmp_path)
            os.replace(tmp_path, path)

        # Save config
        config_path = os.path.join(self.save_path, "config.yaml")
        atomic_write_yaml(self.config, config_path)
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'actor_state_dict': self.actor.model.state_dict(),
            'optimizer_state_dict': self.actor.optimizer.state_dict(),
            'metrics': metrics,
        }
        lr_scheduler = getattr(self.actor, "lr_scheduler", None)
        if lr_scheduler is not None:
            try:
                checkpoint["scheduler_state_dict"] = lr_scheduler.state_dict()
            except Exception:
                checkpoint["scheduler_state_dict"] = None
        
        # Save latest checkpoint
        atomic_torch_save(checkpoint, os.path.join(self.save_path, "checkpoint_latest.pth"))
        
        # Save epoch checkpoint
        atomic_torch_save(checkpoint, os.path.join(self.save_path, f"checkpoint_epoch_{epoch}.pth"))
        
        # Save best checkpoint if metrics are provided
        if metrics is not None:
            metric_value = metrics.get(self.best_checkpoint_metric)
            if metric_value is not None:
                is_better = (
                    metric_value > self.best_metric_value
                    if self.best_checkpoint_mode == "max"
                    else metric_value < self.best_metric_value
                )
                if is_better:
                    self.best_metric_value = metric_value
                    atomic_torch_save(
                        self.model.state_dict(),
                        os.path.join(self.save_path, "generative_model_ema.npy"),
                    )
                    atomic_torch_save(checkpoint, os.path.join(self.save_path, "checkpoint_best.pth"))
            else:
                print(f"Warning: Metric '{self.best_checkpoint_metric}' not found in metrics. "
                      "Skipping best-checkpoint update.")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.actor.model.load_state_dict(checkpoint['actor_state_dict'])
        self.actor.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self._apply_train_hparams_to_optimizer()
        
        if 'metrics' in checkpoint:
            metric_value = checkpoint['metrics'].get(self.best_checkpoint_metric)
            if metric_value is not None:
                self.best_metric_value = metric_value
            else:
                self.best_metric_value = float('-inf') if self.best_checkpoint_mode == "max" else float('inf')

        return checkpoint

    def _apply_train_hparams_to_optimizer(self) -> None:
        """Ensure optimizer hyperparameters reflect the current config.

        When resuming, PyTorch restores optimizer param group settings (including LR) from the
        checkpoint. That is desired for an exact resume, but it also means CLI/config overrides
        (e.g. lowering LR for a fine-tune) would silently have no effect. To make hyperparameter
        sweeps sane, we re-apply the current `train.*` optimizer knobs after loading a checkpoint.
        """
        train_cfg = self.config.get("train") if isinstance(self.config, dict) else {}
        if not isinstance(train_cfg, dict):
            return

        optimizer = getattr(self.actor, "optimizer", None)
        if optimizer is None:
            return

        def _maybe_float(value, default=None):
            if value is None:
                return default
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        lr = _maybe_float(train_cfg.get("learning_rate"))
        beta1 = _maybe_float(train_cfg.get("adam_beta1"))
        beta2 = _maybe_float(train_cfg.get("adam_beta2"))
        weight_decay = _maybe_float(train_cfg.get("adam_weight_decay"))
        eps = _maybe_float(train_cfg.get("adam_epsilon"))

        for group in optimizer.param_groups:
            if lr is not None:
                group["lr"] = lr
            if beta1 is not None or beta2 is not None:
                existing_betas = group.get("betas", (0.9, 0.999))
                group["betas"] = (
                    beta1 if beta1 is not None else existing_betas[0],
                    beta2 if beta2 is not None else existing_betas[1],
                )
            if weight_decay is not None:
                group["weight_decay"] = weight_decay
            if eps is not None:
                group["eps"] = eps

    def fit(self):
        """
        Training loop for DDPO - processes each dataloader batch and updates model parameters
        
        Overrides BaseTrainer.fit() method
        """
        try:
            start_epoch = 0
            global_batch_idx = 0
            loaded_checkpoint = None
            start_time = time.monotonic()
            train_cfg = self.config.get("train") if isinstance(self.config, dict) else {}
            if not isinstance(train_cfg, dict):
                train_cfg = {}

            max_time_hours = train_cfg.get("max_time_hours")
            max_time_seconds = None
            if max_time_hours is not None:
                try:
                    max_time_seconds = float(max_time_hours) * 3600.0
                except (TypeError, ValueError):
                    max_time_seconds = None

            early_stop_metric = train_cfg.get("early_stop_metric")
            early_stop_mode = str(train_cfg.get("early_stop_mode", "max")).lower()
            if early_stop_mode not in {"max", "min"}:
                early_stop_mode = "max"
            early_stop_patience_minutes = train_cfg.get("early_stop_patience_minutes")
            early_stop_patience_seconds = None
            if early_stop_patience_minutes is not None:
                try:
                    early_stop_patience_seconds = float(early_stop_patience_minutes) * 60.0
                except (TypeError, ValueError):
                    early_stop_patience_seconds = None
            try:
                early_stop_min_delta = float(train_cfg.get("early_stop_min_delta", 0.0))
            except (TypeError, ValueError):
                early_stop_min_delta = 0.0

            best_early_stop_value = None
            last_improvement_time = start_time
            # Load checkpoint if resuming
            if self.config.get('resume', False) and self.config.get('checkpoint_path'):
                loaded_checkpoint = self.load_checkpoint(self.config['checkpoint_path'])
                last_epoch = loaded_checkpoint.get('epoch', 0)
                start_epoch = int(last_epoch) + 1
                if self.is_main_process:
                    print(f"Resuming training from epoch {start_epoch}")

            # Initialize learning rate scheduler if configured.
            # IMPORTANT: do this after loading a checkpoint so we don't reset LR when resuming.
            scheduler_cfg = (self.config.get("train") or {}).get("scheduler")
            if isinstance(scheduler_cfg, dict) and scheduler_cfg.get("name"):
                total_training_steps = scheduler_cfg.get("total_steps")
                if total_training_steps is None:
                    total_training_steps = self.epoches
                if total_training_steps is not None:
                    scheduler_state = None
                    if loaded_checkpoint is not None:
                        scheduler_state = loaded_checkpoint.get("scheduler_state_dict")
                    if loaded_checkpoint is None or scheduler_state is not None:
                        self.actor.setup_scheduler(int(total_training_steps))
                        if scheduler_state is not None and getattr(self.actor, "lr_scheduler", None) is not None:
                            try:
                                self.actor.lr_scheduler.load_state_dict(scheduler_state)
                            except Exception as exc:
                                if self.is_main_process:
                                    print(f"Warning: failed to restore LR scheduler state: {exc}")
                    elif self.is_main_process:
                        print(
                            "Warning: resume checkpoint has no scheduler_state_dict; "
                            "skipping LR scheduler to avoid LR resets."
                        )
            global_batch_idx = start_epoch
            prompt_iter = iter(self.dataloader)
            if start_epoch > 0:
                for _ in range(start_epoch):
                    try:
                        next(prompt_iter)
                    except StopIteration:
                        break

            metrics = None
            last_epoch = None
            save_interval = int(self.config.get("save_interval", 10))
            if save_interval <= 0:
                save_interval = 10

            for epoch in range(start_epoch, self.epoches):
                try:
                    prompts = next(prompt_iter)
                except StopIteration:
                    break

                last_epoch = epoch

                samples = self.process_batch(epoch, prompts)
                if not isinstance(samples, DataProto) or len(samples) == 0:
                    if self.is_main_process:
                        print(f"Skipping epoch {epoch} because batch processing failed or returned empty samples.")
                    continue
                rdkit_validity = None
                rdkit_uniqueness = None
                if self.filters is not None:
                    filter_out = self.filters.filter(samples)
                    if isinstance(filter_out, (list, tuple)) and len(filter_out) == 5:
                        samples, filter_ratio, novelty_penalty_ratio, rdkit_validity, rdkit_uniqueness = filter_out
                    elif isinstance(filter_out, (list, tuple)) and len(filter_out) == 4:
                        samples, filter_ratio, novelty_penalty_ratio, rdkit_validity = filter_out
                    else:
                        samples, filter_ratio, novelty_penalty_ratio = filter_out
                else:
                    filter_ratio, novelty_penalty_ratio = 1.0, 1.0
                if len(samples) == 0:
                    if self.is_main_process:
                        print(f"Skipping epoch {epoch} because filtering removed all samples.")
                    continue
                samples = self.compute_advantage(samples)
                metrics = self.actor.update_policy(samples, epoch_callback=None)
                metrics["epoch"] = epoch + 1
                # `reward` is the mean of the final scalar signal used for policy updates,
                # i.e. the force and energy components after weighting plus any shaping bonus.
                # `force_reward_mean` logs the unweighted force term, while
                # `weighted_force_reward_mean` reflects that same term after its configured weight.
                metrics["reward"] = samples.batch["rewards"].mean().item()
                metrics["filter_ratio"] = filter_ratio
                metrics["novelty_penalty_ratio"] = novelty_penalty_ratio
                metrics["molecule_stability"] = samples.batch['stability'].mean().item()
                if "atom_stability" in samples.batch:
                    metrics["atom_stability"] = samples.batch["atom_stability"].mean().item()
                if "valence_underbond" in samples.batch:
                    metrics["valence_underbond"] = samples.batch["valence_underbond"].mean().item()
                if "valence_overbond" in samples.batch:
                    metrics["valence_overbond"] = samples.batch["valence_overbond"].mean().item()
                if "valence_underbond_soft" in samples.batch:
                    metrics["valence_underbond_soft"] = samples.batch["valence_underbond_soft"].mean().item()
                if "valence_overbond_soft" in samples.batch:
                    metrics["valence_overbond_soft"] = samples.batch["valence_overbond_soft"].mean().item()
                if rdkit_validity is not None:
                    metrics["rdkit_validity"] = float(rdkit_validity)
                if rdkit_uniqueness is not None:
                    metrics["rdkit_uniqueness"] = float(rdkit_uniqueness)
                if "rdkit_valid_mask" in samples.batch and "stability" in samples.batch:
                    rdkit_valid_mask = samples.batch["rdkit_valid_mask"].to(samples.batch["stability"].device)
                    valid_denom = rdkit_valid_mask.sum().item()
                    if valid_denom > 0:
                        stable_valid = (samples.batch["stability"] * rdkit_valid_mask).sum().item() / valid_denom
                        metrics["stability_given_rdkit_valid"] = stable_valid
                if rdkit_validity is not None and rdkit_uniqueness is not None:
                    metrics["validity_x_uniqueness"] = float(rdkit_validity) * float(rdkit_uniqueness)
                    metrics["validity_x_uniqueness_x_stability"] = (
                        metrics["validity_x_uniqueness"] * metrics["molecule_stability"]
                    )

                # Collect force and energy reward statistics if available
                if "force_rewards" in samples.batch:
                    metrics["force_reward_mean"] = samples.batch["force_rewards"].mean().item()
                    metrics["force_reward_std"] = samples.batch["force_rewards"].std().item()
                    metrics["force_reward_min"] = samples.batch["force_rewards"].min().item()
                    metrics["force_reward_max"] = samples.batch["force_rewards"].max().item()

                if "energy_rewards" in samples.batch:
                    metrics["energy_reward_mean"] = samples.batch["energy_rewards"].mean().item()
                    metrics["energy_reward_std"] = samples.batch["energy_rewards"].std().item()
                    metrics["energy_reward_min"] = samples.batch["energy_rewards"].min().item()
                    metrics["energy_reward_max"] = samples.batch["energy_rewards"].max().item()

                if "weighted_force_rewards" in samples.batch:
                    metrics["weighted_force_reward_mean"] = samples.batch["weighted_force_rewards"].mean().item()

                if "weighted_energy_rewards" in samples.batch:
                    metrics["weighted_energy_reward_mean"] = samples.batch["weighted_energy_rewards"].mean().item()

                # Collect advantage statistics (GRPO normalized rewards)
                if "force_advantages" in samples.batch:
                    metrics["force_advantage_mean"] = samples.batch["force_advantages"].mean().item()
                    metrics["force_advantage_std"] = samples.batch["force_advantages"].std().item()

                if "energy_advantages" in samples.batch:
                    metrics["energy_advantage_mean"] = samples.batch["energy_advantages"].mean().item()
                    metrics["energy_advantage_std"] = samples.batch["energy_advantages"].std().item()

                if "advantages" in samples.batch:
                    metrics["advantage_mean"] = samples.batch["advantages"].mean().item()
                    metrics["advantage_std"] = samples.batch["advantages"].std().item()

                metrics = self._sync_metrics(metrics)

                # Save checkpoints.
                #
                # - Always persist a best checkpoint immediately when the tracked metric improves.
                #   This avoids missing short-lived spikes in noisy RL metrics (e.g., validity×uniqueness).
                # - Also persist periodic "latest" checkpoints for recovery and monitoring.
                metric_value = metrics.get(self.best_checkpoint_metric) if isinstance(metrics, dict) else None
                saved_this_epoch = False
                if metric_value is not None:
                    try:
                        metric_value_f = float(metric_value)
                    except (TypeError, ValueError):
                        metric_value_f = None
                    if metric_value_f is not None:
                        is_better = (
                            metric_value_f > self.best_metric_value
                            if self.best_checkpoint_mode == "max"
                            else metric_value_f < self.best_metric_value
                        )
                        if is_better:
                            self.save_checkpoint(epoch, metrics)
                            saved_this_epoch = True
                if not saved_this_epoch and (epoch + 1) % save_interval == 0:
                    self.save_checkpoint(epoch, metrics)

                global_batch_idx += 1
                # Log metrics to wandb if enabled
                if self.wandb_enabled:
                    # Combine all metrics into a single log call
                    log_dict = {
                        "train/reward": metrics["reward"],
                        "train/filter_ratio": metrics["filter_ratio"],
                        "train/novelty_penalty_ratio": metrics["novelty_penalty_ratio"],
                        "train/molecule_stability": metrics["molecule_stability"],
                        "train/epoch": metrics["epoch"],
                        "train/step": global_batch_idx,
                    }
                    if "atom_stability" in metrics:
                        log_dict["train/atom_stability"] = metrics["atom_stability"]
                    if "valence_underbond" in metrics:
                        log_dict["train/valence_underbond"] = metrics["valence_underbond"]
                    if "valence_overbond" in metrics:
                        log_dict["train/valence_overbond"] = metrics["valence_overbond"]
                    if "valence_underbond_soft" in metrics:
                        log_dict["train/valence_underbond_soft"] = metrics["valence_underbond_soft"]
                    if "valence_overbond_soft" in metrics:
                        log_dict["train/valence_overbond_soft"] = metrics["valence_overbond_soft"]
                    if "rdkit_validity" in metrics:
                        log_dict["train/rdkit_validity"] = metrics["rdkit_validity"]
                    if "rdkit_uniqueness" in metrics:
                        log_dict["train/rdkit_uniqueness"] = metrics["rdkit_uniqueness"]
                    if "stability_given_rdkit_valid" in metrics:
                        log_dict["train/stability_given_rdkit_valid"] = metrics["stability_given_rdkit_valid"]
                    if "validity_x_uniqueness" in metrics:
                        log_dict["train/validity_x_uniqueness"] = metrics["validity_x_uniqueness"]
                    if "validity_x_uniqueness_x_stability" in metrics:
                        log_dict["train/validity_x_uniqueness_x_stability"] = metrics[
                            "validity_x_uniqueness_x_stability"
                        ]
                    if "ForceAlignPenalty" in metrics:
                        log_dict["train/force_alignment_penalty"] = metrics["ForceAlignPenalty"]
                    if "ForceAlignCosine" in metrics:
                        log_dict["train/force_alignment_cosine"] = metrics["ForceAlignCosine"]
                    if "lr" in metrics:
                        log_dict["train/lr"] = metrics["lr"]
                    if "PolicyEpochs" in metrics:
                        log_dict["train/policy_epochs"] = metrics["PolicyEpochs"]

                    # Add force reward statistics
                    if "force_reward_mean" in metrics:
                        log_dict["train/force_reward_mean"] = metrics["force_reward_mean"]
                        log_dict["train/force_reward_std"] = metrics["force_reward_std"]
                        log_dict["train/force_reward_min"] = metrics["force_reward_min"]
                        log_dict["train/force_reward_max"] = metrics["force_reward_max"]

                    # Add energy reward statistics
                    if "energy_reward_mean" in metrics:
                        log_dict["train/energy_reward_mean"] = metrics["energy_reward_mean"]
                        log_dict["train/energy_reward_std"] = metrics["energy_reward_std"]
                        log_dict["train/energy_reward_min"] = metrics["energy_reward_min"]
                        log_dict["train/energy_reward_max"] = metrics["energy_reward_max"]

                    # Add weighted reward statistics
                    if "weighted_force_reward_mean" in metrics:
                        log_dict["train/weighted_force_reward_mean"] = metrics["weighted_force_reward_mean"]

                    if "weighted_energy_reward_mean" in metrics:
                        log_dict["train/weighted_energy_reward_mean"] = metrics["weighted_energy_reward_mean"]

                    # Add advantage statistics (GRPO normalized rewards)
                    if "force_advantage_mean" in metrics:
                        log_dict["train/force_advantage_mean"] = metrics["force_advantage_mean"]
                        log_dict["train/force_advantage_std"] = metrics["force_advantage_std"]

                    if "energy_advantage_mean" in metrics:
                        log_dict["train/energy_advantage_mean"] = metrics["energy_advantage_mean"]
                        log_dict["train/energy_advantage_std"] = metrics["energy_advantage_std"]

                    if "advantage_mean" in metrics:
                        log_dict["train/advantage_mean"] = metrics["advantage_mean"]
                        log_dict["train/advantage_std"] = metrics["advantage_std"]

                    # Add any additional metrics from actor update
                    for k, v in metrics.items():
                        if k not in [
                            "reward",
                            "filter_ratio",
                            "novelty_penalty_ratio",
                            "molecule_stability",
                            "atom_stability",
                            "valence_underbond",
                            "valence_overbond",
                            "valence_underbond_soft",
                            "valence_overbond_soft",
                            "rdkit_validity",
                            "rdkit_uniqueness",
                            "validity_x_uniqueness",
                            "validity_x_uniqueness_x_stability",
                            "force_reward_mean",
                            "force_reward_std",
                            "force_reward_min",
                            "force_reward_max",
                            "energy_reward_mean",
                            "energy_reward_std",
                            "energy_reward_min",
                            "energy_reward_max",
                            "weighted_force_reward_mean",
                            "weighted_energy_reward_mean",
                            "force_advantage_mean",
                            "force_advantage_std",
                            "energy_advantage_mean",
                            "energy_advantage_std",
                            "advantage_mean",
                            "advantage_std",
                        ]:
                            log_dict[f"train/{k}"] = v
                    wandb.log(log_dict, step=global_batch_idx)

                if self.is_main_process:
                    print(metrics)

                now = time.monotonic()
                if max_time_seconds is not None and (now - start_time) >= max_time_seconds:
                    if self.is_main_process:
                        elapsed_hours = (now - start_time) / 3600.0
                        print(f"Reached train.max_time_hours ({elapsed_hours:.2f}h); stopping training.")
                    break

                if early_stop_metric:
                    metric_value = metrics.get(early_stop_metric) if isinstance(metrics, dict) else None
                    if metric_value is not None:
                        if best_early_stop_value is None:
                            best_early_stop_value = metric_value
                            last_improvement_time = now
                        else:
                            is_better = (
                                metric_value > best_early_stop_value + early_stop_min_delta
                                if early_stop_mode == "max"
                                else metric_value < best_early_stop_value - early_stop_min_delta
                            )
                            if is_better:
                                best_early_stop_value = metric_value
                                last_improvement_time = now
                            elif (
                                early_stop_patience_seconds is not None
                                and (now - last_improvement_time) >= early_stop_patience_seconds
                            ):
                                if self.is_main_process:
                                    elapsed_minutes = (now - last_improvement_time) / 60.0
                                    print(
                                        f"Early stopping: '{early_stop_metric}' has not improved for "
                                        f"{elapsed_minutes:.1f} min (best={best_early_stop_value:.4f})."
                                    )
                                break

            # Save final checkpoint.
            if metrics is not None and last_epoch is not None:
                self.save_checkpoint(last_epoch, metrics)
                if self.is_main_process:
                    print(f"Saved checkpoint at epoch {last_epoch}")
            
        except KeyboardInterrupt:
            if self.is_main_process:
                print("Training interrupted by user")
            # Save checkpoint on interruption
            if metrics is not None and last_epoch is not None:
                self.save_checkpoint(last_epoch, metrics)

        self._export_lr_history()
        return self.model
    
    def _export_lr_history(self):
        """Persist learning rate history and optional plot for visualization."""
        lr_history = getattr(self.actor, "lr_history", None)
        if not lr_history:
            return

        os.makedirs(self.save_path, exist_ok=True)
        csv_path = os.path.join(self.save_path, "learning_rate.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("step,lr\n")
            for idx, value in enumerate(lr_history):
                f.write(f"{idx},{value}\n")

        try:
            import matplotlib.pyplot as plt  # type: ignore

            plt.figure(figsize=(8, 4))
            plt.plot(range(len(lr_history)), lr_history, linewidth=2.0)
            plt.xlabel("Update step")
            plt.ylabel("Learning rate")
            plt.title("Learning Rate Schedule")
            plt.grid(True, alpha=0.3)
            plot_path = os.path.join(self.save_path, "learning_rate.png")
            plt.tight_layout()
            plt.savefig(plot_path, dpi=200)
            plt.close()
        except ImportError:
            if self.is_main_process:
                print("matplotlib not available; skipped saving learning rate plot.")
        except Exception as exc:
            if self.is_main_process:
                print(f"Failed to save learning rate plot: {exc}")
    
    def clean_up(self):
        """Clean up resources"""
        # Shutdown Ray
        if ray.is_initialized():
            ray.shutdown()
            
        # Finish wandb run if enabled
        if self.wandb_enabled:
            wandb.finish() 
