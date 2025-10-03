import torch
import queue
import time
import threading
import ray
from tqdm import tqdm
import wandb
import os
import yaml
from verl_diffusion.trainer.base import BaseTrainer
from verl_diffusion.protocol import DataProto


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
        if "save_path" in config:
            self.save_path = config["save_path"]
        else:
            self.save_path = os.path.join("./exp", config["wandb"]["wandb_name"])
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
        self.best_reward = float('-inf')
        
        # Initialize Ray for parallel processing
        if not ray.is_initialized():
            ray.init()
            
        # Initialize wandb if enabled in config
        if config.get("wandb", False):
            wandb.init(
                project=config["wandb"].get("wandb_project", "edm-ddp"),
                name=config["wandb"].get("wandb_name", "edm-ddp-run"),
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
            except Exception as e:
                print(f"Error in reward calculation: {e}")
                self.samples_queue.task_done()
    
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
            num_chunks = max(batch_size // self.config.get("micro_batch_size", batch_size), 1)
            
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
            print(f"Error processing batch: {e}")
            # Send termination signal (if not already sent)
            try:
                self.samples_queue.put_nowait(None)
            except queue.Full:
                pass
            
            # Wait for reward thread to complete
            if reward_thread.is_alive():
                reward_thread.join(timeout=5)
            
            return []
    def compute_advantage(self, samples):
        group_index = samples.batch["group_index"]
        clip_value = self.config["train"].get("clip_advantage_value", 5.0)

        # Per-timestep rewards available (reward shaping)
        if "rewards_ts" in samples.batch.keys():
            rewards_ts = samples.batch["rewards_ts"].to(group_index.device)  # [B, S]
            unique_groups = torch.unique(group_index)
            advantages_ts = torch.zeros_like(rewards_ts)
            
            # Check if we have separate force and energy rewards for GRPO
            if "force_rewards_ts" in samples.batch.keys() and "energy_rewards_ts" in samples.batch.keys():
                force_rewards_ts = samples.batch["force_rewards_ts"].to(group_index.device)  # [B, S]
                energy_rewards_ts = samples.batch["energy_rewards_ts"].to(group_index.device)  # [B, S]
                
                force_advantages_ts = torch.zeros_like(force_rewards_ts)
                energy_advantages_ts = torch.zeros_like(energy_rewards_ts)
                
                # Normalize force and energy separately within each group (GRPO)
                for group_id in unique_groups:
                    gmask = (group_index == group_id)
                    if gmask.any():
                        # Normalize force rewards
                        grp_force = force_rewards_ts[gmask]  # [B_g, S]
                        mean_force = grp_force.mean(dim=0, keepdim=True)  # [1, S]
                        std_force = grp_force.std(dim=0, keepdim=True)
                        force_advantages_ts[gmask] = (grp_force - mean_force) / (std_force + 1e-8)
                        
                        # Normalize energy rewards
                        grp_energy = energy_rewards_ts[gmask]  # [B_g, S]
                        mean_energy = grp_energy.mean(dim=0, keepdim=True)  # [1, S]
                        std_energy = grp_energy.std(dim=0, keepdim=True)
                        energy_advantages_ts[gmask] = (grp_energy - mean_energy) / (std_energy + 1e-8)
                
                # Combine normalized force and energy advantages
                advantages_ts = force_advantages_ts + energy_advantages_ts
                
                # Store separate advantages for logging/analysis if needed
                samples.batch["force_advantages_ts"] = torch.clamp(force_advantages_ts, -clip_value, clip_value)
                samples.batch["energy_advantages_ts"] = torch.clamp(energy_advantages_ts, -clip_value, clip_value)
            else:
                # Standard normalization for combined rewards
                for group_id in unique_groups:
                    gmask = (group_index == group_id)
                    if gmask.any():
                        grp = rewards_ts[gmask]  # [B_g, S]
                        mean = grp.mean(dim=0, keepdim=True)  # [1, S]
                        std = grp.std(dim=0, keepdim=True)
                        advantages_ts[gmask] = (grp - mean) / (std + 1e-8)

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
                group_mask = (group_index == group_id)
                if group_mask.any():
                    # Normalize force rewards
                    group_force = force_rewards[group_mask]
                    force_mean = group_force.mean()
                    force_std = group_force.std()
                    force_advantages[group_mask] = (group_force - force_mean) / (force_std + 1e-8)
                    
                    # Normalize energy rewards
                    group_energy = energy_rewards[group_mask]
                    energy_mean = group_energy.mean()
                    energy_std = group_energy.std()
                    energy_advantages[group_mask] = (group_energy - energy_mean) / (energy_std + 1e-8)
            
            # Combine normalized advantages
            normalized_rewards = force_advantages + energy_advantages
            
            # Store separate advantages for logging/analysis
            samples.batch["force_advantages"] = torch.clamp(force_advantages, -clip_value, clip_value)
            samples.batch["energy_advantages"] = torch.clamp(energy_advantages, -clip_value, clip_value)
        else:
            # Standard normalization
            for group_id in unique_groups:
                group_mask = (group_index == group_id)
                group_rewards = rewards[group_mask]
                group_mean = group_rewards.mean()
                group_std = group_rewards.std()
                normalized_rewards[group_mask] = (group_rewards - group_mean) / (group_std + 1e-8)

        samples.batch["advantages"] = torch.clamp(normalized_rewards, -clip_value, clip_value)
        return samples

    def save_checkpoint(self, epoch, metrics=None):
        """Save model checkpoint and config"""
        # Save config
        config_path = os.path.join(self.save_path, 'config.yaml')
        if isinstance(self.config, dict):
            with open(config_path, 'w') as f:
                yaml.dump(self.config, f)
        else:
            self.config.to_yaml(config_path)
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'actor_state_dict': self.actor.model.state_dict(),
            'optimizer_state_dict': self.actor.optimizer.state_dict(),
            'metrics': metrics
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, os.path.join(self.save_path, 'checkpoint_latest.pth'))
        
        # Save epoch checkpoint
        torch.save(checkpoint, os.path.join(self.save_path, f'checkpoint_epoch_{epoch}.pth'))
        
        # Save best checkpoint if metrics are provided
        if metrics is not None:
            if metrics['reward'] > self.best_reward:
                self.best_reward = metrics['reward']
                
                torch.save(self.model.state_dict(),os.path.join(self.save_path, 'generative_model_ema.npy'))
                torch.save(checkpoint, os.path.join(self.save_path, 'checkpoint_best.pth'))
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.actor.model.load_state_dict(checkpoint['actor_state_dict'])
        self.actor.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'metrics' in checkpoint:
            self.best_reward = checkpoint['metrics'].get('reward', float('-inf'))
        
        return checkpoint.get('epoch', 0)

    def fit(self):
        """
        Training loop for DDPO - processes each dataloader batch and updates model parameters
        
        Overrides BaseTrainer.fit() method
        """
        try:
            all_results = []
            start_epoch = 0
            global_batch_idx = 0
            # Load checkpoint if resuming
            if self.config.get('resume', False) and self.config.get('checkpoint_path'):
                start_epoch = self.load_checkpoint(self.config['checkpoint_path'])
                print(f"Resuming training from epoch {start_epoch}")
            for epoch in range(start_epoch, self.epoches):
                # Process each batch from the dataloader and update model after each batch
                for batch_idx, prompts in enumerate(self.dataloader):
                    # Process the batch
                    samples = self.process_batch(batch_idx, prompts)
                    samples, filter_ratio, novelty_penalty_ratio = self.filters.filter(samples)
                    samples = self.compute_advantage(samples)
                    metrics = self.actor.update_policy(samples)
                    metrics["reward"] = samples.batch["rewards"].mean().item()
                    metrics["filter_ratio"] = filter_ratio
                    metrics["novelty_penalty_ratio"] = novelty_penalty_ratio
                    metrics["molecule_stability"] = samples.batch['stability'].mean().item()
                    
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
                    
                    # Save checkpoint periodically
                    if (batch_idx + 1) % self.config.get('save_interval', 10) == 0:
                        self.save_checkpoint(batch_idx, metrics)
                    global_batch_idx += 1
                    # Log metrics to wandb if enabled
                    if self.wandb_enabled:
                        # Combine all metrics into a single log call
                        log_dict = {
                            "train/reward": metrics["reward"],
                            "train/filter_ratio": metrics["filter_ratio"],
                            "train/novelty_penalty_ratio": metrics["novelty_penalty_ratio"],
                            "train/molecule_stability": metrics["molecule_stability"],
                            "train/step": global_batch_idx
                        }
                        
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
                            if k not in ["reward", "filter_ratio", "novelty_penalty_ratio", "molecule_stability",
                                        "force_reward_mean", "force_reward_std", "force_reward_min", "force_reward_max",
                                        "energy_reward_mean", "energy_reward_std", "energy_reward_min", "energy_reward_max",
                                        "weighted_force_reward_mean", "weighted_energy_reward_mean",
                                        "force_advantage_mean", "force_advantage_std",
                                        "energy_advantage_mean", "energy_advantage_std",
                                        "advantage_mean", "advantage_std"]:
                                log_dict[f"train/{k}"] = v
                        wandb.log(log_dict)
                    
                    print(metrics)
                
                # Save checkpoint at the end of each epoch
                self.save_checkpoint(epoch, metrics)
                print(f"Saved checkpoint at epoch {epoch}")
                
                # Save final checkpoint
                self.save_checkpoint(len(self.dataloader), metrics)
            
        except KeyboardInterrupt:
            print("Training interrupted by user")
            # Save checkpoint on interruption
            if 'metrics' in locals():
                self.save_checkpoint(batch_idx, metrics)
            
        return self.model
    
    def clean_up(self):
        """Clean up resources"""
        # Shutdown Ray
        if ray.is_initialized():
            ray.shutdown()
            
        # Finish wandb run if enabled
        if self.wandb_enabled:
            wandb.finish() 
