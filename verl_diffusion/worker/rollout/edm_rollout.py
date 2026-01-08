from verl_diffusion.protocol import DataProto, TensorDict
from .base import BaseRollout
import torch
import queue
import threading
import time

class EDMRollout(BaseRollout):
    def __init__(self, model, config):
        """
        Initialize the EDM rollout worker.
        
        Args:
            model: The EDM model to use for rollout
            config: Configuration parameters for the rollout
        """
        super().__init__()  # Call parent constructor with no arguments
        self.model = model
        self.config = config
        self.output_queue = queue.Queue(maxsize=config.get("queue_size", 5))
        self.running = False
        self.thread = None
        dist_cfg = self.config.get("distributed") or {}
        self.is_main_process = bool(dist_cfg.get("is_main_process", True))
        train_cfg = self.config.get("train", {})
        raw_weight = float(train_cfg.get("force_alignment_weight", 0.0))
        enabled_cfg = train_cfg.get("force_alignment_enabled")
        if enabled_cfg is None:
            self.force_alignment_enabled = raw_weight > 0.0
        else:
            self.force_alignment_enabled = bool(enabled_cfg)
        self.force_alignment_weight = raw_weight if self.force_alignment_enabled else 0.0

        reward_cfg = self.config.get("reward") or {}
        shaping_cfg = reward_cfg.get("shaping", {}) if isinstance(reward_cfg, dict) else {}
        scheduler_cfg = {}
        if isinstance(shaping_cfg, dict):
            scheduler_cfg = shaping_cfg.get("scheduler", {}) or {}
        skip_default = shaping_cfg.get("skip_prefix", 0) if isinstance(shaping_cfg, dict) else 0
        skip_value = scheduler_cfg.get("skip_prefix", skip_default)
        try:
            self.skip_prefix = max(0, int(skip_value))
        except Exception:
            self.skip_prefix = 0

        # Prefer an explicit shared-prefix setting under model.* so callers can
        # decouple training truncation from reward shaping schedules.
        model_cfg = self.config.get("model") or {}
        if isinstance(model_cfg, dict) and model_cfg.get("skip_prefix") is not None:
            try:
                self.skip_prefix = max(0, int(model_cfg.get("skip_prefix")))
            except Exception:
                pass
        
        
    def start_async(self, prompts_queue):
        """
        Start asynchronous generation of samples.
        
        Args:
            prompts_queue: Queue containing prompt batches to process
        """
        if self.running:
            return
            
        self.running = True
        self.thread = threading.Thread(
            target=self._async_generation_loop,
            args=(prompts_queue,),
            daemon=True
        )
        self.thread.start()
        
    def stop_async(self):
        """Stop the asynchronous generation thread."""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=10)
            self.thread = None
            
    def _async_generation_loop(self, prompts_queue):
        """
        Continuously generate samples from prompts in the queue.
        
        Args:
            prompts_queue: Queue containing prompt batches to process
        """
        while self.running:
            try:
                # Try to get a prompt batch from the queue with a timeout
                prompts = prompts_queue.get(timeout=0.1)
                
                # Generate samples for this batch
                result = self.generate_minibatch(prompts)
                
                # Put the result in the output queue
                self.output_queue.put(result)
                
                # Mark task as done
                prompts_queue.task_done()
            except queue.Empty:
                # No prompts available, sleep briefly
                time.sleep(0.01)
            except Exception as e:
                # Log any other exceptions
                if self.is_main_process:
                    print(f"Error in async generation: {e}")
                
    def generate_samples(self, prompts: DataProto) -> DataProto:
        batch_size = prompts.batch.batch_size[0]
        # Honor nested config: dataloader.micro_batch_size
        micro_cfg = (self.config.get("dataloader") or {}).get("micro_batch_size", self.config.get("micro_batch_size", batch_size))
        try:
            micro_bs = int(micro_cfg)
        except Exception:
            micro_bs = batch_size
        micro_bs = max(1, micro_bs)

        # Process in micro-batches to control memory footprint.
        results = []
        for start in range(0, batch_size, micro_bs):
            chunk = prompts[start : start + micro_bs]
            results.append(self.generate_minibatch(chunk))

        if len(results) == 1:
            return results[0]
        return DataProto.concat(results)
    
    def generate_minibatch(self, prompts: DataProto) -> DataProto:       
        # Extract necessary data from prompts for the model
        # This will depend on what data is needed by the model.sample() method
        batch_size = prompts.batch.batch_size[0]
        max_n_nodes = prompts.meta_info["max_n_nodes"]
        # we need to extract node information from the prompts
        nodesxsample = prompts.batch['nodesxsample']
        model_ref = self.model.module if hasattr(self.model, "module") else self.model
        node_mask, edge_mask = model_ref.get_mask(nodesxsample, batch_size, max_n_nodes)
        node_mask = node_mask.to(nodesxsample.device)
        edge_mask = edge_mask.to(nodesxsample.device)
        n_samples = batch_size
        n_nodes = max_n_nodes  # node_mask shape is [batch_size, n_nodes]
        model_cfg = self.config.get("model") or {}
        return_suffix_only = bool(model_cfg.get("return_suffix_only", False))
        share_prefix = bool(model_cfg.get("share_initial_noise", False) and self.skip_prefix > 0)
        # Call the model's sample method
        # Delegate shared-prefix handling to the model so each group reuses the
        # same prefix latents before branching into independent continuations.
        x, h, latents, logps, timesteps, mus, sigmas, z0_preds = model_ref.sample(
            n_samples=n_samples,
            n_nodes=n_nodes,
            node_mask=node_mask,
            edge_mask=edge_mask,
            timestep=self.config["model"]["time_step"],
            group_index=prompts.batch["group_index"],
            share_initial_noise=self.config["model"].get("share_initial_noise", False),
            skip_prefix=self.skip_prefix,
            return_suffix_only=return_suffix_only,
        )
        device = x.device

        effective_skip_prefix = 0 if (return_suffix_only and share_prefix) else self.skip_prefix

        latents_tensor = torch.stack(latents, dim=1)
        logps_tensor = torch.stack(logps, dim=1)
        mus_tensor = None
        if self.force_alignment_enabled and self.force_alignment_weight > 0.0:
            mus_tensor = torch.stack(mus, dim=1)
        z0_preds_tensor = torch.stack(z0_preds, dim=1)

        # Convert timesteps to tensor and expand to batch_size
        timesteps_tensor = torch.tensor(timesteps, device=device)
        # Expand timesteps to have the same batch size as the other tensors
        # Shape will be [batch_size, num_timesteps]
        expanded_timesteps = timesteps_tensor.unsqueeze(0).repeat([batch_size,1])

        # Create a dictionary with the results
        batch_data = {
            "x": x,
            "categorical": h["categorical"],
            "latents": latents_tensor,
            "z0_preds": z0_preds_tensor,
            "logps": logps_tensor,
            "nodesxsample": nodesxsample,
            "timesteps": expanded_timesteps,
            "group_index": prompts.batch["group_index"],
        }
        if mus_tensor is not None:
            batch_data["mus"] = mus_tensor

        meta_info = prompts.meta_info.copy()
        meta_info["force_alignment_enabled"] = self.force_alignment_enabled and self.force_alignment_weight > 0.0
        meta_info["skip_prefix"] = effective_skip_prefix
        meta_info["original_skip_prefix"] = self.skip_prefix
        meta_info["share_initial_noise"] = bool(self.config["model"].get("share_initial_noise", False))

        return DataProto(batch=TensorDict(batch_data, batch_size= batch_size), meta_info=meta_info)
    
    def get_next_sample(self, timeout=None):
        """
        Get the next generated sample from the output queue.
        
        Args:
            timeout: Maximum time to wait for a sample
            
        Returns:
            Generated sample or None if timeout occurs
        """
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    
