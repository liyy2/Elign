import logging
import os
import queue
import threading
import time
from typing import Optional, Union

from .base import BaseReward
os.environ["OMP_NUM_THREADS"] = "18"
import numpy as np
import torch
import ray
from tqdm import tqdm as tq


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

try:
    from xtb.ase.calculator import XTB
except ModuleNotFoundError:
    if _is_main_process():
        print('Not importing xtb.')
from ase import Atoms
from verl_diffusion.utils.math import kl_divergence_normal, rmsd
from edm_source.qm9.analyze import check_stability

from edm_source.mlff_modules.mlff_force_computer import MLFFForceComputer
from edm_source.mlff_modules.mlff_utils import get_mlff_predictor

from verl_diffusion.protocol import DataProto, TensorDict
from .scheduler import RewardScheduler


logger = logging.getLogger(__name__)

@ray.remote(num_cpus=8)
def calcuate_xtb_force(mol, calc, dataset_info, atom_encoder):
    pos = mol[0].tolist()
    atom_type = mol[1].tolist()
    validity_results = check_stability(np.array(pos), atom_type, dataset_info)
    atom_type = [atom_encoder[atom] for atom in atom_type]
    atoms = Atoms(symbols=atom_type, positions=pos)
    atoms.calc = calc
    try:
        forces = atoms.get_forces()
        mean_abs_forces = rmsd(forces)
    except:
        mean_abs_forces = 5.0
    return -1 * mean_abs_forces, float(validity_results[0])



class ForceReward(BaseReward):
    def __init__(self, dataset_info:dict, condition:bool=False):
        super().__init__()
        self.is_main_process = _is_main_process()
        self.calc = XTB(method="GFN2-xTB")
        self.dataset_info = dataset_info
        self.atom_encoder = self.dataset_info['atom_decoder']
        self.condition = condition
        self.input_queue = queue.Queue(maxsize=512)
        self.output_queue = queue.Queue()
        self.running = False
        self.thread = None
        
    def start_async(self):
        """Start asynchronous reward calculation thread"""
        if self.running:
            return
            
        self.running = True
        self.thread = threading.Thread(
            target=self._async_reward_loop,
            daemon=True
        )
        self.thread.start()
        
    def stop_async(self):
        """Stop the asynchronous reward calculation thread"""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=10)
            self.thread = None
    
    def _async_reward_loop(self):
        """Continuously calculate rewards for samples in the input queue"""
        while self.running:
            try:
                # Try to get a sample from the input queue
                sample = self.input_queue.get(timeout=0.1)
                
                # Calculate rewards for this sample
                result = self.calculate_rewards(sample)
                
                # Put the results in the output queue
                self.output_queue.put((sample, result))
                
                # Mark task as done
                self.input_queue.task_done()
            except queue.Empty:
                # No samples available, sleep briefly
                time.sleep(0.01)
            except Exception as e:
                # Log any other exceptions
                if self.is_main_process:
                    print(f"Error in async reward calculation: {e}")
                
    def submit_batch(self, samples):
        """
        Submit a batch of samples for reward calculation
        
        Args:
            samples: List of samples to calculate rewards for
            
        Returns:
            List of reward results
        """
        results = []
        result = self.calculate_rewards(samples)
        results.append(result)
        return results
                
    def submit_sample(self, sample):
        """
        Submit a sample for asynchronous reward calculation
        
        Args:
            sample: The sample to calculate rewards for
        
        Returns:
            bool: True if the sample was submitted, False if the queue is full
        """
        try:
            self.input_queue.put_nowait(sample)
            return True
        except queue.Full:
            return False
            
    def get_next_result(self, timeout=None):
        """
        Get the next reward calculation result from the output queue
        
        Args:
            timeout: Maximum time to wait for a result
            
        Returns:
            Tuple of (sample, result) or None if timeout occurs
        """
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def process_data(self, samples:DataProto) -> list:
        """
        Process the DataProto object to prepare it for force calculation.

        Args:
            samples (DataProto): A DataProto object containing the data to process.
            
        Returns:
            list: A list of processed molecule tuples (position, atom_type)
        """
        
        one_hot = samples.batch["categorical"]
        x = samples.batch['x']
        nodesxsample = samples.batch["nodesxsample"]
        node_mask = torch.zeros(x.shape[0], self.dataset_info['max_n_nodes'])
        
        for i in range(x.shape[0]):
            node_mask[i, 0:nodesxsample[i]] = 1
        n_samples = len(x)
        processed_list = []
        
        for i in range(n_samples):
            atom_type = one_hot[i].argmax(1).cpu().detach()
            pos = x[i].cpu().detach()
            atom_type = atom_type[0:int(nodesxsample[i])]
            pos = pos[0:int(nodesxsample[i])]
            if self.condition:
                processed_list.append((pos, atom_type, samples.batch["context"][i][0].cpu().detach()))
            else:
                processed_list.append((pos, atom_type))
        return processed_list
        
    def calculate_rewards(self, data: DataProto) -> DataProto:
        """
        Calculate the force reward for a given DataProto object.

        Args:
            data (DataProto): A DataProto object containing the data to calculate the reward for.
            
        Returns:
            DataProto: A DataProto object containing the original data and rewards
        """
        processed_list = self.process_data(data)
        rewards = []
        molecule_stable = []
        futures = [calcuate_xtb_force.remote(mol, self.calc, self.dataset_info, self.atom_encoder) for mol in processed_list]
        outputs_list = ray.get(futures)
        
        for i in outputs_list:
            rewards.append(i[0])
            molecule_stable.append(i[1])
            
        result = {}
        result['rewards'] = torch.tensor(rewards)
        result['stability'] = torch.tensor(molecule_stable)
        
        return DataProto(batch=TensorDict(result, batch_size=[len(rewards)]), meta_info=data.meta_info.copy())


class UMAForceReward(BaseReward):
    """Reward module that scores molecules with UMA MLFF forces and supports intermediate reward shaping."""

    def __init__(
        self,
        dataset_info: dict,
        condition: bool = False,
        mlff_model: str = "uma-s-1p1",
        mlff_predictor: Optional[object] = None,
        position_scale: Optional[float] = None,
        force_clip_threshold: Optional[float] = None,
        device: Optional[Union[str, torch.device]] = None,
        mlff_device: Optional[str] = None,
        shaping: Optional[dict] = None,
        use_energy: bool = False,
        force_weight: float = 1.0,
        energy_weight: float = 1.0,
        force_aggregation: str = "rms",
        energy_transform_offset: float = 10000.0,
        energy_transform_scale: float = 1000.0,
    ):
        super().__init__()
        self.is_main_process = _is_main_process()
        self.dataset_info = dataset_info
        self.condition = condition
        self.input_queue = queue.Queue(maxsize=512)
        self.output_queue = queue.Queue()
        self.running = False
        self.thread = None

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, torch.device):
            self.device = device
        else:
            self.device = torch.device(device)
        self.mlff_device = mlff_device or ("cuda" if self.device.type == "cuda" else "cpu")
        if position_scale is None:
            norm_values = self.dataset_info.get("normalize_factors")
            if isinstance(norm_values, (list, tuple)) and len(norm_values) > 0:
                position_scale = float(norm_values[0])
            else:
                position_scale = 1.0
        self.position_scale = position_scale
        self.force_clip_threshold = force_clip_threshold
        
        # Energy reward configuration
        self.use_energy = use_energy
        self.force_weight = force_weight
        self.energy_weight = energy_weight
        force_aggregation = (force_aggregation or "rms").lower()
        if force_aggregation not in {"rms", "max"}:
            raise ValueError(f"Unsupported force aggregation '{force_aggregation}'. Use 'rms' or 'max'.")
        self.force_aggregation = force_aggregation
        self.energy_transform_offset = energy_transform_offset
        self.energy_transform_scale = energy_transform_scale
        

        # Reward shaping config
        # Scheduler allows switching between uniform and adaptive sampling of diffusion steps.
        self.shaping_cfg = shaping or {}
        self.shaping_enabled = bool(self.shaping_cfg.get("enabled", False))
        self.shaping_method = self.shaping_cfg.get("method", "potential")
        self.shaping_gamma = float(self.shaping_cfg.get("gamma", 1.0))
        # Compute MLFF only for the last K timesteps to control cost; 0 -> all steps
        self.shaping_horizon = int(self.shaping_cfg.get("horizon", 0))
        default_skip_prefix = int(self.shaping_cfg.get("skip_prefix", 0))
        # Weight for adding shaped sum to final reward
        self.shaping_weight = float(self.shaping_cfg.get("weight", 1.0))
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
        self.terminal_reward_weight = max(0.0, float(self.shaping_cfg.get("terminal_weight", 1.5)))


        if mlff_predictor is not None:
            self.mlff_predictor = mlff_predictor
        else:
            self.mlff_predictor = get_mlff_predictor(mlff_model, self.mlff_device)

        if self.mlff_predictor is not None:
            self.force_computer = MLFFForceComputer(
                mlff_predictor=self.mlff_predictor,
                position_scale=self.position_scale,
                device=self.device,
                compute_energy=self.use_energy,  # Enable energy computation based on config
            )
        else:
            self.force_computer = None
            logger.warning("UMAForceReward: MLFF predictor not loaded, rewards will default to zero.")

    def start_async(self):
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._async_reward_loop, daemon=True)
        self.thread.start()

    def stop_async(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=10)
            self.thread = None

    def _async_reward_loop(self):
        while self.running:
            try:
                sample = self.input_queue.get(timeout=0.1)
                result = self.calculate_rewards(sample)
                self.output_queue.put((sample, result))
                self.input_queue.task_done()
            except queue.Empty:
                time.sleep(0.01)
            except Exception as exc:  # pragma: no cover - safeguard
                logger.error(f"UMAForceReward async error: {exc}")

    def submit_batch(self, samples):
        results = []
        result = self.calculate_rewards(samples)
        results.append(result)
        return results

    def submit_sample(self, sample):
        try:
            self.input_queue.put_nowait(sample)
            return True
        except queue.Full:
            return False

    def get_next_result(self, timeout=None):
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None

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
        processed = self.process_data(data)

        z = processed["z"].to(self.device)
        node_mask = processed["node_mask"].to(self.device)
        positions = processed["positions"].to(self.device)
        categorical = processed["categorical"].to(self.device)
        batch_size = processed["batch_size"]

        # Decide shaping path: if enabled, compute terminal + shaped in one MLFF pass over latents
        shaping_active = self.shaping_enabled and ("latents" in data.batch.keys()) and (self.force_computer is not None)

        force_rewards = torch.zeros(batch_size, device=self.device)
        energy_rewards = torch.zeros(batch_size, device=self.device)
        stability_flags = torch.zeros(batch_size, device=self.device)
        result = {}


        if shaping_active:
            latents = data.batch["latents"]
            if "timesteps" in data.batch.keys():
                T_steps = data.batch["timesteps"].shape[1]
                latents = latents[:, :T_steps]
            else:
                T_steps = latents.shape[1]

            B, S, N, D = latents.shape
            node_mask_b = processed["node_mask"].to(self.device)
            F_force = torch.zeros((B, S), device=self.device)
            F_energy = torch.zeros((B, S), device=self.device) if self.use_energy else None

            if self.shaping_horizon and self.shaping_horizon > 0:
                horizon_start = max(0, S - self.shaping_horizon)
            else:
                horizon_start = 0

            if "timesteps" in data.batch.keys():
                timesteps_schedule = data.batch["timesteps"][0, :S].detach().cpu()
            else:
                timesteps_schedule = torch.linspace(float(S - 1), 0.0, steps=S)

            # Select the diffusion steps that will receive MLFF calls and potential shaping.
            schedule = self.reward_scheduler.build(
                timesteps=timesteps_schedule,
                start_idx=horizon_start,
                end_idx=S - 1,
            )
            schedule_indices = schedule.indices.to(latents.device)
            interval_steps = schedule.intervals.to(latents.device)
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
                transformed_energies = (energies_bt + self.energy_transform_offset) / self.energy_transform_scale
                F_energy.index_copy_(1, schedule_indices, -transformed_energies)

            if alignment_active and force_vectors_flat is not None:
                force_vectors_selected = force_vectors_flat.view(B, selected_count, N, 3)
            else:
                force_vectors_selected = None

            force_rewards = F_force[:, S - 1]
            if self.use_energy:
                energy_rewards = F_energy[:, S - 1]
            else:
                energy_rewards = torch.zeros_like(force_rewards)

            shaped_force = torch.zeros_like(F_force)
            shaped_energy = torch.zeros_like(F_force)
            # Vectorized potential-shaping over the scheduled indices, accounting for stride length.
            selected_count = int(schedule_indices.numel())
            if selected_count > 0:
                last_idx = schedule_indices[-1]
                shaped_force[:, last_idx] = F_force[:, last_idx]
                if self.use_energy and F_energy is not None:
                    shaped_energy[:, last_idx] = F_energy[:, last_idx]

                if selected_count > 1:
                    idx_current = schedule_indices[:-1]
                    idx_next = schedule_indices[1:]
                    intervals = interval_steps[:-1].clamp(min=1)
                    intervals_f = intervals.to(F_force.dtype)
                    gamma_factor = torch.pow(
                        torch.full_like(intervals_f, self.shaping_gamma),
                        intervals_f,
                    )

                    force_current = torch.index_select(F_force, 1, idx_current)
                    force_next = torch.index_select(F_force, 1, idx_next)
                    shaped_vals = intervals_f.unsqueeze(0) * (
                        gamma_factor.unsqueeze(0) * force_next - force_current
                    )
                    shaped_force.index_copy_(1, idx_current, shaped_vals)

                    if self.use_energy and F_energy is not None:
                        energy_current = torch.index_select(F_energy, 1, idx_current)
                        energy_next = torch.index_select(F_energy, 1, idx_next)
                        shaped_energy_vals = intervals_f.unsqueeze(0) * (
                            gamma_factor.unsqueeze(0) * energy_next - energy_current
                        )
                        shaped_energy.index_copy_(1, idx_current, shaped_energy_vals)

            weighted_force_rewards = self.force_weight * self.terminal_reward_weight * force_rewards
            weighted_energy_rewards = self.energy_weight * self.terminal_reward_weight * energy_rewards
            rewards = weighted_force_rewards + weighted_energy_rewards

            weighted_shaped_force = self.force_weight * shaped_force
            weighted_shaped_energy = self.energy_weight * shaped_energy if self.use_energy else torch.zeros_like(shaped_force)
            shaped = weighted_shaped_force + weighted_shaped_energy

            result["rewards_ts"] = shaped.detach().cpu()
            result["force_rewards_ts"] = shaped_force.detach().cpu()
            if self.use_energy:
                result["energy_rewards_ts"] = shaped_energy.detach().cpu()
            shaped_sum = shaped.sum(dim=1)
            result["rewards"] = (rewards + self.shaping_weight * shaped_sum).detach().cpu()

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
                if self.use_energy:
                    forces, energies = self.force_computer.compute_mlff_forces(z, node_mask, self.dataset_info)
                else:
                    forces = self.force_computer.compute_mlff_forces(z, node_mask, self.dataset_info)
                    energies = torch.zeros(z.shape[0], device=self.device)
            else:
                forces = torch.zeros_like(z[:, :, :3], device=self.device)
                energies = torch.zeros(z.shape[0], device=self.device)

            for batch_idx in tq(
                range(batch_size),
                desc="UMAForce terminal reward",
                leave=False,
                disable=batch_size <= 1,
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
                    transformed_energy = (energy + self.energy_transform_offset) / self.energy_transform_scale
                    energy_rewards[batch_idx] = -transformed_energy

            weighted_force_rewards = self.force_weight * self.terminal_reward_weight * force_rewards
            weighted_energy_rewards = self.energy_weight * self.terminal_reward_weight * energy_rewards
            rewards = weighted_force_rewards + weighted_energy_rewards
            result["rewards"] = rewards.detach().cpu()

        # Compute stability flags once
        for batch_idx in tq(
            range(batch_size),
            desc="UMAForce stability",
            leave=False,
            disable=batch_size <= 1,
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
                stability_flags[batch_idx] = float(validity_results[0])
            except Exception:
                stability_flags[batch_idx] = 0.0

        # Populate final fields
        result.setdefault("rewards", rewards.detach().cpu())
        result["force_rewards"] = force_rewards.detach().cpu()
        result["energy_rewards"] = energy_rewards.detach().cpu()
        result["weighted_force_rewards"] = weighted_force_rewards.detach().cpu()
        result["weighted_energy_rewards"] = weighted_energy_rewards.detach().cpu()
        result["stability"] = stability_flags.cpu()

        if self.is_main_process:
            print("Rewards calculated via UMAForceReward")
        return DataProto(batch=TensorDict(result, batch_size=[batch_size]), meta_info=data.meta_info.copy())
