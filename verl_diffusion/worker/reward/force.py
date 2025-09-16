import logging
import os
import queue
import threading
import time
from typing import Optional

from .base import BaseReward
os.environ["OMP_NUM_THREADS"] = "18"
import numpy as np
import torch
import ray
from tqdm import tqdm as tq

from xtb.ase.calculator import XTB
from ase import Atoms
from verl_diffusion.utils.math import kl_divergence_normal, rmsd
from Model.EDM.qm9.analyze import check_stability

from edm_source.mlff_modules.mlff_force_computer import MLFFForceComputer
from edm_source.mlff_modules.mlff_utils import (
    apply_force_clipping,
    get_mlff_predictor,
)

from verl_diffusion.protocol import DataProto, TensorDict


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
    """Reward module that scores molecules with UMA MLFF forces without altering ForceReward."""

    def __init__(
        self,
        dataset_info: dict,
        condition: bool = False,
        mlff_model: str = "uma-s-1p1",
        mlff_predictor: Optional[object] = None,
        position_scale: Optional[float] = None,
        force_clip_threshold: Optional[float] = None,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.dataset_info = dataset_info
        self.condition = condition
        self.input_queue = queue.Queue(maxsize=512)
        self.output_queue = queue.Queue()
        self.running = False
        self.thread = None

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if position_scale is None:
            norm_values = self.dataset_info.get("normalize_factors")
            if isinstance(norm_values, (list, tuple)) and len(norm_values) > 0:
                position_scale = float(norm_values[0])
            else:
                position_scale = 1.0
        self.position_scale = position_scale
        self.force_clip_threshold = force_clip_threshold

        if mlff_predictor is not None:
            self.mlff_predictor = mlff_predictor
        else:
            self.mlff_predictor = get_mlff_predictor(mlff_model, self.device)

        if self.mlff_predictor is not None:
            self.force_computer = MLFFForceComputer(
                mlff_predictor=self.mlff_predictor,
                position_scale=self.position_scale,
                device=self.device,
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
        nodesxsample = samples.batch["nodesxsample"].long().cpu()
        batch_size, max_n_nodes, _ = positions.shape

        node_mask = torch.zeros(batch_size, max_n_nodes, 1, device=positions.device)
        for idx, n_nodes in enumerate(nodesxsample.tolist()):
            node_mask[idx, :n_nodes] = 1

        features = categorical
        z = torch.cat([positions, features], dim=-1)

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

    def calculate_rewards(self, data: DataProto) -> DataProto:
        processed = self.process_data(data)

        z = processed["z"].to(self.device)
        node_mask = processed["node_mask"].to(self.device)

        if self.force_computer is not None:
            forces = self.force_computer.compute_mlff_forces(z, node_mask, self.dataset_info)
            if self.force_clip_threshold is not None:
                forces, _ = apply_force_clipping(forces, self.force_clip_threshold, node_mask)
        else:
            forces = torch.zeros_like(z[:, :, :3], device=self.device)

        rewards = []
        stability_flags = []

        forces_cpu = forces.detach().cpu()
        node_mask_cpu = processed["node_mask"].cpu()
        positions_cpu = processed["positions"].detach().cpu()
        categorical_cpu = processed["categorical"].detach().cpu()

        for batch_idx in range(processed["batch_size"]):
            mask = node_mask_cpu[batch_idx, :, 0].bool()

            if mask.sum() == 0:
                rewards.append(0.0)
                stability_flags.append(0.0)
                continue

            sample_forces = forces_cpu[batch_idx][mask]
            if sample_forces.numel() == 0:
                rms_force = 0.0
            else:
                rms_force = torch.sqrt(torch.mean(sample_forces.pow(2))).item()

            rewards.append(-rms_force)

            atom_type_indices = categorical_cpu[batch_idx].argmax(dim=1)
            atom_types = atom_type_indices[mask].tolist()
            positions = positions_cpu[batch_idx][mask].numpy()

            try:
                validity_results = check_stability(np.array(positions), atom_types, self.dataset_info)
                stability_flags.append(float(validity_results[0]))
            except Exception:
                stability_flags.append(0.0)

        result = {
            "rewards": torch.tensor(rewards, dtype=torch.float32),
            "stability": torch.tensor(stability_flags, dtype=torch.float32),
        }

        return DataProto(batch=TensorDict(result, batch_size=[len(rewards)]), meta_info=data.meta_info.copy())
