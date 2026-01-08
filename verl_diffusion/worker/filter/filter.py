import pickle
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from verl_diffusion.protocol import DataProto

class Filter:
    def __init__(
        self,
        dataset_info,
        file_name,
        condition,
        enable_filtering=True,
        enable_penalty=True,
        penalty_scale=0.1,
        invalid_penalty_scale: float = 0.0,
        duplicate_penalty_scale: float = 0.0,
    ):
        self.dataset_info = dataset_info
        self.file_name = file_name
        self.condition = condition
        self.enable_filtering = bool(enable_filtering)
        self.enable_penalty = bool(enable_penalty)
        self.penalty_scale = float(penalty_scale)
        self.invalid_penalty_scale = float(invalid_penalty_scale or 0.0)
        self.duplicate_penalty_scale = float(duplicate_penalty_scale or 0.0)

        dataset_smiles_list: List[str] = []
        if self.enable_penalty_requires_smiles(self.enable_penalty) and not file_name:
            raise ValueError("filters.enable_penalty=true requires dataloader.smiles_path to be set")
        if self.enable_penalty and file_name:
            with open(file_name, "rb") as f:
                dataset_smiles_list = pickle.load(f)
        self.dataset_smiles = set(dataset_smiles_list)

    @staticmethod
    def enable_penalty_requires_smiles(enable_penalty: bool) -> bool:
        return bool(enable_penalty)

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

    @staticmethod
    def _add_terminal_penalty(data: DataProto, penalty: torch.Tensor) -> None:
        """Add a per-sample terminal penalty across reward tensors.

        Notes on GRPO/DDPO implementation:
        - When both force+energy reward tensors exist, DDPOTrainer computes advantages from
          `force_rewards(_ts)` / `energy_rewards(_ts)` and ignores `rewards(_ts)` for learning.
        - Penalizing only the scalar `rewards` is therefore insufficient when the trainer is using
          separate force/energy channels. To keep the penalty effective regardless of which reward
          channels are present, we add it to *all* available reward tensors.
        """
        if "rewards" in data.batch:
            data.batch["rewards"] = data.batch["rewards"] + penalty
        if "force_rewards" in data.batch:
            data.batch["force_rewards"] = data.batch["force_rewards"] + penalty
        if "energy_rewards" in data.batch:
            data.batch["energy_rewards"] = data.batch["energy_rewards"] + penalty

        if "rewards_ts" in data.batch:
            rewards_ts = data.batch["rewards_ts"]
            if (
                isinstance(rewards_ts, torch.Tensor)
                and rewards_ts.ndim == 2
                and rewards_ts.shape[0] == penalty.shape[0]
            ):
                rewards_ts = rewards_ts.clone()
                rewards_ts[:, -1] = rewards_ts[:, -1] + penalty
                data.batch["rewards_ts"] = rewards_ts

        if "force_rewards_ts" in data.batch:
            force_rewards_ts = data.batch["force_rewards_ts"]
            if (
                isinstance(force_rewards_ts, torch.Tensor)
                and force_rewards_ts.ndim == 2
                and force_rewards_ts.shape[0] == penalty.shape[0]
            ):
                force_rewards_ts = force_rewards_ts.clone()
                force_rewards_ts[:, -1] = force_rewards_ts[:, -1] + penalty
                data.batch["force_rewards_ts"] = force_rewards_ts

        if "energy_rewards_ts" in data.batch:
            energy_rewards_ts = data.batch["energy_rewards_ts"]
            if (
                isinstance(energy_rewards_ts, torch.Tensor)
                and energy_rewards_ts.ndim == 2
                and energy_rewards_ts.shape[0] == penalty.shape[0]
            ):
                energy_rewards_ts = energy_rewards_ts.clone()
                energy_rewards_ts[:, -1] = energy_rewards_ts[:, -1] + penalty
                data.batch["energy_rewards_ts"] = energy_rewards_ts
        
    def filter(self, data: DataProto) -> tuple[DataProto, float, float, float, float]:
        # The filter relies on RDKit for SMILES-based deduplication and penalties.
        try:
            import rdkit  # noqa: F401
        except ImportError as exc:  # pragma: no cover - training expects RDKit installed
            raise ImportError(
                "RDKit is required for filters.enable_filtering / filters.enable_penalty / RDKit validity logging."
            ) from exc

        from verl_diffusion.utils.rdkit_metrics import graph_largest_fragment_smiles

        processed_list = self.process_data(data)
        all_smiles: List[Optional[str]] = []
        for graph in processed_list:
            # `process_data` may include conditioning context as a third element.
            # RDKit evaluation only needs (positions, atom_types).
            if not isinstance(graph, (tuple, list)) or len(graph) < 2:
                all_smiles.append(None)
                continue
            positions, atom_types = graph[0], graph[1]
            # Match `BasicMolecularMetrics`: evaluate uniqueness/novelty on the *largest fragment*.
            # This prevents "cheating" by appending many tiny disconnected fragments that inflate
            # SMILES-level uniqueness during training but collapse under eval-time canonicalization.
            smiles = graph_largest_fragment_smiles(positions, atom_types, self.dataset_info)
            all_smiles.append(smiles)
         
        num_total = len(all_smiles)
        num_valid = sum(smiles is not None for smiles in all_smiles)
        rdkit_validity = num_valid / num_total if num_total > 0 else 0.0
        rdkit_uniqueness = (
            len({smiles for smiles in all_smiles if smiles is not None}) / num_valid if num_valid > 0 else 0.0
        )

        # Store a per-sample RDKit validity mask for downstream metrics/logging.
        #
        # We compute this early (before any filtering) so the trainer can inspect:
        # - stability_given_rdkit_valid
        # - how often rewards are coming from invalid chemistry
        base_tensor = data.batch["rewards"] if "rewards" in data.batch else data.batch["x"]
        rdkit_valid_mask = torch.tensor(
            [1.0 if smiles is not None else 0.0 for smiles in all_smiles],
            device=base_tensor.device,
            dtype=base_tensor.dtype,
        )
        data.batch["rdkit_valid_mask"] = rdkit_valid_mask

        # Important: invalid RDKit molecules can sometimes achieve artificially "good" MLFF
        # scores (forces or energies) due to out-of-distribution artifacts. When we explicitly
        # optimize RDKit validity (invalid_penalty_scale > 0), gate *all* reward channels to
        # valid molecules so the policy cannot be reinforced by invalid chemistry.
        if self.invalid_penalty_scale > 0.0:
            if "force_rewards" in data.batch:
                data.batch["force_rewards"] = data.batch["force_rewards"] * rdkit_valid_mask
            if "weighted_force_rewards" in data.batch:
                data.batch["weighted_force_rewards"] = data.batch["weighted_force_rewards"] * rdkit_valid_mask
            if "force_rewards_ts" in data.batch:
                force_rewards_ts = data.batch["force_rewards_ts"]
                if (
                    isinstance(force_rewards_ts, torch.Tensor)
                    and force_rewards_ts.ndim == 2
                    and force_rewards_ts.shape[0] == rdkit_valid_mask.shape[0]
                ):
                    data.batch["force_rewards_ts"] = force_rewards_ts * rdkit_valid_mask.unsqueeze(1)

            if "energy_rewards" in data.batch:
                data.batch["energy_rewards"] = data.batch["energy_rewards"] * rdkit_valid_mask
            if "weighted_energy_rewards" in data.batch:
                data.batch["weighted_energy_rewards"] = data.batch["weighted_energy_rewards"] * rdkit_valid_mask
            if "energy_rewards_ts" in data.batch:
                energy_rewards_ts = data.batch["energy_rewards_ts"]
                if (
                    isinstance(energy_rewards_ts, torch.Tensor)
                    and energy_rewards_ts.ndim == 2
                    and energy_rewards_ts.shape[0] == rdkit_valid_mask.shape[0]
                ):
                    data.batch["energy_rewards_ts"] = energy_rewards_ts * rdkit_valid_mask.unsqueeze(1)

            if "rewards" in data.batch:
                data.batch["rewards"] = data.batch["rewards"] * rdkit_valid_mask
            if "rewards_ts" in data.batch:
                rewards_ts = data.batch["rewards_ts"]
                if (
                    isinstance(rewards_ts, torch.Tensor)
                    and rewards_ts.ndim == 2
                    and rewards_ts.shape[0] == rdkit_valid_mask.shape[0]
                ):
                    data.batch["rewards_ts"] = rewards_ts * rdkit_valid_mask.unsqueeze(1)

            # Keep scalar `rewards` consistent with weighted components in terminal-only mode.
            if (
                "rewards_ts" not in data.batch
                and "weighted_force_rewards" in data.batch
                and "weighted_energy_rewards" in data.batch
            ):
                data.batch["rewards"] = data.batch["weighted_force_rewards"] + data.batch["weighted_energy_rewards"]

        if (
            not self.enable_filtering
            and not self.enable_penalty
            and self.invalid_penalty_scale <= 0.0
            and self.duplicate_penalty_scale <= 0.0
        ):
            return data, 1.0, 1.0, rdkit_validity, rdkit_uniqueness

        # Map smiles -> indices for per-batch de-dup filtering.
        smiles_indices: Dict[str, List[int]] = {}
        none_indices: List[int] = []
        novelty_penalty: List[int] = []
        novelty_penalty_ratio = 1.0
        
        for idx, smiles in enumerate(all_smiles):
            if smiles is None:
                none_indices.append(idx)
            else:
                smiles_indices.setdefault(smiles, []).append(idx)

            if self.enable_penalty:
                novelty_penalty.append(-1 if (smiles is None or smiles in self.dataset_smiles) else 0)

        if self.enable_penalty and novelty_penalty:
            novelty_penalty_ratio = 1 + sum(novelty_penalty) / len(novelty_penalty)

        if self.enable_filtering:
            keep_mask = [False] * len(all_smiles)
            # For each unique SMILES, keep a single representative.
            #
            # Prefer keeping the *best* sample (highest reward) so that when the policy collapses to
            # a single molecule with slightly different geometries, training reinforces the most
            # stable/chemically plausible geometry rather than a random one.
            reward_tensor = None
            if "force_rewards" in data.batch:
                reward_tensor = data.batch["force_rewards"]
            elif "rewards" in data.batch:
                reward_tensor = data.batch["rewards"]

            if reward_tensor is not None and isinstance(reward_tensor, torch.Tensor):
                rewards_cpu = reward_tensor.detach().cpu().tolist()
                for indices in smiles_indices.values():
                    if not indices:
                        continue
                    best_idx = max(indices, key=lambda idx: rewards_cpu[idx])
                    keep_mask[best_idx] = True
            else:
                # Backward-compatible fallback.
                for indices in smiles_indices.values():
                    if indices:
                        keep_mask[random.choice(indices)] = True
            # Keep invalid molecules as-is so they still receive penalty feedback.
            for idx in none_indices:
                keep_mask[idx] = True
            indices_to_keep = np.where(keep_mask)[0]
        else:
            indices_to_keep = np.arange(len(all_smiles))

        # Optional: penalize duplicates within the rollout batch (helps avoid mode collapse).
        #
        # Implementation detail: when `enable_filtering=true`, DDPO keeps only one representative
        # per unique SMILES. In that setting, assigning a small per-sample penalty to *all*
        # duplicates would mostly be dropped along with the filtered samples. To ensure PPO
        # actually "sees" collapse, we instead apply the full duplicate cost to the kept sample:
        #   penalty(kept) = -duplicate_penalty_scale * (count - 1)
        #
        # When `enable_filtering=false`, we distribute the duplicate cost across all occurrences.
        if self.duplicate_penalty_scale > 0.0 and smiles_indices:
            base_tensor = data.batch["rewards"] if "rewards" in data.batch else data.batch["x"]
            duplicate_penalty = torch.zeros(
                len(all_smiles),
                device=base_tensor.device,
                dtype=base_tensor.dtype,
            )
            if self.enable_filtering:
                keep_set = set(indices_to_keep.tolist())
                for indices in smiles_indices.values():
                    count = len(indices)
                    if count <= 1:
                        continue
                    kept_indices = [idx for idx in indices if idx in keep_set]
                    if not kept_indices:
                        continue
                    kept_idx = kept_indices[0]
                    duplicate_penalty[kept_idx] = -self.duplicate_penalty_scale * (count - 1)
            else:
                for indices in smiles_indices.values():
                    count = len(indices)
                    if count <= 1:
                        continue
                    per_sample_penalty = -self.duplicate_penalty_scale * (count - 1) / count
                    idx_tensor = torch.tensor(indices, device=base_tensor.device, dtype=torch.long)
                    duplicate_penalty.index_fill_(0, idx_tensor, per_sample_penalty)

            self._add_terminal_penalty(data, duplicate_penalty)
            
        # Apply penalty if enabled
        if self.enable_penalty:
            penalty = torch.tensor(novelty_penalty, device=data.batch["rewards"].device)
            penalty = penalty.to(dtype=data.batch["rewards"].dtype) * self.penalty_scale
            self._add_terminal_penalty(data, penalty)

        # Optional: an explicit penalty for RDKit-invalid molecules (smiles == None).
        # This lets us optimize for validity without also penalizing "in-QM9" molecules.
        if self.invalid_penalty_scale > 0.0 and none_indices:
            base_ref = data.batch["rewards"] if "rewards" in data.batch else data.batch["x"]
            base_device = base_ref.device
            invalid_idx = torch.tensor(none_indices, device=base_device, dtype=torch.long)
            valid_mask = rdkit_valid_mask > 0.0

            def _apply_min_margin(key: str) -> None:
                if key not in data.batch:
                    return
                tensor = data.batch[key]
                if not isinstance(tensor, torch.Tensor) or tensor.ndim != 1:
                    return
                if valid_mask.any():
                    min_valid = tensor[valid_mask].min()
                else:
                    min_valid = tensor.min()
                target = min_valid - self.invalid_penalty_scale
                out = tensor.clone()
                out.index_fill_(0, invalid_idx, target)
                data.batch[key] = out

            def _apply_min_margin_ts(key: str) -> None:
                if key not in data.batch:
                    return
                tensor = data.batch[key]
                if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
                    return
                if valid_mask.any():
                    min_valid = tensor[valid_mask, -1].min()
                else:
                    min_valid = tensor[:, -1].min()
                target = min_valid - self.invalid_penalty_scale
                out = tensor.clone()
                out[invalid_idx, -1] = target
                data.batch[key] = out

            # Apply the invalid penalty in a reward-scale-aware way:
            # push invalid samples slightly below the *worst* valid reward in each channel.
            _apply_min_margin("rewards")
            _apply_min_margin("force_rewards")
            _apply_min_margin("energy_rewards")
            _apply_min_margin("weighted_force_rewards")
            _apply_min_margin("weighted_energy_rewards")
            _apply_min_margin_ts("rewards_ts")
            _apply_min_margin_ts("force_rewards_ts")
            _apply_min_margin_ts("energy_rewards_ts")
            
        # filter 
        if self.enable_filtering:
            filtered_data_proto = DataProto.select_idxs(data, indices_to_keep)
        else:
            filtered_data_proto = data
        
        # Calculate filtering ratio
        total_samples = len(all_smiles)
        kept_samples = len(indices_to_keep)
        filtering_ratio = kept_samples / total_samples if total_samples > 0 else 0.0
        
        return filtered_data_proto, filtering_ratio, novelty_penalty_ratio, rdkit_validity, rdkit_uniqueness
    
