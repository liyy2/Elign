import pickle
import random
import math
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

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
        invalid_reward_gate_mode: str = "hard",
        duplicate_penalty_scale: float = 0.0,
        duplicate_penalty_mode: str = "constant",
        history_size: int = 0,
        history_penalty_scale: float = 0.0,
        history_penalty_mode: str = "constant",
        history_penalty_max_multiplier: Optional[float] = None,
    ):
        self.dataset_info = dataset_info
        self.file_name = file_name
        self.condition = condition
        self.enable_filtering = bool(enable_filtering)
        self.enable_penalty = bool(enable_penalty)
        self.penalty_scale = float(penalty_scale)
        self.invalid_penalty_scale = float(invalid_penalty_scale or 0.0)
        gate_mode = str(invalid_reward_gate_mode or "hard").strip().lower()
        if gate_mode not in {"hard", "min"}:
            gate_mode = "hard"
        self.invalid_reward_gate_mode = gate_mode
        self.duplicate_penalty_scale = float(duplicate_penalty_scale or 0.0)
        dup_mode = str(duplicate_penalty_mode or "constant").strip().lower()
        if dup_mode in {"best", "best_only", "keep_best", "leave_one_out"}:
            dup_mode = "best_only"
        elif dup_mode not in {"constant", "uniform"}:
            dup_mode = "constant"
        if dup_mode == "uniform":
            dup_mode = "constant"
        self.duplicate_penalty_mode = dup_mode

        try:
            history_size_int = int(history_size or 0)
        except (TypeError, ValueError):
            history_size_int = 0
        self.history_size = max(0, history_size_int)
        self.history_penalty_scale = float(history_penalty_scale or 0.0)
        history_penalty_mode = str(history_penalty_mode or "constant").strip().lower()
        if history_penalty_mode not in {"constant", "log", "sqrt", "linear"}:
            history_penalty_mode = "constant"
        self.history_penalty_mode = history_penalty_mode
        try:
            max_mult = float(history_penalty_max_multiplier) if history_penalty_max_multiplier is not None else None
        except (TypeError, ValueError):
            max_mult = None
        if max_mult is not None and max_mult <= 0.0:
            max_mult = None
        self.history_penalty_max_multiplier = max_mult
        self._recent_smiles: Deque[str] = deque()
        self._recent_smiles_counts: Dict[str, int] = {}

        dataset_smiles_list: List[str] = []
        if self.enable_penalty_requires_smiles(self.enable_penalty) and not file_name:
            raise ValueError("filters.enable_penalty=true requires dataloader.smiles_path to be set")
        if self.enable_penalty and file_name:
            with open(file_name, "rb") as f:
                dataset_smiles_list = pickle.load(f)
        self.dataset_smiles = set(dataset_smiles_list)

    def _remember_smiles(self, smiles: str) -> None:
        if self.history_size <= 0:
            return
        self._recent_smiles.append(smiles)
        self._recent_smiles_counts[smiles] = self._recent_smiles_counts.get(smiles, 0) + 1
        while len(self._recent_smiles) > self.history_size:
            popped = self._recent_smiles.popleft()
            remaining = self._recent_smiles_counts.get(popped, 0) - 1
            if remaining <= 0:
                self._recent_smiles_counts.pop(popped, None)
            else:
                self._recent_smiles_counts[popped] = remaining

    def _seen_recently(self, smiles: str) -> bool:
        return self._recent_smiles_counts.get(smiles, 0) > 0

    def _history_penalty_multiplier(self, count: int) -> float:
        if count <= 0:
            return 0.0

        if self.history_penalty_mode == "linear":
            mult = float(count)
        elif self.history_penalty_mode == "sqrt":
            mult = math.sqrt(float(count))
        elif self.history_penalty_mode == "log":
            # Normalize so count=1 maps to multiplier=1 (log2 scaling).
            mult = math.log1p(float(count)) / math.log(2.0)
        else:
            mult = 1.0

        if self.history_penalty_max_multiplier is not None:
            mult = min(mult, float(self.history_penalty_max_multiplier))
        return float(mult)

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
        
    def filter(self, data: DataProto) -> tuple[DataProto, float, float, float, float, float, float]:
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
        # optimize RDKit validity (invalid_penalty_scale > 0), gate MLFF-driven reward channels
        # to valid molecules so the policy cannot be reinforced by invalid chemistry.
        #
        # NOTE: we preserve the stability/valence shaping term (when available) for RDKit-invalid
        # samples to keep a graded learning signal even when a prompt-group temporarily collapses
        # to invalid chemistry.
        if self.invalid_penalty_scale > 0.0:
            inv_mask = (1.0 - rdkit_valid_mask).to(dtype=rdkit_valid_mask.dtype)
            stability_rewards = data.batch.get("stability_rewards")
            has_stability_rewards = (
                isinstance(stability_rewards, torch.Tensor)
                and stability_rewards.ndim == 1
                and stability_rewards.shape[0] == rdkit_valid_mask.shape[0]
            )

            stability_for_invalid = None
            if has_stability_rewards:
                stability_on_device = stability_rewards.to(device=base_tensor.device, dtype=base_tensor.dtype)
                stability_for_invalid = torch.minimum(stability_on_device, torch.zeros_like(stability_on_device))

            if "force_rewards" in data.batch:
                force_rewards = data.batch["force_rewards"]
                if isinstance(force_rewards, torch.Tensor) and force_rewards.ndim == 1:
                    if stability_for_invalid is not None:
                        stability_for_invalid = stability_for_invalid.to(
                            device=force_rewards.device, dtype=force_rewards.dtype
                        )
                        if self.invalid_reward_gate_mode == "min":
                            invalid_force = torch.minimum(force_rewards, stability_for_invalid)
                        else:
                            invalid_force = stability_for_invalid
                        data.batch["force_rewards"] = force_rewards * rdkit_valid_mask + invalid_force * inv_mask
                    else:
                        data.batch["force_rewards"] = force_rewards * rdkit_valid_mask
            if "weighted_force_rewards" in data.batch:
                weighted_force_rewards = data.batch["weighted_force_rewards"]
                if isinstance(weighted_force_rewards, torch.Tensor) and weighted_force_rewards.ndim == 1:
                    if stability_for_invalid is not None and "force_rewards" in data.batch:
                        gated_force_rewards = data.batch["force_rewards"]
                        ratio = None
                        try:
                            denom_mask = (rdkit_valid_mask > 0.0) & (gated_force_rewards.abs() > 1e-12)
                            if denom_mask.any():
                                ratio = (weighted_force_rewards[denom_mask] / gated_force_rewards[denom_mask]).median()
                        except Exception:
                            ratio = None

                        if isinstance(ratio, torch.Tensor) and torch.isfinite(ratio).item():
                            ratio = ratio.to(dtype=weighted_force_rewards.dtype)
                            invalid_weighted = ratio * gated_force_rewards
                            data.batch["weighted_force_rewards"] = (
                                weighted_force_rewards * rdkit_valid_mask + invalid_weighted * inv_mask
                            )
                        else:
                            data.batch["weighted_force_rewards"] = weighted_force_rewards * rdkit_valid_mask
                    else:
                        data.batch["weighted_force_rewards"] = weighted_force_rewards * rdkit_valid_mask
            if "force_rewards_ts" in data.batch:
                original_force_rewards_ts = data.batch["force_rewards_ts"]
                force_rewards_ts = original_force_rewards_ts
                if (
                    isinstance(force_rewards_ts, torch.Tensor)
                    and force_rewards_ts.ndim == 2
                    and force_rewards_ts.shape[0] == rdkit_valid_mask.shape[0]
                ):
                    force_rewards_ts = force_rewards_ts * rdkit_valid_mask.unsqueeze(1)
                    if stability_for_invalid is not None:
                        force_rewards_ts = force_rewards_ts.clone()
                        stability_for_invalid_ts = stability_for_invalid.to(
                            device=force_rewards_ts.device, dtype=force_rewards_ts.dtype
                        )
                        if self.invalid_reward_gate_mode == "min":
                            terminal_orig = original_force_rewards_ts[:, -1].to(
                                device=force_rewards_ts.device, dtype=force_rewards_ts.dtype
                            )
                            terminal_gated = torch.minimum(terminal_orig, stability_for_invalid_ts)
                        else:
                            terminal_gated = stability_for_invalid_ts
                        force_rewards_ts[:, -1] = force_rewards_ts[:, -1] + terminal_gated * inv_mask
                    data.batch["force_rewards_ts"] = force_rewards_ts

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
            and (self.history_penalty_scale <= 0.0 or self.history_size <= 0)
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
                if self.duplicate_penalty_mode == "best_only":
                    reward_tensor = None
                    if "force_rewards" in data.batch:
                        reward_tensor = data.batch["force_rewards"]
                    elif "rewards" in data.batch:
                        reward_tensor = data.batch["rewards"]

                    rewards_cpu = None
                    if reward_tensor is not None and isinstance(reward_tensor, torch.Tensor):
                        rewards_cpu = reward_tensor.detach().cpu().tolist()

                    for indices in smiles_indices.values():
                        count = len(indices)
                        if count <= 1:
                            continue
                        if rewards_cpu is not None:
                            best_idx = max(indices, key=lambda idx: rewards_cpu[idx])
                        else:
                            best_idx = random.choice(indices)
                        for idx in indices:
                            if idx == best_idx:
                                continue
                            duplicate_penalty[idx] = -self.duplicate_penalty_scale
                else:
                    for indices in smiles_indices.values():
                        count = len(indices)
                        if count <= 1:
                            continue
                        per_sample_penalty = -self.duplicate_penalty_scale * (count - 1) / count
                        idx_tensor = torch.tensor(indices, device=base_tensor.device, dtype=torch.long)
                        duplicate_penalty.index_fill_(0, idx_tensor, per_sample_penalty)

            self._add_terminal_penalty(data, duplicate_penalty)

        # Optional: penalize repeats across recent rollout batches (helps improve large-sample uniqueness).
        #
        # Unlike `duplicate_penalty_scale` (within-batch), this uses a rolling memory of recent SMILES to
        # discourage cross-batch mode collapse (which tends to show up only in 1024-sample evaluations).
        if self.history_penalty_scale > 0.0 and self.history_size > 0 and self._recent_smiles_counts:
            base_tensor = data.batch["rewards"] if "rewards" in data.batch else data.batch["x"]
            history_penalty = torch.zeros(
                len(all_smiles),
                device=base_tensor.device,
                dtype=base_tensor.dtype,
            )
            if self.enable_filtering:
                keep_set = set(indices_to_keep.tolist())
                for idx in keep_set:
                    smiles = all_smiles[idx]
                    if smiles is None:
                        continue
                    count = self._recent_smiles_counts.get(smiles, 0)
                    if count > 0:
                        history_penalty[idx] = -self.history_penalty_scale * self._history_penalty_multiplier(count)
            else:
                for idx, smiles in enumerate(all_smiles):
                    if smiles is None:
                        continue
                    count = self._recent_smiles_counts.get(smiles, 0)
                    if count > 0:
                        history_penalty[idx] = -self.history_penalty_scale * self._history_penalty_multiplier(count)
            self._add_terminal_penalty(data, history_penalty)
            
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
            invalid_penalty = torch.zeros(
                len(all_smiles),
                device=base_device,
                dtype=base_ref.dtype,
            )
            invalid_penalty.index_fill_(0, invalid_idx, -self.invalid_penalty_scale)
            self._add_terminal_penalty(data, invalid_penalty)
            
        # filter 
        if self.enable_filtering:
            filtered_data_proto = DataProto.select_idxs(data, indices_to_keep)
        else:
            filtered_data_proto = data

        duplicate_hit_ratio = 0.0
        history_hit_ratio = 0.0
        if isinstance(indices_to_keep, np.ndarray) and indices_to_keep.size > 0:
            keep_set = {int(idx) for idx in indices_to_keep.tolist()}
            kept_valid = [idx for idx in keep_set if all_smiles[idx] is not None]
            kept_valid_count = len(kept_valid)
            if kept_valid_count > 0:
                duplicate_hit_count = 0
                for idx in kept_valid:
                    smiles = all_smiles[idx]
                    if smiles is None:
                        continue
                    if len(smiles_indices.get(smiles, [])) > 1:
                        duplicate_hit_count += 1
                duplicate_hit_ratio = duplicate_hit_count / kept_valid_count

                if self.history_size > 0 and self._recent_smiles_counts:
                    history_hit_count = 0
                    for idx in kept_valid:
                        smiles = all_smiles[idx]
                        if smiles is None:
                            continue
                        if self._seen_recently(smiles):
                            history_hit_count += 1
                    history_hit_ratio = history_hit_count / kept_valid_count

        # Update novelty memory with the (kept) unique SMILES for cross-batch de-dup penalties.
        if self.history_penalty_scale > 0.0 and self.history_size > 0:
            unique_kept_smiles = {
                all_smiles[int(idx)]
                for idx in indices_to_keep.tolist()
                if all_smiles[int(idx)] is not None
            }
            for smiles in unique_kept_smiles:
                self._remember_smiles(smiles)
        
        # Calculate filtering ratio
        total_samples = len(all_smiles)
        kept_samples = len(indices_to_keep)
        filtering_ratio = kept_samples / total_samples if total_samples > 0 else 0.0
        
        return (
            filtered_data_proto,
            filtering_ratio,
            novelty_penalty_ratio,
            rdkit_validity,
            rdkit_uniqueness,
            duplicate_hit_ratio,
            history_hit_ratio,
        )
    
