import random
from typing import Optional

import numpy as np
import torch
from verl_diffusion.protocol import DataProto, TensorDict

class EDMDataLoader:
    def __init__(
        self,
        config,
        dataset_info,
        nodes_dist,
        prop_dist,
        device,
        condition: bool = False,
        num_batches: Optional[int] = None,
        rank: int = 0,
        world_size: int = 1,
        base_seed: Optional[int] = None,
    ):
        """
        Initialize the EDM dataloader
        
        Args:
            config: Configuration object containing sampling parameters
            dataset_info: Dictionary containing dataset information
            nodes_dist: Distribution for sampling nodes
            prop_dist: Distribution for sampling properties
            device: Device to put tensors on
            condition: Whether to generate conditional samples
            num_batches: Number of batches per epoch (if None, will run indefinitely)
        """
        self.config = config
        self.group_sample = self.config["dataloader"]["sample_group_size"]
        self.each_prompt_sample = self.config["dataloader"]["each_prompt_sample"]
        self.dataset_info = dataset_info
        self.nodes_dist = nodes_dist
        self.prop_dist = prop_dist
        self.device = device
        self.condition = condition
        self.num_batches = num_batches
        self.current_batch = 0
        self.rank = max(int(rank), 0)
        self.world_size = max(int(world_size), 1)
        self.base_seed = int(base_seed) if base_seed is not None else 0
        if self.num_batches is not None and self.world_size > 1:
            if self.num_batches <= 0:
                self._global_batch_indices = []
            else:
                per_rank = (self.num_batches + self.world_size - 1) // self.world_size
                self._global_batch_indices = [
                    i * self.world_size + self.rank for i in range(per_rank)
                ]
            self._local_num_batches = len(self._global_batch_indices)
        else:
            self._global_batch_indices = None
            self._local_num_batches = self.num_batches
        
    def __iter__(self):
        """Make the class iterable"""
        self.current_batch = 0
        self._infinite_counter = 0
        return self
        
    def __len__(self):
        """Return the number of batches in an epoch"""
        if self.num_batches is None:
            return float('inf')
        if self.world_size <= 1:
            return self.num_batches
        return self._local_num_batches
    
    def __next__(self):
        """Get the next batch of data"""
        if self.num_batches is not None:
            if self.world_size > 1:
                if self.current_batch >= self._local_num_batches:
                    raise StopIteration
                global_batch_idx = self._global_batch_indices[self.current_batch]
            else:
                if self.current_batch >= self.num_batches:
                    raise StopIteration
                global_batch_idx = self.current_batch
        else:
            global_batch_idx = self.rank + self.world_size * self._infinite_counter
            self._infinite_counter += 1

        self.current_batch += 1
        return self.generate_batch(global_batch_idx)
        
        
    def generate_batch(self, global_batch_idx: int):
        """
        Generate a batch of data
        
        Returns:
            DataProto: DataProto containing the generated batch data
        """
        seed = self.base_seed + int(global_batch_idx)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)

        tensors = {}
        max_n_nodes = self.dataset_info['max_n_nodes']
        nodesxsample = self.nodes_dist.sample(self.group_sample)
        prompt_ids = torch.arange(self.group_sample) + self.rank * self.group_sample
        group_index = prompt_ids.repeat_interleave(self.each_prompt_sample, dim=0)
        group_index = group_index.to(self.device)
        self.batch_size = self.each_prompt_sample * self.group_sample
        
        if self.condition:
            context = self.prop_dist.sample_batch(nodesxsample).to(self.device)
            context = context.repeat_interleave(self.each_prompt_sample, dim=0)
            nodesxsample = nodesxsample.repeat_interleave(self.each_prompt_sample, dim=0)
            # Store tensors
            tensors.update({
                'nodesxsample': nodesxsample,
                'context': context,
                'group_index': group_index,
  
            })
        else:
            nodesxsample = nodesxsample.repeat_interleave(self.each_prompt_sample, dim=0)
            
            # Store tensors
            tensors.update({
                'nodesxsample': nodesxsample.to(self.device),
                'group_index' : group_index
            })
        
        # Create meta info
        meta_info = {
            'max_n_nodes': max_n_nodes,
            'condition': self.condition
        }
            
        # Create DataProto
        return DataProto(batch=TensorDict(tensors, batch_size=[self.batch_size]), meta_info=meta_info)
        
    def reset(self):
        """Reset the iterator"""
        self.current_batch = 0
   
