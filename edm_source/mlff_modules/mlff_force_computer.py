"""
MLFF Force Computer Module
Handles force computation using MLFF predictors for molecular configurations.
ONLY computes forces - no logging or statistics.
"""

import torch
import numpy as np
from typing import Dict, List
from ase import Atoms
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.datasets import data_list_collater


class MLFFForceComputer:
    """Handles force computation using MLFF predictors."""
    
    def __init__(self, mlff_predictor, position_scale=1.0, device='cuda', compute_energy=False):
        """
        Initialize the MLFF force computer.
        
        Args:
            mlff_predictor: The MLFF predictor model (e.g., UMA)
            position_scale: Scale factor to convert from normalized to physical positions
            device: Device to run computations on
            compute_energy: Whether to compute energy in addition to forces (default: False)
        """
        self.mlff_predictor = mlff_predictor
        self.position_scale = position_scale
        self.device = device
        self.compute_energy = compute_energy
        
        # Default molecule cell size for molecular systems
        self.molecule_cell_size = 50.0
        
        # Default task name for MLFF predictor
        self.task_name = "omol"  # For molecular systems
        
    def diffusion_to_atomic_data(self, z: torch.Tensor, node_mask: torch.Tensor, 
                                 dataset_info: Dict, batch_size: int) -> List[AtomicData]:
        """
        Convert diffusion state to AtomicData format for MLFF.
        
        Args:
            z: Diffusion state tensor [batch_size, max_n_nodes, n_dims + n_features]
            node_mask: Valid node mask [batch_size, max_n_nodes, 1]
            dataset_info: Dataset information including atom decoder
            batch_size: Number of molecules in batch
            
        Returns:
            List of AtomicData objects
        """
        positions = z[:, :, :3]  # Extract positions [batch_size, max_n_nodes, 3]
        
        # Scale positions to physical units (Angstroms)
        positions_scaled = positions * self.position_scale
        
        # Get per-node features (categorical one-hot [+ optional charge])
        features = z[:, :, 3:]  # [batch_size, max_n_nodes, n_features]
        # Determine number of categorical channels from dataset info
        num_classes = len(dataset_info['atom_decoder'])
        
        # Convert one-hot encoded features to atomic numbers
        atom_decoder = dataset_info['atom_decoder']
        atomic_data_list = []
        
        # Create a mapping from atomic symbols to atomic numbers
        # QM9 contains H, C, N, O, F (F is rare with only ~2300 occurrences)
        atomic_number_map = {
            'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9
        }
        
        for batch_idx in range(batch_size):
            # Get valid atoms for this molecule
            mask = node_mask[batch_idx, :, 0].bool()
            valid_positions = positions_scaled[batch_idx, mask]  # [n_valid_nodes, 3]
            valid_features = features[batch_idx, mask]  # [n_valid_nodes, n_features]
            
            if valid_positions.shape[0] == 0:
                continue
            
            # Convert categorical features to atom types
            # Only use the first num_classes channels for argmax (exclude charge channel if present)
            if valid_features.shape[1] >= num_classes:
                valid_categorical = valid_features[:, :num_classes]
            else:
                # Fallback: if features are shorter than expected, use all
                valid_categorical = valid_features
            atom_type_indices = torch.argmax(valid_categorical, dim=1)  # [n_valid_nodes]
            
            # Map atom types to atomic numbers
            atom_type_indices_cpu = atom_type_indices.cpu()
            
            if 'atomic_nb' in dataset_info:
                # For GEOM dataset
                atomic_numbers = torch.tensor([dataset_info['atomic_nb'][idx] for idx in atom_type_indices_cpu])
            else:
                # For QM9 dataset - map through decoder
                atomic_numbers = torch.tensor([
                    atomic_number_map.get(atom_decoder[idx], 1) 
                    for idx in atom_type_indices_cpu
                ])
            
            # Create ASE atoms object
            atoms = Atoms(
                numbers=atomic_numbers.detach().cpu().numpy(),
                positions=valid_positions.detach().cpu().numpy(),
                cell=np.eye(3) * self.molecule_cell_size,
                pbc=False
            )
            
            # Set charge and spin for molecular systems
            atoms.info['charge'] = 0  # Default charge
            atoms.info['spin'] = 1    # Default spin multiplicity
            
            # Convert to AtomicData
            try:
                atomic_data = AtomicData.from_ase(
                    atoms,
                    r_edges=True,
                    radius=6.0,
                    max_neigh=50,
                    task_name=self.task_name,
                    r_data_keys=["charge", "spin"]
                )
                # Debug: Check natoms for each molecule
                # print(f"Batch {batch_idx}: Created AtomicData with {atomic_data.natoms.item()} atoms")
                atomic_data_list.append(atomic_data)
            except Exception:
                continue
        
        return atomic_data_list
    
    def compute_mlff_forces(self, z: torch.Tensor, node_mask: torch.Tensor, 
                           dataset_info: Dict):
        """
        Compute forces (and optionally energies) using MLFF predictor.
        
        Args:
            z: Current molecular configuration [batch_size, max_n_nodes, n_dims + n_features]
            node_mask: Valid node mask [batch_size, max_n_nodes, 1]
            dataset_info: Dataset information including atom decoder
            
        Returns:
            If compute_energy=False: Forces tensor [batch_size, max_n_nodes, 3]
            If compute_energy=True: Tuple of (forces tensor [batch_size, max_n_nodes, 3], energies tensor [batch_size])
        """
        batch_size, max_n_nodes, _ = z.shape
        
        # Initialize zero forces
        forces = torch.zeros((batch_size, max_n_nodes, 3), device=self.device)
        if self.compute_energy:
            energies = torch.zeros(batch_size, device=self.device)
        
        # Convert to atomic data format
        atomic_data_list = self.diffusion_to_atomic_data(z, node_mask, dataset_info, batch_size)
        
        if not atomic_data_list:
            return (forces, energies) if self.compute_energy else forces
        
        # Batch the atomic data using FAIRChem's collater
        try:
            batch = data_list_collater(atomic_data_list, otf_graph=True)
            batch = batch.to(self.device)
        except Exception:
            return (forces, energies) if self.compute_energy else forces
        
        # Compute forces (and energy) using MLFF
        try:
            with torch.no_grad():
                predictions = self.mlff_predictor.predict(batch)
            
            # Extract forces
            if 'forces' in predictions:
                mlff_forces = predictions['forces']  # [total_atoms, 3]
                
                # Map forces back to original batch structure
                atom_idx = 0
                for batch_idx, atomic_data in enumerate(atomic_data_list):
                    n_atoms = atomic_data.natoms.item()
                    batch_forces = mlff_forces[atom_idx:atom_idx + n_atoms]  # [n_atoms, 3]
                    
                    # Get valid node indices for this batch
                    mask = node_mask[batch_idx, :, 0].bool()
                    valid_indices = torch.where(mask)[0]
                    
                    if len(valid_indices) == n_atoms:
                        # Ensure batch_forces is on the same device as forces
                        batch_forces = batch_forces.to(forces.device)
                        forces[batch_idx, valid_indices] = batch_forces
                        
                    atom_idx += n_atoms
                
                # Scale forces back to normalized space
                # If x_phys = x_norm * position_scale, then F_norm = dE/dx_norm = (dE/dx_phys) * position_scale
                forces = forces * self.position_scale
            
            # Extract energy only if requested
            if self.compute_energy and 'energy' in predictions:
                mlff_energy = predictions['energy']  # [num_systems] or [num_systems, 1]
                if mlff_energy.dim() > 1:
                    mlff_energy = mlff_energy.squeeze(-1)
                # Map energies back to batch (accounting for skipped systems)
                for idx, _ in enumerate(atomic_data_list):
                    energies[idx] = mlff_energy[idx].item()
            
            return (forces, energies) if self.compute_energy else forces
            
        except Exception:
            # Return zero forces (and energies) on error
            return (forces, energies) if self.compute_energy else forces
