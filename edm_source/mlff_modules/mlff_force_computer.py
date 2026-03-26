"""
MLFF force computation utilities.
"""

import logging
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from ase import Atoms

try:
    from fairchem.core.datasets.atomic_data import AtomicData
    from fairchem.core.datasets import data_list_collater
except Exception:
    AtomicData = None
    data_list_collater = None

try:
    from mace import data as mace_data
    from mace.tools import torch_geometric as mace_torch_geometric
except Exception:
    mace_data = None
    mace_torch_geometric = None

logger = logging.getLogger(__name__)


@contextmanager
def _temporary_default_dtype(dtype: Optional[torch.dtype]):
    if dtype is None:
        yield
        return
    previous = torch.get_default_dtype()
    if previous == dtype:
        yield
        return
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


class MLFFForceComputer:
    """Handles force computation using MLFF predictors."""

    def __init__(self, mlff_predictor, position_scale=1.0, device="cuda", compute_energy=False):
        self.mlff_predictor = mlff_predictor
        self.position_scale = position_scale
        self.device = device
        self.compute_energy = compute_energy

        self.molecule_cell_size = 50.0
        self.task_name = "omol"
        self.fallback_force_magnitude = 5.0

    def _predictor_backend(self) -> str:
        return str(getattr(self.mlff_predictor, "backend", "uma")).lower()

    def _extract_atomic_numbers(
        self,
        valid_features: torch.Tensor,
        dataset_info: Dict,
    ) -> torch.Tensor:
        num_classes = len(dataset_info["atom_decoder"])
        atom_decoder = dataset_info["atom_decoder"]
        atomic_number_map = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}

        if valid_features.shape[1] >= num_classes:
            valid_categorical = valid_features[:, :num_classes]
        else:
            valid_categorical = valid_features
        atom_type_indices = torch.argmax(valid_categorical, dim=1)
        atom_type_indices_cpu = atom_type_indices.cpu()

        if "atomic_nb" in dataset_info:
            atomic_numbers = torch.tensor(
                [dataset_info["atomic_nb"][idx] for idx in atom_type_indices_cpu],
                dtype=torch.long,
            )
        else:
            atomic_numbers = torch.tensor(
                [atomic_number_map.get(atom_decoder[idx], 1) for idx in atom_type_indices_cpu],
                dtype=torch.long,
            )
        return atomic_numbers

    def _build_ase_atoms(
        self,
        z: torch.Tensor,
        node_mask: torch.Tensor,
        dataset_info: Dict,
        batch_idx: int,
    ) -> Optional[Atoms]:
        positions = z[:, :, :3]
        positions_scaled = positions * self.position_scale
        features = z[:, :, 3:]

        mask = node_mask[batch_idx, :, 0].bool()
        valid_positions = positions_scaled[batch_idx, mask]
        valid_features = features[batch_idx, mask]
        if valid_positions.shape[0] == 0:
            return None

        atomic_numbers = self._extract_atomic_numbers(valid_features, dataset_info)
        atoms = Atoms(
            numbers=atomic_numbers.detach().cpu().numpy(),
            positions=valid_positions.detach().cpu().numpy(),
            cell=np.eye(3) * self.molecule_cell_size,
            pbc=False,
        )

        charge = int(getattr(self.mlff_predictor, "charge", 0))
        spin = int(getattr(self.mlff_predictor, "spin", 1))
        external_field = tuple(getattr(self.mlff_predictor, "external_field", (0.0, 0.0, 0.0)))

        atoms.info["charge"] = charge
        atoms.info["spin"] = spin
        atoms.info["external_field"] = list(external_field)
        return atoms

    def _build_mace_atomic_data(
        self,
        z: torch.Tensor,
        node_mask: torch.Tensor,
        dataset_info: Dict,
        batch_idx: int,
    ):
        if mace_data is None:
            return None

        positions = z[:, :, :3]
        positions_scaled = positions * self.position_scale
        features = z[:, :, 3:]

        mask = node_mask[batch_idx, :, 0].bool()
        valid_positions = positions_scaled[batch_idx, mask]
        valid_features = features[batch_idx, mask]
        if valid_positions.shape[0] == 0:
            return None

        atomic_numbers = self._extract_atomic_numbers(valid_features, dataset_info)
        properties = {
            "total_charge": float(getattr(self.mlff_predictor, "charge", 0)),
            "total_spin": float(getattr(self.mlff_predictor, "spin", 1)),
            "fermi_level": 0.0,
            "external_field": list(
                tuple(getattr(self.mlff_predictor, "external_field", (0.0, 0.0, 0.0)))
            ),
        }
        config = mace_data.Configuration(
            atomic_numbers=atomic_numbers.detach().cpu().numpy(),
            positions=valid_positions.detach().cpu().numpy(),
            properties=properties,
            property_weights={},
            cell=np.eye(3) * self.molecule_cell_size,
            pbc=(False, False, False),
            head=str(getattr(self.mlff_predictor, "head", "Default")),
        )
        return mace_data.AtomicData.from_config(
            config,
            z_table=getattr(self.mlff_predictor, "z_table"),
            cutoff=float(getattr(self.mlff_predictor, "r_max")),
            heads=list(getattr(self.mlff_predictor, "available_heads", ("Default",))),
        )

    def diffusion_to_atomic_data(
        self,
        z: torch.Tensor,
        node_mask: torch.Tensor,
        dataset_info: Dict,
        batch_size: int,
    ) -> List[Tuple[int, "AtomicData"]]:
        if AtomicData is None:
            logger.warning(
                "FAIRChem is unavailable, so batched UMA-style MLFF inference cannot run."
            )
            return []

        atomic_data_list: List[Tuple[int, AtomicData]] = []
        for batch_idx in range(batch_size):
            atoms = self._build_ase_atoms(z, node_mask, dataset_info, batch_idx)
            if atoms is None:
                continue
            try:
                atomic_data = AtomicData.from_ase(
                    atoms,
                    # NOTE: `r_edges=True` requires `pymatgen` for neighbor construction and will
                    # raise at runtime if the dependency is missing. UMA's predictor can operate
                    # without precomputed edges, so we keep `r_edges=False` to avoid falling back
                    # to the constant-force penalty for every sample.
                    r_edges=False,
                    radius=6.0,
                    max_neigh=50,
                    task_name=self.task_name,
                    r_data_keys=["charge", "spin"],
                )
                atomic_data_list.append((batch_idx, atomic_data))
            except Exception:
                continue

        return atomic_data_list

    def diffusion_to_mace_atomic_data(
        self,
        z: torch.Tensor,
        node_mask: torch.Tensor,
        dataset_info: Dict,
        batch_size: int,
    ) -> List[Tuple[int, object]]:
        if mace_data is None:
            logger.warning(
                "MACE is unavailable, so direct Polar MACE inference cannot run."
            )
            return []

        atomic_data_list: List[Tuple[int, object]] = []
        model_dtype = getattr(self.mlff_predictor, "model_dtype", None)
        with _temporary_default_dtype(model_dtype):
            for batch_idx in range(batch_size):
                try:
                    atomic_data = self._build_mace_atomic_data(
                        z,
                        node_mask,
                        dataset_info,
                        batch_idx,
                    )
                except Exception as exc:
                    logger.warning(
                        "MLFFForceComputer: failed to build direct Polar MACE graph for sample %s (%s). "
                        "Using fallback forces.",
                        batch_idx,
                        exc,
                    )
                    continue
                if atomic_data is not None:
                    atomic_data_list.append((batch_idx, atomic_data))
        return atomic_data_list

    def _compute_with_batched_predictor(
        self,
        z: torch.Tensor,
        node_mask: torch.Tensor,
        dataset_info: Dict,
    ):
        batch_size, max_n_nodes, _ = z.shape
        forces = torch.zeros((batch_size, max_n_nodes, 3), device=self.device)
        forces[:, :, 0] = self.fallback_force_magnitude
        if self.compute_energy:
            energies = torch.zeros(batch_size, device=self.device)

        atomic_data_pairs = self.diffusion_to_atomic_data(z, node_mask, dataset_info, batch_size)
        if not atomic_data_pairs:
            return (forces, energies) if self.compute_energy else forces

        batch_indices, atomic_data_list = zip(*atomic_data_pairs)
        batch_indices = list(batch_indices)
        atomic_data_list = list(atomic_data_list)

        if data_list_collater is None:
            logger.warning(
                "FAIRChem collater is unavailable, returning fallback MLFF forces."
            )
            return (forces, energies) if self.compute_energy else forces

        try:
            batch = data_list_collater(atomic_data_list, otf_graph=True)
            batch = batch.to(self.device)
        except Exception as exc:
            logger.warning(
                "MLFFForceComputer: failed to collate atomic data for MLFF inference (%s). "
                "Returning fallback forces.",
                exc,
            )
            return (forces, energies) if self.compute_energy else forces

        try:
            with torch.no_grad():
                predictions = self.mlff_predictor.predict(batch)

            if "forces" in predictions:
                mlff_forces = predictions["forces"]
                atom_idx = 0
                for slot_idx, atomic_data in enumerate(atomic_data_list):
                    n_atoms = atomic_data.natoms.item()
                    batch_forces = mlff_forces[atom_idx : atom_idx + n_atoms]
                    original_idx = int(batch_indices[slot_idx])
                    mask = node_mask[original_idx, :, 0].bool()
                    valid_indices = torch.where(mask)[0]

                    if len(valid_indices) == n_atoms:
                        batch_forces = batch_forces.to(forces.device)
                        forces[original_idx, valid_indices] = batch_forces
                    atom_idx += n_atoms

                forces = forces * self.position_scale

            if self.compute_energy and "energy" in predictions:
                mlff_energy = predictions["energy"]
                if mlff_energy.dim() > 1:
                    mlff_energy = mlff_energy.squeeze(-1)
                for slot_idx in range(len(atomic_data_list)):
                    original_idx = int(batch_indices[slot_idx])
                    energies[original_idx] = mlff_energy[slot_idx].item()

            return (forces, energies) if self.compute_energy else forces

        except Exception as exc:
            logger.warning(
                "MLFFForceComputer: MLFF predictor execution failed (%s). Returning fallback forces.",
                exc,
            )
            return (forces, energies) if self.compute_energy else forces

    def _compute_with_direct_mace_model(
        self,
        z: torch.Tensor,
        node_mask: torch.Tensor,
        dataset_info: Dict,
    ):
        batch_size, max_n_nodes, _ = z.shape
        forces = torch.zeros((batch_size, max_n_nodes, 3), device=self.device)
        forces[:, :, 0] = self.fallback_force_magnitude
        if self.compute_energy:
            energies = torch.zeros(batch_size, device=self.device)

        model = getattr(self.mlff_predictor, "model", None)
        if model is None or mace_torch_geometric is None:
            logger.warning(
                "Direct Polar MACE model path is unavailable, falling back to calculator execution."
            )
            return self._compute_with_ase_calculator(z, node_mask, dataset_info)

        atomic_data_pairs = self.diffusion_to_mace_atomic_data(
            z,
            node_mask,
            dataset_info,
            batch_size,
        )
        if not atomic_data_pairs:
            return (forces, energies) if self.compute_energy else forces

        batch_indices, atomic_data_list = zip(*atomic_data_pairs)
        batch_indices = list(batch_indices)
        atomic_data_list = list(atomic_data_list)

        try:
            data_loader = mace_torch_geometric.dataloader.DataLoader(
                dataset=atomic_data_list,
                batch_size=len(atomic_data_list),
                shuffle=False,
                drop_last=False,
            )
            batch = next(iter(data_loader)).to(self.device)
        except Exception as exc:
            logger.warning(
                "MLFFForceComputer: failed to collate direct Polar MACE batch (%s). "
                "Returning fallback forces.",
                exc,
            )
            return (forces, energies) if self.compute_energy else forces

        model_dtype = getattr(self.mlff_predictor, "model_dtype", None)
        batch_keys = batch.keys if not callable(batch.keys) else batch.keys()
        for key in batch_keys:
            value = batch[key]
            if torch.is_tensor(value) and torch.is_floating_point(value) and model_dtype is not None:
                batch[key] = value.to(dtype=model_dtype)

        try:
            with torch.enable_grad():
                predictions = model(
                    batch.to_dict(),
                    training=False,
                    compute_force=True,
                    compute_stress=False,
                )

            mlff_forces = predictions.get("forces")
            if mlff_forces is not None:
                atom_idx = 0
                for slot_idx, atomic_data in enumerate(atomic_data_list):
                    n_atoms = int(atomic_data.positions.shape[0])
                    batch_forces = mlff_forces[atom_idx : atom_idx + n_atoms]
                    original_idx = int(batch_indices[slot_idx])
                    mask = node_mask[original_idx, :, 0].bool()
                    valid_indices = torch.where(mask)[0]
                    if len(valid_indices) == n_atoms:
                        forces[original_idx, valid_indices] = batch_forces.to(
                            device=forces.device,
                            dtype=forces.dtype,
                        )
                    atom_idx += n_atoms
                forces = forces * self.position_scale

            if self.compute_energy:
                mlff_energy = predictions.get("energy")
                if mlff_energy is not None:
                    if mlff_energy.dim() > 1:
                        mlff_energy = mlff_energy.squeeze(-1)
                    for slot_idx in range(len(atomic_data_list)):
                        original_idx = int(batch_indices[slot_idx])
                        energies[original_idx] = mlff_energy[slot_idx].to(
                            device=energies.device,
                            dtype=energies.dtype,
                        )

            return (forces, energies) if self.compute_energy else forces
        except Exception as exc:
            logger.warning(
                "MLFFForceComputer: direct Polar MACE execution failed (%s). Returning fallback forces.",
                exc,
            )
            return (forces, energies) if self.compute_energy else forces

    def _compute_with_ase_calculator(
        self,
        z: torch.Tensor,
        node_mask: torch.Tensor,
        dataset_info: Dict,
    ):
        batch_size, max_n_nodes, _ = z.shape
        forces = torch.zeros((batch_size, max_n_nodes, 3), device=self.device)
        forces[:, :, 0] = self.fallback_force_magnitude
        if self.compute_energy:
            energies = torch.zeros(batch_size, device=self.device)

        calculator = getattr(self.mlff_predictor, "calculator", None)
        if calculator is None:
            calculator = getattr(self.mlff_predictor, "predictor", self.mlff_predictor)

        for batch_idx in range(batch_size):
            atoms = self._build_ase_atoms(z, node_mask, dataset_info, batch_idx)
            if atoms is None:
                continue

            try:
                atoms.calc = calculator
                if self.compute_energy:
                    energies[batch_idx] = float(atoms.get_potential_energy())
                batch_forces = atoms.get_forces()
                batch_forces = torch.as_tensor(
                    batch_forces,
                    device=forces.device,
                    dtype=forces.dtype,
                )

                mask = node_mask[batch_idx, :, 0].bool()
                valid_indices = torch.where(mask)[0]
                if len(valid_indices) == batch_forces.shape[0]:
                    forces[batch_idx, valid_indices] = batch_forces
            except Exception as exc:
                logger.warning(
                    "MLFFForceComputer: Polar MACE evaluation failed for sample %s (%s). "
                    "Using fallback forces.",
                    batch_idx,
                    exc,
                )

        forces = forces * self.position_scale
        return (forces, energies) if self.compute_energy else forces

    def compute_mlff_forces(self, z: torch.Tensor, node_mask: torch.Tensor, dataset_info: Dict):
        """
        Compute forces (and optionally energies) using the configured MLFF backend.

        Args:
            z: Current molecular configuration [batch_size, max_n_nodes, n_dims + n_features]
            node_mask: Valid node mask [batch_size, max_n_nodes, 1]
            dataset_info: Dataset information including atom decoder

        Returns:
            If compute_energy=False: forces tensor [batch_size, max_n_nodes, 3]
            If compute_energy=True: tuple of (forces tensor, energies tensor)
        """
        backend = self._predictor_backend()
        if backend == "polar_mace":
            return self._compute_with_direct_mace_model(z, node_mask, dataset_info)
        return self._compute_with_batched_predictor(z, node_mask, dataset_info)
