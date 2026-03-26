"""Minimal MLFF energy oracle for fixed-topology Ala2 coordinate samples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import torch

from edm_source.mlff_modules.mlff_force_computer import MLFFForceComputer
from edm_source.mlff_modules.mlff_utils import get_mlff_predictor

_Z_TO_SYMBOL = {
    1: "H",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    13: "Al",
    14: "Si",
    15: "P",
    16: "S",
    17: "Cl",
    33: "As",
    35: "Br",
    53: "I",
    80: "Hg",
    83: "Bi",
}

KB_EV_PER_K = 8.617333262145e-5


def beta_from_temperature(temperature_kelvin: float) -> float:
    """Return the inverse thermal energy in eV^-1."""
    temperature = float(temperature_kelvin)
    if temperature <= 0.0:
        raise ValueError("temperature_kelvin must be > 0")
    return 1.0 / (KB_EV_PER_K * temperature)


def build_dataset_info_from_atomic_numbers(atomic_numbers: Sequence[int]) -> dict:
    """Build the minimal dataset_info structure required by MLFFForceComputer."""
    ordered_unique = sorted({int(z) for z in atomic_numbers})

    atom_decoder = []
    for z in ordered_unique:
        if z not in _Z_TO_SYMBOL:
            raise ValueError(f"Unsupported atomic number for minimal MLFF oracle: {z}")
        atom_decoder.append(_Z_TO_SYMBOL[z])

    return {
        "name": "ala2_fixed",
        "atom_decoder": atom_decoder,
        "atomic_nb": ordered_unique,
        "max_n_nodes": len(atomic_numbers),
        "n_nodes": {len(atomic_numbers): 1},
    }


def one_hot_from_atomic_numbers(
    atomic_numbers: Sequence[int],
    dataset_info: dict,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create fixed one-hot atom features for the provided topology."""
    atomic_nb = [int(z) for z in dataset_info["atomic_nb"]]
    index_by_atomic_number = {z: idx for idx, z in enumerate(atomic_nb)}
    num_classes = len(atomic_nb)
    one_hot = torch.zeros(len(atomic_numbers), num_classes, device=device, dtype=dtype)
    for atom_idx, z in enumerate(atomic_numbers):
        one_hot[atom_idx, index_by_atomic_number[int(z)]] = 1.0
    return one_hot


@dataclass
class MLFFEnergyConfig:
    """Configuration for loading a fixed-topology MLFF energy oracle."""

    model_name: str = "polar-1-m"
    backend: Optional[str] = None
    device: str = "cuda"
    charge: int = 0
    spin: int = 1
    external_field: Sequence[float] = (0.0, 0.0, 0.0)
    default_dtype: str = "float32"
    position_scale: float = 1.0
    microbatch_size: Optional[int] = 2


class MLFFEnergyOracle:
    """Terminal-energy-only oracle for fixed-topology coordinate samples."""

    def __init__(
        self,
        atomic_numbers: Sequence[int],
        config: MLFFEnergyConfig,
    ) -> None:
        if not atomic_numbers:
            raise ValueError("atomic_numbers must be non-empty")
        self.atomic_numbers = [int(z) for z in atomic_numbers]
        self.config = config
        self.dataset_info = build_dataset_info_from_atomic_numbers(self.atomic_numbers)
        self._predictor = get_mlff_predictor(
            mlff_model=config.model_name,
            device=config.device,
            backend=config.backend,
            charge=config.charge,
            spin=config.spin,
            external_field=config.external_field,
            default_dtype=config.default_dtype,
        )
        if self._predictor is None:
            raise RuntimeError("Failed to initialize MLFF predictor.")
        self._force_computer = MLFFForceComputer(
            self._predictor,
            position_scale=float(config.position_scale),
            device=str(config.device),
            compute_energy=True,
        )

    @property
    def backend(self) -> str:
        return str(getattr(self._predictor, "backend", "unknown"))

    def _build_feature_tensor(self, positions: torch.Tensor) -> torch.Tensor:
        one_hot = one_hot_from_atomic_numbers(
            self.atomic_numbers,
            self.dataset_info,
            device=positions.device,
            dtype=positions.dtype,
        )
        return one_hot.unsqueeze(0).expand(positions.shape[0], -1, -1)

    def __call__(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Evaluate terminal MLFF energies.

        Args:
            positions: Coordinate tensor of shape [batch, n_atoms, 3].

        Returns:
            Tensor of shape [batch] with energies in the MLFF's native units.
        """
        if positions.ndim != 3 or positions.shape[-1] != 3:
            raise ValueError(f"Expected positions with shape [batch, n_atoms, 3], got {positions.shape}")
        if positions.shape[1] != len(self.atomic_numbers):
            raise ValueError(
                f"Expected {len(self.atomic_numbers)} atoms, got {positions.shape[1]}"
            )
        batch_size = int(positions.shape[0])
        microbatch_size = self.config.microbatch_size
        if microbatch_size is None or int(microbatch_size) <= 0:
            microbatch_size = batch_size
        microbatch_size = min(int(microbatch_size), batch_size)

        energy_chunks = []
        with torch.no_grad():
            for start in range(0, batch_size, microbatch_size):
                end = min(batch_size, start + microbatch_size)
                position_chunk = positions[start:end]
                one_hot = self._build_feature_tensor(position_chunk)
                z = torch.cat([position_chunk, one_hot], dim=-1)
                node_mask = torch.ones(
                    position_chunk.shape[0],
                    position_chunk.shape[1],
                    1,
                    device=position_chunk.device,
                    dtype=position_chunk.dtype,
                )
                _, energies = self._force_computer.compute_mlff_forces(z, node_mask, self.dataset_info)
                energy_chunks.append(energies.to(device=positions.device, dtype=positions.dtype))

        return torch.cat(energy_chunks, dim=0)


def atomic_numbers_from_metadata(metadata: dict) -> list[int]:
    """Extract fixed atomic numbers from a metadata dictionary."""
    if "atomic_numbers" not in metadata:
        raise KeyError("metadata must contain `atomic_numbers`")
    return [int(z) for z in metadata["atomic_numbers"]]


def ensure_atomic_numbers(values: Iterable[int]) -> list[int]:
    """Normalize an iterable of atomic numbers into a concrete list."""
    output = [int(v) for v in values]
    if not output:
        raise ValueError("No atomic numbers provided.")
    return output
