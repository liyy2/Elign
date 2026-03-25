import logging
from typing import Optional, Union

import numpy as np
import torch

from verl_diffusion.protocol import DataProto, TensorDict
from .base import BaseReward

logger = logging.getLogger(__name__)

BOHR_TO_ANGSTROM = 0.529177210903
HARTREE_TO_EV = 27.211386245988
HARTREE_PER_BOHR_TO_EV_PER_A = HARTREE_TO_EV / BOHR_TO_ANGSTROM

_ATOMIC_NUMBER_MAP = {
    "H": 1,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "AL": 13,
    "SI": 14,
    "P": 15,
    "S": 16,
    "CL": 17,
    "AS": 33,
    "BR": 35,
    "I": 53,
    "HG": 80,
    "BI": 83,
}


def _to_device(device_like: Optional[Union[str, torch.device]]) -> torch.device:
    if device_like is None:
        return torch.device("cpu")
    if isinstance(device_like, torch.device):
        return device_like
    return torch.device(str(device_like))


def _resolve_position_scale(dataset_info: dict, position_scale: Optional[float]) -> float:
    if position_scale is not None:
        try:
            return float(position_scale)
        except (TypeError, ValueError):
            return 1.0
    norm_values = dataset_info.get("normalize_factors")
    if isinstance(norm_values, (list, tuple)) and len(norm_values) > 0:
        try:
            return float(norm_values[0])
        except (TypeError, ValueError):
            return 1.0
    return 1.0


def _min_pairwise_distance(positions: np.ndarray) -> float:
    if positions.shape[0] <= 1:
        return float("inf")
    diff = positions[:, None, :] - positions[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist, np.inf)
    return float(dist.min())


class PySCFDFTForceReward(BaseReward):
    """DFT (PySCF) force-based reward.

    Runs a PySCF DFT single-point + gradient and returns a scalar reward proportional
    to the negative RMS force magnitude.

    Notes:
    - This is intended for *throughput scaling* benchmarks; for realistic chemistry
      you should tune XC/basis/grid settings.
    - PySCF forces are computed on CPU.
    """

    def __init__(
        self,
        dataset_info: dict,
        position_scale: Optional[float] = None,
        device: Optional[Union[str, torch.device]] = None,
        xc: str = "pbe",
        basis: str = "sto-3g",
        charge: int = 0,
        spin: Optional[int] = None,
        max_cycle: int = 50,
        conv_tol: float = 1e-6,
        grids_level: int = 3,
        density_fit: bool = True,
        accept_unconverged: bool = True,
        min_interatomic_distance: float = 0.6,
        fallback_reward: float = -5.0,
        reward_in_ev_per_a: bool = False,
    ) -> None:
        super().__init__()
        self.dataset_info = dataset_info
        self.device = _to_device(device)
        self.position_scale = _resolve_position_scale(dataset_info, position_scale)
        self.xc = str(xc)
        self.basis = str(basis)
        self.charge = int(charge)
        self.spin = None if spin is None else int(spin)
        self.max_cycle = int(max_cycle)
        self.conv_tol = float(conv_tol)
        self.grids_level = int(grids_level)
        self.density_fit = bool(density_fit)
        self.accept_unconverged = bool(accept_unconverged)
        self.min_interatomic_distance = float(min_interatomic_distance)
        self.fallback_reward = float(fallback_reward)
        self.reward_in_ev_per_a = bool(reward_in_ev_per_a)

        try:
            import pyscf  # noqa: F401
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "PySCF is required. Install into the `edm` conda env: `conda install -n edm -c conda-forge pyscf`."
            ) from exc

    def _symbols_and_numbers_from_types(self, atom_type_indices: torch.Tensor) -> tuple[list[str], np.ndarray]:
        decoder = self.dataset_info.get("atom_decoder")
        if not isinstance(decoder, list):
            raise ValueError("dataset_info is missing atom_decoder")

        symbols = [str(decoder[int(idx)]).strip() for idx in atom_type_indices.tolist()]
        numbers: list[int] = []
        for sym in symbols:
            key = sym.upper()
            z = _ATOMIC_NUMBER_MAP.get(key)
            if z is None:
                raise ValueError(f"Unsupported element '{sym}' for DFT reward")
            numbers.append(int(z))
        return symbols, np.asarray(numbers, dtype=np.int32)

    def _infer_spin(self, atomic_numbers: np.ndarray) -> int:
        nelec = int(np.sum(atomic_numbers)) - int(self.charge)
        if nelec < 0:
            return 0
        # Minimal parity-based guess: even -> singlet, odd -> doublet.
        return int(nelec % 2)

    def calculate_rewards(self, data: DataProto) -> DataProto:
        positions = data.batch["x"].detach()
        categorical = data.batch["categorical"].detach()
        nodesxsample = data.batch["nodesxsample"].long().detach()

        batch_size = int(positions.shape[0])
        rewards = torch.full((batch_size,), self.fallback_reward, device=self.device, dtype=torch.float32)
        energies = torch.zeros((batch_size,), device=self.device, dtype=torch.float32)
        force_rms = torch.zeros((batch_size,), device=self.device, dtype=torch.float32)

        decoder = self.dataset_info.get("atom_decoder")
        if not isinstance(decoder, list):
            raise ValueError("dataset_info is missing atom_decoder")
        num_classes = len(decoder)

        # Local imports to keep reward import cheap for non-DFT runs.
        from pyscf import gto, dft  # noqa: WPS433

        for i in range(batch_size):
            n_atoms = int(nodesxsample[i].item())
            if n_atoms <= 0:
                continue

            try:
                atom_types = torch.argmax(categorical[i, :n_atoms, :num_classes], dim=-1).to("cpu")
                symbols, atomic_numbers = self._symbols_and_numbers_from_types(atom_types)
                pos_ang = (positions[i, :n_atoms, :3] * self.position_scale).to("cpu").numpy().astype(np.float64)

                if _min_pairwise_distance(pos_ang) < self.min_interatomic_distance:
                    continue

                spin = self.spin
                if spin is None:
                    spin = self._infer_spin(atomic_numbers)

                mol = gto.Mole()
                mol.atom = [(sym, tuple(map(float, xyz))) for sym, xyz in zip(symbols, pos_ang)]
                mol.unit = "Angstrom"
                mol.basis = self.basis
                mol.charge = int(self.charge)
                mol.spin = int(spin)
                mol.verbose = 0
                mol.build()

                mf = dft.RKS(mol) if int(spin) == 0 else dft.UKS(mol)
                mf.xc = self.xc
                mf.conv_tol = self.conv_tol
                mf.max_cycle = self.max_cycle
                mf.verbose = 0
                try:
                    mf.grids.level = self.grids_level
                except Exception:
                    pass
                if self.density_fit:
                    try:
                        mf = mf.density_fit()
                    except Exception:
                        pass

                energy = float(mf.kernel())
                converged = bool(getattr(mf, "converged", False))
                if not converged and not self.accept_unconverged:
                    continue

                grad = mf.nuc_grad_method().kernel()  # dE/dR in Hartree/Bohr
                forces = -np.asarray(grad, dtype=np.float64)
                rms = float(np.sqrt(np.mean(np.sum(forces * forces, axis=1))))

                energies[i] = float(energy)
                force_rms[i] = float(rms)
                if self.reward_in_ev_per_a:
                    rewards[i] = -float(rms) * float(HARTREE_PER_BOHR_TO_EV_PER_A)
                else:
                    rewards[i] = -float(rms)
            except Exception as exc:
                logger.debug("PySCF DFT reward failed for sample %d: %s", i, exc)
                continue

        batch = TensorDict(
            {
                "rewards": rewards,
                "dft_energy_hartree": energies,
                "dft_force_rms_hartree_per_bohr": force_rms,
            },
            batch_size=[batch_size],
        )
        return DataProto(batch=batch, meta_info=data.meta_info.copy())

