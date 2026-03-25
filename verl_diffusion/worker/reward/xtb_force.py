import logging
from typing import Optional, Union

import numpy as np
import torch

from verl_diffusion.protocol import DataProto, TensorDict
from .base import BaseReward

logger = logging.getLogger(__name__)

BOHR_TO_ANGSTROM = 0.529177210903

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


class XTBForceReward(BaseReward):
    """xTB (GFN*) force-based reward.

    Computes xTB energy + gradient for each sampled molecule and returns a scalar reward
    proportional to the negative RMS force magnitude (in atomic units).

    Intended primarily for *timing/throughput* benchmarks; the exact scaling/units of
    the reward are not critical for PPO stability.
    """

    def __init__(
        self,
        dataset_info: dict,
        position_scale: Optional[float] = None,
        device: Optional[Union[str, torch.device]] = None,
        param: str = "GFN2xTB",
        charge: float = 0.0,
        uhf: Optional[int] = None,
        fallback_reward: float = -5.0,
    ) -> None:
        super().__init__()
        self.dataset_info = dataset_info
        self.device = _to_device(device)
        self.position_scale = _resolve_position_scale(dataset_info, position_scale)
        self.charge = float(charge)
        self.uhf = None if uhf is None else int(uhf)
        self.fallback_reward = float(fallback_reward)

        try:
            from xtb.interface import Param  # noqa: WPS433
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "xTB python bindings are required. Install into the `edm` conda env: `pip install xtb==22.1`."
            ) from exc

        param_raw = str(param).strip()
        self.param_enum = None
        for candidate in Param:  # type: ignore[assignment]
            if str(getattr(candidate, "name", "")).lower() == param_raw.lower():
                self.param_enum = candidate
                break
        if self.param_enum is None:
            valid = [p.name for p in Param]  # type: ignore[attr-defined]
            raise ValueError(f"Unsupported xTB param '{param_raw}'. Valid: {valid}")

    def _atom_numbers_from_types(self, atom_type_indices: torch.Tensor) -> np.ndarray:
        decoder = self.dataset_info.get("atom_decoder")
        if not isinstance(decoder, list):
            raise ValueError("dataset_info is missing atom_decoder")

        symbols = [str(decoder[int(idx)]).strip() for idx in atom_type_indices.tolist()]
        numbers: list[int] = []
        for sym in symbols:
            key = sym.upper()
            z = _ATOMIC_NUMBER_MAP.get(key)
            if z is None:
                raise ValueError(f"Unsupported element '{sym}' for xTB reward")
            numbers.append(int(z))
        return np.asarray(numbers, dtype=np.int32)

    def calculate_rewards(self, data: DataProto) -> DataProto:
        positions = data.batch["x"].detach()
        categorical = data.batch["categorical"].detach()
        nodesxsample = data.batch["nodesxsample"].long().detach()

        batch_size = int(positions.shape[0])
        rewards = torch.full((batch_size,), self.fallback_reward, device=self.device, dtype=torch.float32)
        energies = torch.zeros((batch_size,), device=self.device, dtype=torch.float32)
        force_rms = torch.zeros((batch_size,), device=self.device, dtype=torch.float32)

        try:
            from xtb.interface import Calculator  # noqa: WPS433
            from xtb.libxtb import VERBOSITY_MUTED  # noqa: WPS433
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "xTB python bindings are required. Install into the `edm` conda env: `pip install xtb==22.1`."
            ) from exc

        decoder = self.dataset_info.get("atom_decoder")
        if not isinstance(decoder, list):
            raise ValueError("dataset_info is missing atom_decoder")
        num_classes = len(decoder)
        for i in range(batch_size):
            n_atoms = int(nodesxsample[i].item())
            if n_atoms <= 0:
                continue

            try:
                atom_types = torch.argmax(categorical[i, :n_atoms, :num_classes], dim=-1).to("cpu")
                numbers = self._atom_numbers_from_types(atom_types)

                pos_ang = (positions[i, :n_atoms, :3] * self.position_scale).to("cpu").numpy().astype(np.float64)
                pos_bohr = pos_ang / BOHR_TO_ANGSTROM

                calc = Calculator(
                    self.param_enum,
                    numbers,
                    pos_bohr,
                    charge=self.charge,
                    uhf=self.uhf,
                )
                calc.set_verbosity(VERBOSITY_MUTED)
                res = calc.singlepoint()
                e = float(res.get_energy())
                grad = np.asarray(res.get_gradient(), dtype=np.float64)
                forces = -grad  # force = -dE/dR
                rms = float(np.sqrt(np.mean(np.sum(forces * forces, axis=1))))

                energies[i] = float(e)
                force_rms[i] = float(rms)
                rewards[i] = -float(rms)
            except Exception as exc:
                logger.debug("xTB reward failed for sample %d: %s", i, exc)
                continue

        batch = TensorDict(
            {
                "rewards": rewards,
                "xtb_energy": energies,
                "xtb_force_rms": force_rms,
            },
            batch_size=[batch_size],
        )
        return DataProto(batch=batch, meta_info=data.meta_info.copy())
