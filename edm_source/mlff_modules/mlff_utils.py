"""
MLFF utility functions.
"""

import logging
from typing import Optional, Sequence, Tuple, Union

import torch

logger = logging.getLogger(__name__)


class LoadedMLFFPredictor:
    """Small wrapper that carries backend metadata alongside the loaded predictor."""

    def __init__(
        self,
        *,
        backend: str,
        predictor: object,
        device: str,
        model_name: str,
        charge: int = 0,
        spin: int = 1,
        external_field: Sequence[float] = (0.0, 0.0, 0.0),
        default_dtype: str = "float32",
    ) -> None:
        self.backend = str(backend).lower()
        self.predictor = predictor
        self.device = str(device)
        self.model_name = str(model_name)
        self.charge = int(charge)
        self.spin = int(spin)
        self.external_field = tuple(float(v) for v in external_field)
        self.default_dtype = str(default_dtype)
        self.calculator = predictor if self.backend == "polar_mace" else None

    def predict(self, *args, **kwargs):
        if not hasattr(self.predictor, "predict"):
            raise AttributeError(f"{self.backend} predictor does not expose a batched `predict` method")
        return self.predictor.predict(*args, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self.predictor, name)


def _normalize_mlff_device(device):
    """Normalize device hints for MLFF predictors."""
    if isinstance(device, torch.device):
        if device.type == "cuda":
            if device.index is not None:
                return f"cuda:{device.index}"
            return "cuda"
        if device.type == "cpu":
            return "cpu"
    elif device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = str(device).lower()
        if device_str == "cuda":
            return "cuda"
        if device_str.startswith("cuda:"):
            index_str = device_str.split(":", 1)[1]
            try:
                index = int(index_str)
            except ValueError as exc:
                raise ValueError(f"Invalid CUDA device string: {device}") from exc
            return f"cuda:{index}"
        if device_str == "cpu":
            return "cpu"
    raise ValueError(f"Unsupported device specification: {device}")


def _normalize_external_field(
    external_field: Optional[Union[Sequence[float], torch.Tensor]]
) -> Tuple[float, float, float]:
    if external_field is None:
        return (0.0, 0.0, 0.0)
    if isinstance(external_field, torch.Tensor):
        values = external_field.detach().cpu().view(-1).tolist()
    else:
        values = list(external_field)
    if len(values) != 3:
        raise ValueError("external_field must contain exactly three components")
    return tuple(float(v) for v in values)


def resolve_mlff_backend(mlff_model: Optional[str], backend: Optional[str] = None) -> str:
    """Infer the MLFF backend from an explicit hint or model name."""
    if backend is not None:
        backend_name = str(backend).strip().lower()
        if backend_name in {"polar_mace", "polar", "mace_polar", "mace-polar"}:
            return "polar_mace"
        if backend_name in {"uma", "mlff", "fairchem"}:
            return "uma"
        raise ValueError(f"Unsupported MLFF backend '{backend}'")

    model_name = "" if mlff_model is None else str(mlff_model).strip().lower()
    if model_name.startswith("polar-") or "polar" in model_name:
        return "polar_mace"
    return "uma"


def get_mlff_predictor(
    mlff_model: str = "polar-1-m",
    device: Union[str, torch.device, None] = "cuda",
    *,
    backend: Optional[str] = None,
    charge: int = 0,
    spin: int = 1,
    external_field: Optional[Union[Sequence[float], torch.Tensor]] = None,
    default_dtype: str = "float32",
):
    """
    Load and initialize the MLFF predictor.

    Args:
        mlff_model: MLFF model name.
        device: Device hint for loading the model. Supports explicit CUDA indices.
        backend: Optional backend override (`uma` or `polar_mace`).
        charge: Molecular charge used by Polar MACE.
        spin: Spin multiplicity used by Polar MACE.
        external_field: External electric field vector used by Polar MACE.
        default_dtype: Default floating-point dtype for Polar MACE.

    Returns:
        Initialized MLFF predictor wrapper or None if loading fails.
    """
    target_device = _normalize_mlff_device(device)
    backend_name = resolve_mlff_backend(mlff_model, backend=backend)
    field = _normalize_external_field(external_field)

    if backend_name == "polar_mace":
        model_name = mlff_model
        if model_name is None or str(model_name).lower() in {"none", "", "null"}:
            model_name = "polar-1-m"
        try:
            from mace.calculators import mace_polar

            calculator = mace_polar(
                model=str(model_name),
                device=target_device,
                default_dtype=str(default_dtype),
            )
            logger.info("Successfully loaded Polar MACE predictor: %s", model_name)
            return LoadedMLFFPredictor(
                backend="polar_mace",
                predictor=calculator,
                device=target_device,
                model_name=str(model_name),
                charge=charge,
                spin=spin,
                external_field=field,
                default_dtype=str(default_dtype),
            )
        except Exception as e:
            logger.error("Failed to load Polar MACE predictor '%s': %s", model_name, e)
            logger.warning(
                "Continuing without MLFF guidance. Polar MACE requires MACE from source and "
                "`graph_electrostatics` (module namespace `graph_longrange`)."
            )
            return None

    try:
        from fairchem.core import pretrained_mlip

        load_device = "cuda" if target_device.startswith("cuda") else "cpu"
        target_index = None
        if target_device.startswith("cuda:"):
            target_index = int(target_device.split(":", 1)[1])

        if load_device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested for MLFF predictor but CUDA is unavailable")
            if target_index is not None and torch.cuda.current_device() != target_index:
                torch.cuda.set_device(target_index)
            if target_index is not None:
                logger.info("Loading MLFF predictor on cuda:%s", target_index)

        model_name = mlff_model
        if model_name is None or str(model_name).lower() in {"none", "", "null"}:
            model_name = "uma-s-1p1"
        if model_name == "uma-s-1":
            model_name = "uma-s-1p1"

        predictor = pretrained_mlip.get_predict_unit(model_name, device=load_device)
        logger.info("Successfully loaded MLFF predictor: %s", model_name)

        if predictor is not None and hasattr(predictor, "lazy_model_initialized"):
            if load_device == "cuda":
                current_index = torch.cuda.current_device()
                predictor.device = f"cuda:{current_index}"
            else:
                predictor.device = "cpu"
            predictor.lazy_model_initialized = False

        return LoadedMLFFPredictor(
            backend="uma",
            predictor=predictor,
            device=target_device,
            model_name=str(model_name),
            charge=charge,
            spin=spin,
            external_field=field,
            default_dtype=str(default_dtype),
        )

    except Exception as e:
        logger.error("Failed to load MLFF predictor: %s", e)
        logger.warning("Continuing without MLFF guidance")
        return None


def remove_mean_with_constraint(x: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    """
    Remove center of mass from positions while respecting node mask.

    Args:
        x: Positions tensor [batch_size, n_nodes, 3]
        node_mask: Valid node mask [batch_size, n_nodes, 1]

    Returns:
        Centered positions tensor
    """
    node_mask_expanded = node_mask.expand_as(x)
    masked_x = x * node_mask_expanded

    sum_x = masked_x.sum(dim=1, keepdim=True)
    n_valid = node_mask.sum(dim=1, keepdim=True).clamp(min=1)
    com = sum_x / n_valid

    x_centered = x - com
    x_centered = x_centered * node_mask_expanded

    return x_centered


def apply_force_clipping(
    forces: torch.Tensor,
    threshold: float,
    node_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, int]:
    """
    Clip force magnitudes that exceed threshold.

    Args:
        forces: Force tensor [batch_size, n_nodes, 3]
        threshold: Maximum allowed force magnitude
        node_mask: Optional mask for valid nodes

    Returns:
        Tuple of (clipped forces, number of clipped forces)
    """
    force_magnitudes = torch.norm(forces, dim=-1, keepdim=True)

    exceed_mask = force_magnitudes > threshold
    if node_mask is not None:
        exceed_mask = exceed_mask & (node_mask > 0)

    n_clipped = exceed_mask.sum().item()
    if n_clipped > 0:
        scale_factors = torch.where(
            exceed_mask,
            threshold / (force_magnitudes + 1e-10),
            torch.ones_like(force_magnitudes),
        )
        forces_clipped = forces * scale_factors
        logger.info("Clipped %s forces exceeding threshold %s", n_clipped, threshold)
        return forces_clipped, n_clipped

    return forces, 0
