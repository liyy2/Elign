from __future__ import annotations

import argparse
import json
from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Type, TypeVar, Union, get_args, get_origin


T = TypeVar("T", bound="JsonDataclassMixin")


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {f.name: _to_jsonable(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    return value


def _coerce_simple(value: Any, target_type: Any) -> Any:
    if target_type is Any:
        return value
    if target_type is Path:
        return Path(value) if value is not None else None
    if target_type in (str, int, float, bool):
        return target_type(value)
    return value


def _coerce_field(value: Any, annotation: Any) -> Any:
    origin = get_origin(annotation)
    args = get_args(annotation)

    if value is None:
        return None

    if origin is Union:
        non_none_args = [arg for arg in args if arg is not type(None)]
        if len(non_none_args) == 1:
            return _coerce_field(value, non_none_args[0])
        return value

    if is_dataclass(annotation):
        return annotation.from_dict(value) if isinstance(value, Mapping) else value

    if origin in (list, List):
        inner = args[0] if args else Any
        return [_coerce_field(item, inner) for item in value]

    if origin in (tuple, Tuple):
        inner = args[0] if args else Any
        return tuple(_coerce_field(item, inner) for item in value)

    if origin in (dict, Dict):
        key_type = args[0] if len(args) > 0 else Any
        val_type = args[1] if len(args) > 1 else Any
        return {
            _coerce_field(k, key_type): _coerce_field(v, val_type)
            for k, v in value.items()
        }

    return _coerce_simple(value, annotation)


def _dataclass_from_dict(cls: Type[T], raw: Mapping[str, Any]) -> T:
    kwargs: Dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in raw:
            continue
        kwargs[f.name] = _coerce_field(raw[f.name], f.type)
    return cls(**kwargs)


def flatten_mapping(
    mapping: Union[Mapping[str, Any], "JsonDataclassMixin"],
    *,
    parent_key: str = "",
    sep: str = ".",
) -> Dict[str, Any]:
    if is_dataclass(mapping):
        mapping = mapping.to_dict()
    flat: Dict[str, Any] = {}
    for key, value in mapping.items():
        joined = f"{parent_key}{sep}{key}" if parent_key else str(key)
        if isinstance(value, Mapping):
            flat.update(flatten_mapping(value, parent_key=joined, sep=sep))
        else:
            flat[joined] = value
    return flat


def _parse_override_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        lowered = raw.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if lowered == "null":
            return None
        return raw


class JsonDataclassMixin:
    def to_dict(self) -> Dict[str, Any]:
        return _to_jsonable(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def save_json(self, path: Union[str, Path], *, indent: int = 2) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")
        return target

    @classmethod
    def from_dict(cls: Type[T], raw: Mapping[str, Any]) -> T:
        return _dataclass_from_dict(cls, raw)

    @classmethod
    def from_json(cls: Type[T], path: Union[str, Path]) -> T:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls.from_dict(payload)


@dataclass
class WandbConfig(JsonDataclassMixin):
    enabled: bool = True
    entity: Optional[str] = None
    project: str = "ala2-diffusion"
    group: str = "ala2-300k"
    name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    mode: str = "online"
    log_code: bool = False
    notes: Optional[str] = None


@dataclass
class DataConfig(JsonDataclassMixin):
    dataset_path: Optional[str] = None
    processed_data_path: Optional[str] = None
    topology_path: Optional[str] = None
    metadata_path: Optional[str] = None
    split_path: Optional[str] = None
    trajectory_npz: Optional[str] = None
    topology_pdb: Optional[str] = None
    metadata_json: Optional[str] = None
    split_json: Optional[str] = None
    cache_dir: str = "outputs/ala2_scratch/data"
    batch_size: int = 256
    num_workers: int = 0
    pin_memory: bool = False
    drop_last_train: bool = False
    train_fraction: float = 0.8
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    frame_stride: int = 10
    align: bool = True
    center: bool = True
    block_size: int = 100
    split_seed: int = 0


@dataclass
class BackboneConfig(JsonDataclassMixin):
    hidden_nf: int = 128
    n_layers: int = 4
    attention: bool = True
    tanh: bool = False
    norm_constant: float = 1.0
    inv_sublayers: int = 1
    normalization_factor: float = 100.0
    aggregation_method: str = "sum"
    atom_type_embed_dim: int = 16
    atom_index_embed_dim: int = 16
    time_embed_dim: int = 32


@dataclass
class DiffusionConfig(JsonDataclassMixin):
    num_steps: int = 100
    beta_schedule: str = "cosine"
    beta_start: float = 1.0e-4
    beta_end: float = 2.0e-2
    predict_target: str = "epsilon"
    recenter_every_step: bool = True
    clip_denoised: bool = False


@dataclass
class OptimizerConfig(JsonDataclassMixin):
    lr: float = 2.0e-4
    weight_decay: float = 1.0e-6
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1.0e-8
    grad_clip_norm: float = 1.0


@dataclass
class SchedulerConfig(JsonDataclassMixin):
    enabled: bool = False
    warmup_steps: int = 0
    total_steps: int = 0
    min_lr_scale: float = 0.1


@dataclass
class CheckpointConfig(JsonDataclassMixin):
    save_dir: str = "/home/yl2428/logs/ala2_scratch"
    save_every: int = 500
    max_to_keep: int = 5


@dataclass
class BaseRunConfig(JsonDataclassMixin):
    stage: str = "base"
    experiment_name: str = "ala2_scratch"
    run_name: Optional[str] = None
    seed: int = 42
    device: str = "cuda"
    dtype: str = "float32"
    output_dir: str = "outputs/ala2_scratch"
    output_root: str = "outputs/ala2_scratch"
    save_dir: Optional[str] = None
    allow_overwrite_save_dir: bool = False
    data: DataConfig = field(default_factory=DataConfig)
    model: Dict[str, Any] = field(default_factory=dict)
    train: Dict[str, Any] = field(default_factory=dict)
    eval: Dict[str, Any] = field(default_factory=dict)
    mlff: Dict[str, Any] = field(default_factory=dict)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


@dataclass
class PretrainConfig(BaseRunConfig):
    stage: str = "pretrain"
    num_train_steps: int = 10_000
    log_every: int = 20
    val_every: int = 200
    sample_every: int = 500
    num_val_batches: int = 8
    num_eval_samples: int = 256


@dataclass
class PosttrainConfig(BaseRunConfig):
    stage: str = "posttrain"
    pretrained_checkpoint: Optional[str] = None
    reference_checkpoint: Optional[str] = None
    mlff_model: str = "polar-1-m"
    mlff_backend: Optional[str] = None
    mlff_device: Optional[str] = None
    num_train_steps: int = 4_000
    rollout_batch_size: int = 64
    log_every: int = 10
    eval_every: int = 100
    save_every: int = 100
    reference_weight: float = 1.0
    reward_temperature_kelvin: float = 300.0
    reward_scale: float = 1.0
    reward_center: bool = True
    reward_normalize: bool = True
    num_eval_samples: int = 256


@dataclass
class EvalConfig(BaseRunConfig):
    stage: str = "eval"
    checkpoint_path: Optional[str] = None
    sample_count: int = 1024
    histogram_bins: int = 64
    output_path: str = "outputs/ala2_scratch/eval"


StageConfig = Union[PretrainConfig, PosttrainConfig, EvalConfig]


_STAGE_TO_CONFIG: Dict[str, Type[StageConfig]] = {
    "pretrain": PretrainConfig,
    "posttrain": PosttrainConfig,
    "eval": EvalConfig,
}


def default_config(stage: str) -> StageConfig:
    stage_name = stage.strip().lower()
    if stage_name not in _STAGE_TO_CONFIG:
        raise ValueError(f"Unsupported stage '{stage}'")
    return _STAGE_TO_CONFIG[stage_name]()


def load_config(path: Union[str, Path], *, expected_stage: Optional[str] = None) -> StageConfig:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    stage = str(payload.get("stage", expected_stage or "pretrain")).lower()
    if expected_stage is not None and stage != expected_stage.lower():
        raise ValueError(f"Expected stage '{expected_stage}', found '{stage}' in {path}")
    config_cls = _STAGE_TO_CONFIG.get(stage)
    if config_cls is None:
        raise ValueError(f"Unsupported stage '{stage}' in {path}")
    return config_cls.from_dict(payload)


def _set_nested_value(target: Any, path: str, value: Any) -> None:
    parts = path.split(".")
    obj = target
    for key in parts[:-1]:
        obj = getattr(obj, key)
    setattr(obj, parts[-1], value)


def apply_overrides(config: StageConfig, overrides: Sequence[str]) -> StageConfig:
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override '{override}' must use key=value syntax")
        key, raw_value = override.split("=", 1)
        _set_nested_value(config, key, _parse_override_value(raw_value))
    return config


def add_config_arguments(parser: argparse.ArgumentParser, *, default_stage: str) -> argparse.ArgumentParser:
    parser.add_argument("--config", type=str, default=None, help="Path to a JSON config file.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a nested config field using dot notation, e.g. --set wandb.mode=offline",
    )
    parser.set_defaults(_ala2_default_stage=default_stage)
    return parser


def config_from_args(args: argparse.Namespace, *, default_stage: Optional[str] = None) -> StageConfig:
    stage = default_stage or getattr(args, "_ala2_default_stage", None) or "pretrain"
    config = load_config(args.config, expected_stage=None) if getattr(args, "config", None) else default_config(stage)
    overrides = getattr(args, "set", []) or []
    if overrides:
        config = apply_overrides(config, overrides)
    return config
