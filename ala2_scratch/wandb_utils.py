from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

import wandb

from .config import JsonDataclassMixin, WandbConfig, flatten_mapping


logger = logging.getLogger(__name__)


def _wandb_enabled(cfg: Union[WandbConfig, Mapping[str, Any], JsonDataclassMixin]) -> bool:
    if isinstance(cfg, WandbConfig):
        return bool(cfg.enabled)
    if isinstance(cfg, JsonDataclassMixin):
        cfg = cfg.to_dict()
    return bool(cfg.get("enabled", True))


def _as_wandb_config(cfg: Union[WandbConfig, Mapping[str, Any], JsonDataclassMixin]) -> WandbConfig:
    if isinstance(cfg, WandbConfig):
        return cfg
    if isinstance(cfg, JsonDataclassMixin):
        cfg = cfg.to_dict()
    return WandbConfig.from_dict(cfg)


def init_wandb(
    full_config: Union[Mapping[str, Any], JsonDataclassMixin],
    *,
    wandb_config: Optional[Union[WandbConfig, Mapping[str, Any], JsonDataclassMixin]] = None,
    job_type: Optional[str] = None,
    name: Optional[str] = None,
    tags: Optional[Sequence[str]] = None,
    reinit: bool = False,
):
    if wandb_config is None:
        if hasattr(full_config, "wandb"):
            wandb_config = getattr(full_config, "wandb")
        elif isinstance(full_config, Mapping) and "wandb" in full_config:
            wandb_config = full_config["wandb"]
        else:
            raise ValueError("wandb_config must be provided when full_config has no `.wandb` field")

    wb_cfg = _as_wandb_config(wandb_config)
    resolved_mode = os.environ.get("WANDB_MODE", wb_cfg.mode)
    if not wb_cfg.enabled or resolved_mode == "disabled":
        return None

    config_payload = flatten_mapping(full_config)
    run = wandb.init(
        entity=wb_cfg.entity,
        project=wb_cfg.project,
        group=wb_cfg.group,
        name=name or wb_cfg.name,
        tags=list(tags or wb_cfg.tags),
        notes=wb_cfg.notes,
        mode=resolved_mode,
        config=config_payload,
        job_type=job_type,
        reinit=reinit,
        settings=wandb.Settings(_disable_stats=True),
    )
    if wb_cfg.log_code:
        try:
            run.log_code(".")
        except Exception as exc:  # pragma: no cover - best effort only
            logger.warning("wandb.log_code failed: %s", exc)
    return run


def define_summary_metrics(run, metric_names: Iterable[str], *, summary: str = "min") -> None:
    if run is None:
        return
    for metric_name in metric_names:
        try:
            wandb.define_metric(metric_name, summary=summary)
        except Exception as exc:  # pragma: no cover - best effort only
            logger.warning("wandb.define_metric(%s) failed: %s", metric_name, exc)


def log_metrics(
    run,
    metrics: Mapping[str, Any],
    *,
    step: Optional[int] = None,
    prefix: Optional[str] = None,
    commit: Optional[bool] = None,
) -> None:
    if run is None:
        return
    payload = {}
    for key, value in metrics.items():
        if prefix:
            payload[f"{prefix}/{key}"] = value
        else:
            payload[str(key)] = value
    kwargs = {}
    if step is not None:
        kwargs["step"] = int(step)
    if commit is not None:
        kwargs["commit"] = bool(commit)
    run.log(payload, **kwargs)


def log_image(run, key: str, image: Union[str, os.PathLike, Any], *, step: Optional[int] = None, caption: Optional[str] = None) -> None:
    if run is None:
        return
    if isinstance(image, (str, os.PathLike)):
        image_obj = wandb.Image(str(image), caption=caption)
    else:
        image_obj = wandb.Image(image, caption=caption)
    kwargs = {"step": int(step)} if step is not None else {}
    run.log({key: image_obj}, **kwargs)


def log_artifact(
    run,
    *,
    name: str,
    artifact_type: str,
    paths: Sequence[Union[str, os.PathLike]],
    aliases: Optional[Sequence[str]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> None:
    if run is None:
        return
    artifact = wandb.Artifact(name=name, type=artifact_type, metadata=dict(metadata or {}))
    for item in paths:
        path = Path(item)
        if path.is_dir():
            artifact.add_dir(str(path))
        else:
            artifact.add_file(str(path))
    run.log_artifact(artifact, aliases=list(aliases or []))


def log_checkpoint(
    run,
    checkpoint_path: Union[str, os.PathLike],
    *,
    aliases: Optional[Sequence[str]] = None,
    name: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> None:
    path = Path(checkpoint_path)
    artifact_name = name or f"{path.stem}-checkpoint"
    log_artifact(
        run,
        name=artifact_name,
        artifact_type="checkpoint",
        paths=[path],
        aliases=aliases,
        metadata=metadata,
    )


def finish_wandb(run, *, exit_code: int = 0) -> None:
    if run is None:
        return
    try:
        wandb.finish(exit_code=exit_code)
    except TypeError:  # pragma: no cover - compatibility fallback
        wandb.finish()
