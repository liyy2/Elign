import argparse
import copy
import importlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
import torch


class _EMA:
    def __init__(self, beta: float) -> None:
        self.beta = float(beta)

    def update_model_average(self, averaged_model: torch.nn.Module, current_model: torch.nn.Module) -> None:
        for current_params, averaged_params in zip(current_model.parameters(), averaged_model.parameters()):
            averaged_params.data = self.update_average(averaged_params.data, current_params.data)

    def update_average(self, old: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
        return old * self.beta + (1.0 - self.beta) * new


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(payload).__name__}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _deep_update(base: Dict[str, Any], updates: Mapping[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def _parse_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _apply_override(config: Dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ValueError(f"Override must have the form key=value, got '{override}'")
    key_path, raw_value = override.split("=", 1)
    keys = [item for item in key_path.split(".") if item]
    if not keys:
        raise ValueError(f"Invalid override path '{key_path}'")
    target = config
    for key in keys[:-1]:
        if key not in target or not isinstance(target[key], dict):
            target[key] = {}
        target = target[key]
    target[keys[-1]] = _parse_value(raw_value)


def _try_import(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


def _load_wandb_helpers():
    helpers = _try_import("ala2_scratch.wandb_utils")
    if helpers is None:
        return None, None, None, None
    init_fn = getattr(helpers, "init_wandb", None)
    log_fn = getattr(helpers, "log_metrics", None)
    image_fn = getattr(helpers, "log_image", None)
    artifact_fn = getattr(helpers, "log_checkpoint", None)
    return init_fn, log_fn, image_fn, artifact_fn


def _load_build_dataloaders():
    module = importlib.import_module("ala2_scratch.data")
    for name in ("build_dataloaders", "create_dataloaders", "build_dataloaders_from_config"):
        fn = getattr(module, name, None)
        if callable(fn):
            return fn
    raise AttributeError("ala2_scratch.data must expose build_dataloaders/create_dataloaders/build_dataloaders_from_config")


def _load_build_model():
    module = importlib.import_module("ala2_scratch.model")
    for name in ("build_model", "create_model"):
        fn = getattr(module, name, None)
        if callable(fn):
            return fn
    raise AttributeError("ala2_scratch.model must expose build_model or create_model")


def _load_build_diffusion():
    module = importlib.import_module("ala2_scratch.diffusion")
    for name in ("build_diffusion", "create_diffusion"):
        fn = getattr(module, name, None)
        if callable(fn):
            return fn
    for name in ("CoordinateDiffusion", "Ala2CoordinateDiffusion"):
        cls = getattr(module, name, None)
        if cls is not None:
            return cls
    raise AttributeError("ala2_scratch.diffusion must expose build_diffusion/create_diffusion or a diffusion class")


def _resolve_device(config: Mapping[str, Any]) -> torch.device:
    explicit = str(config.get("device", "auto"))
    if explicit == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if explicit == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Config requested CUDA but CUDA is unavailable")
    return torch.device(explicit)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_device(batch: Any, device: torch.device) -> Any:
    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, dict):
        return {key: _to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, (list, tuple)):
        values = [_to_device(item, device) for item in batch]
        return type(batch)(values)
    return batch


def _as_mapping(batch: Any) -> Dict[str, Any]:
    if isinstance(batch, dict):
        return batch
    if hasattr(batch, "items"):
        return dict(batch.items())
    raise TypeError(f"Unsupported batch type: {type(batch).__name__}")


def _extract_batch_tensors(batch: Mapping[str, Any]) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    coordinate_keys = ("coordinates", "positions", "x")
    static_keys = ("static_features", "node_features", "features", "atom_features")
    mask_keys = ("node_mask", "mask", "atom_mask")

    coordinates = None
    for key in coordinate_keys:
        value = batch.get(key)
        if torch.is_tensor(value):
            coordinates = value
            break
    if coordinates is None:
        raise KeyError(f"Batch must contain one of {coordinate_keys}")

    static_features = None
    for key in static_keys:
        value = batch.get(key)
        if torch.is_tensor(value):
            static_features = value
            break

    node_mask = None
    for key in mask_keys:
        value = batch.get(key)
        if torch.is_tensor(value):
            node_mask = value
            break

    return coordinates, static_features, node_mask


def _compute_startup_health_metrics(batch: Mapping[str, Any]) -> Dict[str, Any]:
    coordinates, static_features, node_mask = _extract_batch_tensors(batch)
    if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
        raise ValueError(
            f"Expected coordinates shaped [batch, nodes, 3], got {tuple(coordinates.shape)}"
        )
    if not torch.isfinite(coordinates).all():
        raise ValueError("Coordinates contain non-finite values")
    metrics: Dict[str, Any] = {
        "health/data_load_success": 1.0,
        "health/batch_construct_success": 1.0,
        "health/coords_finite": 1.0,
        "health/train_batch_size": int(coordinates.shape[0]),
        "health/n_nodes": int(coordinates.shape[1]),
        "health/coord_dim": int(coordinates.shape[2]),
        "health/coord_abs_mean": float(coordinates.abs().mean().item()),
        "health/coord_abs_max": float(coordinates.abs().max().item()),
    }
    if static_features is not None:
        if static_features.ndim not in {2, 3}:
            raise ValueError(
                f"Expected static features shaped [nodes, feat] or [batch, nodes, feat], got {tuple(static_features.shape)}"
            )
        if not torch.isfinite(static_features).all():
            raise ValueError("Static features contain non-finite values")
        feature_nodes = int(static_features.shape[-2])
        if feature_nodes != int(coordinates.shape[1]):
            raise ValueError(
                "Static feature node dimension does not match coordinate node dimension: "
                f"{feature_nodes} vs {int(coordinates.shape[1])}"
            )
        metrics["health/static_features_present"] = 1.0
        metrics["health/static_feature_ndim"] = int(static_features.ndim)
        metrics["health/static_feature_dim"] = int(static_features.shape[-1])
        metrics["health/static_features_finite"] = 1.0
    else:
        metrics["health/static_features_present"] = 0.0

    if node_mask is not None:
        if not torch.isfinite(node_mask).all():
            raise ValueError("Node mask contains non-finite values")
        metrics["health/node_mask_present"] = 1.0
        metrics["health/node_mask_mean"] = float(node_mask.float().mean().item())
    else:
        metrics["health/node_mask_present"] = 0.0

    return metrics


def _print_health_metrics(metrics: Mapping[str, Any]) -> None:
    print("[startup-health] successful data loading and processing")
    for key in sorted(metrics):
        print(f"[startup-health] {key}={metrics[key]}")


def _unwrap_loss(loss_out: Any) -> Tuple[torch.Tensor, Dict[str, float]]:
    if torch.is_tensor(loss_out):
        return loss_out, {"train/loss": float(loss_out.detach().item())}
    if isinstance(loss_out, dict):
        if "loss" not in loss_out:
            raise KeyError("Diffusion loss dict must include a 'loss' tensor")
        loss = loss_out["loss"]
        metrics = {
            f"train/{key}": float(value.detach().item()) if torch.is_tensor(value) else float(value)
            for key, value in loss_out.items()
            if key != "loss"
        }
        metrics["train/loss"] = float(loss.detach().item())
        return loss, metrics
    raise TypeError(f"Unsupported loss output type: {type(loss_out).__name__}")


def _call_training_loss(
    diffusion: Any,
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    *,
    timesteps: Optional[torch.Tensor] = None,
    noise: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    coordinates, static_features, node_mask = _extract_batch_tensors(batch)
    extra_kwargs = {}
    if timesteps is not None:
        extra_kwargs["timesteps"] = timesteps
    if noise is not None:
        extra_kwargs["noise"] = noise
    candidate_calls = []
    if hasattr(diffusion, "training_loss"):
        candidate_calls.extend(
            [
                lambda: diffusion.training_loss(model=model, batch=batch, **extra_kwargs),
                lambda: diffusion.training_loss(
                    model=model,
                    coordinates=coordinates,
                    static_features=static_features,
                    node_mask=node_mask,
                    **extra_kwargs,
                ),
                lambda: diffusion.training_loss(model, batch, **extra_kwargs),
                lambda: diffusion.training_loss(model, coordinates, static_features, node_mask, **extra_kwargs),
            ]
        )
    if hasattr(diffusion, "loss"):
        candidate_calls.extend(
            [
                lambda: diffusion.loss(model=model, batch=batch, **extra_kwargs),
                lambda: diffusion.loss(
                    model=model,
                    coordinates=coordinates,
                    static_features=static_features,
                    node_mask=node_mask,
                    **extra_kwargs,
                ),
                lambda: diffusion.loss(model, batch, **extra_kwargs),
                lambda: diffusion.loss(model, coordinates, static_features, node_mask, **extra_kwargs),
            ]
        )
    last_error = None
    for call in candidate_calls:
        try:
            return _unwrap_loss(call())
        except TypeError as exc:
            last_error = exc
            continue
    raise RuntimeError(f"Unable to call diffusion training loss with supported signatures: {last_error}")


def _deterministic_eval_noise(
    batch: Mapping[str, Any],
    *,
    batch_index: int,
    eval_seed: int,
    num_timesteps: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    coordinates, _static_features, node_mask = _extract_batch_tensors(batch)
    batch_size = int(coordinates.shape[0])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(eval_seed) + int(batch_index))
    timesteps = torch.randint(
        low=0,
        high=max(int(num_timesteps) + 1, 1),
        size=(batch_size,),
        generator=generator,
        dtype=torch.int64,
    ).to(device=device)
    noise = torch.randn(coordinates.shape, generator=generator, dtype=coordinates.dtype)
    noise = noise.to(device=device)
    if node_mask is None:
        mask = torch.ones(batch_size, coordinates.shape[1], 1, device=device, dtype=coordinates.dtype)
    else:
        mask = node_mask.to(device=device, dtype=coordinates.dtype)
        if mask.dim() == 2:
            mask = mask.unsqueeze(-1)
    from ala2_scratch.diffusion import _remove_mean_with_mask  # local import to avoid cyclic import at module load

    return timesteps, _remove_mean_with_mask(noise, mask)


@torch.no_grad()
def _evaluate_average_loss(
    diffusion: Any,
    model: torch.nn.Module,
    dataloader: Iterable[Any],
    device: torch.device,
    max_batches: int,
    eval_seed: int,
) -> Dict[str, float]:
    model.eval()
    losses = []
    aux_totals: Dict[str, float] = {}
    aux_counts: Dict[str, int] = {}
    for batch_idx, raw_batch in enumerate(dataloader):
        if batch_idx >= max_batches:
            break
        batch = _to_device(_as_mapping(raw_batch), device)
        coordinates, _static_features, _node_mask = _extract_batch_tensors(batch)
        if not hasattr(diffusion, "timesteps"):
            raise AttributeError("Deterministic eval requires diffusion.timesteps")
        timesteps, noise = _deterministic_eval_noise(
            batch,
            batch_index=batch_idx,
            eval_seed=eval_seed,
            num_timesteps=int(getattr(diffusion, "timesteps")),
            device=device,
        )
        timesteps = timesteps.to(device=coordinates.device, dtype=torch.long)
        loss, metrics = _call_training_loss(diffusion, model, batch, timesteps=timesteps, noise=noise)
        losses.append(float(loss.detach().item()))
        for key, value in metrics.items():
            aux_totals[key] = aux_totals.get(key, 0.0) + float(value)
            aux_counts[key] = aux_counts.get(key, 0) + 1
    if not losses:
        return {}
    summary = {"val/loss": float(sum(losses) / len(losses))}
    for key, total in aux_totals.items():
        if key == "train/loss":
            continue
        summary[f"val/{key.split('/', 1)[-1]}"] = total / max(aux_counts.get(key, 1), 1)
    return summary


def _build_optimizer(model: torch.nn.Module, config: Mapping[str, Any]) -> torch.optim.Optimizer:
    train_cfg = config.get("train", {})
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 2e-4)),
        betas=(
            float(train_cfg.get("adam_beta1", 0.9)),
            float(train_cfg.get("adam_beta2", 0.999)),
        ),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        eps=float(train_cfg.get("adam_epsilon", 1e-8)),
    )


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
    *,
    steps_per_epoch: int,
):
    train_cfg = config.get("train", {})
    schedule = str(train_cfg.get("scheduler", "cosine")).lower()
    if schedule in {"none", "constant"}:
        return None
    if schedule == "cosine":
        total_steps = max(int(train_cfg.get("epochs", 1)) * max(int(steps_per_epoch), 1), 1)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    if schedule == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(int(train_cfg.get("step_lr_size", 10)), 1),
            gamma=float(train_cfg.get("step_lr_gamma", 0.5)),
        )
    raise ValueError(f"Unsupported scheduler '{schedule}'")


def _save_checkpoint(
    output_dir: Path,
    model: torch.nn.Module,
    eval_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    global_step: int,
    config: Mapping[str, Any],
    metrics: Mapping[str, Any],
    tag: str,
) -> Path:
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"{tag}.pt"
    uses_ema = eval_model is not model
    payload = {
        "model_state_dict": eval_model.state_dict(),
        "online_model_state_dict": model.state_dict(),
        "ema_state_dict": eval_model.state_dict() if uses_ema else None,
        "uses_ema": uses_ema,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "config": dict(config),
        "metrics": dict(metrics),
    }
    torch.save(payload, path)
    latest_path = ckpt_dir / "latest.pt"
    torch.save(payload, latest_path)
    return path


def _maybe_run_eval_hook(
    config: Mapping[str, Any],
    checkpoint_path: Path,
    output_dir: Path,
    device: torch.device,
    global_step: int,
    wandb_run: Any,
    log_fn,
):
    eval_cfg = config.get("eval", {})
    if not bool(eval_cfg.get("enable_eval_hook", True)):
        return
    eval_module = _try_import("ala2_scratch.eval")
    if eval_module is None:
        return

    for fn_name in ("evaluate_checkpoint", "run_evaluation", "evaluate"):
        fn = getattr(eval_module, fn_name, None)
        if not callable(fn):
            continue
        eval_output_dir = output_dir / "eval" / f"step_{global_step}_{checkpoint_path.stem}"
        kwargs = {
            "checkpoint_path": str(checkpoint_path),
            "config": dict(config),
            "output_dir": str(eval_output_dir),
            "device": str(device),
        }
        try:
            result = fn(**kwargs)
        except TypeError:
            try:
                result = fn(str(checkpoint_path), dict(config), str(eval_output_dir), str(device))
            except TypeError:
                continue
        except Exception as exc:
            print(f"[pretrain] eval_hook_failed checkpoint={checkpoint_path} error={exc}")
            if wandb_run is not None and callable(log_fn):
                log_fn(wandb_run, {"eval_hook/error": str(exc)}, step=global_step)
            if bool(eval_cfg.get("fail_on_hook_error", True)):
                raise
            return
        if isinstance(result, dict) and wandb_run is not None and callable(log_fn):
            log_fn(wandb_run, result, step=global_step)
        break


@torch.no_grad()
def _maybe_sample(
    diffusion: Any,
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    device: torch.device,
    output_dir: Path,
    epoch: int,
    global_step: int,
    wandb_run: Any,
    log_fn,
) -> Dict[str, float]:
    if not hasattr(diffusion, "sample"):
        return {}
    was_training = bool(getattr(model, "training", False))
    model.eval()
    try:
        coordinates, static_features, node_mask = _extract_batch_tensors(batch)
        batch_size = min(int(coordinates.shape[0]), 8)
        n_nodes = int(coordinates.shape[1])
        static_slice = None if static_features is None else static_features[:batch_size]
        mask_slice = None if node_mask is None else node_mask[:batch_size]
        candidate_calls = [
            lambda: diffusion.sample(model=model, batch_size=batch_size, n_nodes=n_nodes, static_features=static_slice, node_mask=mask_slice, device=device),
            lambda: diffusion.sample(model=model, shape=(batch_size, n_nodes, 3), static_features=static_slice, node_mask=mask_slice, device=device),
            lambda: diffusion.sample(model, batch_size, n_nodes, static_slice, mask_slice, device),
        ]
        samples = None
        for call in candidate_calls:
            try:
                samples = call()
                break
            except TypeError:
                continue
        if samples is None:
            return {}
        sample_path = output_dir / "samples"
        sample_path.mkdir(parents=True, exist_ok=True)
        torch.save(samples, sample_path / f"samples_epoch_{epoch:04d}.pt")
        if torch.is_tensor(samples):
            sample_tensor = samples
        elif hasattr(samples, "x0") and torch.is_tensor(samples.x0):
            sample_tensor = samples.x0
        elif isinstance(samples, dict):
            sample_tensor = None
            for key in ("coordinates", "positions", "x", "samples"):
                if torch.is_tensor(samples.get(key)):
                    sample_tensor = samples[key]
                    break
        else:
            sample_tensor = None
        metrics: Dict[str, float] = {}
        if torch.is_tensor(sample_tensor):
            metrics["sample/coord_abs_mean"] = float(sample_tensor.abs().mean().item())
            metrics["sample/coord_abs_max"] = float(sample_tensor.abs().max().item())
        if metrics and wandb_run is not None and callable(log_fn):
            log_fn(wandb_run, metrics, step=global_step)
        return metrics
    finally:
        if was_training:
            model.train()


def _default_output_dir(config: Mapping[str, Any]) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = str(config.get("run_name", "ala2_pretrain"))
    save_dir = config.get("save_dir")
    if save_dir:
        base = Path(str(save_dir))
    else:
        base = Path(str(config.get("output_root", "outputs/ala2_scratch")))
    if bool(config.get("allow_overwrite_save_dir", False)):
        return base
    return base / f"{run_name}_{timestamp}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone Ala2 coordinate-diffusion pretraining")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path("ala2_scratch/configs/ala2_base.json")),
        help="Path to JSON config file.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config values using dotted paths, e.g. train.epochs=10",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional explicit output directory. Defaults to outputs/ala2_scratch/<run>_<timestamp>.",
    )
    parser.add_argument(
        "--resume-if-exists",
        action="store_true",
        help="Allow writing into an existing --output-dir instead of erroring.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = _read_json(config_path)
    for override in args.override:
        _apply_override(config, override)

    output_dir = Path(args.output_dir).resolve() if args.output_dir else _default_output_dir(config).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume_if_exists:
        raise FileExistsError(
            f"Output directory '{output_dir}' already exists and is non-empty. "
            "Pass --resume-if-exists or choose a new --output-dir."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "resolved_config.json", config)

    device = _resolve_device(config)
    _seed_everything(int(config.get("seed", 42)))

    init_wandb, log_fn, _image_fn, artifact_fn = _load_wandb_helpers()
    wandb_run = None
    if callable(init_wandb):
        wandb_run = init_wandb(config, job_type="pretrain")
    else:
        try:
            import wandb

            wandb_cfg = dict(config.get("wandb", {}))
            mode = str(wandb_cfg.get("mode", "offline"))
            enabled = bool(wandb_cfg.get("enabled", True))
            if enabled:
                wandb_run = wandb.init(
                    project=wandb_cfg.get("project", "ala2-diffusion"),
                    entity=wandb_cfg.get("entity"),
                    group=wandb_cfg.get("group"),
                    name=wandb_cfg.get("name") or config.get("run_name"),
                    tags=wandb_cfg.get("tags", []),
                    config=config,
                    mode=mode,
                )
        except Exception:
            wandb_run = None

    def log_metrics(metrics: Mapping[str, Any], step: int) -> None:
        if wandb_run is None:
            return
        if callable(log_fn):
            log_fn(wandb_run, dict(metrics), step=step)
            return
        try:
            wandb_run.log(dict(metrics), step=step)
        except Exception:
            pass

    print(f"[pretrain] output_dir={output_dir}")
    print(f"[pretrain] device={device}")
    print(f"[pretrain] config={config_path}")

    build_dataloaders = _load_build_dataloaders()
    build_model = _load_build_model()
    build_diffusion = _load_build_diffusion()

    dataloaders = build_dataloaders(config)
    if isinstance(dataloaders, tuple):
        train_loader = dataloaders[0]
        val_loader = dataloaders[1] if len(dataloaders) > 1 else None
        metadata = dataloaders[2] if len(dataloaders) > 2 else {}
    elif isinstance(dataloaders, dict):
        train_loader = dataloaders["train"]
        val_loader = dataloaders.get("val") or dataloaders.get("valid")
        metadata = dataloaders.get("metadata", {})
    else:
        raise TypeError(f"Unsupported dataloader container: {type(dataloaders).__name__}")

    first_train_batch = _as_mapping(next(iter(train_loader)))
    startup_metrics = _compute_startup_health_metrics(first_train_batch)
    _print_health_metrics(startup_metrics)
    log_metrics(startup_metrics, step=0)
    _write_json(output_dir / "startup_health.json", startup_metrics)

    train_batch_device = _to_device(first_train_batch, device)
    model = build_model(config=config, metadata=metadata, example_batch=train_batch_device)
    model = model.to(device)

    try:
        diffusion = build_diffusion(config=config, metadata=metadata, example_batch=train_batch_device)
    except TypeError:
        diffusion = build_diffusion(config)

    optimizer = _build_optimizer(model, config)
    scheduler = _build_scheduler(optimizer, config, steps_per_epoch=len(train_loader))

    train_cfg = config.get("train", {})
    epochs = int(train_cfg.get("epochs", 100))
    ema_decay = float(train_cfg.get("ema_decay", 0.999))
    ema_start_step = int(
        train_cfg.get(
            "ema_start_step",
            0 if ema_decay <= 0.0 else max(int(round(1.0 / max(1.0 - ema_decay, 1e-8))), 1),
        )
    )
    log_every = max(int(train_cfg.get("log_every", 10)), 1)
    eval_every = max(int(train_cfg.get("eval_every", 1)), 1)
    sample_every = max(int(train_cfg.get("sample_every", 1)), 1)
    save_every = max(int(train_cfg.get("save_every", 1)), 1)
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
    val_batches = max(int(train_cfg.get("val_batches", 8)), 1)
    eval_seed = int(train_cfg.get("eval_seed", config.get("seed", 42)))
    ema_helper: Optional[_EMA] = None
    ema_model: Optional[torch.nn.Module] = None
    ema_active = False
    if ema_decay > 0.0:
        ema_helper = _EMA(ema_decay)
        ema_model = copy.deepcopy(model).to(device)
        ema_model.eval()
        for parameter in ema_model.parameters():
            parameter.requires_grad_(False)

    best_val = math.inf
    global_step = 0
    best_path: Optional[Path] = None
    final_metrics: Dict[str, Any] = {}

    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        epoch_start = time.time()
        for batch_idx, raw_batch in enumerate(train_loader):
            batch = _to_device(_as_mapping(raw_batch), device)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = _call_training_loss(diffusion, model, batch)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            if ema_helper is not None and ema_model is not None:
                if global_step < ema_start_step:
                    ema_model.load_state_dict(model.state_dict())
                elif not ema_active:
                    ema_model.load_state_dict(model.state_dict())
                    ema_active = True
                else:
                    ema_helper.update_model_average(ema_model, model)
            if scheduler is not None:
                scheduler.step()

            global_step += 1
            epoch_losses.append(float(loss.detach().item()))
            metrics = dict(metrics)
            metrics["train/grad_norm"] = float(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm)
            metrics["train/lr"] = float(optimizer.param_groups[0]["lr"])
            metrics["train/epoch"] = float(epoch + 1)
            metrics["train/step"] = float(global_step)

            if global_step % log_every == 0:
                print(
                    f"[pretrain] epoch={epoch + 1} step={global_step} "
                    f"loss={metrics.get('train/loss', float(loss.detach().item())):.6f} "
                    f"lr={metrics['train/lr']:.3e}"
                )
                log_metrics(metrics, step=global_step)

        epoch_metrics = {
            "train/epoch_loss_mean": float(sum(epoch_losses) / max(len(epoch_losses), 1)),
            "train/epoch_wall_sec": float(time.time() - epoch_start),
            "train/epoch": float(epoch + 1),
        }
        log_metrics(epoch_metrics, step=global_step)

        latest_metrics: Dict[str, Any] = dict(epoch_metrics)
        eval_model = ema_model if ema_model is not None and ema_active else model
        if val_loader is not None and (epoch + 1) % eval_every == 0:
            val_metrics = _evaluate_average_loss(
                diffusion,
                eval_model,
                val_loader,
                device,
                max_batches=val_batches,
                eval_seed=eval_seed,
            )
            if val_metrics:
                print(
                    f"[pretrain] epoch={epoch + 1} val_loss={val_metrics.get('val/loss', float('nan')):.6f}"
                )
                log_metrics(val_metrics, step=global_step)
                latest_metrics.update(val_metrics)
                final_metrics = dict(latest_metrics)

                if val_metrics.get("val/loss", math.inf) < best_val:
                    best_val = float(val_metrics["val/loss"])
                    best_path = _save_checkpoint(
                        output_dir,
                        model,
                        eval_model,
                        optimizer,
                        scheduler,
                        epoch=epoch + 1,
                        global_step=global_step,
                        config=config,
                        metrics=latest_metrics,
                        tag="best",
                    )
                    if wandb_run is not None and callable(artifact_fn):
                        artifact_fn(wandb_run, str(best_path), aliases=["best", "pretrain-best"])
                    _maybe_run_eval_hook(config, best_path, output_dir, device, global_step, wandb_run, log_fn)

        if (epoch + 1) % sample_every == 0:
            _maybe_sample(
                diffusion,
                eval_model,
                train_batch_device,
                device,
                output_dir,
                epoch + 1,
                global_step,
                wandb_run,
                log_fn,
            )

        if (epoch + 1) % save_every == 0:
            ckpt_path = _save_checkpoint(
                output_dir,
                model,
                eval_model,
                optimizer,
                scheduler,
                epoch=epoch + 1,
                global_step=global_step,
                config=config,
                metrics=latest_metrics,
                tag=f"epoch_{epoch + 1:04d}",
            )
            if wandb_run is not None and callable(artifact_fn):
                artifact_fn(wandb_run, str(ckpt_path), aliases=[f"epoch-{epoch + 1:04d}"])

    if best_path is not None and wandb_run is not None:
        try:
            wandb_run.summary["best_checkpoint_path"] = str(best_path)
            wandb_run.summary["best_val_loss"] = float(best_val)
        except Exception:
            pass

    final_path = _save_checkpoint(
        output_dir,
        model,
        ema_model if ema_model is not None else model,
        optimizer,
        scheduler,
        epoch=epochs,
        global_step=global_step,
        config=config,
        metrics=final_metrics or {"val/loss": best_val},
        tag="final",
    )
    _maybe_run_eval_hook(config, final_path, output_dir, device, global_step, wandb_run, log_fn)

    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
