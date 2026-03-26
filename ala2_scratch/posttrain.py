"""Minimal terminal-energy-only post-training for fixed-topology Ala2 diffusion."""

from __future__ import annotations

import argparse
import copy
import importlib
import inspect
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
import wandb

from ala2_scratch.mlff_energy import (
    MLFFEnergyConfig,
    MLFFEnergyOracle,
    atomic_numbers_from_metadata,
    beta_from_temperature,
    ensure_atomic_numbers,
)
from ala2_scratch.topology import build_one_hot_features


def _seed_everything(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _parse_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _apply_override(config: dict, override: str) -> None:
    if "=" not in override:
        raise ValueError(f"Override must have the form key=value, got '{override}'")
    key_path, raw_value = override.split("=", 1)
    value = _parse_value(raw_value)
    cursor = config
    parts = [part for part in key_path.split(".") if part]
    if not parts:
        raise ValueError(f"Invalid override path '{key_path}'")
    for part in parts[:-1]:
        next_cursor = cursor.get(part)
        if not isinstance(next_cursor, dict):
            next_cursor = {}
            cursor[part] = next_cursor
        cursor = next_cursor
    cursor[parts[-1]] = value


def _select_device(device_value: Optional[str]) -> torch.device:
    if device_value:
        device_str = str(device_value)
        if device_str == "cuda" and torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _flatten_dict(data: dict, prefix: str = "") -> dict:
    flat = {}
    for key, value in data.items():
        next_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_dict(value, next_key))
        else:
            flat[next_key] = value
    return flat


def _init_wandb(cfg: dict) -> Optional[wandb.sdk.wandb_run.Run]:
    wandb_cfg = cfg.get("wandb", {})
    if not wandb_cfg.get("enabled", True):
        return None
    mode = os.environ.get("WANDB_MODE", wandb_cfg.get("mode", "online"))
    if mode == "disabled":
        return None
    return wandb.init(
        entity=wandb_cfg.get("entity"),
        project=wandb_cfg.get("project", "ala2-diffusion"),
        group=wandb_cfg.get("group"),
        name=wandb_cfg.get("name"),
        tags=wandb_cfg.get("tags") or [],
        job_type="posttrain",
        mode=mode,
        config=_flatten_dict(cfg),
        settings=wandb.Settings(_disable_stats=True),
        reinit=True,
    )


def _merge_runtime_config(user_cfg: dict, checkpoint_cfg: Optional[dict]) -> dict:
    merged = copy.deepcopy(checkpoint_cfg or {})
    for key, value in user_cfg.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def _validate_checkpoint_section(section_name: str, runtime_cfg: dict, checkpoint_cfg: Optional[dict]) -> None:
    if not isinstance(checkpoint_cfg, dict):
        return
    runtime_section = runtime_cfg.get(section_name) or {}
    checkpoint_section = checkpoint_cfg.get(section_name) or {}
    if not isinstance(runtime_section, dict) or not isinstance(checkpoint_section, dict):
        return
    ignored_keys = {"device"}
    mismatches = {}
    for key, checkpoint_value in checkpoint_section.items():
        if key in ignored_keys or key not in runtime_section:
            continue
        if runtime_section[key] != checkpoint_value:
            mismatches[key] = {"checkpoint": checkpoint_value, "runtime": runtime_section[key]}
    if mismatches:
        raise ValueError(
            f"Runtime {section_name} config does not match the pretrained checkpoint: {mismatches}"
        )


def _resolve_class(module_name: str, candidate_names: Iterable[str]):
    module = importlib.import_module(module_name)
    for name in candidate_names:
        obj = getattr(module, name, None)
        if obj is not None:
            return obj
    raise AttributeError(
        f"None of {list(candidate_names)} found in module `{module_name}`."
    )


def _load_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict:
    checkpoint = torch.load(path, map_location=map_location)
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError(f"Unsupported checkpoint format in {path}")


def _infer_model_config(user_cfg: dict, checkpoint: dict) -> dict:
    checkpoint_cfg = checkpoint.get("config") if isinstance(checkpoint.get("config"), dict) else {}
    runtime_cfg = _merge_runtime_config(user_cfg, checkpoint_cfg)
    _validate_checkpoint_section("model", runtime_cfg, checkpoint_cfg)
    return copy.deepcopy(runtime_cfg.get("model", {}))


def _infer_diffusion_config(user_cfg: dict, checkpoint: dict) -> dict:
    checkpoint_cfg = checkpoint.get("config") if isinstance(checkpoint.get("config"), dict) else {}
    runtime_cfg = _merge_runtime_config(user_cfg, checkpoint_cfg)
    _validate_checkpoint_section("diffusion", runtime_cfg, checkpoint_cfg)
    return copy.deepcopy(runtime_cfg.get("diffusion", {}))


def _instantiate_model(checkpoint: dict, cfg: dict, device: torch.device):
    model_cls = _resolve_class(
        "ala2_scratch.model",
        ("Ala2EGNNDenoiser", "CoordinateEGNNDenoiser", "EGNNDenoiser"),
    )
    model_cfg = _infer_model_config(cfg, checkpoint)
    init_kwargs = dict(model_cfg)
    if "device" not in init_kwargs:
        init_kwargs["device"] = str(device)
    model = model_cls(**init_kwargs)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Pretrained model state does not match runtime architecture. Missing={missing}, unexpected={unexpected}"
        )
    model.to(device)
    return model


def _instantiate_diffusion(model, checkpoint: dict, cfg: dict, device: torch.device):
    diffusion_cls = _resolve_class(
        "ala2_scratch.diffusion",
        ("CoordinateDiffusion", "GaussianCoordinateDiffusion", "Ala2Diffusion"),
    )
    diffusion_cfg = _infer_diffusion_config(cfg, checkpoint)
    init_kwargs = dict(diffusion_cfg)
    if "device" not in init_kwargs:
        init_kwargs["device"] = str(device)
    constructors = (
        lambda: diffusion_cls(model=model, **init_kwargs),
        lambda: diffusion_cls(denoiser=model, **init_kwargs),
        lambda: diffusion_cls(**init_kwargs),
    )
    last_exc = None
    diffusion = None
    for constructor in constructors:
        try:
            diffusion = constructor()
            break
        except TypeError as exc:
            last_exc = exc
    if diffusion is None:
        raise TypeError(f"Unable to instantiate diffusion wrapper: {last_exc}") from last_exc
    if hasattr(diffusion, "model") and getattr(diffusion, "model") is None:
        diffusion.model = model
    if hasattr(diffusion, "denoiser") and getattr(diffusion, "denoiser") is None:
        diffusion.denoiser = model
    state_dict = checkpoint.get("diffusion_state_dict")
    if state_dict and hasattr(diffusion, "load_state_dict"):
        missing, unexpected = diffusion.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Pretrained diffusion state does not match runtime architecture. Missing={missing}, unexpected={unexpected}"
            )
    if isinstance(diffusion, torch.nn.Module):
        diffusion.to(device)
    return diffusion


def _parameter_l2(current: torch.nn.Module, reference: torch.nn.Module) -> torch.Tensor:
    total = None
    for current_param, reference_param in zip(current.parameters(), reference.parameters()):
        term = (current_param - reference_param.detach()).pow(2).mean()
        total = term if total is None else total + term
    if total is None:
        total = torch.tensor(0.0, device=next(current.parameters()).device)
    return total


def _get_trainable_module(model, diffusion):
    if isinstance(diffusion, torch.nn.Module):
        return diffusion
    return model


def _parse_sample_output(sample_output: Any) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    if hasattr(sample_output, "x0") and hasattr(sample_output, "log_prob_sum"):
        positions = getattr(sample_output, "x0")
        log_probs = getattr(sample_output, "log_prob_sum")
        extra = {}
        for key in ("chain", "timesteps", "x0_preds", "log_probs"):
            if hasattr(sample_output, key):
                extra[key] = getattr(sample_output, key)
        if positions is None or log_probs is None:
            raise ValueError("DiffusionSample must contain x0 and log_prob_sum.")
        return positions, log_probs, extra
    if isinstance(sample_output, dict):
        positions = None
        for key in ("samples", "positions", "x0"):
            value = sample_output.get(key)
            if value is not None:
                positions = value
                break
        log_probs = None
        for key in ("log_probs", "log_prob_sum", "trajectory_log_probs"):
            value = sample_output.get(key)
            if value is not None:
                log_probs = value
                break
        extra = {k: v for k, v in sample_output.items() if k not in {"samples", "positions", "x0", "log_probs", "log_prob_sum", "trajectory_log_probs"}}
        if positions is None or log_probs is None:
            raise ValueError("Sample dictionary must contain positions and log_probs.")
        return positions, log_probs, extra
    if isinstance(sample_output, (tuple, list)):
        if len(sample_output) < 2:
            raise ValueError("Sample tuple must contain at least positions and log_probs.")
        positions = sample_output[0]
        log_probs = sample_output[1]
        extra = sample_output[2] if len(sample_output) > 2 and isinstance(sample_output[2], dict) else {}
        return positions, log_probs, extra
    raise TypeError(f"Unsupported sample output type: {type(sample_output)!r}")


def _sample_with_policy(
    diffusion,
    *,
    batch_size: int,
    node_features: torch.Tensor,
    num_nodes: int,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    sample_fns = []
    for fn_name in ("sample_with_logprobs", "sample", "sample_batch"):
        fn = getattr(diffusion, fn_name, None)
        if fn is not None:
            sample_fns.append(fn)
    if not sample_fns:
        raise AttributeError("Diffusion object does not expose a sampling method.")

    kwargs = {
        "batch_size": batch_size,
        "node_features": node_features,
        "num_nodes": num_nodes,
        "return_logprobs": True,
        "return_intermediates": False,
    }
    last_error = None
    for fn in sample_fns:
        try:
            signature = inspect.signature(fn)
            supported = {k: v for k, v in kwargs.items() if k in signature.parameters}
            if "num_samples" in signature.parameters and "batch_size" in supported:
                supported["num_samples"] = supported.pop("batch_size")
            sample_output = fn(**supported)
            return _parse_sample_output(sample_output)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"All sampling entrypoints failed: {last_error}") from last_error


def _load_metadata(cfg: dict) -> dict:
    data_cfg = cfg.get("data", {})
    metadata_path = (
        data_cfg.get("metadata_path")
        or data_cfg.get("topology_path")
        or data_cfg.get("metadata_json")
    )
    if metadata_path:
        return _load_json(metadata_path)
    return {}


def _resolve_atomic_numbers(cfg: dict, metadata: dict) -> list[int]:
    if cfg.get("data", {}).get("atomic_numbers"):
        return ensure_atomic_numbers(cfg["data"]["atomic_numbers"])
    if metadata:
        return atomic_numbers_from_metadata(metadata)
    raise ValueError("Either `data.atomic_numbers` or `data.metadata_path` with atomic numbers is required.")


def _build_node_features(
    atomic_numbers: list[int],
    device: torch.device,
    feature_dim: Optional[int] = None,
) -> torch.Tensor:
    features_np, _feature_atomic_numbers = build_one_hot_features(atomic_numbers)
    features = torch.as_tensor(features_np, device=device, dtype=torch.float32)
    if feature_dim is not None and feature_dim > features.shape[1]:
        padding = torch.zeros(features.shape[0], feature_dim - features.shape[1], device=device)
        features = torch.cat([features, padding], dim=1)
    return features


def _default_output_dir(cfg: dict) -> Path:
    save_dir = Path(str(cfg.get("save_dir", "./outputs/ala2_posttrain")))
    if bool(cfg.get("allow_overwrite_save_dir", False)):
        return save_dir
    run_name = str(cfg.get("wandb", {}).get("name") or cfg.get("run_name") or "ala2_posttrain")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return save_dir / f"{run_name}_{timestamp}"


def _save_checkpoint(
    path: Path,
    *,
    step: int,
    cfg: dict,
    model,
    diffusion,
    metrics: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": int(step),
        "config": cfg,
        "model_state_dict": model.state_dict(),
        "metrics": metrics,
    }
    if isinstance(diffusion, torch.nn.Module):
        payload["diffusion_state_dict"] = diffusion.state_dict()
    torch.save(payload, path)


def _maybe_log_checkpoint(run, path: Path, alias: str) -> None:
    if run is None:
        return
    artifact = wandb.Artifact(run.name or "ala2-posttrain", type="model")
    artifact.add_file(str(path))
    run.log_artifact(artifact, aliases=[alias])


def _run_eval_if_available(
    *,
    cfg: dict,
    model,
    diffusion,
    step: int,
    run,
    save_dir: Path,
) -> dict:
    try:
        eval_mod = importlib.import_module("ala2_scratch.eval")
    except Exception:
        return {}

    for fn_name in ("evaluate_runtime", "evaluate_model", "evaluate_checkpoint"):
        fn = getattr(eval_mod, fn_name, None)
        if fn is None:
            continue
        try:
            metrics = fn(cfg=cfg, model=model, diffusion=diffusion, output_dir=save_dir / "eval" / f"step_{step}")
            if isinstance(metrics, dict):
                if run is not None:
                    run.log({f"eval/{k}": v for k, v in metrics.items()}, step=step)
                return metrics
        except TypeError:
            continue
        except Exception as exc:
            if run is not None:
                run.log({"eval/error": str(exc)}, step=step)
            print(f"[posttrain] eval_failed step={step} error={exc}")
            return {}
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal Ala2 terminal-energy post-training")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent / "configs" / "ala2_posttrain.json"),
        help="Path to JSON config file.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config values using dotted paths, e.g. train.steps=500",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional explicit output directory for post-training artifacts.",
    )
    parser.add_argument(
        "--resume-if-exists",
        action="store_true",
        help="Allow writing into an existing --output-dir instead of erroring.",
    )
    args = parser.parse_args()

    cfg = _load_json(args.config)
    for override in args.override:
        _apply_override(cfg, override)
    save_dir = Path(args.output_dir).resolve() if args.output_dir else _default_output_dir(cfg).resolve()
    if save_dir.exists() and any(save_dir.iterdir()) and not args.resume_if_exists:
        raise FileExistsError(
            f"Output directory '{save_dir}' already exists and is non-empty. "
            "Pass --resume-if-exists or choose a new --output-dir."
        )
    save_dir.mkdir(parents=True, exist_ok=True)
    _save_json(save_dir / "resolved_config.json", cfg)

    _seed_everything(int(cfg.get("seed", 42)))
    device = _select_device(cfg.get("device"))

    metadata = _load_metadata(cfg)
    atomic_numbers = _resolve_atomic_numbers(cfg, metadata)
    node_features = _build_node_features(
        atomic_numbers,
        device=device,
        feature_dim=cfg.get("model", {}).get("in_node_nf"),
    )

    checkpoint_path = cfg["pretrained_checkpoint"]
    reference_path = cfg.get("reference_checkpoint") or checkpoint_path
    base_checkpoint = _load_checkpoint(checkpoint_path, map_location="cpu")
    ref_checkpoint = _load_checkpoint(reference_path, map_location="cpu")

    model = _instantiate_model(base_checkpoint, cfg, device)
    diffusion = _instantiate_diffusion(model, base_checkpoint, cfg, device)

    reference_model = _instantiate_model(ref_checkpoint, cfg, device)
    reference_diffusion = _instantiate_diffusion(reference_model, ref_checkpoint, cfg, device)
    del reference_diffusion
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad_(False)

    mlff_cfg = MLFFEnergyConfig(**cfg.get("mlff", {}))
    oracle = MLFFEnergyOracle(atomic_numbers, mlff_cfg)

    train_cfg = cfg.get("train", {})
    total_steps = int(train_cfg.get("steps", 1000))
    batch_size = int(train_cfg.get("batch_size", 16))
    lr = float(train_cfg.get("lr", 1e-5))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    log_every = int(train_cfg.get("log_every", 10))
    save_every = int(train_cfg.get("save_every", 100))
    eval_every = int(train_cfg.get("eval_every", 200))
    normalize_advantages = bool(train_cfg.get("normalize_advantages", True))
    reference_l2_weight = float(train_cfg.get("reference_l2_weight", 1e-4))
    reward_scale = float(train_cfg.get("reward_scale", 1.0))
    temperature_kelvin = float(train_cfg.get("temperature_kelvin", 300.0))
    energy_offset = train_cfg.get("energy_offset")
    beta = float(train_cfg.get("beta", beta_from_temperature(temperature_kelvin)))

    trainable = _get_trainable_module(model, diffusion)
    optimizer = torch.optim.Adam(trainable.parameters(), lr=lr, weight_decay=weight_decay)

    run = _init_wandb(cfg)
    if run is not None:
        run.summary["mlff/backend"] = oracle.backend
        run.summary["posttrain/beta"] = beta

    best_metric_name = train_cfg.get("best_metric_name", "reward_mean")
    best_metric_mode = train_cfg.get("best_metric_mode", "max")
    best_metric_value = -math.inf if best_metric_mode == "max" else math.inf
    best_checkpoint_path = save_dir / "checkpoint_best.pt"

    train_start = time.time()
    for step in range(1, total_steps + 1):
        trainable.train()
        optimizer.zero_grad(set_to_none=True)

        positions, log_probs, sample_info = _sample_with_policy(
            diffusion,
            batch_size=batch_size,
            node_features=node_features,
            num_nodes=len(atomic_numbers),
        )
        if not isinstance(positions, torch.Tensor):
            raise TypeError("Sampled positions must be a torch.Tensor.")
        if not isinstance(log_probs, torch.Tensor):
            raise TypeError("Sampled log_probs must be a torch.Tensor.")
        positions = positions.to(device=device, dtype=torch.float32)
        log_probs = log_probs.to(device=device, dtype=torch.float32).view(-1)

        energies = oracle(positions).view(-1)
        energy_baseline = torch.as_tensor(float(energy_offset), device=device) if energy_offset is not None else energies.mean().detach()
        rewards = -reward_scale * beta * (energies - energy_baseline)
        if normalize_advantages:
            advantages = (rewards - rewards.mean()) / rewards.std(unbiased=False).clamp(min=1e-6)
        else:
            advantages = rewards

        policy_loss = -(advantages.detach() * log_probs).mean()
        ref_loss = reference_l2_weight * _parameter_l2(model, reference_model)
        total_loss = policy_loss + ref_loss
        total_loss.backward()
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(trainable.parameters(), grad_clip)
        optimizer.step()

        metrics = {
            "step": step,
            "reward_mean": float(rewards.mean().item()),
            "reward_std": float(rewards.std(unbiased=False).item()),
            "energy_mean": float(energies.mean().item()),
            "energy_std": float(energies.std(unbiased=False).item()),
            "adv_mean": float(advantages.mean().item()),
            "adv_std": float(advantages.std(unbiased=False).item()),
            "logprob_mean": float(log_probs.mean().item()),
            "policy_loss": float(policy_loss.item()),
            "ref_loss": float(ref_loss.item()),
            "total_loss": float(total_loss.item()),
            "elapsed_sec": float(time.time() - train_start),
        }
        if isinstance(sample_info, dict):
            for key, value in sample_info.items():
                if isinstance(value, (int, float)):
                    metrics[f"sample/{key}"] = float(value)

        if run is not None and (step % log_every == 0 or step == 1):
            wandb.log({f"train/{k}": v for k, v in metrics.items() if k != "step"}, step=step)

        latest_path = save_dir / "checkpoint_latest.pt"
        if step % save_every == 0 or step == total_steps:
            _save_checkpoint(
                latest_path,
                step=step,
                cfg=cfg,
                model=model,
                diffusion=diffusion,
                metrics=metrics,
            )

        eval_metrics = {}
        if eval_every > 0 and (step % eval_every == 0 or step == total_steps):
            eval_metrics = _run_eval_if_available(
                cfg=cfg,
                model=model,
                diffusion=diffusion,
                step=step,
                run=run,
                save_dir=save_dir,
            )

        tracked_value = eval_metrics.get(best_metric_name)
        if tracked_value is None:
            tracked_value = metrics.get(best_metric_name)
        if tracked_value is not None:
            is_better = (
                float(tracked_value) > best_metric_value
                if best_metric_mode == "max"
                else float(tracked_value) < best_metric_value
            )
            if is_better:
                best_metric_value = float(tracked_value)
                _save_checkpoint(
                    best_checkpoint_path,
                    step=step,
                    cfg=cfg,
                    model=model,
                    diffusion=diffusion,
                    metrics={**metrics, **{f"eval/{k}": v for k, v in eval_metrics.items()}},
                )
                _maybe_log_checkpoint(run, best_checkpoint_path, "best")

    if run is not None:
        if best_metric_value not in (-math.inf, math.inf):
            run.summary[f"best/{best_metric_name}"] = best_metric_value
        run.finish()


if __name__ == "__main__":
    main()
