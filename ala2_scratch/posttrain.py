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
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
import wandb

from ala2_scratch.data import load_processed_dataset
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


def _state_dict_to_cpu(module: torch.nn.Module) -> dict[str, Any]:
    cpu_state = {}
    for key, value in module.state_dict().items():
        if torch.is_tensor(value):
            cpu_state[key] = value.detach().cpu()
        else:
            cpu_state[key] = value
    return cpu_state


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
        extra = {
            k: v
            for k, v in sample_output.items()
            if k not in {"samples", "positions", "x0", "log_probs", "log_prob_sum"}
        }
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


def _sample_rollout_no_grad(
    diffusion,
    *,
    batch_size: int,
    node_features: torch.Tensor,
    num_nodes: int,
) -> Any:
    sample_fn = getattr(diffusion, "sample_rollout", None)
    if sample_fn is None:
        raise AttributeError("Diffusion object does not expose `sample_rollout`.")
    with torch.no_grad():
        return sample_fn(
            batch_size=batch_size,
            node_features=node_features,
            num_nodes=num_nodes,
            move_to_cpu=True,
        )


def _replay_policy_log_probs(
    diffusion,
    *,
    rollout: Any,
    node_features: torch.Tensor,
    skip_prefix: int = 0,
    tail_steps: Optional[int] = None,
) -> torch.Tensor:
    replay_fn = getattr(diffusion, "replay_log_probs", None)
    if replay_fn is None:
        raise AttributeError("Diffusion object does not expose `replay_log_probs`.")
    replay_output = replay_fn(
        rollout=rollout,
        node_features=node_features,
        skip_prefix=skip_prefix,
        tail_steps=tail_steps,
    )
    if isinstance(replay_output, dict):
        for key in ("log_prob_mean", "log_probs", "log_prob_sum"):
            value = replay_output.get(key)
            if isinstance(value, torch.Tensor):
                return value
        raise ValueError("Replay output dict must contain a tensor log-prob entry.")
    if isinstance(replay_output, torch.Tensor):
        return replay_output
    raise TypeError(f"Unsupported replay output type: {type(replay_output)!r}")


def _slice_rollout(rollout: Any, start: int, end: int) -> Any:
    rollout_cls = type(rollout)
    return rollout_cls(
        positions=rollout.positions[start:end],
        z_chain=rollout.z_chain[start:end],
        timesteps=rollout.timesteps[start:end],
    )


def _index_rollout(rollout: Any, indices: torch.Tensor) -> Any:
    rollout_cls = type(rollout)
    index_tensor = indices.detach().to(device=rollout.positions.device, dtype=torch.long)
    return rollout_cls(
        positions=rollout.positions.index_select(0, index_tensor),
        z_chain=rollout.z_chain.index_select(0, index_tensor),
        timesteps=rollout.timesteps.index_select(0, index_tensor),
    )


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
        "model_state_dict": _state_dict_to_cpu(model),
        "metrics": metrics,
    }
    if isinstance(diffusion, torch.nn.Module):
        payload["diffusion_state_dict"] = _state_dict_to_cpu(diffusion)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        torch.save(payload, tmp_path, _use_new_zipfile_serialization=False)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


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
    sampling_microbatch_size = int(train_cfg.get("sampling_microbatch_size", batch_size))
    replay_microbatch_size = int(train_cfg.get("rollout_microbatch_size", batch_size))
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
    logprob_skip_prefix = int(train_cfg.get("logprob_skip_prefix", 0))
    logprob_tail_steps = train_cfg.get("logprob_tail_steps")
    reward_baseline_mode = str(train_cfg.get("reward_baseline_mode", "batch_mean")).strip().lower()
    reward_baseline_momentum = float(train_cfg.get("reward_baseline_momentum", 0.95))
    anchor_weight = float(train_cfg.get("anchor_weight", 0.0))
    anchor_batch_size = int(train_cfg.get("anchor_batch_size", batch_size))
    replay_subset_size = train_cfg.get("replay_subset_size")
    replay_subset_mode = str(train_cfg.get("replay_subset_mode", "abs_advantage")).strip().lower()
    beta = float(train_cfg.get("beta", beta_from_temperature(temperature_kelvin)))

    trainable = _get_trainable_module(model, diffusion)
    optimizer = torch.optim.Adam(trainable.parameters(), lr=lr, weight_decay=weight_decay)

    anchor_positions: Optional[torch.Tensor] = None
    if anchor_weight > 0.0:
        dataset_path = cfg.get("data", {}).get("dataset_path")
        topology_path = (
            cfg.get("data", {}).get("topology_path")
            or cfg.get("data", {}).get("metadata_path")
        )
        split_path = cfg.get("data", {}).get("split_path")
        if not dataset_path or not topology_path:
            raise ValueError("Anchor loss requires data.dataset_path and data.topology_path/metadata_path.")
        processed = load_processed_dataset(dataset_path, topology_path, split_path=split_path)
        anchor_positions = torch.as_tensor(
            processed.positions[processed.split_indices["train"]],
            dtype=torch.float32,
        )

    run = _init_wandb(cfg)
    if run is not None:
        run.summary["mlff/backend"] = oracle.backend
        run.summary["posttrain/beta"] = beta

    best_metric_name = train_cfg.get("best_metric_name", "reward_mean")
    best_metric_mode = train_cfg.get("best_metric_mode", "max")
    best_metric_value = -math.inf if best_metric_mode == "max" else math.inf
    best_checkpoint_path = save_dir / "checkpoint_best.pt"
    running_energy_baseline: Optional[torch.Tensor] = None

    train_start = time.time()
    for step in range(1, total_steps + 1):
        trainable.train()
        optimizer.zero_grad(set_to_none=True)

        sampling_batch_size = min(max(1, sampling_microbatch_size), batch_size)
        sampling_batch_count = max(1, math.ceil(batch_size / sampling_batch_size))
        reward_chunks = []
        energy_chunks = []
        advantage_chunks = []
        logprob_chunks = []
        sample_metric_totals: Dict[str, float] = {}
        policy_loss_value = 0.0

        rollout_chunks = []
        for microbatch_idx in range(sampling_batch_count):
            current_batch_size = min(sampling_batch_size, batch_size - microbatch_idx * sampling_batch_size)
            if current_batch_size <= 0:
                continue
            rollout = _sample_rollout_no_grad(
                diffusion,
                batch_size=current_batch_size,
                node_features=node_features,
                num_nodes=len(atomic_numbers),
            )
            positions = rollout.positions.to(device=device, dtype=torch.float32)
            energies = oracle(positions).view(-1).detach().cpu()
            rollout_chunks.append(
                {
                    "batch_size": current_batch_size,
                    "rollout": rollout,
                    "energies": energies,
                }
            )
            energy_chunks.append(energies)

        if not rollout_chunks:
            raise RuntimeError("No rollout chunks were produced for policy optimization.")

        all_energies = torch.cat(energy_chunks, dim=0).to(device=device, dtype=torch.float32)
        if reward_baseline_mode == "batch_mean":
            energy_baseline = (
                torch.as_tensor(float(energy_offset), device=device, dtype=all_energies.dtype)
                if energy_offset is not None
                else all_energies.mean().detach()
            )
        elif reward_baseline_mode == "fixed":
            if energy_offset is None:
                raise ValueError("train.energy_offset is required when reward_baseline_mode='fixed'.")
            energy_baseline = torch.as_tensor(float(energy_offset), device=device, dtype=all_energies.dtype)
        elif reward_baseline_mode == "ema":
            if running_energy_baseline is None:
                energy_baseline = all_energies.mean().detach()
            else:
                energy_baseline = running_energy_baseline.to(device=device, dtype=all_energies.dtype)
        else:
            raise ValueError(
                f"Unsupported reward_baseline_mode={reward_baseline_mode!r}. "
                "Expected one of {'batch_mean', 'ema', 'fixed'}."
            )

        rewards = -reward_scale * beta * (all_energies - energy_baseline)
        if normalize_advantages:
            advantages = (rewards - rewards.mean()) / rewards.std(unbiased=False).clamp(min=1e-6)
        else:
            advantages = rewards

        selected_global_indices: Optional[torch.Tensor] = None
        total_replay_count = int(advantages.numel())
        if replay_subset_size is not None:
            replay_subset_size_int = max(1, min(int(replay_subset_size), int(advantages.numel())))
            if replay_subset_size_int < int(advantages.numel()):
                if replay_subset_mode == "abs_advantage":
                    selected_global_indices = torch.topk(
                        advantages.abs(),
                        k=replay_subset_size_int,
                        largest=True,
                    ).indices.sort().values
                elif replay_subset_mode == "reward":
                    selected_global_indices = torch.topk(
                        rewards,
                        k=replay_subset_size_int,
                        largest=True,
                    ).indices.sort().values
                elif replay_subset_mode == "random":
                    selected_global_indices = torch.randperm(
                        int(advantages.numel()),
                        device=advantages.device,
                    )[:replay_subset_size_int].sort().values
                else:
                    raise ValueError(
                        f"Unsupported replay_subset_mode={replay_subset_mode!r}. "
                        "Expected one of {'abs_advantage', 'reward', 'random'}."
                    )
                total_replay_count = replay_subset_size_int

        if reward_baseline_mode == "ema":
            step_energy_mean = all_energies.mean().detach().cpu()
            if running_energy_baseline is None:
                running_energy_baseline = step_energy_mean
            else:
                running_energy_baseline = (
                    reward_baseline_momentum * running_energy_baseline
                    + (1.0 - reward_baseline_momentum) * step_energy_mean
                )

        offset = 0
        for chunk in rollout_chunks:
            current_batch_size = int(chunk["batch_size"])
            next_offset = offset + current_batch_size
            chunk_advantages = advantages[offset:next_offset]
            chunk_selected_indices: Optional[torch.Tensor] = None
            if selected_global_indices is not None:
                keep_mask = (selected_global_indices >= offset) & (selected_global_indices < next_offset)
                if bool(keep_mask.any()):
                    chunk_selected_indices = (selected_global_indices[keep_mask] - offset).to(dtype=torch.long)
                else:
                    offset = next_offset
                    continue
            replay_batch_size = min(max(1, replay_microbatch_size), current_batch_size)
            replay_target_count = int(chunk_selected_indices.numel()) if chunk_selected_indices is not None else current_batch_size
            replay_batch_count = max(1, math.ceil(replay_target_count / replay_batch_size))
            local_offset = 0
            chunk_logprob_parts = []
            for replay_idx in range(replay_batch_count):
                local_end = min(replay_target_count, local_offset + replay_batch_size)
                if chunk_selected_indices is None:
                    sub_rollout = _slice_rollout(chunk["rollout"], local_offset, local_end)
                    sub_advantages = chunk_advantages[local_offset:local_end]
                else:
                    sub_indices = chunk_selected_indices[local_offset:local_end]
                    sub_rollout = _index_rollout(chunk["rollout"], sub_indices.cpu())
                    sub_advantages = chunk_advantages.index_select(0, sub_indices)
                sub_log_probs = _replay_policy_log_probs(
                    diffusion,
                    rollout=sub_rollout,
                    node_features=node_features,
                    skip_prefix=logprob_skip_prefix,
                    tail_steps=logprob_tail_steps,
                ).to(device=device, dtype=torch.float32).view(-1)
                micro_policy_loss = -(
                    sub_advantages.detach() * sub_log_probs
                ).sum() / max(1, total_replay_count if selected_global_indices is not None else batch_size)
                micro_policy_loss.backward()
                policy_loss_value += float(micro_policy_loss.item())
                chunk_logprob_parts.append(sub_log_probs.detach())
                local_offset = local_end

            if chunk_selected_indices is None:
                reward_chunks.append(rewards[offset:next_offset].detach())
                advantage_chunks.append(chunk_advantages.detach())
            else:
                reward_chunks.append(rewards[offset:next_offset].index_select(0, chunk_selected_indices).detach())
                advantage_chunks.append(chunk_advantages.index_select(0, chunk_selected_indices).detach())
            logprob_chunks.append(torch.cat(chunk_logprob_parts, dim=0))
            offset = next_offset

        energies = all_energies

        policy_loss = torch.tensor(policy_loss_value, device=device)
        ref_loss = reference_l2_weight * _parameter_l2(model, reference_model)
        anchor_loss = torch.tensor(0.0, device=device)
        anchor_eps_mse = torch.tensor(float("nan"), device=device)
        if anchor_positions is not None and anchor_weight > 0.0:
            anchor_indices = torch.randint(
                low=0,
                high=int(anchor_positions.shape[0]),
                size=(max(1, anchor_batch_size),),
            )
            anchor_coords = anchor_positions[anchor_indices].to(device=device, dtype=torch.float32)
            anchor_terms = diffusion.training_loss(
                model=model,
                coordinates=anchor_coords,
                node_features=node_features,
            )
            anchor_loss = anchor_weight * anchor_terms["loss"]
            anchor_eps_mse = anchor_terms["eps_mse"].detach()
        ref_loss.backward()
        if anchor_loss.requires_grad:
            anchor_loss.backward()
        total_loss = policy_loss + ref_loss + anchor_loss
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(trainable.parameters(), grad_clip)
        optimizer.step()

        replayed_rewards = torch.cat(reward_chunks, dim=0) if reward_chunks else rewards.detach()
        replayed_advantages = torch.cat(advantage_chunks, dim=0) if advantage_chunks else advantages.detach()
        log_probs = torch.cat(logprob_chunks, dim=0) if logprob_chunks else torch.empty(0, device=device)

        metrics = {
            "step": step,
            "reward_mean": float(rewards.mean().item()),
            "reward_std": float(rewards.std(unbiased=False).item()),
            "energy_mean": float(energies.mean().item()),
            "energy_std": float(energies.std(unbiased=False).item()),
            "energy_baseline": float(energy_baseline.item()) if energy_baseline is not None else float("nan"),
            "adv_mean": float(replayed_advantages.mean().item()),
            "adv_std": float(replayed_advantages.std(unbiased=False).item()),
            "logprob_mean": float(log_probs.mean().item()) if log_probs.numel() > 0 else float("nan"),
            "policy_loss": float(policy_loss.item()),
            "ref_loss": float(ref_loss.item()),
            "anchor_loss": float(anchor_loss.item()),
            "anchor_eps_mse": float(anchor_eps_mse.item()) if torch.isfinite(anchor_eps_mse) else float("nan"),
            "total_loss": float(total_loss.item()),
            "replay_count": int(log_probs.numel()),
            "elapsed_sec": float(time.time() - train_start),
        }
        for key, total_value in sample_metric_totals.items():
            metrics[f"sample/{key}"] = float(total_value / batch_size)

        if run is not None and (step % log_every == 0 or step == 1):
            wandb.log({f"train/{k}": v for k, v in metrics.items() if k != "step"}, step=step)
        if step % log_every == 0 or step == 1:
            print(
                "[posttrain] "
                f"step={step} reward_mean={metrics['reward_mean']:.4f} "
                f"energy_mean={metrics['energy_mean']:.4f} "
                f"policy_loss={metrics['policy_loss']:.4f} "
                f"ref_loss={metrics['ref_loss']:.4e}"
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

        latest_path = save_dir / "checkpoint_latest.pt"
        if step % save_every == 0 or step == total_steps:
            _save_checkpoint(
                latest_path,
                step=step,
                cfg=cfg,
                model=model,
                diffusion=diffusion,
                metrics={**metrics, **{f"eval/{k}": v for k, v in eval_metrics.items()}},
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
