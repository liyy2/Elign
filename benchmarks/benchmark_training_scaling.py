#!/usr/bin/env python
import argparse
import csv
import json
import os
import pickle
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
EDM_SOURCE_ROOT = REPO_ROOT / "edm_source"
import sys

for path in (REPO_ROOT, EDM_SOURCE_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from edm_source.configs.datasets_config import get_dataset_info
from edm_source.qm9.dataset import retrieve_dataloaders
from edm_source.qm9.models import get_model
from verl_diffusion.dataloader.dataloader import EDMDataLoader
from verl_diffusion.model.edm_model import EDMModel
from verl_diffusion.worker.actor.edm_actor import EDMActor
from verl_diffusion.worker.rollout.edm_rollout import EDMRollout
from verl_diffusion.trainer.ddpo_trainer import DDPOTrainer


def _parse_int_list(value: str) -> List[int]:
    items = []
    for token in str(value).split(","):
        token = token.strip()
        if not token:
            continue
        items.append(int(token))
    if not items:
        raise ValueError("Empty list")
    return items


def _select_device(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    # Prefer an explicit index for reproducibility and to avoid accidental CPU runs.
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _load_yaml(path: Path) -> Dict[str, Any]:
    data = OmegaConf.to_container(OmegaConf.load(str(path)), resolve=True)
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict YAML at {path}")
    return data


def _load_edm_config(args_pickle: Path) -> Any:
    with open(args_pickle, "rb") as f:
        return pickle.load(f)


def _read_last_jsonl(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    last = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                last = json.loads(line)
            except Exception:
                continue
    return last


def run_one(
    cfg: Dict[str, Any],
    reward_type: str,
    n_nodes: int,
    device: torch.device,
    *,
    epochs: int,
    sample_group_size: int,
    each_prompt_sample: int,
    micro_batch_size: int,
    train_micro_batch_size: int,
    time_step: int,
    dft_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = deepcopy(cfg)

    # Minimal deterministic single-process setup.
    cfg["distributed"] = {"rank": 0, "world_size": 1, "local_rank": 0, "is_main_process": True}

    cfg.setdefault("dataloader", {})
    cfg.setdefault("model", {})
    cfg.setdefault("train", {})
    cfg.setdefault("reward", {})
    cfg.setdefault("wandb", {})

    cfg["dataloader"]["epoches"] = int(epochs)
    cfg["dataloader"]["sample_group_size"] = int(sample_group_size)
    cfg["dataloader"]["each_prompt_sample"] = int(each_prompt_sample)
    cfg["dataloader"]["micro_batch_size"] = int(micro_batch_size)
    cfg["train"]["train_micro_batch_size"] = int(train_micro_batch_size)
    cfg["model"]["time_step"] = int(time_step)

    cfg["wandb"]["enabled"] = False
    cfg["reward"]["type"] = str(reward_type)
    if dft_overrides and str(reward_type).lower() in {"dft", "pyscf"}:
        # Keep DFT settings local to these benchmarks so we can tune speed/robustness.
        cfg["reward"].update({k: v for k, v in dft_overrides.items() if v is not None})

    # Terminal-only rewards for speed benchmarking across oracles.
    cfg["reward"].setdefault("shaping", {})
    cfg["reward"]["shaping"]["enabled"] = False
    cfg["reward"]["use_energy"] = bool(cfg["reward"].get("use_energy", False))

    # Disable extras that add noise to timing.
    cfg["train"]["force_alignment_enabled"] = False
    cfg["train"]["force_alignment_weight"] = 0.0
    cfg["filters"] = {"enable_filtering": False, "enable_penalty": False}

    # Fix molecule size and tensor shape.
    cfg["dataloader"]["nodes_dist_fixed"] = int(n_nodes)
    cfg["dataloader"]["max_n_nodes"] = int(n_nodes)

    # Avoid checkpointing I/O in benchmarks.
    cfg["best_checkpoint_metric"] = "nonexistent_metric"
    cfg["best_checkpoint_mode"] = "max"
    cfg["save_interval"] = 1_000_000

    # Timing log for machine-readable parsing.
    cfg["timing"] = {"enabled": True, "jsonl_path": "timing.jsonl"}

    # Resolve save path.
    run_name = f"bench_train_{reward_type}_n{n_nodes}_{int(time.time())}"
    save_path = Path("benchmarks/runs") / run_name
    save_path.mkdir(parents=True, exist_ok=True)
    cfg["save_path"] = str(save_path)

    # Load EDM args/config.
    edm_config = _load_edm_config(Path(cfg["model"]["config"]))
    dataset_name = getattr(edm_config, "dataset", "")
    if isinstance(dataset_name, str) and "qm9" in dataset_name:
        edm_config.datadir = "qm9/temp"

    # Set device
    edm_config.cuda = device.type == "cuda"
    edm_config.device = device
    if hasattr(edm_config, "no_cuda"):
        edm_config.no_cuda = not edm_config.cuda

    dataset_info = get_dataset_info(edm_config.dataset, edm_config.remove_h)

    dataloaders, _ = retrieve_dataloaders(edm_config)
    flow, nodes_dist, prop_dist = get_model(edm_config, device, dataset_info, dataloaders["train"])
    flow.to(device)

    model = EDMModel(flow, edm_config).to(device)
    model.load(model_path=str(cfg["model"]["model_path"]))
    model.eval()

    dataloader = EDMDataLoader(
        config=cfg,
        dataset_info=dataset_info,
        nodes_dist=nodes_dist,
        prop_dist=prop_dist,
        device=device,
        condition=False,
        num_batches=int(cfg["dataloader"]["epoches"]),
        rank=0,
        world_size=1,
        base_seed=int(cfg.get("seed", 0) or 0),
    )

    rollout = EDMRollout(model, cfg)

    # Rewarder selection mirrors run_verl_diffusion.py
    reward_cfg = cfg.get("reward", {}) or {}
    reward_kind = str(reward_cfg.get("type", reward_type)).lower()

    if reward_kind in {"uma", "mlff", "polar_mace"}:
        from verl_diffusion.worker.reward.force import UMAForceReward

        rewarder = UMAForceReward(
            dataset_info,
            condition=False,
            mlff_model=reward_cfg.get("mlff_model", "polar-1-m"),
            mlff_backend=reward_cfg.get(
                "mlff_backend",
                reward_kind if reward_kind in {"uma", "polar_mace"} else None,
            ),
            position_scale=float(getattr(edm_config, "normalize_factors", [1.0])[0]),
            device=device,
            mlff_device=device if device.type == "cuda" else "cpu",
            mlff_charge=reward_cfg.get("mlff_charge", reward_cfg.get("charge", 0)),
            mlff_spin=reward_cfg.get("mlff_spin", reward_cfg.get("spin", 1)),
            mlff_external_field=reward_cfg.get("mlff_external_field", [0.0, 0.0, 0.0]),
            mlff_default_dtype=reward_cfg.get("mlff_default_dtype", "float32"),
            shaping=reward_cfg.get("shaping", {}),
            use_energy=reward_cfg.get("use_energy", False),
        )
    elif reward_kind == "xtb":
        from verl_diffusion.worker.reward.xtb_force import XTBForceReward

        rewarder = XTBForceReward(
            dataset_info=dataset_info,
            position_scale=float(getattr(edm_config, "normalize_factors", [1.0])[0]),
            device="cpu",
            param=reward_cfg.get("xtb_param", reward_cfg.get("param", "GFN2xTB")),
        )
    elif reward_kind in {"dft", "pyscf"}:
        from verl_diffusion.worker.reward.pyscf_dft_force import PySCFDFTForceReward

        rewarder = PySCFDFTForceReward(
            dataset_info=dataset_info,
            position_scale=float(getattr(edm_config, "normalize_factors", [1.0])[0]),
            device="cpu",
            xc=reward_cfg.get("xc", reward_cfg.get("dft_xc", "pbe")),
            basis=reward_cfg.get("basis", reward_cfg.get("dft_basis", "minao")),
            max_cycle=int(reward_cfg.get("max_cycle", 10)),
            grids_level=int(reward_cfg.get("grids_level", 1)),
            density_fit=bool(reward_cfg.get("density_fit", True)),
            accept_unconverged=bool(reward_cfg.get("accept_unconverged", True)),
            min_interatomic_distance=float(reward_cfg.get("min_interatomic_distance", 0.3)),
        )
    else:
        raise ValueError(f"Unsupported reward.type '{reward_kind}'")

    actor = EDMActor(model, cfg)

    trainer = DDPOTrainer(
        config=cfg,
        model=model,
        dataset_info=dataset_info,
        device=device,
        dataloader=dataloader,
        rollout=rollout,
        rewarder=rewarder,
        actor=actor,
        filters=None,
    )

    trainer.fit()

    last = _read_last_jsonl(save_path / "timing.jsonl") or {}
    # Record device placement explicitly to avoid confusion when comparing runs.
    last["benchmark/model_device"] = str(device)
    reward_device = "unknown"
    if hasattr(rewarder, "mlff_device"):
        reward_device = str(getattr(rewarder, "mlff_device"))
    elif hasattr(rewarder, "device"):
        reward_device = str(getattr(rewarder, "device"))
    last["benchmark/reward_device"] = reward_device
    last["benchmark/torch_cuda_available"] = bool(torch.cuda.is_available())
    # For MLFF rewards, capture whether the model ended up on GPU after the first call.
    pred_device = ""
    param_device = ""
    if hasattr(rewarder, "mlff_predictor") and getattr(rewarder, "mlff_predictor") is not None:
        pred_device = str(getattr(rewarder.mlff_predictor, "device", ""))
        try:
            if hasattr(rewarder.mlff_predictor, "model"):
                param_device = str(next(rewarder.mlff_predictor.model.parameters()).device)
        except Exception:
            param_device = ""
    last["benchmark/mlff_predictor_device"] = pred_device
    last["benchmark/mlff_model_param_device"] = param_device
    last["reward_type"] = reward_kind
    last["n_nodes"] = int(n_nodes)
    last["save_path"] = str(save_path)
    return last


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark training scaling vs molecule size.")
    parser.add_argument("--config-yaml", type=str, default="verl_diffusion/trainer/config/ddpo_config.yaml")
    parser.add_argument("--reward-types", type=str, default="mlff,xtb,dft")
    parser.add_argument("--n-nodes", type=str, default="10,15,20,25,29")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--sample-group-size", type=int, default=1)
    parser.add_argument("--each-prompt-sample", type=int, default=4)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--train-micro-batch-size", type=int, default=4)
    parser.add_argument("--time-step", type=int, default=200)
    parser.add_argument("--device", type=str, default=None)
    # DFT knobs: defaults are set for speed/robustness on large molecules.
    parser.add_argument("--dft-xc", type=str, default="lda")
    parser.add_argument("--dft-basis", type=str, default="minao")
    parser.add_argument("--dft-max-cycle", type=int, default=10)
    parser.add_argument("--dft-grids-level", type=int, default=1)
    parser.add_argument("--dft-density-fit", type=int, default=1, help="1 to enable, 0 to disable.")
    parser.add_argument("--dft-accept-unconverged", type=int, default=1, help="1 to accept, 0 to skip.")
    parser.add_argument("--dft-min-interatomic-distance", type=float, default=0.3)
    parser.add_argument("--out-csv", type=str, default="benchmarks/results/training_scaling.csv")
    args = parser.parse_args()

    device = _select_device(args.device)
    reward_types = [t.strip().lower() for t in str(args.reward_types).split(",") if t.strip()]
    n_nodes_list = _parse_int_list(args.n_nodes)

    base_cfg = _load_yaml(Path(args.config_yaml))
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    dft_overrides = {
        "xc": str(args.dft_xc),
        "basis": str(args.dft_basis),
        "max_cycle": int(args.dft_max_cycle),
        "grids_level": int(args.dft_grids_level),
        "density_fit": bool(int(args.dft_density_fit)),
        "accept_unconverged": bool(int(args.dft_accept_unconverged)),
        "min_interatomic_distance": float(args.dft_min_interatomic_distance),
    }
    for reward_type in reward_types:
        for n_nodes in n_nodes_list:
            print(f"[bench] reward={reward_type} n_nodes={n_nodes}", flush=True)
            row = run_one(
                base_cfg,
                reward_type,
                n_nodes,
                device,
                epochs=int(args.epochs),
                sample_group_size=int(args.sample_group_size),
                each_prompt_sample=int(args.each_prompt_sample),
                micro_batch_size=int(args.micro_batch_size),
                train_micro_batch_size=int(args.train_micro_batch_size),
                time_step=int(args.time_step),
                dft_overrides=dft_overrides,
            )
            rows.append(row)
            if device.type == "cuda":
                torch.cuda.empty_cache()

    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    main()
