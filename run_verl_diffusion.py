import os
import pickle
import random
import sys
from typing import Dict, Optional

import numpy as np
import ray
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from hydra import main as hydra_main
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from verl_diffusion.trainer.ddpo_trainer import DDPOTrainer

sys.path.append("/home/yl2428/project_pi_mg269/yl2428/e3_diffusion_for_molecules-main/edm_source")
from edm_source.configs.datasets_config import get_dataset_info
from edm_source.qm9.dataset import retrieve_dataloaders
from edm_source.qm9.models import get_model
from edm_source.qm9.rdkit_functions import retrieve_qm9_smiles
from verl_diffusion.dataloader.dataloader import EDMDataLoader
from verl_diffusion.model.edm_model import EDMModel
from verl_diffusion.worker.actor.edm_actor import EDMActor
from verl_diffusion.worker.filter.filter import Filter
from verl_diffusion.worker.reward.force import UMAForceReward
from verl_diffusion.worker.rollout.edm_rollout import EDMRollout

os.environ.setdefault("WANDB_MODE", "online")


def _setup_distributed() -> Dict[str, int]:
    """Initialize torch.distributed if launched under torchrun."""

    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return {"rank": rank, "world_size": world_size, "local_rank": local_rank}

    world_size_env = os.environ.get("WORLD_SIZE")
    if world_size_env is None or int(world_size_env) <= 1:
        return {"rank": 0, "world_size": 1, "local_rank": 0}

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return {"rank": rank, "world_size": world_size, "local_rank": local_rank}


def _seed_everything(base_seed: int, rank: int) -> None:
    """Seed RNGs in a rank-aware fashion."""

    seed = int(base_seed) + int(rank)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def _make_absolute(path_value: Optional[str]) -> Optional[str]:
    """Convert relative config paths to absolute paths for Hydra runs."""
    if not path_value:
        return path_value
    return to_absolute_path(path_value)


@hydra_main(config_path="verl_diffusion/trainer/config", config_name="ddpo_config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Entry point for DDPO training managed by Hydra."""

    dist_state = _setup_distributed()
    is_main_process = dist_state["rank"] == 0

    # Set random seed for reproducibility (rank-aware)
    _seed_everything(cfg.seed, dist_state["rank"])

    # Convert relevant paths to absolute to avoid Hydra working directory changes
    cfg.model.config = _make_absolute(cfg.model.get("config"))
    cfg.model.model_path = _make_absolute(cfg.model.get("model_path"))
    cfg.dataloader.smiles_path = _make_absolute(cfg.dataloader.get("smiles_path"))
    cfg.save_path = _make_absolute(cfg.get("save_path"))
    cfg.checkpoint_path = _make_absolute(cfg.get("checkpoint_path"))

    # Convert OmegaConf to standard Python dict for downstream components
    config = OmegaConf.to_container(cfg, resolve=True)
    if isinstance(config, dict):
        config["distributed"] = {
            "rank": dist_state["rank"],
            "world_size": dist_state["world_size"],
            "local_rank": dist_state["local_rank"],
            "is_main_process": is_main_process,
        }
        train_cfg = config.get("train")
        if isinstance(train_cfg, dict):
            grad_accum = train_cfg.get("gradient_accumulation_steps", 1)
            try:
                grad_accum = int(grad_accum)
            except (TypeError, ValueError):
                grad_accum = 1
            if grad_accum <= 0:
                grad_accum = 1
            train_cfg["gradient_accumulation_steps"] = grad_accum

    # Load EDM config
    edm_config_path = config["model"]["config"]
    with open(edm_config_path, "rb") as f:
        edm_config = pickle.load(f)

    # Use the same processed data directory convention as eval_mlff_guided.py
    edm_config.datadir = "qm9/temp"
    if is_main_process:
        print(edm_config)

    # Set default values if not present
    if not hasattr(edm_config, "normalization_factor"):
        edm_config.normalization_factor = 1
    if not hasattr(edm_config, "aggregation_method"):
        edm_config.aggregation_method = "sum"

    # Set up device
    edm_config.cuda = not edm_config.no_cuda and torch.cuda.is_available()
    if edm_config.cuda:
        device = torch.device(f"cuda:{dist_state['local_rank']}")
    else:
        device = torch.device("cpu")
    edm_config.device = device

    # Load dataset info and model
    dataset_info = get_dataset_info(edm_config.dataset, edm_config.remove_h)
    retrieve_qm9_smiles(dataset_info)
    dataloaders, _ = retrieve_dataloaders(edm_config)
    flow, nodes_dist, prop_dist = get_model(
        edm_config, edm_config.device, dataset_info, dataloaders["train"]
    )
    flow.to(device)

    # Initialize EDM model
    model = EDMModel(flow, edm_config)
    model.to(device)
    model.load(model_path=config["model"]["model_path"])

    if dist_state["world_size"] > 1:
        model = DDP(
            model,
            device_ids=[dist_state["local_rank"]] if device.type == "cuda" else None,
            output_device=dist_state["local_rank"] if device.type == "cuda" else None,
            find_unused_parameters=True,
        )

    # Initialize dataloader
    dataloader = EDMDataLoader(
        config=config,
        dataset_info=dataset_info,
        nodes_dist=nodes_dist,
        prop_dist=prop_dist,
        device=device,
        condition=False,
        num_batches=config["dataloader"]["epoches"],
        rank=dist_state["rank"],
        world_size=dist_state["world_size"],
        base_seed=cfg.seed,
    )

    # Initialize rollout and rewarder in main function
    rollout = EDMRollout(model, config)

    reward_cfg = config.get("reward", {})
    reward_device = device
    mlff_device = device if device.type == "cuda" else "cpu"
    rewarder = UMAForceReward(
        dataset_info,
        condition=False,
        mlff_model=reward_cfg.get("mlff_model", "uma-s-1p1"),
        mlff_predictor=None,
        position_scale=None,
        force_clip_threshold=reward_cfg.get("force_clip_threshold", None),
        device=reward_device,
        mlff_device=mlff_device,
        shaping=reward_cfg.get("shaping", {}),
        use_energy=reward_cfg.get("use_energy", False),
        force_weight=reward_cfg.get("force_weight", 1.0),
        energy_weight=reward_cfg.get("energy_weight", 1.0),
        stability_weight=reward_cfg.get("stability_weight", 0.0),
        force_aggregation=reward_cfg.get("force_aggregation", "rms"),
        energy_transform_offset=reward_cfg.get("energy_transform_offset", 10000.0),
        energy_transform_scale=reward_cfg.get("energy_transform_scale", 1000.0),
    )

    filter_cfg = config.get("filters", {})
    filter_condition = filter_cfg.get("condition", False)
    filter_enable_filtering = filter_cfg.get("enable_filtering", False)
    filter_enable_penalty = filter_cfg.get("enable_penalty", False)
    filter_penalty_scale = filter_cfg.get("penalty_scale", 0.5)
    filters = Filter(
        dataset_info,
        config["dataloader"]["smiles_path"],
        filter_condition,
        filter_enable_filtering,
        filter_enable_penalty,
        filter_penalty_scale,
    )
    actor = EDMActor(model, config)

    if is_main_process and not ray.is_initialized():
        ray.init()
        print("Ray initialized")

    trainer = DDPOTrainer(
        config=config,
        model=model,
        dataset_info=dataset_info,
        device=device,
        dataloader=dataloader,
        rollout=rollout,
        rewarder=rewarder,
        filters=filters,
        actor=actor,
    )

    if is_main_process:
        print("DDPO Trainer initialized")

    try:
        if is_main_process:
            print("Training started")
        trainer.fit()
    except KeyboardInterrupt:
        if is_main_process:
            print("Training interrupted by user")
    finally:
        trainer.clean_up()
        if is_main_process:
            print("Training completed")
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
