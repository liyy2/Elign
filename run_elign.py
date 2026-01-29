import os
import pickle
import random
import sys
from typing import Dict, Optional

import numpy as np
import ray
import torch
import torch.distributed as dist
from torch.distributions.categorical import Categorical
from torch.nn.parallel import DistributedDataParallel as DDP
from hydra import main as hydra_main
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from verl_diffusion.trainer.ddpo_trainer import DDPOTrainer

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
EDM_SOURCE_ROOT = os.path.join(REPO_ROOT, "edm_source")
for path in (REPO_ROOT, EDM_SOURCE_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from edm_source.configs.datasets_config import get_dataset_info
from edm_source.qm9.dataset import retrieve_dataloaders
from edm_source.qm9.models import get_model
from edm_source.qm9.rdkit_functions import retrieve_qm9_smiles
from verl_diffusion.dataloader.dataloader import EDMDataLoader
from verl_diffusion.model.edm_model import EDMModel
from verl_diffusion.worker.actor.edm_actor import EDMActor
from verl_diffusion.worker.filter.filter import Filter
from verl_diffusion.worker.reward.dummy import DummyReward
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
    if "geom_data_file" in cfg.dataloader:
        cfg.dataloader.geom_data_file = _make_absolute(cfg.dataloader.get("geom_data_file"))
    cfg.save_path = _make_absolute(cfg.get("save_path"))
    if "checkpoint_path" in cfg:
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

    # Keep QM9 data under the processed cache directory; GEOM uses an explicit `.npy` file path.
    dataset_name = getattr(edm_config, "dataset", "")
    if isinstance(dataset_name, str) and "qm9" in dataset_name:
        edm_config.datadir = "qm9/temp"

    dataloader_cfg = config.get("dataloader") or {}
    if isinstance(dataloader_cfg, dict):
        geom_data_file = dataloader_cfg.get("geom_data_file")
        if geom_data_file:
            setattr(edm_config, "geom_data_file", geom_data_file)
            setattr(edm_config, "geom_data_path", geom_data_file)
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

    filter_cfg = config.get("filters") or {}
    enable_penalty = False
    if isinstance(filter_cfg, dict):
        enable_penalty = bool(filter_cfg.get("enable_penalty", False))
    smiles_path = (config.get("dataloader") or {}).get("smiles_path")
    if enable_penalty and isinstance(smiles_path, str) and smiles_path:
        if not os.path.exists(smiles_path):
            if "qm9" in str(dataset_info.get("name", "")):
                retrieve_qm9_smiles(dataset_info)
            else:
                raise FileNotFoundError(
                    f"SMILES pickle not found at {smiles_path}. "
                    "Provide `dataloader.smiles_path` or disable `filters.enable_penalty`."
                )
    dataloaders, _ = retrieve_dataloaders(edm_config)
    flow, nodes_dist, prop_dist = get_model(
        edm_config, edm_config.device, dataset_info, dataloaders["train"]
    )

    # Optional: reweight the node-count distribution used for RL rollouts.
    #
    # This does *not* change the pretrained EDM prior itself; it only changes how often the PPO
    # loop samples certain molecule sizes. This is useful when some sizes are systematically
    # harder for `check_stability` (e.g., small molecules) and need more training coverage.
    if isinstance(dataloader_cfg, dict) and nodes_dist is not None:
        focus_min = dataloader_cfg.get("nodes_dist_focus_min")
        focus_max = dataloader_cfg.get("nodes_dist_focus_max")
        focus_multiplier = dataloader_cfg.get("nodes_dist_focus_multiplier", 1.0)
        try:
            focus_multiplier = float(focus_multiplier) if focus_multiplier is not None else 1.0
        except (TypeError, ValueError):
            focus_multiplier = 1.0

        if focus_min is not None and focus_max is not None and focus_multiplier != 1.0:
            try:
                focus_min = int(focus_min)
                focus_max = int(focus_max)
            except (TypeError, ValueError):
                focus_min = None
                focus_max = None

        if focus_min is not None and focus_max is not None and focus_multiplier != 1.0:
            if hasattr(nodes_dist, "prob") and hasattr(nodes_dist, "n_nodes") and hasattr(nodes_dist, "m"):
                prob = nodes_dist.prob.detach().clone().to(dtype=torch.float64)
                n_nodes = nodes_dist.n_nodes.detach().to(dtype=torch.long)
                mask = (n_nodes >= focus_min) & (n_nodes <= focus_max)
                if mask.any():
                    prob[mask] = prob[mask] * focus_multiplier
                    prob = prob / prob.sum().clamp(min=1e-12)
                    nodes_dist.prob = prob.to(dtype=torch.float32)
                    nodes_dist.m = Categorical(nodes_dist.prob)
                    if is_main_process:
                        print(
                            f"Reweighted n_nodes prior for RL rollouts: "
                            f"[{focus_min}, {focus_max}] x {focus_multiplier}"
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

    reward_cfg = config.get("reward", {}) or {}
    reward_type = "uma"
    if isinstance(reward_cfg, dict):
        reward_type = str(reward_cfg.get("type", reward_type)).lower()

    reward_device = device
    mlff_device = device if device.type == "cuda" else "cpu"

    if reward_type == "dummy":
        rewarder = DummyReward(
            reward_value=float(reward_cfg.get("reward_value", 0.0)),
            stability_value=float(reward_cfg.get("stability_value", 0.0)),
            device=reward_device,
        )
    elif reward_type in {"uma", "mlff"}:
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
            energy_only_if_stable=reward_cfg.get("energy_only_if_stable", False),
            force_only_if_stable=reward_cfg.get("force_only_if_stable", False),
            force_weight=reward_cfg.get("force_weight", 1.0),
            energy_weight=reward_cfg.get("energy_weight", 1.0),
            stability_weight=reward_cfg.get("stability_weight", 0.0),
            atom_stability_weight=reward_cfg.get("atom_stability_weight", 0.0),
            valence_underbond_weight=reward_cfg.get("valence_underbond_weight", 0.0),
            valence_overbond_weight=reward_cfg.get("valence_overbond_weight", 0.0),
            valence_underbond_soft_weight=reward_cfg.get("valence_underbond_soft_weight", 0.0),
            valence_overbond_soft_weight=reward_cfg.get("valence_overbond_soft_weight", 0.0),
            valence_soft_temperature=reward_cfg.get("valence_soft_temperature", 0.02),
            force_aggregation=reward_cfg.get("force_aggregation", "rms"),
            energy_transform_offset=reward_cfg.get("energy_transform_offset", 10000.0),
            energy_transform_scale=reward_cfg.get("energy_transform_scale", 1000.0),
            energy_transform_clip=reward_cfg.get("energy_transform_clip", None),
            energy_normalize_by_atoms=reward_cfg.get("energy_normalize_by_atoms", False),
            energy_atom_refs=reward_cfg.get("energy_atom_refs", None),
        )
    else:
        raise ValueError(f"Unsupported reward.type '{reward_type}'. Use 'uma' or 'dummy'.")

    filter_cfg = config.get("filters", {})
    filter_condition = filter_cfg.get("condition", False)
    filter_enable_filtering = filter_cfg.get("enable_filtering", False)
    filter_enable_penalty = filter_cfg.get("enable_penalty", False)
    filter_penalty_scale = filter_cfg.get("penalty_scale", 0.5)
    filter_invalid_penalty_scale = filter_cfg.get("invalid_penalty_scale", 0.0)
    filter_duplicate_penalty_scale = filter_cfg.get("duplicate_penalty_scale", 0.0)
    filters = Filter(
        dataset_info,
        config["dataloader"]["smiles_path"],
        filter_condition,
        filter_enable_filtering,
        filter_enable_penalty,
        filter_penalty_scale,
        filter_invalid_penalty_scale,
        filter_duplicate_penalty_scale,
    )
    actor = EDMActor(model, config)

    ray_cfg = config.get("ray", {})
    ray_enabled = False
    if isinstance(ray_cfg, dict):
        ray_enabled = bool(ray_cfg.get("enabled", False))
    elif ray_cfg:
        ray_enabled = True

    if ray_enabled and is_main_process and not ray.is_initialized():
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
