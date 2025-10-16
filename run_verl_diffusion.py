import os
import pickle
import random
import sys
from typing import Optional

import numpy as np
import ray
import torch
from hydra import main as hydra_main
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from verl_diffusion.trainer.ddpo_trainer import DDPOTrainer

sys.path.append("/home/yl2428/scratch_pi_mg269/yl2428/e3_diffusion_for_molecules-main/edm_source")

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


def _make_absolute(path_value: Optional[str]) -> Optional[str]:
    """Convert relative config paths to absolute paths for Hydra runs."""
    if not path_value:
        return path_value
    return to_absolute_path(path_value)


@hydra_main(config_path="verl_diffusion/trainer/config", config_name="ddpo_config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Entry point for DDPO training managed by Hydra."""

    # Set random seed for reproducibility
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Convert relevant paths to absolute to avoid Hydra working directory changes
    cfg.model.config = _make_absolute(cfg.model.get("config"))
    cfg.model.model_path = _make_absolute(cfg.model.get("model_path"))
    cfg.dataloader.smiles_path = _make_absolute(cfg.dataloader.get("smiles_path"))
    cfg.save_path = _make_absolute(cfg.get("save_path"))
    cfg.checkpoint_path = _make_absolute(cfg.get("checkpoint_path"))

    # Convert OmegaConf to standard Python dict for downstream components
    config = OmegaConf.to_container(cfg, resolve=True)

    # Load EDM config
    edm_config_path = config["model"]["config"]
    with open(edm_config_path, "rb") as f:
        edm_config = pickle.load(f)

    # Use the same processed data directory convention as eval_mlff_guided.py
    edm_config.datadir = "qm9/temp"
    print(edm_config)

    # Set default values if not present
    if not hasattr(edm_config, "normalization_factor"):
        edm_config.normalization_factor = 1
    if not hasattr(edm_config, "aggregation_method"):
        edm_config.aggregation_method = "sum"

    # Set up device
    edm_config.cuda = not edm_config.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if edm_config.cuda else "cpu")
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
    model.load(model_path=config["model"]["model_path"])

    # Initialize dataloader
    dataloader = EDMDataLoader(
        config=config,
        dataset_info=dataset_info,
        nodes_dist=nodes_dist,
        prop_dist=prop_dist,
        device=device,
        condition=False,
        num_batches=config["dataloader"]["epoches"],
    )

    # Initialize rollout and rewarder in main function
    rollout = EDMRollout(model, config)
    rollout.model.to(device)

    reward_cfg = config.get("reward", {})
    rewarder = UMAForceReward(
        dataset_info,
        condition=False,
        mlff_model=reward_cfg.get("mlff_model", "uma-s-1p1"),
        mlff_predictor=None,
        position_scale=None,
        force_clip_threshold=reward_cfg.get("force_clip_threshold", None),
        device=str(device),
        shaping=reward_cfg.get("shaping", {}),
        use_energy=reward_cfg.get("use_energy", False),
        force_weight=reward_cfg.get("force_weight", 1.0),
        energy_weight=reward_cfg.get("energy_weight", 1.0),
        force_aggregation=reward_cfg.get("force_aggregation", "rms"),
        energy_transform_offset=reward_cfg.get("energy_transform_offset", 10000.0),
        energy_transform_scale=reward_cfg.get("energy_transform_scale", 1000.0),
    )

    filters = Filter(dataset_info, config["dataloader"]["smiles_path"], False, False, False)
    actor = EDMActor(model, config)

    if not ray.is_initialized():
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

    print("DDPO Trainer initialized")

    try:
        print("Training started")
        trainer.fit()
    except KeyboardInterrupt:
        print("Training interrupted by user")
    finally:
        trainer.clean_up()
        print("Training completed")


if __name__ == "__main__":
    main()
