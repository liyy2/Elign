import argparse
import pickle
import os
import torch
import ray
os.environ["WANDB_MODE"] = "online"
os.environ["WANDB_API_KEY"] = "6f1080f993d5d7ad6103e69ef57dd9291f1bf366"
from verl_diffusion.trainer.config.base import BaseConfig
from verl_diffusion.trainer.ddpo_trainer import DDPOTrainer
import sys
sys.path.append("/nfs/roberts/project/pi_mg269/yl2428/e3_diffusion_for_molecules-main/edm_source")
from edm_source.configs.datasets_config import get_dataset_info
from edm_source.qm9.dataset import retrieve_dataloaders
from edm_source.qm9.models import get_model
from verl_diffusion.model.edm_model import EDMModel
from verl_diffusion.dataloader.dataloader import EDMDataLoader
from verl_diffusion.worker.rollout.edm_rollout import EDMRollout
from verl_diffusion.worker.reward.force import UMAForceReward
from verl_diffusion.worker.filter.filter import Filter
from verl_diffusion.worker.actor.edm_actor import EDMActor
from edm_source.qm9.rdkit_functions import retrieve_qm9_smiles


def parse_args():
    parser = argparse.ArgumentParser(description="DDPO Training for EDM")
    parser.add_argument('--config_path', type=str, default="./verl_diffusion/trainer/config/ddpo_config.yaml", help="Path to the configuration file")
    parser.add_argument('--save_path', type=str, default=None, help="Override path to save checkpoints")
    parser.add_argument('--model_path', type=str, default=None, help="Override pretrained EDM checkpoint path")
    parser.add_argument('--wandb', action='store_true', help="Enable wandb logging")
    parser.add_argument('--wandb_project', type=str, default=None, help="Wandb project name override")
    parser.add_argument('--wandb_name', type=str, default=None, help="Wandb run name override")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--resume', action='store_true', help="Resume training from checkpoint")
    parser.add_argument('--checkpoint_path', type=str, default=None, help="Path to checkpoint to resume from")
    
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    import random
    random.seed(args.seed)
    import numpy as np
    np.random.seed(args.seed)
    
    # Load configuration
    config = BaseConfig()
    config = config.from_yaml(args.config_path)
    config = config.to_dict()

    # CLI overrides take precedence over config defaults
    if args.save_path:
        config["save_path"] = args.save_path
    if args.model_path:
        config.setdefault("model", {})["model_path"] = args.model_path
    if args.wandb_project:
        config.setdefault("wandb", {})["wandb_project"] = args.wandb_project
    if args.wandb_name:
        config.setdefault("wandb", {})["wandb_name"] = args.wandb_name
    if args.resume:
        config["resume"] = True
    if args.checkpoint_path:
        config["checkpoint_path"] = args.checkpoint_path
    print("Config loaded")
    
    # Load EDM config
    edm_config_path = config["model"]["config"]
    with open(edm_config_path, 'rb') as f:
        edm_config = pickle.load(f)
    # Use the same processed data directory convention as eval_mlff_guided.py
    edm_config.datadir = "qm9/temp"
    print(edm_config)
    # Set default values if not present
    if not hasattr(edm_config, 'normalization_factor'):
        edm_config.normalization_factor = 1
    if not hasattr(edm_config, 'aggregation_method'):
        edm_config.aggregation_method = 'sum'

    # Set up device
    edm_config.cuda = not edm_config.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if edm_config.cuda else "cpu")
    edm_config.device = device
    
    # Load dataset info and model
    dataset_info = get_dataset_info(edm_config.dataset, edm_config.remove_h)
    retrieve_qm9_smiles(dataset_info)
    dataloaders, charge_scale = retrieve_dataloaders(edm_config)
    flow, nodes_dist, prop_dist = get_model(edm_config, edm_config.device, dataset_info, dataloaders['train'])
    flow.to(device)
    
    # Initialize EDM model
    model = EDMModel(flow, edm_config)
    model.load(model_path=config["model"]["model_path"])
    
    # Initialize dataloader
    dataloader = EDMDataLoader(config=config,
                               dataset_info=dataset_info,
                               nodes_dist=nodes_dist,
                               prop_dist=prop_dist,
                               device=device,
                               condition=False,
                               num_batches=config["dataloader"]["epoches"])
    
    # Initialize rollout and rewarder in main function
    rollout = EDMRollout(model, config)
    rollout.model.to(device)
    rewarder = UMAForceReward(dataset_info)
    filters = Filter(dataset_info,config["dataloader"]["smiles_path"],False,False,False)
    actor = EDMActor(model, config)
    # Initialize Ray for parallel processing
    if not ray.is_initialized():
        ray.init()
    print("Ray initialized")
    # Initialize DDPO Trainer with rollout and rewarder
    trainer = DDPOTrainer(
        config=config,
        model=model,
        dataset_info=dataset_info,
        device=device,
        dataloader=dataloader,
        rollout=rollout,
        rewarder=rewarder,
        filters=filters,
        actor = actor
    )
    print("DDPO Trainer initialized")
    try:
        print("Training started")
        trainer.fit()
    except KeyboardInterrupt:
        print("Training interrupted by user")
    finally:
        # Clean up resources
        trainer.clean_up()
        print("Training completed")

if __name__ == "__main__":
    main() 
