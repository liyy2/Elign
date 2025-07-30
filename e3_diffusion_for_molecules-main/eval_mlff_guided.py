# Rdkit import should be first, do not move it
try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import sys

"""
MLFF-Guided Diffusion Evaluation with Weights & Biases Logging and Multi-GPU Support

This script evaluates molecular generation using diffusion models with optional 
MLFF (Machine Learning Force Field) guidance. It includes comprehensive W&B logging
to track stability metrics, sampling progress, and model comparisons.
Now supports multi-GPU evaluation for faster sampling.

Usage examples:
    # Single GPU evaluation
    python eval_mlff_guided.py --model_path outputs/edm_1

    # Multi-GPU evaluation
    python -m torch.distributed.launch --nproc_per_node=4 eval_mlff_guided.py \
        --model_path outputs/edm_1 --use_distributed

    # With wandb logging (multi-GPU)
    python -m torch.distributed.launch --nproc_per_node=4 eval_mlff_guided.py \
        --model_path outputs/edm_1 --use_distributed --use_wandb --wandb_project my-project

Logged metrics include:
    - Molecular stability percentages for baseline and guided sampling
    - Stability ratios and comparisons across guidance scales  
    - Sampling configuration and progress
    - Model and dataset information
    - Chain visualization status
"""

import utils
import argparse
from configs.datasets_config import get_dataset_info

# Set matplotlib backend before any plotting imports to prevent hanging on headless servers
import matplotlib
matplotlib.use('Agg')
from qm9 import dataset
from qm9.models import get_model
from fairchem.core import pretrained_mlip, FAIRChemCalculator

from equivariant_diffusion.utils import assert_correctly_masked
import pickle
import qm9.visualizer as vis
from qm9.analyze import check_stability
from os.path import join
from qm9.sampling import sample_chain, sample
from mlff_guided_diffusion import (
    MLFFGuidedDiffusion, 
    create_mlff_guided_model, 
    enhanced_sampling_with_mlff
)
import numpy as np
from tqdm import tqdm

# Wandb import with error handling
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with 'pip install wandb' to enable logging.")


def setup_distributed(rank, world_size, backend='nccl'):
    """Initialize distributed training."""
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '12355'
    
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
    # Suppress stdout on non-main processes for cleaner logging
    if rank != 0:
        sys.stdout = open(os.devnull, 'w')
        # Keep stderr for error reporting


def cleanup_distributed():
    """Clean up distributed training."""
    # Restore stdout on non-main processes
    if dist.is_initialized() and dist.get_rank() != 0:
        sys.stdout.close()
        sys.stdout = sys.__stdout__
    
    if dist.is_initialized():
        dist.destroy_process_group()


def get_rank_and_world_size():
    """Get current rank and world size."""
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    else:
        return 0, 1


def all_gather_results(results, device):
    """Gather results from all processes. Assumes all tensors have same size across processes."""
    rank, world_size = get_rank_and_world_size()
    
    if world_size == 1:
        return results
    
    # Gather baseline results - all tensors now have same size due to even sample splitting
    baseline_gathered = None
    if results['baseline'] is not None:
        baseline_gathered = {}
        for key in ['one_hot', 'charges', 'positions', 'node_mask']:
            local_tensor = results['baseline'][key]
            tensor_list = [torch.empty_like(local_tensor) for _ in range(world_size)]
            dist.all_gather(tensor_list, local_tensor)
            baseline_gathered[key] = torch.cat(tensor_list, dim=0)
    
    # Gather guided results
    guided_gathered = {}
    for guidance_scale, guided_result in results['guided'].items():
        if guided_result is not None:
            guided_gathered[guidance_scale] = {
                'features': {},
                'positions': guided_result['positions'],
                'node_mask': guided_result['node_mask']
            }
            
            # Gather guided results
            for key in ['categorical', 'integer']:
                if key in guided_result['features']:
                    local_tensor = guided_result['features'][key]
                    tensor_list = [torch.empty_like(local_tensor) for _ in range(world_size)]
                    dist.all_gather(tensor_list, local_tensor)
                    guided_gathered[guidance_scale]['features'][key] = torch.cat(tensor_list, dim=0)
            
            # Gather positions and node_mask
            for key in ['positions', 'node_mask']:
                local_tensor = guided_result[key]
                tensor_list = [torch.empty_like(local_tensor) for _ in range(world_size)]
                dist.all_gather(tensor_list, local_tensor)
                guided_gathered[guidance_scale][key] = torch.cat(tensor_list, dim=0)
        else:
            guided_gathered[guidance_scale] = None
    
    return {
        'baseline': baseline_gathered,
        'guided': guided_gathered
    }


def is_main_process():
    """Check if current process is the main process."""
    rank, _ = get_rank_and_world_size()
    return rank == 0


def split_samples_across_gpus(n_samples):
    """Split samples evenly across available GPUs for all_gather compatibility."""
    rank, world_size = get_rank_and_world_size()
    
    # Ensure even division by adjusting total samples if needed
    samples_per_gpu = n_samples // world_size
    if n_samples % world_size != 0:
        # Round up to ensure all requested samples are covered
        samples_per_gpu = (n_samples + world_size - 1) // world_size
        adjusted_total = samples_per_gpu * world_size
        if rank == 0:  # Only print from main process
            print(f"Adjusting samples from {n_samples} to {adjusted_total} for even GPU distribution")
    
    if is_main_process():
        print(f"Distributing {n_samples} samples across {world_size} GPUs ({samples_per_gpu} samples per GPU)")
    return samples_per_gpu


def check_mask_correct(variables, node_mask):
    for variable in variables:
        assert_correctly_masked(variable, node_mask)


def load_mlff_predictor(model_name, device):
    """Load MLFF predictor with error handling."""
    rank, world_size = get_rank_and_world_size()
    
    if is_main_process():
        print(f"Loading MLFF predictor: {model_name}")
    
    try:
        # Convert device to string format expected by predictor
        device_str = 'cuda' if device.type == 'cuda' else 'cpu'
        predictor = pretrained_mlip.get_predict_unit(model_name, device=device_str)
        
        if is_main_process():
            print(f"✓ Successfully loaded MLFF predictor: {model_name}")
        return predictor
    except Exception as e:
        if is_main_process():
            print(f"✗ Error loading MLFF predictor: {e}")
            print("Make sure you have access to the UMA model and proper authentication.")
            print("You can still run baseline sampling without MLFF guidance.")
        return None


def sample_with_mlff_comparison(args, eval_args, device, flow, nodes_dist, 
                               dataset_info, mlff_predictor, prop_dist, n_samples=10):
    """Sample molecules with and without MLFF guidance for comparison."""
    
    # Split samples across GPUs
    local_n_samples = split_samples_across_gpus(n_samples)
    
    if is_main_process():
        print(f"\n{'='*50}")
        print(f"MOLECULAR SAMPLING COMPARISON")
        print(f"{'='*50}")
        print(f"Total samples: {n_samples}")
        print(f"GPUs: {get_rank_and_world_size()[1]}")
        print(f"Samples per GPU: {local_n_samples}")
        print(f"MLFF predictor: {'Available' if mlff_predictor is not None else 'Not available'}")
        print(f"Skip baseline: {eval_args.skip_baseline}")
        if mlff_predictor is not None:
            print(f"Guidance scales: {eval_args.guidance_scales}")
            print(f"Guidance iterations: {eval_args.guidance_iterations}")
            print(f"Noise threshold: {eval_args.noise_threshold} (skip guidance when noise > threshold)")
    
    # Log sampling configuration (only from main process)
    if WANDB_AVAILABLE and wandb.run is not None and is_main_process():
        wandb.log({
            "sampling/n_samples": n_samples,
            "sampling/local_n_samples": local_n_samples,
            "sampling/world_size": get_rank_and_world_size()[1],
            "sampling/mlff_available": mlff_predictor is not None,
            "sampling/guidance_iterations": eval_args.guidance_iterations,
            "sampling/skip_baseline": eval_args.skip_baseline,
        })
    
    # Sample baseline (without guidance) - skip if requested
    baseline_results = None
    if not eval_args.skip_baseline:
        if is_main_process():
            print(f"\n1. BASELINE SAMPLING (No MLFF guidance)")
            print(f"   Sampling {n_samples} molecules...")
        
        # Set seed for reproducible baseline sampling
        rank, _ = get_rank_and_world_size()
        torch.manual_seed(eval_args.seed + rank)
        
        # Use local sample count for each GPU
        nodesxsample = nodes_dist.sample(local_n_samples)
        one_hot_baseline, charges_baseline, x_baseline, node_mask = sample(
            args, device, flow, dataset_info, nodesxsample=nodesxsample)
        
        rank, world_size = get_rank_and_world_size()
        # Synchronize before reporting completion
        if dist.is_initialized():
            dist.barrier()
        
        if is_main_process():
            print(f"   ✓ All ranks completed baseline sampling ({local_n_samples} molecules per rank)")
        
        baseline_results = {
            'one_hot': one_hot_baseline,
            'charges': charges_baseline,
            'positions': x_baseline,
            'node_mask': node_mask
        }
    else:
        if is_main_process():
            print(f"\n1. BASELINE SAMPLING SKIPPED")
        # Still need nodesxsample for guided sampling
        rank, _ = get_rank_and_world_size()
        torch.manual_seed(eval_args.seed + rank)
        nodesxsample = nodes_dist.sample(local_n_samples)
    
    # Sample with MLFF guidance (if predictor available)
    if mlff_predictor is not None:
        # Sample with different guidance scales
        guidance_scales = eval_args.guidance_scales
        
        if is_main_process():
            print(f"\n2. MLFF-GUIDED SAMPLING")
            print(f"   Testing {len(guidance_scales)} guidance scales: {guidance_scales}")
        
        # Setup for batched sampling
        max_n_nodes = dataset_info['max_n_nodes']
        total_samples = len(nodesxsample)
        internal_batch_size = eval_args.batch_size
        
        guided_results = {}
        
        for guidance_scale in guidance_scales:
            if is_main_process():
                print(f"\n   → Guidance scale {guidance_scale}:")
            
            # Set same seed as baseline for reproducible comparison
            torch.manual_seed(eval_args.seed + rank)
            
            # Initialize storage for this guidance scale
            all_x_guided = []
            all_h_guided = []
            all_node_masks = []
            
            # Process in batches with progress bar
            n_batches = (total_samples + internal_batch_size - 1) // internal_batch_size
            batch_iterator = range(0, total_samples, internal_batch_size)
            
            # Show progress bar only on main process
            if is_main_process():
                batch_iterator = tqdm(batch_iterator, desc=f"Guidance scale {guidance_scale}", 
                                    total=n_batches, leave=False)
            
            for batch_start in batch_iterator:
                batch_end = min(batch_start + internal_batch_size, total_samples)
                batch_nodesxsample = nodesxsample[batch_start:batch_end]
                batch_size = len(batch_nodesxsample)
                
                # Create node mask for this batch
                node_mask_batch = torch.zeros(batch_size, max_n_nodes, device=device)
                for i, n_nodes in enumerate(batch_nodesxsample):
                    node_mask_batch[i, 0:n_nodes] = 1
                node_mask_batch = node_mask_batch.unsqueeze(2)
                
                # Create edge mask for this batch
                edge_mask_batch = node_mask_batch.squeeze(2).unsqueeze(1) * node_mask_batch.squeeze(2).unsqueeze(2)
                diag_mask = ~torch.eye(edge_mask_batch.size(1), dtype=torch.bool, device=device).unsqueeze(0)
                edge_mask_batch *= diag_mask
                edge_mask_batch = edge_mask_batch.view(batch_size * max_n_nodes * max_n_nodes, 1)
                
                # Create context for this batch
                if args.context_node_nf > 0:
                    context_batch = torch.zeros(batch_size, args.context_node_nf, device=device)
                    context_batch = context_batch.unsqueeze(1).repeat(1, max_n_nodes, 1) * node_mask_batch
                else:
                    context_batch = None
                
                # Sample this batch
                x_batch, h_batch = enhanced_sampling_with_mlff(
                    flow, mlff_predictor, batch_size, max_n_nodes, 
                    node_mask_batch, edge_mask_batch, context_batch, dataset_info,
                    guidance_scale=guidance_scale, guidance_iterations=eval_args.guidance_iterations, 
                    noise_threshold=eval_args.noise_threshold, fix_noise=False
                )
                
                # Store batch results
                all_x_guided.append(x_batch)
                all_h_guided.append(h_batch)
                all_node_masks.append(node_mask_batch)
            
            # Concatenate all batches for this guidance scale
            x_guided = torch.cat(all_x_guided, dim=0)
            h_guided = {key: torch.cat([batch[key] for batch in all_h_guided], dim=0) 
                       for key in all_h_guided[0].keys()}
            node_mask_guided = torch.cat(all_node_masks, dim=0)
            
            guided_results[guidance_scale] = {
                'positions': x_guided,
                'features': h_guided,
                'node_mask': node_mask_guided
            }
            
            # Synchronize before reporting completion
            if dist.is_initialized():
                dist.barrier()
            
            if is_main_process():
                print(f"   ✓ All ranks completed guidance scale {guidance_scale} ({total_samples} molecules per rank)")
            
            # Log guidance scale completion (only from main process)
            if WANDB_AVAILABLE and wandb.run is not None and is_main_process():
                wandb.log({f"sampling/guidance_scale_{guidance_scale}_completed": True})
            
    else:
        if is_main_process():
            print(f"\n2. MLFF GUIDANCE SKIPPED")
            print(f"   Reason: Predictor not available")
        guided_results = {}
    
    return {
        'baseline': baseline_results,
        'guided': guided_results
    }


def analyze_stability(positions, one_hot, node_mask, dataset_info, name="molecules", log_wandb=True):
    """Analyze stability of generated molecules."""
    if is_main_process():
        print(f"\nAnalyzing stability of {name}...")
    
    n_samples = positions.shape[0]
    stable_count = 0
    stability_results = []
    
    for i in range(n_samples):
        if node_mask.dim() == 3:
            num_atoms = int(node_mask[i].sum().item())
            atom_type = one_hot[i, :num_atoms].argmax(1).cpu().numpy()
        else:
            num_atoms = int(node_mask[i].sum().item())
            atom_type = one_hot[i, :num_atoms].argmax(1).cpu().numpy()
        
        x_mol = positions[i, :num_atoms].cpu().numpy()
        
        try:
            mol_stable, num_stable_atoms, num_atoms_total = check_stability(
                x_mol, atom_type, dataset_info)
            
            if mol_stable:
                stable_count += 1
            
            stability_results.append({
                'stable': mol_stable,
                'stable_atoms': num_stable_atoms,
                'total_atoms': num_atoms_total,
                'stability_ratio': num_stable_atoms / num_atoms_total if num_atoms_total > 0 else 0
            })
        except Exception as e:
            if is_main_process():
                print(f"   Warning: Could not analyze molecule {i}: {e}")
            stability_results.append({
                'stable': False,
                'stable_atoms': 0,
                'total_atoms': num_atoms,
                'stability_ratio': 0.0
            })
    
    stability_percentage = (stable_count / n_samples) * 100
    avg_stability_ratio = np.mean([r['stability_ratio'] for r in stability_results])
    
    if is_main_process():
        print(f"   Stable molecules: {stable_count}/{n_samples} ({stability_percentage:.1f}%)")
        print(f"   Average atom stability ratio: {avg_stability_ratio:.3f}")
    
    # Log to wandb if available and enabled (only from main process)
    if WANDB_AVAILABLE and wandb.run is not None and log_wandb and is_main_process():
        log_data = {
            f"{name}/stable_molecules": stable_count,
            f"{name}/total_molecules": n_samples,
            f"{name}/stability_percentage": stability_percentage,
            f"{name}/avg_stability_ratio": avg_stability_ratio,
        }
        wandb.log(log_data)
    
    return stability_results, stability_percentage


def save_comparison_results(results, eval_args, dataset_info):
    """Save comparison results (only from main process)."""
    # Only save from main process
    if not is_main_process():
        return {}
    
    output_dir = join(eval_args.model_path, 'eval/mlff_guided/')
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize summary for wandb logging
    summary_stats = {}
    
    # Save baseline results (if available)
    baseline_percentage = 0.0
    if results['baseline'] is not None:
        baseline = results['baseline']
        print(f"\nSaving baseline results to {output_dir}")
        
        vis.save_xyz_file(
            join(output_dir, 'baseline/'), 
            baseline['one_hot'], baseline['charges'], baseline['positions'],
            dataset_info, id_from=0, name='molecule_baseline',
            node_mask=baseline['node_mask']
        )
        
        # Analyze baseline stability
        baseline_stability, baseline_percentage = analyze_stability(
            baseline['positions'], baseline['one_hot'], 
            baseline['node_mask'], dataset_info, name="baseline"
        )
        summary_stats['baseline_stability'] = baseline_percentage
    else:
        print(f"\nSkipping baseline results (not computed)")
        summary_stats['baseline_stability'] = None
    
    # Save guided results
    guided = results['guided']
    guidance_scale_stats = {}
    
    for guidance_scale, guided_result in guided.items():
        if guided_result is not None:
            print(f"\nSaving guided results (scale {guidance_scale}) to {output_dir}")
            
            # Extract features
            one_hot = guided_result['features']['categorical']
            charges = guided_result['features']['integer']
            positions = guided_result['positions']
            node_mask = guided_result['node_mask']
            
            # Save files
            subdir = f"guided_scale_{guidance_scale}/"
            vis.save_xyz_file(
                join(output_dir, subdir),
                one_hot, charges, positions,
                dataset_info, id_from=0, name=f'molecule_guided_{guidance_scale}',
                node_mask=node_mask
            )
            
            # Analyze stability
            guided_stability, guided_percentage = analyze_stability(
                positions, one_hot, node_mask, dataset_info, 
                name=f"guided_scale_{guidance_scale}"
            )
            
            # Store stats for summary
            guidance_scale_stats[guidance_scale] = guided_percentage
            summary_stats[f'guided_scale_{guidance_scale}_stability'] = guided_percentage
    
    # Log summary comparison to wandb (main process only)
    if WANDB_AVAILABLE and wandb.run is not None:
        # Create comparison table
        if guidance_scale_stats:
            comparison_data = []
            if results['baseline'] is not None:
                comparison_data.append(['baseline', baseline_percentage])
            for scale, percentage in guidance_scale_stats.items():
                comparison_data.append([f'guided_{scale}', percentage])
            
            # Log summary metrics
            summary_stats['best_guided_stability'] = max(guidance_scale_stats.values()) if guidance_scale_stats else 0
            summary_stats['best_guidance_scale'] = max(guidance_scale_stats.keys(), key=guidance_scale_stats.get) if guidance_scale_stats else None
            
            if results['baseline'] is not None:
                summary_stats['stability_improvement'] = summary_stats['best_guided_stability'] - baseline_percentage if guidance_scale_stats else 0
            else:
                summary_stats['stability_improvement'] = None  # Cannot compute without baseline
            
            wandb.log(summary_stats)
            
            # Create wandb table for comparison (only if we have data)
            if comparison_data:
                table = wandb.Table(data=comparison_data, columns=['method', 'stability_percentage'])
                wandb.log({'stability_comparison': wandb.plot.bar(table, 'method', 'stability_percentage', title='Stability Comparison by Method')})
    
    print(f"\n✓ All results saved to {output_dir}")
    return summary_stats


def sample_chain_with_guidance(args, eval_args, device, flow, mlff_predictor,
                              n_tries, n_nodes, dataset_info):
    """Sample visualization chain with MLFF guidance (only from main process)."""
    
    # Only sample chain from main process to avoid conflicts
    if not is_main_process():
        return
    
    print(f"\nSampling visualization chain with {n_nodes} nodes...")
    
    # Sample baseline chain
    print("1. Sampling baseline chain...")
    one_hot_baseline, charges_baseline, x_baseline = sample_chain(
        args, device, flow, n_tries, dataset_info)
    
    # Save baseline chain
    target_path = 'eval/mlff_guided/chain_baseline/'
    vis.save_xyz_file(
        join(eval_args.model_path, target_path), 
        one_hot_baseline, charges_baseline, x_baseline,
        dataset_info, id_from=0, name='chain_baseline'
    )
    
    print("   ✓ Baseline chain saved")
    
    # Sample guided chain (if predictor available)
    if mlff_predictor is not None:
        print("2. Sampling guided chain...")
        
        # Create guided model with appropriate guidance scale
        guided_model = create_mlff_guided_model(
            flow, mlff_predictor, guidance_scale=0.008, dataset_info=dataset_info, 
            guidance_iterations=eval_args.guidance_iterations, noise_threshold=eval_args.noise_threshold
        )
        
        # Sample guided chain
        n_samples = 1
        node_mask = torch.ones(n_samples, n_nodes, 1, device=device)
        edge_mask = torch.ones(n_samples, n_nodes, n_nodes, device=device)
        diag_mask = ~torch.eye(n_nodes, dtype=torch.bool, device=device).unsqueeze(0)
        edge_mask = edge_mask * diag_mask.unsqueeze(0)
        edge_mask = edge_mask.view(n_samples * n_nodes * n_nodes, 1)
        context = None
        
        # Sample chain with guidance
        chain = guided_model.sample_chain(
            n_samples, n_nodes, node_mask, edge_mask, context, keep_frames=100
        )
        
        # Process chain for visualization
        x_guided = chain[:, :, :3]
        h_guided_cat = chain[:, :, 3:-1] if guided_model.include_charges else chain[:, :, 3:]
        h_guided_int = chain[:, :, -1:] if guided_model.include_charges else torch.zeros(chain.shape[0], chain.shape[1], 1, device=device)
        
        # Keep categorical features as one-hot encoded (don't convert to indices)
        one_hot_guided = h_guided_cat
        charges_guided = h_guided_int if guided_model.include_charges else torch.zeros(chain.shape[0], chain.shape[1], 1, device=device)
        
        # Save guided chain
        target_path = 'eval/mlff_guided/chain_guided/'
        vis.save_xyz_file(
            join(eval_args.model_path, target_path),
            one_hot_guided, charges_guided, x_guided,
            dataset_info, id_from=0, name='chain_guided'
        )
        
        print("   ✓ Guided chain saved")
    
    else:
        print("2. Skipping guided chain (predictor not available)")


def main():
    parser = argparse.ArgumentParser(description='Evaluate diffusion model with MLFF guidance')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, default="outputs/edm_1",
                        help='Path to trained diffusion model')
    
    # Sampling arguments
    parser.add_argument('--n_samples', type=int, default=200,
                        help='Number of samples for comparison')
    parser.add_argument('--n_tries', type=int, default=10,
                        help='Number of tries to find stable molecule for chain visualization')
    parser.add_argument('--n_nodes', type=int, default=19,
                        help='Number of atoms in molecule for chain visualization')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch size for MLFF guided sampling (to manage GPU memory)')
    
    # MLFF arguments
    parser.add_argument('--mlff_model', type=str, default='uma-s-1p1',
                        help='MLFF model name')
    parser.add_argument('--task_name', type=str, default='omol',
                        help='Task name for MLFF predictor')
    parser.add_argument('--guidance_scales', type=float, nargs='+', 
                        default=[0.001, 0.002, 0.003],
                        help='Guidance scales to test')
    parser.add_argument('--guidance_iterations', type=int, default=1,
                        help='Number of iterative force field evaluations per diffusion step')
    parser.add_argument('--noise_threshold', type=float, default=0.8,
                        help='Skip guidance when noise level exceeds this threshold (0.8 = skip first ~20%% of steps)')
    
    # Evaluation options
    parser.add_argument('--skip_comparison', action='store_true',
                        help='Skip comparison sampling')
    parser.add_argument('--skip_baseline', action='store_true',
                        help='Skip baseline comparison sampling (only run MLFF-guided)')
    parser.add_argument('--skip_chain', action='store_true',
                        help='Skip chain visualization')
    parser.add_argument('--skip_visualization', action='store_true',
                        help='Skip 3D visualization')
    
    # Distributed training arguments
    parser.add_argument('--use_distributed', action='store_true',
                        help='Enable distributed multi-GPU evaluation')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='Local rank for distributed training (set automatically by launch script)')
    parser.add_argument('--world_size', type=int, default=1,
                        help='Total number of processes (set automatically by launch script)')
    parser.add_argument('--dist_backend', type=str, default='nccl',
                        help='Distributed backend (nccl or gloo)')
    
    # Random seed argument
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible results')
    
    # Wandb arguments
    parser.add_argument('--use_wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='mlff-guided-diffusion',
                        help='Wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='Wandb entity (username or team name)')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='Wandb run name')
    parser.add_argument('--wandb_tags', type=str, nargs='*', default=[],
                        help='Wandb tags for the run')
    
    eval_args, unparsed_args = parser.parse_known_args()
    
    assert eval_args.model_path is not None, "Model path must be specified"
    
    # Setup distributed training if requested
    if eval_args.use_distributed:
        # Get distributed info from environment or args
        local_rank = eval_args.local_rank
        if local_rank == -1:
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        world_size = eval_args.world_size
        if world_size == 1:
            world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        # Initialize distributed training
        setup_distributed(local_rank, world_size, eval_args.dist_backend)
        
        # Set device to local rank
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        
        if is_main_process():
            print(f"Initialized distributed training with {world_size} GPUs")
    else:
        # Single GPU setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if torch.cuda.is_available():
            print("Using single GPU evaluation")
        else:
            print("Using CPU evaluation")
    
    # Load model arguments
    with open(join(eval_args.model_path, 'args.pickle'), 'rb') as f:
        args = pickle.load(f)
    
    # Handle missing attributes (same as original eval_sample.py)
    if not hasattr(args, 'normalization_factor'):
        args.normalization_factor = 1
    if not hasattr(args, 'aggregation_method'):
        args.aggregation_method = 'sum'
    
    # Setup device
    args.cuda = device.type == 'cuda'
    args.device = device
    dtype = torch.float32
    
    if is_main_process():
        print("MLFF-Guided Diffusion Evaluation")
        print("=" * 40)
        print(f"Model path: {eval_args.model_path}")
        print(f"Device: {device}")
        print(f"Dataset: {args.dataset}")
        if eval_args.use_distributed:
            rank, world_size = get_rank_and_world_size()
            print(f"Distributed: {world_size} GPUs (rank {rank})")
    
    # Create folders (only from main process)
    if is_main_process():
        utils.create_folders(args)
    
    # Synchronize all processes before proceeding
    if eval_args.use_distributed:
        dist.barrier()
    
    # Load dataset info
    dataset_info = get_dataset_info(args.dataset, args.remove_h)
    if is_main_process():
        print(f"Atom decoder: {dataset_info['atom_decoder']}")
        print(f"Include charges: {args.include_charges}")
        print(f"Context node nf: {args.context_node_nf}")
        print(f"Expected input features: {len(dataset_info['atom_decoder']) + int(args.include_charges)}")
    
    # Load dataloaders
    dataloaders, charge_scale = dataset.retrieve_dataloaders(args)
    
    # Load diffusion model
    if is_main_process():
        print("\nLoading diffusion model...")
    
    flow, nodes_dist, prop_dist = get_model(
        args, device, dataset_info, dataloaders['train'])
    flow.to(device)
    
    # Handle conditional models that don't have proper normalizer
    if prop_dist is not None and prop_dist.normalizer is None:
        if is_main_process():
            print("Warning: Model expects conditioning but no normalizer available.")
            print("Setting prop_dist to None but keeping context_node_nf to match model architecture.")
        prop_dist = None
    
    # Load model weights
    fn = 'generative_model_ema.npy' if args.ema_decay > 0 else 'generative_model.npy'
    flow_state_dict = torch.load(join(eval_args.model_path, fn), map_location=device)
    flow.load_state_dict(flow_state_dict)
    
    # For evaluation, we don't need DDP wrapping - it causes issues with sampling
    if is_main_process():
        print("✓ Diffusion model loaded successfully (no DDP for evaluation)")
    
    # Load MLFF predictor
    mlff_predictor = load_mlff_predictor(eval_args.mlff_model, device)
    
    # Initialize wandb if requested and available (only from main process)
    if eval_args.use_wandb and WANDB_AVAILABLE and is_main_process():
        print("\nInitializing Weights & Biases...")
        
        # Create run config
        config = {
            # Model config
            'model_path': eval_args.model_path,
            'dataset': args.dataset,
            'device': str(device),
            'remove_h': args.remove_h,
            'ema_decay': args.ema_decay,
            
            # Sampling config
            'n_samples': eval_args.n_samples,
            'n_tries': eval_args.n_tries,
            'n_nodes': eval_args.n_nodes,
            
            # MLFF config
            'mlff_model': eval_args.mlff_model,
            'mlff_available': mlff_predictor is not None,
            'guidance_scales': eval_args.guidance_scales,
            'guidance_iterations': eval_args.guidance_iterations,
            'noise_threshold': eval_args.noise_threshold,
            
            # Dataset info
            'max_n_nodes': dataset_info['max_n_nodes'],
            'atom_decoder': dataset_info['atom_decoder'],
            'atom_encoder': dataset_info.get('atom_encoder', {}),
            
            # Evaluation options
            'skip_comparison': eval_args.skip_comparison,
            'skip_baseline': eval_args.skip_baseline,
            'skip_chain': eval_args.skip_chain,
            'skip_visualization': eval_args.skip_visualization,
            
            # Distributed config
            'use_distributed': eval_args.use_distributed,
            'world_size': get_rank_and_world_size()[1] if eval_args.use_distributed else 1,
        }
        
        # Initialize wandb run
        wandb.init(
            project=eval_args.wandb_project,
            entity=eval_args.wandb_entity,
            name=eval_args.wandb_run_name,
            config=config,
            tags=eval_args.wandb_tags + ['evaluation', 'mlff-guided'] + (['distributed'] if eval_args.use_distributed else []),
            job_type='evaluation'
        )
        
        print(f"✓ Wandb initialized: {wandb.run.name}")
    
    elif eval_args.use_wandb and not WANDB_AVAILABLE and is_main_process():
        print("Warning: Wandb requested but not available. Install with 'pip install wandb'.")
    
    # Run comparison sampling
    if not eval_args.skip_comparison:
        if is_main_process():
            print("\n" + "="*50)
            print("COMPARISON SAMPLING")
            print("="*50)
        
        # Synchronize before sampling
        if eval_args.use_distributed:
            dist.barrier()
        
        # Sample on each GPU
        local_results = sample_with_mlff_comparison(
            args, eval_args, device, flow, nodes_dist, dataset_info, 
            mlff_predictor, prop_dist, n_samples=eval_args.n_samples
        )
        
        # Gather results from all GPUs
        if eval_args.use_distributed:
            if is_main_process():
                print("\nGathering results from all GPUs...")
            dist.barrier()
            results = all_gather_results(local_results, device)
        else:
            results = local_results
        
        # Save and analyze results (only from main process)
        summary_stats = save_comparison_results(results, eval_args, dataset_info)
        
        # Visualize results (only from main process)
        if not eval_args.skip_visualization and is_main_process():
            print("\nGenerating visualizations...")
            output_dir = join(eval_args.model_path, 'eval/mlff_guided/')
            
            # Visualize baseline (if available)
            if results['baseline'] is not None:
                try:
                    vis.visualize(
                        join(output_dir, 'baseline/'), dataset_info,
                        max_num=100, spheres_3d=True
                    )
                    print("✓ Baseline visualization saved")
                except Exception as e:
                    print(f"Warning: Failed to create baseline visualization: {e}")
            else:
                print("Skipping baseline visualization (baseline not computed)")
            
            # Visualize guided results
            for guidance_scale in results['guided'].keys():
                if results['guided'][guidance_scale] is not None:
                    try:
                        vis.visualize(
                            join(output_dir, f'guided_scale_{guidance_scale}/'), dataset_info,
                            max_num=100, spheres_3d=True
                        )
                        print(f"✓ Guided visualization (scale {guidance_scale}) saved")
                    except Exception as e:
                        print(f"Warning: Failed to create guided visualization (scale {guidance_scale}): {e}")

    
    # Run chain visualization
    if not eval_args.skip_chain:
        if is_main_process():
            print("\n" + "="*50)
            print("CHAIN VISUALIZATION")
            print("="*50)
        
        # Synchronize before chain sampling
        if eval_args.use_distributed:
            dist.barrier()
        
        # Log chain visualization start (only from main process)
        if WANDB_AVAILABLE and wandb.run is not None and is_main_process():
            wandb.log({
                'chain_visualization/started': True,
                'chain_visualization/n_nodes': eval_args.n_nodes,
                'chain_visualization/n_tries': eval_args.n_tries,
            })
        
        # Sample chain (only from main process to avoid conflicts)
        sample_chain_with_guidance(
            args, eval_args, device, flow, mlff_predictor,
            eval_args.n_tries, eval_args.n_nodes, dataset_info
        )
        
        # Synchronize after chain sampling
        if eval_args.use_distributed:
            dist.barrier()
        
        # Create chain visualizations (only from main process)
        if not eval_args.skip_visualization and is_main_process():
            output_dir = join(eval_args.model_path, 'eval/mlff_guided/')
            
            try:
                vis.visualize_chain_uncertainty(
                    join(output_dir, 'chain_baseline/'), dataset_info, spheres_3d=True
                )
                print("✓ Baseline chain visualization saved")
            except Exception as e:
                print(f"Warning: Failed to create baseline chain visualization: {e}")

            if mlff_predictor is not None:
                try:
                    vis.visualize_chain_uncertainty(
                        join(output_dir, 'chain_guided/'), dataset_info, spheres_3d=True
                    )
                    print("✓ Guided chain visualization saved")
                except Exception as e:
                    print(f"Warning: Failed to create guided chain visualization: {e}")
                
        # Log chain visualization completion (only from main process)
        if WANDB_AVAILABLE and wandb.run is not None and is_main_process():
            wandb.log({
                'chain_visualization/completed': True,
                'chain_visualization/baseline_saved': True,
                'chain_visualization/guided_saved': mlff_predictor is not None,
            })

    # Synchronize all processes before final cleanup
    if eval_args.use_distributed:
        dist.barrier()
    
    if is_main_process():
        print("\n" + "="*50)
        print("EVALUATION COMPLETE")
        print("="*50)
        print(f"Results saved to: {join(eval_args.model_path, 'eval/mlff_guided/')}")
    
    # Final wandb logging (only from main process)
    if WANDB_AVAILABLE and wandb.run is not None and is_main_process():
        final_summary = {
            'evaluation_completed': True,
            'comparison_performed': not eval_args.skip_comparison,
            'chain_visualization_performed': not eval_args.skip_chain,
            'visualization_generated': not eval_args.skip_visualization,
            'output_directory': join(eval_args.model_path, 'eval/mlff_guided/'),
            'distributed_used': eval_args.use_distributed,
            'world_size': get_rank_and_world_size()[1],
        }
        wandb.log(final_summary)
        
        # Add evaluation status to summary
        wandb.run.summary.update(final_summary)
        
        print(f"✓ Wandb logging completed: {wandb.run.url}")
        wandb.finish()
    
    if is_main_process():
        if mlff_predictor is not None:
            print("\n✓ Successfully evaluated both baseline and MLFF-guided sampling")
            if eval_args.use_distributed:
                print(f"   Using {get_rank_and_world_size()[1]} GPUs for distributed evaluation")
            print("Compare the results to see the effect of MLFF guidance on:")
            print("  - Molecular stability")
            print("  - Structural quality") 
            print("  - Chemical validity")
        else:
            print("\n! Only baseline sampling was performed (MLFF predictor not available)")
            print("To enable MLFF guidance:")
            print("  1. Set up HuggingFace authentication")
            print("  2. Request access to UMA models")
            print("  3. Re-run the evaluation")
    
    # Clean up distributed training
    if eval_args.use_distributed:
        cleanup_distributed()


if __name__ == "__main__":
    main() 