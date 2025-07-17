# Rdkit import should be first, do not move it
try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass

import utils
import argparse
from configs.datasets_config import get_dataset_info
from qm9 import dataset
from qm9.models import get_model
from fairchem.core import pretrained_mlip, FAIRChemCalculator

from equivariant_diffusion.utils import assert_correctly_masked
import torch
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
import os


def check_mask_correct(variables, node_mask):
    for variable in variables:
        assert_correctly_masked(variable, node_mask)


def load_mlff_predictor(model_name, device):
    """Load MLFF predictor with error handling."""
    print(f"Loading MLFF predictor: {model_name}")
    
    try:
        # Convert device to string format expected by predictor
        device_str = 'cuda' if device.type == 'cuda' else 'cpu'
        predictor = pretrained_mlip.get_predict_unit(model_name, device=device_str)
        print(f"✓ Successfully loaded MLFF predictor: {model_name}")
        return predictor
    except Exception as e:
        print(f"✗ Error loading MLFF predictor: {e}")
        print("Make sure you have access to the UMA model and proper authentication.")
        print("You can still run baseline sampling without MLFF guidance.")
        return None


def sample_with_mlff_comparison(args, eval_args, device, flow, nodes_dist, 
                               dataset_info, mlff_predictor, n_samples=10):
    """Sample molecules with and without MLFF guidance for comparison."""
    
    print(f"\nSampling {n_samples} molecules for comparison...")
    
    # Sample baseline (without guidance)
    print("1. Sampling baseline molecules (without MLFF guidance)...")
    nodesxsample = nodes_dist.sample(n_samples)
    one_hot_baseline, charges_baseline, x_baseline, node_mask = sample(
        args, device, flow, dataset_info, nodesxsample=nodesxsample)
    
    print(f"   ✓ Generated {n_samples} baseline molecules")
    
    # Sample with MLFF guidance (if predictor available)
    if mlff_predictor is not None:
        print("2. Sampling with MLFF guidance...")
        
        # Create sampling inputs
        max_n_nodes = dataset_info['max_n_nodes']
        batch_size = len(nodesxsample)
        
        # Create node mask
        node_mask_guided = torch.zeros(batch_size, max_n_nodes, device=device)
        for i in range(batch_size):
            node_mask_guided[i, 0:nodesxsample[i]] = 1
        node_mask_guided = node_mask_guided.unsqueeze(2)
        
        # Create edge mask
        edge_mask = node_mask_guided.squeeze(2).unsqueeze(1) * node_mask_guided.squeeze(2).unsqueeze(2)
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool, device=device).unsqueeze(0)
        edge_mask *= diag_mask
        edge_mask = edge_mask.view(batch_size * max_n_nodes * max_n_nodes, 1)
        
        # No context for unconditional generation
        context = None
        
        # Sample with different guidance scales
        guidance_scales = [0.0, 0.0005, 0.0007, 0.008]  # Include 0.0 to test consistency with baseline
        guided_results = {}
        
        for guidance_scale in guidance_scales:
            print(f"   Sampling with guidance scale {guidance_scale}...")
            x_guided, h_guided = enhanced_sampling_with_mlff(
                flow, mlff_predictor, batch_size, max_n_nodes, 
                node_mask_guided, edge_mask, context, dataset_info,
                guidance_scale=guidance_scale, fix_noise=False
            )
            
            guided_results[guidance_scale] = {
                'positions': x_guided,
                'features': h_guided,
                'node_mask': node_mask_guided
            }
            print(f"   ✓ Generated molecules with guidance scale {guidance_scale}")
            
    else:
        print("2. Skipping MLFF guidance (predictor not available)")
        guided_results = {}
    
    return {
        'baseline': {
            'one_hot': one_hot_baseline,
            'charges': charges_baseline,
            'positions': x_baseline,
            'node_mask': node_mask
        },
        'guided': guided_results
    }


def analyze_stability(positions, one_hot, node_mask, dataset_info, name="molecules"):
    """Analyze stability of generated molecules."""
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
            print(f"   Warning: Could not analyze molecule {i}: {e}")
            stability_results.append({
                'stable': False,
                'stable_atoms': 0,
                'total_atoms': num_atoms,
                'stability_ratio': 0.0
            })
    
    stability_percentage = (stable_count / n_samples) * 100
    avg_stability_ratio = np.mean([r['stability_ratio'] for r in stability_results])
    
    print(f"   Stable molecules: {stable_count}/{n_samples} ({stability_percentage:.1f}%)")
    print(f"   Average atom stability ratio: {avg_stability_ratio:.3f}")
    
    return stability_results, stability_percentage


def save_comparison_results(results, eval_args, dataset_info):
    """Save comparison results."""
    output_dir = join(eval_args.model_path, 'eval/mlff_guided/')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save baseline results
    baseline = results['baseline']
    print(f"\nSaving baseline results to {output_dir}")
    
    vis.save_xyz_file(
        join(output_dir, 'baseline/'), 
        baseline['one_hot'], baseline['charges'], baseline['positions'],
        dataset_info, id_from=0, name='molecule_baseline',
        node_mask=baseline['node_mask']
    )
    
    # Analyze baseline stability
    analyze_stability(
        baseline['positions'], baseline['one_hot'], 
        baseline['node_mask'], dataset_info, name="baseline molecules"
    )
    
    # Save guided results
    guided = results['guided']
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
            analyze_stability(
                positions, one_hot, node_mask, dataset_info, 
                name=f"guided molecules (scale {guidance_scale})"
            )
    
    print(f"\n✓ All results saved to {output_dir}")


def sample_chain_with_guidance(args, eval_args, device, flow, mlff_predictor,
                              n_tries, n_nodes, dataset_info):
    """Sample visualization chain with MLFF guidance."""
    
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
        
        # Create guided model
        guided_model = create_mlff_guided_model(
            flow, mlff_predictor, guidance_scale=1.0, dataset_info=dataset_info
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
    parser.add_argument('--n_samples', type=int, default=20,
                        help='Number of samples for comparison')
    parser.add_argument('--n_tries', type=int, default=10,
                        help='Number of tries to find stable molecule for chain visualization')
    parser.add_argument('--n_nodes', type=int, default=19,
                        help='Number of atoms in molecule for chain visualization')
    
    # MLFF arguments
    parser.add_argument('--mlff_model', type=str, default='uma-s-1p1',
                        help='MLFF model name')
    parser.add_argument('--task_name', type=str, default='omol',
                        help='Task name for MLFF predictor')
    parser.add_argument('--guidance_scales', type=float, nargs='+', 
                        default=[0.0, 0.5, 1.0, 2.0],
                        help='Guidance scales to test')
    
    # Evaluation options
    parser.add_argument('--skip_comparison', action='store_true',
                        help='Skip comparison sampling')
    parser.add_argument('--skip_chain', action='store_true',
                        help='Skip chain visualization')
    parser.add_argument('--skip_visualization', action='store_true',
                        help='Skip 3D visualization')
    
    eval_args, unparsed_args = parser.parse_known_args()
    
    assert eval_args.model_path is not None, "Model path must be specified"
    
    # Load model arguments
    with open(join(eval_args.model_path, 'args.pickle'), 'rb') as f:
        args = pickle.load(f)
    
    # Handle missing attributes
    if not hasattr(args, 'normalization_factor'):
        args.normalization_factor = 1
    if not hasattr(args, 'aggregation_method'):
        args.aggregation_method = 'sum'
    
    # Setup device
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device
    dtype = torch.float32
    
    print("MLFF-Guided Diffusion Evaluation")
    print("=" * 40)
    print(f"Model path: {eval_args.model_path}")
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    
    # Create folders
    utils.create_folders(args)
    
    # Load dataset info
    dataset_info = get_dataset_info(args.dataset, args.remove_h)
    print(f"Atom decoder: {dataset_info['atom_decoder']}")
    
    # Load dataloaders
    dataloaders, charge_scale = dataset.retrieve_dataloaders(args)
    
    # Load diffusion model
    print("\nLoading diffusion model...")
    flow, nodes_dist, prop_dist = get_model(
        args, device, dataset_info, dataloaders['train'])
    flow.to(device)
    
    # Load model weights
    fn = 'generative_model_ema.npy' if args.ema_decay > 0 else 'generative_model.npy'
    flow_state_dict = torch.load(join(eval_args.model_path, fn), map_location=device)
    flow.load_state_dict(flow_state_dict)
    print("✓ Diffusion model loaded successfully")
    
    # Load MLFF predictor
    mlff_predictor = load_mlff_predictor(eval_args.mlff_model, device)
    
    # Run comparison sampling
    if not eval_args.skip_comparison:
        print("\n" + "="*50)
        print("COMPARISON SAMPLING")
        print("="*50)
        
        results = sample_with_mlff_comparison(
            args, eval_args, device, flow, nodes_dist, dataset_info, 
            mlff_predictor, n_samples=eval_args.n_samples
        )
        
        # Save and analyze results
        save_comparison_results(results, eval_args, dataset_info)
        
        # Visualize results
        if not eval_args.skip_visualization:
            print("\nGenerating visualizations...")
            output_dir = join(eval_args.model_path, 'eval/mlff_guided/')
            
            # Visualize baseline
            vis.visualize(
                join(output_dir, 'baseline/'), dataset_info,
                max_num=100, spheres_3d=True
            )
            print("✓ Baseline visualization saved")
            
            # Visualize guided results
            for guidance_scale in eval_args.guidance_scales:
                if guidance_scale in results['guided'] and results['guided'][guidance_scale] is not None:

                    vis.visualize(
                        join(output_dir, f'guided_scale_{guidance_scale}/'), dataset_info,
                        max_num=100, spheres_3d=True
                    )
                    print(f"✓ Guided visualization (scale {guidance_scale}) saved")

    
    # Run chain visualization
    if not eval_args.skip_chain:
        print("\n" + "="*50)
        print("CHAIN VISUALIZATION")
        print("="*50)
        
        sample_chain_with_guidance(
            args, eval_args, device, flow, mlff_predictor,
            eval_args.n_tries, eval_args.n_nodes, dataset_info
        )
        
        # Create chain visualizations
        if not eval_args.skip_visualization:
            output_dir = join(eval_args.model_path, 'eval/mlff_guided/')
            
            vis.visualize_chain_uncertainty(
                join(output_dir, 'chain_baseline/'), dataset_info, spheres_3d=True
            )
            print("✓ Baseline chain visualization saved")


            
            if mlff_predictor is not None:

                vis.visualize_chain_uncertainty(
                    join(output_dir, 'chain_guided/'), dataset_info, spheres_3d=True
                )
                print("✓ Guided chain visualization saved")

    print("\n" + "="*50)
    print("EVALUATION COMPLETE")
    print("="*50)
    print(f"Results saved to: {join(eval_args.model_path, 'eval/mlff_guided/')}")
    
    if mlff_predictor is not None:
        print("\n✓ Successfully evaluated both baseline and MLFF-guided sampling")
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


if __name__ == "__main__":
    main() 