import torch
import numpy as np
from typing import Optional, Dict

# Optional wandb
try:
    import wandb  # type: ignore
    WANDB_AVAILABLE = True
except Exception:
    WANDB_AVAILABLE = False

import torch.distributed as dist
from fairchem.core import pretrained_mlip
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.datasets import data_list_collater


def is_main_process() -> bool:
    """Return True if current process is rank 0 or DDP not initialized."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def load_mlff_predictor_for_energy(model_name: Optional[str], device: torch.device):
    """Load UMA predictor for energy evaluation with simple messaging."""
    if model_name is None or str(model_name).lower() in {"", "none", "null"}:
        model_name = "uma-s-1p1"
        if is_main_process():
            print("Energy eval: defaulting UMA model to 'uma-s-1p1'")

    if model_name == 'uma-s-1':
        model_name = 'uma-s-1p1'
        if is_main_process():
            print("Energy eval: mapping 'uma-s-1' -> 'uma-s-1p1'")

    try:
        device_str = 'cuda' if device.type == 'cuda' else 'cpu'
        predictor = pretrained_mlip.get_predict_unit(model_name, device=device_str)
        if is_main_process():
            print(f"Energy eval: loaded UMA predictor '{model_name}'")
        return predictor
    except Exception as e:
        if is_main_process():
            print(f"Energy eval: failed to load UMA predictor '{model_name}': {e}")
        return None


def one_hot_to_atomic_numbers(one_hot: torch.Tensor, dataset_info: Dict) -> torch.Tensor:
    """Convert one-hot atom type encodings to atomic numbers using dataset_info."""
    num_classes = len(dataset_info['atom_decoder'])
    categorical = one_hot[:, :num_classes] if one_hot.shape[1] >= num_classes else one_hot
    type_indices = torch.argmax(categorical, dim=1).cpu()

    if 'atomic_nb' in dataset_info:
        atomic_numbers = torch.tensor([dataset_info['atomic_nb'][idx] for idx in type_indices])
        return atomic_numbers

    # Default QM9 mapping
    atom_decoder = dataset_info['atom_decoder']
    symbol_to_z = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}
    atomic_numbers = torch.tensor([symbol_to_z.get(atom_decoder[idx], 1) for idx in type_indices])
    return atomic_numbers


def molecules_to_atomicdata_list(
    positions: torch.Tensor,
    one_hot: torch.Tensor,
    node_mask: torch.Tensor,
    dataset_info: Dict,
    task_name: str = 'omol',
):
    """Convert batched diffusion outputs to a list of AtomicData for UMA.

    Note: Final sampled positions from the diffusion model are already unnormalized
    to physical units (Angstroms). We therefore pass them directly without extra scaling.
    """
    B, N, _ = positions.shape
    atomic_data_list = []

    # Positions are already in physical units from sampling
    pos_phys = positions.detach().cpu()
    feat = one_hot.detach().cpu()
    mask = node_mask.detach().cpu().squeeze(-1).bool()

    from ase import Atoms

    for b in range(B):
        valid = mask[b]
        if valid.sum().item() == 0:
            continue
        pos_b = pos_phys[b, valid]
        onehot_b = feat[b, valid]

        atomic_numbers = one_hot_to_atomic_numbers(onehot_b, dataset_info)

        atoms = Atoms(
            numbers=atomic_numbers.numpy(),
            positions=pos_b.numpy(),
            cell=np.eye(3) * 50.0,
            pbc=False,
        )
        atoms.info['charge'] = 0
        atoms.info['spin'] = 1

        try:
            ad = AtomicData.from_ase(
                atoms,
                r_edges=True,
                radius=6.0,
                max_neigh=50,
                task_name=task_name,
                r_data_keys=["charge", "spin"],
            )
            atomic_data_list.append(ad)
        except Exception:
            continue

    return atomic_data_list


def evaluate_energy_stats(
    name: str,
    positions: torch.Tensor,
    one_hot: torch.Tensor,
    node_mask: torch.Tensor,
    dataset_info: Dict,
    mlff_predictor,
    device: torch.device,
    task_name: str,
    energy_batch_size: int = 256,
):
    """Compute energy stats for a block of samples; returns dict or None."""
    ad_list = molecules_to_atomicdata_list(
        positions, one_hot, node_mask, dataset_info, task_name
    )
    if not ad_list:
        return None
    # Chunked prediction to control memory
    all_e_list = []
    start = 0
    total = len(ad_list)
    while start < total:
        end = min(start + energy_batch_size, total)
        chunk = ad_list[start:end]
        try:
            batch = data_list_collater(chunk, otf_graph=True)
            batch = batch.to(device)
        except Exception as e:
            if is_main_process():
                print(f"Energy eval: collate failed for {name} [{start}:{end}]: {e}")
            # Skip this chunk
            start = end
            continue

        try:
            with torch.no_grad():
                preds = mlff_predictor.predict(batch)
        except Exception as e:
            if is_main_process():
                print(f"Energy eval: UMA prediction failed for {name} [{start}:{end}]: {e}")
            start = end
            continue

        energies = None
        for key in ["energy", "energies", "total_energy"]:
            if key in preds:
                energies = preds[key]
                break
        if energies is not None:
            e = energies.detach().cpu().view(-1).numpy()
            if e.size > 0:
                all_e_list.append(e)

        start = end

    if not all_e_list:
        return None
    e = np.concatenate(all_e_list, axis=0)
    stats = {
        'count': int(e.size),
        'mean': float(np.mean(e)),
        'median': float(np.median(e)),
        'std': float(np.std(e)),
        'min': float(np.min(e)),
        'max': float(np.max(e)),
    }
    return stats


def compare_energy_distributions(
    results: Dict,
    eval_args,
    dataset_info: Dict,
    mlff_predictor,
    device: torch.device,
):
    """Evaluate UMA energies for baseline and each guidance scale and print stats."""
    if not is_main_process():
        return

    # Ensure UMA predictor exists
    if mlff_predictor is None:
        print("UMA predictor not loaded during sampling; loading now for energy evaluation...")
        mlff_predictor = load_mlff_predictor_for_energy(eval_args.mlff_model, device)
        if mlff_predictor is None:
            print("Skipping energy comparison: UMA model unavailable.")
            return

    print("\n" + "="*50)
    print("ENERGY COMPARISON (UMA)")
    print("="*50)

    baseline_stats = None
    if results.get('baseline') is not None:
        b = results['baseline']
        baseline_stats = evaluate_energy_stats(
            'baseline', b['positions'], b['one_hot'], b['node_mask'],
            dataset_info, mlff_predictor, device,
            task_name=getattr(eval_args, 'task_name', 'omol'),
            energy_batch_size=getattr(eval_args, 'energy_batch_size', 256),
        )
        if baseline_stats is not None:
            print(f"Baseline energy (eV): mean={baseline_stats['mean']:.4f}, "
                  f"median={baseline_stats['median']:.4f}, std={baseline_stats['std']:.4f}, "
                  f"min={baseline_stats['min']:.4f}, max={baseline_stats['max']:.4f}, n={baseline_stats['count']}")
        else:
            print("Baseline energy: unavailable")
    else:
        print("Baseline energy: skipped (no baseline samples)")

    guided_stats = {}
    for scale, guided in results.get('guided', {}).items():
        if guided is None:
            print(f"Guided scale {scale}: no samples")
            continue
        oh = guided['features'].get('categorical') if 'features' in guided else None
        pos = guided.get('positions')
        nm = guided.get('node_mask')
        stats = evaluate_energy_stats(
            f'guided_scale_{scale}', pos, oh, nm,
            dataset_info, mlff_predictor, device,
            task_name=getattr(eval_args, 'task_name', 'omol'),
            energy_batch_size=getattr(eval_args, 'energy_batch_size', 256),
        )
        guided_stats[scale] = stats
        if stats is not None:
            line = (f"Guided {scale}: mean={stats['mean']:.4f}, median={stats['median']:.4f}, "
                    f"std={stats['std']:.4f}, min={stats['min']:.4f}, max={stats['max']:.4f}, n={stats['count']}")
            if baseline_stats is not None:
                delta = stats['mean'] - baseline_stats['mean']
                line += f"  (Δmean vs baseline: {delta:+.4f} eV)"
            print(line)
        else:
            print(f"Guided scale {scale}: energy unavailable")

    if WANDB_AVAILABLE and 'wandb' in globals():
        log_data = {}
        if baseline_stats is not None:
            for k, v in baseline_stats.items():
                log_data[f"energy/baseline_{k}"] = v
        for scale, st in guided_stats.items():
            if st is None:
                continue
            for k, v in st.items():
                log_data[f"energy/guided_{scale}_{k}"] = v
            if baseline_stats is not None:
                log_data[f"energy/guided_{scale}_delta_mean"] = st['mean'] - baseline_stats['mean']
        if log_data:
            wandb.log(log_data)

    print("="*50 + "\n")
