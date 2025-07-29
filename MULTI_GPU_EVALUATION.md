# Multi-GPU Support for MLFF-Guided Diffusion Evaluation

This document describes the multi-GPU support implementation for the MLFF-guided diffusion evaluation script (`eval_mlff_guided.py`).

## Overview

The multi-GPU implementation uses PyTorch's Distributed Data Parallel (DDP) to parallelize the evaluation process across multiple GPUs. This significantly reduces evaluation time for large sample counts.

## Key Features

- **Distributed Sampling**: Splits sample generation across multiple GPUs
- **Result Aggregation**: Automatically gathers and combines results from all processes
- **Coordinated Logging**: Only the main process handles file I/O and Wandb logging
- **MLFF Compatibility**: Each GPU loads its own MLFF predictor instance
- **Chain Visualization**: Sampling chains are generated only on the main process to avoid conflicts

## Usage

### Method 1: Using the Convenience Script

```bash
# Make the script executable
chmod +x e3_diffusion_for_molecules-main/run_distributed_eval.sh

# Basic 4-GPU evaluation
./run_distributed_eval.sh --model_path outputs/my_model --n_gpus 4

# With Wandb logging and more samples
./run_distributed_eval.sh --model_path outputs/my_model --n_gpus 8 --n_samples 2000 --use_wandb

# See all options
./run_distributed_eval.sh --help
```

### Method 2: Direct PyTorch Launch

```bash
# Basic distributed evaluation
python -m torch.distributed.launch --nproc_per_node=4 e3_diffusion_for_molecules-main/eval_mlff_guided.py \
    --model_path outputs/edm_1 --use_distributed --n_samples 1000

# With Wandb logging
python -m torch.distributed.launch --nproc_per_node=4 e3_diffusion_for_molecules-main/eval_mlff_guided.py \
    --model_path outputs/edm_1 --use_distributed --n_samples 1000 \
    --use_wandb --wandb_project my-project
```

### Method 3: Single GPU (Original)

```bash
# Single GPU evaluation (no changes required)
python e3_diffusion_for_molecules-main/eval_mlff_guided.py --model_path outputs/edm_1
```

## Command Line Arguments

### New Distributed Arguments

- `--use_distributed`: Enable distributed multi-GPU evaluation
- `--local_rank`: Local rank for distributed training (set automatically by launch script)
- `--world_size`: Total number of processes (set automatically by launch script)
- `--dist_backend`: Distributed backend (default: 'nccl')

### Existing Arguments (All Still Supported)

- `--model_path`: Path to trained diffusion model
- `--n_samples`: Number of samples for comparison
- `--mlff_model`: MLFF model name (default: 'uma-s-1p1')
- `--guidance_iterations`: Number of iterative force field evaluations per diffusion step
- `--use_wandb`: Enable Weights & Biases logging
- `--wandb_project`: Wandb project name
- And all other existing evaluation options...

## Technical Implementation

### 1. Distributed Setup

```python
def setup_distributed(rank, world_size, backend='nccl'):
    """Initialize distributed training."""
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '12355'
    
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
```

### 2. Sample Distribution

```python
def split_samples_across_gpus(n_samples):
    """Split samples across available GPUs."""
    rank, world_size = get_rank_and_world_size()
    
    # Calculate samples per GPU
    samples_per_gpu = n_samples // world_size
    remaining_samples = n_samples % world_size
    
    # Add extra sample to first few GPUs if remainder exists
    if rank < remaining_samples:
        local_samples = samples_per_gpu + 1
    else:
        local_samples = samples_per_gpu
```

### 3. Result Gathering

```python
def all_gather_results(results, device):
    """Gather results from all processes."""
    # Gather baseline results
    for key in ['one_hot', 'charges', 'positions', 'node_mask']:
        tensor_list = [torch.zeros_like(results['baseline'][key]) for _ in range(world_size)]
        dist.all_gather(tensor_list, results['baseline'][key])
        baseline_gathered[key] = torch.cat(tensor_list, dim=0)
```

### 4. Process Coordination

- **Sampling**: Each GPU processes a portion of the total samples
- **Model Loading**: Each process loads the diffusion model and wraps it with DDP
- **MLFF Predictor**: Each GPU loads its own MLFF predictor instance
- **File I/O**: Only the main process (rank 0) saves files and logs to Wandb
- **Synchronization**: `dist.barrier()` ensures all processes stay synchronized

## Performance Benefits

### Sample Distribution Example

For 1000 samples on 4 GPUs:
- GPU 0: 250 samples
- GPU 1: 250 samples  
- GPU 2: 250 samples
- GPU 3: 250 samples

For 1003 samples on 4 GPUs:
- GPU 0: 251 samples
- GPU 1: 251 samples
- GPU 2: 251 samples
- GPU 3: 250 samples

### Expected Speedup

- **Linear scaling**: Near-linear speedup with number of GPUs for sampling
- **I/O bottleneck**: File saving and analysis still single-threaded on main process
- **MLFF overhead**: Each GPU loads its own MLFF predictor (memory trade-off for speed)

## Output Structure

Results are saved exactly as in single-GPU mode:

```
{model_path}/eval/mlff_guided/
├── baseline/
│   ├── molecule_baseline_*.xyz
│   └── ... (visualization files)
├── guided_scale_0.0/
├── guided_scale_0.0005/
├── guided_scale_0.0007/
├── guided_scale_0.008/
├── chain_baseline/
└── chain_guided/
```

## Wandb Logging

- **Distributed Configuration**: Automatically logs world size and distributed status
- **Process Coordination**: Only rank 0 logs to avoid conflicts
- **Complete Metrics**: All stability metrics and comparisons are aggregated correctly

## Requirements

- PyTorch with distributed support
- Multiple CUDA-capable GPUs
- NCCL backend (recommended for GPU communication)
- Same dependencies as single-GPU version

## Troubleshooting

### Common Issues

1. **Port conflicts**: Change `--master_port` if 29500 is in use
2. **CUDA_VISIBLE_DEVICES**: Ensure proper GPU visibility
3. **Memory issues**: Reduce `n_samples` if running out of GPU memory
4. **MLFF loading**: Each GPU needs access to the MLFF model

### Debug Mode

Add these environment variables for debugging:

```bash
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO
```

## Migration from Single-GPU

Existing single-GPU evaluation scripts work unchanged. To enable multi-GPU:

1. Add `--use_distributed` flag
2. Use `torch.distributed.launch` or the convenience script
3. Optionally adjust `n_samples` for better GPU utilization

## Limitations

- Chain visualization runs only on main process (not parallelized)
- File I/O and analysis remain single-threaded
- Each GPU loads its own MLFF predictor (increased memory usage)
- Requires homogeneous GPU setup for optimal performance 