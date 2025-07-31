# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is an implementation of E(3) Equivariant Diffusion Models for 3D molecular generation, with an enhanced MLFF (Machine Learning Force Field) guidance system. The codebase generates molecules through diffusion processes while maintaining E(3) equivariance and optionally incorporating physics-based constraints from pretrained force fields.

**IMPORTANT**: The MLFF enhancement is the most critical and novel component of this codebase. The remaining components are adaptations from an existing diffusion model framework. When making modifications, prioritize MLFF-related functionality and minimize edits to the existing codebase infrastructure.

## Core Architecture

### Main Components

1. **Equivariant Diffusion (`equivariant_diffusion/`)**
   - `en_diffusion.py`: Core EnVariationalDiffusion class implementing the diffusion process
   - `distributions.py`: Probability distributions for molecular properties
   - `utils.py`: Utility functions for equivariant operations

2. **QM9 Dataset Integration (`qm9/`)**
   - `models.py`: Model factory functions, creates EGNN_dynamics_QM9 and EnVariationalDiffusion
   - `dataset.py`: Data loading and preprocessing for QM9 molecular dataset
   - `sampling.py`: Molecular sampling functions (sample, sample_chain)
   - `visualizer.py`: 3D molecular visualization tools
   - `analyze.py`: Molecular stability and quality analysis

3. **EGNN Dynamics (`egnn/`)**
   - `models.py`: E(3) Equivariant Graph Neural Network implementations
   - `egnn.py`: Core EGNN layers with equivariant message passing

4. MOST IMPORTANT **MLFF Enhancement**
   - `mlff_guided_diffusion.py`: MLFFGuidedDiffusion class extending base diffusion with physics guidance
   - `eval_mlff_guided.py`: Comprehensive evaluation script with multi-GPU support

5. Always conda activate edm

## Key Training Commands

### Basic QM9 Training
```bash
python main_qm9.py --n_epochs 3000 --exp_name edm_qm9 --n_stability_samples 1000 \
    --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 \
    --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 64 --nf 256 \
    --n_layers 9 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 20 --ema_decay 0.9999
```

### Conditional Training (Property-Conditioned)
```bash
python main_qm9.py --exp_name exp_cond_alpha --model egnn_dynamics --lr 1e-4 --nf 192 \
    --n_layers 9 --save_model True --diffusion_steps 1000 --sin_embedding False \
    --n_epochs 3000 --conditioning alpha --dataset qm9_second_half
```

### GEOM-Drugs Training
```bash
python main_geom_drugs.py --n_epochs 3000 --exp_name edm_geom_drugs \
    --n_stability_samples 500 --diffusion_noise_schedule polynomial_2 \
    --diffusion_steps 1000 --batch_size 64 --nf 256 --n_layers 4 --lr 1e-4 \
    --normalize_factors [1,4,10] --normalization_factor 1 --model egnn_dynamics
```

## Evaluation Commands

### Standard Evaluation
```bash
# Analyze molecular quality and stability
python eval_analyze.py --model_path outputs/edm_qm9 --n_samples 10000

# Generate sample visualizations (default DDPM sampling)
python eval_sample.py --model_path outputs/edm_qm9 --n_samples 10000

# Generate samples with DPM-Solver++ (50x faster)
python eval_sample.py --model_path outputs/edm_qm9 --n_samples 10000 \
    --sampling_method dpm_solver++ --dpm_solver_steps 20 --dpm_solver_order 2
```

### MLFF-Guided Evaluation
```bash
# Full evaluation with custom parameters and force clipping (default DDPM)
python eval_mlff_guided.py --model_path outputs/edm_1 --n_samples 200 \
    --mlff_model uma-s-1p1 --guidance_scales 0.1 0.5 1.0 --force_clip_threshold 15.0

# MLFF evaluation with DPM-Solver++ for faster sampling
python eval_mlff_guided.py --model_path outputs/edm_1 --n_samples 200 \
    --mlff_model uma-s-1p1 --guidance_scales 0.1 0.5 1.0 \
    --sampling_method dpm_solver++ --dpm_solver_steps 20 --dpm_solver_order 2

# Multi-GPU distributed evaluation
python -m torch.distributed.launch --nproc_per_node=4 eval_mlff_guided.py \
    --model_path outputs/edm_1 --use_distributed --n_samples 200 \
    --sampling_method dpm_solver++
```

### Conditional Evaluation
```bash
python eval_conditional_qm9.py --generators_path outputs/exp_cond_alpha \
    --property alpha --n_sweeps 10 --task qualitative
```

## Development Patterns

**Note**: The base diffusion framework is well-established. Focus development efforts on MLFF integration and enhancement while preserving existing functionality.

### Model Architecture Flow
1. **Dataset Info**: `get_dataset_info()` provides atom decoders and molecular constraints
2. **Model Creation**: `get_model()` constructs EGNN_dynamics_QM9 wrapped in EnVariationalDiffusion
3. **Training Loop**: `train_epoch()` handles equivariant loss computation and EMA updates
4. **Sampling**: `sample()` and `sample_chain()` generate molecules with optional guidance

### Sampling Method Details

The codebase uses **DDPM-style reverse diffusion sampling** with the following characteristics:

#### Core Sampling Algorithm
- **Method**: Reverse diffusion process implementing DDPM (Denoising Diffusion Probabilistic Models)
- **Implementation**: Located in `equivariant_diffusion/en_diffusion.py:717` (`sample_p_zs_given_zt`)
- **Process**: Iterative denoising from pure noise to molecular structure over T timesteps (typically 1000)

#### Noise Schedules Available
- **Primary**: `polynomial_2` (polynomial schedule with power=2) - used in most training examples
- **Alternative**: `cosine` (cosine beta schedule)  
- **Learned**: Adaptive noise schedule via `GammaNetwork` (requires VLB loss)

#### Parametrization
- **Method**: Epsilon parameterization (`parametrization='eps'`)
- **Prediction**: Model predicts the noise ε added at each timestep
- **Reconstruction**: Uses predicted noise to estimate clean molecular structure

#### Posterior Mean Calculation
During reverse diffusion sampling, the **posterior mean** is calculated at each timestep to estimate the less noisy state. At `en_diffusion.py:734`:

```python
mu = zt / alpha_t_given_s - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_t
```

**Mathematical Interpretation:**
- **Context**: Computing μ for the posterior distribution p(z_s | z_t) 
- **z_t**: Current noisy molecular state at timestep t
- **ε_t**: Neural network's prediction of noise that was added
- **Formula**: Uses Bayes' theorem applied to the diffusion process
- **Purpose**: Best estimate of cleaner molecular state given current observation and predicted noise

**Physical Meaning:** The posterior mean represents our best guess of a slightly cleaner molecular structure. By iteratively sampling around these means with controlled variance, the process gradually transforms random noise into valid 3D molecular structures while maintaining E(3) equivariance.

#### Sampling Variants
- **Standard Sampling** (`sample`): Sequential reverse diffusion from t=T to t=0
- **Chain Sampling** (`sample_chain`): Same as standard but saves intermediate states for visualization
- **MLFF-Guided Sampling**: Incorporates physics-based force field guidance during the reverse process

#### DPM-Solver++ Integration

**NEW**: The codebase now supports DPM-Solver++ as an alternative sampling method for dramatically faster inference:

**Key Benefits:**
- **Speed**: 50x faster sampling (20 steps vs 1000 steps)
- **Quality**: Maintains or improves sample quality 
- **Full MLFF Compatibility**: ✅ **NEW!** Complete integration with MLFF guidance system
- **Drop-in**: Non-breaking addition with DDPM as default

**Configuration Options:**
- `sampling_method`: 'ddpm' (default) or 'dpm_solver++'
- `dpm_solver_order`: 2 (recommended for guided sampling) or 3 (unconditional)
- `dpm_solver_steps`: Number of sampling steps (typically 10-20)

**Usage Examples:**
```bash
# Fast sampling with DPM-Solver++
python eval_sample.py --model_path outputs/edm_1 --sampling_method dpm_solver++ \
    --dpm_solver_steps 20 --dpm_solver_order 2

# Combine with MLFF guidance for fast physics-guided generation (FULLY SUPPORTED!)
python eval_mlff_guided.py --model_path outputs/edm_1 --sampling_method dpm_solver++ \
    --guidance_scales 0.5 --dpm_solver_steps 20 --dpm_solver_order 2
```

**Implementation Details:**
- Maintains E(3) equivariance throughout sampling process
- Preserves center-of-mass constraints via `remove_mean_with_mask()`
- Compatible with chain visualization (adapts frame count to solver steps)
- Uses data prediction formulation for improved stability

**MLFF + DPM-Solver++ Integration:**
- ✅ **Fully Implemented**: MLFF forces are applied at each DPM-Solver++ step
- **Smart Guidance**: Respects noise threshold (skip guidance at high noise levels)
- **Force Scaling**: Automatically adapts force magnitude based on current noise level
- **Performance**: ~50x faster than DDPM+MLFF while maintaining physics accuracy
- **Automatic Routing**: Simply add `--sampling_method dpm_solver++` to existing MLFF commands

**Advanced MLFF + DPM-Solver++ Usage:**
```bash
# High-quality physics-guided generation with different guidance scales
python eval_mlff_guided.py --model_path outputs/edm_1 \
    --sampling_method dpm_solver++ --dpm_solver_steps 20 --dpm_solver_order 2 \
    --guidance_scales 0.1 0.5 1.0 2.0 --guidance_iterations 2 \
    --noise_threshold 0.8 --force_clip_threshold 15.0

# Fast distributed evaluation with MLFF + DPM-Solver++
python -m torch.distributed.launch --nproc_per_node=4 eval_mlff_guided.py \
    --model_path outputs/edm_1 --use_distributed --n_samples 1000 \
    --sampling_method dpm_solver++ --guidance_scales 0.5
```

### MLFF Integration Pattern
The MLFF guidance system follows this pattern:
1. **Data Conversion**: Convert diffusion tensors to AtomicData format via `diffusion_to_atomic_data()`
2. **Force Computation**: Use pretrained MLFF to compute forces on atomic positions
3. **Guidance Application**: Scale and apply forces during sampling via `apply_mlff_guidance()`
4. **Constraint Maintenance**: Preserve equivariance and center-of-mass properties

### Key Data Structures
- **Diffusion State**: `[batch_size, n_nodes, n_dims + n_features]` where first 3 dims are positions
- **Node Mask**: `[batch_size, n_nodes, 1]` indicating valid atoms
- **Edge Mask**: `[batch_size * n_nodes * n_nodes, 1]` for molecular connectivity
- **AtomicData**: FAIRChem format with pos, atomic_numbers, cell, pbc, etc.

## Important Configuration Notes

### Training Hyperparameters
- **Normalization factors**: `[1,4,10]` for QM9, `[1,8,1]` for conditional models
- **Diffusion steps**: 1000 for production, 500 for faster debugging
- **Noise schedule**: `polynomial_2` generally works best
- **EMA decay**: 0.9999 for stable training

### MLFF Guidance Parameters
- **Guidance scale**: 0.0 (no guidance) to 2.0 (strong guidance)
- **Guidance iterations**: 1 for efficiency, higher for iterative refinement
- **Noise threshold**: 0.8 (skip guidance when noise > threshold, provides ~2-3x speedup)
- **Force clip threshold**: Maximum force magnitude allowed (None = no clipping, 10.0-20.0 recommended)
- **MLFF model**: `uma-s-1p1` is the default pretrained model

### Memory Considerations
- EGNN uses fully connected message passing - memory scales as O(n²)
- Reduce batch size for larger molecules (GEOM dataset)
- Use `--use_distributed` for multi-GPU evaluation to handle large sample counts

## Dependencies and Setup

### Core Requirements
Install from `requirements.txt`: torch, numpy, scipy, wandb, tqdm, imageio

### Optional Dependencies
- **RDKit**: `conda create -c conda-forge -n my-rdkit-env rdkit` (for molecular analysis)
- **FAIRChem**: `pip install fairchem-core` (for MLFF guidance)
- **HuggingFace CLI**: Required for UMA model access authentication

### Dataset Setup
- QM9: Automatically downloaded to `qm9/temp/`
- GEOM: Follow instructions in `data/geom/README.md`

## Testing and Debugging

### Quick Validation
```bash
# Test basic training with minimal epochs
conda activae edm
python main_qm9.py --n_epochs 1 --exp_name debug_test --batch_size 8

# Test MLFF guidance with small samples
python eval_mlff_guided.py --model_path outputs/edm_1 --n_samples 5 --skip_visualization
```

### Common Debug Patterns
- Use `guidance_scale=0.0` to disable MLFF guidance for baseline comparison
- Check `node_mask` and `edge_mask` shapes for dimension mismatches
- Monitor GPU memory usage during EGNN forward passes
- Verify atom decoder consistency between training and evaluation

## Output Structure

Training outputs go to `outputs/{exp_name}/`:
- `args.pickle`: Serialized training arguments
- `generative_model.npy` / `generative_model_ema.npy`: Model checkpoints
- `epoch_{n}/`: Sample molecules at training checkpoints

Evaluation outputs go to `outputs/{model_path}/eval/`:
- `mlff_guided/`: MLFF comparison results with stability metrics
- Molecular XYZ files and 3D visualizations organized by guidance scale