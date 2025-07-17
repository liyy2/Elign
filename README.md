# E3 Diffusion for Molecules with MLFF Guidance

This repository contains an E(3) Equivariant Diffusion Model for Molecule Generation in 3D, enhanced with Machine Learning Force Field (MLFF) guidance for physics-informed molecular generation.

## Prerequisites

### 1. Access to UMA via Hugging Face

First, you need to have access to the UMA (Unified Molecular Assay) models via Hugging Face:

1. **Create a Hugging Face account** at [huggingface.co](https://huggingface.co)
2. **Request access** to the UMA model repository
3. **Generate a token** with appropriate permissions from your HF settings
4. **Login** using one of the following methods:
   ```bash
   # Option 1: Interactive login
   huggingface-cli login
   
   # Option 2: Set environment variable
   export HF_TOKEN=your_token_here
   ```

### 2. Install FAIRChem Package

Install the package for MLFF predictor functionality:
```bash
# For rdkit environment (recommended)
conda create -c conda-forge -n molecular-diffusion rdkit
conda activate molecular-diffusion

# Install other requirements
pip install -r e3_diffusion_for_molecules-main/requirements.txt
```

```bash
cd fairchem_repo
pip install -e packages/fairchem-core
```



## Training a Diffusion Model

You can train diffusion models on different molecular datasets:

### QM9 Dataset
```bash
cd e3_diffusion_for_molecules-main
python main_qm9.py \
    --n_epochs 3000 \
    --exp_name edm_qm9 \
    --n_stability_samples 1000 \
    --diffusion_noise_schedule polynomial_2 \
    --diffusion_noise_precision 1e-5 \
    --diffusion_steps 1000 \
    --diffusion_loss_type l2 \
    --batch_size 64 \
    --nf 256 \
    --n_layers 9 \
    --lr 1e-4 \
    --normalize_factors [1,4,10] \
    --test_epochs 20 \
    --ema_decay 0.9999
```

### GEOM-Drugs Dataset
```bash
python main_geom_drugs.py \
    --n_epochs 3000 \
    --exp_name edm_geom_drugs \
    --n_stability_samples 500 \
    --diffusion_noise_schedule polynomial_2 \
    --diffusion_steps 1000 \
    --diffusion_noise_precision 1e-5 \
    --diffusion_loss_type l2 \
    --batch_size 64 \
    --nf 256 \
    --n_layers 4 \
    --lr 1e-4 \
    --normalize_factors [1,4,10] \
    --test_epochs 1 \
    --ema_decay 0.9999 \
    --normalization_factor 1 \
    --model egnn_dynamics \
    --visualize_every_batch 10000
```

### Conditional Generation
Train a conditional diffusion model for property-specific generation:
```bash
python main_qm9.py \
    --exp_name exp_cond_alpha \
    --model egnn_dynamics \
    --lr 1e-4 \
    --nf 192 \
    --n_layers 9 \
    --save_model True \
    --diffusion_steps 1000 \
    --sin_embedding False \
    --n_epochs 3000 \
    --n_stability_samples 500 \
    --diffusion_noise_schedule polynomial_2 \
    --diffusion_noise_precision 1e-5 \
    --dequantization deterministic \
    --include_charges False \
    --diffusion_loss_type l2 \
    --batch_size 64 \
    --normalize_factors [1,8,1] \
    --conditioning alpha \
    --dataset qm9_second_half
```

## Running Guidance with MLFF Experiments

### Quick Start

### Manual MLFF Guidance Evaluation
Run physics-informed molecular generation with MLFF guidance:

```bash
# Basic usage
python eval_mlff_guided.py --model_path outputs/edm_qm9

# With custom parameters
python eval_mlff_guided.py \
    --model_path outputs/edm_qm9 \
    --n_samples 100 \
    --mlff_model uma-s-1p1 \
    --guidance_scale 1.0 \
    --guidance_steps 10
```


## Key Features

- **Physics-informed sampling**: Uses MLFF forces to guide coordinate generation
- **E(3) equivariant architecture**: Maintains rotational and translational equivariance
- **Flexible guidance**: Adjustable guidance scale for different applications
- **Multiple datasets**: Supports QM9 and GEOM datasets
- **Conditional generation**: Generate molecules with specific properties


## Project Structure

```
├── e3_diffusion_for_molecules-main/    # Main project directory
│   ├── equivariant_diffusion/          # Core diffusion model code
│   ├── egnn/                           # EGNN implementation
│   ├── configs/                        # Configuration files
│   ├── data/                           # Dataset utilities
│   ├── qm9/                            # QM9-specific code
│   ├── main_qm9.py                     # QM9 training script
│   ├── main_geom_drugs.py              # GEOM training script
│   ├── eval_mlff_guided.py             # MLFF guidance evaluation
│   ├── mlff_guided_diffusion.py        # MLFF-guided training
│   ├── quick_start_mlff.py             # Quick start script
│   └── README_MLFF_GUIDANCE.md         # Detailed MLFF guidance docs
├── fairchem/                           # FAIRChem integration
├── uma_notebook.ipynb                  # UMA example notebook
└── README.md                           # This file
```


