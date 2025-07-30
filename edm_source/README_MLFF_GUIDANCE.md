# MLFF-Guided Diffusion for Molecular Generation

This enhancement adds Machine Learning Force Field (MLFF) guidance to the equivariant diffusion model, enabling physics-informed molecular generation with improved structural quality and stability.

## Overview

The MLFF-guided diffusion model combines the generative capabilities of diffusion models with the physical constraints from pretrained force fields. During the sampling process, the model uses forces from a pretrained MLFF to guide the coordinate generation, resulting in more realistic and stable molecular structures.

### Key Features

- **Physics-informed sampling**: Uses MLFF forces to guide coordinate generation
- **Equivariant architecture**: Maintains the E(3) equivariance of the original model
- **Flexible guidance**: Adjustable guidance scale for different applications
- **Compatibility**: Works with existing trained diffusion models
- **Multiple datasets**: Supports QM9 and GEOM datasets

## Installation

### Prerequisites

1. Install the base diffusion model dependencies
2. Install FAIRChem for MLFF predictor:
   ```bash
   pip install fairchem-core
   ```

### Authentication for UMA Models

To use the UMA MLFF models, you need to:

1. Get a HuggingFace account and request access to the UMA model
2. Create a HuggingFace token with appropriate permissions
3. Login using `huggingface-cli login` or set the `HF_TOKEN` environment variable

## Quick Start

To evaluate your pretrained diffusion model with MLFF guidance:

```bash
# Quick start (automatically finds pretrained models)
python quick_start_mlff.py

# Or run evaluation directly
python eval_mlff_guided.py --model_path /path/to/your/model

# Example with specific settings
python eval_mlff_guided.py \
    --model_path outputs/edm_1 \
    --n_samples 20 \
    --mlff_model uma-s-1p1 \
    --guidance_scales 0.5 1.0 2.0
```

## Usage

### Basic Usage

```python
import torch
from fairchem.core import pretrained_mlip
from configs.datasets_config import get_dataset_info
from qm9.models import get_model
from mlff_guided_diffusion import enhanced_sampling_with_mlff

# Load dataset info
dataset_info = get_dataset_info('qm9', remove_h=False)

# Load pretrained diffusion model
model, nodes_dist, prop_dist = get_model(args, device, dataset_info, None)
model.load_state_dict(torch.load('path/to/model.pth'))

# Load MLFF predictor
predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cuda")

# Create sampling inputs
n_samples, n_nodes = 10, 19
node_mask = torch.ones(n_samples, n_nodes, 1, device=device)
edge_mask = torch.ones(n_samples * n_nodes * n_nodes, 1, device=device)
context = None

# Sample with MLFF guidance
x, h = enhanced_sampling_with_mlff(
    model, predictor, n_samples, n_nodes, node_mask, edge_mask,
    context, dataset_info, guidance_scale=1.0
)
```

### Advanced Usage

```python
from mlff_guided_diffusion import MLFFGuidedDiffusion, create_mlff_guided_model

# Create guided model from existing model
guided_model = create_mlff_guided_model(
    original_model=model,
    mlff_predictor=predictor,
    guidance_scale=1.0,
    dataset_info=dataset_info
)

# Custom sampling with different guidance scales
for guidance_scale in [0.0, 0.5, 1.0, 2.0]:
    guided_model.guidance_scale = guidance_scale
    x, h = guided_model.sample_with_mlff_guidance(
        n_samples, n_nodes, node_mask, edge_mask, context, dataset_info
    )
```

### Command Line Evaluation

Run the full evaluation script with your pretrained model:

```bash
python eval_mlff_guided.py \
    --model_path outputs/edm_1 \
    --n_samples 20 \
    --n_nodes 19 \
    --mlff_model uma-s-1p1
```

This will:
- Load your pretrained diffusion model
- Sample molecules with and without MLFF guidance
- Compare stability and quality metrics
- Generate visualization chains
- Save results and 3D visualizations

### Command Line Arguments

- `--model_path`: Path to trained diffusion model
- `--n_samples`: Number of samples for comparison (default: 20)
- `--n_nodes`: Number of nodes for chain visualization (default: 19)
- `--n_tries`: Number of tries for stable molecule chain (default: 10)
- `--mlff_model`: MLFF model name (default: uma-s-1p1)
- `--task_name`: Task name for MLFF predictor (default: omol)
- `--guidance_scales`: List of guidance scales to test (default: [0.5, 1.0, 2.0])
- `--skip_comparison`: Skip comparison sampling
- `--skip_chain`: Skip chain visualization
- `--skip_visualization`: Skip 3D visualization generation

### Evaluation Output

The evaluation script generates:

```
outputs/your_model/eval/mlff_guided/
├── baseline/                    # Baseline sampling results
│   ├── molecule_baseline_*.xyz  # XYZ files
│   └── vis/                     # 3D visualizations
├── guided_scale_0.5/           # Results with guidance scale 0.5
├── guided_scale_1.0/           # Results with guidance scale 1.0 
├── guided_scale_2.0/           # Results with guidance scale 2.0
├── chain_baseline/             # Baseline diffusion chain
│   ├── chain_baseline_*.xyz
│   └── vis/
└── chain_guided/               # MLFF-guided diffusion chain
    ├── chain_guided_*.xyz
    └── vis/
```

Each run provides:
- **Stability metrics**: Percentage of stable molecules
- **Atom stability ratio**: Average stability per atom
- **XYZ files**: For external analysis tools
- **3D visualizations**: HTML files for interactive viewing

## Implementation Details

### Data Conversion

The system converts diffusion model intermediate states to AtomicData format required by the MLFF predictor:

```python
def diffusion_to_atomic_data(self, z, node_mask, dataset_info):
    """
    Convert diffusion data to AtomicData format.
    
    Args:
        z: [batch_size, n_nodes, n_dims + n_features]
        node_mask: [batch_size, n_nodes, 1]
        dataset_info: Dataset information with atom decoders
    
    Returns:
        List of AtomicData objects
    """
```

### Force Guidance

The MLFF forces are applied during the sampling process:

```python
def apply_mlff_guidance(self, z, node_mask, dataset_info, sigma):
    """
    Apply MLFF force guidance to coordinates.
    
    Forces are scaled by guidance_scale and current noise level.
    Only coordinates are modified, not atom types.
    """
```

### Sampling Enhancement

The enhanced sampling process includes:

1. **Force computation**: Get forces from MLFF predictor for current state
2. **Guidance application**: Apply scaled forces to coordinate sampling
3. **Constraint maintenance**: Ensure equivariance and mean-zero properties

## Required Data Keys

The system ensures all required keys for AtomicData are present:

```python
_REQUIRED_KEYS = [
    "pos",           # Atomic positions
    "atomic_numbers", # Atomic numbers
    "cell",          # Unit cell
    "pbc",           # Periodic boundary conditions
    "natoms",        # Number of atoms
    "edge_index",    # Edge connectivity
    "cell_offsets",  # Cell offsets for edges
    "nedges",        # Number of edges
    "charge",        # Total charge
    "spin",          # Spin multiplicity
    "fixed",         # Fixed atoms
    "tags",          # Atom tags
]
```

## Performance Considerations

- **Memory usage**: MLFF predictor requires additional GPU memory
- **Computational cost**: Force computation adds overhead to sampling
- **Batch processing**: Efficient batching for MLFF predictions
- **Error handling**: Graceful fallback when MLFF prediction fails

## Troubleshooting

### Common Issues

1. **MLFF model access**: Ensure proper HuggingFace authentication
2. **Memory errors**: Reduce batch size or use CPU for MLFF predictor
3. **Dimension mismatches**: Check dataset info and atom decoder consistency
4. **Import errors**: Ensure all dependencies are installed

### Debug Tips

- Use `guidance_scale=0.0` to disable guidance for debugging
- Check atomic data conversion with small test cases
- Monitor GPU memory usage during sampling
- Use try-catch blocks around MLFF predictions

## Examples

### QM9 Dataset

```python
# Load QM9 dataset info
dataset_info = get_dataset_info('qm9', remove_h=False)

# Sample small molecules
x, h = enhanced_sampling_with_mlff(
    model, predictor, n_samples=5, n_nodes=19,
    node_mask=node_mask, edge_mask=edge_mask,
    context=None, dataset_info=dataset_info,
    guidance_scale=1.0
)
```

### GEOM Dataset

```python
# Load GEOM dataset info
dataset_info = get_dataset_info('geom', remove_h=False)

# Sample larger molecules
x, h = enhanced_sampling_with_mlff(
    model, predictor, n_samples=3, n_nodes=44,
    node_mask=node_mask, edge_mask=edge_mask,
    context=None, dataset_info=dataset_info,
    guidance_scale=0.5
)
```

## Citation

If you use this code, please cite the original E(3) Diffusion paper and the UMA paper:

```bibtex
@article{hoogeboom2022equivariant,
  title={Equivariant Diffusion for Molecule Generation in 3D},
  author={Hoogeboom, Emiel and Satorras, V{\'\i}ctor Garcia and Vignac, Cl{\'e}ment and Welling, Max},
  journal={arXiv preprint arXiv:2203.17003},
  year={2022}
}

@article{uma2024,
  title={UMA: A Universal Model for Atoms},
  author={...},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This enhancement maintains the same license as the original E(3) Diffusion codebase. 