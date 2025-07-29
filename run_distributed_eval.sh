#!/bin/bash

# Distributed MLFF-Guided Diffusion Evaluation Script
# This script simplifies running multi-GPU evaluation

# Default values
MODEL_PATH="outputs/edm_1"
N_GPUS=4
N_SAMPLES=1000
MLFF_MODEL="uma-s-1p1"
WANDB_PROJECT="mlff-guided-diffusion"
USE_WANDB=false
GUIDANCE_ITERATIONS=1

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --n_gpus)
            N_GPUS="$2"
            shift 2
            ;;
        --n_samples)
            N_SAMPLES="$2"
            shift 2
            ;;
        --mlff_model)
            MLFF_MODEL="$2"
            shift 2
            ;;
        --wandb_project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --use_wandb)
            USE_WANDB=true
            shift
            ;;
        --guidance_iterations)
            GUIDANCE_ITERATIONS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model_path PATH         Path to trained diffusion model (default: outputs/edm_1)"
            echo "  --n_gpus N               Number of GPUs to use (default: 4)"
            echo "  --n_samples N            Number of samples to generate (default: 1000)"
            echo "  --mlff_model MODEL       MLFF model name (default: uma-s-1p1)"
            echo "  --wandb_project PROJECT  Wandb project name (default: mlff-guided-diffusion)"
            echo "  --use_wandb              Enable Weights & Biases logging"
            echo "  --guidance_iterations N  Number of guidance iterations (default: 1)"
            echo "  -h, --help               Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Basic distributed evaluation with 4 GPUs"
            echo "  $0 --model_path outputs/my_model --n_gpus 4"
            echo ""
            echo "  # With wandb logging and more samples"
            echo "  $0 --model_path outputs/my_model --n_gpus 8 --n_samples 2000 --use_wandb"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if model path exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path '$MODEL_PATH' does not exist!"
    exit 1
fi

# Check if we have the required files
if [ ! -f "$MODEL_PATH/args.pickle" ]; then
    echo "Error: args.pickle not found in model path '$MODEL_PATH'"
    exit 1
fi

# Check for CUDA availability
if ! nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found. CUDA may not be available."
    exit 1
fi

# Check number of available GPUs
AVAILABLE_GPUS=$(nvidia-smi --list-gpus | wc -l)
if [ "$N_GPUS" -gt "$AVAILABLE_GPUS" ]; then
    echo "Warning: Requested $N_GPUS GPUs but only $AVAILABLE_GPUS available."
    echo "Setting N_GPUS to $AVAILABLE_GPUS"
    N_GPUS=$AVAILABLE_GPUS
fi

echo "Starting distributed MLFF-guided diffusion evaluation..."
echo "========================================================"
echo "Configuration:"
echo "  Model path: $MODEL_PATH"
echo "  Number of GPUs: $N_GPUS"
echo "  Number of samples: $N_SAMPLES"
echo "  MLFF model: $MLFF_MODEL"
echo "  Guidance iterations: $GUIDANCE_ITERATIONS"
echo "  Wandb logging: $USE_WANDB"
if [ "$USE_WANDB" = true ]; then
    echo "  Wandb project: $WANDB_PROJECT"
fi
echo "========================================================"

# Build the command
CMD="python -m torch.distributed.launch"
CMD="$CMD --nproc_per_node=$N_GPUS"
CMD="$CMD --master_port=29500"
CMD="$CMD e3_diffusion_for_molecules-main/eval_mlff_guided.py"
CMD="$CMD --model_path $MODEL_PATH"
CMD="$CMD --use_distributed"
CMD="$CMD --n_samples $N_SAMPLES"
CMD="$CMD --mlff_model $MLFF_MODEL"
CMD="$CMD --guidance_iterations $GUIDANCE_ITERATIONS"

if [ "$USE_WANDB" = true ]; then
    CMD="$CMD --use_wandb --wandb_project $WANDB_PROJECT"
fi

echo "Running command:"
echo "$CMD"
echo ""

# Export CUDA_VISIBLE_DEVICES to ensure proper GPU usage
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((N_GPUS-1)))

# Run the command
eval $CMD

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================"
    echo "✓ Distributed evaluation completed successfully!"
    echo "Results saved to: $MODEL_PATH/eval/mlff_guided/"
    echo "========================================================"
else
    echo ""
    echo "========================================================"
    echo "✗ Distributed evaluation failed!"
    echo "========================================================"
    exit 1
fi 