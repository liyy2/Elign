#!/usr/bin/env python3
"""
Quick start script for MLFF-guided diffusion evaluation.

This script demonstrates how to evaluate a pretrained diffusion model 
with MLFF guidance in just a few lines of code.
"""

import torch
import os
from pathlib import Path

def main():
    print("MLFF-Guided Diffusion Quick Start")
    print("=" * 40)
    
    # Check for pretrained model
    model_paths = [
        "outputs/edm_1",
        "outputs/qm9_default", 
        "outputs/geom_default"
    ]
    
    found_model = None
    for path in model_paths:
        if os.path.exists(path) and os.path.exists(os.path.join(path, "args.pickle")):
            found_model = path
            break
    
    if found_model:
        print(f"✓ Found pretrained model at: {found_model}")
        
        # Run evaluation
        print("\nRunning MLFF-guided evaluation...")
        print("This will compare baseline vs. MLFF-guided sampling")
        
        # Import and run evaluation
        try:
            import subprocess
            import sys
            
            cmd = [
                sys.executable, "eval_mlff_guided.py",
                "--model_path", found_model,
                "--n_samples", "10",  # Small number for quick demo
                "--n_nodes", "19"
            ]
            
            print(f"Command: {' '.join(cmd)}")
            print("\nStarting evaluation...")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✓ Evaluation completed successfully!")
                print(f"\nResults saved to: {found_model}/eval/mlff_guided/")
                print("\nTo view results:")
                print(f"  - Check XYZ files in {found_model}/eval/mlff_guided/*/")
                print(f"  - Open HTML visualizations in {found_model}/eval/mlff_guided/*/vis/")
            else:
                print("✗ Evaluation failed:")
                print(result.stderr)
                
        except Exception as e:
            print(f"✗ Error running evaluation: {e}")
            print("\nYou can run manually with:")
            print(f"python eval_mlff_guided.py --model_path {found_model}")
    
    else:
        print("✗ No pretrained model found")
        print("\nTo use this script, you need a trained diffusion model with:")
        print("  - args.pickle (model configuration)")
        print("  - generative_model.npy or generative_model_ema.npy (weights)")
        print("\nExample model structure:")
        print("  outputs/your_model/")
        print("  ├── args.pickle")
        print("  ├── generative_model.npy")
        print("  └── (other training files)")
        print("\nTrain a model first or specify the correct path:")
        print("  python eval_mlff_guided.py --model_path /path/to/your/model")
    
    print("\n" + "=" * 40)
    print("Quick Start Complete")


if __name__ == "__main__":
    main() 