#!/bin/bash
#SBATCH --job-name=guidance_sweep
#SBATCH --output=guidance_sweep_%A_%a.out
#SBATCH --error=guidance_sweep_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:8
#SBATCH --partition=gpu_h200

# Load DSQ module
module load DSQ

# Submit the job array using DSQ
dsq --job-file guidance_sweep_jobs.txt --mem-per-cpu 4G --cpus-per-task 8 --gres gpu:8 --time 48:00:00 --partition gpu_h200 --batch-file guidance_sweep_batch.sh

# Submit the generated batch file
sbatch guidance_sweep_batch.sh