#!/bin/bash
#SBATCH --job-name=histo_pipeline
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH --array=0-99%10         # Adjust range based on manifest size, %10 limits concurrent jobs
#SBATCH --time=04:00:00        # Adjust time limit
#SBATCH --mem=16G              # Adjust memory
#SBATCH --cpus-per-task=4      # Adjust CPUs
#SBATCH --gres=gpu:1           # Request GPU if needed

# Load environment
# module load python/3.8
# source activate myenv

# Define paths
MANIFEST="data/manifest.csv"
OUT_DIR="results"
MODEL_PATH="checkpoints/hovernet_model.pth"
GNN_MODEL_PATH="checkpoints/gnn_model.pth"

# Create logs directory
mkdir -p logs

# Run batch runner
# SLURM_ARRAY_TASK_ID is automatically set by SLURM and picked up by the script
python batch_runner.py \
    --manifest "$MANIFEST" \
    --out_dir "$OUT_DIR" \
    --model_path "$MODEL_PATH" \
    --gnn_model_path "$GNN_MODEL_PATH" \
    --slide_col "slide_path" \
    --skip_errors

# To run a specific subset or single slide without array:
# python batch_runner.py ... --slurm_array_idx 0
