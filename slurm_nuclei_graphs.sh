#!/bin/bash
#SBATCH -J nuclei_graphs            # Job name
#SBATCH -o logs/nuclei_graphs_%A_%a.out   # Output file (%A=array job ID, %a=task ID)
#SBATCH -e logs/nuclei_graphs_%A_%a.err   # Error file
#SBATCH -p gh                       # GPU partition (Grace Hopper)
#SBATCH -N 1                        # Number of nodes (required)
#SBATCH -n 1                        # Total tasks
#SBATCH -t 02:00:00                 # Wall time (max 48:00:00)
#SBATCH -A ASC25123
#SBATCH --array=0-31%10             # Array range (adjust based on slide count), %10 = max concurrent

# =============================================================================
# TACC Vista - Nuclei Graph Feature Extraction Pipeline
# =============================================================================
# Usage:
#   1. Edit USER CONFIG section below
#   2. Submit: sbatch slurm_nuclei_graphs.sh
#   3. Monitor: squeue -u $USER
# =============================================================================

# --- USER CONFIG (edit these) ------------------------------------------------
# Input
MANIFEST="scratch/11090/joakimchi/Pediatric-Brain-Tumor/data/manifests/v1.0.1/manifest.csv"                          # Path to manifest CSV (REQUIRED)
SLIDE_COL="path"                                     # Column name containing slide paths

# Output
OUT_DIR="scratch/11090/joakimchi/Pediatric-Brain-Tumor/data/wsi_preprocessed"                            # Output directory

# Models
MODEL_PATH="checkpoints/hovernet_pannuke.pth"              # HoVerNet model checkpoint
GNN_MODEL_PATH=""                                          # Optional: GNN model (leave empty for stats mode)

# Environment
VENV="/scratch/11090/joakimchi/histocart_env"         # Conda/venv environment path
PROJECT_ROOT="/scratch/11090/joakimchi/histocartography"       # Histocartography repo path

# Pipeline options
GRAPH_METHOD="knn"                                         # "knn" or "radius"
K_NEIGHBORS=5                                              # k for kNN graph
FEAT_MODE="stats"                                          # "stats" or "gnn"
# -----------------------------------------------------------------------------

# --- ENVIRONMENT SETUP -------------------------------------------------------
set -eo pipefail

echo "=== Job Info ==="
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "Node: ${HOSTNAME:-unknown}"
echo "Start time: $(date)"
echo ""

# Load TACC modules (Vista-specific)
# Note: Vista inherits environment at submission time.
module load gcc cuda
module load python3

# Activate environment
if [[ -d "$VENV" ]]; then
    source "$VENV/bin/activate" 2>/dev/null || source activate "$VENV"
fi

# Change to repo directory
cd "$PROJECT_ROOT" || { echo "ERROR: Cannot cd to $PROJECT_ROOT"; exit 1; }

# Create necessary directories
mkdir -p logs
mkdir -p "$OUT_DIR"

# --- VALIDATE MANIFEST -------------------------------------------------------
if [[ ! -f "$MANIFEST" ]]; then
    echo "ERROR: Manifest not found: $MANIFEST"
    exit 1
fi
echo "Using manifest: $MANIFEST"

# --- CHECK TASK VALIDITY -----------------------------------------------------
NUM_SLIDES=$(tail -n +2 "$MANIFEST" | wc -l)

if [[ "$SLURM_ARRAY_TASK_ID" -ge "$NUM_SLIDES" ]]; then
    echo "Task ID $SLURM_ARRAY_TASK_ID >= number of slides ($NUM_SLIDES). Exiting."
    exit 0
fi

# --- RUN PIPELINE ------------------------------------------------------------
echo ""
echo "=== Running Pipeline ==="
echo "Manifest: $MANIFEST"
echo "Output: $OUT_DIR"
echo "Graph method: $GRAPH_METHOD (k=$K_NEIGHBORS)"
echo "Feature mode: $FEAT_MODE"
echo ""

# Build command
CMD="python batch_runner.py \
    --manifest \"$MANIFEST\" \
    --out_dir \"$OUT_DIR\" \
    --model_path \"$MODEL_PATH\" \
    --slide_col "$SLIDE_COL" \
    --graph_method $GRAPH_METHOD \
    --k $K_NEIGHBORS \
    --feat_mode $FEAT_MODE \
    --skip_errors"

# Add GNN model if specified
if [[ -n "$GNN_MODEL_PATH" && -f "$GNN_MODEL_PATH" ]]; then
    CMD="$CMD --gnn_model_path \"$GNN_MODEL_PATH\""
fi

echo "Command: $CMD"
echo ""

# Run
eval $CMD
EXIT_CODE=$?

# --- CLEANUP -----------------------------------------------------------------
echo ""
echo "=== Job Complete ==="
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"

exit $EXIT_CODE
