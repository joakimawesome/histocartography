#!/bin/bash
#SBATCH -J nuclei_graphs            # Job name
#SBATCH -o logs/nuclei_graphs_%A_%a.out   # Output file (%A=array job ID, %a=task ID)
#SBATCH -e logs/nuclei_graphs_%A_%a.err   # Error file
#SBATCH -p gh                       # GPU partition (Grace Hopper)
#SBATCH -N 1                        # Number of nodes
#SBATCH -n 1                        # Total tasks
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH -t 04:00:00                 # Wall time (increased for global stitching + deep features)
#SBATCH -A ASC25123
#SBATCH --array=0-239%10            # 240 slides; %10 = max 10 concurrent jobs

# =============================================================================
# TACC Vista - Nuclei Graph Feature Extraction Pipeline (v2 — aligned)
# =============================================================================
# Aligned with original BiomedSciAI/histocartography:
#   - Global pred_map stitching (no tile-boundary artefacts)
#   - Per-node handcrafted features (24-dim) + optional deep CNN features
#   - Full model object loading only (no state_dict fallback)
#
# Usage:
#   1. Edit USER CONFIG section below
#   2. Submit: sbatch slurm_nuclei_graphs.sh
#   3. Monitor: squeue -u $USER
#   4. Check output: ls $OUT_DIR/<run_name>/<slide_id>/
# =============================================================================

# --- USER CONFIG (edit these) ------------------------------------------------
# Input
MANIFEST="/scratch/11090/joakimchi/Pediatric-Brain-Tumor/data/manifests/v1.0.1/manifest.csv"
SLIDE_COL="path"
SLIDES_ROOT="/scratch/11090/joakimchi/Pediatric-Brain-Tumor"

# Output
OUT_DIR="/scratch/11090/joakimchi/Pediatric-Brain-Tumor/data/wsi_processed"

# Models
MODEL_PATH="checkpoints/hovernet_pannuke.pth"              # HoVer-Net (full model .pth)
GNN_MODEL_PATH=""                                          # Optional GNN model (for feat_mode=gnn only)

# Environment
VENV="/scratch/11090/joakimchi/histocart_env"              # Conda/venv path
PROJECT_ROOT="/scratch/11090/joakimchi/histocartography"   # Repo clone on TACC

# Pipeline options
GRAPH_METHOD="knn"                                         # "knn" or "radius"
K_NEIGHBORS=5                                              # k for kNN graph
FEAT_MODE="handcrafted"                                    # "handcrafted" | "deep" | "stats" | "gnn"
FEAT_ARCHITECTURE="resnet50"                               # CNN arch for deep features
FEAT_PATCH_SIZE=72                                         # Patch size for deep features
FEAT_RESIZE_SIZE=""                                        # Resize before CNN (empty = no resize)

# Segmentation options
STITCH_MODE="global"                                       # "global" (TACC, faithful) or "tile" (local)
SEG_DEVICE="cuda"                                          # "cuda" or "cpu"
SEG_BATCH_SIZE=32                                          # HoVer-Net inference batch size
SEG_TILE_SIZE=256                                          # Tile size (HoVer-Net native is 256)
SEG_OVERLAP=0                                              # Tile overlap (only used in tile mode)
SEG_LEVEL=0                                                # WSI pyramid level (0 = full resolution)
SEG_MIN_NUCLEUS_AREA=10                                    # Filter small nuclei

# PyTorch allocator config
PYTORCH_ALLOC_CONF="expandable_segments:True"
# -----------------------------------------------------------------------------

# --- ENVIRONMENT SETUP -------------------------------------------------------
set -eo pipefail

echo "=== Job Info ==="
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "Node: ${HOSTNAME:-unknown}"
echo "Start time: $(date)"
echo ""

# Load TACC modules
module load gcc cuda
module load python3

# Activate environment
if [[ -d "$VENV" ]]; then
    source "$VENV/bin/activate" 2>/dev/null || source activate "$VENV"
fi

# CUDA allocator
if [[ -n "${PYTORCH_ALLOC_CONF:-}" ]]; then
    export PYTORCH_ALLOC_CONF="$PYTORCH_ALLOC_CONF"
    export PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_ALLOC_CONF"
fi

cd "$PROJECT_ROOT" || { echo "ERROR: Cannot cd to $PROJECT_ROOT"; exit 1; }

mkdir -p logs
mkdir -p "$OUT_DIR"

# Sanity check
python -c "import torch; print(f'torch={torch.__version__} cuda={torch.version.cuda} avail={torch.cuda.is_available()} gpus={torch.cuda.device_count()}')" || true

# --- VALIDATE MANIFEST -------------------------------------------------------
if [[ ! -f "$MANIFEST" ]]; then
    echo "ERROR: Manifest not found: $MANIFEST"
    exit 1
fi
echo "Using manifest: $MANIFEST"

NUM_SLIDES=$(tail -n +2 "$MANIFEST" | wc -l)
echo "Total slides in manifest: $NUM_SLIDES"

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    if [[ "$SLURM_ARRAY_TASK_ID" -ge "$NUM_SLIDES" ]]; then
        echo "Task ID $SLURM_ARRAY_TASK_ID >= slides ($NUM_SLIDES). Exiting gracefully."
        exit 0
    fi
else
    echo "WARNING: SLURM_ARRAY_TASK_ID not set. Running in sequential mode."
fi

# --- RUN PIPELINE ------------------------------------------------------------
echo ""
echo "=== Running Pipeline ==="
echo "Manifest: $MANIFEST"
echo "Output: $OUT_DIR"
echo "Stitch mode: $STITCH_MODE"
echo "Feature mode: $FEAT_MODE (arch=$FEAT_ARCHITECTURE)"
echo "Graph: $GRAPH_METHOD (k=$K_NEIGHBORS)"
echo "Segmentation: device=$SEG_DEVICE batch=$SEG_BATCH_SIZE tile=$SEG_TILE_SIZE"
echo ""

# Build command
CMD=(
    python batch_runner.py
    --manifest "$MANIFEST"
    --out_dir "$OUT_DIR"
    --model_path "$MODEL_PATH"
    --slides_root "$SLIDES_ROOT"
    --slide_col "$SLIDE_COL"
    --graph_method "$GRAPH_METHOD"
    --k "$K_NEIGHBORS"
    --feat_mode "$FEAT_MODE"
    --stitch_mode "$STITCH_MODE"
    --skip_errors
)

# Optional args
if [[ -n "$GNN_MODEL_PATH" && -f "$GNN_MODEL_PATH" ]]; then
    CMD+=(--gnn_model_path "$GNN_MODEL_PATH")
fi
if [[ -n "$FEAT_ARCHITECTURE" ]]; then
    CMD+=(--feat_architecture "$FEAT_ARCHITECTURE")
fi
if [[ -n "$FEAT_PATCH_SIZE" ]]; then
    CMD+=(--feat_patch_size "$FEAT_PATCH_SIZE")
fi
if [[ -n "$FEAT_RESIZE_SIZE" ]]; then
    CMD+=(--feat_resize_size "$FEAT_RESIZE_SIZE")
fi

# Segmentation overrides
if [[ -n "${SEG_DEVICE:-}" ]]; then
    CMD+=(--seg_device "$SEG_DEVICE")
fi
if [[ -n "${SEG_BATCH_SIZE:-}" ]]; then
    CMD+=(--seg_batch_size "$SEG_BATCH_SIZE")
fi
if [[ -n "${SEG_TILE_SIZE:-}" ]]; then
    CMD+=(--seg_tile_size "$SEG_TILE_SIZE")
fi
if [[ -n "${SEG_OVERLAP:-}" ]]; then
    CMD+=(--seg_overlap "$SEG_OVERLAP")
fi
if [[ -n "${SEG_LEVEL:-}" ]]; then
    CMD+=(--seg_level "$SEG_LEVEL")
fi
if [[ -n "${SEG_MIN_NUCLEUS_AREA:-}" ]]; then
    CMD+=(--seg_min_nucleus_area "$SEG_MIN_NUCLEUS_AREA")
fi

echo "Command: ${CMD[*]}"
echo ""

# Run
"${CMD[@]}"
EXIT_CODE=$?

# --- CLEANUP -----------------------------------------------------------------
echo ""
echo "=== Job Complete ==="
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"

exit $EXIT_CODE
