#!/bin/bash
#SBATCH -J nuclei_graphs            # Job name
#SBATCH -o logs/nuclei_graphs_%A_%a.out   # Output file (%A=array job ID, %a=task ID)
#SBATCH -e logs/nuclei_graphs_%A_%a.err   # Error file
#SBATCH -p gh-dev                       # GPU partition (Grace Hopper)
#SBATCH -N 1                        # Number of nodes
#SBATCH -n 1                        # Total tasks
#SBATCH -t 02:30:00                 # Wall time (global stitching + hybrid features)
#SBATCH -A ASC25123
#SBATCH --array=0-239%20            # 240 slides; %20 = max 20 concurrent jobs

# =============================================================================
# TACC Vista - Nuclei Graph Feature Extraction Pipeline (v2 — aligned)
# =============================================================================
# Aligned with original BiomedSciAI/histocartography:
#   - Global pred_map stitching (no tile-boundary artefacts)
#   - Per-node handcrafted features (24-dim) + optional deep CNN features
#   - State_dict standard + legacy 'desc' checkpoint fallback
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
MODEL_PATH="checkpoints/hovernet_pannuke.pth"              # HoVer-Net checkpoint (.pth)
GNN_MODEL_PATH=""                                          # Optional GNN model (for feat_mode=gnn only)

# Environment
VENV="/scratch/11090/joakimchi/histocart_env"              # Conda/venv path
PROJECT_ROOT="/scratch/11090/joakimchi/histocartography"   # Repo clone on TACC

# Pipeline options
GRAPH_METHOD="knn"                                         # "knn" or "radius"
K_NEIGHBORS=5                                              # k for kNN graph
FEAT_MODE="hybrid"                                         # "handcrafted" | "deep" | "hybrid" | "stats" | "gnn"
FEAT_ARCHITECTURE="resnet50"                               # CNN arch for deep features
FEAT_PATCH_SIZE=72                                         # Patch size for deep features
FEAT_RESIZE_SIZE=""                                        # Resize before CNN (empty = no resize)
FEAT_BATCH_SIZE="auto"                                     # Auto-tuned if set to "auto"
FEAT_NUM_WORKERS=16                                         # DataLoader workers for deep/hybrid
FEAT_PIN_MEMORY="true"                                     # Pin memory for deep/hybrid

# Segmentation options
STITCH_MODE="global"                                       # "global" (TACC, faithful) or "tile" (local)
SEG_DEVICE="cuda"                                          # "cuda" or "cpu"
SEG_BATCH_SIZE="auto"                                      # Auto-tuned if set to "auto"
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

# Threading hints
if [[ -n "${SLURM_CPUS_PER_TASK:-}" ]]; then
    export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
    export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
    export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
fi

cd "$PROJECT_ROOT" || { echo "ERROR: Cannot cd to $PROJECT_ROOT"; exit 1; }

mkdir -p logs
mkdir -p "$OUT_DIR"

# Sanity check
python -c "import torch; print(f'torch={torch.__version__} cuda={torch.version.cuda} avail={torch.cuda.is_available()} gpus={torch.cuda.device_count()}')" || true

# Auto-tune batch sizes based on GPU memory (if set to "auto")
GPU_MEM_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1)"
if [[ "${SEG_BATCH_SIZE}" == "auto" || -z "${SEG_BATCH_SIZE}" ]]; then
    if [[ -n "${GPU_MEM_MIB}" ]]; then
        if (( GPU_MEM_MIB >= 80000 )); then
            SEG_BATCH_SIZE=64
        elif (( GPU_MEM_MIB >= 40000 )); then
            SEG_BATCH_SIZE=32
        else
            SEG_BATCH_SIZE=16
        fi
    else
        SEG_BATCH_SIZE=16
    fi
fi
if [[ "${FEAT_BATCH_SIZE}" == "auto" || -z "${FEAT_BATCH_SIZE}" ]]; then
    if [[ -n "${GPU_MEM_MIB}" ]]; then
        if (( GPU_MEM_MIB >= 80000 )); then
            FEAT_BATCH_SIZE=128
        elif (( GPU_MEM_MIB >= 40000 )); then
            FEAT_BATCH_SIZE=64
        else
            FEAT_BATCH_SIZE=32
        fi
    else
        FEAT_BATCH_SIZE=32
    fi
fi

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
if [[ -n "${FEAT_BATCH_SIZE:-}" ]]; then
    CMD+=(--feat_batch_size "$FEAT_BATCH_SIZE")
fi
if [[ -n "${FEAT_NUM_WORKERS:-}" ]]; then
    CMD+=(--feat_num_workers "$FEAT_NUM_WORKERS")
fi
if [[ -n "${FEAT_PIN_MEMORY:-}" ]]; then
    if [[ "$FEAT_PIN_MEMORY" == "true" ]]; then
        CMD+=(--feat_pin_memory)
    else
        CMD+=(--no-feat_pin_memory)
    fi
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
