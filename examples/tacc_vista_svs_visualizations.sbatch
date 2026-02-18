#!/bin/bash
#SBATCH -J hc_svs_viz
#SBATCH -o %x.%A_%a.out
#SBATCH -e %x.%A_%a.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH -A ASC25123
#SBATCH --array=0-23%12

set -euo pipefail

echo "[INFO] Job started at $(date)"
echo "[INFO] Host: $(hostname)"
echo "[INFO] Working dir: ${SLURM_SUBMIT_DIR}"

# Assume this script is submitted from the repository root.
REPO_ROOT="${SLURM_SUBMIT_DIR}"

cd "${REPO_ROOT}"
REPO_ROOT="$(pwd)"

# Vista docs recommend managing software via modules in job scripts.
module reset
module load gcc
module load cuda
module load python3

# Activate your Python environment.
# Set HC_ENV before submission, or edit this path.
HC_ENV_DEFAULT="$SCRATCH/histocart_env"
HC_ENV="${HC_ENV:-$HC_ENV_DEFAULT}"

if [ ! -f "${HC_ENV}/bin/activate" ]; then
  echo "[ERROR] Python environment not found at: ${HC_ENV}"
  echo "[ERROR] Set HC_ENV=/path/to/venv before sbatch or edit this script."
  exit 2
fi

source "${HC_ENV}/bin/activate"

PYTHON_BIN="$(command -v python)"

# Runtime tuning.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

# Ensure local package imports are resolvable after modules/venv are active.
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/histocartography:${PYTHONPATH:-}"

echo "[INFO] Python: ${PYTHON_BIN}"
echo "[INFO] Repo root: ${REPO_ROOT}"
echo "[INFO] PYTHONPATH: ${PYTHONPATH}"

# Preflight import check for fast failure with clear diagnostics.
python - << 'PY'
import os
import sys
print("[INFO] CWD:", os.getcwd())
print("[INFO] sys.executable:", sys.executable)
print("[INFO] sys.path[0:5]:", sys.path[:5])
import histocartography
print("[INFO] histocartography:", histocartography.__file__)
PY

SHARD_INDEX="${SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"
NUM_SHARDS="${NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}"

# Must match your embeddings output layout from svs_graph_embeddings.py.
INPUT_DIR="${INPUT_DIR:-$SCRATCH/Pediatric-Brain-Tumor/data/wsi_raw}"
EMBEDDINGS_ROOT="${EMBEDDINGS_ROOT:-$SCRATCH/Pediatric-Brain-Tumor/data/wsi_processed}"
EMBEDDINGS_SUBDIR="${EMBEDDINGS_SUBDIR:-histocartography_graph_embeddings_vista}"
VISUALIZATION_SUBDIR="${VISUALIZATION_SUBDIR:-histocartography_graph_visualizations_vista}"

# Rendering controls
MAX_TILES_PER_SLIDE="${MAX_TILES_PER_SLIDE:-0}"
TILE_STEP="${TILE_STEP:-1}"
NODE_STYLE="${NODE_STYLE:-outline}"
NODE_COLOR="${NODE_COLOR:-yellow}"
NODE_RADIUS="${NODE_RADIUS:-5}"
EDGE_COLOR="${EDGE_COLOR:-blue}"
EDGE_THICKNESS="${EDGE_THICKNESS:-2}"

echo "[INFO] Shard: ${SHARD_INDEX}/${NUM_SHARDS}"
echo "[INFO] Embeddings dir: ${EMBEDDINGS_ROOT}/${EMBEDDINGS_SUBDIR}"
echo "[INFO] Visualization dir: ${EMBEDDINGS_ROOT}/${VISUALIZATION_SUBDIR}"

echo "[INFO] Rendering overlays from saved graph tiles..."
python "${REPO_ROOT}/examples/svs_graph_visualizations.py" \
  --input-dir "${INPUT_DIR}" \
  --embeddings-root "${EMBEDDINGS_ROOT}" \
  --embeddings-subdir "${EMBEDDINGS_SUBDIR}" \
  --visualization-subdir "${VISUALIZATION_SUBDIR}" \
  --shard-index "${SHARD_INDEX}" \
  --num-shards "${NUM_SHARDS}" \
  --tile-size 2048 \
  --level 0 \
  --max-tiles-per-slide "${MAX_TILES_PER_SLIDE}" \
  --tile-step "${TILE_STEP}" \
  --node-style "${NODE_STYLE}" \
  --node-color "${NODE_COLOR}" \
  --node-radius "${NODE_RADIUS}" \
  --edge-color "${EDGE_COLOR}" \
  --edge-thickness "${EDGE_THICKNESS}" \
  --continue-on-error

echo "[INFO] Job finished at $(date)"
