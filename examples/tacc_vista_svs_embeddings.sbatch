#!/bin/bash
#SBATCH -J hc_svs_embed
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A ASC25123

set -euo pipefail

echo "[INFO] Job started at $(date)"
echo "[INFO] Host: $(hostname)"
echo "[INFO] Working dir: ${SLURM_SUBMIT_DIR}"

# Assume this script is submitted from the repository root.
REPO_ROOT="${SLURM_SUBMIT_DIR}"

cd "${REPO_ROOT}"
REPO_ROOT="$(pwd)"
mkdir -p logs

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

# Ensure local package imports (histocartography/) are resolvable after modules/venv are active.
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

# Tunable runtime knobs (override at submit time: VAR=value sbatch ...)
NUM_WORKERS="${NUM_WORKERS:-8}"
NUCLEI_BATCH="${NUCLEI_BATCH:-16}"
CELL_BATCH="${CELL_BATCH:-64}"
PATCH_BATCH="${PATCH_BATCH:-128}"
MAX_TILES_PER_SLIDE="${MAX_TILES_PER_SLIDE:-0}"
NUCLEI_MODEL_PATH="${NUCLEI_MODEL_PATH:-}"

# Repo-aligned defaults requested by user
INPUT_DIR="${INPUT_DIR:-$SCRATCH/Pediatric-Brain-Tumor/data/wsi_raw}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRATCH/Pediatric-Brain-Tumor/data/wsi_processed}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-histocartography_graph_embeddings_vista}"

NUCLEI_MODEL_ARGS=()
if [ -z "${NUCLEI_MODEL_PATH}" ] && [ -f "${REPO_ROOT}/checkpoints/hovernet_pannuke.pth" ]; then
  NUCLEI_MODEL_PATH="${REPO_ROOT}/checkpoints/hovernet_pannuke.pth"
fi
if [ -n "${NUCLEI_MODEL_PATH}" ]; then
  NUCLEI_MODEL_ARGS=(--nuclei-model-path "${NUCLEI_MODEL_PATH}")
fi

python "${REPO_ROOT}/examples/svs_graph_embeddings.py" \
  --input-dir "${INPUT_DIR}" \
  --output-root "${OUTPUT_ROOT}" \
  --output-subdir "${OUTPUT_SUBDIR}" \
  --tile-size 2048 \
  --tile-stride 2048 \
  --nuclei-batch-size "${NUCLEI_BATCH}" \
  --cell-feature-batch-size "${CELL_BATCH}" \
  --patch-feature-batch-size "${PATCH_BATCH}" \
  --num-workers "${NUM_WORKERS}" \
  --max-tiles-per-slide "${MAX_TILES_PER_SLIDE}" \
  "${NUCLEI_MODEL_ARGS[@]}" \
  --save-graphs \
  --export-patch-embeddings \
  --continue-on-error

echo "[INFO] Job finished at $(date)"
