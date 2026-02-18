#!/bin/bash
#SBATCH -J hc_svs_embed
#SBATCH -o %x.%A_%a.out
#SBATCH -e %x.%A_%a.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
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

# Runtime tuning for throughput/stability on GH nodes.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TORCH_HOME="${TORCH_HOME:-$SCRATCH/.cache/torch}"
mkdir -p "${TORCH_HOME}"

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
NUM_WORKERS="${NUM_WORKERS:-12}"
NUCLEI_BATCH="${NUCLEI_BATCH:-16}"
CELL_BATCH="${CELL_BATCH:-64}"
PATCH_BATCH="${PATCH_BATCH:-128}"
MAX_TILES_PER_SLIDE="${MAX_TILES_PER_SLIDE:-0}"
NUCLEI_MODE="${NUCLEI_MODE:-classical}"
NUCLEI_MODEL_PATH="${NUCLEI_MODEL_PATH:-}"
HOVERNET_PKG_MODEL_MODE="${HOVERNET_PKG_MODEL_MODE:-auto}"
HOVERNET_PKG_NR_TYPES="${HOVERNET_PKG_NR_TYPES:--1}"
EXPORT_PATCH_EMBEDDINGS="${EXPORT_PATCH_EMBEDDINGS:-0}"

SHARD_INDEX="${SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"
NUM_SHARDS="${NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}"

# Repo-aligned defaults requested by user
INPUT_DIR="${INPUT_DIR:-$SCRATCH/Pediatric-Brain-Tumor/data/wsi_raw}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRATCH/Pediatric-Brain-Tumor/data/wsi_processed}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-histocartography_graph_embeddings}"

NUCLEI_MODEL_ARGS=()
if [ -n "${NUCLEI_MODEL_PATH}" ]; then
  NUCLEI_MODEL_ARGS=(--nuclei-model-path "${NUCLEI_MODEL_PATH}")
fi

HOVERNET_PKG_ARGS=()
if [ "${NUCLEI_MODE}" = "hovernet-package" ]; then
  HOVERNET_PKG_ARGS=(
    --hovernet-pkg-model-mode "${HOVERNET_PKG_MODEL_MODE}"
    --hovernet-pkg-nr-types "${HOVERNET_PKG_NR_TYPES}"
  )
fi

PATCH_ARGS=()
if [ "${EXPORT_PATCH_EMBEDDINGS}" = "1" ]; then
  PATCH_ARGS=(--export-patch-embeddings)
fi

echo "[INFO] Shard: ${SHARD_INDEX}/${NUM_SHARDS}"
echo "[INFO] Nuclei mode: ${NUCLEI_MODE}"
echo "[INFO] Nuclei model path: ${NUCLEI_MODEL_PATH:-<none>}"
echo "[INFO] hover-net package model mode: ${HOVERNET_PKG_MODEL_MODE}"
echo "[INFO] hover-net package nr_types: ${HOVERNET_PKG_NR_TYPES}"
echo "[INFO] Export patch embeddings: ${EXPORT_PATCH_EMBEDDINGS}"

python "${REPO_ROOT}/examples/svs_graph_embeddings.py" \
  --input-dir "${INPUT_DIR}" \
  --output-root "${OUTPUT_ROOT}" \
  --output-subdir "${OUTPUT_SUBDIR}" \
  --shard-index "${SHARD_INDEX}" \
  --num-shards "${NUM_SHARDS}" \
  --nuclei-mode "${NUCLEI_MODE}" \
  --tile-size 2048 \
  --tile-stride 2048 \
  --nuclei-batch-size "${NUCLEI_BATCH}" \
  --cell-feature-batch-size "${CELL_BATCH}" \
  --patch-feature-batch-size "${PATCH_BATCH}" \
  --num-workers "${NUM_WORKERS}" \
  --max-tiles-per-slide "${MAX_TILES_PER_SLIDE}" \
  "${NUCLEI_MODEL_ARGS[@]}" \
  "${HOVERNET_PKG_ARGS[@]}" \
  "${PATCH_ARGS[@]}" \
  --save-graphs \
  --continue-on-error

echo "[INFO] Job finished at $(date)"
