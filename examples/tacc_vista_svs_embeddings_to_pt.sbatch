#!/bin/bash
#SBATCH -J hc_svs_to_pt
#SBATCH -o %x.%A_%a.out
#SBATCH -e %x.%A_%a.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH -A ASC25123

set -euo pipefail

echo "[INFO] Job started at $(date)"
echo "[INFO] Host: $(hostname)"
echo "[INFO] Working dir: ${SLURM_SUBMIT_DIR}"

# Assume this script is submitted from repository root.
REPO_ROOT="${SLURM_SUBMIT_DIR}"

cd "${REPO_ROOT}"
REPO_ROOT="$(pwd)"

# Vista docs recommend managing software via modules in job scripts.
module reset
module load gcc
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

export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/histocartography:${PYTHONPATH:-}"

# Repo-aligned defaults requested by user
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRATCH/Pediatric-Brain-Tumor/data/wsi_processed}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-histocartography_graph_embeddings_vista}"

SHARD_INDEX="${SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"
NUM_SHARDS="${NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}"
OVERWRITE="${OVERWRITE:-0}"
MERGE_SINGLE_PT="${MERGE_SINGLE_PT:-1}"
MERGED_FILENAME="${MERGED_FILENAME:-all_wsi_fused_embeddings.pt}"

OVERWRITE_ARGS=()
if [ "${OVERWRITE}" = "1" ]; then
  OVERWRITE_ARGS=(--overwrite)
fi

MERGE_ARGS=()
if [ "${MERGE_SINGLE_PT}" = "1" ]; then
  MERGE_ARGS=(--export-merged --merged-filename "${MERGED_FILENAME}")
fi

echo "[INFO] Python: ${PYTHON_BIN}"
echo "[INFO] Repo root: ${REPO_ROOT}"
echo "[INFO] Output root: ${OUTPUT_ROOT}"
echo "[INFO] Output subdir: ${OUTPUT_SUBDIR}"
echo "[INFO] Shard: ${SHARD_INDEX}/${NUM_SHARDS}"
echo "[INFO] Overwrite existing .pt: ${OVERWRITE}"
echo "[INFO] Export merged .pt: ${MERGE_SINGLE_PT}"
echo "[INFO] Merged filename: ${MERGED_FILENAME}"

python "${REPO_ROOT}/examples/svs_embeddings_to_pt.py" \
  --output-root "${OUTPUT_ROOT}" \
  --output-subdir "${OUTPUT_SUBDIR}" \
  --shard-index "${SHARD_INDEX}" \
  --num-shards "${NUM_SHARDS}" \
  "${OVERWRITE_ARGS[@]}" \
  "${MERGE_ARGS[@]}"

echo "[INFO] Job finished at $(date)"
