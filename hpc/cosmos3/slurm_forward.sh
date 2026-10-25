#!/usr/bin/env bash
# =============================================================================
# Cosmos 3 — Job C: Cosmos3-Nano forward dynamics
#
#   first frame + planned action sequence  ->  predicted video
#   (domain_name: camera_pose; action_path JSON from posebuilder)
#
# Submit (from the repository root):
#
#   sbatch hpc/cosmos3/slurm_forward.sh jobs/flyby_w03
#
# The job directory must already contain a job.json (mode: forward_dynamics)
# with a first frame and an actions path. The worker writes the generated clip
# and, last, manifest.json back into the same directory.
#
# Environment overrides (export them before sbatch; sbatch forwards the
# submitting environment by default, or pass --export=ALL):
#
#   COSMOS_REPO         repository root            (default: $SLURM_SUBMIT_DIR)
#   COSMOS_VENV         virtualenv to activate     (default: <repo>/.venv)
#   COSMOS_MODULES      modules to load            (default: "cuda/12.8 python/3.12")
#   COSMOS_WORKER_CMD   real worker command prefix (default: ml/cosmos/run.py)
#   COSMOS_MODEL        model id                   (default: Cosmos3-Nano)
#   COSMOS_DOMAIN       action domain              (default: camera_pose)
#
# Site-specific values (partition, GRES name, module names, wall time) are
# marked [verify] — confirm them against your cluster's documentation.
# =============================================================================
#SBATCH --job-name=cosmos3-forward
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:b200:1                 # [verify] GRES name for one B200
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

# --- locate the repository ---------------------------------------------------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="${COSMOS_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ ! -f "$REPO/hpc/cosmos3/smoke.py" ]]; then
    REPO="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
fi
export COSMOS_REPO="$REPO"
echo "repo:    $REPO"
echo "host:    $(hostname)"
echo "started: $(date -Is)"

# --- job directory (first positional argument) -------------------------------
if [[ $# -lt 1 ]]; then
    echo "usage: sbatch $0 <job-dir> [worker-args...]" >&2
    exit 2
fi
JOB_ARG="$1"
shift
JOB_DIR="$(cd -- "$JOB_ARG" 2>/dev/null && pwd)" || {
    echo "error: job directory not found: $JOB_ARG" >&2
    exit 2
}
if [[ ! -f "$JOB_DIR/job.json" ]]; then
    echo "error: $JOB_DIR/job.json not found (run posebuilder first)" >&2
    exit 2
fi
export COSMOS_JOB_DIR="$JOB_DIR"
echo "job dir: $JOB_DIR"

# --- environment modules -----------------------------------------------------
if type module >/dev/null 2>&1; then
    module purge || true
    # shellcheck disable=SC2086
    module load ${COSMOS_MODULES:-cuda/12.8 python/3.12} || true
fi

# --- python environment ------------------------------------------------------
VENV="${COSMOS_VENV:-$REPO/.venv}"
if [[ -f "$VENV/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    echo "venv:    $VENV"
elif [[ -n "${COSMOS_CONDA_ENV:-}" ]] && type conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    eval "$(conda shell.bash hook)"
    conda activate "$COSMOS_CONDA_ENV"
    echo "conda:   $COSMOS_CONDA_ENV"
else
    echo "warning: no virtualenv at $VENV; falling back to system python" >&2
fi
PYTHON="${COSMOS_PYTHON:-python}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

# --- caches on shared storage (compute nodes are assumed offline) ------------
export HF_HOME="${HF_HOME:-$REPO/.cache/huggingface}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$REPO/.cache/uv}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$REPO/.cache/triton}"
export TORCH_HOME="${TORCH_HOME:-$REPO/.cache/torch}"
mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$TRITON_CACHE_DIR" "$TORCH_HOME"

# --- error reporting ---------------------------------------------------------
on_error() {
    local code=$?
    echo "ERROR: $(basename -- "$0") exited with status $code (line ${1:-?})" >&2
    echo "job dir: $JOB_DIR" >&2
    exit "$code"
}
trap 'on_error $LINENO' ERR

nvidia-smi -L 2>/dev/null || echo "warning: nvidia-smi not available" >&2

# --- run the worker ----------------------------------------------------------
export COSMOS_MODE="forward_dynamics"
export COSMOS_MODEL="${COSMOS_MODEL:-Cosmos3-Nano}"
export COSMOS_DOMAIN="${COSMOS_DOMAIN:-camera_pose}"

if [[ -n "${COSMOS_WORKER_CMD:-}" ]]; then
    read -r -a WORKER <<< "$COSMOS_WORKER_CMD"
elif [[ -f "$REPO/ml/cosmos/run.py" ]]; then
    WORKER=("$PYTHON" "$REPO/ml/cosmos/run.py" --mode "$COSMOS_MODE")
else
    echo "note: ml/cosmos/run.py not found; using the file-contract mock worker." >&2
    echo "note: set COSMOS_WORKER_CMD to the Cosmos framework CLI for real runs." >&2
    WORKER=("$PYTHON" "$REPO/scripts/mock_cosmos_worker.py")
fi

echo "worker:  ${WORKER[*]} $JOB_DIR $*"
"${WORKER[@]}" "$JOB_DIR" "$@"

echo "finished: $(date -Is)"
