#!/usr/bin/env bash
#SBATCH --job-name=cosmos3-super
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0

# Submit via ./submit.sh so the site-specific partition/GRES can be supplied.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
cd "$REPO_ROOT"

: "${COSMOS_BACKEND:?Submit with hpc/cosmos3/submit.sh reasoner|omni}"
: "${COSMOS_VENV:?Set COSMOS_VENV to the environment containing vLLM or vLLM-Omni}"
[[ -x "$COSMOS_VENV/bin/python" ]] || { echo "No Python at $COSMOS_VENV/bin/python" >&2; exit 2; }
source "$COSMOS_VENV/bin/activate"

# Use persistent project/scratch storage for both cache and generated media.
export HF_HOME="${HF_HOME:-${SCRATCH:-$HOME}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export COSMOS_MEDIA_ROOT="${COSMOS_MEDIA_ROOT:-${SCRATCH:-$HOME}}"
mkdir -p "$HF_HOME" "${COSMOS_OUTPUT_DIR:-$HERE/outputs}"

# B200 requires a CUDA 13 capable stack. Leave module selection to the site's
# environment; fail early with useful GPU visibility diagnostics.
command -v nvidia-smi >/dev/null || { echo "nvidia-smi is not available in this job" >&2; exit 2; }
nvidia-smi -L

exec python "$HERE/launch_server.py" --backend "$COSMOS_BACKEND"
