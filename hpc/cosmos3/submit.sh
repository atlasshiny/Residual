#!/usr/bin/env bash
# Submit one Cosmos3-Super serving job.
# Required site-specific settings: HPC_PARTITION and (if required) HPC_ACCOUNT.
# Optional: GPU_GRES (defaults to gpu:b200:8), COSMOS_JOB_TIME (24:00:00).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-}"
case "$MODE" in
  reasoner|omni) ;;
  *) echo "usage: HPC_PARTITION=<partition> [HPC_ACCOUNT=<account>] $0 {reasoner|omni}" >&2; exit 2 ;;
esac

: "${HPC_PARTITION:?Set HPC_PARTITION to the B200 partition name on your cluster}"
GPU_GRES="${GPU_GRES:-gpu:b200:8}"
COSMOS_JOB_TIME="${COSMOS_JOB_TIME:-24:00:00}"
LOG_DIR="${COSMOS_LOG_DIR:-$HERE/logs}"
mkdir -p "$LOG_DIR"

SBATCH_ARGS=(
  --partition="$HPC_PARTITION"
  --gres="$GPU_GRES"
  --time="$COSMOS_JOB_TIME"
  --output="$LOG_DIR/cosmos3-super-${MODE}-%j.out"
  --error="$LOG_DIR/cosmos3-super-${MODE}-%j.err"
  --export="ALL,COSMOS_BACKEND=$MODE"
)
if [[ -n "${HPC_ACCOUNT:-}" ]]; then
  SBATCH_ARGS+=(--account="$HPC_ACCOUNT")
fi

exec sbatch "${SBATCH_ARGS[@]}" "$HERE/serve.sbatch"
