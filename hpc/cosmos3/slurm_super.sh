#!/usr/bin/env bash
# =============================================================================
# Cosmos 3 — Optional visuals: Cosmos3-Super text2video / image2video
#
#   prompt                ->  cinematic clip   (text2video)
#   prompt + first frame  ->  cinematic clip   (image2video)
#
# Cosmos3-Super has NO action modes (ARCHITECTURE §7): it cannot run jobs
# B/C. Use it only for optional, non-pose-locked cinematic clips.
#
# This script starts a vLLM-Omni server on the allocated GPUs
# (hpc/cosmos3/launch_server.py), waits for it to become ready, drives it
# through the OpenAI-compatible video endpoint with the same fields
# hpc/cosmos3/smoke.py uses, then shuts the server down.
#
# Submit (from the repository root):
#
#   sbatch hpc/cosmos3/slurm_super.sh jobs/super_clip
#
# The job directory must contain job.json. The worker writes output.mp4 and,
# last, manifest.json into it. Network is assumed unavailable on compute
# nodes, so stage the weights (and a serving venv) on shared storage first.
#
# Environment overrides (export before sbatch, or pass --export=ALL):
#
#   COSMOS_REPO         repository root          (default: $SLURM_SUBMIT_DIR)
#   COSMOS_VENV         virtualenv to activate   (default: <repo>/.venv)
#   COSMOS_MODULES      modules to load          (default: "cuda/12.8 python/3.12")
#   COSMOS_MODEL_PATH   HF checkpoint / local dir (default: nvidia/Cosmos3-Super)
#   COSMOS_PROMPT       override the job.json prompt
#   COSMOS_IMAGE        first-frame path -> image2video (else text2video)
#   COSMOS_CFG_PARALLEL / COSMOS_ULYSSES / COSMOS_HSDP_SHARD
#                       vLLM-Omni parallelism (see launch_server.py) [verify]
#   COSMOS_INIT_TIMEOUT seconds to wait for the server (default: 1800)
#
# Site-specific values (partition, GRES name, module names, wall time) are
# marked [verify] — confirm them against your cluster's documentation.
# =============================================================================
#SBATCH --job-name=cosmos3-super
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:b200:4                 # [verify] 2-4 B200s for Super
#SBATCH --time=04:00:00
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
    echo "error: $JOB_DIR/job.json not found" >&2
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

# --- server configuration ----------------------------------------------------
export COSMOS_MODEL="${COSMOS_MODEL:-nvidia/Cosmos3-Super}"
export COSMOS_MODEL_PATH="${COSMOS_MODEL_PATH:-$COSMOS_MODEL}"
export COSMOS_HOST="${COSMOS_HOST:-127.0.0.1}"
export COSMOS_PORT="${COSMOS_PORT:-8000}"
COSMOS_BASE_URL="http://${COSMOS_HOST}:${COSMOS_PORT}"
export COSMOS_BASE_URL

# vLLM-Omni topology for the 4x B200 node. launch_server.py defaults assume an
# 8-GPU layout, so we override for 4 GPUs here. [verify] against the framework.
export COSMOS_CFG_PARALLEL="${COSMOS_CFG_PARALLEL:-2}"
export COSMOS_ULYSSES="${COSMOS_ULYSSES:-2}"
export COSMOS_HSDP_SHARD="${COSMOS_HSDP_SHARD:-2}"
export COSMOS_INIT_TIMEOUT="${COSMOS_INIT_TIMEOUT:-1800}"

SERVER_LOG="$JOB_DIR/server.log"
echo "starting vLLM-Omni server -> $SERVER_LOG"
"$PYTHON" "$REPO/hpc/cosmos3/launch_server.py" --backend omni >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

cleanup() {
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "stopping Cosmos3-Super server (pid $SERVER_PID)"
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# --- wait for the server to come up ------------------------------------------
echo "waiting for $COSMOS_BASE_URL/v1/models (up to ${COSMOS_INIT_TIMEOUT}s) ..."
READY=0
DEADLINE=$(( $(date +%s) + COSMOS_INIT_TIMEOUT ))
while (( $(date +%s) < DEADLINE )); do
    if curl -fsS "$COSMOS_BASE_URL/v1/models" >/dev/null 2>&1; then
        READY=1
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "error: Cosmos3-Super server exited before becoming ready" >&2
        tail -n 50 "$SERVER_LOG" >&2 || true
        exit 1
    fi
    sleep 10
done
if (( READY == 0 )); then
    echo "error: Cosmos3-Super server not ready within ${COSMOS_INIT_TIMEOUT}s" >&2
    tail -n 50 "$SERVER_LOG" >&2 || true
    exit 1
fi
echo "server is ready"

# --- drive the endpoint and write outputs ------------------------------------
export COSMOS_OUTPUT="$JOB_DIR/output.mp4"
"$PYTHON" - <<'PY'
"""Generate one Cosmos3-Super clip from job.json via /v1/videos/sync.

Mirrors the fields used by hpc/cosmos3/smoke.py, and adds an optional first
frame so the same worker covers image2video as well as text2video.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import requests

job_dir = Path(os.environ["COSMOS_JOB_DIR"])
base_url = os.environ["COSMOS_BASE_URL"].rstrip("/")
model = os.environ.get("COSMOS_MODEL", "nvidia/Cosmos3-Super")
output = Path(os.environ.get("COSMOS_OUTPUT", str(job_dir / "output.mp4")))

job: dict = {}
job_file = job_dir / "job.json"
try:
    job = json.loads(job_file.read_text(encoding="utf-8"))
except (OSError, ValueError) as exc:
    print(f"warning: could not read {job_file}: {exc}", file=sys.stderr)
    job = {}

params = job.get("params") if isinstance(job.get("params"), dict) else {}
inputs = job.get("inputs") if isinstance(job.get("inputs"), dict) else {}


def pick(*names, default=None):
    """Top-level job fields win over params; skip explicit nulls."""
    for name in names:
        if job.get(name) is not None:
            return job[name]
    for name in names:
        if params.get(name) is not None:
            return params[name]
    return default


prompt = (
    os.environ.get("COSMOS_PROMPT")
    or pick("prompt", "caption")
    or inputs.get("prompt")
    or "A spacecraft drifts past the Moon as Earth rises over the limb."
)
image = os.environ.get("COSMOS_IMAGE") or inputs.get("first_frame") or inputs.get("image")

headers = {}
if token := os.environ.get("COSMOS_API_KEY"):
    headers["Authorization"] = f"Bearer {token}"

fields = {
    "model": model,
    "prompt": str(prompt),
    "size": str(pick("size", "resolution", default="1280x720")),
    "num_frames": str(pick("num_frames", "frames", default=189)),
    "fps": str(pick("fps", default=24)),
    "num_inference_steps": str(pick("num_inference_steps", "steps", default=35)),
    "guidance_scale": str(pick("guidance_scale", default=6.0)),
    "max_sequence_length": str(pick("max_sequence_length", default=4096)),
    "flow_shift": str(pick("flow_shift", default=10.0)),
    "seed": str(pick("seed", default=17)),
    "extra_params": json.dumps(
        {"use_resolution_template": False, "use_duration_template": False, "guardrails": True}
    ),
}
if image:
    # image2video: launch_server.py allows local media under COSMOS_MEDIA_ROOT
    # (default "/"), so a path is accepted. Field name [verify] vs vLLM-Omni.
    fields["image"] = str(image)

mode = "image2video" if image else "text2video"
print(f"requesting {mode}: model={model} size={fields['size']} frames={fields['num_frames']}")

response = requests.post(
    f"{base_url}/v1/videos/sync",
    headers={**headers, "Accept": "video/mp4"},
    data=fields,
    timeout=1800,
)
response.raise_for_status()
output.parent.mkdir(parents=True, exist_ok=True)
output.write_bytes(response.content)
if len(response.content) < 1024 or b"ftyp" not in response.content[:32]:
    raise RuntimeError(f"{output} does not look like an MP4 ({len(response.content)} bytes)")

manifest = {
    "job_id": job.get("job_id") or job.get("id") or job_dir.name,
    "mode": mode,
    "artifacts": [
        {"name": "video", "path": output.name, "kind": "video", "bytes": len(response.content)}
    ],
    "created_at_s": int(time.time()),
    "metadata": {
        "worker": "hpc/cosmos3/slurm_super.sh",
        "model": model,
        "prompt": str(prompt),
        "seed": fields["seed"],
        "size": fields["size"],
        "num_frames": fields["num_frames"],
    },
}
# manifest.json is the completion marker: write it last.
(job_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(f"wrote {output} ({len(response.content)} bytes) and manifest.json")
PY

echo "finished: $(date -Is)"
