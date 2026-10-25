<<<<<<< HEAD
# Cosmos 3 on HPC (4× B200)

SLURM deployment for the Cosmos 3 render-and-validate layer described in
[`ARCHITECTURE.md` §7–8](../../ARCHITECTURE.md). Four job scripts cover the three
Nano jobs (one GPU each) and the optional Super visuals (2–4 GPUs).

| Script | Job | Checkpoint | GPUs | Wall time |
| --- | --- | --- | --- | --- |
| `slurm_transfer.sh` | A. Transfer (control video → photoreal video) | `Cosmos3-Nano-Sim-Transfer` | 1 | 2 h |
| `slurm_inverse.sh` | B. Inverse dynamics (video → 9D actions) | `Cosmos3-Nano` | 1 | 2 h |
| `slurm_forward.sh` | C. Forward dynamics (first frame + actions → video) | `Cosmos3-Nano` | 1 | 2 h |
| `slurm_super.sh` | Optional cinematic clip (text2video / image2video) | `Cosmos3-Super` | 4 (2–4) | 4 h |

`Cosmos3-Super` has **no action modes** — it cannot run jobs B or C. Use it only
for non-pose-locked cinematic clips.

```
hpc/cosmos3/
├─ slurm_transfer.sh     # Job A  (job dir → worker)
├─ slurm_inverse.sh      # Job B
├─ slurm_forward.sh      # Job C
├─ slurm_super.sh        # Cosmos3-Super: launches vLLM-Omni + drives the endpoint
├─ launch_server.py      # starts the vLLM Reasoner / vLLM-Omni server
├─ smoke.py              # reference client for the Super endpoint
├─ download_weights.py   # stage gated checkpoints into HF_HOME
└─ README.md
```

---

## 1. Prerequisites

* A SLURM cluster with 4× B200 (or comparable) nodes. The scripts assume a
  `gpu` partition and a `gpu:b200:N` GRES name — **verify these** for your site
  (`sinfo`, `scontrol show partition`). Edit the `#SBATCH` header or add a
  site-specific `#SBATCH --partition=…` / `--gres=…` line.
* CUDA 12.8+ toolkit (driver 580+ for CUDA 13), Python 3.11+.
* A Hugging Face token (`HF_TOKEN`) with access to the gated `nvidia/Cosmos3-*`
  checkpoints.
* **Compute nodes are assumed to have no internet.** Do all downloads on a
  login/build node and keep caches on shared storage (see §2).
* ~150 GiB free on shared storage for weights and caches.

## 2. One-time setup (login node, internet available)

```sh
cd /path/to/gatorhacks-v

# Python environment (uv). The GPU extra pulls torch.
uv sync --extra dev --extra data --extra gpu

# For the Super server, install vLLM / vLLM-Omni into the same venv.
#   (follow the Cosmos-Framework install notes for exact extras)
# uv pip install vllm vllm-omni   # [verify] package names vs the framework docs

# Accept the model terms on Hugging Face, then stage the checkpoints.
export HF_TOKEN=hf_...
python hpc/cosmos3/download_weights.py nvidia/Cosmos3-Super
python hpc/cosmos3/download_weights.py nvidia/Cosmos3-Nano
python hpc/cosmos3/download_weights.py nvidia/Cosmos3-Nano-Sim-Transfer
```

Downloaded weights land in `$HF_HOME` (default `<repo>/.cache/huggingface`).
Keep that directory on shared storage so batch jobs can read it.

## 3. Preparing a job

Every script takes a **job directory** (created by the Rust `posebuilder` or the
`ipc` queue — see [`docs/cosmos-contract.md`](../../docs/cosmos-contract.md)) as
its first argument. The directory must contain `job.json`:

```sh
# Native queue entry:
cargo run -p ipc --example enqueue_job -- jobs flyby_w03 inverse_dynamics clip.mp4

# Or from a trajectory CSV (writes job.json + actions.json):
cargo run -p posebuilder --example pose_pipeline -- \
  trajectory.csv jobs/flyby_w03 \
  --job-id flyby_w03 --mode forward_dynamics --model Cosmos3-Nano \
  --t0 1000 --t1 1005 --fps 24 --frames 120 --scale-k 1000 --pointing earth
```

The worker writes its outputs back into the same directory and finishes by
writing `manifest.json` (the completion marker). Submit from the repository
root so `SLURM_SUBMIT_DIR` resolves to the repo.

## 4. Submitting jobs

```sh
cd /path/to/gatorhacks-v

# Job A — transfer
sbatch hpc/cosmos3/slurm_transfer.sh jobs/flyby_w03

# Job B — inverse dynamics
sbatch hpc/cosmos3/slurm_inverse.sh  jobs/flyby_w03

# Job C — forward dynamics
sbatch hpc/cosmos3/slurm_forward.sh  jobs/flyby_w03

# Optional Super clip
sbatch hpc/cosmos3/slurm_super.sh    jobs/super_clip
```

Override environment values by exporting them first (SLURM forwards the
submitting environment; add `--export=ALL` if your site restricts this):

```sh
COSMOS_MODEL_PATH=/shared/models/Cosmos3-Super \
HF_HOME=/shared/hf \
sbatch hpc/cosmos3/slurm_super.sh jobs/super_clip
```

A **job array** runs the same job type over several directories in parallel,
one GPU each (the recommended Nano layout):

```sh
# jobs/array.txt lists one job directory per line.
sbatch --array=0-5 --gres=gpu:b200:1 hpc/cosmos3/slurm_inverse.sh \
  "$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" jobs/array.txt)"
```

## 5. Monitoring

```sh
squeue -u "$USER"                 # queue state
squeue --start -j <jobid>         # estimated start (backfill)
scontrol show job <jobid>         # full detail
sacct -j <jobid> --format=JobID,JobName,State,Elapsed,MaxRSS,ExitCode

# Live logs (written next to the submit dir):
tail -f cosmos3-transfer-<jobid>.out
tail -f cosmos3-transfer-<jobid>.err

# Per-job server log (Super):
tail -f jobs/super_clip/server.log

# Confirm GPUs are visible inside the job:
#   the scripts run `nvidia-smi -L` at startup.
```

`job_state` meanings: `PENDING`, `RUNNING`, `COMPLETED` (exit 0),
`FAILED` (non-zero). A successful worker always leaves `manifest.json` in the
job directory; a failed run does not (by design — see the contract §4.2).

## 6. Retrieving outputs

Artifacts are ordinary files in the job directory, so retrieve them with
`rsync`/`scp` or by reading them from shared storage directly:

```sh
# Inspect:
cat jobs/flyby_w03/manifest.json        # artifact list + metadata
cat jobs/flyby_w03/sample_outputs.json  # poses[] (jobs B/C)
ls  jobs/flyby_w03                      # output.mp4 (jobs A/Super)

# Pull to a laptop:
rsync -avz login.example.edu:/shared/gatorhacks/jobs/flyby_w03 ./local-jobs/
```

Send completed artifacts to the evaluator and the Three.js UI:

| Artifact | `kind` | Consumer |
| --- | --- | --- |
| `sample_outputs.json` | `pose_json` | Rust `evaluator` (RRE/RTE/ATE), UI |
| `output.mp4` | `video` | UI playback; Job B input for transfer clips |
| `manifest.json` | — | completion marker / provenance |

## 7. Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `COSMOS_REPO` | `$SLURM_SUBMIT_DIR` | repository root |
| `COSMOS_VENV` | `<repo>/.venv` | virtualenv to activate |
| `COSMOS_CONDA_ENV` | — | conda env fallback if no venv |
| `COSMOS_PYTHON` | `python` | interpreter (after activation) |
| `COSMOS_MODULES` | `cuda/12.8 python/3.12` | modules to `module load` |
| `COSMOS_WORKER_CMD` | `ml/cosmos/run.py` if present, else mock | real worker command prefix |
| `COSMOS_MODEL` | per job | served model id |
| `COSMOS_DOMAIN` | `camera_pose` | action domain (jobs B/C) |
| `HF_HOME` | `<repo>/.cache/huggingface` | checkpoint cache (put on shared storage) |
| `COSMOS_MODEL_PATH` | `nvidia/Cosmos3-Super` | local checkpoint for the server |
| `COSMOS_HOST` / `COSMOS_PORT` | `127.0.0.1` / `8000` | Super server bind (non-loopback needs `COSMOS_API_KEY`) |
| `COSMOS_CFG_PARALLEL` / `COSMOS_ULYSSES` / `COSMOS_HSDP_SHARD` | `2` / `2` / `2` | vLLM-Omni parallelism for the 4× B200 node |
| `COSMOS_INIT_TIMEOUT` | `1800` | seconds to wait for the Super server |
| `COSMOS_PROMPT` / `COSMOS_IMAGE` | from `job.json` | override Super prompt / first frame (→ image2video) |

## 8. How each script works

All four share the same skeleton:

1. **Locate the repo** (`COSMOS_REPO` → `SLURM_SUBMIT_DIR` → script path) and
   resolve the **job directory** from `$1`; fail fast if `job.json` is missing.
2. **Load modules** (`module purge` + `module load $COSMOS_MODULES`) when a
   `module` command exists; otherwise skip.
3. **Set up the environment**: activate the venv (or conda env), prepend the
   repo to `PYTHONPATH`, and create shared caches (`HF_HOME`, `UV_CACHE_DIR`,
   `TRITON_CACHE_DIR`, `TORCH_HOME`).
4. **Run the worker** and let its exit code propagate:
   * Jobs A/B/C invoke the file-contract worker on the job directory. The
     default is the real runner at `ml/cosmos/run.py` when it exists, otherwise
     `scripts/mock_cosmos_worker.py` (deterministic, no GPU) so the pipeline is
     testable before weights are staged. Point `COSMOS_WORKER_CMD` at the Cosmos
     framework CLI for production runs.
   * Super starts `launch_server.py --backend omni`, waits for
     `GET /v1/models`, then calls `POST /v1/videos/sync` with the same fields
     `smoke.py` uses (plus an optional `image` for image2video) and writes
     `output.mp4` + `manifest.json`. The server is always stopped on exit via a
     trap.
5. **Report errors**: an `ERR` trap prints the exit status and line number and
   re-exits non-zero, so `sacct`/`squeue` and the `.err` log agree.

## 9. Troubleshooting

| Symptom | Likely cause / fix |
| --- | --- |
| `error: job directory not found` | pass an existing path containing `job.json`; run `sbatch` from the repo root |
| `vllm executable not found` | install vLLM/vLLM-Omni into `COSMOS_VENV`; see §2 |
| Super server never becomes ready | check `jobs/<id>/server.log`; increase `COSMOS_INIT_TIMEOUT`; confirm the parallelism env matches the GPU count |
| `CUDA out of memory` on Super | reduce `COSMOS_ULYSSES`/`--gres` or use a 4-GPU node; Nano jobs need only 1 GPU |
| Weights not found at runtime | compute nodes are offline — stage them into `HF_HOME` on shared storage and pass `HF_HOME` to the job |
| `module: command not found` | the site uses a different module system; set `COSMOS_MODULES=""` and provide the environment another way |
| Wrong GRES/partition | edit the `#SBATCH` header or add a site-specific line |

## 10. Notes to verify (`[verify]`)

The following are **placeholders** copied from `ARCHITECTURE.md` §8 and must be
confirmed against your cluster and the Cosmos-Framework docs before a real run:
scheduler, partition, GRES name, wall-time limits, GPU memory per device, and
whether compute nodes have internet; the vLLM-Omni parallelism values for a
4× B200 node; the image2video request field name; and the module names.
=======
# Cosmos3-Super on an 8-GPU B200 Slurm node

This directory is a standalone HPC deployment path. It does not start or depend on the Reconix Electron application.

It exposes two serving choices over the same Hugging Face checkpoint:

- `reasoner`: vLLM OpenAI-compatible `/v1/chat/completions` for text/image/video understanding.
- `omni`: vLLM-Omni OpenAI-compatible image/video generation APIs (`/v1/images/generations`, `/v1/videos/sync`).

The Reasoner and Generator are different model towers. Pick a backend per Slurm job; do not try to run both in the same eight-GPU allocation.

# One-time: choose where venvs, weights, and caches live.
# Keep this OUTSIDE the git repo so big files never get committed.
export COSMOS_ROOT="$HOME/cosmos3-data"
mkdir -p "$COSMOS_ROOT"/{venvs,models,huggingface,uv-cache}
export UV_CACHE_DIR="$COSMOS_ROOT/uv-cache"

## Prerequisites

1. Request and accept the Hugging Face access terms for `nvidia/Cosmos3-Super`. The Omni Generator keeps Cosmos safety guardrails enabled by default; accept access for `nvidia/Cosmos-1.0-Guardrail` as well.
2. Use a CUDA 13.0 compatible driver/software environment on the B200 node. Build the serving environment on a compute node or a site-supported container; the frontend is for editing/submission, not model inference.
3. Install a current Cosmos-compatible serving stack in a persistent virtual environment. Official recipes currently require native vLLM Cosmos 3 support for Reasoner and vLLM-Omni for Generator. Example starting points:

   ```bash
   # Reasoner environment
  uv venv --python 3.13 "$COSMOS_ROOT/venvs/cosmos3-reasoner"
  source "$COSMOS_ROOT/venvs/cosmos3-reasoner/bin/activate"
  uv pip install --torch-backend=cu130 'vllm>=0.23.0' requests huggingface_hub

  # Generator environment
  uv venv --python 3.13 "$COSMOS_ROOT/venvs/cosmos3-omni"
  source "$COSMOS_ROOT/venvs/cosmos3-omni/bin/activate"
  uv pip install --torch-backend=cu130 \
  'vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git@main' \
  requests huggingface_hub
   ```

   vLLM-Omni `main` is intentionally shown as a bring-up command, not a reproducible production pin. After the first successful run, pin the exact commit (or a tested site container digest) and record the Python/CUDA/driver versions.

4. Store the HF token in the cluster's approved secret mechanism or export it in the submission shell. Do not put it in this repository, a Slurm script, or job output.

## Stage the Hugging Face weights

Run this in a job/environment that can reach Hugging Face and write to persistent storage:

```bash
source "$COSMOS_ROOT/venvs/cosmos3-omni/bin/activate"
export HF_HOME="$COSMOS_ROOT/huggingface"
python hpc/cosmos3/download_weights.py nvidia/Cosmos3-Super \
  --local-dir "$COSMOS_ROOT/models/Cosmos3-Super"
python hpc/cosmos3/download_weights.py nvidia/Cosmos-1.0-Guardrail \
  --local-dir "$COSMOS_ROOT/models/Cosmos-1.0-Guardrail"
```

If the cluster compute nodes have no external network access, stage the snapshots from an approved transfer node first. Then run with `COSMOS_MODEL_PATH` set to the local snapshot directory and `HF_HUB_OFFLINE=1`. Keep `HF_HOME` on persistent storage so a job restart does not repeat the large download.

## Submit

Site partition and account names differ, so supply them at submission time rather than hardcoding one cluster's names:

```bash
export HPC_PARTITION=longq7-dgxb200
export COSMOS_VENV="$COSMOS_ROOT/venvs/cosmos3-omni"
export HF_HOME="$COSMOS_ROOT/huggingface"
export COSMOS_MODEL_PATH="$COSMOS_ROOT/models/Cosmos3-Super"
export HF_TOKEN=<read-token-from-your-secret-store>
export COSMOS_PORT=8000

bash hpc/cosmos3/submit.sh omni
# or, in a separate allocation/environment:
bash hpc/cosmos3/submit.sh reasoner
```

The submit wrapper requests one node and eight B200 GPUs with `--gres=gpu:b200:8`. Override `GPU_GRES` if the site uses a different GRES spelling, for example `gpu:8`. The batch job runs the server in the foreground so Slurm owns its lifetime and terminates all ranks when the allocation ends.

The Omni launch uses NVIDIA's documented 8-GPU topology: CFG parallel size 2, Ulysses degree 4, and HSDP shard size 8. Initialization can take several minutes; `--init-timeout 1800` is set. Guardrails remain enabled.

The Reasoner uses the unified `nvidia/Cosmos3-Super` checkpoint with the Reasoner architecture override and TP=8. Set `COSMOS_TP_SIZE=4` to test the documented four-GPU model-card topology; that requires changing the Slurm request to four GPUs (or changing the launcher to use another parallelism layout).

## Call the server

By default, the service binds to loopback on the allocated node. Use the cluster-approved SSH tunnel or forwarding method to reach it from your workstation. If your site's internal routing requires binding to a non-loopback interface, set `COSMOS_HOST=0.0.0.0` and a strong `COSMOS_API_KEY`; the launcher refuses a non-loopback bind without an API key. Do not publish the port to the public network. From a shell on the compute node:

```bash
# Smoke test
python hpc/cosmos3/smoke.py --backend omni --output "$COSMOS_ROOT/cosmos-smoke.mp4"
python hpc/cosmos3/smoke.py --backend reasoner
```

The Omni smoke request follows NVIDIA's Cosmos3-Super example settings (720p, 189 frames, 35 steps), so it is a real generation workload and may take a few minutes. The Reasoner smoke request checks a text completion; use the official vLLM Cosmos 3 sample for image/video inputs.

Logs are written under `hpc/cosmos3/logs/`. Cancel the job with `scancel <job-id>`; Slurm then tears down the foreground server and its distributed workers.
>>>>>>> 5426358 (directions for cosmos on hpc)
