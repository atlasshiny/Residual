# Residual — Architecture

A CUDA/C++ orbital propagator that replays an Artemis/Orion trajectory, measures where physics diverges from reality, and uses PINN/SINDy to output a readable closed-form correction. **Cosmos 3** is the render-and-validate layer: it turns the physics into photoreal video and independently checks that the video still encodes the motion we gave it.

> **Status legend.** `[paper]` = stated in the Cosmos 3 technical report (arXiv 2606.02800v4, 23 Jun 2026). `[infer]` = our design decision or inference, needs testing. `[verify]` = must be confirmed against the repo, model card, or data source before relying on it.

---

## 1. Why this is "fresh off the frontier"

| Item | Detail |
|---|---|
| Frontier artifact | **Cosmos 3** (Nano 16B, Super 64B), omnimodal world model with a native action modality `[paper]` |
| Release | Hugging Face and GitHub release listed as 05/31/2026 on a copy of the Cosmos3-Super model card (`[verify]` on the official `nvidia/Cosmos3-Super` card); arXiv 2606.02800, v4 dated 23 Jun 2026; OpenMDW-1.1 license `[paper]` |
| Load-bearing use | Action tokens + inverse dynamics give a *video → pose* path that previous Cosmos releases (separate Predict/Transfer/Reason models) did not unify `[paper]` |
| Second recent asset | Artemis/Orion trajectory data `[verify]` what exists, its format, and its release date |

If a judge swaps Cosmos 3 for an older video model, **inverse dynamics (job B below) and the unified transfer/forward/inverse interface disappear**. That is the dependency we want.

## 2. Goals and non-goals

**Goals**
1. FP64 n-body propagation in Rust, validated against SPICE/Horizons.
2. Residual between propagated and reference trajectory, with burn windows masked.
3. PINN/SINDy → symbolic correction term, compared honestly against baselines.
4. Cosmos 3 renders the trajectory, then recovers pose from the video and reports error against the input.

**Non-goals**
- Cosmos does **not** improve the state estimate or feed the PINN. Video models do not conserve energy or obey orbital mechanics.
- No real-time (60 FPS) Cosmos loop. Generation is offline (see §7).
- No FP4/FP8 in the propagator. Low precision is acceptable only inside the PINN and Cosmos.
- Burn Δv or thrust are **not** Cosmos inputs. The Cosmos action space is relative pose deltas `[paper]`.

## 3. System diagram

```
 Artemis/Orion ephemeris + telemetry (SPICE kernels via Python `spiceypy`, other formats [verify])
                         │
                         ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │ RUST CORE                                                             │
 │  ingest ► frames/time ► propagator (FP64, Dormand–Prince) ► state(t)  │
 │                                   │                                   │
 │                    residual = reference − propagated  (burns masked)  │
 │                                   │                                   │
 │                         ┌─────────┴──────────┐                        │
 │                         ▼                    ▼                        │
 │                  PINN branch (§5)     Pose/Action builder (§6)         │
 └─────────────────────────┬────────────────────┬────────────────────────┘
                           │                    │ ΔT (9D) + control video
                           ▼                    ▼
                  symbolic equation   ┌──────────────────────────┐
                  + baselines         │ COSMOS 3 (Python workers) │
                                      │  A Transfer               │
                                      │  B Inverse dynamics       │
                                      │  C Forward dynamics       │
                                      └────────────┬──────────────┘
                                                   ▼
                                  Evaluator (Rust): integrate recovered ΔT,
                                  align, compute RRE / RTE / ATE
                                                   ▼
                              Artifact store (clips, tensors, metrics, JSON)
                                                   ▼
                              UI: orbit viewport · photoreal clip · PINN equation · error plots
```

**Data flow rule:** physics is authoritative. Arrows into Cosmos carry pose and geometry. Arrows out of Cosmos go only to the evaluator and the UI.

## 4. Components

### 4.1 CUDA/C++ core (`src/kernels/`, with Rust helpers where useful)

The performance-critical propagator stays in CUDA/C++. Rust crates may still be used for lightweight helpers, schemas, or orchestration, but the authoritative numerical propagation is CUDA.

| Component | Responsibility | Notes |
|---|---|---|
| `src/kernels/multibody_solver` | FP64 Earth/Moon n-body propagation | CUDA shared/static libraries; currently fixed-step velocity Verlet with optional J2 |
| `ingest` (future) | Load mission files and telemetry; SPICE ingestion is Python-only (`scripts/fetch_spice.py`) | Python/Rust helper around data files; SPICE coverage `[verify]` |
| `frames` | Time systems and reference frames | Largest source of silent bugs; unit-test every transform |
| `residual` | Reference − propagated, burn masking, residual metric | Burn windows and attitude events must be masked or modeled |
| `posebuilder` | State + pointing law → camera pose → ΔT sequence (§6) | Owns scale mapping `k` |
| `evaluator` | Integrate recovered ΔT, Sim(3) align, compute errors | Same metrics as the paper `[paper]` |
| `schema` | Vehicle and mission schemas (serde/JSON/YAML) | Stretch: "universal" spacecraft |
| `ipc` | Job queue and artifact manifest | Talks to Cosmos workers over files + a small RPC |

Validation targets for `dynamics`: energy and angular-momentum conservation on a two-body case, then agreement with Horizons over a chosen arc. The default `AnalyticEphemeris` is the circular Earth-Moon model shared with the CUDA kernel; `TabulatedEphemeris` interpolates the Python-ingested Horizons/SPICE arc so the propagator sees the measured Earth-Moon geometry instead. Building it from the reference Moon track (`TabulatedEphemeris::from_moon_track`) removes the circular-model error from the residual, leaving integrator error plus genuinely missing physics (the Sun, non-gravitational forces) for the PINN.

```bash
# opt-in: measured geometry instead of the circular model
make residual-tabulated      # writes data/residual_table_tabulated.csv
make train DATA=data/residual_table_tabulated.csv
```

The tabulated path needs a reference that keeps the real separation (`scripts/convert_reference.py --preserve-distance`); the default `data/reference.csv` is rescaled onto the circular radius and is intended for the analytic model. The example warns when it detects such a constant-radius arc. With the 30-day Horizons arc, the measured-geometry model cuts the RMS position residual by roughly 4.5× (64,800 km → 14,400 km); the remainder is dominated by the unmodelled solar perturbation, while the fixed-step integrator itself contributes about 1.4 km RMS (velocity Verlet at 600 s versus an adaptive DOP853 solve of the same force model).

### 4.2 PINN branch (`ml/pinn/`)

- Input: residual time series (burn windows masked), plus features such as position relative to Earth and Moon.
- Output: pruned symbolic expression for the correction term.
- Required baselines: SINDy or standard symbolic regression, and a plain MLP. The comparison is what makes the result credible.
- Runs separately from Cosmos on its own GPU allocation (§8).

### 4.3 Cosmos workers (`ml/cosmos/`)

Python, reference PyTorch path first `[paper]` (§5.3.1: reference implementation in native PyTorch; production paths are vLLM-Omni for the Generator). Each worker pulls a job manifest, runs one mode, writes outputs and a result manifest.

### 4.4 UI (`ui/`)

Tauri + Three.js. Shows predicted vs actual path, the symbolic equation, error plots, and plays pre-rendered Cosmos clips. It never waits on live generation.

## 5. Residual and PINN contract

```
residual_table.parquet
  t_tdb, r_ref[3], v_ref[3], r_prop[3], v_prop[3], dr[3], dv[3], burn_mask, features...
pinn_result.json
  expression, per-term coefficients, fit error, baselines {sindy, mlp}, masked_fraction
```

## 6. Pose/Action builder: the Cosmos interface

This is the part most likely to go wrong, so it is specified tightly.

### 6.1 What Cosmos 3 actually accepts `[paper]`

- An action token `a_t` represents the transition from video state `v_{t-1}` to `v_t`.
- Ego and effector motion are **relative transforms**: `ΔT_t = T_{t-1}⁻¹ · T_t`, encoded as **3D translation + 6D rotation** (Zhou et al. 2019 representation), i.e. **9D** for an ego pose.
- Rotations follow the **OpenCV camera convention**. The 6D rotation is converted back to SO(3) via SVD on output.
- Each embodiment domain has its own input/output projection. Actions are normalized per dimension to roughly [−1, 1] using normalizers computed from training data. Camera poses are in **metric scale**.

### 6.2 Our mapping

| Step | Operation |
|---|---|
| 1 | Sample propagator state at the clip frame times (see §6.3). |
| 2 | Apply a **pointing law** to get spacecraft attitude, then the camera pose `T_t` in a fixed inertial frame. Use real attitude only if the data provides it `[verify]`; otherwise define Earth-pointing or Moon-pointing and state that in every result. |
| 3 | Compute `ΔT_t = T_{t-1}⁻¹ T_t`, convert rotation to 6D. |
| 4 | **Scale**: divide positions by `k` km per Cosmos-metre so per-frame translations fall inside the range the model saw in training. `[infer]` `k` is a tuned parameter; log it with every clip. |
| 5 | Normalize with the checkpoint's own normalizers `[verify]` locate them in `nvidia/cosmos-framework`. |
| 6 | Set `domain_name: camera_pose` for the virtual camera (named in the framework's action docs; model card lists camera motion as 9D input) `[framework docs]` `[verify]` raw vs normalized rows in `inputs/omni/action_forward_dynamics_camera.json` |

### 6.3 Clip windows

Trajectories span days, but clips are short (the paper's action evaluations use 5–6 s clips; the generator supports 5–400 frames at 10–30 FPS `[paper]`). Render **time-lapse windows** around events: TLI, outbound coast, lunar flyby, return coast, entry. Each window gets its own `k`, time-compression factor, and manifest entry.

### 6.4 Job manifest

```json
{
  "job_id": "flyby_w03_transfer_v1",
  "mode": "transfer | inverse_dynamics | forward_dynamics",
  "model": "Cosmos3-Nano | Cosmos3-Super",
  "window": {"t0_tdb": "...", "t1_tdb": "...", "time_compression": 0},
  "scale_k_km_per_m": 0,
  "pointing_law": "earth | moon | data | observer",
  "fps": 24, "frames": 120, "resolution": [704, 1280],
  "inputs": {"control_video": "path", "first_frame": "path", "actions": "path.npy", "prompt_json": "path"},
  "seed": 0
}
```

## 7. Cosmos jobs

**Checkpoint routing (from the Cosmos-Framework docs, which override the paper where they differ).** The paper discusses one unified model, but the released framework CLI splits capabilities across checkpoints:

| Job | Checkpoint | Framework mode |
|---|---|---|
| A. Transfer | `Cosmos3-Nano-Sim-Transfer` | One precomputed control video (edge, blur, depth, or segmentation) + a pre-upsampled caption → video, generated autoregressively with a fixed four-step schedule. Sample specs use 121 frames at 30 FPS, 832×480 (blur 736×544). |
| B. Inverse dynamics | `Cosmos3-Nano` | `inverse_dynamics`: video → action sequence in `sample_outputs.json` |
| C. Forward dynamics | `Cosmos3-Nano` | `forward_dynamics`: image/video + `action_path` JSON → video, `domain_name: camera_pose` |
| Optional visuals | `Cosmos3-Super` | **Only** `text2image`, `text2video`, `image2video` in the framework CLI. No action modes. |

**Consequence:** Super cannot run jobs B or C. The paper's action results for Super were post-trained variants, not the released base checkpoint `[framework docs]`. Use Super only for optional cinematic clips (for example, image-to-video from a rendered frame). Those clips are not pose-locked and are not evaluated by job B.

### A. Transfer: physics → photoreal video

- **Input:** one control video rendered by our own renderer plus a caption. Pick one control type per run (depth is the natural fit for Earth/Moon/spacecraft geometry `[infer]`).
- **Settings:** use the recipe's own flags (`guidance 1`, `control_guidance 1`, fixed four-step SDE schedule). The paper's 50-step transfer settings do not apply to this checkpoint.
- **Constraint:** supply enough control frames for the requested frame count, and keep `num_frames`, `max_frames`, and `num_video_frames_per_chunk` equal when adapting the spec `[framework docs]`.
- **Prompt:** set `prompt_upsampling: false` and write the caption yourself, or generate it with Claude beforehand.
- **Why it is the safest job:** structure comes from the control video, so out-of-distribution scale hurts least.

### B. Inverse dynamics: video → pose (the headline result)

- **Input:** video from job A, plus real Artemis/Orion footage if available `[verify]`.
- **Output:** action sequence, integrated by the evaluator into a trajectory.
- **Caveat 1:** the shipped inverse-dynamics examples cover `av` and `robot` domains only. Try `camera_pose` first and fall back to `av` if it is unsupported `[verify]`.
- **Caveat 2:** the paper's inverse-dynamics evaluation used 6-second driving clips at 10 FPS. Metric translation in space is likely unreliable, so Sim(3)-align (Umeyama) before reporting translation and treat rotation error as primary `[infer]`.
- **Metrics:** RRE (°), RTE (m), ATE (m), as in the paper `[paper]`.

### C. Forward dynamics: planned trajectory → predicted view

- **Input:** first frame plus the planned pose sequence, one row per action step, one column per raw action dimension `[framework docs]`. Action length is 16–400 frames `[model card]`.
- **Evaluation:** run B on the output and check the recovered path against the commanded path.
- **Honest framing:** Cosmos sees only the pose path. It does not know a burn happened.

### Throughput

Generation is offline. The paper's Figure 16 shows roughly 108–115 s on one B200 for a Nano 720p text-to-video clip `[paper]` `[verify]` (figure text extracted garbled). Plan on minutes per clip, queue jobs, and cache everything.

## 8. Compute plan: 8× B200 (FAU HPC)

`[verify]` scheduler, partition, GRES name, wall-time limits, GPU memory per device, and whether compute nodes have internet. Assumed SLURM, no internet on compute nodes.

Nano's weights take about 32 GB of GPU memory `[framework docs]`, so one Nano job fits on one GPU with room for activations. The natural layout is **many single-GPU Nano jobs in parallel**, not one big sharded job.

| GPUs | Role |
|---|---|
| 0–5 | Nano workers: Sim-Transfer, inverse dynamics, forward dynamics, and `k` sweeps (job array, one GPU each) |
| 6 | PINN training and baselines |
| 7 | Control-video rendering and batched Monte Carlo (stretch) |
| (on demand) | 4–8 GPU `torchrun` allocation for optional Super clips, separate from the above |

**HPC practicalities**
- Install and download on a node with internet, then run batch jobs with caches on shared storage (`HF_HOME`, `UV_CACHE_DIR`). The framework docs ask for about 150 GiB free.
- System packages the docs install with `sudo apt-get` (ffmpeg, git-lfs, libgl1, libglib2.0-0) need modules or a container on HPC.
- CUDA 13.0 build needs driver 580 or newer, otherwise use the CUDA 12.8 group `[framework docs]`.
- Render control videos headless on a node. The Three.js UI runs on a laptop and reads synced artifacts.
- Keep a no-network demo fallback: pre-rendered clips, cached metrics, recorded UI run.
- Ready-to-edit scripts are in `slurm/`.

## 9. Evaluation plan

| Question | Method | Reported |
|---|---|---|
| Is the propagator right? | Compare to Horizons/SPICE; conservation tests | Position error vs time |
| Does the PINN find real physics? | Hold out time windows; compare to SINDy and MLP | Residual reduction, equation, physical plausibility |
| Does Cosmos preserve geometry? | Job B on job A output, Sim(3) aligned | RRE, RTE, ATE per window and per `k` |
| How far can the scale be pushed? | Sweep `k` | Error vs `k`, point where it breaks |
| Does video hallucinate? | Compare generated frames to real imagery at known timestamps (apparent Earth size, phase) `[verify]` data | Where it holds and where it fails |
| Zero-shot gap | Compare against the paper's driving/camera numbers | Honest delta |

The `k` sweep and the failure analysis are the "difficulties we faced" story. Log every run with model, seed, `k`, pointing law, and settings.

## 10. Repository layout

```
residual/
├─ architecture.md
├─ crates/ { ingest frames dynamics residual posebuilder evaluator schema ipc }
├─ ml/
│  ├─ pinn/        training, pruning, symbolic extraction, baselines
│  └─ cosmos/     job runner, prompt upsampler, mode wrappers
├─ render/        headless control-video renderer
├─ ui/            Tauri + Three.js
├─ data/          manifests only; large files on shared storage
├─ jobs/          SLURM scripts, job manifests
└─ results/       metrics, clips, equations
```

## 11. Milestones

| Phase | Deliverable | Gate |
|---|---|---|
| 0 (first hours) | Confirm Artemis/Orion data, formats, release dates; confirm HPC access and weights staged | Go/no-go on data; fallback to other lunar trajectories in Horizons |
| 1 | CUDA propagator + frame tests + residual table | Matches Horizons within tolerance |
| 2 | PINN + baselines | Equation produced, comparison table |
| 3 | Pose builder + control renderer + job A on one window | One transfer clip with correct geometry |
| 4 | Jobs B and C + evaluator | First RRE/RTE/ATE numbers, `k` sweep |
| 5 | UI + cached clips + fallback demo | Full run without network |
| Stretch | Monte Carlo cloud, vehicle/mission schema, Super clips | Only after phase 4 |

## 12. Roles

| Person | Owns |
|---|---|
| **1. CUDA/physics lead** | `src/kernels`, `frames`, `ingest`, `posebuilder`, integrator tests |
| **2. Data and validation lead** | Data sourcing, time systems, burn masking, `evaluator`, Horizons comparison |
| **3. ML lead** | PINN, baselines, Cosmos job runner, SLURM scripts |
| **4. Viz and demo lead** | Control renderer, UI, clip cache, fallback demo, slides |

## 13. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Space is out of distribution; scale far beyond training range `[paper]` action data is egocentric, driving, robotics, camera motion | Jobs B and C may fail | Scaled virtual camera; `k` sweep; lead with job A; report failures as findings |
| No spacecraft attitude in the data `[verify]` | Pose is partly synthetic | Explicit pointing law, labeled in every result |
| Inverse-dynamics metric scale unreliable `[infer]` | Translation error misleading | Sim(3) alignment; rotation as primary |
| Best paper numbers used domain post-training `[paper]` Tab. 18 | Zero-shot weaker | Do not claim parity; present as zero-shot |
| Maneuvers pollute residuals | Junk symbolic equation | Mask or model burns |
| Offline generation latency | Cannot be live | Pre-render, cache, async queue |
| Normalizers or domain IDs not exposed as assumed `[verify]` | Wrong action encoding | Read `cosmos-framework` before writing `posebuilder` |
