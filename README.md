# GatorHacks-V: CUDA Earth-Moon Multibody Propagator

GatorHacks-V is a high-performance orbital mechanics workbench for lunar trajectory propagation. It combines a CUDA-accelerated propagator, a Rust data/validation core, a PINN/SINDy/MLP residual-correction pipeline, and a Cosmos-based render/validation layer.

## 🚀 Project Overview

The project implements the restricted Earth-Moon multibody problem with a symplectic integrator on the GPU, allowing propagation of large numbers of test particles (Monte Carlo, debris clouds, etc.). A machine-learning pipeline learns the residual between the numerical propagator and reference ephemerides (JPL Horizons or SPICE) and produces a correction term.

### Key Technical Features

- **CUDA-Accelerated Propagator**: A custom C++/CUDA kernel with a grid-stride loop for scalability across particles and GPU architectures.
- **Symplectic + Adaptive Integration**: Velocity-Verlet (kick-drift-kick) in both CUDA and Rust, plus an adaptive Dormand–Prince RK45 propagator in the Rust core.
- **High-Fidelity Physics**: J2 perturbation with configurable spin axis, gravitational softening for close encounters, and impulsive burn support.
- **Rust Validation Core**: A type-safe workspace (`dynamics`, `ingest`, `frames`, `residual`, `schema`, `ipc`, `posebuilder`, `evaluator`) that keeps the CPU reference aligned with the GPU implementation and owns the Cosmos job/pose contracts.
- **Symbolic Correction Pipeline**:
  - **PINN (Physics-Informed Neural Network)**: predicts position residuals with a velocity (derivative) consistency loss.
  - **MLP (PyTorch)**: plain feed-forward baseline.
  - **SINDy**: sparse polynomial baseline.
  - **Linear regression**: linear baseline.
- **Cosmos Integration**: Python workers (`ml/cosmos/`) implement transfer, inverse-dynamics, and forward-dynamics jobs over a file contract. `scripts/pinn_to_cosmos.py` reconstructs a PINN-corrected trajectory, runs the Rust `posebuilder`, and optionally submits 9D actions to vLLM-Omni.
- **Tauri + Three.js Viewer**: visualizes the trajectory and the PINN/SINDy metrics and expression, plus a Sun-direction marker that tracks the current epoch (its distance is compressed to fit the lunar scene; the UI label says "not to scale"). The orbit view renders guide-style visuals: a dimmed full reference path with a bright growing traveled trail, exaggerated Earth/Moon markers and labels, a deterministic starfield, and a static "Top view" framing the arc face-on down the orbit normal.

## 📊 Results & Analysis

Current results on the 30-day Moon reference arc (`data/residual_table.csv`, 4321 samples, 0 burn-masked) with the default pipeline. The train/test split is a **seeded random interpolation split** (`--test-fraction 0.2`); a temporal `split_by_time` helper remains available for extrapolation studies.

| Model | Train R² | Test R² | Behavior |
| :--- | :---: | :---: | :--- |
| **MLP** | 0.995 | **0.995** | Best fit; generalizes well on the interpolation split. |
| **PINN** | 0.867 | 0.859 | Physics-consistent correction; competitive and stable. |
| **SINDy (degree 1)** | 0.374 | 0.350 | Linear trend only; identical to linear regression at degree 1. |
| **Linear regression** | 0.374 | 0.350 | Same as degree-1 SINDy. |

Notes:

- These numbers reflect **interpolation**, not temporal extrapolation. On a temporal holdout the residuals are non-stationary and all models previously scored negative R²; use `split_by_time` when evaluating extrapolation.
- The PINN's velocity loss uses the chain-rule factor `σ_pos / (σ_t · σ_vel)` so `d(pos_norm)/d(t_norm)` is compared in standardized-velocity units.
- SINDy at degree 1 collapses to linear regression; raise the degree with a stronger sparsity threshold if a distinct symbolic term is desired.
- Runs are reproducible: `run()` seeds `random`, NumPy, and PyTorch.
- `ml/pinn/pinn_result.json` holds metrics; `ml/pinn/pinn_predictions.csv` holds per-sample residuals and predictions; `ml/pinn/pinn_models.pt` holds the checkpoints (gitignored).

## 🛠 Technical Guide

### Repository Layout

| Path | Contents |
| --- | --- |
| `src/kernels/` | FP64 CUDA propagator (`multibody_solver`) |
| `crates/` | Rust workspace: `dynamics`, `ingest`, `frames`, `residual`, `schema`, `ipc`, `posebuilder`, `evaluator`, `mission` |
| `ml/pinn/` | PINN, MLP, SINDy, linear regression, dataset/feature helpers, config |
| `ml/cosmos/` | Cosmos transfer / inverse-dynamics / forward-dynamics workers and vLLM-Omni client |
| `ml/validation/` | Reference comparison helpers (`horizons.py`) |
| `scripts/` | Horizons/SPICE fetch, frame conversion, Artemis data, `pinn_to_cosmos.py`, mock Cosmos worker |
| `data/` | Raw/reference CSV outputs and `manifest.json` provenance |
| `render/`, `hpc/`, `tauri-src/` | Control-video renderer, HPC launch scripts, Tauri viewer |
| `tests/` | Python and C++/CUDA tests |
| `ARCHITECTURE.md` | System design and contracts |

### Prerequisites

- Python 3.11+ and [uv](https://docs.astral.sh/uv/)
- Rust stable via `rustup`
- CMake 3.24+ and a C++17 compiler
- CUDA Toolkit 12.8+ (for the CUDA path)

### One-command setup

```bash
./setup_and_run.sh              # fetch → residual → train (Horizons)
./setup_and_run.sh --spice      # use SPICE kernels instead of Horizons
./setup_and_run.sh --skip-fetch # reuse existing data/
./setup_and_run.sh --test       # run the test suite after training
```

`setup_and_run.sh` installs the `dev,data,gpu` extras (plus `spice` with `--spice`), builds the Rust workspace, fetches data, generates the residual table, and trains the models.

### Manual setup & build

```bash
# Python environment (torch is in the optional "gpu" extra)
uv sync --extra dev --extra data --extra gpu
# SPICE support (optional)
uv sync --extra spice

# CUDA build
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build -j

# Rust workspace
cargo build --workspace
```

### Data & ML Pipeline

1. **Fetch Reference**: `make fetch-data` (JPL Horizons) or `make fetch-spice` (offline SPICE kernels).
2. **Generate Residuals**: `make residual` (propagates and computes `reference − propagated`).
3. **Train Correction**: `make train` (trains PINN, MLP, SINDy, linear regression).
4. **Build a Cosmos job** (optional):
   ```bash
   uv run --extra data python scripts/pinn_to_cosmos.py \
     --reference data/reference.csv \
     --predictions ml/pinn/pinn_predictions.csv \
     --out-dir jobs/pinn-cosmos --frames 121 --fps 24
   ```

SPICE ingestion is Python-only (`scripts/fetch_spice.py` with `spiceypy`); it produces the same canonical `data/reference.csv` as Horizons, so the residual and ML stages are source-agnostic.

### Generalization Roadmap

The correction models currently train on one 30-day Moon arc and report a random-split R², which measures interpolation rather than transfer. [`docs/generalization-plan.md`](docs/generalization-plan.md) lays out the path to a genuinely generalized model: temporal and mission-level holdouts, state-based features instead of time, acceleration-residual targets, multi-arc training data, uncertainty/OOD gating, and a benchmark suite. It also lists the P0 work that should be done before quoting any model as "generalized."

### Transfer trajectory data

The transfer study now has separate trajectory sources for the Artemis I and
Artemis II Moon and Orion arcs, plus an LRO arc over the Artemis II window.
LRO adds a different spacecraft trajectory near the Moon, which helps test
whether a correction generalizes across vehicle dynamics and orbital states.
It does not by itself teach the model realistic rocket forces: the Moon/Orion
arcs are still a small and highly correlated set, and transfer claims need
whole-trajectory holdouts. Fetch or refresh the LRO reference with
`make fetch-lro-transfer`; the default window is 2026-04-02 through
2026-04-10. The source is JPL Horizons target `-85` (LRO). These files are
inputs for a separate experiment and must not replace the Artemis-specific
model weights or viewer predictions.

Generate a reproducible synthetic cislunar rocket dataset with
`make synthetic-rocket-data`. The default run creates 12 four-day arcs with
finite thrust burns, rocket-equation propellant mass flow, Earth J2, and solar
radiation pressure. Rows pair the state and vehicle/control inputs with a
truth-minus-baseline acceleration label. The training model applies known
thrust as `F/m` and learns the remaining J2/SRP residual from state and vehicle
features. The CSV and a manifest containing
arc-specific burn schedules are written under `data/`; override settings with
`make synthetic-rocket-data SYNTHETIC_ARGS="--arcs 20 --seed 17"`. This is a
separate research dataset. Artemis I/II stay outside synthetic training and
serve as mission-level transfer tests. The existing time-input PINN cannot
train on this table directly; it needs a separate state-conditioned
acceleration-model trainer. Burn rows are sparse, so training uses arc-level
splits and burn-weighted loss. Prepare a separate 2025 Moon/Sun residual arc
with `make fetch-moon-training-data`; Artemis data is not part of this step.
Then train with `make train-synthetic-dynamics`.
Run the untouched Artemis arcs with `make evaluate-artemis-transfer`. Orion
scores remain diagnostic until actual thrust and mass histories are available.
`make evaluate-synthetic-rollout` compares corrected and physics-only free
propagation on the held-out synthetic arcs. Training and evaluation artifacts
write under `ml/pinn/synthetic_transfer/` and leave production checkpoints
untouched.

The current pilot is not ready for Artemis correction: its high all-sample
R² is dominated by the explicit thrust term, while learned coast correction
R² is about 0.006 and free-rollout results are inconsistent across the two
held-out arcs. Both Artemis Moon arcs trigger the correction gate; the Orion
acceleration checks are inconclusive without burn and mass histories.

### Satellite path prediction experiment

For direct path prediction, use a separate sequence model instead of the
Moon-residual correction PINN. `make synthetic-satellite-data` generates
Earth LEO/MEO/high-orbit and low/high lunar-orbit tracks under Earth, Moon, and
Sun gravity, Earth/Moon J2, and solar radiation pressure. Each row contains a
state, body ephemerides, cadence, and vehicle properties; the target is the
next observed position and velocity. The trainer splits by complete orbit arc
within each regime, so neighboring samples from a trajectory cannot leak into
validation or test. Run `make train-satellite-path-model` and then
`make evaluate-satellite-path-model` for held-out free rollouts and an external
LRO path comparison. Training/checkpoints live in
`ml/pinn/satellite_path_model/`; the existing PINN and Artemis weights and
viewer predictions are not used or overwritten. The synthetic generator does
not yet model stationkeeping burns, and LRO vehicle mass/area are placeholders
in the external evaluation. A useful next training-data step is to replace
those placeholders with mission vehicle data and add maneuver-tagged satellite
arcs; Artemis remains a test set, never training input. In the current 40-arc
pilot, the model beats a constant-velocity baseline but still has about 64,000
km mean position RMSE over two-day synthetic holdouts and about 392,000 km on
the LRO arc. It is a pipeline prototype, not a useful navigation prediction.
The next iteration should learn short-step acceleration/dynamics and integrate
those predictions, with real multi-mission orbit histories and maneuver
metadata; direct 600-second next-state regression drifts too much.

### Make Targets

| Target | Description |
| --- | --- |
| `make python` | Install Python dependencies (`dev,data,gpu`) |
| `make rust` | Build Rust workspace |
| `make cuda` | Build CUDA C++ project |
| `make fetch-data` | Fetch and convert Horizons data |
| `make fetch-sun` | Fetch a Sun arc (target 10) over the same window |
| `make fetch-spice` | Fetch and convert SPICE data (`SPICE_START`, `SPICE_STOP`, `SPICE_STEP`, `SPICE_MODE`) |
| `make residual` | Generate `data/residual_table.csv` with the circular analytic ephemeris |
| `make residual-tabulated` | Generate `data/residual_table_tabulated.csv` from the measured Earth-Moon geometry (`--preserve-distance` reference) and print the analytic/tabulated RMS comparison |
| `make residual-tabulated-sun` | Same as `residual-tabulated` but also attaches the Sun as a tidal third body (`data/reference_sun.csv`) |
| `make train` | Train models on `data/residual_table.csv` (or `DATA=...`); writes `ml/pinn/pinn_result.json` |
| `make ui-data` | Rebuild the Tauri viewer bundle in `tauri-src/src/data/` (embeds the per-epoch Sun bearing from `data/reference_sun.csv` when present) |
| `make ensemble-data` | Regenerate the demo Monte-Carlo cloud (`ensemble.json`) |
| `make test` | Full test suite (CUDA + Rust + Python) |
| `make test-cpu` | CPU-only validation (`SKIP_CUDA_TESTS=1`) |
| `make test-cuda` | Full CUDA validation |
| `make bench` | Build and run the full CUDA benchmark suite |
| `make bench-quick` | Short benchmark smoke run (0.2 s per case) |
| `make bench-json` | Write JSON results to `results/bench.json` (`BENCH_OUT=...`) |
| `make bench-list` | List registered benchmark cases without running them |
| `make bench-kernel` | Kernel-only throughput series |
| `make bench-api` | Full `mbs_propagate` API series |
| `make bench-transfer` | Host→device and device→host copy bandwidth |
| `make bench-physics` | J2, ephemeris, force-model, softening, and epoch series |
| `make bench-trajectory` | Sustained trajectory-write series |
| `make bench-burns` | Burn-count overhead series |
| `make bench-timestep` | Step-size scaling series |
| `make bench-backends` | Direct CUDA vs Rust comparison on identical states (requires CUDA + cargo) |

Benchmarks use CUDA events and require the CUDA toolkit plus a GPU. Focused
targets accept `BENCH_ARGS`, and `make bench` also accepts `BENCH_FILTER`; see
[`benchmarks/README.md`](benchmarks/README.md) for the full case inventory.

### Tests

```bash
# Python (skip the CUDA shared-library tests requires SKIP_CUDA_TESTS=1)
SKIP_CUDA_TESTS=1 uv run --extra dev --extra data --extra gpu pytest -q tests

# Rust
cargo test --workspace
```

`tests/test_horizons.py` loads a native CUDA `.so` via `ctypes`; on machines without a compatible driver it segfaults, so use `SKIP_CUDA_TESTS=1` (the `test-cpu` target does this).
