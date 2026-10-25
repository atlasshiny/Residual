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
- **Live Mission Runner**: the same viewer can propagate a new mission in-app from an Earth-centred state vector, duration, and impulsive burns. The Tauri backend calls the pure `mission` crate (Rust Dormand–Prince), returns a `points`/`times` bundle, and draws it with a cyan spacecraft marker instead of replaying the static JSON. See [`docs/mission-runner.md`](docs/mission-runner.md).

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

### Live Mission Runner

The desktop viewer's **Mission** card propagates a state vector locally instead of loading the bundled 30-day reference arc. Start the Tauri app (`cd tauri-src && npm run dev`), then:

1. Pick a centre (Earth-centred inertial by default, or Earth–Moon barycentric).
2. Enter duration, position (`km`), and velocity (`km/s`); optionally add up to 8 impulsive burns (`t` in seconds, `Δv` in `m/s`).
3. Press **Run mission**. The backend runs the Rust Dormand–Prince core on a blocking thread and returns a uniformly sampled `points`/`times` bundle; the viewport swaps to a cyan **Craft** marker and the trajectory card shows the live frame and sample count.
4. **Load bundle** restores the static reference arc. In a plain browser (no Tauri IPC) the mission card reports that the desktop backend is required and the bundled viewer keeps working.

Presets fill a 400 km LEO orbit and an approximate translunar injection. The form is saved in `localStorage`.

Current limitations: the propagator uses the analytic circular Earth–Moon ephemeris (no SPICE kernels in Rust), there is no finite-thrust, atmosphere, or launch model yet, and Earth-centred inputs are converted with the Earth's position at the start epoch but do not subtract its ~12 m/s barycentric velocity. Those approximation notes are documented in [`docs/mission-runner.md`](docs/mission-runner.md) and in the crate docs.

### Generalization Roadmap

The correction models currently train on one 30-day Moon arc and report a random-split R², which measures interpolation rather than transfer. [`docs/generalization-plan.md`](docs/generalization-plan.md) lays out the path to a genuinely generalized model: temporal and mission-level holdouts, state-based features instead of time, acceleration-residual targets, multi-arc training data, uncertainty/OOD gating, and a benchmark suite. It also lists the P0 work that should be done before quoting any model as "generalized."

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
