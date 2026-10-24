# GatorHacks-V: CUDA Earth-Moon Multibody Propagator

GatorHacks-V is a high-performance orbital mechanics workbench designed for precise lunar trajectory propagation. It combines a massively parallel CUDA-accelerated propagator, a robust Rust data-validation core, and a symbolic-correction pipeline using Kolmogorov-Arnold Networks (KAN) and SINDy to minimize propagation residuals.

## 🚀 Project Overview

The project implements a restricted multibody problem (Earth-Moon) using a symplectic integrator on the GPU, allowing for the propagation of millions of test particles (e.g., for Monte Carlo analysis or debris clouds) with high efficiency. To address the accumulation of numerical residuals over long arcs, the system employs a machine learning pipeline that learns the symbolic error function of the propagator and applies corrections to the state.

### Key Technical Features

- **CUDA-Accelerated Propagator**: A custom C++/CUDA kernel implementing the multibody equations of motion. It uses a grid-stride loop to ensure scalability across any number of test particles and GPU architectures.
- **Symplectic Integration**: Implementation of the **Velocity-Verlet** (kick-drift-kick) algorithm in both CUDA and Rust, ensuring long-term energy conservation and orbital stability.
- **High-Fidelity Physics**:
    - **J2 Perturbation**: Support for Earth's oblateness with a configurable unit spin axis, allowing the solver to operate in any coordinate frame (e.g., lunar-orbit or equatorial).
    - **Softening**: Adaptive gravitational softening to prevent singularities during close encounters.
    - **Impulsive Burns**: Support for scheduled $\Delta V$ maneuvers applied instantaneously during propagation.
- **Rust Validation Core**: A type-safe Rust workspace (`dynamics`, `residual`, `schema`) that ensures the CPU reference model is perfectly aligned with the GPU implementation.
- **Symbolic Correction Pipeline**:
    - **KAN (Kolmogorov-Arnold Networks)**: A novel neural network architecture that learns additive functions on edges, providing better interpretability than standard MLPs.
    - **SINDy (Sparse Identification of Non-linear Dynamics)**: Used as a baseline to identify the sparse symbolic form of the residuals.
    - **MLP (Multi-Layer Perceptron)**: A vectorized NumPy implementation for baseline comparison.
- **Tauri Viewer**: A modern frontend (Three.js + React/Vue) for visualizing trajectories and monitoring Cosmos-style job status.

## 📊 Results & Analysis

The system was evaluated using 30 days of real JPL Horizons data for the Moon, producing a residual table of position and velocity errors. We tested three models to learn the residual function and extrapolate the error for a future window.

### Extrapolation Performance (Test $R^2$)

| Model | Training $R^2$ | Test $R^2$ (Extrapolation) | Behavior |
| :--- | :---: | :---: | :--- |
| **SINDy (Quadratic)** | 0.982 | **0.966** | **Generalizes best; captures the linear/quadratic drift.** |
| **KAN** | 0.991 | -0.142 | Overfits the training arc; fails to extrapolate. |
| **MLP** | 0.994 | -0.885 | High training accuracy but poor extrapolation. |

**Conclusion**: For the 30-day Moon residual dataset, the residuals are nearly linear/quadratic. While KAN and MLP can perfectly memorize the training data, **SINDy's sparse symbolic nature** allows it to generalize to the extrapolation window, making it the ideal choice for long-term trajectory correction.

---

## 🛠 Technical Guide

### Repository Layout

| Path | Contents |
| --- | --- |
| `src/kernels/` | FP64 CUDA propagator (`multibody_solver`) |
| `crates/` | Rust workspace: `dynamics`, `ingest`, `frames`, `residual`, `schema`, `ipc`, `posebuilder`, `evaluator` |
| `ml/kan/` | KAN, MLP, and SINDy training plus residual-dataset helpers |
| `ml/validation/` | Reference comparison helpers (`horizons.py`) |
| `scripts/` | Horizons fetch and frame-conversion scripts |
| `data/` | Raw/reference CSV outputs and `manifest.json` provenance |
| `tests/` | Python and C++/CUDA tests |
| `ARCHITECTURE.md` | System design and contracts |

### Prerequisites

- Python 3.11+ and [uv](https://docs.astral.sh/uv/)
- Rust stable via `rustup`
- CMake 3.24+ and a C++17 compiler
- CUDA Toolkit 12.8+ (for the CUDA path)

### Setup & Build

```bash
# Python environment
uv sync --extra dev --extra data

# CUDA build
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build -j

# Rust workspace
cargo build --workspace
```

### Data & ML Pipeline

1. **Fetch Reference**: `make fetch-data` (queries JPL Horizons).
2. **Generate Residuals**: `make residual` (propagates and calculates error).
3. **Train Correction**: `make kan` (trains KAN, MLP, and SINDy).

### Make Targets

| Target | Description |
| --- | --- |
| `make python` | Install Python dependencies |
| `make rust` | Build Rust workspace |
| `make cuda` | Build CUDA C++ project |
| `make fetch-data` | Fetch and convert Horizons data |
| `make residual` | Generate `data/residual_table.csv` |
| `make kan` | Train ML models and write results to `ml/kan/kan_result.json` |
| `make test-cpu` | Run non-CUDA test suite |
| `make test-cuda` | Run full test suite (requires GPU) |
