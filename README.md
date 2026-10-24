# GatorHacks-V

Orbital-mechanics workbench for an Artemis/Orion-style lunar trajectory:
a CUDA/C++ n-body propagator, Rust data/validation crates, a residual
pipeline, and a KAN symbolic-correction branch compared against MLP/SINDy
baselines.

## Repository layout

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

## Prerequisites

- Python 3.11+ and [uv](https://docs.astral.sh/uv/)
- Rust stable via `rustup` (with `rustfmt` and `clippy` for CI parity)
- CMake 3.24+ and a C++17 compiler
- CUDA Toolkit 12.8+ and a compatible driver for the CUDA path
- Network access for `make fetch-data` (JPL Horizons) and for CMake's
  GoogleTest/Google Benchmark downloads

## Setup

```bash
# Python environment with test and data extras (pandas, pyarrow, pytest)
uv sync --extra dev --extra data
```

## Build

CPU-only C++ test build (no CUDA toolkit required):

```bash
cmake -S . -B target/build-cpu -DBUILD_TESTING=ON -DSKIP_CUDA_TESTS=ON
cmake --build target/build-cpu -j
ctest --test-dir target/build-cpu --output-on-failure
```

CUDA build (default `build/` directory):

```bash
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

The CUDA build writes the shared library used by `ml/validation/horizons.py`
to `build/src/kernels/libmultibody_solver_shared.so`.

CUDA benchmarks (event-timed; warm-up excluded):

```bash
make bench
# or
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
cmake --build build -j --target multibody_bench
./build/benchmarks/multibody_bench --benchmark_min_time=1s
```

See `benchmarks/README.md` for the series definitions and counters.

Rust workspace:

```bash
cargo build --workspace
cargo test --workspace
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
```

## Reference data

`scripts/fetch_horizons.py` queries JPL Horizons for the Moon (`301`) relative
to the Earth body centre (`500@399`) over a short window (one day, 10-minute
steps by default). `scripts/convert_reference.py` then:

1. converts km and km/s to metres and m/s;
2. splits the geocentric vector into Earth/Moon barycentric states using
   `mu_earth = 3.986004418e14` and `mu_moon = 4.9048695e12`;
3. normalises the separation to the propagator's nominal `D = 3.844e8 m` and
   rotates the arc so epoch Moon is `+x` and orbital angular momentum is `+z`;
4. writes `data/reference.csv` and `data/manifest.json` (source, frame,
   epoch, cadence, units, constants, and conversion notes).

```bash
make fetch-data
# or with a custom window:
make fetch-data HORIZONS_ARGS="--start '2025-06-01' --stop '2025-06-02' --step 5m"

# equivalent direct invocation
uv run --extra data python scripts/fetch_horizons.py --start 2025-01-01
uv run --extra data python scripts/convert_reference.py
```

Generated files:

- `data/raw_moon.csv` — `t` (s from start), `x,y,z` (km), `vx,vy,vz` (km/s)
- `data/raw_moon.meta.json` — exact Horizons query and JDTDB epoch
- `data/reference.csv` — solver-frame SI states (m, m/s)
- `data/manifest.json` — provenance and conversion metadata

The committed `data/manifest.json` is an example; running the commands above
regenerates it.

## Residual pipeline and KAN

The Rust residual example propagates from the first reference state and writes
per-sample position/velocity residuals:

```bash
make residual
# runs: cargo run -p residual --example residual_pipeline -- \
#         data/reference.csv data/residual_table.csv
```

Train the KAN and baselines. `make kan` uses `data/residual_table.csv` when it
exists, otherwise it falls back to the synthetic demo:

```bash
make kan
make kan KAN_DATA=data/residual_table.csv \
  KAN_ARGS="--data data/residual_table.csv --epochs 100"
```

Results are written to `ml/kan/kan_result.json`.

## Tests

```bash
# CPU-only: CMake with CUDA disabled, Python tests with CUDA skipped, Rust tests
make test-cpu

# Full CUDA build + Python + C++/CUDA + Rust tests
make test-cuda

# Python only
SKIP_CUDA_TESTS=1 uv run --extra dev --extra data pytest -q

# C++/CUDA only
ctest --test-dir build --output-on-failure
```

`SKIP_CUDA_TESTS=1` makes the Python CUDA-dependent tests skip cleanly on
machines without the shared library.

## Make targets

| Target | Description |
| --- | --- |
| `make python` | `uv sync --extra dev --extra data` |
| `make rust` | `cargo build --workspace` |
| `make cuda` | Configure and build the CUDA C++ project in `build/` |
| `make fetch-data` | Fetch Horizons raw data and convert it to `data/reference.csv` |
| `make residual` | Build the reference if needed and write `data/residual_table.csv` |
| `make kan` | Train KAN/MLP/SINDy (residual table if present, else synthetic) |
| `make test-cpu` | CPU-only build and full non-CUDA test suite |
| `make test-cuda` | CUDA build and full test suite |
| `make test` | Original all-in-one test target (`cuda`, Python, CTest, Rust) |
| `make clean` | Remove build directories, `.venv`, and Cargo artifacts |

## CI

- `.github/workflows/rust.yml` runs `cargo fmt --check`, non-blocking
  `cargo clippy -- -D warnings`, `cargo build`, and `cargo test`.
- `.github/workflows/cpp.yml` builds the CPU-only C++ test configuration.
- `.github/workflows/python.yml` runs pytest with `SKIP_CUDA_TESTS=1`.
