.PHONY: all cuda rust python test fetch-data fetch-spice residual train test-cpu test-cuda tauri bench clean ui-data

BUILD_DIR ?= build
BUILD_CPU_DIR ?= target/build-cpu
HORIZONS_ARGS ?=
CONVERT_ARGS ?=
SPICE_START ?= 2022-08-01
SPICE_STOP ?=
SPICE_STEP ?= 600
SPICE_MODE ?= moon
SPICE_ARGS ?=
DATA ?= data/residual_table.csv
ARGS ?= $(if $(wildcard $(DATA)),--data $(DATA),)
BENCH_ARGS ?= --benchmark_min_time=1s

all: cuda rust python

cuda:
	cmake -S . -B $(BUILD_DIR)
	cmake --build $(BUILD_DIR) -j

rust:
	@if [ -f Cargo.toml ]; then cargo build --workspace; else echo "No Cargo.toml found, skipping Rust build."; fi

tauri:
	cd tauri-src/src-tauri && cargo build

python:
	uv sync --extra dev --extra data --extra gpu

ui-data:
	uv run python scripts/make_ui_data.py

# Download a short Earth-centred Moon arc with JPL Horizons and convert it
# into the solver's metres/seconds barycentric frame.  Writes:
#   data/raw_moon.csv, data/raw_moon.meta.json,
#   data/reference.csv, data/manifest.json
fetch-data:
	uv run --extra data python scripts/fetch_horizons.py $(HORIZONS_ARGS)
	uv run --extra data python scripts/convert_reference.py $(CONVERT_ARGS)

# Fetch an ephemeris arc from SPICE kernels (offline) and convert it.
# Requires `uv sync --extra spice` and SPICE_KERNELS (or --kernel).
#   make fetch-spice SPICE_START="2022-08-01" SPICE_STOP="2022-09-01" SPICE_STEP=3600
fetch-spice:
	uv run --extra spice python scripts/fetch_spice.py \
	  --start "$(SPICE_START)" $(if $(SPICE_STOP),--stop "$(SPICE_STOP)",) \
	  --step $(SPICE_STEP) $(SPICE_ARGS)
	uv run --extra data python scripts/convert_reference.py \
	  --raw data/raw_spice.csv --mode $(SPICE_MODE) $(CONVERT_ARGS)

# Reference trajectory -> residual table (dr, dv per sample).
data/reference.csv:
	$(MAKE) fetch-data

residual: data/reference.csv
	cargo run -p residual --example residual_pipeline -- data/reference.csv data/residual_table.csv

# Train PINN/MLP/SINDy.  Uses data/residual_table.csv when present, otherwise
# falls back to the synthetic demo.  Override with e.g.
#   make train DATA=other.csv ARGS="--data other.csv --epochs 50"
train:
	uv run --extra data --extra gpu python -m ml.pinn.run $(ARGS)

test: cuda python
	uv run --extra dev --extra data --extra gpu pytest -q tests ml
	@if [ -f $(BUILD_DIR)/CTestTestfile.cmake ]; then ctest --test-dir $(BUILD_DIR) --output-on-failure; else echo "No CTest file found, skipping C++ tests."; fi
	@if [ -f Cargo.toml ]; then cargo test --workspace; fi

# CPU-only validation: configure CMake with CUDA disabled, run Python tests
# with the CUDA library skipped, then Rust tests.
test-cpu:
	SKIP_CUDA_TESTS=1 cmake -S . -B $(BUILD_CPU_DIR) -DBUILD_TESTING=ON -DSKIP_CUDA_TESTS=ON
	SKIP_CUDA_TESTS=1 cmake --build $(BUILD_CPU_DIR) -j
	SKIP_CUDA_TESTS=1 uv run --extra dev --extra data --extra gpu pytest -q tests ml
	@if [ -f $(BUILD_CPU_DIR)/CTestTestfile.cmake ]; then ctest --test-dir $(BUILD_CPU_DIR) --output-on-failure; else echo "No CTest file found, skipping C++ tests."; fi
	cargo test --workspace

# Build and run the CUDA propagator benchmark suite.  Requires the CUDA
# toolkit; pass extra Google Benchmark flags with BENCH_ARGS.
bench:
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
	cmake --build $(BUILD_DIR) -j --target multibody_bench
	mkdir -p results
	$(BUILD_DIR)/benchmarks/multibody_bench $(BENCH_ARGS)

# Full validation including the CUDA kernel and comparison tests.
test-cuda:
	SKIP_CUDA_TESTS=0 cmake -S . -B $(BUILD_DIR) -DBUILD_TESTING=ON
	SKIP_CUDA_TESTS=0 cmake --build $(BUILD_DIR) -j
	SKIP_CUDA_TESTS=0 uv run --extra dev --extra data --extra gpu pytest -q tests ml
	@if [ -f $(BUILD_DIR)/CTestTestfile.cmake ]; then ctest --test-dir $(BUILD_DIR) --output-on-failure; else echo "No CTest file found, skipping C++ tests."; fi
	SKIP_CUDA_TESTS=0 cargo test --workspace

clean:
	rm -rf build target/build-cpu .venv
	@if [ -f Cargo.toml ]; then cargo clean; else rm -rf target; fi
