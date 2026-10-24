.PHONY: all cuda rust python test fetch-data residual kan test-cpu test-cuda tauri bench clean

BUILD_DIR ?= build
BUILD_CPU_DIR ?= target/build-cpu
HORIZONS_ARGS ?=
CONVERT_ARGS ?=
KAN_DATA ?= data/residual_table.csv
KAN_ARGS ?= $(if $(wildcard $(KAN_DATA)),--data $(KAN_DATA),)
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
	uv sync --extra dev --extra data

# Download a short Earth-centred Moon arc with JPL Horizons and convert it
# into the solver's metres/seconds barycentric frame.  Writes:
#   data/raw_moon.csv, data/raw_moon.meta.json,
#   data/reference.csv, data/manifest.json
fetch-data:
	uv run --extra data python scripts/fetch_horizons.py $(HORIZONS_ARGS)
	uv run --extra data python scripts/convert_reference.py $(CONVERT_ARGS)

# Reference trajectory -> residual table (dr, dv per sample).
data/reference.csv:
	$(MAKE) fetch-data

residual: data/reference.csv
	cargo run -p residual --example residual_pipeline -- data/reference.csv data/residual_table.csv

# Train KAN/MLP/SINDy.  Uses data/residual_table.csv when present, otherwise
# falls back to the synthetic demo.  Override with e.g.
#   make kan KAN_DATA=other.csv KAN_ARGS="--data other.csv --epochs 50"
kan:
	uv run --extra data python -m ml.kan.run $(KAN_ARGS)

test: cuda python
	uv run --extra dev --extra data pytest -q tests ml
	@if [ -f $(BUILD_DIR)/CTestTestfile.cmake ]; then ctest --test-dir $(BUILD_DIR) --output-on-failure; else echo "No CTest file found, skipping C++ tests."; fi
	@if [ -f Cargo.toml ]; then cargo test --workspace; fi

# CPU-only validation: configure CMake with CUDA disabled, run Python tests
# with the CUDA library skipped, then Rust tests.
test-cpu:
	SKIP_CUDA_TESTS=1 cmake -S . -B $(BUILD_CPU_DIR) -DBUILD_TESTING=ON -DSKIP_CUDA_TESTS=ON
	SKIP_CUDA_TESTS=1 cmake --build $(BUILD_CPU_DIR) -j
	SKIP_CUDA_TESTS=1 uv run --extra dev --extra data pytest -q tests ml
	@if [ -f $(BUILD_CPU_DIR)/CTestTestfile.cmake ]; then ctest --test-dir $(BUILD_CPU_DIR) --output-on-failure; else echo "No CTest file found, skipping C++ tests."; fi
	cargo test --workspace

# Build and run the CUDA propagator benchmark suite.  Requires the CUDA
# toolkit; pass extra Google Benchmark flags with BENCH_ARGS.
bench:
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
	cmake --build $(BUILD_DIR) -j --target multibody_bench
	$(BUILD_DIR)/benchmarks/multibody_bench $(BENCH_ARGS)

# Full validation including the CUDA kernel and comparison tests.
test-cuda:
	SKIP_CUDA_TESTS=0 cmake -S . -B $(BUILD_DIR) -DBUILD_TESTING=ON
	SKIP_CUDA_TESTS=0 cmake --build $(BUILD_DIR) -j
	SKIP_CUDA_TESTS=0 uv run --extra dev --extra data pytest -q tests ml
	@if [ -f $(BUILD_DIR)/CTestTestfile.cmake ]; then ctest --test-dir $(BUILD_DIR) --output-on-failure; else echo "No CTest file found, skipping C++ tests."; fi
	SKIP_CUDA_TESTS=0 cargo test --workspace

clean:
	rm -rf build target/build-cpu .venv
	@if [ -f Cargo.toml ]; then cargo clean; else rm -rf target; fi
