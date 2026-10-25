.PHONY: all cuda rust python test fetch-data fetch-sun fetch-spice fetch-lro-transfer fetch-moon-training-data synthetic-rocket-data train-synthetic-dynamics evaluate-artemis-transfer evaluate-synthetic-rollout residual residual-tabulated residual-tabulated-sun train evaluate-pinn-splits test-cpu test-cuda tauri bench clean ui-data ensemble-data

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
BENCH_MIN_TIME ?= 1s
BENCH_ARGS ?= --benchmark_min_time=$(BENCH_MIN_TIME)
BENCH_FILTER ?=
BENCH_OUT ?= results/bench.json
BENCH_BIN ?= $(BUILD_DIR)/benchmarks/multibody_bench
BENCH_FILTER_ARG = $(if $(strip $(BENCH_FILTER)),--benchmark_filter='$(BENCH_FILTER)',)
BENCH_RUN_ARGS = $(BENCH_FILTER_ARG) $(BENCH_ARGS)

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

ensemble-data:
	uv run python scripts/make_ensemble_data.py

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

# Fetch a Sun arc (target 10) over the same window as the Moon arc.
fetch-sun:
	uv run --extra data python scripts/fetch_horizons.py \
	  --target 10 --start "$(SPICE_START)" $(if $(SPICE_STOP),--stop "$(SPICE_STOP)",) \
	  --step $(SPICE_STEP) --out data/raw_sun.csv $(HORIZONS_ARGS)

# Fetch LRO over the Artemis II interval and convert it into the solver's
# barycentric frame using the matching Moon arc. This is transfer-study data;
# it does not change the production PINN or Artemis viewer outputs.
LRO_START ?= 2026-04-02 02:00
LRO_STOP ?= 2026-04-10 23:54
LRO_STEP ?= 10m
fetch-lro-transfer:
	uv run --extra data python scripts/fetch_horizons.py \
	  --target -85 --start "$(LRO_START)" --stop "$(LRO_STOP)" \
	  --step $(LRO_STEP) --out data/raw_lro_artemis2_window.csv $(HORIZONS_ARGS)
	uv run --extra data python scripts/convert_spacecraft.py \
	  data/raw_lro_artemis2_window.csv data/raw_moon_artemis2.csv \
	  data/reference_lro_artemis2_window_preserve.csv --preserve-distance

# Reproducible synthetic finite-burn/mass-flow trajectories for transfer-model
# research. Artemis I/II remain held-out external validation cases.
SYNTHETIC_ARGS ?=
synthetic-rocket-data:
	uv run --extra data python scripts/generate_synthetic_rocket_data.py $(SYNTHETIC_ARGS)

MOON_TRAIN_START ?= 2025-01-01
MOON_TRAIN_STOP ?= 2025-01-31
MOON_TRAIN_STEP ?= 10m
fetch-moon-training-data:
	uv run --extra data python scripts/fetch_horizons.py --target 301 \
	  --start "$(MOON_TRAIN_START)" --stop "$(MOON_TRAIN_STOP)" \
	  --step $(MOON_TRAIN_STEP) --out data/raw_moon_training.csv
	uv run --extra data python scripts/fetch_horizons.py --target 10 \
	  --start "$(MOON_TRAIN_START)" --stop "$(MOON_TRAIN_STOP)" \
	  --step $(MOON_TRAIN_STEP) --out data/raw_sun_moon_training.csv
	uv run --extra data python scripts/convert_reference.py \
	  --raw data/raw_moon_training.csv --out data/reference_moon_training.csv \
	  --manifest data/reference_moon_training.manifest.json --preserve-distance \
	  --sun-raw data/raw_sun_moon_training.csv --sun-out data/reference_sun_moon_training.csv
	uv run --extra data python scripts/build_ephemeris_accel_residuals.py \
	  --target moon --reference data/reference_moon_training.csv \
	  --sun data/reference_sun_moon_training.csv --out data/moon_training_accel_residuals.csv

# Train a new dynamics-residual model in a separate directory. This does not
# read or overwrite the existing PINN / Artemis checkpoints.
SYNTHETIC_TRAIN_ARGS ?= --moon-data data/moon_training_accel_residuals.csv
train-synthetic-dynamics:
	uv run --extra data --extra gpu python scripts/train_synthetic_dynamics.py $(SYNTHETIC_TRAIN_ARGS)

# Rebuild Artemis test labels from the held-out Horizons arcs. Never trains
# on or modifies the Artemis-specific prediction/model files.
prepare-artemis-tests:
	uv run --extra data python scripts/convert_reference.py \
	  --raw data/raw_moon_artemis1.csv --out data/reference_artemis1_preserve.csv \
	  --manifest data/reference_artemis1_preserve.manifest.json --preserve-distance \
	  --sun-raw data/raw_sun_artemis1.csv --sun-out data/reference_sun_artemis1.csv
	uv run --extra data python scripts/convert_reference.py \
	  --raw data/raw_moon_artemis2.csv --out data/reference_artemis2_preserve.csv \
	  --manifest data/reference_artemis2_preserve.manifest.json --preserve-distance \
	  --sun-raw data/raw_sun_artemis2.csv --sun-out data/reference_sun_artemis2.csv
	uv run --extra data python scripts/convert_spacecraft.py \
	  data/raw_artemis1.csv data/raw_moon_artemis1.csv \
	  data/reference_artemis1_spacecraft_preserve.csv --preserve-distance
	uv run --extra data python scripts/convert_spacecraft.py \
	  data/raw_artemis2.csv data/raw_moon_artemis2.csv \
	  data/reference_artemis2_spacecraft_preserve.csv --preserve-distance
	uv run --extra data python scripts/build_ephemeris_accel_residuals.py \
	  --target moon --reference data/reference_artemis1_preserve.csv \
	  --sun data/reference_sun_artemis1.csv --out data/artemis1_moon_accel_test.csv
	uv run --extra data python scripts/build_ephemeris_accel_residuals.py \
	  --target moon --reference data/reference_artemis2_preserve.csv \
	  --sun data/reference_sun_artemis2.csv --out data/artemis2_moon_accel_test.csv
	uv run --extra data python scripts/build_ephemeris_accel_residuals.py \
	  --target spacecraft --reference data/reference_artemis1_spacecraft_preserve.csv \
	  --moon data/reference_artemis1_preserve.csv --sun data/reference_sun_artemis1.csv \
	  --out data/artemis1_orion_accel_diagnostic.csv
	uv run --extra data python scripts/build_ephemeris_accel_residuals.py \
	  --target spacecraft --reference data/reference_artemis2_spacecraft_preserve.csv \
	  --moon data/reference_artemis2_preserve.csv --sun data/reference_sun_artemis2.csv \
	  --out data/artemis2_orion_accel_diagnostic.csv

# External evaluation only; these Artemis arcs are never used in training.
ARTEMIS_EVAL_ARGS ?= --data data/artemis1_moon_accel_test.csv data/artemis2_moon_accel_test.csv data/artemis1_orion_accel_diagnostic.csv data/artemis2_orion_accel_diagnostic.csv
evaluate-artemis-transfer: prepare-artemis-tests
	uv run --extra data --extra gpu python scripts/evaluate_state_acceleration_model.py $(ARTEMIS_EVAL_ARGS)

evaluate-synthetic-rollout:
	uv run --extra data --extra gpu python scripts/evaluate_synthetic_rollout.py

# Train a separate next-state satellite predictor on synthetic Earth/lunar
# orbit tracks; whole arcs are held out and LRO remains an external test.
SATELLITE_ARGS ?=
synthetic-satellite-data:
	uv run --extra data python scripts/generate_synthetic_satellite_data.py $(SATELLITE_ARGS)

SATELLITE_TRAIN_ARGS ?=
train-satellite-path-model:
	uv run --extra data --extra gpu python scripts/train_satellite_path_model.py $(SATELLITE_TRAIN_ARGS)

SATELLITE_EVAL_ARGS ?=
evaluate-satellite-path-model:
	uv run --extra data --extra gpu python scripts/evaluate_satellite_path_model.py $(SATELLITE_EVAL_ARGS)

# Reference trajectory -> residual table (dr, dv per sample).
# Uses the circular Earth-Moon ephemeris, matching the CUDA kernel.
data/reference.csv:
	$(MAKE) fetch-data

residual: data/reference.csv
	cargo run -p residual --example residual_pipeline -- data/reference.csv data/residual_table.csv

# Same pipeline but driven by the measured Earth-Moon geometry.  The reference
# must keep the real separation: the default reference is rescaled onto the
# circular model's radius, which throws away the geometry the table needs.
PRESERVED_REFERENCE ?= data/reference_preserve.csv

$(PRESERVED_REFERENCE): data/raw_moon.csv data/raw_sun.csv
	uv run --extra data python scripts/convert_reference.py \
	  --raw data/raw_moon.csv --out $(PRESERVED_REFERENCE) \
	  --manifest data/reference_preserve.manifest.json --preserve-distance \
	  --sun-raw data/raw_sun.csv --sun-out data/reference_sun.csv

data/raw_moon.csv:
	$(MAKE) fetch-data

data/raw_sun.csv:
	$(MAKE) fetch-sun

residual-tabulated: $(PRESERVED_REFERENCE)
	cargo run -p residual --example residual_pipeline -- \
	  $(PRESERVED_REFERENCE) data/residual_table_tabulated.csv \
	  --ephemeris tabulated --compare

# Same as residual-tabulated but also attaches the Sun as a tidal third body.
residual-tabulated-sun: $(PRESERVED_REFERENCE) data/reference_sun.csv
	cargo run -p residual --example residual_pipeline -- \
	  $(PRESERVED_REFERENCE) data/residual_table_tabulated_sun.csv \
	  --ephemeris tabulated --sun data/reference_sun.csv --compare

# Train PINN/MLP/SINDy.  Uses data/residual_table.csv when present, otherwise
# falls back to the synthetic demo.  Override with e.g.
#   make train DATA=other.csv
#   make train ARGS="--epochs 50 --targets y"
RESIDUAL_TABLE ?= $(firstword $(wildcard data/residual_table.csv))
train:
	uv run --extra data --extra gpu python -m ml.pinn.run $(if $(RESIDUAL_TABLE),--data $(RESIDUAL_TABLE)) $(ARGS)

# Re-run the models across split modes and seeds, then aggregate mean/spread.
# Group splits require a multi-arc table with an arc_id column.
SPLIT_EVAL_ARGS ?= --splits random,time --seeds 0,1,2
evaluate-pinn-splits:
	@test -f "$(DATA)" || { \
	  echo "No residual table at $(DATA); run 'make residual' or pass DATA=path"; \
	  exit 1; \
	}
	uv run --extra data --extra gpu python -m ml.pinn.evaluate_splits \
	  --data $(DATA) $(SPLIT_EVAL_ARGS)

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
# toolkit; pass extra Google Benchmark flags with BENCH_ARGS, or select series
# with BENCH_FILTER.  Examples:
#   make bench
#   make bench BENCH_ARGS="--benchmark_min_time=0.5s --benchmark_repetitions=3"
#   make bench BENCH_FILTER="BM_PropagateKernelBurns"
#   make bench-json BENCH_OUT=results/quick.json
bench-build:
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
	cmake --build $(BUILD_DIR) -j --target multibody_bench

bench: bench-build
	mkdir -p results $(dir $(BENCH_OUT))
	$(BENCH_BIN) $(BENCH_RUN_ARGS)

# Short sampling window for a fast smoke run.
bench-quick: BENCH_ARGS = --benchmark_min_time=0.2s
bench-quick: bench

# Machine-readable output for the viewer, CI, or a report.
bench-json: BENCH_ARGS = --benchmark_min_time=$(BENCH_MIN_TIME) --benchmark_format=json --benchmark_out=$(BENCH_OUT)
bench-json: bench

# List every registered case without executing it.
bench-list: bench-build
	$(BENCH_BIN) --benchmark_list_tests $(BENCH_FILTER_ARG)

# Focused subsets.  Filters are POSIX regular expressions matched against the
# full benchmark name, including its /arg/arg suffix.
bench-kernel: BENCH_FILTER = ^BM_PropagateKernel/
bench-kernel: bench

bench-api: BENCH_FILTER = ^BM_PropagateApi/
bench-api: bench

bench-transfer: BENCH_FILTER = ^BM_(HostToDevice|DeviceToHost)
bench-transfer: bench

bench-physics: BENCH_FILTER = ^BM_PropagateKernel(J2|Ephemeris|Forces|Full|Softening|Epoch)
bench-physics: bench

bench-trajectory: BENCH_FILTER = ^BM_PropagateTrajectory
bench-trajectory: bench

bench-burns: BENCH_FILTER = ^BM_PropagateKernelBurns
bench-burns: bench

bench-timestep: BENCH_FILTER = ^BM_PropagateKernelTimestep
bench-timestep: bench

# Direct CUDA vs Rust comparison.  Builds the Rust dynamics static library via
# cargo, links it into backend_bench, then runs both backends on identical
# states.  Requires CUDA and cargo.
backend-build:
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
	cmake --build $(BUILD_DIR) -j --target backend_bench

bench-backends: backend-build
	mkdir -p results $(dir $(BENCH_OUT))
	$(BUILD_DIR)/benchmarks/backend_bench $(BENCH_RUN_ARGS)

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
