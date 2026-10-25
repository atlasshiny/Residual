# CUDA multibody benchmarks

`multibody_bench.cpp` times the propagator with **CUDA events**
(`cudaEventRecord` / `cudaEventElapsedTime`), so the reported numbers are GPU
timeline time rather than host wall-clock time.

| Series | What it measures |
|---|---|
| `BM_HostToDevice` | `cudaMemcpy` of initial states, host → device |
| `BM_DeviceToHost` | `cudaMemcpy` of final states, device → host |
| `BM_PropagateKernel` | kernel only on resident device buffers (`mbs_propagate_device`) |
| `BM_PropagateApi` | full `mbs_propagate`: allocate + upload + kernel + download |
| `BM_PropagateTrajectory` | sustained trajectory writes at large body × step shapes |
| `BM_PropagateKernelJ2` | Earth J2 off vs on |
| `BM_PropagateKernelEphemeris` | Earth-only vs Earth+Moon |
| `BM_PropagateKernelForces` | J2 × ephemeris matrix, including the combined case |
| `BM_PropagateKernelBurns` | impulsive-burn bookkeeping for 0–8 burns |
| `BM_PropagateKernelTimestep` | equal simulated span at different `dt` |
| `BM_PropagateKernelSoftening` | gravitational softening values |
| `BM_PropagateKernelEpoch` | epoch and lunar-phase variation |
| `BM_PropagateKernelLaunch` | tiny batches / launch overhead |
| `BM_PropagateKernelFull` | combined J2 + Moon + softening + burns, with/without trajectory |

Every benchmark performs one **untimed warm-up call** (plus
`cudaDeviceSynchronize`) before sampling, so CUDA context creation, module
loading, and the first allocation are excluded. `DeviceWorkload` also restores
the initial states between samples outside the recorded events, so every
sample performs identical work.

## Build and run

Directly:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
cmake --build build -j --target multibody_bench
./build/benchmarks/multibody_bench --benchmark_min_time=1s
```

Through Make:

```sh
make bench              # full suite, 1 s per case
make bench-quick        # 0.2 s per case, for a fast smoke run
make bench-json         # writes results/bench.json
make bench-list         # lists every registered case without running it
make bench-kernel       # BM_PropagateKernel/* only
make bench-api          # BM_PropagateApi/* only
make bench-transfer     # host→device and device→host copies
make bench-physics      # J2, ephemeris, forces, softening, epoch
make bench-trajectory   # sustained trajectory writes
make bench-burns        # burn-count sweep
make bench-timestep     # step-size sweep
```

`make bench` accepts the same variables as the focused targets:

```sh
make bench BENCH_ARGS="--benchmark_min_time=0.5s --benchmark_repetitions=3"
make bench BENCH_FILTER="BM_PropagateKernelBurns"
make bench-json BENCH_OUT=results/quick.json BENCH_MIN_TIME=0.2s
```

`bench-build` builds only the benchmark target, so repeated runs skip
reconfiguration work.

## Case inventory

| Benchmark | Arguments | Purpose |
|---|---|---|
| `BM_HostToDevice` | 1 k–1 M states | upload bandwidth |
| `BM_DeviceToHost` | 1 k–1 M states | download bandwidth |
| `BM_PropagateKernel` | 1 k–1 M bodies × 100 steps | pure kernel throughput |
| `BM_PropagateKernel` | 100 k bodies × 1000 steps | sustained kernel work |
| `BM_PropagateKernel` (trajectory) | 1 k–100 k bodies × 100 steps | trajectory write cost |
| `BM_PropagateApi` | 1 k–1 M bodies × 100 steps | end-to-end API cost |
| `BM_PropagateApi` | 1 M bodies × 1 step | allocation/launch overhead |
| `BM_PropagateApi` (trajectory) | 100 k × 100; 1 k/10 k × 1000 | full-path trajectory cost |
| `BM_PropagateTrajectory` | 1 k/10 k × 1000; 1 k × 10000 | sustained trajectory writes |
| `BM_PropagateKernelJ2` | perturbation 0 / 1 | J2 on vs off |
| `BM_PropagateKernelEphemeris` | moon distance 0 / 384,400 km | Earth-only vs Earth+Moon |
| `BM_PropagateKernelForces` | J2 × moon distance | combined force model |
| `BM_PropagateKernelBurns` | 0, 1, 2, 4, 8 burns | burn-loop overhead |
| `BM_PropagateKernelTimestep` | 600/60/30/10/5 s over 6 h | step-size scaling |
| `BM_PropagateKernelSoftening` | 0, 1 km, 100 km, 1000 km | close-encounter safeguard |
| `BM_PropagateKernelEpoch` | epoch 0/7/182 d; phase 0–270° | ephemeris phase variation |
| `BM_PropagateKernelLaunch` | 1–1024 bodies, 0–1 steps | launch/event overhead |
| `BM_PropagateKernelFull` | 100 k bodies × 100 steps, trajectory 0/1 | combined production configuration |

All trajectory shapes keep the device buffer at or below ~480 MB
(`bodies × (steps + 1) × 48 B`).

## Counters

- `gpu_ms_per_call` — event-measured milliseconds per call.
- `particle_steps_per_s` / `items_per_second` — `bodies × steps / GPU seconds`.
- `GB_per_s` — host↔device bandwidth (copy benchmarks only).
- `MB_per_call` — bytes moved or produced by one call.
- `dt_s`, `sim_hours` — integration step and simulated span.
- `burns`, `softening_km` — burn count and softening radius.
- `moon_phase_deg`, `epoch_days` — ephemeris configuration.
- `bodies`, `steps`, `trajectory`, `perturbation`, `moon_distance_km` — the
  workload shape, so JSON results are self-describing.

## Reproducible output

```sh
make bench-json BENCH_MIN_TIME=2s
# equivalent to:
./build/benchmarks/multibody_bench \
  --benchmark_min_time=2s \
  --benchmark_format=json \
  --benchmark_out=results/bench.json
```

## Notes

- Google Benchmark is used for iteration control and reporting, but timing is
  manual (`UseManualTime` + `SetIterationTime`), sourced from CUDA events.
- `BM_PropagateApi` brackets the *entire* `mbs_propagate` call, which includes
  `cudaMalloc`/`cudaFree` and blocking `cudaMemcpy`; the GPU is idle during
  those host-side steps, so `gpu_ms_per_call` for that series is wall time
  inside the API call, not pure GPU busy time. Use `BM_PropagateKernel` for
  compute throughput.
- `BM_PropagateKernelTimestep` holds the simulated span fixed, so throughput
  counters are directly comparable across `dt`; total wall time still grows as
  `dt` shrinks.
- `BM_PropagateKernelBurns` measures the per-step burn check, not the cost of a
  physically different trajectory. The applied `Δv` is small (0.5 m/s) and only
  keeps the batch away from a degenerate state.
- The benchmark is only built when CUDA is enabled (`SKIP_CUDA_TESTS=OFF`).
  `make bench` therefore requires a CUDA toolkit and a GPU; `make -n bench`
  shows the exact commands without running them.
