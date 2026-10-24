# CUDA multibody benchmark

`multibody_bench.cpp` times three separable costs of the propagator with
**CUDA events** (`cudaEventRecord` / `cudaEventElapsedTime`), so the reported
numbers are GPU timeline time rather than host wall-clock time:

| Series | What it times |
|---|---|
| `BM_HostToDevice` | `cudaMemcpy` of the initial states, host → device |
| `BM_PropagateKernel` | kernel only on resident device buffers (`mbs_propagate_device`) |
| `BM_PropagateApi` | full `mbs_propagate`: allocate + upload + kernel + download |

Every benchmark performs one **untimed warm-up call** (plus
`cudaDeviceSynchronize`) before sampling, so CUDA context creation, module
loading, and the first allocation are excluded. `DeviceWorkload` also restores
the initial states between samples outside the recorded events, so every
sample performs identical work.

## Build and run

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
cmake --build build -j --target multibody_bench
./build/benchmarks/multibody_bench --benchmark_min_time=1s
```

or:

```sh
make bench
```

## What is measured

| Benchmark | Shape | Purpose |
|---|---|---|
| `BM_HostToDevice` | 1 k–1 M states | upload bandwidth |
| `BM_PropagateKernel` | 1 k–1 M bodies × 100 steps | pure kernel throughput |
| `BM_PropagateKernel` | 100 k bodies × 1000 steps | sustained kernel work |
| `BM_PropagateKernel` (trajectory) | 1 k–100 k bodies × 100 steps | trajectory write cost |
| `BM_PropagateApi` | 1 k–1 M bodies × 100 steps | end-to-end API cost |
| `BM_PropagateApi` | 1 M bodies × 1 step | allocation/launch overhead |
| `BM_PropagateKernelJ2` | 100 k bodies × 100 steps | J2 on vs off |
| `BM_PropagateKernelEphemeris` | 100 k bodies × 100 steps | Earth-only vs Earth+Moon |

Counters written by the harness:

- `gpu_ms_per_call` — event-measured milliseconds per call.
- `particle_steps_per_s` / `items_per_second` — `bodies × steps / GPU seconds`.
- `GB_per_s` — host→device bandwidth (copy benchmark only).
- `MB_per_call` — bytes moved or produced by one call.
- `bodies`, `steps`, `trajectory`, `perturbation`, `moon_distance_km` — the
  workload shape, so JSON results are self-describing.

## Reproducible output

```sh
./build/benchmarks/multibody_bench \
  --benchmark_min_time=2s \
  --benchmark_format=json \
  --benchmark_out=results/bench.json
```

## Notes

- Google Benchmark is used for iteration control and reporting, but timing is
  manual (`UseManualTime` + `SetIterationTime`), sourced from CUDA events.
- `BM_PropagateApi` includes `cudaMalloc`/`cudaFree` on every call. For small
  batches the allocator dominates — that is real API cost. Use
  `BM_PropagateKernel` for pure compute throughput.
- `mbs_propagate_device` is the device-resident entry point added for this
  split: it takes device pointers, launches on the default stream, and returns
  without synchronising.
- Trajectory cases are sized so the device buffer stays at or below ~1 GiB
  (`100000 bodies × 101 frames × 48 B ≈ 485 MB`).
- The benchmark is only built when CUDA is enabled (`SKIP_CUDA_TESTS=OFF`).
