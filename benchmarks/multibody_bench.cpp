// Benchmarks for the CUDA Earth-Moon multibody propagator.
//
// Timing uses CUDA events, not the host clock, so the reported numbers are GPU
// timeline time.  The series isolate the costs a caller pays:
//
//   HostToDevice        cudaMemcpy of the initial states, host -> device
//   DeviceToHost        cudaMemcpy of the final states, device -> host
//   PropagateKernel     kernel only, on resident device buffers
//   PropagateApi        full mbs_propagate: allocate + upload + kernel + download
//   PropagateTrajectory sustained trajectory writes at several shapes
//   PropagateKernelJ2   Earth J2 on vs off
//   PropagateKernelEphemeris  Earth-only vs Earth+Moon
//   PropagateKernelForces     J2 x ephemeris matrix
//   PropagateKernelBurns      impulsive-burn bookkeeping, 0..8 burns
//   PropagateKernelTimestep   equal simulated duration at different dt
//   PropagateKernelSoftening  gravitational softening values
//   PropagateKernelEpoch      epoch and lunar-phase variation
//   PropagateKernelLaunch     tiny batches / launch overhead
//   PropagateKernelFull       combined J2 + Moon + softening + burns (+ trajectory)
//
// Every benchmark runs one untimed warm-up call first so CUDA context creation,
// module loading, and the first allocation are excluded from the samples.
//
// Build and run on a CUDA machine:
//   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
//   cmake --build build -j --target multibody_bench
//   ./build/benchmarks/multibody_bench --benchmark_min_time=1s
//
// Or through Make:
//   make bench              # full suite
//   make bench-quick        # short sampling window
//   make bench-json         # JSON into results/bench.json
//   make bench-list         # list cases without running them
//   make bench-kernel       # kernel-only series
//   make bench-physics      # J2/ephemeris/forces/softening/epoch
//   make bench-burns        # burn-count series
//   make bench-trajectory   # trajectory write series
//
// Useful flags (Google Benchmark):
//   --benchmark_format=json --benchmark_out=results/bench.json
//   --benchmark_filter=PropagateKernel
//
// Counters:
//   gpu_ms_per_call        event-measured time per call
//   particle_steps_per_s   bodies * steps / GPU seconds
//   items_per_second       same value, reported by Google Benchmark
//   MB_per_call            bytes moved by the call (custom, for context)
//   dt_s, sim_hours        integration step and simulated span
//   burns, softening_km    burn count and softening radius
//   moon_phase_deg, epoch_days  ephemeris configuration

#include "bench_support.h"

#include <benchmark/benchmark.h>

using namespace bench_support;

// ---------------------------------------------------------------------------
// Host <-> device copies
// ---------------------------------------------------------------------------

void BM_HostToDevice(benchmark::State& state) {
    const int bodies = static_cast<int>(state.range(0));
    const std::vector<MbsState> host = initial_states(bodies, WorkloadConfig{});
    const std::size_t bytes = host.size() * sizeof(MbsState);

    MbsState* device = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&device), bytes) != cudaSuccess) {
        state.SkipWithError("cudaMalloc failed");
        return;
    }
    auto cleanup = [&]() { cudaFree(device); };

    // Warm-up copy, untimed.
    if (cudaMemcpy(device, host.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess) {
        cleanup();
        state.SkipWithError("warm-up copy failed");
        return;
    }

    GpuTimer timer;
    if (!timer.ok()) {
        cleanup();
        state.SkipWithError("cudaEventCreate failed");
        return;
    }

    double total_ms = 0.0;
    for (auto _ : state) {
        MBS_CUDA_CHECK_OR_SKIP(state, timer.begin(), "event record failed");
        const cudaError_t rc =
            cudaMemcpy(device, host.data(), bytes, cudaMemcpyHostToDevice);
        float ms = 0.0f;
        MBS_CUDA_CHECK_OR_SKIP(state, timer.end_ms(&ms), "event elapsed failed");
        if (rc != cudaSuccess) {
            cleanup();
            state.SkipWithError("cudaMemcpy H2D failed");
            return;
        }
        state.SetIterationTime(static_cast<double>(ms) / 1.0e3);
        total_ms += static_cast<double>(ms);
    }

    const double seconds = total_ms / 1.0e3;
    const double copies = static_cast<double>(state.iterations());
    state.counters["bodies"] = static_cast<double>(bodies);
    state.counters["gpu_ms_per_call"] = copies > 0.0 ? total_ms / copies : 0.0;
    state.counters["GB_per_s"] =
        seconds > 0.0 ? (copies * static_cast<double>(bytes)) / seconds / 1.0e9 : 0.0;
    state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations()) * bodies);
    state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations()) *
                            static_cast<std::int64_t>(bytes));

    cleanup();
}

void BM_DeviceToHost(benchmark::State& state) {
    const int bodies = static_cast<int>(state.range(0));
    std::vector<MbsState> host = initial_states(bodies, WorkloadConfig{});
    const std::size_t bytes = host.size() * sizeof(MbsState);

    MbsState* device = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&device), bytes) != cudaSuccess) {
        state.SkipWithError("cudaMalloc failed");
        return;
    }
    auto cleanup = [&]() { cudaFree(device); };

    if (cudaMemcpy(device, host.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess) {
        cleanup();
        state.SkipWithError("warm-up upload failed");
        return;
    }

    GpuTimer timer;
    if (!timer.ok()) {
        cleanup();
        state.SkipWithError("cudaEventCreate failed");
        return;
    }

    double total_ms = 0.0;
    for (auto _ : state) {
        MBS_CUDA_CHECK_OR_SKIP(state, timer.begin(), "event record failed");
        const cudaError_t rc =
            cudaMemcpy(host.data(), device, bytes, cudaMemcpyDeviceToHost);
        float ms = 0.0f;
        MBS_CUDA_CHECK_OR_SKIP(state, timer.end_ms(&ms), "event elapsed failed");
        if (rc != cudaSuccess) {
            cleanup();
            state.SkipWithError("cudaMemcpy D2H failed");
            return;
        }
        state.SetIterationTime(static_cast<double>(ms) / 1.0e3);
        total_ms += static_cast<double>(ms);
    }

    const double seconds = total_ms / 1.0e3;
    const double copies = static_cast<double>(state.iterations());
    state.counters["bodies"] = static_cast<double>(bodies);
    state.counters["gpu_ms_per_call"] = copies > 0.0 ? total_ms / copies : 0.0;
    state.counters["GB_per_s"] =
        seconds > 0.0 ? (copies * static_cast<double>(bytes)) / seconds / 1.0e9 : 0.0;
    state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations()) * bodies);
    state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations()) *
                            static_cast<std::int64_t>(bytes));

    cleanup();
}

// ---------------------------------------------------------------------------
// Kernel and end-to-end benchmarks
// ---------------------------------------------------------------------------

void BM_PropagateKernel(benchmark::State& state) {
    const int bodies = static_cast<int>(state.range(0));
    const int steps = static_cast<int>(state.range(1));
    const bool trajectory = state.range(2) != 0;
    time_workload(state, DeviceWorkload(bodies, steps, trajectory, WorkloadConfig{}), true);
}

void BM_PropagateApi(benchmark::State& state) {
    const int bodies = static_cast<int>(state.range(0));
    const int steps = static_cast<int>(state.range(1));
    const bool trajectory = state.range(2) != 0;
    time_workload(
        state,
        HostWorkload(bodies, steps, trajectory, WorkloadConfig{}),
        false);
}

void BM_PropagateTrajectory(benchmark::State& state) {
    const int bodies = static_cast<int>(state.range(0));
    const int steps = static_cast<int>(state.range(1));
    time_workload(
        state,
        DeviceWorkload(bodies, steps, true, WorkloadConfig{}),
        true);
}

void BM_PropagateKernelJ2(benchmark::State& state) {
    WorkloadConfig shape;
    shape.perturbation = static_cast<int>(state.range(0));
    time_workload(state, DeviceWorkload(100000, 100, false, shape), true);
}

void BM_PropagateKernelEphemeris(benchmark::State& state) {
    WorkloadConfig shape;
    shape.moon_distance = static_cast<double>(state.range(0));
    time_workload(state, DeviceWorkload(100000, 100, false, shape), true);
}

void BM_PropagateKernelForces(benchmark::State& state) {
    WorkloadConfig shape;
    shape.perturbation = static_cast<int>(state.range(0));
    shape.moon_distance = static_cast<double>(state.range(1));
    time_workload(state, DeviceWorkload(100000, 100, false, shape), true);
}

void BM_PropagateKernelBurns(benchmark::State& state) {
    WorkloadConfig shape;
    shape.num_burns = static_cast<int>(state.range(0));
    time_workload(state, DeviceWorkload(100000, 100, false, shape), true);
}

void BM_PropagateKernelTimestep(benchmark::State& state) {
    WorkloadConfig shape;
    shape.dt = static_cast<double>(state.range(0));
    const int steps = static_cast<int>(state.range(1));
    const int bodies = static_cast<int>(state.range(2));
    time_workload(state, DeviceWorkload(bodies, steps, false, shape), true);
}

void BM_PropagateKernelSoftening(benchmark::State& state) {
    WorkloadConfig shape;
    shape.softening = static_cast<double>(state.range(0));
    time_workload(state, DeviceWorkload(100000, 100, false, shape), true);
}

void BM_PropagateKernelEpoch(benchmark::State& state) {
    WorkloadConfig shape;
    shape.epoch = static_cast<double>(state.range(0));
    shape.moon_phase =
        static_cast<double>(state.range(1)) * mbs::kPi / 180.0;
    time_workload(state, DeviceWorkload(100000, 100, false, shape), true);
}

void BM_PropagateKernelLaunch(benchmark::State& state) {
    const int bodies = static_cast<int>(state.range(0));
    const int steps = static_cast<int>(state.range(1));
    time_workload(
        state,
        DeviceWorkload(bodies, steps, false, WorkloadConfig{}),
        true);
}

// Realistic combined configuration: Earth+Moon, J2, softening, four burns,
// and an optional trajectory write.  This is the closest single number to a
// production particle batch.
void BM_PropagateKernelFull(benchmark::State& state) {
    WorkloadConfig shape;
    shape.perturbation = 1;
    shape.moon_distance = kMoonDistance;
    shape.softening = 1000.0;
    shape.num_burns = 4;
    const bool trajectory = state.range(0) != 0;
    time_workload(state, DeviceWorkload(100000, 100, trajectory, shape), true);
}

// Host -> device bandwidth for the initial-state upload.
BENCHMARK(BM_HostToDevice)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(100000)
    ->Arg(1000000)
    ->ArgNames({"bodies"})
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();

// Device -> host bandwidth for the final-state download.
BENCHMARK(BM_DeviceToHost)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(100000)
    ->Arg(1000000)
    ->ArgNames({"bodies"})
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();

// Kernel-only: batch scaling, API-free.  The 1 M x 100 case is the peak
// throughput point; the 100 k x 1000 case shows sustained work per launch.
BENCHMARK(BM_PropagateKernel)
    ->Args({1000, 100, 0})
    ->Args({10000, 100, 0})
    ->Args({100000, 100, 0})
    ->Args({1000000, 100, 0})
    ->Args({100000, 1000, 0})
    ->Args({1000, 100, 1})
    ->Args({10000, 100, 1})
    ->Args({100000, 100, 1})
    ->ArgNames({"bodies", "steps", "trajectory"})
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();

// Full public API, including per-call allocation and transfers.
BENCHMARK(BM_PropagateApi)
    ->Args({1000, 100, 0})
    ->Args({10000, 100, 0})
    ->Args({100000, 100, 0})
    ->Args({1000000, 100, 0})
    ->Args({1000000, 1, 0})
    ->Args({100000, 100, 1})
    ->Args({1000, 1000, 1})
    ->Args({10000, 1000, 1})
    ->ArgNames({"bodies", "steps", "trajectory"})
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();

// Sustained trajectory writes.  Each shape keeps the trajectory buffer at or
// below ~480 MB (bodies * (steps + 1) * 48 B).
BENCHMARK(BM_PropagateTrajectory)
    ->Args({1000, 1000})
    ->Args({10000, 1000})
    ->Args({1000, 10000})
    ->ArgNames({"bodies", "steps"})
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();

BENCHMARK(BM_PropagateKernelJ2)
    ->Arg(0)
    ->Arg(1)
    ->ArgNames({"perturbation"})
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();

BENCHMARK(BM_PropagateKernelEphemeris)
    ->Arg(0)
    ->Arg(384400000)
    ->ArgNames({"moon_distance_m"})
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();

// J2 x ephemeris matrix, including the combined Earth+Moon+J2 case.
BENCHMARK(BM_PropagateKernelForces)
    ->Args({0, 0})
    ->Args({0, 384400000})
    ->Args({1, 0})
    ->Args({1, 384400000})
    ->ArgNames({"perturbation", "moon_distance_m"})
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();

// Impulsive-burn bookkeeping.  Every particle gets the same burn list, so the
// difference between 0 and 8 burns is the per-step burn-loop overhead.
BENCHMARK(BM_PropagateKernelBurns)
    ->Arg(0)
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Arg(8)
    ->ArgNames({"burns"})
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();

// Equal simulated span (6 hours) at different step sizes.  particle_steps_per_s
// normalises the extra steps, so the counters show cost per particle-step.
BENCHMARK(BM_PropagateKernelTimestep)
    ->Args({600, 36, 50000})
    ->Args({60, 360, 50000})
    ->Args({30, 720, 50000})
    ->Args({10, 2160, 50000})
    ->Args({5, 4320, 50000})
    ->ArgNames({"dt_s", "steps", "bodies"})
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();

// Softening replaces |d|^3 with (|d|^2 + eps^2)^(3/2); this checks that the
// close-encounter safeguard does not add a branch or change throughput.
BENCHMARK(BM_PropagateKernelSoftening)
    ->Arg(0)
    ->Arg(1000)
    ->Arg(100000)
    ->Arg(1000000)
    ->ArgNames({"softening_m"})
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();

// Ephemeris epoch and lunar-phase variation.  The initial states are built from
// the same circular ephemeris, so the batch stays Earth-centred.
BENCHMARK(BM_PropagateKernelEpoch)
    ->Args({0, 0})
    ->Args({604800, 0})
    ->Args({0, 90})
    ->Args({0, 180})
    ->Args({0, 270})
    ->Args({15724800, 45})
    ->ArgNames({"epoch_s", "moon_phase_deg"})
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();

// Tiny batches isolate launch/event overhead from particle work.
BENCHMARK(BM_PropagateKernelLaunch)
    ->Args({1, 0})
    ->Args({1, 1})
    ->Args({32, 1})
    ->Args({1024, 1})
    ->ArgNames({"bodies", "steps"})
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();

// Combined production configuration, with and without trajectory writes.
BENCHMARK(BM_PropagateKernelFull)
    ->Arg(0)
    ->Arg(1)
    ->ArgNames({"trajectory"})
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();

// Aggregate statistics over a resident device buffer.  This isolates the
// cost of the tiled reduction from allocation and host/device traffic.
void BM_EnsembleStats(benchmark::State& state) {
    const int bodies = static_cast<int>(state.range(0));
    const std::vector<MbsState> host = initial_states(bodies, WorkloadConfig{});

    MbsState* device_states = nullptr;
    MbsEnsembleStats* device_stats = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&device_states),
                   host.size() * sizeof(MbsState)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&device_stats), sizeof(MbsEnsembleStats)) != cudaSuccess) {
        state.SkipWithError("cudaMalloc failed");
        return;
    }
    if (cudaMemcpy(device_states, host.data(), host.size() * sizeof(MbsState),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(device_stats);
        cudaFree(device_states);
        state.SkipWithError("cudaMemcpy failed");
        return;
    }

    GpuTimer timer;
    if (!timer.ok()) {
        cudaFree(device_stats);
        cudaFree(device_states);
        state.SkipWithError("cudaEventCreate failed");
        return;
    }

    double total_ms = 0.0;
    for (auto _ : state) {
        MBS_CUDA_CHECK_OR_SKIP(state, timer.begin(), "event record failed");
        const int rc = mbs_ensemble_stats_device(device_states, bodies, device_stats);
        float ms = 0.0f;
        MBS_CUDA_CHECK_OR_SKIP(state, timer.end_ms(&ms), "event elapsed failed");
        MBS_CUDA_CHECK_OR_SKIP(state, rc, "ensemble stats launch failed");
        MBS_CUDA_LAST_ERROR_OR_SKIP(state, "ensemble stats execution failed");
        state.SetIterationTime(static_cast<double>(ms) / 1.0e3);
        total_ms += static_cast<double>(ms);
    }

    cudaFree(device_stats);
    cudaFree(device_states);

    state.counters["bodies"] = static_cast<double>(bodies);
    state.counters["gpu_ms_per_call"] =
        state.iterations() > 0 ? total_ms / static_cast<double>(state.iterations()) : 0.0;
    state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations()) * bodies);
}

BENCHMARK(BM_EnsembleStats)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(100000)
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();

BENCHMARK_MAIN();
