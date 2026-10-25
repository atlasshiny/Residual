// Direct CUDA vs Rust benchmark for the fixed-step multibody propagator.
//
// The two backends share the same initial states, configuration, and burn
// window:
//
//   BM_CudaKernel      mbs_propagate_device, timed with CUDA events
//   BM_RustBatch       dynamics::propagate_batch_in_place through the
//                      dynamics_bench static library, timed with the host clock
//   BM_BackendAgreement  runs both once and reports final-state differences
//
// BM_RustBatch also calibrates the CUDA kernel for the same shape and reports
// cuda_ms_per_call and cuda_speedup, so a single row contains the head-to-head
// comparison. The headline Time column for that row is the Rust time.
//
// Build and run (requires CUDA and cargo):
//   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
//   cmake --build build -j --target backend_bench
//   ./build/benchmarks/backend_bench --benchmark_min_time=1s
//
// Or:
//   make bench-backends
//   make bench-backends BENCH_ARGS="--benchmark_filter=BM_BackendAgreement"
//
// Counters:
//   cuda_ms_per_call / rust_ms_per_call   event / host-clock time per call
//   cuda_speedup                          rust_ms / cuda_ms
//   max_pos_diff_m, rms_pos_diff_m        final position difference
//   max_vel_diff_m_s                      final velocity difference
//   particle_steps_per_s                  bodies * steps / seconds

#include "bench_support.h"

#include <benchmark/benchmark.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <vector>

// Matches DynamicsBenchState in crates/dynamics-bench/src/lib.rs.
struct DynBenchState {
    double r[3];
    double v[3];
};

extern "C" int dynamics_bench_propagate(
    DynBenchState* states,
    int num_bodies,
    int num_steps,
    double dt,
    double epoch,
    int perturbation,
    double moon_distance,
    double moon_phase,
    double softening,
    int num_burns,
    int burn_start_step,
    int burn_stride_steps,
    double burn_dv);

namespace {

using namespace bench_support;

std::vector<DynBenchState> to_dyn(const std::vector<MbsState>& states) {
    std::vector<DynBenchState> out(states.size());
    for (std::size_t i = 0; i < states.size(); ++i) {
        out[i].r[0] = states[i].r.x;
        out[i].r[1] = states[i].r.y;
        out[i].r[2] = states[i].r.z;
        out[i].v[0] = states[i].v.x;
        out[i].v[1] = states[i].v.y;
        out[i].v[2] = states[i].v.z;
    }
    return out;
}

struct RustWorkload {
    std::vector<DynBenchState> initial;
    std::vector<DynBenchState> working;
    MbsConfig config{};
    WorkloadConfig shape;
    int bodies = 0;
    int steps = 0;

    RustWorkload(int body_count, int step_count, const WorkloadConfig& workload_shape)
        : initial(to_dyn(initial_states(body_count, workload_shape))),
          working(initial),
          shape(workload_shape),
          bodies(body_count),
          steps(step_count) {
        apply_workload_config(config, workload_shape);
    }

    bool ok() const { return !initial.empty(); }
    const char* error() const { return "empty Rust workload"; }

    // Match DeviceWorkload::reset(): copy the initial states outside the timed
    // region so every sample does identical work.
    void reset() { working = initial; }

    int run() {
        return dynamics_bench_propagate(
            working.data(), bodies, steps, shape.dt, shape.epoch,
            shape.perturbation, shape.moon_distance, shape.moon_phase,
            shape.softening, shape.num_burns, shape.burn_start_step,
            shape.burn_stride_steps, shape.burn_dv);
    }
};

// Average CUDA kernel time for one propagation, used as the comparison point
// in the Rust rows. The warm-up and per-call resets happen outside the events.
double measure_cuda_ms(int bodies, int steps, const WorkloadConfig& shape,
                       int reps = 5) {
    DeviceWorkload workload(bodies, steps, false, shape);
    if (!workload.ok()) {
        return 0.0;
    }
    GpuTimer timer;
    if (!timer.ok()) {
        return 0.0;
    }

    workload.reset();
    if (workload.run() != 0 || cudaDeviceSynchronize() != cudaSuccess) {
        return 0.0;
    }

    double total_ms = 0.0;
    for (int i = 0; i < reps; ++i) {
        if (workload.reset() != cudaSuccess || timer.begin() != cudaSuccess) {
            return 0.0;
        }
        if (workload.run() != 0) {
            return 0.0;
        }
        float ms = 0.0f;
        if (timer.end_ms(&ms) != cudaSuccess) {
            return 0.0;
        }
        total_ms += static_cast<double>(ms);
    }
    return total_ms / static_cast<double>(reps);
}

void time_rust(benchmark::State& state, RustWorkload workload) {
    if (!workload.ok()) {
        state.SkipWithError(workload.error());
        return;
    }

    // Three untimed warm-up calls: allocator caches, page faults, and CPU
    // frequency ramp-up should not land in the first sample.
    for (int i = 0; i < 3; ++i) {
        workload.reset();
        if (workload.run() != 0) {
            state.SkipWithError("Rust warm-up propagation failed");
            return;
        }
    }

    double total_ms = 0.0;
    for (auto _ : state) {
        workload.reset();
        const auto start = std::chrono::steady_clock::now();
        const int rc = workload.run();
        const auto stop = std::chrono::steady_clock::now();
        if (rc != 0) {
            state.SkipWithError("Rust propagation failed");
            return;
        }
        const double ms =
            std::chrono::duration<double, std::milli>(stop - start).count();
        state.SetIterationTime(ms / 1.0e3);
        total_ms += ms;
    }

    set_shape_counters(state, workload.bodies, workload.steps, false,
                       workload.config);
    set_throughput(state, total_ms, workload.bodies, workload.steps,
                   "cpu_ms_per_call");

    const double calls = static_cast<double>(state.iterations());
    const double rust_ms = calls > 0.0 ? total_ms / calls : 0.0;
    state.counters["rust_ms_per_call"] = rust_ms;

    const double cuda_ms =
        measure_cuda_ms(workload.bodies, workload.steps, workload.shape);
    state.counters["cuda_ms_per_call"] = cuda_ms;
    if (cuda_ms > 0.0) {
        state.counters["cuda_speedup"] = rust_ms / cuda_ms;
    }

    // Two state transfers (read + write) per body; the Rust shim works in place.
    const double bytes = 2.0 * static_cast<double>(workload.bodies) *
                         static_cast<double>(sizeof(DynBenchState));
    state.counters["MB_per_call"] = bytes / (1024.0 * 1024.0);
}

}  // namespace

void BM_CudaKernel(benchmark::State& state) {
    const int bodies = static_cast<int>(state.range(0));
    const int steps = static_cast<int>(state.range(1));
    WorkloadConfig shape;
    shape.perturbation = static_cast<int>(state.range(2));
    shape.num_burns = static_cast<int>(state.range(3));
    time_workload(state, DeviceWorkload(bodies, steps, false, shape), true);
}

void BM_RustBatch(benchmark::State& state) {
    const int bodies = static_cast<int>(state.range(0));
    const int steps = static_cast<int>(state.range(1));
    WorkloadConfig shape;
    shape.perturbation = static_cast<int>(state.range(2));
    shape.num_burns = static_cast<int>(state.range(3));
    time_rust(state, RustWorkload(bodies, steps, shape));
}

// Correctness companion to the timing rows. The reported time is the combined
// cost of both propagations; the counters are the interesting part.
void BM_BackendAgreement(benchmark::State& state) {
    constexpr int kBodies = 1000;
    constexpr int kSteps = 100;

    WorkloadConfig shape;
    shape.perturbation = static_cast<int>(state.range(0));
    shape.num_burns = static_cast<int>(state.range(1));

    DeviceWorkload cuda(kBodies, kSteps, false, shape);
    RustWorkload rust(kBodies, kSteps, shape);
    if (!cuda.ok() || !rust.ok()) {
        state.SkipWithError("backend setup failed");
        return;
    }

    std::vector<MbsState> cuda_final(static_cast<std::size_t>(kBodies));
    const std::size_t bytes =
        static_cast<std::size_t>(kBodies) * sizeof(MbsState);

    double max_pos = 0.0;
    double sum_sq = 0.0;
    double max_vel = 0.0;

    for (auto _ : state) {
        if (cuda.reset() != cudaSuccess || cuda.run() != 0) {
            state.SkipWithError("CUDA propagation failed");
            return;
        }
        if (cudaDeviceSynchronize() != cudaSuccess ||
            cudaMemcpy(cuda_final.data(), cuda.device_states, bytes,
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
            state.SkipWithError("CUDA download failed");
            return;
        }

        rust.reset();
        if (rust.run() != 0) {
            state.SkipWithError("Rust propagation failed");
            return;
        }

        max_pos = 0.0;
        sum_sq = 0.0;
        max_vel = 0.0;
        for (int i = 0; i < kBodies; ++i) {
            const MbsState& c = cuda_final[static_cast<std::size_t>(i)];
            const DynBenchState& r = rust.working[static_cast<std::size_t>(i)];

            const double dx = c.r.x - r.r[0];
            const double dy = c.r.y - r.r[1];
            const double dz = c.r.z - r.r[2];
            const double pos_sq = dx * dx + dy * dy + dz * dz;
            max_pos = std::max(max_pos, std::sqrt(pos_sq));
            sum_sq += pos_sq;

            const double dvx = c.v.x - r.v[0];
            const double dvy = c.v.y - r.v[1];
            const double dvz = c.v.z - r.v[2];
            max_vel = std::max(max_vel, std::sqrt(dvx * dvx + dvy * dvy + dvz * dvz));
        }
    }

    set_shape_counters(state, kBodies, kSteps, false, cuda.config);
    state.counters["max_pos_diff_m"] = max_pos;
    state.counters["rms_pos_diff_m"] =
        std::sqrt(sum_sq / static_cast<double>(kBodies));
    state.counters["max_vel_diff_m_s"] = max_vel;
}

BENCHMARK(BM_CudaKernel)
    ->Args({1000, 100, 0, 0})
    ->Args({10000, 100, 0, 0})
    ->Args({100000, 100, 0, 0})
    ->Args({100000, 1000, 0, 0})
    ->Args({100000, 100, 0, 4})
    ->Args({100000, 100, 1, 4})
    ->ArgNames({"bodies", "steps", "perturbation", "burns"})
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();

BENCHMARK(BM_RustBatch)
    ->Args({1000, 100, 0, 0})
    ->Args({10000, 100, 0, 0})
    ->Args({100000, 100, 0, 0})
    ->Args({100000, 1000, 0, 0})
    ->Args({100000, 100, 0, 4})
    ->Args({100000, 100, 1, 4})
    ->ArgNames({"bodies", "steps", "perturbation", "burns"})
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();

BENCHMARK(BM_BackendAgreement)
    ->Args({0, 0})
    ->Args({1, 4})
    ->ArgNames({"perturbation", "burns"});

BENCHMARK_MAIN();
