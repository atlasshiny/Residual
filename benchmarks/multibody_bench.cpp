// Benchmarks for the CUDA Earth-Moon multibody propagator.
//
// Timing uses CUDA events, not the host clock, so the reported numbers are GPU
// timeline time.  Three series isolate the costs a caller pays:
//
//   HostToDevice     cudaMemcpy of the initial states
//   PropagateKernel  kernel only, on resident device buffers (mbs_propagate_device)
//   PropagateApi     full mbs_propagate: allocate + upload + kernel + download
//
// Every benchmark runs one untimed warm-up call first so CUDA context creation,
// module loading, and the first allocation are excluded from the samples.
//
// Build and run on a CUDA machine:
//   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
//   cmake --build build -j --target multibody_bench
//   ./build/benchmarks/multibody_bench --benchmark_min_time=1s
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

#include "multibody_solver.cuh"

#include <benchmark/benchmark.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

// Circular orbit around the Earth with a small spread in radius so the batch
// is not perfectly degenerate.  Values stay well outside the Earth and well
// inside the Moon's orbit.
// Earth position at t=0 for a given Moon distance (barycentric frame).
// With moon_distance <= 0 both bodies sit at the origin (Earth-only mode).
MbsVec3 earth_position_at_epoch(double moon_distance) {
    if (moon_distance <= 0.0) {
        return {};
    }
    constexpr double kMuEarth = 3.986004418e14;
    constexpr double kMuMoon = 4.9048695e12;
    const double scale = moon_distance * kMuMoon / (kMuEarth + kMuMoon);
    MbsVec3 earth{};
    earth.x = -scale;
    return earth;
}

MbsState initial_state_for(int index, double moon_distance) {
    constexpr double kTwoPi = 6.283185307179586476925286766559;
    constexpr double kMuEarth = 3.986004418e14;

    const double radius = 6.8e6 + 1.0e5 * static_cast<double>(index % 50);
    const double angle = kTwoPi * static_cast<double>(index % 360) / 360.0;
    const double speed = std::sqrt(kMuEarth / radius);

    const MbsVec3 earth = earth_position_at_epoch(moon_distance);

    MbsState state{};
    state.r.x = earth.x + radius * std::cos(angle);
    state.r.y = earth.y + radius * std::sin(angle);
    state.r.z = 0.0;
    state.v.x = -speed * std::sin(angle);
    state.v.y = speed * std::cos(angle);
    state.v.z = 0.0;
    return state;
}

std::vector<MbsState> initial_states(int bodies, double moon_distance) {
    std::vector<MbsState> states(static_cast<std::size_t>(bodies));
    for (int i = 0; i < bodies; ++i) {
        states[static_cast<std::size_t>(i)] = initial_state_for(i, moon_distance);
    }
    return states;
}

// Records events on the default stream.  mbs_propagate and
// mbs_propagate_device both launch on that stream, so a start/stop pair around
// the call brackets exactly the GPU work it enqueues.
class GpuTimer {
 public:
    GpuTimer()
        : ok_(cudaEventCreate(&start_) == cudaSuccess &&
              cudaEventCreate(&stop_) == cudaSuccess) {}

    ~GpuTimer() {
        if (start_ != nullptr) {
            cudaEventDestroy(start_);
        }
        if (stop_ != nullptr) {
            cudaEventDestroy(stop_);
        }
    }

    GpuTimer(const GpuTimer&) = delete;
    GpuTimer& operator=(const GpuTimer&) = delete;

    bool ok() const { return ok_; }

    void begin() { cudaEventRecord(start_); }

    // Records the stop event, waits for it, and returns elapsed milliseconds.
    float end_ms() {
        cudaEventRecord(stop_);
        cudaEventSynchronize(stop_);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start_, stop_);
        return ms;
    }

 private:
    cudaEvent_t start_ = nullptr;
    cudaEvent_t stop_ = nullptr;
    bool ok_ = false;
};

void set_shape_counters(benchmark::State& state, int bodies, int steps, bool trajectory, const MbsConfig& config) {
    state.counters["bodies"] = static_cast<double>(bodies);
    state.counters["steps"] = static_cast<double>(steps);
    state.counters["trajectory"] = trajectory ? 1.0 : 0.0;
    state.counters["perturbation"] = static_cast<double>(config.perturbation);
    state.counters["moon_distance_km"] = config.moon_distance / 1.0e3;
}

void set_throughput(benchmark::State& state, double total_ms, int bodies, int steps) {
    const double seconds = total_ms / 1.0e3;
    const double calls = static_cast<double>(state.iterations());
    const double particle_steps = calls * static_cast<double>(bodies) * steps;
    state.counters["gpu_ms_per_call"] = calls > 0.0 ? total_ms / calls : 0.0;
    state.counters["particle_steps_per_s"] = seconds > 0.0 ? particle_steps / seconds : 0.0;
    state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations()) * bodies * steps);
}

// ---------------------------------------------------------------------------
// End-to-end public API
// ---------------------------------------------------------------------------

struct HostWorkload {
    std::vector<MbsState> initial;
    std::vector<MbsState> final;
    std::vector<MbsState> trajectory;
    MbsConfig config{};
    int bodies = 0;
    int steps = 0;
    bool write_trajectory = false;

    HostWorkload(int body_count, int step_count, bool with_trajectory, int perturbation, double moon_distance)
        : initial(initial_states(body_count, moon_distance)),
          final(static_cast<std::size_t>(body_count)),
          bodies(body_count),
          steps(step_count),
          write_trajectory(with_trajectory) {
        if (with_trajectory) {
            const std::size_t frames = static_cast<std::size_t>(step_count) + 1;
            trajectory.resize(frames * static_cast<std::size_t>(body_count));
        }
        mbs_default_config(&config);
        config.dt = 60.0;
        config.perturbation = perturbation;
        config.moon_distance = moon_distance;
    }

    bool ok() const { return true; }
    const char* error() const { return ""; }
    cudaError_t reset() { return cudaSuccess; }

    int run() {
        return mbs_propagate(initial.data(), bodies, steps, &config, final.data(),
                             write_trajectory ? trajectory.data() : nullptr);
    }

    // Upload + download + optional trajectory download.
    std::int64_t bytes_per_call() const {
        const std::int64_t state_bytes = static_cast<std::int64_t>(sizeof(MbsState));
        std::int64_t bytes = 2 * static_cast<std::int64_t>(bodies) * state_bytes;
        if (write_trajectory) {
            bytes += static_cast<std::int64_t>(trajectory.size()) * state_bytes;
        }
        return bytes;
    }
};

// ---------------------------------------------------------------------------
// Kernel-only, device resident
// ---------------------------------------------------------------------------

struct DeviceWorkload {
    std::vector<MbsState> host_initial;
    MbsState* device_states = nullptr;
    MbsState* device_trajectory = nullptr;
    MbsConfig config{};
    int bodies = 0;
    int steps = 0;
    bool write_trajectory = false;
    cudaError_t alloc_error = cudaSuccess;

    DeviceWorkload(int body_count, int step_count, bool with_trajectory, int perturbation, double moon_distance)
        : host_initial(initial_states(body_count, moon_distance)),
          bodies(body_count),
          steps(step_count),
          write_trajectory(with_trajectory) {
        mbs_default_config(&config);
        config.dt = 60.0;
        config.perturbation = perturbation;
        config.moon_distance = moon_distance;

        alloc_error = cudaMalloc(reinterpret_cast<void**>(&device_states),
                                 static_cast<std::size_t>(body_count) * sizeof(MbsState));
        if (alloc_error == cudaSuccess && with_trajectory) {
            const std::size_t frames = static_cast<std::size_t>(step_count) + 1;
            alloc_error = cudaMalloc(
                reinterpret_cast<void**>(&device_trajectory),
                frames * static_cast<std::size_t>(body_count) * sizeof(MbsState));
        }
        if (alloc_error == cudaSuccess) {
            const std::size_t bytes = static_cast<std::size_t>(body_count) * sizeof(MbsState);
            alloc_error = cudaMemcpy(device_states, host_initial.data(), bytes,
                                     cudaMemcpyHostToDevice);
        }
    }

    ~DeviceWorkload() {
        if (device_trajectory != nullptr) {
            cudaFree(device_trajectory);
        }
        if (device_states != nullptr) {
            cudaFree(device_states);
        }
    }

    DeviceWorkload(const DeviceWorkload&) = delete;
    DeviceWorkload& operator=(const DeviceWorkload&) = delete;

    DeviceWorkload(DeviceWorkload&& other) noexcept
        : host_initial(std::move(other.host_initial)),
          device_states(other.device_states),
          device_trajectory(other.device_trajectory),
          config(other.config),
          bodies(other.bodies),
          steps(other.steps),
          write_trajectory(other.write_trajectory),
          alloc_error(other.alloc_error) {
        other.device_states = nullptr;
        other.device_trajectory = nullptr;
    }

    DeviceWorkload& operator=(DeviceWorkload&&) = delete;

    bool ok() const { return alloc_error == cudaSuccess; }
    const char* error() const { return "device allocation/copy failed"; }

    // Restore the initial states before each timed call so every sample does
    // identical work.  This copy happens outside the recorded events.
    cudaError_t reset() {
        const std::size_t bytes = static_cast<std::size_t>(bodies) * sizeof(MbsState);
        return cudaMemcpy(device_states, host_initial.data(), bytes,
                          cudaMemcpyHostToDevice);
    }

    int run() {
        return mbs_propagate_device(device_states, bodies, steps, &config,
                                    write_trajectory ? device_trajectory : nullptr);
    }

    // One read and one write per particle, plus the optional trajectory writes.
    std::int64_t bytes_per_call() const {
        const std::int64_t state_bytes = static_cast<std::int64_t>(sizeof(MbsState));
        std::int64_t bytes = 2 * static_cast<std::int64_t>(bodies) * state_bytes;
        if (write_trajectory) {
            bytes += static_cast<std::int64_t>(bodies) * (steps + 1) * state_bytes;
        }
        return bytes;
    }
};

// ---------------------------------------------------------------------------
// Shared runner
// ---------------------------------------------------------------------------

template <typename Workload>
void time_workload(benchmark::State& state, Workload workload, bool reset_before_call) {
    if (!workload.ok()) {
        state.SkipWithError(workload.error());
        return;
    }

    // Warm-up: one full call plus synchronisation, untimed.
    MBS_CUDA_CHECK_OR_SKIP(state, workload.run(), "warm-up call failed");
    MBS_CUDA_CHECK_OR_SKIP(state, cudaDeviceSynchronize(), "warm-up synchronisation failed");

    GpuTimer timer;
    if (!timer.ok()) {
        state.SkipWithError("cudaEventCreate failed");
        return;
    }

    double total_ms = 0.0;
    for (auto _ : state) {
        if (reset_before_call) {
            if (workload.reset() != cudaSuccess) { state.SkipWithError("device reset failed"); return; }
        }
        timer.begin();
        const int rc = workload.run();
        const float ms = timer.end_ms();
        MBS_CUDA_CHECK_OR_SKIP(state, rc, "mbs_propagate launch failed");
        MBS_CUDA_LAST_ERROR_OR_SKIP(state, "kernel execution failed");
        state.SetIterationTime(static_cast<double>(ms) / 1.0e3);
        total_ms += static_cast<double>(ms);
    }

    set_shape_counters(state, workload.bodies, workload.steps,
                       workload.write_trajectory, workload.config);
    set_throughput(state, total_ms, workload.bodies, workload.steps);
    state.counters["MB_per_call"] =
        static_cast<double>(workload.bytes_per_call()) / (1024.0 * 1024.0);
    state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations()) *
                            workload.bytes_per_call());
    return;
}

}  // namespace

// ---------------------------------------------------------------------------
// Host -> device copy
// ---------------------------------------------------------------------------

void BM_HostToDevice(benchmark::State& state) {
    const int bodies = static_cast<int>(state.range(0));
    const std::vector<MbsState> host = initial_states(bodies, 3.844e8);
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
        cudaFree(device);
        state.SkipWithError("cudaEventCreate failed");
        return;
    }

    double total_ms = 0.0;
    for (auto _ : state) {
        timer.begin();
        const cudaError_t rc =
            cudaMemcpy(device, host.data(), bytes, cudaMemcpyHostToDevice);
        const float ms = timer.end_ms();
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

// ---------------------------------------------------------------------------
// Kernel and end-to-end benchmarks
// ---------------------------------------------------------------------------

void BM_PropagateKernel(benchmark::State& state) {
    const int bodies = static_cast<int>(state.range(0));
    const int steps = static_cast<int>(state.range(1));
    const bool trajectory = state.range(2) != 0;
    DeviceWorkload workload(bodies, steps, trajectory, 0, 3.844e8);
    time_workload(state, std::move(workload), true);
}

void BM_PropagateApi(benchmark::State& state) {
    const int bodies = static_cast<int>(state.range(0));
    const int steps = static_cast<int>(state.range(1));
    const bool trajectory = state.range(2) != 0;
    time_workload(
        state,
        HostWorkload(bodies, steps, trajectory, 0, 3.844e8),
        false);
}

void BM_PropagateKernelJ2(benchmark::State& state) {
    const int perturbation = static_cast<int>(state.range(0));
    time_workload(
        state,
        DeviceWorkload(100000, 100, false, perturbation, 3.844e8),
        true);
}

void BM_PropagateKernelEphemeris(benchmark::State& state) {
    const double moon_distance = static_cast<double>(state.range(0));
    time_workload(
        state,
        DeviceWorkload(100000, 100, false, 0, moon_distance),
        true);
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
    ->ArgNames({"bodies", "steps", "trajectory"})
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
    ->Arg(3.844e8)
    ->ArgNames({"moon_distance_m"})
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime();

BENCHMARK_MAIN();
