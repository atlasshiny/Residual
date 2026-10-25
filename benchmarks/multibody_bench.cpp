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

#include "multibody_solver.cuh"

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace {

constexpr double kMuEarth = 3.986004418e14;  // m^3/s^2
constexpr double kMuMoon = 4.9048695e12;     // m^3/s^2
constexpr double kMoonDistance = 3.844e8;    // m

// Everything a benchmark can vary beyond the batch shape.  mbs_default_config
// supplies the rest of the physical constants.
struct WorkloadConfig {
    double dt = 60.0;                 // integration step, seconds
    double epoch = 0.0;               // initial time, seconds
    int perturbation = 0;             // 0 = none, 1 = Earth J2
    double moon_distance = kMoonDistance;
    double moon_phase = 0.0;          // radians at t = epoch
    double softening = 0.0;           // metres
    int num_burns = 0;                // 0..8
    int burn_start_step = 10;         // first burn at epoch + step * dt
    int burn_stride_steps = 10;       // spacing between burns
    double burn_dv = 0.5;             // m/s, applied along +y
};

void apply_workload_config(MbsConfig& config, const WorkloadConfig& shape) {
    mbs_default_config(&config);
    config.dt = shape.dt;
    config.epoch = shape.epoch;
    config.perturbation = shape.perturbation;
    config.moon_distance = shape.moon_distance;
    config.moon_phase = shape.moon_phase;
    config.softening = shape.softening;

    config.num_burns = std::min(shape.num_burns, 8);
    for (int b = 0; b < config.num_burns; ++b) {
        const int step = shape.burn_start_step + b * shape.burn_stride_steps;
        config.burns[b] =
            MbsBurn{shape.epoch + static_cast<double>(step) * shape.dt,
                    0.0, shape.burn_dv, 0.0};
    }
}

// Earth position at the workload epoch for the circular barycentric ephemeris.
// With moon_distance <= 0 both bodies sit at the origin (Earth-only mode).
MbsVec3 earth_position_at_epoch(const WorkloadConfig& shape) {
    if (shape.moon_distance <= 0.0) {
        return {};
    }
    const double total_mu = kMuEarth + kMuMoon;
    const double omega = std::sqrt(
        total_mu / (shape.moon_distance * shape.moon_distance *
                    shape.moon_distance));
    const double theta = shape.moon_phase + omega * shape.epoch;
    const double moon_fraction = kMuMoon / total_mu;
    const double scale = shape.moon_distance * moon_fraction;
    MbsVec3 earth{};
    earth.x = -scale * std::cos(theta);
    earth.y = -scale * std::sin(theta);
    return earth;
}

MbsState initial_state_for(int index, const MbsVec3& earth) {
    constexpr double kTwoPi = 6.283185307179586476925286766559;

    const double radius = 6.8e6 + 1.0e5 * static_cast<double>(index % 50);
    const double angle = kTwoPi * static_cast<double>(index % 360) / 360.0;
    const double speed = std::sqrt(kMuEarth / radius);

    MbsState state{};
    state.r.x = earth.x + radius * std::cos(angle);
    state.r.y = earth.y + radius * std::sin(angle);
    state.r.z = 0.0;
    state.v.x = -speed * std::sin(angle);
    state.v.y = speed * std::cos(angle);
    state.v.z = 0.0;
    return state;
}

std::vector<MbsState> initial_states(int bodies, const WorkloadConfig& shape) {
    const MbsVec3 earth = earth_position_at_epoch(shape);
    std::vector<MbsState> states(static_cast<std::size_t>(bodies));
    for (int i = 0; i < bodies; ++i) {
        states[static_cast<std::size_t>(i)] = initial_state_for(i, earth);
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

    cudaError_t begin() { return cudaEventRecord(start_); }

    // Records the stop event, waits for it, and returns elapsed milliseconds.
    // Returns cudaSuccess on success; *ms is set only on success.
    cudaError_t end_ms(float* ms_out) {
        cudaError_t rc = cudaEventRecord(stop_);
        if (rc != cudaSuccess) return rc;
        rc = cudaEventSynchronize(stop_);
        if (rc != cudaSuccess) return rc;
        float ms = 0.0f;
        rc = cudaEventElapsedTime(&ms, start_, stop_);
        if (rc == cudaSuccess && ms_out != nullptr) *ms_out = ms;
        return rc;
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
    state.counters["dt_s"] = config.dt;
    state.counters["sim_hours"] = config.dt * static_cast<double>(steps) / 3600.0;
    state.counters["burns"] = static_cast<double>(config.num_burns);
    state.counters["softening_km"] = config.softening / 1.0e3;
    state.counters["moon_phase_deg"] = config.moon_phase * 180.0 / mbs::kPi;
    state.counters["epoch_days"] = config.epoch / 86400.0;
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

    HostWorkload(int body_count, int step_count, bool with_trajectory, const WorkloadConfig& shape)
        : initial(initial_states(body_count, shape)),
          final(static_cast<std::size_t>(body_count)),
          bodies(body_count),
          steps(step_count),
          write_trajectory(with_trajectory) {
        if (with_trajectory) {
            const std::size_t frames = static_cast<std::size_t>(step_count) + 1;
            trajectory.resize(frames * static_cast<std::size_t>(body_count));
        }
        apply_workload_config(config, shape);
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

    DeviceWorkload(int body_count, int step_count, bool with_trajectory, const WorkloadConfig& shape)
        : host_initial(initial_states(body_count, shape)),
          bodies(body_count),
          steps(step_count),
          write_trajectory(with_trajectory) {
        apply_workload_config(config, shape);

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

    // Warm-up: run until ~50 ms of GPU time has elapsed so clocks ramp up
    // and the first (cold) sample is not folded into the reported average.
    {
        GpuTimer w;
        if (!w.ok()) { state.SkipWithError("cudaEventCreate failed"); return; }
        float acc = 0.0f;
        while (acc < 50.0f) {
            MBS_CUDA_CHECK_OR_SKIP(state, w.begin(), "warm-up record failed");
            MBS_CUDA_CHECK_OR_SKIP(state, workload.run(), "warm-up call failed");
            float ms = 0.0f;
            MBS_CUDA_CHECK_OR_SKIP(state, w.end_ms(&ms), "warm-up elapsed failed");
            MBS_CUDA_CHECK_OR_SKIP(state, cudaDeviceSynchronize(), "warm-up sync failed");
            acc += ms;
            if (acc == 0.0f) break;  // avoid infinite loop on zero-time stubs
        }
    }

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
        MBS_CUDA_CHECK_OR_SKIP(state, timer.begin(), "event record failed");
        const int rc = workload.run();
        float ms = 0.0f;
        MBS_CUDA_CHECK_OR_SKIP(state, timer.end_ms(&ms), "event elapsed failed");
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

BENCHMARK_MAIN();
