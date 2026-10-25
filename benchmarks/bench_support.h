// Shared helpers for the CUDA and backend-comparison benchmarks.
//
// Kept header-only so multibody_bench.cpp and backend_bench.cpp use exactly
// the same workload shapes, initial states, and CUDA-event timing convention.
#pragma once

#include "multibody_solver.cuh"

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace bench_support {

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

inline void apply_workload_config(MbsConfig& config, const WorkloadConfig& shape) {
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
inline MbsVec3 earth_position_at_epoch(const WorkloadConfig& shape) {
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

inline MbsState initial_state_for(int index, const MbsVec3& earth) {
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

inline std::vector<MbsState> initial_states(int bodies, const WorkloadConfig& shape) {
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

inline void set_shape_counters(benchmark::State& state, int bodies, int steps, bool trajectory, const MbsConfig& config) {
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

inline void set_throughput(benchmark::State& state, double total_ms, int bodies, int steps,
                           const char* ms_counter = "gpu_ms_per_call") {
    const double seconds = total_ms / 1.0e3;
    const double calls = static_cast<double>(state.iterations());
    const double particle_steps = calls * static_cast<double>(bodies) * steps;
    state.counters[ms_counter] = calls > 0.0 ? total_ms / calls : 0.0;
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
}

}  // namespace bench_support
