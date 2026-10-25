// CUDA multibody solver for the Earth-Moon restricted problem.
//
// Build (standalone object):
//   nvcc -O3 -std=c++17 -c multibody_solver.cu
//
// All units are SI: metres, seconds, m^3/s^2.

#include "multibody_solver.cuh"

#include <algorithm>
#include <cstddef>
#include <limits>

namespace {

// One thread owns one particle for its whole trajectory. The grid-stride loop
// keeps the kernel valid for batches larger than the launch grid. When
// trajectory is non-null, frame 0 is the initial state and frame step + 1 is
// written after each integration step.
__global__ void __launch_bounds__(MBS_THREADS_PER_BLOCK)
multibody_kernel(MbsState* __restrict__ states, int num_bodies, int num_steps,
                 MbsConfig config, MbsState* __restrict__ trajectory) {
    const int stride = blockDim.x * gridDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_bodies;
         i += stride) {
        // Keep the particle state in registers across the time loop.
        MbsState state = states[i];
        double t = config.epoch;

        // Burns are scheduled on integration steps.  Precompute the nearest
        // step once per particle so the hot loop avoids a floating-point
        // comparison against every burn on every step.
        int burn_steps[8];
        for (int b = 0; b < config.num_burns && b < 8; ++b) {
            burn_steps[b] = static_cast<int>(std::llround(
                (config.burns[b].t - config.epoch) / config.dt));
        }

        if (trajectory != nullptr) {
            trajectory[i] = state;  // frame 0
        }

        for (int step = 0; step < num_steps; ++step) {
            // Apply impulsive burns firing at this step's start time.
            for (int b = 0; b < config.num_burns && b < 8; ++b) {
                if (step == burn_steps[b] &&
                    std::fabs(t - config.burns[b].t) < 0.5 * config.dt) {
                    state.v.x += config.burns[b].dvx;
                    state.v.y += config.burns[b].dvy;
                    state.v.z += config.burns[b].dvz;
                }
            }
            mbs::velocity_verlet_step(state, t, config.dt, config);
            t = config.epoch + static_cast<double>(step + 1) * config.dt;

            if (trajectory != nullptr) {
                // Layout is [frame][body], so frames stay contiguous.
                trajectory[(static_cast<std::size_t>(step) + 1) *
                               static_cast<std::size_t>(num_bodies) +
                           static_cast<std::size_t>(i)] = state;
            }
        }

        states[i] = state;
    }
}

// RAII owner for a cudaMalloc buffer. Every early return in mbs_propagate
// releases both device allocations through this destructor.
template <typename T>
struct DeviceArray {
    T* data = nullptr;

    DeviceArray() = default;
    ~DeviceArray() {
        if (data != nullptr) {
            cudaFree(data);
        }
    }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
    DeviceArray(DeviceArray&& other) noexcept : data(other.data) {
        other.data = nullptr;
    }
    DeviceArray& operator=(DeviceArray&& other) noexcept {
        if (this != &other) {
            if (data != nullptr) {
                cudaFree(data);
            }
            data = other.data;
            other.data = nullptr;
        }
        return *this;
    }

    cudaError_t allocate(std::size_t count) {
        return cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T));
    }
};

// Reject configurations that would produce NaNs or divide by zero.
bool valid_config(const MbsConfig* config) {
    if (config == nullptr) {
        return false;
    }
    if (!std::isfinite(config->dt) || config->dt <= 0.0 ||
        !std::isfinite(config->epoch)) {
        return false;
    }
    if (!std::isfinite(config->mu_earth) || config->mu_earth < 0.0 ||
        !std::isfinite(config->mu_moon) || config->mu_moon < 0.0) {
        return false;
    }
    if (!std::isfinite(config->earth_radius) || config->earth_radius < 0.0 ||
        !std::isfinite(config->j2)) {
        return false;
    }
    if (!std::isfinite(config->moon_distance) ||
        config->moon_distance < 0.0 ||
        (config->moon_distance > 0.0 && config->moon_distance < config->earth_radius) ||
        !std::isfinite(config->moon_period) ||
        config->moon_period < 0.0 ||
        !std::isfinite(config->moon_phase)) {
        return false;
    }
    if (!std::isfinite(config->softening) || config->softening < 0.0) {
        return false;
    }
    if (!(config->perturbation == 0 || config->perturbation == 1)) {
        return false;
    }
    if (config->num_burns < 0 || config->num_burns > 8) {
        return false;
    }
    if (!std::isfinite(config->spin_axis_x) ||
        !std::isfinite(config->spin_axis_y) ||
        !std::isfinite(config->spin_axis_z)) {
        return false;
    }
    for (int i = 0; i < config->num_burns; ++i) {
        if (!std::isfinite(config->burns[i].t) ||
            !std::isfinite(config->burns[i].dvx) ||
            !std::isfinite(config->burns[i].dvy) ||
            !std::isfinite(config->burns[i].dvz)) {
            return false;
        }
    }
    return true;
}

}  // namespace

extern "C" {

void mbs_default_config(MbsConfig* config) {
    if (config == nullptr) {
        return;
    }
    // J2 is opt-in; set perturbation = 1 to enable it.
    config->dt = 60.0;
    config->epoch = 0.0;
    config->mu_earth = mbs::kMuEarth;
    config->mu_moon = mbs::kMuMoon;
    config->earth_radius = mbs::kEarthRadius;
    config->j2 = mbs::kEarthJ2;
    config->moon_distance = mbs::kEarthMoonDistance;
    config->moon_period = mbs::kMoonPeriod;
    config->moon_phase = 0.0;
    config->softening = 0.0;
    config->perturbation = 0;
    config->num_burns = 0;
    config->spin_axis_x = 0.0;
    config->spin_axis_y = 0.0;
    config->spin_axis_z = 1.0;
}

// Host orchestration: validate, size buffers, upload, launch, download.
// The call is synchronous and returns the raw cudaError_t value.
int mbs_propagate(const MbsState* __restrict__ initial_states, int num_bodies, int num_steps, const MbsConfig* config, MbsState* __restrict__ final_states, MbsState* __restrict__ trajectory) {
    if (initial_states == nullptr || !valid_config(config) ||
        num_bodies <= 0 || num_steps < 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (final_states == nullptr && trajectory == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    // frames includes the initial state, hence num_steps + 1.
    const std::size_t bodies = static_cast<std::size_t>(num_bodies);
    const std::size_t frames = static_cast<std::size_t>(num_steps) + 1;

    std::size_t trajectory_states = 0;
    if (trajectory != nullptr) {
        // Check element-count overflow before allocating or copying.
        if (frames > std::numeric_limits<std::size_t>::max() / bodies) {
            return static_cast<int>(cudaErrorInvalidValue);
        }
        trajectory_states = frames * bodies;
        if (trajectory_states >
            std::numeric_limits<std::size_t>::max() / sizeof(MbsState)) {
            return static_cast<int>(cudaErrorInvalidValue);
        }
    }

    const std::size_t state_bytes = bodies * sizeof(MbsState);

    DeviceArray<MbsState> device_states;
    MBS_CUDA_CHECK(device_states.allocate(bodies));

    DeviceArray<MbsState> device_trajectory;
    if (trajectory != nullptr) {
        MBS_CUDA_CHECK(device_trajectory.allocate(trajectory_states));
    }

    // Upload the initial states before the kernel starts.
    MBS_CUDA_CHECK(cudaMemcpy(device_states.data, initial_states, state_bytes, cudaMemcpyHostToDevice));

    // One thread per particle; the grid-stride loop covers larger batches.
    const int blocks = MBS_BLOCKS_FOR(num_bodies);

    multibody_kernel<<<blocks, MBS_THREADS_PER_BLOCK>>>(device_states.data, num_bodies, num_steps, *config, trajectory != nullptr ? device_trajectory.data : nullptr);

    // Surface launch errors before attempting the device-to-host copies.
    MBS_CUDA_CHECK(cudaGetLastError());
    if (final_states != nullptr) {
        MBS_CUDA_CHECK(cudaMemcpy(final_states, device_states.data, state_bytes, cudaMemcpyDeviceToHost));
    }
    if (trajectory != nullptr) {
        MBS_CUDA_CHECK(cudaMemcpy(trajectory, device_trajectory.data, trajectory_states * sizeof(MbsState), cudaMemcpyDeviceToHost));
    }
    // Also catches asynchronous kernel failures.
    MBS_CUDA_CHECK(cudaStreamSynchronize(0));

    return static_cast<int>(cudaSuccess);
}

// Kernel-only path: no allocation, no copies, no synchronisation.  The
// benchmark suite uses this to separate GPU compute time from memory traffic.
int mbs_propagate_device(MbsState* __restrict__ device_states, int num_bodies, int num_steps, const MbsConfig* config, MbsState* __restrict__ device_trajectory) {
    if (device_states == nullptr || !valid_config(config) ||
        num_bodies <= 0 || num_steps < 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    // One thread per particle; the grid-stride loop covers larger batches.
    const int blocks = MBS_BLOCKS_FOR(num_bodies);

    // Drop any sticky error left by the caller so it is not attributed here.
    cudaGetLastError();
    multibody_kernel<<<blocks, MBS_THREADS_PER_BLOCK>>>(device_states, num_bodies, num_steps, *config, device_trajectory);
    return static_cast<int>(cudaGetLastError());
}

namespace {

constexpr int MBS_ENSEMBLE_TILE_SIZE = 256;

__device__ void atomic_min_double(double* address, double value) {
    unsigned long long* address_as_ull = reinterpret_cast<unsigned long long*>(address);
    unsigned long long old = *address_as_ull;
    while (value < __longlong_as_double(old)) {
        unsigned long long assumed = old;
        unsigned long long result = atomicCAS(address_as_ull, assumed, __double_as_longlong(value));
        if (result == assumed) {
            break;
        }
        old = result;
    }
}

__device__ void atomic_max_double(double* address, double value) {
    unsigned long long* address_as_ull = reinterpret_cast<unsigned long long*>(address);
    unsigned long long old = *address_as_ull;
    while (value > __longlong_as_double(old)) {
        unsigned long long assumed = old;
        unsigned long long result = atomicCAS(address_as_ull, assumed, __double_as_longlong(value));
        if (result == assumed) {
            break;
        }
        old = result;
    }
}

// One block cooperatively loads a tile of particles into shared memory, then
// each thread accumulates the contribution of its assigned particle.  The
// tile keeps the global loads coalesced and gives the reduction a bounded
// working set; atomics combine the per-block partial results.
__global__ void __launch_bounds__(MBS_THREADS_PER_BLOCK)
ensemble_stats_kernel(const MbsState* __restrict__ states, int num_bodies,
                      MbsEnsembleStats* __restrict__ stats) {
    __shared__ MbsState tile[MBS_ENSEMBLE_TILE_SIZE];

    const int tid = threadIdx.x;
    double sx = 0.0;
    double sy = 0.0;
    double sz = 0.0;
    double svx = 0.0;
    double svy = 0.0;
    double svz = 0.0;
    double sr2 = 0.0;
    double sv2 = 0.0;
    double min_r2 = 0.0;
    double max_r2 = 0.0;
    bool has_particle = false;

    for (int tile_start = 0; tile_start < num_bodies; tile_start += MBS_ENSEMBLE_TILE_SIZE) {
        const int index = tile_start + tid;
        if (index < num_bodies) {
            tile[tid] = states[index];
        }
        __syncthreads();

        const int tile_count = std::min(MBS_ENSEMBLE_TILE_SIZE, num_bodies - tile_start);
        if (tid < tile_count) {
            const MbsState state = tile[tid];
            const double r2 = state.r.x * state.r.x + state.r.y * state.r.y + state.r.z * state.r.z;
            const double v2 = state.v.x * state.v.x + state.v.y * state.v.y + state.v.z * state.v.z;

            sx += state.r.x;
            sy += state.r.y;
            sz += state.r.z;
            svx += state.v.x;
            svy += state.v.y;
            svz += state.v.z;
            sr2 += r2;
            sv2 += v2;

            if (!has_particle) {
                min_r2 = r2;
                max_r2 = r2;
                has_particle = true;
            } else {
                min_r2 = (r2 < min_r2) ? r2 : min_r2;
                max_r2 = (r2 > max_r2) ? r2 : max_r2;
            }
        }
        __syncthreads();
    }

    if (has_particle) {
        atomicAdd(&stats->sum_x, sx);
        atomicAdd(&stats->sum_y, sy);
        atomicAdd(&stats->sum_z, sz);
        atomicAdd(&stats->sum_vx, svx);
        atomicAdd(&stats->sum_vy, svy);
        atomicAdd(&stats->sum_vz, svz);
        atomicAdd(&stats->sum_r2, sr2);
        atomicAdd(&stats->sum_v2, sv2);
        atomicAdd(&stats->count, 1);
        atomic_min_double(&stats->min_r2, min_r2);
        atomic_max_double(&stats->max_r2, max_r2);
    }
}

}  // namespace

int mbs_ensemble_stats(const MbsState* __restrict__ states, int num_bodies, MbsEnsembleStats* __restrict__ stats) {
    if (states == nullptr || stats == nullptr || num_bodies <= 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    MbsEnsembleStats initial{};
    initial.min_r2 = std::numeric_limits<double>::infinity();
    initial.max_r2 = -std::numeric_limits<double>::infinity();

    DeviceArray<MbsEnsembleStats> device_stats;
    MBS_CUDA_CHECK(device_stats.allocate(1));
    MBS_CUDA_CHECK(cudaMemcpy(device_stats.data, &initial, sizeof(MbsEnsembleStats), cudaMemcpyHostToDevice));

    const int blocks = MBS_BLOCKS_FOR(num_bodies);
    ensemble_stats_kernel<<<blocks, MBS_THREADS_PER_BLOCK>>>(states, num_bodies, device_stats.data);
    MBS_CUDA_CHECK(cudaGetLastError());
    MBS_CUDA_CHECK(cudaStreamSynchronize(0));
    MBS_CUDA_CHECK(cudaMemcpy(stats, device_stats.data, sizeof(MbsEnsembleStats), cudaMemcpyDeviceToHost));

    return static_cast<int>(cudaSuccess);
}

int mbs_ensemble_stats_device(const MbsState* __restrict__ device_states, int num_bodies, MbsEnsembleStats* __restrict__ device_stats) {
    if (device_states == nullptr || device_stats == nullptr || num_bodies <= 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    MbsEnsembleStats initial{};
    initial.min_r2 = std::numeric_limits<double>::infinity();
    initial.max_r2 = -std::numeric_limits<double>::infinity();
    MBS_CUDA_CHECK(cudaMemcpy(device_stats, &initial, sizeof(MbsEnsembleStats), cudaMemcpyHostToDevice));

    cudaGetLastError();
    const int blocks = MBS_BLOCKS_FOR(num_bodies);
    ensemble_stats_kernel<<<blocks, MBS_THREADS_PER_BLOCK>>>(device_states, num_bodies, device_stats);
    return static_cast<int>(cudaGetLastError());
}

}  // extern "C"
