// CUDA multibody solver for the Earth-Moon restricted problem.
//
// Build (standalone object):
//   nvcc -O3 -std=c++17 -c multibody_solver.cu
//
// All units are SI: metres, seconds, m^3/s^2.

#include "multibody_solver.cuh"

#include <cstddef>
#include <limits>

namespace {

// One thread owns one particle for its whole trajectory. The grid-stride loop
// keeps the kernel valid for batches larger than the launch grid. When
// trajectory is non-null, frame 0 is the initial state and frame step + 1 is
// written after each integration step.
__global__ void multibody_kernel(MbsState* states, int num_bodies, int num_steps, MbsConfig config, MbsState* trajectory) {
    const int stride = blockDim.x * gridDim.x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_bodies;
         i += stride) {
        // Keep the particle state in registers across the time loop.
        MbsState state = states[i];
        double t = config.epoch;

        if (trajectory != nullptr) {
            trajectory[i] = state;  // frame 0
        }

        for (int step = 0; step < num_steps; ++step) {
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

    ~DeviceArray() {
        if (data != nullptr) {
            cudaFree(data);
        }
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
        !std::isfinite(config->moon_period) ||
        config->moon_period < 0.0 ||
        !std::isfinite(config->moon_phase)) {
        return false;
    }
    if (!std::isfinite(config->softening) || config->softening < 0.0) {
        return false;
    }
    return config->perturbation == 0 || config->perturbation == 1;
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
}

// Host orchestration: validate, size buffers, upload, launch, download.
// The call is synchronous and returns the raw cudaError_t value.
int mbs_propagate(const MbsState* initial_states, int num_bodies, int num_steps, const MbsConfig* config, MbsState* final_states, MbsState* trajectory) {
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
int mbs_propagate_device(MbsState* device_states, int num_bodies, int num_steps, const MbsConfig* config, MbsState* device_trajectory) {
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

}  // extern "C"
