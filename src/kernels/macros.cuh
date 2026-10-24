// Shared CUDA helpers for the multibody solver and its benchmarks.
//
// Keep this header dependency-free apart from the CUDA runtime so it can be
// included from both .cu translation units and host-only benchmark code.
// Every failure macro returns, so only use it where returning from the
// enclosing function is the right recovery.
#pragma once

#include <cuda_runtime.h>

#include <cstdio>

// ---------------------------------------------------------------------------
// Kernel launch geometry
// ---------------------------------------------------------------------------

// Threads per block used by every kernel in this project.
#ifndef MBS_THREADS_PER_BLOCK
#define MBS_THREADS_PER_BLOCK 256
#endif

// Ceiling division: number of blocks needed to cover `count` elements.
// `count` must be a non-negative int; the result is clamped to the CUDA grid
// limit.  Evaluates `count` once.
#define MBS_BLOCKS_FOR(count) \
    ((count) > 0 ? ((count) + MBS_THREADS_PER_BLOCK - 1) / MBS_THREADS_PER_BLOCK : 0)

// ---------------------------------------------------------------------------
// Attributes
// ---------------------------------------------------------------------------

#define MBS_DEVICE_INLINE __device__ __forceinline__
#define MBS_HOST_DEVICE_INLINE __host__ __device__ __forceinline__

// Silences unused-parameter warnings in stubs and partial configurations.
#define MBS_UNUSED(value) static_cast<void>(value)

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

// Check a CUDA call in a function that returns an int cudaError_t value.
// (The macro hard-codes `return static_cast<int>(...)`; only use it in
// functions whose return type is `int` and whose failure mode is "propagate
// the cudaError_t".  For other return types use MBS_CUDA_CHECK_RETURN.)
#define MBS_CUDA_CHECK(call)                                     \
    do {                                                         \
        const cudaError_t mbs_status_ = (call);                  \
        if (mbs_status_ != cudaSuccess) {                        \
            return static_cast<int>(mbs_status_);                \
        }                                                        \
    } while (0)

// Check a CUDA call and return a caller-supplied value on failure.
#define MBS_CUDA_CHECK_RETURN(call, on_error)                    \
    do {                                                         \
        const cudaError_t mbs_status_ = (call);                  \
        if (mbs_status_ != cudaSuccess) {                        \
            return (on_error);                                   \
        }                                                        \
    } while (0)

// Check a CUDA call in a void function: print a diagnostic and return.
#define MBS_CUDA_CHECK_VOID(call)                                \
    do {                                                         \
        const cudaError_t mbs_status_ = (call);                  \
        if (mbs_status_ != cudaSuccess) {                        \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", \
                         #call, __FILE__, __LINE__,              \
                         cudaGetErrorString(mbs_status_));       \
            return;                                              \
        }                                                        \
    } while (0)

// Check a CUDA call inside a Google Benchmark body: report to the benchmark
// state and return.  `state` must be the enclosing `benchmark::State&`.
#define MBS_CUDA_CHECK_OR_SKIP(state, call, message)             \
    do {                                                         \
        const cudaError_t mbs_status_ =                          \
            static_cast<cudaError_t>(call);                      \
        if (mbs_status_ != cudaSuccess) {                        \
            (state).SkipWithError(message);                      \
            return;                                              \
        }                                                        \
    } while (0)

// Check the result of the most recently launched kernel.
#define MBS_CUDA_LAST_ERROR_OR_SKIP(state, message)              \
    MBS_CUDA_CHECK_OR_SKIP(state, cudaGetLastError(), message)

// Human-readable name for a cudaError_t (or an int returned by a CUDA call).
#define MBS_CUDA_ERROR_NAME(status) \
    cudaGetErrorString(static_cast<cudaError_t>(status))
