//! C ABI wrapper used by `benchmarks/backend_bench.cpp` to compare the Rust
//! fixed-step propagator with the CUDA kernel in the same process and with the
//! same initial states.
//!
//! The wrapper is deliberately narrow: it accepts a flat array of states and
//! the scalar configuration fields the CUDA benchmark varies, calls
//! [`dynamics::propagate_batch_in_place`], and writes the final states back.
//! It is a benchmark shim, not a general-purpose FFI surface.

use std::slice;

use dynamics::{propagate_batch_in_place, Burn, Config, State, Vec3, MAX_BURNS};

/// C-compatible state. Matches `DynBenchState` in `backend_bench.cpp`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct DynamicsBenchState {
    pub r: [f64; 3],
    pub v: [f64; 3],
}

fn to_state(state: &DynamicsBenchState) -> State {
    State {
        r: Vec3::new(state.r[0], state.r[1], state.r[2]),
        v: Vec3::new(state.v[0], state.v[1], state.v[2]),
    }
}

fn from_state(state: &State) -> DynamicsBenchState {
    DynamicsBenchState {
        r: [state.r.x, state.r.y, state.r.z],
        v: [state.v.x, state.v.y, state.v.z],
    }
}

/// Propagate `num_bodies` states in place for `num_steps` fixed steps.
///
/// Returns `0` on success, `-1` for invalid pointers/counts/step size, and
/// `-2` when the generated configuration fails [`Config::validate`].
///
/// # Safety
///
/// `states` must point to `num_bodies` writable, properly aligned
/// [`DynamicsBenchState`] values for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn dynamics_bench_propagate(
    states: *mut DynamicsBenchState,
    num_bodies: i32,
    num_steps: i32,
    dt: f64,
    epoch: f64,
    perturbation: i32,
    moon_distance: f64,
    moon_phase: f64,
    softening: f64,
    num_burns: i32,
    burn_start_step: i32,
    burn_stride_steps: i32,
    burn_dv: f64,
) -> i32 {
    if states.is_null() || num_bodies <= 0 || num_steps < 0 {
        return -1;
    }
    if !dt.is_finite() || dt <= 0.0 {
        return -1;
    }

    let burn_count = num_burns.clamp(0, MAX_BURNS as i32) as usize;
    let config = Config {
        dt,
        epoch,
        perturbation,
        moon_distance,
        moon_phase,
        softening,
        burns: (0..burn_count)
            .map(|b| {
                let step = burn_start_step + (b as i32) * burn_stride_steps;
                Burn {
                    t: epoch + f64::from(step) * dt,
                    dv: Vec3::new(0.0, burn_dv, 0.0),
                }
            })
            .collect(),
        ..Config::default()
    };

    if config.validate().is_err() {
        return -2;
    }

    let bodies = num_bodies as usize;
    let ffi_states = slice::from_raw_parts_mut(states, bodies);
    let mut rust_states: Vec<State> = ffi_states.iter().map(to_state).collect();
    propagate_batch_in_place(&mut rust_states, num_steps as usize, dt, &config);
    for (dst, src) in ffi_states.iter_mut().zip(rust_states.iter()) {
        *dst = from_state(src);
    }

    0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state() -> DynamicsBenchState {
        DynamicsBenchState {
            r: [7.0e6, 0.0, 0.0],
            v: [0.0, 7.5e3, 0.0],
        }
    }

    #[test]
    fn propagates_and_writes_back() {
        let mut states = [state()];
        let rc = unsafe {
            dynamics_bench_propagate(
                states.as_mut_ptr(),
                states.len() as i32,
                10,
                60.0,
                0.0,
                0,
                3.844e8,
                0.0,
                0.0,
                0,
                0,
                0,
                0.0,
            )
        };
        assert_eq!(rc, 0);
        assert_ne!(states[0], state());
        assert!(states[0].r.iter().all(|v| v.is_finite()));
        assert!(states[0].v.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn matches_the_direct_batch_api() {
        let mut via_ffi = [state()];
        let rc = unsafe {
            dynamics_bench_propagate(
                via_ffi.as_mut_ptr(),
                1,
                25,
                60.0,
                0.0,
                1,
                3.844e8,
                0.5,
                1000.0,
                4,
                10,
                10,
                0.5,
            )
        };
        assert_eq!(rc, 0);

        let config = Config {
            dt: 60.0,
            perturbation: 1,
            moon_phase: 0.5,
            softening: 1000.0,
            burns: (0..4)
                .map(|b| Burn {
                    t: 60.0 * f64::from(10 + b * 10),
                    dv: Vec3::new(0.0, 0.5, 0.0),
                })
                .collect(),
            ..Config::default()
        };
        let mut direct = [to_state(&state())];
        propagate_batch_in_place(&mut direct, 25, 60.0, &config);
        assert_eq!(via_ffi[0], from_state(&direct[0]));
    }

    #[test]
    fn rejects_invalid_arguments() {
        let mut states = [state()];
        let rc = unsafe {
            dynamics_bench_propagate(
                std::ptr::null_mut(),
                1,
                1,
                60.0,
                0.0,
                0,
                3.844e8,
                0.0,
                0.0,
                0,
                0,
                0,
                0.0,
            )
        };
        assert_eq!(rc, -1);
        let rc = unsafe {
            dynamics_bench_propagate(
                states.as_mut_ptr(),
                1,
                1,
                -1.0,
                0.0,
                0,
                3.844e8,
                0.0,
                0.0,
                0,
                0,
                0,
                0.0,
            )
        };
        assert_eq!(rc, -1);
        let rc = unsafe {
            dynamics_bench_propagate(
                states.as_mut_ptr(),
                1,
                1,
                60.0,
                0.0,
                7,
                3.844e8,
                0.0,
                0.0,
                0,
                0,
                0,
                0.0,
            )
        };
        assert_eq!(rc, -2);
    }
}
