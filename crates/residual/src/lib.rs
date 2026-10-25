//! Residual computation, burn masking and conservation diagnostics.
//!
//! Beyond raw residuals this module reports energy and angular-momentum
//! conservation drift and splits position error by burn window via
//! [`ResidualMetrics`].

use dynamics::{State, Vec3};

/// (dr, dv) residual pair.
pub type Residual = (Vec3, Vec3);

/// One time-tagged residual row, ready to be written to the residual table.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResidualRow {
    pub t: f64,
    pub dr: Vec3,
    pub dv: Vec3,
}

impl ResidualRow {
    pub fn new(t: f64, dr: Vec3, dv: Vec3) -> Self {
        Self { t, dr, dv }
    }

    /// Euclidean position error in metres.
    pub fn position_norm(&self) -> f64 {
        self.dr.norm()
    }

    /// Euclidean velocity error in m/s.
    pub fn velocity_norm(&self) -> f64 {
        self.dv.norm()
    }
}

/// One residual row together with the propagated state it was computed from.
///
/// The propagated state is retained so conservation metrics (specific energy
/// and angular momentum) can be derived alongside the raw residual.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResidualSample {
    pub t: f64,
    pub dr: Vec3,
    pub dv: Vec3,
    /// Propagated state at `t`.
    pub propagated: State,
}

impl ResidualSample {
    pub fn new(t: f64, dr: Vec3, dv: Vec3, propagated: State) -> Self {
        Self {
            t,
            dr,
            dv,
            propagated,
        }
    }
}

/// Trait allowing residual() to accept State or &State.
pub trait AsState {
    fn as_state(&self) -> State;
}
impl AsState for State {
    fn as_state(&self) -> State {
        *self
    }
}
impl AsState for &State {
    fn as_state(&self) -> State {
        **self
    }
}

/// Compute residual = reference - propagated.
pub fn residual<R, P>(reference: R, propagated: P) -> Residual
where
    R: AsState,
    P: AsState,
{
    let r = reference.as_state();
    let p = propagated.as_state();
    (r.r.sub(p.r), r.v.sub(p.v))
}

/// Trait for mask entries that can be interpreted as keep/drop.
pub trait MaskBit {
    fn keep(self) -> bool;
}
impl MaskBit for bool {
    fn keep(self) -> bool {
        self
    }
}
impl MaskBit for i32 {
    fn keep(self) -> bool {
        self != 0
    }
}
impl MaskBit for i64 {
    fn keep(self) -> bool {
        self != 0
    }
}
impl MaskBit for u8 {
    fn keep(self) -> bool {
        self != 0
    }
}
impl MaskBit for usize {
    fn keep(self) -> bool {
        self != 0
    }
}

/// Filter residuals by a mask (truthy keeps).
pub fn mask_burns<M>(residuals: &[Residual], mask: &[M]) -> Vec<Residual>
where
    M: MaskBit + Copy,
{
    residuals
        .iter()
        .zip(mask.iter())
        .filter_map(|(r, &m)| if m.keep() { Some(*r) } else { None })
        .collect()
}

/// Linear interpolation of a regularly sampled trajectory.
///
/// `trajectory[0]` is assumed to be at `epoch`, with consecutive samples
/// separated by `dt`.  Returns `None` when `t` falls outside the trajectory.
pub fn resample(trajectory: &[State], epoch: f64, dt: f64, t: f64) -> Option<State> {
    if trajectory.is_empty() || !dt.is_finite() || dt <= 0.0 || !t.is_finite() {
        return None;
    }
    let last_index = trajectory.len() - 1;
    let last_time = epoch + last_index as f64 * dt;
    if t < epoch || t > last_time {
        return None;
    }
    let x = (t - epoch) / dt;
    let lo = x.floor() as usize;
    if lo >= last_index {
        return Some(trajectory[last_index]);
    }
    let alpha = x - lo as f64;
    let a = trajectory[lo];
    let b = trajectory[lo + 1];
    Some(State {
        r: lerp(a.r, b.r, alpha),
        v: lerp(a.v, b.v, alpha),
    })
}

fn lerp(a: Vec3, b: Vec3, alpha: f64) -> Vec3 {
    Vec3::new(
        a.x + (b.x - a.x) * alpha,
        a.y + (b.y - a.y) * alpha,
        a.z + (b.z - a.z) * alpha,
    )
}

pub fn residual_samples(
    reference: &[(f64, State)],
    trajectory: &[State],
    epoch: f64,
    dt: f64,
) -> Vec<ResidualSample> {
    reference
        .iter()
        .filter_map(|(t, reference_state)| {
            let propagated = resample(trajectory, epoch, dt, *t)?;
            let (dr, dv) = residual(*reference_state, propagated);
            Some(ResidualSample::new(*t, dr, dv, propagated))
        })
        .collect()
}

/// Residual rows for every reference epoch that the propagated trajectory
/// covers.  Epochs outside the propagation span are skipped rather than
/// extrapolated.
pub fn residual_series(
    reference: &[(f64, State)],
    trajectory: &[State],
    epoch: f64,
    dt: f64,
) -> Vec<ResidualRow> {
    residual_samples(reference, trajectory, epoch, dt)
        .into_iter()
        .map(|s| ResidualRow::new(s.t, s.dr, s.dv))
        .collect()
}

/// Cross product `a × b` (`dynamics::Vec3` does not provide one).
fn cross(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    )
}

/// Specific orbital energy (kinetic + potential) of a state.
///
/// `mu` is the gravitational parameter of the central body; the potential is
/// `-mu / |r|`, so the total is `0.5 * |v|^2 - mu / |r|`.
pub fn specific_energy(state: &State, mu: f64) -> f64 {
    let r = state.r.norm();
    let kinetic = 0.5 * state.v.dot(state.v);
    let potential = if r > 0.0 { -mu / r } else { 0.0 };
    kinetic + potential
}

/// Specific angular momentum magnitude `|r × v|`.
pub fn specific_angular_momentum(state: &State) -> f64 {
    cross(state.r, state.v).norm()
}

/// Largest absolute drift of specific energy across `states`, measured against
/// the first sample.  Returns `0.0` for an empty slice.
pub fn energy_drift_max(states: &[State], mu: f64) -> f64 {
    let Some(first) = states.first() else {
        return 0.0;
    };
    let e0 = specific_energy(first, mu);
    states[1..]
        .iter()
        .map(|s| (specific_energy(s, mu) - e0).abs())
        .fold(0.0_f64, f64::max)
}

/// Largest absolute drift of `|r × v|` across `states`, measured against the
/// first sample.  Returns `0.0` for an empty slice.
pub fn angular_momentum_drift_max(states: &[State]) -> f64 {
    let Some(first) = states.first() else {
        return 0.0;
    };
    let l0 = specific_angular_momentum(first);
    states[1..]
        .iter()
        .map(|s| (specific_angular_momentum(s) - l0).abs())
        .fold(0.0_f64, f64::max)
}

/// Aggregate residual and conservation metrics over a sample set.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ResidualMetrics {
    /// Root-mean-square position residual (metres) over all samples.
    pub pos_rmse: f64,
    /// Root-mean-square velocity residual (m/s) over all samples.
    pub vel_rmse: f64,
    /// Maximum absolute drift of specific orbital energy (J/kg), measured over
    /// the out-of-burn samples.
    pub energy_drift_max: f64,
    /// Maximum absolute drift of specific angular momentum (m²/s), measured
    /// over the out-of-burn samples.
    pub angular_momentum_drift_max: f64,
    /// Number of samples flagged as inside a burn window.
    pub burn_masked_count: usize,
    /// Position RMSE (metres) over in-burn samples; `0.0` when there are none.
    pub in_burn_pos_rmse: f64,
    /// Position RMSE (metres) over out-of-burn samples; `0.0` when there are none.
    pub out_of_burn_pos_rmse: f64,
}

/// Root-mean-square of a value slice; `0.0` for an empty slice.
fn rms(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        (values.iter().map(|v| v * v).sum::<f64>() / values.len() as f64).sqrt()
    }
}

/// Aggregate residual, energy-conservation, angular-momentum and burn-window
/// metrics for a set of [`ResidualSample`]s.
///
/// `burn_mask`, when supplied, is a per-sample flag whose truthy entries mark
/// samples inside a burn window.  Energy and angular-momentum drifts are only
/// measured over out-of-burn samples because impulsive burns legitimately
/// change both quantities.
pub fn compute_metrics<M>(
    samples: &[ResidualSample],
    mu: f64,
    burn_mask: Option<&[M]>,
) -> ResidualMetrics
where
    M: MaskBit + Copy,
{
    let is_burn = |i: usize| {
        burn_mask
            .and_then(|mask| mask.get(i))
            .map(|m| m.keep())
            .unwrap_or(false)
    };

    let mut all_pos = Vec::with_capacity(samples.len());
    let mut all_vel = Vec::with_capacity(samples.len());
    let mut in_burn_pos = Vec::new();
    let mut out_of_burn_pos = Vec::new();
    let mut out_of_burn_states = Vec::new();
    let mut burn_masked_count = 0usize;

    for (i, sample) in samples.iter().enumerate() {
        let dr = sample.dr.norm();
        all_pos.push(dr);
        all_vel.push(sample.dv.norm());
        if is_burn(i) {
            burn_masked_count += 1;
            in_burn_pos.push(dr);
        } else {
            out_of_burn_pos.push(dr);
            out_of_burn_states.push(sample.propagated);
        }
    }

    ResidualMetrics {
        pos_rmse: rms(&all_pos),
        vel_rmse: rms(&all_vel),
        energy_drift_max: energy_drift_max(&out_of_burn_states, mu),
        angular_momentum_drift_max: angular_momentum_drift_max(&out_of_burn_states),
        burn_masked_count,
        in_burn_pos_rmse: rms(&in_burn_pos),
        out_of_burn_pos_rmse: rms(&out_of_burn_pos),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamics::Config;

    #[test]
    fn residual_difference() {
        let cfg = Config::default();
        let _ = cfg;
        let r_state = State {
            r: Vec3::new(1.0, 2.0, 3.0),
            v: Vec3::new(4.0, 5.0, 6.0),
        };
        let p_state = State {
            r: Vec3::new(0.5, 1.0, 1.5),
            v: Vec3::new(1.0, 2.0, 3.0),
        };
        let (dr, dv) = residual(r_state, p_state);
        assert_eq!(dr, Vec3::new(0.5, 1.0, 1.5));
        assert_eq!(dv, Vec3::new(3.0, 3.0, 3.0));
    }

    #[test]
    fn resample_hits_samples_and_midpoints() {
        let traj = vec![
            State {
                r: Vec3::new(0.0, 0.0, 0.0),
                v: Vec3::new(0.0, 0.0, 0.0),
            },
            State {
                r: Vec3::new(10.0, 0.0, 0.0),
                v: Vec3::new(2.0, 0.0, 0.0),
            },
        ];
        assert_eq!(resample(&traj, 0.0, 60.0, 0.0), Some(traj[0]));
        assert_eq!(resample(&traj, 0.0, 60.0, 60.0), Some(traj[1]));
        let mid = resample(&traj, 0.0, 60.0, 30.0).unwrap();
        assert!((mid.r.x - 5.0).abs() < 1e-12);
        assert!((mid.v.x - 1.0).abs() < 1e-12);
        assert!(resample(&traj, 0.0, 60.0, -1.0).is_none());
        assert!(resample(&traj, 0.0, 60.0, 120.0).is_none());
    }

    #[test]
    fn residual_series_skips_out_of_range_epochs() {
        let traj = vec![
            State {
                r: Vec3::new(1.0, 0.0, 0.0),
                v: Vec3::ZERO,
            },
            State {
                r: Vec3::new(2.0, 0.0, 0.0),
                v: Vec3::ZERO,
            },
        ];
        let reference = vec![
            (
                0.0,
                State {
                    r: Vec3::new(1.5, 0.0, 0.0),
                    v: Vec3::ZERO,
                },
            ),
            (
                60.0,
                State {
                    r: Vec3::new(2.5, 0.0, 0.0),
                    v: Vec3::ZERO,
                },
            ),
            (
                120.0,
                State {
                    r: Vec3::new(9.0, 0.0, 0.0),
                    v: Vec3::ZERO,
                },
            ),
        ];
        let rows = residual_series(&reference, &traj, 0.0, 60.0);
        assert_eq!(rows.len(), 2);
        assert!((rows[0].dr.x - 0.5).abs() < 1e-12);
        assert!((rows[1].dr.x - 0.5).abs() < 1e-12);
    }

    #[test]
    fn mask_filters() {
        let residuals = vec![
            (Vec3::new(1.0, 0.0, 0.0), Vec3::ZERO),
            (Vec3::new(2.0, 0.0, 0.0), Vec3::ZERO),
            (Vec3::new(3.0, 0.0, 0.0), Vec3::ZERO),
        ];
        let mask = vec![true, false, true];
        let out = mask_burns(&residuals, &mask);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].0.x, 1.0);
        assert_eq!(out[1].0.x, 3.0);
    }

    fn zero_state() -> State {
        State {
            r: Vec3::ZERO,
            v: Vec3::ZERO,
        }
    }

    fn circular_state(r0: f64, mu: f64) -> State {
        let v0 = (mu / r0).sqrt();
        State {
            r: Vec3::new(r0, 0.0, 0.0),
            v: Vec3::new(0.0, v0, 0.0),
        }
    }

    #[test]
    fn specific_energy_matches_circular_orbit() {
        let mu = dynamics::K_MU_EARTH;
        let r0 = 7.0e6;
        let state = circular_state(r0, mu);
        let expected = -0.5 * mu / r0;
        let got = specific_energy(&state, mu);
        assert!(
            (got - expected).abs() < 1e-9 * expected.abs(),
            "energy {got} != {expected}"
        );
    }

    #[test]
    fn specific_angular_momentum_is_cross_product_magnitude() {
        let state = State {
            r: Vec3::new(7.0e6, 0.0, 0.0),
            v: Vec3::new(0.0, 7.5e3, 0.0),
        };
        let got = specific_angular_momentum(&state);
        assert!((got - 7.0e6 * 7.5e3).abs() < 1e-6, "ang-mom {got}");
    }

    #[test]
    fn energy_drift_detects_velocity_change() {
        let mu = 3.986004418e14;
        let a = State {
            r: Vec3::new(7.0e6, 0.0, 0.0),
            v: Vec3::new(0.0, 7500.0, 0.0),
        };
        let b = State {
            r: Vec3::new(7.0e6, 0.0, 0.0),
            v: Vec3::new(0.0, 7600.0, 0.0),
        };
        let expected = 0.5 * (7600.0f64.powi(2) - 7500.0f64.powi(2));
        let drift = energy_drift_max(&[a, b], mu);
        assert!(
            (drift - expected).abs() < 1e-9 * expected.abs(),
            "drift {drift} != {expected}"
        );
        // A single sample has no drift to report.
        assert_eq!(energy_drift_max(&[a], mu), 0.0);
        assert_eq!(energy_drift_max(&[], mu), 0.0);
    }

    #[test]
    fn angular_momentum_drift_detects_velocity_change() {
        let a = State {
            r: Vec3::new(7.0e6, 0.0, 0.0),
            v: Vec3::new(0.0, 7500.0, 0.0),
        };
        let b = State {
            r: Vec3::new(7.0e6, 0.0, 0.0),
            v: Vec3::new(0.0, 7600.0, 0.0),
        };
        let expected = 7.0e6 * (7600.0 - 7500.0);
        let drift = angular_momentum_drift_max(&[a, b]);
        assert!((drift - expected).abs() < 1e-6, "drift {drift}");
        assert_eq!(angular_momentum_drift_max(&[]), 0.0);
    }

    #[test]
    fn conservation_over_propagated_circular_orbit() {
        let cfg = Config {
            moon_distance: 0.0,
            mu_moon: 0.0,
            ..Config::default()
        };
        let mu = cfg.mu_earth;
        let r0 = 7.0e6;
        let start = circular_state(r0, mu);
        let traj = dynamics::propagate(start, 200usize, 60.0, &cfg);

        let e0 = specific_energy(&start, mu);
        let l0 = specific_angular_momentum(&start);
        let e_drift = energy_drift_max(&traj, mu);
        let l_drift = angular_momentum_drift_max(&traj);
        assert!(
            e_drift < 1e-4 * e0.abs(),
            "relative energy drift {}",
            e_drift / e0.abs()
        );
        assert!(
            l_drift < 1e-6 * l0.abs(),
            "relative angular-momentum drift {}",
            l_drift / l0.abs()
        );
    }

    #[test]
    fn metrics_default_is_zero() {
        let m = ResidualMetrics::default();
        assert_eq!(m.pos_rmse, 0.0);
        assert_eq!(m.vel_rmse, 0.0);
        assert_eq!(m.energy_drift_max, 0.0);
        assert_eq!(m.angular_momentum_drift_max, 0.0);
        assert_eq!(m.burn_masked_count, 0);
        assert_eq!(m.in_burn_pos_rmse, 0.0);
        assert_eq!(m.out_of_burn_pos_rmse, 0.0);
    }

    #[test]
    fn burn_mask_partitions_metrics() {
        let samples = vec![
            ResidualSample::new(
                0.0,
                Vec3::new(3.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                zero_state(),
            ),
            ResidualSample::new(
                1.0,
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(2.0, 0.0, 0.0),
                zero_state(),
            ),
            ResidualSample::new(
                2.0,
                Vec3::new(2.0, 0.0, 0.0),
                Vec3::new(3.0, 0.0, 0.0),
                zero_state(),
            ),
        ];
        let mask = [true, false, false];
        let m = compute_metrics(&samples, 0.0, Some(&mask[..]));

        assert_eq!(m.burn_masked_count, 1);
        assert!((m.in_burn_pos_rmse - 3.0).abs() < 1e-12);
        assert!((m.out_of_burn_pos_rmse - (2.5f64).sqrt()).abs() < 1e-12);
        assert!((m.pos_rmse - (14.0f64 / 3.0).sqrt()).abs() < 1e-12);
        assert!((m.vel_rmse - (14.0f64 / 3.0).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn no_burn_mask_treats_all_samples_as_out_of_burn() {
        let samples = vec![
            ResidualSample::new(0.0, Vec3::new(1.0, 0.0, 0.0), Vec3::ZERO, zero_state()),
            ResidualSample::new(1.0, Vec3::new(2.0, 0.0, 0.0), Vec3::ZERO, zero_state()),
        ];
        let m = compute_metrics(&samples, 0.0, None::<&[bool]>);
        assert_eq!(m.burn_masked_count, 0);
        assert_eq!(m.in_burn_pos_rmse, 0.0);
        assert!((m.out_of_burn_pos_rmse - m.pos_rmse).abs() < 1e-12);
        assert!((m.pos_rmse - (2.5f64).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn drift_ignores_in_burn_samples() {
        let mu = dynamics::K_MU_EARTH;
        let r0 = 7.0e6;
        let orbit = circular_state(r0, mu);
        let mut perturbed = orbit;
        perturbed.v.y += 500.0;

        let samples = vec![
            ResidualSample::new(0.0, Vec3::ZERO, Vec3::ZERO, orbit),
            ResidualSample::new(1.0, Vec3::new(9.0, 0.0, 0.0), Vec3::ZERO, perturbed),
            ResidualSample::new(2.0, Vec3::ZERO, Vec3::ZERO, orbit),
        ];
        let mask = [false, true, false];
        let m = compute_metrics(&samples, mu, Some(&mask[..]));

        assert_eq!(m.burn_masked_count, 1);
        // The perturbed in-burn sample is excluded, so the out-of-burn
        // trajectory is perfectly conserved.
        assert!(
            m.energy_drift_max < 1e-9,
            "energy drift {}",
            m.energy_drift_max
        );
        assert!(
            m.angular_momentum_drift_max < 1e-9,
            "angular-momentum drift {}",
            m.angular_momentum_drift_max
        );
    }

    #[test]
    fn residual_samples_keep_propagated_state_and_align_with_series() {
        let traj = vec![
            State {
                r: Vec3::new(1.0, 0.0, 0.0),
                v: Vec3::ZERO,
            },
            State {
                r: Vec3::new(2.0, 0.0, 0.0),
                v: Vec3::ZERO,
            },
        ];
        let reference = vec![(
            0.0,
            State {
                r: Vec3::new(1.5, 0.0, 0.0),
                v: Vec3::ZERO,
            },
        )];
        let samples = residual_samples(&reference, &traj, 0.0, 60.0);
        assert_eq!(samples.len(), 1);
        assert!((samples[0].propagated.r.x - 1.0).abs() < 1e-12);

        let rows = residual_series(&reference, &traj, 0.0, 60.0);
        assert_eq!(rows.len(), samples.len());
        assert!((rows[0].dr.x - samples[0].dr.x).abs() < 1e-12);
        assert!((rows[0].dv.x - samples[0].dv.x).abs() < 1e-12);
    }
}
