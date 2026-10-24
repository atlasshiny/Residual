//! Residual computation and burn masking.
//!
//! TODO: richer residual metrics and burn-window models.

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

/// Residual rows for every reference epoch that the propagated trajectory
/// covers.  Epochs outside the propagation span are skipped rather than
/// extrapolated.
pub fn residual_series(
    reference: &[(f64, State)],
    trajectory: &[State],
    epoch: f64,
    dt: f64,
) -> Vec<ResidualRow> {
    reference
        .iter()
        .filter_map(|(t, reference_state)| {
            let propagated = resample(trajectory, epoch, dt, *t)?;
            let (dr, dv) = residual(*reference_state, propagated);
            Some(ResidualRow::new(*t, dr, dv))
        })
        .collect()
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
}
