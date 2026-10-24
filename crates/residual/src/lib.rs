//! Residual computation and burn masking.
//!
//! TODO: richer residual metrics and burn-window models.

use dynamics::{State, Vec3};

/// (dr, dv) residual pair.
pub type Residual = (Vec3, Vec3);

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
