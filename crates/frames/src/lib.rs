//! Time and reference-frame primitives for the orbital pipeline.
//!
//! The pipeline uses **TDB seconds past J2000** as its canonical time base
//! (`docs/cosmos-contract.md`). This crate provides:
//!
//! * [`Seconds`] — a duration/sample-offset newtype with arithmetic.
//! * [`Epoch`] — an absolute TDB epoch with Julian Date / MJD conversions and
//!   a Gregorian calendar projection.
//! * [`InertialFrame`] — identifiers for the quasi-inertial frames used by the
//!   ingest/posebuilder layers.
//!
//! No leap-second arithmetic is performed here: UTC↔TDB requires a leap-second
//! table and belongs in the ingest layer. `Epoch` is always TDB.

use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Julian Date of the J2000.0 reference epoch (2000-01-01 12:00:00 TDB).
pub const J2000_JD: f64 = 2_451_545.0;
/// Seconds in one day.
pub const SECONDS_PER_DAY: f64 = 86_400.0;
/// Julian Date of the Modified Julian Date (MJD) epoch.
pub const MJD_EPOCH_JD: f64 = 2_400_000.5;

/// A finite time offset in seconds.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default)]
pub struct Seconds(pub f64);

impl Seconds {
    pub const ZERO: Self = Self(0.0);

    /// Construct a duration, rejecting non-finite values.
    pub fn new(value: f64) -> Result<Self, FrameError> {
        if value.is_finite() {
            Ok(Self(value))
        } else {
            Err(FrameError::NonFinite("seconds"))
        }
    }

    /// The underlying value.
    pub const fn value(self) -> f64 {
        self.0
    }

    /// Absolute value.
    pub fn abs(self) -> Self {
        Self(self.0.abs())
    }

    /// Whether the value is finite.
    pub fn is_finite(self) -> bool {
        self.0.is_finite()
    }
}

impl From<f64> for Seconds {
    fn from(v: f64) -> Self {
        Self(v)
    }
}

impl From<Seconds> for f64 {
    fn from(s: Seconds) -> Self {
        s.0
    }
}

impl Add for Seconds {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl Sub for Seconds {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}

impl Neg for Seconds {
    type Output = Self;
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

impl Mul<f64> for Seconds {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self(self.0 * rhs)
    }
}

impl Div<f64> for Seconds {
    type Output = Self;
    fn div(self, rhs: f64) -> Self {
        Self(self.0 / rhs)
    }
}

/// An absolute TDB epoch, stored as seconds past J2000.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Epoch {
    tdb_seconds: f64,
}

impl Epoch {
    /// Epoch at J2000.0 (JD 2451545.0 TDB).
    pub const J2000: Self = Self { tdb_seconds: 0.0 };

    /// Construct from TDB seconds past J2000.
    pub fn from_tdb_seconds(seconds: f64) -> Result<Self, FrameError> {
        if seconds.is_finite() {
            Ok(Self {
                tdb_seconds: seconds,
            })
        } else {
            Err(FrameError::NonFinite("tdb_seconds"))
        }
    }

    /// Construct from a Julian Date (TDB).
    pub fn from_jd(jd: f64) -> Result<Self, FrameError> {
        if !jd.is_finite() {
            return Err(FrameError::NonFinite("jd"));
        }
        Self::from_tdb_seconds((jd - J2000_JD) * SECONDS_PER_DAY)
    }

    /// Construct from a Modified Julian Date (TDB).
    pub fn from_mjd(mjd: f64) -> Result<Self, FrameError> {
        if !mjd.is_finite() {
            return Err(FrameError::NonFinite("mjd"));
        }
        Self::from_jd(mjd + MJD_EPOCH_JD)
    }

    /// Construct from days past J2000.
    pub fn from_days_since_j2000(days: f64) -> Result<Self, FrameError> {
        Self::from_tdb_seconds(days * SECONDS_PER_DAY)
    }

    /// TDB seconds past J2000.
    pub const fn tdb_seconds(self) -> f64 {
        self.tdb_seconds
    }

    /// Julian Date (TDB).
    pub fn jd(self) -> f64 {
        J2000_JD + self.tdb_seconds / SECONDS_PER_DAY
    }

    /// Modified Julian Date (TDB).
    pub fn mjd(self) -> f64 {
        self.jd() - MJD_EPOCH_JD
    }

    /// Days past J2000.
    pub fn days_since_j2000(self) -> f64 {
        self.tdb_seconds / SECONDS_PER_DAY
    }

    /// Offset from this epoch to `other`, in seconds.
    pub fn seconds_until(self, other: Self) -> Seconds {
        Seconds(other.tdb_seconds - self.tdb_seconds)
    }

    /// Add a duration, rejecting a non-finite result.
    pub fn checked_add(self, delta: Seconds) -> Result<Self, FrameError> {
        Self::from_tdb_seconds(self.tdb_seconds + delta.0)
    }

    /// Subtract a duration, rejecting a non-finite result.
    pub fn checked_sub(self, delta: Seconds) -> Result<Self, FrameError> {
        Self::from_tdb_seconds(self.tdb_seconds - delta.0)
    }

    /// Project the epoch onto the Gregorian calendar (TDB, no leap seconds).
    ///
    /// Returns `(year, month, day, hour, minute, second)`. The algorithm is the
    /// standard Fliegel–Van Flandern JD conversion; it is exact for the JD
    /// range used by this project and is intended for display/manifest output,
    /// not for UTC arithmetic.
    pub fn to_gregorian(self) -> (i32, u32, u32, u32, u32, f64) {
        let jd = self.jd() + 0.5;
        let z = jd.floor() as i64;
        let fraction = jd - z as f64;

        let a = if z < 2_299_161 {
            z
        } else {
            let alpha = ((z as f64 - 1_867_216.25) / 36_524.25).floor();
            z + 1 + alpha as i64 - (alpha / 4.0).floor() as i64
        };
        let b = a + 1_524;
        let c = ((b as f64 - 122.1) / 365.25).floor();
        let d = (365.25 * c).floor();
        let e = ((b as f64 - d) / 30.6001).floor();

        let day = b - d as i64 - (30.6001 * e).floor() as i64;
        let month = if e < 14.0 { e - 1.0 } else { e - 13.0 };
        let year = if month > 2.0 { c - 4716.0 } else { c - 4715.0 };

        let day_seconds = fraction * SECONDS_PER_DAY;
        let hour = (day_seconds / 3_600.0).floor() as u32;
        let minute = ((day_seconds - hour as f64 * 3_600.0) / 60.0).floor() as u32;
        let second = day_seconds - hour as f64 * 3_600.0 - minute as f64 * 60.0;

        (year as i32, month as u32, day as u32, hour, minute, second)
    }
}

impl fmt::Display for Epoch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (year, month, day, hour, minute, second) = self.to_gregorian();
        write!(
            f,
            "{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:06.3} TDB"
        )
    }
}

/// Quasi-inertial reference frames used by the pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InertialFrame {
    /// Generic Earth-centred inertial frame.
    #[default]
    Eci,
    /// Earth mean equator and equinox of J2000.
    Eme2000,
    /// International Celestial Reference Frame.
    Icrf,
}

impl InertialFrame {
    /// Stable manifest/display name.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Eci => "eci",
            Self::Eme2000 => "eme2000",
            Self::Icrf => "icrf",
        }
    }

    /// Every variant is treated as quasi-inertial by the propagator.
    pub const fn is_quasi_inertial(self) -> bool {
        true
    }

    /// Whether two frames can be compared without a rotation.
    ///
    /// The EME2000/ICRF frame bias is below the project's comparison
    /// tolerances, and `Eci` is a generic alias here, so all variants are
    /// compatible for this pipeline.
    pub const fn is_compatible_with(self, other: Self) -> bool {
        matches!(
            (self, other),
            (Self::Eci, _)
                | (_, Self::Eci)
                | (Self::Eme2000, Self::Icrf)
                | (Self::Icrf, Self::Eme2000)
                | (Self::Eme2000, Self::Eme2000)
                | (Self::Icrf, Self::Icrf)
        )
    }
}

impl fmt::Display for InertialFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Alias retained for callers that used the old name.
pub type Frame = InertialFrame;
/// Alias for the canonical TDB-seconds type.
pub type EpochSeconds = Seconds;

/// Errors from frame/time construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameError {
    /// A value was NaN or infinite.
    NonFinite(&'static str),
}

impl fmt::Display for FrameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite(field) => write!(f, "{field} must be finite"),
        }
    }
}

impl std::error::Error for FrameError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seconds_roundtrip_and_arithmetic() {
        let s = Seconds::from(123.456);
        let v: f64 = s.into();
        assert_eq!(v, 123.456);
        assert_eq!((s + Seconds(1.0)).value(), 124.456);
        assert_eq!((s - Seconds(1.0)).value(), 122.456);
        assert_eq!((s * 2.0).value(), 246.912);
        assert_eq!((-s).value(), -123.456);
        assert_eq!(Seconds::ZERO.abs().value(), 0.0);
        assert!(Seconds::new(f64::NAN).is_err());
    }

    #[test]
    fn epoch_roundtrips_through_jd_and_mjd() {
        let epoch = Epoch::from_jd(2_451_545.0).unwrap();
        assert_eq!(epoch.tdb_seconds(), 0.0);
        assert_eq!(epoch.jd(), J2000_JD);
        assert_eq!(epoch.mjd(), J2000_JD - MJD_EPOCH_JD);

        let later = Epoch::from_tdb_seconds(86_400.0).unwrap();
        assert!((later.jd() - (J2000_JD + 1.0)).abs() < 1e-12);
        assert!((later.mjd() - (J2000_JD + 1.0 - MJD_EPOCH_JD)).abs() < 1e-12);
        assert_eq!(epoch.seconds_until(later).value(), 86_400.0);
    }

    #[test]
    fn epoch_rejects_non_finite_inputs() {
        assert!(Epoch::from_tdb_seconds(f64::INFINITY).is_err());
        assert!(Epoch::from_jd(f64::NAN).is_err());
        assert!(Epoch::from_mjd(f64::NEG_INFINITY).is_err());
    }

    #[test]
    fn j2000_projects_to_gregorian_noon() {
        let (year, month, day, hour, minute, second) = Epoch::J2000.to_gregorian();
        assert_eq!((year, month, day), (2000, 1, 1));
        assert_eq!((hour, minute), (12, 0));
        assert!((second - 0.0).abs() < 1e-6);
    }

    #[test]
    fn gregorian_handles_next_day() {
        let epoch = Epoch::from_tdb_seconds(86_400.0).unwrap();
        let (year, month, day, hour, _, _) = epoch.to_gregorian();
        assert_eq!((year, month, day), (2000, 1, 2));
        assert_eq!(hour, 12);
    }

    #[test]
    fn frame_names_and_compatibility() {
        assert_eq!(InertialFrame::Eci.name(), "eci");
        assert_eq!(InertialFrame::Eme2000.name(), "eme2000");
        assert_eq!(InertialFrame::Icrf.name(), "icrf");
        assert!(InertialFrame::Eme2000.is_compatible_with(InertialFrame::Icrf));
        assert!(InertialFrame::Eci.is_compatible_with(InertialFrame::Eme2000));
        assert!(InertialFrame::default().is_quasi_inertial());
        assert_eq!(Frame::Icrf.to_string(), "icrf");
    }
}
