//! Rust core for the Earth-Moon restricted dynamics.
//!
//! Provides a fixed-step velocity-Verlet propagator ([`propagate`]) and an
//! adaptive Dormand-Prince (RK45) propagator ([`propagate_adaptive`]) built on
//! the [`dormand_prince_step`] embedded 4(5) error estimate.  Both integrators
//! share one acceleration function and one burn model, so they cannot drift
//! apart.
//!
//! # Ephemeris
//!
//! The default [`AnalyticEphemeris`] is a circular, coplanar Earth-Moon
//! barycentric model that matches the CUDA kernel exactly.  That model is a
//! deliberate approximation: a real lunar trajectory does not follow a circle,
//! so it cannot reproduce a JPL Horizons or SPICE arc to better than the
//! circular-model error.
//!
//! Real positions are supplied through the [`Ephemeris`] trait.
//! [`TabulatedEphemeris`] linearly interpolates a time series — for example the
//! Moon/Sun positions written by the Python SPICE or Horizons ingestion step —
//! so the Rust propagator can consume a real ephemeris without linking a SPICE
//! toolkit.  SPICE stays Python-side (`scripts/fetch_spice.py`); Rust never
//! reads a kernel directly.
//!
//! # Validation
//!
//! [`Config::validate`] mirrors the CUDA kernel's `valid_config`, including the
//! [`MAX_BURNS`] cap.  Propagators run a debug assertion on it, so an invalid
//! configuration fails loudly in tests instead of silently producing NaNs.

use std::f64::consts::PI;
use std::fmt;

/// Allow `propagate` to accept any common integer step count.
pub trait IntoSteps {
    fn into_steps(self) -> usize;
}
impl IntoSteps for usize {
    fn into_steps(self) -> usize {
        self
    }
}
impl IntoSteps for u32 {
    fn into_steps(self) -> usize {
        self as usize
    }
}
impl IntoSteps for u64 {
    fn into_steps(self) -> usize {
        self as usize
    }
}
impl IntoSteps for i32 {
    fn into_steps(self) -> usize {
        self.max(0) as usize
    }
}
impl IntoSteps for i64 {
    fn into_steps(self) -> usize {
        self.max(0) as usize
    }
}
impl IntoSteps for isize {
    fn into_steps(self) -> usize {
        self.max(0) as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }

    pub fn scale(self, s: f64) -> Self {
        Self {
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
        }
    }

    pub fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn norm(self) -> f64 {
        self.dot(self).sqrt()
    }

    pub fn length(self) -> f64 {
        self.norm()
    }

    pub fn magnitude(self) -> f64 {
        self.norm()
    }

    /// Linear interpolation: `self` at `w = 0`, `other` at `w = 1`.
    pub fn lerp(self, other: Self, w: f64) -> Self {
        Self {
            x: self.x + (other.x - self.x) * w,
            y: self.y + (other.y - self.y) * w,
            z: self.z + (other.z - self.z) * w,
        }
    }

    /// Whether all three components are finite.
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }
}

impl std::ops::Add for Vec3 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        self.add(rhs)
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        self.sub(rhs)
    }
}

impl std::ops::Mul<f64> for Vec3 {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        self.scale(rhs)
    }
}

impl std::ops::Mul<Vec3> for f64 {
    type Output = Vec3;
    fn mul(self, rhs: Vec3) -> Vec3 {
        rhs.scale(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct State {
    pub r: Vec3,
    pub v: Vec3,
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Burn {
    pub t: f64,
    pub dv: Vec3,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub dt: f64,
    pub epoch: f64,
    pub mu_earth: f64,
    pub mu_moon: f64,
    /// Solar gravitational parameter. Zero by default, which keeps the model
    /// identical to the CUDA kernel. Set it (and supply a Sun position through
    /// an [`Ephemeris`]) to enable the third-body term.
    pub mu_sun: f64,
    pub earth_radius: f64,
    pub j2: f64,
    pub moon_distance: f64,
    pub moon_period: f64,
    pub moon_phase: f64,
    pub softening: f64,
    pub perturbation: i32,
    pub spin_axis: Vec3,
    /// Impulsive burns. At most [`MAX_BURNS`] entries are accepted so a
    /// configuration valid here is also valid on the GPU.
    pub burns: Vec<Burn>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            dt: 60.0,
            epoch: 0.0,
            mu_earth: K_MU_EARTH,
            mu_moon: K_MU_MOON,
            mu_sun: 0.0,
            earth_radius: K_EARTH_RADIUS,
            j2: K_EARTH_J2,
            moon_distance: K_EARTH_MOON_DISTANCE,
            moon_period: K_MOON_PERIOD,
            moon_phase: 0.0,
            softening: 0.0,
            perturbation: 0,
            spin_axis: Vec3::new(0.0, 0.0, 1.0),
            burns: Vec::new(),
        }
    }
}

/// Maximum number of impulsive burns per propagation.
///
/// The CUDA kernel stores burns in a fixed `burns[8]` array and rejects
/// configurations with more, so the Rust core enforces the same cap.
pub const MAX_BURNS: usize = 8;

/// A configuration that would produce NaNs, divide by zero, or diverge from
/// the CUDA kernel's accepted inputs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConfigError {
    /// A scalar field was NaN or infinite.
    NotFinite(&'static str),
    /// A field was finite but outside its accepted range.
    OutOfRange { field: &'static str, value: f64 },
    /// More burns than [`MAX_BURNS`].
    TooManyBurns { count: usize, max: usize },
    /// `perturbation` is not a supported mode.
    UnsupportedPerturbation(i32),
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFinite(field) => write!(f, "{field} must be finite"),
            Self::OutOfRange { field, value } => {
                write!(f, "{field} is out of range: {value}")
            }
            Self::TooManyBurns { count, max } => {
                write!(f, "{count} burns supplied, at most {max} are supported")
            }
            Self::UnsupportedPerturbation(mode) => {
                write!(f, "unsupported perturbation mode {mode} (expected 0 or 1)")
            }
        }
    }
}

impl std::error::Error for ConfigError {}

impl Config {
    /// Check every field against the constraints enforced by the CUDA kernel.
    ///
    /// This is intentionally identical in spirit to `valid_config` in
    /// `src/kernels/multibody_solver.cu`, plus the Sun parameter which CUDA
    /// does not model yet.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if !self.dt.is_finite() || self.dt <= 0.0 {
            return Err(ConfigError::OutOfRange {
                field: "dt",
                value: self.dt,
            });
        }
        if !self.epoch.is_finite() {
            return Err(ConfigError::NotFinite("epoch"));
        }
        for (name, value) in [
            ("mu_earth", self.mu_earth),
            ("mu_moon", self.mu_moon),
            ("mu_sun", self.mu_sun),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(ConfigError::OutOfRange { field: name, value });
            }
        }
        if !self.earth_radius.is_finite() || self.earth_radius < 0.0 {
            return Err(ConfigError::OutOfRange {
                field: "earth_radius",
                value: self.earth_radius,
            });
        }
        if !self.j2.is_finite() {
            return Err(ConfigError::NotFinite("j2"));
        }
        if !self.moon_distance.is_finite()
            || self.moon_distance < 0.0
            || (self.moon_distance > 0.0 && self.moon_distance < self.earth_radius)
        {
            return Err(ConfigError::OutOfRange {
                field: "moon_distance",
                value: self.moon_distance,
            });
        }
        if !self.moon_period.is_finite() || self.moon_period < 0.0 {
            return Err(ConfigError::OutOfRange {
                field: "moon_period",
                value: self.moon_period,
            });
        }
        if !self.moon_phase.is_finite() {
            return Err(ConfigError::NotFinite("moon_phase"));
        }
        if !self.softening.is_finite() || self.softening < 0.0 {
            return Err(ConfigError::OutOfRange {
                field: "softening",
                value: self.softening,
            });
        }
        if self.perturbation != 0 && self.perturbation != 1 {
            return Err(ConfigError::UnsupportedPerturbation(self.perturbation));
        }
        for (name, value) in [
            ("spin_axis.x", self.spin_axis.x),
            ("spin_axis.y", self.spin_axis.y),
            ("spin_axis.z", self.spin_axis.z),
        ] {
            if !value.is_finite() {
                return Err(ConfigError::NotFinite(name));
            }
        }
        if self.burns.len() > MAX_BURNS {
            return Err(ConfigError::TooManyBurns {
                count: self.burns.len(),
                max: MAX_BURNS,
            });
        }
        for burn in &self.burns {
            if !burn.t.is_finite() {
                return Err(ConfigError::NotFinite("burn.t"));
            }
            if !burn.dv.x.is_finite() || !burn.dv.y.is_finite() || !burn.dv.z.is_finite() {
                return Err(ConfigError::NotFinite("burn.dv"));
            }
        }
        Ok(())
    }
}

// Physical constants (SI), matching CUDA solver.
pub const K_PI: f64 = PI;
pub const K_MU_EARTH: f64 = 3.986004418e14;
pub const K_MU_MOON: f64 = 4.9048695e12;
/// Solar gravitational parameter, SI. Not used unless `Config::mu_sun` is set.
pub const K_MU_SUN: f64 = 1.32712440018e20;
pub const K_EARTH_RADIUS: f64 = 6.378137e6;
pub const K_EARTH_J2: f64 = 1.08262668e-3;
pub const K_EARTH_MOON_DISTANCE: f64 = 3.844e8;
pub const K_MOON_PERIOD: f64 = 2.3605915e6;

pub fn default_config() -> Config {
    Config::default()
}

/// Circular barycentric ephemeris matching the CUDA solver.
pub fn earth_moon_positions(t: f64, config: &Config) -> (Vec3, Vec3) {
    if config.moon_distance <= 0.0 {
        return (Vec3::ZERO, Vec3::ZERO);
    }

    let mut theta = config.moon_phase;
    let total_mu = config.mu_earth + config.mu_moon;
    if config.moon_distance > 0.0 && total_mu > 0.0 {
        theta += (total_mu / (config.moon_distance * config.moon_distance * config.moon_distance))
            .sqrt()
            * t;
    }

    let moon_fraction = if total_mu > 0.0 {
        config.mu_moon / total_mu
    } else {
        0.0
    };

    let c = theta.cos();
    let s = theta.sin();

    let earth = Vec3::new(
        -config.moon_distance * moon_fraction * c,
        -config.moon_distance * moon_fraction * s,
        0.0,
    );
    let moon = Vec3::new(
        config.moon_distance * (1.0 - moon_fraction) * c,
        config.moon_distance * (1.0 - moon_fraction) * s,
        0.0,
    );
    (earth, moon)
}

// ---------------------------------------------------------------------------
// Ephemeris abstraction.
// ---------------------------------------------------------------------------

/// Positions of the perturbing bodies at one instant, in the same inertial
/// frame (and the same time base) as the propagated state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BodyPositions {
    /// Barycentric Earth position.
    pub earth: Vec3,
    /// Moon position, when a Moon term should be included.
    pub moon: Option<Vec3>,
    /// Sun position, when a third-body term should be included.
    pub sun: Option<Vec3>,
}

/// Source of perturbing-body positions for the propagator.
///
/// The default [`AnalyticEphemeris`] reproduces the CUDA kernel's circular
/// model.  Supply a [`TabulatedEphemeris`] to propagate against real positions
/// (for example a Horizons/SPICE sampled arc) without linking SPICE in Rust.
///
/// Implementations must be deterministic and must not depend on `t` in a way
/// that is discontinuous across an integration step; the adaptive controller
/// assumes a smooth field.
pub trait Ephemeris {
    /// Body positions at absolute time `t` (the same base as `Config::epoch`).
    fn positions(&self, t: f64, config: &Config) -> BodyPositions;

    /// Time span `[start, end]` covered by this ephemeris, if bounded.
    ///
    /// Unbounded sources (such as the analytic model) return `None`.  Bounded
    /// sources are clamped outside their span; use
    /// [`TabulatedEphemeris::covers`] before propagating to reject an arc that
    /// falls outside the data.
    fn span(&self) -> Option<(f64, f64)> {
        None
    }
}

/// The circular, coplanar Earth-Moon model baked into the CUDA kernel.
///
/// The Sun is not modelled here; provide a [`TabulatedEphemeris`] with Sun
/// samples and set `Config::mu_sun` to enable the third-body term.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AnalyticEphemeris;

impl Ephemeris for AnalyticEphemeris {
    fn positions(&self, t: f64, config: &Config) -> BodyPositions {
        let (earth, moon) = earth_moon_positions(t, config);
        BodyPositions {
            earth,
            // `moon_distance == 0` is the documented way to disable the Moon,
            // matching the CUDA `acceleration` guard.
            moon: if config.moon_distance > 0.0 {
                Some(moon)
            } else {
                None
            },
            sun: None,
        }
    }
}

/// One row of a tabulated ephemeris, in the propagator's inertial frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EphemerisSample {
    /// Absolute time, same base as `Config::epoch`.
    pub t: f64,
    /// Barycentric Earth position, metres.
    pub earth: Vec3,
    /// Barycentric Moon position, metres.
    pub moon: Vec3,
    /// Barycentric Sun position, metres. Must be `Some` for every sample or
    /// for none of them.
    pub sun: Option<Vec3>,
}

impl EphemerisSample {
    fn positions(self) -> BodyPositions {
        BodyPositions {
            earth: self.earth,
            moon: Some(self.moon),
            sun: self.sun,
        }
    }
}

/// Errors from constructing or using a [`TabulatedEphemeris`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EphemerisError {
    /// No samples were supplied.
    Empty,
    /// A time or position was NaN or infinite.
    NonFinite { index: usize },
    /// Sample times are not strictly increasing.
    NotSorted { index: usize },
    /// Some samples carry a Sun position and others do not.
    InconsistentSun { index: usize },
    /// The requested time lies outside the tabulated span.
    OutOfRange { t: f64, start: f64, end: f64 },
    /// A mass ratio was zero or not finite, so the Earth track cannot be
    /// derived from the Moon track.
    NonPositiveMass,
}

impl fmt::Display for EphemerisError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "ephemeris table is empty"),
            Self::NonFinite { index } => write!(f, "ephemeris sample {index} is not finite"),
            Self::NotSorted { index } => {
                write!(f, "ephemeris sample {index} does not increase in time")
            }
            Self::InconsistentSun { index } => {
                write!(
                    f,
                    "ephemeris sample {index} disagrees about Sun availability"
                )
            }
            Self::OutOfRange { t, start, end } => {
                write!(f, "time {t} outside ephemeris span [{start}, {end}]")
            }
            Self::NonPositiveMass => write!(
                f,
                "cannot derive an Earth track without positive, finite Earth and Moon masses"
            ),
        }
    }
}

impl std::error::Error for EphemerisError {}

/// Linearly interpolated ephemeris built from a time series.
///
/// Construct it with [`TabulatedEphemeris::new`], which validates ordering and
/// finiteness.  Lookups outside the tabulated span clamp to the nearest sample;
/// call [`TabulatedEphemeris::covers`] first to reject such an arc.
#[derive(Debug, Clone, PartialEq)]
pub struct TabulatedEphemeris {
    samples: Vec<EphemerisSample>,
    has_sun: bool,
}

impl TabulatedEphemeris {
    /// Build a table from samples sorted by strictly increasing time.
    pub fn new(samples: Vec<EphemerisSample>) -> Result<Self, EphemerisError> {
        if samples.is_empty() {
            return Err(EphemerisError::Empty);
        }
        let has_sun = samples[0].sun.is_some();
        for (index, sample) in samples.iter().enumerate() {
            if !sample.t.is_finite()
                || !sample.earth.is_finite()
                || !sample.moon.is_finite()
                || sample.sun.is_some_and(|s| !s.is_finite())
            {
                return Err(EphemerisError::NonFinite { index });
            }
            if index > 0 && sample.t <= samples[index - 1].t {
                return Err(EphemerisError::NotSorted { index });
            }
            if sample.sun.is_some() != has_sun {
                return Err(EphemerisError::InconsistentSun { index });
            }
        }
        Ok(Self { samples, has_sun })
    }

    /// Build a real-geometry table from a barycentric Moon track.
    ///
    /// The Earth track is derived from the two-body barycentre condition
    /// `mu_earth * r_earth + mu_moon * r_moon = 0`, so a reference arc written
    /// in the Earth-Moon barycentric frame (the canonical `reference.csv` from
    /// `scripts/convert_reference.py`) is enough to drive the propagator with
    /// the measured geometry instead of the circular model.
    ///
    /// `mu_earth` and `mu_moon` must be the *physical* gravitational
    /// parameters, not the values used to propagate a body that should not
    /// attract itself.  The residual pipeline zeroes `mu_moon` for the Moon;
    /// build the table before doing so.
    ///
    /// The table carries no Sun positions.  Leave `Config::mu_sun` at zero (or
    /// attach a Sun track separately) when using it.
    pub fn from_moon_track<I>(track: I, mu_earth: f64, mu_moon: f64) -> Result<Self, EphemerisError>
    where
        I: IntoIterator<Item = (f64, Vec3)>,
    {
        if !(mu_earth.is_finite() && mu_earth > 0.0 && mu_moon.is_finite() && mu_moon > 0.0) {
            return Err(EphemerisError::NonPositiveMass);
        }
        let earth_per_moon = -mu_moon / mu_earth;
        let samples = track
            .into_iter()
            .map(|(t, moon)| EphemerisSample {
                t,
                earth: moon.scale(earth_per_moon),
                moon,
                sun: None,
            })
            .collect();
        Self::new(samples)
    }

    /// Number of tabulated samples.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Whether the table has no samples (never true after `new`).
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Whether the table carries Sun positions.
    pub fn has_sun(&self) -> bool {
        self.has_sun
    }

    /// The tabulated samples.
    pub fn samples(&self) -> &[EphemerisSample] {
        &self.samples
    }

    /// First and last tabulated time.
    pub fn time_span(&self) -> (f64, f64) {
        (self.samples[0].t, self.samples[self.samples.len() - 1].t)
    }

    /// Validate that every time in `[start, end]` is covered by the table.
    pub fn check_span(&self, start: f64, end: f64) -> Result<(), EphemerisError> {
        let (lo, hi) = if start <= end {
            (start, end)
        } else {
            (end, start)
        };
        let (first, last) = self.time_span();
        if lo < first {
            return Err(EphemerisError::OutOfRange {
                t: lo,
                start: first,
                end: last,
            });
        }
        if hi > last {
            return Err(EphemerisError::OutOfRange {
                t: hi,
                start: first,
                end: last,
            });
        }
        Ok(())
    }

    /// Whether every time in `[start, end]` is covered by the table.
    pub fn covers(&self, start: f64, end: f64) -> bool {
        self.check_span(start, end).is_ok()
    }

    /// Position lookup with linear interpolation between samples.
    fn lookup(&self, t: f64) -> BodyPositions {
        let last = self.samples.len() - 1;
        if t <= self.samples[0].t {
            return self.samples[0].positions();
        }
        if t >= self.samples[last].t {
            return self.samples[last].positions();
        }
        // `idx` is the first sample strictly after `t`, so `idx >= 1`.
        let idx = self.samples.partition_point(|s| s.t <= t);
        let a = self.samples[idx - 1];
        let b = self.samples[idx];
        let span = b.t - a.t;
        let w = if span > 0.0 { (t - a.t) / span } else { 0.0 };
        BodyPositions {
            earth: a.earth.lerp(b.earth, w),
            moon: Some(a.moon.lerp(b.moon, w)),
            sun: a.sun.zip(b.sun).map(|(sa, sb)| sa.lerp(sb, w)),
        }
    }
}

impl Ephemeris for TabulatedEphemeris {
    fn positions(&self, t: f64, _config: &Config) -> BodyPositions {
        self.lookup(t)
    }

    fn span(&self) -> Option<(f64, f64)> {
        Some(TabulatedEphemeris::time_span(self))
    }
}

/// Default analytic ephemeris instance used by the infallible propagators.
static ANALYTIC: AnalyticEphemeris = AnalyticEphemeris;

/// J2 acceleration relative to Earth's centre.
pub fn earth_j2_acceleration(d: Vec3, config: &Config) -> Vec3 {
    let r2 = d.dot(d);
    if r2 <= 0.0 {
        return Vec3::ZERO;
    }
    let r = r2.sqrt();
    if r <= config.earth_radius {
        return Vec3::ZERO;
    }
    let s = config.spin_axis;
    let sdotr = d.x * s.x + d.y * s.y + d.z * s.z;
    let cos2 = (sdotr * sdotr) / r2;
    let k = -1.5 * config.j2 * config.mu_earth * config.earth_radius * config.earth_radius
        / (r2 * r2 * r);
    let factor = 1.0 - 5.0 * cos2;
    Vec3::new(
        k * (factor * d.x + 2.0 * sdotr * s.x),
        k * (factor * d.y + 2.0 * sdotr * s.y),
        k * (factor * d.z + 2.0 * sdotr * s.z),
    )
}

/// Acceleration with Earth/Moon point masses and optional J2, using the
/// analytic ephemeris.
pub fn acceleration(r: Vec3, config: &Config, t: f64) -> Vec3 {
    acceleration_with(r, config, t, &ANALYTIC)
}

/// Acceleration with Earth/Moon/Sun point masses and optional J2, using a
/// caller-supplied ephemeris.
///
/// A body contributes only when the ephemeris supplies its position *and* the
/// matching gravitational parameter is positive.  With the default
/// [`AnalyticEphemeris`] and `mu_sun = 0`, this is identical to the CUDA
/// kernel.
pub fn acceleration_with(r: Vec3, config: &Config, t: f64, ephemeris: &dyn Ephemeris) -> Vec3 {
    let bodies = ephemeris.positions(t, config);
    let eps2 = config.softening * config.softening;

    let d_earth = r.sub(bodies.earth);
    let r_earth2 = d_earth.dot(d_earth) + eps2;

    let mut a = Vec3::ZERO;
    if config.mu_earth > 0.0 && r_earth2 > 0.0 {
        a = a.add(d_earth.scale(-config.mu_earth / (r_earth2 * r_earth2.sqrt())));
    }
    if let Some(moon) = bodies.moon {
        let d_moon = r.sub(moon);
        let r_moon2 = d_moon.dot(d_moon) + eps2;
        if config.mu_moon > 0.0 && r_moon2 > 0.0 {
            a = a.add(d_moon.scale(-config.mu_moon / (r_moon2 * r_moon2.sqrt())));
        }
    }
    if let Some(sun) = bodies.sun {
        let d_sun = r.sub(sun);
        let r_sun2 = d_sun.dot(d_sun) + eps2;
        if config.mu_sun > 0.0 && r_sun2 > 0.0 {
            a = a.add(d_sun.scale(-config.mu_sun / (r_sun2 * r_sun2.sqrt())));
        }
    }
    if config.perturbation == 1 {
        a = a.add(earth_j2_acceleration(d_earth, config));
    }
    a
}

/// Shared physics for both integrators: one acceleration field and one burn
/// model.  Keeping them here is what guarantees `propagate` and
/// `propagate_adaptive` cannot diverge.
struct System<'a> {
    config: &'a Config,
    ephemeris: &'a dyn Ephemeris,
}

impl<'a> System<'a> {
    fn new(config: &'a Config, ephemeris: &'a dyn Ephemeris) -> Self {
        Self { config, ephemeris }
    }

    fn acceleration(&self, r: Vec3, t: f64) -> Vec3 {
        acceleration_with(r, self.config, t, self.ephemeris)
    }

    /// Burn Δv for the fixed-step window that matches the CUDA kernel.
    fn burn_dv_window(&self, t: f64, dt: f64) -> Vec3 {
        self.config
            .burns
            .iter()
            .filter(|burn| (t - burn.t).abs() < 0.5 * dt)
            .fold(Vec3::ZERO, |acc, burn| acc.add(burn.dv))
    }

    /// Burn Δv for burns at an exact epoch, within `tol`.
    fn burn_dv_at(&self, t: f64, tol: f64) -> Vec3 {
        self.config
            .burns
            .iter()
            .filter(|burn| (burn.t - t).abs() <= tol)
            .fold(Vec3::ZERO, |acc, burn| acc.add(burn.dv))
    }
}

/// Fixed-step velocity-Verlet (kick-drift-kick) propagation, matching the CUDA
/// solver. Returns the trajectory including the initial state.
///
/// Uses the analytic ephemeris.  See [`propagate_with`] for a real ephemeris.
///
/// # Panics
///
/// Debug builds assert that `config` passes [`Config::validate`] and that `dt`
/// is finite and positive.
pub fn propagate(state: State, steps: impl IntoSteps, dt: f64, config: &Config) -> Vec<State> {
    propagate_with(state, steps, dt, config, &ANALYTIC)
}

/// Fixed-step velocity-Verlet propagation using a caller-supplied ephemeris.
///
/// A bounded ephemeris such as [`TabulatedEphemeris`] clamps outside its span;
/// call [`TabulatedEphemeris::covers`] first if the arc must be fully covered.
///
/// # Panics
///
/// Debug builds assert that `config` passes [`Config::validate`] and that `dt`
/// is finite and positive.
pub fn propagate_with(
    state: State,
    steps: impl IntoSteps,
    dt: f64,
    config: &Config,
    ephemeris: &dyn Ephemeris,
) -> Vec<State> {
    let steps = steps.into_steps();
    debug_assert!(
        config.validate().is_ok(),
        "invalid dynamics config: {:?}",
        config.validate()
    );
    debug_assert!(dt.is_finite() && dt > 0.0, "dt must be finite and positive");
    if steps == 0 || !dt.is_finite() || dt <= 0.0 {
        return vec![state];
    }

    let system = System::new(config, ephemeris);
    let mut out = Vec::with_capacity(steps + 1);
    let mut s = state;
    let mut t = config.epoch;
    out.push(s);
    for _ in 0..steps {
        // Apply impulsive burns firing at this step's start time.
        s.v = s.v.add(system.burn_dv_window(t, dt));
        let a0 = system.acceleration(s.r, t);
        s.v = s.v.add(a0.scale(0.5 * dt));
        s.r = s.r.add(s.v.scale(dt));
        let a1 = system.acceleration(s.r, t + dt);
        s.v = s.v.add(a1.scale(0.5 * dt));
        t += dt;
        out.push(s);
    }
    out
}

// ---------------------------------------------------------------------------
// Adaptive Dormand-Prince (RK45 / DOPRI5) integrator.
// ---------------------------------------------------------------------------

// Butcher tableau for the Dormand-Prince 5(4) pair. The 5th-order weights
// coincide with the final stage row (the method is FSAL), so the seventh stage
// state is reused as the step output.

// Stage nodes c_i.
const DP_C2: f64 = 1.0 / 5.0;
const DP_C3: f64 = 3.0 / 10.0;
const DP_C4: f64 = 4.0 / 5.0;
const DP_C5: f64 = 8.0 / 9.0;
const DP_C6: f64 = 1.0;

// Stage coupling coefficients a_ij (a_72, a_77 are zero).
const DP_A21: f64 = 1.0 / 5.0;

const DP_A31: f64 = 3.0 / 40.0;
const DP_A32: f64 = 9.0 / 40.0;

const DP_A41: f64 = 44.0 / 45.0;
const DP_A42: f64 = -56.0 / 15.0;
const DP_A43: f64 = 32.0 / 9.0;

const DP_A51: f64 = 19372.0 / 6561.0;
const DP_A52: f64 = -25360.0 / 2187.0;
const DP_A53: f64 = 64448.0 / 6561.0;
const DP_A54: f64 = -212.0 / 729.0;

const DP_A61: f64 = 9017.0 / 3168.0;
const DP_A62: f64 = -355.0 / 33.0;
const DP_A63: f64 = 46732.0 / 5247.0;
const DP_A64: f64 = 49.0 / 176.0;
const DP_A65: f64 = -5103.0 / 18656.0;

// Final row, which doubles as the 5th-order solution weights b_i.
const DP_A71: f64 = 35.0 / 384.0;
const DP_A73: f64 = 500.0 / 1113.0;
const DP_A74: f64 = 125.0 / 192.0;
const DP_A75: f64 = -2187.0 / 6784.0;
const DP_A76: f64 = 11.0 / 84.0;

// 4th-order solution weights b*_i (b*_2 is zero).
const DP_B4_1: f64 = 5179.0 / 57600.0;
const DP_B4_3: f64 = 7571.0 / 16695.0;
const DP_B4_4: f64 = 393.0 / 640.0;
const DP_B4_5: f64 = -92097.0 / 339200.0;
const DP_B4_6: f64 = 187.0 / 2100.0;
const DP_B4_7: f64 = 1.0 / 40.0;

/// Normalised (dimensionless) relative error estimate for one DP step.
///
/// Position and velocity error contributions are each scaled by the larger of
/// the old/new magnitude and combined in an L2 sense. This makes the value
/// directly comparable to a relative tolerance regardless of the units of the
/// state.
fn relative_error(old: &State, new: &State, dr: Vec3, dv: Vec3) -> f64 {
    let scale_r = old.r.norm().max(new.r.norm()).max(f64::MIN_POSITIVE);
    let scale_v = old.v.norm().max(new.v.norm()).max(f64::MIN_POSITIVE);
    let er = dr.norm() / scale_r;
    let ev = dv.norm() / scale_v;
    ((er * er + ev * ev) / 2.0).sqrt()
}

/// Advise a next step size from a normalised error and target tolerance.
///
/// `err` is the dimensionless estimate returned by [`dormand_prince_step`] and
/// `rtol` is the tolerance stored on the controller. The classic elementary
/// controller `h_new = h * safety * (rtol / err)^(1/5)` is used, clamped to the
/// configured growth/shrink limits.
fn recommend_dt(
    dt: f64,
    err: f64,
    rtol: f64,
    safety: f64,
    min_factor: f64,
    max_factor: f64,
) -> f64 {
    if err.is_finite() && err > 0.0 {
        let factor = safety * (rtol / err).powf(0.2);
        dt * factor.clamp(min_factor, max_factor)
    } else {
        dt * max_factor
    }
}

/// Perform one adaptive Dormand-Prince 5(4) step of size `dt`.
///
/// `accel` returns the acceleration at a position and time; the state
/// derivative is `(r', v') = (v, a(r, t))`. Returns the 5th-order solution and
/// a dimensionless relative-error estimate suitable for step-size control.
pub fn dormand_prince_step<F>(state: State, accel: &F, t: f64, dt: f64) -> (State, f64)
where
    F: Fn(Vec3, f64) -> Vec3,
{
    // Stage 1 (FSAL: k1 is the derivative at the step start).
    let k1r = state.v;
    let k1v = accel(state.r, t);

    // Stage 2.
    let r2 = state.r.add(k1r.scale(dt * DP_A21));
    let v2 = state.v.add(k1v.scale(dt * DP_A21));
    let k2r = v2;
    let k2v = accel(r2, t + DP_C2 * dt);

    // Stage 3.
    let r3 = state
        .r
        .add(k1r.scale(dt * DP_A31))
        .add(k2r.scale(dt * DP_A32));
    let v3 = state
        .v
        .add(k1v.scale(dt * DP_A31))
        .add(k2v.scale(dt * DP_A32));
    let k3r = v3;
    let k3v = accel(r3, t + DP_C3 * dt);

    // Stage 4.
    let r4 = state
        .r
        .add(k1r.scale(dt * DP_A41))
        .add(k2r.scale(dt * DP_A42))
        .add(k3r.scale(dt * DP_A43));
    let v4 = state
        .v
        .add(k1v.scale(dt * DP_A41))
        .add(k2v.scale(dt * DP_A42))
        .add(k3v.scale(dt * DP_A43));
    let k4r = v4;
    let k4v = accel(r4, t + DP_C4 * dt);

    // Stage 5.
    let r5 = state
        .r
        .add(k1r.scale(dt * DP_A51))
        .add(k2r.scale(dt * DP_A52))
        .add(k3r.scale(dt * DP_A53))
        .add(k4r.scale(dt * DP_A54));
    let v5 = state
        .v
        .add(k1v.scale(dt * DP_A51))
        .add(k2v.scale(dt * DP_A52))
        .add(k3v.scale(dt * DP_A53))
        .add(k4v.scale(dt * DP_A54));
    let k5r = v5;
    let k5v = accel(r5, t + DP_C5 * dt);

    // Stage 6.
    let r6 = state
        .r
        .add(k1r.scale(dt * DP_A61))
        .add(k2r.scale(dt * DP_A62))
        .add(k3r.scale(dt * DP_A63))
        .add(k4r.scale(dt * DP_A64))
        .add(k5r.scale(dt * DP_A65));
    let v6 = state
        .v
        .add(k1v.scale(dt * DP_A61))
        .add(k2v.scale(dt * DP_A62))
        .add(k3v.scale(dt * DP_A63))
        .add(k4v.scale(dt * DP_A64))
        .add(k5v.scale(dt * DP_A65));
    let k6r = v6;
    let k6v = accel(r6, t + DP_C6 * dt);

    // 5th-order solution (uses the final tableau row as its weights).
    let new_r = state
        .r
        .add(k1r.scale(dt * DP_A71))
        .add(k3r.scale(dt * DP_A73))
        .add(k4r.scale(dt * DP_A74))
        .add(k5r.scale(dt * DP_A75))
        .add(k6r.scale(dt * DP_A76));
    let new_v = state
        .v
        .add(k1v.scale(dt * DP_A71))
        .add(k3v.scale(dt * DP_A73))
        .add(k4v.scale(dt * DP_A74))
        .add(k5v.scale(dt * DP_A75))
        .add(k6v.scale(dt * DP_A76));

    // Stage 7 evaluated at the 5th-order solution (FSAL reuse).
    let k7r = new_v;
    let k7v = accel(new_r, t + dt);

    // 4th-order solution for the embedded error estimate.
    let r4s = state
        .r
        .add(k1r.scale(dt * DP_B4_1))
        .add(k3r.scale(dt * DP_B4_3))
        .add(k4r.scale(dt * DP_B4_4))
        .add(k5r.scale(dt * DP_B4_5))
        .add(k6r.scale(dt * DP_B4_6))
        .add(k7r.scale(dt * DP_B4_7));
    let v4s = state
        .v
        .add(k1v.scale(dt * DP_B4_1))
        .add(k3v.scale(dt * DP_B4_3))
        .add(k4v.scale(dt * DP_B4_4))
        .add(k5v.scale(dt * DP_B4_5))
        .add(k6v.scale(dt * DP_B4_6))
        .add(k7v.scale(dt * DP_B4_7));

    let new_state = State { r: new_r, v: new_v };
    let err = relative_error(&state, &new_state, new_r.sub(r4s), new_v.sub(v4s));
    (new_state, err)
}

/// Adaptive step-size controller for the Dormand-Prince integrator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdaptiveController {
    /// Target relative tolerance (normalised error threshold).
    pub rtol: f64,
    /// Safety factor applied to the raw step-size recommendation.
    pub safety: f64,
    /// Minimum allowed step-size shrink factor per step.
    pub min_factor: f64,
    /// Maximum allowed step-size growth factor per step.
    pub max_factor: f64,
    /// Absolute lower bound on the step size (seconds).
    pub dt_min: f64,
    /// Absolute upper bound on the step size (seconds).
    pub dt_max: f64,
}

impl Default for AdaptiveController {
    fn default() -> Self {
        Self {
            rtol: 1e-9,
            safety: 0.9,
            min_factor: 0.2,
            max_factor: 5.0,
            dt_min: 1e-6,
            dt_max: 1e6,
        }
    }
}

impl AdaptiveController {
    /// Build a controller for a given relative tolerance with sensible defaults.
    pub fn with_tolerance(rtol: f64) -> Self {
        Self {
            rtol,
            ..Self::default()
        }
    }

    /// Is the error estimate acceptable at this tolerance?
    pub fn accept(&self, err: f64) -> bool {
        err <= self.rtol
    }

    /// Recommend the next step size, clamped to `[dt_min, dt_max]`.
    pub fn next_dt(&self, dt: f64, err: f64) -> f64 {
        recommend_dt(
            dt,
            err,
            self.rtol,
            self.safety,
            self.min_factor,
            self.max_factor,
        )
        .clamp(self.dt_min, self.dt_max)
    }
}

/// Result of an adaptive propagation, including the sampled times and step
/// statistics (useful for diagnostics and tests).
#[derive(Debug, Clone, PartialEq)]
pub struct AdaptiveTrajectory {
    /// Sample times, starting at the configured epoch.
    pub t: Vec<f64>,
    /// States at each sample time (`states.len() == t.len()`).
    pub states: Vec<State>,
    /// Number of steps that satisfied the tolerance.
    pub accepted: usize,
    /// Number of steps that were rejected and retried.
    pub rejected: usize,
}

impl AdaptiveTrajectory {
    /// Final state of the propagation (the initial state if nothing was stepped).
    pub fn last(&self) -> &State {
        self.states.last().expect("trajectory is never empty")
    }
}

/// Propagate `state` from `config.epoch` to `t_end` using adaptive
/// Dormand-Prince steps with the supplied controller.
///
/// Impulsive burns from the configuration are honoured in both integration
/// directions: the step size is clamped so a step never crosses a burn epoch,
/// and the burn Δv is applied exactly at that epoch.
pub fn propagate_adaptive_with_controller(
    state: State,
    t_end: f64,
    dt0: f64,
    controller: &AdaptiveController,
    config: &Config,
) -> AdaptiveTrajectory {
    propagate_adaptive_with_ephemeris(state, t_end, dt0, controller, config, &ANALYTIC)
}

/// Adaptive propagation using a caller-supplied ephemeris.
///
/// Negative `h` is used internally for backward arcs; the Dormand-Prince
/// tableau is direction-agnostic, so the same error controller applies.
///
/// # Panics
///
/// Debug builds assert that `config` passes [`Config::validate`].
pub fn propagate_adaptive_with_ephemeris(
    state: State,
    t_end: f64,
    dt0: f64,
    controller: &AdaptiveController,
    config: &Config,
    ephemeris: &dyn Ephemeris,
) -> AdaptiveTrajectory {
    debug_assert!(
        config.validate().is_ok(),
        "invalid dynamics config: {:?}",
        config.validate()
    );
    let system = System::new(config, ephemeris);
    let accel = |r: Vec3, t: f64| system.acceleration(r, t);

    let mut s = state;
    let mut t = config.epoch;
    let mut dt = if dt0.is_finite() && dt0 > 0.0 {
        dt0.clamp(controller.dt_min, controller.dt_max)
    } else {
        controller.dt_min
    };

    let mut traj = AdaptiveTrajectory {
        t: vec![t],
        states: vec![s],
        accepted: 0,
        rejected: 0,
    };

    let forward = t_end >= t;
    if t_end == t {
        return traj;
    }

    let tol_eps = 1e-12 * t_end.abs().max(1.0);
    let start_tol = 1e-9 * dt.max(1.0);

    // Burns strictly inside the span, ordered in the direction of travel.
    let mut burn_times: Vec<f64> = config
        .burns
        .iter()
        .map(|b| b.t)
        .filter(|bt| {
            if forward {
                *bt > t + start_tol && *bt < t_end - tol_eps
            } else {
                *bt < t - start_tol && *bt > t_end + tol_eps
            }
        })
        .collect();
    if forward {
        burn_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    } else {
        burn_times.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    }
    let mut next_burn = 0usize;

    // Apply any burns scheduled exactly at the start epoch.
    s.v = s.v.add(system.burn_dv_at(t, start_tol));
    if let Some(first) = traj.states.first_mut() {
        *first = s;
    }

    let mut guard = 0usize;
    while (forward && t < t_end && (t_end - t).abs() > tol_eps)
        || (!forward && t > t_end && (t - t_end).abs() > tol_eps)
    {
        guard += 1;
        if guard > 50_000_000 {
            break;
        }

        let remaining = (t_end - t).abs();
        let mut h = dt.min(remaining);

        // Never overshoot the next scheduled burn.
        if let Some(&nb) = burn_times.get(next_burn) {
            let delta = nb - t;
            let distance = if forward { delta } else { -delta };
            if distance > 0.0 && distance < h {
                h = distance;
            }
        }

        if h <= 0.0 || !h.is_finite() {
            break;
        }

        let h_signed = if forward { h } else { -h };
        let (candidate, err) = dormand_prince_step(s, &accel, t, h_signed);

        if !candidate.r.is_finite() || !candidate.v.is_finite() {
            // Numerical breakdown: shrink and retry, or give up.
            if h <= controller.dt_min {
                break;
            }
            dt = (h * 0.5).clamp(controller.dt_min, controller.dt_max);
            traj.rejected += 1;
            continue;
        }

        let hit_min = h <= controller.dt_min;
        if controller.accept(err) || hit_min {
            s = candidate;
            t += h_signed;
            if (t_end - t).abs() <= tol_eps {
                t = t_end;
            }
            traj.accepted += 1;
            dt = controller.next_dt(h, err);
            traj.t.push(t);
            traj.states.push(s);

            // Apply burns whose epoch this accepted step reached, in either
            // direction.
            while let Some(&nb) = burn_times.get(next_burn) {
                let reached = if forward {
                    t >= nb - tol_eps
                } else {
                    t <= nb + tol_eps
                };
                if !reached {
                    break;
                }
                s.v = s.v.add(system.burn_dv_at(nb, tol_eps));
                if let Some(last) = traj.states.last_mut() {
                    *last = s;
                }
                next_burn += 1;
            }
        } else {
            traj.rejected += 1;
            let mut next = controller.next_dt(h, err);
            if !next.is_finite() || next >= h {
                next = h * 0.5;
            }
            dt = next.clamp(controller.dt_min, controller.dt_max);
        }
    }

    traj
}

/// Propagate `state` over `[config.epoch, t_end]` with adaptive Dormand-Prince
/// steps targeting the relative tolerance `rtol`.
pub fn propagate_adaptive_detailed(
    state: State,
    t_end: f64,
    dt0: f64,
    rtol: f64,
    config: &Config,
) -> AdaptiveTrajectory {
    let controller = AdaptiveController::with_tolerance(rtol);
    propagate_adaptive_with_controller(state, t_end, dt0, &controller, config)
}

/// Convenience wrapper matching [`propagate`]: adaptively propagate over the
/// span `[config.epoch, t_end]` and return the sampled states only.
pub fn propagate_adaptive(
    state: State,
    t_end: f64,
    dt0: f64,
    rtol: f64,
    config: &Config,
) -> Vec<State> {
    propagate_adaptive_detailed(state, t_end, dt0, rtol, config).states
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_body_config() -> Config {
        Config {
            moon_distance: 0.0,
            mu_moon: 0.0,
            ..Config::default()
        }
    }

    #[test]
    fn two_body_energy_conservation() {
        let cfg = two_body_config();
        let r0 = 7.0e6;
        let v0 = (cfg.mu_earth / r0).sqrt();
        let state = State {
            r: Vec3::new(r0, 0.0, 0.0),
            v: Vec3::new(0.0, v0, 0.0),
        };
        let steps = 200;
        let dt = 60.0;
        let traj = propagate(state, steps, dt, &cfg);
        let energy = |st: &State| -> f64 { 0.5 * st.v.dot(st.v) - cfg.mu_earth / st.r.norm() };
        let e0 = energy(&state);
        let e1 = energy(traj.last().unwrap());
        let rel = (e1 - e0).abs() / e0.abs();
        assert!(rel < 1e-6, "relative energy drift {rel}");
    }

    #[test]
    fn circular_orbit_closure() {
        let cfg = two_body_config();
        let r0 = 7.0e6;
        let v0 = (cfg.mu_earth / r0).sqrt();
        let state = State {
            r: Vec3::new(r0, 0.0, 0.0),
            v: Vec3::new(0.0, v0, 0.0),
        };
        // One orbital period.
        let period = 2.0 * PI * (r0.powi(3) / cfg.mu_earth).sqrt();
        let dt = 60.0;
        let steps = (period / dt).round() as usize;
        let traj = propagate(state, steps, dt, &cfg);
        let last = traj.last().unwrap();
        let pos_err = last.r.sub(state.r).norm();
        assert!(pos_err < 5e5, "position error {pos_err}");
    }

    #[test]
    fn earth_moon_finite_step() {
        let cfg = Config::default();
        let state = State {
            r: Vec3::new(6.6e6, 0.0, 0.0),
            v: Vec3::new(0.0, 7.5e3, 0.0),
        };
        let traj = propagate(state, 5, 60.0, &cfg);
        for s in &traj {
            assert!(s.r.x.is_finite() && s.r.y.is_finite() && s.r.z.is_finite());
            assert!(s.v.x.is_finite() && s.v.y.is_finite() && s.v.z.is_finite());
        }
    }

    fn circular_two_body(r0: f64) -> (Config, State, f64) {
        let cfg = two_body_config();
        let v0 = (cfg.mu_earth / r0).sqrt();
        let state = State {
            r: Vec3::new(r0, 0.0, 0.0),
            v: Vec3::new(0.0, v0, 0.0),
        };
        let period = 2.0 * PI * (r0.powi(3) / cfg.mu_earth).sqrt();
        (cfg, state, period)
    }

    fn specific_energy(st: &State, mu: f64) -> f64 {
        0.5 * st.v.dot(st.v) - mu / st.r.norm()
    }

    #[test]
    fn dp_step_error_is_fifth_order() {
        let (cfg, state, _) = circular_two_body(7.0e6);
        let accel = |r: Vec3, t: f64| acceleration(r, &cfg, t);
        let h = 600.0;
        let (_, err_full) = dormand_prince_step(state, &accel, 0.0, h);
        let (_, err_half) = dormand_prince_step(state, &accel, 0.0, h * 0.5);
        assert!(err_full > 0.0 && err_half > 0.0);
        let ratio = err_full / err_half;
        // Error estimate is O(dt^5): halving dt should cut it by ~2^5 = 32.
        assert!(
            ratio > 16.0 && ratio < 64.0,
            "unexpected convergence ratio {ratio}"
        );
    }

    #[test]
    fn dp_step_tiny_dt_is_identity() {
        let (cfg, state, _) = circular_two_body(7.0e6);
        let accel = |r: Vec3, t: f64| acceleration(r, &cfg, t);
        let (next, err) = dormand_prince_step(state, &accel, 0.0, 1e-9);
        assert!(err < 1e-12, "error estimate {err}");
        assert!(next.r.sub(state.r).norm() < 1e-3);
        assert!(next.v.sub(state.v).norm() < 1e-6);
    }

    #[test]
    fn adaptive_energy_conservation_two_body() {
        let (cfg, state, period) = circular_two_body(7.0e6);
        let e0 = specific_energy(&state, cfg.mu_earth);
        let traj = propagate_adaptive_detailed(state, period, 60.0, 1e-10, &cfg);
        let e1 = specific_energy(traj.last(), cfg.mu_earth);
        let rel = (e1 - e0).abs() / e0.abs();
        assert!(rel < 1e-9, "adaptive relative energy drift {rel}");
        assert!(traj.accepted > 0, "no adaptive steps were taken");
        assert!(traj.t.len() == traj.states.len());
    }

    #[test]
    fn adaptive_closure_over_one_period() {
        let (cfg, state, period) = circular_two_body(7.0e6);
        let traj = propagate_adaptive_detailed(state, period, 60.0, 1e-10, &cfg);
        assert_eq!(traj.t.last().copied(), Some(period));
        let pos_err = traj.last().r.sub(state.r).norm();
        assert!(pos_err < 1.0e3, "adaptive position error {pos_err}");
    }

    #[test]
    fn adaptive_beats_fixed_step_verlet() {
        let (cfg, state, period) = circular_two_body(7.0e6);
        let e0 = specific_energy(&state, cfg.mu_earth);

        // Fixed-step velocity Verlet over the same span.
        let dt = 60.0;
        let steps = (period / dt).round() as usize;
        let verlet = propagate(state, steps, dt, &cfg);
        let verlet_last = verlet.last().unwrap();
        let verlet_energy_drift =
            (specific_energy(verlet_last, cfg.mu_earth) - e0).abs() / e0.abs();
        let verlet_pos_err = verlet_last.r.sub(state.r).norm();

        // Adaptive Dormand-Prince over the same span.
        let traj = propagate_adaptive_detailed(state, period, dt, 1e-11, &cfg);
        let dp_last = traj.last();
        let dp_energy_drift = (specific_energy(dp_last, cfg.mu_earth) - e0).abs() / e0.abs();
        let dp_pos_err = dp_last.r.sub(state.r).norm();

        assert!(
            dp_energy_drift < verlet_energy_drift,
            "DP energy drift {dp_energy_drift} !< Verlet {verlet_energy_drift}"
        );
        assert!(
            dp_pos_err < verlet_pos_err,
            "DP position error {dp_pos_err} !< Verlet {verlet_pos_err}"
        );
    }

    #[test]
    fn adaptive_accepts_looser_tolerance_with_larger_steps() {
        let (cfg, state, period) = circular_two_body(7.0e6);
        let loose = propagate_adaptive_detailed(state, period, 60.0, 1e-6, &cfg);
        let tight = propagate_adaptive_detailed(state, period, 60.0, 1e-11, &cfg);
        assert!(
            loose.accepted < tight.accepted,
            "looser tolerance should need fewer steps ({} vs {})",
            loose.accepted,
            tight.accepted
        );
    }

    #[test]
    fn test_vec3_ops() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        assert_eq!(a.add(b), Vec3::new(5.0, 7.0, 9.0));
        assert_eq!(a.sub(b), Vec3::new(-3.0, -3.0, -3.0));
        assert_eq!(a.scale(2.0), Vec3::new(2.0, 4.0, 6.0));
        assert_eq!(a.dot(b), 1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0);
        assert_eq!(Vec3::ZERO, Vec3::new(0.0, 0.0, 0.0));
    }
}

#[cfg(test)]
mod burn_tests {
    use super::*;

    #[test]
    fn impulsive_burn_changes_velocity() {
        let cfg = Config::default();
        let state = State {
            r: Vec3::new(7.0e6, 0.0, 0.0),
            v: Vec3::new(0.0, 7.5e3, 0.0),
        };
        let mut cfg_burn = cfg.clone();
        cfg_burn.burns.push(Burn {
            t: 0.0,
            dv: Vec3::new(0.0, 10.0, 0.0),
        });
        let traj_none = propagate(state, 2, 60.0, &cfg);
        let traj_burn = propagate(state, 2, 60.0, &cfg_burn);
        // After the burn, velocity should differ.
        let v_none = traj_none.last().unwrap().v.y;
        let v_burn = traj_burn.last().unwrap().v.y;
        assert!(
            (v_burn - v_none).abs() > 5.0,
            "burn did not affect velocity"
        );
    }

    #[test]
    fn burn_cap_matches_cuda() {
        let mut cfg = Config {
            burns: (0..MAX_BURNS)
                .map(|i| Burn {
                    t: i as f64,
                    dv: Vec3::new(0.0, 0.0, 0.0),
                })
                .collect(),
            ..Config::default()
        };
        assert!(cfg.validate().is_ok());

        cfg.burns.push(Burn {
            t: 9.0,
            dv: Vec3::ZERO,
        });
        assert!(matches!(
            cfg.validate(),
            Err(ConfigError::TooManyBurns {
                count: 9,
                max: MAX_BURNS
            })
        ));
    }

    #[test]
    fn adaptive_applies_burn_on_backward_arc() {
        let cfg = Config {
            moon_distance: 0.0,
            mu_moon: 0.0,
            epoch: 1_000.0,
            ..Config::default()
        };

        let r0 = 7.0e6;
        let v0 = (cfg.mu_earth / r0).sqrt();
        let state = State {
            r: Vec3::new(r0, 0.0, 0.0),
            v: Vec3::new(0.0, v0, 0.0),
        };

        let mut burned = cfg.clone();
        burned.burns.push(Burn {
            t: 400.0,
            dv: Vec3::new(0.0, 5.0, 0.0),
        });

        let coast = propagate_adaptive_detailed(state, 0.0, 60.0, 1e-10, &cfg);
        let with_burn = propagate_adaptive_detailed(state, 0.0, 60.0, 1e-10, &burned);

        assert_eq!(coast.t.last().copied(), Some(0.0));
        assert_eq!(with_burn.t.last().copied(), Some(0.0));
        let dv = with_burn.last().v.y - coast.last().v.y;
        assert!(
            (dv - 5.0).abs() < 0.5,
            "backward burn delta-v {dv}, expected ~5"
        );
    }
}

#[cfg(test)]
mod validation_tests {
    use super::*;

    #[test]
    fn default_config_is_valid() {
        assert!(Config::default().validate().is_ok());
    }

    #[test]
    fn constants_match_cuda_kernel() {
        // These are the literals in src/kernels/multibody_solver.cuh.  If one
        // side changes, this test forces the other side to follow.
        assert_eq!(K_MU_EARTH, 3.986004418e14);
        assert_eq!(K_MU_MOON, 4.9048695e12);
        assert_eq!(K_EARTH_RADIUS, 6.378137e6);
        assert_eq!(K_EARTH_J2, 1.08262668e-3);
        assert_eq!(K_EARTH_MOON_DISTANCE, 3.844e8);
        assert_eq!(K_MOON_PERIOD, 2.3605915e6);
        assert_eq!(MAX_BURNS, 8);
    }

    #[test]
    fn rejects_invalid_fields_like_cuda() {
        let cfg = Config {
            dt: 0.0,
            ..Config::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(ConfigError::OutOfRange { field: "dt", .. })
        ));

        let cfg = Config {
            epoch: f64::NAN,
            ..Config::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(ConfigError::NotFinite("epoch"))
        ));

        let cfg = Config {
            mu_earth: -1.0,
            ..Config::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(ConfigError::OutOfRange {
                field: "mu_earth",
                ..
            })
        ));

        let cfg = Config {
            moon_distance: 1.0,
            ..Config::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(ConfigError::OutOfRange {
                field: "moon_distance",
                ..
            })
        ));

        let cfg = Config {
            perturbation: 2,
            ..Config::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(ConfigError::UnsupportedPerturbation(2))
        ));

        let cfg = Config {
            burns: vec![Burn {
                t: f64::INFINITY,
                dv: Vec3::ZERO,
            }],
            ..Config::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(ConfigError::NotFinite("burn.t"))
        ));
    }

    #[test]
    fn negative_step_count_yields_initial_state_only() {
        let cfg = Config::default();
        let state = State {
            r: Vec3::new(7.0e6, 0.0, 0.0),
            v: Vec3::new(0.0, 7.5e3, 0.0),
        };
        let traj = propagate(state, -5i32, 60.0, &cfg);
        assert_eq!(traj.len(), 1);
        assert_eq!(traj[0], state);
    }

    #[test]
    fn rust_matches_cuda_two_body_fixture() {
        // Same fixture and thresholds as tests/test_multibody.cpp.
        let cfg = Config {
            dt: 60.0,
            perturbation: 0,
            softening: 0.0,
            moon_distance: 0.0,
            ..Config::default()
        };

        let r0 = 7.0e6;
        let v0 = (cfg.mu_earth / r0).sqrt();
        let state = State {
            r: Vec3::new(r0, 0.0, 0.0),
            v: Vec3::new(0.0, v0, 0.0),
        };
        let period = 2.0 * PI * (r0 * r0 * r0 / cfg.mu_earth).sqrt();
        let steps = (period / cfg.dt) as usize;

        let traj = propagate(state, steps, cfg.dt, &cfg);
        let last = traj.last().unwrap();

        let radius_error = (last.r.norm() - r0).abs();
        assert!(radius_error < 10.0, "radius error {radius_error} m");

        let energy = |st: &State| 0.5 * st.v.dot(st.v) - cfg.mu_earth / st.r.norm();
        let drift = (energy(last) - energy(&state)).abs() / energy(&state).abs();
        assert!(drift < 5e-6, "energy drift {drift}");
    }
}

#[cfg(test)]
mod ephemeris_tests {
    use super::*;

    fn two_body_config() -> Config {
        Config {
            moon_distance: 0.0,
            mu_moon: 0.0,
            ..Config::default()
        }
    }

    #[test]
    fn analytic_ephemeris_matches_the_free_function() {
        let cfg = Config::default();
        for &t in &[0.0, 123.0, 86_400.0, -4_000.0] {
            let (earth, moon) = earth_moon_positions(t, &cfg);
            let bodies = AnalyticEphemeris.positions(t, &cfg);
            assert_eq!(bodies.earth, earth);
            assert_eq!(bodies.moon, Some(moon));
            assert_eq!(bodies.sun, None);
            assert_eq!(AnalyticEphemeris.span(), None);
        }
    }

    #[test]
    fn disabling_the_moon_removes_the_body_from_the_field() {
        let mut cfg = Config::default();
        assert!(AnalyticEphemeris.positions(0.0, &cfg).moon.is_some());
        cfg.moon_distance = 0.0;
        assert!(AnalyticEphemeris.positions(0.0, &cfg).moon.is_none());
    }

    #[test]
    fn tabulated_ephemeris_interpolates_linearly() {
        let sample = |t: f64| EphemerisSample {
            t,
            earth: Vec3::new(t, 2.0 * t, -t),
            moon: Vec3::new(3.0 * t, 4.0, 5.0),
            sun: None,
        };
        let table = TabulatedEphemeris::new(vec![sample(0.0), sample(10.0)]).unwrap();
        assert_eq!(table.len(), 2);
        assert!(!table.is_empty());
        assert!(!table.has_sun());
        assert_eq!(table.time_span(), (0.0, 10.0));
        assert!(table.covers(0.0, 10.0));
        assert!(table.covers(10.0, 2.5));
        assert!(!table.covers(-1.0, 5.0));
        assert!(!table.covers(5.0, 11.0));
        assert!(matches!(
            table.check_span(-1.0, 5.0),
            Err(EphemerisError::OutOfRange { .. })
        ));

        let cfg = Config::default();
        let bodies = table.positions(2.5, &cfg);
        assert!((bodies.earth.x - 2.5).abs() < 1e-12);
        assert!((bodies.earth.y - 5.0).abs() < 1e-12);
        let moon = bodies.moon.unwrap();
        assert!((moon.x - 7.5).abs() < 1e-12);
        assert!((moon.y - 4.0).abs() < 1e-12);
        assert_eq!(bodies.sun, None);

        // Outside the span the nearest sample is clamped.
        assert_eq!(table.positions(-5.0, &cfg).earth, sample(0.0).earth);
        assert_eq!(table.positions(50.0, &cfg).earth, sample(10.0).earth);
    }

    #[test]
    fn tabulated_ephemeris_rejects_bad_tables() {
        assert!(matches!(
            TabulatedEphemeris::new(Vec::new()),
            Err(EphemerisError::Empty)
        ));

        let sample = |t: f64| EphemerisSample {
            t,
            earth: Vec3::ZERO,
            moon: Vec3::ZERO,
            sun: None,
        };
        assert!(matches!(
            TabulatedEphemeris::new(vec![sample(10.0), sample(5.0)]),
            Err(EphemerisError::NotSorted { index: 1 })
        ));
        assert!(matches!(
            TabulatedEphemeris::new(vec![sample(f64::NAN)]),
            Err(EphemerisError::NonFinite { index: 0 })
        ));

        let mut first = sample(0.0);
        first.sun = Some(Vec3::new(1.0, 0.0, 0.0));
        assert!(matches!(
            TabulatedEphemeris::new(vec![first, sample(1.0)]),
            Err(EphemerisError::InconsistentSun { index: 1 })
        ));
    }

    #[test]
    fn moon_track_reconstructs_the_analytic_barycentric_geometry() {
        let cfg = Config::default();
        let samples: Vec<EphemerisSample> = (0..=720)
            .map(|i| {
                let t = i as f64 * 10.0;
                let (earth, moon) = earth_moon_positions(t, &cfg);
                EphemerisSample {
                    t,
                    earth,
                    moon,
                    sun: None,
                }
            })
            .collect();
        let table = TabulatedEphemeris::from_moon_track(
            samples.iter().map(|s| (s.t, s.moon)),
            cfg.mu_earth,
            cfg.mu_moon,
        )
        .unwrap();

        for expected in samples.iter().step_by(97) {
            let bodies = table.positions(expected.t, &cfg);
            assert!((bodies.earth.sub(expected.earth)).norm() < 1e-6);
            assert!((bodies.moon.unwrap().sub(expected.moon)).norm() < 1e-6);
            // mu_e * r_e + mu_m * r_m must cancel at the barycentre.
            let balance = bodies
                .earth
                .scale(cfg.mu_earth)
                .add(bodies.moon.unwrap().scale(cfg.mu_moon));
            let scale = cfg.moon_distance * cfg.mu_moon;
            assert!(
                balance.norm() / scale < 1e-12,
                "barycentre residual {}",
                balance.norm()
            );
        }

        let state = State {
            r: Vec3::new(6.6e6, 0.0, 0.0),
            v: Vec3::new(0.0, 7.5e3, 0.0),
        };
        let analytic = propagate(state, 720, 10.0, &cfg);
        let tabulated = propagate_with(state, 720, 10.0, &cfg, &table);
        let diff = analytic
            .last()
            .unwrap()
            .r
            .sub(tabulated.last().unwrap().r)
            .norm();
        assert!(diff < 1e-6, "derived Earth track diverged by {diff} m");
    }

    #[test]
    fn moon_track_requires_physical_masses() {
        let track = || vec![(0.0, Vec3::new(1.0, 0.0, 0.0))];
        assert_eq!(
            TabulatedEphemeris::from_moon_track(track(), 0.0, 1.0),
            Err(EphemerisError::NonPositiveMass)
        );
        assert_eq!(
            TabulatedEphemeris::from_moon_track(track(), 1.0, 0.0),
            Err(EphemerisError::NonPositiveMass)
        );
        assert_eq!(
            TabulatedEphemeris::from_moon_track(track(), f64::NAN, 1.0),
            Err(EphemerisError::NonPositiveMass)
        );
        assert!(TabulatedEphemeris::from_moon_track(track(), 398_600.0, 4_904.0).is_ok());
    }

    #[test]
    fn tabulated_ephemeris_reproduces_the_analytic_trajectory() {
        let cfg = Config::default();
        let state = State {
            r: Vec3::new(6.6e6, 0.0, 0.0),
            v: Vec3::new(0.0, 7.5e3, 0.0),
        };
        let samples = (0..=720)
            .map(|i| {
                let t = i as f64 * 10.0;
                let (earth, moon) = earth_moon_positions(t, &cfg);
                EphemerisSample {
                    t,
                    earth,
                    moon,
                    sun: None,
                }
            })
            .collect();
        let table = TabulatedEphemeris::new(samples).unwrap();

        let analytic = propagate(state, 720, 10.0, &cfg);
        let tabulated = propagate_with(state, 720, 10.0, &cfg, &table);
        let diff = analytic
            .last()
            .unwrap()
            .r
            .sub(tabulated.last().unwrap().r)
            .norm();
        assert!(diff < 1e-6, "tabulated/analytic divergence {diff} m");
    }

    #[test]
    fn sun_term_is_opt_in() {
        let cfg = Config {
            mu_earth: 0.0,
            mu_moon: 0.0,
            mu_sun: K_MU_SUN,
            ..Config::default()
        };

        let table = TabulatedEphemeris::new(vec![EphemerisSample {
            t: 0.0,
            earth: Vec3::ZERO,
            moon: Vec3::ZERO,
            sun: Some(Vec3::new(1.5e11, 0.0, 0.0)),
        }])
        .unwrap();
        assert!(table.has_sun());

        let r = Vec3::new(1.5e11 + 1.0e9, 0.0, 0.0);
        let a = acceleration_with(r, &cfg, 0.0, &table);
        let expected = -K_MU_SUN / 1.0e18;
        assert!(
            (a.x - expected).abs() / expected.abs() < 1e-12,
            "sun acceleration {}, expected {expected}",
            a.x
        );
        assert_eq!(a.y, 0.0);
        assert_eq!(a.z, 0.0);

        // With the default solar parameter the third body contributes nothing.
        let no_sun = Config {
            mu_earth: 0.0,
            mu_moon: 0.0,
            ..Config::default()
        };
        assert_eq!(acceleration_with(r, &no_sun, 0.0, &table), Vec3::ZERO);
    }

    #[test]
    fn adaptive_integrates_backwards_to_the_start_epoch() {
        let cfg = two_body_config();
        let r0 = 7.0e6;
        let v0 = (cfg.mu_earth / r0).sqrt();
        let start = State {
            r: Vec3::new(r0, 0.0, 0.0),
            v: Vec3::new(0.0, v0, 0.0),
        };
        let period = 2.0 * PI * (r0.powi(3) / cfg.mu_earth).sqrt();

        let forward = propagate_adaptive_detailed(start, period, 60.0, 1e-10, &cfg);
        let s_period = *forward.last();

        let mut back_cfg = cfg.clone();
        back_cfg.epoch = period;
        let backward = propagate_adaptive_detailed(s_period, 0.0, 60.0, 1e-10, &back_cfg);

        assert_eq!(backward.t.last().copied(), Some(0.0));
        let closure = backward.last().r.sub(start.r).norm();
        assert!(closure < 1.0e3, "backward closure error {closure} m");
    }
}
