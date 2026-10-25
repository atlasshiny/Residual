//! SPICE kernel loading and ephemeris queries.
//!
//! The rest of the ingest pipeline compares a propagated trajectory against a
//! reference trajectory.  That reference can come from a CSV export (see
//! [`crate::Telemetry`]) *or* from a SPICE kernel.  This module provides the
//! second path:
//!
//! * [`KernelSet`] discovers and loads kernel files by extension (`.bsp`,
//!   `.tls`, `.bpc`, `.tpc`, `.pca`, …).  Leap-second kernels are parsed
//!   directly (see [`LeapSecondTable`]); ephemeris kernels are handed to a
//!   backend provider.
//! * [`Ephemeris`] is the small abstraction over "give me a state vector for
//!   `target` relative to `observer` at TDB seconds past J2000".  It can be
//!   backed by:
//!   * [`MockEphemeris`] — a deterministic analytic ephemeris with no external
//!     data, used by the unit tests and by pipelines that don't have real
//!     kernels on disk.
//!   * [`AniseEphemeris`] — real SPICE support via the `anise` crate, enabled
//!     with the `anise` feature.
//!
//! States are always returned in **SI units** (metres and metres per second),
//! matching [`dynamics::State`].  SPICE/ANISE natively reports kilometres, so
//! the ANISE backend converts on the way out.
//!
//! Time is the SPICE "ephemeris time": TDB seconds past the J2000 reference
//! epoch (2000-01-01 12:00:00 TDB).

use std::error::Error;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use dynamics::{Config, State, Vec3};

/// A NAIF body identifier, with named variants for the bodies this pipeline
/// cares about.
///
/// Any body not enumerated here can still be addressed with [`Body::Naif`],
/// which is how spacecraft (e.g. `-999`) are referenced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Body {
    /// NAIF id `0`.
    SolarSystemBarycenter,
    /// NAIF id `10`.
    Sun,
    /// NAIF id `3` (Earth-Moon barycentre).
    EarthMoonBarycenter,
    /// NAIF id `399`.
    Earth,
    /// NAIF id `301`.
    Moon,
    /// Any other NAIF id, for example a spacecraft (`-999`).
    Naif(i32),
}

impl Body {
    /// NAIF id of the Solar System barycentre.
    pub const SSB: i32 = 0;
    /// NAIF id of the Sun.
    pub const SUN: i32 = 10;
    /// NAIF id of the Earth-Moon barycentre.
    pub const EMB: i32 = 3;
    /// NAIF id of the Earth.
    pub const EARTH: i32 = 399;
    /// NAIF id of the Moon.
    pub const MOON: i32 = 301;

    /// The NAIF integer id for this body.
    pub const fn naif_id(self) -> i32 {
        match self {
            Body::SolarSystemBarycenter => Self::SSB,
            Body::Sun => Self::SUN,
            Body::EarthMoonBarycenter => Self::EMB,
            Body::Earth => Self::EARTH,
            Body::Moon => Self::MOON,
            Body::Naif(id) => id,
        }
    }

    /// The named body for a NAIF id, or [`Body::Naif`] for anything else.
    pub const fn from_naif_id(id: i32) -> Self {
        match id {
            Self::SSB => Body::SolarSystemBarycenter,
            Self::SUN => Body::Sun,
            Self::EMB => Body::EarthMoonBarycenter,
            Self::EARTH => Body::Earth,
            Self::MOON => Body::Moon,
            other => Body::Naif(other),
        }
    }
}

impl fmt::Display for Body {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Body::SolarSystemBarycenter => write!(f, "Solar System Barycenter (0)"),
            Body::Sun => write!(f, "Sun (10)"),
            Body::EarthMoonBarycenter => write!(f, "Earth-Moon Barycenter (3)"),
            Body::Earth => write!(f, "Earth (399)"),
            Body::Moon => write!(f, "Moon (301)"),
            Body::Naif(id) => write!(f, "NAIF body {id}"),
        }
    }
}

/// The kind of SPICE/ANISE kernel a file holds, inferred from its extension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelKind {
    /// Binary SPK ephemeris (`.bsp`).
    BinarySpk,
    /// Binary planetary constants (`.bpc`).
    BinaryPck,
    /// Leap-second kernel, text (`.tls`).
    LeapSeconds,
    /// Text planetary constants (`.tpc`, `.pck`).
    TextPck,
    /// ANISE-native dataset (`.pca`, `.epa`).
    AniseKernel,
    /// Meta-kernel listing other kernels (`.tm`).
    MetaKernel,
}

impl KernelKind {
    /// Infer the kernel kind from a file extension (case-insensitive).
    pub fn from_extension(extension: &str) -> Option<Self> {
        match extension
            .trim_start_matches('.')
            .to_ascii_lowercase()
            .as_str()
        {
            "bsp" => Some(Self::BinarySpk),
            "bpc" => Some(Self::BinaryPck),
            "tls" => Some(Self::LeapSeconds),
            "tpc" | "pck" => Some(Self::TextPck),
            "pca" | "epa" | "asi" => Some(Self::AniseKernel),
            "tm" => Some(Self::MetaKernel),
            _ => None,
        }
    }

    /// Whether this kernel can carry state vectors you can query.
    pub fn is_ephemeris(self) -> bool {
        matches!(self, KernelKind::BinarySpk | KernelKind::AniseKernel)
    }
}

/// A kernel file plus its resolved [`KernelKind`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelPath {
    pub path: PathBuf,
    pub kind: KernelKind,
}

impl KernelPath {
    /// Resolve a path to a kernel by its extension.
    pub fn new(path: impl Into<PathBuf>) -> Result<Self, SpiceError> {
        let path = path.into();
        let extension = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or_default()
            .to_string();
        let kind = KernelKind::from_extension(&extension).ok_or_else(|| {
            SpiceError::UnsupportedExtension {
                path: path.display().to_string(),
                extension,
            }
        })?;
        Ok(Self { path, kind })
    }
}

/// One leap-second record: the accumulated TAI-UTC offset from `effective`
/// onwards.
#[derive(Debug, Clone, PartialEq)]
pub struct LeapSecondEntry {
    /// TAI-UTC in seconds (integer-valued in practice).
    pub tai_minus_utc_s: f64,
    /// Effective date as written in the kernel, e.g. `1972-JAN-1`.
    pub effective: String,
}

/// The subset of a NAIF leap-seconds (`.tls`) kernel this pipeline needs.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct LeapSecondTable {
    entries: Vec<LeapSecondEntry>,
    delta_t_a: Option<f64>,
}

impl LeapSecondTable {
    /// Parse a leap-second kernel already in memory.
    pub fn from_tls_str(text: &str) -> Result<Self, SpiceError> {
        match parse_tls(text) {
            Ok((entries, delta_t_a)) => Ok(Self { entries, delta_t_a }),
            Err(msg) => Err(SpiceError::Parse {
                path: "<inline>".to_string(),
                msg,
            }),
        }
    }

    /// Load and parse a `.tls` file from disk.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, SpiceError> {
        let path = path.as_ref();
        let text = fs::read_to_string(path).map_err(|e| SpiceError::io(path, e))?;
        match parse_tls(&text) {
            Ok((entries, delta_t_a)) => Ok(Self { entries, delta_t_a }),
            Err(msg) => Err(SpiceError::Parse {
                path: path.display().to_string(),
                msg,
            }),
        }
    }

    /// The leap-second records, in kernel order (ascending effective date).
    pub fn entries(&self) -> &[LeapSecondEntry] {
        &self.entries
    }

    /// The `DELTET/DELTA_T_A` constant (TT-TAI offset) if present, in seconds.
    pub fn delta_t_a(&self) -> Option<f64> {
        self.delta_t_a
    }

    /// The most recent (largest) TAI-UTC offset, i.e. the current leap-second
    /// count.
    pub fn latest_tai_minus_utc(&self) -> Option<f64> {
        self.entries.last().map(|e| e.tai_minus_utc_s)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

fn parse_tls(text: &str) -> Result<(Vec<LeapSecondEntry>, Option<f64>), String> {
    let delta_t_a =
        find_scalar(text, "DELTET/DELTA_T_A").or_else(|| find_scalar(text, "DELTA_T_A"));
    let entries = parse_delta_at(text)?;
    Ok((entries, delta_t_a))
}

/// Read a `KEY = value` scalar assignment, returning the parsed value.
fn find_scalar(text: &str, key: &str) -> Option<f64> {
    for line in text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix(key) {
            if let Some(rest) = rest.trim_start().strip_prefix('=') {
                let value = rest.split_whitespace().next()?;
                return value.parse::<f64>().ok();
            }
        }
    }
    None
}

/// Parse the `DELTET/DELTA_AT = ( offset, @date, ... )` array.
fn parse_delta_at(text: &str) -> Result<Vec<LeapSecondEntry>, String> {
    const KEY: &str = "DELTET/DELTA_AT";
    let start = text
        .find(KEY)
        .ok_or_else(|| format!("missing {KEY} record"))?;
    let after = &text[start + KEY.len()..];
    let open = after
        .find('(')
        .ok_or_else(|| format!("missing '(' after {KEY}"))?;
    let rest = &after[open + 1..];
    let close = rest
        .find(')')
        .ok_or_else(|| format!("missing ')' closing {KEY}"))?;

    let tokens: Vec<&str> = rest[..close]
        .split(|c: char| c == ',' || c.is_whitespace())
        .filter(|t| !t.is_empty())
        .collect();

    let mut entries = Vec::new();
    let mut i = 0;
    while i < tokens.len() {
        let tai_minus_utc_s = tokens[i]
            .parse::<f64>()
            .map_err(|_| format!("invalid leap-second value '{}'", tokens[i]))?;
        i += 1;
        let effective = if i < tokens.len() && tokens[i].starts_with('@') {
            let date = tokens[i].trim_start_matches('@').to_string();
            i += 1;
            date
        } else {
            String::new()
        };
        entries.push(LeapSecondEntry {
            tai_minus_utc_s,
            effective,
        });
    }

    if entries.is_empty() {
        return Err(format!("{KEY} contains no records"));
    }
    Ok(entries)
}

/// A source of SPICE state vectors.
///
/// Implementors return a [`State`] for `target` relative to `observer` at the
/// requested epoch, expressed in the J2000 / ICRF inertial frame and in SI
/// units (metres, metres per second).
pub trait Ephemeris {
    /// Position and velocity of `target` relative to `observer` at `tdb_seconds`
    /// (TDB seconds past J2000), in the J2000 inertial frame and SI units.
    fn state_relative(
        &self,
        target: Body,
        observer: Body,
        tdb_seconds: f64,
    ) -> Result<State, SpiceError>;

    /// The loaded leap-second table, if any.
    fn leap_seconds(&self) -> Option<&LeapSecondTable> {
        None
    }

    /// How many kernels back this provider.
    fn kernel_count(&self) -> usize {
        0
    }

    /// Convenience: state of `target` relative to the Solar System barycentre.
    fn state(&self, target: Body, tdb_seconds: f64) -> Result<State, SpiceError> {
        self.state_relative(target, Body::SolarSystemBarycenter, tdb_seconds)
    }
}

/// A deterministic, data-free ephemeris used by tests and by pipelines that
/// don't have real SPICE kernels available.
///
/// Earth and Moon follow the same circular barycentric model as
/// [`dynamics::earth_moon_positions`], with analytic velocities.  An optional
/// [`SpacecraftMock`] rides on a circular orbit about the Earth.  This is a
/// *mock*: it is smooth and self-consistent, but it is not a high-fidelity
/// ephemeris.
#[derive(Debug, Clone)]
pub struct MockEphemeris {
    config: Config,
    spacecraft: Option<SpacecraftMock>,
    leap_seconds: Option<LeapSecondTable>,
    kernel_count: usize,
}

impl Default for MockEphemeris {
    fn default() -> Self {
        Self::new()
    }
}

impl MockEphemeris {
    pub fn new() -> Self {
        Self {
            config: Config::default(),
            spacecraft: None,
            leap_seconds: None,
            kernel_count: 0,
        }
    }

    pub fn with_config(mut self, config: Config) -> Self {
        self.config = config;
        self
    }

    pub fn with_spacecraft(mut self, spacecraft: SpacecraftMock) -> Self {
        self.spacecraft = Some(spacecraft);
        self
    }

    pub fn with_leap_seconds(mut self, table: LeapSecondTable) -> Self {
        self.leap_seconds = Some(table);
        self
    }

    pub fn with_kernel_count(mut self, count: usize) -> Self {
        self.kernel_count = count;
        self
    }

    /// Absolute (barycentric) state of a body in the mock's inertial frame.
    fn absolute_state(&self, body: Body, t: f64) -> Result<State, SpiceError> {
        let (earth, moon) = circular_earth_moon_states(t, &self.config);
        match body {
            Body::SolarSystemBarycenter | Body::EarthMoonBarycenter => Ok(State {
                r: Vec3::ZERO,
                v: Vec3::ZERO,
            }),
            // The mock has no solar ephemeris; the Sun is placed at the
            // barycentre.  Use the ANISE backend for a real Sun state.
            Body::Sun => Ok(State {
                r: Vec3::ZERO,
                v: Vec3::ZERO,
            }),
            Body::Earth => Ok(earth),
            Body::Moon => Ok(moon),
            Body::Naif(id) => match &self.spacecraft {
                Some(sc) if sc.body.naif_id() == id => {
                    let rel = sc.relative_state(t);
                    Ok(State {
                        r: earth.r.add(rel.r),
                        v: earth.v.add(rel.v),
                    })
                }
                _ => Err(SpiceError::UnknownBody(body)),
            },
        }
    }
}

impl Ephemeris for MockEphemeris {
    fn state_relative(
        &self,
        target: Body,
        observer: Body,
        tdb_seconds: f64,
    ) -> Result<State, SpiceError> {
        if !tdb_seconds.is_finite() {
            return Err(SpiceError::Ephemeris(
                "epoch must be a finite TDB second count".to_string(),
            ));
        }
        let target_state = self.absolute_state(target, tdb_seconds)?;
        let observer_state = self.absolute_state(observer, tdb_seconds)?;
        Ok(State {
            r: target_state.r.sub(observer_state.r),
            v: target_state.v.sub(observer_state.v),
        })
    }

    fn leap_seconds(&self) -> Option<&LeapSecondTable> {
        self.leap_seconds.as_ref()
    }

    fn kernel_count(&self) -> usize {
        self.kernel_count
    }
}

/// A spacecraft on a circular two-body orbit about the Earth, for use with
/// [`MockEphemeris`].
#[derive(Debug, Clone, PartialEq)]
pub struct SpacecraftMock {
    /// NAIF id this spacecraft answers to (typically [`Body::Naif`]`(-999)`).
    pub body: Body,
    /// Central-body gravitational parameter, m^3/s^2.
    pub mu: f64,
    /// Circular orbit radius, m.
    pub radius: f64,
    /// Initial phase in the orbital plane, rad.
    pub phase: f64,
    /// Orbit-plane inclination about the x-axis, rad.
    pub inclination: f64,
}

impl SpacecraftMock {
    pub fn circular(body: Body, mu: f64, radius: f64, phase: f64, inclination: f64) -> Self {
        Self {
            body,
            mu,
            radius,
            phase,
            inclination,
        }
    }

    /// Position and velocity relative to the Earth at `t` (TDB seconds).
    fn relative_state(&self, t: f64) -> State {
        let n = if self.radius > 0.0 && self.mu > 0.0 {
            (self.mu / self.radius.powi(3)).sqrt()
        } else {
            0.0
        };
        let theta = self.phase + n * t;
        let (sin_t, cos_t) = theta.sin_cos();
        let (sin_i, cos_i) = self.inclination.sin_cos();
        let speed = (self.mu / self.radius).sqrt();

        // In-plane basis tilted by `inclination` about the x-axis:
        //   p_hat = (1, 0, 0), q_hat = (0, cos i, sin i).
        let p_hat = Vec3::new(1.0, 0.0, 0.0);
        let q_hat = Vec3::new(0.0, cos_i, sin_i);
        let r_hat = p_hat.scale(cos_t).add(q_hat.scale(sin_t));
        let t_hat = p_hat.scale(-sin_t).add(q_hat.scale(cos_t));

        State {
            r: r_hat.scale(self.radius),
            v: t_hat.scale(speed),
        }
    }
}

/// Circular barycentric Earth/Moon states with analytic velocities.
///
/// Mirrors the position model in [`dynamics::earth_moon_positions`].
fn circular_earth_moon_states(t: f64, config: &Config) -> (State, State) {
    let zero = State {
        r: Vec3::ZERO,
        v: Vec3::ZERO,
    };
    if config.moon_distance <= 0.0 {
        return (zero, zero);
    }

    let total_mu = config.mu_earth + config.mu_moon;
    let n = if total_mu > 0.0 {
        (total_mu / config.moon_distance.powi(3)).sqrt()
    } else {
        0.0
    };
    let theta = config.moon_phase + n * t;
    let (sin_t, cos_t) = theta.sin_cos();

    let f = if total_mu > 0.0 {
        config.mu_moon / total_mu
    } else {
        0.0
    };
    let d = config.moon_distance;

    let earth = State {
        r: Vec3::new(-d * f * cos_t, -d * f * sin_t, 0.0),
        v: Vec3::new(d * f * n * sin_t, -d * f * n * cos_t, 0.0),
    };
    let moon = State {
        r: Vec3::new(d * (1.0 - f) * cos_t, d * (1.0 - f) * sin_t, 0.0),
        v: Vec3::new(-d * (1.0 - f) * n * sin_t, d * (1.0 - f) * n * cos_t, 0.0),
    };
    (earth, moon)
}

/// A collection of discovered kernel files plus any parsed leap-second table.
#[derive(Debug, Clone, Default)]
pub struct KernelSet {
    kernels: Vec<KernelPath>,
    leap_seconds: Option<LeapSecondTable>,
}

impl KernelSet {
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a set by loading every path, resolving kinds from extensions.
    pub fn load<I, P>(paths: I) -> Result<Self, SpiceError>
    where
        I: IntoIterator<Item = P>,
        P: AsRef<Path>,
    {
        let mut set = Self::new();
        for path in paths {
            set.add(path)?;
        }
        Ok(set)
    }

    /// Add one kernel.  `.tls` files are parsed immediately; other kinds are
    /// only recorded (their contents are read by a backend on demand).
    pub fn add<P: AsRef<Path>>(&mut self, path: P) -> Result<&mut Self, SpiceError> {
        let kernel = KernelPath::new(path.as_ref().to_path_buf())?;
        if kernel.kind == KernelKind::LeapSeconds {
            self.leap_seconds = Some(LeapSecondTable::load(&kernel.path)?);
        }
        self.kernels.push(kernel);
        Ok(self)
    }

    pub fn kernels(&self) -> &[KernelPath] {
        &self.kernels
    }

    pub fn leap_seconds(&self) -> Option<&LeapSecondTable> {
        self.leap_seconds.as_ref()
    }

    pub fn len(&self) -> usize {
        self.kernels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.kernels.is_empty()
    }

    /// Build a [`MockEphemeris`] seeded with this set's metadata.  Useful for
    /// exercising the pipeline without real kernels on disk.
    pub fn mock_provider(&self) -> MockEphemeris {
        let mut mock = MockEphemeris::new().with_kernel_count(self.kernels.len());
        if let Some(table) = &self.leap_seconds {
            mock = mock.with_leap_seconds(table.clone());
        }
        mock
    }

    /// Build a real, ANISE-backed provider from this set.  Requires the
    /// `anise` feature.
    #[cfg(feature = "anise")]
    pub fn anise_provider(&self) -> Result<AniseEphemeris, SpiceError> {
        AniseEphemeris::from_kernel_set(self)
    }
}

/// Uniformly spaced epochs, `count` samples starting at `t0` with step `dt`.
pub fn uniform_times(t0: f64, dt: f64, count: usize) -> Vec<f64> {
    (0..count).map(|i| t0 + dt * i as f64).collect()
}

/// Real SPICE ephemeris backed by the `anise` crate.
///
/// Load [`KernelSet`]s and query state vectors for any NAIF body present in the
/// loaded ephemeris.  Requires the `anise` feature.
#[cfg(feature = "anise")]
pub struct AniseEphemeris {
    almanac: anise::prelude::Almanac,
    leap_seconds: Option<LeapSecondTable>,
    kernel_count: usize,
}

#[cfg(feature = "anise")]
impl AniseEphemeris {
    /// Build a provider by handing every non-leap-second kernel to ANISE.
    pub fn from_kernel_set(set: &KernelSet) -> Result<Self, SpiceError> {
        let mut almanac = anise::prelude::Almanac::default();
        let mut kernel_count = 0;
        for kernel in set.kernels() {
            if kernel.kind == KernelKind::LeapSeconds {
                // Handled by LeapSecondTable; ANISE's time system already
                // knows the leap seconds it needs.
                continue;
            }
            let path_str = kernel.path.to_str().ok_or_else(|| {
                SpiceError::Ephemeris(format!(
                    "kernel path is not valid UTF-8: {}",
                    kernel.path.display()
                ))
            })?;
            almanac = almanac
                .load(path_str)
                .map_err(|e| SpiceError::Ephemeris(format!("{}: {e}", kernel.path.display())))?;
            kernel_count += 1;
        }
        Ok(Self {
            almanac,
            leap_seconds: set.leap_seconds().cloned(),
            kernel_count,
        })
    }

    /// The J2000 frame for a body's NAIF id.
    fn frame(body: Body) -> anise::prelude::Frame {
        anise::prelude::Frame::from_ephem_j2000(body.naif_id())
    }
}

#[cfg(feature = "anise")]
impl Ephemeris for AniseEphemeris {
    fn state_relative(
        &self,
        target: Body,
        observer: Body,
        tdb_seconds: f64,
    ) -> Result<State, SpiceError> {
        if !tdb_seconds.is_finite() {
            return Err(SpiceError::Ephemeris(
                "epoch must be a finite TDB second count".to_string(),
            ));
        }
        let epoch = anise::prelude::Epoch::from_et_seconds(tdb_seconds);
        let orbit = self
            .almanac
            .translate(Self::frame(target), Self::frame(observer), epoch, None)
            .map_err(|e| SpiceError::Ephemeris(e.to_string()))?;

        // ANISE reports km and km/s; convert to SI metres.
        Ok(State {
            r: Vec3::new(orbit.radius_km.x, orbit.radius_km.y, orbit.radius_km.z).scale(1000.0),
            v: Vec3::new(
                orbit.velocity_km_s.x,
                orbit.velocity_km_s.y,
                orbit.velocity_km_s.z,
            )
            .scale(1000.0),
        })
    }

    fn leap_seconds(&self) -> Option<&LeapSecondTable> {
        self.leap_seconds.as_ref()
    }

    fn kernel_count(&self) -> usize {
        self.kernel_count
    }
}

/// Errors raised while loading kernels or querying ephemerides.
#[derive(Debug)]
pub enum SpiceError {
    Io { path: String, err: String },
    UnsupportedExtension { path: String, extension: String },
    UnknownBody(Body),
    Parse { path: String, msg: String },
    Ephemeris(String),
}

impl SpiceError {
    fn io(path: &Path, err: std::io::Error) -> Self {
        SpiceError::Io {
            path: path.display().to_string(),
            err: err.to_string(),
        }
    }
}

impl fmt::Display for SpiceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SpiceError::Io { path, err } => write!(f, "io error for {path}: {err}"),
            SpiceError::UnsupportedExtension { path, extension } => {
                write!(f, "unsupported kernel extension '{extension}' for {path}")
            }
            SpiceError::UnknownBody(body) => write!(f, "no ephemeris for {body}"),
            SpiceError::Parse { path, msg } => write!(f, "failed to parse {path}: {msg}"),
            SpiceError::Ephemeris(msg) => write!(f, "ephemeris query failed: {msg}"),
        }
    }
}

impl Error for SpiceError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Telemetry;
    use dynamics::K_MU_EARTH;

    const TLS: &str = "\
KPL/LSK

      DELTA_T_A = 32.184
      DELTET/DELTA_T_A = 32.184
      DELTET/DELTA_AT  = ( 10, @1972-JAN-1
                           11, @1972-JUL-1
                           36, @2015-JUL-1
                           37, @2017-JAN-1 )

\\begintext
";

    #[test]
    fn body_naif_round_trip() {
        for body in [
            Body::SolarSystemBarycenter,
            Body::Sun,
            Body::EarthMoonBarycenter,
            Body::Earth,
            Body::Moon,
            Body::Naif(-999),
        ] {
            assert_eq!(Body::from_naif_id(body.naif_id()), body);
        }
        assert_eq!(Body::Earth.naif_id(), 399);
        assert_eq!(Body::from_naif_id(42), Body::Naif(42));
    }

    #[test]
    fn kernel_kind_from_extension() {
        assert_eq!(
            KernelKind::from_extension("BSP"),
            Some(KernelKind::BinarySpk)
        );
        assert_eq!(
            KernelKind::from_extension(".tls"),
            Some(KernelKind::LeapSeconds)
        );
        assert_eq!(
            KernelKind::from_extension("pca"),
            Some(KernelKind::AniseKernel)
        );
        assert_eq!(KernelKind::from_extension("txt"), None);
        assert!(KernelKind::BinarySpk.is_ephemeris());
        assert!(!KernelKind::LeapSeconds.is_ephemeris());
    }

    #[test]
    fn rejects_unknown_extension() {
        let err = KernelPath::new("/tmp/de440.txt").unwrap_err();
        assert!(matches!(err, SpiceError::UnsupportedExtension { .. }));
    }

    #[test]
    fn parses_leap_second_kernel() {
        let table = LeapSecondTable::from_tls_str(TLS).unwrap();
        assert_eq!(table.len(), 4);
        assert_eq!(table.latest_tai_minus_utc(), Some(37.0));
        assert_eq!(table.delta_t_a(), Some(32.184));
        assert_eq!(table.entries()[0].tai_minus_utc_s, 10.0);
        assert_eq!(table.entries()[0].effective, "1972-JAN-1");
        assert_eq!(table.entries()[3].effective, "2017-JAN-1");
    }

    #[test]
    fn rejects_leap_second_kernel_without_array() {
        assert!(matches!(
            LeapSecondTable::from_tls_str("KPL/LSK\nDELTA_T_A = 32.184\n"),
            Err(SpiceError::Parse { .. })
        ));
    }

    #[test]
    fn kernel_set_loads_tls_from_disk() {
        let dir = std::env::temp_dir().join("gatorhacks-ingest-spice-tests");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("synthetic.tls");
        std::fs::write(&path, TLS).unwrap();

        let set = KernelSet::load([&path]).unwrap();
        assert_eq!(set.len(), 1);
        assert_eq!(
            set.leap_seconds().unwrap().latest_tai_minus_utc(),
            Some(37.0)
        );
        assert_eq!(set.mock_provider().kernel_count(), 1);
    }

    #[test]
    fn mock_earth_moon_geometry() {
        let eph = MockEphemeris::new();
        let rel = eph.state_relative(Body::Moon, Body::Earth, 0.0).unwrap();
        // Earth-Moon separation tracks the configured distance.
        let d = Config::default().moon_distance;
        assert!((rel.r.norm() - d).abs() / d < 1e-12);

        // Relative speed is d * n for the circular model.
        let total_mu = Config::default().mu_earth + Config::default().mu_moon;
        let n = (total_mu / d.powi(3)).sqrt();
        assert!((rel.v.norm() - d * n).abs() / (d * n) < 1e-12);
    }

    #[test]
    fn mock_spacecraft_circular_orbit() {
        let radius = 7.0e6;
        let eph = MockEphemeris::new().with_spacecraft(SpacecraftMock::circular(
            Body::Naif(-999),
            K_MU_EARTH,
            radius,
            0.0,
            0.0,
        ));
        let rel = eph
            .state_relative(Body::Naif(-999), Body::Earth, 0.0)
            .unwrap();
        assert!((rel.r.norm() - radius).abs() < 1e-6);
        let expected_speed = (K_MU_EARTH / radius).sqrt();
        assert!((rel.v.norm() - expected_speed).abs() < 1e-9);
        // Circular orbit: velocity is perpendicular to position.
        assert!(rel.r.dot(rel.v).abs() / (rel.r.norm() * rel.v.norm()) < 1e-12);
    }

    #[test]
    fn state_defaults_to_ssb() {
        let eph = MockEphemeris::new();
        let a = eph.state(Body::Moon, 12.0).unwrap();
        let b = eph
            .state_relative(Body::Moon, Body::SolarSystemBarycenter, 12.0)
            .unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn unknown_body_errors() {
        let eph = MockEphemeris::new();
        assert!(matches!(
            eph.state_relative(Body::Naif(42), Body::Earth, 0.0),
            Err(SpiceError::UnknownBody(_))
        ));
    }

    #[test]
    fn epochs_must_be_finite() {
        let eph = MockEphemeris::new();
        assert!(eph
            .state_relative(Body::Moon, Body::Earth, f64::NAN)
            .is_err());
    }

    #[test]
    fn telemetry_from_ephemeris_pipeline() {
        let eph = MockEphemeris::new().with_spacecraft(SpacecraftMock::circular(
            Body::Naif(-999),
            K_MU_EARTH,
            7.0e6,
            0.0,
            0.0,
        ));
        let times = uniform_times(0.0, 60.0, 11);
        let telemetry =
            Telemetry::from_ephemeris(&eph, Body::Naif(-999), Body::Earth, &times).unwrap();
        assert_eq!(telemetry.len(), 11);

        // The interpolated state at a sample epoch matches the kernel value.
        let mid = telemetry.sample_at(300.0).unwrap();
        let direct = eph
            .state_relative(Body::Naif(-999), Body::Earth, 300.0)
            .unwrap();
        assert!((mid.r.norm() - direct.r.norm()).abs() < 1e-6);
        assert!((mid.v.norm() - direct.v.norm()).abs() < 1e-6);

        // Round-trips through the CSV representation.
        let again = Telemetry::from_csv_str(&telemetry.to_csv_string()).unwrap();
        assert_eq!(telemetry, again);
    }

    #[cfg(feature = "anise")]
    #[test]
    fn anise_provider_queries_fail_cleanly_without_data() {
        let provider = KernelSet::new().anise_provider().unwrap();
        assert_eq!(provider.kernel_count(), 0);
        // No ephemeris was loaded, so a query must error rather than panic.
        assert!(provider
            .state_relative(Body::Moon, Body::Earth, 0.0)
            .is_err());
    }

    #[cfg(feature = "anise")]
    #[test]
    fn anise_provider_rejects_invalid_kernel() {
        let dir = std::env::temp_dir().join("gatorhacks-ingest-spice-tests");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("garbage.bsp");
        std::fs::write(&path, b"not a real DAF file").unwrap();

        // Discovery succeeds; the backend reports the failure on load.
        let set = KernelSet::load([&path]).unwrap();
        assert_eq!(set.len(), 1);
        assert!(set.anise_provider().is_err());
    }
}
