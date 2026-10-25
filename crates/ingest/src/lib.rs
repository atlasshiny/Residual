//! Telemetry ingestion: CSV reference trajectories → typed states.
//!
//! Reference data (Horizons exports, flight telemetry dumps, and the synthetic
//! tables used by the tests) all share the same shape: a time column plus a
//! position and velocity per row.  This crate parses that shape without pulling
//! in a CSV dependency, validates it, and offers linear resampling so a
//! propagated trajectory can be compared against the reference at exactly the
//! reference epochs.
//!
//! Canonical header:
//!
//! ```text
//! t,x,y,z,vx,vy,vz
//! 0,7000000,0,0,0,7546.05,0
//! ```
//!
//! Column names are matched case-insensitively and a few common aliases are
//! accepted (`time`, `epoch`, `seconds` for `t`).  Unknown columns are ignored.

use std::error::Error;
use std::fmt;
use std::fs;
use std::path::Path;

use dynamics::{State, Vec3};

pub mod spice;

pub use spice::{
    Body, Ephemeris, KernelKind, KernelPath, KernelSet, LeapSecondTable, MockEphemeris,
    SpacecraftMock, SpiceError,
};

/// One reference epoch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReferenceSample {
    /// Seconds from the mission epoch.
    pub t: f64,
    pub state: State,
}

/// A validated, monotonically increasing reference trajectory.
#[derive(Debug, Clone, PartialEq)]
pub struct Telemetry {
    samples: Vec<ReferenceSample>,
}

impl Telemetry {
    pub fn new(samples: Vec<ReferenceSample>) -> Result<Self, IngestError> {
        if samples.is_empty() {
            return Err(IngestError::Empty);
        }
        for (i, sample) in samples.iter().enumerate() {
            if !sample.t.is_finite() || !sample.state.is_finite() {
                return Err(IngestError::NotFinite(i + 1));
            }
            if i > 0 && sample.t <= samples[i - 1].t {
                return Err(IngestError::NonMonotonic {
                    line: i + 1,
                    previous: samples[i - 1].t,
                    current: sample.t,
                });
            }
        }
        Ok(Self { samples })
    }

    /// Load a CSV file from disk.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, IngestError> {
        let path = path.as_ref();
        let text = fs::read_to_string(path)
            .map_err(|e| IngestError::Io(path.display().to_string(), e.to_string()))?;
        Self::from_csv_str(&text)
    }

    /// Write the trajectory back out with the canonical header.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), IngestError> {
        let path = path.as_ref();
        fs::write(path, self.to_csv_string())
            .map_err(|e| IngestError::Io(path.display().to_string(), e.to_string()))
    }

    pub fn from_csv_str(text: &str) -> Result<Self, IngestError> {
        let mut lines = text.lines().enumerate();
        let (_, header_line) = lines.next().ok_or(IngestError::Empty)?;
        let header = parse_header(header_line)?;
        let columns = Columns::from_header(&header)?;

        let mut samples = Vec::new();
        for (idx, line) in lines {
            let line_number = idx + 1;
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let fields = split_fields(line);
            if fields.len() != header.len() {
                return Err(IngestError::BadFieldCount {
                    line: line_number,
                    expected: header.len(),
                    found: fields.len(),
                });
            }
            let value = |name: &str| -> Result<f64, IngestError> {
                let col = columns.get(name).expect("validated header");
                parse_f64(fields[col], line_number, name)
            };
            samples.push(ReferenceSample {
                t: value("t")?,
                state: State {
                    r: Vec3::new(value("x")?, value("y")?, value("z")?),
                    v: Vec3::new(value("vx")?, value("vy")?, value("vz")?),
                },
            });
        }
        Self::new(samples)
    }

    pub fn to_csv_string(&self) -> String {
        let mut out = String::from("t,x,y,z,vx,vy,vz\n");
        for sample in &self.samples {
            out.push_str(&format!(
                "{},{},{},{},{},{},{}\n",
                sample.t,
                sample.state.r.x,
                sample.state.r.y,
                sample.state.r.z,
                sample.state.v.x,
                sample.state.v.y,
                sample.state.v.z
            ));
        }
        out
    }

    pub fn samples(&self) -> &[ReferenceSample] {
        &self.samples
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    pub fn times(&self) -> Vec<f64> {
        self.samples.iter().map(|s| s.t).collect()
    }

    pub fn states(&self) -> Vec<State> {
        self.samples.iter().map(|s| s.state).collect()
    }

    /// Build a reference trajectory by sampling an ephemeris.
    ///
    /// `times` are TDB seconds past J2000 (the SPICE ephemeris-time
    /// convention); each becomes the `t` column of the returned telemetry.
    /// `ephemeris` yields the state of `target` relative to `observer` in SI
    /// units, so a real SPICE kernel and the analytic [`spice::MockEphemeris`]
    /// produce the same shape of data.
    pub fn from_ephemeris(
        ephemeris: &dyn Ephemeris,
        target: Body,
        observer: Body,
        times: &[f64],
    ) -> Result<Self, IngestError> {
        let mut samples = Vec::with_capacity(times.len());
        for &t in times {
            let state = ephemeris
                .state_relative(target, observer, t)
                .map_err(|e| IngestError::Ephemeris(e.to_string()))?;
            samples.push(ReferenceSample { t, state });
        }
        Self::new(samples)
    }

    /// Linear interpolation at `t`, or `None` outside the sampled range.
    ///
    /// Velocities are interpolated the same way; for a dense table this is
    /// well inside the comparison tolerance.
    pub fn sample_at(&self, t: f64) -> Option<State> {
        if !t.is_finite() || self.samples.is_empty() {
            return None;
        }
        let first = &self.samples[0];
        let last = &self.samples[self.samples.len() - 1];
        if t < first.t || t > last.t {
            return None;
        }
        if t == first.t {
            return Some(first.state);
        }
        if t == last.t {
            return Some(last.state);
        }

        // partition_point gives the first sample with time > t; the interval
        // containing t is therefore (idx - 1, idx).
        let idx = self.samples.partition_point(|s| s.t <= t);
        let hi = &self.samples[idx];
        let lo = &self.samples[idx - 1];
        if hi.t == lo.t {
            return Some(hi.state);
        }
        let alpha = (t - lo.t) / (hi.t - lo.t);
        Some(lerp_state(lo.state, hi.state, alpha))
    }
}

fn lerp_state(a: State, b: State, alpha: f64) -> State {
    State {
        r: lerp(a.r, b.r, alpha),
        v: lerp(a.v, b.v, alpha),
    }
}

fn lerp(a: Vec3, b: Vec3, alpha: f64) -> Vec3 {
    Vec3::new(
        a.x + (b.x - a.x) * alpha,
        a.y + (b.y - a.y) * alpha,
        a.z + (b.z - a.z) * alpha,
    )
}

fn parse_header(line: &str) -> Result<Vec<String>, IngestError> {
    // Strip a UTF-8 BOM, which spreadsheet exports like to add.
    let line = line.trim_start_matches('\u{feff}').trim();
    if line.is_empty() {
        return Err(IngestError::Empty);
    }
    Ok(split_fields(line)
        .into_iter()
        .map(|name| name.trim().to_ascii_lowercase())
        .collect())
}

fn split_fields(line: &str) -> Vec<&str> {
    line.split(',').map(str::trim).collect()
}

fn parse_f64(field: &str, line: usize, name: &str) -> Result<f64, IngestError> {
    field.parse::<f64>().map_err(|_| IngestError::BadNumber {
        line,
        column: name.to_string(),
        value: field.to_string(),
    })
}

/// Resolved column indices for the canonical fields.
struct Columns {
    t: usize,
    x: usize,
    y: usize,
    z: usize,
    vx: usize,
    vy: usize,
    vz: usize,
}

impl Columns {
    fn from_header(header: &[String]) -> Result<Self, IngestError> {
        let find = |aliases: &[&str]| -> Option<usize> {
            header
                .iter()
                .position(|name| aliases.iter().any(|alias| name == alias))
        };
        let required = |aliases: &[&str], display: &str| -> Result<usize, IngestError> {
            find(aliases).ok_or_else(|| IngestError::MissingColumn(display.to_string()))
        };
        Ok(Self {
            t: required(&["t", "time", "epoch", "seconds"], "t")?,
            x: required(&["x"], "x")?,
            y: required(&["y"], "y")?,
            z: required(&["z"], "z")?,
            vx: required(&["vx", "v_x"], "vx")?,
            vy: required(&["vy", "v_y"], "vy")?,
            vz: required(&["vz", "v_z"], "vz")?,
        })
    }

    fn get(&self, name: &str) -> Option<usize> {
        match name {
            "t" => Some(self.t),
            "x" => Some(self.x),
            "y" => Some(self.y),
            "z" => Some(self.z),
            "vx" => Some(self.vx),
            "vy" => Some(self.vy),
            "vz" => Some(self.vz),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub enum IngestError {
    Io(String, String),
    Empty,
    MissingColumn(String),
    BadFieldCount {
        line: usize,
        expected: usize,
        found: usize,
    },
    BadNumber {
        line: usize,
        column: String,
        value: String,
    },
    NotFinite(usize),
    NonMonotonic {
        line: usize,
        previous: f64,
        current: f64,
    },
    Ephemeris(String),
}

impl fmt::Display for IngestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IngestError::Io(path, err) => write!(f, "io error for {path}: {err}"),
            IngestError::Empty => write!(f, "telemetry is empty"),
            IngestError::MissingColumn(name) => write!(f, "missing required column '{name}'"),
            IngestError::BadFieldCount {
                line,
                expected,
                found,
            } => write!(f, "line {line}: expected {expected} fields, found {found}"),
            IngestError::BadNumber {
                line,
                column,
                value,
            } => write!(
                f,
                "line {line}: column '{column}' is not a number: '{value}'"
            ),
            IngestError::NotFinite(line) => {
                write!(f, "line {line}: value is not finite")
            }
            IngestError::NonMonotonic {
                line,
                previous,
                current,
            } => write!(
                f,
                "line {line}: time must increase (previous {previous}, current {current})"
            ),
            IngestError::Ephemeris(msg) => write!(f, "ephemeris error: {msg}"),
        }
    }
}

impl Error for IngestError {}

/// Re-exported so downstream crates only need `ingest` in their dependency list.
pub use dynamics;

/// `State::is_finite` is not part of the physics API; keep the check local.
trait FiniteState {
    fn is_finite(&self) -> bool;
}

impl FiniteState for State {
    fn is_finite(&self) -> bool {
        self.r.x.is_finite()
            && self.r.y.is_finite()
            && self.r.z.is_finite()
            && self.v.x.is_finite()
            && self.v.y.is_finite()
            && self.v.z.is_finite()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CSV: &str =
        "t,x,y,z,vx,vy,vz\n0,7000000,0,0,0,7546.05,0\n60,6999962,452761,0,-488.1,7545.9,0\n";

    #[test]
    fn parses_and_interpolates() {
        let telemetry = Telemetry::from_csv_str(CSV).unwrap();
        assert_eq!(telemetry.len(), 2);
        let mid = telemetry.sample_at(30.0).unwrap();
        assert!((mid.r.x - 6_999_981.0).abs() < 1e-6);
        assert!((mid.v.y - 7545.975).abs() < 1e-6);
    }

    #[test]
    fn rejects_unsorted_times() {
        let csv = "t,x,y,z,vx,vy,vz\n10,1,2,3,4,5,6\n5,1,2,3,4,5,6\n";
        assert!(matches!(
            Telemetry::from_csv_str(csv),
            Err(IngestError::NonMonotonic { .. })
        ));
    }

    #[test]
    fn accepts_alias_header_and_extra_columns() {
        let csv = "Time, x, y, z, vx, vy, vz, note\n0,1,2,3,4,5,6,hi\n1,1,2,3,4,5,6,bye\n";
        let telemetry = Telemetry::from_csv_str(csv).unwrap();
        assert_eq!(telemetry.len(), 2);
        assert_eq!(telemetry.samples()[1].t, 1.0);
    }

    #[test]
    fn rejects_bad_numbers() {
        let csv = "t,x,y,z,vx,vy,vz\n0,not-a-number,2,3,4,5,6\n";
        assert!(matches!(
            Telemetry::from_csv_str(csv),
            Err(IngestError::BadNumber { .. })
        ));
    }

    #[test]
    fn csv_round_trip() {
        let telemetry = Telemetry::from_csv_str(CSV).unwrap();
        let again = Telemetry::from_csv_str(&telemetry.to_csv_string()).unwrap();
        assert_eq!(telemetry, again);
    }

    #[test]
    fn out_of_range_samples_are_none() {
        let telemetry = Telemetry::from_csv_str(CSV).unwrap();
        assert!(telemetry.sample_at(-1.0).is_none());
        assert!(telemetry.sample_at(61.0).is_none());
        assert_eq!(
            telemetry.sample_at(60.0),
            Some(telemetry.samples()[1].state)
        );
    }
}
