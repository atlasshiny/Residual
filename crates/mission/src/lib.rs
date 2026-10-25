//! Mission execution: initial state + solver settings -> sampled trajectory.
//!
//! This is the bridge between the serialisable mission contract and the
//! [`dynamics`] propagator.  It deliberately performs no file or Tauri I/O, so
//! the same code path is exercised by unit tests, the desktop UI, and any
//! future CLI.
//!
//! # Frames
//!
//! Inputs are given in kilometres (position) and kilometres per second
//! (velocity).  With [`MissionCenter::Earth`] the state is interpreted as
//! Earth-centred inertial and converted to the solver's Earth-Moon barycentric
//! frame by adding the analytic Earth position at the start epoch.  Output
//! positions are returned in the same frame the state was supplied in, so the
//! viewer can draw Earth at the origin.
//!
//! The Earth's barycentric velocity (~12 m/s in the analytic model) is not
//! subtracted from Earth-centred inputs.  That is well below the accuracy of
//! the circular ephemeris and is documented rather than hidden.
//!
//! # Sampling
//!
//! The propagator runs adaptively; the result is resampled onto a uniform grid
//! with cubic Hermite interpolation (position and velocity), which keeps a
//! smooth curve even when the adaptive controller takes long coast steps.

use dynamics::{
    earth_moon_positions, propagate_adaptive_detailed, AnalyticEphemeris, Burn, Config, Ephemeris,
    State, Vec3, MAX_BURNS,
};
use schema::SolverConfig;
use serde::{Deserialize, Serialize};

/// Schema tag written into the UI bundle.
pub const SCHEMA: &str = "mission-trajectory/v1";
/// Default number of output samples.
pub const DEFAULT_SAMPLES: usize = 1200;
/// Upper bound on output samples (keeps the JSON bundle and the viewer sane).
pub const MAX_SAMPLES: usize = 20_000;
/// Longest mission we accept in one call (400 days).
pub const MAX_DURATION_S: f64 = 400.0 * 86_400.0;

const RTOL: f64 = 1e-9;
const MIN_GEOCENTRIC_RADIUS_M: f64 = 1_000.0;

/// Frame the input state is expressed in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum MissionCenter {
    /// Earth-centred inertial (the natural frame for a rocket).
    #[default]
    Earth,
    /// Earth-Moon barycentric inertial (the solver's native frame).
    Barycentric,
}

impl MissionCenter {
    pub fn frame_name(self) -> &'static str {
        match self {
            Self::Earth => "earth-centered-inertial",
            Self::Barycentric => "earth-moon-barycentric",
        }
    }
}

/// One impulsive manoeuvre, relative to the start of the mission.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct MissionBurn {
    /// Burn epoch in seconds from the start of the mission.
    pub t_s: f64,
    /// Impulsive Δv in the solver frame, metres per second.
    pub dv_m_s: [f64; 3],
}

/// Input accepted by the mission runner.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct MissionRequest {
    /// Initial position, kilometres.
    pub position_km: [f64; 3],
    /// Initial velocity, kilometres per second.
    pub velocity_km_s: [f64; 3],
    /// Propagation span, seconds.
    pub duration_s: f64,
    /// Number of uniformly spaced output samples.
    #[serde(default)]
    pub samples: Option<usize>,
    /// Input/output frame.
    #[serde(default)]
    pub center: MissionCenter,
    /// Absolute epoch of the initial state, seconds.  Defaults to zero.
    #[serde(default)]
    pub start_epoch_s: Option<f64>,
    /// Impulsive burns, relative to the start epoch.
    #[serde(default)]
    pub burns: Option<Vec<MissionBurn>>,
    /// Optional solver overrides.  Missing fields take the dynamics defaults.
    #[serde(default)]
    pub solver: Option<SolverConfig>,
    /// Display name for the run.
    #[serde(default)]
    pub name: Option<String>,
}

/// Position/velocity at one sample.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct StateSample {
    pub position_km: [f64; 3],
    pub velocity_km_s: [f64; 3],
    pub radius_km: f64,
    pub speed_km_s: f64,
}

/// Run summary for the mission card.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MissionStats {
    pub samples: usize,
    pub min_radius_km: f64,
    pub max_radius_km: f64,
    pub final_radius_km: f64,
    pub min_speed_km_s: f64,
    pub max_speed_km_s: f64,
    pub accepted_steps: usize,
    pub rejected_steps: usize,
}

/// Result written back to the viewer.  The field names match the static
/// `trajectory.json` bundle (`points`, `times`, `units`) so the renderer can
/// consume both without a separate path.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MissionTrajectory {
    pub schema: &'static str,
    pub name: String,
    pub frame: &'static str,
    pub units: &'static str,
    pub points: Vec<[f64; 3]>,
    /// Seconds from the start of the mission (what the scrubber shows).
    pub times: Vec<f64>,
    /// Absolute solver times, for provenance.
    pub times_absolute: Vec<f64>,
    pub count: usize,
    pub duration_s: f64,
    pub start_epoch_s: f64,
    pub initial_state: StateSample,
    pub final_state: StateSample,
    pub stats: MissionStats,
}

/// Inputs that would make the propagator produce nonsense.
#[derive(Debug, Clone, PartialEq)]
pub enum MissionError {
    NotFinite { field: &'static str },
    OutOfRange { field: &'static str, value: f64 },
    TooManyBurns { count: usize, max: usize },
    InvalidSolver(String),
    EmptyTrajectory,
    PropagationFailed,
}

impl std::fmt::Display for MissionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFinite { field } => write!(f, "{field} must be finite"),
            Self::OutOfRange { field, value } => {
                write!(f, "{field} is out of range: {value}")
            }
            Self::TooManyBurns { count, max } => {
                write!(f, "{count} burns supplied, at most {max} are supported")
            }
            Self::InvalidSolver(detail) => write!(f, "invalid solver configuration: {detail}"),
            Self::EmptyTrajectory => write!(f, "propagation produced no trajectory samples"),
            Self::PropagationFailed => write!(f, "propagation did not advance the state"),
        }
    }
}

impl std::error::Error for MissionError {}

/// Propagate one mission and return a viewer-ready trajectory.
pub fn run(request: &MissionRequest) -> Result<MissionTrajectory, MissionError> {
    for value in request.position_km {
        if !value.is_finite() {
            return Err(MissionError::NotFinite {
                field: "position_km",
            });
        }
    }
    for value in request.velocity_km_s {
        if !value.is_finite() {
            return Err(MissionError::NotFinite {
                field: "velocity_km_s",
            });
        }
    }

    let samples = request.samples.unwrap_or(DEFAULT_SAMPLES);
    if !(2..=MAX_SAMPLES).contains(&samples) {
        return Err(MissionError::OutOfRange {
            field: "samples",
            value: samples as f64,
        });
    }

    let duration = request.duration_s;
    if !duration.is_finite() || duration <= 0.0 || duration > MAX_DURATION_S {
        return Err(MissionError::OutOfRange {
            field: "duration_s",
            value: duration,
        });
    }

    let start_epoch = request.start_epoch_s.unwrap_or(0.0);
    if !start_epoch.is_finite() {
        return Err(MissionError::NotFinite {
            field: "start_epoch_s",
        });
    }

    let mut solver = request.solver.clone().unwrap_or_default();
    solver.epoch = start_epoch;
    let mut config = Config::from(solver);
    config
        .validate()
        .map_err(|e| MissionError::InvalidSolver(e.to_string()))?;

    let burns = request.burns.clone().unwrap_or_default();
    if config.burns.len() + burns.len() > MAX_BURNS {
        return Err(MissionError::TooManyBurns {
            count: config.burns.len() + burns.len(),
            max: MAX_BURNS,
        });
    }
    for burn in &burns {
        if !burn.t_s.is_finite() || burn.t_s < 0.0 || burn.t_s >= duration {
            return Err(MissionError::OutOfRange {
                field: "burns.t_s",
                value: burn.t_s,
            });
        }
        if burn.dv_m_s.iter().any(|v| !v.is_finite()) {
            return Err(MissionError::NotFinite {
                field: "burns.dv_m_s",
            });
        }
        config.burns.push(Burn {
            t: config.epoch + burn.t_s,
            dv: Vec3::new(burn.dv_m_s[0], burn.dv_m_s[1], burn.dv_m_s[2]),
        });
    }
    config
        .validate()
        .map_err(|e| MissionError::InvalidSolver(e.to_string()))?;

    let input_r = Vec3::new(
        request.position_km[0] * 1000.0,
        request.position_km[1] * 1000.0,
        request.position_km[2] * 1000.0,
    );
    let input_v = Vec3::new(
        request.velocity_km_s[0] * 1000.0,
        request.velocity_km_s[1] * 1000.0,
        request.velocity_km_s[2] * 1000.0,
    );

    let center = request.center;
    let earth_at_epoch = earth_moon_positions(config.epoch, &config).0;
    if center == MissionCenter::Earth && input_r.norm() < MIN_GEOCENTRIC_RADIUS_M {
        return Err(MissionError::OutOfRange {
            field: "position_km",
            value: input_r.norm() / 1000.0,
        });
    }

    let state = State {
        r: if center == MissionCenter::Earth {
            input_r.add(earth_at_epoch)
        } else {
            input_r
        },
        v: input_v,
    };

    let t_end = config.epoch + duration;
    let dt0 = (duration / samples as f64).clamp(0.25, 600.0);
    let trajectory = propagate_adaptive_detailed(state, t_end, dt0, RTOL, &config);
    if trajectory.states.len() < 2 {
        return Err(MissionError::PropagationFailed);
    }

    let ephemeris = AnalyticEphemeris;
    let mut points = Vec::with_capacity(samples);
    let mut times = Vec::with_capacity(samples);
    let mut times_absolute = Vec::with_capacity(samples);
    let mut states = Vec::with_capacity(samples);

    let mut segment = 0usize;
    for index in 0..samples {
        let fraction = index as f64 / (samples - 1) as f64;
        let relative_t = duration * fraction;
        let t = config.epoch + relative_t;

        while segment + 2 < trajectory.t.len() && trajectory.t[segment + 1] < t {
            segment += 1;
        }

        let sample = if t <= trajectory.t[0] {
            trajectory.states[0]
        } else if t >= *trajectory.t.last().expect("trajectory is never empty") {
            *trajectory.states.last().expect("trajectory is never empty")
        } else {
            let t0 = trajectory.t[segment];
            let t1 = trajectory.t[segment + 1];
            let h = t1 - t0;
            if h <= 0.0 {
                trajectory.states[segment]
            } else {
                hermite(
                    trajectory.states[segment],
                    trajectory.states[segment + 1],
                    h,
                    (t - t0) / h,
                )
            }
        };

        let earth = if center == MissionCenter::Earth {
            ephemeris.positions(t, &config).earth
        } else {
            Vec3::ZERO
        };
        let position = sample.r.sub(earth);
        let velocity = sample.v;

        points.push([
            position.x / 1000.0,
            position.y / 1000.0,
            position.z / 1000.0,
        ]);
        times.push(relative_t);
        times_absolute.push(t);
        states.push(StateSample {
            position_km: [
                position.x / 1000.0,
                position.y / 1000.0,
                position.z / 1000.0,
            ],
            velocity_km_s: [
                velocity.x / 1000.0,
                velocity.y / 1000.0,
                velocity.z / 1000.0,
            ],
            radius_km: position.norm() / 1000.0,
            speed_km_s: velocity.norm() / 1000.0,
        });
    }

    let initial_state = states
        .first()
        .cloned()
        .ok_or(MissionError::EmptyTrajectory)?;
    let final_state = states
        .last()
        .cloned()
        .ok_or(MissionError::EmptyTrajectory)?;

    let (min_radius_km, max_radius_km) = radius_bounds(&states);
    let (min_speed_km_s, max_speed_km_s) = speed_bounds(&states);

    Ok(MissionTrajectory {
        schema: SCHEMA,
        name: request
            .name
            .clone()
            .filter(|n| !n.trim().is_empty())
            .unwrap_or_else(|| "mission".to_string()),
        frame: center.frame_name(),
        units: "km",
        count: points.len(),
        points,
        times,
        times_absolute,
        duration_s: duration,
        start_epoch_s: config.epoch,
        initial_state,
        final_state,
        stats: MissionStats {
            samples: states.len(),
            min_radius_km,
            max_radius_km,
            final_radius_km: states.last().map(|s| s.radius_km).unwrap_or(0.0),
            min_speed_km_s,
            max_speed_km_s,
            accepted_steps: trajectory.accepted,
            rejected_steps: trajectory.rejected,
        },
    })
}

fn radius_bounds(states: &[StateSample]) -> (f64, f64) {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for state in states {
        min = min.min(state.radius_km);
        max = max.max(state.radius_km);
    }
    if min.is_finite() && max.is_finite() {
        (min, max)
    } else if let Some(first) = states.first() {
        (first.radius_km, first.radius_km)
    } else {
        (0.0, 0.0)
    }
}

fn speed_bounds(states: &[StateSample]) -> (f64, f64) {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for state in states {
        min = min.min(state.speed_km_s);
        max = max.max(state.speed_km_s);
    }
    if min.is_finite() && max.is_finite() {
        (min, max)
    } else if let Some(first) = states.first() {
        (first.speed_km_s, first.speed_km_s)
    } else {
        (0.0, 0.0)
    }
}

/// Cubic Hermite interpolation between two states, parameterised by
/// `w ∈ [0, 1]` over a segment of length `h` seconds.
fn hermite(s0: State, s1: State, h: f64, w: f64) -> State {
    let w = w.clamp(0.0, 1.0);
    let w2 = w * w;
    let w3 = w2 * w;

    let h00 = 2.0 * w3 - 3.0 * w2 + 1.0;
    let h10 = w3 - 2.0 * w2 + w;
    let h01 = -2.0 * w3 + 3.0 * w2;
    let h11 = w3 - w2;

    let r =
        s0.r.scale(h00)
            .add(s0.v.scale(h10 * h))
            .add(s1.r.scale(h01))
            .add(s1.v.scale(h11 * h));

    let d00 = 6.0 * w2 - 6.0 * w;
    let d10 = 3.0 * w2 - 4.0 * w + 1.0;
    let d01 = -6.0 * w2 + 6.0 * w;
    let d11 = 3.0 * w2 - 2.0 * w;

    let v =
        s0.r.scale(d00 / h)
            .add(s0.v.scale(d10))
            .add(s1.r.scale(d01 / h))
            .add(s1.v.scale(d11));

    State { r, v }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LEO_RADIUS_KM: f64 = 6_778.137;
    const LEO_SPEED_KM_S: f64 = 7.668_6;

    fn leo() -> MissionRequest {
        MissionRequest {
            position_km: [LEO_RADIUS_KM, 0.0, 0.0],
            velocity_km_s: [0.0, LEO_SPEED_KM_S, 0.0],
            duration_s: 5_550.0,
            samples: Some(240),
            center: MissionCenter::Earth,
            start_epoch_s: Some(0.0),
            burns: None,
            solver: None,
            name: Some("LEO test".to_string()),
        }
    }

    #[test]
    fn earth_centered_orbit_starts_at_the_requested_state() {
        let trajectory = run(&leo()).expect("LEO run");
        assert_eq!(trajectory.count, 240);
        assert_eq!(trajectory.points.len(), 240);
        assert_eq!(trajectory.times.len(), 240);
        assert!((trajectory.points[0][0] - LEO_RADIUS_KM).abs() < 1e-6);
        assert!(trajectory.points[0][1].abs() < 1e-6);
        assert!(trajectory.points[0][2].abs() < 1e-6);
        assert!((trajectory.times[0]).abs() < 1e-12);
        assert!((trajectory.times[239] - 5_550.0).abs() < 1e-6);
        assert_eq!(trajectory.frame, "earth-centered-inertial");
    }

    #[test]
    fn circular_orbit_stays_round() {
        let trajectory = run(&leo()).expect("LEO run");
        let stats = &trajectory.stats;
        assert!(
            stats.min_radius_km > LEO_RADIUS_KM * 0.97,
            "perigee {} km",
            stats.min_radius_km
        );
        assert!(
            stats.max_radius_km < LEO_RADIUS_KM * 1.03,
            "apogee {} km",
            stats.max_radius_km
        );
    }

    #[test]
    fn prograde_burn_raises_the_apogee() {
        let mut request = leo();
        request.duration_s = 2_000.0;
        request.samples = Some(200);
        let coast = run(&request).expect("coast run");

        request.burns = Some(vec![MissionBurn {
            t_s: 600.0,
            dv_m_s: [0.0, 1_000.0, 0.0],
        }]);
        let burned = run(&request).expect("burn run");

        assert!(
            burned.stats.max_radius_km > coast.stats.max_radius_km,
            "burn should widen the orbit: {} vs {}",
            burned.stats.max_radius_km,
            coast.stats.max_radius_km
        );
        assert!(burned.stats.max_radius_km > LEO_RADIUS_KM * 1.1);
    }

    #[test]
    fn translunar_injection_reaches_lunar_distance() {
        let mut request = leo();
        request.velocity_km_s = [0.0, 10.75, 0.0];
        request.duration_s = 3.0 * 86_400.0;
        request.samples = Some(600);
        let trajectory = run(&request).expect("TLI run");
        assert!(
            trajectory.stats.max_radius_km > 300_000.0,
            "apogee only {} km",
            trajectory.stats.max_radius_km
        );
        assert!(trajectory.stats.max_radius_km.is_finite());
    }

    #[test]
    fn barycentric_input_is_used_verbatim() {
        let mut request = leo();
        request.center = MissionCenter::Barycentric;
        request.position_km = [100_000.0, 0.0, 0.0];
        request.duration_s = 60.0;
        request.samples = Some(4);
        let trajectory = run(&request).expect("barycentric run");
        assert!((trajectory.points[0][0] - 100_000.0).abs() < 1e-6);
        assert!(trajectory.points[0][1].abs() < 1e-6);
        assert_eq!(trajectory.frame, "earth-moon-barycentric");
    }

    #[test]
    fn earth_centered_start_cancels_the_earth_offset_at_any_epoch() {
        for epoch in [0.0, 12_345.0, 864_000.0] {
            let mut request = leo();
            request.start_epoch_s = Some(epoch);
            request.duration_s = 10.0;
            request.samples = Some(3);
            let trajectory = run(&request).expect("epoch run");
            assert!(
                (trajectory.points[0][0] - LEO_RADIUS_KM).abs() < 1e-6,
                "epoch {epoch}: x = {}",
                trajectory.points[0][0]
            );
            assert!(trajectory.points[0][1].abs() < 1e-6);
            assert!(trajectory.points[0][2].abs() < 1e-6);
        }
    }

    #[test]
    fn rejects_invalid_inputs() {
        let mut request = leo();
        request.samples = Some(1);
        assert!(matches!(
            run(&request),
            Err(MissionError::OutOfRange {
                field: "samples",
                ..
            })
        ));

        let mut request = leo();
        request.duration_s = 0.0;
        assert!(matches!(
            run(&request),
            Err(MissionError::OutOfRange {
                field: "duration_s",
                ..
            })
        ));

        let mut request = leo();
        request.position_km = [f64::NAN, 0.0, 0.0];
        assert!(matches!(
            run(&request),
            Err(MissionError::NotFinite {
                field: "position_km"
            })
        ));

        let mut request = leo();
        request.burns = Some(
            (0..=MAX_BURNS)
                .map(|index| MissionBurn {
                    t_s: index as f64 * 10.0,
                    dv_m_s: [0.0, 1.0, 0.0],
                })
                .collect(),
        );
        assert!(matches!(
            run(&request),
            Err(MissionError::TooManyBurns { .. })
        ));
    }

    #[test]
    fn samples_are_uniform_and_the_final_state_is_reported() {
        let mut request = leo();
        request.duration_s = 1_000.0;
        request.samples = Some(11);
        let trajectory = run(&request).expect("run");
        for pair in trajectory.times.windows(2) {
            let step = pair[1] - pair[0];
            assert!((step - 100.0).abs() < 1e-9, "step {step}");
        }
        assert!(
            (trajectory.final_state.radius_km - trajectory.stats.final_radius_km).abs() < 1e-12
        );
        assert_eq!(
            trajectory.times_absolute.last().copied(),
            Some(trajectory.start_epoch_s + trajectory.duration_s)
        );
    }
}
