//! Mission, vehicle, and solver schemas for the Residual pipeline.
//!
//! The types here are the on-disk contract shared by the Rust core, the KAN
//! training script, and the Cosmos job runners.  Everything is plain serde so
//! a mission can be written as JSON or YAML without bespoke code.
//!
//! `SolverConfig` deliberately mirrors [`dynamics::Config`] instead of deriving
//! serde on the physics crate: the propagator stays dependency-free (which
//! keeps it cheap to use from CUDA-side tests) while the schema crate owns the
//! serialised representation.

use std::error::Error;
use std::fmt;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use dynamics::Config as DynConfig;

// ---------------------------------------------------------------------------
// Solver configuration
// ---------------------------------------------------------------------------

/// Serialisable form of the propagator configuration.
///
/// Field names intentionally match the CUDA `MbsConfig` layout so the same
/// JSON can be handed to either implementation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SolverConfig {
    pub dt: f64,
    pub epoch: f64,
    pub mu_earth: f64,
    pub mu_moon: f64,
    pub earth_radius: f64,
    pub j2: f64,
    pub moon_distance: f64,
    pub moon_period: f64,
    pub moon_phase: f64,
    pub softening: f64,
    pub perturbation: i32,
    #[serde(default = "default_spin_axis")]
    pub spin_axis: (f64, f64, f64),
    #[serde(default)]
    pub burns: Vec<dynamics::Burn>,
}

fn default_spin_axis() -> (f64, f64, f64) {
    (0.0, 0.0, 1.0)
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self::from(DynConfig::default())
    }
}

impl From<DynConfig> for SolverConfig {
    fn from(c: DynConfig) -> Self {
        Self {
            dt: c.dt,
            epoch: c.epoch,
            mu_earth: c.mu_earth,
            mu_moon: c.mu_moon,
            earth_radius: c.earth_radius,
            j2: c.j2,
            moon_distance: c.moon_distance,
            moon_period: c.moon_period,
            moon_phase: c.moon_phase,
            softening: c.softening,
            perturbation: c.perturbation,
            spin_axis: (c.spin_axis.x, c.spin_axis.y, c.spin_axis.z),
            burns: c.burns.clone(),
        }
    }
}

impl From<SolverConfig> for DynConfig {
    fn from(c: SolverConfig) -> Self {
        DynConfig {
            dt: c.dt,
            epoch: c.epoch,
            mu_earth: c.mu_earth,
            mu_moon: c.mu_moon,
            earth_radius: c.earth_radius,
            j2: c.j2,
            moon_distance: c.moon_distance,
            moon_period: c.moon_period,
            moon_phase: c.moon_phase,
            softening: c.softening,
            perturbation: c.perturbation,
            spin_axis: dynamics::Vec3::new(c.spin_axis.0, c.spin_axis.1, c.spin_axis.2),
            burns: c.burns.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Vehicle and burns
// ---------------------------------------------------------------------------

/// Physical properties used by optional non-gravitational perturbations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Vehicle {
    pub name: String,
    /// Wet mass, kilograms.
    pub mass_kg: f64,
    /// Cross-sectional area used for drag, m^2.
    pub drag_area_m2: f64,
    /// Drag coefficient (dimensionless).
    pub drag_coefficient: f64,
    /// Cross-sectional area used for solar radiation pressure, m^2.
    pub srp_area_m2: f64,
    /// Reflectivity coefficient (1.0 = fully absorbing, 2.0 = mirror).
    pub reflectivity: f64,
}

impl Default for Vehicle {
    fn default() -> Self {
        Self {
            name: "Orion".to_string(),
            mass_kg: 26_500.0,
            drag_area_m2: 0.0,
            drag_coefficient: 2.2,
            srp_area_m2: 0.0,
            reflectivity: 1.3,
        }
    }
}

/// Half-open time interval `[start_s, end_s)` during which the physics model is
/// unreliable (a manoeuvre, attitude slew, or unmodelled thrust).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BurnWindow {
    pub label: String,
    pub start_s: f64,
    pub end_s: f64,
}

impl BurnWindow {
    pub fn new(label: impl Into<String>, start_s: f64, end_s: f64) -> Self {
        Self {
            label: label.into(),
            start_s,
            end_s,
        }
    }

    /// True when `t` falls inside the window.
    pub fn covers(&self, t: f64) -> bool {
        t >= self.start_s && t < self.end_s
    }

    pub fn is_valid(&self) -> bool {
        self.start_s.is_finite() && self.end_s.is_finite() && self.end_s >= self.start_s
    }
}

// ---------------------------------------------------------------------------
// Mission
// ---------------------------------------------------------------------------

/// Top-level mission description.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Mission {
    pub name: String,
    /// Inertial frame the states are expressed in (`eci`, `eme2000`, ...).
    pub frame: String,
    /// Epoch of the first sample, seconds.
    pub epoch_s: f64,
    pub vehicle: Vehicle,
    pub solver: SolverConfig,
    pub burns: Vec<BurnWindow>,
}

impl Default for Mission {
    fn default() -> Self {
        Self {
            name: "Artemis".to_string(),
            frame: "eci".to_string(),
            epoch_s: 0.0,
            vehicle: Vehicle::default(),
            solver: SolverConfig::default(),
            burns: Vec::new(),
        }
    }
}

impl Mission {
    /// Read a mission from JSON.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, SchemaError> {
        let path = path.as_ref();
        let text = fs::read_to_string(path)
            .map_err(|e| SchemaError::Io(path.display().to_string(), e.to_string()))?;
        Self::from_json(&text)
    }

    pub fn from_json(text: &str) -> Result<Self, SchemaError> {
        let mission: Mission =
            serde_json::from_str(text).map_err(|e| SchemaError::Json(e.to_string()))?;
        mission.validate()?;
        Ok(mission)
    }

    /// Write the mission as pretty-printed JSON.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), SchemaError> {
        let path = path.as_ref();
        let text = self
            .to_json()
            .map_err(|e| SchemaError::Json(e.to_string()))?;
        fs::write(path, text)
            .map_err(|e| SchemaError::Io(path.display().to_string(), e.to_string()))
    }

    pub fn to_json(&self) -> Result<String, SchemaError> {
        serde_json::to_string_pretty(self).map_err(|e| SchemaError::Json(e.to_string()))
    }

    /// Reject missions that would silently produce nonsense downstream.
    pub fn validate(&self) -> Result<(), SchemaError> {
        if self.name.trim().is_empty() {
            return Err(SchemaError::Invalid("mission name is empty".into()));
        }
        if !self.epoch_s.is_finite() {
            return Err(SchemaError::Invalid("epoch_s is not finite".into()));
        }
        if !self.solver.dt.is_finite() || self.solver.dt <= 0.0 {
            return Err(SchemaError::Invalid("solver.dt must be positive".into()));
        }
        if self.solver.mu_earth < 0.0 || self.solver.mu_moon < 0.0 {
            return Err(SchemaError::Invalid(
                "gravitational parameters must be non-negative".into(),
            ));
        }
        for burn in &self.burns {
            if !burn.is_valid() {
                return Err(SchemaError::Invalid(format!(
                    "burn window '{}' has invalid bounds",
                    burn.label
                )));
            }
        }
        Ok(())
    }

    /// Per-sample keep mask: `true` means the sample is usable for training.
    ///
    /// Burn samples are dropped, matching the Python residual-table builder.
    pub fn burn_mask(&self, times: &[f64]) -> Vec<bool> {
        times
            .iter()
            .map(|&t| !self.burns.iter().any(|b| b.covers(t)))
            .collect()
    }

    /// Fraction of samples that fall inside a burn window.
    pub fn masked_fraction(&self, times: &[f64]) -> f64 {
        if times.is_empty() {
            return 0.0;
        }
        let masked = self.burn_mask(times).iter().filter(|keep| !**keep).count();
        masked as f64 / times.len() as f64
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum SchemaError {
    Io(String, String),
    Json(String),
    Invalid(String),
}

impl fmt::Display for SchemaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SchemaError::Io(path, err) => write!(f, "io error for {path}: {err}"),
            SchemaError::Json(err) => write!(f, "invalid JSON: {err}"),
            SchemaError::Invalid(msg) => write!(f, "invalid mission: {msg}"),
        }
    }
}

impl Error for SchemaError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solver_config_round_trips() {
        let dyn_cfg = DynConfig::default();
        let serde_cfg = SolverConfig::from(dyn_cfg.clone());
        let back: DynConfig = serde_cfg.into();
        assert_eq!(dyn_cfg, back);
    }

    #[test]
    fn mission_json_round_trip() {
        let mut mission = Mission {
            name: "Artemis II".into(),
            ..Mission::default()
        };
        mission.burns.push(BurnWindow::new("TLI", 100.0, 200.0));
        let json = mission.to_json().unwrap();
        let parsed = Mission::from_json(&json).unwrap();
        assert_eq!(parsed, mission);
    }

    #[test]
    fn burn_mask_drops_windows() {
        let mission = Mission {
            burns: vec![BurnWindow::new("TLI", 10.0, 20.0)],
            ..Mission::default()
        };
        let times = vec![0.0, 10.0, 15.0, 20.0];
        assert_eq!(mission.burn_mask(&times), vec![true, false, false, true]);
        assert_eq!(mission.masked_fraction(&times), 0.5);
    }

    #[test]
    fn rejects_zero_dt() {
        let mission = Mission {
            solver: SolverConfig {
                dt: 0.0,
                ..SolverConfig::default()
            },
            ..Mission::default()
        };
        assert!(matches!(mission.validate(), Err(SchemaError::Invalid(_))));
    }

    #[test]
    fn missing_fields_use_defaults() {
        let mission = Mission::from_json(r#"{"name": "minimal"}"#).unwrap();
        assert_eq!(mission.vehicle.name, "Orion");
        assert_eq!(mission.solver.dt, SolverConfig::default().dt);
    }
}
