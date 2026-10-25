//! Reference trajectory → residual table.
//!
//! Usage:
//!
//! ```text
//! cargo run -p residual --example residual_pipeline -- reference.csv out.csv [mission.json]
//!     [--ephemeris analytic|tabulated] [--compare]
//! ```
//!
//! The reference CSV uses the canonical `t,x,y,z,vx,vy,vz` header.  The
//! propagator starts from the first reference state and uses either the
//! mission's solver settings or the reference cadence.  Burn windows declared
//! in the mission are masked out of the output table.
//!
//! # Ephemeris
//!
//! `--ephemeris analytic` (the default) keeps the circular Earth-Moon model
//! that matches the CUDA kernel.  Its residual therefore contains both
//! integrator error and circular-model error.
//!
//! `--ephemeris tabulated` builds a [`TabulatedEphemeris`] from the reference
//! Moon track: the Earth position is recovered from the two-body barycentre
//! condition, so the propagator sees the measured Earth-Moon geometry.  The
//! remaining residual is integrator error plus genuinely missing physics
//! (the Sun, non-gravitational forces), which is what should be handed to the
//! PINN.
//!
//! `--sun <csv>` attaches a Sun track (same solver frame, `t,x,y,z` columns)
//! to the tabulated ephemeris and enables the (tidal) third-body term.  The
//! default `mu_sun` is zero, so the Sun contributes nothing unless a track
//! is supplied.
//!
//! The tabulated model is only meaningful with a reference that keeps the real
//! Earth-Moon separation (`scripts/convert_reference.py --preserve-distance`).
//! The default reference is rescaled onto the circular model's radius, so its
//! geometry is *designed* for `analytic`; feeding it to `tabulated` makes the
//! residual worse, and the example warns when it detects a constant-radius arc.

use std::env;
use std::error::Error;
use std::fs;

use dynamics::{propagate, propagate_with, Config, State, TabulatedEphemeris, K_MU_SUN};
use ingest::Telemetry;
use residual::{residual_series, ResidualRow};
use schema::Mission;

/// Which force-field geometry drives the propagation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EphemerisModel {
    Analytic,
    Tabulated,
}

impl EphemerisModel {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "analytic" | "circular" => Ok(Self::Analytic),
            "tabulated" | "table" | "reference" => Ok(Self::Tabulated),
            other => Err(format!(
                "unknown ephemeris '{other}' (expected 'analytic' or 'tabulated')"
            )),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Analytic => "analytic (circular)",
            Self::Tabulated => "tabulated (reference geometry)",
        }
    }
}

fn print_usage() {
    eprintln!(
        "usage: residual_pipeline <reference.csv> <out.csv> [mission.json]\n\
                [--ephemeris analytic|tabulated] [--sun <sun.csv>] [--compare]"
    );
}

/// RMS position error (metres) and velocity error (m/s).
fn rms(rows: &[ResidualRow]) -> (f64, f64) {
    if rows.is_empty() {
        return (0.0, 0.0);
    }
    let n = rows.len() as f64;
    let position = (rows.iter().map(|r| r.dr.norm().powi(2)).sum::<f64>() / n).sqrt();
    let velocity = (rows.iter().map(|r| r.dv.norm().powi(2)).sum::<f64>() / n).sqrt();
    (position, velocity)
}

/// Load a Sun track from a CSV with at least `t,x,y,z` columns (metres).
fn load_sun_track(path: &str) -> Result<Vec<(f64, dynamics::Vec3)>, Box<dyn Error>> {
    let text = std::fs::read_to_string(path)?;
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let header = lines
        .next()
        .ok_or_else(|| format!("empty sun file: {path}"))?;
    let columns: Vec<&str> = header.split(',').map(str::trim).collect();
    let ti = columns
        .iter()
        .position(|c| *c == "t")
        .ok_or_else(|| format!("sun CSV missing 't' column: {header}"))?;
    let xi = columns
        .iter()
        .position(|c| *c == "x")
        .ok_or_else(|| format!("sun CSV missing 'x' column: {header}"))?;
    let yi = columns
        .iter()
        .position(|c| *c == "y")
        .ok_or_else(|| format!("sun CSV missing 'y' column: {header}"))?;
    let zi = columns
        .iter()
        .position(|c| *c == "z")
        .ok_or_else(|| format!("sun CSV missing 'z' column: {header}"))?;

    let mut out = Vec::new();
    for (lineno, line) in lines.enumerate() {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() <= ti.max(xi).max(yi).max(zi) {
            return Err(format!("short row at line {}", lineno + 2).into());
        }
        let t: f64 = fields[ti].trim().parse()?;
        let x: f64 = fields[xi].trim().parse()?;
        let y: f64 = fields[yi].trim().parse()?;
        let z: f64 = fields[zi].trim().parse()?;
        out.push((t, dynamics::Vec3::new(x, y, z)));
    }
    if out.is_empty() {
        return Err(format!("no data rows in {path}").into());
    }
    Ok(out)
}
///
/// `convert_reference.py` rescales every sample onto the propagator's nominal
/// Earth-Moon distance by default.  Such an arc is circular by construction and
/// is meant for the analytic model; the tabulated model wants the measured
/// geometry instead.
fn has_constant_radius(positions: &[dynamics::Vec3]) -> bool {
    if positions.len() < 2 {
        return false;
    }
    let radii: Vec<f64> = positions.iter().map(|p| p.norm()).collect();
    let mean = radii.iter().sum::<f64>() / radii.len() as f64;
    if !(mean.is_finite() && mean > 0.0) {
        return false;
    }
    let max_dev = radii
        .iter()
        .map(|r| (r - mean).abs())
        .fold(0.0_f64, f64::max);
    max_dev / mean < 1e-4
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut positional = Vec::new();
    let mut model = EphemerisModel::Analytic;
    let mut compare = false;
    let mut sun_path: Option<String> = None;

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--ephemeris" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--ephemeris needs a value".to_string())?;
                model = EphemerisModel::parse(&value)?;
            }
            "--sun" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--sun needs a path".to_string())?;
                sun_path = Some(value);
            }
            "--compare" => compare = true,
            "--help" | "-h" => {
                print_usage();
                return Ok(());
            }
            other if other.starts_with("--") => {
                return Err(format!("unknown flag '{other}'").into());
            }
            other => positional.push(other.to_string()),
        }
    }

    if sun_path.is_some() && model == EphemerisModel::Analytic {
        model = EphemerisModel::Tabulated;
        eprintln!("note: --sun implies --ephemeris tabulated");
    }

    if positional.len() < 2 || positional.len() > 3 {
        print_usage();
        std::process::exit(2);
    }
    let reference_path = &positional[0];
    let output_path = &positional[1];

    let telemetry = Telemetry::load(reference_path)?;
    let samples = telemetry.samples();
    let start_time = samples.first().expect("non-empty").t;
    let end_time = samples.last().expect("non-empty").t;

    let (mut config, mission) = match positional.get(2) {
        Some(path) => {
            let mission = Mission::load(path)?;
            (Config::from(mission.solver.clone()), Some(mission))
        }
        None => (Config::default(), None),
    };
    config.epoch = start_time;

    // The reference is the Moon itself, so it must not be attracted by its own
    // gravity.  The *table* still needs the physical masses to recover the
    // Earth track, so build it before disabling the Moon term.
    let (mu_earth, mu_moon) = (config.mu_earth, config.mu_moon);

    // Prefer the mission cadence; otherwise match the reference spacing.
    let dt = match &mission {
        Some(m) => m.solver.dt,
        None if samples.len() > 1 => samples[1].t - samples[0].t,
        None => config.dt,
    };
    if !dt.is_finite() || dt <= 0.0 {
        return Err("propagation step must be positive".into());
    }

    let steps = ((end_time - start_time) / dt).ceil() as usize;

    let mut table = if model == EphemerisModel::Tabulated || compare || sun_path.is_some() {
        let table = TabulatedEphemeris::from_moon_track(
            samples.iter().map(|s| (s.t, s.state.r)),
            mu_earth,
            mu_moon,
        )?;
        // A partitioned integration would silently clamp outside the arc, so
        // reject a reference that the table cannot cover.
        table.check_span(start_time, end_time)?;
        Some(table)
    } else {
        None
    };

    if let Some(path) = &sun_path {
        let track = load_sun_track(path)?;
        let attached = table
            .take()
            .expect("sun implies a tabulated table")
            .with_sun_track(track)?;
        println!("sun: attached {} samples from {}", attached.len(), path);
        table = Some(attached);
        config.mu_sun = K_MU_SUN;
    }

    if model == EphemerisModel::Tabulated {
        let positions: Vec<dynamics::Vec3> = samples.iter().map(|s| s.state.r).collect();
        if has_constant_radius(&positions) {
            eprintln!(
                "warning: reference radius is constant to 1e-4, so this looks like the \
                 circular-model normalized arc. Regenerate it with \
                 `scripts/convert_reference.py --preserve-distance` before using the \
                 tabulated ephemeris."
            );
        }
    }

    config.mu_moon = 0.0;

    let start_state: State = samples[0].state;
    let reference: Vec<(f64, State)> = samples.iter().map(|s| (s.t, s.state)).collect();

    let propagate_model = |choice: EphemerisModel| -> Vec<State> {
        match choice {
            EphemerisModel::Analytic => propagate(start_state, steps, dt, &config),
            EphemerisModel::Tabulated => {
                let table = table.as_ref().expect("tabulated model built a table");
                propagate_with(start_state, steps, dt, &config, table)
            }
        }
    };

    let trajectory = propagate_model(model);
    let rows = residual_series(&reference, &trajectory, config.epoch, dt);

    match model {
        EphemerisModel::Analytic => {
            println!("ephemeris: analytic circular Earth-Moon model");
        }
        EphemerisModel::Tabulated => {
            let table = table.as_ref().expect("tabulated model built a table");
            let (first, last) = table.time_span();
            println!(
                "ephemeris: tabulated reference geometry ({} samples, t = [{first}, {last}] s){}",
                table.len(),
                if table.has_sun() { " + Sun" } else { "" }
            );
        }
    }

    if compare {
        println!(
            "ephemeris comparison over {} samples (dt = {dt} s):",
            reference.len()
        );
        let (position, velocity) = rms(&rows);
        println!(
            "  {:<32} rms |dr| = {:>14.3} m   rms |dv| = {:>10.4} m/s",
            model.label(),
            position,
            velocity
        );
        let other = match model {
            EphemerisModel::Analytic => EphemerisModel::Tabulated,
            EphemerisModel::Tabulated => EphemerisModel::Analytic,
        };
        let other_rows = residual_series(&reference, &propagate_model(other), config.epoch, dt);
        let (position, velocity) = rms(&other_rows);
        println!(
            "  {:<32} rms |dr| = {:>14.3} m   rms |dv| = {:>10.4} m/s",
            other.label(),
            position,
            velocity
        );
    }

    let mut csv = String::from("t,drx,dry,drz,dvx,dvy,dvz\n");
    let mut written = 0usize;
    for row in &rows {
        if let Some(m) = &mission {
            if m.burns.iter().any(|burn| burn.covers(row.t)) {
                continue;
            }
        }
        csv.push_str(&format!(
            "{},{},{},{},{},{},{}\n",
            row.t, row.dr.x, row.dr.y, row.dr.z, row.dv.x, row.dv.y, row.dv.z
        ));
        written += 1;
    }
    fs::write(output_path, csv)?;

    let masked = rows.len().saturating_sub(written);
    println!("wrote {written} residual rows ({masked} masked) to {output_path}");
    if let Some(m) = &mission {
        println!(
            "mission '{}', frame {}, masked fraction {:.3}",
            m.name,
            m.frame,
            m.masked_fraction(&telemetry.times())
        );
    }
    Ok(())
}
