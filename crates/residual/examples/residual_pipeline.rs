//! Reference trajectory → residual table.
//!
//! Usage:
//!
//! ```text
//! cargo run -p residual --example residual_pipeline -- reference.csv out.csv [mission.json]
//! ```
//!
//! The reference CSV uses the canonical `t,x,y,z,vx,vy,vz` header.  The
//! propagator starts from the first reference state and uses either the
//! mission's solver settings or the reference cadence.  Burn windows declared
//! in the mission are masked out of the output table.

use std::env;
use std::error::Error;
use std::fs;

use dynamics::{propagate, Config, State};
use ingest::Telemetry;
use residual::residual_series;
use schema::Mission;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: residual_pipeline <reference.csv> <out.csv> [mission.json]");
        std::process::exit(2);
    }

    let reference_path = &args[1];
    let output_path = &args[2];

    let telemetry = Telemetry::load(reference_path)?;
    let samples = telemetry.samples();
    let start_time = samples.first().expect("non-empty").t;

    let (mut config, mission) = match args.get(3) {
        Some(path) => {
            let mission = Mission::load(path)?;
            (Config::from(mission.solver.clone()), Some(mission))
        }
        None => (Config::default(), None),
    };
    config.epoch = start_time;

    // Prefer the mission cadence; otherwise match the reference spacing.
    let dt = match &mission {
        Some(m) => m.solver.dt,
        None if samples.len() > 1 => samples[1].t - samples[0].t,
        None => config.dt,
    };
    if !dt.is_finite() || dt <= 0.0 {
        return Err("propagation step must be positive".into());
    }

    let end_time = samples.last().expect("non-empty").t;
    let steps = ((end_time - start_time) / dt).ceil() as usize;

    let start_state: State = samples[0].state;
    let trajectory = propagate(start_state, steps, dt, &config);

    let reference: Vec<(f64, State)> = samples.iter().map(|s| (s.t, s.state)).collect();
    let rows = residual_series(&reference, &trajectory, config.epoch, dt);

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
