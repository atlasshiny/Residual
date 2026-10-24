//! Enqueue a Cosmos job into a directory-backed queue.
//!
//! Usage:
//!
//! ```text
//! cargo run -p ipc --example enqueue_job -- <jobs-root> <job-id> <mode> [inputs...]
//! ```
//!
//! `mode` is `transfer`, `inverse_dynamics`, or `forward_dynamics` (hyphens are
//! accepted too). Each remaining argument is recorded as an input path.
//!
//! The result is `<jobs-root>/<job-id>/job.json`:
//!
//! ```json
//! {"id":"demo","mode":"inverse_dynamics","inputs":["clip.mp4"],"seed":0,
//!  "params":null,"created_at_s":1700000000}
//! ```
//!
//! Run `scripts/mock_cosmos_worker.py <jobs-root>/<job-id>` to produce
//! `sample_outputs.json` and `manifest.json` without a GPU.

use std::env;
use std::error::Error;

use ipc::{Job, JobMode, JobStore};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() < 3 {
        eprintln!("usage: enqueue_job <jobs-root> <job-id> <mode> [inputs...]");
        eprintln!("  mode: transfer | inverse_dynamics | forward_dynamics");
        std::process::exit(2);
    }

    let jobs_root = &args[0];
    let job_id = &args[1];
    let mode = parse_mode(&args[2])?;

    let mut job = Job::new(job_id, mode);
    for input in &args[3..] {
        job = job.with_input(input.clone());
    }

    let store = JobStore::new(jobs_root);
    let path = store.enqueue(&job)?;
    println!(
        "enqueued {} (mode={}) -> {}",
        job.id,
        job.mode.as_str(),
        path.display()
    );
    if store.pending()?.iter().any(|pending| pending.id == job.id) {
        println!(
            "next: python scripts/mock_cosmos_worker.py {}",
            store.job_dir(&job.id)?.display()
        );
    }
    Ok(())
}

fn parse_mode(value: &str) -> Result<JobMode, Box<dyn Error>> {
    match value.replace('-', "_").as_str() {
        "transfer" => Ok(JobMode::Transfer),
        "inverse_dynamics" => Ok(JobMode::InverseDynamics),
        "forward_dynamics" => Ok(JobMode::ForwardDynamics),
        _ => Err(format!(
            "unknown mode '{value}' (expected transfer, inverse_dynamics, or forward_dynamics)"
        )
        .into()),
    }
}
