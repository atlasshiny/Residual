//! Evaluate a completed Cosmos job: integrate the recovered `sample_outputs.json`
//! action rows, align them against a reference trajectory with a Sim(3)
//! (Umeyama) fit, and write a metrics JSON document.
//!
//! Usage:
//!
//! ```text
//! cargo run -p evaluator --example evaluate_job -- <job-dir> [options]
//! ```
//!
//! Reference resolution order:
//!
//! 1. `--reference <path>`: another `sample_outputs.json` (for example the
//!    commanded path from job A).  Its `first_pose` + `poses` rows are the
//!    reference trajectory.
//! 2. `<job-dir>/job.json`: the manifest's ground-truth `first_pose` + `actions`
//!    rows (written by `posebuilder`).
//!
//! The recovered sequence is always `<job-dir>/sample_outputs.json` (override
//! with `--sample`).  Metrics are written to `<job-dir>/metrics.json` unless
//! `--out` is given.
//!
//! Options:
//!
//! ```text
//! --reference PATH   reference sample_outputs.json (default: job.json actions)
//! --job-json PATH    job manifest (default: <job-dir>/job.json)
//! --sample PATH      recovered outputs (default: <job-dir>/sample_outputs.json)
//! --out PATH         metrics output (default: <job-dir>/metrics.json)
//! --k K              override kilometres per Cosmos-metre (uses both files)
//! ```
//!
//! See `docs/cosmos-contract.md` §4 for the file format and §9 for the metrics.

use std::env;
use std::error::Error;
use std::path::PathBuf;

use evaluator::{build_report, write_metrics, JobManifest, MetricsReport, SampleOutputs};

fn main() -> Result<(), Box<dyn Error>> {
    let args = match Args::parse(env::args().skip(1).collect()) {
        Ok(Some(args)) => args,
        Ok(None) => return Ok(()),
        Err(err) => {
            eprintln!("error: {err}\n\n{}", usage());
            std::process::exit(2);
        }
    };

    match run(&args) {
        Ok(report) => {
            println!(
                "job {}: RRE {:.6} deg | RTE {:.6} m | ATE {:.6} m \
                 ({} frames, k={} km/cosmos-m, aligned={}) -> {}",
                if report.job_id.is_empty() {
                    "<unknown>"
                } else {
                    &report.job_id
                },
                report.rre_degrees,
                report.rte_metres,
                report.ate_metres,
                report.frames,
                report.scale_k_km_per_m,
                report.aligned,
                args.metrics_path().display(),
            );
            Ok(())
        }
        Err(err) => Err(err),
    }
}

fn run(args: &Args) -> Result<MetricsReport, Box<dyn Error>> {
    let sample_path = args.sample_path();
    let outputs = SampleOutputs::load(&sample_path).map_err(|e| {
        format!("cannot read recovered outputs {}: {e}", sample_path.display())
    })?;
    if !outputs.has_poses() {
        return Err(format!(
            "{} has no poses (mode '{}'); nothing to evaluate for transfer jobs",
            sample_path.display(),
            if outputs.mode.is_empty() {
                "<unknown>"
            } else {
                &outputs.mode
            }
        )
        .into());
    }

    // job.json is optional: needed only when it supplies the reference.
    let job_path = args.job_json.clone().unwrap_or_else(|| args.job_dir.join("job.json"));
    let job: Option<JobManifest> = if job_path.is_file() {
        Some(JobManifest::load(&job_path).map_err(|e| {
            format!("cannot read job manifest {}: {e}", job_path.display())
        })?)
    } else {
        None
    };
    let fallback_first = job
        .as_ref()
        .and_then(|j| j.first_pose.as_ref())
        .map(|pose| pose.to_pose());

    // Recovered sequence (may borrow the job's first_pose as a fallback).
    let mut recovered = outputs
        .sequence(fallback_first)
        .map_err(|e| format!("cannot reconstruct recovered trajectory: {e}"))?;

    // Reference sequence.
    let (mut reference, reference_label) = match &args.reference {
        Some(path) => {
            let ref_outputs = SampleOutputs::load(path)
                .map_err(|e| format!("cannot read reference {}: {e}", path.display()))?;
            let seq = ref_outputs
                .sequence(fallback_first)
                .map_err(|e| format!("cannot reconstruct reference trajectory: {e}"))?;
            (seq, path.display().to_string())
        }
        None => {
            let job = job.as_ref().ok_or_else(|| {
                format!(
                    "no reference: {} is missing or has no actions; pass --reference <sample_outputs.json>",
                    job_path.display()
                )
            })?;
            if !job.has_actions() {
                return Err(format!(
                    "{} has no first_pose+actions reference; pass --reference <sample_outputs.json>",
                    job_path.display()
                )
                .into());
            }
            let seq = job
                .sequence()
                .map_err(|e| format!("cannot reconstruct reference trajectory: {e}"))?;
            (seq, format!("{} (actions)", job_path.display()))
        }
    };

    if let Some(k) = args.k {
        recovered.km_per_unit = k;
        reference.km_per_unit = k;
    }

    let report = build_report(
        &outputs,
        job.as_ref(),
        &recovered,
        &reference,
        &reference_label,
        &sample_path.display().to_string(),
    )
    .map_err(|e| format!("evaluation failed: {e}"))?;

    write_metrics(args.metrics_path(), &report)
        .map_err(|e| format!("cannot write metrics: {e}"))?;
    Ok(report)
}

fn usage() -> &'static str {
    "usage: evaluate_job <job-dir> [--reference <sample_outputs.json>] \
     [--job-json <job.json>] [--sample <sample_outputs.json>] [--out <metrics.json>] [--k K]"
}

// ---------------------------------------------------------------------------
// Argument parsing (std only; no clap dependency)
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct Args {
    job_dir: PathBuf,
    reference: Option<PathBuf>,
    job_json: Option<PathBuf>,
    sample: Option<PathBuf>,
    out: Option<PathBuf>,
    k: Option<f64>,
}

impl Args {
    fn parse(argv: Vec<String>) -> Result<Option<Self>, String> {
        if argv.is_empty() {
            return Ok(None);
        }
        if argv.iter().any(|a| a == "-h" || a == "--help") {
            println!("{}", help_text());
            return Ok(None);
        }

        let mut positional: Vec<String> = Vec::new();
        let mut reference = None;
        let mut job_json = None;
        let mut sample = None;
        let mut out = None;
        let mut k = None;

        let mut i = 0;
        while i < argv.len() {
            let arg = argv[i].as_str();
            let mut next = |name: &str| -> Result<String, String> {
                i += 1;
                argv.get(i)
                    .cloned()
                    .ok_or_else(|| format!("{name} requires a value"))
            };
            match arg {
                "--reference" => reference = Some(PathBuf::from(next("--reference")?)),
                "--job-json" => job_json = Some(PathBuf::from(next("--job-json")?)),
                "--sample" => sample = Some(PathBuf::from(next("--sample")?)),
                "--out" => out = Some(PathBuf::from(next("--out")?)),
                "--k" => {
                    let value = parser_f64("--k", &next("--k")?)?;
                    if value <= 0.0 {
                        return Err("--k must be positive".into());
                    }
                    k = Some(value);
                }
                other if other.starts_with('-') => {
                    return Err(format!("unknown option '{other}'"));
                }
                other => positional.push(other.to_string()),
            }
            i += 1;
        }

        if positional.len() != 1 {
            return Err(format!(
                "expected exactly one <job-dir>, got {} positional arguments",
                positional.len()
            ));
        }

        Ok(Some(Self {
            job_dir: PathBuf::from(&positional[0]),
            reference,
            job_json,
            sample,
            out,
            k,
        }))
    }

    fn sample_path(&self) -> PathBuf {
        self.sample
            .clone()
            .unwrap_or_else(|| self.job_dir.join("sample_outputs.json"))
    }

    fn metrics_path(&self) -> PathBuf {
        self.out
            .clone()
            .unwrap_or_else(|| self.job_dir.join("metrics.json"))
    }
}

fn help_text() -> String {
    [
        usage(),
        "",
        "options:",
        "  --reference PATH   reference sample_outputs.json (default: job.json actions)",
        "  --job-json PATH    job manifest (default: <job-dir>/job.json)",
        "  --sample PATH      recovered outputs (default: <job-dir>/sample_outputs.json)",
        "  --out PATH         metrics output (default: <job-dir>/metrics.json)",
        "  --k K              override kilometres per Cosmos-metre (applies to both)",
    ]
    .join("\n")
}

fn parser_f64(name: &str, value: &str) -> Result<f64, String> {
    let parsed: f64 = value
        .parse()
        .map_err(|_| format!("{name} must be a number, got '{value}'"))?;
    if !parsed.is_finite() {
        return Err(format!("{name} must be finite"));
    }
    Ok(parsed)
}
