//! Trajectory CSV -> Cosmos pose/action job manifest.
//!
//! Usage:
//!
//! ```text
//! cargo run -p posebuilder --example pose_pipeline -- <trajectory.csv> <out> [options]
//! ```
//!
//! The trajectory uses the canonical `t,x,y,z,vx,vy,vz` header (seconds and
//! metres). The example:
//!
//! 1. resamples the trajectory at `fps` inside the requested window,
//! 2. applies the pointing law to get a camera pose per frame
//!    (`observer` parks the camera at `--observer-offset`, points it at the
//!    spacecraft, and also records absolute spacecraft positions),
//! 3. computes `ΔT_t = T_{t-1}^-1 T_t` and encodes it as
//!    `[tx, ty, tz, r6]` (3D translation + 6D rotation, 9D total),
//! 4. writes `<out>/job.json` (the §6.4 manifest plus the action array)
//!    and `<out>/actions.json`, or a single combined manifest file when
//!    `<out>` ends in `.json`.
//!
//! See `docs/cosmos-contract.md` for the file contract.
//!
//! Options:
//!
//! ```text
//! --job-id ID              default: <csv stem>_<mode>
//! --mode MODE              transfer | inverse_dynamics | forward_dynamics
//!                          (default: forward_dynamics)
//! --model NAME             default: Cosmos3-Nano
//! --t0 SECONDS             window start (default: first CSV sample)
//! --t1 SECONDS             window end   (default: last CSV sample)
//! --fps N                  default: 24
//! --frames N               default: 120
//! --scale-k K              km per Cosmos-metre (default: 1000)
//! --pointing LAW           earth | moon | data | observer (default: earth)
//! --observer-offset X,Y,Z  third-person camera offset from the Earth-Moon
//!                          barycenter, kilometres (default: 1e6,0,0)
//!                          only used with --pointing observer
//! --time-compression X     simulation seconds per playback second
//!                          (default: chosen so the clip covers t1-t0)
//! --resolution WxH         default: 704x1280
//! --seed N                 default: 0
//! --control-video PATH     recorded in inputs.control_video
//! --first-frame PATH       recorded in inputs.first_frame
//! --prompt-json PATH       recorded in inputs.prompt_json
//! ```

use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use ingest::Telemetry;
use posebuilder::{
    spacecraft_position_km, Config, ObserverConfig, PointingLaw, Pose, PoseBuilder, Vec3,
};
use serde_json::{json, Value};

const DEFAULT_FPS: f64 = 24.0;
const DEFAULT_FRAMES: usize = 120;
const DEFAULT_SCALE_K: f64 = 1000.0;
const DEFAULT_RESOLUTION: [u32; 2] = [704, 1280];
/// Third-person camera offset from the Earth-Moon barycenter, kilometres.
const DEFAULT_OBSERVER_OFFSET_KM: Vec3 = Vec3 {
    x: 1.0e6,
    y: 0.0,
    z: 0.0,
};

fn main() -> Result<(), Box<dyn Error>> {
    let args = match Args::parse(env::args().skip(1).collect()) {
        Ok(Some(args)) => args,
        Ok(None) => return Ok(()),
        Err(err) => {
            eprintln!("error: {err}\n\n{}", usage());
            std::process::exit(2);
        }
    };

    run(&args)
}

fn run(args: &Args) -> Result<(), Box<dyn Error>> {
    let telemetry = Telemetry::load(&args.trajectory)?;
    let data_t0 = telemetry.samples().first().expect("validated non-empty").t;
    let data_t1 = telemetry.samples().last().expect("validated non-empty").t;

    let t0 = args.t0.unwrap_or(data_t0);
    let t1 = args.t1.unwrap_or(data_t1);
    if !t0.is_finite() || !t1.is_finite() || t1 <= t0 {
        return Err(format!("window must satisfy t1 > t0 (got t0={t0}, t1={t1})").into());
    }
    if t0 < data_t0 || t1 > data_t1 {
        return Err(format!(
            "window [{t0}, {t1}] is outside the trajectory range [{data_t0}, {data_t1}]"
        )
        .into());
    }

    // Simulation seconds per playback second. The default makes the last
    // frame land exactly on t1, so the window and the clip agree.
    let time_compression = match args.time_compression {
        Some(value) => value,
        None => (t1 - t0) * args.fps / (args.frames.saturating_sub(1) as f64),
    };
    if !time_compression.is_finite() || time_compression <= 0.0 {
        return Err("time compression must be positive and finite".into());
    }

    let frame_times: Vec<f64> = (0..args.frames)
        .map(|i| t0 + (i as f64 / args.fps) * time_compression)
        .collect();

    let mut states = Vec::with_capacity(frame_times.len());
    for &t in &frame_times {
        let state = telemetry.sample_at(t).ok_or_else(|| {
            format!("frame time {t} is outside the trajectory range or not finite")
        })?;
        states.push(state);
    }

    let config = Config::default();
    let observer = ObserverConfig::new(args.observer_offset_km.scale(1.0e3));
    let builder =
        PoseBuilder::new_with_pointing(args.pointing, args.scale_k).with_observer(observer);
    let poses = builder.poses_from_states(&states, &frame_times, &config);
    let actions = builder.actions_from_poses(&poses);
    // Absolute inertial spacecraft positions (km), independent of camera mode.
    let spacecraft_positions: Vec<Vec<f64>> = states
        .iter()
        .map(|state| {
            let p = spacecraft_position_km(state);
            vec![p.x, p.y, p.z]
        })
        .collect();

    let first_pose = poses.first().ok_or("no frames were sampled")?;
    let job_id = effective_job_id(args);
    validate_job_id(&job_id)?;

    let (out_dir, job_path) = output_paths(&args.output);
    fs::create_dir_all(&out_dir)?;

    let action_rows: Vec<Vec<f64>> = actions.iter().map(|action| action.to_vec()).collect();
    let action_doc = json!({
        "schema": "cosmos-actions/v1",
        "job_id": job_id,
        "action_dim": 9,
        "action_encoding": "translation_xyz + rot6d_first_two_columns",
        "translation_unit": "cosmos_metre",
        "scale_k_km_per_m": args.scale_k,
        "count": action_rows.len(),
        "actions": action_rows,
    });
    write_json(&out_dir.join("actions.json"), &action_doc)?;

    // Third-person vantage point, recorded only when the observer law is used.
    let observer_json = match args.pointing {
        PointingLaw::Observer => {
            let camera = observer.camera_position_km();
            json!({
                "origin": "earth_moon_barycenter",
                "offset_km": [
                    args.observer_offset_km.x,
                    args.observer_offset_km.y,
                    args.observer_offset_km.z
                ],
                "camera_position_km": [camera.x, camera.y, camera.z],
            })
        }
        _ => Value::Null,
    };

    let job = json!({
        // §6.4 manifest fields ------------------------------------------------
        "schema": "cosmos-job/v1",
        "job_id": job_id,
        "id": job_id,
        "mode": args.mode,
        "model": args.model,
        "window": {
            "t0_tdb": t0,
            "t1_tdb": t1,
            "time_compression": time_compression,
        },
        "scale_k_km_per_m": args.scale_k,
        "pointing_law": args.pointing_name(),
        "observer": observer_json,
        "fps": args.fps,
        "frames": args.frames,
        "resolution": [args.resolution[0], args.resolution[1]],
        "inputs": {
            "control_video": args.control_video,
            "first_frame": args.first_frame,
            "actions": "actions.json",
            "prompt_json": args.prompt_json,
        },
        "seed": args.seed,
        // Contract extensions (documented in docs/cosmos-contract.md) --------
        "time_system": "TDB",
        "time_unit": "s",
        "frame_times_tdb_s": frame_times,
        "action_dim": 9,
        "action_encoding": "translation_xyz + rot6d_first_two_columns",
        "action_count": action_rows.len(),
        "first_pose": pose_json(first_pose),
        "actions": action_rows,
        "spacecraft_positions_km": spacecraft_positions,
        "units": {
            "trajectory_position": "m",
            "pose_translation": "km",
            "action_translation": "cosmos_m",
            "spacecraft_position": "km",
            "observer_position": "km",
            "rotation": "6d_zhuo2019_first_two_columns",
        },
        "pointing_note": args.pointing_note(),
        "generated_by": "posebuilder/examples/pose_pipeline.rs",
    });
    write_json(&job_path, &job)?;

    println!(
        "wrote {} frames, {} actions to {} (job_id={}, k={} km/cosmos-m, pointing={})",
        args.frames,
        action_rows.len(),
        job_path.display(),
        job_id,
        args.scale_k,
        args.pointing_name()
    );
    Ok(())
}

fn write_json(path: &Path, value: &Value) -> Result<(), Box<dyn Error>> {
    let text = serde_json::to_string_pretty(value)?;
    fs::write(path, text)?;
    Ok(())
}

/// Accept either a job directory or a `.json` file path. `actions.json` always
/// lands next to the manifest.
fn output_paths(output: &Path) -> (PathBuf, PathBuf) {
    let is_json = output
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("json"))
        .unwrap_or(false);
    if is_json {
        let parent = output
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        (parent.to_path_buf(), output.to_path_buf())
    } else {
        (output.to_path_buf(), output.join("job.json"))
    }
}

fn pose_json(pose: &Pose) -> Value {
    let rotation: Vec<Vec<f64>> = pose.rotation.iter().map(|row| row.to_vec()).collect();
    json!({
        "rotation": rotation,
        "translation_km": [pose.translation.x, pose.translation.y, pose.translation.z],
    })
}

fn effective_job_id(args: &Args) -> String {
    match &args.job_id {
        Some(id) => id.clone(),
        None => {
            let stem = args
                .trajectory
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("pose_job");
            let raw = format!("{stem}_{}", args.mode);
            raw.chars()
                .map(|c| {
                    if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
                        c
                    } else {
                        '_'
                    }
                })
                .collect()
        }
    }
}

fn validate_job_id(id: &str) -> Result<(), Box<dyn Error>> {
    let ok = !id.is_empty()
        && !id.starts_with('.')
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'));
    if ok {
        Ok(())
    } else {
        Err(format!("job id '{id}' must be non-empty and contain only [A-Za-z0-9._-]").into())
    }
}

fn usage() -> &'static str {
    "usage: pose_pipeline <trajectory.csv> <out-dir-or-json> [options]\n\
     run with --help for the option list"
}

// ---------------------------------------------------------------------------
// Argument parsing (std only; no clap dependency)
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct Args {
    trajectory: PathBuf,
    output: PathBuf,
    job_id: Option<String>,
    mode: String,
    model: String,
    t0: Option<f64>,
    t1: Option<f64>,
    fps: f64,
    frames: usize,
    scale_k: f64,
    pointing: PointingLaw,
    observer_offset_km: Vec3,
    time_compression: Option<f64>,
    resolution: [u32; 2],
    seed: u64,
    control_video: Option<String>,
    first_frame: Option<String>,
    prompt_json: Option<String>,
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
        let mut job_id = None;
        let mut mode = "forward_dynamics".to_string();
        let mut model = "Cosmos3-Nano".to_string();
        let mut t0 = None;
        let mut t1 = None;
        let mut fps = DEFAULT_FPS;
        let mut frames = DEFAULT_FRAMES;
        let mut scale_k = DEFAULT_SCALE_K;
        let mut pointing = PointingLaw::Earth;
        let mut observer_offset_km = DEFAULT_OBSERVER_OFFSET_KM;
        let mut time_compression = None;
        let mut resolution = DEFAULT_RESOLUTION;
        let mut seed = 0u64;
        let mut control_video = None;
        let mut first_frame = None;
        let mut prompt_json = None;

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
                "--job-id" => job_id = Some(next("--job-id")?),
                "--mode" => mode = parse_mode(&next("--mode")?)?,
                "--model" => model = next("--model")?,
                "--t0" => t0 = Some(parse_f64("--t0", &next("--t0")?)?),
                "--t1" => t1 = Some(parse_f64("--t1", &next("--t1")?)?),
                "--fps" => fps = parse_f64("--fps", &next("--fps")?)?,
                "--frames" => {
                    frames = next("--frames")?
                        .parse()
                        .map_err(|_| "--frames must be a positive integer".to_string())?
                }
                "--scale-k" => scale_k = parse_f64("--scale-k", &next("--scale-k")?)?,
                "--pointing" => pointing = parse_pointing(&next("--pointing")?)?,
                "--observer-offset" => {
                    observer_offset_km =
                        parse_vec3_km("--observer-offset", &next("--observer-offset")?)?
                }
                "--time-compression" => {
                    time_compression = Some(parse_f64(
                        "--time-compression",
                        &next("--time-compression")?,
                    )?)
                }
                "--resolution" => resolution = parse_resolution(&next("--resolution")?)?,
                "--seed" => {
                    seed = next("--seed")?
                        .parse()
                        .map_err(|_| "--seed must be an unsigned integer".to_string())?
                }
                "--control-video" => control_video = Some(next("--control-video")?),
                "--first-frame" => first_frame = Some(next("--first-frame")?),
                "--prompt-json" => prompt_json = Some(next("--prompt-json")?),
                other if other.starts_with('-') => {
                    return Err(format!("unknown option '{other}'"));
                }
                other => positional.push(other.to_string()),
            }
            i += 1;
        }

        if positional.len() != 2 {
            return Err(format!(
                "expected <trajectory.csv> and <out-dir-or-json>, got {} positional arguments",
                positional.len()
            ));
        }
        if !fps.is_finite() || fps <= 0.0 {
            return Err("--fps must be positive and finite".into());
        }
        if frames < 2 {
            return Err("--frames must be at least 2 (one action needs two poses)".into());
        }
        if !scale_k.is_finite() || scale_k <= 0.0 {
            return Err("--scale-k must be positive and finite".into());
        }

        Ok(Some(Self {
            trajectory: PathBuf::from(&positional[0]),
            output: PathBuf::from(&positional[1]),
            job_id,
            mode,
            model,
            t0,
            t1,
            fps,
            frames,
            scale_k,
            pointing,
            observer_offset_km,
            time_compression,
            resolution,
            seed,
            control_video,
            first_frame,
            prompt_json,
        }))
    }

    fn pointing_name(&self) -> &'static str {
        match self.pointing {
            PointingLaw::Earth => "earth",
            PointingLaw::Moon => "moon",
            PointingLaw::Data => "data",
            PointingLaw::Observer => "observer",
        }
    }

    fn pointing_note(&self) -> &'static str {
        match self.pointing {
            PointingLaw::Earth => "synthetic Earth-pointing; no measured attitude used",
            PointingLaw::Moon => "synthetic Moon-pointing; no measured attitude used",
            PointingLaw::Data => {
                "requested 'data' pointing but the CSV has no attitude; fell back to Earth"
            }
            PointingLaw::Observer => {
                "third-person observer; camera parked at observer offset and pointed at the spacecraft"
            }
        }
    }
}

fn help_text() -> String {
    [
        usage(),
        "",
        "options:",
        "  --job-id ID              default: <csv stem>_<mode>",
        "  --mode MODE              transfer | inverse_dynamics | forward_dynamics",
        "                           (default: forward_dynamics)",
        "  --model NAME             default: Cosmos3-Nano",
        "  --t0 SECONDS             window start (default: first CSV sample)",
        "  --t1 SECONDS             window end   (default: last CSV sample)",
        "  --fps N                  default: 24",
        "  --frames N               default: 120",
        "  --scale-k K              km per Cosmos-metre (default: 1000)",
        "  --pointing LAW           earth | moon | data | observer (default: earth)",
        "  --observer-offset X,Y,Z  observer camera offset from the Earth-Moon",
        "                           barycenter, km (default: 1e6,0,0); with",
        "                           --pointing observer",
        "  --time-compression X     simulation seconds per playback second",
        "  --resolution WxH         default: 704x1280",
        "  --seed N                 default: 0",
        "  --control-video PATH     recorded in inputs.control_video",
        "  --first-frame PATH       recorded in inputs.first_frame",
        "  --prompt-json PATH       recorded in inputs.prompt_json",
    ]
    .join("\n")
}

fn parse_mode(value: &str) -> Result<String, String> {
    let normalized = value.replace('-', "_");
    match normalized.as_str() {
        "transfer" | "inverse_dynamics" | "forward_dynamics" => Ok(normalized),
        _ => Err(format!(
            "unknown mode '{value}' (expected transfer, inverse_dynamics, or forward_dynamics)"
        )),
    }
}

fn parse_pointing(value: &str) -> Result<PointingLaw, String> {
    match value.to_ascii_lowercase().as_str() {
        "earth" => Ok(PointingLaw::Earth),
        "moon" => Ok(PointingLaw::Moon),
        "data" => Ok(PointingLaw::Data),
        "observer" | "third_person" | "third-person" => Ok(PointingLaw::Observer),
        _ => Err(format!(
            "unknown pointing law '{value}' (expected earth, moon, data, or observer)"
        )),
    }
}

/// Parse a comma/x/space separated `X,Y,Z` triplet in kilometres.
fn parse_vec3_km(name: &str, value: &str) -> Result<Vec3, String> {
    let parts: Vec<&str> = value
        .split(|c: char| c == ',' || c == 'x' || c == 'X' || c == ':' || c.is_whitespace())
        .filter(|p| !p.is_empty())
        .collect();
    if parts.len() != 3 {
        return Err(format!(
            "{name} must be X,Y,Z in kilometres (for example 1e6,0,0), got '{value}'"
        ));
    }
    let mut out = [0.0f64; 3];
    for (i, part) in parts.iter().enumerate() {
        let parsed: f64 = part
            .parse()
            .map_err(|_| format!("{name} component '{}' is not a number", part))?;
        if !parsed.is_finite() {
            return Err(format!("{name} components must be finite"));
        }
        out[i] = parsed;
    }
    Ok(Vec3::new(out[0], out[1], out[2]))
}

fn parse_f64(name: &str, value: &str) -> Result<f64, String> {
    let parsed: f64 = value
        .parse()
        .map_err(|_| format!("{name} must be a number, got '{value}'"))?;
    if !parsed.is_finite() {
        return Err(format!("{name} must be finite"));
    }
    Ok(parsed)
}

fn parse_resolution(value: &str) -> Result<[u32; 2], String> {
    let parts: Vec<&str> = value
        .split(|c| c == 'x' || c == 'X' || c == ',' || c == ':')
        .collect();
    if parts.len() != 2 {
        return Err(format!(
            "--resolution must be WxH (for example 704x1280), got '{value}'"
        ));
    }
    let w: u32 = parts[0]
        .trim()
        .parse()
        .map_err(|_| format!("bad resolution width '{}'", parts[0]))?;
    let h: u32 = parts[1]
        .trim()
        .parse()
        .map_err(|_| format!("bad resolution height '{}'", parts[1]))?;
    if w == 0 || h == 0 {
        return Err("--resolution dimensions must be positive".into());
    }
    Ok([w, h])
}
