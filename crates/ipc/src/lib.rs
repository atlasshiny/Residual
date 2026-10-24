//! File-based job queue and artifact manifests for the Cosmos pipeline.
//!
//! The GPU workers are deliberately decoupled from the rest of the system: a
//! job is a directory containing `job.json`, the runner writes its outputs into
//! the same directory, and `manifest.json` records what was produced.  No
//! database, no RPC, trivially inspectable after a failed run.
//!
//! Layout:
//!
//! ```text
//! jobs/
//! └─ <job-id>/
//!    ├─ job.json
//!    ├─ manifest.json
//!    └─ ... output artifacts ...
//! ```

use std::error::Error;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::Value;

// ---------------------------------------------------------------------------
// Jobs
// ---------------------------------------------------------------------------

/// The three Cosmos entry points from ARCHITECTURE.md section 7.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobMode {
    Transfer,
    InverseDynamics,
    ForwardDynamics,
}

impl JobMode {
    pub fn as_str(self) -> &'static str {
        match self {
            JobMode::Transfer => "transfer",
            JobMode::InverseDynamics => "inverse_dynamics",
            JobMode::ForwardDynamics => "forward_dynamics",
        }
    }
}

/// One unit of work for a single GPU worker.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Job {
    /// Unique, filesystem-safe identifier.
    pub id: String,
    pub mode: JobMode,
    /// Input file paths, relative to the job directory or absolute.
    #[serde(default)]
    pub inputs: Vec<String>,
    #[serde(default)]
    pub seed: u64,
    /// Mode-specific parameters (`prompt`, `domain_name`, `k`, ...).
    #[serde(default)]
    pub params: Value,
    /// Unix seconds when the job was created.
    #[serde(default)]
    pub created_at_s: u64,
}

impl Job {
    pub fn new(id: impl Into<String>, mode: JobMode) -> Self {
        Self {
            id: id.into(),
            mode,
            inputs: Vec::new(),
            seed: 0,
            params: Value::Null,
            created_at_s: unix_now(),
        }
    }

    pub fn with_input(mut self, path: impl Into<String>) -> Self {
        self.inputs.push(path.into());
        self
    }

    /// Reject ids that could escape the queue root when used as a directory.
    pub fn validate(&self) -> Result<(), IpcError> {
        if !is_safe_id(&self.id) {
            return Err(IpcError::Invalid(format!(
                "job id '{}' must be non-empty and contain only [A-Za-z0-9._-]",
                self.id
            )));
        }
        Ok(())
    }
}

fn is_safe_id(id: &str) -> bool {
    !id.is_empty()
        && !id.starts_with('.')
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
}

// ---------------------------------------------------------------------------
// Artifacts and manifests
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactKind {
    Video,
    PoseJson,
    Metrics,
    Image,
    Log,
    Other,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Artifact {
    pub name: String,
    /// Path relative to the job directory (or absolute for external stages).
    pub path: String,
    pub kind: ArtifactKind,
    #[serde(default)]
    pub bytes: Option<u64>,
}

impl Artifact {
    pub fn new(name: impl Into<String>, path: impl Into<String>, kind: ArtifactKind) -> Self {
        Self {
            name: name.into(),
            path: path.into(),
            kind,
            bytes: None,
        }
    }
}

/// What a completed job produced.  Written once at the end of a run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    pub job_id: String,
    pub mode: JobMode,
    #[serde(default)]
    pub artifacts: Vec<Artifact>,
    #[serde(default)]
    pub created_at_s: u64,
    #[serde(default)]
    pub metadata: Value,
}

impl Manifest {
    pub fn new(job: &Job) -> Self {
        Self {
            job_id: job.id.clone(),
            mode: job.mode,
            artifacts: Vec::new(),
            created_at_s: unix_now(),
            metadata: Value::Null,
        }
    }

    pub fn add(&mut self, artifact: Artifact) {
        self.artifacts.push(artifact);
    }

    /// Find an artifact by its logical name.
    pub fn artifact(&self, name: &str) -> Option<&Artifact> {
        self.artifacts.iter().find(|a| a.name == name)
    }
}

// ---------------------------------------------------------------------------
// Job store
// ---------------------------------------------------------------------------

/// Directory-backed queue rooted at a jobs folder.
#[derive(Debug, Clone)]
pub struct JobStore {
    root: PathBuf,
}

impl JobStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn job_dir(&self, id: &str) -> Result<PathBuf, IpcError> {
        if !is_safe_id(id) {
            return Err(IpcError::Invalid(format!("unsafe job id '{id}'")));
        }
        Ok(self.root.join(id))
    }

    /// Create `<root>/<id>/job.json`.
    pub fn enqueue(&self, job: &Job) -> Result<PathBuf, IpcError> {
        job.validate()?;
        let dir = self.job_dir(&job.id)?;
        fs::create_dir_all(&dir)
            .map_err(|e| IpcError::Io(dir.display().to_string(), e.to_string()))?;
        let path = dir.join("job.json");
        let text = serde_json::to_string_pretty(job).map_err(|e| IpcError::Json(e.to_string()))?;
        fs::write(&path, text)
            .map_err(|e| IpcError::Io(path.display().to_string(), e.to_string()))?;
        Ok(path)
    }

    pub fn read_job(&self, id: &str) -> Result<Job, IpcError> {
        let path = self.job_dir(id)?.join("job.json");
        read_json(&path)
    }

    pub fn write_manifest(&self, manifest: &Manifest) -> Result<PathBuf, IpcError> {
        manifest.validate()?;
        let dir = self.job_dir(&manifest.job_id)?;
        fs::create_dir_all(&dir)
            .map_err(|e| IpcError::Io(dir.display().to_string(), e.to_string()))?;
        let path = dir.join("manifest.json");
        let text =
            serde_json::to_string_pretty(manifest).map_err(|e| IpcError::Json(e.to_string()))?;
        fs::write(&path, text)
            .map_err(|e| IpcError::Io(path.display().to_string(), e.to_string()))?;
        Ok(path)
    }

    pub fn read_manifest(&self, id: &str) -> Result<Manifest, IpcError> {
        let path = self.job_dir(id)?.join("manifest.json");
        read_json(&path)
    }

    /// Every job directory that contains a `job.json`, sorted by id.
    pub fn list_jobs(&self) -> Result<Vec<Job>, IpcError> {
        let entries = match fs::read_dir(&self.root) {
            Ok(entries) => entries,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(IpcError::Io(self.root.display().to_string(), e.to_string())),
        };
        let mut jobs: Vec<Job> = Vec::new();
        for entry in entries {
            let entry =
                entry.map_err(|e| IpcError::Io(self.root.display().to_string(), e.to_string()))?;
            if !entry.path().is_dir() {
                continue;
            }
            let job_path = entry.path().join("job.json");
            if job_path.is_file() {
                jobs.push(read_json(&job_path)?);
            }
        }
        jobs.sort_by(|a, b| a.id.cmp(&b.id));
        Ok(jobs)
    }

    /// Jobs that have no manifest yet.
    pub fn pending(&self) -> Result<Vec<Job>, IpcError> {
        Ok(self
            .list_jobs()?
            .into_iter()
            .filter(|job| {
                !self
                    .job_dir(&job.id)
                    .map(|d| d.join("manifest.json").is_file())
                    .unwrap_or(false)
            })
            .collect())
    }

    /// Completed jobs, paired with their manifests.
    pub fn completed(&self) -> Result<Vec<(Job, Manifest)>, IpcError> {
        let mut out = Vec::new();
        for job in self.list_jobs()? {
            if self.job_dir(&job.id)?.join("manifest.json").is_file() {
                out.push((job.clone(), self.read_manifest(&job.id)?));
            }
        }
        Ok(out)
    }
}

impl Manifest {
    fn validate(&self) -> Result<(), IpcError> {
        if !is_safe_id(&self.job_id) {
            return Err(IpcError::Invalid(format!(
                "unsafe job id '{}'",
                self.job_id
            )));
        }
        for artifact in &self.artifacts {
            if artifact.name.trim().is_empty() {
                return Err(IpcError::Invalid("artifact name is empty".into()));
            }
        }
        Ok(())
    }
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, IpcError> {
    let text = fs::read_to_string(path)
        .map_err(|e| IpcError::Io(path.display().to_string(), e.to_string()))?;
    serde_json::from_str(&text).map_err(|e| IpcError::Json(e.to_string()))
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum IpcError {
    Io(String, String),
    Json(String),
    Invalid(String),
}

impl fmt::Display for IpcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IpcError::Io(path, err) => write!(f, "io error for {path}: {err}"),
            IpcError::Json(err) => write!(f, "invalid JSON: {err}"),
            IpcError::Invalid(msg) => write!(f, "invalid job: {msg}"),
        }
    }
}

impl Error for IpcError {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn temp_root(tag: &str) -> PathBuf {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir =
            std::env::temp_dir().join(format!("residual-ipc-{tag}-{}-{n}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn enqueue_list_and_manifest() {
        let root = temp_root("roundtrip");
        let store = JobStore::new(&root);

        let job = Job::new("job-001", JobMode::InverseDynamics).with_input("clip.mp4");
        store.enqueue(&job).unwrap();

        let jobs = store.list_jobs().unwrap();
        assert_eq!(jobs, vec![job.clone()]);
        assert_eq!(store.pending().unwrap().len(), 1);

        let mut manifest = Manifest::new(&job);
        manifest.add(Artifact::new("poses", "poses.json", ArtifactKind::PoseJson));
        store.write_manifest(&manifest).unwrap();

        assert_eq!(store.read_manifest("job-001").unwrap(), manifest);
        assert!(store.pending().unwrap().is_empty());
        assert_eq!(store.completed().unwrap().len(), 1);

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn rejects_traversal_ids() {
        let store = JobStore::new(temp_root("traversal"));
        let job = Job::new("../escape", JobMode::Transfer);
        assert!(store.enqueue(&job).is_err());
    }

    #[test]
    fn missing_root_is_empty() {
        let store = JobStore::new(temp_root("missing"));
        assert!(store.list_jobs().unwrap().is_empty());
    }

    #[test]
    fn mode_names_are_stable() {
        assert_eq!(JobMode::Transfer.as_str(), "transfer");
        let json = serde_json::to_string(&JobMode::ForwardDynamics).unwrap();
        assert_eq!(json, "\"forward_dynamics\"");
    }
}
