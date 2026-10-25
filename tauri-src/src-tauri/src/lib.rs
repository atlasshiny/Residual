use std::path::PathBuf;
use std::process::Command;

/// Make targets the UI is allowed to run.  Keeping this an allowlist means a
/// compromised webview cannot turn the IPC bridge into arbitrary `make`.
const ALLOWED_TARGETS: &[&str] = &["ui-data", "residual", "train"];

/// Repository root, resolved at compile time.  The Tauri process starts in
/// `tauri-src` (or `src-tauri`), never in the repo root, so `make` needs an
/// explicit working directory.
fn project_root() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    if let Some(root) = manifest.parent().and_then(|p| p.parent()) {
        if root.join("Makefile").is_file() {
            return root.to_path_buf();
        }
    }
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            read_text_file,
            list_dir,
            pick_folder,
            run_make,
            propagate_mission
        ])
        .run(tauri::generate_context!())
        .expect("error while running residual ui");
}

#[tauri::command]
fn read_text_file(path: String) -> Result<String, String> {
    std::fs::read_to_string(&path).map_err(|e| format!("{path}: {e}"))
}

#[tauri::command]
fn list_dir(path: String) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(&path).map_err(|e| format!("{path}: {e}"))? {
        let entry = entry.map_err(|e| e.to_string())?;
        out.push(entry.file_name().to_string_lossy().to_string());
    }
    out.sort();
    Ok(out)
}

#[tauri::command]
async fn pick_folder(app: tauri::AppHandle) -> Option<String> {
    use tauri_plugin_dialog::DialogExt;

    tauri::async_runtime::spawn_blocking(move || {
        app.dialog()
            .file()
            .blocking_pick_folder()
            .map(|path| path.to_string())
    })
    .await
    .ok()
    .flatten()
}

#[tauri::command]
async fn run_make(target: String) -> Result<String, String> {
    if !ALLOWED_TARGETS.contains(&target.as_str()) {
        return Err(format!(
            "target '{target}' is not allowed (choose one of {})",
            ALLOWED_TARGETS.join(", ")
        ));
    }

    tauri::async_runtime::spawn_blocking(move || {
        let root = project_root();
        let output = Command::new("make")
            .arg(&target)
            .current_dir(&root)
            .output()
            .map_err(|e| format!("could not run make in {}: {e}", root.display()))?;

        let mut text = format!("$ make {target}  (in {})\n", root.display());
        text.push_str(&String::from_utf8_lossy(&output.stdout));
        let stderr = String::from_utf8_lossy(&output.stderr);
        if !stderr.trim().is_empty() {
            text.push_str("\n--- stderr ---\n");
            text.push_str(&stderr);
        }
        if output.status.success() {
            Ok(text)
        } else {
            Err(format!("{text}\n---\nmake exited with {}", output.status))
        }
    })
    .await
    .map_err(|e| format!("make task failed: {e}"))?
}

#[tauri::command]
async fn propagate_mission(
    request: mission::MissionRequest,
) -> Result<mission::MissionTrajectory, String> {
    tauri::async_runtime::spawn_blocking(move || mission::run(&request))
        .await
        .map_err(|e| format!("mission task failed: {e}"))?
        .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn project_root_resolves_to_the_repo() {
        let root = project_root();
        assert!(
            root.join("Makefile").is_file(),
            "project_root() returned {}",
            root.display()
        );
    }

    #[test]
    fn make_targets_are_allowlisted() {
        assert!(ALLOWED_TARGETS.contains(&"ui-data"));
        assert!(!ALLOWED_TARGETS.contains(&"clean"));
        assert!(!ALLOWED_TARGETS.contains(&"$(touch /tmp/pwned)"));
    }
}
