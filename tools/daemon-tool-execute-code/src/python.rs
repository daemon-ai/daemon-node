// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Interpreter resolution for execute_code (hermes `_resolve_child_python` parity).
//!
//! `project` mode prefers the user's active virtualenv / conda env, then a workspace-local `.venv` /
//! `venv`, then `python3`/`python` on `PATH` — so `import pandas` and project packages resolve like
//! the shell tool. `strict` mode uses only `python3`/`python` on `PATH` (reproducible; no project
//! deps). Every candidate must exist, be executable, and pass a cached Python >= 3.8 version probe.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use crate::Mode;

/// The interpreter version-probe timeout (hermes forks `python -c` with a 5 s guard).
const PROBE_TIMEOUT: Duration = Duration::from_secs(5);

/// Resolve the child interpreter for `mode`, rooted at the session workspace `ws_root`. Returns the
/// first candidate that exists, is executable, and is Python >= 3.8; `None` if none qualifies.
pub(crate) async fn resolve_interpreter(mode: Mode, ws_root: &Path) -> Option<PathBuf> {
    for cand in candidate_paths(mode, ws_root) {
        if is_executable_file(&cand) && is_usable_python(&cand).await {
            return Some(cand);
        }
    }
    None
}

/// The ordered interpreter candidates for `mode`.
fn candidate_paths(mode: Mode, ws_root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if mode == Mode::Project {
        for var in ["VIRTUAL_ENV", "CONDA_PREFIX"] {
            if let Some(root) = std::env::var_os(var).filter(|v| !v.is_empty()) {
                push_venv(&mut out, Path::new(&root));
            }
        }
        push_venv(&mut out, &ws_root.join(".venv"));
        push_venv(&mut out, &ws_root.join("venv"));
    }
    for name in exe_names() {
        if let Some(found) = which(name) {
            out.push(found);
        }
    }
    out
}

/// Append the interpreter paths inside a virtualenv/conda root (`<root>/bin/python[3]`).
fn push_venv(out: &mut Vec<PathBuf>, root: &Path) {
    for name in exe_names() {
        out.push(root.join(bin_subdir()).join(name));
    }
}

/// Interpreter executable names, most-specific first.
fn exe_names() -> &'static [&'static str] {
    #[cfg(windows)]
    {
        &["python.exe", "python3.exe"]
    }
    #[cfg(not(windows))]
    {
        &["python3", "python"]
    }
}

/// The venv interpreter subdirectory (`Scripts` on Windows, `bin` elsewhere).
fn bin_subdir() -> &'static str {
    #[cfg(windows)]
    {
        "Scripts"
    }
    #[cfg(not(windows))]
    {
        "bin"
    }
}

/// First `name` found on `PATH`, if any.
fn which(name: &str) -> Option<PathBuf> {
    let paths = std::env::var_os("PATH")?;
    std::env::split_paths(&paths)
        .map(|dir| dir.join(name))
        .find(|cand| is_executable_file(cand))
}

/// Whether `p` is an existing, executable regular file.
fn is_executable_file(p: &Path) -> bool {
    let Ok(meta) = std::fs::metadata(p) else {
        return false;
    };
    if !meta.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        meta.permissions().mode() & 0o111 != 0
    }
    #[cfg(not(unix))]
    {
        true
    }
}

/// Whether `path` is a usable Python >= 3.8 (cached across calls so we fork at most once per path).
async fn is_usable_python(path: &Path) -> bool {
    static CACHE: OnceLock<Mutex<HashMap<PathBuf, bool>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(cached) = cache.lock().expect("probe cache poisoned").get(path) {
        return *cached;
    }
    let ok = probe(path).await;
    cache
        .lock()
        .expect("probe cache poisoned")
        .insert(path.to_path_buf(), ok);
    ok
}

/// Fork `<path> -c "…"` and report whether it exits 0 for Python >= 3.8 within [`PROBE_TIMEOUT`].
async fn probe(path: &Path) -> bool {
    let fut = tokio::process::Command::new(path)
        .arg("-c")
        .arg("import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
    matches!(
        tokio::time::timeout(PROBE_TIMEOUT, fut).await,
        Ok(Ok(status)) if status.success()
    )
}
