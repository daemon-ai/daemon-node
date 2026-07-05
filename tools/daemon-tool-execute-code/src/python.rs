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
///
/// `trusted` reflects whether `ws_root` is a node-managed isolated sandbox (`true`) or an
/// operator-bound external directory whose contents may be attacker-influenced (`false`; Cluster E).
/// On an untrusted root, venv auto-discovery is suppressed so a planted `.venv`/`venv` (or an
/// inherited `VIRTUAL_ENV`/`CONDA_PREFIX`) is never auto-executed — see [`candidate_paths`].
pub(crate) async fn resolve_interpreter(
    mode: Mode,
    ws_root: &Path,
    trusted: bool,
) -> Option<PathBuf> {
    for cand in candidate_paths(mode, ws_root, trusted) {
        if is_executable_file(&cand) && is_usable_python(&cand).await {
            return Some(cand);
        }
    }
    None
}

/// The ordered interpreter candidates for `mode`. In `Project` mode on a `trusted` root the venv-aware
/// candidates (`VIRTUAL_ENV`/`CONDA_PREFIX`, then workspace-local `.venv`/`venv`) precede the system
/// PATH interpreters. On an *untrusted* root (an operator-bound directory) the venv candidates are
/// dropped — we never auto-trust a venv discovered under a root whose contents may be
/// attacker-planted (Cluster E), so project mode resolves the same system-PATH set as `strict`.
fn candidate_paths(mode: Mode, ws_root: &Path, trusted: bool) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if mode == Mode::Project && trusted {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Cluster E: on an *untrusted* (operator-bound) workspace root, project mode must not offer the
    /// workspace-local `.venv`/`venv` (nor `VIRTUAL_ENV`/`CONDA_PREFIX`) as interpreter candidates —
    /// a planted venv there would be silently executed. It falls back to the system PATH interpreters
    /// only (the same set `strict` mode uses).
    #[test]
    fn untrusted_root_skips_workspace_venv_candidates() {
        let root = Path::new("/ws/session");
        let trusted = candidate_paths(Mode::Project, root, true);
        let untrusted = candidate_paths(Mode::Project, root, false);

        let under_venv = |cands: &[PathBuf]| {
            cands
                .iter()
                .any(|c| c.starts_with(root.join(".venv")) || c.starts_with(root.join("venv")))
        };
        // Trusted (isolated sandbox) still auto-discovers the workspace venv.
        assert!(
            under_venv(&trusted),
            "trusted project mode should include the workspace .venv/venv"
        );
        // Untrusted (Bound) root: no workspace venv candidate is offered.
        assert!(
            !under_venv(&untrusted),
            "untrusted project mode must not include a workspace-local venv candidate"
        );
        // Strict mode never adds venv candidates regardless of trust.
        assert!(!under_venv(&candidate_paths(Mode::Strict, root, true)));
    }
}
