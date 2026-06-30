// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Computes the SemVer build-metadata suffix appended to `CARGO_PKG_VERSION` to form
//! [`daemon_common::VERSION`]. Two producers, following phosphor's grammar:
//!
//! - Nix builds (reproducible, no `.git` in the sandbox): the flake injects `DAEMON_BUILD_ID`
//!   (e.g. `g1a2b3c4` or `g1a2b3c4.dirty`), which we wrap verbatim as `+<id>`.
//! - Dev builds: best-effort `git describe --tags --dirty --always`, mapped to a suffix.
//!
//! The suffix is empty on a clean exact tag, so a tagged release reports the bare `X.Y.Z`.

use std::path::Path;
use std::process::Command;

fn main() {
    // Refresh when the injected id changes, or (dev) when the git HEAD/index moves so the
    // off-tag distance / short hash / dirty marker stay current.
    println!("cargo:rerun-if-env-changed=DAEMON_BUILD_ID");
    // build.rs runs with CWD = this crate dir (crates/contracts/daemon-common); the repo `.git`
    // lives three levels up at the daemon-node root. Best-effort: only emit if present.
    for rel in ["../../../.git/HEAD", "../../../.git/index"] {
        if Path::new(rel).exists() {
            println!("cargo:rerun-if-changed={rel}");
        }
    }

    println!("cargo:rustc-env=DAEMON_BUILD_SUFFIX={}", build_suffix());
}

/// The SemVer build-metadata suffix: `""` or `+<dot-separated-identifiers>`.
fn build_suffix() -> String {
    // Nix path: the flake hands us a ready identifier (no `.git` available).
    if let Ok(id) = std::env::var("DAEMON_BUILD_ID") {
        let id = id.trim();
        return if id.is_empty() {
            String::new()
        } else {
            format!("+{id}")
        };
    }
    // Dev path: derive from git, falling back to a bare version if git is unavailable.
    match git_describe() {
        Some(desc) => suffix_from_describe(&desc),
        None => String::new(),
    }
}

fn git_describe() -> Option<String> {
    let out = Command::new("git")
        .args([
            "describe", "--tags", "--match", "v[0-9]*", "--dirty", "--always",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let desc = String::from_utf8(out.stdout).ok()?.trim().to_string();
    (!desc.is_empty()).then_some(desc)
}

/// Map `git describe` output to a build-metadata suffix (phosphor's grammar):
/// - `vX.Y.Z-N-gHASH` -> `+N.gHASH`
/// - bare `HASH` (no tags yet) -> `+gHASH`
/// - exact tag -> `` (empty)
/// - `-dirty` appends `.dirty` (or `+dirty` on an otherwise-clean exact tag).
fn suffix_from_describe(raw: &str) -> String {
    let (raw, dirty) = match raw.strip_suffix("-dirty") {
        Some(rest) => (rest, true),
        None => (raw, false),
    };
    let raw = raw.strip_prefix('v').unwrap_or(raw);

    // `tag-N-gHASH`: N commits past the most recent tag.
    if let Some((tag_and_n, hash)) = raw.rsplit_once("-g") {
        if let Some((_tag, n)) = tag_and_n.rsplit_once('-') {
            let mut meta = format!("{n}.g{hash}");
            if dirty {
                meta.push_str(".dirty");
            }
            return format!("+{meta}");
        }
    }

    // `--always` with no reachable tag yields a bare abbreviated hash.
    if raw.len() >= 7 && raw.chars().all(|c| c.is_ascii_hexdigit()) {
        let mut meta = format!("g{raw}");
        if dirty {
            meta.push_str(".dirty");
        }
        return format!("+{meta}");
    }

    // Exact tag: a clean release reports the bare base version.
    if dirty {
        "+dirty".to_string()
    } else {
        String::new()
    }
}
