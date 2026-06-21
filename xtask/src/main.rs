//! `xtask` — repo automation (codegen, CI helpers).
//!
//! Subcommands:
//! - `gen-headers` — run `cbindgen` over `bindings/daemon-core-ffi` to (re)generate the committed C
//!   header `bindings/daemon-core-ffi/include/daemon_core.h`. The generated header plus the
//!   published `daemon-api.cddl` are the complete non-Rust contract (daemon-ffi-spec §3.6).
//! - `cddl` — a light presence/sanity check of the `daemon-api` mirror CDDL artifact.

#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};
use std::process::Command;

fn main() -> anyhow::Result<()> {
    let sub = std::env::args().nth(1).unwrap_or_default();
    match sub.as_str() {
        "gen-headers" => gen_headers(),
        "cddl" => check_cddl(),
        other => {
            eprintln!("usage: xtask <gen-headers|cddl>");
            anyhow::bail!("unknown xtask subcommand: {other:?}");
        }
    }
}

/// The workspace root (xtask's manifest dir is `<root>/xtask`).
fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask lives under the workspace root")
        .to_path_buf()
}

/// Generate `bindings/daemon-core-ffi/include/daemon_core.h` via `cbindgen`.
fn gen_headers() -> anyhow::Result<()> {
    let root = workspace_root();
    let crate_dir = root.join("bindings/daemon-core-ffi");
    let config = crate_dir.join("cbindgen.toml");
    let out = crate_dir.join("include/daemon_core.h");
    std::fs::create_dir_all(out.parent().unwrap())?;

    let status = Command::new("cbindgen")
        .arg("--config")
        .arg(&config)
        .arg("--crate")
        .arg("daemon-core-ffi")
        .arg("--output")
        .arg(&out)
        .arg(&crate_dir)
        .status()
        .map_err(|e| anyhow::anyhow!("failed to run cbindgen (is it on PATH?): {e}"))?;
    anyhow::ensure!(status.success(), "cbindgen exited with {status}");

    println!("generated {}", out.display());
    Ok(())
}

/// A light check of the `daemon-api` CDDL mirror artifact: it exists, is non-empty, and references
/// the top-level request/response rules. A full validator is out of scope.
fn check_cddl() -> anyhow::Result<()> {
    let path = workspace_root().join("crates/contracts/daemon-api/daemon-api.cddl");
    let text = std::fs::read_to_string(&path)
        .map_err(|e| anyhow::anyhow!("reading {}: {e}", path.display()))?;
    anyhow::ensure!(!text.trim().is_empty(), "{} is empty", path.display());
    for rule in [
        "api-request",
        "api-response",
        "wire_version",
        // wire v2: the merged live session event log shapes.
        "session-log-entry",
        "session-payload",
        "log-page-view",
        "direction",
        "disposition",
        "origin",
        // wire v2: outbound delivery targets + handover (§5.4).
        "delivery-target",
        "sink-kind",
        "route-addr",
    ] {
        anyhow::ensure!(
            text.contains(rule),
            "{} is missing the `{rule}` rule",
            path.display()
        );
    }
    println!("ok: {} defines the api mirror", path.display());
    Ok(())
}
