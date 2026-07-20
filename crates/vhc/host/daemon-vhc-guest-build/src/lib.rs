// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The ONE guest-workspace builder every wasm-backed test harness shares.
//!
//! `crates/vhc/guests` is its own cargo workspace; every test binary that runs a production guest
//! blob builds it on demand (`cargo build --release --target wasm32-unknown-unknown`). Before this
//! crate, each harness carried its own copy of that shell-out, and the copies had to stay in
//! lockstep on the reproducibility env by hand — a straggler that forgot the RUSTFLAGS remap
//! rebuilt the guests with differently-hashed bytes, and under an overlapped lane schedule could
//! rewrite the on-disk `.wasm` files while another suite was mid-read (spurious content-hash
//! mismatches against the committed `guests.blake3` pins). Centralizing the builder makes that
//! drift class unrepresentable, and the advisory build lock makes concurrent builders safe even
//! if a future call site regresses:
//!
//! - **One env**: [`remap_rustflags`] + the `CARGO_TARGET_DIR`/`RUSTC_WRAPPER` scrubs are applied
//!   here, once, for every caller (tests, xtask, the harness-gated replay sandbox).
//! - **One lock**: the build and every artifact read go through a cross-process advisory lockfile
//!   in the guests target dir (the same pattern as the acceptance suite's `serial_guard`), so a
//!   concurrent rebuild can never clobber `.wasm` bytes another test binary is reading.
//!
//! Dev/gate tooling by charter: linked only by test targets, xtask, and harness-feature-gated
//! oracle code — never by a production binary.

// Dev/gate tooling: shells `cargo` for the guests workspace and reads its build artifacts, so the
// fs/process choke-point bans do not apply (the established harness allowance, now in one place).
#![allow(clippy::disallowed_methods)]

use std::path::PathBuf;
use std::sync::OnceLock;
use std::time::Duration;

/// The guests workspace root (`crates/vhc/guests`), resolved from this crate's manifest dir.
///
/// # Panics
/// If the guests workspace is missing (a structurally broken checkout).
#[must_use]
pub fn guests_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../guests")
        .canonicalize()
        .expect("guests workspace path")
}

/// RUSTFLAGS that make the guest `.wasm` byte-reproducible across checkouts/machines by remapping
/// the absolute prefixes rustc embeds in panic locations: the `<checkout>` root (workspace + path
/// deps like the guest SDK crates) and the cargo registry (`$CARGO_HOME`, else `$HOME/.cargo`).
/// With the guests' committed `Cargo.lock` this makes clean rebuilds byte-identical within one
/// checkout path; the cross-worktree `-C metadata` reordering this remap does NOT rewrite is
/// handled by the guests workspace's config-wired `rustc-wrapper` (`guest-rustc-shim.sh`).
///
/// Every builder MUST pass the identical value: cargo fingerprints RUSTFLAGS, so a builder with
/// different flags does not just produce different bytes — it *rewrites* the shared on-disk
/// artifacts other suites read (the drift class this crate exists to end).
#[must_use]
pub fn remap_rustflags() -> String {
    let root = guests_root();
    let checkout = root.ancestors().nth(3).unwrap_or(&root).to_path_buf();
    let cargo_home = std::env::var_os("CARGO_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".cargo"));
    format!(
        "--remap-path-prefix={}=/daemon-node --remap-path-prefix={}=/cargo",
        checkout.display(),
        cargo_home.display(),
    )
}

/// A cross-process advisory guard serializing guest-workspace builds and artifact reads (the
/// acceptance suite's `serial_guard` pattern): holders exclude each other via a `create_new`
/// lockfile in the guests target dir, so a rebuild can never rewrite `.wasm` bytes while another
/// test binary is mid-read.
struct BuildGuard {
    path: PathBuf,
}

impl Drop for BuildGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Acquire the guest-build guard (blocks until free; stale locks older than 30 min are reclaimed
/// so a killed builder never wedges the suites).
fn build_guard() -> BuildGuard {
    let target = guests_root().join("target");
    // First build in a fresh checkout: the target dir does not exist yet.
    let _ = std::fs::create_dir_all(&target);
    let path = target.join(".guest-build.lock");
    loop {
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
        {
            Ok(mut f) => {
                use std::io::Write as _;
                let _ = writeln!(f, "{}", std::process::id());
                return BuildGuard { path };
            }
            Err(_) => {
                // Reclaim a stale lock (a builder that was killed before its guard dropped).
                if let Ok(meta) = std::fs::metadata(&path) {
                    if meta
                        .modified()
                        .ok()
                        .and_then(|m| m.elapsed().ok())
                        .is_some_and(|e| e > Duration::from_secs(1800))
                    {
                        let _ = std::fs::remove_file(&path);
                        continue;
                    }
                }
                std::thread::sleep(Duration::from_millis(250));
            }
        }
    }
}

/// Build the guests workspace for `wasm32-unknown-unknown` under the build guard — unconditionally
/// (cargo's own freshness check makes a repeat a no-op). Callers inside test binaries should
/// prefer [`ensure_built`] (once per process).
///
/// # Errors
/// A human-readable description of the spawn or build failure.
pub fn build_guests() -> Result<(), String> {
    let _guard = build_guard();
    let status = std::process::Command::new("cargo")
        .current_dir(guests_root())
        // The devShell pins `CARGO_TARGET_DIR` to the parent checkout's `target/`; left inherited
        // it redirects the guests' wasm out of `guests/target/` (where the harnesses read them).
        // The guests are their own workspace, so clear it and let cargo default to `guests/target/`.
        .env_remove("CARGO_TARGET_DIR")
        // The devShell also exports `RUSTC_WRAPPER=sccache`; an env wrapper OVERRIDES the guests
        // workspace's config-wired `rustc-wrapper` reproducibility shim (`guest-rustc-shim.sh`),
        // so strip it to keep the shim authoritative.
        .env_remove("RUSTC_WRAPPER")
        .env("RUSTFLAGS", remap_rustflags())
        .args(["build", "--release", "--target", "wasm32-unknown-unknown"])
        .status()
        .map_err(|e| format!("spawn cargo for guests (dev shell provides the wasm target): {e}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("building guest modules failed with {status}"))
    }
}

/// Build the guests workspace at most once per process (the shared replacement for the per-file
/// `Once` copies). The first caller's outcome is cached, so a failed build fails every dependent
/// test instead of silently proceeding.
///
/// # Errors
/// See [`build_guests`].
pub fn ensure_built() -> Result<(), String> {
    static BUILT: OnceLock<Result<(), String>> = OnceLock::new();
    BUILT.get_or_init(build_guests).clone()
}

/// The path of a built guest module (`<guests>/target/wasm32-unknown-unknown/release/<name>.wasm`),
/// building the workspace first if this process has not yet.
///
/// # Panics
/// If the guest build fails (the harness convention: a broken guest build is a suite failure).
#[must_use]
pub fn built_module_path(name: &str) -> PathBuf {
    ensure_built().unwrap_or_else(|e| panic!("{e}"));
    guests_root().join(format!("target/wasm32-unknown-unknown/release/{name}.wasm"))
}

/// Read a built guest module's bytes (building first if needed), under the build guard — so a
/// concurrent straggler build can never clobber the bytes mid-read.
///
/// # Errors
/// A human-readable description of the build or read failure.
pub fn module_bytes(name: &str) -> Result<Vec<u8>, String> {
    ensure_built()?;
    let path = guests_root().join(format!("target/wasm32-unknown-unknown/release/{name}.wasm"));
    let _guard = build_guard();
    std::fs::read(&path).map_err(|e| format!("read {}: {e}", path.display()))
}

/// [`module_bytes`], panicking on failure (the harness convention).
///
/// # Panics
/// If the guest build or the artifact read fails.
#[must_use]
pub fn guest_wasm(name: &str) -> Vec<u8> {
    module_bytes(name).unwrap_or_else(|e| panic!("{e}"))
}
