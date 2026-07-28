// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The **fit-probe directory contract** — the on-disk seam between a probe orchestrator and the
//! worker binary's probe mode (`DAEMON_TRAIN_FIT_PROBE`).
//!
//! The probe that produces a [`FitVerdict`](crate::verdict::FitVerdict) must run inside the very
//! binary the ceremony will run — the verdict's backend-revision digest names that binary's
//! sealed identity, and a probe hosted anywhere else would record evidence about an executable
//! nobody executes. But the production worker build carries **no round vocabulary** (the
//! opaque-host boundary: the SDK schema crates are absent from its normal dependency graph,
//! dep-check-enforced), so the worker cannot *author* the frames and staged batches a drive
//! needs. This contract is the resolution: an orchestrator that already owns the vocabulary
//! (the testkit's authoring seam, driven by `xtask vhc-fit-probe`) writes a directory of opaque
//! artifacts, and the worker's probe mode consumes them exactly the way production frames reach
//! it — as bytes it never decodes.
//!
//! ## Directory layout
//!
//! | entry | contents |
//! |---|---|
//! | [`MODULE_FILE`] | the module wasm to probe (its blake3 keys the verdict) |
//! | [`CONFIG_FILE`] | the role's opaque canonical-CBOR config (the geometry) |
//! | [`REQUIREMENTS_FILE`] | canonical `RoleExecutionRequirements` (the run's certification policy + frozen grant) |
//! | [`OPEN_FRAME_FILE`] | the one pre-authored control frame the drive delivers (a round open) |
//! | [`STAGE_DIR`]`/`… | pre-authored staged payloads (host-staged batches), delivered in name order |
//! | [`DRIVE_FILE`] | the [`FitProbeDrive`] — how to drive and when the probe is complete |
//! | `fit-verdict-<key-digest>.cbor` | **output**: the canonical [`FitVerdict`](crate::verdict::FitVerdict) |

use serde::{Deserialize, Serialize};

/// The module wasm under probe.
pub const MODULE_FILE: &str = "module.wasm";
/// The role's opaque canonical-CBOR configuration.
pub const CONFIG_FILE: &str = "config.cbor";
/// The canonical `RoleExecutionRequirements` the admission composes under.
pub const REQUIREMENTS_FILE: &str = "requirements.cbor";
/// The pre-authored control frame the drive delivers after staging.
pub const OPEN_FRAME_FILE: &str = "open-frame.cbor";
/// The staged-payload directory; entries are delivered in ascending file-name order.
pub const STAGE_DIR: &str = "stage";
/// The drive parameters ([`FitProbeDrive`], canonical CBOR).
pub const DRIVE_FILE: &str = "drive.cbor";
/// The verdict output file-name prefix; the suffix is the probe key's digest in hex.
pub const VERDICT_FILE_PREFIX: &str = "fit-verdict-";

/// How the worker's probe mode drives the admitted instance, as **pure data** — the completion
/// condition is a `(tag, round)` pair the orchestrator states, so the worker compares two
/// integers it was handed instead of linking a message schema.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FitProbeDrive {
    /// The envelope-level role label being probed (the reservation's scope).
    pub role: String,
    /// The leading integer of the published payload that marks the probe complete
    /// (the module's committed-container voice).
    pub commit_tag: u64,
    /// The round member of that same payload head.
    pub commit_round: u64,
    /// Wall-clock deadline for the whole drive, seconds. A run that neither completes nor ends
    /// inside it is a probe FAILURE (no verdict) — a wedged probe is not evidence.
    pub deadline_s: u64,
    /// The run-pinned `state_chunk_size` (what the genesis' state contract pins on a real join).
    /// `0` leaves the driver default.
    pub state_chunk_size: u64,
    /// Compute-queue depth override (the genesis role grants pin this on a real join).
    /// `None` leaves the admitted/driver value.
    pub compute_queue_depth: Option<u64>,
    /// Per-slice readback allowance override, bytes.
    pub max_readback_bytes_per_slice: Option<u64>,
    /// Live-buffer byte allowance override.
    pub max_live_buffer_bytes: Option<u64>,
    /// Live-buffer handle allowance override.
    pub max_live_buffer_handles: Option<u64>,
}

/// The verdict output path for a probe key digest, under `dir`.
#[must_use]
pub fn verdict_path(dir: &std::path::Path, key_digest_hex: &str) -> std::path::PathBuf {
    dir.join(format!("{VERDICT_FILE_PREFIX}{key_digest_hex}.cbor"))
}
