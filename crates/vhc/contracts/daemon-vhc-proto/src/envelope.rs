// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The envelope vocabulary shared by the genesis (schema-2) envelope and the coordinator's run
//! configuration, plus the signed-envelope transport wrapper.
//!
//! The schema-major-1 (v1) envelope machinery is RETIRED: its typed sections, freeze/open chain,
//! and validation are gone. A schema-1 envelope is detected by the **outer schema-major read
//! alone** ([`crate::peek_schema`] — both schemas nest the major under `[run].schema`) and meets
//! a typed refusal at every configuration seat (assess: `EnvelopeSchemaRetired`; the coordinator
//! configuration: `EnvelopeCannotConfigure`); no v1 payload decode exists or is needed for the
//! refusal. The only resolvable run description is the genesis envelope v2 ([`crate::genesis`]).

use serde::{Deserialize, Serialize};

use crate::bytes::{PeerId, Signature};

/// Who may join a run (`[run].access`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Access {
    /// Members of the authoring org.
    Org,
    /// An explicit allowlist.
    Allowlist,
    /// Open enrolment.
    Open,
}

/// `global_batch` — sequences per round, ramped linearly over `ramp_rounds` (consumed by the
/// coordinator's run configuration).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GlobalBatch {
    /// Starting sequences-per-round.
    pub start: u32,
    /// Final sequences-per-round.
    pub end: u32,
    /// Rounds over which to ramp `start` → `end`.
    pub ramp_rounds: u32,
}

/// `stop` — the `Finished` trigger (§6.2), evaluated at round boundaries.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopCondition {
    /// Terminate after a target token count.
    Tokens(u64),
    /// Terminate after a fixed number of rounds.
    Rounds(u64),
}

/// The `device_min` role section (ABI §9.3 `device-minimums`; raw bytes and bps, never MB/Mbps):
/// the run author's device floor, evaluated at funnel stage 3 **before the module is
/// downloaded**. All fields optional — absent means "no constraint". `gpu` follows the lane
/// convention: 0 = forbidden, 1 = optional, 2 = required.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceMinimums {
    /// 0 = forbidden, 1 = optional, 2 = required.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu: Option<u64>,
    /// Minimum device memory, bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vram_bytes: Option<u64>,
    /// Minimum host RAM, bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ram_bytes: Option<u64>,
    /// Minimum free disk, bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub disk_bytes: Option<u64>,
    /// Minimum uplink, bits/s.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub up_bps: Option<u64>,
    /// Minimum downlink, bits/s.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub down_bps: Option<u64>,
    /// Acceptable device backend classes (e.g. `"cuda"`, `"vulkan"`); empty means "no
    /// constraint" (ABI §9.3 `backend_class`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub backend_class: Vec<String>,
}

/// The serializable transport form of a frozen envelope: the canonical envelope bytes plus the
/// author's signature + signer — the shape that crosses an opaque byte seam (e.g. the worker
/// protocol's `AssessRun { envelope }`). The receiver routes on the **outer schema-major read**
/// ([`crate::peek_schema`]) and reopens a genesis (schema-2) payload with
/// [`crate::FrozenGenesis::open`], which re-derives the hash and verifies the signature; any
/// other schema major meets a typed refusal before any payload decode.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignedEnvelope {
    /// The canonical CBOR bytes of the resolved envelope.
    pub bytes: Vec<u8>,
    /// The author's ed25519 signature over the envelope hash.
    pub signature: Signature,
    /// The author's node identity.
    pub signer: PeerId,
}
