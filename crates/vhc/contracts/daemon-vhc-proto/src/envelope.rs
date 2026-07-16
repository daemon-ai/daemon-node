// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The run envelope: schema, validation, and the freeze → hash → sign / verify chain (spec §6.1,
//! §16; TDD PROTO-11).
//!
//! The TOML in spec §6.1 is the *authoring* surface only. The types here model the **resolved**
//! envelope; the field names mirror the TOML tables (`[run]`, `[experiment]`, `[artifacts]`,
//! `[data]`, `[requirements]`, `[phases]`) because that TOML is normative. Freezing is:
//!
//! 1. `validate()` — reject unknown schema majors and dangling artifact references;
//! 2. serialize the whole resolved envelope to **canonical CBOR** ([`crate::canonical`]);
//! 3. the envelope hash is blake3 over those bytes; the signature covers that hash;
//! 4. `[experiment.config]`'s canonical sub-encoding is byte-identically what `da_build` receives —
//!    and, because canonical encoding emits a map value contiguously, it is a literal **subslice**
//!    of the frozen bytes: one unambiguous byte chain from author signature to guest input.
//!
//! This crate never parses TOML (it stays `wasm32`-clean); layering/authoring lives in the node's
//! figment config. `[experiment.config]` is carried as an opaque CBOR value — displayed raw, never
//! interpreted (the seam rule, §4.3).

use serde::{Deserialize, Serialize};

use crate::bytes::{Hash, PeerId, Signature};
use crate::canonical::to_canonical_vec;
use crate::error::SwarmProtoError;
use crate::hash::blake3_hash;
use crate::sign::{peer_id, sign_canonical, verify_canonical, SigningKey};

/// The envelope schema major this build understands (spec §16 — `[run].schema`).
pub const ENVELOPE_SCHEMA_MAJOR: u32 = 1;

/// Who may join a run (`[run].access`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Access {
    /// Members of the authoring org.
    Org,
    /// An explicit allowlist.
    Allowlist,
    /// Open enrolment (v2).
    Open,
}

/// The apply cadence of a round (`[phases].round_mode`, spec §6.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoundMode {
    /// v1: ingest at the round boundary as a barrier (invariant I2).
    Barrier,
    /// Reserved: one-round-delayed apply; the module manifest must declare support.
    Pipelined,
}

/// `[run]` — identity + membership.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunSection {
    /// Envelope schema major (§16).
    pub schema: u32,
    /// Stable run identifier.
    pub run_id: String,
    /// Minimum healthy peers to leave `WaitingForMembers` (§6.2).
    pub min_peers: u32,
    /// Roster ceiling.
    pub max_peers: u32,
    /// Admission policy.
    pub access: Access,
}

/// `[experiment]` — carried by the swarm, interpreted only by the module (§5, §4.3).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExperimentSection {
    /// Artifact name of the wasm module (a key in [`Envelope::artifacts`]).
    pub module: String,
    /// Tensor-ABI major the module targets (e.g. `tensor-abi@1`).
    pub abi: String,
    /// `[experiment.config]` — opaque bytes handed to `da_build`; never interpreted by the swarm.
    pub config: ciborium::value::Value,
}

/// A single external object: named, pinned by content hash, host-fetched (`[artifacts]`, §8).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Artifact {
    /// Source URL (`r2://` / `hf://@rev` / `https://`).
    pub url: String,
    /// blake3 content hash the host verifies on fetch.
    pub blake3: Hash,
}

/// `[data].global_batch` — sequences per round, ramped linearly over `ramp_rounds`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GlobalBatch {
    /// Starting sequences-per-round.
    pub start: u32,
    /// Final sequences-per-round.
    pub end: u32,
    /// Rounds over which to ramp `start` → `end`.
    pub ramp_rounds: u32,
}

/// `[data].stop` — the `Finished` trigger (§6.2), evaluated at round boundaries.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopCondition {
    /// Terminate after a target token count.
    Tokens(u64),
    /// Terminate after a fixed number of rounds.
    Rounds(u64),
}

/// `[data]` — the coordination-consumed schedule (assignment math, §6.3).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DataSection {
    /// Artifact name of the data manifest (a key in [`Envelope::artifacts`]).
    pub manifest: String,
    /// Inner steps per round (H) — module-derived, copied at freeze (§6.1).
    pub steps_per_round: u32,
    /// Sequences per round schedule.
    pub global_batch: GlobalBatch,
    /// Termination condition.
    pub stop: StopCondition,
}

/// `[requirements]` — published guidance; peers re-derive locally, never trust it (§6.5).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Requirements {
    /// Minimum VRAM in MiB.
    pub vram_mb_min: u32,
    /// Minimum host RAM in GiB.
    pub ram_gb_min: u32,
    /// Minimum uplink in Mbps.
    pub uplink_mbps_min: u32,
    /// Minimum downlink in Mbps.
    pub downlink_mbps_min: u32,
    /// Minimum free disk in GiB.
    pub disk_gb_min: u32,
    /// Measured tokens/s class floor (§6.3), e.g. `"c2"`.
    pub throughput_floor: String,
    /// Per-peer round-payload cap in MiB, receive-side enforced (§7.3).
    pub update_mb_max: u32,
    /// The module's static import list; peers re-derive at assess (§6.5).
    pub capabilities: Vec<String>,
    /// Bulk payload plane (§7.1), e.g. `"r2"`.
    pub payload_store: String,
}

/// `[phases]` — round-protocol parameters (timeouts in seconds; §6.2/§6.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Phases {
    /// Apply cadence.
    pub round_mode: RoundMode,
    /// Warmup timeout.
    pub warmup: u32,
    /// Max training time per round.
    pub round_train_max: u32,
    /// Witness grace window.
    pub round_witness: u32,
    /// Cooldown duration.
    pub cooldown: u32,
    /// Rounds per epoch (roster-stable span).
    pub epoch_rounds: u32,
    /// Checkpoint cadence in epochs.
    pub checkpoint_every_epochs: u32,
    /// Fetch-recovery budget before a peer must leave (§6.4).
    pub stall_rounds_max: u32,
    /// R2 lifecycle floor (≥ `stall_rounds_max` + resync window, §9).
    pub payload_retention_rounds: u32,
}

/// A resolved run envelope (spec §6.1).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Envelope {
    /// `[run]`.
    pub run: RunSection,
    /// `[experiment]`.
    pub experiment: ExperimentSection,
    /// `[artifacts]` — name → pinned object.
    pub artifacts: std::collections::BTreeMap<String, Artifact>,
    /// `[data]`.
    pub data: DataSection,
    /// `[requirements]`.
    pub requirements: Requirements,
    /// `[phases]`.
    pub phases: Phases,
}

impl Envelope {
    /// Validate the resolved envelope against the schema rules the coordinator enforces (§6.1, §16):
    /// a known schema major, a sane peer floor/ceiling, and no dangling artifact references.
    pub fn validate(&self) -> Result<(), SwarmProtoError> {
        if self.run.schema != ENVELOPE_SCHEMA_MAJOR {
            return Err(SwarmProtoError::Validation(format!(
                "unknown envelope schema major {} (this build understands {ENVELOPE_SCHEMA_MAJOR})",
                self.run.schema
            )));
        }
        if self.run.min_peers == 0 {
            return Err(SwarmProtoError::Validation("min_peers must be >= 1".into()));
        }
        if self.run.max_peers < self.run.min_peers {
            return Err(SwarmProtoError::Validation(
                "max_peers must be >= min_peers".into(),
            ));
        }
        if !self.artifacts.contains_key(&self.experiment.module) {
            return Err(SwarmProtoError::Validation(format!(
                "experiment.module `{}` is not present in [artifacts]",
                self.experiment.module
            )));
        }
        if !self.artifacts.contains_key(&self.data.manifest) {
            return Err(SwarmProtoError::Validation(format!(
                "data.manifest `{}` is not present in [artifacts]",
                self.data.manifest
            )));
        }
        Ok(())
    }

    /// Freeze the envelope: validate, serialize to canonical CBOR, hash (blake3), and sign the hash
    /// with the run author's key. The returned [`FrozenEnvelope`] is the only form peers and the
    /// coordinator ever see (§6.1).
    pub fn freeze(&self, key: &SigningKey) -> Result<FrozenEnvelope, SwarmProtoError> {
        self.validate()?;
        let bytes = to_canonical_vec(self)?;
        let hash = blake3_hash(&bytes);
        let config_bytes = to_canonical_vec(&self.experiment.config)?;
        let signature = sign_canonical(key, &hash)?;
        Ok(FrozenEnvelope {
            bytes,
            hash,
            config_bytes,
            signature,
            signer: peer_id(key),
        })
    }
}

/// The additive `device_min` envelope section (ABI §9.3 `device-minimums`; raw bytes and bps,
/// never MB/Mbps): the run author's device floor, evaluated at funnel stage 3 **before the
/// module is downloaded**. All fields optional — absent means "no constraint". `gpu` follows the
/// lane convention: 0 = forbidden, 1 = optional, 2 = required.
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
    /// constraint" (ABI §9.3 `backend_class`). Additive: absent in the v1 `device_min` section.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub backend_class: Vec<String>,
}

/// A frozen, hashed, signed envelope — the immutable run snapshot (spec §6.1, §11.3).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrozenEnvelope {
    bytes: Vec<u8>,
    hash: Hash,
    config_bytes: Vec<u8>,
    signature: Signature,
    signer: PeerId,
}

impl FrozenEnvelope {
    /// Reconstruct a frozen envelope from bytes received over the wire, verifying the signature.
    /// The canonical form is re-derived so a peer never trusts a supplied hash or config slice.
    pub fn open(
        bytes: Vec<u8>,
        signature: Signature,
        signer: PeerId,
    ) -> Result<Self, SwarmProtoError> {
        let envelope: Envelope = crate::canonical::from_canonical_slice(&bytes)?;
        envelope.validate()?;
        let config_bytes = to_canonical_vec(&envelope.experiment.config)?;
        let hash = blake3_hash(&bytes);
        let frozen = Self {
            bytes,
            hash,
            config_bytes,
            signature,
            signer,
        };
        frozen.verify()?;
        Ok(frozen)
    }

    /// The canonical CBOR bytes of the resolved envelope.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// The blake3 envelope hash (the signed digest).
    #[must_use]
    pub fn hash(&self) -> &Hash {
        &self.hash
    }

    /// The canonical CBOR of `[experiment.config]` — byte-identically the `da_build` input, and a
    /// subslice of [`FrozenEnvelope::bytes`].
    #[must_use]
    pub fn config_bytes(&self) -> &[u8] {
        &self.config_bytes
    }

    /// The author's ed25519 signature over [`FrozenEnvelope::hash`].
    #[must_use]
    pub fn signature(&self) -> &Signature {
        &self.signature
    }

    /// The author's node identity.
    #[must_use]
    pub fn signer(&self) -> &PeerId {
        &self.signer
    }

    /// Decode the resolved envelope from the frozen bytes.
    pub fn decode(&self) -> Result<Envelope, SwarmProtoError> {
        crate::canonical::from_canonical_slice(&self.bytes)
    }

    /// The additive host-readable `device_min` section (ABI §9.3 pre-D0; D3 cell 5), parsed from
    /// the **raw frozen bytes at the CBOR-value level** — never through the typed [`Envelope`],
    /// which silently discards unknown keys (the exact decode→re-freeze trap the cell-5 provisos
    /// guard against). `None` when the envelope carries no such section (the pre-A2 common case);
    /// a malformed section is `None` too — an author who wants pre-screening writes it correctly.
    #[must_use]
    pub fn device_min(&self) -> Option<DeviceMinimums> {
        let v: ciborium::value::Value = ciborium::de::from_reader(self.bytes.as_slice()).ok()?;
        let ciborium::value::Value::Map(entries) = v else {
            return None;
        };
        let section = entries.iter().find_map(|(k, v)| match k {
            ciborium::value::Value::Text(t) if t == "device_min" => Some(v),
            _ => None,
        })?;
        let ciborium::value::Value::Map(fields) = section else {
            return None;
        };
        let uint = |name: &str| -> Option<u64> {
            fields.iter().find_map(|(k, v)| match k {
                ciborium::value::Value::Text(t) if t == name => v
                    .as_integer()
                    .and_then(|n| u64::try_from(i128::from(n)).ok()),
                _ => None,
            })
        };
        let backend_class = fields
            .iter()
            .find_map(|(k, v)| match k {
                ciborium::value::Value::Text(t) if t == "backend_class" => match v {
                    ciborium::value::Value::Array(items) => Some(
                        items
                            .iter()
                            .filter_map(|i| match i {
                                ciborium::value::Value::Text(s) => Some(s.clone()),
                                _ => None,
                            })
                            .collect::<Vec<_>>(),
                    ),
                    _ => None,
                },
                _ => None,
            })
            .unwrap_or_default();
        Some(DeviceMinimums {
            gpu: uint("gpu"),
            vram_bytes: uint("vram_bytes"),
            ram_bytes: uint("ram_bytes"),
            disk_bytes: uint("disk_bytes"),
            up_bps: uint("up_bps"),
            down_bps: uint("down_bps"),
            backend_class,
        })
    }

    /// Verify integrity: the stored hash matches blake3 of the bytes, and the signature verifies.
    pub fn verify(&self) -> Result<(), SwarmProtoError> {
        let recomputed = blake3_hash(&self.bytes);
        if recomputed != self.hash {
            return Err(SwarmProtoError::Validation(
                "envelope hash does not match its bytes".into(),
            ));
        }
        verify_canonical(&self.signer, &self.signature, &self.hash)
    }

    /// The serializable wire form of this frozen envelope (canonical bytes + author signature +
    /// signer), for transport over an opaque byte seam like the worker protocol's `AssessRun`.
    #[must_use]
    pub fn to_wire(&self) -> SignedEnvelope {
        SignedEnvelope {
            bytes: self.bytes.clone(),
            signature: self.signature,
            signer: self.signer,
        }
    }
}

/// The serializable transport form of a [`FrozenEnvelope`]: the canonical envelope bytes plus the
/// author's signature + signer. A [`FrozenEnvelope`] itself is not `Serialize` (its hash/config are
/// re-derived, never trusted from the wire), so this is the shape that crosses an opaque byte seam
/// (e.g. the worker protocol's `AssessRun { envelope }`). Reopen it with [`SignedEnvelope::open`],
/// which re-derives the hash + config and verifies the signature (`FrozenEnvelope::open`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignedEnvelope {
    /// The canonical CBOR bytes of the resolved envelope.
    pub bytes: Vec<u8>,
    /// The author's ed25519 signature over the envelope hash.
    pub signature: Signature,
    /// The author's node identity.
    pub signer: PeerId,
}

impl SignedEnvelope {
    /// Reopen into a verified [`FrozenEnvelope`] (re-derives the hash + config, checks the signature).
    pub fn open(self) -> Result<FrozenEnvelope, SwarmProtoError> {
        FrozenEnvelope::open(self.bytes, self.signature, self.signer)
    }
}
