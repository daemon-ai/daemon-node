// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The **genesis envelope v2** — the D0 schema migration (architecture §5.1; refactor §8/D0).
//!
//! The genesis envelope (`[run].schema == 2`) is the ONLY resolvable run description — the
//! schema-major-1 (v1) envelope form is retired and meets a typed refusal decided by the outer
//! schema-major read alone ([`peek_schema`]). A genesis envelope carries **mechanism only**
//! (architecture §5.1):
//!
//! - **role set** — `{role → (lane selector, module blake3, opaque config, grant list)}`
//!   ([`RoleEntry`]). Roles are envelope-level labels the host never interprets beyond the lane
//!   selection and the per-role device minimums (architecture §3.5, §5.1). The same module hash
//!   may serve several roles with different configs.
//! - **host-readable minimum device requirements per role** ([`crate::envelope::DeviceMinimums`])
//!   — the pre-screen filter (ABI §9.3), checked from the envelope alone before any module fetch;
//!   tighten-only against the selected lane (the host computes `max(lane floor, envelope
//!   minimums)` at admission).
//! - **artifact map with pinned snapshot descriptors** ([`SnapshotArtifact`]) — the mutable-source
//!   pin at the edge (architecture §3.4/§5.1): a source like `hf://repo@rev` is resolved **once,
//!   globally**, into a content-addressed descriptor whose `blake3` is committed into the genesis
//!   hash; hosts only fetch-and-verify.
//! - **opaque `Authority` configuration** ([`GenesisEnvelope::authority`]) — a raw CBOR value the
//!   host never interprets; D1's `daemon-vhc-sdk-consensus` gives it meaning (architecture §4.2).
//! - **transport selection** ([`TransportSelection`]) — which control-plane transports the hosts
//!   bring up; the always-on `DualPlane` becomes envelope config, iroh-only default (architecture
//!   §2).
//! - **identities** ([`Identities`]) — the coordinator identity/keyset and the upgrade-authority
//!   keys (architecture §5.1). The cryptographic **`RunId`** is *not* stored: it **is** the
//!   genesis hash ([`FrozenGenesis::run_id`]); the human/registry-facing **`RunLabel`** lives in
//!   [`RunSection::run_label`] (decisions D1 — the string→`RunLabel`, hash→`RunId` split).
//!
//! The host-visible `[data]` schedule and `[phases]` policy of the v1 envelope **do not exist**
//! here — they became worker-module and coordinator-module opaque config respectively (each role's
//! [`RoleEntry::config`]); admission no longer pre-screens `round_mode` (refactor §8/D0).

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::bytes::{Hash, PeerId, Seed, Signature};
use crate::canonical::{from_canonical_slice, to_canonical_vec};
use crate::envelope::{Access, DeviceMinimums};
use crate::error::VhcProtoError;
use crate::hash::blake3_hash;
use crate::sign::{peer_id, sign_canonical, verify_canonical, SigningKey};

/// The genesis-envelope schema major this build understands (architecture §5.1; the v2 cell of the
/// ratified mixed-fleet matrix — decisions D3). Distinct from the retired v1 schema major (`1`),
/// whose only surviving trace is the typed refusal keyed off the outer schema read.
pub const GENESIS_SCHEMA_MAJOR: u32 = 2;

/// A control-plane transport a run may bring up (architecture §2 — "the envelope declares which
/// transports a run uses; the default is iroh-only"). `DualPlane` stops being an always-on host
/// constant and becomes this selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ControlTransport {
    /// iroh gossip/streams (with a self-run relay) — the default.
    Iroh,
    /// WebSocket control plane.
    WebSocket,
    /// In-memory (simulation only).
    Mem,
}

/// `[transport]` — control-plane transports + the bulk payload store selector (architecture §2,
/// §7.1). `control` is ordered by preference; the default is iroh-only.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransportSelection {
    /// The control-plane transports the hosts bring up (ordered; iroh-only default).
    pub control: Vec<ControlTransport>,
    /// The bulk payload plane selector (e.g. `"r2"`, `"fs"`); §7.1.
    pub payload_store: String,
}

impl Default for TransportSelection {
    fn default() -> Self {
        Self {
            control: vec![ControlTransport::Iroh],
            payload_store: "r2".to_string(),
        }
    }
}

/// A single external object, **pinned by a content-addressed snapshot descriptor** (architecture
/// §3.4/§5.1). `url` is the author's source (`hf://repo@rev`, `r2://`, `https://`, `file://`); it
/// was resolved once at authoring time into `blake3`, the committed content hash the host verifies
/// on fetch. An unpinned `hf://` source (no `@rev`) is an authoring error the author must fix
/// before freeze — the host never resolves a mutable source itself (admission-time resolution
/// could bind different snapshots on different hosts).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnapshotArtifact {
    /// The author's source URL (already edge-resolved to a pinned revision).
    pub url: String,
    /// The committed blake3 content hash the host fetches and verifies against.
    pub blake3: Hash,
    /// The pinned byte size where the author recorded it (`None` = not pinned).
    pub size: Option<u64>,
}

/// The **init pin** of a genesis state contract ([`StateContract::init`]): how a fresh joiner
/// obtains the run's matched initial canonical state. Two shapes, both content-cross-checked:
///
/// - **Seed-derived**: the guest expands a pinned `(seed, dist)` deterministically, chunk-wise,
///   in registration order, and the sealed family fold MUST equal `expected_root` — a mismatch
///   is a typed init failure, never a silent divergence. Zero storage, zero transfer.
/// - **Content-addressed artifact**: a det-state manifest hash
///   ([`crate::det_state::DetStateManifest::state_root`]) whose family artifacts publish like
///   corpus shards; MUST also name a fetchable `[artifacts]` entry. The only shape that can
///   express a warm start, and the golden-fixture continuity carrier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StateInit {
    /// `{seed, dist, expected_root}` — deterministic seed expansion under a versioned
    /// distribution id, cross-checked against the pinned root.
    Seed {
        /// The 32-byte expansion seed.
        seed: Seed,
        /// The versioned distribution id the guest expands under (a derivation identity: any
        /// change to the expansion scheme is a new id).
        dist: u64,
        /// The family fold the sealed expansion MUST reproduce.
        expected_root: Hash,
    },
    /// `{manifest}` — the det-state manifest hash of a published init artifact.
    Manifest {
        /// blake3 of the init det-state manifest (canonical CBOR).
        manifest: Hash,
    },
}

/// The genesis **state contract** (additive key, the corpus-pin precedent): the chunk-addressed
/// canonical-state geometry + the init pin. Envelope-internal validation covers what the host
/// can judge from the envelope alone (non-degenerate geometry, a mapped artifact-form pin, a
/// pinned root); the layout-aware rules — the profile chunk dividing every parameter numel
/// ([`crate::det_state::validate_profile_chunk`]), `chunk_size` being an integer multiple of
/// the profile chunk's byte width ([`crate::det_state::validate_state_chunk_size`]), and the
/// checkpoint cadence↔retention bound ([`crate::det_state::validate_checkpoint_cadence`]) —
/// run at the authoring/admission seat that can see the (host-opaque) module config.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateContract {
    /// The state chunk size in bytes ([`crate::det_state::derive_state_chunk_size`] is the
    /// authoring derivation).
    pub chunk_size: u64,
    /// The init pin.
    pub init: StateInit,
}

/// A per-grant numeric/enumerated bound (ABI §2.3 `grant-bound`). Every field is optional; an
/// absent field means "unbounded by this grant" (still bounded by the lane ceiling at admission).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GrantBound {
    /// Per-item byte ceiling (e.g. publish payload, readback value).
    pub max_bytes: Option<u64>,
    /// Per-event-slice call ceiling for this grant.
    pub max_per_slice: Option<u64>,
    /// Sustained rate ceiling (token bucket, per minute).
    pub rate_per_min: Option<u64>,
    /// Concurrent-operation ceiling (Phase B completions).
    pub max_outstanding: Option<u64>,
    /// Enumerated allowed values (topics, dataset hashes, sources).
    pub values: Vec<String>,
}

/// A per-world grant (ABI §2.6 `world-grant`): the namespace minor + its per-grant bounds.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorldGrant {
    /// The negotiated namespace minor.
    pub minor: u64,
    /// Per-grant bounds keyed by grant name.
    pub bounds: BTreeMap<String, GrantBound>,
}

/// A channel declaration (ABI §6.2 `channel-decl`). From D0 the per-role channel table lives in
/// the envelope (the ABI surface — `publish(channel_id, …)`, `frame-ev.channel`, the §12 scope
/// tuple — is unchanged). Authoritative-channel bounds (`spool_frames`, `replay_window`,
/// `per_sender_quota`) are `None` for advisory/gossip channels.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChannelDecl {
    /// The channel id the guest selects in `publish` / matches in `frame-ev`.
    pub id: u32,
    /// A human name.
    pub name: String,
    /// Delivery class: 0 = authoritative, 1 = advisory/gossip.
    pub class: u8,
    /// Direction: 0 = rx-only, 1 = tx-only, 2 = bidirectional.
    pub direction: u8,
    /// Per-frame byte ceiling.
    pub max_frame_bytes: u64,
    /// tx token bucket, per minute.
    pub rate_per_min: u64,
    /// Authoritative-only: bounded durable spool depth.
    pub spool_frames: Option<u64>,
    /// Authoritative-only: dedup window (frames).
    pub replay_window: Option<u64>,
    /// Authoritative-only: rx per-sender outstanding quota.
    pub per_sender_quota: Option<u64>,
}

/// One advisory-class queue declaration (ABI §2.3 `event-caps`): `{depth, coalesce}` where
/// `coalesce` is the fixed per-class code (0 = dedup-by-hash, 1 = latest-wins, 2 = drop-oldest).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventCap {
    /// Bounded queue depth.
    pub depth: u64,
    /// The fixed coalescing code for this class.
    pub coalesce: u64,
}

/// The advisory-class depth/coalescing declarations (ABI §2.3 `event-caps`), keyed by class name
/// (`"payload-ready"` / `"timer"` / `"gossip"`). Authoritative channels are NOT declared here —
/// their bounds live in the channel table.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventCaps {
    /// Class → `{depth, coalesce}`.
    pub classes: BTreeMap<String, EventCap>,
}

/// Live-resource quotas (ABI §2.3 `buffer-req`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BufferReq {
    /// Standing live-resource ceiling (all instance-class resources).
    pub max_live_handles: u64,
    /// Standing live-buffer byte ceiling (Phase B buffers).
    pub max_live_bytes: u64,
    /// Per-slice ceiling on bytes crossing into linear memory.
    pub max_readback_bytes: u64,
}

/// The migration grant (ABI §2.6 `migration-grant`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MigrationGrant {
    /// Whether `read_back(kind = state-section)` restore is permitted during `da_migrate`.
    pub restore: bool,
    /// Max migration sections.
    pub max_sections: u64,
    /// Max bytes per section.
    pub max_section_bytes: u64,
}

/// The **envelope role grant list** (architecture §5.1; the "envelope role grant list" contributor
/// to the ABI §2.6 derived grants document). This is what a role's module *may* reach; the host
/// intersects it with the selected lane's ceilings and the owner's standing policy to derive the
/// admitted grants (tighten-only — a role whose grants exceed the lane's bounds is refused at
/// admission, architecture §3.5). The fields mirror the ABI grant vocabulary and map onto the
/// Phase-B `RunConfig` quotas (`payload_depth`/`advisory_depth`/`gossip_depth`, spool bounds,
/// stream credit, `granted_artifacts`).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoleGrants {
    /// Per-world grants (`"vhc"`/`"net"`/`"sys"`/`"data"`/`"compute"`).
    pub worlds: BTreeMap<String, WorldGrant>,
    /// Versioned host custom-op names required (architecture §3.2).
    pub custom_ops: Vec<String>,
    /// The role's channel table (ABI §6.2).
    pub channels: Vec<ChannelDecl>,
    /// Advisory-class depths + coalescing.
    pub events: EventCaps,
    /// Live-resource quotas.
    pub buffers: BufferReq,
    /// The granted artifact hashes — a subset of the run's [`GenesisEnvelope::artifacts`] this role
    /// may `data.fetch` ("which artifacts a module may touch is a grant", architecture §3.2).
    pub artifacts: BTreeSet<Hash>,
    /// Async-completion concurrent-operation ceiling (`grant-bound.max_outstanding`).
    pub max_outstanding_ops: u64,
    /// Cumulative `data@2` **read budget** in bytes for the whole role instance (`0` =
    /// unbounded by this grant): the total artifact bytes the module may fetch across the run —
    /// the admitted-grant bound the ratified corpus contract draws cache/read ceilings from.
    #[serde(default)]
    pub data_read_budget_bytes: u64,
    /// compute@2 command-queue depth grant (C1, ABI §15; D0∩C1 union). `0` = unspecified —
    /// inherits the lane ceiling at derivation (tighten-only, like the other quotas).
    #[serde(default)]
    pub compute_queue_depth: u64,
    /// The migration grant, when the role participates in live upgrades.
    pub migration: Option<MigrationGrant>,
}

/// One role in the run's role set (architecture §5.1): `role → (lane selector, module blake3,
/// opaque config, grant list)` plus the host-readable device minimums.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RoleEntry {
    /// The **lane selector** — which versioned `ParticipationLane` this role admits under
    /// (`"trainer"` / `"verifier"` / `"coordinator"`). The host maps this to a node-side lane
    /// profile at admission; the envelope may tighten a lane, never weaken it.
    pub lane: String,
    /// The artifact-map key of this role's wasm module (its `blake3` is the pinned module hash).
    pub module: String,
    /// The tensor/loop ABI the module targets, e.g. `"vhc@2"` (informational; the host derives the
    /// driver from the module's static imports, ABI §1.3).
    pub abi: String,
    /// The role's **opaque module config** — the worker's data schedule / the coordinator's phase
    /// policy live here now (refactor §8/D0). Never interpreted by the host (the seam rule).
    pub config: ciborium::value::Value,
    /// The role's grant list.
    pub grants: RoleGrants,
    /// The host-readable per-role device minimums (ABI §9.3), tighten-only vs the lane floor.
    pub device_min: DeviceMinimums,
}

/// `[identities]` — the run's cryptographic identities (architecture §5.1). Opaque-ish to the host
/// (it stamps the coordinator identity into certificates); D1's `Authority` gives the keyset
/// meaning.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Identities {
    /// The launch coordinator identity (`SingleKey`, architecture §4.2).
    pub coordinator: Option<PeerId>,
    /// The coordinator keyset (`ThresholdKeys`, D1); empty for `SingleKey`.
    pub coordinator_set: Vec<PeerId>,
    /// The upgrade-authority keys (architecture §5.4 two-key model).
    pub upgrade_authority: Vec<PeerId>,
}

/// `[run]` for a genesis envelope (architecture §5.1). Distinct from v1 [`crate::envelope::
/// RunSection`] in that identity is the cryptographic **`RunId`** (the genesis hash) — carried
/// here only as the human/registry-facing **`RunLabel`** string (decisions D1).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunSection {
    /// Envelope schema major — MUST be [`GENESIS_SCHEMA_MAJOR`].
    pub schema: u32,
    /// The human/registry-facing run handle (the old string `run_id`, renamed — decisions D1). The
    /// cryptographic `RunId` is the genesis hash, not stored here.
    pub run_label: String,
    /// Minimum healthy peers to leave `WaitingForMembers`.
    pub min_peers: u32,
    /// Roster ceiling.
    pub max_peers: u32,
    /// Admission policy.
    pub access: Access,
}

/// A resolved **genesis envelope** (architecture §5.1) — envelope schema v2.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GenesisEnvelope {
    /// `[run]`.
    pub run: RunSection,
    /// The role set — `role → (lane, module, opaque config, grants, device minimums)`.
    pub roles: BTreeMap<String, RoleEntry>,
    /// `[artifacts]` — name → pinned snapshot descriptor.
    pub artifacts: BTreeMap<String, SnapshotArtifact>,
    /// The pinned **corpus-manifest hash** (the chunk-addressed data root,
    /// [`crate::corpus::CorpusManifest::manifest_hash`]) for a run that trains on a published
    /// corpus. When present it MUST also be an `[artifacts]` entry (the manifest is fetched like
    /// any other pinned artifact); the pin here is what commits the run's data identity into the
    /// genesis hash. `None` for runs without a corpus (pure-consensus roles, conformance runs).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub corpus_manifest: Option<Hash>,
    /// The chunk-addressed canonical-state contract (geometry + init pin) for a run whose
    /// modules stream det-lane state through the host store. Additive, like the corpus pin;
    /// `None` for runs without host-side canonical state. The artifact-form init manifest MUST
    /// also be an `[artifacts]` entry; the pin here is what commits the run's initial-state
    /// identity into the genesis hash.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state_contract: Option<StateContract>,
    /// Opaque `Authority` configuration — interpreted by modules (D1), never by the host.
    pub authority: ciborium::value::Value,
    /// Control-plane transport selection + payload store.
    pub transport: TransportSelection,
    /// The run's cryptographic identities.
    pub identities: Identities,
}

impl GenesisEnvelope {
    /// Validate the resolved genesis envelope against the schema rules the host enforces from the
    /// envelope **alone** (architecture §5.1) — a known schema major, a sane peer floor/ceiling,
    /// at least a coordinator and a worker role, every role's module + granted artifacts present
    /// in the artifact map, and a non-empty lane selector per role.
    ///
    /// Grants-exceed-lane and device-below-floor are **admission-time** judgments (they need the
    /// node's lane profiles and device probe, architecture §3.5), not envelope-internal — they are
    /// not checked here.
    pub fn validate(&self) -> Result<(), VhcProtoError> {
        if self.run.schema != GENESIS_SCHEMA_MAJOR {
            return Err(VhcProtoError::Validation(format!(
                "unknown genesis schema major {} (this build understands \
                 {GENESIS_SCHEMA_MAJOR})",
                self.run.schema
            )));
        }
        if self.run.min_peers == 0 {
            return Err(VhcProtoError::Validation("min_peers must be >= 1".into()));
        }
        if self.run.max_peers < self.run.min_peers {
            return Err(VhcProtoError::Validation(
                "max_peers must be >= min_peers".into(),
            ));
        }
        if self.roles.is_empty() {
            return Err(VhcProtoError::Validation(
                "a genesis envelope must declare at least one role".into(),
            ));
        }
        // "Every run has at least a coordinator role and one worker role" (architecture §5.1).
        // Roles are opaque labels the host never interprets, so this checks for the two canonical
        // label prefixes only as a well-formedness floor (an author may name either explicitly).
        let has_coordinator = self.roles.keys().any(|r| r.contains("coordinator"));
        let has_worker = self.roles.keys().any(|r| !r.contains("coordinator"));
        if !has_coordinator || !has_worker {
            return Err(VhcProtoError::Validation(
                "a genesis envelope must declare at least a coordinator role and one worker role \
                 (architecture §5.1)"
                    .into(),
            ));
        }
        let artifact_hashes: BTreeSet<Hash> = self.artifacts.values().map(|a| a.blake3).collect();
        for (name, role) in &self.roles {
            if role.lane.is_empty() {
                return Err(VhcProtoError::Validation(format!(
                    "role `{name}` has an empty lane selector"
                )));
            }
            let Some(module) = self.artifacts.get(&role.module) else {
                return Err(VhcProtoError::Validation(format!(
                    "role `{name}` module `{}` is not present in [artifacts]",
                    role.module
                )));
            };
            // Snapshot descriptors must be pinned: a zero size where declared is fine, but the
            // blake3 is the pin and is always present (byte newtype). Guard the obvious authoring
            // slip of an all-zero hash on a role's module.
            if module.blake3 == Hash([0u8; 32]) {
                return Err(VhcProtoError::Validation(format!(
                    "role `{name}` module `{}` has an unpinned (all-zero) blake3",
                    role.module
                )));
            }
            for granted in &role.grants.artifacts {
                if !artifact_hashes.contains(granted) {
                    return Err(VhcProtoError::Validation(format!(
                        "role `{name}` grants artifact {} absent from the artifact map",
                        granted.to_hex()
                    )));
                }
            }
        }
        if let Some(manifest) = &self.corpus_manifest {
            if !artifact_hashes.contains(manifest) {
                return Err(VhcProtoError::Validation(format!(
                    "corpus_manifest pin {} is absent from the artifact map (the manifest must \
                     be a fetchable pinned artifact)",
                    manifest.to_hex()
                )));
            }
        }
        if let Some(contract) = &self.state_contract {
            if contract.chunk_size == 0 {
                return Err(VhcProtoError::Validation(
                    "state contract chunk_size must be > 0".into(),
                ));
            }
            match &contract.init {
                StateInit::Seed { expected_root, .. } => {
                    // Guard the authoring slip of an unpinned (all-zero) root — the seed form
                    // is only cross-checkable through this pin.
                    if *expected_root == Hash([0u8; 32]) {
                        return Err(VhcProtoError::Validation(
                            "state contract seed init has an unpinned (all-zero) expected_root"
                                .into(),
                        ));
                    }
                }
                StateInit::Manifest { manifest } => {
                    if !artifact_hashes.contains(manifest) {
                        return Err(VhcProtoError::Validation(format!(
                            "state contract init manifest {} is absent from the artifact map \
                             (the init manifest must be a fetchable pinned artifact)",
                            manifest.to_hex()
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    /// Freeze the genesis envelope: validate, serialize to canonical CBOR, hash (blake3), and sign
    /// the hash with the author's key. The hash **is** the run's cryptographic `RunId`
    /// (architecture §5.1). The returned [`FrozenGenesis`] is the only form peers and the
    /// coordinator ever see.
    pub fn freeze(&self, key: &SigningKey) -> Result<FrozenGenesis, VhcProtoError> {
        self.validate()?;
        let bytes = to_canonical_vec(self)?;
        let hash = blake3_hash(&bytes);
        let signature = sign_canonical(key, &hash)?;
        Ok(FrozenGenesis {
            bytes,
            hash,
            signature,
            signer: peer_id(key),
        })
    }
}

/// A frozen, hashed, signed genesis envelope — the immutable run snapshot whose hash is the
/// cryptographic `RunId` (architecture §5.1).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrozenGenesis {
    bytes: Vec<u8>,
    hash: Hash,
    signature: Signature,
    signer: PeerId,
}

impl FrozenGenesis {
    /// Reconstruct from bytes received over the wire, verifying the signature. The canonical form
    /// is re-derived so a peer never trusts a supplied hash (the hash is always recomputed over
    /// the received bytes).
    pub fn open(
        bytes: Vec<u8>,
        signature: Signature,
        signer: PeerId,
    ) -> Result<Self, VhcProtoError> {
        let envelope: GenesisEnvelope = from_canonical_slice(&bytes)?;
        envelope.validate()?;
        let hash = blake3_hash(&bytes);
        let frozen = Self {
            bytes,
            hash,
            signature,
            signer,
        };
        frozen.verify()?;
        Ok(frozen)
    }

    /// The canonical CBOR bytes of the resolved genesis envelope.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// The blake3 genesis hash — **this is the cryptographic `RunId`** (architecture §5.1; ABI
    /// §8.1 `run_id: bstr .size 32`).
    #[must_use]
    pub fn run_id(&self) -> &Hash {
        &self.hash
    }

    /// The human/registry-facing `RunLabel` (decisions D1). When a node carries both a `RunLabel`
    /// and a `RunId`, they must agree — a `RunLabel` resolving to a different genesis hash is a
    /// typed refusal (that cross-check is a host/node concern; this accessor is the label side).
    pub fn run_label(&self) -> Result<String, VhcProtoError> {
        Ok(self.decode()?.run.run_label)
    }

    /// The author's ed25519 signature over [`FrozenGenesis::run_id`].
    #[must_use]
    pub fn signature(&self) -> &Signature {
        &self.signature
    }

    /// The author's node identity.
    #[must_use]
    pub fn signer(&self) -> &PeerId {
        &self.signer
    }

    /// Decode the resolved genesis envelope from the frozen bytes.
    pub fn decode(&self) -> Result<GenesisEnvelope, VhcProtoError> {
        from_canonical_slice(&self.bytes)
    }

    /// The canonical CBOR of a role's opaque module config — byte-identically what that role's
    /// module receives as its `da_init`/`da_build` config input (the per-role analogue of v1's
    /// single `[experiment.config]`). `None` for an unknown role.
    pub fn role_config_bytes(&self, role: &str) -> Result<Option<Vec<u8>>, VhcProtoError> {
        let env = self.decode()?;
        match env.roles.get(role) {
            Some(entry) => Ok(Some(to_canonical_vec(&entry.config)?)),
            None => Ok(None),
        }
    }

    /// Verify integrity: the stored hash matches blake3 of the bytes, and the signature verifies.
    pub fn verify(&self) -> Result<(), VhcProtoError> {
        let recomputed = blake3_hash(&self.bytes);
        if recomputed != self.hash {
            return Err(VhcProtoError::Validation(
                "genesis hash does not match its bytes".into(),
            ));
        }
        verify_canonical(&self.signer, &self.signature, &self.hash)
    }
}

/// Peek at the `[run].schema` major of a frozen envelope's raw bytes **without** committing to a
/// typed decode — the outer schema-major read every configuration seat routes on: schema 2
/// resolves as a genesis envelope; any other major (notably the retired schema-1 form) meets a
/// typed refusal with no payload decode. Returns `None` if the bytes are not a CBOR map or carry
/// no integer `schema` key.
#[must_use]
pub fn peek_schema(bytes: &[u8]) -> Option<u32> {
    let v: ciborium::value::Value = ciborium::de::from_reader(bytes).ok()?;
    let ciborium::value::Value::Map(entries) = v else {
        return None;
    };
    // Both v1 `Envelope` and v2 `GenesisEnvelope` nest schema under `[run].schema`.
    let run = entries.iter().find_map(|(k, val)| match k {
        ciborium::value::Value::Text(t) if t == "run" => Some(val),
        _ => None,
    })?;
    let ciborium::value::Value::Map(run_fields) = run else {
        return None;
    };
    run_fields.iter().find_map(|(k, val)| match k {
        ciborium::value::Value::Text(t) if t == "schema" => val
            .as_integer()
            .and_then(|n| u32::try_from(i128::from(n)).ok()),
        _ => None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sign::SigningKey;

    fn hash(n: u8) -> Hash {
        Hash([n; 32])
    }

    fn key() -> SigningKey {
        SigningKey::from_bytes(&[7u8; 32])
    }

    fn sample() -> GenesisEnvelope {
        let mut artifacts = BTreeMap::new();
        artifacts.insert(
            "worker-mod".to_string(),
            SnapshotArtifact {
                url: "r2://mods/worker.wasm".into(),
                blake3: hash(1),
                size: Some(4096),
            },
        );
        artifacts.insert(
            "coord-mod".to_string(),
            SnapshotArtifact {
                url: "r2://mods/coord.wasm".into(),
                blake3: hash(2),
                size: Some(2048),
            },
        );
        artifacts.insert(
            "corpus".to_string(),
            SnapshotArtifact {
                url: "hf://org/corpus@abc123".into(),
                blake3: hash(3),
                size: None,
            },
        );

        let mut roles = BTreeMap::new();
        let mut worker_grants = RoleGrants::default();
        worker_grants.artifacts.insert(hash(3));
        worker_grants.max_outstanding_ops = 16;
        roles.insert(
            "worker".to_string(),
            RoleEntry {
                lane: "trainer".into(),
                module: "worker-mod".into(),
                abi: "vhc@2".into(),
                config: ciborium::value::Value::Map(vec![]),
                grants: worker_grants,
                device_min: DeviceMinimums {
                    gpu: Some(2),
                    vram_bytes: Some(16 << 30),
                    backend_class: vec!["cuda".into()],
                    ..Default::default()
                },
            },
        );
        roles.insert(
            "coordinator".to_string(),
            RoleEntry {
                lane: "coordinator".into(),
                module: "coord-mod".into(),
                abi: "vhc@2".into(),
                config: ciborium::value::Value::Map(vec![]),
                grants: RoleGrants::default(),
                device_min: DeviceMinimums::default(),
            },
        );

        GenesisEnvelope {
            run: RunSection {
                schema: GENESIS_SCHEMA_MAJOR,
                run_label: "demo-run".into(),
                min_peers: 1,
                max_peers: 64,
                access: Access::Org,
            },
            roles,
            artifacts,
            corpus_manifest: None,
            state_contract: None,
            authority: ciborium::value::Value::Map(vec![]),
            transport: TransportSelection::default(),
            identities: Identities::default(),
        }
    }

    #[test]
    fn freeze_open_roundtrip_and_run_id_is_hash() {
        let env = sample();
        let frozen = env.freeze(&key()).unwrap();
        let wire = frozen.bytes().to_vec();
        let reopened = FrozenGenesis::open(wire, *frozen.signature(), *frozen.signer()).unwrap();
        assert_eq!(reopened.run_id(), frozen.run_id());
        assert_eq!(reopened.decode().unwrap(), env);
        assert_eq!(reopened.run_label().unwrap(), "demo-run");
    }

    #[test]
    fn schema_sniff_distinguishes_retired_v1_from_genesis() {
        let frozen = sample().freeze(&key()).unwrap();
        assert_eq!(peek_schema(frozen.bytes()), Some(GENESIS_SCHEMA_MAJOR));
        // The retired v1 form is detected by the outer schema-major read alone: a synthetic
        // canonical-CBOR map carrying `[run].schema = 1` sniffs as major 1 — the input every
        // typed schema-retired refusal keys on, with no v1 payload machinery involved.
        let run = ciborium::value::Value::Map(vec![(
            ciborium::value::Value::Text("schema".into()),
            ciborium::value::Value::from(1u32),
        )]);
        let v1 =
            ciborium::value::Value::Map(vec![(ciborium::value::Value::Text("run".into()), run)]);
        let bytes = crate::to_canonical_vec(&v1).unwrap();
        assert_eq!(peek_schema(&bytes), Some(1));
        assert_eq!(peek_schema(b"not cbor"), None);
    }

    #[test]
    fn tamper_breaks_hash_and_signature() {
        let frozen = sample().freeze(&key()).unwrap();
        let mut bytes = frozen.bytes().to_vec();
        bytes[0] ^= 0xff;
        assert!(FrozenGenesis::open(bytes, *frozen.signature(), *frozen.signer()).is_err());
    }

    #[test]
    fn validate_requires_coordinator_and_worker() {
        let mut env = sample();
        env.roles.remove("coordinator");
        assert!(env.validate().is_err());
    }

    #[test]
    fn validate_rejects_grant_of_unmapped_artifact() {
        let mut env = sample();
        env.roles
            .get_mut("worker")
            .unwrap()
            .grants
            .artifacts
            .insert(hash(99));
        assert!(env.validate().is_err());
    }

    #[test]
    fn validate_rejects_module_absent_from_artifacts() {
        let mut env = sample();
        env.roles.get_mut("worker").unwrap().module = "nope".into();
        assert!(env.validate().is_err());
    }

    /// The state contract is additive: absent it changes nothing; present it commits into the
    /// genesis hash, round-trips through the frozen form in both init shapes, and its
    /// envelope-internal rules refuse degenerate geometry, an unpinned seed root, and an
    /// unmapped artifact-form init manifest.
    #[test]
    fn state_contract_commits_into_the_hash_and_validates_envelope_internal_rules() {
        let baseline_id = *sample().freeze(&key()).unwrap().run_id();

        // Seed form: valid, hash-committed, round-trips.
        let mut env = sample();
        env.state_contract = Some(StateContract {
            chunk_size: 4 << 20,
            init: StateInit::Seed {
                seed: Seed([5u8; 32]),
                dist: 1,
                expected_root: hash(9),
            },
        });
        let frozen = env.freeze(&key()).unwrap();
        assert_ne!(
            *frozen.run_id(),
            baseline_id,
            "the contract is hash-committed"
        );
        assert_eq!(
            frozen.decode().unwrap().state_contract,
            env.state_contract,
            "seed form round-trips through the frozen form"
        );

        // Artifact form: must name a mapped artifact; mapped it round-trips.
        let mut env = sample();
        env.state_contract = Some(StateContract {
            chunk_size: 4 << 20,
            init: StateInit::Manifest { manifest: hash(99) },
        });
        assert!(env.validate().is_err(), "unmapped init manifest refused");
        env.state_contract = Some(StateContract {
            chunk_size: 4 << 20,
            init: StateInit::Manifest { manifest: hash(3) },
        });
        let frozen = env.freeze(&key()).unwrap();
        assert_eq!(frozen.decode().unwrap().state_contract, env.state_contract);

        // Degenerate geometry and an unpinned seed root are authoring errors.
        let mut env = sample();
        env.state_contract = Some(StateContract {
            chunk_size: 0,
            init: StateInit::Manifest { manifest: hash(3) },
        });
        assert!(env.validate().is_err(), "zero chunk_size refused");
        let mut env = sample();
        env.state_contract = Some(StateContract {
            chunk_size: 4 << 20,
            init: StateInit::Seed {
                seed: Seed([5u8; 32]),
                dist: 1,
                expected_root: Hash([0u8; 32]),
            },
        });
        assert!(env.validate().is_err(), "all-zero expected_root refused");
    }

    /// The corpus-manifest pin must name a fetchable `[artifacts]` entry; a mapped pin commits
    /// into the genesis hash (a different pin is a different `RunId`).
    #[test]
    fn corpus_manifest_pin_must_be_a_mapped_artifact_and_commits_into_the_hash() {
        let mut env = sample();
        env.corpus_manifest = Some(hash(99));
        assert!(env.validate().is_err(), "unmapped pin refused");

        // Pin the mapped corpus artifact: valid, and the genesis hash moves with the pin.
        let unpinned_id = *sample().freeze(&key()).unwrap().run_id();
        env.corpus_manifest = Some(hash(3));
        let frozen = env.freeze(&key()).unwrap();
        assert_ne!(*frozen.run_id(), unpinned_id, "the pin is hash-committed");
        assert_eq!(
            frozen.decode().unwrap().corpus_manifest,
            Some(hash(3)),
            "the pin round-trips through the frozen form"
        );
    }
}
