// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The typed engine snapshot (lifecycle §2) — the *only* durable engine state.
//!
//! Everything an incarnation needs to be reconstructed: its identity, epoch, the typed
//! [`Conversation`], the [`References`] to re-establish on rehydration (never live resources), and
//! the outstanding background work it suspended for. Persisted as opaque CBOR
//! ([`SnapshotBlob`](daemon_common::SnapshotBlob)) so the durable substrate stays engine-agnostic.

use crate::approval::ApprovalPolicy;
use crate::conversation::{Conversation, ToolCall};
use daemon_common::{DaemonError, Epoch, JobId, SessionId, SnapshotBlob};
use serde::{Deserialize, Serialize};

/// A host-owned OS process handle, re-attached by the host on rehydration (lifecycle §2).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcHandle {
    /// Opaque host-assigned process key.
    pub key: String,
}

/// A tool identity plus the key the tool uses to reload its own external state (lifecycle §1.2).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolBinding {
    /// The tool's stable name.
    pub name: String,
    /// The key the tool reloads its own state from.
    pub state_key: String,
}

/// Handles to re-establish on rehydration — never live resources (lifecycle §2).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct References {
    /// Delegated child engines, by id (recursive composition).
    pub children: Vec<SessionId>,
    /// Host-owned OS processes, re-attached by the host.
    pub processes: Vec<ProcHandle>,
    /// Tool identities + the keys tools use to reload their own state.
    pub tools: Vec<ToolBinding>,
}

/// The complete, serializable state of one engine incarnation. Nothing else is durable
/// (lifecycle §2). Tool working state and live resources are never included.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Snapshot {
    /// Stable logical identity (not a live task handle).
    pub session_id: SessionId,
    /// Monotonic epoch; bumped on every suspension; fences stale incarnations.
    pub epoch: Epoch,
    /// The typed conversation body (source of truth).
    pub conversation: Conversation,
    /// Handles to re-establish on rehydration.
    pub references: References,
    /// Outstanding background work this incarnation suspended for.
    pub waiting_for: Vec<JobId>,
    /// Tool iterations accumulated since the last skill review / `skill_manage` use. Drives the
    /// engine-native skill-review nudge ([`Config::skill_review_interval`](crate::Config)); durable
    /// so the cadence survives suspension. `#[serde(default)]` keeps pre-existing snapshots decodable.
    #[serde(default)]
    pub iters_since_skill: u32,
    /// Completed turns since the last memory review / memory write. Drives the engine-native
    /// memory-review nudge ([`Config::memory_review_interval`](crate::Config)); durable across
    /// suspension. `#[serde(default)]` keeps pre-existing snapshots decodable.
    #[serde(default)]
    pub turns_since_memory: u32,
    /// The session's explicit edit-approval policy (§12 session mode), set via `SetSessionMode`.
    /// `None` falls back to the engine [`Config::approval_policy`](crate::Config). Durable so a
    /// supervised session's mode survives suspension/restart. `#[serde(default)]` keeps pre-existing
    /// snapshots decodable.
    #[serde(default)]
    pub approval_policy: Option<ApprovalPolicy>,
    /// Gated tool calls this incarnation suspended on, awaiting a durable operator decision (§12
    /// HITL). Each carries the original [`ToolCall`] so the engine can re-run it verbatim on
    /// approval (allow -> execute, deny -> tool-error). Durable so a parked approval survives
    /// restart; cleared as each is resolved. `#[serde(default)]` keeps pre-existing snapshots
    /// decodable.
    #[serde(default)]
    pub pending_approvals: Vec<PendingApproval>,
    /// Provenance: the node build that last wrote this snapshot (`daemon_common::VERSION`, e.g.
    /// `0.0.1+g1a2b3c4`). Stamped on [`Snapshot::encode`], not a migration key — the snapshot format
    /// itself evolves via `#[serde(default)]`. Empty on snapshots written before this field existed.
    /// [`Snapshot::decode`] warns (but still decodes) when a snapshot was written by a newer build.
    #[serde(default)]
    pub writer_version: String,
}

/// A gated tool call parked for a durable human-in-the-loop decision (§12). Persisted on the
/// [`Snapshot`] so the engine can re-run the exact call once the operator answers.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PendingApproval {
    /// The decision's correlation id (the suspension job id); the operator answers by this id and
    /// the completion that wakes the session carries it.
    pub job_id: JobId,
    /// The original tool call to re-run verbatim on approval.
    pub call: ToolCall,
    /// The human-readable approval prompt (diff summary / command), surfaced to the operator.
    pub prompt: String,
    /// The target path for an fs edit (used for the sensitive-path carve-out + display), if any.
    #[serde(default)]
    pub path: Option<String>,
    /// The §12 exec-approval fingerprint (Cluster B): a hash of the fully-resolved command tuple
    /// `(abs-binary, argv, env-delta, cwd, exec-surface)` the operator approved. On the durable
    /// re-run the engine recomputes the tuple and refuses if it no longer matches (the approve-then-swap
    /// TOCTOU gate). `None` for non-command approvals (fs edits) and for pre-existing snapshots
    /// (`#[serde(default)]` keeps them decodable); a `None` fingerprint runs verbatim as before.
    #[serde(default)]
    pub fingerprint: Option<crate::exec::CommandFingerprint>,
}

impl Snapshot {
    /// A fresh snapshot for a newly created session at epoch 0.
    pub fn fresh(session_id: SessionId) -> Self {
        Self {
            session_id,
            epoch: Epoch::ZERO,
            conversation: Conversation::default(),
            references: References::default(),
            waiting_for: Vec::new(),
            iters_since_skill: 0,
            turns_since_memory: 0,
            approval_policy: None,
            pending_approvals: Vec::new(),
            writer_version: String::new(),
        }
    }

    /// Encode to the opaque persisted CBOR form, stamping this build's version as provenance.
    pub fn encode(&self) -> Result<SnapshotBlob, DaemonError> {
        let mut to_write = self.clone();
        to_write.writer_version = daemon_common::VERSION.to_string();
        let mut bytes = Vec::new();
        ciborium::into_writer(&to_write, &mut bytes)
            .map_err(|e| DaemonError::Codec(e.to_string()))?;
        Ok(SnapshotBlob::new(bytes))
    }

    /// Decode from the opaque persisted CBOR form. Decoding stays tolerant (the format evolves via
    /// `#[serde(default)]`); a snapshot stamped by a *newer* build is decoded but logged, since this
    /// (older) build may not understand fields the newer one wrote.
    pub fn decode(blob: &SnapshotBlob) -> Result<Self, DaemonError> {
        let snapshot: Self = ciborium::from_reader(blob.as_bytes())
            .map_err(|e| DaemonError::Codec(e.to_string()))?;
        if base_semver(&snapshot.writer_version)
            .zip(base_semver(daemon_common::VERSION))
            .is_some_and(|(writer, current)| writer > current)
        {
            tracing::warn!(
                writer = %snapshot.writer_version,
                current = %daemon_common::VERSION,
                session = %snapshot.session_id,
                "decoding a snapshot written by a newer node build; fields it added may be lost",
            );
        }
        Ok(snapshot)
    }
}

/// Parse the `MAJOR.MINOR.PATCH` core out of a version string, ignoring any `+build`/`-pre` suffix.
/// Returns `None` for an empty or unparseable value (e.g. a pre-provenance snapshot), so the
/// newer-than comparison is simply skipped.
fn base_semver(version: &str) -> Option<(u64, u64, u64)> {
    let core = version.split(['+', '-']).next()?;
    let mut parts = core.split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next()?.parse().ok()?;
    let patch = parts.next()?.parse().ok()?;
    Some((major, minor, patch))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Serialize;

    /// Forward-compat: a blob written by an older build (no `writer_version` and none of the later
    /// `#[serde(default)]` fields) must still decode, with the new fields defaulted. This is the
    /// snapshot-format analogue of the wire layer's additive-field tolerance.
    #[test]
    fn decodes_blob_written_before_new_fields_existed() {
        #[derive(Serialize)]
        struct OldSnapshot {
            session_id: SessionId,
            epoch: Epoch,
            conversation: Conversation,
            references: References,
            waiting_for: Vec<JobId>,
        }
        let old = OldSnapshot {
            session_id: SessionId::new("s1"),
            epoch: Epoch::ZERO,
            conversation: Conversation::default(),
            references: References::default(),
            waiting_for: Vec::new(),
        };
        let mut bytes = Vec::new();
        ciborium::into_writer(&old, &mut bytes).unwrap();

        let decoded = Snapshot::decode(&SnapshotBlob::new(bytes)).expect("old snapshot decodes");
        assert_eq!(decoded.session_id, SessionId::new("s1"));
        assert!(decoded.writer_version.is_empty());
        assert!(decoded.pending_approvals.is_empty());
        assert_eq!(decoded.iters_since_skill, 0);
    }

    /// `encode` stamps the running build's version, and a round-trip preserves it.
    #[test]
    fn encode_stamps_writer_version() {
        let blob = Snapshot::fresh(SessionId::new("s2")).encode().unwrap();
        let decoded = Snapshot::decode(&blob).unwrap();
        assert_eq!(decoded.writer_version, daemon_common::VERSION);
    }

    #[test]
    fn base_semver_parses_and_orders() {
        assert_eq!(base_semver("1.2.3"), Some((1, 2, 3)));
        assert_eq!(base_semver("0.0.1+g1a2b3c4.dirty"), Some((0, 0, 1)));
        assert_eq!(base_semver(""), None);
        assert!(base_semver("0.10.0") > base_semver("0.9.9"));
    }
}
