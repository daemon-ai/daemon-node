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
        }
    }

    /// Encode to the opaque persisted CBOR form.
    pub fn encode(&self) -> Result<SnapshotBlob, DaemonError> {
        let mut bytes = Vec::new();
        ciborium::into_writer(self, &mut bytes).map_err(|e| DaemonError::Codec(e.to_string()))?;
        Ok(SnapshotBlob::new(bytes))
    }

    /// Decode from the opaque persisted CBOR form.
    pub fn decode(blob: &SnapshotBlob) -> Result<Self, DaemonError> {
        ciborium::from_reader(blob.as_bytes()).map_err(|e| DaemonError::Codec(e.to_string()))
    }
}
