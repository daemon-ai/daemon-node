//! The typed engine snapshot (lifecycle §2) — the *only* durable engine state.
//!
//! Everything an incarnation needs to be reconstructed: its identity, epoch, the typed
//! [`Conversation`], the [`References`] to re-establish on rehydration (never live resources), and
//! the outstanding background work it suspended for. Persisted as opaque CBOR
//! ([`SnapshotBlob`](daemon_common::SnapshotBlob)) so the durable substrate stays engine-agnostic.

use crate::conversation::Conversation;
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
