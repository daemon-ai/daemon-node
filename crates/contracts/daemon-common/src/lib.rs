//! `daemon-common` — shared primitives across the workspace.
//!
//! Stable identifiers (`SessionId`, `UnitId`, `JobId`), `Budget`, `FenceToken`, `Epoch`,
//! error scaffolding, wire-version, and the opaque persisted-snapshot newtype. Pure types only;
//! no runtime. This is the root of the crate DAG — it depends on nothing internal.
//!
//! See `docs/daemon-workspace-layout.md` and `docs/specs/`.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::fmt;

/// Macro to declare a string-backed, stable logical identifier newtype.
macro_rules! string_id {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
        pub struct $name(pub String);

        impl $name {
            /// Construct from anything string-like.
            pub fn new(s: impl Into<String>) -> Self {
                Self(s.into())
            }

            /// Borrow the underlying string.
            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl From<String> for $name {
            fn from(s: String) -> Self {
                Self(s)
            }
        }

        impl From<&str> for $name {
            fn from(s: &str) -> Self {
                Self(s.to_owned())
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.0)
            }
        }
    };
}

string_id! {
    /// Stable logical identity of a durable engine incarnation. Never a live task handle.
    SessionId
}
string_id! {
    /// Stable identity of a managed unit in the supervision tree.
    UnitId
}
string_id! {
    /// Stable identity of a unit of background work delegated by a session.
    JobId
}

/// Partition / ownership domain. The activation lease is scoped per partition.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PartitionId(pub u64);

impl PartitionId {
    /// The default single-partition (in-process) domain.
    pub const DEFAULT: Self = Self(0);
}

impl Default for PartitionId {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// A monotonic fencing token guarding a single activation.
///
/// Ordering is load-bearing: only the holder of the highest token for a `SessionId` may commit
/// durable state (lifecycle §4 invariant #5; acceptance tests #4/#6).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct FenceToken(pub u64);

impl FenceToken {
    /// The pre-acquisition token (no activation has held the lease yet).
    pub const ZERO: Self = Self(0);

    /// The next monotonic token after this one.
    pub fn next(self) -> Self {
        Self(self.0 + 1)
    }
}

impl Default for FenceToken {
    fn default() -> Self {
        Self::ZERO
    }
}

/// Monotonic incarnation epoch, bumped on every suspension; part of the idempotency key
/// `UNIQUE(session_id, epoch, job_id)` (lifecycle §4 invariant #2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Epoch(pub u64);

impl Epoch {
    /// The initial epoch of a freshly created session.
    pub const ZERO: Self = Self(0);

    /// The next epoch (bumped at each suspension boundary).
    pub fn next(self) -> Self {
        Self(self.0 + 1)
    }
}

impl Default for Epoch {
    fn default() -> Self {
        Self::ZERO
    }
}

/// A resource budget carried alongside delegated work.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Budget {
    /// Optional token ceiling (`None` = unbounded).
    pub tokens: Option<u64>,
    /// Optional wall-clock ceiling in milliseconds (`None` = unbounded).
    pub wall_ms: Option<u64>,
}

impl Budget {
    /// An explicitly unbounded budget.
    pub fn unlimited() -> Self {
        Self {
            tokens: None,
            wall_ms: None,
        }
    }
}

/// Correlation id for a request/response pair on the live protocols.
///
/// Shared by the §17 host protocol (`daemon-protocol`) and the generic management protocol
/// (`daemon-supervision`) so a `request_id` means the same thing at every level of the tree
/// (supervision spec §2.3; §17.1 item 2). Mandatory on every correlated command/request.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ReqId(pub u64);

/// An incremental usage measurement, identical at every level of the supervision tree.
///
/// `Usage` is first-class on both the §17 and management event streams precisely because it
/// aggregates up the tree by construction: an orchestrator's usage is the fold of its children's
/// (supervision spec §2.2 / §4, "identical at every level"). Deltas are additive.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct UsageDelta {
    /// Prompt/input tokens consumed by this step.
    pub input_tokens: u64,
    /// Completion/output tokens produced by this step.
    pub output_tokens: u64,
    /// Provider API calls made by this step.
    pub api_calls: u32,
}

impl UsageDelta {
    /// Fold another delta into this one (the tree aggregation, supervision invariant #4).
    pub fn add(&mut self, other: &UsageDelta) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
        self.api_calls += other.api_calls;
    }
}

/// A point-in-time view of a provider rate-limit window, identical at every level (supervision
/// spec §2.2). Fields are `None` when the provider does not surface them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RateLimitSnapshot {
    /// Requests/tokens remaining in the current window.
    pub remaining: Option<u64>,
    /// The window ceiling.
    pub limit: Option<u64>,
    /// Milliseconds until the window resets.
    pub reset_ms: Option<u64>,
}

/// Version of the live host wire protocol (§17 envelopes / CDDL contract).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct WireVersion(pub u16);

impl WireVersion {
    /// The version this build speaks.
    pub const CURRENT: Self = Self(1);

    /// The version this build speaks (alias for [`WireVersion::CURRENT`]).
    pub fn current() -> Self {
        Self::CURRENT
    }

    /// Whether a peer's version is compatible with this one.
    pub fn is_compatible(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Default for WireVersion {
    fn default() -> Self {
        Self::CURRENT
    }
}

/// The opaque, persisted form of an engine snapshot. The typed `Snapshot` lives in `daemon-core`
/// (§5); the durable layer (`daemon-store` / `daemon-activation`) handles them only as CBOR bytes,
/// keeping those crates free of an engine/protocol dependency (lifecycle §2; layout §3 DAG).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnapshotBlob(pub Vec<u8>);

impl SnapshotBlob {
    /// Wrap raw CBOR bytes.
    pub fn new(bytes: Vec<u8>) -> Self {
        Self(bytes)
    }

    /// Borrow the raw CBOR bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    /// Whether the blob carries no bytes (e.g. a freshly created session before first checkpoint).
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl From<Vec<u8>> for SnapshotBlob {
    fn from(bytes: Vec<u8>) -> Self {
        Self(bytes)
    }
}

/// The shared base error reused/wrapped by layer-specific errors (`StoreError`, `SubErr`).
#[derive(Debug, thiserror::Error)]
pub enum DaemonError {
    /// A stale incarnation attempted to commit after losing the lease.
    #[error("fenced: holder token {have} is stale (current is {current})")]
    Fenced {
        /// The token the caller presented.
        have: u64,
        /// The current (highest) token for the session.
        current: u64,
    },
    /// The requested session/record does not exist.
    #[error("not found")]
    NotFound,
    /// (De)serialization failure.
    #[error("codec: {0}")]
    Codec(String),
    /// An injected fault boundary fired (test-only crash simulation).
    #[error("injected fault: {0}")]
    Fault(String),
    /// Any other failure.
    #[error("{0}")]
    Other(String),
}
