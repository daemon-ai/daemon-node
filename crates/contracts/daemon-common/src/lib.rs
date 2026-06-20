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
string_id! {
    /// The stream a verifiable journal is keyed by: any addressable agent in the tree (a durable
    /// session, a live interactive session, a fleet/foreign unit). Decouples the journal from the
    /// durable activation identity (`SessionId`/`Epoch`) so non-durable units journal too.
    JournalStreamId
}

impl JournalStreamId {
    /// The journal stream for a durable/live session.
    pub fn session(id: &SessionId) -> Self {
        Self(id.0.clone())
    }

    /// The journal stream for a managed unit (fleet/foreign).
    pub fn unit(id: &UnitId) -> Self {
        Self(id.0.clone())
    }
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

/// A logical correlation id that rides every message boundary ("trace context").
///
/// Modelled on elfo's `TraceId`: a process generates one at an ingress point, stamps it onto
/// every outbound frame, and the receiver *restores* it into its task-local scope so logs,
/// spans, and the verifiable journal on both sides of a cut correlate. This is a correlation
/// handle only — **not** an integrity primitive (the signed Merkle journal provides that).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TraceId(pub u64);

impl TraceId {
    /// The absence of a trace context.
    pub const NONE: Self = Self(0);

    /// Generate a fresh, process-locally-unique, nonzero trace id.
    ///
    /// Combines a monotonic counter with a nanosecond time seed and a mixing constant so ids do
    /// not collide within a run. No crypto dependency: this keeps `daemon-common` at the root of
    /// the DAG (layout §3).
    pub fn generate() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        use std::time::{SystemTime, UNIX_EPOCH};
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        let mixed = seed.rotate_left(17) ^ n.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        Self(if mixed == 0 { 1 } else { mixed })
    }

    /// Whether this is the absent trace.
    pub fn is_none(self) -> bool {
        self.0 == 0
    }
}

impl Default for TraceId {
    fn default() -> Self {
        Self::NONE
    }
}

impl fmt::Display for TraceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:016x}", self.0)
    }
}

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

/// Macro for a 32-byte opaque digest newtype (SHA-256 sized).
macro_rules! hash32 {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
        pub struct $name(pub [u8; 32]);

        impl $name {
            /// Wrap raw 32 bytes.
            pub const fn new(bytes: [u8; 32]) -> Self {
                Self(bytes)
            }

            /// Borrow the raw bytes.
            pub fn as_bytes(&self) -> &[u8; 32] {
                &self.0
            }

            /// Lowercase hex rendering.
            pub fn to_hex(&self) -> String {
                let mut s = String::with_capacity(64);
                for b in &self.0 {
                    s.push_str(&format!("{b:02x}"));
                }
                s
            }
        }

        impl From<[u8; 32]> for $name {
            fn from(bytes: [u8; 32]) -> Self {
                Self(bytes)
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, concat!(stringify!($name), "({})"), self.to_hex())
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.to_hex())
            }
        }
    };
}

hash32! {
    /// A 32-byte content hash of a deterministically-encoded value.
    ///
    /// Opaque at this layer: `daemon-store` persists it without depending on the crypto stack,
    /// while `daemon-telemetry` computes it (the Gordian Envelope / dCBOR digest).
    ContentHash
}

hash32! {
    /// A 32-byte Merkle root over a trace segment's digest tree (one per `(session, epoch)`).
    ///
    /// Folds every journal entry's digest plus the prior epoch's root (a rolling hash chain),
    /// bound into the durable incarnation under the same fence the checkpoint commits under.
    MerkleRoot
}

// ---------------------------------------------------------------------------
// Credential primitives (phase 7)
// ---------------------------------------------------------------------------
//
// The credential authority brokers short-lived, scoped *capability leases* down the supervision
// tree (host-spec §6; supervision-spec rules #6, #142). These are the serializable primitives that
// ride a cut: the authority (an ancestor host that owns secret material) mints a signed
// `CapabilityLease`; descendants several cuts down acquire it by re-brokering upward, with the
// `CredScope` intersected at each hop. The crypto (ed25519 signing of the capability) lives in
// `daemon-credentials` — this layer stays codec/crypto-free (layout §3), holding only opaque
// signature bytes alongside the public fields.

string_id! {
    /// Stable identity of one credential / capability lease minted by the authority.
    CredId
}
string_id! {
    /// A reference to a provider profile (which provider/key family a credential serves).
    ProfileRef
}

/// How a `CapabilityLease` resolves at the point of use — three modes trading isolation for cost.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CredMode {
    /// The owner mints a genuinely short-lived provider token (OAuth/STS); the holder calls the
    /// provider directly with the embedded `LeaseSecret`. The provider enforces the TTL.
    Native,
    /// The owner hands over a usable (often non-expiring) key in the lease; the holder calls the
    /// provider directly. Honest that the holder effectively keeps it for the key's lifetime, so
    /// the compensating control is the mandatory audit record, not the TTL. A fresh per-grant key
    /// (where the source can mint one) is genuinely revocable; otherwise the grant is audit-only.
    Bearer,
    /// The owner holds a non-expiring key; the lease is a handle only, and the actual provider call
    /// is proxied to the owner (who attaches the real key). The holder never sees secret material.
    Proxied,
}

/// A short-lived secret token embedded in a `CredMode::Native` lease. Its `Debug` is redacted so a
/// token never leaks into logs or the verifiable trace.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LeaseSecret(pub String);

impl LeaseSecret {
    /// Wrap a token string.
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
    /// Borrow the raw token (use only at the provider boundary).
    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for LeaseSecret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("LeaseSecret(***)")
    }
}

/// An attenuable capability scope (macaroon-style): the set of profiles and actions a holder may
/// use, plus an optional cost ceiling. Attenuation is set intersection + the tighter ceiling, so a
/// child's scope can only ever *narrow* its parent's (least privilege enforced by the authority).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CredScope {
    /// The profiles this scope may serve.
    pub profiles: std::collections::BTreeSet<String>,
    /// The named operations permitted (e.g. `"chat"`, `"embed"`).
    pub actions: std::collections::BTreeSet<String>,
    /// Optional token ceiling this scope authorizes (`None` = unbounded).
    pub max_tokens: Option<u64>,
}

impl CredScope {
    /// The empty scope (grants nothing) — the identity for "deny".
    pub fn nothing() -> Self {
        Self {
            profiles: Default::default(),
            actions: Default::default(),
            max_tokens: Some(0),
        }
    }

    /// A scope over the given profile and actions with an optional ceiling.
    pub fn new<P, A>(profiles: P, actions: A, max_tokens: Option<u64>) -> Self
    where
        P: IntoIterator,
        P::Item: Into<String>,
        A: IntoIterator,
        A::Item: Into<String>,
    {
        Self {
            profiles: profiles.into_iter().map(Into::into).collect(),
            actions: actions.into_iter().map(Into::into).collect(),
            max_tokens,
        }
    }

    /// Attenuate: the intersection of two scopes (profiles ∩, actions ∩, tighter ceiling). The
    /// result authorizes only what *both* allow — the per-hop narrowing the broker applies.
    pub fn intersect(&self, other: &CredScope) -> CredScope {
        let tighter = match (self.max_tokens, other.max_tokens) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        };
        CredScope {
            profiles: self.profiles.intersection(&other.profiles).cloned().collect(),
            actions: self.actions.intersection(&other.actions).cloned().collect(),
            max_tokens: tighter,
        }
    }

    /// Whether this scope authorizes `action` on `profile`.
    pub fn allows(&self, profile: &ProfileRef, action: &str) -> bool {
        self.profiles.contains(profile.as_str()) && self.actions.contains(action)
    }

    /// Whether this scope is a superset of `other` (so `other` is a valid attenuation of it). Used
    /// by the broker to reject a request for *more* than the hop's own grant.
    pub fn contains(&self, other: &CredScope) -> bool {
        other.profiles.is_subset(&self.profiles)
            && other.actions.is_subset(&self.actions)
            && match (self.max_tokens, other.max_tokens) {
                (None, _) => true,
                (Some(_), None) => false,
                (Some(a), Some(b)) => b <= a,
            }
    }

    /// Whether this scope grants nothing.
    pub fn is_empty(&self) -> bool {
        self.profiles.is_empty() || self.actions.is_empty() || self.max_tokens == Some(0)
    }
}

/// A minted, signed, short-lived capability the authority hands a holder. It is **not** the raw
/// provider key: `Native` embeds a genuinely-expiring token; `Proxied` carries only a handle and the
/// real call is proxied to the owner. The `signature` is an opaque ed25519 detached signature over
/// the capability's canonical form (produced/verified by `daemon-credentials`); this layer treats it
/// as bytes so the DAG root stays crypto-free.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityLease {
    /// The capability's stable id (audit correlation).
    pub cap_id: CredId,
    /// The profile this capability serves.
    pub profile: ProfileRef,
    /// The (attenuated) scope this capability authorizes.
    pub scope: CredScope,
    /// How the capability resolves at use.
    pub mode: CredMode,
    /// Wall-clock expiry, in milliseconds since the Unix epoch.
    pub expires_at_ms: u64,
    /// The short-lived token, present only in `Native` mode.
    pub secret: Option<LeaseSecret>,
    /// The authority's detached signature over the canonical capability bytes.
    pub signature: Vec<u8>,
}

impl CapabilityLease {
    /// Whether this capability has expired relative to `now_ms`.
    pub fn is_expired(&self, now_ms: u64) -> bool {
        now_ms >= self.expires_at_ms
    }
}

/// Why a credential acquire / use / verify failed. Crosses a cut (the brokered `CredReply`), so it
/// is serializable; the authority's verdict (notably `Fenced`/`Expired`) round-trips faithfully.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, thiserror::Error)]
pub enum CredError {
    /// No usable credential is available for the profile (pool exhausted/dead).
    #[error("no credential available for profile {0}")]
    Unavailable(String),
    /// The requested scope exceeds what this hop (or the authority) may grant.
    #[error("scope denied: requested capability exceeds the grant")]
    ScopeDenied,
    /// The capability has expired.
    #[error("capability expired")]
    Expired,
    /// The capability signature did not verify (tampered or wrong authority).
    #[error("capability signature invalid")]
    BadSignature,
    /// A stale incarnation attempted to acquire/use under a superseded fence.
    #[error("fenced: a stale incarnation cannot acquire credentials")]
    Fenced,
    /// This host is not the authority and has no upstream broker to forward to.
    #[error("no credential authority reachable")]
    NoAuthority,
    /// Any other failure.
    #[error("{0}")]
    Other(String),
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
