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
use std::path::PathBuf;

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
    /// Prompt tokens served from the provider's prompt cache (a subset of `input_tokens` billed at a
    /// reduced rate). `0` when the provider does not surface cache reads.
    #[serde(default)]
    pub cache_read_tokens: u64,
    /// Prompt tokens written to the provider's prompt cache this step (the cache-creation surcharge).
    /// `0` when the provider does not surface cache writes.
    #[serde(default)]
    pub cache_write_tokens: u64,
    /// Reasoning/thinking tokens (a subset of `output_tokens` for reasoning models). `0` when none.
    #[serde(default)]
    pub reasoning_tokens: u64,
    /// The estimated cost of this step in micro-USD (millionths of a dollar), when a pricing table is
    /// available. `0`/unset where cost is not computed.
    #[serde(default)]
    pub cost_micros: u64,
}

impl UsageDelta {
    /// Fold another delta into this one (the tree aggregation, supervision invariant #4).
    pub fn add(&mut self, other: &UsageDelta) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
        self.api_calls += other.api_calls;
        self.cache_read_tokens += other.cache_read_tokens;
        self.cache_write_tokens += other.cache_write_tokens;
        self.reasoning_tokens += other.reasoning_tokens;
        self.cost_micros += other.cost_micros;
    }

    /// Estimate this step's cost in micro-USD under `pricing`, returning the value `cost_micros`
    /// should carry. Fresh (un-cached) input is `input_tokens - cache_read_tokens -
    /// cache_write_tokens` billed at the base input rate, cache reads/writes at their own rates, and
    /// all output at the output rate. `reasoning_tokens` are a billed subset of `output_tokens`, so
    /// they are *not* charged again. Token counts are per-step; rates are micro-USD per million
    /// tokens (see [`Pricing`]).
    pub fn estimate_cost_micros(&self, pricing: &Pricing) -> u64 {
        let fresh_input = self
            .input_tokens
            .saturating_sub(self.cache_read_tokens)
            .saturating_sub(self.cache_write_tokens);
        let per_mtok = |tokens: u64, rate: u64| -> u64 {
            // tokens * rate / 1_000_000, in u128 to avoid overflow on large windows.
            ((tokens as u128 * rate as u128) / 1_000_000u128) as u64
        };
        per_mtok(fresh_input, pricing.input_micros_per_mtok)
            + per_mtok(self.cache_read_tokens, pricing.cache_read_micros_per_mtok)
            + per_mtok(self.cache_write_tokens, pricing.cache_write_micros_per_mtok)
            + per_mtok(self.output_tokens, pricing.output_micros_per_mtok)
    }
}

/// A per-model price sheet in micro-USD per **million** tokens (the unit cloud providers publish:
/// e.g. $3.00 / Mtok => `3_000_000`). Used by [`UsageDelta::estimate_cost_micros`] to fill in
/// `cost_micros` at the provider boundary. Cache read/write rates default to the Anthropic public
/// ratios relative to base input (reads at 0.1x, writes at 1.25x) when not set explicitly.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Pricing {
    /// Base (un-cached) input rate, micro-USD per million tokens.
    pub input_micros_per_mtok: u64,
    /// Output rate, micro-USD per million tokens.
    pub output_micros_per_mtok: u64,
    /// Cache-read rate (prompt tokens served from cache), micro-USD per million tokens.
    pub cache_read_micros_per_mtok: u64,
    /// Cache-write rate (cache-creation surcharge), micro-USD per million tokens.
    pub cache_write_micros_per_mtok: u64,
}

impl Pricing {
    /// A price sheet from base input/output rates (micro-USD per million tokens), deriving the
    /// cache rates from Anthropic's public ratios: reads at 0.1x input, writes at 1.25x input.
    pub fn from_io(input_micros_per_mtok: u64, output_micros_per_mtok: u64) -> Self {
        Self {
            input_micros_per_mtok,
            output_micros_per_mtok,
            cache_read_micros_per_mtok: input_micros_per_mtok / 10,
            cache_write_micros_per_mtok: input_micros_per_mtok * 5 / 4,
        }
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
    ///
    /// v2 (event-io edge): adds the merged session-event-log surface (`Origin`, `Disposition`,
    /// `Direction`, `SessionLogEntry`), the `subscribe`/`log_after` api ops, and the
    /// `TurnTrigger::Scheduled` arm.
    ///
    /// v3 (genai-native providers): collapses `ProviderSelector` to `mock | genai | llama_cpp |
    /// mistral_rs` (the genai adapter is inferred from the model id; legacy per-provider names
    /// deserialize to `genai` via serde aliases) and lists networked models live from genai.
    ///
    /// v4 (background spawn): adds the fire-and-forget `HostRequestKind::Spawn { SpawnSpec }` host
    /// request (engine-native `Effect::Spawn`) materializing an attached, self-closing background
    /// child (skill/memory review) that records a child edge without suspending or waking the parent.
    ///
    /// v5 (runtime control): adds the live per-session model switch (`SetSessionModel`), the §12
    /// edit-approval session modes (`SetSessionMode` + `ApprovalMode`), and the durable HITL approval
    /// surface (`HostResponseBody::Deferred`, `ApprovalsPending`/`ApprovalDecide`, `ApprovalInfo`).
    ///
    /// v6 (profile distributions + versioning): adds profile clone/export/import
    /// (`ProfileClone`/`ProfileExport`/`ProfileImport`, the `Distribution` bundle = spec + local
    /// skills, `credential_ref` kept) and a native append-only revision history shared by profiles
    /// and skills (`Profile{History,At,Revert}`, `Skill{History,At,Revert}`, `Revision`/`Author`,
    /// `SkillBundle`) with non-destructive revert / roll-forward.
    ///
    /// v7 (routing): adds the host-routed submit (`SubmitRouted { origin, command }` ->
    /// `ApiResponse::Routed { session }`), the seam a transport uses to hand the host an `Origin` and
    /// let the §5.9 routing capability resolve the session + profile + delivery (rather than deriving
    /// the `SessionId` itself).
    ///
    /// v8 (observe): adds the context-only `AgentCommand::Observe { input, request_id }` — appends
    /// inbound context to the conversation **without** opening a turn (the multi-party accumulation
    /// seam, event-io §5.9): chatter folds in while idle and lands in the following turn while busy,
    /// so a shared room can feed the agent context it sees on its next mention-gated turn.
    ///
    /// v9 (delivery sessions): adds owned-session discovery (`DeliverySessions { transport }` ->
    /// `ApiResponse::DeliverySessions([session-id])`), the outbound-symmetry seam a transport calls on
    /// (re)connect to enumerate the sessions whose `Primary` it owns and resume delivery (event-io
    /// §5.9.3). The in-process `DeliverySink` push path is a live trait object and does not cross the
    /// wire, so it adds no op.
    pub const CURRENT: Self = Self(9);

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
            profiles: self
                .profiles
                .intersection(&other.profiles)
                .cloned()
                .collect(),
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

// ---------------------------------------------------------------------------
// Model management primitives (the unified local-inference model surface)
// ---------------------------------------------------------------------------
//
// These are the transport-stable shapes the model-management surface (`ModelApi` in `daemon-api`)
// marshals and the `daemon-models` crate produces: a `ModelRef` names a model for a local engine,
// `ModelSource` says where its bytes come from (a Hugging Face repo or a local path), and the
// search / file / download / catalog DTOs carry discovery + acquisition state. They live here (the
// DAG root) so the contract crate, the implementer (`daemon-models`), and every transport share one
// definition without dragging engine types into the contract.

string_id! {
    /// Stable catalog identity of an installed model (a content-derived handle, so the same model
    /// resolves to the same id regardless of which engine/profile activated it).
    ModelId
}

/// Which local inference engine a model targets. Mirrors `daemon_infer::protocol::Engine`, kept
/// independent so this DAG-root crate carries no engine dependency.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelEngine {
    /// llama.cpp (GGUF) via the `llama-cpp-4` bindings.
    Llama,
    /// mistral.rs (Hugging Face repo / UQFF / GGUF) via the `mistralrs` crate.
    MistralRs,
}

impl ModelEngine {
    /// Parse an engine selector (`llama`, `mistralrs`, and common spellings).
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "llama" | "llama-cpp" | "llamacpp" | "gguf" => Some(ModelEngine::Llama),
            "mistralrs" | "mistral-rs" | "mistral.rs" => Some(ModelEngine::MistralRs),
            _ => None,
        }
    }

    /// The canonical lowercase spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            ModelEngine::Llama => "llama",
            ModelEngine::MistralRs => "mistralrs",
        }
    }
}

impl fmt::Display for ModelEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Where a model's bytes come from.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelSource {
    /// A Hugging Face Hub repo. `file` selects a single artifact within the repo (the GGUF file for
    /// llama; `None` means "the repo" — mistral.rs loads a repo directory). `revision` pins a
    /// branch / tag / commit (`"main"` by default).
    Hf {
        /// The `org/name` repo id.
        repo: String,
        /// The artifact path within the repo (e.g. `Model-Q4_K_M.gguf`), if a single file is named.
        file: Option<String>,
        /// The git revision (branch / tag / commit) to pin.
        revision: String,
    },
    /// An already-present local path (a GGUF file or a model directory).
    Local {
        /// The local filesystem path.
        path: PathBuf,
    },
}

impl ModelSource {
    /// A Hugging Face repo source at the default (`main`) revision.
    pub fn hf(repo: impl Into<String>) -> Self {
        ModelSource::Hf {
            repo: repo.into(),
            file: None,
            revision: "main".to_string(),
        }
    }

    /// A Hugging Face single-file source (e.g. one GGUF in a repo) at the default revision.
    pub fn hf_file(repo: impl Into<String>, file: impl Into<String>) -> Self {
        ModelSource::Hf {
            repo: repo.into(),
            file: Some(file.into()),
            revision: "main".to_string(),
        }
    }
}

/// A model named for a specific local engine — the unit a client downloads, activates, and the
/// daemon resolves to a ready on-disk artifact before loading.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelRef {
    /// The engine that will load this model.
    pub engine: ModelEngine,
    /// Where the model's bytes come from.
    pub source: ModelSource,
}

impl ModelRef {
    /// A reference to a model `source` for `engine`.
    pub fn new(engine: ModelEngine, source: ModelSource) -> Self {
        Self { engine, source }
    }
}

/// How a model search result set is ordered (Hugging Face `sort` values).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum SearchSort {
    /// Trending score (the Hub default).
    #[default]
    Trending,
    /// Most downloaded.
    Downloads,
    /// Most liked.
    Likes,
    /// Most recently modified.
    Modified,
    /// Most recently created.
    Created,
}

impl SearchSort {
    /// The Hugging Face `sort` query value.
    pub fn as_query(self) -> &'static str {
        match self {
            SearchSort::Trending => "trending",
            SearchSort::Downloads => "downloads",
            SearchSort::Likes => "likes",
            SearchSort::Modified => "lastModified",
            SearchSort::Created => "createdAt",
        }
    }
}

/// A model-search request a client issues (step 1 of the two-step search→select→download flow).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchQuery {
    /// The free-text query (matched against repo id / name).
    pub text: String,
    /// The engine the results must be loadable by (filters the file/format the repo must carry).
    pub engine: ModelEngine,
    /// The result ordering.
    pub sort: SearchSort,
    /// The 0-based result page.
    pub page: u32,
    /// The page size (results per page).
    pub limit: u32,
}

impl SearchQuery {
    /// A first-page query for `text` against `engine`, ordered by the Hub default.
    pub fn new(text: impl Into<String>, engine: ModelEngine) -> Self {
        Self {
            text: text.into(),
            engine,
            sort: SearchSort::default(),
            page: 0,
            limit: 25,
        }
    }
}

/// One repo in a search result page.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchHit {
    /// The `org/name` repo id.
    pub repo: String,
    /// The repo author / org, when distinct from the id prefix.
    pub author: Option<String>,
    /// Cumulative download count.
    pub downloads: u64,
    /// Like count.
    pub likes: u64,
    /// The model's parameter count, when the Hub reports it.
    pub num_parameters: Option<u64>,
    /// The pipeline tag (e.g. `text-generation`).
    pub pipeline_tag: Option<String>,
    /// ISO-8601 last-modified timestamp, when present.
    pub last_modified: Option<String>,
    /// Whether the repo is gated (requires accepting terms / a token).
    pub gated: bool,
    /// Whether the repo is private.
    pub private: bool,
}

/// A page of search results (step 1 result).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchPage {
    /// The 0-based page index this set corresponds to.
    pub page: u32,
    /// The repos on this page.
    pub results: Vec<SearchHit>,
    /// Whether another page is likely available (the page came back full).
    pub has_more: bool,
}

/// One downloadable file within a repo (step 2 result — what the client selects to download).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelFile {
    /// The file path within the repo (e.g. `Model-Q4_K_M.gguf`).
    pub path: String,
    /// The file size in bytes, when the Hub reports it.
    pub size_bytes: u64,
    /// The quantization label parsed from the filename (e.g. `Q4_K_M`), for GGUF artifacts.
    pub quant: Option<String>,
    /// Whether this file is one shard of a multi-part (split) GGUF.
    pub is_split: bool,
    /// Whether this is the *first* shard of a split set (the file to name when downloading the set).
    pub is_first_shard: bool,
}

/// A handle to one in-flight or completed download job.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct DownloadId(pub u64);

impl fmt::Display for DownloadId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "dl-{}", self.0)
    }
}

/// The lifecycle state of a download job.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DownloadState {
    /// Accepted, not yet started transferring.
    Queued,
    /// Actively transferring bytes.
    Downloading,
    /// All selected files are present and verified.
    Completed,
    /// Paused by the client (partial bytes kept for resume).
    Paused,
    /// Cancelled by the client (partial bytes discarded).
    Cancelled,
    /// Failed; `error` on the [`DownloadStatus`] carries the reason.
    Failed,
}

/// A point-in-time snapshot of a download job's progress.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DownloadStatus {
    /// The job handle.
    pub id: DownloadId,
    /// The model being acquired.
    pub model: ModelRef,
    /// The current lifecycle state.
    pub state: DownloadState,
    /// Bytes transferred so far (across all selected files).
    pub downloaded_bytes: u64,
    /// Total bytes to transfer, when known (the sum of selected file sizes).
    pub total_bytes: u64,
    /// Files completed so far.
    pub files_done: u32,
    /// Total files selected for this job.
    pub files_total: u32,
    /// A failure reason when `state == Failed`.
    pub error: Option<String>,
}

/// An installed (downloaded + cataloged) model the daemon can activate and load.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstalledModel {
    /// The stable catalog id.
    pub id: ModelId,
    /// The reference that resolves this model.
    pub model: ModelRef,
    /// A human-friendly display name (the repo id, or file stem).
    pub display_name: String,
    /// The resolved on-disk artifact: the GGUF file (llama) or the model directory (mistral.rs).
    pub local_path: PathBuf,
    /// Total on-disk size in bytes.
    pub size_bytes: u64,
    /// The quantization label, when known.
    pub quant: Option<String>,
    /// Milliseconds since the Unix epoch when the model was installed.
    pub installed_at_ms: u64,
    /// The model architecture (e.g. `llama`, `qwen2`), read from GGUF metadata when available.
    #[serde(default)]
    pub arch: Option<String>,
    /// The training context length, read from GGUF metadata when available.
    #[serde(default)]
    pub context_length: Option<u32>,
    /// The authoritative GGUF file-type label (from metadata), more reliable than the filename guess.
    #[serde(default)]
    pub file_type: Option<String>,
}

// ---------------------------------------------------------------------------
// Quantization recommendation + local quantize + GGUF introspection
// ---------------------------------------------------------------------------

/// A recommended quantization for a repo given detected hardware — the "tune"-like selection that
/// helps a user get running quickly regardless of engine. For llama it names a GGUF file to pull;
/// for mistral.rs it names an in-engine ISQ level to apply.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuantRecommendation {
    /// The engine the recommendation targets.
    pub engine: ModelEngine,
    /// The `org/name` repo the recommendation is for.
    pub repo: String,
    /// For llama: the chosen GGUF file to download. `None` for mistral.rs (whole-repo + ISQ).
    pub file: Option<String>,
    /// The chosen quant label: a GGUF quant (llama, e.g. `Q4_K_M`) or an ISQ level (mistral.rs).
    pub quant: String,
    /// Estimated on-disk / resident bytes for the choice, when known.
    pub size_bytes: Option<u64>,
    /// The memory budget (bytes) the choice was fit against.
    pub budget_bytes: u64,
    /// Whether the choice is expected to fit the budget.
    pub fits: bool,
    /// A short human-readable rationale.
    pub reason: String,
    /// The full ranked candidate list (best-first), so a client can override the pick.
    pub candidates: Vec<QuantCandidate>,
}

/// One ranked quantization candidate the recommender considered.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuantCandidate {
    /// The quant label (GGUF quant or ISQ level).
    pub quant: String,
    /// The candidate GGUF file (llama), if applicable.
    pub file: Option<String>,
    /// Size in bytes, when known.
    pub size_bytes: Option<u64>,
    /// Whether it fits the budget.
    pub fits: bool,
}

/// A handle to a local quantization job.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct QuantizeId(pub u64);

impl fmt::Display for QuantizeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "qz-{}", self.0)
    }
}

/// The lifecycle state of a quantization job.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizeState {
    /// Accepted, not yet started.
    Queued,
    /// Acquiring the high-precision source GGUF.
    Preparing,
    /// Running the quantizer (worker process).
    Quantizing,
    /// Done; the result is cataloged.
    Completed,
    /// Failed; `error` on the [`QuantizeStatus`] carries the reason.
    Failed,
}

/// A point-in-time snapshot of a quantization job.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuantizeStatus {
    /// The job handle.
    pub id: QuantizeId,
    /// The source repo.
    pub repo: String,
    /// The high-precision source GGUF file being quantized.
    pub source_file: String,
    /// The target quant label (e.g. `Q4_K_M`).
    pub target_quant: String,
    /// The current lifecycle state.
    pub state: QuantizeState,
    /// The produced GGUF path, once quantization finishes.
    pub output_path: Option<PathBuf>,
    /// The catalog id of the produced model, once cataloged.
    pub model_id: Option<ModelId>,
    /// A failure reason when `state == Failed`.
    pub error: Option<String>,
}

/// Metadata read from a GGUF file header (via `gguf-rs`) without loading the model.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GgufInfo {
    /// The model architecture (`general.architecture`, e.g. `llama`, `qwen2`).
    pub architecture: Option<String>,
    /// The model name (`general.name`).
    pub name: Option<String>,
    /// The GGUF file-type / quant label (`general.file_type`).
    pub file_type: Option<String>,
    /// The training context length (`<arch>.context_length`).
    pub context_length: Option<u32>,
    /// The transformer block count (`<arch>.block_count`).
    pub block_count: Option<u32>,
    /// The GGUF quantization version (`general.quantization_version`).
    pub quantization_version: Option<u32>,
    /// The total parameter count, when derivable.
    pub parameter_count: Option<u64>,
    /// The file size in bytes.
    pub size_bytes: u64,
}

/// A portable snapshot of one skill bundle: its identity plus every text file under the bundle dir
/// (`SKILL.md` + support files), keyed by bundle-relative path. This is both the skill revision
/// snapshot blob and the unit carried in a profile [distribution]. Text-only (skills are markdown +
/// support docs); binary assets are out of scope.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SkillBundle {
    /// The bundle (directory) name — the canonical skill name.
    pub name: String,
    /// The category path segment under the skills root (`None` for a top-level skill).
    pub category: Option<String>,
    /// Bundle-relative path -> file contents (includes `SKILL.md`).
    pub files: std::collections::BTreeMap<String, String>,
}

/// Which versioned artifact a revision history tracks. One [`RevisionLog`] keys its history by
/// `(kind, id)`, so profiles and skills share one append-only mechanism.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RevisionKind {
    /// A profile bundle (`ProfileSpec`).
    Profile,
    /// A skill bundle (`SKILL.md` + support files).
    Skill,
}

impl RevisionKind {
    /// The on-disk/segment slug for this kind.
    pub fn as_str(self) -> &'static str {
        match self {
            RevisionKind::Profile => "profile",
            RevisionKind::Skill => "skill",
        }
    }
}

/// Who authored a revision — the provenance that matters when the agent edits its own
/// profile/skills. Distinguishes a human operator (a NodeApi call) from the agent itself (a tool
/// write, labeled by source, e.g. `skill_manage`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Author {
    /// A human operator acting over the NodeApi control surface.
    Operator,
    /// The agent itself, labeled by the write source (e.g. `skill_manage`).
    Agent(String),
}

/// One recorded revision of a versioned artifact. The full snapshot lives in a content-addressed
/// blob (keyed by `content_hash`); this is the metadata row a `history` query returns.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Revision {
    /// The 1-based monotonic sequence within `(kind, id)`.
    pub seq: u64,
    /// The previous head's `seq` (`None` for the first revision).
    pub parent: Option<u64>,
    /// The content hash of this revision's snapshot blob (dedupe + integrity).
    pub content_hash: ContentHash,
    /// Who made the change.
    pub author: Author,
    /// A short human-readable reason (`create`, `update`, `revert to 3`, `import`, …).
    pub reason: String,
    /// Wall-clock milliseconds since the Unix epoch at append time.
    pub ts_ms: u64,
}

/// Errors a [`RevisionLog`] can surface.
#[derive(Debug, thiserror::Error)]
pub enum RevisionError {
    /// No revision with that `seq` exists for `(kind, id)`.
    #[error("revision not found: {kind}/{id}@{seq}")]
    NotFound {
        /// The artifact kind slug.
        kind: String,
        /// The artifact id.
        id: String,
        /// The requested sequence.
        seq: u64,
    },
    /// An underlying I/O failure.
    #[error("revision log io: {0}")]
    Io(String),
    /// A (de)serialization failure of an index entry or blob.
    #[error("revision log codec: {0}")]
    Codec(String),
}

/// An append-only, content-addressed revision history shared by versioned artifacts (profiles and
/// skills). Every mutation appends a revision capturing the full snapshot; **revert is
/// non-destructive** — it appends a new head equal to an older revision's content, so nothing is
/// ever lost and roll-forward is simply reverting to a later `seq`.
///
/// The trait is a storage contract (sync, opaque-byte blobs); the file-backed implementation lives
/// in `daemon-host`. Profiles record through the NodeApi layer; skills record through the skill
/// store so the agent's own background-review writes are versioned too.
pub trait RevisionLog: Send + Sync {
    /// Append a new revision of `(kind, id)` carrying `blob` (the full snapshot), returning the
    /// recorded metadata. The new revision becomes the head.
    fn append(
        &self,
        kind: RevisionKind,
        id: &str,
        blob: &[u8],
        author: Author,
        reason: &str,
    ) -> Result<Revision, RevisionError>;

    /// The full revision history of `(kind, id)`, oldest first. Empty if none recorded yet.
    fn history(&self, kind: RevisionKind, id: &str) -> Result<Vec<Revision>, RevisionError>;

    /// The snapshot blob recorded at `seq` for `(kind, id)`.
    fn get_at(&self, kind: RevisionKind, id: &str, seq: u64) -> Result<Vec<u8>, RevisionError>;

    /// The current head revision of `(kind, id)`, if any.
    fn head(&self, kind: RevisionKind, id: &str) -> Result<Option<Revision>, RevisionError>;
}

impl fmt::Display for RevisionKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod pricing_tests {
    use super::{Pricing, UsageDelta};

    #[test]
    fn from_io_derives_anthropic_cache_ratios() {
        // $3 / $15 per Mtok => cache read 0.1x input, cache write 1.25x input.
        let p = Pricing::from_io(3_000_000, 15_000_000);
        assert_eq!(p.input_micros_per_mtok, 3_000_000);
        assert_eq!(p.output_micros_per_mtok, 15_000_000);
        assert_eq!(p.cache_read_micros_per_mtok, 300_000);
        assert_eq!(p.cache_write_micros_per_mtok, 3_750_000);
    }

    #[test]
    fn cost_splits_fresh_cached_and_output() {
        let pricing = Pricing::from_io(3_000_000, 15_000_000);
        let usage = UsageDelta {
            // 1M input tokens of which 600k are cache reads and 100k cache writes => 300k fresh.
            input_tokens: 1_000_000,
            output_tokens: 500_000,
            api_calls: 1,
            cache_read_tokens: 600_000,
            cache_write_tokens: 100_000,
            reasoning_tokens: 0,
            cost_micros: 0,
        };
        // fresh: 300_000 * 3_000_000 / 1e6 = 900_000
        // cache_read: 600_000 * 300_000 / 1e6 = 180_000
        // cache_write: 100_000 * 3_750_000 / 1e6 = 375_000
        // output: 500_000 * 15_000_000 / 1e6 = 7_500_000
        assert_eq!(usage.estimate_cost_micros(&pricing), 900_000 + 180_000 + 375_000 + 7_500_000);
    }

    #[test]
    fn no_cache_charges_all_input_fresh() {
        let pricing = Pricing::from_io(2_000_000, 8_000_000);
        let usage = UsageDelta {
            input_tokens: 1_000_000,
            output_tokens: 1_000_000,
            api_calls: 1,
            ..Default::default()
        };
        // 1M * 2.0 + 1M * 8.0 = 2_000_000 + 8_000_000
        assert_eq!(usage.estimate_cost_micros(&pricing), 10_000_000);
    }
}
