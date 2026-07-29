// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The **coordinator seat lease** — an Authority-signed, fenced claim on a run's coordinator
//! role, stored and compare-and-swapped by the registry (architecture §6.3; the registry is
//! untrusted storage and never becomes consensus authority).
//!
//! A seat claim is a canonical-CBOR [`SeatLeaseBody`] signed by the **claimant's certified
//! per-run key**, distributed as a [`SeatLease`] that carries the claimant's
//! [`RunKeyCertificate`] beside the body (a separate distribution record, never a frame-envelope
//! field — ABI §12.1). The registry stores the signed object and CASes on the **leadership
//! term** but creates no authority: peers verify the lease signature, the certificate chain to a
//! genesis-named base identity, and the term floor themselves. A stale claimant's records are
//! refused once a higher term exists, regardless of what the registry says.
//!
//! # Two identities, two order relations (scheme v2)
//!
//! - **Execution identity** — [`SeatLeaseBody::incarnation`] is the claimant's node-local,
//!   never-reused role-instance incarnation (ABI §8.1 `instance`): it names a sandbox, a per-run
//!   key, a journal stream, and a channel sequence namespace, and is meaningful only within one
//!   base identity's ladder.
//! - **Leadership identity** — [`SeatLeaseBody::leadership_term`] is the run-role-global
//!   monotonic term ordering seat ownership ACROSS base identities. The registry CAS accepts any
//!   **strictly greater** term (sparse, never `floor + 1` exactly): local counters on different
//!   boxes are unrelated, so dense increments cannot be authored honestly.
//!
//! The retired v1 scheme bound them together (`fencing_token == incarnation`), which broke as
//! soon as leadership could move between nodes whose local counters diverged. **Renew = same
//! execution, same term; every NEW execution key — takeover, same-base restart, or upgrade —
//! obtains a new grant at `term > floor`** (a new incarnation mints a new per-run key, hence a
//! new claimant).
//!
//! **Wall clock is liveness, never safety.** Expiry (`expires_at_ms` + a bounded skew grace)
//! only gates when a takeover may be attempted; the term is what fences the superseded claimant.
//! A premature, skew-driven takeover costs liveness, not safety.
//!
//! **Bounded integer domain.** Every `u64` ordinal in a seat body (term, incarnation) is pinned
//! to `<= i64::MAX` — one domain across Rust, SQLite, and the TypeScript registry port, so no
//! implementation can wrap, truncate, or disagree at the edges. Out-of-domain values are a
//! structural refusal.
//!
//! The registry-side acceptance rule is the pure [`SeatSlot::fold`] — normative CAS semantics any
//! conforming seat registry implements bit-for-bit (the shared test vectors pin it), exactly as
//! [`crate::revocation::RevocationLedger`] pins receiver-side revocation semantics. The fold
//! validates **structure only** (domain tag, bounds, windows, endpoint presence, slot
//! consistency); it MUST NOT verify signatures or judge authority — peers do that.

use serde::{Deserialize, Serialize};

use crate::bytes::{Hash, PeerId, Signature};
use crate::cert::{CertError, CertScope, RunKeyCertificate};
use crate::domains::{
    SEAT_LEASE_DOMAIN, SEAT_LEASE_DOMAIN_V1, SEAT_RELEASE_DOMAIN, SEAT_RELEASE_DOMAIN_V1,
};
use crate::error::VhcProtoError;
use crate::sign::{peer_id, sign_canonical, verify_canonical, SigningKey};

/// Default lease time-to-live: a claim/renew is live for this long past `issued_at_ms`.
pub const DEFAULT_SEAT_TTL_MS: u64 = 30_000;
/// Default heartbeat (renew) cadence — the TTL is 3× this, so two missed heartbeats still renew.
pub const DEFAULT_SEAT_HEARTBEAT_MS: u64 = 10_000;
/// Default clock-skew grace added to `expires_at_ms` before a lease is treated as expired.
pub const DEFAULT_SEAT_SKEW_MS: u64 = 5_000;
/// Ceiling any configured skew grace is clamped to (a larger grace only delays takeover).
pub const MAX_SEAT_SKEW_MS: u64 = 30_000;

/// The inclusive upper bound of the shared ordinal domain (terms, incarnations): `i64::MAX`,
/// the largest value every store in the system (SQLite integers, JS bigint folds pinned by the
/// shared vectors) represents exactly and compares identically.
pub const MAX_ORDINAL: u64 = i64::MAX as u64;

/// The control-plane endpoint a leased coordinator publishes for peers to dial. At least one of
/// the fields must be present; both may be (a dual-plane coordinator). Part of the signed body,
/// re-signed per incarnation.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ControlEndpoint {
    /// The WebSocket control-plane URL (e.g. `wss://…/runs/<label>/ws`).
    pub ws: Option<String>,
    /// The iroh ticket peers join gossip through.
    pub iroh_ticket: Option<String>,
}

impl ControlEndpoint {
    /// Whether the endpoint names at least one dialable plane.
    #[must_use]
    pub fn is_dialable(&self) -> bool {
        self.ws.is_some() || self.iroh_ticket.is_some()
    }
}

/// The signed body of a coordinator seat lease (scheme v2). Every field is part of the signed
/// preimage.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SeatLeaseBody {
    /// Domain-separation tag — MUST be [`SEAT_LEASE_DOMAIN`].
    pub domain: String,
    /// The run's cryptographic identity: the genesis-envelope hash (ABI §8.1 `run_id`).
    pub run_id: Hash,
    /// The envelope role label being claimed (e.g. `coordinator`).
    pub role: String,
    /// The run epoch this lease is scoped to. An epoch change REBINDS on renew (same execution
    /// identity, same term, reissued certificate) — never a takeover.
    pub epoch: u64,
    /// The claimant's **execution incarnation** (ABI §8.1 `instance`): node-local, never-reused,
    /// minted only by the claimant's durable counter. Names the sandbox/key/journal/sequence
    /// namespace this lease's holder runs as; meaningful only within the claimant's base
    /// identity. `<= MAX_ORDINAL`.
    pub incarnation: u64,
    /// The **leadership term**: the run-role-global monotonic seat order. The registry CAS
    /// accepts a claim only at a term STRICTLY GREATER than the slot's floor (sparse — local
    /// counters across boxes are unrelated). Peers fence the superseded claimant on this term,
    /// never on the incarnation. `<= MAX_ORDINAL`.
    pub leadership_term: u64,
    /// The claimant's certified per-run public key — the §12.1 frame-envelope `sender` this
    /// incarnation signs with, and the key that signs this lease.
    pub claimant: PeerId,
    /// The pinned module blob the claimant runs (ABI §8.1 `module_hash`; must match the
    /// embedded certificate's binding and the genesis role module).
    pub module_hash: Hash,
    /// The control-plane endpoint peers dial while this lease holds the seat.
    pub endpoint: ControlEndpoint,
    /// Claimant wall clock at issue (milliseconds; skew diagnostics).
    pub issued_at_ms: u64,
    /// Claimant wall clock at expiry (milliseconds). Liveness only — never safety (the term is
    /// the safety mechanism).
    pub expires_at_ms: u64,
    /// Advisory renew cadence; the TTL should be ≥ 3× this.
    pub heartbeat_interval_ms: u64,
}

/// A coordinator seat lease: the signed [`SeatLeaseBody`], the claimant's [`RunKeyCertificate`]
/// (travelling beside the body as a separate distribution record), and the claimant per-run key's
/// ed25519 signature over the canonical CBOR of the body.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SeatLease {
    /// The signed claim.
    pub body: SeatLeaseBody,
    /// The certificate authorizing [`SeatLeaseBody::claimant`] for
    /// `(run_id, epoch, role, incarnation, module_hash)`.
    pub certificate: RunKeyCertificate,
    /// ed25519 signature by [`SeatLeaseBody::claimant`] over the canonical CBOR of `body`.
    pub sig: Signature,
}

/// The signed body of a seat release: the claimant's statement that it gives the seat up. Every
/// field is part of the signed preimage.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SeatReleaseBody {
    /// Domain-separation tag — MUST be [`SEAT_RELEASE_DOMAIN`].
    pub domain: String,
    /// The run whose seat is released.
    pub run_id: Hash,
    /// The released role label.
    pub role: String,
    /// The releasing claimant's execution incarnation (must match the held lease).
    pub incarnation: u64,
    /// The released leadership term (must match the held lease).
    pub leadership_term: u64,
    /// The releasing claimant's per-run key (must match the held lease).
    pub claimant: PeerId,
}

/// A seat release: the signed [`SeatReleaseBody`] plus the claimant per-run key's ed25519
/// signature over its canonical CBOR.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SeatRelease {
    /// The signed statement.
    pub body: SeatReleaseBody,
    /// ed25519 signature by [`SeatReleaseBody::claimant`] over the canonical CBOR of `body`.
    pub sig: Signature,
}

/// Why a seat lease (or release) was refused by a **verifier** (a peer, or the typed layers of
/// the claim constructors). Registry-side refusals are [`SeatDecision`]s — structural only.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum SeatLeaseError {
    /// The body's domain tag is not the expected seat domain.
    WrongDomain {
        /// The tag actually carried.
        got: String,
    },
    /// An ordinal (term / incarnation) is outside the shared bounded domain
    /// (`> MAX_ORDINAL`) — a value no store in the system may represent.
    OutOfDomain {
        /// The offending field.
        field: &'static str,
        /// The carried value.
        value: u64,
    },
    /// `expires_at_ms <= issued_at_ms`, or a zero heartbeat interval.
    InvalidWindow,
    /// The endpoint names no dialable control plane.
    MissingEndpoint,
    /// The claimant per-run key's signature over the body does not verify.
    BadSignature,
    /// The embedded certificate does not authorize the claimant for the lease's scope
    /// (chain / scope / epoch / module / sender refusal, carried typed).
    Cert(CertError),
    /// The certificate's base identity is not one the run's genesis/Authority names.
    UntrustedBase,
    /// The lease is past `expires_at_ms` plus the skew grace.
    Expired {
        /// The lease's expiry (ms).
        expires_at_ms: u64,
        /// The verifier's clock (ms).
        now_ms: u64,
    },
    /// The lease's term is at or below a higher verified leadership term the receiver has
    /// observed — a fenced (superseded) claimant, dead regardless of the registry.
    TermSuperseded {
        /// The lease's term.
        got: u64,
        /// The highest verified term the receiver holds.
        floor: u64,
    },
}

impl core::fmt::Display for SeatLeaseError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::WrongDomain { got } => {
                write!(f, "seat domain `{got}` is not the expected seat domain")
            }
            Self::OutOfDomain { field, value } => write!(
                f,
                "seat {field} {value} is outside the bounded ordinal domain (max {MAX_ORDINAL})"
            ),
            Self::InvalidWindow => write!(f, "lease expiry/heartbeat window is invalid"),
            Self::MissingEndpoint => write!(f, "lease publishes no dialable control endpoint"),
            Self::BadSignature => write!(f, "lease signature does not verify to the claimant"),
            Self::Cert(e) => write!(f, "lease certificate refused: {e}"),
            Self::UntrustedBase => {
                write!(f, "lease certificate base identity is not genesis-trusted")
            }
            Self::Expired {
                expires_at_ms,
                now_ms,
            } => write!(f, "lease expired at {expires_at_ms} (now {now_ms})"),
            Self::TermSuperseded { got, floor } => write!(
                f,
                "lease term {got} is superseded (highest verified term {floor})"
            ),
        }
    }
}

impl std::error::Error for SeatLeaseError {}

impl From<CertError> for SeatLeaseError {
    fn from(e: CertError) -> Self {
        Self::Cert(e)
    }
}

impl From<SeatLeaseError> for VhcProtoError {
    fn from(e: SeatLeaseError) -> Self {
        VhcProtoError::Validation(e.to_string())
    }
}

/// Bounded-domain check for one ordinal field.
fn in_domain(field: &'static str, value: u64) -> Result<(), SeatLeaseError> {
    if value > MAX_ORDINAL {
        return Err(SeatLeaseError::OutOfDomain { field, value });
    }
    Ok(())
}

impl SeatLeaseBody {
    /// Structural validation — the invariants BOTH sides enforce (the registry before storing,
    /// every verifier before trusting): domain tag, bounded ordinals, a positive expiry window,
    /// a non-zero heartbeat, and a dialable endpoint. Signature-free by design.
    ///
    /// # Errors
    /// The applicable [`SeatLeaseError`].
    pub fn validate(&self) -> Result<(), SeatLeaseError> {
        if self.domain != SEAT_LEASE_DOMAIN {
            return Err(SeatLeaseError::WrongDomain {
                got: self.domain.clone(),
            });
        }
        in_domain("incarnation", self.incarnation)?;
        in_domain("leadership_term", self.leadership_term)?;
        if self.expires_at_ms <= self.issued_at_ms || self.heartbeat_interval_ms == 0 {
            return Err(SeatLeaseError::InvalidWindow);
        }
        if !self.endpoint.is_dialable() {
            return Err(SeatLeaseError::MissingEndpoint);
        }
        Ok(())
    }

    /// The execution-identity scope this lease claims — what the embedded certificate must bind.
    /// Scoped by the **incarnation** (execution identity), never the term (leadership identity):
    /// certificates certify executions; grants order leadership.
    #[must_use]
    pub fn cert_scope(&self) -> CertScope {
        CertScope {
            run_id: self.run_id,
            epoch: self.epoch,
            role: self.role.clone(),
            instance: self.incarnation,
            module_hash: self.module_hash,
        }
    }
}

impl SeatLease {
    /// Author a claim: validate the body structurally, require `run_key` to be the body's
    /// claimant, and sign the canonical CBOR of the body with the claimant per-run key. The
    /// certificate is issued separately by the base identity (once per binding) and travels here.
    ///
    /// # Errors
    /// A structural violation, a claimant/key mismatch, or a signing failure.
    pub fn claim(
        run_key: &SigningKey,
        certificate: RunKeyCertificate,
        body: SeatLeaseBody,
    ) -> Result<Self, VhcProtoError> {
        body.validate()?;
        if peer_id(run_key) != body.claimant {
            return Err(VhcProtoError::Validation(
                "seat lease claimant is not the signing per-run key".into(),
            ));
        }
        let sig = sign_canonical(run_key, &body)?;
        Ok(Self {
            body,
            certificate,
            sig,
        })
    }

    /// Verify structure + the claimant's self-signature over the body. This is the registry-free
    /// half of acceptance; authority is [`SeatLease::authorize`].
    ///
    /// # Errors
    /// The applicable [`SeatLeaseError`].
    pub fn verify_signature(&self) -> Result<(), SeatLeaseError> {
        self.body.validate()?;
        verify_canonical(&self.body.claimant, &self.sig, &self.body)
            .map_err(|_| SeatLeaseError::BadSignature)
    }

    /// The full peer-side acceptance: structure, self-signature, the certificate chain to a
    /// **genesis-trusted** base identity, the certificate's binding over exactly this lease's
    /// scope and claimant, and expiry (with the skew grace). Compose with
    /// [`crate::revocation::RevocationLedger::judge`] over [`SeatLeaseBody::cert_scope`] (the
    /// per-base execution judgment) AND [`SeatTermLedger::judge`] (the cross-base leadership
    /// judgment) — a lease below either floor is dead even if everything here verifies.
    ///
    /// # Errors
    /// The applicable [`SeatLeaseError`]. The registry MUST NOT run this check — authority is
    /// never the registry's judgment.
    pub fn authorize(
        &self,
        trusted_bases: &[PeerId],
        now_ms: u64,
        skew_ms: u64,
    ) -> Result<(), SeatLeaseError> {
        self.verify_signature()?;
        if !trusted_bases.contains(&self.certificate.base_identity) {
            return Err(SeatLeaseError::UntrustedBase);
        }
        self.certificate
            .authorizes_sender(&self.body.cert_scope(), &self.body.claimant)?;
        if self.is_expired(now_ms, skew_ms) {
            return Err(SeatLeaseError::Expired {
                expires_at_ms: self.body.expires_at_ms,
                now_ms,
            });
        }
        Ok(())
    }

    /// The observation-grade acceptance for **grant distribution** (feeding a
    /// [`SeatTermLedger`]): structure, self-signature, and the genesis-trusted certificate
    /// chain binding exactly this lease's scope and claimant — everything
    /// [`authorize`](Self::authorize) checks EXCEPT expiry. An expired grant is still the latest
    /// verified ownership statement: the term floor is monotonic ownership HISTORY, while expiry
    /// gates takeover LIVENESS — a receiver that refused to learn "term 7 was won" just because
    /// the lease has since lapsed would happily keep honoring term 3.
    ///
    /// # Errors
    /// The applicable [`SeatLeaseError`].
    pub fn verify_grant(&self, trusted_bases: &[PeerId]) -> Result<(), SeatLeaseError> {
        self.verify_signature()?;
        if !trusted_bases.contains(&self.certificate.base_identity) {
            return Err(SeatLeaseError::UntrustedBase);
        }
        self.certificate
            .authorizes_sender(&self.body.cert_scope(), &self.body.claimant)?;
        Ok(())
    }

    /// Whether the lease is past `expires_at_ms` plus the skew grace. Expiry gates takeover
    /// liveness only; fencing is the safety mechanism.
    #[must_use]
    pub fn is_expired(&self, now_ms: u64, skew_ms: u64) -> bool {
        now_ms > self.body.expires_at_ms.saturating_add(skew_ms)
    }
}

impl SeatReleaseBody {
    /// Structural validation: domain tag + bounded ordinals.
    ///
    /// # Errors
    /// The applicable [`SeatLeaseError`].
    pub fn validate(&self) -> Result<(), SeatLeaseError> {
        if self.domain != SEAT_RELEASE_DOMAIN {
            return Err(SeatLeaseError::WrongDomain {
                got: self.domain.clone(),
            });
        }
        in_domain("incarnation", self.incarnation)?;
        in_domain("leadership_term", self.leadership_term)?;
        Ok(())
    }
}

impl SeatRelease {
    /// Author a release: validate structurally, require `run_key` to be the body's claimant, and
    /// sign the canonical CBOR of the body.
    ///
    /// # Errors
    /// A structural violation, a claimant/key mismatch, or a signing failure.
    pub fn sign(run_key: &SigningKey, body: SeatReleaseBody) -> Result<Self, VhcProtoError> {
        body.validate()?;
        if peer_id(run_key) != body.claimant {
            return Err(VhcProtoError::Validation(
                "seat release claimant is not the signing per-run key".into(),
            ));
        }
        let sig = sign_canonical(run_key, &body)?;
        Ok(Self { body, sig })
    }

    /// Verify structure + the claimant's self-signature over the body.
    ///
    /// # Errors
    /// The applicable [`SeatLeaseError`].
    pub fn verify_signature(&self) -> Result<(), SeatLeaseError> {
        self.body.validate()?;
        verify_canonical(&self.body.claimant, &self.sig, &self.body)
            .map_err(|_| SeatLeaseError::BadSignature)
    }
}

// -- the receiver-side leadership-term ledger ---------------------------------------------------

/// A receiver's **leadership-term floor** for the coordinator seats it observes: the highest
/// VERIFIED term per `(run, role)`, and the claimant that term binds. Fed **only** by seat
/// grants that passed the full peer-side acceptance ([`SeatLease::authorize`] + the caller's
/// revocation judgment) — never by generic certificates (a coordinator-role certificate that
/// never won the seat must not fence the incumbent) and never by naked registry metadata (the
/// registry is untrusted storage; "accepted" means stored, not authorized).
///
/// This is the cross-base half of coordinator-sender authorization: ordinary frame acceptance
/// uses the per-base execution floors ([`crate::revocation::RevocationLedger`]); coordinator
/// frames additionally require the sender to be the claimant bound at the highest verified term.
/// A partitioned receiver enforces its highest *observed* term — not necessarily the globally
/// latest one (architecture §4.4 posture: peer-local floors protect after observation; signer
/// transfer under equivocation remains an explicit trust-statement item).
#[derive(Debug, Default)]
pub struct SeatTermLedger {
    /// `(run, role)` → the highest verified term and the claimant it binds.
    floors: std::collections::BTreeMap<(Hash, String), (u64, PeerId)>,
}

impl SeatTermLedger {
    /// An empty ledger.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Observe one VERIFIED seat grant (the caller has already run the full peer-side
    /// acceptance — this method deliberately takes the lease, not raw bytes, so an unverified
    /// object cannot reach it by construction). Advances the `(run, role)` floor monotonically;
    /// an equal-term re-observation of the SAME claimant refreshes nothing and changes nothing;
    /// a lower term is ignored (stale grant, kept out by the floor).
    pub fn observe_verified_grant(&mut self, lease: &SeatLease) {
        let key = (lease.body.run_id, lease.body.role.clone());
        let (term, claimant) = (lease.body.leadership_term, lease.body.claimant);
        match self.floors.get(&key) {
            Some((floor, _)) if term <= *floor => {}
            _ => {
                self.floors.insert(key, (term, claimant));
            }
        }
    }

    /// Restore a persisted floor — a term this receiver ITSELF verified and durably recorded
    /// earlier (node-restart continuity; the durable row is written only downstream of
    /// [`observe_verified_grant`](Self::observe_verified_grant)-grade acceptance). Same
    /// monotonic fold as observation: a lower/equal restore never regresses a live floor.
    pub fn restore_floor(&mut self, run_id: Hash, role: &str, term: u64, claimant: PeerId) {
        let key = (run_id, role.to_string());
        match self.floors.get(&key) {
            Some((floor, _)) if term <= *floor => {}
            _ => {
                self.floors.insert(key, (term, claimant));
            }
        }
    }

    /// The highest verified term for `(run, role)`, if any grant has been observed.
    #[must_use]
    pub fn floor(&self, run_id: &Hash, role: &str) -> Option<u64> {
        self.floors
            .get(&(*run_id, role.to_string()))
            .map(|(t, _)| *t)
    }

    /// Judge a seat lease against the floor: refused typed when its term is BELOW the highest
    /// verified term (a fenced predecessor). An equal term is accepted only for the exact bound
    /// claimant (the incumbent's own re-presentation); an equal term under a different claimant
    /// is a collision — fail closed.
    ///
    /// # Errors
    /// [`SeatLeaseError::TermSuperseded`].
    pub fn judge(&self, lease: &SeatLease) -> Result<(), SeatLeaseError> {
        let Some((floor, bound)) = self
            .floors
            .get(&(lease.body.run_id, lease.body.role.clone()))
        else {
            return Ok(());
        };
        let term = lease.body.leadership_term;
        if term < *floor || (term == *floor && lease.body.claimant != *bound) {
            return Err(SeatLeaseError::TermSuperseded {
                got: term,
                floor: *floor,
            });
        }
        Ok(())
    }

    /// Whether `sender` is the claimant bound at the highest verified term for `(run, role)` —
    /// the coordinator-frame authorization predicate. `None` when no grant has been observed
    /// (the caller decides whether an ungoverned seat is acceptable in its context).
    #[must_use]
    pub fn binds(&self, run_id: &Hash, role: &str, sender: &PeerId) -> Option<bool> {
        self.floors
            .get(&(*run_id, role.to_string()))
            .map(|(_, bound)| bound == sender)
    }
}

// -- the explicit v1 interpretation (archived state only) ---------------------------------------

/// The retired v1 lease body (`fencing_token == incarnation`), decodable for **interpreting
/// archived state only** (e.g. the run-g evidence fixture): `execution_incarnation =
/// incarnation`, `leadership_term = fencing_token`. The v1 signature preimage is the v1 byte
/// layout, so a v1 object can never be re-verified or re-presented as v2 — this type carries no
/// authority and no product path authors or accepts it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SeatLeaseBodyV1 {
    /// v1 domain tag — [`SEAT_LEASE_DOMAIN_V1`].
    pub domain: String,
    /// The run's cryptographic identity.
    pub run_id: Hash,
    /// The claimed role label.
    pub role: String,
    /// The run epoch.
    pub epoch: u64,
    /// The conflated ordinal (v1: `== fencing_token`). Interpret as the execution incarnation.
    pub incarnation: u64,
    /// The conflated ordinal (v1: `== incarnation`). Interpret as the leadership term.
    pub fencing_token: u64,
    /// The claimant's per-run key.
    pub claimant: PeerId,
    /// The pinned module hash.
    pub module_hash: Hash,
    /// The published control endpoint.
    pub endpoint: ControlEndpoint,
    /// Issue stamp (ms).
    pub issued_at_ms: u64,
    /// Expiry stamp (ms).
    pub expires_at_ms: u64,
    /// Advisory renew cadence (ms).
    pub heartbeat_interval_ms: u64,
}

impl SeatLeaseBodyV1 {
    /// The v2 interpretation of an archived v1 body: `(execution_incarnation, leadership_term)`.
    /// Read-only forensics — never authority (see the type docs).
    ///
    /// # Errors
    /// [`SeatLeaseError::WrongDomain`] when the body is not a v1 body.
    pub fn interpret(&self) -> Result<(u64, u64), SeatLeaseError> {
        if self.domain != SEAT_LEASE_DOMAIN_V1 {
            return Err(SeatLeaseError::WrongDomain {
                got: self.domain.clone(),
            });
        }
        Ok((self.incarnation, self.fencing_token))
    }
}

/// The retired v1 release domain check (companion to [`SeatLeaseBodyV1`]; see
/// [`SEAT_RELEASE_DOMAIN_V1`]).
#[must_use]
pub fn is_v1_release_domain(domain: &str) -> bool {
    domain == SEAT_RELEASE_DOMAIN_V1
}

// -- the registry-side slot + the normative CAS fold --------------------------------------------

/// The readable projection of one run-role seat slot.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SeatState {
    /// No lease holds the seat. `last_leadership_term` is the release/expiry tombstone floor —
    /// terms never reset; the next claim must present a strictly greater term (`None` = never
    /// claimed: the first claim's term is accepted as presented).
    Unclaimed {
        /// The tombstone floor.
        last_leadership_term: Option<u64>,
    },
    /// A signed lease holds the seat. Whether it is *live* is the reader's judgment (expiry +
    /// authority verification are peer-side; the registry only stores and CASes). Boxed for
    /// enum-size hygiene only — `Box<T>` serializes exactly as `T`.
    Leased(Box<SeatLease>),
}

/// One mutation request against a seat slot — the wire vocabulary of the seat endpoints.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SeatRequest {
    /// Claim / take over the seat (CAS at a strictly greater term — sparse).
    Claim(SeatLease),
    /// Renew (heartbeat) the held lease — same claimant, same incarnation, same term; the
    /// re-signed body extends expiry and may rebind the epoch (same key, reissued certificate).
    Renew(SeatLease),
    /// Release the seat, leaving the tombstone floor.
    Release(SeatRelease),
}

/// The registry's structural verdict on one [`SeatRequest`]. Purely structural — an `Accepted`
/// says nothing about authority (peers judge that; the registry is untrusted storage).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SeatDecision {
    /// Stored; the slot now reflects the request.
    Accepted,
    /// The object violates a structural invariant (domain, bounds, window, endpoint).
    RejectedStructural {
        /// A human-readable reason (diagnostic only; never authority).
        reason: String,
    },
    /// A live (unexpired, by the registry clock + skew) lease by a DIFFERENT claimant holds the
    /// seat.
    RejectedHeld,
    /// Nothing to renew/release — the slot is unclaimed.
    RejectedNotHeld,
    /// The compare-and-swap failed: the presented term does not satisfy what the slot requires.
    /// The loser re-reads and either accepts the incumbent or retries above the floor.
    RejectedFencingConflict {
        /// The floor the request must relate to (claim: must be STRICTLY GREATER than this;
        /// renew/release: must EQUAL the held term).
        expected: u64,
        /// The term the request presented.
        got: u64,
    },
}

/// One run-role seat slot as the registry stores it: the current lease (if any) plus the
/// monotonic term floor. INVARIANT: while leased, `last_leadership_term ==
/// Some(lease.body.leadership_term)`; after a release/takeover the floor persists (terms never
/// reset).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SeatSlot {
    /// The stored lease, if the seat is claimed.
    pub lease: Option<SeatLease>,
    /// The highest leadership term this slot has ever stored (the tombstone floor).
    pub last_leadership_term: Option<u64>,
}

impl SeatSlot {
    /// A never-claimed slot.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The readable projection.
    #[must_use]
    pub fn state(&self) -> SeatState {
        match &self.lease {
            Some(lease) => SeatState::Leased(Box::new(lease.clone())),
            None => SeatState::Unclaimed {
                last_leadership_term: self.last_leadership_term,
            },
        }
    }

    /// The **normative registry CAS fold** — pure: `(slot, request, now, skew) → (next slot,
    /// decision)`. A refused request never mutates the slot. Every conforming seat registry
    /// (the local fake, the cloud implementation) applies exactly this function under its
    /// single-writer serialization; the shared test vectors pin it.
    ///
    /// Structural only (registry posture): domain/invariant checks, slot-key consistency
    /// (`run_id`/`role` against the stored lease), expiry by the registry clock + skew grace,
    /// and the term CAS. NO signature verification, NO authority judgment.
    ///
    /// Claim acceptance (**sparse monotonic** — never `floor + 1` exactly: leadership terms
    /// derive from claimant-local state, so dense increments cannot be authored honestly):
    /// - held by the SAME claimant: the held term (idempotent refresh) or any STRICTLY GREATER
    ///   term (the claimant's own voluntary advance);
    /// - held by another claimant and NOT expired: refused (`RejectedHeld`);
    /// - expired or unclaimed: any term STRICTLY GREATER than the floor (`None` floor = never
    ///   claimed: the presented term stands).
    ///
    /// Renew requires the identity five-tuple `(run_id, role, incarnation, leadership_term,
    /// claimant)` to match the held lease exactly; epoch/module/endpoint/expiry may change (the
    /// epoch-rebind rule). An expired-but-untaken lease may still renew — grace, not safety.
    /// **A new execution identity (new incarnation ⇒ new per-run key ⇒ new claimant) can never
    /// renew**: it claims fresh at a higher term.
    ///
    /// Release requires `(run_id, role, leadership_term, claimant)` to match; the floor persists.
    #[must_use]
    pub fn fold(&self, request: &SeatRequest, now_ms: u64, skew_ms: u64) -> (Self, SeatDecision) {
        let refuse = |d: SeatDecision| (self.clone(), d);
        match request {
            SeatRequest::Claim(lease) => {
                if let Err(e) = lease.body.validate() {
                    return refuse(SeatDecision::RejectedStructural {
                        reason: e.to_string(),
                    });
                }
                if let Some(cur) = &self.lease {
                    if !Self::keys_match(&cur.body, &lease.body) {
                        return refuse(SeatDecision::RejectedStructural {
                            reason: "claim run/role does not match the slot".into(),
                        });
                    }
                    if cur.body.claimant == lease.body.claimant {
                        // Idempotent refresh at the held term, or the claimant's own advance.
                        if lease.body.leadership_term >= cur.body.leadership_term {
                            return self.store(lease);
                        }
                        return refuse(SeatDecision::RejectedFencingConflict {
                            expected: cur.body.leadership_term,
                            got: lease.body.leadership_term,
                        });
                    }
                    if !cur.is_expired(now_ms, skew_ms) {
                        return refuse(SeatDecision::RejectedHeld);
                    }
                }
                match self.last_leadership_term {
                    None => self.store(lease),
                    Some(floor) if lease.body.leadership_term > floor => self.store(lease),
                    Some(floor) => refuse(SeatDecision::RejectedFencingConflict {
                        expected: floor,
                        got: lease.body.leadership_term,
                    }),
                }
            }
            SeatRequest::Renew(lease) => {
                if let Err(e) = lease.body.validate() {
                    return refuse(SeatDecision::RejectedStructural {
                        reason: e.to_string(),
                    });
                }
                let Some(cur) = &self.lease else {
                    return refuse(SeatDecision::RejectedNotHeld);
                };
                if !Self::keys_match(&cur.body, &lease.body) {
                    return refuse(SeatDecision::RejectedStructural {
                        reason: "renew run/role does not match the slot".into(),
                    });
                }
                if lease.body.claimant != cur.body.claimant
                    || lease.body.leadership_term != cur.body.leadership_term
                    || lease.body.incarnation != cur.body.incarnation
                {
                    return refuse(SeatDecision::RejectedFencingConflict {
                        expected: cur.body.leadership_term,
                        got: lease.body.leadership_term,
                    });
                }
                self.store(lease)
            }
            SeatRequest::Release(release) => {
                if let Err(e) = release.body.validate() {
                    return refuse(SeatDecision::RejectedStructural {
                        reason: e.to_string(),
                    });
                }
                let Some(cur) = &self.lease else {
                    return refuse(SeatDecision::RejectedNotHeld);
                };
                if release.body.run_id != cur.body.run_id || release.body.role != cur.body.role {
                    return refuse(SeatDecision::RejectedStructural {
                        reason: "release run/role does not match the slot".into(),
                    });
                }
                if release.body.claimant != cur.body.claimant
                    || release.body.leadership_term != cur.body.leadership_term
                {
                    return refuse(SeatDecision::RejectedFencingConflict {
                        expected: cur.body.leadership_term,
                        got: release.body.leadership_term,
                    });
                }
                (
                    Self {
                        lease: None,
                        last_leadership_term: self.last_leadership_term,
                    },
                    SeatDecision::Accepted,
                )
            }
        }
    }

    /// Whether two lease bodies address the same slot key.
    fn keys_match(a: &SeatLeaseBody, b: &SeatLeaseBody) -> bool {
        a.run_id == b.run_id && a.role == b.role
    }

    /// Store `lease`, advancing the floor.
    fn store(&self, lease: &SeatLease) -> (Self, SeatDecision) {
        let floor = self
            .last_leadership_term
            .map_or(lease.body.leadership_term, |f| {
                f.max(lease.body.leadership_term)
            });
        (
            Self {
                lease: Some(lease.clone()),
                last_leadership_term: Some(floor),
            },
            SeatDecision::Accepted,
        )
    }
}

/// The response every seat mutation endpoint returns: the structural verdict plus the slot's
/// state after the fold (the current state, on a refusal — what the loser re-reads).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SeatMutationResponse {
    /// The registry's structural verdict.
    pub decision: SeatDecision,
    /// The slot state after the fold.
    pub state: SeatState,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cert::RunKeyCertificate;

    fn key(seed: u8) -> SigningKey {
        SigningKey::from_bytes(&[seed; 32])
    }

    fn run(n: u8) -> Hash {
        Hash([n; 32])
    }

    fn body(claimant: PeerId, incarnation: u64, term: u64) -> SeatLeaseBody {
        SeatLeaseBody {
            domain: SEAT_LEASE_DOMAIN.to_string(),
            run_id: run(1),
            role: "coordinator".into(),
            epoch: 0,
            incarnation,
            leadership_term: term,
            claimant,
            module_hash: Hash([0xCC; 32]),
            endpoint: ControlEndpoint {
                ws: Some("wss://registry.example/runs/demo/ws".into()),
                iroh_ticket: None,
            },
            issued_at_ms: 1_000,
            expires_at_ms: 31_000,
            heartbeat_interval_ms: DEFAULT_SEAT_HEARTBEAT_MS,
        }
    }

    /// A lease claimed by `run_key(seed)` at `(incarnation, term)`, certified by `base`.
    fn lease(base: &SigningKey, seed: u8, incarnation: u64, term: u64) -> SeatLease {
        let run_key = key(seed);
        let claimant = peer_id(&run_key);
        let b = body(claimant, incarnation, term);
        let cert = RunKeyCertificate::issue(base, b.cert_scope(), claimant).unwrap();
        SeatLease::claim(&run_key, cert, b).unwrap()
    }

    #[test]
    fn a_lease_round_trips_through_canonical_cbor_and_authorizes() {
        let base = key(1);
        // Term and incarnation are independent ordinals: a low local incarnation may hold a
        // high leadership term (leadership moved to a fresh box).
        let l = lease(&base, 2, 2, 40);
        let bytes = crate::canonical::to_canonical_vec(&l).unwrap();
        let back: SeatLease = crate::canonical::from_canonical_slice(&bytes).unwrap();
        assert_eq!(l, back);
        let trusted = [peer_id(&base)];
        assert!(back
            .authorize(&trusted, 5_000, DEFAULT_SEAT_SKEW_MS)
            .is_ok());
    }

    #[test]
    fn tampered_body_wrong_domain_and_out_of_domain_ordinals_are_refused_typed() {
        let base = key(1);
        let mut l = lease(&base, 2, 3, 7);
        l.body.epoch = 9; // re-scope without re-signing
        assert_eq!(l.verify_signature(), Err(SeatLeaseError::BadSignature));

        let run_key = key(2);
        let mut b = body(peer_id(&run_key), 0, 1);
        b.domain = SEAT_LEASE_DOMAIN_V1.into(); // the retired scheme's tag is not this scheme
        assert!(matches!(
            b.validate(),
            Err(SeatLeaseError::WrongDomain { .. })
        ));

        // Ordinals beyond i64::MAX are outside the shared bounded domain — structural refusal
        // (SQLite and the TS bigint fold could not represent/compare them identically).
        let mut b = body(peer_id(&run_key), 0, 1);
        b.leadership_term = MAX_ORDINAL + 1;
        assert_eq!(
            b.validate(),
            Err(SeatLeaseError::OutOfDomain {
                field: "leadership_term",
                value: MAX_ORDINAL + 1
            })
        );
        let mut b = body(peer_id(&run_key), 0, 1);
        b.incarnation = u64::MAX;
        assert!(matches!(
            b.validate(),
            Err(SeatLeaseError::OutOfDomain {
                field: "incarnation",
                ..
            })
        ));
        // The boundary value itself is in-domain.
        let b = body(peer_id(&run_key), MAX_ORDINAL, MAX_ORDINAL);
        assert!(b.validate().is_ok());
    }

    #[test]
    fn claim_constructor_refuses_invalid_bodies_and_foreign_keys() {
        let base = key(1);
        let run_key = key(2);
        let claimant = peer_id(&run_key);
        let cert =
            RunKeyCertificate::issue(&base, body(claimant, 4, 9).cert_scope(), claimant).unwrap();

        // No dialable endpoint.
        let mut b = body(claimant, 4, 9);
        b.endpoint = ControlEndpoint::default();
        assert!(SeatLease::claim(&run_key, cert.clone(), b).is_err());

        // A key that is not the body's claimant cannot author the lease.
        let b = body(claimant, 4, 9);
        assert!(SeatLease::claim(&key(3), cert, b).is_err());
    }

    #[test]
    fn authorize_refuses_untrusted_base_cert_mismatch_and_expiry() {
        let base = key(1);
        let trusted = [peer_id(&base)];
        let l = lease(&base, 2, 0, 1);

        // Untrusted base: the same object under a different trust set is refused.
        assert_eq!(
            l.authorize(&[peer_id(&key(9))], 5_000, 0),
            Err(SeatLeaseError::UntrustedBase)
        );

        // A certificate for a different scope (wrong module) is a typed cert refusal.
        let run_key = key(2);
        let claimant = peer_id(&run_key);
        let b = body(claimant, 0, 1);
        let mut wrong_scope = b.cert_scope();
        wrong_scope.module_hash = Hash([0xDD; 32]);
        let cert = RunKeyCertificate::issue(&base, wrong_scope, claimant).unwrap();
        let l2 = SeatLease::claim(&run_key, cert, b).unwrap();
        assert!(matches!(
            l2.authorize(&trusted, 5_000, 0),
            Err(SeatLeaseError::Cert(CertError::ModuleMismatch))
        ));

        // Expiry honors the skew grace: expired at expiry+skew+1, live at expiry+skew.
        assert!(!l.is_expired(l.body.expires_at_ms + 5_000, 5_000));
        assert!(l.is_expired(l.body.expires_at_ms + 5_001, 5_000));
        assert!(matches!(
            l.authorize(&trusted, l.body.expires_at_ms + 5_001, 5_000),
            Err(SeatLeaseError::Expired { .. })
        ));
    }

    #[test]
    fn sparse_terms_claim_over_the_floor_and_dense_arithmetic_is_never_required() {
        // The scope separation's operational core: a takeover claims at ANY term strictly above
        // the floor — the successor's term derives from ITS state, not the predecessor's + 1.
        let base = key(1);
        let slot = SeatSlot::new();
        // First claim: the presented term stands (floor None).
        let l0 = lease(&base, 2, 5, 17);
        let (slot, d) = slot.fold(&SeatRequest::Claim(l0.clone()), 2_000, 5_000);
        assert_eq!(d, SeatDecision::Accepted);
        assert_eq!(slot.last_leadership_term, Some(17));

        // A foreign claim while the incumbent is live is held out regardless of term.
        let foreign = lease(&base, 6, 1, 40);
        let (_, d) = slot.fold(&SeatRequest::Claim(foreign.clone()), 2_000, 5_000);
        assert_eq!(d, SeatDecision::RejectedHeld);

        // After expiry, a sparse jump (17 → 40) is accepted; the floor follows.
        let (slot2, d) = slot.fold(&SeatRequest::Claim(foreign), 99_999, 5_000);
        assert_eq!(d, SeatDecision::Accepted);
        assert_eq!(slot2.last_leadership_term, Some(40));

        // An equal-or-lower term is a fencing conflict against the floor.
        let stale = lease(&base, 7, 9, 40);
        let (_, d) = slot2.fold(&SeatRequest::Claim(stale), 99_999, 5_000);
        assert_eq!(
            d,
            SeatDecision::RejectedFencingConflict {
                expected: 40,
                got: 40
            }
        );
    }

    #[test]
    fn a_new_execution_identity_never_renews_it_claims_fresh_above_the_floor() {
        // Renew is bound to the exact held execution identity: a new incarnation mints a new
        // per-run key, hence a new claimant — the fold sees a foreign renew and refuses. The
        // successor execution claims fresh at a higher term instead.
        let base = key(1);
        let held = lease(&base, 2, 3, 10);
        let slot = SeatSlot {
            lease: Some(held.clone()),
            last_leadership_term: Some(10),
        };
        // Same box, NEW incarnation (new key seed): renew refused — not the held claimant.
        let successor = lease(&base, 4, 4, 10);
        let (_, d) = slot.fold(&SeatRequest::Renew(successor), 2_000, 5_000);
        assert_eq!(
            d,
            SeatDecision::RejectedFencingConflict {
                expected: 10,
                got: 10
            }
        );
        // The held claimant renewing with a CHANGED incarnation is equally refused (the lease
        // binds one execution identity).
        let run_key = key(2);
        let claimant = peer_id(&run_key);
        let mut b = body(claimant, 4, 10);
        b.expires_at_ms = 90_000;
        let cert = RunKeyCertificate::issue(&base, b.cert_scope(), claimant).unwrap();
        let rebound = SeatLease::claim(&run_key, cert, b).unwrap();
        let (_, d) = slot.fold(&SeatRequest::Renew(rebound), 2_000, 5_000);
        assert!(matches!(d, SeatDecision::RejectedFencingConflict { .. }));
        // After expiry the successor claims fresh above the floor — the honest path.
        let successor = lease(&base, 4, 4, 11);
        let (next, d) = slot.fold(&SeatRequest::Claim(successor), 99_999, 5_000);
        assert_eq!(d, SeatDecision::Accepted);
        assert_eq!(next.last_leadership_term, Some(11));
    }

    #[test]
    fn an_epoch_rebind_renews_under_the_same_term_with_a_reissued_certificate() {
        // Contract: a renew across an epoch boundary rebinds the certificate (same key, new
        // epoch, possibly a new module), same incarnation, same term — a renew, not a takeover.
        let base = key(1);
        let run_key = key(2);
        let claimant = peer_id(&run_key);
        let l0 = lease(&base, 2, 3, 12);

        let mut b1 = body(claimant, 3, 12);
        b1.epoch = 1;
        b1.module_hash = Hash([0xEE; 32]);
        b1.expires_at_ms = 90_000;
        let cert1 = RunKeyCertificate::issue(&base, b1.cert_scope(), claimant).unwrap();
        let l1 = SeatLease::claim(&run_key, cert1, b1).unwrap();

        let slot = SeatSlot {
            lease: Some(l0),
            last_leadership_term: Some(12),
        };
        let (next, decision) = slot.fold(&SeatRequest::Renew(l1.clone()), 40_000, 5_000);
        assert_eq!(decision, SeatDecision::Accepted);
        assert_eq!(next.lease, Some(l1.clone()));
        assert_eq!(next.last_leadership_term, Some(12));
        // And the rebound lease authorizes under the new epoch's certificate.
        assert!(l1.authorize(&[peer_id(&base)], 40_000, 5_000).is_ok());
    }

    #[test]
    fn release_signs_verifies_and_folds_to_a_tombstoned_slot() {
        let base = key(1);
        let run_key = key(2);
        let l = lease(&base, 2, 4, 20);
        let release = SeatRelease::sign(
            &run_key,
            SeatReleaseBody {
                domain: SEAT_RELEASE_DOMAIN.to_string(),
                run_id: l.body.run_id,
                role: l.body.role.clone(),
                incarnation: 4,
                leadership_term: 20,
                claimant: l.body.claimant,
            },
        )
        .unwrap();
        assert!(release.verify_signature().is_ok());

        let slot = SeatSlot {
            lease: Some(l),
            last_leadership_term: Some(20),
        };
        let (next, decision) = slot.fold(&SeatRequest::Release(release), 2_000, 5_000);
        assert_eq!(decision, SeatDecision::Accepted);
        assert_eq!(
            next.state(),
            SeatState::Unclaimed {
                last_leadership_term: Some(20)
            }
        );
        // The floor persists: the next claim must exceed 20 — sparsely.
        let next_claim = lease(&base, 6, 1, 33);
        let (after, decision) = next.fold(&SeatRequest::Claim(next_claim), 3_000, 5_000);
        assert_eq!(decision, SeatDecision::Accepted);
        assert_eq!(after.last_leadership_term, Some(33));
        // A re-claim at the released term is a fencing conflict.
        let stale = lease(&base, 7, 9, 20);
        let (_, decision) = next.fold(&SeatRequest::Claim(stale), 3_000, 5_000);
        assert_eq!(
            decision,
            SeatDecision::RejectedFencingConflict {
                expected: 20,
                got: 20
            }
        );
    }

    #[test]
    fn a_release_signature_is_never_a_lease_signature() {
        // Domain separation: the release preimage can never verify under the lease domain
        // (distinct domain strings; distinct body shapes are not relied upon).
        let run_key = key(2);
        let claimant = peer_id(&run_key);
        let mut b = SeatReleaseBody {
            domain: SEAT_LEASE_DOMAIN.to_string(), // a lease tag on a release body
            run_id: run(1),
            role: "coordinator".into(),
            incarnation: 1,
            leadership_term: 1,
            claimant,
        };
        assert!(matches!(
            b.validate(),
            Err(SeatLeaseError::WrongDomain { .. })
        ));
        b.domain = SEAT_RELEASE_DOMAIN.to_string();
        assert!(b.validate().is_ok());
    }

    #[test]
    fn the_term_ledger_advances_only_on_verified_grants_and_fences_predecessors() {
        let base = key(1);
        let incumbent = lease(&base, 2, 3, 10);
        let mut ledger = SeatTermLedger::new();
        assert_eq!(ledger.floor(&run(1), "coordinator"), None);
        assert_eq!(
            ledger.binds(&run(1), "coordinator", &incumbent.body.claimant),
            None
        );

        ledger.observe_verified_grant(&incumbent);
        assert_eq!(ledger.floor(&run(1), "coordinator"), Some(10));
        assert!(ledger.judge(&incumbent).is_ok());
        assert_eq!(
            ledger.binds(&run(1), "coordinator", &incumbent.body.claimant),
            Some(true)
        );

        // A successor's verified grant at a sparse higher term fences the incumbent.
        let successor = lease(&base, 4, 1, 25);
        ledger.observe_verified_grant(&successor);
        assert_eq!(ledger.floor(&run(1), "coordinator"), Some(25));
        assert_eq!(
            ledger.judge(&incumbent),
            Err(SeatLeaseError::TermSuperseded { got: 10, floor: 25 })
        );
        assert_eq!(
            ledger.binds(&run(1), "coordinator", &incumbent.body.claimant),
            Some(false)
        );
        assert_eq!(
            ledger.binds(&run(1), "coordinator", &successor.body.claimant),
            Some(true)
        );

        // A stale grant never rolls the floor back.
        ledger.observe_verified_grant(&incumbent);
        assert_eq!(ledger.floor(&run(1), "coordinator"), Some(25));

        // An equal term under a DIFFERENT claimant is a collision — fail closed.
        let equivocation = lease(&base, 6, 9, 25);
        assert_eq!(
            ledger.judge(&equivocation),
            Err(SeatLeaseError::TermSuperseded { got: 25, floor: 25 })
        );
    }

    #[test]
    fn v1_bodies_interpret_read_only_and_never_validate_as_v2() {
        // The archived v1 shape decodes for forensics; its interpretation maps the conflated
        // ordinal into the two scopes. It can never re-enter the live system: the v2 validate
        // refuses the v1 domain tag, and the v1 preimage cannot carry a v2 signature.
        let v1 = SeatLeaseBodyV1 {
            domain: SEAT_LEASE_DOMAIN_V1.to_string(),
            run_id: run(1),
            role: "coordinator".into(),
            epoch: 0,
            incarnation: 19,
            fencing_token: 19,
            claimant: peer_id(&key(2)),
            module_hash: Hash([0xCC; 32]),
            endpoint: ControlEndpoint {
                ws: Some("wss://x".into()),
                iroh_ticket: None,
            },
            issued_at_ms: 1,
            expires_at_ms: 2,
            heartbeat_interval_ms: 3,
        };
        assert_eq!(v1.interpret().unwrap(), (19, 19));
        let bytes = crate::canonical::to_canonical_vec(&v1).unwrap();
        let back: SeatLeaseBodyV1 = crate::canonical::from_canonical_slice(&bytes).unwrap();
        assert_eq!(back, v1);
        assert!(is_v1_release_domain(SEAT_RELEASE_DOMAIN_V1));
        assert!(!is_v1_release_domain(SEAT_RELEASE_DOMAIN));
    }

    #[test]
    fn seat_state_and_response_round_trip_through_canonical_cbor() {
        let base = key(1);
        let l = lease(&base, 2, 0, 1);
        for state in [
            SeatState::Unclaimed {
                last_leadership_term: None,
            },
            SeatState::Unclaimed {
                last_leadership_term: Some(7),
            },
            SeatState::Leased(Box::new(l)),
        ] {
            let resp = SeatMutationResponse {
                decision: SeatDecision::Accepted,
                state: state.clone(),
            };
            let bytes = crate::canonical::to_canonical_vec(&resp).unwrap();
            let back: SeatMutationResponse =
                crate::canonical::from_canonical_slice(&bytes).unwrap();
            assert_eq!(back.state, state);
        }
        // The refusal variants round-trip too.
        for decision in [
            SeatDecision::RejectedStructural { reason: "r".into() },
            SeatDecision::RejectedHeld,
            SeatDecision::RejectedNotHeld,
            SeatDecision::RejectedFencingConflict {
                expected: 2,
                got: 1,
            },
        ] {
            let bytes = crate::canonical::to_canonical_vec(&decision).unwrap();
            let back: SeatDecision = crate::canonical::from_canonical_slice(&bytes).unwrap();
            assert_eq!(back, decision);
        }
    }
}
