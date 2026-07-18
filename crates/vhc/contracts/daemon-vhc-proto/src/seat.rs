// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The **coordinator seat lease** — an Authority-signed, fenced claim on a run's coordinator
//! role, stored and compare-and-swapped by the registry (architecture §6.3; the registry is
//! untrusted storage and never becomes consensus authority).
//!
//! A seat claim is a canonical-CBOR [`SeatLeaseBody`] signed by the **claimant's certified
//! per-run key**, distributed as a [`SeatLease`] that carries the claimant's
//! [`RunKeyCertificate`] beside the body (a separate distribution record, never a frame-envelope
//! field — ABI §12.1). The registry stores the signed object and CASes on the **fencing token**
//! (classic compare-and-set with increment) but creates no authority: peers verify the lease
//! signature, the certificate chain to a genesis-named base identity, and the supersession floor
//! (architecture §6.3.1) themselves. A stale claimant's records are refused once a higher fencing
//! token exists, regardless of what the registry says.
//!
//! **Fencing token ≡ incarnation.** The fencing token is bound to the role-instance incarnation
//! (`fencing_token == incarnation`, both carried explicitly; verifiers assert equality). Because
//! the registry accepts a claim only at `stored + 1`, the seat slot allocates the coordinator
//! role's incarnations in step with the certificate chain: a takeover is a new incarnation, which
//! is exactly what advances the peers' supersession floor (implicit revocation of the fenced
//! predecessor — [`crate::revocation::RevocationLedger`]).
//!
//! **Wall clock is liveness, never safety.** Expiry (`expires_at_ms` + a bounded skew grace)
//! only gates when a takeover may be attempted; the fencing token is what fences the superseded
//! claimant. A premature, skew-driven takeover costs liveness, not safety.
//!
//! The registry-side acceptance rule is the pure [`SeatSlot::fold`] — normative CAS semantics any
//! conforming seat registry implements bit-for-bit (the shared test vectors pin it), exactly as
//! [`crate::revocation::RevocationLedger`] pins receiver-side revocation semantics. The fold
//! validates **structure only** (domain tag, token≡incarnation, windows, endpoint presence, slot
//! consistency); it MUST NOT verify signatures or judge authority — peers do that.

use serde::{Deserialize, Serialize};

use crate::bytes::{Hash, PeerId, Signature};
use crate::cert::{CertError, CertScope, RunKeyCertificate};
use crate::domains::{SEAT_LEASE_DOMAIN, SEAT_RELEASE_DOMAIN};
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

/// The signed body of a coordinator seat lease. Every field is part of the signed preimage.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SeatLeaseBody {
    /// Domain-separation tag — MUST be [`SEAT_LEASE_DOMAIN`].
    pub domain: String,
    /// The run's cryptographic identity: the genesis-envelope hash (ABI §8.1 `run_id`).
    pub run_id: Hash,
    /// The envelope role label being claimed (e.g. `coordinator`).
    pub role: String,
    /// The run epoch this lease is scoped to. An epoch change REBINDS on renew (same
    /// incarnation, same fencing token, reissued certificate) — never a takeover.
    pub epoch: u64,
    /// The never-reused monotonic role-instance incarnation id (ABI §8.1 `instance`).
    pub incarnation: u64,
    /// The monotonic fencing token. INVARIANT: `fencing_token == incarnation` (both carried for
    /// explicitness; every verifier asserts equality). The registry accepts a claim only at
    /// `stored + 1`.
    pub fencing_token: u64,
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
    /// Claimant wall clock at expiry (milliseconds). Liveness only — never safety (the fencing
    /// token is the safety mechanism).
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
    /// The releasing incarnation.
    pub incarnation: u64,
    /// The released fencing token. INVARIANT: `fencing_token == incarnation`.
    pub fencing_token: u64,
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
    /// `fencing_token != incarnation` — the bound-token invariant is violated.
    TokenIncarnationMismatch {
        /// The carried fencing token.
        fencing_token: u64,
        /// The carried incarnation.
        incarnation: u64,
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
}

impl core::fmt::Display for SeatLeaseError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::WrongDomain { got } => {
                write!(f, "seat domain `{got}` is not the expected seat domain")
            }
            Self::TokenIncarnationMismatch {
                fencing_token,
                incarnation,
            } => write!(
                f,
                "fencing token {fencing_token} is not bound to incarnation {incarnation}"
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

impl SeatLeaseBody {
    /// Structural validation — the invariants BOTH sides enforce (the registry before storing,
    /// every verifier before trusting): domain tag, token≡incarnation, a positive expiry window,
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
        if self.fencing_token != self.incarnation {
            return Err(SeatLeaseError::TokenIncarnationMismatch {
                fencing_token: self.fencing_token,
                incarnation: self.incarnation,
            });
        }
        if self.expires_at_ms <= self.issued_at_ms || self.heartbeat_interval_ms == 0 {
            return Err(SeatLeaseError::InvalidWindow);
        }
        if !self.endpoint.is_dialable() {
            return Err(SeatLeaseError::MissingEndpoint);
        }
        Ok(())
    }

    /// The execution-identity scope this lease claims — what the embedded certificate must bind.
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
    /// [`crate::revocation::RevocationLedger::judge`] over [`SeatLeaseBody::cert_scope`] for
    /// explicit-revocation + supersession-floor enforcement (architecture §6.3.1) — a lease below
    /// the floor is dead even if everything here verifies.
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

    /// Whether the lease is past `expires_at_ms` plus the skew grace. Expiry gates takeover
    /// liveness only; fencing is the safety mechanism.
    #[must_use]
    pub fn is_expired(&self, now_ms: u64, skew_ms: u64) -> bool {
        now_ms > self.body.expires_at_ms.saturating_add(skew_ms)
    }
}

impl SeatReleaseBody {
    /// Structural validation: domain tag + token≡incarnation.
    ///
    /// # Errors
    /// The applicable [`SeatLeaseError`].
    pub fn validate(&self) -> Result<(), SeatLeaseError> {
        if self.domain != SEAT_RELEASE_DOMAIN {
            return Err(SeatLeaseError::WrongDomain {
                got: self.domain.clone(),
            });
        }
        if self.fencing_token != self.incarnation {
            return Err(SeatLeaseError::TokenIncarnationMismatch {
                fencing_token: self.fencing_token,
                incarnation: self.incarnation,
            });
        }
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

// -- the registry-side slot + the normative CAS fold ------------------------------------------------

/// The readable projection of one run-role seat slot.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SeatState {
    /// No lease holds the seat. `last_fencing_token` is the release/expiry tombstone floor —
    /// tokens never reset; the next claim must present `floor + 1` (`None` = never claimed:
    /// the first claim's token is accepted as presented).
    Unclaimed {
        /// The tombstone floor.
        last_fencing_token: Option<u64>,
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
    /// Claim / take over the seat (CAS at `stored + 1`).
    Claim(SeatLease),
    /// Renew (heartbeat) the held lease — same claimant, same incarnation, same token; the
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
    /// The object violates a structural invariant (domain, token≡incarnation, window, endpoint).
    RejectedStructural {
        /// A human-readable reason (diagnostic only; never authority).
        reason: String,
    },
    /// A live (unexpired, by the registry clock + skew) lease by a DIFFERENT claimant holds the
    /// seat.
    RejectedHeld,
    /// Nothing to renew/release — the slot is unclaimed.
    RejectedNotHeld,
    /// The compare-and-swap failed: the presented fencing token does not match what the slot
    /// requires. The loser re-reads and either accepts the incumbent or retries at the floor + 1.
    RejectedFencingConflict {
        /// The token the slot requires (claim: `stored + 1`; renew/release: the held token).
        expected: u64,
        /// The token the request presented.
        got: u64,
    },
}

/// One run-role seat slot as the registry stores it: the current lease (if any) plus the
/// monotonic token floor. INVARIANT: while leased, `last_fencing_token ==
/// Some(lease.body.fencing_token)`; after a release/takeover the floor persists (tokens never
/// reset).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SeatSlot {
    /// The stored lease, if the seat is claimed.
    pub lease: Option<SeatLease>,
    /// The highest fencing token this slot has ever stored (the tombstone floor).
    pub last_fencing_token: Option<u64>,
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
                last_fencing_token: self.last_fencing_token,
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
    /// and the token CAS. NO signature verification, NO authority judgment.
    ///
    /// Claim acceptance:
    /// - held by the SAME claimant: the held token (idempotent refresh) or `held + 1`
    ///   (self-supersession — a planned handover to the claimant's own next incarnation);
    /// - held by another claimant and NOT expired: refused (`RejectedHeld`);
    /// - expired or unclaimed: `floor + 1` exactly (`None` floor = first claim, any token).
    ///
    /// Renew requires the identity five-tuple `(run_id, role, incarnation, fencing_token,
    /// claimant)` to match the held lease exactly; epoch/module/endpoint/expiry may change (the
    /// epoch-rebind rule). An expired-but-untaken lease may still renew — grace, not safety.
    ///
    /// Release requires `(run_id, role, fencing_token, claimant)` to match; the floor persists.
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
                        // Idempotent refresh at the held token, or self-supersession at +1.
                        if lease.body.fencing_token == cur.body.fencing_token
                            || lease.body.fencing_token == cur.body.fencing_token + 1
                        {
                            return self.store(lease);
                        }
                        return refuse(SeatDecision::RejectedFencingConflict {
                            expected: cur.body.fencing_token + 1,
                            got: lease.body.fencing_token,
                        });
                    }
                    if !cur.is_expired(now_ms, skew_ms) {
                        return refuse(SeatDecision::RejectedHeld);
                    }
                }
                match self.last_fencing_token {
                    None => self.store(lease),
                    Some(floor) if lease.body.fencing_token == floor + 1 => self.store(lease),
                    Some(floor) => refuse(SeatDecision::RejectedFencingConflict {
                        expected: floor + 1,
                        got: lease.body.fencing_token,
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
                    || lease.body.fencing_token != cur.body.fencing_token
                    || lease.body.incarnation != cur.body.incarnation
                {
                    return refuse(SeatDecision::RejectedFencingConflict {
                        expected: cur.body.fencing_token,
                        got: lease.body.fencing_token,
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
                    || release.body.fencing_token != cur.body.fencing_token
                {
                    return refuse(SeatDecision::RejectedFencingConflict {
                        expected: cur.body.fencing_token,
                        got: release.body.fencing_token,
                    });
                }
                (
                    Self {
                        lease: None,
                        last_fencing_token: self.last_fencing_token,
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
            .last_fencing_token
            .map_or(lease.body.fencing_token, |f| {
                f.max(lease.body.fencing_token)
            });
        (
            Self {
                lease: Some(lease.clone()),
                last_fencing_token: Some(floor),
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

    fn body(claimant: PeerId, incarnation: u64) -> SeatLeaseBody {
        SeatLeaseBody {
            domain: SEAT_LEASE_DOMAIN.to_string(),
            run_id: run(1),
            role: "coordinator".into(),
            epoch: 0,
            incarnation,
            fencing_token: incarnation,
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

    /// A lease claimed by `run_key(seed)` at `incarnation`, certified by `base`.
    fn lease(base: &SigningKey, seed: u8, incarnation: u64) -> SeatLease {
        let run_key = key(seed);
        let claimant = peer_id(&run_key);
        let b = body(claimant, incarnation);
        let cert = RunKeyCertificate::issue(base, b.cert_scope(), claimant).unwrap();
        SeatLease::claim(&run_key, cert, b).unwrap()
    }

    #[test]
    fn a_lease_round_trips_through_canonical_cbor_and_authorizes() {
        let base = key(1);
        let l = lease(&base, 2, 0);
        let bytes = crate::canonical::to_canonical_vec(&l).unwrap();
        let back: SeatLease = crate::canonical::from_canonical_slice(&bytes).unwrap();
        assert_eq!(l, back);
        let trusted = [peer_id(&base)];
        assert!(back
            .authorize(&trusted, 5_000, DEFAULT_SEAT_SKEW_MS)
            .is_ok());
    }

    #[test]
    fn tampered_body_and_wrong_domain_are_refused_typed() {
        let base = key(1);
        let mut l = lease(&base, 2, 0);
        l.body.fencing_token = 1; // breaks token≡incarnation AND the signature
        assert_eq!(
            l.verify_signature(),
            Err(SeatLeaseError::TokenIncarnationMismatch {
                fencing_token: 1,
                incarnation: 0
            })
        );
        let mut l = lease(&base, 2, 3);
        l.body.epoch = 9; // re-scope without re-signing
        assert_eq!(l.verify_signature(), Err(SeatLeaseError::BadSignature));

        let run_key = key(2);
        let mut b = body(peer_id(&run_key), 0);
        b.domain = "daemon-vhc/cert/2".into();
        assert!(matches!(
            b.validate(),
            Err(SeatLeaseError::WrongDomain { .. })
        ));
    }

    #[test]
    fn claim_constructor_refuses_invalid_bodies_and_foreign_keys() {
        let base = key(1);
        let run_key = key(2);
        let claimant = peer_id(&run_key);

        // Token not bound to the incarnation.
        let mut b = body(claimant, 4);
        b.fencing_token = 5;
        let cert = RunKeyCertificate::issue(&base, b.cert_scope(), claimant).unwrap();
        assert!(SeatLease::claim(&run_key, cert.clone(), b).is_err());

        // No dialable endpoint.
        let mut b = body(claimant, 4);
        b.endpoint = ControlEndpoint::default();
        assert!(SeatLease::claim(&run_key, cert.clone(), b).is_err());

        // A key that is not the body's claimant cannot author the lease.
        let b = body(claimant, 4);
        assert!(SeatLease::claim(&key(3), cert, b).is_err());
    }

    #[test]
    fn authorize_refuses_untrusted_base_cert_mismatch_and_expiry() {
        let base = key(1);
        let trusted = [peer_id(&base)];
        let l = lease(&base, 2, 0);

        // Untrusted base: the same object under a different trust set is refused.
        assert_eq!(
            l.authorize(&[peer_id(&key(9))], 5_000, 0),
            Err(SeatLeaseError::UntrustedBase)
        );

        // A certificate for a different scope (wrong module) is a typed cert refusal.
        let run_key = key(2);
        let claimant = peer_id(&run_key);
        let b = body(claimant, 0);
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
    fn an_epoch_rebind_renews_under_the_same_token_with_a_reissued_certificate() {
        // Contract: a renew across an epoch boundary rebinds the certificate (same key, new epoch,
        // possibly a new module), same incarnation, same fencing token — a renew, not a takeover.
        let base = key(1);
        let run_key = key(2);
        let claimant = peer_id(&run_key);
        let l0 = lease(&base, 2, 3);

        let mut b1 = body(claimant, 3);
        b1.epoch = 1;
        b1.module_hash = Hash([0xEE; 32]);
        b1.expires_at_ms = 90_000;
        let cert1 = RunKeyCertificate::issue(&base, b1.cert_scope(), claimant).unwrap();
        let l1 = SeatLease::claim(&run_key, cert1, b1).unwrap();

        let slot = SeatSlot {
            lease: Some(l0),
            last_fencing_token: Some(3),
        };
        let (next, decision) = slot.fold(&SeatRequest::Renew(l1.clone()), 40_000, 5_000);
        assert_eq!(decision, SeatDecision::Accepted);
        assert_eq!(next.lease, Some(l1.clone()));
        assert_eq!(next.last_fencing_token, Some(3));
        // And the rebound lease authorizes under the new epoch's certificate.
        assert!(l1.authorize(&[peer_id(&base)], 40_000, 5_000).is_ok());
    }

    #[test]
    fn release_signs_verifies_and_folds_to_a_tombstoned_slot() {
        let base = key(1);
        let run_key = key(2);
        let l = lease(&base, 2, 4);
        let release = SeatRelease::sign(
            &run_key,
            SeatReleaseBody {
                domain: SEAT_RELEASE_DOMAIN.to_string(),
                run_id: l.body.run_id,
                role: l.body.role.clone(),
                incarnation: 4,
                fencing_token: 4,
                claimant: l.body.claimant,
            },
        )
        .unwrap();
        assert!(release.verify_signature().is_ok());

        let slot = SeatSlot {
            lease: Some(l),
            last_fencing_token: Some(4),
        };
        let (next, decision) = slot.fold(&SeatRequest::Release(release), 2_000, 5_000);
        assert_eq!(decision, SeatDecision::Accepted);
        assert_eq!(
            next.state(),
            SeatState::Unclaimed {
                last_fencing_token: Some(4)
            }
        );
        // The floor persists: the next claim must present 5.
        let next_claim = lease(&base, 6, 5);
        let (after, decision) = next.fold(&SeatRequest::Claim(next_claim), 3_000, 5_000);
        assert_eq!(decision, SeatDecision::Accepted);
        assert_eq!(after.last_fencing_token, Some(5));
        // A re-claim at the released token is a fencing conflict.
        let stale = lease(&base, 7, 4);
        let (_, decision) = next.fold(&SeatRequest::Claim(stale), 3_000, 5_000);
        assert_eq!(
            decision,
            SeatDecision::RejectedFencingConflict {
                expected: 5,
                got: 4
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
            fencing_token: 1,
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
    fn seat_state_and_response_round_trip_through_canonical_cbor() {
        let base = key(1);
        let l = lease(&base, 2, 0);
        for state in [
            SeatState::Unclaimed {
                last_fencing_token: None,
            },
            SeatState::Unclaimed {
                last_fencing_token: Some(7),
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
