// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The coordinator **seat manager** (architecture §6.3; ABI §12.4; D-P9) — an admitted node's
//! claim on a run's coordinator role through an Authority-signed, fenced lease.
//!
//! The registry is UNTRUSTED STORAGE: it stores the signed lease and compare-and-swaps on the
//! fencing token, but never judges authority. This manager owns the CLAIMANT half:
//!
//! - **bid derivation** from the slot's current state (unclaimed → `floor + 1`; an expired lease
//!   → `held + 1`; a live lease → stand by);
//! - **lease authorship**: provision the coordinator per-run identity (mint key + certificate
//!   under the base identity, [`crate::credentials`]/[`daemon_vhc_session::provisioning`]) at the
//!   bid incarnation (`fencing_token == incarnation`, [SEAT-1]), and sign the lease with it;
//! - **renew** (heartbeat): re-sign the body with a fresh expiry under the SAME token;
//! - **release**: sign a release the registry tombstones (the floor persists);
//! - **peer-side acceptance** ([`authorize_incumbent`]): the trainer's judgment of a stored lease
//!   — signature + certificate chain to a genesis-trusted base + the supersession floor. A stale
//!   claimant is refused once a higher fencing token exists, regardless of the registry ([SEAT-3]).
//!
//! Fencing is safety; wall-clock expiry gates takeover liveness only. Failover is mechanism-only
//! for this program (a standby claim at `floor + 1` after expiry) — nothing here precludes it, and
//! nothing here makes it automatic (fencing-is-safe-not-seamless).

use daemon_vhc_proto::domains::{SEAT_LEASE_DOMAIN, SEAT_RELEASE_DOMAIN};
use daemon_vhc_proto::Hash;
use daemon_vhc_proto::{
    peer_id, ControlEndpoint, PeerId, RevocationLedger, SeatLease, SeatLeaseBody, SeatLeaseError,
    SeatRelease, SeatReleaseBody, SeatState, SigningKey, DEFAULT_SEAT_HEARTBEAT_MS,
    DEFAULT_SEAT_SKEW_MS, DEFAULT_SEAT_TTL_MS,
};
use daemon_vhc_session::keystore::VhcKeystore;
use daemon_vhc_session::provisioning::{provision_run_identity, ProvisionScope};

/// The scope a coordinator seat claim is authored under.
pub struct CoordinatorSeat<'a> {
    /// The run label (keystore namespace + registry run key).
    pub run_label: &'a str,
    /// The run's cryptographic identity (genesis hash).
    pub genesis_hash: [u8; 32],
    /// The claimed role label (the envelope's coordinator role).
    pub role: &'a str,
    /// The run epoch the lease is scoped to.
    pub epoch: u64,
    /// The pinned coordinator module hash.
    pub module_hash: [u8; 32],
    /// The control-plane endpoint peers dial while this lease holds the seat.
    pub endpoint: ControlEndpoint,
}

/// A seat-manager error.
#[derive(Debug, thiserror::Error)]
pub enum SeatError {
    /// The slot is held by a live incumbent — stand by (no bid).
    #[error("seat held by a live incumbent (stand by)")]
    HeldByIncumbent,
    /// Identity provisioning / signing failed.
    #[error("seat identity: {0}")]
    Identity(String),
    /// Lease authorship failed structurally.
    #[error("seat lease: {0}")]
    Lease(#[from] SeatLeaseError),
    /// A proto encode/sign failure.
    #[error("seat proto: {0}")]
    Proto(String),
}

/// Derive the fencing-token bid for a fresh claim from the slot's current state, or `None` when a
/// LIVE incumbent holds the seat (the caller stands by). Both carried tokens equal the
/// incarnation the claimant will mint ([SEAT-1]).
#[must_use]
pub fn derive_bid(state: &SeatState, now_ms: u64, skew_ms: u64) -> Option<u64> {
    match state {
        SeatState::Unclaimed { last_fencing_token } => {
            Some(last_fencing_token.map_or(0, |f| f + 1))
        }
        SeatState::Leased(lease) => {
            if lease.is_expired(now_ms, skew_ms) {
                Some(lease.body.fencing_token + 1)
            } else {
                None
            }
        }
    }
}

/// Author a seat lease at `bid`: provision the coordinator per-run identity (mint key + issue the
/// certificate under the base identity) and sign the lease with it. `bid == incarnation ==
/// fencing_token` ([SEAT-1]).
///
/// # Errors
/// Identity provisioning, keystore, or signing failure.
pub fn author_claim(
    keystore: &VhcKeystore,
    seat: &CoordinatorSeat<'_>,
    bid: u64,
    now_ms: u64,
) -> Result<SeatLease, SeatError> {
    let cert = provision_run_identity(
        keystore,
        &ProvisionScope {
            run_label: seat.run_label,
            genesis_hash: seat.genesis_hash,
            epoch: seat.epoch,
            role: seat.role,
            incarnation: bid,
            module_hash: seat.module_hash,
        },
    )
    .map_err(|e| SeatError::Identity(e.to_string()))?;
    let run_key = coordinator_key(keystore, seat, bid)?;
    let body = SeatLeaseBody {
        domain: SEAT_LEASE_DOMAIN.to_string(),
        run_id: Hash(seat.genesis_hash),
        role: seat.role.to_string(),
        epoch: seat.epoch,
        incarnation: bid,
        fencing_token: bid,
        claimant: peer_id(&run_key),
        module_hash: Hash(seat.module_hash),
        endpoint: seat.endpoint.clone(),
        issued_at_ms: now_ms,
        expires_at_ms: now_ms + DEFAULT_SEAT_TTL_MS,
        heartbeat_interval_ms: DEFAULT_SEAT_HEARTBEAT_MS,
    };
    SeatLease::claim(&run_key, cert, body).map_err(|e| SeatError::Proto(e.to_string()))
}

/// Re-sign the held lease with a fresh expiry under the SAME incarnation/token — the heartbeat
/// renew (never a takeover; the epoch/module/endpoint may change, [SEAT-2]).
///
/// # Errors
/// Keystore or signing failure.
pub fn author_renew(
    keystore: &VhcKeystore,
    seat: &CoordinatorSeat<'_>,
    held: &SeatLease,
    now_ms: u64,
) -> Result<SeatLease, SeatError> {
    let run_key = coordinator_key(keystore, seat, held.body.incarnation)?;
    let mut body = held.body.clone();
    body.issued_at_ms = now_ms;
    body.expires_at_ms = now_ms + DEFAULT_SEAT_TTL_MS;
    SeatLease::claim(&run_key, held.certificate.clone(), body)
        .map_err(|e| SeatError::Proto(e.to_string()))
}

/// Author a release for the held lease (the registry tombstones the token; the floor persists).
///
/// # Errors
/// Keystore or signing failure.
pub fn author_release(
    keystore: &VhcKeystore,
    seat: &CoordinatorSeat<'_>,
    incarnation: u64,
) -> Result<SeatRelease, SeatError> {
    let run_key = coordinator_key(keystore, seat, incarnation)?;
    let body = SeatReleaseBody {
        domain: SEAT_RELEASE_DOMAIN.to_string(),
        run_id: Hash(seat.genesis_hash),
        role: seat.role.to_string(),
        incarnation,
        fencing_token: incarnation,
        claimant: peer_id(&run_key),
    };
    SeatRelease::sign(&run_key, body).map_err(|e| SeatError::Proto(e.to_string()))
}

/// The peer-side acceptance of a stored lease (the TRAINER's judgment): signature + certificate
/// chain to a genesis-trusted base + expiry, AND the supersession floor (a lease at or below the
/// floor is dead even if it verifies — [SEAT-3], architecture §6.3.1). Returns the authorized
/// lease's endpoint + claimant on success.
///
/// # Errors
/// The applicable [`SeatLeaseError`]; a below-floor lease is [`SeatLeaseError::Expired`]-adjacent
/// (surfaced as a typed refusal by the ledger judgment).
pub fn authorize_incumbent(
    lease: &SeatLease,
    trusted_bases: &[PeerId],
    revocations: &RevocationLedger,
    now_ms: u64,
    skew_ms: u64,
) -> Result<AuthorizedSeat, SeatLeaseError> {
    lease.authorize(trusted_bases, now_ms, skew_ms)?;
    // Supersession floor / explicit revocation: a stale claimant is dead once a higher fencing
    // token (incarnation) exists, regardless of the registry.
    revocations
        .judge(&lease.body.cert_scope(), &lease.body.claimant)
        .map_err(|_| SeatLeaseError::Expired {
            // A superseded/revoked lease is not authoritative; reuse the typed refusal surface
            // (the ledger's own error is folded to the seat vocabulary here).
            expires_at_ms: lease.body.expires_at_ms,
            now_ms,
        })?;
    Ok(AuthorizedSeat {
        endpoint: lease.body.endpoint.clone(),
        claimant: lease.body.claimant,
        certificate: lease.certificate.clone(),
        incarnation: lease.body.incarnation,
    })
}

/// The result of a successful peer-side authorization: what the trainer dials + trusts.
pub struct AuthorizedSeat {
    /// The coordinator's published control-plane endpoint.
    pub endpoint: ControlEndpoint,
    /// The coordinator's certified per-run key (the frame sender the trainer's attach expects).
    pub claimant: PeerId,
    /// The coordinator's certificate (rides the trainer's credentials as a bootstrap peer cert).
    pub certificate: daemon_vhc_proto::RunKeyCertificate,
    /// The seat incarnation (the trainer's supersession floor for the coordinator slot).
    pub incarnation: u64,
}

/// The default skew grace this manager judges expiry under.
#[must_use]
pub fn default_skew_ms() -> u64 {
    DEFAULT_SEAT_SKEW_MS
}

/// Resolve the coordinator per-run signing key read from the keystore (the claim path provisions
/// it first, so it exists).
fn coordinator_key(
    keystore: &VhcKeystore,
    seat: &CoordinatorSeat<'_>,
    incarnation: u64,
) -> Result<SigningKey, SeatError> {
    keystore
        .existing_run_signing_key(seat.run_label, seat.role, incarnation)
        .map_err(|e| SeatError::Identity(e.to_string()))?
        .ok_or_else(|| {
            SeatError::Identity(format!("no coordinator key at incarnation {incarnation}"))
        })
}
