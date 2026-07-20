// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The node's **iroh roster** duties (architecture §6.3; the transport analogue of the seat
//! manager): author this node's signed reachability record, and judge everyone else's.
//!
//! The registry is UNTRUSTED STORAGE: it stores signed roster records under a monotonic
//! freshness upsert but never judges authority. This module owns the two node-side halves:
//!
//! - **record authorship**: derive the iroh `EndpointId` from the keystore's transport secret
//!   (the iroh key is ed25519, so its public half IS the endpoint id — no iroh dependency
//!   node-side), resolve the provisioned per-run key + certificate, and sign the record;
//! - **peer-side acceptance** ([`verified_iroh_roster`]): verify every fetched record
//!   (signature, certificate chain to a genesis-trusted base, exact scope), reduce to the
//!   freshest record per `(role, base identity)` node, drop this node's own entry, and map the
//!   survivors onto the credentials' [`IrohRosterPeer`] vocabulary.
//!
//! Staleness is precedence, never wall clock: a rejoined node's higher incarnation supersedes
//! its past records; a re-addressed node's later `issued_at_ms` supersedes within the
//! incarnation (`daemon_vhc_proto::roster`).

use daemon_vhc_proto::bytes::IrohId;
use daemon_vhc_proto::domains::ROSTER_RECORD_DOMAIN;
use daemon_vhc_proto::{freshest_per_node, peer_id, Hash, PeerId, RosterRecord, RosterRecordBody};
use daemon_vhc_session::keystore::VhcKeystore;
use daemon_vhc_session::protocol::IrohRosterPeer;
use daemon_vhc_session::provisioning::{provision_run_identity, ProvisionScope};

use crate::credentials::RunInstanceIdentity;

/// A roster-authorship error.
#[derive(Debug, thiserror::Error)]
pub enum RosterError {
    /// Keystore / identity resolution failed.
    #[error("roster identity: {0}")]
    Identity(String),
    /// Record authorship failed structurally or on signing.
    #[error("roster record: {0}")]
    Record(String),
}

/// The iroh `EndpointId` this node's keystore transport secret derives to (the iroh secret is an
/// ed25519 key; its public half is the endpoint id — computed proto-side, no iroh linkage).
///
/// # Errors
/// A keystore failure.
pub fn local_endpoint_id(keystore: &VhcKeystore) -> Result<IrohId, RosterError> {
    let secret = keystore
        .iroh_secret()
        .map_err(|e| RosterError::Identity(format!("iroh transport secret: {e}")))?;
    Ok(IrohId(peer_id(&secret.signing_key()).0))
}

/// Author this node's roster record for `identity`: provision (or recover — idempotent within an
/// incarnation, the [`provision_run_identity`] contract) the per-run key + its certificate, and
/// sign a record binding the keystore-derived endpoint id to `direct_addrs` + `relay_url` at
/// `(incarnation, now_ms)` freshness — the seat manager's authoring shape.
///
/// # Errors
/// Keystore resolution, provisioning, or record authorship failure.
pub fn author_roster_record(
    keystore: &VhcKeystore,
    identity: &RunInstanceIdentity<'_>,
    direct_addrs: Vec<String>,
    relay_url: Option<String>,
    now_ms: u64,
) -> Result<RosterRecord, RosterError> {
    let endpoint_id = local_endpoint_id(keystore)?;
    let certificate = provision_run_identity(
        keystore,
        &ProvisionScope {
            run_label: identity.run_label,
            genesis_hash: identity.genesis_hash,
            epoch: identity.epoch,
            role: identity.role,
            incarnation: identity.incarnation,
            module_hash: identity.module_hash,
        },
    )
    .map_err(|e| RosterError::Identity(format!("provision run identity: {e}")))?;
    let run_key = keystore
        .run_signing_key(identity.run_label, identity.role, identity.incarnation)
        .map_err(|e| RosterError::Identity(format!("run signing key: {e}")))?;
    let body = RosterRecordBody {
        domain: ROSTER_RECORD_DOMAIN.to_string(),
        run_id: Hash(identity.genesis_hash),
        role: identity.role.to_string(),
        epoch: identity.epoch,
        incarnation: identity.incarnation,
        sender: peer_id(&run_key),
        module_hash: Hash(identity.module_hash),
        endpoint_id,
        direct_addrs,
        relay_url,
        issued_at_ms: now_ms,
    };
    RosterRecord::publish(&run_key, certificate, body)
        .map_err(|e| RosterError::Record(e.to_string()))
}

/// The peer-side roster judgment: authorize every fetched record (signature + certificate chain
/// to a genesis-trusted base + exact scope — `RosterRecord::authorize`), reduce to the freshest
/// record per `(role, base identity)` node, drop this node's own entry (never dial yourself),
/// and map onto the credentials' [`IrohRosterPeer`] form. A record that fails verification is
/// silently excluded from the roster — an unverifiable address is simply not reachability, and
/// gossip needs only SOME verified peers to form the mesh (the registry can withhold entries
/// anyway; a forged one must never be dialed).
#[must_use]
pub fn verified_iroh_roster(
    records: Vec<RosterRecord>,
    trusted_bases: &[PeerId],
    own_endpoint: IrohId,
) -> Vec<IrohRosterPeer> {
    let verified: Vec<RosterRecord> = records
        .into_iter()
        .filter(|record| record.authorize(trusted_bases).is_ok())
        .collect();
    freshest_per_node(verified)
        .into_iter()
        .filter(|record| record.body.endpoint_id != own_endpoint)
        .map(|record| IrohRosterPeer {
            endpoint_id: record.body.endpoint_id.0,
            direct_addrs: record.body.direct_addrs,
            relay_url: record.body.relay_url,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use daemon_vhc_proto::cert::RunKeyCertificate;
    use daemon_vhc_proto::SigningKey;

    fn identity<'a>(
        run_label: &'a str,
        role: &'a str,
        incarnation: u64,
    ) -> RunInstanceIdentity<'a> {
        RunInstanceIdentity {
            run_label,
            genesis_hash: [0x11; 32],
            epoch: 0,
            role,
            incarnation,
            module_hash: [0xCC; 32],
        }
    }

    fn keystore() -> (tempfile::TempDir, VhcKeystore) {
        let dir = tempfile::tempdir().expect("keystore dir");
        let store = VhcKeystore::open(dir.path()).expect("open keystore");
        (dir, store)
    }

    /// The authored record verifies end-to-end against the node's own base identity, binds the
    /// keystore-derived endpoint id, and carries the provisioned certificate.
    #[test]
    fn authored_record_authorizes_against_the_own_base() {
        let (_dir, store) = keystore();
        let id = identity("run-a", "trainer", 1);
        let record = author_roster_record(&store, &id, vec!["127.0.0.1:4550".into()], None, 1_000)
            .expect("author");
        assert_eq!(record.body.endpoint_id, local_endpoint_id(&store).unwrap());
        let base = peer_id(&store.base_identity().unwrap());
        record.authorize(&[base]).expect("authorizes");
    }

    /// Authoring provisions on demand (idempotent within an incarnation — the crash-safe
    /// resume): re-authoring recovers the SAME per-run key, and a republish at a later stamp is
    /// exactly the re-address form the fold supersedes on.
    #[test]
    fn re_authoring_recovers_the_same_identity() {
        let (_dir, store) = keystore();
        let id = identity("run-a", "trainer", 1);
        let first =
            author_roster_record(&store, &id, vec!["127.0.0.1:1".into()], None, 1_000).unwrap();
        let again =
            author_roster_record(&store, &id, vec!["127.0.0.1:2".into()], None, 2_000).unwrap();
        assert_eq!(first.body.sender, again.body.sender, "same recovered key");
        assert_eq!(first.body.endpoint_id, again.body.endpoint_id);
        assert!(again.body.freshness() > first.body.freshness());
    }

    /// Verification excludes untrusted records, applies incarnation-then-issue precedence per
    /// node, and never returns this node's own entry.
    #[test]
    fn verified_roster_reduces_excludes_and_never_self_dials() {
        let (_dir, store_a) = keystore();
        let (_dir_b, store_b) = keystore();

        // Node A publishes at incarnation 1 then rejoins at 2; node B publishes once.
        let a1 = author_roster_record(
            &store_a,
            &identity("run-a", "trainer", 1),
            vec!["127.0.0.1:4001".into()],
            None,
            5_000,
        )
        .unwrap();
        let a2 = author_roster_record(
            &store_a,
            &identity("run-a", "trainer", 2),
            vec!["127.0.0.1:4002".into()],
            None,
            1_000,
        )
        .unwrap();
        let b1 = author_roster_record(
            &store_b,
            &identity("run-a", "trainer", 1),
            vec!["127.0.0.1:4003".into()],
            None,
            2_000,
        )
        .unwrap();

        // A rogue record chained to a base no genesis names.
        let rogue_key = SigningKey::from_bytes(&[7; 32]);
        let rogue_base = SigningKey::from_bytes(&[8; 32]);
        let rogue_body = RosterRecordBody {
            domain: ROSTER_RECORD_DOMAIN.to_string(),
            run_id: Hash([0x11; 32]),
            role: "trainer".into(),
            epoch: 0,
            incarnation: 9,
            sender: peer_id(&rogue_key),
            module_hash: Hash([0xCC; 32]),
            endpoint_id: IrohId([0x99; 32]),
            direct_addrs: vec!["10.0.0.9:9".into()],
            relay_url: None,
            issued_at_ms: 9_000,
        };
        let rogue_cert =
            RunKeyCertificate::issue(&rogue_base, rogue_body.cert_scope(), peer_id(&rogue_key))
                .unwrap();
        let rogue = RosterRecord::publish(&rogue_key, rogue_cert, rogue_body).unwrap();

        let trusted = [
            peer_id(&store_a.base_identity().unwrap()),
            peer_id(&store_b.base_identity().unwrap()),
        ];

        // As node B sees it: A's incarnation-2 record wins (never the older-incarnation one,
        // despite its later wall clock), the rogue is excluded, and B's own entry is dropped.
        let own = local_endpoint_id(&store_b).unwrap();
        let peers = verified_iroh_roster(vec![a1, a2, b1, rogue], &trusted, own);
        assert_eq!(peers.len(), 1, "A's freshest record only: {peers:?}");
        assert_eq!(peers[0].direct_addrs, vec!["127.0.0.1:4002".to_string()]);
    }
}
