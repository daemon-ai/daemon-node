// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `Committed<T>` — the SDK wrapper consensus transitions require (architecture §4.2; refactor
//! §8/D1).
//!
//! This is the **re-typing of A2's bridged `Staged`** (refactor §5 A2 item 3 / §8 D1): the ordering
//! and verification semantics are byte-identical to the `Staged` bridging oracle — RECORD-LISTED
//! order, per-item blake3 verification, all-or-nothing mint, the same `Missing`/`HashMismatch`
//! refusals — so the existing oracle/parity lanes (the barrier-round pins, the TinyLlama det-digest
//! parity across backends) reproduce the identical digests. The mint **adds one thing and only one
//! thing** over `Staged`: it is constructible only from an [`Authorized`] token
//! (`crate::authority`), so "consensus state is a pure function of an authority-verified record plus
//! hash-verified content, in the record's listed order" is a compile-time property of cooperative
//! module code, not a convention (architecture §4.2). The verified *content* and its *order* are
//! unchanged, which is why the digest a `RoundExperiment::ingest` computes over
//! [`Committed::items`] does not move.
//!
//! The two soundness properties architecture §4.2 separates hold here as stated: the *determinism of
//! the mint* is this wasm/SDK property; *agreement on which records exist* is the [`Authority`]
//! protocol's job and is out of the mint's scope (the mint only decides "authoritative under this
//! topology, content matches the listed hashes, in listed order").

use daemon_vhc_proto::messages::RecordEntry;
use daemon_vhc_proto::{blake3_hash, Hash, PeerId};

use crate::authority::Authorized;

/// How a payload representation proves itself against the record-listed hash at mint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PayloadCheck {
    /// The repr carries the bytes and they hash to the listed value.
    Verified,
    /// The repr carries the bytes and they DO NOT hash to the listed value (tamper — refuse).
    Mismatch,
    /// The repr is a host staging token: the host hash-verified the content before announcing it
    /// (`PayloadReady` is delivered only after blake3 verification, ABI §4.3), so the in-guest
    /// check is delegated — the mint's ordering / all-or-nothing semantics still hold.
    HostVerified,
}

/// A committed payload's representation at the barrier: in-guest bytes (native tests, the session's
/// verified cache) or a host staging token (the bridge's `read_back` kinds — guest payloads never
/// enter linear memory wholesale, architecture §3.4).
pub trait PayloadRepr: Clone + PartialEq {
    /// Check this repr against the record-listed blake3.
    fn check(&self, expected: &Hash) -> PayloadCheck;
}

impl PayloadRepr for Vec<u8> {
    fn check(&self, expected: &Hash) -> PayloadCheck {
        if blake3_hash(self) == *expected {
            PayloadCheck::Verified
        } else {
            PayloadCheck::Mismatch
        }
    }
}

/// A host-staged payload token: the staging id / `upd_*` index `read_back` yielded. The host
/// verified the content hash before announcing it (ABI §4.3), so the mint delegates the check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostStaged(pub u64);

impl PayloadRepr for HostStaged {
    fn check(&self, _expected: &Hash) -> PayloadCheck {
        PayloadCheck::HostVerified
    }
}

/// Where committed payloads come from at the barrier. Sans-io: `None` = not (yet) fetchable
/// (→ the stall ladder), never an await. Guests answer with staged tokens; the session answers from
/// its verified cache; tests answer from a map.
pub trait PayloadSource<P = Vec<u8>> {
    /// The payload committed by `peer` for `round`, if fetchable now.
    fn payload(&mut self, round: u64, peer: &PeerId) -> Option<P>;
}

impl<P: Clone> PayloadSource<P> for std::collections::BTreeMap<(u64, PeerId), P> {
    fn payload(&mut self, round: u64, peer: &PeerId) -> Option<P> {
        self.get(&(round, *peer)).cloned()
    }
}

/// One verified committed payload (the former `StagedItem`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommittedItem<P = Vec<u8>> {
    /// The contributing peer.
    pub peer: PeerId,
    /// The record-listed blake3 (verified against `bytes` at mint).
    pub hash: Hash,
    /// The payload representation (bytes, or a host staging token).
    pub bytes: P,
}

/// Why a [`Committed::mint`] refused. Identical surface to the former `MintError` (the barrier
/// ladder branches on these unchanged).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MintError {
    /// A record-listed payload is not fetchable yet (the caller stalls — never a panic).
    Missing {
        /// The peer whose payload is missing.
        peer: PeerId,
    },
    /// Fetched bytes do not hash to the record-listed value (tamper — refuse, propagate).
    HashMismatch {
        /// The offending peer.
        peer: PeerId,
    },
}

/// The verified, record-ordered committed set consensus transitions ingest — the re-typed `Staged`.
/// Constructible **only** through [`Committed::mint`], which requires an [`Authorized`] token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Committed<P = Vec<u8>> {
    items: Vec<CommittedItem<P>>,
    channel: u32,
}

impl<P: PayloadRepr> Committed<P> {
    /// Mint the committed set from an **authority-verified** record's listed entries (architecture
    /// §4.2). The [`Authorized`] token proves the record carried sufficient authority under the
    /// run's topology (`crate::authority`) — this is the D1 addition over the A2 `Staged` mint; the
    /// verification/ordering below is byte-identical to it:
    ///
    /// - **record-listed order** (I3 — never arrival or map order),
    /// - every item's bytes **blake3-verified** against its listed hash (or delegated for a
    ///   host-verified staging token),
    /// - a missing payload refuses with [`MintError::Missing`] (the stall ladder's input),
    /// - a mismatch refuses with [`MintError::HashMismatch`],
    /// - on any error NOTHING is minted (all-or-nothing, the I2 barrier).
    ///
    /// # Errors
    /// [`MintError`] as above.
    pub fn mint(
        authorized: &Authorized,
        round: u64,
        entries: &[RecordEntry],
        source: &mut impl PayloadSource<P>,
    ) -> Result<Self, MintError> {
        let mut items = Vec::with_capacity(entries.len());
        for entry in entries {
            let Some(bytes) = source.payload(round, &entry.peer) else {
                return Err(MintError::Missing { peer: entry.peer });
            };
            if bytes.check(&entry.hash) == PayloadCheck::Mismatch {
                return Err(MintError::HashMismatch { peer: entry.peer });
            }
            items.push(CommittedItem {
                peer: entry.peer,
                hash: entry.hash,
                bytes,
            });
        }
        Ok(Self {
            items,
            channel: authorized.channel(),
        })
    }

    /// The verified items, in record-listed order.
    #[must_use]
    pub fn items(&self) -> &[CommittedItem<P>] {
        &self.items
    }

    /// The authoritative channel the record that produced this set arrived on (from the
    /// [`Authorized`] token — a channel declaration, architecture §4.2 / ABI §6.2).
    #[must_use]
    pub fn channel(&self) -> u32 {
        self.channel
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::authority::Authorized;

    fn peer(b: u8) -> PeerId {
        PeerId([b; 32])
    }

    fn entry(p: PeerId, bytes: &[u8]) -> RecordEntry {
        RecordEntry {
            peer: p,
            hash: blake3_hash(bytes),
            size: bytes.len() as u64,
        }
    }

    /// The mint preserves record-listed order (not map/pubkey order) and is byte-identical across
    /// mints — the Phase-D re-typing contract this module discharges.
    #[test]
    fn mint_pins_record_listed_order_and_byte_identity() {
        let (a, b, c) = (peer(3), peer(1), peer(2));
        let entries = vec![entry(a, b"pay-a"), entry(b, b"pay-b"), entry(c, b"pay-c")];
        let mut source: std::collections::BTreeMap<(u64, PeerId), Vec<u8>> = Default::default();
        source.insert((7, a), b"pay-a".to_vec());
        source.insert((7, b), b"pay-b".to_vec());
        source.insert((7, c), b"pay-c".to_vec());

        let auth = Authorized::from_authoritative_channel(0);
        let committed = Committed::mint(&auth, 7, &entries, &mut source).expect("mint");
        let order: Vec<PeerId> = committed.items().iter().map(|i| i.peer).collect();
        assert_eq!(order, vec![a, b, c], "record-listed order");
        assert_eq!(
            committed,
            Committed::mint(&auth, 7, &entries, &mut source).expect("mint again")
        );
    }

    #[test]
    fn mint_is_all_or_nothing_missing_then_mismatch() {
        let (a, b) = (peer(3), peer(1));
        let entries = vec![entry(a, b"pay-a"), entry(b, b"pay-b")];
        let mut source: std::collections::BTreeMap<(u64, PeerId), Vec<u8>> = Default::default();
        source.insert((7, a), b"pay-a".to_vec());
        let auth = Authorized::from_authoritative_channel(0);

        assert_eq!(
            Committed::mint(&auth, 7, &entries, &mut source).unwrap_err(),
            MintError::Missing { peer: b }
        );
        source.insert((7, b), b"pay-B".to_vec());
        assert_eq!(
            Committed::mint(&auth, 7, &entries, &mut source).unwrap_err(),
            MintError::HashMismatch { peer: b }
        );
    }

    #[test]
    fn host_staged_repr_delegates_verification_but_keeps_order() {
        let (a, b) = (peer(2), peer(1));
        let entries = vec![entry(a, b"pay-a"), entry(b, b"pay-b")];
        let mut source: std::collections::BTreeMap<(u64, PeerId), HostStaged> = Default::default();
        source.insert((3, a), HostStaged(11));
        source.insert((3, b), HostStaged(12));
        let auth = Authorized::from_authoritative_channel(0);

        let committed = Committed::mint(&auth, 3, &entries, &mut source).expect("mint");
        let order: Vec<(PeerId, u64)> = committed
            .items()
            .iter()
            .map(|i| (i.peer, i.bytes.0))
            .collect();
        assert_eq!(order, vec![(a, 11), (b, 12)]);

        source.remove(&(3, b));
        assert_eq!(
            Committed::mint(&auth, 3, &entries, &mut source).unwrap_err(),
            MintError::Missing { peer: b }
        );
    }

    #[test]
    fn committed_carries_the_authorized_channel() {
        let a = peer(1);
        let entries = vec![entry(a, b"x")];
        let mut source: std::collections::BTreeMap<(u64, PeerId), Vec<u8>> = Default::default();
        source.insert((1, a), b"x".to_vec());
        let committed = Committed::mint(
            &Authorized::from_authoritative_channel(4),
            1,
            &entries,
            &mut source,
        )
        .unwrap();
        assert_eq!(committed.channel(), 4);
    }
}
