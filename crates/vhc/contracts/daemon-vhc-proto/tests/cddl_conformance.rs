// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Swarm control-plane CDDL conformance (TDD PROTO-19, spec §6.4/§7.3).
//!
//! Mirrors the `daemon-api` pattern: real canonical-CBOR bytes emitted by the Rust serde types are
//! validated against the authoritative `daemon-swarm.cddl`, so serde↔CDDL drift becomes a failing
//! test. Fixtures are generated in-process (no committed blobs, no `xtask` edit — the whole crate
//! surface is here). Plus signature-reject and roster-size-invariant checks.

use daemon_swarm_proto::bytes::{Hash, IrohId, PeerId, Seed, StateDigest};
use daemon_swarm_proto::capability::CapabilitySet;
use daemon_swarm_proto::merkle::commit_set;
use daemon_swarm_proto::messages::{
    Attestation, BatchWindow, Commitment, Digest, Heartbeat, Join, Locator, RecordEntry, RoundOpen,
    RoundRecord, SignedMessage, StorageReceipt, Straggle, StraggleStatus, SwarmMessage,
    ThroughputClass,
};
use daemon_swarm_proto::version::SWARM_PROTO_VERSION;
use daemon_swarm_proto::{to_canonical_vec, SigningKey};

const CDDL: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/daemon-swarm.cddl"));

fn key() -> SigningKey {
    SigningKey::from_bytes(&[0x21; 32])
}

fn peer(n: u8) -> PeerId {
    PeerId([n; 32])
}

fn hash(n: u8) -> Hash {
    Hash([n; 32])
}

fn signed(payload: SwarmMessage) -> Vec<u8> {
    let msg = SignedMessage::sign(&key(), SWARM_PROTO_VERSION, payload).unwrap();
    to_canonical_vec(&msg).unwrap()
}

fn validate(root: &str, bytes: &[u8]) {
    cddl_cat::validate_cbor_bytes(root, CDDL, bytes)
        .unwrap_or_else(|e| panic!("`{root}` failed to validate: {e:?}"));
}

fn all_sample_messages() -> Vec<SwarmMessage> {
    let set = commit_set(&[(peer(1), hash(9)), (peer(2), hash(8))]);
    vec![
        SwarmMessage::RoundOpen(RoundOpen {
            round: 42,
            seed: Seed([7; 32]),
            roster_digest: hash(5),
            batch: BatchWindow { start: 0, end: 256 },
            deadline_unix_s: 1_800_000_000,
        }),
        SwarmMessage::Commitment(Commitment {
            round: 42,
            payload: hash(3),
            size: 40 * 1024 * 1024,
            locators: vec![
                Locator::StoreKey("runs/x/r42/peer1.bin".into()),
                Locator::BlobTicket("blobabc".into()),
            ],
        }),
        SwarmMessage::Attestation(Attestation {
            round: 42,
            set: set.commitment(),
            inline: None,
        }),
        SwarmMessage::StorageReceipt(StorageReceipt {
            round: 42,
            verified: vec![RecordEntry {
                peer: peer(1),
                hash: hash(9),
                size: 1234,
            }],
        }),
        SwarmMessage::RoundRecord(RoundRecord {
            round: 42,
            set: set.commitment(),
            drops: vec![peer(7)],
            next_seed: Seed([8; 32]),
            set_locator: Locator::StoreKey("runs/x/r42/record-set.cbor".into()),
            inline: Some(vec![RecordEntry {
                peer: peer(1),
                hash: hash(9),
                size: 1234,
            }]),
        }),
        SwarmMessage::Digest(Digest {
            round: 42,
            digest: StateDigest([0xab; 16]),
        }),
        SwarmMessage::Straggle(Straggle {
            round: 42,
            status: StraggleStatus::Stalled,
        }),
        SwarmMessage::Join(Join {
            run_id: "smollm-500m-01".into(),
            iroh_id: IrohId([4; 32]),
            class: ThroughputClass::C2,
            capabilities: CapabilitySet::from_tokens(["tensor-abi@1", "det_sum@1"]).unwrap(),
            // Exercise the additive optional envelope-hash carrier against the `? "envelope_hash"`
            // CDDL rule (Wave 3).
            envelope_hash: Some(daemon_swarm_proto::blake3_hash(b"smollm-500m-01-envelope")),
        }),
        SwarmMessage::Heartbeat(Heartbeat {
            round: 42,
            ready: None,
        }),
    ]
}

#[test]
fn heartbeat_ready_flag_cddl_conforms_and_roundtrips() {
    use daemon_swarm_proto::from_canonical_slice;

    // A legacy heartbeat omits `ready` on the wire; a Wave-3 ready heartbeat carries the bool.
    let legacy = Heartbeat {
        round: 3,
        ready: None,
    };
    let ready = Heartbeat {
        round: 3,
        ready: Some(true),
    };
    for hb in [legacy, ready] {
        let bytes = signed(SwarmMessage::Heartbeat(hb));
        validate("signed-message", &bytes);
    }
    // The optional field is absent from the canonical bytes when `None` (back-compat), present when set.
    let legacy_bytes = to_canonical_vec(&legacy).unwrap();
    let ready_bytes = to_canonical_vec(&ready).unwrap();
    assert!(ready_bytes.len() > legacy_bytes.len());
    assert_eq!(
        from_canonical_slice::<Heartbeat>(&legacy_bytes).unwrap(),
        legacy
    );
    assert_eq!(
        from_canonical_slice::<Heartbeat>(&ready_bytes).unwrap(),
        ready
    );
    // A legacy heartbeat (no `ready` key) still decodes, defaulting to `None`.
    assert_eq!(
        from_canonical_slice::<Heartbeat>(&legacy_bytes)
            .unwrap()
            .ready,
        None
    );
}

#[test]
fn round_messages_cddl_conformance() {
    let messages = all_sample_messages();
    assert_eq!(messages.len(), 9, "seven round messages + join + heartbeat");
    for payload in messages {
        validate("signed-message", &signed(payload));
    }
}

#[test]
fn invalid_payloads_are_rejected() {
    use ciborium::value::Value;

    fn enc(v: &Value) -> Vec<u8> {
        let mut b = Vec::new();
        ciborium::ser::into_writer(v, &mut b).unwrap();
        b
    }

    // Not a map at all.
    assert!(cddl_cat::validate_cbor_bytes("signed-message", CDDL, &enc(&Value::Null)).is_err());

    // A signed frame whose payload is an unknown variant tag.
    let bad_variant = Value::Map(vec![
        (Value::Text("version".into()), Value::Integer(1.into())),
        (
            Value::Text("payload".into()),
            Value::Map(vec![(Value::Text("Bogus".into()), Value::Null)]),
        ),
        (Value::Text("signer".into()), Value::Bytes(vec![0u8; 32])),
        (Value::Text("sig".into()), Value::Bytes(vec![0u8; 64])),
    ]);
    assert!(cddl_cat::validate_cbor_bytes("signed-message", CDDL, &enc(&bad_variant)).is_err());

    // A heartbeat with a wrong-typed field (`round` must be a uint, not text).
    let bad_field = Value::Map(vec![(
        Value::Text("Heartbeat".into()),
        Value::Map(vec![(Value::Text("round".into()), Value::Text("x".into()))]),
    )]);
    assert!(cddl_cat::validate_cbor_bytes("swarm-message", CDDL, &enc(&bad_field)).is_err());
}

#[test]
fn record_bad_sig_rejected() {
    let set = commit_set(&[(peer(1), hash(9))]);
    let mut msg = SignedMessage::sign(
        &key(),
        SWARM_PROTO_VERSION,
        SwarmMessage::RoundRecord(RoundRecord {
            round: 7,
            set: set.commitment(),
            drops: vec![],
            next_seed: Seed([1; 32]),
            set_locator: Locator::StoreKey("k".into()),
            inline: None,
        }),
    )
    .unwrap();
    assert!(msg.verify().is_ok());
    // The signed bytes still validate structurally against the CDDL …
    validate("signed-message", &to_canonical_vec(&msg).unwrap());
    // … but a corrupted signature must not verify.
    msg.sig.0[0] ^= 0xff;
    assert!(msg.verify().is_err());
}

#[test]
fn wrong_run_version_rejected() {
    use daemon_swarm_proto::version::SwarmProtoVersion;
    let msg = SignedMessage::sign(
        &key(),
        SwarmProtoVersion(2),
        SwarmMessage::Heartbeat(Heartbeat {
            round: 1,
            ready: None,
        }),
    )
    .unwrap();
    // Signature is valid, but the run is pinned to a different version → join gate rejects it.
    assert!(msg.verify().is_ok());
    assert!(msg.verify_for_run(SWARM_PROTO_VERSION).is_err());
}

#[test]
fn message_size_roster_invariant() {
    // Attestation/RoundRecord sign a constant-size set commitment; omitting the inline set keeps
    // the message O(1) in the roster. Going 4 → 512 peers changes only the small `count` varint —
    // it does NOT grow with the roster — whereas an inline set grows linearly.
    fn attestation_size(n: u32, inline: bool) -> usize {
        // Distinct peers/hashes so the tree really holds n leaves.
        let entries: Vec<_> = (0..n)
            .map(|i| {
                let mut p = [0u8; 32];
                p[..4].copy_from_slice(&i.to_be_bytes());
                let mut h = [0u8; 32];
                h[..4].copy_from_slice(&(i * 3 + 1).to_be_bytes());
                (PeerId(p), Hash(h))
            })
            .collect();
        let tree = commit_set(&entries);
        let inline_set = if inline {
            Some(
                tree.entries()
                    .iter()
                    .map(|(p, h)| daemon_swarm_proto::messages::AttestEntry { peer: *p, hash: *h })
                    .collect(),
            )
        } else {
            None
        };
        let payload = SwarmMessage::Attestation(Attestation {
            round: 1,
            set: tree.commitment(),
            inline: inline_set,
        });
        signed(payload).len()
    }

    let s4 = attestation_size(4, false);
    let s512 = attestation_size(512, false);
    // No roster-proportional growth: only the count varint (≤ 2 extra bytes) may differ.
    assert!(
        s512 <= s4 + 2,
        "inline-free attestation must not grow with the roster (s4={s4}, s512={s512})"
    );

    // Contrast: an inline set at 512 peers is an order of magnitude larger — proving the root form
    // is what buys scale-invariance.
    let inline512 = attestation_size(512, true);
    assert!(
        inline512 > s512 * 10,
        "inline set should grow with the roster (inline512={inline512}, s512={s512})"
    );
}
