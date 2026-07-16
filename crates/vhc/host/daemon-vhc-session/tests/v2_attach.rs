// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// The inbound §12.1 seam's pinned negatives (refactor §5 A2; ABI §12.1/§12.2): a bad signature
// is refused and never delivered; a duplicate scope tuple is idempotently dropped; a sequence
// gap is detected, surfaced typed, and the frame HELD (no silent skip). The positive path
// (verified frames delivered to a real pump) is exercised end-to-end by the worker's v2 join
// (daemon-vhc-worker/tests/v2_join.rs).

use ciborium::value::Value;
use daemon_vhc_proto::sign::{peer_id, sign_canonical};
use daemon_vhc_proto::{Hash, RunKeyCertificate, SigningKey};
use daemon_vhc_session::v2_attach::{InboundFrames, InboundVerdict};

const RUN: [u8; 32] = [0xA1; 32];

fn frame(key: &SigningKey, channel: u64, seq: u64, payload: &[u8]) -> Vec<u8> {
    let sender = peer_id(key).0;
    let envelope = Value::Map(vec![
        (Value::from("domain"), Value::from("daemon-vhc/frame/2")),
        (Value::from("run_id"), Value::Bytes(RUN.to_vec())),
        (Value::from("epoch"), Value::from(0u64)),
        (Value::from("role"), Value::from("trainer")),
        (Value::from("instance"), Value::from(1u64)),
        (Value::from("module"), Value::Bytes(vec![0; 32])),
        (Value::from("sender"), Value::Bytes(sender.to_vec())),
        (Value::from("channel"), Value::from(channel)),
        (Value::from("seq"), Value::from(seq)),
        (
            Value::from("payload_hash"),
            Value::Bytes(blake3::hash(payload).as_bytes().to_vec()),
        ),
    ]);
    let sig = sign_canonical(key, &envelope).expect("sign");
    let wire = Value::Array(vec![
        envelope,
        Value::Bytes(payload.to_vec()),
        Value::Bytes(sig.0.to_vec()),
    ]);
    let mut out = Vec::new();
    ciborium::into_writer(&wire, &mut out).expect("frame cbor");
    out
}

#[test]
fn verified_in_sequence_frames_deliver_and_advance_the_cursor() {
    let key = SigningKey::from_bytes(&[9; 32]);
    let mut v = InboundFrames::new(RUN, 0);
    for seq in 0..3u64 {
        match v.accept(&frame(&key, 0, seq, b"hello")) {
            InboundVerdict::Deliver {
                seq: s, payload, ..
            } => {
                assert_eq!(s, seq);
                assert_eq!(payload, b"hello");
            }
            other => panic!("seq {seq}: expected Deliver, got {other:?}"),
        }
    }
}

#[test]
fn a_bad_signature_is_refused_and_never_delivered() {
    let key = SigningKey::from_bytes(&[9; 32]);
    let mut wire = frame(&key, 0, 0, b"hello");
    let n = wire.len();
    wire[n - 10] ^= 1; // flip a signature byte
    let mut v = InboundFrames::new(RUN, 0);
    assert!(matches!(v.accept(&wire), InboundVerdict::BadSignature(_)));
    // The cursor did NOT advance: the genuine seq-0 frame still delivers.
    assert!(matches!(
        v.accept(&frame(&key, 0, 0, b"hello")),
        InboundVerdict::Deliver { .. }
    ));
}

#[test]
fn a_tampered_payload_is_refused_even_with_a_valid_envelope_signature() {
    let key = SigningKey::from_bytes(&[9; 32]);
    let wire = frame(&key, 0, 0, b"hello");
    // Re-splice the frame with a different payload under the SAME signed envelope.
    let v: Value = ciborium::de::from_reader(wire.as_slice()).expect("cbor");
    let Value::Array(mut parts) = v else { panic!() };
    parts[1] = Value::Bytes(b"evil!".to_vec());
    let mut spliced = Vec::new();
    ciborium::into_writer(&Value::Array(parts), &mut spliced).expect("cbor");
    let mut ver = InboundFrames::new(RUN, 0);
    assert_eq!(ver.accept(&spliced), InboundVerdict::TamperedPayload);
}

#[test]
fn a_duplicate_scope_tuple_is_idempotently_dropped() {
    let key = SigningKey::from_bytes(&[9; 32]);
    let mut v = InboundFrames::new(RUN, 0);
    let wire = frame(&key, 0, 0, b"hello");
    assert!(matches!(v.accept(&wire), InboundVerdict::Deliver { .. }));
    assert!(matches!(
        v.accept(&wire),
        InboundVerdict::Duplicate { seq: 0, .. }
    ));
}

#[test]
fn a_sequence_gap_is_surfaced_and_the_frame_held() {
    let key = SigningKey::from_bytes(&[9; 32]);
    let mut v = InboundFrames::new(RUN, 0);
    assert!(matches!(
        v.accept(&frame(&key, 0, 0, b"a")),
        InboundVerdict::Deliver { .. }
    ));
    // seq jumps 1 -> 3: gap surfaced, frame held, cursor unmoved (§12.2 — no silent skip).
    assert_eq!(
        v.accept(&frame(&key, 0, 3, b"c")),
        InboundVerdict::Gap {
            sender: peer_id(&key).0,
            channel: 0,
            expected: 1,
            got: 3
        }
    );
    // The in-sequence frame still delivers afterwards (the gap held nothing back silently).
    assert!(matches!(
        v.accept(&frame(&key, 0, 1, b"b")),
        InboundVerdict::Deliver { seq: 1, .. }
    ));
}

#[test]
fn a_frame_from_another_run_scope_is_refused() {
    let key = SigningKey::from_bytes(&[9; 32]);
    let mut v = InboundFrames::new([0xB2; 32], 0);
    assert!(matches!(
        v.accept(&frame(&key, 0, 0, b"x")),
        InboundVerdict::ScopeMismatch(_)
    ));
}

// -- D1 certified per-run keys, layered around the retained A2 verifier (architecture §4.3) --------

#[test]
fn certified_sender_delivers_and_an_uncertified_one_is_downgrade_refused() {
    // The frame builder signs with `key` under scope (RUN, epoch 0, role "trainer", instance 1).
    let key = SigningKey::from_bytes(&[9; 32]);
    let base = SigningKey::from_bytes(&[7; 32]); // the trusted base machine identity
                                                 // The base certifies `key`'s per-run key for exactly this scope, epochs 0..=3.
    let cert = RunKeyCertificate::issue(&base, Hash(RUN), "trainer", 1, 0, 3, peer_id(&key))
        .expect("issue cert");
    let mut v = InboundFrames::with_certs(RUN, 0, peer_id(&base), vec![cert]);

    // A v2 signer (certified per-run key) is accepted — the cell that should accept it does.
    assert!(matches!(
        v.accept(&frame(&key, 0, 0, b"hello")),
        InboundVerdict::Deliver { .. }
    ));

    // An uncertified sender (a different per-run key, no cert) — signature verifies, but the key is
    // not certified: the signature-downgrade refusal, never a delivery.
    let uncertified = SigningKey::from_bytes(&[42; 32]);
    match v.accept(&frame(&uncertified, 0, 0, b"hello")) {
        InboundVerdict::UncertifiedSender { sender, .. } => {
            assert_eq!(sender, peer_id(&uncertified).0);
        }
        other => panic!("expected UncertifiedSender, got {other:?}"),
    }
}

#[test]
fn the_transition_path_without_certs_delivers_an_uncertified_sender() {
    // `InboundFrames::new` is the retained A2 verifier: it checks the frame signature over `sender`
    // but does NOT require certification. An uncertified sender still delivers — the dual-support
    // transition (old verifier retained through D1).
    let uncertified = SigningKey::from_bytes(&[42; 32]);
    let mut v = InboundFrames::new(RUN, 0);
    assert!(matches!(
        v.accept(&frame(&uncertified, 0, 0, b"hello")),
        InboundVerdict::Deliver { .. }
    ));
}
