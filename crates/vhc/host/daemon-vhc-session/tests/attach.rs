// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// The inbound §12.1 seam's pinned negatives (refactor §5 A2; ABI §12.1/§12.2): a bad signature
// is refused and never delivered; a duplicate scope tuple is idempotently dropped; a sequence
// gap is detected, surfaced typed, and the frame HELD (no silent skip); an uncertified sender is
// the signature-downgrade refusal and a revoked/superseded one the CertRevoked refusal. The
// positive path (verified frames delivered to a real pump) is exercised end-to-end by the
// worker's v2 join (daemon-vhc-worker/tests/join.rs).

use ciborium::value::Value;
use daemon_vhc_proto::sign::{peer_id, sign_canonical};
use daemon_vhc_proto::{CertScope, Hash, RunKeyCertificate, RunKeyRevocation, SigningKey};
use daemon_vhc_session::attach::{CertCheck, InboundFrames, InboundVerdict};

const RUN: [u8; 32] = [0xA1; 32];

/// The scope every [`frame`] below signs under: (RUN, epoch 0, "trainer", instance 1,
/// module all-zero).
fn frame_scope() -> CertScope {
    CertScope {
        run_id: Hash(RUN),
        epoch: 0,
        role: "trainer".into(),
        instance: 1,
        module_hash: Hash([0; 32]),
    }
}

/// A CertCheck trusting `base` with certificates for the given signer keys at [`frame_scope`].
fn check_for(base: &SigningKey, signers: &[&SigningKey]) -> CertCheck {
    let certs = signers
        .iter()
        .map(|k| RunKeyCertificate::issue(base, frame_scope(), peer_id(k)).expect("issue cert"))
        .collect();
    CertCheck::new(vec![peer_id(base)], certs)
}

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
    let base = SigningKey::from_bytes(&[7; 32]);
    let mut v = InboundFrames::new(RUN, 0, check_for(&base, &[&key]));
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
    let base = SigningKey::from_bytes(&[7; 32]);
    let mut v = InboundFrames::new(RUN, 0, check_for(&base, &[&key]));
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
    let base = SigningKey::from_bytes(&[7; 32]);
    let mut ver = InboundFrames::new(RUN, 0, check_for(&base, &[&key]));
    assert_eq!(ver.accept(&spliced), InboundVerdict::TamperedPayload);
}

#[test]
fn a_duplicate_scope_tuple_is_idempotently_dropped() {
    let key = SigningKey::from_bytes(&[9; 32]);
    let base = SigningKey::from_bytes(&[7; 32]);
    let mut v = InboundFrames::new(RUN, 0, check_for(&base, &[&key]));
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
    let base = SigningKey::from_bytes(&[7; 32]);
    let mut v = InboundFrames::new(RUN, 0, check_for(&base, &[&key]));
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
    let base = SigningKey::from_bytes(&[7; 32]);
    let mut v = InboundFrames::new([0xB2; 32], 0, CertCheck::new(vec![peer_id(&base)], vec![]));
    assert!(matches!(
        v.accept(&frame(&key, 0, 0, b"x")),
        InboundVerdict::ScopeMismatch(_)
    ));
}

// -- certified per-run keys, layered around the retained frame verifier (architecture §4.3) -------

#[test]
fn certified_sender_delivers_and_an_uncertified_one_is_downgrade_refused() {
    // The frame builder signs with `key` under the fixture scope — the base certifies `key`'s
    // per-run key for exactly that binding.
    let key = SigningKey::from_bytes(&[9; 32]);
    let base = SigningKey::from_bytes(&[7; 32]); // the trusted base machine identity
    let mut v = InboundFrames::new(RUN, 0, check_for(&base, &[&key]));

    // A certified per-run key is accepted — the cell that should accept it does.
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
fn a_key_reconstructed_from_public_run_data_cannot_authenticate() {
    // The retired derivation shape: a signing key expanded from a blake3 of a public run label.
    // Anyone can rebuild it from public data — the attach must refuse it as uncertified, because
    // no base identity ever certified it (no production path issues certificates over derived
    // keys; per-run keys are CSPRNG-generated).
    let derived_seed = *blake3::hash(b"vhc-worker/genesis-t2").as_bytes();
    let derived = SigningKey::from_bytes(&derived_seed);
    let base = SigningKey::from_bytes(&[7; 32]);
    let honest = SigningKey::from_bytes(&[9; 32]);
    let mut v = InboundFrames::new(RUN, 0, check_for(&base, &[&honest]));
    match v.accept(&frame(&derived, 0, 0, b"hello")) {
        InboundVerdict::UncertifiedSender { sender, .. } => {
            assert_eq!(sender, peer_id(&derived).0);
        }
        other => panic!("expected UncertifiedSender, got {other:?}"),
    }
}

#[test]
fn a_revoked_key_is_refused_typed_even_with_a_valid_certificate() {
    let key = SigningKey::from_bytes(&[9; 32]);
    let base = SigningKey::from_bytes(&[7; 32]);
    let mut v = InboundFrames::new(RUN, 0, check_for(&base, &[&key]));
    // Live before the record...
    assert!(matches!(
        v.accept(&frame(&key, 0, 0, b"hello")),
        InboundVerdict::Deliver { .. }
    ));
    // ...the base signs the revocation, the attach ingests it...
    let record = RunKeyRevocation::issue(&base, Hash(RUN), "trainer", 1, peer_id(&key), 1)
        .expect("issue revocation");
    v.certs_mut()
        .expect("production attach carries certs")
        .ingest_revocation(&record)
        .expect("trusted, in-sequence record ingests");
    // ...and the key is dead from here on: CertRevoked, never a delivery.
    assert_eq!(
        v.accept(&frame(&key, 0, 1, b"hello")),
        InboundVerdict::CertRevoked {
            sender: peer_id(&key).0
        }
    );
}

#[test]
fn a_superseding_incarnations_certificate_fences_the_old_one() {
    // Supersession is the safety floor: ingesting the certificate of a HIGHER incarnation for the
    // same (run, role) slot fences the old incarnation — no explicit revocation record needed.
    let old_key = SigningKey::from_bytes(&[9; 32]);
    let base = SigningKey::from_bytes(&[7; 32]);
    let mut v = InboundFrames::new(RUN, 0, check_for(&base, &[&old_key]));
    assert!(matches!(
        v.accept(&frame(&old_key, 0, 0, b"hello")),
        InboundVerdict::Deliver { .. }
    ));
    let new_key = SigningKey::from_bytes(&[10; 32]);
    let successor = RunKeyCertificate::issue(
        &base,
        CertScope {
            instance: 2,
            ..frame_scope()
        },
        peer_id(&new_key),
    )
    .expect("issue successor cert");
    v.certs_mut()
        .expect("production attach carries certs")
        .ingest_certificate(successor);
    assert_eq!(
        v.accept(&frame(&old_key, 0, 1, b"hello")),
        InboundVerdict::CertRevoked {
            sender: peer_id(&old_key).0
        }
    );
}

/// §12.3 on-plane distribution: a certificate record from a genesis-trusted base ingests and its
/// sender then authenticates; a record chained to an UNTRUSTED base is refused and advances no
/// trust state (its sender stays uncertified — a forged record can never fence anyone).
#[test]
fn distribution_certs_ingest_only_from_trusted_bases() {
    let base = SigningKey::from_bytes(&[7; 32]);
    let rogue_base = SigningKey::from_bytes(&[8; 32]);
    let key = SigningKey::from_bytes(&[9; 32]);
    // Trusts `base`, holds NO certificates yet.
    let mut v = InboundFrames::new(RUN, 0, CertCheck::new(vec![peer_id(&base)], vec![]));

    // Before any distribution: the sender is uncertified.
    assert!(matches!(
        v.accept(&frame(&key, 0, 0, b"early")),
        InboundVerdict::UncertifiedSender { .. }
    ));

    // A record chained to a rogue base refuses typed and changes nothing.
    let rogue_cert =
        RunKeyCertificate::issue(&rogue_base, frame_scope(), peer_id(&key)).expect("issue");
    let err = v
        .ingest_distribution(daemon_vhc_session::distribution::DistributionRecord::Cert(
            rogue_cert,
        ))
        .unwrap_err();
    assert!(err.contains("not genesis-trusted"), "got: {err}");
    assert!(matches!(
        v.accept(&frame(&key, 0, 0, b"still-early")),
        InboundVerdict::UncertifiedSender { .. }
    ));

    // A record with a TAMPERED chain (bad signature) refuses even when it names a trusted base.
    let mut forged = RunKeyCertificate::issue(&base, frame_scope(), peer_id(&key)).expect("issue");
    forged.body.scope.instance = 99; // body no longer matches the signature
    let err = v
        .ingest_distribution(daemon_vhc_session::distribution::DistributionRecord::Cert(
            forged,
        ))
        .unwrap_err();
    assert!(err.contains("certificate chain"), "got: {err}");

    // The honest record ingests; the sender now delivers. Re-delivery is an idempotent no-op.
    let cert = RunKeyCertificate::issue(&base, frame_scope(), peer_id(&key)).expect("issue");
    v.ingest_distribution(daemon_vhc_session::distribution::DistributionRecord::Cert(
        cert.clone(),
    ))
    .expect("trusted record ingests");
    v.ingest_distribution(daemon_vhc_session::distribution::DistributionRecord::Cert(
        cert,
    ))
    .expect("re-delivery is idempotent");
    assert!(matches!(
        v.accept(&frame(&key, 0, 0, b"hello")),
        InboundVerdict::Deliver { .. }
    ));
}

/// §12.3 on-plane distribution: a revocation record rides the same surface — after it, the
/// revoked sender is the typed CertRevoked refusal.
#[test]
fn distribution_revocations_ride_the_same_surface() {
    let base = SigningKey::from_bytes(&[7; 32]);
    let key = SigningKey::from_bytes(&[9; 32]);
    let mut v = InboundFrames::new(RUN, 0, check_for(&base, &[&key]));
    assert!(matches!(
        v.accept(&frame(&key, 0, 0, b"pre-revocation")),
        InboundVerdict::Deliver { .. }
    ));

    let record = RunKeyRevocation::issue(&base, Hash(RUN), "trainer", 1, peer_id(&key), 1)
        .expect("issue revocation");
    v.ingest_distribution(daemon_vhc_session::distribution::DistributionRecord::Revocation(record))
        .expect("trusted revocation ingests");
    assert_eq!(
        v.accept(&frame(&key, 0, 1, b"post-revocation")),
        InboundVerdict::CertRevoked {
            sender: peer_id(&key).0
        }
    );
}
