// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// **The D3 cell-5 three-proviso envelope fixture** (ABI §9.3; decisions D3, ratified
// conditionally): a v2 worker module under envelope v1 + the additive `device_min` section is
// "Pending fixture: interim-supported iff all three provisos pass, otherwise refused until D0".
// This is that standing fixture. The provisos, verbatim:
//
//   1. the old (pre-A2) `FrozenEnvelope::open` accepts and signature-verifies the new RAW bytes
//      carrying `device_min`;
//   2. the original bytes and their blake3 hash are preserved END-TO-END through every path that
//      stores or forwards the envelope (the hash is always over received bytes, never re-derived
//      from a re-encode);
//   3. no code path decodes into the old typed `Envelope` and RE-FREEZES it — the typed struct
//      silently discards unknown fields, so a decode→re-freeze round-trip would strip
//      `device_min` and change the hash (the failure mode provisos 1–2 alone would miss).
//
// Proviso 3 is a statement about every path; this fixture pins it two ways: the strip canary
// (proving the trap is REAL, so the end-to-end byte-identity assertions are load-bearing) and
// identity through the wire carrier (`SignedEnvelope`, the worker protocol's `AssessRun` seam).
// The admission-side consumption (`admit_v2` stage 3) and the worker thread-through are pinned
// in `daemon-vhc-host/tests/v2_claim_funnel.rs`. What this fixture does NOT cover — recorded
// honestly — is a live t2 whole-run (a native coordinator driving a v2 round through the worker
// binary): that needs the v2 join/session wiring, which is not built yet.

use std::collections::BTreeMap;

use ciborium::value::Value;
use daemon_vhc_proto::envelope::{
    Access, Artifact, DataSection, DeviceMinimums, ExperimentSection, GlobalBatch, Phases,
    Requirements, RoundMode, RunSection, StopCondition,
};
use daemon_vhc_proto::{
    blake3_hash, from_canonical_slice, to_canonical_vec, Envelope, FrozenEnvelope, Hash,
    SignedEnvelope, SigningKey,
};

fn text(s: &str) -> Value {
    Value::Text(s.into())
}

fn author() -> SigningKey {
    SigningKey::from_bytes(&[0x42; 32])
}

/// A minimal valid v1 envelope (the pre-A2 schema — no `device_min` field exists in the type).
fn v1_envelope() -> Envelope {
    let mut artifacts = BTreeMap::new();
    artifacts.insert(
        "experiment.wasm".to_string(),
        Artifact {
            url: "file:///dev/null".into(),
            blake3: Hash([1; 32]),
        },
    );
    artifacts.insert(
        "data.manifest".to_string(),
        Artifact {
            url: "file:///dev/null".into(),
            blake3: Hash([2; 32]),
        },
    );
    Envelope {
        run: RunSection {
            schema: 1,
            run_id: "cell5-fixture".into(),
            min_peers: 1,
            max_peers: 8,
            access: Access::Org,
        },
        experiment: ExperimentSection {
            module: "experiment.wasm".into(),
            abi: "tensor-abi@1".into(),
            config: Value::Map(vec![(text("h"), Value::Integer(2.into()))]),
        },
        artifacts,
        data: DataSection {
            manifest: "data.manifest".into(),
            steps_per_round: 2,
            global_batch: GlobalBatch {
                start: 4,
                end: 4,
                ramp_rounds: 1,
            },
            stop: StopCondition::Tokens(1),
        },
        requirements: Requirements {
            vram_mb_min: 0,
            ram_gb_min: 1,
            uplink_mbps_min: 1,
            downlink_mbps_min: 1,
            disk_gb_min: 1,
            throughput_floor: "c1".into(),
            update_mb_max: 1,
            capabilities: vec!["tensor-abi@1".into()],
            payload_store: "r2".into(),
        },
        phases: Phases {
            round_mode: RoundMode::Barrier,
            warmup: 1,
            round_train_max: 60,
            round_witness: 1,
            cooldown: 1,
            epoch_rounds: 1,
            checkpoint_every_epochs: 1,
            stall_rounds_max: 2,
            payload_retention_rounds: 4,
        },
    }
}

/// Inject the additive `device_min` section at the RAW-CBOR level (an author-side operation the
/// typed v1 `Envelope` cannot express — exactly how a v2-aware author extends a v1 envelope),
/// then re-sign over the new bytes.
fn modified_envelope_bytes() -> (Vec<u8>, SignedEnvelope) {
    let frozen = v1_envelope().freeze(&author()).expect("freeze");
    let v: Value = ciborium::de::from_reader(frozen.bytes()).expect("decode value");
    let Value::Map(mut entries) = v else {
        panic!("envelope is a map")
    };
    entries.push((
        text("device_min"),
        Value::Map(vec![
            (text("gpu"), Value::Integer(1.into())),
            (text("ram_bytes"), Value::Integer((1u64 << 30).into())),
            (text("disk_bytes"), Value::Integer((1u64 << 30).into())),
        ]),
    ));
    let bytes = to_canonical_vec(&Value::Map(entries)).expect("re-encode");
    // Sign over the NEW bytes' hash (the author's act; the signature scheme is unchanged).
    let hash = blake3_hash(&bytes);
    let signature = daemon_vhc_proto::sign::sign_canonical(&author(), &hash).expect("sign");
    let signer = daemon_vhc_proto::sign::peer_id(&author());
    let wire = SignedEnvelope {
        bytes: bytes.clone(),
        signature,
        signer,
    };
    (bytes, wire)
}

/// Proviso 1: the pre-A2 decoder accepts + signature-verifies the raw bytes carrying `device_min`.
#[test]
fn proviso_1_old_reader_opens_and_verifies_device_min_bytes() {
    let (bytes, wire) = modified_envelope_bytes();
    let frozen = wire
        .open()
        .expect("FrozenEnvelope::open accepts the additive section");
    frozen
        .verify()
        .expect("signature verifies over the new bytes");
    // The typed view still decodes (unknown keys ignored) and validates.
    let env = frozen.decode().expect("typed decode");
    assert_eq!(env.run.run_id, "cell5-fixture");
    // The host-readable section parses from the RAW bytes.
    assert_eq!(
        frozen.device_min(),
        Some(DeviceMinimums {
            gpu: Some(1),
            ram_bytes: Some(1 << 30),
            disk_bytes: Some(1 << 30),
            ..DeviceMinimums::default()
        })
    );
    assert_eq!(frozen.bytes(), &bytes[..]);
}

/// Proviso 2: bytes + blake3 are preserved end-to-end through the wire carrier the worker
/// protocol forwards (`SignedEnvelope`, the `AssessRun` seam) — hash over RECEIVED bytes only.
#[test]
fn proviso_2_bytes_and_hash_survive_the_wire_carrier_end_to_end() {
    let (bytes, wire) = modified_envelope_bytes();
    let original_hash = blake3_hash(&bytes);
    // The transport round-trip: exactly what AssessRun{envelope} carries.
    let carried = to_canonical_vec(&wire).expect("wire encode");
    let received: SignedEnvelope = from_canonical_slice(&carried).expect("wire decode");
    assert_eq!(received.bytes, bytes, "byte-identity through the carrier");
    let frozen = received.open().expect("open");
    assert_eq!(
        frozen.hash(),
        &original_hash,
        "hash re-derived from the received bytes"
    );
    // And config extraction (the worker's next step) reads a subslice of those bytes.
    let frozen2 = FrozenEnvelope::open(
        frozen.bytes().to_vec(),
        *frozen.signature(),
        *frozen.signer(),
    )
    .expect("second open (store→reload path)");
    assert_eq!(frozen2.hash(), &original_hash);
    assert_eq!(frozen2.config_bytes(), frozen.config_bytes());
}

/// Proviso 3's canary: a decode→re-freeze round-trip STRIPS `device_min` and changes the hash —
/// the silent-field-discard trap is real, which is exactly why the identity assertions above are
/// load-bearing and why `FrozenEnvelope` exposes no re-freeze (freeze is an authoring API on the
/// typed `Envelope` only; `open` stores the received bytes verbatim).
#[test]
fn proviso_3_canary_decode_refreeze_strips_the_section_and_changes_the_hash() {
    let (bytes, wire) = modified_envelope_bytes();
    let frozen = wire.open().expect("open");
    let refrozen = frozen
        .decode()
        .expect("typed decode")
        .freeze(&author())
        .expect("re-freeze (authoring API)");
    assert_ne!(
        refrozen.bytes(),
        &bytes[..],
        "the typed round-trip must NOT be able to reproduce the bytes"
    );
    assert_ne!(refrozen.hash(), &blake3_hash(&bytes));
    assert_eq!(
        refrozen.device_min(),
        None,
        "the section is silently stripped — the trap the provisos guard"
    );
}

/// A pre-A2 envelope (no section) parses as `None` — the common case constrains nothing.
#[test]
fn absent_device_min_is_none() {
    let frozen = v1_envelope().freeze(&author()).expect("freeze");
    assert_eq!(frozen.device_min(), None);
}
