// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// **The D2 failover drill — the non-Authority half** (refactor §8/D2: "kill coordinator node;
// standby resumes from archive + journal"; architecture §4.4: "state reconstruction is
// deterministic; authority transfer remains a distributed-systems problem").
//
// The drill, in the host testkit against the PRODUCTION coordinator blob:
//
// 1. A **primary** `coordinator_quorum.wasm` (configured from a genesis envelope v2) drives
//    rounds while the harness — playing the session's persistence seat — journals every delivered
//    input into the on-disk A1 journal (tag-1 events + tag-3 clocks, mirroring the guest's
//    synthetic per-event clock) plus the tag-10 initial state.
// 2. Mid-run, the sealed journal prefix is **published to the record archive** under signed chain
//    heads; the current segment stays on disk unsealed — the local journal *tail*.
// 3. The primary is **killed** mid-round-stream.
// 4. A **standby** reconstructs the coordinator state deterministically from the **archive prefix
//    + the journal tail alone** (chain-walked, content-re-hashed, tail chained against the last
//    archived segment), re-folds the pure `tick` natively — legitimate because the D2
//    dual-compilation gate proves native ≡ blob — and boots a FRESH incarnation of the production
//    blob with the reconstructed state as its config. The guest resumes its synthetic clock from
//    the restored state, so its decisions continue the same logical timeline.
// 5. The remaining rounds are delivered to the standby; its decisions must be **byte-identical**
//    to what an uninterrupted reference run would have produced — resumption, not approximation.
//
// **The explicitly-marked gap (sitting 3, post-D1):** the standby signs its §12.1 frames under a
// NEW key — this drill asserts that honestly (`standby_sender != envelope authority`). Deciding
// who may SIGN next — split-brain prevention, fencing the old signer, seq continuity — is the
// `Authority` contract's signer-transfer protocol (architecture §4.4), deliberately NOT built
// here. Until D1, records from a standby are correct but not yet *authoritative*.
//
// Dev/test harness: shells `cargo build` for guests; journal writes real files.
#![allow(clippy::disallowed_methods)]

use std::path::PathBuf;
use std::process::Command;
use std::sync::Once;
use std::time::Duration;

use ciborium::value::Value;

use daemon_vhc_host::v2::RunEnd;
use daemon_vhc_observe::journal::archive::ChainHead;
use daemon_vhc_observe::journal::oracle::{record_initial_state, record_input, record_run_header};
use daemon_vhc_observe::journal::record::ExecIdentity;
use daemon_vhc_observe::journal::segment::scan_file;
use daemon_vhc_observe::journal::store::{Journal, RotatePolicy};
use daemon_vhc_observe::journal::StaticKey;
use daemon_vhc_observe::{
    extract_consensus_capture, recover_chain_from_archive, RecordArchive, ReplicationPolicy,
    RetentionPolicy,
};
use daemon_vhc_proto::messages::{
    Commitment, Heartbeat, Join, RecordEntry, StorageReceipt, ThroughputClass,
};
use daemon_vhc_proto::sign::Signed;
use daemon_vhc_proto::{
    blake3_hash, peer_id, to_canonical_vec, CapabilitySet, Hash, IrohId, PeerId, SignedMessage,
    SigningKey, SwarmMessage, SWARM_PROTO_VERSION,
};
use daemon_vhc_sdk_consensus::coordinator::{
    tick, tick_authenticated, CoordinatorState, Input, Output,
};
use daemon_vhc_testkit::barrier::phase_a_grants;
use daemon_vhc_testkit::{
    cell8_genesis, configure_wasm_coordinator, WasmCoordinator, WasmCoordinatorSpec,
};

const ROUNDS_BEFORE_KILL: u64 = 4;
const ROUNDS_AFTER: u64 = 2;
const RUN_LABEL: &str = "failover-drill";

// -- guest build (the established testkit pattern) -------------------------------------------------

fn guests_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../guests")
        .canonicalize()
        .expect("guests workspace path")
}

fn guest_remap_rustflags() -> String {
    let root = guests_root();
    let checkout = root.ancestors().nth(3).unwrap_or(&root).to_path_buf();
    let cargo_home = std::env::var_os("CARGO_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".cargo"));
    format!(
        "--remap-path-prefix={}=/daemon-node --remap-path-prefix={}=/cargo",
        checkout.display(),
        cargo_home.display(),
    )
}

static BUILD: Once = Once::new();

fn coordinator_quorum_wasm() -> Vec<u8> {
    BUILD.call_once(|| {
        let status = Command::new("cargo")
            .current_dir(guests_root())
            .env_remove("CARGO_TARGET_DIR")
            .env("RUSTFLAGS", guest_remap_rustflags())
            .args(["build", "--release", "--target", "wasm32-unknown-unknown"])
            .status()
            .expect("run cargo for guests");
        assert!(status.success(), "building guest modules failed");
    });
    let path = guests_root().join("target/wasm32-unknown-unknown/release/coordinator_quorum.wasm");
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

fn tempdir() -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    let mut base = std::env::temp_dir();
    base.push(format!(
        "dvhc-failover-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&base).unwrap();
    base
}

// -- the deterministic input script ----------------------------------------------------------------

struct ScriptMsg {
    key: SigningKey,
    msg: SwarmMessage,
}

fn sign(k: &SigningKey, m: &SwarmMessage) -> SignedMessage {
    SignedMessage::sign(k, SWARM_PROTO_VERSION, m.clone()).expect("sign")
}

/// The join/warmup prologue + one commitment pair and covering receipt per round.
fn build_script(worker_keys: &[SigningKey; 2], rounds: std::ops::Range<u64>) -> Vec<ScriptMsg> {
    let peers: Vec<PeerId> = worker_keys.iter().map(peer_id).collect();
    let mut script = Vec::new();
    if rounds.start == 0 {
        for k in worker_keys {
            script.push(ScriptMsg {
                key: k.clone(),
                msg: SwarmMessage::Join(Join {
                    run_id: RUN_LABEL.into(),
                    iroh_id: IrohId([0x44; 32]),
                    class: ThroughputClass::C1,
                    capabilities: CapabilitySet::new(),
                    envelope_hash: None,
                }),
            });
        }
        for k in worker_keys {
            script.push(ScriptMsg {
                key: k.clone(),
                msg: SwarmMessage::Heartbeat(Heartbeat {
                    round: 0,
                    ready: Some(true),
                }),
            });
        }
    }
    for round in rounds {
        let mut entries = Vec::new();
        for (i, k) in worker_keys.iter().enumerate() {
            let bytes = format!("update/{i}/{round}").into_bytes();
            let hash = blake3_hash(&bytes);
            script.push(ScriptMsg {
                key: k.clone(),
                msg: SwarmMessage::Commitment(Commitment {
                    round,
                    payload: hash,
                    size: bytes.len() as u64,
                    locators: Vec::new(),
                }),
            });
            entries.push(RecordEntry {
                peer: peers[i],
                hash,
                size: bytes.len() as u64,
            });
        }
        script.push(ScriptMsg {
            key: worker_keys[0].clone(),
            msg: SwarmMessage::StorageReceipt(StorageReceipt {
                round,
                verified: entries,
            }),
        });
    }
    script
}

/// The uninterrupted native reference: fold `tick` over the whole script exactly as the guest
/// does (authenticated dispatch + one synthetic clock tick per frame), collecting every publish.
/// The synthetic clock CONTINUES from the initial state's clock — the same resume rule the guest
/// applies (`da_init` resumes `now_s` from the restored state), so a reference over a
/// reconstructed mid-run state stays on the primary's logical timeline.
fn reference_publishes(initial: CoordinatorState, script: &[ScriptMsg]) -> Vec<SwarmMessage> {
    let version = initial.config.proto_version;
    let mut now_s = initial.now_s;
    let mut state = initial;
    let mut published = Vec::new();
    for sm in script {
        let (next, outputs) = tick_authenticated(state, peer_id(&sm.key), version, sm.msg.clone());
        state = next;
        for o in &outputs {
            if let Output::Publish(m) = o {
                published.push((**m).clone());
            }
        }
        now_s += 1;
        let (next, outputs) = tick(state, Input::Clock(now_s));
        state = next;
        for o in &outputs {
            if let Output::Publish(m) = o {
                published.push((**m).clone());
            }
        }
    }
    published
}

/// Decode the coordinator's initial state out of its opaque `{state: …}` config bytes.
fn initial_state_of(spec: &WasmCoordinatorSpec) -> CoordinatorState {
    let v: Value = ciborium::de::from_reader(spec.config_bytes.as_slice()).expect("config cbor");
    let Value::Map(entries) = v else {
        panic!("coordinator config is a map")
    };
    let state = entries
        .iter()
        .find_map(|(k, val)| match k {
            Value::Text(t) if t == "state" => Some(val.clone()),
            _ => None,
        })
        .expect("config carries state");
    state.deserialized().expect("state decodes")
}

#[test]
#[allow(clippy::too_many_lines)]
fn standby_resumes_from_archive_plus_journal_tail_byte_identically() {
    let wasm = coordinator_quorum_wasm();
    let coord_hash = Hash(*blake3::hash(&wasm).as_bytes());

    // Identities: the primary's §12.1 frame key IS the envelope-named SingleKey identity.
    let primary_key_seed = *blake3::hash(b"failover/primary-key").as_bytes();
    let authority = peer_id(&SigningKey::from_bytes(&primary_key_seed));
    let worker_keys = [
        SigningKey::from_bytes(blake3::hash(b"failover/worker/0").as_bytes()),
        SigningKey::from_bytes(blake3::hash(b"failover/worker/1").as_bytes()),
    ];

    // The genesis envelope v2 + the coordinator spec derived from it (the cell-8 seat).
    let genesis = cell8_genesis(RUN_LABEL, coord_hash, Hash([0x77; 32]), authority, 2, 2, 4);
    let author = SigningKey::from_bytes(blake3::hash(b"failover/author").as_bytes());
    let frozen = genesis.freeze(&author).expect("genesis freeze");
    let spec = configure_wasm_coordinator(&frozen).expect("coordinator configurable");
    let initial = initial_state_of(&spec);

    // The persistence seat: the A1 journal the primary's inputs are recorded into.
    let ident = ExecIdentity {
        run_id: spec.run_id,
        epoch: 0,
        role: "coordinator".into(),
        instance: 0,
        module: coord_hash,
    };
    let root = tempdir();
    let mut journal = Journal::create(
        &root,
        ident.clone(),
        StaticKey::new([7u8; 32]),
        RotatePolicy {
            max_records: 10_000,
        }, // explicit rolls below control the seal points
    )
    .expect("journal create");
    record_run_header(&mut journal, &ident, Vec::new()).expect("run header");
    record_initial_state(&mut journal, &initial).expect("snapshot");

    // -- 1. the primary drives rounds 0..4, every input journaled (message + clock pair) ---------
    let mut primary =
        WasmCoordinator::start(&wasm, &spec, phase_a_grants(), 0, primary_key_seed).unwrap();
    let pre_kill = build_script(&worker_keys, 0..ROUNDS_BEFORE_KILL);
    let mut at = 0u64;
    let mut now_s = 0u64;
    // Seal + archive the prefix after this many delivered frames; the rest stays in the tail.
    let archive_cut = pre_kill.len() * 2 / 3;
    let mut cut_ord: Option<u64> = None;
    for (i, sm) in pre_kill.iter().enumerate() {
        primary.deliver(&sm.key, &sm.msg).expect("primary deliver");
        let signed = sign(&sm.key, &sm.msg);
        record_input(&mut journal, at, &Input::Message(signed)).expect("record msg");
        at += 1;
        now_s += 1;
        record_input(&mut journal, at, &Input::Clock(now_s)).expect("record clock");
        at += 1;
        if i + 1 == archive_cut {
            journal.commit().expect("barrier");
            journal.roll().expect("seal the archived prefix");
            cut_ord = Some(journal.current_segment());
        }
    }
    journal.commit().expect("tail barrier (§8.4)");
    let tail_segment = cut_ord.expect("the cut happened");

    // Drain the primary's decisions up to the kill point (records 0..4 + opens) and assert the
    // primary signed as the envelope-named identity.
    let mut primary_decisions = Vec::new();
    let expected_pre = reference_publishes(initial.clone(), &pre_kill);
    while primary_decisions.len() < expected_pre.len() {
        let (sender, _, msg) = primary
            .next_decision(Duration::from_secs(60))
            .expect("primary decision");
        assert_eq!(
            sender, authority.0,
            "primary signs as the SingleKey identity"
        );
        primary_decisions.push(msg);
    }

    // -- 2. publish the sealed prefix to the record archive under signed heads -------------------
    let head_signer = SigningKey::from_bytes(&primary_key_seed);
    let mut archive = RecordArchive::new(
        authority,
        ReplicationPolicy { factor: 1 },
        RetentionPolicy::default(),
    );
    let mut heads = Vec::new();
    let mut prev = Hash([0u8; 32]);
    for ord in 0..tail_segment {
        let scan = scan_file(journal.paths().segment(ord)).expect("scan sealed");
        assert!(scan.sealed, "archived prefix segments are sealed");
        let bytes = std::fs::read(journal.paths().segment(ord)).expect("read segment");
        let addr = archive.publish_segment(bytes).expect("publish");
        let head = Signed::seal(
            &head_signer,
            ChainHead {
                run_id: ident.run_id,
                epoch: 0,
                role: "coordinator".into(),
                instance: 0,
                module: ident.module,
                segment: ord,
                segment_hash: addr,
                prev_hash: prev,
                records: scan.records.len() as u64,
            },
        )
        .expect("sign head");
        archive.ingest_head(head.clone()).expect("head accepted");
        heads.push(head);
        prev = addr;
    }
    assert!(!heads.is_empty(), "an archived prefix exists");

    // -- 3. kill the primary mid-stream -----------------------------------------------------------
    let end = primary.kill().expect("primary killed");
    assert!(matches!(end, RunEnd::Outcome(_)), "guest thread joined");

    // -- 4. the standby reconstructs from ARCHIVE + JOURNAL TAIL alone ----------------------------
    // Archive half: authenticated heads, contiguous chain, content re-hashed.
    let chain = recover_chain_from_archive(&archive, &heads).expect("archive recovery");
    // Journal-tail half: the unsealed segment on disk, chained against the last archived segment.
    let tail_scan = scan_file(journal.paths().segment(tail_segment)).expect("tail scan");
    assert!(!tail_scan.sealed, "the tail was never sealed");
    assert_eq!(
        Hash(tail_scan.header.prev_blake3),
        heads.last().expect("heads").body.segment_hash,
        "the tail chains off the archived prefix"
    );
    let mut records = chain.records;
    records.extend(tail_scan.records);
    let capture = extract_consensus_capture(&records).expect("capture extraction");
    let rec_initial = capture
        .initial
        .expect("the archived prefix carries the snapshot");

    // Deterministic state reconstruction (architecture §4.4): fold the pure tick natively —
    // native ≡ blob by the D2 dual-compilation gate.
    let mut state = rec_initial;
    for input in capture.inputs {
        let (next, _) = tick(state, input);
        state = next;
    }
    assert_eq!(
        state.round, ROUNDS_BEFORE_KILL,
        "the reconstructed state stands at the next un-opened round"
    );

    // -- 5. boot the standby (fresh incarnation, reconstructed state as config) and resume -------
    let standby_config = {
        let v = Value::Map(vec![(
            Value::Text("state".into()),
            Value::serialized(&state).expect("state value"),
        )]);
        to_canonical_vec(&v).expect("standby config")
    };
    let standby_spec = WasmCoordinatorSpec {
        module_hash: spec.module_hash,
        config_bytes: standby_config,
        authority: spec.authority,
        run_id: spec.run_id,
    };
    // The standby's OWN key (incarnation 1): signer transfer is the marked Authority gap.
    let standby_key_seed = *blake3::hash(b"failover/standby-key").as_bytes();
    let mut standby =
        WasmCoordinator::start(&wasm, &standby_spec, phase_a_grants(), 1, standby_key_seed)
            .unwrap();

    let post_kill = build_script(
        &worker_keys,
        ROUNDS_BEFORE_KILL..ROUNDS_BEFORE_KILL + ROUNDS_AFTER,
    );
    for sm in &post_kill {
        standby.deliver(&sm.key, &sm.msg).expect("standby deliver");
    }

    // The uninterrupted reference over the SAME post-kill inputs, continuing from the
    // reconstructed state: the standby's decisions must be byte-identical (resumption).
    let expected_post = reference_publishes(state, &post_kill);
    assert!(
        expected_post
            .iter()
            .filter(|m| matches!(m, SwarmMessage::RoundRecord(_)))
            .count()
            == ROUNDS_AFTER as usize,
        "the reference records the post-kill rounds"
    );
    let standby_sender = peer_id(&SigningKey::from_bytes(&standby_key_seed));
    for (i, expected) in expected_post.iter().enumerate() {
        let (sender, _, msg) = standby
            .next_decision(Duration::from_secs(60))
            .unwrap_or_else(|e| panic!("standby decision {i}: {e}"));
        assert_eq!(
            to_canonical_vec(&msg).unwrap(),
            to_canonical_vec(expected).unwrap(),
            "standby decision {i} diverged from the uninterrupted reference"
        );
        // THE MARKED GAP (sitting 3, post-D1): the standby's records are byte-correct but signed
        // under a NEW identity — not the envelope-named authority. Judging/fencing that signer is
        // the Authority contract's signer-transfer protocol (architecture §4.4), not built here.
        assert_eq!(sender, standby_sender.0, "standby signs under its own key");
        assert_ne!(
            sender, authority.0,
            "signer transfer is NOT performed by this drill (the Authority gap)"
        );
    }

    // The resumed decisions continue the primary's timeline: together they equal one
    // uninterrupted run over the full script.
    let total_rounds = ROUNDS_BEFORE_KILL + ROUNDS_AFTER;
    let full_script = build_script(&worker_keys, 0..total_rounds);
    let uninterrupted = reference_publishes(initial, &full_script);
    let mut resumed = primary_decisions;
    resumed.extend(expected_post);
    assert_eq!(
        uninterrupted.len(),
        resumed.len(),
        "kill+resume produced the same number of decisions as an uninterrupted run"
    );
    for (a, b) in uninterrupted.iter().zip(resumed.iter()) {
        assert_eq!(
            to_canonical_vec(a).unwrap(),
            to_canonical_vec(b).unwrap(),
            "kill+resume ≡ uninterrupted, byte-for-byte"
        );
    }

    standby.stop().expect("standby stops clean");
}
