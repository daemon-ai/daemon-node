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
//    archived segment) — **through the sandbox, never a native `tick` fold**: it replays the
//    recovered inputs into a fresh `coordinator_quorum.wasm` instance seeded from the archived
//    initial state and EXPORTS the rebuilt `CoordinatorState` via the typed quiesce→snapshot path
//    (§10.2). It then boots a FRESH incarnation of the production blob and re-instantiates it from
//    that exported snapshot (`da_migrate`). The guest resumes its synthetic clock from the restored
//    state, so its decisions continue the same logical timeline. Consensus never runs outside the
//    content-addressed module, even to rebuild a standby (architecture §4.1/§4.4).
// 5. The remaining rounds are delivered to the standby; its decisions must be **byte-identical**
//    to what an uninterrupted reference run would have produced — resumption, not approximation.
//
// **Signer transfer per the Authority contract (sitting 3, D1 merged):** the run's trust root is
// the envelope-named `SingleKey` **base identity** (`Reconfiguration::SingleSigner`: "a standby
// is a journal-replicated warm spare and transfer is a key-custody problem — fence the old
// signer via a signed epoch-fence record"). Run traffic is signed by **certified per-run keys**
// (D1's `RunKeyCertificate`, chained to the base): the primary's incarnation-0 run key carries
// cert 0; at takeover the base issues the standby's incarnation-1 certificate — that base-signed
// issuance IS the transfer statement, and ingesting it **fences every lower incarnation** (the
// cert store refuses instance-0 senders from then on; incarnation monotonicity gives the
// ordering, §8.1 never-reused). The drill asserts: pre-kill frames authenticate via the cert
// chain; an UNcertified standby key is refused (`NoCertifiedChain`); after the transfer the
// standby authenticates via ITS certificate and the fenced old signer is refused. What remains
// for tier-3 live: the same protocol over a real fleet (lease expiry, WAN cert distribution).
//
// Dev/test harness: shells `cargo build` for guests; journal writes real files.
#![allow(clippy::disallowed_methods)]

use std::path::PathBuf;
use std::process::Command;
use std::sync::Once;
use std::time::Duration;

use ciborium::value::Value;

use daemon_vhc_host::v2::RunEnd;
use daemon_vhc_observe::journal::archive::{AttestedHead, ChainHead};
use daemon_vhc_observe::journal::oracle::{record_initial_state, record_input, record_run_header};
use daemon_vhc_observe::journal::record::ExecIdentity;
use daemon_vhc_observe::journal::segment::scan_file;
use daemon_vhc_observe::journal::store::{Journal, RotatePolicy};
use daemon_vhc_observe::journal::StaticKey;
use daemon_vhc_observe::{
    extract_consensus_capture, recover_chain_from_archive, RecordArchive, ReplicationPolicy,
    RetentionPolicy,
};
use daemon_vhc_proto::cert::{verify_certified_sender, CertError, RunKeyCertificate};
use daemon_vhc_proto::messages::{
    Commitment, Heartbeat, Join, RecordEntry, StorageReceipt, ThroughputClass,
};
use daemon_vhc_proto::{
    blake3_hash, peer_id, to_canonical_vec, CapabilitySet, Hash, IrohId, PeerId, SignedMessage,
    SigningKey, SwarmMessage, SWARM_PROTO_VERSION,
};
use daemon_vhc_sdk_consensus::coordinator::{CoordinatorState, Input};
use daemon_vhc_testkit::cell8::phase_a_grants;
use daemon_vhc_testkit::{
    cell8_genesis, configure_wasm_coordinator, coordinator_state_from_capture, WasmCoordinator,
    WasmCoordinatorSpec,
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

/// The uninterrupted reference is itself a wasm-coordinator run (consensus never runs outside the
/// sandbox — the D2 dual-compilation gate proves the module ≡ the native tick, so no separate
/// native oracle is authored here): drive a FRESH `coordinator_quorum.wasm` blob over `script`
/// from the same genesis config, drain exactly `decisions` published decisions in order, and stop
/// it clean. Every barrier round contributes a `RoundOpen` + `RoundRecord`; a round that opens but
/// is never recorded (the trailing open past the last committed round) adds one more open — so
/// `decisions = 2 * committed_rounds (+ 1 trailing open)`.
fn wasm_reference(
    wasm: &[u8],
    spec: &WasmCoordinatorSpec,
    key_seed: [u8; 32],
    script: &[ScriptMsg],
    decisions: usize,
) -> Vec<SwarmMessage> {
    let mut coord = WasmCoordinator::start(wasm, spec, phase_a_grants(), 0, key_seed).unwrap();
    for sm in script {
        coord.deliver(&sm.key, &sm.msg).expect("reference deliver");
    }
    let mut published = Vec::new();
    while published.len() < decisions {
        let (_, _, msg) = coord
            .next_decision(Duration::from_secs(60))
            .expect("reference decision");
        published.push(msg);
    }
    coord.stop().expect("reference stops clean");
    published
}

/// Published decisions a `committed`-round barrier drive produces, plus the trailing open of the
/// round that opened but never recorded (`with_trailing_open`): `2 * committed (+ 1)`.
fn decision_count(committed: u64, with_trailing_open: bool) -> usize {
    2 * committed as usize + usize::from(with_trailing_open)
}

/// The receivers' run-key certificate store — the signer-transfer seat (architecture §4.4 under
/// `Reconfiguration::SingleSigner`). It trusts certificates chained to the run's base identity
/// and enforces **fencing by incarnation monotonicity**: ingesting the base-signed certificate of
/// a NEWER coordinator incarnation (the transfer statement) fences every lower incarnation — a
/// fenced instance's frames are refused even though its certificate once verified. Incarnations
/// are never reused (§8.1), so the ordering is total and rollback-free.
struct CoordinatorCertStore {
    run_id: Hash,
    trusted_base: PeerId,
    certs: Vec<RunKeyCertificate>,
    /// The lowest live coordinator incarnation; anything below is fenced.
    live_floor: u64,
}

impl CoordinatorCertStore {
    fn new(run_id: Hash, trusted_base: PeerId) -> Self {
        Self {
            run_id,
            trusted_base,
            certs: Vec::new(),
            live_floor: 0,
        }
    }

    /// Ingest a base-signed run-key certificate. A certificate for a newer incarnation is the
    /// signer-transfer statement: the live floor advances and every lower incarnation is fenced.
    fn ingest(&mut self, cert: RunKeyCertificate) {
        assert!(cert.verify_chain().is_ok(), "only chain-valid certs ingest");
        self.live_floor = self.live_floor.max(cert.body.instance);
        self.certs.push(cert);
    }

    /// Authenticate a coordinator frame's `sender` for `instance` at `epoch`: refused if the
    /// incarnation is fenced, else the D1 certified-sender check against the trusted base.
    fn accept(&self, instance: u64, epoch: u64, sender: &PeerId) -> Result<(), CertError> {
        if instance < self.live_floor {
            // The fence: a base-signed transfer to a newer incarnation supersedes this signer.
            return Err(CertError::NoCertifiedChain);
        }
        verify_certified_sender(
            &self.run_id,
            "coordinator",
            instance,
            epoch,
            sender,
            &self.trusted_base,
            &self.certs,
        )
    }
}

/// Wrap a `CoordinatorState` in the module's opaque `{state: …}` `da_init` config shape.
fn state_config(state: &CoordinatorState) -> Vec<u8> {
    let v = Value::Map(vec![(
        Value::Text("state".into()),
        Value::serialized(state).expect("state value"),
    )]);
    to_canonical_vec(&v).expect("state config")
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

    // Identities per D1's cert layering (architecture §4.3): the envelope-named SingleKey
    // authority is the coordinator's BASE identity (the trust root; it signs certificates and
    // holds head custody); run traffic is signed by per-run keys certified by it. The primary is
    // incarnation 0 under run key 0; the standby will be incarnation 1 under run key 1.
    let base_key = SigningKey::from_bytes(blake3::hash(b"failover/base-identity").as_bytes());
    let base_id = peer_id(&base_key);
    let primary_key_seed = *blake3::hash(b"failover/primary-key").as_bytes();
    let primary_sender = peer_id(&SigningKey::from_bytes(&primary_key_seed));
    let standby_key_seed = *blake3::hash(b"failover/standby-key").as_bytes();
    let standby_sender = peer_id(&SigningKey::from_bytes(&standby_key_seed));
    let worker_keys = [
        SigningKey::from_bytes(blake3::hash(b"failover/worker/0").as_bytes()),
        SigningKey::from_bytes(blake3::hash(b"failover/worker/1").as_bytes()),
    ];

    // The genesis envelope v2 + the coordinator spec derived from it (the cell-8 seat).
    let genesis = cell8_genesis(RUN_LABEL, coord_hash, Hash([0x77; 32]), base_id, 2, 2, 4);
    let author = SigningKey::from_bytes(blake3::hash(b"failover/author").as_bytes());
    let frozen = genesis.freeze(&author).expect("genesis freeze");
    let spec = configure_wasm_coordinator(&frozen).expect("coordinator configurable");
    let initial = initial_state_of(&spec);

    // The receivers' cert store: the base certifies the primary's incarnation-0 run key.
    let mut certs = CoordinatorCertStore::new(spec.run_id, base_id);
    certs.ingest(
        RunKeyCertificate::issue(
            &base_key,
            spec.run_id,
            "coordinator",
            0,
            0,
            0,
            primary_sender,
        )
        .expect("issue primary cert"),
    );

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

    // Drain the primary's decisions up to the kill point: the ROUNDS_BEFORE_KILL records + their
    // opens, plus the trailing open of the next (never-recorded) round. Each frame's sender
    // authenticates through the D1 certified-sender check — the cert chain to the base identity,
    // incarnation 0, epoch 0 (the reconciled signer seat).
    let mut primary_decisions = Vec::new();
    let expected_pre = decision_count(ROUNDS_BEFORE_KILL, true);
    while primary_decisions.len() < expected_pre {
        let (sender, _, msg) = primary
            .next_decision(Duration::from_secs(60))
            .expect("primary decision");
        certs
            .accept(0, 0, &PeerId(sender))
            .expect("primary frame authenticates via its run-key certificate");
        primary_decisions.push(msg);
    }

    // -- 2. publish the sealed prefix to the record archive under attested heads (base custody) --
    let mut archive = RecordArchive::new(
        spec.authority.clone(),
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
        let head = AttestedHead::single(
            &base_key,
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
        .expect("attest head");
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

    // Deterministic state reconstruction THROUGH THE SANDBOX (architecture §4.1/§4.4): consensus
    // never runs outside the content-addressed module, even to rebuild a standby — so instead of
    // folding the pure native `tick`, replay the recovered inputs into a fresh coordinator-quorum
    // instance seeded from the archived initial state, then EXPORT its rebuilt state via the typed
    // quiesce→snapshot path (§10.2). The module owns a one-tick-per-frame synthetic clock, so the
    // recovered `Input::Message` frames alone reproduce the primary's state (the recovered
    // `Input::Clock`s are redundant with the module's own per-frame ticks).
    let rebuild_spec = WasmCoordinatorSpec {
        module_hash: spec.module_hash,
        config_bytes: state_config(&rec_initial),
        authority: spec.authority.clone(),
        run_id: spec.run_id,
    };
    let rebuild_key_seed = *blake3::hash(b"failover/rebuild-key").as_bytes();
    let mut rebuild =
        WasmCoordinator::start(&wasm, &rebuild_spec, phase_a_grants(), 2, rebuild_key_seed)
            .unwrap();
    for input in &capture.inputs {
        if let Input::Message(signed) = input {
            rebuild
                .deliver_signed(signed)
                .expect("replay a recovered frame into the reconstruction instance");
        }
    }
    // Drain the rebuilt decisions before exporting: a `Quiesce` freezes delivery (queued frames
    // spool), so the module must have finished folding the replayed inputs first. The rebuild
    // re-derives the same prefix the primary published — its trailing open marks the round the
    // exported state stands in.
    let expected_rebuild = decision_count(ROUNDS_BEFORE_KILL, true);
    let mut drained = 0usize;
    while drained < expected_rebuild {
        rebuild
            .next_decision(Duration::from_secs(60))
            .expect("rebuild decision");
        drained += 1;
    }
    let exported = rebuild
        .quiesce_snapshot(60_000)
        .expect("the reconstruction instance exports its state through the sandbox");
    let state = coordinator_state_from_capture(&exported).expect("exported state decodes");
    assert_eq!(
        state.round, ROUNDS_BEFORE_KILL,
        "the reconstructed state stands at the next un-opened round"
    );

    // -- 5. SIGNER TRANSFER per the Authority contract (architecture §4.4, SingleSigner) ---------
    // Before the transfer, the standby's run key is UNcertified: receivers refuse it typed.
    assert_eq!(
        certs.accept(1, 0, &standby_sender),
        Err(CertError::NoCertifiedChain),
        "an uncertified standby key is refused before the transfer"
    );
    // The base identity issues the standby's incarnation-1 certificate — the base-signed transfer
    // statement. Ingesting it advances the live floor: incarnation 0 is FENCED from here on.
    certs.ingest(
        RunKeyCertificate::issue(
            &base_key,
            spec.run_id,
            "coordinator",
            1,
            0,
            0,
            standby_sender,
        )
        .expect("issue standby cert"),
    );
    assert_eq!(
        certs.accept(0, 0, &primary_sender),
        Err(CertError::NoCertifiedChain),
        "the fenced old signer is refused after the transfer (split-brain prevention)"
    );

    // -- 6. boot the standby (fresh incarnation, RE-INSTANTIATED from the exported snapshot) -----
    // `da_init` runs with the genesis config, then `da_migrate` restores the exported state through
    // the sandbox (ABI §10.3 step 4) — the module continues the same logical timeline from there.
    let mut standby = WasmCoordinator::start_migrating(
        &wasm,
        &spec,
        phase_a_grants(),
        1,
        standby_key_seed,
        exported,
    )
    .unwrap();

    let post_kill = build_script(
        &worker_keys,
        ROUNDS_BEFORE_KILL..ROUNDS_BEFORE_KILL + ROUNDS_AFTER,
    );
    for sm in &post_kill {
        standby.deliver(&sm.key, &sm.msg).expect("standby deliver");
    }

    // Drain the standby's decisions: the round the reconstructed state already stands in was
    // opened during reconstruction, so the standby's stream starts at that round's RECORD — the
    // post-kill rounds contribute 2 decisions each (record + the next round's open), no leading
    // open. The standby's decisions must be byte-identical to what an uninterrupted run produces
    // over the same tail (resumption, not approximation).
    let mut standby_decisions = Vec::new();
    let expected_post = decision_count(ROUNDS_AFTER, false);
    for i in 0..expected_post {
        let (sender, _, msg) = standby
            .next_decision(Duration::from_secs(60))
            .unwrap_or_else(|e| panic!("standby decision {i}: {e}"));
        // The sitting-2 gap, CLOSED: the standby's frames authenticate through the certified path
        // — its incarnation-1 run key chains to the base identity via the transfer cert.
        certs
            .accept(1, 0, &PeerId(sender))
            .expect("standby frame authenticates via the transfer certificate");
        assert_eq!(sender, standby_sender.0, "standby signs under its own key");
        standby_decisions.push(msg);
    }
    assert_eq!(
        standby_decisions
            .iter()
            .filter(|m| matches!(m, SwarmMessage::RoundRecord(_)))
            .count(),
        ROUNDS_AFTER as usize,
        "the standby records the post-kill rounds"
    );

    // The uninterrupted reference is a fresh wasm-coordinator run over the full script (no native
    // oracle): the resumed decisions — the killed primary's prefix followed by the standby's
    // resumption — must equal it byte-for-byte.
    let total_rounds = ROUNDS_BEFORE_KILL + ROUNDS_AFTER;
    let full_script = build_script(&worker_keys, 0..total_rounds);
    let uninterrupted = wasm_reference(
        &wasm,
        &spec,
        primary_key_seed,
        &full_script,
        decision_count(total_rounds, true),
    );
    let mut resumed = primary_decisions;
    resumed.extend(standby_decisions);
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

/// The coordinator module **state-export** proof (architecture §4.1/§4.4; ABI §10.2): a running
/// `coordinator_quorum.wasm` is quiesced, snapshots its consensus state through the sandbox, is torn
/// down, and is re-instantiated from that manifest (`da_migrate`) — and the resumed run's published
/// decisions are byte-identical to an uninterrupted reference run's. This is the direct proof of
/// the export → re-instantiate cycle (the failover drill exercises the same export on a rebuilt
/// instance recovered from the archive); it is the prerequisite for standby reconstruction running
/// entirely through the sandbox.
#[test]
fn coordinator_state_export_resumes_byte_identically() {
    let wasm = coordinator_quorum_wasm();
    let coord_hash = Hash(*blake3::hash(&wasm).as_bytes());

    let base_key = SigningKey::from_bytes(blake3::hash(b"export/base-identity").as_bytes());
    let base_id = peer_id(&base_key);
    let primary_key_seed = *blake3::hash(b"export/primary-key").as_bytes();
    let standby_key_seed = *blake3::hash(b"export/standby-key").as_bytes();
    let worker_keys = [
        SigningKey::from_bytes(blake3::hash(b"export/worker/0").as_bytes()),
        SigningKey::from_bytes(blake3::hash(b"export/worker/1").as_bytes()),
    ];

    let genesis = cell8_genesis(RUN_LABEL, coord_hash, Hash([0x77; 32]), base_id, 2, 2, 4);
    let author = SigningKey::from_bytes(blake3::hash(b"export/author").as_bytes());
    let frozen = genesis.freeze(&author).expect("genesis freeze");
    let spec = configure_wasm_coordinator(&frozen).expect("coordinator configurable");

    // The primary drives the first rounds, then we drain its published decisions.
    let mut primary =
        WasmCoordinator::start(&wasm, &spec, phase_a_grants(), 0, primary_key_seed).unwrap();
    let pre_kill = build_script(&worker_keys, 0..ROUNDS_BEFORE_KILL);
    for sm in &pre_kill {
        primary.deliver(&sm.key, &sm.msg).expect("primary deliver");
    }
    let mut primary_decisions = Vec::new();
    let expected_pre = decision_count(ROUNDS_BEFORE_KILL, true);
    while primary_decisions.len() < expected_pre {
        let (_, _, msg) = primary
            .next_decision(Duration::from_secs(60))
            .expect("primary decision");
        primary_decisions.push(msg);
    }

    // Quiesce → snapshot → tear down: the module exports its consensus state through the sandbox.
    let exported = primary
        .quiesce_snapshot(60_000)
        .expect("the primary exports its state through the sandbox");
    let state = coordinator_state_from_capture(&exported).expect("exported state decodes");
    assert_eq!(
        state.round, ROUNDS_BEFORE_KILL,
        "the exported state stands at the round the primary had opened but not recorded"
    );

    // Re-instantiate a FRESH incarnation from the exported manifest and resume the remaining rounds.
    let mut standby = WasmCoordinator::start_migrating(
        &wasm,
        &spec,
        phase_a_grants(),
        1,
        standby_key_seed,
        exported,
    )
    .unwrap();
    let post_kill = build_script(
        &worker_keys,
        ROUNDS_BEFORE_KILL..ROUNDS_BEFORE_KILL + ROUNDS_AFTER,
    );
    for sm in &post_kill {
        standby.deliver(&sm.key, &sm.msg).expect("standby deliver");
    }
    let mut standby_decisions = Vec::new();
    let expected_post = decision_count(ROUNDS_AFTER, false);
    for i in 0..expected_post {
        let (_, _, msg) = standby
            .next_decision(Duration::from_secs(60))
            .unwrap_or_else(|e| panic!("standby decision {i}: {e}"));
        standby_decisions.push(msg);
    }
    standby.stop().expect("standby stops clean");

    // The uninterrupted reference: a single fresh instance over the full script.
    let total_rounds = ROUNDS_BEFORE_KILL + ROUNDS_AFTER;
    let full_script = build_script(&worker_keys, 0..total_rounds);
    let uninterrupted = wasm_reference(
        &wasm,
        &spec,
        primary_key_seed,
        &full_script,
        decision_count(total_rounds, true),
    );
    let mut resumed = primary_decisions;
    resumed.extend(standby_decisions);
    assert_eq!(
        uninterrupted.len(),
        resumed.len(),
        "export+resume produced the same number of decisions as an uninterrupted run"
    );
    for (a, b) in uninterrupted.iter().zip(resumed.iter()) {
        assert_eq!(
            to_canonical_vec(a).unwrap(),
            to_canonical_vec(b).unwrap(),
            "export+resume ≡ uninterrupted, byte-for-byte"
        );
    }
}
