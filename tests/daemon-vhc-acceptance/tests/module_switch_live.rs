// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

//! G7 — the in-suite LIVE MODULE SWITCH through the product path (architecture §5.4; ABI §10.3):
//! the same three-node topology as the baseline gate trains real rounds, then an authorized
//! run-level module-upgrade record — authored by the genesis-named upgrade authority, targeting
//! the trainer role at epoch 1 with a byte-distinct (appended custom section) but semantically
//! identical trainer module — is consumed on BOTH trainer nodes through `vhc_switch_module`
//! over their Unix-socket product API. Nothing is reached around: the node validates the record
//! fail-closed against its rebuilt transition chain, provisions the post-switch identity, and
//! drives the worker's switch transaction (quiesce → snapshot → owner-law re-admission →
//! migrate → validate → activate).
//!
//! Asserted, all offline/product-path:
//!   - both consumptions answer `Activated { epoch: 1 }` over the API;
//!   - the trainers' pre-switch det digests agree (the run really trained first);
//!   - each trainer's DURABLE journal shows the §8.1 continuation seam: a sealed retired span
//!     under the old identity and a new span whose header carries epoch 1, the new module hash,
//!     and a strictly higher incarnation;
//!   - a stale record (epoch 1 re-presented after activation... a replayed consumption) and a
//!     record signed by a non-authority key are REFUSED typed, every node process unharmed.
//!
//! Post-switch cross-peer round progression is deliberately NOT asserted: re-scoping every
//! role's live frame stream to the new epoch at a coordinated fence is run-level (coordinator
//! module / SDK) choreography, outside the node-side record-consumption seam this gate proves.

mod harness;

use std::time::Duration;

use daemon_api::{ApiRequest, ApiResponse, VhcSwitchOutcome};
use daemon_vhc_proto::{blake3_hash, to_canonical_vec, Hash, UpgradeRecord};
use harness::{
    assert_digests_agree, base_peer, guest_wasm, join, journal_digests, leave, seed_corpus_fs,
    spawn_node, start_cluster_on, upgrade_authority_key, wait_rounds, Node, NodeSpec,
};

const RUN: &str = "acceptance-live-switch";

/// The upgraded trainer artifact: the pinned trainer bytes plus an appended wasm CUSTOM section
/// (id 0) — byte-distinct (a different content hash, so the artifact plane and the hash pins are
/// genuinely re-exercised) yet semantically identical (custom sections are ignored by the
/// runtime), so the migrated state remains valid under the target.
fn upgraded_trainer_wasm() -> Vec<u8> {
    let mut bytes = guest_wasm("tiny_llama");
    let name = b"acceptance-upgrade";
    let payload = b"epoch-1";
    let content_len = 1 + name.len() + payload.len();
    assert!(content_len < 0x80, "single-byte LEB128 section size");
    bytes.push(0x00); // custom section id
    bytes.push(content_len as u8);
    bytes.push(name.len() as u8);
    bytes.extend_from_slice(name);
    bytes.extend_from_slice(payload);
    bytes
}

/// Seed one content-addressed object into the shared filesystem payload plane (the same layout
/// `seed_corpus_fs` writes and the worker's `FsContentStore` opens).
fn seed_payload_object(shared_root: &std::path::Path, run_label: &str, bytes: &[u8]) -> [u8; 32] {
    let hash = *blake3::hash(bytes).as_bytes();
    let dir = shared_root
        .join(blake3::hash(run_label.as_bytes()).to_hex().as_str())
        .join("payload");
    std::fs::create_dir_all(&dir).expect("shared payload dir");
    std::fs::write(dir.join(Hash(hash).to_hex()), bytes).expect("seed payload object");
    hash
}

/// Consume an upgrade record on a node through the product API.
async fn switch_module(node: &Node, record: &UpgradeRecord, op: &str) -> VhcSwitchOutcome {
    let resp = node
        .client()
        .call(ApiRequest::VhcSwitchModule {
            run_id: RUN.to_string(),
            upgrade_record: to_canonical_vec(record).expect("record wire"),
            op_id: op.to_string(),
        })
        .await
        .expect("vhc_switch_module call");
    match resp {
        ApiResponse::VhcSwitchOutcome(outcome) => outcome,
        other => panic!("vhc_switch_module on `{}`: {other:?}", node.name),
    }
}

/// The `(epoch, instance, module)` identity of every journal SEGMENT header for a run on a node,
/// in segment order — the offline §8.1 seam evidence (the continuation rolls a segment whose
/// header carries the incoming identity).
fn journal_spans(node: &Node, run_label: &str) -> Vec<(u64, u64, [u8; 32])> {
    let run_state = node
        .run_dir()
        .join(blake3::hash(run_label.as_bytes()).to_hex().as_str());
    let mut spans = Vec::new();
    let Ok(entries) = std::fs::read_dir(&run_state) else {
        return spans;
    };
    for entry in entries.flatten() {
        let journal = entry.path().join("journal");
        let Ok(paths) = daemon_vhc_journal::JournalPaths::open(&journal) else {
            continue;
        };
        for ord in paths.existing_segments().unwrap_or_default() {
            let Ok(scan) = daemon_vhc_journal::segment::scan_file(paths.segment(ord)) else {
                continue;
            };
            spans.push((
                scan.header.id.epoch,
                scan.header.id.instance,
                scan.header.id.module.0,
            ));
        }
    }
    spans
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn live_module_switch_activates_through_the_product_api() {
    let _serial = harness::serial_guard();
    let payload_root = tempfile::tempdir().expect("shared payload root");
    let base_port = harness::free_port();
    let base_url = format!("http://127.0.0.1:{base_port}/api/v1/vhc");

    let mk = |name: &'static str, seat_claim: bool| NodeSpec {
        name,
        registry_base: Box::leak(base_url.clone().into_boxed_str()),
        seat_claim,
        payload_dir: Some(payload_root.path()),
        allowlist: Box::leak(base_url.clone().into_boxed_str()),
        reconcile_tick_ms: 500,
        initial_backoff_ms: 0,
    };
    let coord = spawn_node(&mk("coordinator", true));
    let trainer_a = spawn_node(&mk("trainer-a", false));
    let trainer_b = spawn_node(&mk("trainer-b", false));

    let bases = [
        base_peer(&coord),
        base_peer(&trainer_a),
        base_peer(&trainer_b),
    ];
    let cluster = start_cluster_on(base_port, RUN, &bases, 0, 2).await;
    seed_corpus_fs(payload_root.path(), RUN, &cluster.genesis);

    join(&coord, RUN, "op-coord").await;
    join(&trainer_a, RUN, "op-a").await;
    join(&trainer_b, RUN, "op-b").await;

    // The run really trains first: both trainers voice agreeing det digests.
    let rounds = 3u64;
    let timeout = Duration::from_secs(180);
    wait_rounds(&trainer_a, RUN, rounds, timeout).await;
    wait_rounds(&trainer_b, RUN, rounds, timeout).await;

    // The committed target: a byte-distinct trainer artifact on the shared payload plane.
    let old_module = *blake3::hash(&guest_wasm("tiny_llama")).as_bytes();
    let new_wasm = upgraded_trainer_wasm();
    let new_module = seed_payload_object(payload_root.path(), RUN, &new_wasm);
    assert_ne!(old_module, new_module, "the target is byte-distinct");
    let grants_hash =
        daemon_vhc_testkit::role_grants_hash(&cluster.genesis.wire, "trainer", &new_wasm)
            .expect("derive the target's grants anchor");

    // The run-level event: the genesis-named upgrade authority authorizes epoch 1 for the
    // trainer role (committed once, globally; each node validates it against its own rebuilt
    // chain before any local switch).
    let genesis_hash = Hash(cluster.genesis.genesis_hash);
    let record = UpgradeRecord::author(
        genesis_hash,
        1,
        genesis_hash,
        "trainer",
        Hash(old_module),
        Hash(new_module),
        rounds,
        Hash(grants_hash),
        blake3_hash(&[]),
        &[&upgrade_authority_key()],
    )
    .expect("author the upgrade record");

    // A record signed by a NON-authority key is refused typed (fail closed), instance untouched.
    let rogue = UpgradeRecord::author(
        genesis_hash,
        1,
        genesis_hash,
        "trainer",
        Hash(old_module),
        Hash(new_module),
        rounds,
        Hash(grants_hash),
        blake3_hash(&[]),
        &[&harness::key_from("acceptance/rogue-authority")],
    )
    .expect("author the rogue record");
    match switch_module(&trainer_a, &rogue, "op-rogue").await {
        VhcSwitchOutcome::Refused { reason } => {
            assert!(
                reason.contains("not authorized"),
                "the refusal names the authority failure: {reason}"
            );
        }
        other => panic!("a non-authority record must refuse, got {other:?}"),
    }

    // Consume the authorized record on BOTH trainer nodes through the product API.
    for (node, op) in [(&trainer_a, "op-switch-a"), (&trainer_b, "op-switch-b")] {
        match switch_module(node, &record, op).await {
            VhcSwitchOutcome::Activated {
                epoch, module_hash, ..
            } => {
                assert_eq!(epoch, 1, "`{}` activated the target epoch", node.name);
                assert_eq!(
                    module_hash,
                    Hash(new_module).to_hex(),
                    "`{}` activated the committed module",
                    node.name
                );
            }
            other => panic!("switch on `{}` did not activate: {other:?}", node.name),
        }
    }

    // A replayed consumption of the SAME record is refused typed (the chain already advanced;
    // strictly-monotone epochs admit no re-append), instances unharmed.
    match switch_module(&trainer_a, &record, "op-replay").await {
        VhcSwitchOutcome::Refused { reason } => {
            assert!(
                reason.contains("epoch"),
                "the refusal names the epoch monotonicity: {reason}"
            );
        }
        other => panic!("a replayed record must refuse, got {other:?}"),
    }

    // Let the new spans' run-headers settle on disk, then read the offline seam evidence.
    tokio::time::sleep(Duration::from_secs(2)).await;
    for node in [&trainer_a, &trainer_b] {
        let spans = journal_spans(node, RUN);
        let old_span = spans
            .iter()
            .find(|(epoch, _, module)| *epoch == 0 && *module == old_module)
            .unwrap_or_else(|| panic!("`{}` has an epoch-0 span: {spans:?}", node.name));
        let new_span = spans
            .iter()
            .find(|(epoch, _, module)| *epoch == 1 && *module == new_module)
            .unwrap_or_else(|| {
                panic!(
                    "`{}` journal shows the epoch-1 continuation span: {spans:?}",
                    node.name
                )
            });
        assert!(
            new_span.1 > old_span.1,
            "`{}`: the post-switch incarnation {} supersedes {}",
            node.name,
            new_span.1,
            old_span.1
        );
    }

    // Pre-switch digest agreement stands (the run trained real rounds before the fence), and
    // every node process survived the whole exercise (typed refusals never crash).
    let a = journal_digests(&trainer_a, RUN);
    let b = journal_digests(&trainer_b, RUN);
    assert_digests_agree(&a, &b, rounds as usize);
    for node in [&coord, &trainer_a, &trainer_b] {
        assert!(node.is_alive(), "node `{}` survived", node.name);
    }

    leave(&trainer_a, RUN, daemon_api::VhcLeaveMode::Graceful, "op-la").await;
    leave(&trainer_b, RUN, daemon_api::VhcLeaveMode::Graceful, "op-lb").await;
    leave(&coord, RUN, daemon_api::VhcLeaveMode::Graceful, "op-lc").await;

    drop(cluster);
}
