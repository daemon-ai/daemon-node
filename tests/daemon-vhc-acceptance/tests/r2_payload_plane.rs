// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

//! G2 — WS control + R2-compatible payload plane: identical topology to the baseline gate, but the
//! payload/artifact plane is the presigned R2 content store (the fixture's presign + object store)
//! instead of the shared filesystem. Proves the presign/receipt plane end to end — committed
//! payloads are `put` via presign and fetched via presigned GET, the coordinator's availability
//! check reads them back, and the run converges with agreeing per-round digests.

mod harness;

use std::time::Duration;

use harness::{
    assert_digests_agree, base_peer, join, journal_digests, leave, seed_corpus_r2, spawn_node,
    start_cluster_on, wait_rounds, NodeSpec,
};

const RUN: &str = "acceptance-r2";

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn three_node_training_over_presigned_r2_payload_plane() {
    let base_port = harness::free_port();
    let base_url = format!("http://127.0.0.1:{base_port}/api/v1/vhc");

    // No shared filesystem payload root: with `payload_dir` absent, the node authors credentials
    // with the registry presign base, selecting the R2 content store on every worker.
    let mk = |name: &str, seat_claim: bool| NodeSpec {
        name: Box::leak(name.to_string().into_boxed_str()),
        registry_base: Box::leak(base_url.clone().into_boxed_str()),
        seat_claim,
        payload_dir: None,
        allowlist: Box::leak(base_url.clone().into_boxed_str()),
        reconcile_tick_ms: 500,
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
    seed_corpus_r2(&cluster, RUN, &cluster.genesis);

    join(&coord, RUN, "op-coord").await;
    join(&trainer_a, RUN, "op-a").await;
    join(&trainer_b, RUN, "op-b").await;

    let rounds = 3u64;
    let timeout = Duration::from_secs(180);
    wait_rounds(&trainer_a, RUN, rounds, timeout).await;
    wait_rounds(&trainer_b, RUN, rounds, timeout).await;

    leave(&trainer_a, RUN, daemon_api::VhcLeaveMode::Graceful, "op-la").await;
    leave(&trainer_b, RUN, daemon_api::VhcLeaveMode::Graceful, "op-lb").await;
    leave(&coord, RUN, daemon_api::VhcLeaveMode::Graceful, "op-lc").await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    let a = journal_digests(&trainer_a, RUN);
    let b = journal_digests(&trainer_b, RUN);
    assert_digests_agree(&a, &b, rounds as usize);

    drop(cluster);
}
