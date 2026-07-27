// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

//! C1's transport delta — training convergence and churn when the gossip mesh is **relay-carried**
//! (`docs/specs/vhc-fleet-ceremony-runbook.md` §4.2: WAN peers behind NAT reach each other through
//! the pinned iroh relay, not through published direct addresses).
//!
//! The dual-plane gate (`iroh_dual_plane.rs`) proves the second transport with direct loopback
//! dialing — the roster's `ip:port` addresses ARE the reachability, which is the one thing a WAN
//! fleet does not have. This gate removes them: every node runs `[vhc.iroh]` with
//! `advertise_ips = []` and a pinned relay URL, so its roster record carries **no direct
//! addresses**. The net crate's endpoint uses `presets::Minimal` (no DNS/pkarr/mDNS discovery)
//! seeded only from the verified roster, so a formed QUIC gossip connection (`NeighborUp`) has
//! exactly one possible dial path: through the relay. That makes the marker sufficient, not just
//! suggestive — if the relay plane were broken, no mesh could form at all.
//!
//! On top of the relay-carried mesh the gate runs the C1 churn drill (the graceful choreography
//! the loopback churn gate proved): a trainer drains out mid-run, the survivor settles, the second
//! trainer drains too, both rejoin as fresh incarnations — every rejoin re-publishing a relay-only
//! roster record and re-forming the mesh through the relay — and training resumes past the
//! checkpoint with byte-identical per-round det digests.
//!
//! The relay itself is the REAL `iroh-relay` binary in its localhost dev mode (plain HTTP — the
//! same runner `crates/vhc/host/daemon-vhc-net/dev/run-relay.sh` wraps), spawned per gate on a
//! free port. When `iroh-relay` is not on PATH (a standalone checkout outside the devShell) the
//! gate skips cleanly with a loud note, mirroring the net crate's relay unit test — inside the
//! acceptance lane the devShell always ships it.

mod harness;

use std::io::Write as _;
use std::process::{Child, Command, Stdio};
use std::time::Duration;

use harness::{
    assert_digests_agree, base_peer, join, journal_digests, leave, node_log_contains,
    seed_corpus_fs, spawn_node_with, start_cluster_with, wait_rounds, NodeSpec,
};

const RUN: &str = "acceptance-iroh-relay-plane";
/// The mesh-formation marker `daemon-vhc-net`'s gossip plane logs on a formed QUIC connection.
const NEIGHBOR_MARKER: &str = "iroh gossip neighbor up";

/// The spawned dev relay (`iroh-relay --dev` on a free port), killed on drop.
struct DevRelay {
    child: Child,
    port: u16,
    /// The throwaway `--config-path` TOML (`--dev` on a non-default port needs `http_bind_addr`;
    /// dev mode still forces plain HTTP and ignores TLS fields). Owned so it outlives the child.
    _config: tempfile::NamedTempFile,
}

impl DevRelay {
    /// Spawn the relay and block until its HTTP listener accepts, or `None` when the binary is
    /// not on PATH (skip-clean seam for standalone checkouts).
    fn spawn() -> Option<Self> {
        let port = harness::free_port();
        let mut config = tempfile::NamedTempFile::new().expect("relay config tempfile");
        writeln!(config, "http_bind_addr = \"127.0.0.1:{port}\"").expect("write relay config");
        writeln!(config, "enable_metrics = false").expect("write relay config");
        config.flush().expect("flush relay config");

        let child = match Command::new("iroh-relay")
            .arg("--dev")
            .arg("--config-path")
            .arg(config.path())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
        {
            Ok(child) => child,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return None,
            Err(e) => panic!("spawn iroh-relay: {e}"),
        };
        let mut relay = Self {
            child,
            port,
            _config: config,
        };
        relay.wait_ready();
        Some(relay)
    }

    /// The relay URL the nodes pin (`[vhc.iroh] relays`).
    fn url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    /// Block until the relay's HTTP listener accepts a TCP connection (the readiness probe the
    /// runbook's `generate_204` check reduces to on loopback).
    fn wait_ready(&mut self) {
        let deadline = std::time::Instant::now() + Duration::from_secs(20);
        loop {
            if std::net::TcpStream::connect(("127.0.0.1", self.port)).is_ok() {
                return;
            }
            if let Ok(Some(status)) = self.child.try_wait() {
                panic!("iroh-relay exited before serving: {status}");
            }
            assert!(
                std::time::Instant::now() < deadline,
                "iroh-relay never served port {}",
                self.port
            );
            std::thread::sleep(Duration::from_millis(100));
        }
    }
}

impl Drop for DevRelay {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Wait until the registry holds a trainer checkpoint pointer of `kind` at (or past) `min_round`.
async fn wait_checkpoint(
    cluster: &harness::Cluster,
    kind: &str,
    min_round: u64,
    timeout: Duration,
) -> harness::registry::Checkpoint {
    let deadline = std::time::Instant::now() + timeout;
    loop {
        if let Some(c) = cluster.registry.checkpoint("trainer", kind) {
            if c.round >= min_round {
                return c;
            }
        }
        assert!(
            std::time::Instant::now() < deadline,
            "no trainer {kind} checkpoint pointer at round >= {min_round} within {timeout:?} \
             (got {:?})",
            cluster
                .registry
                .checkpoint("trainer", kind)
                .map(|c| c.round)
        );
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn relay_carried_mesh_trains_and_survives_graceful_churn() {
    let _serial = harness::serial_guard();
    let Some(relay) = DevRelay::spawn() else {
        eprintln!(
            "SKIP: `iroh-relay` is not on PATH — run inside `nix develop` (the devShell ships \
             iroh-relay; the acceptance lane always has it)"
        );
        return;
    };
    let payload_root = tempfile::tempdir().expect("shared payload root");

    let base_port = harness::free_port();
    let base_url = format!("http://127.0.0.1:{base_port}/api/v1/vhc");

    // Relay-only reachability: no advertised direct addresses, one pinned relay URL — the WAN
    // posture. (An empty `advertise_ips` still binds loopback; nothing is published to dial it.)
    let iroh_toml = format!(
        "[vhc.iroh]\nenabled = true\nrelays = \"{}\"\nadvertise_ips = []\n",
        relay.url()
    );

    let mk = |name: &str, seat_claim: bool| NodeSpec {
        name: Box::leak(name.to_string().into_boxed_str()),
        registry_base: Box::leak(base_url.clone().into_boxed_str()),
        seat_claim,
        payload_dir: Some(payload_root.path()),
        allowlist: Box::leak(base_url.clone().into_boxed_str()),
        reconcile_tick_ms: 500,
        initial_backoff_ms: 0,
    };
    let coord = spawn_node_with(&mk("relay-coordinator", true), &iroh_toml);
    let trainer_a = spawn_node_with(&mk("relay-trainer-a", false), &iroh_toml);
    let trainer_b = spawn_node_with(&mk("relay-trainer-b", false), &iroh_toml);

    let bases = [
        base_peer(&coord),
        base_peer(&trainer_a),
        base_peer(&trainer_b),
    ];
    // The churn tier: real coordinator timer, deadline-survivable absences, a one-round absence
    // budget, prompt cooldown (where the staged rejoins materialize).
    let cluster = start_cluster_with(
        base_port,
        RUN,
        &bases,
        0,
        2,
        1,
        daemon_vhc_testkit::live_genesis::LiveTiming::churn(),
    )
    .await;
    seed_corpus_fs(payload_root.path(), RUN, &cluster.genesis);

    join(&coord, RUN, "op-coord").await;
    join(&trainer_a, RUN, "op-a").await;
    join(&trainer_b, RUN, "op-b").await;

    // A healthy pre-churn baseline over the relay-carried mesh.
    let timeout = Duration::from_secs(180);
    wait_rounds(&trainer_a, RUN, 2, timeout).await;
    wait_rounds(&trainer_b, RUN, 2, timeout).await;

    // The sufficiency signal: a formed QUIC gossip connection per node. With no direct addresses
    // in any roster record and no ambient discovery, the relay is the only dial path — this
    // marker cannot appear unless the relay actually carried the mesh.
    for node in ["relay-coordinator", "relay-trainer-a", "relay-trainer-b"] {
        assert!(
            node_log_contains(node, NEIGHBOR_MARKER),
            "node `{node}` never logged `{NEIGHBOR_MARKER}` — the relay-carried mesh did not form"
        );
    }

    // The C1 churn drill (the graceful choreography the loopback churn gate proved), now with
    // every rejoin re-publishing a relay-only roster record: trainer A drains out; the survivor
    // settles; B drains too (its later checkpoint folds the interregnum record and wins the
    // pointer); both rejoin as fresh incarnations and resume from the checkpoint document.
    leave(&trainer_a, RUN, daemon_api::VhcLeaveMode::Graceful, "op-la").await;
    wait_checkpoint(&cluster, "drain", 0, Duration::from_secs(60)).await;

    let after_a = journal_digests(&trainer_b, RUN)
        .keys()
        .max()
        .copied()
        .unwrap_or(0);
    let deadline = std::time::Instant::now() + Duration::from_secs(90);
    loop {
        let b_now = journal_digests(&trainer_b, RUN)
            .keys()
            .max()
            .copied()
            .unwrap_or(0);
        if b_now >= after_a {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "survivor never settled after the graceful leave"
        );
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    tokio::time::sleep(Duration::from_secs(2)).await;
    leave(&trainer_b, RUN, daemon_api::VhcLeaveMode::Graceful, "op-lb").await;
    let pointer = wait_checkpoint(&cluster, "drain", after_a, Duration::from_secs(60)).await;

    join(&trainer_a, RUN, "op-a2").await;
    join(&trainer_b, RUN, "op-b2").await;

    // Training resumes past the checkpoint through the re-formed relay mesh (the ghost
    // incarnation costs one deadline round before it drops).
    let target = pointer.round + 2;
    let churn_timeout = Duration::from_secs(240);
    wait_rounds(&trainer_a, RUN, target, churn_timeout).await;
    wait_rounds(&trainer_b, RUN, target, churn_timeout).await;

    leave(
        &trainer_a,
        RUN,
        daemon_api::VhcLeaveMode::Graceful,
        "op-la2",
    )
    .await;
    leave(
        &trainer_b,
        RUN,
        daemon_api::VhcLeaveMode::Graceful,
        "op-lb2",
    )
    .await;
    leave(&coord, RUN, daemon_api::VhcLeaveMode::Graceful, "op-lc").await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    // The digest oracle across the churn: every shared round — pre-churn and post-restore alike —
    // carries one agreed det digest.
    let a = journal_digests(&trainer_a, RUN);
    let b = journal_digests(&trainer_b, RUN);
    assert!(
        a.keys().any(|r| *r > pointer.round),
        "trainer-a produced no post-restore digests (restore did not resume training)"
    );
    assert_digests_agree(&a, &b, 3);

    drop(cluster);
    drop(relay);
}
