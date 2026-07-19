// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// The env/spawn/fs bans target the shipped node, not an integration harness that spawns real
// product binaries and inspects their on-disk state: allowed file-wide here.
#![allow(clippy::disallowed_methods, clippy::disallowed_types, dead_code)]

//! The multi-process acceptance harness: three REAL `daemon` node processes over Unix sockets,
//! plus the local in-process fixtures they talk to (a run registry + seat-CAS server backed by
//! the normative fold, and — for the R2 payload tier — the presign/object store fixture). No
//! testkit relay, no pump handle: everything here is a spawned binary, a socket, a fixture
//! server, or an on-disk artifact.
//!
//! The digest oracle is OFFLINE and product-path: each node writes a durable §8 journal under
//! its own state dir; after the run the harness scans the two trainers' journals, decodes the
//! guest's tag-4 det-digest publishes, and asserts per-round agreement (the reused det-lane
//! digest comparison, now over durable product artifacts instead of an in-process pump).

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};

use daemon_api::{ApiRequest, ApiResponse, VhcEvent, VhcLeaveMode, VhcPolicy, VhcPolicyMode};
use daemon_host::ApiClient;
use daemon_vhc_journal::record::Body;
use daemon_vhc_journal::segment::scan_file;
use daemon_vhc_net::FakeSeatRegistry;
use daemon_vhc_proto::{peer_id, SeatState, SigningKey};
use daemon_vhc_testkit::{live_genesis, LiveGenesis, LiveGenesisSpec};

pub mod registry;
pub use registry::FixtureRegistry;

/// The vhc control channel the guest publishes its tag-N voices on.
const CONTROL_CHANNEL: u64 = 0;

/// Locate a sibling product binary in the cargo target profile dir (the xtask acceptance lane
/// builds `daemon` + `daemon-vhc-worker` before the suite runs). Walks up from the test
/// executable to the profile dir (the parent of `deps/`) and expects `<profile>/<name>`.
pub fn locate_bin(name: &str) -> PathBuf {
    let exe = std::env::current_exe().expect("test executable path");
    // .../target/<profile>/deps/<test>-<hash>  →  .../target/<profile>
    let profile_dir = exe
        .ancestors()
        .find(|p| p.join(name).is_file())
        .unwrap_or_else(|| {
            panic!(
                "could not locate `{name}` next to the test binary ({}) — the xtask acceptance \
                 lane must `cargo build -p daemon -p daemon-vhc-worker` first",
                exe.display()
            )
        });
    profile_dir.join(name)
}

/// A free loopback TCP port (bind :0, read the assignment, drop the listener).
pub fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .expect("bind ephemeral port")
        .local_addr()
        .expect("local addr")
        .port()
}

/// The path to a built guest blob (the guests workspace's release target).
pub fn guest_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../crates/vhc/guests/target/wasm32-unknown-unknown/release")
        .join(format!("{name}.wasm"))
}

/// The workspace's built guest blob (the guests workspace's release target).
pub fn guest_wasm(name: &str) -> Vec<u8> {
    let path = guest_path(name);
    std::fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "read guest {} ({}): {e} — run `cargo run -p xtask -- build-guests` first",
            name,
            path.display()
        )
    })
}

/// Seed the run's chunk-addressed corpus objects into the shared filesystem content plane so the
/// trainers' `data@2` fetches resolve them (the local stand-in for a shared object store). Each
/// object lands under its content-hash hex at `<shared>/<blake3(label)>/payload/<hex>` — exactly
/// the `FsContentStore` layout the worker opens.
pub fn seed_corpus_fs(shared_payload_root: &Path, run_label: &str, genesis: &LiveGenesis) {
    let dir = shared_payload_root
        .join(blake3::hash(run_label.as_bytes()).to_hex().as_str())
        .join("payload");
    std::fs::create_dir_all(&dir).expect("shared payload dir");
    for (hash, bytes) in &genesis.corpus_objects {
        let hex = daemon_vhc_proto::Hash(*hash).to_hex();
        std::fs::write(dir.join(hex.as_str()), bytes).expect("seed corpus object");
    }
}

/// The vendored chunk-addressed corpus fixture directory.
pub fn corpus_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/corpus")
}

/// One node process on the product path: a spawned `daemon` in its host role, its own isolated
/// state dir + Unix socket, driven only through [`ApiClient`].
pub struct Node {
    pub name: String,
    pub data_dir: tempfile::TempDir,
    pub socket: PathBuf,
    pub base_key: SigningKey,
    child: Child,
}

impl Node {
    /// The node's product API client (a fresh one-shot connection per call).
    pub fn client(&self) -> ApiClient {
        ApiClient::new(&self.socket)
    }

    /// The node's identity keystore dir (base identity lives here; the harness reads the base
    /// pubkey to author the genesis trust set).
    pub fn identity_dir(&self) -> PathBuf {
        self.data_dir.path().join("vhc").join("identity")
    }

    /// The run-state root (per-incarnation journals + the fs payload plane live under it).
    pub fn run_dir(&self) -> PathBuf {
        self.data_dir.path().join("vhc").join("runs")
    }
}

impl Drop for Node {
    fn drop(&mut self) {
        // Best-effort graceful stop; SIGKILL fallback. The worker child is reaped by the node's
        // own shutdown, but confirm no orphaned worker survives (resource discipline).
        let pid = self.child.id();
        let _ = nix::sys::signal::kill(
            nix::unistd::Pid::from_raw(pid as i32),
            nix::sys::signal::Signal::SIGTERM,
        );
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            match self.child.try_wait() {
                Ok(Some(_)) => break,
                _ if Instant::now() >= deadline => {
                    let _ = self.child.kill();
                    let _ = self.child.wait();
                    break;
                }
                _ => std::thread::sleep(Duration::from_millis(50)),
            }
        }
    }
}

/// How a node is configured before it boots.
pub struct NodeSpec<'a> {
    pub name: &'a str,
    /// The registry base URL (discovery + seat CAS + presign).
    pub registry_base: &'a str,
    /// Enable coordinator duty (the seat-claiming node).
    pub seat_claim: bool,
    /// The shared filesystem payload-plane root (the multi-node single-host plane). `None` ⇒
    /// the presigned R2 tier (the registry base's presign endpoint).
    pub payload_dir: Option<&'a Path>,
    /// The allowlisted coordinator endpoint (the WS/registry base the run is served from).
    pub allowlist: &'a str,
    /// Reconcile tick (ms) — kept short so churn reconvergence is prompt in-test.
    pub reconcile_tick_ms: u64,
}

/// Spawn a node process with a seeded base identity, blocking until it serves its socket.
pub fn spawn_node(spec: &NodeSpec<'_>) -> Node {
    let data_dir = tempfile::tempdir().expect("node data dir");
    let socket = data_dir.path().join("api.sock");
    // Pre-create the identity keystore so the harness can read the node's base pubkey BEFORE the
    // node boots (the genesis trust set names every node's base identity). The node opens the
    // same keystore at boot (idempotent).
    let identity_dir = data_dir.path().join("vhc").join("identity");
    std::fs::create_dir_all(&identity_dir).expect("identity dir");
    let keystore =
        daemon_vhc_session::keystore::VhcKeystore::open(&identity_dir).expect("open node keystore");
    let base_key = keystore.base_identity().expect("node base identity");

    // The node config TOML (the `[vhc]` section drives everything the suite needs).
    let worker_bin = locate_bin("daemon-vhc-worker");
    let worker_bin = worker_bin.display().to_string();
    let mut toml = String::new();
    toml.push_str("[vhc]\n");
    toml.push_str("enabled = true\n");
    writeln!(toml, "worker_path = {worker_bin:?}").unwrap();
    writeln!(
        toml,
        "identity_dir = {:?}",
        identity_dir.display().to_string()
    )
    .unwrap();
    writeln!(toml, "seat_claim = {}", spec.seat_claim).unwrap();
    writeln!(toml, "coordinator_allowlist = [{:?}]", spec.allowlist).unwrap();
    if let Some(dir) = spec.payload_dir {
        writeln!(toml, "payload_dir = {:?}", dir.display().to_string()).unwrap();
    }
    toml.push_str("[vhc.owner_budget]\nunbounded = true\n");
    toml.push_str("[vhc.registry]\n");
    writeln!(toml, "base = {:?}", spec.registry_base).unwrap();
    toml.push_str("[vhc.retry]\n");
    writeln!(toml, "reconcile_tick_ms = {}", spec.reconcile_tick_ms).unwrap();
    let config_path = data_dir.path().join("node.toml");
    std::fs::write(&config_path, toml).expect("write node config");

    let mut cmd = Command::new(locate_bin("daemon"));
    cmd.env_clear();
    if let Ok(p) = std::env::var("PATH") {
        cmd.env("PATH", p);
    }
    for var in ["SSL_CERT_FILE", "NIX_SSL_CERT_FILE", "SSL_CERT_DIR"] {
        if let Ok(v) = std::env::var(var) {
            cmd.env(var, v);
        }
    }
    cmd.env("DAEMON_STORE", "memory")
        .env("DAEMON_DATA_DIR", data_dir.path())
        .env("DAEMON_SOCKET_PATH", &socket)
        .env("DAEMON_CONFIG", &config_path)
        // The training worker's compute lane needs no GPU; keep it CPU/ndarray.
        .env("DAEMON_VHC_LANE_GPU_OPTIONAL", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    let child = cmd.spawn().expect("spawn daemon node");

    let node = Node {
        name: spec.name.to_string(),
        data_dir,
        socket,
        base_key,
        child,
    };
    // Block until the socket is served (a fail-fast boot exits first).
    let deadline = Instant::now() + Duration::from_secs(30);
    while !node.socket.exists() {
        if Instant::now() >= deadline {
            panic!("node `{}` did not serve its socket within 30s", spec.name);
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    node
}

/// A default participation policy (always-on, uncapped — the arbiter is unbounded in-suite).
pub fn always_policy() -> VhcPolicy {
    VhcPolicy {
        mode: VhcPolicyMode::Always,
        vram_cap_mb: 0,
        duty_cycle_pct: 100,
        schedule: None,
    }
}

/// Issue `vhc_join` over the socket.
pub async fn join(node: &Node, run_id: &str, op: &str) {
    let resp = node
        .client()
        .call(ApiRequest::VhcJoin {
            run_id: run_id.to_string(),
            policy: always_policy(),
            op_id: op.to_string(),
        })
        .await
        .expect("vhc_join call");
    assert!(
        matches!(resp, ApiResponse::Ok),
        "vhc_join for `{run_id}` on `{}`: {resp:?}",
        node.name
    );
}

/// Issue `vhc_leave` over the socket.
pub async fn leave(node: &Node, run_id: &str, mode: VhcLeaveMode, op: &str) {
    let _ = node
        .client()
        .call(ApiRequest::VhcLeave {
            run_id: run_id.to_string(),
            mode,
            op_id: op.to_string(),
        })
        .await
        .expect("vhc_leave call");
}

/// Read a run's recent [`VhcEvent`]s from the node's `vhc_run_detail` window.
pub async fn recent_events(node: &Node, run_id: &str) -> Vec<VhcEvent> {
    let resp = node
        .client()
        .call(ApiRequest::VhcRunDetail {
            run_id: run_id.to_string(),
        })
        .await
        .expect("vhc_run_detail call");
    match resp {
        ApiResponse::VhcRunDetail(Some(detail)) => detail.recent_events,
        _ => Vec::new(),
    }
}

/// Poll until `node`'s run has progressed to at least `rounds` distinct round outcomes, or the
/// deadline elapses. Returns the highest round observed. A persisted `VhcEvent::Error` fails the
/// wait loudly (never a silent drop).
pub async fn wait_rounds(node: &Node, run_id: &str, rounds: u64, timeout: Duration) -> u64 {
    let deadline = Instant::now() + timeout;
    let mut best = 0u64;
    loop {
        for ev in recent_events(node, run_id).await {
            match ev {
                VhcEvent::RoundOutcome { round, .. } => best = best.max(round + 1),
                VhcEvent::Error { class, detail, .. } => {
                    panic!(
                        "node `{}` run `{run_id}` errored: {class}: {detail}",
                        node.name
                    )
                }
                _ => {}
            }
        }
        if best >= rounds {
            return best;
        }
        if Instant::now() >= deadline {
            panic!(
                "node `{}` run `{run_id}` reached only {best} rounds (< {rounds}) within {timeout:?}",
                node.name
            );
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
}

/// The per-round tag-4 det digests decoded from a node's durable journals for `run_id` (the
/// offline oracle input — pure product artifacts). Scans every incarnation's segments.
pub fn journal_digests(node: &Node, run_label: &str) -> BTreeMap<u64, [u8; 16]> {
    let run_state = node
        .run_dir()
        .join(blake3::hash(run_label.as_bytes()).to_hex().as_str());
    let mut digests = BTreeMap::new();
    let Ok(entries) = std::fs::read_dir(&run_state) else {
        return digests;
    };
    for entry in entries.flatten() {
        let journal = entry.path().join("journal");
        let paths = match daemon_vhc_journal::JournalPaths::open(&journal) {
            Ok(p) => p,
            Err(_) => continue,
        };
        for ord in paths.existing_segments().unwrap_or_default() {
            let Ok(scan) = scan_file(paths.segment(ord)) else {
                continue;
            };
            for record in scan.records {
                if let Body::Publish(p) = record.body {
                    if p.channel != CONTROL_CHANNEL {
                        continue;
                    }
                    if let Some((tag, round, bytes)) = decode_tagged(&p.frame) {
                        if tag == 4 {
                            if let Ok(d) = <[u8; 16]>::try_from(bytes.as_slice()) {
                                digests.insert(round, d);
                            }
                        }
                    }
                }
            }
        }
    }
    digests
}

/// Decode a §12.1 signed wire frame `[envelope, payload, sig]` and its inner module payload
/// `[tag, round, bytes]` (the guest's own publish shape) — the reused det-lane `decode_tagged`.
fn decode_tagged(frame: &[u8]) -> Option<(u64, u64, Vec<u8>)> {
    let v: ciborium::value::Value = ciborium::de::from_reader(frame).ok()?;
    let ciborium::value::Value::Array(parts) = v else {
        return None;
    };
    let ciborium::value::Value::Bytes(payload) = parts.get(1)? else {
        return None;
    };
    let inner: ciborium::value::Value = ciborium::de::from_reader(payload.as_slice()).ok()?;
    let ciborium::value::Value::Array(items) = inner else {
        return None;
    };
    let uint = |i: usize| -> Option<u64> {
        items
            .get(i)
            .and_then(ciborium::value::Value::as_integer)
            .and_then(|n| u64::try_from(i128::from(n)).ok())
    };
    let bytes = match items.get(2) {
        Some(ciborium::value::Value::Bytes(b)) => b.clone(),
        _ => Vec::new(),
    };
    Some((uint(0)?, uint(1)?, bytes))
}

/// Assert the two trainers agree on the tag-4 det digest for every round both reported, over at
/// least `min_rounds` shared rounds — the cross-peer digest-agreement oracle (§7 pass criterion).
pub fn assert_digests_agree(
    a: &BTreeMap<u64, [u8; 16]>,
    b: &BTreeMap<u64, [u8; 16]>,
    min_rounds: usize,
) {
    let shared: Vec<u64> = a.keys().filter(|r| b.contains_key(r)).copied().collect();
    assert!(
        shared.len() >= min_rounds,
        "the two trainers share only {} rounds of digests (< {min_rounds}); A={:?} B={:?}",
        shared.len(),
        a.keys().collect::<Vec<_>>(),
        b.keys().collect::<Vec<_>>()
    );
    for round in shared {
        assert_eq!(
            a[&round], b[&round],
            "trainers disagree on the round-{round} det digest"
        );
    }
}

/// The shared cluster fixtures for one gate: the registry + seat server and the authored genesis.
pub struct Cluster {
    pub registry: Arc<FixtureRegistry>,
    pub genesis: LiveGenesis,
    pub run_label: String,
    pub base_url: String,
    _registry_task: tokio::task::JoinHandle<()>,
}

/// Author the acceptance genesis for the nodes' base identities + the vendored corpus and start
/// the registry/seat HTTP fixture on a pre-chosen `port` (so nodes can be configured with the
/// base URL before the fixture binds).
pub async fn start_cluster_on(
    port: u16,
    run_label: &str,
    trusted_bases: &[daemon_vhc_proto::PeerId],
    epoch_rounds: u32,
    global_batch: u32,
) -> Cluster {
    let coordinator_wasm = guest_wasm("coordinator_quorum");
    let trainer_wasm = guest_wasm("tiny_llama");
    let corpus = corpus_dir();
    let genesis = live_genesis(&LiveGenesisSpec {
        run_label,
        coordinator_wasm: &coordinator_wasm,
        coordinator_url: format!("file://{}", guest_path("coordinator_quorum").display()),
        trainer_wasm: &trainer_wasm,
        trainer_url: format!("file://{}", guest_path("tiny_llama").display()),
        corpus_dir: &corpus,
        trusted_bases,
        min_peers: 2,
        max_peers: 4,
        epoch_rounds,
        global_batch,
        steps_per_round: 2,
        k_absences: 6,
    });

    let (registry, base_url, task) = registry::serve(&genesis, run_label, port).await;
    Cluster {
        registry,
        genesis,
        run_label: run_label.to_string(),
        base_url,
        _registry_task: task,
    }
}

/// Peer-verify a stored seat lease the way a trainer node does (used by the isolation +
/// negative gates to assert the seat surface behaves).
pub fn authorize_seat(
    registry: &FakeSeatRegistry,
    run: &str,
    role: &str,
    trusted: &[daemon_vhc_proto::PeerId],
    now_ms: u64,
) -> bool {
    match registry.read(run, role) {
        SeatState::Leased(lease) => lease
            .authorize(trusted, now_ms, daemon_vhc_proto::DEFAULT_SEAT_SKEW_MS)
            .is_ok(),
        SeatState::Unclaimed { .. } => false,
    }
}

/// A signing helper for the malformed-cert / seat negative fixtures.
pub fn key_from(seed: &str) -> SigningKey {
    SigningKey::from_bytes(blake3::hash(seed.as_bytes()).as_bytes())
}

/// The base peer id of a node.
pub fn base_peer(node: &Node) -> daemon_vhc_proto::PeerId {
    peer_id(&node.base_key)
}
