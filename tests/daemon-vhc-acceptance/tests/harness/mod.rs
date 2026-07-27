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

/// The suite's **development authority** (`[PC-12]`): the identity whose vouching every node's
/// provisioned profile carries, and which the genesis names in each role's
/// `accepted_development_authorities`. Pure data — nothing signs with it — and acceptance
/// requires BOTH sides to name it, which is exactly what this harness does. A development
/// authority satisfies integration evidence and can never certify a ceremony; that fence lives
/// in the authentication result, not here.
pub fn development_authority() -> daemon_vhc_proto::PeerId {
    daemon_vhc_proto::PeerId(*blake3::hash(b"vhc-acceptance/development-authority").as_bytes())
}

/// The run-side profile-certification policy this suite authors into every role: name the
/// development authority, defer everything else.
pub fn profile_certification() -> daemon_vhc_proto::ProfileCertificationRequirements {
    daemon_vhc_proto::ProfileCertificationRequirements {
        accepted_development_authorities: vec![development_authority()],
        ..Default::default()
    }
}

/// The worker binary's own CPU-lane revision record, exported once per suite process through the
/// worker's `DAEMON_TRAIN_REVISION_OUT` seam and cached.
///
/// The record must come from the binary the nodes will spawn — authentication compares the
/// sealed-binary identity (blake3+size of that very file) and the implementation revision against
/// what actually runs, so a record the harness derived itself would vouch for a worker nobody
/// executes.
fn worker_cpu_revision_record() -> daemon_vhc_resource::BackendImplementationRevision {
    use std::sync::OnceLock;
    static RECORD: OnceLock<daemon_vhc_resource::BackendImplementationRevision> = OnceLock::new();
    RECORD
        .get_or_init(|| {
            let worker = locate_bin("daemon-vhc-worker");
            let out_dir = tempfile::tempdir().expect("revision export dir");
            let status = Command::new(&worker)
                .env_clear()
                .env("PATH", std::env::var("PATH").unwrap_or_default())
                .env("DAEMON_TRAIN_REVISION_OUT", out_dir.path())
                .stdout(Stdio::null())
                .stderr(Stdio::inherit())
                .status()
                .expect("run the worker's revision export");
            assert!(status.success(), "worker revision export failed");
            let bytes = std::fs::read(out_dir.path().join("revision-cpu.cbor"))
                .expect("the worker exported its CPU-lane revision record");
            daemon_vhc_proto::from_canonical_slice(&bytes)
                .expect("the exported revision record decodes")
        })
        .clone()
}

/// Provision one node's profile home (`<data_dir>/vhc/profiles`) with the development-authority
/// profile set for the worker this suite spawns. Written BEFORE the node boots, exactly like the
/// pre-created identity keystore: the node hands the directory to its workers by path reference.
fn provision_dev_profiles(data_dir: &Path) {
    let set = daemon_vhc_resource::test_support::development_provisioned_profiles(
        &worker_cpu_revision_record(),
        development_authority(),
    );
    let dir = data_dir.join("vhc").join("profiles");
    daemon_vhc_resource::provision::write(&dir, &set).expect("write provisioned profiles");
}

/// Locate a sibling product binary in the cargo target profile dir (the xtask acceptance lane
/// builds `daemon` + `daemon-vhc-worker` before the suite runs). Walks up from the test
/// executable to the profile dir (the parent of `deps/`) and expects `<profile>/<name>`.
pub fn locate_bin(name: &str) -> PathBuf {
    // An explicit override (the xtask acceptance lane builds the node + worker in RELEASE — debug
    // wasmtime compilation of the ~1 MB trainer module can exceed the supervisor's assess
    // watchdog — and points the suite at them).
    if let Some(dir) = std::env::var_os("VHC_ACCEPTANCE_BIN_DIR") {
        let p = PathBuf::from(dir).join(name);
        if p.is_file() {
            return p;
        }
    }
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

/// A cross-process serialization guard: each heavy multi-process gate spawns three real node
/// processes doing burn compute, so running several gates concurrently (cargo runs test binaries
/// in parallel) oversubscribes the host and makes round timing flaky. Every gate takes this guard
/// first; it blocks (advisory lockfile) until any other gate finishes, so gates run one at a time
/// regardless of how the suite is invoked — deterministic timing, never a retry-to-green.
pub struct SerialGuard {
    path: PathBuf,
}

impl Drop for SerialGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Acquire the acceptance-suite serial guard (blocks until free; stale locks older than 30 min
/// are reclaimed so a killed gate never wedges the suite).
pub fn serial_guard() -> SerialGuard {
    let path = std::env::temp_dir().join("vhc-acceptance-serial.lock");
    loop {
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
        {
            Ok(mut f) => {
                use std::io::Write as _;
                let _ = writeln!(f, "{}", std::process::id());
                return SerialGuard { path };
            }
            Err(_) => {
                // Reclaim a stale lock (a gate that was killed before its guard dropped).
                if let Ok(meta) = std::fs::metadata(&path) {
                    if meta
                        .modified()
                        .ok()
                        .and_then(|m| m.elapsed().ok())
                        .is_some_and(|e| e > Duration::from_secs(1800))
                    {
                        let _ = std::fs::remove_file(&path);
                        continue;
                    }
                }
                std::thread::sleep(Duration::from_millis(250));
            }
        }
    }
}

/// A free loopback TCP port (bind :0, read the assignment, drop the listener).
pub fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .expect("bind ephemeral port")
        .local_addr()
        .expect("local addr")
        .port()
}

/// A free loopback **UDP** port — what an iroh endpoint binds. The colocation gates PIN this into
/// `[vhc.iroh] bind_port` exactly as the ceremony runbook pins a per-box port, so the gate
/// exercises the pinned-socket path (a node-picked `bind_port = 0` cannot collide and would hide
/// the co-located bind collision the fleet smoke hit).
pub fn free_udp_port() -> u16 {
    std::net::UdpSocket::bind("127.0.0.1:0")
        .expect("bind ephemeral udp port")
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
    for object in &genesis.corpus_objects {
        let hex = daemon_vhc_proto::Hash(object.id).to_hex();
        std::fs::write(dir.join(hex.as_str()), &object.bytes).expect("seed corpus object");
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

    /// The node process id.
    pub fn pid(&self) -> u32 {
        self.child.id()
    }

    /// Whether the node process is still alive (a typed refusal must never crash it).
    pub fn is_alive(&self) -> bool {
        nix::sys::signal::kill(nix::unistd::Pid::from_raw(self.child.id() as i32), None).is_ok()
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
    /// The first reconvergence backoff (ms); `0` = the node default. The hard-crash continuity
    /// gate raises it so the rejoin resolves its restore AFTER the survivor's post-deadline
    /// live checkpoint lands (deterministic restore freshness, never a race).
    pub initial_backoff_ms: u64,
}

/// Spawn a node process with a seeded base identity, blocking until it serves its socket.
pub fn spawn_node(spec: &NodeSpec<'_>) -> Node {
    spawn_node_with(spec, "")
}

/// [`spawn_node`] with extra `[vhc]`-scoped TOML appended to the node config (e.g. the
/// dual-plane gate's `[vhc.iroh]` table) — additive, so the shared spec shape stays untouched.
/// The owner budget defaults to the permissive `unbounded = true` posture (most gates are about
/// transport / restore / upgrade, not arbitration); a gate that must exercise the FINITE owner
/// ledgers the live node runs (the coordinator+trainer duty arbitration) uses
/// [`spawn_node_with_budget`] instead.
pub fn spawn_node_with(spec: &NodeSpec<'_>, extra_vhc_toml: &str) -> Node {
    spawn_node_with_budget(spec, extra_vhc_toml, "unbounded = true\n")
}

/// [`spawn_node_with`] with an explicit `[vhc.owner_budget]` body (the TOML that follows the
/// `[vhc.owner_budget]` header — e.g. `"duty_pct = 100\n"` for a finite ledger, or
/// `"unbounded = true\n"` for the permissive default). This is the seam that lets a gate run
/// against the SAME finite arbitration ledgers the shipped node derives, so a defect in owner
/// admission (e.g. a coordinator seat starving its co-located trainer on the duty ledger) is
/// caught by the gate instead of only on real hardware.
pub fn spawn_node_with_budget(
    spec: &NodeSpec<'_>,
    extra_vhc_toml: &str,
    owner_budget_body: &str,
) -> Node {
    let mut data_dir = tempfile::tempdir().expect("node data dir");
    // Forensics seam: `VHC_ACCEPTANCE_KEEP_STATE=1` leaves every node's state dir (journals,
    // payload plane, config) on disk after the test — a failing gate's evidence would otherwise
    // unwind with the tempdirs. The kept path is printed so a human (or the failing assert's
    // reader) can find it.
    if std::env::var_os("VHC_ACCEPTANCE_KEEP_STATE").is_some_and(|v| v == "1") {
        data_dir.disable_cleanup(true);
        eprintln!(
            "keep-state: node `{}` state dir kept at {}",
            spec.name,
            data_dir.path().display()
        );
    }
    let socket = data_dir.path().join("api.sock");
    // Pre-create the identity keystore so the harness can read the node's base pubkey BEFORE the
    // node boots (the genesis trust set names every node's base identity). The node opens the
    // same keystore at boot (idempotent).
    let identity_dir = data_dir.path().join("vhc").join("identity");
    std::fs::create_dir_all(&identity_dir).expect("identity dir");
    let keystore =
        daemon_vhc_session::keystore::VhcKeystore::open(&identity_dir).expect("open node keystore");
    let base_key = keystore.base_identity().expect("node base identity");
    // Provision the box (`[PC-12]`): the development-authority profile set the certification-minor
    // trainer composes its Physical Estimate against. Pre-created like the keystore; the node
    // hands `<data_dir>/vhc/profiles` to its workers by path reference.
    provision_dev_profiles(data_dir.path());

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
    // Additive `[vhc]`-scoped TOML injected here — BEFORE the sub-tables — so a caller-supplied
    // bare key (e.g. `coordinator_trains = true`) lands under `[vhc]`, and a caller-supplied
    // sub-table (e.g. `[vhc.iroh]`) simply precedes the ones below (TOML table order is
    // irrelevant). Appending it at the END would misfile a bare key under the last sub-table.
    if !extra_vhc_toml.is_empty() {
        toml.push_str(extra_vhc_toml);
        if !extra_vhc_toml.ends_with('\n') {
            toml.push('\n');
        }
    }
    toml.push_str("[vhc.owner_budget]\n");
    toml.push_str(owner_budget_body);
    if !owner_budget_body.ends_with('\n') {
        toml.push('\n');
    }
    toml.push_str("[vhc.registry]\n");
    writeln!(toml, "base = {:?}", spec.registry_base).unwrap();
    toml.push_str("[vhc.retry]\n");
    writeln!(toml, "reconcile_tick_ms = {}", spec.reconcile_tick_ms).unwrap();
    if spec.initial_backoff_ms > 0 {
        writeln!(toml, "initial_backoff_ms = {}", spec.initial_backoff_ms).unwrap();
    }
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
    let log_dir = std::path::PathBuf::from(
        std::env::var("VHC_ACCEPTANCE_LOG_DIR")
            .unwrap_or_else(|_| "/tmp/vhc-acceptance-logs".into()),
    );
    std::fs::create_dir_all(&log_dir).ok();
    let log = std::fs::File::create(log_dir.join(format!("{}.log", spec.name))).expect("node log");
    let log2 = log.try_clone().expect("clone node log");
    cmd.env("DAEMON_STORE", "memory")
        .env("DAEMON_DATA_DIR", data_dir.path())
        .env("DAEMON_SOCKET_PATH", &socket)
        .env("DAEMON_CONFIG", &config_path)
        // The training worker's compute lane needs no GPU; keep it CPU/ndarray.
        .env("DAEMON_VHC_LANE_GPU_OPTIONAL", "1")
        // The worker inherits the node's env (provisioner policy): selecting the CPU backend
        // EXPLICITLY also selects the roomy real-training engine budgets (fuel 1<<34 per slice,
        // 600s epoch) — the tiny-model defaults trap `BudgetFuel` on a real trainer's quiesce
        // slice (the drain snapshot's manifest/flatten work), losing the leave checkpoint.
        .env("DAEMON_TRAIN_BACKEND", "cpu")
        .env(
            "RUST_LOG",
            std::env::var("VHC_ACCEPTANCE_RUST_LOG").unwrap_or_else(|_| {
                "daemon_vhc_node=debug,daemon_vhc_session=debug,daemon_vhc_supervisor=debug,\
                 daemon_vhc_host=info,daemon_vhc_worker=debug,info"
                    .to_string()
            }),
        )
        .stdin(Stdio::null());
    if std::env::var_os("VHC_ACCEPTANCE_INHERIT_STDIO").is_some() {
        cmd.stdout(Stdio::inherit()).stderr(Stdio::inherit());
        drop((log, log2));
    } else {
        cmd.stdout(Stdio::from(log)).stderr(Stdio::from(log2));
    }
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

/// Seed the run's chunk-addressed corpus objects into the fixture's presigned OBJECT store (the
/// R2-compatible tier) **at the keys the run's own publisher writes** — the genesis artifact map's
/// url for each object, mapped to a bucket key by the one §11.3 layout function both the registry
/// and the client share (`daemon_vhc_net::r2_object_key`). The live trainer fetches them by content
/// hash via `data@2`; the pump verifies covering chunks (the store is untrusted).
///
/// **Why not the payload key.** This used to stage every corpus object under
/// `runs/<run>/payload/<hex>` — the committed-PAYLOAD plane's key ([RS-4]) — which is not where any
/// publisher puts a corpus object: `xtask publish-corpus` writes `corpus/<manifest blake3>.cbor`,
/// `corpus/<tokenizer blake3>.json` and `corpus/<fold>.bin` (ABI §12.7 [CC-7]). Staging at the
/// payload key made the suite agree with the runtime about a layout NOBODY publishes, so the whole
/// module-driven corpus plane passed green while being unreachable on a real fleet box: the first
/// trainer to get past init died on a typed `payload miss` fetching its genesis-pinned corpus
/// manifest.
///
/// **What this lane does NOT prove.** The staging key is derived from the genesis url under test,
/// so this lane agrees with whatever genesis it is handed — it is a gate on the *runtime* resolving
/// a pinned id at the *committed* url, not on that url being a key any publisher writes. It is
/// honest only because `live_genesis` derives its urls from `daemon_vhc_net::PublishedArtifact`,
/// the one definition `publish-corpus` also writes at. The self-consistency was load-bearing once:
/// while this lane was green, the FROZEN `ceremony_genesis` — the only genesis a fleet box runs —
/// pinned an extensionless `corpus/<hash>` that nothing published, and both fleet trainers took a
/// 404. The gate that forces the publisher and the ceremony authoring to agree by construction is
/// `xtask` `genesis_pinned_urls_resolve_against_the_publishers_own_keys`, which stages with the
/// real publisher instead of with the artifact under test.
pub fn seed_corpus_r2(cluster: &Cluster, run_label: &str, genesis: &LiveGenesis) {
    for object in &genesis.corpus_objects {
        cluster.registry.put_object(
            &published_object_key(run_label, &object.url),
            object.bytes.clone(),
        );
    }
}

/// The bucket key one genesis-pinned `r2://<path>` url resolves to for `run_label`, through the
/// same §11.3 layout the presign surface mints (`runs/<run>/<path>`).
fn published_object_key(run_label: &str, url: &str) -> String {
    let path = url
        .strip_prefix("r2://")
        .unwrap_or_else(|| panic!("a published corpus artifact must carry an r2:// url, got {url}"))
        .trim_start_matches('/');
    daemon_vhc_net::r2_object_key(
        &daemon_vhc_net::RunId::new(run_label),
        &daemon_vhc_net::PresignRequest::artifact(daemon_vhc_net::PresignOp::Get, path),
    )
    .expect("artifact presign requests carry a path")
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

/// Read a run's full `VhcRunDetail` snapshot (`None` if the node does not know the run).
pub async fn run_detail(node: &Node, run_id: &str) -> Option<daemon_api::VhcRunDetail> {
    let resp = node
        .client()
        .call(ApiRequest::VhcRunDetail {
            run_id: run_id.to_string(),
        })
        .await
        .expect("vhc_run_detail call");
    match resp {
        ApiResponse::VhcRunDetail(d) => d,
        _ => None,
    }
}

/// Assert the wire-v44 per-round digest surfaces on the product API consistently with the durable
/// journal (the offline truth): any `VhcEvent::RoundOutcome` in the snapshot window carries the
/// journal's digest for its round, and `last_round_digest` (when present) equals the journal's
/// digest for the round it reports. This is the G-2 digest-agreement evidence collected through the
/// node's public API rather than a journal reader.
///
/// The digest reaches the API through the OPACITY-SAFE live producer: the trainer guest reports
/// its per-round outcome (digest + committed/ingested/stalled) over the reserved `round_metrics`
/// metric plane, the role session folds the reserved metric group into a `RoundOutcome` session
/// event (decoding no module frame), and the node projects it — the conversion no longer drops the
/// digest, and the snapshot carries `last_round_digest`. See
/// [`assert_api_digests_cover_and_match_journal`] for the stronger progression assertion.
pub fn assert_api_digest_matches_journal(
    detail: &daemon_api::VhcRunDetail,
    journal: &BTreeMap<u64, [u8; 16]>,
) {
    for ev in &detail.recent_events {
        if let VhcEvent::RoundOutcome { round, digest, .. } = ev {
            if let Some(j) = journal.get(round) {
                assert_eq!(
                    digest, j,
                    "RoundOutcome API digest for round {round} disagrees with the journal"
                );
            }
        }
    }
    if let Some(d) = &detail.last_round_digest {
        // The snapshot pointer is the digest of the HIGHEST-round `RoundOutcome` in the window
        // (the node's projection) — NOT `summary.last_round`, which tracks the phase round. Pair
        // it with the max event round so the journal cross-check is over the right round.
        let round = detail
            .recent_events
            .iter()
            .filter_map(|ev| match ev {
                VhcEvent::RoundOutcome { round, .. } => Some(*round),
                _ => None,
            })
            .max();
        if let Some(round) = round {
            if let Some(j) = journal.get(&round) {
                assert_eq!(
                    d, j,
                    "VhcRunDetail.last_round_digest for round {round} disagrees with the journal"
                );
            }
        }
    }
}

/// Poll a node's `vhc_run_detail` (a PRODUCT API — no pump access, no journal read, no SDK schema
/// decode) and accumulate the per-round det digests it surfaces, until digests for at least
/// `want_rounds` distinct rounds have been observed or the deadline elapses. Returns the collected
/// `round → digest` map (product-API-sourced).
///
/// Both API surfaces are folded: every `VhcEvent::RoundOutcome` in the windowed event stream AND
/// the `last_round_digest`/`summary.last_round` progression — so a `--watch`-style client that only
/// sampled the snapshot pointer, and one that read the event feed, both collect the same evidence.
pub async fn collect_api_digests(
    node: &Node,
    run_id: &str,
    want_rounds: u64,
    timeout: Duration,
) -> BTreeMap<u64, [u8; 16]> {
    let deadline = Instant::now() + timeout;
    let mut digests: BTreeMap<u64, [u8; 16]> = BTreeMap::new();
    loop {
        if let Some(detail) = run_detail(node, run_id).await {
            let mut max_round: Option<u64> = None;
            for ev in &detail.recent_events {
                if let VhcEvent::Error { class, detail, .. } = ev {
                    panic!(
                        "node `{}` run `{run_id}` errored: {class}: {detail}",
                        node.name
                    );
                }
                if let VhcEvent::RoundOutcome { round, digest, .. } = ev {
                    // The event carries its OWN round — the authoritative key.
                    digests.insert(*round, *digest);
                    max_round = Some(max_round.map_or(*round, |m| m.max(*round)));
                }
            }
            // The `last_round_digest` snapshot pointer is the node's projection of the
            // HIGHEST-round `RoundOutcome` in the window (service.rs); exercise that surface too
            // and cross-check it agrees with the event we recorded for that round (never keyed by
            // `summary.last_round`, which tracks the phase round, not the digest's round).
            if let (Some(d), Some(mr)) = (detail.last_round_digest, max_round) {
                assert_eq!(
                    digests.get(&mr),
                    Some(&d),
                    "VhcRunDetail.last_round_digest disagrees with the max-round RoundOutcome event"
                );
                digests.insert(mr, d);
            }
        }
        if digests.keys().filter(|r| **r < want_rounds).count() as u64 >= want_rounds {
            return digests;
        }
        if Instant::now() >= deadline {
            panic!(
                "node `{}` run `{run_id}`: the product API surfaced digests for only {:?} \
                 (< {want_rounds} rounds) within {timeout:?}",
                node.name,
                digests.keys().collect::<Vec<_>>()
            );
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
}

/// The strengthened G-2 assertion: EVERY completed round in `0..want_rounds` has a digest that is
/// observable via the product API (`api`, collected by [`collect_api_digests`]) and is byte-equal
/// to the offline journal oracle (`journal`) for that round. Proves the live opacity-safe producer
/// actually surfaces each round's digest — not merely that any present digest happens to agree.
pub fn assert_api_digests_cover_and_match_journal(
    api: &BTreeMap<u64, [u8; 16]>,
    journal: &BTreeMap<u64, [u8; 16]>,
    want_rounds: u64,
) {
    for round in 0..want_rounds {
        let observed = api.get(&round).unwrap_or_else(|| {
            panic!(
                "the product API surfaced NO digest for completed round {round} (observed rounds: \
                 {:?}) — the live per-round digest producer is not driving the API",
                api.keys().collect::<Vec<_>>()
            )
        });
        let truth = journal.get(&round).unwrap_or_else(|| {
            panic!("the journal oracle has no digest for round {round} (cannot cross-check)")
        });
        assert_eq!(
            observed, truth,
            "product-API digest for round {round} disagrees with the journal oracle"
        );
    }
}

/// Poll until `node`'s run has progressed to at least `rounds` distinct round outcomes, or the
/// deadline elapses. Returns the highest round observed. A persisted `VhcEvent::Error` fails the
/// wait loudly (never a silent drop).
pub async fn wait_rounds(node: &Node, run_id: &str, rounds: u64, timeout: Duration) -> u64 {
    let deadline = Instant::now() + timeout;
    loop {
        for ev in recent_events(node, run_id).await {
            if let VhcEvent::Error { class, detail, .. } = ev {
                panic!(
                    "node `{}` run `{run_id}` errored: {class}: {detail}",
                    node.name
                )
            }
        }
        // Progress is read from the durable journal (the product-path oracle): the highest
        // round this node voiced a tag-4 det digest for. The node's `RoundOutcome` app event is
        // not emitted on the opaque live path, so the journal is the truth.
        let best = journal_digests(node, run_id)
            .keys()
            .last()
            .map_or(0, |r| r + 1);
        if best >= rounds {
            return best;
        }
        if Instant::now() >= deadline {
            panic!(
                "node `{}` run `{run_id}` reached only {best} rounds (< {rounds}) within {timeout:?}",
                node.name
            );
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
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
    start_cluster_with(
        port,
        run_label,
        trusted_bases,
        epoch_rounds,
        global_batch,
        6,
        daemon_vhc_testkit::live_genesis::LiveTiming::default(),
    )
    .await
}

/// [`start_cluster_on`] with churn knobs: the absence budget (`k_absences`) and the coordinator's
/// liveness timing (a churn tier arms the real timer so a vanished member is survivable).
pub async fn start_cluster_with(
    port: u16,
    run_label: &str,
    trusted_bases: &[daemon_vhc_proto::PeerId],
    epoch_rounds: u32,
    global_batch: u32,
    k_absences: u32,
    timing: daemon_vhc_testkit::live_genesis::LiveTiming,
) -> Cluster {
    let coordinator_wasm = guest_wasm("coordinator_quorum");
    let trainer_wasm = guest_wasm("tiny_llama");
    let corpus = corpus_dir();
    let steps_per_round = 2;
    // Derived from the real guests this cluster runs — over the same corpus manifest and
    // steps-per-round the genesis pins — so the envelope carries what those modules' own
    // assessment said rather than a stand-in.
    let execution = daemon_vhc_testkit::live_genesis::live_execution_with_certification(
        &coordinator_wasm,
        &trainer_wasm,
        &corpus,
        steps_per_round,
        // The run names the suite's development authority (`[PC-12]`): acceptance requires BOTH
        // the owner policy (the provisioned file) and the run to opt in.
        profile_certification(),
    )
    .expect("derive the live cluster execution requirements");
    let genesis = live_genesis(&LiveGenesisSpec {
        execution,
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
        steps_per_round,
        k_absences,
        timing,
        upgrade_authority: vec![peer_id(&upgrade_authority_key())],
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

/// [`start_cluster_on`] with an explicit membership floor/ceiling — used by the single-node
/// coordinator+trainer gate (min=max=1: the one box is its own coordinator AND its only trainer).
pub async fn start_cluster_membership(
    port: u16,
    run_label: &str,
    trusted_bases: &[daemon_vhc_proto::PeerId],
    epoch_rounds: u32,
    global_batch: u32,
    min_peers: u32,
    max_peers: u32,
) -> Cluster {
    let coordinator_wasm = guest_wasm("coordinator_quorum");
    let trainer_wasm = guest_wasm("tiny_llama");
    let corpus = corpus_dir();
    let steps_per_round = 2;
    // Derived from the real guests this cluster runs — over the same corpus manifest and
    // steps-per-round the genesis pins — so the envelope carries what those modules' own
    // assessment said rather than a stand-in.
    let execution = daemon_vhc_testkit::live_genesis::live_execution_with_certification(
        &coordinator_wasm,
        &trainer_wasm,
        &corpus,
        steps_per_round,
        // The run names the suite's development authority (`[PC-12]`): acceptance requires BOTH
        // the owner policy (the provisioned file) and the run to opt in.
        profile_certification(),
    )
    .expect("derive the live cluster execution requirements");
    let genesis = live_genesis(&LiveGenesisSpec {
        execution,
        run_label,
        coordinator_wasm: &coordinator_wasm,
        coordinator_url: format!("file://{}", guest_path("coordinator_quorum").display()),
        trainer_wasm: &trainer_wasm,
        trainer_url: format!("file://{}", guest_path("tiny_llama").display()),
        corpus_dir: &corpus,
        trusted_bases,
        min_peers,
        max_peers,
        epoch_rounds,
        global_batch,
        steps_per_round,
        k_absences: 6,
        timing: daemon_vhc_testkit::live_genesis::LiveTiming::default(),
        upgrade_authority: vec![peer_id(&upgrade_authority_key())],
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

/// The suite's run-level upgrade authority signing key: every acceptance genesis names its
/// public half as the (single-key, hence unanimous) upgrade authority, so the live-switch gate
/// can author an authorized module-upgrade record the product path validates fail-closed.
pub fn upgrade_authority_key() -> SigningKey {
    key_from("acceptance/upgrade-authority")
}

/// The base peer id of a node.
pub fn base_peer(node: &Node) -> daemon_vhc_proto::PeerId {
    peer_id(&node.base_key)
}

/// Whether a spawned node's captured log (node stderr + the worker child's inherited stderr)
/// contains `needle` — the black-box observability seat for transport-formation markers the
/// dual-plane gate asserts (e.g. the iroh `NeighborUp` marker: gossip only logs it when a real
/// QUIC connection formed).
pub fn node_log_contains(name: &str, needle: &str) -> bool {
    let log_dir = std::path::PathBuf::from(
        std::env::var("VHC_ACCEPTANCE_LOG_DIR")
            .unwrap_or_else(|_| "/tmp/vhc-acceptance-logs".into()),
    );
    std::fs::read_to_string(log_dir.join(format!("{name}.log")))
        .map(|s| s.contains(needle))
        .unwrap_or(false)
}

/// Count a node's live `daemon-vhc-worker` children (via `/proc`) — the process-isolation +
/// churn checks read this without reaching any product internal.
pub fn worker_children(node: &Node) -> Vec<u32> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir("/proc") else {
        return out;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(pid) = name.to_str().and_then(|s| s.parse::<u32>().ok()) else {
            continue;
        };
        let status = std::fs::read_to_string(format!("/proc/{pid}/status")).unwrap_or_default();
        let comm = std::fs::read_to_string(format!("/proc/{pid}/comm")).unwrap_or_default();
        if !comm.contains("daemon-vhc-work") {
            continue;
        }
        // PPid: <node pid> — a child of this node.
        if status
            .lines()
            .find_map(|l| {
                l.strip_prefix("PPid:")
                    .map(|v| v.trim().parse::<u32>().ok())
            })
            .flatten()
            == Some(node.pid())
        {
            out.push(pid);
        }
    }
    out
}

/// A malformed control-plane distribution record: a `RunKeyCertificate` issued by an UNTRUSTED
/// base (a random key absent from the genesis identities), wrapped as a §12.3 distribution record.
/// A conforming node refuses it typed (the base is not genesis-trusted) — never a panic.
pub fn untrusted_cert_record(genesis_hash: [u8; 32]) -> Vec<u8> {
    use daemon_vhc_proto::{CertScope, Hash, RunKeyCertificate};
    let rogue_base = key_from("acceptance/rogue-base");
    let rogue_run_key = peer_id(&key_from("acceptance/rogue-run-key"));
    let scope = CertScope {
        run_id: Hash(genesis_hash),
        epoch: 0,
        role: "coordinator".to_string(),
        instance: 0,
        module_hash: Hash([0u8; 32]),
    };
    let cert = RunKeyCertificate::issue(&rogue_base, scope, rogue_run_key)
        .expect("issue a (chain-valid but untrusted-base) certificate");
    daemon_vhc_session::distribution::DistributionRecord::Cert(cert)
        .to_bytes()
        .expect("encode distribution record")
}
