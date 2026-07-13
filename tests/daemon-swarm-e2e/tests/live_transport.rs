// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
//! The **live-transport exit gate** (spec §6.4, §7.1; TDD §3.8, RUN-5/8, PROTO-20 over a real
//! plane; B3). Every test here drives the frozen `RoundEngine` + the real `daemon-swarm-coordinator`
//! `tick` loop over a **real per-node `IrohGossip` mesh** (QUIC gossip on loopback, explicit roster
//! addressing, no public discovery) + a shared `FsPayloadStore` — the transports swapped behind the
//! frozen `ControlPlane`/`PayloadStore` traits, nothing else changed. This is where B2's
//! plane-level partition/rejoin proof becomes an **engine-level** recovery proof: the stall ladder,
//! late-join resync, and mid-run drop all recover over the real gossip mesh.
//!
//! Gated behind the `iroh` feature so the default e2e gate stays fast + iroh-free; run with
//! `cargo test -p daemon-swarm-e2e --features iroh`. The relay variant additionally shells the
//! devShell `iroh-relay --dev` binary and skips cleanly when it is absent.
#![cfg(feature = "iroh")]
// The relay-path test shells the dev relay binary (a known dev tool from a test — the sanctioned
// exception to the workspace `Command` ban, mirroring B2's `spawn_dev_relay`).
#![allow(clippy::disallowed_methods)]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Once;
use std::time::Duration;

use ciborium::into_writer;
use daemon_swarm_proto::peer_id;
use daemon_swarm_run::backend::{
    AssessMeta, Assessment, BatchRef, StagedPayload, StateDigest, StepCtx, StepStats,
    TrainerBackend,
};
use daemon_swarm_run::engine::EngineEvent;
use daemon_swarm_run::harness::{peer_key, LateJoin, SilentDeath, StallFault, SwarmConfig};
use daemon_swarm_run::live_harness::{run_live_swarm, run_live_swarm_with, LiveSwarmConfig};
use daemon_swarm_run::seam::RoundId;
use daemon_train::{EngineConfig, WasmBackend, WasmBackendConfig, WasmBackendError};
use daemon_train_sdk::models::TinyLlamaCfg;

/// Assert every round in `run` has a single digest shared by all peers that reported it.
fn assert_all_agree(run: &daemon_swarm_run::harness::SwarmRun) {
    assert!(
        run.all_agree(),
        "surviving peers must agree on every round's digest over the live plane"
    );
}

// ---- flagship: 3 peers × ≥10 rounds, real iroh mesh, byte-identical per-round digests -----------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn live_flagship_three_peers_ten_rounds_all_agree() {
    // 3 peers form a real iroh gossip mesh on loopback, run 12 rounds over a shared FS store with
    // the concurrent barrier fetch enabled, agree on every round's post-ingest digest, and the run
    // terminates on the envelope stop condition (`Rounds(12)`).
    let cfg = LiveSwarmConfig::new(SwarmConfig {
        num_peers: 3,
        num_rounds: 12,
        ..SwarmConfig::small(12)
    });
    let run = run_live_swarm(cfg).await.expect("live flagship run");

    assert!(run.left_peers().is_empty(), "no peer should leave");
    assert_all_agree(&run);

    let by_round = run.digests_by_round();
    assert_eq!(
        by_round.len(),
        12,
        "12 rounds ingested over the live mesh (stop condition honored)"
    );
    for (round, peers) in &by_round {
        assert_eq!(peers.len(), 3, "all 3 peers report round {round}");
    }

    // PROTO-20 over a live-transport run: the coordinator's tick trajectory replays byte-identically
    // from its recorded inputs.
    let replay = run.replay.as_ref().expect("coordinator replay captured");
    assert_eq!(replay.recorded_rounds(), 12, "a state snapshot per round");
    assert!(
        replay.verify(),
        "the live-run tick trajectory replays byte-identically"
    );
}

// ---- stall ladder over the real plane (RUN-8, engine-level recovery) ----------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn live_stall_ladder_recovers_over_iroh() {
    // Peer 1 cannot fetch peer 0's round-5 payload for its first 2 gets (prefetch + barrier), stalls
    // over the real mesh, and catches it up on a later round — the ENGINE-level recovery of the
    // gossip-delivery gap B2 proved at the plane level. Concurrent fetch OFF so the injected-miss
    // count matches the barrier semantics.
    let cfg = LiveSwarmConfig::new(SwarmConfig {
        num_peers: 3,
        num_rounds: 12,
        stall_rounds_max: 3,
        fault: Some(StallFault {
            peer_index: 1,
            missing_peer_index: 0,
            round: 5,
            first_n_gets: 2,
        }),
        ..SwarmConfig::small(12)
    })
    .with_concurrent_fetch(false);
    let run = run_live_swarm(cfg).await.expect("live stall run");

    assert!(
        run.left_peers().is_empty(),
        "the stall ladder absorbs the miss within budget"
    );
    assert_all_agree(&run);

    let stalled = peer_id(&peer_key(1));
    let mine: Vec<&EngineEvent> = run
        .events
        .iter()
        .filter(|(p, _)| *p == stalled)
        .map(|(_, e)| e)
        .collect();
    assert!(
        mine.iter()
            .any(|e| matches!(e, EngineEvent::Straggling { round: 5, .. })),
        "peer 1 straggles round 5 over the live plane"
    );
    assert!(
        mine.iter()
            .any(|e| matches!(e, EngineEvent::CaughtUp { round: 5, .. })),
        "peer 1 catches round 5 up over the live plane"
    );
    assert_eq!(
        run.digests_by_round()[&5].len(),
        3,
        "all 3 peers report round 5"
    );
}

// ---- late-join resync over the real plane (§6.4 admission + §9 rejoin) ---------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn live_late_join_resyncs_over_iroh() {
    // Epoch 0 = rounds 0..2 (3 peers); a 4th peer joins epoch 1 through the real admission path over
    // the live mesh, resyncs from the round-2 checkpoint (pulled from the shared store), and
    // contributes rounds 3..5 with consensus digests.
    let cfg = LiveSwarmConfig::new(SwarmConfig {
        num_peers: 3,
        num_rounds: 6,
        steps_per_round: 1,
        micro_batch: 1,
        epoch_rounds: 3,
        checkpoint_every_rounds: 3, // checkpoint at round 2 (and 5)
        min_peers: Some(3),
        late_join: Some(LateJoin { resume_round: 2 }),
        ..SwarmConfig::small(6)
    });
    let run = run_live_swarm(cfg).await.expect("live late-join run");

    assert_all_agree(&run);
    assert!(run.left_peers().is_empty(), "no peer leaves");

    let late = peer_id(&peer_key(3));
    let by_round = run.digests_by_round();
    for round in 3..6 {
        let peers = by_round.get(&round).expect("round reported");
        assert!(
            peers.contains_key(&late),
            "the late peer reports round {round}"
        );
        assert_eq!(peers.len(), 4, "all 4 peers report round {round}");
    }
    let quorum = run.quorum_digests();
    for round in 3..6 {
        assert_eq!(
            by_round[&round][&late], quorum[&round],
            "late peer's round-{round} digest matches consensus (resync correct)"
        );
    }
}

// ---- mid-run drop over the real plane (§6.4, hard peer death) ------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn live_mid_run_drop_dropped_after_absences() {
    // Peer 2 goes silent after round 2 (its engine stops publishing over its iroh node). With
    // k_absences = 2 the coordinator drops it; min_peers = 1 keeps the run alive on peers 0 and 1.
    let cfg = LiveSwarmConfig::new(SwarmConfig {
        num_peers: 3,
        num_rounds: 8,
        min_peers: Some(1),
        k_absences: 2,
        silent_death: Some(SilentDeath {
            peer_index: 2,
            after_round: 2,
        }),
        ..SwarmConfig::small(8)
    });
    let run = run_live_swarm(cfg).await.expect("live drop run");

    assert_all_agree(&run);
    let dead = peer_id(&peer_key(2));
    assert!(
        run.dropped_peers().contains(&dead),
        "the silent peer is dropped after K record-absences, got {:?}",
        run.dropped_peers()
    );
    let by_round = run.digests_by_round();
    let last = &by_round[&7];
    assert_eq!(last.len(), 2, "two survivors report the last round");
    assert!(
        !last.contains_key(&dead),
        "the dropped peer contributes nothing past its death"
    );
}

// ---- relay variant: route through B2's self-hosted dev relay (skip-clean if absent) --------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn live_run_through_self_hosted_relay() {
    // Spawn the devShell `iroh-relay --dev` (plain HTTP, default port 3340 — B2's dev runner) and
    // run a small flagship with the nodes configured for `RelayMode::Custom(<that relay>)`. Skips
    // cleanly when the binary is not on PATH (a standalone checkout without the devShell). On
    // loopback the direct path dominates (B2 finding 1), so the run is robust even if the relay port
    // is busy — the value here is exercising the relay-configured construction end to end.
    let Some(mut relay) = spawn_dev_relay() else {
        eprintln!("SKIP live_run_through_self_hosted_relay: iroh-relay not on PATH");
        return;
    };
    // Give the relay a moment to bind its HTTP listener.
    tokio::time::sleep(Duration::from_millis(800)).await;

    let cfg = LiveSwarmConfig::new(SwarmConfig {
        num_peers: 3,
        num_rounds: 6,
        ..SwarmConfig::small(6)
    })
    .with_relay("http://127.0.0.1:3340");
    let result = run_live_swarm(cfg).await;

    let _ = relay.kill();
    let run = result.expect("live relay run");
    assert_all_agree(&run);
    assert_eq!(
        run.digests_by_round().len(),
        6,
        "6 rounds ingested through the relay-configured mesh"
    );
}

/// Spawn `iroh-relay --dev` (plain HTTP, default port 3340), returning the child (or `None` if the
/// binary is absent — the skip-clean path for a standalone checkout).
fn spawn_dev_relay() -> Option<std::process::Child> {
    std::process::Command::new("iroh-relay")
        .args(["--dev"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .ok()
}

// ---- flagship on the REAL tiny-llama guest: WasmBackend peers over the live iroh mesh -----------

// -- guest module loading (mirrors tests/wasm_profiles.rs / daemon-train/tests) --

fn guests_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../guests")
        .canonicalize()
        .expect("guests workspace path")
}

fn guest_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("SWARM_TEST_GUEST_DIR") {
        return PathBuf::from(dir);
    }
    guests_root().join("target/wasm32-unknown-unknown/release")
}

/// RUSTFLAGS that make the guest `.wasm` byte-reproducible across checkouts/machines by remapping the
/// absolute prefixes rustc embeds in panic locations (the `<checkout>` root + the cargo registry).
/// MUST match `xtask build-guests` (`guest_remap_rustflags`) so a local rebuild reproduces the bytes
/// recorded in the committed `guests/guests.blake3`.
fn guest_remap_rustflags() -> String {
    let root = guests_root();
    let checkout = root.parent().unwrap_or(&root).to_path_buf();
    let cargo_home = std::env::var_os("CARGO_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".cargo"));
    format!(
        "--remap-path-prefix={}=/daemon-node --remap-path-prefix={}=/cargo",
        checkout.display(),
        cargo_home.display(),
    )
}

/// Stale-guest guard (Merge-1 adjudication): compare every module named in the committed
/// `guests/guests.blake3` against the `.wasm` in `dir`. A **missing / unreadable** module still
/// fails loud — a genuinely absent or stale guest would otherwise surface downstream as a NaN loss,
/// which is the failure this guard exists to prevent. A **hash mismatch**, by contrast, only WARNS:
/// the guest `.wasm` is byte-reproducible run-to-run within one checkout but NOT across worktrees /
/// machines. cargo derives each path-package's crate-disambiguator (`-C metadata`) from its absolute
/// manifest dir, and `--remap-path-prefix` does not rewrite that hash, so symbol-hash-ordered codegen
/// reorders the module's code/type/func/elem sections between worktrees (the remapped path *strings*
/// are identical; only the ordering shifts). The committed manifest is therefore an advisory record
/// of one canonical (trunk) build, NOT a cross-machine identity gate — see the Merge-1 decision in
/// `docs/specs/swarm-p2-ledger.md`. Callers rebuild before loading, so the module in use is fresh.
fn verify_guest_manifest(dir: &Path) {
    let manifest = guests_root().join("guests.blake3");
    let text = std::fs::read_to_string(&manifest).unwrap_or_else(|e| {
        panic!(
            "read guest manifest {}: {e} — run `cargo run -p xtask -- build-guests`",
            manifest.display()
        )
    });
    for line in text.lines().map(str::trim).filter(|l| !l.is_empty()) {
        let (hex, name) = line
            .split_once("  ")
            .expect("guests.blake3 line must be `<blake3-hex>  <name>.wasm`");
        let bytes = std::fs::read(dir.join(name))
            .unwrap_or_else(|e| panic!("read guest module {}/{name}: {e}", dir.display()));
        let got = blake3::hash(&bytes).to_hex();
        if got.as_str() != hex {
            eprintln!(
                "warning: guest `{name}` in {} hashes {got} but committed guests.blake3 records \
                 {hex}. This is expected across worktrees/machines (path-keyed codegen ordering, \
                 not a stale artifact); the freshly-built module is used. If you changed guest \
                 source, run `cargo run -p xtask -- build-guests` and commit guests/guests.blake3.",
                dir.display()
            );
        }
    }
}

static BUILD: Once = Once::new();

fn ensure_built() {
    BUILD.call_once(|| {
        if std::env::var("SWARM_TEST_GUEST_DIR").is_ok() {
            verify_guest_manifest(&guest_dir());
            return;
        }
        let status = Command::new("cargo")
            .current_dir(guests_root())
            // Clear the devShell's `CARGO_TARGET_DIR` (pinned to the parent checkout) so the guests
            // build into their own `guests/target/` where `guest_dir()` reads them, and remap the
            // absolute source/registry prefixes so the built `.wasm` bytes stay byte-reproducible
            // (matching the committed `guests.blake3` the stale-guest guard asserts).
            .env_remove("CARGO_TARGET_DIR")
            .env("RUSTFLAGS", guest_remap_rustflags())
            .args(["build", "--release", "--target", "wasm32-unknown-unknown"])
            .status()
            .expect("run cargo for guests (dev shell provides the wasm target)");
        assert!(status.success(), "building guest modules failed");
        verify_guest_manifest(&guest_dir());
    });
}

fn tiny_llama_wasm() -> Vec<u8> {
    let path = guest_dir().join("tiny_llama.wasm");
    if !path.exists() {
        ensure_built();
    }
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// The live-harness corpus is `Corpus::synthetic` with `seq_len = 8`; the guest is built for
/// `seq_len = 9` (predicts positions `1..9` from `0..8`) over a small vocab.
const GUEST_VOCAB: u32 = 64;

fn tiny_cfg_cbor() -> Vec<u8> {
    let cfg = TinyLlamaCfg {
        n_layers: 1,
        seq_len: 9, // corpus seq_len (8) + 1
        vocab: GUEST_VOCAB,
        profile: "sparse_loco".to_string(),
        ..TinyLlamaCfg::default()
    };
    let mut b = Vec::new();
    into_writer(&cfg, &mut b).expect("cbor");
    b
}

/// A [`TrainerBackend`] adapter over the real tiny-llama [`WasmBackend`] that (1) maps the synthetic
/// corpus's u16 token ids (`< 32000`) into the tiny guest's vocabulary (`token % GUEST_VOCAB`)
/// before each `train_step` — deterministic and identical across peers, so the consensus digests
/// are unaffected (a test-side stand-in for tokenizing the corpus at this vocab) — and (2) wraps
/// the `Send`-only `WasmBackend` in a `Mutex` to satisfy the harness's `Sync` bound (the engine
/// owns its backend exclusively, so the lock is uncontended by construction).
struct VocabClampBackend {
    inner: std::sync::Mutex<WasmBackend>,
}

impl VocabClampBackend {
    fn lock(&self) -> std::sync::MutexGuard<'_, WasmBackend> {
        self.inner.lock().expect("wasm backend lock")
    }
}

impl TrainerBackend for VocabClampBackend {
    type Error = WasmBackendError;

    fn build(&mut self, config: &[u8]) -> Result<(), Self::Error> {
        self.lock().build(config)
    }
    fn assess(&self, meta: &AssessMeta) -> Result<Assessment, Self::Error> {
        self.lock().assess(meta)
    }
    fn train_step(&mut self, batch: &BatchRef, ctx: StepCtx) -> Result<StepStats, Self::Error> {
        let clamped = BatchRef {
            tokens: batch.tokens.iter().map(|t| t % GUEST_VOCAB).collect(),
            seq_len: batch.seq_len,
        };
        self.lock().train_step(&clamped, ctx)
    }
    fn inner_update(&mut self, inner_step: u32) -> Result<(), Self::Error> {
        self.lock().inner_update(inner_step)
    }
    fn make_update(&mut self, round: RoundId) -> Result<Vec<u8>, Self::Error> {
        self.lock().make_update(round)
    }
    fn ingest(
        &mut self,
        round: RoundId,
        staged: &[StagedPayload],
    ) -> Result<StateDigest, Self::Error> {
        self.lock().ingest(round, staged)
    }
    fn checkpoint_save(&self) -> Result<Vec<u8>, Self::Error> {
        self.lock().checkpoint_save()
    }
    fn checkpoint_load(&mut self, bytes: &[u8]) -> Result<(), Self::Error> {
        self.lock().checkpoint_load(bytes)
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn live_flagship_tiny_llama_wasm_over_iroh() {
    // THE transport exit-gate flagship (TDD §3.8): 3 peers running the REAL tiny-llama guest
    // (`WasmBackend` — wasmtime host training) drive N≥10 rounds over the real iroh mesh + the
    // shared payload store, and every round's post-ingest det digest is byte-identical across
    // peers; the replay verification is green over the live transport's message log.
    let wasm = tiny_llama_wasm();
    let config = tiny_cfg_cbor();

    let cfg = LiveSwarmConfig::new(SwarmConfig {
        num_peers: 3,
        num_rounds: 10,
        steps_per_round: 2,
        micro_batch: 2,
        ..SwarmConfig::small(10)
    });
    let run = run_live_swarm_with(cfg, |_i| {
        let mut inner = WasmBackend::new(WasmBackendConfig {
            wasm: wasm.clone(),
            engine: EngineConfig::default(),
        })
        .expect("construct WasmBackend");
        inner.build(&config).expect("da_build");
        VocabClampBackend {
            inner: std::sync::Mutex::new(inner),
        }
    })
    .await
    .expect("tiny-llama live run");

    assert!(run.left_peers().is_empty(), "no peer should leave");
    assert_all_agree(&run);

    let by_round = run.digests_by_round();
    assert_eq!(
        by_round.len(),
        10,
        "10 wasm-backed rounds over the live mesh"
    );
    for (round, peers) in &by_round {
        assert_eq!(peers.len(), 3, "all 3 peers report round {round}");
    }
    // The digest transcript evolves (the swarm is learning, not a vacuous constant).
    let transcript: Vec<_> = run.agreed_transcript().into_values().collect();
    assert!(
        transcript.windows(2).any(|w| w[0] != w[1]),
        "the tiny-llama digest transcript must evolve across rounds"
    );

    // PROTO-20 from the live-transport run's message log.
    let replay = run.replay.as_ref().expect("coordinator replay captured");
    assert_eq!(replay.recorded_rounds(), 10);
    assert!(
        replay.verify(),
        "tiny-llama live-run replay is byte-identical"
    );
}
