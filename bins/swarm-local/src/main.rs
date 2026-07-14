// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `swarm-local` — the runnable local-mode swarm runner (spec §10.4 local run mode, §6.1).
//!
//! It reads an authoring TOML envelope, builds the resolved [`Envelope`], **freezes + verifies** it
//! (the real §6.1 chain), and stands up an in-process run:
//!
//! - `--backend stub` (default): N [`RoundEngine`](daemon_swarm_run::engine::RoundEngine) peers over
//!   the deterministic `StubBackend` + the [`LocalCoordinator`](daemon_swarm_run::local_coordinator)
//!   shell (tick + signing + receipt production over a shared payload store). Prints the agreed
//!   per-round digest transcript + the offline replay check (PROTO-20).
//! - `--backend worker`: one supervised `daemon-train-worker` per peer over the frozen worker
//!   protocol (probe → assess the **real** signed envelope → join). The worker verifies the frozen
//!   envelope, extracts `[experiment.config]`, resolves its `.wasm` module from the envelope artifact
//!   map (`file://`, blake3-verified), and drives real host training (`WasmBackend`).
//!
//! This is a local developer runnable (like `xtask`): it reads an operator-supplied envelope path on
//! the operator's own machine, so the file read carries a declared `#[allow]` rather than the
//! `ContainedRoot` ceremony reserved for attacker-influenced paths.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use serde::Deserialize;

use daemon_swarm_proto::envelope::{
    Access, Artifact, DataSection, Envelope, ExperimentSection, GlobalBatch, Phases, Requirements,
    RoundMode, RunSection, StopCondition, ENVELOPE_SCHEMA_MAJOR,
};
use daemon_swarm_proto::{to_canonical_vec, FrozenEnvelope, Hash, SigningKey};
use daemon_swarm_run::harness::{run_swarm, SwarmConfig};
use daemon_swarm_run::protocol::{JoinPolicy, LeaveMode, PolicyMode};
use daemon_train_client::{TrainClientConfig, TrainSupervisor};

/// The peer backend the runner drives.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Backend {
    /// In-process `RoundEngine` peers over the deterministic `StubBackend`.
    Stub,
    /// One supervised `daemon-train` worker per peer over the frozen worker protocol.
    Worker,
}

/// The `swarm-local` command line (frozen at Merge 3).
#[derive(Parser, Debug)]
#[command(
    name = "swarm-local",
    about = "Run a swarm training round loop locally (spec §10.4)."
)]
struct Args {
    /// Path to the authoring TOML envelope (§6.1).
    #[arg(long)]
    envelope: PathBuf,
    /// Number of in-process peers (clamped to the envelope's `[run].min_peers..=max_peers`).
    #[arg(long, default_value_t = 3)]
    peers: usize,
    /// Rounds to drive (defaults to the envelope's `[data].rounds`).
    #[arg(long)]
    rounds: Option<u64>,
    /// Corpus + coordinator seed (`0x`-prefixed hex or decimal).
    #[arg(long, default_value = "0xDAE07E57")]
    seed: String,
    /// Payload-store / worker state directory (informational for stub mode).
    #[arg(long)]
    state_dir: Option<PathBuf>,
    /// Peer backend.
    #[arg(long, value_enum, default_value_t = Backend::Stub)]
    backend: Backend,
    /// Experiment profile passthrough (recorded in the run header).
    #[arg(long)]
    profile: Option<String>,
    /// Worker binary for `--backend worker` (the real `daemon-train-worker`).
    #[arg(long, default_value = "daemon-train-worker")]
    worker_bin: PathBuf,
    /// Write the registry `CreateRunRequest` JSON for this envelope to a file and exit (A3 —
    /// the run-authoring half of Merge-1 Decision 1). The request carries the **declared
    /// RunConfig** fields (`warmup_timeout_s`/`round_timeout_s`/`cooldown_s`/`global_batch`)
    /// derived from the frozen envelope by THIS authoring tool, so the cloud registry forwards
    /// them verbatim to the DO `init` with zero envelope parsing cloud-side (§11.1/§12). POST it:
    /// `curl -X POST {base}/runs -H content-type:application/json -d @<path>`.
    #[arg(long)]
    emit_create_request: Option<PathBuf>,
}

/// The authoring TOML the runner parses (a small, operator-friendly subset of §6.1). The proto crate
/// never parses TOML (it stays wasm32-clean), so the mapping to the resolved [`Envelope`] lives here.
#[derive(Debug, Deserialize)]
struct EnvelopeToml {
    run: RunToml,
    experiment: ExperimentToml,
    data: DataToml,
    #[serde(default)]
    requirements: RequirementsToml,
}

#[derive(Debug, Deserialize)]
struct RunToml {
    run_id: String,
    min_peers: u32,
    max_peers: u32,
}

#[derive(Debug, Deserialize)]
struct ExperimentToml {
    module: String,
    abi: String,
    #[serde(default)]
    config: String,
}

#[derive(Debug, Deserialize)]
struct DataToml {
    #[serde(default = "default_manifest")]
    manifest: String,
    steps_per_round: u32,
    #[serde(default = "default_global_batch")]
    global_batch: u32,
    rounds: u64,
}

#[derive(Debug, Default, Deserialize)]
struct RequirementsToml {
    #[serde(default = "default_throughput_floor")]
    throughput_floor: String,
}

fn default_manifest() -> String {
    "manifest.json".to_string()
}
fn default_global_batch() -> u32 {
    12
}
fn default_throughput_floor() -> String {
    "c1".to_string()
}

impl EnvelopeToml {
    /// Build the resolved proto [`Envelope`] this authoring TOML describes.
    fn into_envelope(self) -> Envelope {
        // Artifacts are pinned by content hash on the real plane; a local demo carries placeholder
        // hashes (validation only checks the module + manifest names resolve).
        let mut artifacts = std::collections::BTreeMap::new();
        artifacts.insert(
            self.experiment.module.clone(),
            Artifact {
                url: format!("file://{}", self.experiment.module),
                blake3: Hash([0u8; 32]),
            },
        );
        artifacts.insert(
            self.data.manifest.clone(),
            Artifact {
                url: format!("file://{}", self.data.manifest),
                blake3: Hash([0u8; 32]),
            },
        );
        Envelope {
            run: RunSection {
                schema: ENVELOPE_SCHEMA_MAJOR,
                run_id: self.run.run_id,
                min_peers: self.run.min_peers,
                max_peers: self.run.max_peers,
                access: Access::Org,
            },
            experiment: ExperimentSection {
                module: self.experiment.module,
                abi: self.experiment.abi,
                // An empty authoring `config` embeds the tiny-llama preset default (a valid, non-trivial
                // experiment config), so `--backend worker` drives a real training round rather than
                // choking on a placeholder. A non-empty string is carried verbatim (opaque, §4.3).
                config: if self.experiment.config.trim().is_empty() {
                    ciborium::value::Value::serialized(&daemon_train_sdk::models::TinyLlamaCfg {
                        n_layers: 1,
                        seq_len: 9,
                        vocab: 64,
                        ..daemon_train_sdk::models::TinyLlamaCfg::default()
                    })
                    .expect("tiny-llama default config is serializable")
                } else {
                    ciborium::value::Value::Text(self.experiment.config)
                },
            },
            artifacts,
            data: DataSection {
                manifest: self.data.manifest,
                steps_per_round: self.data.steps_per_round,
                global_batch: GlobalBatch {
                    start: self.data.global_batch,
                    end: self.data.global_batch,
                    ramp_rounds: 0,
                },
                stop: StopCondition::Rounds(self.data.rounds),
            },
            requirements: Requirements {
                vram_mb_min: 0,
                ram_gb_min: 0,
                uplink_mbps_min: 0,
                downlink_mbps_min: 0,
                disk_gb_min: 0,
                throughput_floor: self.requirements.throughput_floor,
                update_mb_max: 64,
                capabilities: Vec::new(),
                payload_store: "r2".to_string(),
            },
            phases: Phases {
                round_mode: RoundMode::Barrier,
                warmup: 1,
                round_train_max: 1,
                round_witness: 1,
                cooldown: 1,
                epoch_rounds: 0,
                checkpoint_every_epochs: 0,
                stall_rounds_max: 2,
                payload_retention_rounds: 8,
            },
        }
    }
}

/// Read the operator-supplied envelope file. Declared `#[allow]`: a local dev-run path on the
/// operator's own machine, not an attacker-influenced input (the `ContainedRoot` guard's domain).
#[allow(clippy::disallowed_methods)]
fn read_envelope(path: &Path) -> Result<String> {
    std::fs::read_to_string(path).with_context(|| format!("read envelope {}", path.display()))
}

/// Read the operator-supplied module `.wasm` (to content-address it into the envelope artifact).
/// Same declared `#[allow]` rationale as [`read_envelope`]: a local dev-run path, not an
/// attacker-influenced input.
#[allow(clippy::disallowed_methods)]
fn read_module(path: &Path) -> std::io::Result<Vec<u8>> {
    std::fs::read(path)
}

fn parse_seed(s: &str) -> Result<u64> {
    let s = s.trim();
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u64::from_str_radix(hex, 16).context("parse hex seed")
    } else {
        s.parse::<u64>().context("parse decimal seed")
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let seed = parse_seed(&args.seed)?;

    let toml_text = read_envelope(&args.envelope)?;
    let spec: EnvelopeToml = toml::from_str(&toml_text).context("parse envelope TOML")?;
    let rounds = args.rounds.unwrap_or(spec.data.rounds);
    let steps_per_round = spec.data.steps_per_round;
    let (min_peers, max_peers) = (spec.run.min_peers as usize, spec.run.max_peers as usize);
    let module_name = spec.experiment.module.clone();
    let mut envelope = spec.into_envelope();

    // Resolve the experiment module file so the worker path carries a real, blake3-verified artifact
    // (relative to the envelope dir, or `DAEMON_TRAIN_MODULE`). The frozen envelope then binds the
    // module by content hash; the worker fetches it via `ArtifactResolver`. Stub mode ignores this.
    let module_abs = resolve_module_path(&args.envelope, &module_name);
    if let Some(abs) = &module_abs {
        if let Ok(bytes) = read_module(abs) {
            if let Some(art) = envelope.artifacts.get_mut(&module_name) {
                art.url = format!("file://{}", abs.display());
                art.blake3 = daemon_swarm_proto::blake3_hash(&bytes);
            }
        }
    }

    // The real §6.1 freeze → hash → sign → verify chain (a demo author key for local runs).
    envelope.validate().context("envelope validation")?;
    let author = SigningKey::from_bytes(&[0xA1u8; 32]);
    let frozen: FrozenEnvelope = envelope.freeze(&author).context("freeze envelope")?;
    frozen.verify().context("verify frozen envelope")?;

    // A3 (Merge-1 Decision 1, the authoring half): emit the registry CreateRunRequest with the
    // DECLARED RunConfig fields — the run author derives them from the envelope it just froze, so
    // the cloud forwards them verbatim to the DO `init` with zero envelope parsing (§11.1/§12).
    if let Some(path) = &args.emit_create_request {
        let request = create_run_request(&envelope, &frozen)?;
        write_create_request(path, &request)?;
        println!(
            "create-run request written : {} (declared warmup/round/cooldown/global_batch)",
            path.display()
        );
        return Ok(());
    }

    let peers = args.peers.clamp(min_peers.max(1), max_peers.max(1));

    println!("swarm-local — local run");
    println!("  run_id        : {}", frozen_run_id(&frozen)?);
    println!("  envelope hash : {}", frozen.hash().to_hex());
    println!("  peers         : {peers}  (min {min_peers}, max {max_peers})");
    println!("  rounds        : {rounds}");
    println!("  steps/round   : {steps_per_round}");
    println!("  seed          : {seed:#x}");
    if let Some(profile) = &args.profile {
        println!("  profile       : {profile}");
    }
    if let Some(dir) = &args.state_dir {
        println!("  state dir     : {}", dir.display());
    }
    println!("  backend       : {:?}", args.backend);

    match args.backend {
        Backend::Stub => run_stub(peers, rounds, steps_per_round, seed).await,
        Backend::Worker => {
            // The worker receives the full signed envelope (it verifies + resolves its module).
            let wire = to_canonical_vec(&frozen.to_wire()).context("encode signed envelope")?;
            run_worker(peers, &args.worker_bin, &wire, module_abs.as_deref()).await
        }
    }
}

/// Resolve the experiment `.wasm` path for worker mode: `DAEMON_TRAIN_MODULE` if set, else the
/// module name resolved relative to the envelope file's directory. `None` if neither exists.
fn resolve_module_path(envelope: &Path, module_name: &str) -> Option<PathBuf> {
    if let Ok(p) = std::env::var("DAEMON_TRAIN_MODULE") {
        let p = PathBuf::from(p);
        if p.exists() {
            return Some(p);
        }
    }
    let candidate = envelope
        .parent()
        .unwrap_or(Path::new("."))
        .join(module_name);
    candidate.exists().then_some(candidate)
}

/// Stub mode: the in-process peer harness + local coordinator over `StubBackend`.
async fn run_stub(peers: usize, rounds: u64, steps_per_round: u32, seed: u64) -> Result<()> {
    let cfg = SwarmConfig {
        num_peers: peers,
        num_rounds: rounds,
        steps_per_round,
        micro_batch: 2,
        corpus_seed: seed,
        ..SwarmConfig::small(rounds)
    };
    let run = run_swarm(cfg).await.context("run local swarm")?;

    println!(
        "\nagreed digest transcript ({} rounds):",
        run.agreed_transcript().len()
    );
    for (round, digest) in run.agreed_transcript() {
        println!("  round {round:>4}  {}", digest.to_hex());
    }
    let agree = run.all_agree();
    println!("\nall peers agree every round : {agree}");
    if let Some(replay) = &run.replay {
        println!(
            "coordinator replay verified : {} ({} rounds)",
            replay.verify(),
            replay.recorded_rounds()
        );
    }
    anyhow::ensure!(agree, "peers disagreed on at least one round");
    Ok(())
}

/// Worker mode: spawn one supervised `daemon-train-worker` per peer over the frozen worker protocol
/// (probe → assess the **real** signed envelope → join). Each worker verifies the envelope, resolves
/// its module from the artifact map (or the `DAEMON_TRAIN_MODULE` override), and drives real host
/// training (`WasmBackend`, self-driven round).
async fn run_worker(
    peers: usize,
    worker_bin: &Path,
    envelope_bytes: &[u8],
    module_path: Option<&Path>,
) -> Result<()> {
    let policy = JoinPolicy {
        mode: PolicyMode::Always,
        vram_cap_mb: 0,
        duty_cycle_pct: 100,
        schedule: None,
    };
    println!("\nspawning {peers} worker(s) from {}", worker_bin.display());
    if let Some(m) = module_path {
        println!("  module        : {}", m.display());
    }
    for i in 0..peers {
        let mut client_cfg = TrainClientConfig::new(worker_bin);
        if let Some(m) = module_path {
            client_cfg.env.push((
                "DAEMON_TRAIN_MODULE".to_string(),
                m.to_string_lossy().into_owned(),
            ));
        }
        let sup = TrainSupervisor::new(client_cfg);
        let hw = sup.probe().await.context("probe worker")?;
        let elig = sup
            .assess(envelope_bytes.to_vec())
            .await
            .context("assess run")?;
        println!(
            "  peer {i}: gpus={} vram_mb={} eligible={} {:?}",
            hw.gpus, hw.vram_mb, elig.eligible, elig.reasons
        );
        if elig.eligible {
            sup.join(
                "local-demo",
                "local://coordinator",
                Vec::new(),
                policy.clone(),
            )
            .await
            .context("join run")?;
            sup.leave("local-demo", LeaveMode::Graceful)
                .await
                .context("leave run")?;
        }
        sup.shutdown().await;
    }
    println!("\nworker-backed run over the frozen protocol (real host training via WasmBackend).");
    Ok(())
}

/// The run id from the frozen envelope's decoded body.
fn frozen_run_id(frozen: &FrozenEnvelope) -> Result<String> {
    Ok(frozen.decode().context("decode envelope")?.run.run_id)
}

/// Build the cloud registry `CreateRunRequest` JSON body for a frozen envelope, carrying the
/// **declared RunConfig** (Merge-1 Decision 1; A3): the coordination params
/// (`warmup_timeout_s`/`round_timeout_s`/`cooldown_s`/`global_batch`) come from the envelope THIS
/// authoring tool resolved + froze — the cloud never re-derives them. Matches the
/// `packages/shared/src/swarm/types.ts` `CreateRunRequest` shape verbatim.
fn create_run_request(envelope: &Envelope, frozen: &FrozenEnvelope) -> Result<serde_json::Value> {
    use base64::Engine as _;
    let artifacts: Vec<serde_json::Value> = envelope
        .artifacts
        .iter()
        .map(|(name, art)| {
            serde_json::json!({
                "path": name,
                "blake3": art.blake3.to_hex(),
                "size": 0,
            })
        })
        .collect();
    let rounds = match envelope.data.stop {
        StopCondition::Rounds(r) => Some(r),
        _ => None,
    };
    Ok(serde_json::json!({
        "run_id": envelope.run.run_id,
        "schema": ENVELOPE_SCHEMA_MAJOR,
        "proto_version": daemon_swarm_proto::SWARM_PROTO_VERSION,
        "envelope_b64": base64::engine::general_purpose::STANDARD.encode(frozen.bytes()),
        "author_pubkey": frozen.signer().to_hex(),
        "signature": frozen.signature().to_hex(),
        "artifacts": artifacts,
        "update_max_bytes": u64::from(envelope.requirements.update_mb_max) * 1024 * 1024,
        "min_peers": envelope.run.min_peers,
        "max_peers": envelope.run.max_peers,
        "rounds": rounds,
        // Declared RunConfig (Merge-1 Decision 1) — envelope-derived, author-declared.
        "warmup_timeout_s": envelope.phases.warmup,
        "round_timeout_s": envelope.phases.round_train_max,
        "cooldown_s": envelope.phases.cooldown,
        "global_batch": envelope.data.global_batch.start,
    }))
}

/// Write the emitted create-run request (a local dev-authoring output path — the same declared
/// `#[allow]` rationale as [`read_envelope`]).
#[allow(clippy::disallowed_methods)]
fn write_create_request(path: &Path, request: &serde_json::Value) -> Result<()> {
    std::fs::write(
        path,
        serde_json::to_vec_pretty(request).context("encode request")?,
    )
    .with_context(|| format!("write create request {}", path.display()))
}
