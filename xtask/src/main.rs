// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

// Phase 4: xtask is dev/build tooling (codegen, CI helpers) run by maintainers, not a runtime
// security surface. Its fs (build artifacts) and spawns (cbindgen/cc/bash build steps) are
// developer-controlled; the hardening bans target the shipped node, so xtask is allowed crate-wide.
#![allow(clippy::disallowed_methods)]

//! `xtask` — repo automation (codegen, CI helpers).
//!
//! Subcommands:
//! - `gen-headers` — run `cbindgen` over both binding crates to (re)generate the committed C
//!   headers `bindings/daemon-core-ffi/include/daemon_core.h` (the L1 brain seam) and
//!   `bindings/daemon-ffi/include/daemon.h` (the L2 durable-host seam). The generated headers plus
//!   the published `daemon-api.cddl` are the complete non-Rust contract (daemon-ffi-spec §3.6).
//! - `cddl` — check the `daemon-api` mirror CDDL artifact covers the Rust wire enum variants.
//! - `api-fixtures` — write canonical CBOR request/response fixtures for non-Rust clients.
//! - `gen-zcbor` — generate the client zcbor C codec from a CDDL (the artifact `daemon-app` vendors).
//! - `verify-codec` — decode every CBOR fixture with the generated C codec, proving the CDDL/zcbor
//!   path stays byte-compatible with the serde/ciborium runtime wire format.

#![forbid(unsafe_code)]

mod ceremony;
mod publish;
mod replay;
mod tokenize;

use clap::{Parser, Subcommand};
use daemon_vhc_proto::corpus::TokenWidth;
use std::path::{Path, PathBuf};
use std::process::Command;

/// `xtask` — repo automation (codegen, CI helpers).
#[derive(Parser)]
#[command(name = "xtask", about)]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// (Re)generate the committed C headers for both binding crates via `cbindgen`.
    GenHeaders,
    /// Check the `daemon-api` mirror CDDL covers the Rust wire enum variants.
    Cddl,
    /// Write canonical CBOR request/response fixtures for non-Rust clients.
    ApiFixtures,
    /// Generate the client zcbor C codec from a CDDL.
    GenZcbor {
        /// The CDDL contract (defaults to the pinned `daemon-api.cddl`).
        #[arg(long)]
        cddl: Option<PathBuf>,
        /// The output directory (defaults to `target/zcbor-codec`).
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Decode every CBOR fixture with the generated C codec (wire-compat gate).
    VerifyCodec,
    /// Check the tracked ABI specification against the constants the code defines (drift gate).
    VhcAbiSpecDrift,
    /// Scan the code for this program's private vocabulary (red-lined scan gate).
    VhcCodenameScan,
    /// Build the vhc guest experiment modules (`guests/`) for `wasm32-unknown-unknown`.
    BuildGuests,
    /// Run the vhc **CI tier-1** suite: the CPU-only, consensus-critical determinism / round-
    /// protocol / codec / wasm-guest suites (TDD §8.1 tier 1). Builds the guests first, then runs the
    /// pinned suite list, failing on the first red. No GPU, no live substrate (env-gated live tests
    /// skip). This is the single in-repo definition of the per-PR vhc gate — the superproject CI
    /// job and a local operator both invoke `cargo run -p xtask -- vhc-ci-det`.
    ///
    /// Scheduling (coverage is identical in every mode — the same pinned suites, args, and
    /// pass/fail semantics): by default suites run as 3 concurrent package groups with
    /// `--test-threads=4` each (total cap 12 threads; same-package suites stay sequential in
    /// pinned order). `VHC_DET_SERIAL=1` restores the historical fully-serial schedule;
    /// `VHC_DET_GROUPS` / `VHC_DET_TEST_THREADS` tune the caps.
    VhcCiDet,
    /// Run the vhc **CI tier-2** whole-run suites (decisions D4): the deterministic sim/testkit
    /// whole runs as they land — SDK-side `daemon-vhc-sim` native whole runs (the SPARTA
    /// continuous-averaging toy over the virtual worlds) and host-side `daemon-vhc-testkit` whole
    /// runs over the PRODUCTION wasm blobs (wasmtime + simulated capability providers, journaled,
    /// §8.7 replay-verified). Heavier than tier-1 (wasmtime + guest builds), so it is a separate
    /// gate invoked as `cargo run -p xtask -- vhc-ci-t2`, never part of `vhc-ci-det`.
    VhcCiT2,
    /// Run the vhc **node + supervisor lifecycle** suites: the run-instance state machine
    /// (terminal transitions, generation gating, retry budget, crash-window repair, pause/resume),
    /// the resident reconciliation + seat-keeper passes, owner arbitration, and the supervisor's
    /// spawn/respawn/meltdown + event-stream observability over the REAL worker subprocess. This
    /// lane is FOLDED INTO the mandatory `vhc-ci-det` aggregate (never a side gate); the
    /// standalone command serves focused iteration.
    VhcCiNode,
    /// Enforce the daemon-vhc dependency-direction rules (architecture §7): `host/*` never links
    /// `sdk/*`, `contracts/*` links neither, `sdk/*` never links `host/*`. The honest current
    /// exceptions are listed inline and each is tracked to the phase that removes it.
    VhcDepCheck,
    /// Run the vhc **multi-process acceptance** suite: three REAL `daemon` node processes
    /// (seat-claimed coordinator + two trainers) over Unix sockets on the full product path,
    /// driven only through the node API, over WS control + the filesystem / R2 payload planes.
    /// Builds the node + worker binaries in RELEASE first (debug wasmtime compilation of the
    /// multi-layer trainer module exceeds the supervisor's assess watchdog) and points the suite
    /// at them via `VHC_ACCEPTANCE_BIN_DIR`; the required gates run as named tests. Heavy
    /// (multi-process + burn compute), so it is its own command, folded into `vhc-production-gate`.
    ///
    /// The release build is cached under a sound source key (HEAD + dirty/untracked content +
    /// toolchain + codegen env — see `acceptance_release_bins`): a byte-identical workspace
    /// reuses its binaries instead of rebuilding (~12 min). `VHC_ACCEPTANCE_BIN_CACHE=0`
    /// disables; any other value overrides the cache dir (default
    /// `$HOME/.cache/vhc-acceptance-bins`).
    VhcAcceptance,
    /// The merge gate (D-P10): the tier-1 det aggregate (which folds `vhc-ci-node`), the tier-2
    /// whole-run suites, and the multi-process acceptance suite.
    ///
    /// DIFF-SCOPED by default, exactly like `just lint`: history at the base has already been
    /// gated, so re-testing the whole tree on every iteration is waste. Changed files vs the
    /// merge-base with the base ref (default `vhc-integration`; override via GATE_BASE or
    /// `--base` — the LINT_BASE convention), unioned with staged/unstaged/untracked, map to
    /// workspace crates and expand through the cargo dependency graph to reverse-dependents;
    /// only that cone of the pinned battery runs, with identical per-suite semantics. Non-crate
    /// inputs map conservatively: `crates/vhc/guests/` selects every guest-linked suite; xtask,
    /// Cargo.lock, the root Cargo.toml, the flake, `.cargo/`, `vendor/`, and any UNMAPPED path
    /// fail CLOSED into the full battery — selection can only over-include, never under. The
    /// acceptance lane runs whenever the cone reaches the product binaries.
    ///
    /// Lanes memoize green runs by workspace fingerprint (`target/vhc-green-ledger/`;
    /// VHC_GATE_MEMO=0 disables), so re-gating a byte-identical tree — e.g. right after a no-ff
    /// merge of an already-gated tip — re-verifies nothing and reports "green (memoized)".
    ///
    /// The full-tree battery remains `--all` (manual pre-release / post-rebase only —
    /// deliberately wired into NO workflow). `VHC_GATE_PARALLEL=1` (opt-in) overlaps the det and
    /// t2 lanes in `--all` runs, per-lane thread caps summing to <=16 on a 32-thread host.
    VhcProductionGate {
        /// Run the FULL pinned battery regardless of the diff (the `lint-all` analogue: a manual
        /// pre-release / post-rebase pass, deliberately wired into no workflow).
        #[arg(long)]
        all: bool,
        /// Ignore the green ledger: every selected lane RUNS, even for a byte-identical workspace.
        ///
        /// The certification switch. A full-scope battery whose lanes were skipped because a
        /// fingerprint matched has verified the ledger, not the tree — so a certification run states
        /// `--all --no-memo` and the flag is recorded with the verdict.
        #[arg(long)]
        no_memo: bool,
        /// Base ref for the diff scope (default: env GATE_BASE, else `vhc-integration`). The
        /// changed set is the merge-base diff vs HEAD unioned with staged/unstaged/untracked.
        #[arg(long)]
        base: Option<String>,
        /// Print the selection (changed set -> crate cone -> suites) and exit WITHOUT running
        /// anything or touching the green ledger. For inspecting what a diff would gate.
        #[arg(long)]
        dry_run: bool,
    },
    /// Author + freeze the fleet-ceremony genesis (a thin wrapper around the reviewed
    /// `daemon_vhc_testkit::ceremony::ceremony_genesis` — never a reimplementation). Emits
    /// `envelope.cbor`, `envelope.b64` (the cloud seeder's `VHC_ENVELOPE_B64`), `run-id.txt` (the
    /// genesis hash hex), and `authoring-report.txt` (every frozen pin, for human ratification).
    /// Also authors single-peer smoke geneses (`--min-peers 1 --max-peers 1 --stop-rounds <small>`).
    AuthorCeremonyGenesis {
        /// The human/registry-facing run label.
        #[arg(long)]
        run_label: String,
        /// The genesis author signing key: a file (32 raw bytes or 64-hex text) or a bare 64-hex.
        #[arg(long)]
        author_key: String,
        /// The pinned coordinator module blake3 (64-hex), from `guests.blake3` / `build-guests`.
        #[arg(long)]
        coordinator_module: String,
        /// The pinned trainer module blake3 (64-hex).
        #[arg(long)]
        trainer_module: String,
        /// The coordinator module FILE. Required alongside its digest, and checked against it: the
        /// role's resource requirements are derived by running the module's own assessment export, and
        /// a digest cannot be asked what it needs.
        #[arg(long)]
        coordinator_wasm: PathBuf,
        /// The trainer module FILE, checked against `--trainer-module`.
        #[arg(long)]
        trainer_wasm: PathBuf,
        /// The published `corpus-manifest.cbor` (its blake3 is the genesis corpus pin; its shards
        /// + tokenizer become the trainer's `data@2` fetch grants).
        #[arg(long)]
        corpus_manifest: PathBuf,
        /// A genesis-trusted base identity PeerId (64-hex). Repeat; ORDERED — the FIRST is the
        /// coordinator authority.
        #[arg(long = "trusted-base")]
        trusted_base: Vec<String>,
        /// A trainer roster PeerId (64-hex). Repeat.
        #[arg(long)]
        roster: Vec<String>,
        /// An upgrade-authority PeerId (64-hex). Repeat; empty authors an immutable run.
        #[arg(long = "upgrade-authority")]
        upgrade_authority: Vec<String>,
        /// Minimum healthy peers to leave WaitingForMembers (the fleet floor; `1` for a smoke).
        #[arg(long, default_value_t = 3)]
        min_peers: u32,
        /// Roster ceiling.
        #[arg(long, default_value_t = 3)]
        max_peers: u32,
        /// The remote checkpoint cadence in rounds (validated against retention at authoring).
        #[arg(long = "ckpt-cadence", default_value_t = 8)]
        ckpt_cadence: u64,
        /// The payload retention floor in rounds (`0` = unbounded).
        #[arg(long = "payload-retention", default_value_t = 64)]
        payload_retention: u64,
        /// Real run timer: join/warmup wall (seconds).
        #[arg(long = "warmup-s", default_value_t = 1_000_000)]
        warmup_s: u64,
        /// Real run timer: per-round training-phase wall ceiling (seconds).
        #[arg(long = "round-max-s", default_value_t = 1_000_000)]
        round_max_s: u64,
        /// Real run timer: witness/finalization-phase wall (seconds).
        #[arg(long = "witness-s", default_value_t = 1_000_000)]
        witness_s: u64,
        /// Real run timer: end-of-run cooldown wall (seconds).
        #[arg(long = "cooldown-s", default_value_t = 1_000_000)]
        cooldown_s: u64,
        /// Stop after this many completed rounds.
        #[arg(long = "stop-rounds", default_value_t = 1_000_000)]
        stop_rounds: u64,
        /// The output directory for the four ceremony artifacts.
        #[arg(long)]
        out: PathBuf,
    },
    /// Run BOTH replay-oracle modes (input replay + sandboxed consensus re-derivation) over an
    /// on-disk archive directory and emit a per-round, per-peer machine-readable verdict
    /// (agree/disagree + first divergence round). Green over a genuine archive; red with the
    /// divergence round over a corrupted copy. See the archive-layout contract in `--help`.
    VhcReplay {
        /// The archive directory (see the layout contract in this command's long help / runbook).
        #[arg(long)]
        archive: PathBuf,
        /// The run id (genesis hash hex) the archive belongs to.
        #[arg(long)]
        run: String,
        /// Emit the verdict as machine-readable JSON (default: human-readable text).
        #[arg(long)]
        json: bool,
    },
    /// Tokenize a corpus into fixed-width shards + `manifest.json` (spec §8; M1 seam).
    TokenizeCorpus {
        /// HF dataset repo id (e.g. `roneneldan/TinyStories`); omit when using `--text`.
        #[arg(long)]
        dataset: Option<String>,
        /// The file within the dataset repo (e.g. `TinyStories-valid.txt`).
        #[arg(long)]
        dataset_file: Option<String>,
        /// Pinned dataset revision (commit SHA / tag).
        #[arg(long, default_value = "main")]
        revision: String,
        /// A local corpus text file — bypasses the HF dataset pull (offline / synthetic).
        #[arg(long)]
        text: Option<PathBuf>,
        /// HF model id for the tokenizer (e.g. `gpt2`) OR a local `tokenizer.json` path.
        #[arg(long)]
        tokenizer: String,
        /// Pinned tokenizer revision (defaults to `main` when pulling from HF).
        #[arg(long)]
        tokenizer_revision: Option<String>,
        /// Output directory for shards + `manifest.json`.
        #[arg(long)]
        out_dir: PathBuf,
        /// Tokens per shard (rounded down to a multiple of `--seq-len`).
        #[arg(long, default_value_t = 1_048_576)]
        shard_tokens: u64,
        /// Sequence length (tokens per training sequence).
        #[arg(long, default_value_t = 1024)]
        seq_len: u32,
        /// Token element width: `u16` (vocab ≤ 65 536) or `u32`.
        #[arg(long, default_value = "u16")]
        token_width: String,
        /// Chunk size in bytes (pinned in the manifest; must be a multiple of the token width).
        #[arg(long, default_value_t = daemon_vhc_proto::CORPUS_DEFAULT_CHUNK_SIZE)]
        chunk_size: u64,
        /// The tokenizer's end-of-sequence token id, where known (recorded in the manifest).
        #[arg(long)]
        eos_id: Option<u32>,
        /// The padding token id, where the pipeline pads (recorded in the manifest).
        #[arg(long)]
        pad_id: Option<u32>,
        /// Optional cap on total tokens emitted (keeps a vendored fixture small).
        #[arg(long)]
        max_tokens: Option<u64>,
    },
    /// Publish an experiment module to the payload store at `modules/<blake3>.wasm` (the fleet artifact-distribution path).
    PublishModule {
        /// The `.wasm` module to upload.
        #[arg(long)]
        module: PathBuf,
        /// The run id whose prefix the object lives under (`runs/<run>/modules/…`).
        #[arg(long)]
        run: String,
        /// The coordinator presign base (e.g. `https://…/api/v1/vhc`).
        #[arg(long)]
        presign_base: String,
        /// `vhc:*`-scoped bearer token (gateway path).
        #[arg(long)]
        bearer: Option<String>,
        /// Internal identity org id (direct-to-`apps/vhc` dev path; pair with `--actor`).
        #[arg(long)]
        org: Option<String>,
        /// Internal identity actor (pair with `--org`).
        #[arg(long)]
        actor: Option<String>,
    },
    /// Publish a pre-tokenized corpus (shards + manifest) to the payload store by content hash (the fleet artifact-distribution path).
    PublishCorpus {
        /// The `corpus-manifest.cbor` produced by `tokenize-corpus` (shards + tokenizer.json beside it).
        #[arg(long)]
        manifest: PathBuf,
        /// The run id whose prefix the objects live under (`runs/<run>/corpus/…`).
        #[arg(long)]
        run: String,
        /// The coordinator presign base.
        #[arg(long)]
        presign_base: String,
        /// `vhc:*`-scoped bearer token (gateway path).
        #[arg(long)]
        bearer: Option<String>,
        /// Internal identity org id (pair with `--actor`).
        #[arg(long)]
        org: Option<String>,
        /// Internal identity actor (pair with `--org`).
        #[arg(long)]
        actor: Option<String>,
    },
}

/// Build a [`publish::Target`] from the shared CLI auth flags.
fn publish_target(
    run: String,
    presign_base: String,
    bearer: Option<String>,
    org: Option<String>,
    actor: Option<String>,
) -> publish::Target {
    let internal = match (org, actor) {
        (Some(org), Some(actor)) => Some((org, actor)),
        _ => None,
    };
    publish::Target {
        presign_base,
        run,
        bearer,
        internal,
    }
}

fn main() -> anyhow::Result<()> {
    match Cli::parse().command {
        Cmd::GenHeaders => gen_headers(),
        Cmd::Cddl => check_cddl(),
        Cmd::ApiFixtures => gen_api_fixtures(),
        Cmd::GenZcbor { cddl, out } => gen_zcbor(cddl, out),
        Cmd::VerifyCodec => verify_codec(),
        Cmd::VhcAbiSpecDrift => vhc_abi_spec_drift(),
        Cmd::VhcCodenameScan => vhc_codename_scan(),
        Cmd::BuildGuests => build_guests(),
        Cmd::VhcCiDet => vhc_ci_det(),
        Cmd::VhcCiT2 => vhc_ci_t2(),
        Cmd::VhcCiNode => vhc_ci_node(),
        Cmd::VhcDepCheck => vhc_dep_check(),
        Cmd::VhcAcceptance => vhc_acceptance(),
        Cmd::VhcProductionGate {
            all,
            base,
            dry_run,
            no_memo,
        } => vhc_production_gate(all, base, dry_run, no_memo),
        Cmd::AuthorCeremonyGenesis {
            run_label,
            author_key,
            coordinator_module,
            trainer_module,
            coordinator_wasm,
            trainer_wasm,
            corpus_manifest,
            trusted_base,
            roster,
            upgrade_authority,
            min_peers,
            max_peers,
            ckpt_cadence,
            payload_retention,
            warmup_s,
            round_max_s,
            witness_s,
            cooldown_s,
            stop_rounds,
            out,
        } => ceremony::run(ceremony::Args {
            run_label,
            author_key,
            coordinator_module,
            trainer_module,
            coordinator_wasm,
            trainer_wasm,
            corpus_manifest,
            trusted_base,
            roster,
            upgrade_authority,
            min_peers,
            max_peers,
            ckpt_cadence,
            payload_retention,
            warmup_s,
            round_max_s,
            witness_s,
            cooldown_s,
            stop_rounds,
            out,
        }),
        Cmd::VhcReplay { archive, run, json } => replay::run(replay::Args { archive, run, json }),
        Cmd::TokenizeCorpus {
            dataset,
            dataset_file,
            revision,
            text,
            tokenizer,
            tokenizer_revision,
            out_dir,
            shard_tokens,
            seq_len,
            token_width,
            chunk_size,
            eos_id,
            pad_id,
            max_tokens,
        } => {
            let token_width = match token_width.as_str() {
                "u16" => TokenWidth::U16,
                "u32" => TokenWidth::U32,
                other => anyhow::bail!("--token-width must be u16 or u32, got {other}"),
            };
            tokenize::run(tokenize::Args {
                dataset,
                dataset_file,
                revision,
                text,
                tokenizer,
                tokenizer_revision,
                out_dir,
                shard_tokens,
                seq_len,
                token_width,
                chunk_size,
                eos_id,
                pad_id,
                max_tokens,
            })
        }
        Cmd::PublishModule {
            module,
            run,
            presign_base,
            bearer,
            org,
            actor,
        } => publish::publish_module(
            module,
            publish_target(run, presign_base, bearer, org, actor),
        ),
        Cmd::PublishCorpus {
            manifest,
            run,
            presign_base,
            bearer,
            org,
            actor,
        } => publish::publish_corpus(
            manifest,
            publish_target(run, presign_base, bearer, org, actor),
        ),
    }
}

/// Build the vhc guest experiment modules for `wasm32-unknown-unknown`.
///
/// `guests/` is its OWN cargo workspace (excluded from the root workspace), so the host's native
/// `cargo build/clippy/test` never tries to build a `cdylib` for wasm. This target runs
/// `cargo build --release --target wasm32-unknown-unknown` inside it (swarm-training-spec.md §10.1).
/// The `wasm32-unknown-unknown` rust-std is provided by the flake devShell toolchain.
fn build_guests() -> anyhow::Result<()> {
    let root = workspace_root();
    let guests = root.join("crates/vhc/guests");
    anyhow::ensure!(
        guests.join("Cargo.toml").is_file(),
        "no guests workspace at {}",
        guests.display()
    );

    // The SHARED builder (`daemon-vhc-guest-build`): the identical env-scrubbed, remapped,
    // lock-serialized `cargo build` every wasm-backed test harness goes through. One code path
    // means the manifest this command pins and the bytes the suites rebuild can never diverge on
    // build env (the remap/RUSTFLAGS rationale lives with the builder). Together with the guests
    // workspace's COMMITTED Cargo.lock (B3 sitting 2) and its config-wired `rustc-wrapper`
    // (`guest-rustc-shim.sh` — the `-C metadata` pin + the wasm32 getrandom backend cfg), the
    // `.wasm` bytes (hence `guests.blake3`) are byte-identical across checkout paths (C2 lead-in).
    daemon_vhc_guest_build::build_guests().map_err(|e| anyhow::anyhow!("{e}"))?;

    // Stale-guest guard (an archived engineering-ledger Merge-1 follow-on): write the committed
    // blake3 manifest of
    // the built modules. The wasm-backed test harness asserts the module it loads matches this file,
    // so a stale/mismatched guest fails loud instead of surfacing downstream as a NaN loss.
    let manifest = write_guest_manifest(&guests)?;
    println!(
        "built guests in {} (manifest {})",
        guests.display(),
        manifest.display()
    );
    Ok(())
}

/// Run the vhc CI tier-1 suite (TDD §8.1 tier 1: the per-PR, hosted-CI, no-GPU gate).
///
/// Every bit-exact / cross-peer-consensus claim is a CPU property by contract (the det lane is CPU
/// fp32, spec §5.6), so this tier runs on plain runners. It covers the shared det kernels; the round
/// protocol, assignment, envelope schema, and canonical CBOR; the harness, assess, and replay
/// (loopback); the observe/replay oracle; the WS/gossip framing and dedupe codecs (no network); the
/// worker det lane, cross-backend digest identity, and wasm-guest determinism; the SDK profile
/// goldens; the e2e drills and observe-replay (no iroh/live); the wire codec conformance; and — from
/// A1 — the crash-safe segmented journal substrate: the ABI §8.3 record-grammar validity + per-tag
/// round-trips, crash safety (torn-write/CRC/chain recovery, durable seq never reused), and the
/// coordinator oracle re-derived byte-identically over the journal (the journal-soak gate, refactor
/// G6 / Decision 4) — see the pinned list below.
///
/// The GPU (`wgpu`/`cuda`) and live-substrate lanes are deliberately EXCLUDED: those are the scheduled
/// per-lane tier 2 and the manual hardware-in-loop gate tier 3 (see swarm-p2-gate-runbook.md).
///
/// Scheduling knobs for a CI lane's pinned suite list. This is SCHEDULING ONLY: every schedule
/// runs the identical suite list with identical per-suite arguments and pass/fail semantics —
/// only ordering/overlap and the libtest thread count change, never what is tested.
#[derive(Clone, Copy)]
struct LaneSchedule {
    /// How many package groups run concurrently. Suites of the SAME package always run
    /// sequentially in their pinned order (same-package suites share test binaries and
    /// crate-scoped fixtures — e.g. the det lane's `daemon-vhc-observe` journal subset re-runs
    /// binaries its full-suite entry built — so they must never overlap themselves).
    groups: usize,
    /// `--test-threads` handed to every test binary (`None` = the libtest default). Bounds
    /// TOTAL lane threads at `groups * test_threads` — the host-discipline cap.
    test_threads: Option<usize>,
}

impl LaneSchedule {
    /// The historical schedule: one suite at a time, libtest default threads, live output.
    fn serial() -> Self {
        LaneSchedule {
            groups: 1,
            test_threads: None,
        }
    }
}

/// The det lane's schedule: 3 concurrent package groups x 4 libtest threads (total cap 12 of the
/// 32 hardware threads) by default — the lane is dominated by serially-executed test binaries
/// (measured ~1.5 cores busy of 32), not compilation, so bounded overlap cuts wall time without
/// threatening the host. `VHC_DET_SERIAL=1` restores the historical fully-serial schedule;
/// `VHC_DET_GROUPS` / `VHC_DET_TEST_THREADS` tune the caps.
fn det_schedule_from_env() -> LaneSchedule {
    if std::env::var("VHC_DET_SERIAL").is_ok_and(|v| v == "1") {
        return LaneSchedule::serial();
    }
    let groups = std::env::var("VHC_DET_GROUPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&g: &usize| g >= 1)
        .unwrap_or(3);
    let test_threads = std::env::var("VHC_DET_TEST_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&t: &usize| t >= 1)
        .unwrap_or(4);
    LaneSchedule {
        groups,
        test_threads: Some(test_threads),
    }
}

/// One pinned suite entry: (human label, `cargo test` args).
type SuiteEntry<'a> = (&'a str, &'a [&'a str]);

/// The package a suite entry tests (the value after `-p` — every pinned suite names exactly one).
fn suite_package<'a>(args: &[&'a str]) -> &'a str {
    args.iter()
        .position(|a| *a == "-p")
        .and_then(|i| args.get(i + 1))
        .copied()
        .expect("every pinned suite entry names its package via -p")
}

/// Serialize the interleaved completion prints of concurrently running lanes/suites (the det and
/// t2 lanes can run at once under `VHC_GATE_PARALLEL=1`, each with its own internal overlap).
static SUITE_PRINT_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Run a lane's pinned suite list under a [`LaneSchedule`].
///
/// The serial schedule streams each suite's output live (the historical behavior). Any
/// overlapped schedule captures per-suite output and prints it whole on completion (under a
/// global print lock), so concurrent suites never interleave lines; the first red stops new
/// suites from starting (in-flight suites finish) and the lane fails with the red suite's label.
fn run_lane_suites(
    lane: &str,
    suites: &[(&str, &[&str])],
    schedule: LaneSchedule,
) -> anyhow::Result<()> {
    let root = workspace_root();
    if schedule.groups <= 1 && schedule.test_threads.is_none() {
        for (label, args) in suites {
            println!("\n== {lane}: {label} ==");
            let started = std::time::Instant::now();
            let status = Command::new("cargo")
                .current_dir(&root)
                .arg("test")
                .args(*args)
                .status()
                .map_err(|e| anyhow::anyhow!("running cargo test {args:?}: {e}"))?;
            anyhow::ensure!(status.success(), "{lane} suite failed: {label}");
            println!("== {lane}: {label} — green in {:.0?} ==", started.elapsed());
        }
        return Ok(());
    }

    // Group by package, preserving pinned intra-group order; schedule bigger groups first so the
    // longest sequential chain (e.g. the det lane's 12 daemon-vhc-host entries) starts earliest.
    let mut groups: Vec<(&str, Vec<SuiteEntry<'_>>)> = Vec::new();
    for &(label, args) in suites {
        let pkg = suite_package(args);
        match groups.iter_mut().find(|(p, _)| *p == pkg) {
            Some((_, entries)) => entries.push((label, args)),
            None => groups.push((pkg, vec![(label, args)])),
        }
    }
    groups.sort_by_key(|(_, entries)| std::cmp::Reverse(entries.len()));

    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    let next_group = AtomicUsize::new(0);
    let stop = AtomicBool::new(false);
    let failures: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

    std::thread::scope(|scope| {
        for _ in 0..schedule.groups.min(groups.len()) {
            scope.spawn(|| loop {
                let g = next_group.fetch_add(1, Ordering::SeqCst);
                let Some((_, entries)) = groups.get(g) else {
                    return;
                };
                for &(label, args) in entries {
                    if stop.load(Ordering::SeqCst) {
                        return;
                    }
                    let started = std::time::Instant::now();
                    let mut cmd = Command::new("cargo");
                    cmd.current_dir(&root).arg("test").args(args);
                    if let Some(threads) = schedule.test_threads {
                        cmd.args(["--", &format!("--test-threads={threads}")]);
                    }
                    let output = cmd.output();
                    let _print = SUITE_PRINT_LOCK.lock().expect("suite print lock");
                    match output {
                        Ok(out) => {
                            let verdict = if out.status.success() {
                                "green"
                            } else {
                                "FAILED"
                            };
                            println!(
                                "\n== {lane}: {label} — {verdict} in {:.0?} ==",
                                started.elapsed()
                            );
                            print!("{}", String::from_utf8_lossy(&out.stdout));
                            eprint!("{}", String::from_utf8_lossy(&out.stderr));
                            if !out.status.success() {
                                failures
                                    .lock()
                                    .expect("failures lock")
                                    .push(label.to_string());
                                stop.store(true, Ordering::SeqCst);
                                return;
                            }
                        }
                        Err(e) => {
                            failures
                                .lock()
                                .expect("failures lock")
                                .push(format!("{label} (spawn: {e})"));
                            stop.store(true, Ordering::SeqCst);
                            return;
                        }
                    }
                }
            });
        }
    });

    let failures = failures.into_inner().expect("failures lock");
    anyhow::ensure!(
        failures.is_empty(),
        "{lane} suite failed: {}",
        failures.join("; ")
    );
    Ok(())
}

/// The `daemon-conformance` detached-delegation trio (a known parallel-load flake, pass-in-isolation
/// = green) is NOT a vhc crate and NOT in this list, so it never gates the vhc tier.
fn vhc_ci_det() -> anyhow::Result<()> {
    run_lane_memoized(&workspace_root(), "vhc-ci-det", &[], || {
        // Dependency-direction invariant (architecture §7) first — cheap (metadata only) and
        // fails fast on a host/*->sdk/* regression before spending a compile.
        println!("\n== vhc-ci-det: daemon-vhc dependency-direction check ==");
        vhc_dep_check()?;
        // The tracked specification against the code it documents — cheap, and it fails before a
        // compile on a spec sentence a reader would act on and be wrong about.
        println!("\n== vhc-ci-det: tracked ABI specification vs the code ==");
        vhc_abi_spec_drift()?;
        build_guests()?;
        vhc_ci_det_suites(det_schedule_from_env())?;
        vhc_cross_lane()
    })
}

/// Cross-compile the fleet worker for the platforms this machine is not.
///
/// **In the mandatory aggregate because a platform arm that is never compiled is not gated.** The
/// Windows and macOS supply arms sat referencing a struct field that does not exist for four commits,
/// through two coordinator dispositions implemented into them, because nothing on a Linux build ever
/// type-checked that code and no lane ever cross-compiled it. The fixtures beside those arms now check
/// the arithmetic on every build; this checks that the arms themselves still assemble against the real
/// target's libraries and cfg set, which fixtures cannot do.
///
/// Sequential with the cargo suites, never beside them: one build at a time is the standing rule, and a
/// nix build stacked on a cargo build is the failure mode that rule exists for.
///
/// # Errors
/// The lane's own output when the cross build fails.
fn vhc_cross_lane() -> anyhow::Result<()> {
    println!("\n== vhc-ci-det: fleet cross-compile lane (x86_64-pc-windows-gnu) ==");
    let started = std::time::Instant::now();
    let status = std::process::Command::new("nix")
        .args([
            "build",
            "--max-jobs",
            "1",
            "--cores",
            "5",
            ".#daemon-vhc-worker-windows",
            "--no-link",
        ])
        .current_dir(workspace_root())
        .status()
        .map_err(|e| anyhow::anyhow!("run the windows cross lane: {e}"))?;
    anyhow::ensure!(
        status.success(),
        "the fleet worker does not cross-compile for x86_64-pc-windows-gnu; no fleet Windows worker \
         can be built from this revision"
    );
    println!(
        "vhc-ci-det: windows cross lane green in {}s",
        started.elapsed().as_secs()
    );
    // macOS has no cross target on this machine (the devShell carries wasm32 and linux-gnu only), so
    // the Metal arm is covered by the always-compiled mappers and their fixtures rather than by a
    // compiler run against Apple's libraries. Named here rather than left as a silence: it is the one
    // arm this lane does not reach, and closing it needs a darwin std in the shell.
    println!(
        "vhc-ci-det: macOS arm not cross-compiled (no darwin target in this shell); its mapping is \
         compiled and fixture-pinned on every platform instead"
    );
    Ok(())
}

/// The det-lane suite list + scheduler, WITHOUT the dep-check/guest preflight (so the production
/// gate can run the preflight once and overlap this lane with tier-2). The suite list and each
/// suite's pass/fail semantics are identical in every schedule — only ordering/overlap changes.
fn vhc_ci_det_suites(schedule: LaneSchedule) -> anyhow::Result<()> {
    // The node + supervisor lifecycle suites are part of the SAME mandatory aggregate (the
    // run-instance state machine and supervision are consensus-adjacent product paths, not a
    // side lane); `vhc-ci-node` also runs them standalone for focused iteration.
    let all: Vec<SuiteEntry<'_>> = VHC_DET_SUITES
        .iter()
        .chain(VHC_NODE_SUITES)
        .copied()
        .collect();
    run_lane_suites("vhc-ci-det", &all, schedule)?;
    println!("\nvhc-ci-det: all tier-1 (CPU consensus-critical) vhc suites green");
    Ok(())
}

/// The pinned det-lane suite list (label, cargo test args). Each suite runs in its own process;
/// the first red aborts the lane.
const VHC_DET_SUITES: &[SuiteEntry<'static>] = &[
        (
            "daemon-vhc-abi (journal §8.3 CDDL grammar validity + per-tag samples)",
            &["-p", "daemon-vhc-abi"],
        ),
        (
            // The det-lane kernels' tier-1 suite. This is the ONLY det implementation: the
            // former host-op ≡ in-guest-crate conformance lane (det_conformance) retired
            // WITH the tabi@1 bridge — the host-side det_* acceleration dispatch it compared
            // was the bridge's, and no host-executed det-kernel surface exists any more (the
            // compute@2 runner executes burn_ir ops, the tolerance-class native lane; det math
            // runs exclusively in-guest via this crate compiled to wasm). End-to-end det
            // coverage through the production trainer is the trainer-goldens digest equality.
            "daemon-vhc-det (the det-lane kernels — the single implementation)",
            &["-p", "daemon-vhc-det"],
        ),
        (
            "daemon-vhc-proto (wire mechanism: envelopes v1+v2, grants, canonical CBOR)",
            &["-p", "daemon-vhc-proto"],
        ),
        (
            // The host half of the three-object resource model: the composition planner and its
            // canonical vectors, the Backend Execution Profile and its trust envelope, profile
            // authentication and the candidate store, the Device Capability Report's supply
            // derivation, the governor's reservation arithmetic, and the composition-evidence
            // encoder/validator whose fail-closed behaviour gates certification evidence.
            //
            // In the mandatory lane because a resource-subsystem crate whose tests are skippable is
            // the gate-blindness failure the audit documented: every one of these suites decides
            // whether a claim is admitted, and none of them was run by the gate.
            "daemon-vhc-resource (composition, profiles + trust, capability supply, governor, evidence)",
            &["-p", "daemon-vhc-resource"],
        ),
        (
            // D0: assignment math moved out of the proto (refactor §8/D0). The golden vectors
            // (LCG stream, shuffle, quorum ladder, class weights) moved with it — this lane keeps
            // them tier-1 so any drift in the moved math stays a visible, deliberate break.
            "daemon-vhc-sdk-consensus (assignment math + golden vectors, moved at D0)",
            &["-p", "daemon-vhc-sdk-consensus"],
        ),
        (
            // `--features harness` compiles the harness-gated round machinery (engine /
            // checkpoint / upgrade / coordinator shell) into the integration-test lib build;
            // the default (production) session build carries none of it (dep-check-enforced).
            "daemon-vhc-session (harness + assess + replay, loopback)",
            &["-p", "daemon-vhc-session", "--features", "harness"],
        ),
        (
            "daemon-vhc-observe (MessageLog + replay oracle + desync tally)",
            &["-p", "daemon-vhc-observe"],
        ),
        (
            // A1 journal-soak gate (refactor G6 / Decision 4): the crash-safe segmented journal
            // (grammar conformance + per-tag round-trips), crash safety (torn-write/CRC/chain
            // recovery, seq never reused), and the coordinator oracle re-derived byte-identically
            // over the journal substrate. Additive + fast (re-runs already-built test binaries).
            "daemon-vhc-observe journal + input replay (grammar, crash-safety, oracle-over-journal)",
            &[
                "-p",
                "daemon-vhc-observe",
                "--test",
                "journal",
                "--test",
                "journal_crash",
                "--test",
                "journal_oracle",
            ],
        ),
        (
            "daemon-vhc-net (framing + dedupe codecs, no network)",
            &["-p", "daemon-vhc-net"],
        ),
        (
            // The v1 admission refusal is proven over SYNTHETIC inputs (retirement plan §3): the
            // recorded pre-refactor bundle retired once refusal coverage went synthetic. A
            // hand-assembled ABI-major-1 module (empty imports + the v1 lifecycle exports + a
            // `da_abi` declaring major 1) meets a clean typed `AbiUnsupportedMajor` at the §1.3
            // front door — the standing regression that v1 support is gone and gone gracefully.
            // The protocol twin over the real worker binary is `worker_protocol` (below).
            "daemon-vhc-host driver selection (ABI §1.3 typed refusals incl. synthetic major-1)",
            &["-p", "daemon-vhc-host", "--test", "driver_selection"],
        ),
        (
            // The A2 event-loop acceptance (refactor §5 A2): the non-round toy-averager guest
            // (timers + publish only) end-to-end under the real major-2 driver — selection
            // admits, da_init/da_run dispatch, §12.1 signed frames with durable seqs, journaled
            // through the real A1 substrate — plus the undeclared-channel GrantViolation
            // negative. Named as its own lane (like the driver-selection refusal lane) so the
            // standing expressiveness proof is visible; also covered by the host crate suite above.
            "A2 event loop (toy-averager expressiveness + typed channel trap)",
            &["-p", "daemon-vhc-host", "--test", "event_loop"],
        ),
        (
            // The A2 claim + admission-funnel acceptance (refactor §5 A2; §10 gate row "Claim
            // rejection / over-claim / under-claim traps"): over-claim vs owner policy (stage 5),
            // claim outside lane bounds (stage 4), ClaimInconsistent, GrantsExceedLane, the
            // attributable under-claim cap trap at run time, and claim determinism — all through
            // the real restricted assessment instance (test-claim-v2 guest).
            "A2 claim + admission funnel (over/under-claim, lane bounds, typed refusals)",
            &["-p", "daemon-vhc-host", "--test", "claim_funnel"],
        ),
        (
            // The input-replay step (refactor §5 A1→A2 acceptance; §12.6 journal soak for
            // v2): recorded runs (toy averager: timers/clock; bridge guest: nr readouts +
            // staged kinds 1/2) re-driven from the journal alone through the §8.7 verifier
            // (observe contract over the host replay engine) — every decision bit-for-bit;
            // tampered/incomplete journals are typed divergences. The compute@2 trainer's
            // journal-replay soak lives in the trainer-goldens lane below.
            "A2 input-replay: journal-only re-drive ≡ recorded decisions (§8.7)",
            &["-p", "daemon-vhc-host", "--test", "replay"],
        ),
        (
            // The sys@2 crypto-acceleration conformance gate (Phase B; architecture §3.2/§3.7,
            // refactor §6): the host `hash`/`verify_sig` accel bodies ≡ the dual-compiled
            // `daemon_vhc_proto::crypto` contract (the in-guest fallback is that same contract
            // compiled to wasm — bit-exact by construction, the det-lane pattern) over a wide
            // deterministic sweep + known-answer vectors + tri-state verify semantics. Named as
            // its own lane (also covered by the host crate suite below).
            "B2 sys@2 crypto accel conformance (host ≡ in-guest contract: hash/verify_sig)",
            &["-p", "daemon-vhc-host", "--test", "crypto"],
        ),
        (
            // The Phase-C custom-op registry gate (architecture §3.2, refactor §7): versioned
            // named fused kernels register host-side (flash_attn@1 the first entry); a manifest
            // requiring an op the host does not advertise is refused CLEANLY (typed
            // CustomOpUnsupported, never a trap). Pins the shared ABI vocabulary (the seam C1's
            // compute@2 OperationIr::Custom resolves through) + the registry admission behaviour.
            "C2 custom-op registry (flash_attn@1; typed refusal on absent required op)",
            &["-p", "daemon-vhc-host", "--test", "custom_op"],
        ),
        (
            // The Phase-C MODEL-AGNOSTIC acceptance (refactor §7: "a non-LLaMA toy authored with
            // zero host changes … proving the compute ABI is model-agnostic"): the `toy-mlp` guest
            // — a two-layer MLP trained by SGD, authored purely over daemon-vhc-sdk-compute +
            // daemon-vhc-sdk — runs against the SAME compute@2 runner/driver/journal as the
            // LLaMA reference, exports a trained weight bit-exact vs a native Autodiff<NdArray> run
            // of the identical loop, and replays bit-for-bit (§8.7). No host code is model-specific.
            "model-agnostic compute@2 (toy-mlp: distinct model, zero host changes, bit-exact + replay)",
            &["-p", "daemon-vhc-host", "--test", "toy_mlp"],
        ),
        (
            // The Phase-C compute-REPLAY tier (refactor §7: "compute replay … the second of the
            // three replay tiers"; architecture §3.6 "compute replay, tolerance-equivalent"): a
            // recorded compute@2 op-journal, re-executed against a (possibly different) backend,
            // reproduces the trajectory within the native lane's tolerance class. The tier-1 lane
            // pins the ndarray↔ndarray DEGENERATE case (tolerance 0 — bit-exact — so the harness
            // itself is always exercised); the wgpu cross-backend tier is hardware-gated in the
            // same file.
            "compute replay (ndarray↔ndarray degenerate: same op-journal, bit-exact re-execution)",
            &["-p", "daemon-vhc-host", "--test", "compute_replay"],
        ),
        (
            // The B2 data@2 fetch conformance (architecture §3.2 the data world): the corpus
            // window fetched by committed hash + policy-chosen range, completing Ok(BufferHandle)
            // (tag 6) after whole-artifact verification; grant negative (GrantViolation),
            // range negative (StoreRefused), tamper negative (HashMismatch — fetch-and-verify
            // against the committed hash); tag-14 journaling + bit-exact replay with artifacts
            // materialized from the content-addressed payload table.
            "B2 data@2 fetch conformance (grants, pinning, range, journal + replay)",
            &["-p", "daemon-vhc-host", "--test", "data_fetch"],
        ),
        (
            "daemon-vhc-host (det lane + cross-backend digests + the driver suites)",
            &["-p", "daemon-vhc-host", "--features", "burn-ndarray"],
        ),
        (
            // The frozen worker protocol over the REAL `daemon-vhc-worker` binary (probe → assess
            // → join → one self-driven round; envelope seam; preemption churn). Lived inside the
            // `daemon-vhc-host` suite above until the A2 worker-bin split moved the bin (and its
            // CARGO_BIN_EXE-spawning test) to `crates/vhc/bins/daemon-vhc-worker`; same features
            // as before the split (burn-ndarray forwards into the host lib), so coverage is
            // unchanged.
            // `harness` builds the binary WITH the in-process self-driven join so the join suite
            // exercises it; the shipped default worker refuses JoinRun typed and links no SDK
            // schema crate (the dep-check's negative architecture test pins that).
            "daemon-vhc-worker (frozen worker protocol over the real binary)",
            &[
                "-p",
                "daemon-vhc-worker",
                "--features",
                "burn-ndarray,harness",
            ],
        ),
        (
            // The profiles gate (refactor §7 "profiles re-express over Burn tensors + det
            // math"): SparseLoco/DiLoCo/Demo reproduce their pinned goldens bit-for-bit (pinned
            // trajectory inputs → pinned post-ingest outputs — the standing literals that carry
            // the retired reference implementation's equivalence proof), and the Section payload
            // wire is byte-identical to the container encoding the committed payloads ride.
            "sdk-profiles pinned goldens (bit-exact) + payload wire",
            &["-p", "daemon-vhc-sdk-profiles"],
        ),
        (
            // The native trainer goldens (retirement plan §3): the compute@2 trainer guest
            // (`tiny-llama`) reproduces a recorded, content-addressed golden bundle (per-round
            // det digests, the trainer's own committed payload bytes, the matched-init
            // trained-theta trajectory). This is the SUCCESSOR drift oracle that superseded and
            // retired the recorded v1 parity oracle (the retired trainer-parity lanes, D-3): the
            // det lane is an equality class (digests reproduce bit-for-bit through the full
            // wasm32 + CBOR + driver path) and the native lane a tolerance class (theta within
            // the OpClass::Optimizer band), plus the straggle -> catch-up leg (ported from
            // parity) and a checkpoint/migration continuity pin. wgpu/cuda device tiers are
            // hardware-gated in the same file (op-journal replay of the compute@2 kernels, with a
            // bit-exact ndarray self-check of the recording).
            "trainer goldens: native det digests + Optimizer band + straggle catch-up (cpu + burn-ndarray)",
            &[
                "-p",
                "daemon-vhc-host",
                "--test",
                "trainer_goldens",
                "--features",
                "burn-ndarray",
            ],
        ),
        (
            // A2 migrate/main! scaffolding (refactor §5 A2 item 4; ABI §10): state round-trips
            // in sim through the typed manifest protocol; the SDK-derived claim/manifest match
            // the §9.1/§6.2 wire schema the admission funnel decodes. The macro's exports are
            // exercised for real by the compute@2 trainer guest in the whole-run suites.
            "daemon-vhc-sdk (main!/migrate scaffolding: sim round-trips + derivations)",
            &["-p", "daemon-vhc-sdk"],
        ),
        (
            // The worker binary's own integration drills: the framed command protocol, join refusal
            // without a provisioned identity, the command loop across join/leave/shutdown, module
            // switch with journal continuity, the live attach, and the seat smoke.
            //
            // In the mandatory lane because leaving it out is how a red suite survived: these drills
            // sat failing for two sittings — every one of them refused at its first line by a genesis
            // fixture the authoring migration had superseded — while the gate ran the crates whose
            // tests were fast. A suite nobody runs is not coverage, and the drills are the only place
            // the framed protocol and the durable journal are exercised through the real binary.
            "daemon-vhc-worker (framed protocol, join refusal, command loop, switch, live attach)",
            &["-p", "daemon-vhc-worker"],
        ),
        (
            "daemon-vhc-e2e (drills + observe-replay, no iroh/live)",
            &["-p", "daemon-vhc-e2e"],
        ),
        (
            "daemon-api conformance (serde wire ↔ CDDL, pos+neg)",
            &["-p", "daemon-api", "--test", "conformance"],
        ),
        (
            "daemon-api conformance_proptest (arbitrary values ↔ CDDL)",
            &[
                "-p",
                "daemon-api",
                "--features",
                "arbitrary",
                "--test",
                "conformance_proptest",
            ],
        ),
];

/// The node + supervisor lifecycle suites (one list, two entry points): the `daemon-vhc-node`
/// suites cover the durable run-instance state machine (terminal transitions with observed
/// teardown ordering, generation gating, the bounded retry budget, crash-window repair,
/// pause/resume, restart reconvergence), the resident reconciliation + seat-keeper passes, and
/// owner arbitration; the `daemon-vhc-supervisor` suites cover spawn → handshake → respawn →
/// crash-loop meltdown and the event-pump observability contract over the REAL scripted worker
/// subprocess (`fake-train-worker`).
const VHC_NODE_SUITES: &[(&str, &[&str])] = &[
    (
        "daemon-vhc-node (run-instance state machine + reconciliation + seat keeper + arbitration)",
        &["-p", "daemon-vhc-node"],
    ),
    (
        "daemon-vhc-supervisor (spawn/respawn/meltdown + stream observability, real subprocess)",
        &["-p", "daemon-vhc-supervisor"],
    ),
];

/// Run the node + supervisor lifecycle suites standalone (the same list `vhc-ci-det` folds in —
/// D-P5: the node lane is mandatory in the det aggregate, this entry point is for iteration).
fn vhc_ci_node() -> anyhow::Result<()> {
    let root = workspace_root();
    for (label, args) in VHC_NODE_SUITES {
        println!("\n== vhc-ci-node: {label} ==");
        let status = Command::new("cargo")
            .current_dir(&root)
            .arg("test")
            .args(*args)
            .status()
            .map_err(|e| anyhow::anyhow!("running cargo test {args:?}: {e}"))?;
        anyhow::ensure!(
            status.success(),
            "vhc node/supervisor suite failed: {label}"
        );
    }
    println!("\nvhc-ci-node: node + supervisor lifecycle suites green");
    Ok(())
}

/// Run the vhc **CI tier-2** whole-run suites (decisions D4; refactor §6, §10 gate table).
///
/// The two-layer simulation split (architecture §6): SDK-side `daemon-vhc-sim` runs NATIVE policy
/// code (the SPARTA continuous-averaging toy over the virtual worlds — deterministic whole run),
/// and host-side `daemon-vhc-testkit` runs the PRODUCTION wasm blobs under wasmtime + simulated
/// capability providers, journaled and §8.7 replay-verified. This is heavier than tier-1 (it builds
/// the wasm guests + compiles wasmtime), so it is a separate gate — never folded into
/// `vhc-ci-det`, which stays the CPU-only deterministic tier-1 bar.
fn vhc_ci_t2() -> anyhow::Result<()> {
    run_lane_memoized(&workspace_root(), "vhc-ci-t2", &[], || {
        // Same dependency-direction preflight as tier-1, then the guests the testkit runs.
        println!("\n== vhc-ci-t2: daemon-vhc dependency-direction check ==");
        vhc_dep_check()?;
        build_guests()?;
        vhc_ci_t2_suites(LaneSchedule::serial())
    })
}

/// The tier-2 suite list, WITHOUT the dep-check/guest preflight (see `vhc_ci_det_suites` — same
/// split, same "scheduling only, never coverage" contract).
fn vhc_ci_t2_suites(schedule: LaneSchedule) -> anyhow::Result<()> {
    run_lane_suites("vhc-ci-t2", VHC_T2_SUITES, schedule)?;
    println!("\nvhc-ci-t2: all tier-2 (sim/testkit) whole-run suites green");
    Ok(())
}

/// The pinned tier-2 suite list (label, cargo test args).
const VHC_T2_SUITES: &[SuiteEntry<'static>] = &[
    (
        // SDK-side native whole run (architecture §6): the SPARTA-shaped continuous-averaging
        // toy (timers + gossip, no rounds, no coordinator) converges over the virtual worlds;
        // the run is bit-for-bit deterministic (the SDK-side analogue of §8.7 input replay) and
        // stays bounded under a lossy/churn trace.
        "daemon-vhc-sim (SDK-side native whole run: SPARTA averager over the virtual worlds)",
        &["-p", "daemon-vhc-sim"],
    ),
    (
        // Host-side whole runs over the PRODUCTION blobs: wasmtime + simulated capability
        // providers, journaled end-to-end, re-driven through the §8.7 input-replay engine —
        // every decision reproduced bit-for-bit. Covers the toy_averager whole run, the
        // compute@2 trainer barrier whole runs under the production coordinator
        // (single- and 2-worker with cross-worker agreement over the guest-voiced det
        // digests; SDK-free raw-CBOR config), the adversarial-rig pinned cases (duplicate
        // record deduped; delayed payloads → stall → catch-up), the mixed-fleet matrix
        // cells (the whole-run positive + the envelope-v1 typed negatives), the pump-hold
        // back-pressure rig, and the failover drill.
        "daemon-vhc-testkit (production-blob whole runs + D2 coordinator lanes)",
        &["-p", "daemon-vhc-testkit"],
    ),
    (
        // The real-geometry round WITH the training math (`ceremony_training_step`): the frozen
        // ceremony parameter set actually stepped through forward/backward/AdamW before the θ
        // export and commit — the one seam the training-free round walk and the toy-geometry
        // trainer goldens leave between them. RELEASE, and `--ignored`, because the optimizer
        // steps are host fp32 matmuls: 229 s here against >9 min under the test profile, which
        // would make this lane's wall a property of the profile rather than of the code (the
        // suite's own `#[ignore]` reason records the measurement).
        "daemon-vhc-testkit ceremony training-step lane (real geometry WITH the optimizer, release)",
        &[
            "-p",
            "daemon-vhc-testkit",
            "--release",
            "--test",
            "ceremony_training_step",
            "--",
            "--ignored",
            "--nocapture",
        ],
    ),
    (
        // The MODULE-DRIVEN corpus lane (`ceremony_live_staging`): the trainer feeds ITSELF —
        // manifest fetch, per-shard chunk registration, the round's own segment plan, thirty
        // `data@2` ranges over one covering chunk, and the staged batches trained through at the
        // FROZEN `(seq_len, vocab)`. Every other lane hands the guest host-staged batches, so this
        // is the only place the `live` contract the fleet genesis pins is ever executed. Same
        // release/`--ignored` reason as the training-step lane above.
        "daemon-vhc-testkit live corpus staging lane (module-driven data plane, release)",
        &[
            "-p",
            "daemon-vhc-testkit",
            "--release",
            "--test",
            "ceremony_live_staging",
            "--",
            "--ignored",
            "--nocapture",
        ],
    ),
    (
        // D2's CONSENSUS REPLAY — the third replay tier (architecture §3.6; refactor §10 gate
        // row "Consensus replay from archive alone", tier-2): a third party re-verifies every
        // consensus decision and every digest from the record archive's signed, hash-chained,
        // content-addressed sealed segments + the content-addressed payloads ALONE (no live
        // journal, no coordinator). Positive + the typed incompleteness negatives (missing
        // payload, withheld segment, forged/gappy heads).
        "D2 consensus replay (third tier: digests re-verified from archive + payloads alone)",
        &["-p", "daemon-vhc-observe", "--test", "consensus_replay"],
    ),
];

/// Run the multi-process acceptance suite (spec §7): three REAL `daemon` node processes on the
/// full product path. Builds the node + worker binaries in RELEASE first — a debug-compiled
/// wasmtime instantiation of the multi-layer trainer module exceeds the supervisor's 30s assess
/// watchdog, so the acceptance suite spawns the release binaries (located via
/// `VHC_ACCEPTANCE_BIN_DIR`, the durable finding baked into the lane). The suite itself is a
/// debug test binary; only the spawned product binaries are release.
///
/// This lane is **C0 — the ceremony ladder's one-box rung** (`docs/vhc-program-state.md` §5,
/// ceremony runbook): coordinator + two
/// trainers as three real processes at the pinned structural geometry, with digest agreement,
/// checkpoint restore, churn/hard-kill drills, and a live module switch. It runs on every merge as
/// the production gate's non-negotiable core beside the det suites; the higher rungs (C1 two-box,
/// C2 the fleet ceremony) are strict supersets and never the first place a defect can surface.
fn vhc_acceptance() -> anyhow::Result<()> {
    let root = workspace_root();
    run_lane_memoized(&root, "vhc-acceptance", &[], || {
        build_guests()?;

        let bin_dir = acceptance_release_bins(&root)?;
        println!(
            "\n== vhc-acceptance: multi-process gates (bin dir {}) ==",
            bin_dir.display()
        );
        let status = Command::new("cargo")
            .current_dir(&root)
            .env("VHC_ACCEPTANCE_BIN_DIR", &bin_dir)
            .args(["test", "-p", "daemon-vhc-acceptance"])
            .status()
            .map_err(|e| anyhow::anyhow!("running the acceptance suite: {e}"))?;
        anyhow::ensure!(
            status.success(),
            "vhc-acceptance: a multi-process gate failed"
        );
        println!("\nvhc-acceptance: all multi-process gates green");
        Ok(())
    })
}

/// The exact release build the acceptance lane spawns (see `vhc_acceptance` — debug wasmtime
/// instantiation exceeds the assess watchdog, so the PRODUCT binaries are release).
const ACCEPTANCE_BUILD_ARGS: &[&str] = &[
    "build",
    "--release",
    "-p",
    "daemon",
    "-p",
    "daemon-vhc-worker",
    "--features",
    "daemon-vhc-worker/vhc-net,daemon-vhc-worker/burn-ndarray",
];

/// The product binaries that build emits and the acceptance harness spawns.
const ACCEPTANCE_BINS: &[&str] = &["daemon", "daemon-vhc-worker"];

/// Produce the acceptance lane's release product binaries, reusing a source-keyed cache when the
/// workspace is byte-identical to a previously built state (the release rebuild is ~12 min of
/// every acceptance run; across worktrees of the same commit it is pure waste).
///
/// Soundness: the cache key hashes EVERY input that feeds the binaries — the HEAD commit id
/// (which also pins the embedded `git describe` build-metadata suffix), the content of every
/// tracked-but-modified AND untracked (non-ignored) file in the working tree (deletions
/// included), the exact build args, the toolchain (`rustc -vV`), and the ambient env knobs that
/// alter codegen (`RUSTFLAGS` family, `DAEMON_BUILD_ID`). Any input this key cannot prove
/// (key computation failing, cache disabled, no `HOME`) falls back to an unconditional build —
/// a stale binary is impossible, the failure mode is only a redundant rebuild.
///
/// `VHC_ACCEPTANCE_BIN_CACHE=0` disables the cache; any other value overrides the cache dir
/// (default `$HOME/.cache/vhc-acceptance-bins`). Entries are pruned to the 8 most recent.
fn acceptance_release_bins(root: &Path) -> anyhow::Result<PathBuf> {
    let fresh = root.join("target").join("release");
    let Some(cache_root) = acceptance_cache_root() else {
        println!("\n== vhc-acceptance: binary cache disabled — building release binaries ==");
        build_acceptance_bins(root)?;
        return Ok(fresh);
    };
    let key = match acceptance_source_key(root) {
        Ok(key) => key,
        Err(e) => {
            // No provable key (e.g. not a git checkout) -> never guess, always build.
            println!(
                "\n== vhc-acceptance: no sound cache key ({e:#}) — building release binaries =="
            );
            build_acceptance_bins(root)?;
            return Ok(fresh);
        }
    };
    let entry = cache_root.join(&key);
    if ACCEPTANCE_BINS.iter().all(|b| entry.join(b).is_file()) {
        println!(
            "\n== vhc-acceptance: release binaries cache HIT ({key}) — skipping the release build =="
        );
        // Refresh the entry's recency (a file write bumps the dir mtime the pruner sorts by).
        let _ = std::fs::write(
            entry.join("last-used"),
            format!("{:?}\n", std::time::SystemTime::now()),
        );
        return Ok(entry);
    }
    println!("\n== vhc-acceptance: release binaries cache MISS ({key}) — building ==");
    build_acceptance_bins(root)?;
    // Populate atomically: stage into a tmp dir, then rename into place. A concurrent populator
    // of the same key loses the rename race harmlessly (the winner's binaries are, by key
    // construction, byte-equivalent inputs).
    let staging = cache_root.join(format!(".tmp-{}-{}", key, std::process::id()));
    std::fs::create_dir_all(&staging)
        .map_err(|e| anyhow::anyhow!("create cache staging dir {}: {e}", staging.display()))?;
    for bin in ACCEPTANCE_BINS {
        let from = fresh.join(bin);
        let to = staging.join(bin);
        std::fs::copy(&from, &to)
            .map_err(|e| anyhow::anyhow!("cache {} -> {}: {e}", from.display(), to.display()))?;
    }
    match std::fs::rename(&staging, &entry) {
        Ok(()) => {}
        Err(_) if ACCEPTANCE_BINS.iter().all(|b| entry.join(b).is_file()) => {
            let _ = std::fs::remove_dir_all(&staging);
        }
        Err(e) => {
            let _ = std::fs::remove_dir_all(&staging);
            anyhow::bail!("publish cache entry {}: {e}", entry.display());
        }
    }
    prune_acceptance_cache(&cache_root, 8);
    Ok(entry)
}

/// Run the pinned acceptance release build (the historical unconditional path).
fn build_acceptance_bins(root: &Path) -> anyhow::Result<()> {
    println!("\n== vhc-acceptance: building the node + worker product binaries (release) ==");
    let build = Command::new("cargo")
        .current_dir(root)
        .args(ACCEPTANCE_BUILD_ARGS)
        .status()
        .map_err(|e| anyhow::anyhow!("building acceptance product binaries: {e}"))?;
    anyhow::ensure!(
        build.success(),
        "vhc-acceptance: product binary build failed"
    );
    Ok(())
}

/// The acceptance binary cache root, or `None` when disabled (`VHC_ACCEPTANCE_BIN_CACHE=0`, or
/// no `HOME` to anchor the default under).
fn acceptance_cache_root() -> Option<PathBuf> {
    match std::env::var("VHC_ACCEPTANCE_BIN_CACHE") {
        Ok(v) if v == "0" => None,
        Ok(v) if !v.is_empty() => Some(PathBuf::from(v)),
        _ => std::env::var_os("HOME").map(|home| {
            PathBuf::from(home)
                .join(".cache")
                .join("vhc-acceptance-bins")
        }),
    }
}

/// Hash every input that feeds the acceptance release binaries into one cache key (see
/// `acceptance_release_bins` for the soundness argument). Fails (rather than guessing) when the
/// workspace state cannot be proven — the caller then builds unconditionally.
fn acceptance_source_key(root: &Path) -> anyhow::Result<String> {
    let mut hasher = workspace_fingerprint_hasher(root)?;
    hasher.update(b"args=");
    hasher.update(ACCEPTANCE_BUILD_ARGS.join(" ").as_bytes());
    Ok(hasher.finalize().to_hex().to_string())
}

/// A blake3 hasher primed with the COMPLETE workspace state that can feed a build or test
/// outcome: the HEAD commit + `git describe` (pins the embedded build metadata), the
/// working-tree content of every tracked-but-modified and untracked non-ignored file (deletions
/// included), the toolchain (`rustc -vV`), and the codegen env knobs. Fails (rather than
/// guessing) when that state cannot be proven — callers then fall back to doing the work.
fn workspace_fingerprint_hasher(root: &Path) -> anyhow::Result<blake3::Hasher> {
    let git = |args: &[&str]| -> anyhow::Result<String> {
        let out = Command::new("git")
            .current_dir(root)
            .args(args)
            .output()
            .map_err(|e| anyhow::anyhow!("git {args:?}: {e}"))?;
        anyhow::ensure!(
            out.status.success(),
            "git {args:?} failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        Ok(String::from_utf8_lossy(&out.stdout).into_owned())
    };

    let mut hasher = blake3::Hasher::new();
    hasher.update(b"commit=");
    hasher.update(git(&["rev-parse", "HEAD"])?.as_bytes());
    // The build-metadata suffix daemon-common's build.rs embeds (tag moves change it without a
    // commit change, so hash the description itself, not just HEAD).
    hasher.update(b"describe=");
    hasher.update(git(&["describe", "--always", "--dirty"])?.as_bytes());

    // Every divergence from HEAD: tracked modifications/deletions (staged or not) and untracked
    // non-ignored files. Hash the WORKING-TREE content — that is what cargo compiles.
    hasher.update(b"status=");
    let status = git(&["status", "--porcelain=v1", "-uall", "--no-renames"])?;
    for line in status.lines() {
        // Format: `XY <path>` (v1, no renames). Hash the path plus its current content.
        let Some(path) = line.get(3..) else { continue };
        // git quotes paths with special characters; un-escaping them here is not worth the
        // soundness risk (a mis-resolved path would hash the wrong content) — refuse the key
        // and let the caller build unconditionally.
        anyhow::ensure!(
            !path.starts_with('"'),
            "dirty path needs git unquoting: {path}"
        );
        hasher.update(path.as_bytes());
        hasher.update(b"=");
        match std::fs::read(root.join(path)) {
            Ok(bytes) => hasher.update(&bytes),
            Err(_) => hasher.update(b"<absent>"),
        };
        hasher.update(b";");
    }

    // Toolchain + codegen environment.
    let rustc = Command::new("rustc")
        .arg("-vV")
        .output()
        .map_err(|e| anyhow::anyhow!("rustc -vV: {e}"))?;
    anyhow::ensure!(rustc.status.success(), "rustc -vV failed");
    hasher.update(b"rustc=");
    hasher.update(&rustc.stdout);
    for var in [
        "RUSTFLAGS",
        "CARGO_ENCODED_RUSTFLAGS",
        "CARGO_BUILD_RUSTFLAGS",
        "DAEMON_BUILD_ID",
    ] {
        hasher.update(var.as_bytes());
        hasher.update(b"=");
        hasher.update(std::env::var(var).unwrap_or_default().as_bytes());
        hasher.update(b";");
    }

    Ok(hasher)
}

/// The green-run ledger: one file per lane under the (gitignored, repo-local) `target/`, holding
/// the workspace fingerprint of that lane's last green run. Committed history has already been
/// gated; a lane requested again for a byte-identical workspace re-verifies nothing, so it is
/// skipped and reported as memoized. `VHC_GATE_MEMO=0` disables both reading and recording.
fn lane_ledger_path(root: &Path, lane: &str) -> PathBuf {
    root.join("target").join("vhc-green-ledger").join(lane)
}

/// The memo key for a lane: the workspace fingerprint plus any lane-specific identity (e.g. the
/// diff-scoped gate folds its selected suite set in). `None` = memoization unavailable (disabled
/// via env, or the workspace state cannot be proven) — the lane then just runs.
fn lane_memo_key(root: &Path, extra: &[u8]) -> Option<String> {
    if memoization_disabled() {
        return None;
    }
    let mut hasher = workspace_fingerprint_hasher(root).ok()?;
    hasher.update(b"lane-extra=");
    hasher.update(extra);
    Some(hasher.finalize().to_hex().to_string())
}

/// Whether lane memoization is switched off for this process.
///
/// Two switches, and the difference matters. `VHC_GATE_MEMO=0` is the developer's: ambient, convenient,
/// and exactly the kind of thing that is set in one shell and forgotten in another. `--no-memo` is the
/// **certification** switch: it is written in the command an operator runs and recorded in the evidence
/// beside its verdict, so a full-scope battery cannot be satisfied by a ledger entry from an earlier
/// tree. A certification claim that rests on "the fingerprint matched last time" rests on the ledger
/// rather than on the run, and the whole point of the full battery is that it ran.
fn memoization_disabled() -> bool {
    std::env::var("VHC_GATE_MEMO").is_ok_and(|v| v == "0")
        || NO_MEMO.load(std::sync::atomic::Ordering::Relaxed)
}

/// Set by `--no-memo` before any lane runs; never cleared.
static NO_MEMO: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Does the ledger record a green run of `lane` for exactly this memo key?
fn lane_memo_green(root: &Path, lane: &str, key: &Option<String>) -> bool {
    let Some(key) = key else { return false };
    std::fs::read_to_string(lane_ledger_path(root, lane)).is_ok_and(|s| s.trim() == key)
}

/// Record a green run of `lane` (best-effort — a write failure only costs a future re-run).
fn lane_record_green(root: &Path, lane: &str, key: &Option<String>) {
    let Some(key) = key else { return };
    let path = lane_ledger_path(root, lane);
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let _ = std::fs::write(path, format!("{key}\n"));
}

/// Run `lane` through the green ledger: skip (and say so) when the workspace fingerprint matches
/// the lane's last green run, record the fingerprint after a fresh green.
fn run_lane_memoized(
    root: &Path,
    lane: &str,
    extra: &[u8],
    run: impl FnOnce() -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    let key = lane_memo_key(root, extra);
    if lane_memo_green(root, lane, &key) {
        println!(
            "\n== {lane}: green (memoized — workspace identical to the last green run; \
             VHC_GATE_MEMO=0 forces a re-run) =="
        );
        return Ok(());
    }
    run()?;
    lane_record_green(root, lane, &key);
    Ok(())
}

/// Keep the newest `keep` cache entries (by directory mtime — bumped on every hit), delete the
/// rest. Best-effort: pruning failures never fail the lane.
fn prune_acceptance_cache(cache_root: &Path, keep: usize) {
    let Ok(entries) = std::fs::read_dir(cache_root) else {
        return;
    };
    let mut dirs: Vec<(std::time::SystemTime, PathBuf)> = entries
        .flatten()
        .filter(|d| d.file_type().is_ok_and(|t| t.is_dir()))
        .filter(|d| !d.file_name().to_string_lossy().starts_with(".tmp-"))
        .filter_map(|d| {
            let mtime = d.metadata().ok()?.modified().ok()?;
            Some((mtime, d.path()))
        })
        .collect();
    dirs.sort_by_key(|(mtime, _)| std::cmp::Reverse(*mtime));
    for (_, dir) in dirs.into_iter().skip(keep) {
        let _ = std::fs::remove_dir_all(dir);
    }
}

/// The single merge gate (D-P10): the deterministic tier-1 aggregate (which already folds
/// `vhc-ci-node` + the dependency-direction / negative-architecture check), the tier-2 whole-run
/// suites, and the multi-process acceptance suite. Every integration branch passes this
/// aggregate before it merges; nothing merges on the deterministic subset alone.
fn vhc_production_gate(
    all: bool,
    base: Option<String>,
    dry_run: bool,
    no_memo: bool,
) -> anyhow::Result<()> {
    if no_memo {
        // Before any lane consults the ledger. Recorded in the output as well as taken, so the verdict
        // says which mode produced it — a full-scope claim and a memoized one are different claims.
        NO_MEMO.store(true, std::sync::atomic::Ordering::Relaxed);
        println!("vhc-production-gate: --no-memo — the green ledger is ignored; every lane runs");
    }
    if all {
        if dry_run {
            println!(
                "vhc-production-gate (--all, dry run): would run the FULL battery — {} suite(s) + acceptance",
                VHC_DET_SUITES.len() + VHC_NODE_SUITES.len() + VHC_T2_SUITES.len()
            );
            return Ok(());
        }
        vhc_production_gate_all()
    } else {
        vhc_gate_diff(base, dry_run)
    }
}

/// The FULL pinned battery — the `just lint-all` analogue: a manual pre-release / post-rebase
/// pass, deliberately wired into no workflow (the default gate entry point is diff-scoped, and
/// the fail-closed mappings escalate here on their own when a global input changes).
fn vhc_production_gate_all() -> anyhow::Result<()> {
    let gate_started = std::time::Instant::now();
    let parallel = std::env::var("VHC_GATE_PARALLEL").is_ok_and(|v| v == "1");
    println!(
        "== vhc-production-gate (--all): vhc-ci-det + vhc-ci-t2 + vhc-acceptance{} ==",
        if parallel { " (det ∥ t2)" } else { "" }
    );
    if parallel {
        // Opt-in bounded lane overlap (`VHC_GATE_PARALLEL=1`): the det and t2 lanes are both
        // dominated by serially-executed test binaries, so overlapping them recovers idle cores
        // while the per-lane libtest caps keep the TOTAL at det (3 groups x 4 threads = 12) +
        // t2 (1 x 4) = 16 threads. The shared dep-check + guest-build preflight runs ONCE up
        // front (both lanes start with the identical preflight; running it twice concurrently
        // would race the guest-manifest write). The heavyweight acceptance lane (release build +
        // three real node processes per gate) stays strictly after both. Default (env unset)
        // remains the fully serial historical order.
        let root = workspace_root();
        let memo_key = lane_memo_key(&root, &[]);
        let det_green = lane_memo_green(&root, "vhc-ci-det", &memo_key);
        let t2_green = lane_memo_green(&root, "vhc-ci-t2", &memo_key);
        for (lane, green) in [("vhc-ci-det", det_green), ("vhc-ci-t2", t2_green)] {
            if green {
                println!(
                    "\n== {lane}: green (memoized — workspace identical to the last green run) =="
                );
            }
        }
        if !det_green || !t2_green {
            println!("\n== vhc-production-gate: shared preflight (dep-check + guests) ==");
            vhc_dep_check()?;
            build_guests()?;
        }
        let det_schedule = det_schedule_from_env();
        let t2_schedule = LaneSchedule {
            groups: 1,
            test_threads: Some(4),
        };
        match (det_green, t2_green) {
            (true, true) => {}
            (false, true) => {
                vhc_ci_det_suites(det_schedule)?;
                lane_record_green(&root, "vhc-ci-det", &memo_key);
            }
            (true, false) => {
                vhc_ci_t2_suites(t2_schedule)?;
                lane_record_green(&root, "vhc-ci-t2", &memo_key);
            }
            (false, false) => {
                let (det_result, t2_result) = std::thread::scope(|scope| {
                    let det = scope.spawn(move || {
                        let started = std::time::Instant::now();
                        let result = vhc_ci_det_suites(det_schedule);
                        (result, started.elapsed())
                    });
                    let t2_started = std::time::Instant::now();
                    let t2 = (vhc_ci_t2_suites(t2_schedule), t2_started.elapsed());
                    (det.join().expect("det lane thread panicked"), t2)
                });
                println!(
                    "\nvhc-production-gate: det lane {:.0?}, t2 lane {:.0?}",
                    det_result.1, t2_result.1
                );
                if det_result.0.is_ok() {
                    lane_record_green(&root, "vhc-ci-det", &memo_key);
                }
                if t2_result.0.is_ok() {
                    lane_record_green(&root, "vhc-ci-t2", &memo_key);
                }
                // Report BOTH lanes before failing so a double-red run shows both reds.
                match (det_result.0, t2_result.0) {
                    (Ok(()), Ok(())) => {}
                    (det, t2) => {
                        if let Err(e) = &det {
                            eprintln!("vhc-production-gate: det lane RED: {e:#}");
                        }
                        if let Err(e) = &t2 {
                            eprintln!("vhc-production-gate: t2 lane RED: {e:#}");
                        }
                        det?;
                        t2?;
                    }
                }
            }
        }
    } else {
        vhc_ci_det()?;
        vhc_ci_t2()?;
    }
    let acceptance_started = std::time::Instant::now();
    vhc_acceptance()?;
    println!(
        "\nvhc-production-gate: GREEN (det + t2 + node + acceptance; acceptance {:.0?}, total {:.0?})",
        acceptance_started.elapsed(),
        gate_started.elapsed()
    );
    Ok(())
}

/// The packages whose test suites LOAD the built guest `.wasm` bytes at run time (the
/// `ensure_built()` harness copies + the replay sandbox + the acceptance product path). A change
/// under `crates/vhc/guests/` compiles into no host crate — it changes these suites' runtime
/// inputs — so the diff-scoped gate maps guests changes to exactly this set.
const GUEST_LINKED_PACKAGES: &[&str] = &[
    "daemon-vhc-host",
    "daemon-vhc-testkit",
    "daemon-vhc-session",
    "daemon-vhc-worker",
    "daemon-vhc-acceptance",
];

/// Path prefixes that escalate the diff-scoped gate to the FULL battery: they alter the gate
/// itself, the dependency resolution, the toolchain, or vendored sources — inputs whose blast
/// radius no crate cone bounds.
const FULL_GATE_PREFIXES: &[&str] = &["xtask/", ".cargo/", "vendor/"];

/// Root files with the same "no crate cone bounds this" property (the root `Cargo.toml` carries
/// the workspace dependency + lint tables; the flake pins the devShell toolchain).
const FULL_GATE_FILES: &[&str] = &[
    "Cargo.toml",
    "Cargo.lock",
    "flake.nix",
    "flake.lock",
    "rust-toolchain.toml",
];

/// Paths that cannot alter any test outcome: documentation, licensing metadata, and the
/// lint-layer configs (they gate `just lint` / cargo-deny, which have their own diff-scoped
/// runners — never `cargo test`).
const GATE_IGNORED_PREFIXES: &[&str] = &["docs/", ".plans/", "LICENSES/"];
const GATE_IGNORED_FILES: &[&str] = &[
    "README.md",
    "AGENTS.md",
    "NOTICE",
    "LICENSE-APACHE",
    "LICENSE-MIT",
    "REUSE.toml",
    ".gitignore",
    ".gitleaks.toml",
    "typos.toml",
    "deny.toml",
    "clippy.toml",
    "rustfmt.toml",
];

/// How one changed path maps into the gate.
enum GateInput {
    /// Cannot alter a test outcome — selects nothing.
    Ignored,
    /// A guests-workspace change — selects every guest-linked suite.
    Guests,
    /// Maps to a workspace crate — seeds the reverse-dependency cone.
    Crate(String),
    /// A global input (or an unmapped path — fail CLOSED) — escalates to the full battery.
    FullGate(String),
}

/// Classify one repo-relative changed path (see the constants above for each bucket's rationale;
/// order matters: guests and the full-gate globals are carved out before crate mapping, and the
/// terminal arm fails CLOSED so an unmapped path can only over-select).
fn classify_gate_input(path: &str, crate_dirs: &[(String, String)]) -> GateInput {
    if path.starts_with("crates/vhc/guests/") {
        return GateInput::Guests;
    }
    if FULL_GATE_PREFIXES.iter().any(|p| path.starts_with(p)) || FULL_GATE_FILES.contains(&path) {
        return GateInput::FullGate(format!("global input changed: {path}"));
    }
    // Longest-prefix crate match, so a nested fixture project maps to the crate that owns it.
    if let Some((_, name)) = crate_dirs
        .iter()
        .filter(|(dir, _)| path.starts_with(dir.as_str()))
        .max_by_key(|(dir, _)| dir.len())
    {
        return GateInput::Crate(name.clone());
    }
    if GATE_IGNORED_PREFIXES.iter().any(|p| path.starts_with(p))
        || GATE_IGNORED_FILES.contains(&path)
    {
        return GateInput::Ignored;
    }
    GateInput::FullGate(format!("unmapped path (failing closed): {path}"))
}

/// The changed-file set the diff-scoped gate selects over — the `_lint-changed` convention:
/// committed changes vs the merge-base with the base ref, unioned with staged/unstaged/untracked.
fn gate_changed_files(root: &Path, base: &str) -> anyhow::Result<Vec<String>> {
    let git = |args: &[&str]| -> anyhow::Result<String> {
        let out = Command::new("git")
            .current_dir(root)
            .args(args)
            .output()
            .map_err(|e| anyhow::anyhow!("git {args:?}: {e}"))?;
        anyhow::ensure!(
            out.status.success(),
            "git {args:?} failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        Ok(String::from_utf8_lossy(&out.stdout).into_owned())
    };
    let merge_base = git(&["merge-base", "HEAD", base])?.trim().to_string();
    let mut files: std::collections::BTreeSet<String> =
        git(&["diff", "--name-only", "--no-renames", &merge_base, "HEAD"])?
            .lines()
            .map(str::to_string)
            .collect();
    for line in git(&["status", "--porcelain=v1", "-uall", "--no-renames"])?.lines() {
        let Some(path) = line.get(3..) else { continue };
        // git quotes paths with special characters; refuse to guess at un-escaping (the caller
        // fails closed into the full battery).
        anyhow::ensure!(
            !path.starts_with('"'),
            "dirty path needs git unquoting: {path}"
        );
        files.insert(path.to_string());
    }
    Ok(files.into_iter().collect())
}

/// The workspace members from `cargo metadata --no-deps`: (name, repo-relative crate dir,
/// declared dependency names — every kind, optional included, so feature edges over-select
/// rather than under).
#[allow(clippy::type_complexity)]
fn workspace_members(root: &Path) -> anyhow::Result<Vec<(String, String, Vec<String>)>> {
    let out = Command::new("cargo")
        .current_dir(root)
        .args(["metadata", "--no-deps", "--format-version", "1"])
        .output()
        .map_err(|e| anyhow::anyhow!("cargo metadata: {e}"))?;
    anyhow::ensure!(
        out.status.success(),
        "cargo metadata failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let meta: serde_json::Value = serde_json::from_slice(&out.stdout)
        .map_err(|e| anyhow::anyhow!("parse cargo metadata: {e}"))?;
    let packages = meta["packages"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("cargo metadata: no packages array"))?;
    let mut members = Vec::new();
    for pkg in packages {
        let name = pkg["name"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("package without a name"))?
            .to_string();
        let manifest = Path::new(
            pkg["manifest_path"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("package {name} without manifest_path"))?,
        );
        let dir = manifest
            .parent()
            .and_then(|d| d.strip_prefix(root).ok())
            .map(|d| format!("{}/", d.display()))
            .ok_or_else(|| anyhow::anyhow!("crate dir outside the workspace for {name}"))?;
        let deps = pkg["dependencies"]
            .as_array()
            .map(|deps| {
                deps.iter()
                    .filter_map(|d| d["name"].as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();
        members.push((name, dir, deps));
    }
    Ok(members)
}

/// Expand seed crates through the REVERSE dependency graph (workspace members only, every
/// dependency kind): everything whose build or tests can observe a seed changes with it.
fn reverse_dependents(
    members: &[(String, String, Vec<String>)],
    seeds: &std::collections::BTreeSet<String>,
) -> std::collections::BTreeSet<String> {
    let mut affected = seeds.clone();
    loop {
        let mut grew = false;
        for (name, _, deps) in members {
            if !affected.contains(name) && deps.iter().any(|d| affected.contains(d)) {
                affected.insert(name.clone());
                grew = true;
            }
        }
        if !grew {
            return affected;
        }
    }
}

/// The DIFF-SCOPED gate — the default behavior of `vhc-production-gate`, applying the repo's
/// lint doctrine to tests: history at the base has already been gated, so only the delta's cone
/// needs re-checking. Changed files (merge-base with the base ref + staged/unstaged/untracked)
/// map to workspace crates and expand through the reverse dependency graph; only the pinned
/// battery suites of affected packages run (identical per-suite semantics), plus the acceptance
/// lane whenever the affected cone reaches the product binaries. Every non-crate input maps
/// conservatively (see `classify_gate_input`) and unknowns fail CLOSED into the full battery, so
/// selection can only over-include, never under.
fn vhc_gate_diff(base: Option<String>, dry_run: bool) -> anyhow::Result<()> {
    let started = std::time::Instant::now();
    let root = workspace_root();
    let base = base
        .or_else(|| std::env::var("GATE_BASE").ok().filter(|b| !b.is_empty()))
        .unwrap_or_else(|| "vhc-integration".to_string());
    println!("== vhc-production-gate (diff-scoped vs `{base}`; --all for the full battery) ==");

    let changed = match gate_changed_files(&root, &base) {
        Ok(changed) => changed,
        Err(e) => {
            println!("vhc-production-gate: cannot compute the changed set ({e:#}) — failing closed into the full battery");
            if dry_run {
                println!("(dry run) would run the full battery");
                return Ok(());
            }
            return vhc_production_gate_all();
        }
    };
    let members = workspace_members(&root)?;
    let crate_dirs: Vec<(String, String)> = members
        .iter()
        .map(|(name, dir, _)| (dir.clone(), name.clone()))
        .collect();

    let mut seeds: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    let mut guests_changed = false;
    let mut ignored = 0usize;
    for path in &changed {
        match classify_gate_input(path, &crate_dirs) {
            GateInput::Ignored => ignored += 1,
            GateInput::Guests => guests_changed = true,
            GateInput::Crate(name) => {
                seeds.insert(name);
            }
            GateInput::FullGate(reason) => {
                println!("vhc-production-gate: {reason}");
                if dry_run {
                    println!("(dry run) would run the full battery");
                    return Ok(());
                }
                return vhc_production_gate_all();
            }
        }
    }

    let mut affected = reverse_dependents(&members, &seeds);
    if guests_changed {
        // Guest bytes are runtime inputs to the guest-linked suites, not compile inputs to any
        // host crate — select those suites directly (no reverse expansion needed or sound).
        affected.extend(GUEST_LINKED_PACKAGES.iter().map(|p| p.to_string()));
    }

    let battery: Vec<SuiteEntry<'_>> = VHC_DET_SUITES
        .iter()
        .chain(VHC_NODE_SUITES)
        .chain(VHC_T2_SUITES)
        .copied()
        .collect();
    let selected: Vec<SuiteEntry<'_>> = battery
        .into_iter()
        .filter(|(_, args)| affected.contains(suite_package(args)))
        .collect();
    // The acceptance lane gates the PRODUCT binaries: it is affected exactly when the cone
    // reaches them (or the suite itself changed).
    let acceptance = ["daemon", "daemon-vhc-worker", "daemon-vhc-acceptance"]
        .iter()
        .any(|p| affected.contains(*p));

    println!(
        "vhc-production-gate: {} changed file(s) ({} test-neutral) -> {} seed crate(s){} -> {} affected package(s) -> {} suite(s){}",
        changed.len(),
        ignored,
        seeds.len(),
        if guests_changed { " + guests" } else { "" },
        affected.len(),
        selected.len(),
        if acceptance { " + acceptance" } else { "" },
    );
    if !seeds.is_empty() {
        println!(
            "  seeds: {}",
            seeds.iter().cloned().collect::<Vec<_>>().join(", ")
        );
    }
    if dry_run {
        for (label, _) in &selected {
            println!("  would run: {label}");
        }
        if acceptance {
            println!("  would run: the multi-process acceptance suite");
        }
        println!("(dry run) nothing executed, ledger untouched");
        return Ok(());
    }
    if selected.is_empty() && !acceptance {
        println!(
            "\nvhc-production-gate: no affected test targets — green (nothing to re-verify) in {:.0?}",
            started.elapsed()
        );
        return Ok(());
    }

    // Memoize the whole selection (the workspace fingerprint plus the selected suite identity),
    // so re-gating an identical tree — e.g. right after a no-ff merge of a gated tip — is free.
    let mut selection_id = String::new();
    for (label, _) in &selected {
        selection_id.push_str(label);
        selection_id.push('\n');
    }
    selection_id.push_str(if acceptance {
        "+acceptance"
    } else {
        "-acceptance"
    });
    run_lane_memoized(&root, "vhc-gate-diff", selection_id.as_bytes(), || {
        if !selected.is_empty() {
            println!("\n== vhc-production-gate: preflight (dep-check + guests) ==");
            vhc_dep_check()?;
            build_guests()?;
            run_lane_suites("vhc-gate-diff", &selected, det_schedule_from_env())?;
        }
        if acceptance {
            vhc_acceptance()?;
        }
        Ok(())
    })?;
    println!(
        "\nvhc-production-gate: GREEN (diff-scoped: {} suite(s){}) in {:.0?}",
        selected.len(),
        if acceptance { " + acceptance" } else { "" },
        started.elapsed()
    );
    Ok(())
}

/// Enforce the daemon-vhc dependency-direction rules (architecture §7): the wasm boundary is
/// visible as `sdk/` vs `host/`, and `contracts/` is the only shared ground.
///
/// - `host/*` (incl. `bins/*`) never links `sdk/*` in its PRODUCTION graph: the SDK layer owns
///   the round message schemas (`daemon-vhc-sdk-consensus`, `daemon-vhc-sdk-rounds`), and a
///   production host routes opaque signed frames — it never decodes a round message. Dev edges
///   (test authoring) and feature-gated OPTIONAL edges behind a harness feature are exempt when
///   listed below; harness/oracle tooling crates are exempt wholesale, also listed below.
/// - `contracts/*` links neither `sdk/*` nor `host/*`.
/// - `sdk/*` never links `host/*`.
/// - `daemon-vhc-proto` is algorithm-free AND round-vocabulary-free (source-level check below).
/// - NEGATIVE ARCHITECTURE TEST: the resolved default-feature normal graph of every production
///   host crate (worker binary above all) contains no SDK schema crate — `cargo tree -e normal`
///   per crate, so a feature-unification accident can't smuggle a schema decode into a shipped
///   binary.
///
/// Enforced over `cargo metadata` (normal + dev + build edges) + `cargo tree` (resolved default
/// graphs). The real `sdk/*` consumers are the `guests/` modules, which are a separate cargo
/// workspace outside this gate.
fn vhc_dep_check() -> anyhow::Result<()> {
    use std::collections::{BTreeMap, BTreeSet};

    // The SDK crates that own message SCHEMAS (round vocabulary): a production host crate must
    // never link these in its normal, non-optional graph.
    const SCHEMA_CRATES: &[&str] = &["daemon-vhc-sdk-consensus", "daemon-vhc-sdk-rounds"];

    // Harness/oracle tooling crates, exempt WHOLESALE from the host->sdk prohibition (each with
    // its documented rationale). They are never linked by the production node/worker graph —
    // the negative architecture test below proves that stays true.
    const EXEMPT_HARNESS_CRATES: &[(&str, &str)] = &[
        (
            "daemon-vhc-testkit",
            "whole-run harness: authors genesis coordinator configs + re-derives worker windows \
             with the guests' own assignment math; test tooling by charter, linked only by test \
             targets",
        ),
        (
            "daemon-vhc-observe",
            "replay/audit oracle tooling: re-derives + inspects recorded runs, which requires \
             decoding the SDK round schemas; never on the production node/worker path",
        ),
        (
            "daemon-vhc-e2e",
            "leaf end-to-end test crate (tests/): the one place SDK policy and host pipeline \
             legally meet for equivalence oracles",
        ),
    ];

    // host/* crates whose OPTIONAL sdk edges are permitted because they are gated behind a
    // harness feature that is off by default (the production build never activates them). A new
    // optional edge from a crate not listed here still fails the gate.
    const OPTIONAL_HARNESS_EDGES: &[(&str, &str, &str)] = &[
        (
            "daemon-vhc-host",
            "daemon-vhc-sdk-consensus",
            "the whole-run coordinator drive seat (`coordinator` module): decodes coordinator \
             round decisions + the typed AuthorityConfig; `harness`-feature-gated, off default",
        ),
        (
            "daemon-vhc-session",
            "daemon-vhc-sdk-consensus",
            "the retained RoundEngine + coordinator recording shell + typed checkpoint/upgrade \
             machinery; `harness`-feature-gated, off default",
        ),
        (
            "daemon-vhc-worker",
            "daemon-vhc-sdk-consensus",
            "the in-process self-driven join (round-decoding harness seat); \
             `harness`-feature-gated, off default — the shipped worker joins through the \
             role session (opaque frames only; no transport binding = a typed refusal)",
        ),
    ];

    // The production host crates whose resolved default-feature normal graph must be free of the
    // schema crates (the negative architecture test).
    const PRODUCTION_HOST_CRATES: &[&str] = &[
        "daemon-vhc-host",
        "daemon-vhc-net",
        "daemon-vhc-session",
        "daemon-vhc-node",
        "daemon-vhc-supervisor",
        "daemon-vhc-worker",
        // The extracted journal substrate: linked by the production durable sink, so it must be
        // schema-free like every other production host crate (the oracle tooling that decodes
        // schemas stays in daemon-vhc-observe, which re-exports this crate).
        "daemon-vhc-journal",
    ];

    let root = workspace_root();
    let out = Command::new("cargo")
        .current_dir(&root)
        .args(["metadata", "--format-version", "1", "--no-deps"])
        .output()
        .map_err(|e| anyhow::anyhow!("running cargo metadata: {e}"))?;
    anyhow::ensure!(
        out.status.success(),
        "cargo metadata failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let meta: serde_json::Value = serde_json::from_slice(&out.stdout)
        .map_err(|e| anyhow::anyhow!("parsing cargo metadata json: {e}"))?;

    // Classify a workspace crate by where its manifest lives in the crates/vhc/ tree.
    // `bins/` (the worker binary, split out of daemon-vhc-host at A2) is host-side by
    // construction — same rule as host/*: it never links sdk/* (architecture §7).
    fn role(manifest_path: &str) -> Option<&'static str> {
        if manifest_path.contains("/crates/vhc/contracts/") {
            Some("contracts")
        } else if manifest_path.contains("/crates/vhc/sdk/") {
            Some("sdk")
        } else if manifest_path.contains("/crates/vhc/host/")
            || manifest_path.contains("/crates/vhc/bins/")
        {
            Some("host")
        } else {
            None
        }
    }

    let packages = meta["packages"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("metadata.packages is not an array"))?;
    let mut roles: BTreeMap<String, &'static str> = BTreeMap::new();
    for p in packages {
        if let (Some(name), Some(mp)) = (p["name"].as_str(), p["manifest_path"].as_str()) {
            if let Some(r) = role(mp) {
                roles.insert(name.to_string(), r);
            }
        }
    }

    let is_exempt_crate = |from: &str| EXEMPT_HARNESS_CRATES.iter().any(|(name, _)| *name == from);
    let is_optional_harness_edge = |from: &str, to: &str| {
        OPTIONAL_HARNESS_EDGES
            .iter()
            .any(|(f, t, _)| *f == from && *t == to)
    };

    let mut violations: Vec<String> = Vec::new();
    let mut seen: BTreeSet<(String, String)> = BTreeSet::new();

    for p in packages {
        let from = p["name"].as_str().unwrap_or_default().to_string();
        let from_role = roles.get(&from).copied();
        let deps = p["dependencies"].as_array().cloned().unwrap_or_default();
        for d in &deps {
            let to = d["name"].as_str().unwrap_or_default().to_string();
            // Only edges within the crates/vhc/ tree are governed by these rules.
            let Some(to_role) = roles.get(&to).copied() else {
                continue;
            };
            let kind = d["kind"].as_str().unwrap_or("normal"); // null == normal
            let optional = d["optional"].as_bool().unwrap_or(false);

            // Edges into sdk/*. The SDK-side native sim (`daemon-vhc-sim`) is the DESIGNED entry
            // for native harnesses (architecture §6: "policy code compiled natively runs against
            // it"): any crate that is NOT host/* or contracts/* may link it without an exception.
            // The wasm-boundary wall still holds for production graphs — a host/* crate reaches
            // sdk/* only through a dev edge (test authoring, can't ship), a LISTED harness-gated
            // optional edge (off default), or by being a LISTED harness/oracle tooling crate.
            // Anything else is a violation: production hosts route opaque signed frames and never
            // link the SDK schema layer.
            if to_role == "sdk" && from_role != Some("sdk") {
                let sim_native_harness = to == "daemon-vhc-sim"
                    && from_role != Some("host")
                    && from_role != Some("contracts");
                if sim_native_harness {
                    // allowed: a native harness (e.g. the vhc-sim examples, tests/*) linking the
                    // SDK-side sim — its whole purpose (refactor §6/§11).
                } else if kind == "dev" {
                    // allowed: dev edges compile into test targets only — they cannot ship.
                    seen.insert((from.clone(), to.clone()));
                } else if optional && is_optional_harness_edge(&from, &to) {
                    // allowed: a harness-feature-gated optional edge, off the default build.
                    seen.insert((from.clone(), to.clone()));
                } else if is_exempt_crate(&from) {
                    // allowed: a documented harness/oracle tooling crate.
                    seen.insert((from.clone(), to.clone()));
                } else {
                    violations.push(format!(
                        "{from} -> {to} [{kind}{}]: a production crate must not link the SDK \
                         layer (round schemas are module vocabulary; hosts route opaque signed \
                         frames) — dev edges, listed harness-gated optional edges, and listed \
                         harness tooling crates are the only exemptions",
                        if optional { ", optional" } else { "" }
                    ));
                }
            }
            // contracts/* links neither sdk/* nor host/* (hard).
            if from_role == Some("contracts") && (to_role == "sdk" || to_role == "host") {
                violations.push(format!(
                    "{from} -> {to} [{kind}]: contracts/* must link neither sdk/* nor host/*"
                ));
            }
            // sdk/* never links host/* (hard).
            if from_role == Some("sdk") && to_role == "host" {
                violations.push(format!(
                    "{from} -> {to} [{kind}]: sdk/* must not link host/*"
                ));
            }
            // The Backend Execution Profile must stay unreachable from a guest (architecture §9.6
            // [RC-4], §9.7 [PC-11]). The two hard rules above already imply it, but this one is
            // named and separate on purpose: the consequence of getting it wrong is silent and
            // expensive — a profile type reachable from a guest-linked crate makes every profile
            // revision change every guest hash, so a driver update re-pins and re-certifies a
            // training algorithm that did not change. That is the coupling the three-object model
            // exists to remove, and a crate-layout slip is all it takes to reintroduce it. A
            // failure here should say so rather than read as a generic layering complaint.
            if to == "daemon-vhc-resource"
                && (from_role == Some("contracts") || from_role == Some("sdk"))
            {
                violations.push(format!(
                    "{from} -> {to} [{kind}]: the resource crate carries the Backend Execution \
                     Profile and MUST NOT be reachable from a guest — a profile revision would \
                     then change every guest hash, so a driver update would re-pin the fleet"
                ));
            }
            // The resource crate's `test-support` feature exposes the fixture constructors that can
            // MINT a Backend Execution Profile — the one act the store's crate-private surface exists
            // to keep out of a shipping binary (architecture §9.6: composition takes an authenticated
            // profile because there is no other constructor to reach for). A test in another crate
            // legitimately needs them, so the feature exists; it may travel on a `dev` edge only. A
            // production edge that enables it would hand a shipping binary the ability to author its
            // own trust, which is a hole no amount of downstream care closes.
            if to == "daemon-vhc-resource" && kind != "dev" {
                let enables_test_support = d["features"]
                    .as_array()
                    .is_some_and(|fs| fs.iter().any(|f| f.as_str() == Some("test-support")));
                if enables_test_support {
                    violations.push(format!(
                        "{from} -> {to} [{kind}]: this edge enables the `test-support` feature, \
                         whose fixture constructors can mint a Backend Execution Profile — it is \
                         permitted on `dev-dependencies` edges only, so that no shipping binary can \
                         author the trust it is supposed to be authenticated against"
                    ));
                }
            }
            // The A2 dependency inversion (refactor §5 A2 item 3; architecture §7 SESS → HOSTC):
            // the session links the host — the host must NEVER re-grow a runtime edge onto the
            // session (run policy). A dev-only edge (fixture/parity tests) is permitted.
            if from == "daemon-vhc-host" && to == "daemon-vhc-session" && kind != "dev" {
                violations.push(format!(
                    "{from} -> {to} [{kind}]: the A2 inversion is one-way — the host must not \
                     link the session at runtime (session → host is the direction)"
                ));
            }
        }
    }

    // --- daemon-vhc-proto is ALGORITHM-FREE and ROUND-VOCABULARY-FREE (architecture §7 rule 1 —
    // "no assignment math, no round vocabulary"). The assignment module moved to
    // sdk/daemon-vhc-sdk-consensus first; the round message schemas, the round state-digest
    // schedule, and the record-set object followed with the round-vocabulary move. A re-grown module
    // (or file) in the proto fails this gate from now on.
    {
        let proto_src = root.join("crates/vhc/contracts/daemon-vhc-proto/src");
        for module in ["assignment", "messages", "digest", "record_set"] {
            if proto_src.join(format!("{module}.rs")).exists() {
                violations.push(format!(
                    "daemon-vhc-proto: src/{module}.rs exists — the proto is algorithm-free and \
                     round-vocabulary-free; that vocabulary lives in sdk/daemon-vhc-sdk-consensus"
                ));
            }
        }
        let lib = std::fs::read_to_string(proto_src.join("lib.rs")).unwrap_or_default();
        for module in ["assignment", "messages", "digest", "record_set"] {
            if lib.contains(&format!("mod {module}")) {
                violations.push(format!(
                    "daemon-vhc-proto: lib.rs declares a `{module}` module — the proto is \
                     algorithm-free and round-vocabulary-free (architecture §7 rule 1)"
                ));
            }
        }
    }

    // --- The fixture feature is off by default, at the source. The edge check above catches a
    // consumer that asks for `test-support`; this catches the far worse slip of the resource crate
    // handing it out unasked, which would make every production edge a profile-minting edge without
    // any consumer manifest showing it.
    {
        let manifest = root.join("crates/vhc/host/daemon-vhc-resource/Cargo.toml");
        let text = std::fs::read_to_string(&manifest)
            .map_err(|e| anyhow::anyhow!("read {}: {e}", manifest.display()))?;
        let default_line = text
            .lines()
            .find(|l| l.trim_start().starts_with("default ="))
            .unwrap_or_default();
        if default_line.contains("test-support") {
            violations.push(format!(
                "daemon-vhc-resource: `test-support` is in the crate's DEFAULT feature set \
                 ({}) — the fixture constructors that can mint a Backend Execution Profile must \
                 never be on unless a dev edge asks for them by name",
                default_line.trim()
            ));
        }
        anyhow::ensure!(
            text.contains("test-support = []"),
            "daemon-vhc-resource: the `test-support` feature is not declared in {} — this gate \
             asserts a property of a feature that must exist; if the feature was removed, remove \
             the check with it deliberately rather than letting it pass vacuously",
            manifest.display()
        );
    }

    // --- HARNESS QUARANTINE (source-level): the engine-era surfaces stay unmistakably
    // harness-only. The RoundEngine orbit in the session crate (engine, checkpoint, upgrade,
    // receipt, harness, replay_sandbox, coordinator_shell) plus the legacy JSON corpus pipeline
    // (`data` — the production corpus contract is chunk-addressed), the host's coordinator drive
    // seat, and the worker's in-process self-driven join must each sit directly behind their
    // harness cfg gate. A module that loses its gate (or moves without updating this check)
    // fails the gate from now on.
    {
        let session_gate = r#"#[cfg(any(test, feature = "harness"))]"#;
        let worker_gate = r#"#[cfg(feature = "harness")]"#;
        let checks: &[(&str, &str, &str, &[&str])] = &[
            (
                "daemon-vhc-session",
                "crates/vhc/host/daemon-vhc-session/src/lib.rs",
                session_gate,
                &[
                    "pub mod data;",
                    "pub mod checkpoint;",
                    "pub mod engine;",
                    "pub mod upgrade;",
                    "pub mod receipt;",
                    "pub mod harness;",
                    "pub mod replay_sandbox;",
                    "pub mod coordinator_shell;",
                ],
            ),
            (
                "daemon-vhc-host",
                "crates/vhc/host/daemon-vhc-host/src/lib.rs",
                session_gate,
                &["pub mod coordinator;"],
            ),
            (
                "daemon-vhc-worker",
                "crates/vhc/bins/daemon-vhc-worker/src/main.rs",
                worker_gate,
                &["mod session;"],
            ),
        ];
        for (krate, path, gate, decls) in checks {
            let src = std::fs::read_to_string(root.join(path)).unwrap_or_default();
            let lines: Vec<&str> = src.lines().map(str::trim).collect();
            for decl in *decls {
                match lines.iter().position(|l| l == decl) {
                    None => violations.push(format!(
                        "{krate}: `{decl}` not found in {path} — if the harness-era module \
                         moved, move its quarantine check with it"
                    )),
                    Some(i) => {
                        if i == 0 || lines[i - 1] != *gate {
                            violations.push(format!(
                                "{krate}: `{decl}` in {path} is not immediately preceded by \
                                 `{gate}` — the engine-era surface must stay harness-gated \
                                 (production builds route opaque frames and read chunk-addressed \
                                 corpora only)"
                            ));
                        }
                    }
                }
            }
        }
    }

    // --- NEGATIVE ARCHITECTURE TEST: no production host crate can decode an SDK round message.
    // The structural form: the resolved DEFAULT-FEATURE normal dependency graph of each
    // production host crate (the worker binary above all) must not contain a schema crate. This
    // is what the metadata edge rules above cannot see — feature unification. `cargo tree -p X
    // -e normal` resolves X like `cargo build -p X` (workspace-member features do NOT unify in),
    // so this is exactly the shipped graph.
    for krate in PRODUCTION_HOST_CRATES {
        let out = Command::new("cargo")
            .current_dir(&root)
            .args([
                "tree", "-p", krate, "-e", "normal", "--prefix", "none", "--quiet",
            ])
            .output()
            .map_err(|e| anyhow::anyhow!("running cargo tree -p {krate}: {e}"))?;
        anyhow::ensure!(
            out.status.success(),
            "cargo tree -p {krate} failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let tree = String::from_utf8_lossy(&out.stdout);
        for schema in SCHEMA_CRATES {
            if tree.lines().any(|l| l.trim().starts_with(schema)) {
                violations.push(format!(
                    "{krate}: the SDK schema crate `{schema}` is reachable in the resolved \
                     default-feature normal graph — a production host must not be able to \
                     decode an SDK round message (round schemas are module vocabulary)"
                ));
            }
        }
    }

    println!("daemon-vhc dependency-direction check (architecture §7)");
    println!(
        "  rule: host/* never links sdk/* in production graphs · contracts/* links neither · \
         sdk/* never links host/*"
    );
    println!(
        "  rule: daemon-vhc-proto is algorithm-free and round-vocabulary-free (schemas live in \
         sdk-consensus)"
    );
    println!(
        "  rule: daemon-vhc-resource (the Backend Execution Profile) is unreachable from a guest \
         — a profile revision must never move a guest hash"
    );
    println!(
        "  rule: no production host crate resolves a schema crate ({}) in its default normal \
         graph",
        SCHEMA_CRATES.join(", ")
    );
    println!(
        "  rule: the engine-era surfaces (RoundEngine orbit, the legacy JSON corpus pipeline, \
         the coordinator drive seats) stay harness-gated at their declaration sites"
    );
    println!("\nexempt harness/oracle tooling crates:");
    for (name, note) in EXEMPT_HARNESS_CRATES {
        println!("  {name}: {note}");
    }
    println!("\nlisted harness-gated optional edges:");
    for (f, t, note) in OPTIONAL_HARNESS_EDGES {
        let mark = if seen.contains(&((*f).to_string(), (*t).to_string())) {
            "present"
        } else {
            "STALE — listed but not in the graph; drop it from OPTIONAL_HARNESS_EDGES"
        };
        println!("  [{mark}] {f} -> {t}: {note}");
    }

    if !violations.is_empty() {
        eprintln!("\ndependency-direction VIOLATIONS:");
        for v in &violations {
            eprintln!("  x {v}");
        }
        anyhow::bail!("{} dependency-direction violation(s)", violations.len());
    }
    println!("\nok: no dependency-direction violations");
    provenance_scan()?;
    vhc_codename_scan()?;
    Ok(())
}

/// Hash every built `.wasm` in `guests/target/wasm32-unknown-unknown/release` and write the sorted
/// `guests/guests.blake3` manifest (`<blake3-hex>  <name>.wasm` per line). Returns the manifest path.
fn write_guest_manifest(guests: &Path) -> anyhow::Result<PathBuf> {
    let release = guests.join("target/wasm32-unknown-unknown/release");
    let mut entries: Vec<(String, String)> = Vec::new();
    for dent in std::fs::read_dir(&release)
        .map_err(|e| anyhow::anyhow!("read guest output dir {}: {e}", release.display()))?
    {
        let path = dent?.path();
        if path.extension().and_then(|e| e.to_str()) == Some("wasm") {
            let name = path
                .file_name()
                .and_then(|n| n.to_str())
                .expect("wasm file name")
                .to_string();
            let bytes = std::fs::read(&path)
                .map_err(|e| anyhow::anyhow!("read {}: {e}", path.display()))?;
            entries.push((name, blake3::hash(&bytes).to_hex().to_string()));
        }
    }
    anyhow::ensure!(
        !entries.is_empty(),
        "no guest .wasm modules found under {}",
        release.display()
    );
    entries.sort();
    let body: String = entries
        .iter()
        .map(|(name, hex)| format!("{hex}  {name}\n"))
        .collect();
    let manifest = guests.join("guests.blake3");
    std::fs::write(&manifest, body)
        .map_err(|e| anyhow::anyhow!("write {}: {e}", manifest.display()))?;
    Ok(manifest)
}

/// The workspace root (xtask's manifest dir is `<root>/xtask`).
fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask lives under the workspace root")
        .to_path_buf()
}

/// Generate the committed C headers for both binding crates via `cbindgen`.
fn gen_headers() -> anyhow::Result<()> {
    let root = workspace_root();
    // (crate name, crate dir relative to root, output header relative to the crate dir).
    let crates = [
        (
            "daemon-core-ffi",
            "bindings/daemon-core-ffi",
            "include/daemon_core.h",
        ),
        ("daemon-ffi", "bindings/daemon-ffi", "include/daemon.h"),
    ];
    for (name, dir, header) in crates {
        gen_one_header(&root, name, dir, header)?;
    }
    Ok(())
}

/// Run `cbindgen` over one binding crate, writing its committed header.
fn gen_one_header(root: &Path, name: &str, dir: &str, header: &str) -> anyhow::Result<()> {
    let crate_dir = root.join(dir);
    let config = crate_dir.join("cbindgen.toml");
    let out = crate_dir.join(header);
    std::fs::create_dir_all(out.parent().unwrap())?;

    let status = Command::new("cbindgen")
        .arg("--config")
        .arg(&config)
        .arg("--crate")
        .arg(name)
        .arg("--output")
        .arg(&out)
        .arg(&crate_dir)
        .status()
        .map_err(|e| anyhow::anyhow!("failed to run cbindgen (is it on PATH?): {e}"))?;
    anyhow::ensure!(status.success(), "cbindgen exited with {status} for {name}");

    println!("generated {}", out.display());
    Ok(())
}

/// Check that the `daemon-api` CDDL mirror artifact exists and names every Rust request/response
/// variant. This is intentionally a syntactic parity gate: schema validation/codegen is handled by
/// downstream CDDL tooling, but adding a Rust wire variant without updating the published contract
/// must fail CI.
fn check_cddl() -> anyhow::Result<()> {
    let root = workspace_root();
    let path = root.join("crates/contracts/daemon-api/daemon-api.cddl");
    let text = read_to_string(&path)?;
    anyhow::ensure!(!text.trim().is_empty(), "{} is empty", path.display());
    for rule in [
        "api-request",
        "api-response",
        "wire_version",
        // wire v2: the merged live session event log shapes.
        "session-log-entry",
        "session-payload",
        "log-page-view",
        "direction",
        "disposition",
        "origin",
        // wire v2: outbound delivery targets + handover (§5.4).
        "delivery-target",
        "sink-kind",
        "route-addr",
    ] {
        anyhow::ensure!(
            text.contains(rule),
            "{} is missing the `{rule}` rule",
            path.display()
        );
    }
    // ApiRequest / ApiResponse live in the `wire` submodule of daemon-api.
    let rust = read_to_string(&root.join("crates/contracts/daemon-api/src/wire.rs"))?;
    assert_cddl_covers_enum(&text, &rust, "ApiRequest", "api-request")?;
    assert_cddl_covers_enum(&text, &rust, "ApiResponse", "api-response")?;
    println!("ok: {} defines the api mirror", path.display());
    Ok(())
}

fn read_to_string(path: &Path) -> anyhow::Result<String> {
    std::fs::read_to_string(path).map_err(|e| anyhow::anyhow!("reading {}: {e}", path.display()))
}

fn assert_cddl_covers_enum(
    cddl: &str,
    rust: &str,
    enum_name: &str,
    rule_name: &str,
) -> anyhow::Result<()> {
    let variants = rust_enum_variants(rust, enum_name)?;
    let missing: Vec<_> = variants
        .iter()
        .filter(|variant| !cddl_rule_mentions_variant(cddl, rule_name, variant))
        .cloned()
        .collect();
    anyhow::ensure!(
        missing.is_empty(),
        "{rule_name} is missing Rust {enum_name} variants: {}",
        missing.join(", ")
    );
    Ok(())
}

fn rust_enum_variants(rust: &str, enum_name: &str) -> anyhow::Result<Vec<String>> {
    let marker = format!("pub enum {enum_name}");
    let start = rust
        .find(&marker)
        .ok_or_else(|| anyhow::anyhow!("could not find `{marker}`"))?;
    let after_marker = &rust[start + marker.len()..];
    let open = after_marker
        .find('{')
        .ok_or_else(|| anyhow::anyhow!("could not find body for `{enum_name}`"))?;
    let body_start = start + marker.len() + open + 1;
    let mut depth = 1i32;
    let mut end = None;
    for (offset, ch) in rust[body_start..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    end = Some(body_start + offset);
                    break;
                }
            }
            _ => {}
        }
    }
    let body_end = end.ok_or_else(|| anyhow::anyhow!("unterminated `{enum_name}` body"))?;
    let mut variants = Vec::new();
    let mut depth = 1i32;
    for line in rust[body_start..body_end].lines() {
        let trimmed = line.trim();
        if depth == 1
            && !trimmed.is_empty()
            && !trimmed.starts_with("///")
            && !trimmed.starts_with("#[")
            && !trimmed.starts_with("//")
            && !trimmed.starts_with('}')
        {
            let ident: String = trimmed
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect();
            if !ident.is_empty() {
                variants.push(ident);
            }
        }
        for ch in line.chars() {
            match ch {
                '{' | '(' => depth += 1,
                '}' | ')' => depth -= 1,
                _ => {}
            }
        }
    }
    variants.sort();
    variants.dedup();
    Ok(variants)
}

/// A Rust enum variant is "covered" when the CDDL carries its externally-tagged wire key as a
/// quoted string `"Variant"`. In the unified CDDL each `api-request`/`api-response` arm is its own
/// named rule (e.g. `request-submit = { "Submit": ... }`, `request-health = "Health"`), so the key
/// lives in the arm rule rather than inline in the union block; searching the whole file is the
/// format-stable parity check. `rule_name` is kept for call-site clarity.
fn cddl_rule_mentions_variant(cddl: &str, rule_name: &str, variant: &str) -> bool {
    let _ = rule_name;
    cddl.contains(&format!("\"{variant}\""))
}

fn gen_api_fixtures() -> anyhow::Result<()> {
    use daemon_api::{
        AccountSettingsSchema, AdapterCapabilities, AdapterInfo, ApiRequest, ApiResponse,
        ApprovalInfo, ChatMessage, CommandInvocation, CommandOutput, ConnectionState, ContactInfo,
        ContactsOps, ConvChange, ConversationOps, CredentialInfo, DisconnectReason, EventsPage,
        HealthReport, JournalPageView, JournalRecord, JournalRecordPayload, LogPageView,
        MembershipChange, MembershipOps, MessageAttachment, ModelDescriptor, NodeEvent,
        Participant, PolicyEntry, PresenceState, ProfileInfo, ProfileSpec, ProviderDescriptor,
        ProviderKindWire, ProviderSelector, ProviderSignIn, RosterOps, ServiceHealth, SessionPage,
        TransportInstanceInfo,
    };
    use daemon_common::{Author, ProfileRef, ReqId, SessionId};
    use daemon_protocol::{AgentCommand, ToolDetail, TransportId, UserMsg};

    let root = workspace_root();
    let out = root.join("crates/contracts/daemon-api/fixtures/cbor");
    std::fs::create_dir_all(&out)?;

    write_cbor(&out, "request-health.cbor", &ApiRequest::Health)?;
    write_cbor(
        &out,
        "request-sessions-query.cbor",
        &ApiRequest::SessionsQuery {
            query: daemon_api::SessionQuery {
                scope: daemon_api::SessionScope::TopLevel,
                after: None,
                limit: 25,
                since_rev: None,
            },
        },
    )?;
    write_cbor(
        &out,
        "request-subscribe.cbor",
        &ApiRequest::Subscribe {
            session: SessionId::new("fixture-session"),
            after_seq: 0,
            max: 64,
        },
    )?;
    // rung 2 (api/39): the generalized backward windows — a forward SessionHistory resume
    // (before_cursor absent, never null) and the newest-anchored backward forms of all three
    // journal reads, so verify-codec proves the generated zcbor decoder accepts both shapes.
    write_cbor(
        &out,
        "request-session-history.cbor",
        &ApiRequest::SessionHistory {
            session: SessionId::new("fixture-session"),
            after_cursor: 128,
            before_cursor: None,
            max: 64,
        },
    )?;
    write_cbor(
        &out,
        "request-session-history-before.cbor",
        &ApiRequest::SessionHistory {
            session: SessionId::new("fixture-session"),
            after_cursor: 0,
            before_cursor: Some(u64::MAX),
            max: 64,
        },
    )?;
    write_cbor(
        &out,
        "request-unit-history-before.cbor",
        &ApiRequest::UnitHistory {
            unit: daemon_common::UnitId::new("fixture-unit"),
            after_cursor: 0,
            before_cursor: Some(4096),
            max: 32,
        },
    )?;
    write_cbor(
        &out,
        "request-events-since.cbor",
        &ApiRequest::EventsSince {
            cursor: 0,
            wait_ms: Some(1000),
        },
    )?;
    write_cbor(
        &out,
        "request-submit.cbor",
        &ApiRequest::Submit {
            session: SessionId::new("fixture-session"),
            command: AgentCommand::StartTurn {
                input: UserMsg::new("hello from daemon-app"),
                request_id: ReqId(1),
            },
            origin: None,
            profile: Some(ProfileRef::new("default")),
        },
    )?;
    write_cbor(
        &out,
        "request-session-create.cbor",
        &ApiRequest::SessionCreate {
            session: Some(SessionId::new("fixture-session")),
            profile: Some(ProfileRef::new("default")),
        },
    )?;
    // Cluster B / allow_permanent: a committed fixture exercising the additive optional field at v28,
    // so the CDDL↔Rust agreement on `request-approval-decide` is proven on a real ciborium payload.
    write_cbor(
        &out,
        "request-approval-decide.cbor",
        &ApiRequest::ApprovalDecide {
            session: SessionId::new("fixture-session"),
            request_id: "fixture-request".into(),
            allow: true,
            allow_permanent: true,
            reason: Some("fixture reason".into()),
        },
    )?;
    // The read-only guardrail caps (wire v29).
    write_cbor(&out, "request-caps.cbor", &ApiRequest::Caps)?;
    write_cbor(
        &out,
        "response-caps.cbor",
        &ApiResponse::Caps(daemon_api::CapsReport {
            orchestrate_max_depth: 1,
            orchestrate_max_fanout: 8,
            // wire v31: the agent-created-agents guardrail caps.
            max_composed_profiles: 32,
            max_ephemeral_per_session: 8,
        }),
    )?;
    // Fingerprint management (wire v29): the allow-list list/revoke ops + the list response, so
    // `verify-codec` proves the generated zcbor C decoder accepts the new shapes.
    write_cbor(
        &out,
        "request-fingerprint-list.cbor",
        &ApiRequest::FingerprintList {
            session: SessionId::new("fixture-session"),
        },
    )?;
    write_cbor(
        &out,
        "request-fingerprint-revoke.cbor",
        &ApiRequest::FingerprintRevoke {
            session: SessionId::new("fixture-session"),
            fingerprint: "ab12cd34".into(),
        },
    )?;
    write_cbor(
        &out,
        "response-fingerprints.cbor",
        &ApiResponse::Fingerprints(vec![daemon_api::RememberedFingerprint {
            fingerprint: "ab12cd34".into(),
            // Provenance (wire v30): a populated label + capture timestamp.
            label: Some("git status".into()),
            remembered_at_ms: 1_700_000_000_000,
        }]),
    )?;
    write_cbor(
        &out,
        "response-session-created.cbor",
        &ApiResponse::SessionCreated {
            session: SessionId::new("fixture-session"),
        },
    )?;
    // ----- wire v30 batch -----
    // Item 1: transport lifecycle ops.
    write_cbor(
        &out,
        "request-transport-disconnect.cbor",
        &ApiRequest::TransportDisconnect {
            transport: TransportId::new("matrix/@bot:hs.org"),
        },
    )?;
    write_cbor(
        &out,
        "request-transport-remove.cbor",
        &ApiRequest::TransportRemove {
            transport: TransportId::new("matrix/@bot:hs.org"),
        },
    )?;
    // Item 2: an instance carrying a fatal auth failure (reason/message/fatal + Error state).
    write_cbor(
        &out,
        "response-transport-instances.cbor",
        &ApiResponse::TransportInstances(vec![TransportInstanceInfo {
            transport: TransportId::new("matrix/@bot:hs.org"),
            family: "matrix".into(),
            display_name: "@bot:hs.org".into(),
            connection: ConnectionState::Error,
            presence: PresenceState::Offline,
            bound_profile: Some(ProfileRef::new("default")),
            reason: Some(DisconnectReason::AuthenticationFailed),
            message: Some("M_FORBIDDEN: invalid access token".into()),
            fatal: true,
            // Wire v35: this instance carries a custom label + is enabled (the desired-state
            // overlay), so the one fixture exercises the populated decode of both new fields.
            enabled: true,
            label: Some("Work bot".into()),
        }]),
    )?;
    // Item 4: adapter policies — matrix reports auto_accept_invites; a second adapter reports none.
    // Wire v33: the matrix row also carries the per-verb ops descriptors (Some(..) with mixed
    // flags + directory=true) while the room row leaves every new field at its default (None ops +
    // directory=false, encoded as absent) — so the one fixture proves both the populated and the
    // back-compat "absent" decode of the v33 additive fields.
    write_cbor(
        &out,
        "response-adapters.cbor",
        &ApiResponse::Adapters(vec![
            AdapterInfo {
                family: "matrix".into(),
                display_name: "Matrix".into(),
                capabilities: AdapterCapabilities {
                    rooms: true,
                    direct_messages: true,
                    presence: true,
                    room_enumeration: true,
                    file_transfer: false,
                    interactive_auth: true,
                },
                account_schema: AccountSettingsSchema::default(),
                policies: vec![PolicyEntry {
                    key: "auto_accept_invites".into(),
                    label: "Automatically accept room invites".into(),
                    value: "true".into(),
                }],
                conversation_ops: Some(ConversationOps {
                    create: true,
                    join_channel: true,
                    leave: true,
                    delete: false,
                    send: true,
                    set_topic: true,
                    set_title: true,
                    set_description: true,
                }),
                membership_ops: Some(MembershipOps {
                    invite: true,
                    remove: true,
                    ban: true,
                    set_role: true,
                }),
                contacts_ops: Some(ContactsOps {
                    get_profile: true,
                    action_menu: false,
                    set_alias: false,
                }),
                roster_ops: Some(RosterOps {
                    list: true,
                    add: false,
                    update: false,
                    remove: false,
                }),
                directory: true,
            },
            AdapterInfo {
                family: "room".into(),
                display_name: "Rooms (internal)".into(),
                capabilities: AdapterCapabilities::default(),
                account_schema: AccountSettingsSchema::default(),
                policies: Vec::new(),
                conversation_ops: None,
                membership_ops: None,
                contacts_ops: None,
                roster_ops: None,
                directory: false,
            },
        ]),
    )?;
    // Item 6: the tool-override op.
    write_cbor(
        &out,
        "request-tool-set-enabled.cbor",
        &ApiRequest::ToolSetEnabled {
            tool: "browser".into(),
            enabled: false,
        },
    )?;
    // Item 7: an fs/edit approval carrying a node-computed diff detail.
    write_cbor(
        &out,
        "response-approvals.cbor",
        &ApiResponse::Approvals(daemon_api::WirePage {
            items: vec![ApprovalInfo {
                session: SessionId::new("fixture-session"),
                request_id: "fixture-approval".into(),
                prompt: "Apply edit to src/lib.rs".into(),
                path: Some("src/lib.rs".into()),
                fingerprint: None,
                detail: Some(ToolDetail::new(
                    "fs.diff",
                    br#"{"path":"src/lib.rs","diff":"@@ -1 +1 @@\n-old\n+new\n"}"#.to_vec(),
                )),
            }],
            next: None,
        }),
    )?;
    write_cbor(&out, "request-profile-list.cbor", &ApiRequest::ProfileList)?;
    write_cbor(
        &out,
        "request-model-current.cbor",
        &ApiRequest::ModelCurrent {
            profile: Some("default".into()),
        },
    )?;
    write_cbor(&out, "request-fs-roots.cbor", &ApiRequest::FsRoots)?;
    // Paged fs_list (wire v24/v25, the uniform WirePage shape): a resume request (after = the
    // previous page's `next`) and a page response carrying items + a set `next` cursor, so
    // `verify-codec` proves the generated zcbor C decoder accepts the fs-list-page shape.
    write_cbor(
        &out,
        "request-fs-list.cbor",
        &ApiRequest::FsList {
            root: daemon_api::FsRootId::Workspace,
            dir: "src".into(),
            show_ignored: false,
            after: Some("src/main.rs".into()),
        },
    )?;
    write_cbor(
        &out,
        "response-fs-list.cbor",
        &ApiResponse::FsList(daemon_api::FsListPage {
            items: vec![
                daemon_api::FsEntry {
                    name: "vendor".into(),
                    path: "src/vendor".into(),
                    kind: daemon_api::FsEntryKind::Dir,
                    size: 0,
                    mtime_ms: 1_700_000_000_000,
                    ignored: false,
                },
                daemon_api::FsEntry {
                    name: "lib.rs".into(),
                    path: "src/lib.rs".into(),
                    kind: daemon_api::FsEntryKind::File,
                    size: 4096,
                    mtime_ms: 1_700_000_000_001,
                    ignored: false,
                },
            ],
            next: Some("src/lib.rs".into()),
        }),
    )?;
    // Paged conv_list (wire v25): a resume request + a page with a set `next` cursor, proving the
    // generated zcbor C decoder accepts the conv-page shape. rung 2 (api/39): the request
    // carries a `since_rev` delta anchor and the page carries `removed` tombstones.
    write_cbor(
        &out,
        "request-conv-list.cbor",
        &ApiRequest::ConvList {
            transport: daemon_protocol::TransportId::new("rooms"),
            after: Some("conv-063".into()),
            since_rev: Some(7),
        },
    )?;
    write_cbor(
        &out,
        "response-conv-list.cbor",
        &ApiResponse::Conversations(daemon_api::ConvPage {
            items: vec![daemon_api::ConversationInfo {
                transport: daemon_protocol::TransportId::new("rooms"),
                id: "conv-064".into(),
                kind: daemon_api::ConversationType::Channel,
                title: Some("General".into()),
                topic: None,
                description: None,
                members: Vec::new(),
                parent: None,
            }],
            next: Some("conv-064".into()),
            // rung 1 (api/39): the transport's conversation-set rev the client compares against
            // the `ConversationsChanged.rev` pointer.
            rev: 8,
            // rung 2 (api/39): a delta read's removal tombstone (the client prunes it).
            removed: vec!["conv-007".into()],
            // rung 3 (api/39): the page-side `origin_ops` map — the changed conversation's
            // latest reflected mutation carried this client op_id (carrier 2).
            origin_ops: [("conv-064".to_string(), "018f3b9c-op".to_string())]
                .into_iter()
                .collect(),
        }),
    )?;
    // Conversation hierarchy (wire v38): a structural `Space` container (a root — no `parent`) and
    // a child `Channel` naming that space via `parent`, so verify-codec proves the generated zcbor C
    // decoder accepts the new `ConversationType::Space` variant + the additive `parent` member.
    write_cbor(
        &out,
        "response-conv-hierarchy.cbor",
        &ApiResponse::Conversations(daemon_api::ConvPage {
            items: vec![
                daemon_api::ConversationInfo {
                    transport: daemon_protocol::TransportId::new("matrix/@me:hs.org"),
                    id: "!space:hs.org".into(),
                    kind: daemon_api::ConversationType::Space,
                    title: Some("Engineering".into()),
                    topic: None,
                    description: None,
                    members: Vec::new(),
                    parent: None,
                },
                daemon_api::ConversationInfo {
                    transport: daemon_protocol::TransportId::new("matrix/@me:hs.org"),
                    id: "!room:hs.org".into(),
                    kind: daemon_api::ConversationType::Channel,
                    title: Some("general".into()),
                    topic: Some("chit-chat".into()),
                    description: None,
                    members: Vec::new(),
                    parent: Some("!space:hs.org".into()),
                },
            ],
            next: None,
            rev: 2,
            removed: Vec::new(),
            origin_ops: std::collections::BTreeMap::new(),
        }),
    )?;
    // rung 2 (api/39): the ConvHistory backward window — `before_cursor` present (a scroll-back
    // fetch of the newest 64 records below an anchor), `after_cursor` at its default.
    write_cbor(
        &out,
        "request-conv-history-before.cbor",
        &ApiRequest::ConvHistory(daemon_api::ConvHistoryArgs {
            transport: daemon_protocol::TransportId::new("rooms"),
            conv: "conv-064".into(),
            after_cursor: 0,
            before_cursor: Some(9000),
            max: 64,
        }),
    )?;
    // rung 3 (api/39): a ConvSend carrying the client-minted op_id (the idempotency key + the
    // provenance token), so verify-codec proves the generated zcbor C decoder accepts the additive
    // `? op_id` member on conv-send-args.
    write_cbor(
        &out,
        "request-conv-send.cbor",
        &ApiRequest::ConvSend(daemon_api::ConvSendArgs {
            transport: daemon_protocol::TransportId::new("rooms"),
            conv: "conv-064".into(),
            from: None,
            message: daemon_protocol::UserMsg::new("hello"),
            op_id: Some("018f3b9c-op".into()),
        }),
    )?;
    // rung 3 (api/39): the Bootstrap probe request + a response snapshot (revs + cursor +
    // epoch), so verify-codec proves the generated decoder accepts the new request/response arms
    // and the `{ * tstr => uint64 }` revs map.
    write_cbor(&out, "request-bootstrap.cbor", &ApiRequest::Bootstrap)?;
    write_cbor(
        &out,
        "response-bootstrap.cbor",
        &ApiResponse::Bootstrap(daemon_api::BootstrapReport {
            cursor: 42,
            epoch: 1,
            revs: [
                ("roster".to_string(), 7u64),
                ("fleet".to_string(), 9),
                ("persons".to_string(), 6),
                ("conv:rooms".to_string(), 8),
                ("contacts:matrix/@me:hs.org".to_string(), 5),
            ]
            .into_iter()
            .collect(),
        }),
    )?;
    // Server-side roster (wire v34): a paged list request + resume, the mutation requests carrying a
    // ContactInfo, and a ContactPage response with a set `next` cursor — so verify-codec proves the
    // generated zcbor C decoder accepts the contact-page shape + the four new request variants.
    write_cbor(
        &out,
        "request-roster-list.cbor",
        &ApiRequest::RosterList {
            transport: TransportId::new("matrix/@me:hs.org"),
            after: Some("@aaa:matrix.org".into()),
            // rung 2 (api/39): the delta anchor (the contact-roster rev last reflected).
            since_rev: Some(5),
        },
    )?;
    write_cbor(
        &out,
        "request-roster-add.cbor",
        &ApiRequest::RosterAdd {
            transport: TransportId::new("matrix/@me:hs.org"),
            contact: daemon_api::ContactInfo {
                id: "@bob:matrix.org".into(),
                display_name: Some("Bob".into()),
                presence: daemon_api::Presence::default(),
                permission: daemon_api::ContactPermission::Allow,
            },
            // rung 3 (api/39): the client-minted idempotency key on a roster-edit lane verb.
            op_id: Some("018f3b9c-roster".into()),
        },
    )?;
    write_cbor(
        &out,
        "request-roster-remove.cbor",
        &ApiRequest::RosterRemove {
            transport: TransportId::new("matrix/@me:hs.org"),
            contact: daemon_api::ContactInfo {
                id: "@bob:matrix.org".into(),
                display_name: None,
                presence: daemon_api::Presence::default(),
                permission: daemon_api::ContactPermission::Unset,
            },
            // rung 3 (api/39): a token-less roster edit (the `? op_id` absent form).
            op_id: None,
        },
    )?;
    write_cbor(
        &out,
        "response-contact-page.cbor",
        &ApiResponse::ContactPage(daemon_api::ContactPage {
            items: vec![daemon_api::ContactInfo {
                id: "@bob:matrix.org".into(),
                display_name: Some("Bob".into()),
                presence: daemon_api::Presence::default(),
                permission: daemon_api::ContactPermission::Allow,
            }],
            next: Some("@bob:matrix.org".into()),
            // rung 1 (api/39): the transport's contact-roster rev (vs `ContactsChanged.rev`).
            rev: 5,
            // rung 2 (api/39): a delta read's removal tombstone (the client prunes it).
            removed: vec!["@gone:matrix.org".into()],
            // rung 3 (api/39): the page-side `origin_ops` map (carrier 2).
            origin_ops: [("@bob:matrix.org".to_string(), "018f3b9c-roster".to_string())]
                .into_iter()
                .collect(),
        }),
    )?;
    // Notifications (wire v37; port-notify): the read-only list op + a response carrying an
    // authorization-request and a connection-error notification, so verify-codec proves the
    // generated zcbor C decoder accepts the notification-info shape + its typed kinds.
    write_cbor(
        &out,
        "request-notification-list.cbor",
        &ApiRequest::NotificationList,
    )?;
    // Deterministic `created_ms`: the `NotificationInfo::new_*` constructors stamp wall-clock
    // `now_ms()`, which would churn this fixture on every run. Pin it to a fixed epoch AFTER
    // construction (runtime behavior is untouched — this is a fixture-only override).
    const FIXTURE_NOTIF_CREATED_MS: u64 = 1_700_000_000_000;
    let mut notif_authz = daemon_api::NotificationInfo::new_authorization(
        Some("notif-authz".into()),
        daemon_api::AuthorizationRequest::new(daemon_api::ContactInfo {
            id: "@bob:matrix.org".into(),
            display_name: Some("Bob".into()),
            presence: daemon_api::Presence::default(),
            permission: daemon_api::ContactPermission::Unset,
        }),
    );
    notif_authz.created_ms = FIXTURE_NOTIF_CREATED_MS;
    let mut notif_conn = daemon_api::NotificationInfo::new_connection_error(
        Some("notif-conn".into()),
        TransportId::new("matrix/@me:hs.org"),
    );
    notif_conn.created_ms = FIXTURE_NOTIF_CREATED_MS;
    write_cbor(
        &out,
        "response-notifications.cbor",
        &ApiResponse::Notifications(daemon_api::RevList {
            // rung 1 (api/39): the notifications rev the client compares against
            // `NotificationsChanged.rev` to skip an unchanged re-list.
            rev: 4,
            items: vec![notif_authz, notif_conn],
        }),
    )?;
    // Persons / metacontacts (wire v37; port-person): the read-only list op + a response carrying
    // an aliased, avatared, multi-endpoint person, so verify-codec proves the generated zcbor C
    // decoder accepts the person shape (incl. the first wire-reachable `image` rule).
    // rung 2 (api/39): the former unit variant becomes a map carrying the delta anchor.
    write_cbor(
        &out,
        "request-person-list.cbor",
        &ApiRequest::PersonList { since_rev: Some(6) },
    )?;
    write_cbor(
        &out,
        "response-persons.cbor",
        &ApiResponse::Persons(daemon_api::RevDeltaList {
            // rung 1 (api/39): the persons rev the client compares against `PersonsChanged.rev`.
            rev: 6,
            items: vec![daemon_api::Person {
                id: "person-ada".into(),
                alias: Some("Ada".into()),
                avatar: Some(daemon_api::Image {
                    blob: daemon_common::BlobRef::new(
                        daemon_common::ContentHash::new([7u8; 32]),
                        3,
                    ),
                }),
                endpoints: vec![
                    daemon_api::PersonEndpoint::new(
                        TransportId::new("matrix/@me:hs.org"),
                        daemon_api::ContactInfo {
                            id: "@ada:hs.org".into(),
                            display_name: Some("Ada L.".into()),
                            presence: daemon_api::Presence::default(),
                            permission: daemon_api::ContactPermission::Allow,
                        },
                    ),
                    daemon_api::PersonEndpoint::new(
                        TransportId::new("discord/bot"),
                        daemon_api::ContactInfo {
                            id: "ada#1234".into(),
                            display_name: None,
                            presence: daemon_api::Presence::default(),
                            permission: daemon_api::ContactPermission::Unset,
                        },
                    ),
                ],
            }],
            // rung 2 (api/39): a delta read's removal tombstone (the client prunes it).
            removed: vec!["person-gone".into()],
            // rung 3 (api/39): the page-side `origin_ops` map (carrier 2).
            origin_ops: [("person-ada".to_string(), "018f3b9c-person".to_string())]
                .into_iter()
                .collect(),
        }),
    )?;
    // Account management (wire v35): the reversible-connect + persisted enabled/label + credential
    // rename requests, so verify-codec proves the generated zcbor C decoder accepts all four new
    // request variants (including the `? label` optional in both set/clear shapes).
    write_cbor(
        &out,
        "request-transport-connect.cbor",
        &ApiRequest::TransportConnect {
            transport: TransportId::new("matrix/@bot:hs.org"),
        },
    )?;
    write_cbor(
        &out,
        "request-transport-set-enabled.cbor",
        &ApiRequest::TransportSetEnabled {
            transport: TransportId::new("matrix/@bot:hs.org"),
            enabled: false,
        },
    )?;
    write_cbor(
        &out,
        "request-transport-set-label.cbor",
        &ApiRequest::TransportSetLabel {
            transport: TransportId::new("matrix/@bot:hs.org"),
            label: Some("Work bot".into()),
        },
    )?;
    write_cbor(
        &out,
        "request-credential-set-label.cbor",
        &ApiRequest::CredentialSetLabel {
            profile: "default".into(),
            label: Some("Personal key".into()),
        },
    )?;
    write_cbor(&out, "request-command-list.cbor", &ApiRequest::CommandList)?;
    write_cbor(
        &out,
        "request-command-invoke.cbor",
        &ApiRequest::CommandInvoke {
            invocation: CommandInvocation {
                name: "help".into(),
                ..Default::default()
            },
        },
    )?;
    // Onboarding (CON-4 / CON-6): credentials + model discovery/selection.
    write_cbor(
        &out,
        "request-credential-set.cbor",
        &ApiRequest::CredentialSet {
            profile: "default".into(),
            secret: "sk-fixture-secret".into(),
        },
    )?;
    write_cbor(
        &out,
        "request-credential-list.cbor",
        &ApiRequest::CredentialList,
    )?;
    write_cbor(
        &out,
        "request-credential-remove.cbor",
        &ApiRequest::CredentialRemove {
            profile: "default".into(),
        },
    )?;
    // Multi-step interactive auth (wire v31): the AuthStep op across every AuthStepInput arm, the
    // reshaped AuthBegun (initial challenge), and AuthStepped across every AuthChallenge +
    // AuthStepResult arm — so verify-codec proves the generated zcbor decoder accepts the new shapes.
    {
        use daemon_api::{
            AuthBeginResponse, AuthChallenge, AuthCompleteResponse, AuthFieldKind, AuthFlowKind,
            AuthParamField, AuthProviderInfo, AuthStepInput, AuthStepRequest, AuthStepResult,
        };
        write_cbor(
            &out,
            "request-auth-step-fields.cbor",
            &ApiRequest::AuthStep(AuthStepRequest {
                flow_id: "flow-1".into(),
                input: AuthStepInput::Fields(std::collections::BTreeMap::from([(
                    "otp".to_string(),
                    "123456".to_string(),
                )])),
            }),
        )?;
        write_cbor(
            &out,
            "request-auth-step-callback.cbor",
            &ApiRequest::AuthStep(AuthStepRequest {
                flow_id: "flow-1".into(),
                input: AuthStepInput::Callback("https://cb.example/?code=xyz&state=s".into()),
            }),
        )?;
        write_cbor(
            &out,
            "request-auth-step-poll.cbor",
            &ApiRequest::AuthStep(AuthStepRequest {
                flow_id: "flow-1".into(),
                input: AuthStepInput::Poll,
            }),
        )?;
        write_cbor(
            &out,
            "response-auth-begun.cbor",
            &ApiResponse::AuthBegun(AuthBeginResponse {
                flow_id: "flow-1".into(),
                challenge: AuthChallenge::Redirect {
                    authorization_url: "https://idp.example/authorize?state=s".into(),
                },
                expires_at: 1_700_000_600,
            }),
        )?;
        write_cbor(
            &out,
            "response-auth-stepped-form.cbor",
            &ApiResponse::AuthStepped(AuthStepResult::Challenge(AuthChallenge::Form {
                title: "Enter the code we texted you".into(),
                fields: vec![AuthParamField {
                    key: "otp".into(),
                    label: "One-time code".into(),
                    required: true,
                    // wire v38: exercise the enriched metadata (a numeric OTP with a hint) so
                    // verify-codec proves the generated C decoder accepts the new optional members.
                    kind: daemon_api::AuthFieldKind::Number,
                    placeholder: Some("123456".into()),
                    ..Default::default()
                }],
            })),
        )?;
        write_cbor(
            &out,
            "response-auth-stepped-qr.cbor",
            &ApiResponse::AuthStepped(AuthStepResult::Challenge(AuthChallenge::Qr {
                payload: "wa://link?token=abc".into(),
                image: Some(vec![0x89, 0x50, 0x4e, 0x47]),
                poll_interval_ms: 2000,
            })),
        )?;
        write_cbor(
            &out,
            "response-auth-stepped-message.cbor",
            &ApiResponse::AuthStepped(AuthStepResult::Challenge(AuthChallenge::Message {
                text: "Approve the login on your other device".into(),
            })),
        )?;
        write_cbor(
            &out,
            "response-auth-stepped-completed.cbor",
            &ApiResponse::AuthStepped(AuthStepResult::Completed(AuthCompleteResponse {
                credential_ref: "matrix/@bot:hs.org".into(),
                account_label: "@bot:hs.org".into(),
                transport_instance: daemon_protocol::TransportId::new("matrix/@bot:hs.org"),
                bound_profile: Some(ProfileRef::new("default")),
            })),
        )?;
        // wire v38: an AuthProviders discovery response advertising the new UserPassword flow with
        // an enriched params schema across every AuthFieldKind (a plain-text username, a MASKED
        // password, and a defaulted Choice) — so verify-codec proves the generated zcbor C decoder
        // accepts the enriched auth-param-field + the new auth-flow-kind arm.
        write_cbor(
            &out,
            "response-auth-providers.cbor",
            &ApiResponse::AuthProviders(vec![AuthProviderInfo {
                family: "userpass".into(),
                flow_kind: AuthFlowKind::UserPassword,
                display_name: "Username & password".into(),
                params_schema: vec![
                    AuthParamField {
                        key: "username".into(),
                        label: "Username".into(),
                        required: true,
                        kind: AuthFieldKind::Text,
                        placeholder: Some("you@example.org".into()),
                        ..Default::default()
                    },
                    AuthParamField {
                        key: "password".into(),
                        label: "Password".into(),
                        required: true,
                        kind: AuthFieldKind::Password,
                        ..Default::default()
                    },
                    AuthParamField {
                        key: "region".into(),
                        label: "Region".into(),
                        required: false,
                        kind: AuthFieldKind::Choice,
                        default: Some("us".into()),
                        choices: vec!["us".into(), "eu".into()],
                        ..Default::default()
                    },
                ],
            }]),
        )?;
    }
    write_cbor(
        &out,
        "request-models.cbor",
        &ApiRequest::Models { after: None },
    )?;
    write_cbor(
        &out,
        "request-set-session-model.cbor",
        &ApiRequest::SetSessionModel {
            session: SessionId::new("fixture-session"),
            model: "claude-opus-4-8".into(),
            provider: Some(ProviderSelector::GenAi),
        },
    )?;
    // Profiles CRUD (PRO-2/3/4): exercise the now-concrete profile-spec (the optional arms -
    // tool_allowlist Some - and the nested budget/tunables maps).
    let mut fixture_spec = ProfileSpec::new("work", ProviderSelector::GenAi, "claude-opus-4-8");
    fixture_spec.tool_allowlist = Some(vec!["read".into(), "search".into()]);
    write_cbor(
        &out,
        "request-profile-create.cbor",
        &ApiRequest::ProfileCreate {
            spec: fixture_spec.clone(),
        },
    )?;
    write_cbor(
        &out,
        "request-profile-update.cbor",
        &ApiRequest::ProfileUpdate {
            spec: fixture_spec.clone(),
        },
    )?;
    write_cbor(
        &out,
        "request-profile-get.cbor",
        &ApiRequest::ProfileGet { id: "work".into() },
    )?;
    write_cbor(
        &out,
        "request-profile-clone.cbor",
        &ApiRequest::ProfileClone {
            source: "default".into(),
            new_id: "work".into(),
        },
    )?;
    write_cbor(
        &out,
        "response-profile.cbor",
        &ApiResponse::Profile(Some(fixture_spec)),
    )?;
    // Persona ops (wire v36): the SoulGet/SoulSet requests + the SoulText response, so
    // `verify-codec` proves the generated zcbor C decoder accepts the new persona shapes (the
    // composed system prompt itself never travels — this is the SOUL.md source text only).
    write_cbor(
        &out,
        "request-soul-get.cbor",
        &ApiRequest::SoulGet { id: "work".into() },
    )?;
    write_cbor(
        &out,
        "request-soul-set.cbor",
        &ApiRequest::SoulSet {
            id: "work".into(),
            text: "You are a focused work assistant.".into(),
        },
    )?;
    write_cbor(
        &out,
        "response-soul-text.cbor",
        &ApiResponse::SoulText("You are a focused work assistant.".into()),
    )?;
    // The profile listing (PRO-1) exercising the wire v31 provenance on `profile-info`: one
    // operator-authored (created_by "operator", no owner) and one agent-authored
    // (created_by {agent}, owner = the authoring session) row, so `verify-codec` proves the
    // generated zcbor C decoder accepts the new optional `created_by`/`owner` fields on both arms.
    let mut op_info = ProfileInfo::from_spec(
        &ProfileSpec::new("work", ProviderSelector::GenAi, "claude-opus-4-8"),
        true,
    );
    op_info.created_by = Some(Author::Operator);
    let mut agent_info = ProfileInfo::from_spec(
        &ProfileSpec::new("agent/s1/helper", ProviderSelector::Mock, "m"),
        false,
    );
    agent_info.created_by = Some(Author::Agent("profile_manage".into()));
    agent_info.owner = Some("s1".into());
    write_cbor(
        &out,
        "response-profiles.cbor",
        &ApiResponse::Profiles(vec![op_info, agent_info]),
    )?;
    // The daemon-api gateway selector (wire `"daemon_api"`): a full profile-spec exercising the new
    // additive `provider-selector` value so `verify-codec` proves the generated zcbor C decoder
    // accepts it (OpenRouter-style `author/slug` model id + the pinned OpenAI-compatible base URL).
    let daemon_api_spec = ProfileSpec {
        base_url: Some("https://api.daemon.ai/api/v1/".into()),
        ..ProfileSpec::new(
            "daemon",
            ProviderSelector::DaemonApi,
            "anthropic/claude-sonnet-4-5",
        )
    };
    write_cbor(
        &out,
        "response-profile-daemon-api.cbor",
        &ApiResponse::Profile(Some(daemon_api_spec)),
    )?;
    // The foreign-engine selector (wire v23; generalized v29): a profile-spec whose `engine` is
    // the foreign arm (`{"Foreign": {"agent": tstr}}` — catalog name only, never a recipe), so
    // `verify-codec` proves the generated zcbor C decoder accepts the `engine-selector` union. The
    // other profile fixtures above exercise the default "Core" arm (always present on new
    // encodings).
    let foreign_engine_spec = ProfileSpec {
        engine: daemon_api::EngineSelector::Foreign {
            agent: "gemini".into(),
        },
        ..ProfileSpec::new("foreign", ProviderSelector::Mock, "")
    };
    write_cbor(
        &out,
        "response-profile-foreign-engine.cbor",
        &ApiResponse::Profile(Some(foreign_engine_spec)),
    )?;
    // The `NodeProvider` foreign backend (wire v30): a Foreign profile routed through the node
    // gateway to a provider+model, so `verify-codec` proves the generated zcbor C decoder accepts
    // the `foreign-backend` union's `NodeProvider` arm (the `foreign-engine` fixture above exercises
    // the default `AgentNative` arm, present on every profile encoding).
    let foreign_node_provider_spec = ProfileSpec {
        engine: daemon_api::EngineSelector::Foreign {
            agent: "codex".into(),
        },
        foreign_backend: daemon_api::ForeignBackend::NodeProvider {
            provider: ProviderSelector::GenAi,
            model: "gpt-4o".into(),
            credential_ref: Some("openai".into()),
        },
        ..ProfileSpec::new("routed", ProviderSelector::Mock, "")
    };
    write_cbor(
        &out,
        "response-profile-foreign-node-provider.cbor",
        &ApiResponse::Profile(Some(foreign_node_provider_spec)),
    )?;
    // The foreign-agent catalog (wire v29): one ACP entry + one stream-json entry, so
    // `verify-codec` proves the generated zcbor C decoder accepts the renamed `agent-entry` shape
    // and both `agent-protocol` values.
    write_cbor(
        &out,
        "response-agent-catalog.cbor",
        &ApiResponse::AgentCatalog(vec![
            daemon_api::AgentEntry {
                name: "gemini".into(),
                recipe: daemon_api::AgentRecipe {
                    program: Some("gemini".into()),
                    args: vec!["--experimental-acp".into()],
                    env: Vec::new(),
                    endpoint: None,
                },
                source: daemon_api::AgentSource::Builtin,
                protocol: daemon_api::AgentProtocol::Acp,
                installed: true,
                version: Some("1".into()),
                capabilities: vec![("fs".into(), "true".into())],
                verification: daemon_api::AgentVerification::Verified,
            },
            daemon_api::AgentEntry {
                name: "claude".into(),
                recipe: daemon_api::AgentRecipe {
                    program: Some("claude".into()),
                    args: vec!["--output-format".into(), "stream-json".into()],
                    env: Vec::new(),
                    endpoint: None,
                },
                source: daemon_api::AgentSource::Manual,
                protocol: daemon_api::AgentProtocol::StreamJson,
                installed: true,
                version: None,
                capabilities: Vec::new(),
                verification: daemon_api::AgentVerification::Unverified,
            },
        ]),
    )?;

    let fixture_descriptor = ModelDescriptor {
        id: "claude-opus-4-8".into(),
        provider: ProviderSelector::GenAi,
        display_name: None,
        context_length: Some(200_000),
        input_price_micros_per_mtok: Some(15_000_000),
        output_price_micros_per_mtok: Some(75_000_000),
        local: false,
    };
    write_cbor(
        &out,
        "response-credentials.cbor",
        &ApiResponse::Credentials(vec![CredentialInfo {
            profile: "default".into(),
            present: true,
            hint: "\u{2026}cret".into(),
            // Wire v35: the node-overlaid human label (`None` on an un-labeled credential).
            label: Some("Personal key".into()),
        }]),
    )?;
    write_cbor(
        &out,
        "response-models.cbor",
        &ApiResponse::Models(daemon_api::WirePage {
            items: vec![fixture_descriptor.clone()],
            next: Some(fixture_descriptor.id.clone()),
        }),
    )?;
    write_cbor(
        &out,
        "response-model-current.cbor",
        &ApiResponse::ModelCurrent(Some(fixture_descriptor)),
    )?;
    // Provider + model discovery (v22): the enumeration op, a credential-aware per-provider listing
    // (with a transient key), and their responses. The response descriptors exercise the additive
    // `provider-descriptor` shape and a `model-descriptor` carrying the optional `display_name`.
    write_cbor(
        &out,
        "request-provider-catalog.cbor",
        &ApiRequest::ProviderCatalog,
    )?;
    write_cbor(
        &out,
        "request-provider-models.cbor",
        &ApiRequest::ProviderModels {
            provider: "anthropic".into(),
            credential_ref: None,
            transient_key: Some("sk-fixture-transient".into()),
            after: None,
        },
    )?;
    write_cbor(
        &out,
        "response-provider-catalog.cbor",
        &ApiResponse::ProviderCatalog(vec![
            ProviderDescriptor {
                id: "daemon_cloud".into(),
                display_name: "Daemon Cloud".into(),
                kind: ProviderKindWire::DaemonCloud,
                wire_selector: ProviderSelector::DaemonApi,
                // Daemon Cloud needs a key to run turns (lists keyless — host-spec semantics).
                requires_key: true,
                supports_model_discovery: true,
                default_base_url: Some("https://api.daemon.ai/api/v1/".into()),
                sign_in: None,
            },
            // The OpenRouter genai row advertises interactive sign-in (wire v30, CON-15): the node
            // states the auth family + label; the client calls `auth_begin { family, params: {} }`.
            ProviderDescriptor {
                id: "open_router".into(),
                display_name: "OpenRouter".into(),
                kind: ProviderKindWire::Cloud,
                wire_selector: ProviderSelector::GenAi,
                requires_key: true,
                supports_model_discovery: true,
                default_base_url: None,
                sign_in: Some(ProviderSignIn {
                    family: "provider/openrouter".into(),
                    label: "Sign in with OpenRouter".into(),
                }),
            },
        ]),
    )?;
    write_cbor(
        &out,
        "response-provider-models.cbor",
        &ApiResponse::ProviderModels(daemon_api::WirePage {
            items: vec![ModelDescriptor {
                id: "anthropic/claude-sonnet-4-5".into(),
                provider: ProviderSelector::DaemonApi,
                display_name: Some("Claude Sonnet 4.5".into()),
                context_length: Some(200_000),
                input_price_micros_per_mtok: Some(3_000_000),
                output_price_micros_per_mtok: Some(15_000_000),
                local: false,
            }],
            next: None,
        }),
    )?;
    // Custom providers (generalized Daemon Cloud): the write-model CRUD ops + the list response, so
    // a non-Rust client and verify-codec exercise the `custom-provider` shape end-to-end.
    write_cbor(
        &out,
        "request-custom-provider-list.cbor",
        &ApiRequest::CustomProviderList,
    )?;
    write_cbor(
        &out,
        "request-custom-provider-set.cbor",
        &ApiRequest::CustomProviderSet {
            provider: daemon_api::CustomProvider {
                id: "custom/my-gateway".into(),
                display_name: "My Gateway".into(),
                base_url: "https://my-gateway.example/v1/".into(),
                wire_selector: ProviderSelector::DaemonApi,
                requires_key: true,
                credential_ref: Some("custom/my-gateway".into()),
                source: daemon_api::CustomProviderSource::User,
            },
        },
    )?;
    write_cbor(
        &out,
        "request-custom-provider-remove.cbor",
        &ApiRequest::CustomProviderRemove {
            id: "custom/my-gateway".into(),
        },
    )?;
    write_cbor(
        &out,
        "response-custom-providers.cbor",
        &ApiResponse::CustomProviders(vec![daemon_api::CustomProvider {
            id: "custom/my-gateway".into(),
            display_name: "My Gateway".into(),
            base_url: "https://my-gateway.example/v1/".into(),
            wire_selector: ProviderSelector::DaemonApi,
            requires_key: true,
            credential_ref: None,
            source: daemon_api::CustomProviderSource::Config,
        }]),
    )?;
    write_cbor(&out, "response-ok.cbor", &ApiResponse::Ok)?;
    write_cbor(
        &out,
        "response-session-page.cbor",
        &ApiResponse::SessionPage(SessionPage {
            sessions: Vec::new(),
            next_cursor: None,
            rev: 0,
            removed: Vec::new(),
            origin_ops: std::collections::BTreeMap::new(),
        }),
    )?;
    // Session recap: the pure-local session recap op (request + a populated response), so
    // verify-codec proves the generated C decoder takes the new shapes end-to-end.
    write_cbor(
        &out,
        "request-session-recap.cbor",
        &ApiRequest::SessionRecap {
            session: SessionId::new("fixture-session"),
        },
    )?;
    write_cbor(
        &out,
        "response-session-recap.cbor",
        &ApiResponse::SessionRecap(Some(daemon_api::SessionRecap {
            title: Some("Docker Networking Help".into()),
            user_turns: 3,
            assistant_turns: 4,
            tool_results: 2,
            top_tools: vec![("fs".into(), 2), ("web_search".into(), 1)],
            files_touched: vec!["src/lib.rs".into()],
            last_ask: Some("why does the bridge drop packets".into()),
            last_reply: Some("the MTU mismatch was the culprit".into()),
        })),
    )?;
    write_cbor(
        &out,
        "response-log-page.cbor",
        &ApiResponse::LogPage(LogPageView {
            entries: Vec::new(),
            next_seq: 0,
            head_seq: 0,
            epoch: 0,
        }),
    )?;
    // wire v37: the richer ChatMessage on the conversation-history surface, so conformance
    // + verify-codec prove the CDDL↔Rust agreement on the new `JournalRecordPayload::Chat` arm and
    // the `chat-message` / `message-attachment` shapes on a real ciborium payload.
    write_cbor(
        &out,
        "response-journal.cbor",
        &ApiResponse::Journal(JournalPageView {
            entries: vec![JournalRecord {
                cursor: 3,
                segment: 1,
                seq: 3,
                epoch: 0,
                trace: 0,
                kind: "block.message".into(),
                timestamp_ms: 911_347_200_000,
                verified: true,
                // rung 3 (api/39): the node-owned envelope's uniform operation provenance
                // (carrier 1) — this record was caused by a client op carrying this id.
                origin_op: Some("018f3b9c-op".into()),
                payload: JournalRecordPayload::Chat {
                    message: Box::new(ChatMessage {
                        id: Some("$evt:hs.org".into()),
                        author: Some(Participant::Contact(ContactInfo {
                            id: "@alice:hs.org".into(),
                            display_name: Some("Alice Smith".into()),
                            ..Default::default()
                        })),
                        replying_to: Some("$prev:hs.org".into()),
                        text: "Now that is a big door".into(),
                        attachments: vec![MessageAttachment {
                            id: "att-1".into(),
                            content_type: Some("image/png".into()),
                            is_inline: true,
                            local_uri: None,
                            remote_uri: Some("mxc://hs.org/abc".into()),
                            size: 4096,
                        }],
                        timestamp: Some(911_347_200),
                        delivered_at: Some(911_347_201),
                        edited_at: None,
                        error: None,
                        title: Some("Titled".into()),
                        highlight_color: Some("#FF00FF".into()),
                        action: false,
                        event: false,
                        notice: false,
                        system: false,
                        highlighted: true,
                    }),
                },
            }],
            next_cursor: 3,
            head_cursor: 3,
            sealed_after: None,
        }),
    )?;
    write_cbor(
        &out,
        "response-events-page.cbor",
        &ApiResponse::EventsPage(EventsPage {
            events: vec![
                NodeEvent::RosterChanged { rev: 7 },
                // v31: the profile-list-changed pointer, so verify-codec proves the generated
                // decoder accepts the new node-event arm.
                NodeEvent::ProfilesChanged { rev: 3 },
                NodeEvent::ApprovalPending {
                    session: SessionId::new("fixture-session"),
                    request_id: "req-1".into(),
                },
                // v26: byte counters on the throttled download-progress event + the payload-free
                // catalog-changed pointer, so verify-codec proves the generated decoder takes both.
                NodeEvent::DownloadProgress {
                    id: daemon_common::DownloadId(1),
                    pct: 50,
                    state: "Downloading".into(),
                    downloaded_bytes: 46_000_000,
                    total_bytes: 92_000_000,
                },
                NodeEvent::CatalogChanged { rev: 2 },
                // v29: the presence-push event, so verify-codec proves the generated decoder
                // accepts the new node-event arm + the connection/presence enums it carries.
                NodeEvent::TransportChanged {
                    transport: TransportId::new("matrix/@bot:hs.org"),
                    connection: ConnectionState::Connected,
                    presence: PresenceState::Unknown,
                    reason: None,
                    message: None,
                    fatal: false,
                    origin_op: None,
                },
                // v30: a disconnect transition carrying a reason/message + the transient
                // Disconnecting state (reconnect/backoff is node-owned; fatal:false = will retry).
                NodeEvent::TransportChanged {
                    transport: TransportId::new("matrix/@bot:hs.org"),
                    connection: ConnectionState::Disconnecting,
                    presence: PresenceState::Offline,
                    reason: Some(DisconnectReason::NetworkError),
                    message: Some("connection reset by peer".into()),
                    fatal: false,
                    origin_op: None,
                },
                // v30: the two membership-push tiers.
                NodeEvent::ConversationsChanged {
                    transport: TransportId::new("matrix/@bot:hs.org"),
                    conv: "!room:hs.org".into(),
                    change: ConvChange::Added,
                    // rung 1 (api/39): the per-transport conversation-set rev.
                    rev: 2,
                    // rung 3 (api/39): carrier-3 provenance (null on adapter-reported changes).
                    origin_op: None,
                },
                NodeEvent::MembershipChanged {
                    transport: TransportId::new("matrix/@bot:hs.org"),
                    conv: "!room:hs.org".into(),
                    member: "@bot:hs.org".into(),
                    change: MembershipChange::Kicked,
                    actor: Some("@admin:hs.org".into()),
                    reason: Some("cleanup".into()),
                    is_self: true,
                    origin_op: None,
                },
                // v34: the roster-changed pointer, so verify-codec proves the generated decoder
                // accepts the new node-event arm.
                NodeEvent::ContactsChanged {
                    transport: TransportId::new("matrix/@bot:hs.org"),
                    // rung 1 (api/39): the per-transport contact-roster rev.
                    rev: 5,
                },
                // wire v37 + rung 1 (api/39): the notifications-changed pointer with its rev.
                NodeEvent::NotificationsChanged { rev: 4 },
                // wire v37 + rung 1 (api/39): the persons-changed pointer with its rev.
                NodeEvent::PersonsChanged { rev: 6 },
                // wire v38: the per-message conversation-history pointer (chat journal), so
                // verify-codec proves the generated decoder accepts the new node-event arm.
                NodeEvent::MessagesChanged {
                    transport: TransportId::new("matrix/@bot:hs.org"),
                    conv: "!room:hs.org".into(),
                    // rung 3 (api/39): carrier-3 provenance — the client send op that caused it.
                    origin_op: Some("018f3b9c-op".into()),
                },
            ],
            next_cursor: 13,
            head_cursor: 13,
            // rung 1 (api/39): the feed generation stamped on every page.
            epoch: Some(1),
        }),
    )?;
    // Tree report (rung 1, api/39): the fleet-rev echo `tree-report.rev` (of `FleetChanged.rev`),
    // so verify-codec proves the generated zcbor C decoder accepts the additive `rev` member. An
    // empty node list keeps the fixture deterministic while exercising the new field + null root.
    write_cbor(
        &out,
        "response-tree.cbor",
        &ApiResponse::Tree(daemon_api::TreeReport {
            root: None,
            nodes: Vec::new(),
            next: None,
            rev: 9,
        }),
    )?;
    write_cbor(
        &out,
        "response-fs-roots.cbor",
        &ApiResponse::FsRoots(Vec::new()),
    )?;
    write_cbor(
        &out,
        "response-commands.cbor",
        &ApiResponse::Commands(Vec::new()),
    )?;
    write_cbor(
        &out,
        "response-command-output.cbor",
        &ApiResponse::CommandOutput(CommandOutput::default()),
    )?;
    write_cbor(
        &out,
        "response-health.cbor",
        &ApiResponse::Health(HealthReport {
            all_ok: true,
            services: vec![ServiceHealth {
                name: "fixture".into(),
                ok: true,
                restarts: 0,
                detail: None,
            }],
        }),
    )?;
    // Local model track (Phase 2): exercise the model arrays + ModelRef/ModelSource through the
    // regenerated (cap-bumped to 64) C codec. The quant is the chosen GGUF file carried as
    // ModelRef::Hf{ file: Some(...) }; ModelId is content-derived (no quant in it).
    {
        use daemon_common::{
            DownloadState, DownloadStatus, InstalledModel, ModelEngine, ModelFile, ModelId,
            ModelRef, ModelSource, QuantCandidate, QuantRecommendation, SearchHit, SearchPage,
            SearchQuery, SearchSort,
        };
        let repo = "bartowski/SmolLM2-135M-Instruct-GGUF";
        let gguf = "SmolLM2-135M-Instruct-Q4_K_M.gguf";
        let hf_ref = || {
            ModelRef::new(
                ModelEngine::Llama,
                ModelSource::Hf {
                    repo: repo.into(),
                    file: Some(gguf.into()),
                    revision: "main".into(),
                },
            )
        };
        write_cbor(
            &out,
            "request-model-search.cbor",
            &ApiRequest::ModelSearch {
                query: SearchQuery {
                    text: "SmolLM2".into(),
                    engine: ModelEngine::Llama,
                    sort: SearchSort::Trending,
                    page: 0,
                    limit: 25,
                },
            },
        )?;
        write_cbor(
            &out,
            "request-model-files.cbor",
            &ApiRequest::ModelFiles {
                repo: repo.into(),
                revision: None,
                engine: ModelEngine::Llama,
                after: None,
            },
        )?;
        write_cbor(
            &out,
            "request-model-download.cbor",
            &ApiRequest::ModelDownload { model: hf_ref() },
        )?;
        write_cbor(
            &out,
            "request-model-downloads.cbor",
            &ApiRequest::ModelDownloads,
        )?;
        write_cbor(
            &out,
            "request-model-catalog.cbor",
            &ApiRequest::ModelCatalog,
        )?;
        write_cbor(
            &out,
            "request-model-recommend.cbor",
            &ApiRequest::ModelRecommend(daemon_api::ModelRecommendArgs {
                repo: repo.into(),
                revision: None,
                engine: ModelEngine::Llama,
                budget_bytes: Some(6 * 1024 * 1024 * 1024),
            }),
        )?;
        // Responses exercise the bumped array caps: search-page.results, [model-file],
        // [download-status], [installed-model], plus the nested quant candidate list.
        write_cbor(
            &out,
            "response-model-search.cbor",
            &ApiResponse::ModelSearch(SearchPage {
                page: 0,
                results: vec![SearchHit {
                    repo: repo.into(),
                    author: Some("bartowski".into()),
                    downloads: 12_345,
                    likes: 42,
                    num_parameters: Some(135_000_000),
                    pipeline_tag: Some("text-generation".into()),
                    last_modified: Some("2025-01-01T00:00:00Z".into()),
                    gated: false,
                    private: false,
                }],
                has_more: false,
            }),
        )?;
        write_cbor(
            &out,
            "response-model-files.cbor",
            &ApiResponse::ModelFiles(daemon_api::WirePage {
                items: vec![
                    ModelFile {
                        path: gguf.into(),
                        size_bytes: 92_000_000,
                        quant: Some("Q4_K_M".into()),
                        is_split: false,
                        is_first_shard: false,
                        is_mmproj: false,
                    },
                    ModelFile {
                        path: "SmolLM2-135M-Instruct-Q8_0.gguf".into(),
                        size_bytes: 145_000_000,
                        quant: Some("Q8_0".into()),
                        is_split: false,
                        is_first_shard: false,
                        is_mmproj: false,
                    },
                    // A vision-projector companion row (wire v27): listed + downloadable, badged
                    // by the client, never a chat model.
                    ModelFile {
                        path: "mmproj-SmolLM2-135M-Instruct-Q8_0.gguf".into(),
                        size_bytes: 6_000_000,
                        quant: Some("Q8_0".into()),
                        is_split: false,
                        is_first_shard: false,
                        is_mmproj: true,
                    },
                ],
                next: None,
            }),
        )?;
        write_cbor(
            &out,
            "response-model-downloads.cbor",
            &ApiResponse::ModelDownloads(vec![DownloadStatus {
                id: daemon_common::DownloadId(1),
                model: hf_ref(),
                state: DownloadState::Downloading,
                downloaded_bytes: 46_000_000,
                total_bytes: 92_000_000,
                files_done: 0,
                files_total: 1,
                error: None,
            }]),
        )?;
        write_cbor(
            &out,
            "response-model-catalog.cbor",
            &ApiResponse::ModelCatalog(vec![InstalledModel {
                id: ModelId::new("smollm2-135m-q4km"),
                model: hf_ref(),
                display_name: "SmolLM2-135M-Instruct".into(),
                local_path: "/cache/models/SmolLM2-135M-Instruct-Q4_K_M.gguf".into(),
                size_bytes: 92_000_000,
                quant: Some("Q4_K_M".into()),
                installed_at_ms: 1_700_000_000_000,
                arch: Some("llama".into()),
                context_length: Some(8192),
                file_type: Some("Q4_K_M".into()),
                // The paired vision-projector companion (wire v27); null for text-only models.
                mmproj_path: Some("/cache/models/mmproj-SmolLM2-135M-Instruct-Q8_0.gguf".into()),
                // The node-local pinned artifact hash surfaced for display (wire v28).
                sha256: Some(
                    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855".into(),
                ),
            }]),
        )?;
        write_cbor(
            &out,
            "response-model-recommend.cbor",
            &ApiResponse::ModelRecommend(QuantRecommendation {
                engine: ModelEngine::Llama,
                repo: repo.into(),
                file: Some(gguf.into()),
                quant: "Q4_K_M".into(),
                size_bytes: Some(92_000_000),
                budget_bytes: 6 * 1024 * 1024 * 1024,
                fits: true,
                reason: "best quality that fits the detected ~6 GiB budget".into(),
                candidates: vec![
                    QuantCandidate {
                        quant: "Q8_0".into(),
                        file: Some("SmolLM2-135M-Instruct-Q8_0.gguf".into()),
                        size_bytes: Some(145_000_000),
                        fits: true,
                    },
                    QuantCandidate {
                        quant: "Q4_K_M".into(),
                        file: Some(gguf.into()),
                        size_bytes: Some(92_000_000),
                        fits: true,
                    },
                ],
            }),
        )?;
        write_cbor(
            &out,
            "response-model-download-started.cbor",
            &ApiResponse::ModelDownloadStarted(daemon_common::DownloadId(1)),
        )?;
    }
    // Multiplexed/server-streaming envelope (wire L0): prove the Rust serde shapes match the
    // wire-c2s / wire-s2c CDDL rules. The client hand-codes this envelope, so these fixtures are the
    // schema gate that keeps both sides in agreement.
    {
        use daemon_api::{WireC2S, WireS2C, WIRE_FEATURE_MUX, WIRE_FEATURE_STREAM, WIRE_VERSION};
        let features = vec![
            WIRE_FEATURE_MUX.to_string(),
            WIRE_FEATURE_STREAM.to_string(),
        ];
        write_cbor(
            &out,
            "wire-c2s-hello.cbor",
            &WireC2S::Hello {
                wire_version: WIRE_VERSION,
                features: features.clone(),
            },
        )?;
        write_cbor(
            &out,
            "wire-c2s-call.cbor",
            &WireC2S::Call {
                id: 1,
                req: ApiRequest::Subscribe {
                    session: SessionId::new("fixture-session"),
                    after_seq: 0,
                    max: 64,
                },
            },
        )?;
        write_cbor(
            &out,
            "wire-c2s-open.cbor",
            &WireC2S::Open {
                id: 2,
                req: ApiRequest::Subscribe {
                    session: SessionId::new("fixture-session"),
                    after_seq: 0,
                    max: 64,
                },
            },
        )?;
        write_cbor(&out, "wire-c2s-cancel.cbor", &WireC2S::Cancel { id: 1 })?;
        write_cbor(
            &out,
            "wire-s2c-hello.cbor",
            &WireS2C::Hello {
                wire_version: WIRE_VERSION,
                features,
                auth_mechanisms: Vec::new(),
            },
        )?;
        write_cbor(
            &out,
            "wire-s2c-reply.cbor",
            &WireS2C::Reply {
                id: 1,
                res: ApiResponse::Ok,
            },
        )?;
        write_cbor(
            &out,
            "wire-s2c-item.cbor",
            &WireS2C::Item {
                id: 1,
                res: ApiResponse::LogPage(LogPageView {
                    entries: Vec::new(),
                    next_seq: 0,
                    head_seq: 0,
                    epoch: 0,
                }),
            },
        )?;
        write_cbor(
            &out,
            "wire-s2c-end.cbor",
            &WireS2C::End { id: 1, error: None },
        )?;
        write_cbor(
            &out,
            "wire-s2c-reset.cbor",
            &WireS2C::Reset {
                id: 1,
                epoch: 0,
                head_seq: 0,
            },
        )?;
    }

    // ----- access control (Auth 5) -----
    write_cbor(
        &out,
        "request-user-create.cbor",
        &ApiRequest::UserCreate {
            username: "alice".into(),
            password: "correct horse".into(),
            roles: vec!["user".into()],
        },
    )?;
    write_cbor(&out, "request-user-list.cbor", &ApiRequest::UserList)?;
    write_cbor(&out, "request-who-am-i.cbor", &ApiRequest::WhoAmI)?;
    write_cbor(
        &out,
        "request-session-revoke.cbor",
        &ApiRequest::SessionRevoke {
            user_id: "u1".into(),
        },
    )?;
    write_cbor(
        &out,
        "request-resource-grant-create.cbor",
        &ApiRequest::ResourceGrantCreate {
            user_id: "u1".into(),
            resource_kind: "session".into(),
            resource_id: "s1".into(),
            capability: "session_read".into(),
        },
    )?;
    write_cbor(
        &out,
        "response-access-user.cbor",
        &ApiResponse::AccessUser(daemon_api::AccessUser {
            user_id: "u1".into(),
            username: "alice".into(),
            disabled: false,
            created_at: 0,
            roles: vec!["user".into()],
        }),
    )?;
    write_cbor(
        &out,
        "response-access-users.cbor",
        &ApiResponse::AccessUsers(Vec::new()),
    )?;
    write_cbor(
        &out,
        "response-access-roles.cbor",
        &ApiResponse::AccessRoles(vec![daemon_api::RoleInfo {
            role: "admin".into(),
            capabilities: vec!["access_admin".into()],
        }]),
    )?;
    write_cbor(
        &out,
        "response-who-am-i.cbor",
        &ApiResponse::WhoAmI(daemon_api::PrincipalView {
            user_id: "u1".into(),
            username: "alice".into(),
            roles: vec!["admin".into()],
            capabilities: vec!["access_admin".into()],
        }),
    )?;

    // -- user feedback over OpenTelemetry (N1; wire v31) -----------------------------------------
    write_cbor(
        &out,
        "request-feedback-submit.cbor",
        &ApiRequest::FeedbackSubmit {
            kind: daemon_api::FeedbackKind::Response,
            target: Some(daemon_api::FeedbackTarget {
                session: "s-fixture".into(),
                cursor: 42,
                trace: Some(daemon_common::TraceId(0x1234)),
            }),
            rating: Some(daemon_api::FeedbackRating::Up),
            comment: Some("nailed it".into()),
            include_content: true,
            diagnostics: Some(daemon_api::FeedbackDiagnostics {
                app_version: Some("1.2.3".into()),
                os: Some("linux".into()),
            }),
            surface: "transcript".into(),
        },
    )?;
    write_cbor(
        &out,
        "request-telemetry-consent-get.cbor",
        &ApiRequest::TelemetryConsentGet,
    )?;
    write_cbor(
        &out,
        "request-telemetry-consent-set.cbor",
        &ApiRequest::TelemetryConsentSet { enabled: true },
    )?;
    write_cbor(
        &out,
        "response-feedback-ack.cbor",
        &ApiResponse::FeedbackAck(daemon_api::FeedbackAck {
            accepted: true,
            queued: true,
        }),
    )?;
    write_cbor(
        &out,
        "response-telemetry-consent.cbor",
        &ApiResponse::TelemetryConsent { enabled: true },
    )?;
    // Crash-reporting consent (wire v41): get/set ops + the reply (mirrors telemetry-consent).
    write_cbor(
        &out,
        "request-crash-consent-get.cbor",
        &ApiRequest::CrashConsentGet,
    )?;
    write_cbor(
        &out,
        "request-crash-consent-set.cbor",
        &ApiRequest::CrashConsentSet { enabled: true },
    )?;
    write_cbor(
        &out,
        "response-crash-consent.cbor",
        &ApiResponse::CrashConsent { enabled: true },
    )?;
    // Saved presences (wire v37): the list/save/delete/set-active ops + the listing reply.
    {
        use daemon_api::{PresencePrimitive, SavedPresence};
        write_cbor(
            &out,
            "request-presence-list.cbor",
            &ApiRequest::PresenceList,
        )?;
        let fixture_presence = SavedPresence {
            id: "ffffffff-ffff-ffff-ffff-ffffffffffff".into(),
            name: Some("Streaming".into()),
            primitive: PresencePrimitive::Streaming,
            message: Some("live on twitch".into()),
            emoji: Some("💀".into()),
            last_used: Some(1_700_000_000),
            use_count: 7,
        };
        write_cbor(
            &out,
            "request-presence-save.cbor",
            &ApiRequest::PresenceSave {
                presence: fixture_presence.clone(),
            },
        )?;
        write_cbor(
            &out,
            "request-presence-delete.cbor",
            &ApiRequest::PresenceDelete {
                id: "ffffffff-ffff-ffff-ffff-ffffffffffff".into(),
            },
        )?;
        write_cbor(
            &out,
            "request-presence-set-active.cbor",
            &ApiRequest::PresenceSetActive {
                id: "ffffffff-ffff-ffff-ffff-ffffffffffff".into(),
            },
        )?;
        write_cbor(
            &out,
            "response-saved-presences.cbor",
            &ApiResponse::SavedPresences(vec![fixture_presence]),
        )?;
    }

    // -- file transfer (wire v37) --------------------------------------------------------------
    {
        use daemon_api::{FileTransfer, FileTransferDirection, FileTransferState};
        use daemon_common::{BlobRef, ContentHash};

        let blob = BlobRef {
            hash: ContentHash::new([7u8; 32]),
            size: 1337,
            name: Some("cat.png".into()),
            mime: Some("image/png".into()),
        };
        write_cbor(
            &out,
            "request-ft-send.cbor",
            &ApiRequest::FtSend {
                transport: TransportId::new("matrix/@bot:localhost"),
                transfer: FileTransfer {
                    name: "cat.png".into(),
                    blob: blob.clone(),
                    direction: FileTransferDirection::Send,
                    state: FileTransferState::Negotiating,
                    file_size: 1337,
                    content_type: Some("image/png".into()),
                    message: Some("here you go".into()),
                    ..Default::default()
                },
                // rung 3 (api/39): the client-minted idempotency key on the retry-sensitive
                // direct FtSend verb (FtSend as an outboxed lane is deferred; §15).
                op_id: Some("018f3b9c-ft".into()),
            },
        )?;
        write_cbor(
            &out,
            "request-ft-receive.cbor",
            &ApiRequest::FtReceive {
                transport: TransportId::new("matrix/@bot:localhost"),
                transfer: FileTransfer {
                    name: "cat.png".into(),
                    blob,
                    direction: FileTransferDirection::Receive,
                    file_size: 1337,
                    source: Some("mxc://localhost/abc123".into()),
                    ..Default::default()
                },
            },
        )?;
    }

    // -- transport account settings (N2; wire v38) --------------------------------------------
    // The settings read + merge-edit of a transport instance's persisted NON-SECRET values, so
    // verify-codec proves the generated zcbor C decoder accepts the map-carrying shapes.
    {
        use daemon_api::AccountSettingsValues;

        let mut values = std::collections::BTreeMap::new();
        values.insert("server".to_string(), "hs.example.org".to_string());
        values.insert("nick".to_string(), "daemon-bot".to_string());
        write_cbor(
            &out,
            "request-transport-settings.cbor",
            &ApiRequest::TransportSettings {
                transport: TransportId::new("matrix/@bot:hs.org"),
            },
        )?;
        write_cbor(
            &out,
            "request-transport-configure.cbor",
            &ApiRequest::TransportConfigure {
                transport: TransportId::new("matrix/@bot:hs.org"),
                settings: AccountSettingsValues {
                    values: values.clone(),
                },
                // rung 3 (api/39): the client-minted idempotency key (retry-safe settings apply).
                op_id: Some("018f3b9c-cfg".into()),
            },
        )?;
        write_cbor(
            &out,
            "response-transport-settings.cbor",
            &ApiResponse::TransportSettings(AccountSettingsValues { values }),
        )?;
    }

    println!("generated CBOR fixtures in {}", out.display());
    Ok(())
}

fn write_cbor<T: serde::Serialize>(dir: &Path, name: &str, value: &T) -> anyhow::Result<()> {
    let bytes = daemon_api::to_cbor(value);
    std::fs::write(dir.join(name), bytes)?;
    Ok(())
}

/// Base name passed to the codegen script; the generated entry types are `api_request`/`api_response`.
const ZCBOR_BASENAME: &str = "daemon_api_client";

fn codegen_script(root: &Path) -> PathBuf {
    root.join("crates/contracts/daemon-api/zcbor-codegen.sh")
}

/// The single authoritative CDDL. It is authored in zcbor dialect (quoted map keys, named union
/// arms, labeled tuples, `any` for opaque fields, plus a few `-t` rule-name disambiguators) so the
/// one file both documents the full wire contract and generates the client C codec. `verify-codec`
/// proves the generated decoder accepts real ciborium fixtures; the `daemon-api` cddl-cat
/// conformance tests prove the schema matches the serde wire format.
fn default_cddl(root: &Path) -> PathBuf {
    root.join("crates/contracts/daemon-api/daemon-api.cddl")
}

/// Run the canonical codegen script. `extra` forwards flags such as `--copy-sources`.
fn run_codegen(root: &Path, cddl: &Path, out: &Path, extra: &[&str]) -> anyhow::Result<()> {
    let status = Command::new("bash")
        .arg(codegen_script(root))
        .arg(cddl)
        .arg(out)
        .args(extra)
        .status()
        .map_err(|e| {
            anyhow::anyhow!(
                "running zcbor-codegen.sh (is zcbor on PATH / in the flake shell?): {e}"
            )
        })?;
    anyhow::ensure!(status.success(), "zcbor codegen failed with {status}");
    Ok(())
}

/// `gen-zcbor [--cddl <path>] [--out <dir>]` — (re)generate the client CBOR codec.
///
/// A thin dev wrapper over `zcbor-codegen.sh`. daemon-node owns generation because the CDDL is
/// authoritative here and zcbor lives in this flake; the output is the committed artifact
/// `daemon-app` vendors (no Python/zcbor in the Qt build). The superproject's pure
/// `packages.daemon-zcbor-codec` derivation invokes the same script.
fn gen_zcbor(cddl: Option<PathBuf>, out: Option<PathBuf>) -> anyhow::Result<()> {
    let root = workspace_root();
    let cddl = cddl.unwrap_or_else(|| default_cddl(&root));
    let out = out.unwrap_or_else(|| root.join("target/zcbor-codec"));
    run_codegen(&root, &cddl, &out, &[])?;
    println!(
        "generated zcbor codec from {} in {}",
        cddl.display(),
        out.display()
    );
    Ok(())
}

/// The verify-codec harness: decode every ciborium-produced fixture with the zcbor-generated decoder.
/// A `response-*` filename is decoded as `api_response`, anything else as `api_request`; success
/// means the generated decoder accepted the bytes (ZCBOR_SUCCESS) and consumed all of them.
const VERIFY_CODEC_C: &str = r#"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "daemon_api_client_decode.h"

static unsigned char buf[1u << 20];

int main(int argc, char **argv) {
    int failures = 0;
    for (int i = 1; i < argc; i++) {
        const char *path = argv[i];
        FILE *f = fopen(path, "rb");
        if (!f) {
            fprintf(stderr, "FAIL %s: cannot open\n", path);
            failures++;
            continue;
        }
        size_t n = fread(buf, 1, sizeof buf, f);
        fclose(f);

        const char *base = strrchr(path, '/');
        base = base ? base + 1 : path;

        size_t consumed = 0;
        int ret;
        if (strncmp(base, "response", 8) == 0) {
            struct api_response_r *r = calloc(1, sizeof *r);
            ret = cbor_decode_api_response(buf, n, r, &consumed);
            free(r);
        } else {
            struct api_request_r *r = calloc(1, sizeof *r);
            ret = cbor_decode_api_request(buf, n, r, &consumed);
            free(r);
        }

        if (ret != 0) {
            fprintf(stderr, "FAIL %s: zcbor decode error %d\n", base, ret);
            failures++;
        } else if (consumed != n) {
            fprintf(stderr, "FAIL %s: decoded %zu of %zu bytes\n", base, consumed, n);
            failures++;
        } else {
            fprintf(stderr, "ok   %s (%zu bytes)\n", base, n);
        }
    }

    if (failures) {
        fprintf(stderr, "%d fixture(s) failed to decode\n", failures);
        return 1;
    }
    fprintf(stderr, "all fixtures decoded with the generated zcbor codec\n");
    return 0;
}
"#;

/// `verify-codec` — prove the generated C codec accepts real ciborium wire bytes.
///
/// Closes the loop the syntactic `cddl` gate cannot: generate the codec from the CDDL, compile its
/// decoder with the zcbor runtime, then decode every `fixtures/cbor/*.cbor` (each emitted by
/// `api-fixtures` through ciborium — the runtime truth) and assert success + full consumption. Any
/// drift between the serde wire format and the CDDL/zcbor path fails here.
fn verify_codec() -> anyhow::Result<()> {
    let root = workspace_root();

    let fixtures_dir = root.join("crates/contracts/daemon-api/fixtures/cbor");
    if !fixtures_dir.exists() {
        gen_api_fixtures()?;
    }
    let mut fixtures: Vec<PathBuf> = std::fs::read_dir(&fixtures_dir)
        .map_err(|e| anyhow::anyhow!("reading {}: {e}", fixtures_dir.display()))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().map(|ext| ext == "cbor").unwrap_or(false))
        // The multiplexed-envelope fixtures (`wire-c2s-*` / `wire-s2c-*`) are NOT `api-request` /
        // `api-response`, and the vendored C codec is deliberately scoped to those two entry types
        // (the client hand-codes the tiny envelope). Their schema is covered by the cddl-cat
        // conformance test against `wire-c2s` / `wire-s2c`, not this generated-decoder harness.
        .filter(|path| {
            !path
                .file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with("wire-"))
                .unwrap_or(false)
        })
        .collect();
    fixtures.sort();
    anyhow::ensure!(
        !fixtures.is_empty(),
        "no CBOR fixtures in {}",
        fixtures_dir.display()
    );

    // Decode every committed fixture with the generated codec (an independent C cross-check of the
    // serde wire bytes). Per-variant coverage is no longer asserted here: the unified CDDL now spans
    // the full surface (~150 variants), and the comprehensive "Rust output always matches the CDDL"
    // gate is the cddl-cat round-trip + proptest conformance in the `daemon-api` crate. This harness
    // proves the zcbor-generated decoder agrees with ciborium on the fixtures that exist.
    let work = std::env::temp_dir().join(format!("daemon-verify-codec-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&work);
    let codec = work.join("codec");
    std::fs::create_dir_all(&codec)?;
    // `--copy-sources` drops the zcbor C runtime flat alongside the generated codec.
    run_codegen(&root, &default_cddl(&root), &codec, &["--copy-sources"])?;

    let harness_c = work.join("verify_codec.c");
    std::fs::write(&harness_c, VERIFY_CODEC_C)?;
    let bin = work.join("verify-codec");

    let status = Command::new("cc")
        .arg(&harness_c)
        .arg(codec.join(format!("{ZCBOR_BASENAME}_decode.c")))
        .arg(codec.join("zcbor_decode.c"))
        .arg(codec.join("zcbor_common.c"))
        .arg(format!("-I{}", codec.display()))
        .arg("-o")
        .arg(&bin)
        .status()
        .map_err(|e| anyhow::anyhow!("failed to run cc (is it in the flake shell?): {e}"))?;
    anyhow::ensure!(
        status.success(),
        "compiling the verify harness failed with {status}"
    );

    let status = Command::new(&bin).args(&fixtures).status()?;
    anyhow::ensure!(
        status.success(),
        "codec verification failed: a fixture did not decode with the generated codec"
    );
    let _ = std::fs::remove_dir_all(&work);
    println!(
        "verified {} fixtures decode with the generated zcbor codec",
        fixtures.len()
    );
    Ok(())
}

/// The red-lined scan: the provenance constructors must not be reachable from non-test code.
///
/// The authoring migration made the module's own assessment the single source of a resource plan, and
/// it made that structural — a seat receives derived requirements and has no constructor to invent one
/// with. But three constructors *do* mint provenance, and for them the guarantee is about **who may
/// call them**, which no type can express:
///
/// - `from_module_assessment` stamps a plan as module-derived. Called anywhere but the assessment path,
///   it would let a hand-authored plan claim a module produced it — the exact drift the invariant
///   exists to prevent, wearing the invariant's own badge.
/// - `ModuleDerivedPlan::fixture` and `fixture_authored_execution` mint a plan no module produced.
///   Honest in a fixture that pins digests resolving to no bytes; a silent bypass on the real path.
///
/// A naming convention is not a gate, so this is a gate. The heuristic — an occurrence is allowed if a
/// `#[cfg(test)]` appears earlier in the file — is sound *here* because clippy's `items after a test
/// module` already forbids the arrangement that would defeat it: non-test items cannot follow a test
/// module, so anything after the first `#[cfg(test)]` is test code.
fn provenance_scan() -> anyhow::Result<()> {
    /// `(symbol, allowed non-test sites, why)`. An allowed site is a path suffix.
    const GUARDED: &[(&str, &[&str], &str)] = &[
        (
            "ModuleDerivedPlan::from_module_assessment",
            &[
                // The definition, and the assessment path that is the single source.
                "contracts/daemon-vhc-proto/src/execution_requirements.rs",
                "host/daemon-vhc-host/src/run/admission.rs",
            ],
            "only the module's own assessment may stamp a plan as module-derived",
        ),
        (
            "ModuleDerivedPlan::fixture",
            &[
                "contracts/daemon-vhc-proto/src/execution_requirements.rs",
                // The testkit's labelled fixture helper. The testkit is test tooling by charter and
                // is already exempt wholesale from the dependency-direction rules above; the gate
                // that matters for it is that its CALLERS are test targets, which is checked below
                // for `fixture_authored_execution`.
                "host/daemon-vhc-testkit/src/live_genesis.rs",
            ],
            "a fixture's plan must not be mintable on the real authoring path",
        ),
        (
            "fixture_authored_execution",
            &[
                "host/daemon-vhc-testkit/src/live_genesis.rs",
                // The ceremony seat's own in-crate tests live beside it in a `#[cfg(test)]` module,
                // which the earlier-cfg rule already covers; this entry is for the module path.
                "host/daemon-vhc-testkit/src/ceremony.rs",
            ],
            "authoring a genesis from a fixture's requirements must stay in test targets",
        ),
        (
            // The fourth minting constructor, found while verifying the authoring-site migration. It
            // stamps a role's execution requirements over a trivial plan no module produced — the same
            // provenance forgery as the three above, and it was outside the gate because the gate was
            // written before it existed. Every current caller is a test target; this is what keeps
            // that true.
            "RoleExecutionRequirements::fixture_over_trivial_plan",
            &["contracts/daemon-vhc-proto/src/execution_requirements.rs"],
            "a role's execution requirements must come from a module's own assessment, never from a \
             fixture's trivial plan, on any shipping path",
        ),
    ];

    let mut violations: Vec<String> = Vec::new();
    let mut checked = 0usize;
    for path in rust_sources(std::path::Path::new("crates"))? {
        let display = path.display().to_string();
        // A test target is test code wholesale: an integration test directory, or the conventional
        // in-module `tests.rs`.
        let is_test_target = display.contains("/tests/")
            || display.ends_with("/tests.rs")
            || display.contains("/benches/");
        if is_test_target {
            continue;
        }
        let text = std::fs::read_to_string(&path)?;
        checked += 1;
        let first_cfg_test = text.find("#[cfg(test)]").unwrap_or(usize::MAX);
        for (symbol, allowed, why) in GUARDED {
            // The literal spelling. All three are associated functions or a free function reached by
            // path, so the qualified form is the only way to call them — a bare-name search instead
            // matched the word "fixture" in unrelated crates and reported 58 phantom violations.
            let needle = *symbol;
            let mut from = 0usize;
            while let Some(at) = text[from..].find(needle) {
                let at = from + at;
                from = at + needle.len();
                if at > first_cfg_test {
                    continue; // test code, per the note above
                }
                if allowed.iter().any(|ok| display.ends_with(ok)) {
                    continue;
                }
                let line = text[..at].lines().count();
                violations.push(format!("{display}:{line} reaches `{symbol}` — {why}"));
            }
        }
    }

    if !violations.is_empty() {
        eprintln!("\nprovenance scan violations:");
        for v in &violations {
            eprintln!("  x {v}");
        }
        anyhow::bail!(
            "{} provenance violation(s): a fixture or an unstamped plan is reachable from non-test \
             code",
            violations.len()
        );
    }
    println!(
        "ok: provenance constructors unreachable from non-test code ({checked} files scanned)"
    );
    Ok(())
}

/// Every `.rs` file under `root`, excluding build artifacts.
fn rust_sources(root: &std::path::Path) -> anyhow::Result<Vec<std::path::PathBuf>> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir)? {
            let entry = entry?;
            let path = entry.path();
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if entry.file_type()?.is_dir() {
                if name != "target" && name != ".git" {
                    stack.push(path);
                }
            } else if name.ends_with(".rs") {
                out.push(path);
            }
        }
    }
    out.sort();
    Ok(out)
}

/// The tracked ABI specification states values the code defines — check that they still agree.
///
/// A text mirror can only drift from another text. A specification drifts from the **implementation**,
/// and that is the drift that costs something: an author reading a stated constant, writing a profile
/// range or a permitted minor against it, and being wrong. So the check compares the spec's own
/// sentences against `daemon-vhc-abi`'s constants rather than against a copy of itself.
///
/// It is deliberately narrow: the values a reader would act on. A spec sentence the code contradicts
/// fails; prose the code has no opinion about is not this gate's business.
///
/// # Errors
/// Names every statement that no longer matches, with the value the code defines.
fn vhc_abi_spec_drift() -> anyhow::Result<()> {
    let spec_path = workspace_root().join("docs/specs/vhc-module-abi-spec.md");
    let spec = std::fs::read_to_string(&spec_path)
        .map_err(|e| anyhow::anyhow!("read {}: {e}", spec_path.display()))?;

    // `(what the spec must state, the value the code defines)`. Each is a value a reader acts on: a
    // declared minor, an export name a module must provide, a journal tag a replay must decode, or a
    // bound a guest is clamped to.
    let expectations: Vec<(String, String)> = vec![
        (
            format!("`DA_ABI_MAJOR_V2` | `{}`", daemon_vhc_abi::DA_ABI_MAJOR_V2),
            "the ABI major".into(),
        ),
        (
            format!("`DA_ABI_MINOR_V2` | `{}`", daemon_vhc_abi::DA_ABI_MINOR_V2),
            "the highest implemented minor".into(),
        ),
        (
            format!(
                "`CERTIFICATION_MINOR_V2` | `{}`",
                daemon_vhc_abi::CERTIFICATION_MINOR_V2
            ),
            "the certification minor".into(),
        ),
        (
            format!(
                "`LEGACY_CONTEXT_MAX_MINOR` | `{}`",
                daemon_vhc_abi::LEGACY_CONTEXT_MAX_MINOR
            ),
            "the highest legacy minor".into(),
        ),
        (
            daemon_vhc_abi::DA_RESOURCE_PLAN_EXPORT.to_string(),
            "the resource-plan export name".into(),
        ),
        (
            daemon_vhc_abi::DA_APPLY_EXECUTION_GRANT_EXPORT.to_string(),
            "the grant-application export name".into(),
        ),
        (
            format!("**Tag {}**", daemon_vhc_abi::JOURNAL_TAG_EXECUTION_GRANT),
            "the grant-application journal tag".into(),
        ),
        (
            "`LOG_CALLS_PER_PHASE_MAX`".into(),
            "the per-phase log call bound".into(),
        ),
        (
            "`LOG_BYTES_PER_PHASE_MAX`".into(),
            "the per-phase log byte bound".into(),
        ),
        (
            "`LOG_MESSAGE_BYTES_MAX`".into(),
            "the per-message log byte bound".into(),
        ),
    ];

    let mut missing: Vec<String> = Vec::new();
    for (needle, what) in &expectations {
        if !spec.contains(needle) {
            missing.push(format!("  {what}: the spec does not state `{needle}`"));
        }
    }

    // The closed context domain: the spec's table must carry every canonical string the code renders,
    // or a reader cannot tell which context a record's string belongs to. Enumerated here rather than
    // iterated, so adding a twelfth variant to the code fails this gate until the table says so — which
    // is the point, the domain being closed at eleven.
    use daemon_vhc_abi::execution_context::ExecutionContext;
    let domain = [
        ExecutionContext::Init,
        ExecutionContext::Migrate,
        ExecutionContext::RunBeforeFirstSlice,
        ExecutionContext::RunBetweenSlices,
        ExecutionContext::RunAfterLastSlice,
        ExecutionContext::RunSlice(0),
        ExecutionContext::Assessment,
        ExecutionContext::Claim,
        ExecutionContext::ResourcePlan,
        ExecutionContext::Manifest,
        ExecutionContext::ExecutionGrant,
    ];
    for context in &domain {
        let rendered = context.render();
        // The slice context is parameterized, so the table states its shape rather than one instance.
        let needle = if rendered.starts_with(daemon_vhc_abi::SLICE_CONTEXT_PREFIX) {
            format!("`{}<canonical u64>`", daemon_vhc_abi::SLICE_CONTEXT_PREFIX)
        } else {
            format!("`{rendered}`")
        };
        if !spec.contains(&needle) {
            missing.push(format!(
                "  the execution-context table is missing {needle}, which the code renders"
            ));
        }
    }

    anyhow::ensure!(
        missing.is_empty(),
        "the tracked ABI specification has drifted from the code it documents:\n{}\n\
         Fix the specification (or the code, if the code is what moved) — a stated constant a reader \
         acts on must be the one the implementation defines.",
        missing.join("\n")
    );
    println!(
        "ok: the tracked ABI specification agrees with the code on {} stated values + the {}-value \
         execution-context domain",
        expectations.len(),
        domain.len()
    );
    Ok(())
}

/// The red-lined scan: this program's private vocabulary MUST NOT reach the code.
///
/// Defect letters, wave ids, measurement finding ids and divergence ids are how a program talks about
/// itself while it is running. They are useless to anyone reading the code afterwards — worse than
/// useless, because they look like references to something a reader could go and find, and the thing
/// they name is a status document that was never shipped. What a comment must carry is the *substance*:
/// what is true and why, in words that survive the program that discovered them.
///
/// Specification **rule identifiers** are the opposite and are deliberately not scanned: `[RC-4]`,
/// `[SF-6]`, `[EB-4]` and their kin name normative text that ships in `docs/specs`, and tracked code
/// cites them precisely so a reader can find the rule.
///
/// **The needles are assembled from fragments at runtime**, so this function's own source contains none
/// of them whole. A scanner that had to exempt its own file would be a scanner with a hole in it, and
/// the hole would be in exactly the file someone edits when they want to add a codename.
///
/// # Errors
/// Every occurrence, with its file, line and why the vocabulary is forbidden.
fn vhc_codename_scan() -> anyhow::Result<()> {
    // `(prefix, suffix, what it is)` — the needle is `prefix + suffix`.
    const FORBIDDEN: &[(&str, &str, &str)] = &[
        ("MEAS", "-F", "a measurement-wave finding id"),
        ("MEAS", "-O", "a measurement-wave observation id"),
        ("DV", "-1", "a program divergence id"),
        ("DV", "-2", "a program divergence id"),
        ("W-", "SF", "a program wave id"),
        ("W-", "host", "a program wave id"),
        ("W-", "guest", "a program wave id"),
        ("W-", "measure", "a program wave id"),
        (
            "wait",
            "point",
            "the program's own status-document vocabulary",
        ),
        ("Stage", " R ", "a program stage label"),
    ];

    let mut violations: Vec<String> = Vec::new();
    let mut scanned = 0usize;
    // The tracked specifications are scanned too. They ship inside every certification candidate, so a
    // spec citing a wave id names a document its reader does not have — the same defect as a comment
    // doing it, in a document with a wider audience.
    let mut targets: Vec<PathBuf> = [
        "vhc-architecture-spec.md",
        "vhc-module-abi-spec.md",
        "vhc-fleet-ceremony-runbook.md",
    ]
    .iter()
    .map(|name| workspace_root().join("docs/specs").join(name))
    .filter(|p| p.is_file())
    .collect();
    for root in ["crates", "xtask", "bins", "tests"] {
        let dir = workspace_root().join(root);
        if dir.is_dir() {
            targets.extend(rust_sources(&dir)?);
        }
    }
    {
        for path in targets {
            let text = std::fs::read_to_string(&path)?;
            scanned += 1;
            for (prefix, suffix, what) in FORBIDDEN {
                let needle = format!("{prefix}{suffix}");
                for (lineno, line) in text.lines().enumerate() {
                    if line.contains(&needle) {
                        violations.push(format!(
                            "  {}:{}: `{needle}` is {what}; say what is true instead",
                            path.display(),
                            lineno + 1
                        ));
                    }
                }
            }
        }
    }

    anyhow::ensure!(
        violations.is_empty(),
        "{} occurrence(s) of program vocabulary in the code:\n{}\n\
         Program vocabulary lives in the plan and status documents. A comment that cites one names a \
         document the reader does not have; write the substance instead.",
        violations.len(),
        violations.join("\n")
    );
    println!(
        "ok: no program vocabulary in {scanned} tracked sources and specifications ({} needles)",
        FORBIDDEN.len()
    );
    Ok(())
}
