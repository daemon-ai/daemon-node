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

mod publish;
mod tokenize;

use clap::{Parser, Subcommand};
use daemon_vhc_session::data::TokenWidth;
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
    /// Build the swarm guest experiment modules (`guests/`) for `wasm32-unknown-unknown`.
    BuildGuests,
    /// Run the swarm **CI tier-1** suite: the CPU-only, consensus-critical determinism / round-
    /// protocol / codec / wasm-guest suites (TDD §8.1 tier 1). Builds the guests first, then runs the
    /// pinned suite list, failing on the first red. No GPU, no live substrate (env-gated live tests
    /// skip). This is the single in-repo definition of the per-PR swarm gate — the superproject CI
    /// job and a local operator both invoke `cargo run -p xtask -- swarm-ci-det`.
    SwarmCiDet,
    /// Run the swarm **CI tier-2** whole-run suites (decisions D4): the deterministic sim/testkit
    /// whole runs as they land — SDK-side `daemon-vhc-sim` native whole runs (the SPARTA
    /// continuous-averaging toy over the virtual worlds) and host-side `daemon-vhc-testkit` whole
    /// runs over the PRODUCTION wasm blobs (wasmtime + simulated capability providers, journaled,
    /// §8.7 replay-verified). Heavier than tier-1 (wasmtime + guest builds), so it is a separate
    /// gate invoked as `cargo run -p xtask -- swarm-ci-t2`, never part of `swarm-ci-det`.
    SwarmCiT2,
    /// Enforce the daemon-vhc dependency-direction rules (architecture §7): `host/*` never links
    /// `sdk/*`, `contracts/*` links neither, `sdk/*` never links `host/*`. The honest current
    /// exceptions are listed inline and each is tracked to the phase that removes it.
    VhcDepCheck,
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
        /// Optional cap on total tokens emitted (keeps a vendored fixture small).
        #[arg(long)]
        max_tokens: Option<u64>,
    },
    /// Publish an experiment module to the payload store at `modules/<blake3>.wasm` (P3 lane S).
    PublishModule {
        /// The `.wasm` module to upload.
        #[arg(long)]
        module: PathBuf,
        /// The run id whose prefix the object lives under (`runs/<run>/modules/…`).
        #[arg(long)]
        run: String,
        /// The coordinator presign base (e.g. `https://…/api/v1/swarm`).
        #[arg(long)]
        presign_base: String,
        /// `swarm:*`-scoped bearer token (gateway path).
        #[arg(long)]
        bearer: Option<String>,
        /// Internal identity org id (direct-to-`apps/swarm` dev path; pair with `--actor`).
        #[arg(long)]
        org: Option<String>,
        /// Internal identity actor (pair with `--org`).
        #[arg(long)]
        actor: Option<String>,
    },
    /// Publish a pre-tokenized corpus (shards + manifest) to the payload store by content hash (P3 S).
    PublishCorpus {
        /// The `manifest.json` produced by `tokenize-corpus` (its shards sit beside it).
        #[arg(long)]
        manifest: PathBuf,
        /// The run id whose prefix the objects live under (`runs/<run>/corpus/…`).
        #[arg(long)]
        run: String,
        /// The coordinator presign base.
        #[arg(long)]
        presign_base: String,
        /// `swarm:*`-scoped bearer token (gateway path).
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
        Cmd::BuildGuests => build_guests(),
        Cmd::SwarmCiDet => swarm_ci_det(),
        Cmd::SwarmCiT2 => swarm_ci_t2(),
        Cmd::VhcDepCheck => vhc_dep_check(),
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

/// Build the swarm guest experiment modules for `wasm32-unknown-unknown`.
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

    // xtask is dev tooling; the crate-level `#![allow(clippy::disallowed_methods)]` covers this
    // developer-controlled spawn.
    let status = Command::new("cargo")
        .current_dir(&guests)
        // The devShell pins `CARGO_TARGET_DIR` to the parent checkout's `target/`; left inherited it
        // redirects the guests' wasm out of `guests/target/` (where the test harness reads them). The
        // guests are their own workspace, so clear it and let cargo default to `guests/target/`.
        .env_remove("CARGO_TARGET_DIR")
        // Remap the absolute checkout + cargo-registry prefixes rustc bakes into panic locations.
        // Together with the guests workspace's COMMITTED Cargo.lock (B3 sitting 2 — without it,
        // floating registry patch versions re-hashed every SDK-linking guest between builds), this
        // makes the `.wasm` bytes byte-reproducible across clean rebuilds within one checkout path.
        // The remaining cross-worktree/machine variance — cargo derives each path package's
        // `-C metadata` crate-disambiguator from its absolute manifest dir, which reorders the
        // linked module (remap rewrites path *strings*, not that hash) — is removed by the guests
        // workspace's `.cargo/config.toml` `rustc-wrapper` (`guest-rustc-shim.sh`), so `.wasm` bytes
        // (hence `guests.blake3`) are now byte-identical across checkout paths (C2 lead-in). That
        // wrapper is wired via config, so it applies here AND to the test-harness `ensure_built()`
        // copies (which apply the SAME remap) without per-call-site coordination.
        .env("RUSTFLAGS", guest_remap_rustflags(&root))
        .args(["build", "--release", "--target", "wasm32-unknown-unknown"])
        .status()
        .map_err(|e| anyhow::anyhow!("failed to run cargo for the guests workspace: {e}"))?;
    anyhow::ensure!(status.success(), "building guests failed with {status}");

    // Stale-guest guard (swarm-p1-ledger Merge-1 follow-on): write the committed blake3 manifest of
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

/// Run the swarm CI tier-1 suite (TDD §8.1 tier 1: the per-PR, hosted-CI, no-GPU gate).
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
/// The `daemon-conformance` detached-delegation trio (a known parallel-load flake, pass-in-isolation
/// = green) is NOT a swarm crate and NOT in this list, so it never gates the swarm tier.
fn swarm_ci_det() -> anyhow::Result<()> {
    let root = workspace_root();
    // Dependency-direction invariant (architecture §7) first — cheap (metadata only) and fails fast
    // on a host/*->sdk/* regression before spending a compile.
    println!("\n== swarm-ci-det: daemon-vhc dependency-direction check ==");
    vhc_dep_check()?;
    build_guests()?;
    // (label, cargo test args). Each runs in its own process; the first red aborts.
    let suites: &[(&str, &[&str])] = &[
        (
            "daemon-vhc-abi (journal §8.3 CDDL grammar validity + per-tag samples)",
            &["-p", "daemon-vhc-abi"],
        ),
        (
            "daemon-vhc-det (shared det kernels: sim ≡ host)",
            &["-p", "daemon-vhc-det"],
        ),
        (
            "daemon-vhc-proto (wire mechanism: envelopes v1+v2, grants, canonical CBOR)",
            &["-p", "daemon-vhc-proto"],
        ),
        (
            // D0: assignment math moved out of the proto (refactor §8/D0). The golden vectors
            // (LCG stream, shuffle, quorum ladder, class weights) moved with it — this lane keeps
            // them tier-1 so any drift in the moved math stays a visible, deliberate break.
            "daemon-vhc-sdk-consensus (assignment math + golden vectors, moved at D0)",
            &["-p", "daemon-vhc-sdk-consensus"],
        ),
        (
            "daemon-vhc-session (harness + assess + replay, loopback)",
            &["-p", "daemon-vhc-session"],
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
            // The A0 frozen v1 compatibility fixture (refactor §5 A0; decisions D3 cell 1): the
            // pinned pre-refactor tiny-llama bundle replays bit-exact under the v1 driver. Named
            // explicitly (also covered by the crate suite below) so the standing regression is a
            // visible tier-1 lane of its own.
            "A0 frozen v1 fixture replay (pre-refactor tiny-llama digest parity)",
            &["-p", "daemon-vhc-host", "--test", "a0_frozen_fixture"],
        ),
        (
            "daemon-vhc-host driver selection (ABI §1.3 typed refusals)",
            &["-p", "daemon-vhc-host", "--test", "driver_selection"],
        ),
        (
            // The A2 event-loop acceptance (refactor §5 A2): the non-round toy-averager guest
            // (timers + publish only) end-to-end under the real major-2 driver — selection
            // admits, da_init/da_run dispatch, §12.1 signed frames with durable seqs, journaled
            // through the real A1 substrate — plus the undeclared-channel GrantViolation
            // negative. Named as its own lane (like the A0 fixture) so the standing
            // expressiveness proof is visible; also covered by the host crate suite above.
            "A2 v2 event loop (toy-averager expressiveness + typed channel trap)",
            &["-p", "daemon-vhc-host", "--test", "v2_event_loop"],
        ),
        (
            // The A2 claim + admission-funnel acceptance (refactor §5 A2; §10 gate row "Claim
            // rejection / over-claim / under-claim traps"): over-claim vs owner policy (stage 5),
            // claim outside lane bounds (stage 4), ClaimInconsistent, GrantsExceedLane, the
            // attributable under-claim cap trap at run time, and claim determinism — all through
            // the real restricted assessment instance (test-claim-v2 guest).
            "A2 claim + admission funnel (over/under-claim, lane bounds, typed refusals)",
            &["-p", "daemon-vhc-host", "--test", "v2_claim_funnel"],
        ),
        (
            // The §2.5 tabi@1 bridge under major-2 (the choreography sitting): the SAME frozen
            // dispatch as the v1 driver, genericized over the store — registration only in
            // da_init (PhaseViolation otherwise), slice-class arenas cleared per Delivered
            // (StaleHandle across a boundary), nr-class readouts journaled under §2.7 kinds.
            // The A0 fixture lane above is the byte-for-byte v1-untouched proof.
            "A2 tabi@1 bridge under the v2 driver (§2.5 legality + slice arenas + nr journal)",
            &["-p", "daemon-vhc-host", "--test", "v2_bridge"],
        ),
        (
            // THE Phase-A acceptance (refactor §5 A2): TinyLlama on BarrierRound under the v2
            // driver + bridge reproduces the v1 WasmBackend's det-lane state digests — cpu +
            // burn-ndarray tiers here; wgpu/cuda tiers are hardware-gated in the same test file
            // (the scheduled GPU lanes, like reference_parity_{wgpu,cuda}).
            "A2 det-digest parity: TinyLlama-on-BarrierRound v2 ≡ v1 (cpu + burn-ndarray)",
            &[
                "-p",
                "daemon-vhc-host",
                "--test",
                "v2_parity",
                "--features",
                "burn-ndarray",
            ],
        ),
        (
            // The v2 input-replay step (refactor §5 A1→A2 acceptance; §12.6 journal soak for
            // v2): recorded runs (toy averager: timers/clock; bridge guest: nr readouts +
            // staged kinds 1/2) re-driven from the journal alone through the §8.7 verifier
            // (observe contract over the host replay engine) — every decision bit-for-bit;
            // tampered/incomplete journals are typed divergences. The TinyLlama acceptance run
            // replays inside the parity lane above.
            "A2 v2 input-replay: journal-only re-drive ≡ recorded decisions (§8.7)",
            &["-p", "daemon-vhc-host", "--test", "v2_replay"],
        ),
        (
            // The sys@2 crypto-acceleration conformance gate (Phase B; architecture §3.2/§3.7,
            // refactor §6): the host `hash`/`verify_sig` accel bodies ≡ the dual-compiled
            // `daemon_vhc_proto::crypto` contract (the in-guest fallback is that same contract
            // compiled to wasm — bit-exact by construction, the det-lane pattern) over a wide
            // deterministic sweep + known-answer vectors + tri-state verify semantics. Named as
            // its own lane (also covered by the host crate suite below).
            "B2 sys@2 crypto accel conformance (host ≡ in-guest contract: hash/verify_sig)",
            &["-p", "daemon-vhc-host", "--test", "v2_crypto"],
        ),
        (
            // The Phase-C det-reclassification conformance gate (architecture §3.2/§3.6, refactor
            // §7; §10 gate row "Det host-op ≡ in-guest-crate"): the host `det_*` accel bodies the
            // worker runs (OpBackend, via the reference CpuBackend) ≡ the normative dual-compiled
            // `daemon_vhc_det` crate the in-guest fallback also compiles — bit-identical (equality
            // class) for EVERY det accel op over a wide deterministic sweep, plus the DET_ACCEL_OPS
            // coverage guard. The det twin of the crypto lane above; also covered by the host crate
            // suite below.
            "C2 det reclassification conformance (host det_* ≡ in-guest daemon-vhc-det)",
            &["-p", "daemon-vhc-host", "--test", "v2_det_conformance"],
        ),
        (
            // The Phase-C custom-op registry gate (architecture §3.2, refactor §7): versioned
            // named fused kernels register host-side (flash_attn@1 the first entry); a manifest
            // requiring an op the host does not advertise is refused CLEANLY (typed
            // CustomOpUnsupported, never a trap). Pins the shared ABI vocabulary (the seam C1's
            // compute@2 OperationIr::Custom resolves through) + the registry admission behaviour.
            "C2 custom-op registry (flash_attn@1; typed refusal on absent required op)",
            &["-p", "daemon-vhc-host", "--test", "v2_custom_op"],
        ),
        (
            // The Phase-C MODEL-AGNOSTIC acceptance (refactor §7: "a non-LLaMA toy authored with
            // zero host changes … proving the compute ABI is model-agnostic"): the `toy-mlp` guest
            // — a two-layer MLP trained by SGD, authored purely over daemon-vhc-sdk-compute +
            // daemon-vhc-sdk-v2 — runs against the SAME compute@2 runner/driver/journal as the
            // LLaMA reference, exports a trained weight bit-exact vs a native Autodiff<NdArray> run
            // of the identical loop, and replays bit-for-bit (§8.7). No host code is model-specific.
            "C3 model-agnostic compute@2 (toy-mlp: distinct model, zero host changes, bit-exact + replay)",
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
            "C3 compute replay (ndarray↔ndarray degenerate: same op-journal, bit-exact re-execution)",
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
            &["-p", "daemon-vhc-host", "--test", "v2_data_fetch"],
        ),
        (
            "daemon-vhc-host (det lane + cross-backend digests + wasm-guest determinism)",
            &["-p", "daemon-vhc-host", "--features", "burn-ndarray"],
        ),
        (
            // The frozen worker protocol over the REAL `daemon-vhc-worker` binary (probe → assess
            // → join → one self-driven round; envelope seam; preemption churn). Lived inside the
            // `daemon-vhc-host` suite above until the A2 worker-bin split moved the bin (and its
            // CARGO_BIN_EXE-spawning test) to `crates/vhc/bins/daemon-vhc-worker`; same features
            // as before the split (burn-ndarray forwards into the host lib), so coverage is
            // unchanged.
            "daemon-vhc-worker (frozen worker protocol over the real binary)",
            &["-p", "daemon-vhc-worker", "--features", "burn-ndarray"],
        ),
        (
            "daemon-vhc-sdk (SDK profile goldens: sparse_loco/diloco)",
            &["-p", "daemon-vhc-sdk", "--features", "sim"],
        ),
        (
            // The C3a models-exodus profiles gate (refactor §7 "profiles re-express over Burn
            // tensors + det math in sdk/daemon-vhc-sdk-profiles"): the re-expressed
            // SparseLoco/DiLoCo/Demo reproduce the CURRENT SDK profile implementation bit-for-bit
            // (live A/B over the sim oracle + the pinned sparse_loco_golden literals), and the
            // Section payload wire is byte-identical to the v1 container encoding.
            "C3a sdk-profiles ≡ current SDK profiles (bit-exact A/B + pinned goldens + wire)",
            &["-p", "daemon-vhc-sdk-profiles"],
        ),
        (
            // The C3 models-exodus acceptance (refactor §7 "models leave the SDK" + "re-authored
            // tiny-llama matches reference parity within the existing tolerance class"): the
            // re-authored `tiny-llama-c3` guest — a real Burn model over Autodiff<HostBackend> +
            // the C3a profiles' in-guest det lane — through a 2-round barrier whole-run vs the
            // frozen v1 digest oracle. C3b: guest training ≡ native Autodiff<NdArray> of the SAME
            // dual-compiled model source, bit-exact. C3c: det-lane digests ≡ the v1 oracle
            // bit-exact (equality class); trained θ within the OpClass::Optimizer band
            // (tolerance class). The frozen pins (v2_parity) and the A0 fixture run beside this
            // lane in the same gate.
            "C3 re-authored tiny-llama parity (bit-exact lowering + det digests ≡ v1 + Optimizer band)",
            &[
                "-p",
                "daemon-vhc-host",
                "--test",
                "c3_parity",
                "--features",
                "burn-ndarray",
            ],
        ),
        (
            // A2 migrate/main! scaffolding (refactor §5 A2 item 4; ABI §10): state round-trips
            // in sim through the typed manifest protocol; the SDK-derived claim/manifest match
            // the §9.1/§6.2 wire schema the admission funnel decodes. The macro's exports are
            // exercised for real by the tiny-llama-v2 guest under the parity lane.
            "daemon-vhc-sdk-v2 (main!/migrate scaffolding: sim round-trips + derivations)",
            &["-p", "daemon-vhc-sdk-v2"],
        ),
        (
            "daemon-swarm-e2e (drills + observe-replay, no iroh/live)",
            &["-p", "daemon-swarm-e2e"],
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
    for (label, args) in suites {
        println!("\n== swarm-ci-det: {label} ==");
        let status = Command::new("cargo")
            .current_dir(&root)
            .arg("test")
            .args(*args)
            .status()
            .map_err(|e| anyhow::anyhow!("running cargo test {args:?}: {e}"))?;
        anyhow::ensure!(status.success(), "swarm CI tier-1 suite failed: {label}");
    }
    println!("\nswarm-ci-det: all tier-1 (CPU consensus-critical) swarm suites green");
    Ok(())
}

/// Run the swarm **CI tier-2** whole-run suites (decisions D4; refactor §6, §10 gate table).
///
/// The two-layer simulation split (architecture §6): SDK-side `daemon-vhc-sim` runs NATIVE policy
/// code (the SPARTA continuous-averaging toy over the virtual worlds — deterministic whole run),
/// and host-side `daemon-vhc-testkit` runs the PRODUCTION wasm blobs under wasmtime + simulated
/// capability providers, journaled and §8.7 replay-verified. This is heavier than tier-1 (it builds
/// the wasm guests + compiles wasmtime), so it is a separate gate — never folded into
/// `swarm-ci-det`, which stays the CPU-only deterministic tier-1 bar.
fn swarm_ci_t2() -> anyhow::Result<()> {
    let root = workspace_root();
    // Same dependency-direction preflight as tier-1, then the guests the testkit runs.
    println!("\n== swarm-ci-t2: daemon-vhc dependency-direction check ==");
    vhc_dep_check()?;
    build_guests()?;
    let suites: &[(&str, &[&str])] = &[
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
            // tiny_llama_v2 barrier whole runs under the in-process native coordinator (single-
            // and 2-worker with cross-worker det-digest agreement; SDK-free raw-CBOR config), and
            // the adversarial-rig pinned cases (duplicate record deduped; delayed payloads →
            // straggle → catch-up).
            "daemon-vhc-testkit (production-blob whole runs + barrier rounds + adversarial rig)",
            &["-p", "daemon-vhc-testkit"],
        ),
    ];
    for (label, args) in suites {
        println!("\n== swarm-ci-t2: {label} ==");
        let status = Command::new("cargo")
            .current_dir(&root)
            .arg("test")
            .args(*args)
            .status()
            .map_err(|e| anyhow::anyhow!("running cargo test {args:?}: {e}"))?;
        anyhow::ensure!(
            status.success(),
            "swarm CI tier-2 whole-run suite failed: {label}"
        );
    }
    println!("\nswarm-ci-t2: all tier-2 (sim/testkit) whole-run suites green");
    Ok(())
}

/// Enforce the daemon-vhc dependency-direction rules (architecture §7): the wasm boundary is
/// visible as `sdk/` vs `host/`, and `contracts/` is the only shared ground.
///
/// - `host/*` never links `sdk/*` (the host runs production wasm blobs; native policy testing is
///   `vhc-sim`'s job SDK-side, integration testing is the testkit's job host-side).
/// - `contracts/*` links neither `sdk/*` nor `host/*`.
/// - `sdk/*` never links `host/*`.
///
/// Enforced over `cargo metadata` (normal + dev + build edges). The real `sdk/*` consumers are the
/// `guests/` modules, which are a separate cargo workspace outside this gate — so *every* edge into
/// `sdk/*` from this workspace is a transitional wart, listed as an honest exception and tracked to
/// the phase that removes it. A new, un-listed `*/ -> sdk/*` edge fails the gate.
fn vhc_dep_check() -> anyhow::Result<()> {
    use std::collections::{BTreeMap, BTreeSet};

    // The honest current exceptions (Phase 0): each is a transitional edge into `sdk/*` that a
    // later phase removes. Format: (dependent crate, sdk crate, why it exists / when it goes).
    const EXCEPTIONS: &[(&str, &str, &str)] = &[
        (
            "daemon-swarm-e2e",
            "daemon-vhc-sdk",
            "Phase B — e2e runs production wasm blobs under host/daemon-vhc-testkit [dev-dep]",
        ),
        (
            "daemon-swarm-e2e",
            "daemon-vhc-sdk-rounds",
            "Phase E — the A2 choreography bridging oracle (relocated round logic vs the v1 \
             engine) retires with the v1 engine at sunset [dev-dep]",
        ),
        (
            "daemon-swarm-e2e",
            "daemon-vhc-sdk-v2",
            "Phase E — the B2 corpus-windowing equivalence oracle (SDK policy vs the v1 host \
             pipeline `session::data`) retires with the v1 pipeline at sunset [dev-dep]",
        ),
        (
            "daemon-vhc-safetensors",
            "daemon-vhc-sdk",
            "Phase E — safetensors is wired into the checkpoint path (state-dict layout) [dev-dep]",
        ),
        (
            "daemon-vhc-host",
            "daemon-vhc-sdk",
            "Phase C — model presets (TinyLlamaCfg/profiles) leave the SDK for guests/ [dev-dep]",
        ),
        (
            "daemon-vhc-host",
            "daemon-vhc-sdk-profiles",
            "Phase E — the C3 parity harness runs the re-expressed profile natively as the \
             lowering oracle beside the dual-compiled guest model; retires with the v1 oracle at \
             sunset [dev-dep]",
        ),
        (
            "daemon-vhc-worker",
            "daemon-vhc-sdk",
            "Phase C — TinyLlamaCfg in the moved worker-protocol test leaves the SDK for guests/ \
             [dev-dep] (split from daemon-vhc-host's identical exception at the A2 bin split)",
        ),
        // --- D0: proto::assignment -> sdk/daemon-vhc-sdk-consensus (refactor §8/D0). The proto
        // is algorithm-free from D0 (enforced below); its old host-side assignment consumers
        // relink to the consensus SDK layer as explicit transitional edges, each retiring at D2.
        (
            "daemon-vhc-coordinator",
            "daemon-vhc-sdk-consensus",
            "D2 — the native coordinator dissolves at D2 into sdk-consensus + \
             guests/coordinator-quorum; native coordination for tests moves to SDK-side vhc-sim \
             (which links sdk-consensus legitimately) [normal]",
        ),
        (
            "daemon-vhc-session",
            "daemon-vhc-sdk-consensus",
            "D2 — the retained v1 RoundEngine's assignment consumption; reviewed/retired as D2 \
             re-seats consumers (the engine itself retires with the v1 driver at the Phase-E \
             sunset) [normal]",
        ),
        (
            "daemon-vhc-testkit",
            "daemon-vhc-sdk-consensus",
            "D2 — the barrier harness re-derives worker windows natively; reviewed/retired as D2 \
             re-seats consumers on the wasm coordinator [normal]",
        ),
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

    let is_exception =
        |from: &str, to: &str| EXCEPTIONS.iter().any(|(f, t, _)| *f == from && *t == to);

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

            // Edges into sdk/*. The SDK-side native sim (`daemon-vhc-sim`) is the DESIGNED entry
            // for native harnesses (architecture §6: "policy code compiled natively runs against
            // it"): any crate that is NOT host/* or contracts/* may link it without an exception.
            // The wasm-boundary wall still holds — host/* and contracts/* linking sdk/* (including
            // vhc-sim) remains a violation (enforced by the same branch + the hard rules below), so
            // host/daemon-vhc-testkit can never reach across into the SDK sim. Every OTHER sdk/*
            // edge from a non-sdk crate must be a tracked exception.
            if to_role == "sdk" && from_role != Some("sdk") {
                let sim_native_harness = to == "daemon-vhc-sim"
                    && from_role != Some("host")
                    && from_role != Some("contracts");
                if sim_native_harness {
                    // allowed: a native harness (e.g. bins/swarm-local, tests/*) linking the
                    // SDK-side sim — its whole purpose (refactor §6/§11).
                } else if is_exception(&from, &to) {
                    seen.insert((from.clone(), to.clone()));
                } else {
                    violations.push(format!(
                        "{from} -> {to} [{kind}]: only the guests workspace and native harnesses \
                         (via daemon-vhc-sim) may link sdk/* (no tracked exception)"
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

    // --- D0 tightening: daemon-vhc-proto is ALGORITHM-FREE from D0 on (refactor §8/D0;
    // architecture §7 rule 1 — "no assignment math, no round vocabulary"). The assignment module
    // moved to sdk/daemon-vhc-sdk-consensus; a re-grown `assignment` module (or file) in the
    // proto fails this gate from now on.
    {
        let proto_src = root.join("crates/vhc/contracts/daemon-vhc-proto/src");
        if proto_src.join("assignment.rs").exists() {
            violations.push(
                "daemon-vhc-proto: src/assignment.rs exists — the proto is algorithm-free from \
                 D0; assignment math lives in sdk/daemon-vhc-sdk-consensus"
                    .to_string(),
            );
        }
        let lib = std::fs::read_to_string(proto_src.join("lib.rs")).unwrap_or_default();
        if lib.contains("mod assignment") {
            violations.push(
                "daemon-vhc-proto: lib.rs declares an `assignment` module — the proto is \
                 algorithm-free from D0 (refactor §8/D0)"
                    .to_string(),
            );
        }
    }

    println!("daemon-vhc dependency-direction check (architecture §7)");
    println!(
        "  rule: host/* never links sdk/* · contracts/* links neither · sdk/* never links host/*"
    );
    println!("  rule (D0): daemon-vhc-proto is algorithm-free (assignment lives in sdk-consensus)");
    println!("\ntracked exceptions (honest; each removed by the noted phase):");
    for (f, t, note) in EXCEPTIONS {
        let mark = if seen.contains(&((*f).to_string(), (*t).to_string())) {
            "present"
        } else {
            "STALE — listed but not in the graph; drop it from EXCEPTIONS"
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
    Ok(())
}

/// RUSTFLAGS that remap the absolute source prefixes rustc embeds in panic locations: the
/// `<checkout>` root (workspace + path deps like `daemon-vhc-sdk`) and the cargo registry
/// (`$CARGO_HOME`, else `$HOME/.cargo`). With the guests' committed `Cargo.lock` this makes the
/// `.wasm` bytes byte-reproducible across clean rebuilds within one checkout path. The
/// cross-worktree `-C metadata` reordering that this remap does NOT rewrite is handled separately
/// by the guests workspace's `rustc-wrapper` (`guest-rustc-shim.sh`, wired in
/// `crates/vhc/guests/.cargo/config.toml`), so the guests are byte-identical across checkout paths.
/// Kept in lockstep with the `ensure_built()` copies in the wasm-backed test harnesses.
fn guest_remap_rustflags(checkout: &Path) -> String {
    let cargo_home = std::env::var_os("CARGO_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".cargo"));
    format!(
        "--remap-path-prefix={}=/daemon-node --remap-path-prefix={}=/cargo",
        checkout.display(),
        cargo_home.display(),
    )
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
    // W6: the pure-local session recap op (request + a populated response), so verify-codec proves
    // the generated C decoder takes the new shapes end-to-end.
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
    // wire v37 (W2-E): the richer ChatMessage on the conversation-history surface, so conformance
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
    // Saved presences (W2-F; wire v37): the list/save/delete/set-active ops + the listing reply.
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

    // -- file transfer (W2-H; wire v37) --------------------------------------------------------
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
