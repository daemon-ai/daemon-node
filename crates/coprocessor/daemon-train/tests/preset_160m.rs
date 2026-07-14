// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// M1: the 160M LLaMA preset driven through the real wasm host on `CpuBackend` — build + 1 step +
// make_update (the smoke that the preset config is expressible and trains a finite step), plus a
// `meta`-mode footprint reconciliation against spec §5.1's VRAM/RAM planning table.
//
// Two variants:
//   * a FAST, always-on reduced-layer variant (1 layer, small vocab, short seq) that exercises the
//     identical preset code path in seconds;
//   * an #[ignore]d FULL 160M variant (12 layers, vocab 50257, seq 1024) — a real execute-mode pass
//     over ~152M params costs GBs + minutes on the CPU backend (spec §5.1 / Risk 3), so it is
//     opt-in (`cargo test -p daemon-train --test preset_160m -- --ignored`).
//
// Dev/test harness (shells `cargo build` for the guests + reads the `.wasm`, like `xtask
// build-guests`); the fs/process hardening bans target the shipped node, not tests.
#![allow(clippy::disallowed_methods)]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Once;
use std::time::Duration;

use daemon_train::{EngineConfig, Worker};
use daemon_train_sdk::models::TinyLlamaCfg;

/// Sandbox budgets scaled for the 160M-shaped preset. The default `EngineConfig` is tuned for the
/// tiny (`d_model 64`) reference model; a 768-wide model's real fp32 matmuls take longer wall-clock
/// (tripping the 5 s epoch watchdog) though it issues the same *number* of host ops. Budgets are
/// self-protection, not domain limits (ABI §8) — a runnable host sizes them to the model.
fn roomy_engine() -> EngineConfig {
    EngineConfig {
        fuel_per_call: 1 << 34,
        epoch_deadline: Duration::from_secs(600),
        op_budget: 1 << 30,
        max_step_handles: 1 << 24,
        ..EngineConfig::default()
    }
}

fn guests_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../guests")
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

fn cbor(cfg: &TinyLlamaCfg) -> Vec<u8> {
    let mut b = Vec::new();
    ciborium::into_writer(cfg, &mut b).expect("cbor");
    b
}

/// A reduced preset config for a fast always-on smoke: keeps the preset's `sparse_loco` profile +
/// `chunk 256` and a `d_model` that is a multiple of 256 (so top-k needs no padding), but shrinks
/// everything else (1 layer, `d_model 256`, small vocab, short seq) so the real fp32 execute pass
/// runs in a couple of seconds. The full-fidelity 160M shape is checked analytically (models.rs) and
/// in the `#[ignore]`d full test below.
fn fast_cfg() -> TinyLlamaCfg {
    TinyLlamaCfg {
        n_layers: 1,
        d_model: 256,
        n_heads: 4,
        n_kv_heads: 4,
        head_dim: 64,
        vocab: 512,
        seq_len: 32,
        ..TinyLlamaCfg::llama_160m()
    }
}

/// Run build + 1 step + make_update for `cfg` and return the reported loss + the sealed update size.
fn build_step_update(cfg: &TinyLlamaCfg, batch: u32, seq: u32) -> (f32, usize) {
    let worker = Worker::new(roomy_engine()).unwrap();
    let module = worker.load_module(&tiny_llama_wasm()).unwrap();
    let mut inst = worker.instantiate(&module).unwrap();
    inst.build(&cbor(cfg)).unwrap();

    // The registration matches the canonical param layout (name + order + count).
    let params = inst.params();
    let layout = cfg.canonical_param_layout();
    assert_eq!(
        params.len(),
        layout.len(),
        "param count == canonical layout"
    );
    for (p, (name, shape)) in params.iter().zip(layout.iter()) {
        assert_eq!(&p.name, name, "canonical registration order");
        assert_eq!(&p.shape, shape, "param {name} shape");
    }

    // One micro-batch: forward + backward → a finite loss (the NaN tripwire).
    let tokens: Vec<u32> = (0..batch * seq).map(|i| i % cfg.vocab).collect();
    let h = inst.register_batch(tokens, batch, seq);
    inst.step(h, 0, 0, 1, batch).unwrap();
    let loss = inst
        .metrics()
        .into_iter()
        .rev()
        .find(|(n, _)| n == "loss")
        .map(|(_, v)| v)
        .expect("step reports a loss");

    // make_update seals a non-empty container (sparse_loco top-k over every param).
    let container = inst.make_update(0).unwrap();
    let bytes = inst.update_bytes(container).unwrap();
    (loss, bytes.len())
}

/// FAST always-on: the preset shape builds, trains a finite step, and seals an update — on the real
/// wasm host + CpuBackend, through the identical `TinyLlama` code the 160M module runs.
#[test]
fn preset_160m_reduced_smoke() {
    let cfg = fast_cfg();
    let (loss, upd) = build_step_update(&cfg, 2, cfg.seq_len);
    assert!(
        loss.is_finite(),
        "reduced-160M step loss must be finite, got {loss}"
    );
    assert!(upd > 0, "make_update sealed a non-empty payload");
}

/// FAST always-on: a `meta` pass over the reduced preset yields a schema-valid report whose byte
/// footprints are exactly `4 · param_count` (fp32 masters + fp32 grads), the arithmetic the full
/// reconciliation below scales up. Exercises the real Wave-2 op vocabulary.
#[test]
fn preset_160m_reduced_meta_report() {
    let cfg = fast_cfg();
    let worker = Worker::new(roomy_engine()).unwrap();
    let module = worker.load_module(&tiny_llama_wasm()).unwrap();
    let mut inst = worker.instantiate(&module).unwrap();
    let report = inst.meta(&cbor(&cfg), 2, cfg.seq_len).unwrap();

    let n = cfg.param_count();
    assert_eq!(report.master_bytes, n * 4, "fp32 masters = 4·params");
    assert_eq!(report.grad_bytes, n * 4, "fp32 grads = 4·params");
    for op in [
        "embedding@1",
        "rmsnorm@1",
        "rope@1",
        "flash_attn@1",
        "silu@1",
    ] {
        assert!(report.ops_used.iter().any(|o| o == op), "meta missed {op}");
    }
}

/// FULL 160M (expensive; opt-in via `--ignored`): build + 1 step (b=1, seq 1024) + make_update over
/// the real ~152M-param preset, and reconcile the `meta` footprint with spec §5.1's planning table.
///
/// The spec row assumes **bf16 weights** (0.3 GB at 160M); the P1 preset stores **fp32** masters for
/// det-lane exactness, so `master_bytes ≈ 0.6 GB` (2×) — a documented spec-amendment candidate
/// (`swarm-ledger-m1.md`). The invariant that must hold either way: the fp32 steady state fits an
/// 8 GB card (spec's "fits on" conclusion).
#[test]
#[ignore = "expensive: a real 160M execute pass costs GBs + minutes on the CPU backend (Risk 3)"]
fn preset_160m_full_smoke_and_reconcile() {
    let cfg = TinyLlamaCfg::llama_160m();
    assert_eq!(
        cfg.param_count(),
        151_862_784,
        "exact 160M-preset param count"
    );

    let (loss, upd) = build_step_update(&cfg, 1, cfg.seq_len);
    assert!(
        loss.is_finite(),
        "full-160M step loss must be finite, got {loss}"
    );
    assert!(upd > 0);

    // Meta reconciliation against spec §5.1.
    let worker = Worker::new(roomy_engine()).unwrap();
    let module = worker.load_module(&tiny_llama_wasm()).unwrap();
    let mut inst = worker.instantiate(&module).unwrap();
    let report = inst.meta(&cbor(&cfg), 2, cfg.seq_len).unwrap();

    let n = cfg.param_count();
    let gib = 1u64 << 30;
    assert_eq!(report.master_bytes, n * 4); // ~0.57 GiB fp32 (spec bf16 row: 0.3 GB → 2× delta)
    assert_eq!(report.grad_bytes, n * 4); // ~0.57 GiB (spec: 0.6 GB ✓)
                                          // params + master + grad + fp32 Adam(m+v) — the steady-state VRAM excl. activations.
    let steady = report.param_bytes + report.master_bytes + report.grad_bytes + n * 4 * 2;
    assert!(
        steady < 8 * gib,
        "160M fp32 steady state fits an 8 GB card (spec conclusion)"
    );
    // Host RAM (masters + round base materialized at ingest) ~ spec §5.1's ~2 GB host row.
    assert!(
        report.host_ram_bytes_est >= gib,
        "host RAM estimate is params-scale"
    );
}
