// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// G2 — lifecycle-level wgpu tests: HOST-8 (`meta_mode_vram_ram_estimates` against real device
// numbers) and HOST-10 (`ingest_budget_scales_with_count` re-run on the wgpu backend).
//
// GPU-skip convention (TDD §8.1 tier-2): each test checks `wgpu_adapter_available()` and skips
// with a loud stderr note when absent. The `.#vulkan` devShell is the runnable lane.
//
// Guest loading mirrors guest_lifecycle.rs (always-rebuild stale-guest guard); this is a dev/test
// harness, so the fs/process bans (which target the shipped node) are allowed file-wide.
#![cfg(feature = "wgpu")]
#![allow(clippy::disallowed_methods)]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Once;

use daemon_train::autotune::{probe_wgpu, Autotune, DeviceLimits, DEFAULT_MAX_MICROBATCH};
use daemon_train::{wgpu_adapter_available, BackendKind, EngineConfig, TrapCode, Worker};
use daemon_train_sdk::models::TinyLlamaCfg;
use serde::Serialize;

macro_rules! require_gpu {
    () => {
        if !wgpu_adapter_available() {
            eprintln!(
                "SKIP {}: no usable wgpu adapter on this runner (run in the .#vulkan devShell — \
                 TDD §8.1 tier-2)",
                module_path!()
            );
            return;
        }
    };
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
    ensure_built();
    let path = guest_dir().join("tiny_llama.wasm");
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

fn cbor<T: Serialize>(v: &T) -> Vec<u8> {
    let mut b = Vec::new();
    ciborium::into_writer(v, &mut b).expect("cbor");
    b
}

fn tiny_cfg() -> TinyLlamaCfg {
    TinyLlamaCfg {
        n_layers: 1,
        seq_len: 9,
        ..TinyLlamaCfg::default()
    }
}

/// HOST-8 `meta_mode_vram_ram_estimates`: the meta-pass byte footprints feed the G2 autotune, whose
/// verdict against THIS machine's real device numbers (wgpu adapter probe + /proc/meminfo host RAM)
/// is eligible with internally consistent estimates. The estimates themselves are backend-
/// independent (shapes/dtypes), so a CPU meta pass compared against real GPU limits is the honest
/// §6.5 admission shape.
#[test]
fn meta_mode_vram_ram_estimates() {
    require_gpu!();
    let worker = Worker::new(EngineConfig::default()).unwrap();
    let module = worker.load_module(&tiny_llama_wasm()).unwrap();
    let mut inst = worker.instantiate(&module).unwrap();
    let report = inst
        .meta(&cbor(&tiny_cfg()), 1, tiny_cfg().seq_len)
        .unwrap();

    // Byte-footprint consistency: fp32 masters == grads; params in storage dtype ≤ masters (all
    // tiny-llama params are F32 here, so equal); every estimate is non-zero for a real model.
    assert_eq!(report.master_bytes, report.grad_bytes);
    assert!(report.param_bytes > 0 && report.master_bytes > 0);
    assert!(report.act_bytes_est > 0 && report.host_ram_bytes_est > 0);
    assert!(report.payload_bytes_est > 0);

    // The autotune model derived from the report: fixed VRAM = params (storage dtype) + fp32
    // masters + fp32 grads + the registered persistents (tiny-llama registers Adam moments etc.).
    let autotune = Autotune::from_meta(&report);
    let core = report.param_bytes + report.master_bytes + report.grad_bytes;
    assert!(
        autotune.fixed_vram_bytes >= core,
        "fixed VRAM covers at least params+masters+grads ({} >= {core})",
        autotune.fixed_vram_bytes
    );
    assert!(
        !report.persistent.is_empty(),
        "tiny-llama registers optimizer persistents (they enter the fixed VRAM term)"
    );

    // Real device numbers: the wgpu adapter probe (max single allocation — the one wgpu-queryable
    // budget number) + real host RAM.
    let probe = probe_wgpu().expect("adapter probed (require_gpu passed)");
    assert!(probe.gpus >= 1);
    assert!(probe.max_alloc_mb > 0, "max_buffer_size is queryable");
    eprintln!(
        "wgpu probe: adapter={} backend={} max_alloc={} MiB",
        probe.adapter, probe.backend, probe.max_alloc_mb
    );

    let limits = DeviceLimits {
        vram_mb: probe.max_alloc_mb,
        ram_mb: 8192, // a conservative host-RAM floor; the worker probes /proc/meminfo
        max_alloc_mb: probe.max_alloc_mb,
        ..Default::default()
    };
    let v = autotune.verdict(&limits, DEFAULT_MAX_MICROBATCH);
    eprintln!(
        "meta_mode_vram_ram_estimates: verdict={v:?} (fixed={} MiB, act/mb={} B, host_ram={} MiB)",
        autotune.fixed_vram_bytes >> 20,
        autotune.act_bytes_per_mb,
        autotune.host_ram_bytes >> 20
    );
    assert!(
        v.eligible,
        "the 1-layer tiny-llama must fit this machine's real device budget: {:?}",
        v.reasons
    );
    assert!(v.micro_batch >= 1);
    assert!(v.vram_mb_estimate >= 1 && v.ram_mb_estimate >= 1);
    assert_eq!(v.payload_bytes_estimate, report.payload_bytes_est);
}

/// HOST-10 `ingest_budget_scales_with_count` (re-run on wgpu): the per-peer ingest cost measured by
/// the meta two-point fit scales the execute op budget (ABI §8) — a budget sized for the staged
/// count passes on the wgpu backend, and the SAME staged count under a budget sized for 1 peer
/// traps `BudgetOps` (typed, worker intact).
#[test]
fn ingest_budget_scales_with_count() {
    require_gpu!();
    let config = cbor(&tiny_cfg());
    let wasm = tiny_llama_wasm();

    // Meta two-point fit (CPU meta pass; op counts are backend-independent — the guest's import
    // stream is bit-deterministic, ABI §7.1).
    let meta_worker = Worker::new(EngineConfig::default()).unwrap();
    let module = meta_worker.load_module(&wasm).unwrap();
    let mut meta_inst = meta_worker.instantiate(&module).unwrap();
    let report = meta_inst.meta(&config, 1, tiny_cfg().seq_len).unwrap();
    let base = report.op_calls["da_ingest_updates"]; // cost at count = 1
    let slope = report.ingest_op_calls_per_peer;
    let op_build = report.op_calls["da_build"];
    let op_make = report.op_calls["da_make_update"];
    assert!(slope > 0, "ingest cost must grow with the staged count");

    // Pick a count whose ingest cost strictly exceeds every other entry point's cost, so the tight
    // budget below can pass build/make_update yet trap ONLY on the under-scaled ingest.
    let n = usize::try_from((op_build.max(op_make) / slope) + 4).unwrap();
    let cost = |count: u64| base + (count - 1) * slope;

    let run = |op_budget: u64| -> Result<(), daemon_train::TrainError> {
        let worker = Worker::new(EngineConfig {
            backend: BackendKind::Wgpu,
            op_budget,
            ..EngineConfig::default()
        })
        .unwrap();
        let module = worker.load_module(&wasm).unwrap();
        let mut inst = worker.instantiate(&module).unwrap();
        inst.build(&config)?;
        let container = inst.make_update(0)?;
        let payload = inst.update_bytes(container)?;
        let staged: Vec<Vec<u8>> = vec![payload; n];
        inst.ingest_payloads(0, &staged)
    };

    // Budget scaled for count = n (the ABI §8 shape: base + count × per-peer) → passes on wgpu.
    let scaled = cost(n as u64) + slope; // headroom of one peer
    run(scaled).expect("a count-scaled ingest budget must pass");

    // Budget scaled for count = 1 (still enough for build/make_update by construction of n) → the
    // count-n ingest traps typed BudgetOps.
    let tight = cost(1).max(op_build).max(op_make) + 1;
    assert!(
        tight < cost(n as u64),
        "n was chosen so the tight budget under-scales ingest"
    );
    let err = run(tight).expect_err("an unscaled budget must trap on the count-n ingest");
    assert_eq!(
        err.trap_code(),
        Some(TrapCode::BudgetOps),
        "typed BudgetOps, worker intact: {err}"
    );
    eprintln!(
        "ingest_budget_scales_with_count (wgpu): base={base} slope={slope} n={n} \
         scaled_budget={scaled} tight_budget={tight}"
    );
}

/// Read an amdgpu sysfs memory-total file (bytes) for the first DRM card, in MiB (test-side mirror
/// of the worker's `amdgpu_sysfs_mem_mb`, using the same public parser). `0` if absent.
fn sysfs_mem_mb(file: &str) -> u64 {
    let Ok(cards) = std::fs::read_dir("/sys/class/drm") else {
        return 0;
    };
    for entry in cards.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        if !(name.starts_with("card") && name[4..].bytes().all(|b| b.is_ascii_digit())) {
            continue;
        }
        if let Ok(s) = std::fs::read_to_string(entry.path().join("device").join(file)) {
            if let Some(mb) = daemon_train::autotune::parse_amdgpu_mem_mb(&s) {
                if mb > 0 {
                    return mb;
                }
            }
        }
    }
    0
}

fn host_ram_mb() -> u64 {
    let Ok(text) = std::fs::read_to_string("/proc/meminfo") else {
        return 0;
    };
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            if let Some(kb) = rest
                .split_whitespace()
                .next()
                .and_then(|n| n.parse::<u64>().ok())
            {
                return kb / 1024;
            }
        }
    }
    0
}

/// HOST-8 (Merge-2 UMA re-run) `preset_160m_eligible_on_unified_device`: the headline autotune
/// check. With the real device probe + sysfs on THIS machine — dedicated VRAM (`mem_info_vram_total`
/// = 4096 MiB), the GTT/shared pool (`mem_info_gtt_total` = 120000 MiB), the wgpu `max_buffer_size`
/// per-buffer clamp (2047 MiB), and the adapter device type (IntegratedGpu → unified) — the 160M
/// preset is **eligible**. The same model against the pre-fix clamped limits (VRAM = the 2047 MiB
/// per-buffer proxy, non-unified, no shared pool) is REJECTED, documenting the fix delta.
///
/// The 160M footprint is built analytically from the preset's canonical param layout + the fp32
/// steady-state (params + master + grad + AdamW m/v = 20·N; host ≈ 8·N; spec §5.1), avoiding the
/// minutes-long full 160M meta execute pass (program Risk 3) — the full pass is exercised by the
/// wgpu 160M training smoke.
#[test]
fn preset_160m_eligible_on_unified_device() {
    require_gpu!();
    let probe = probe_wgpu().expect("adapter probed (require_gpu passed)");

    let cfg = TinyLlamaCfg::llama_160m();
    let n = cfg.param_count();
    let max_tensor_bytes = cfg
        .canonical_param_layout()
        .iter()
        .map(|(_, dims)| dims.iter().map(|&d| u64::from(d)).product::<u64>() * 4)
        .max()
        .expect("160M has params");
    const MIB: u64 = 1 << 20;
    let m160 = Autotune {
        fixed_vram_bytes: 20 * n, // 4N storage + 4N fp32 master + 4N fp32 grad + 8N AdamW m/v
        act_bytes_per_mb: 128 * MIB, // representative per-micro-batch activation
        host_ram_bytes: 8 * n,
        payload_bytes: 4 * n / 64,
        max_tensor_bytes,
    };

    // Real device limits, exactly as the worker's `device_limits()` builds them on this machine.
    let vram_mb = sysfs_mem_mb("mem_info_vram_total");
    let shared_mb = sysfs_mem_mb("mem_info_gtt_total");
    let ram_mb = host_ram_mb().max(1);
    let limits = DeviceLimits {
        vram_mb: if vram_mb > 0 {
            vram_mb
        } else {
            probe.max_alloc_mb
        },
        ram_mb,
        max_alloc_mb: probe.max_alloc_mb,
        shared_mb,
        unified: probe.unified,
    };
    let v = m160.verdict(&limits, DEFAULT_MAX_MICROBATCH);
    eprintln!(
        "preset_160m_eligible_on_unified_device: adapter={} device_type={} unified={} \
         vram_mb={} shared_mb={} max_alloc_mb={} ram_mb={} => {v:?}",
        probe.adapter,
        probe.device_type,
        probe.unified,
        limits.vram_mb,
        limits.shared_mb,
        limits.max_alloc_mb,
        limits.ram_mb
    );
    assert!(
        v.eligible,
        "160M must be ELIGIBLE with the real unified-memory probe on this machine: {:?}",
        v.reasons
    );
    assert!(v.micro_batch >= 1);

    // The pre-fix interpretation (VRAM = the 2047 MiB per-buffer clamp, non-unified, no GTT pool)
    // rejects the same model — this is the exact Merge-2 blocker the UMA fix resolves.
    let clamped = DeviceLimits {
        vram_mb: probe.max_alloc_mb, // 2047 on RADV/Mesa
        ram_mb,
        max_alloc_mb: probe.max_alloc_mb,
        shared_mb: 0,
        unified: false,
    };
    let before = m160.verdict(&clamped, DEFAULT_MAX_MICROBATCH);
    assert!(
        !before.eligible,
        "the pre-fix clamped budget must reject 160M (the blocker): {before:?}"
    );
    eprintln!("preset_160m (pre-fix clamped 2047 MiB) verdict: {before:?}");
}
