// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// P3 Lane G — lifecycle-level CUDA tests (the CUDA analogue of `wgpu_lifecycle.rs`): HOST-8
// (`meta_mode_estimates_vs_cuda_probe` against the real driver-reported VRAM) and the headline
// autotune verdict (`preset_160m_eligible_on_cuda_discrete` — 160M fits the 4090's 24 GB discrete
// budget, no UMA).
//
// GPU-skip convention (TDD §8.1 tier-2): each test checks `cuda_adapter_available()` and skips with a
// loud stderr note when absent. The `.#cuda-train` devShell on a CUDA box is the runnable lane.
#![cfg(feature = "cuda")]
#![allow(clippy::disallowed_methods)]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Once;

use daemon_train::autotune::{cuda_device_limits, probe_cuda, Autotune, DEFAULT_MAX_MICROBATCH};
use daemon_train::{cuda_adapter_available, EngineConfig, Worker};
use daemon_train_sdk::models::TinyLlamaCfg;
use serde::Serialize;

const MIB: u64 = 1 << 20;

macro_rules! require_gpu {
    () => {
        if !cuda_adapter_available() {
            eprintln!(
                "SKIP {}: no usable CUDA device (run in .#cuda-train on a CUDA box — TDD §8.1 tier-2)",
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
                 {hex}. Expected across worktrees/machines (path-keyed codegen ordering); the \
                 freshly-built module is used.",
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

/// HOST-8 `meta_mode_estimates_vs_cuda_probe`: the meta-pass byte footprints feed the autotune, whose
/// verdict against THIS box's real CUDA device numbers (`probe_cuda` total VRAM + /proc/meminfo host
/// RAM, mapped by `cuda_device_limits` — discrete, no UMA) is eligible with internally consistent
/// estimates. The estimates themselves are backend-independent (shapes/dtypes), so a CPU meta pass
/// compared against the real GPU limits is the honest §6.5 admission shape.
#[test]
fn meta_mode_estimates_vs_cuda_probe() {
    require_gpu!();
    let worker = Worker::new(EngineConfig::default()).unwrap();
    let module = worker.load_module(&tiny_llama_wasm()).unwrap();
    let mut inst = worker.instantiate(&module).unwrap();
    let report = inst
        .meta(&cbor(&tiny_cfg()), 1, tiny_cfg().seq_len)
        .unwrap();

    assert_eq!(report.master_bytes, report.grad_bytes);
    assert!(report.param_bytes > 0 && report.master_bytes > 0);
    assert!(report.act_bytes_est > 0 && report.host_ram_bytes_est > 0);
    assert!(report.payload_bytes_est > 0);

    let autotune = Autotune::from_meta(&report);
    let core = report.param_bytes + report.master_bytes + report.grad_bytes;
    assert!(
        autotune.fixed_vram_bytes >= core,
        "fixed VRAM covers at least params+masters+grads ({} >= {core})",
        autotune.fixed_vram_bytes
    );

    // Real device numbers: the CUDA driver's total-VRAM query (24564 MiB on the 4090).
    let probe = probe_cuda().expect("device probed (require_gpu passed)");
    assert_eq!(probe.gpus, 1);
    assert!(probe.vram_mb > 0, "cuDeviceTotalMem is queryable");
    assert!(!probe.unified, "the 4090 is discrete — no UMA");
    eprintln!(
        "cuda probe: adapter={} vram_mb={} (max_alloc_mb={})",
        probe.adapter, probe.vram_mb, probe.max_alloc_mb
    );

    let ram_mb = host_ram_mb().max(1);
    let limits = cuda_device_limits(probe.vram_mb, probe.max_alloc_mb, ram_mb);
    assert_eq!(limits.shared_mb, 0, "discrete: no shared spill pool");
    assert!(!limits.unified);
    let v = autotune.verdict(&limits, DEFAULT_MAX_MICROBATCH);
    eprintln!("meta_mode_estimates_vs_cuda_probe: verdict={v:?} (limits={limits:?})");
    assert!(
        v.eligible,
        "the 1-layer tiny-llama must fit the 4090's real device budget: {:?}",
        v.reasons
    );
    assert!(v.micro_batch >= 1);
    assert_eq!(v.payload_bytes_estimate, report.payload_bytes_est);
}

/// The headline autotune check on CUDA: with the real driver-reported VRAM on the 4090 (24 GB
/// discrete, no UMA), the 160M preset is **eligible** on the discrete verdict path. The 160M footprint
/// is built analytically from the fp32 steady state (params + master + grad + AdamW m/v = 20·N; host ≈
/// 8·N; §5.1), avoiding the minutes-long full 160M meta execute pass (the full pass is the
/// `preset_160m_cuda` / `reference_parity_cuda` smoke).
#[test]
fn preset_160m_eligible_on_cuda_discrete() {
    require_gpu!();
    let probe = probe_cuda().expect("device probed (require_gpu passed)");
    let cfg = TinyLlamaCfg::llama_160m();
    let n = cfg.param_count();
    let max_tensor_bytes = cfg
        .canonical_param_layout()
        .iter()
        .map(|(_, dims)| dims.iter().map(|&d| u64::from(d)).product::<u64>() * 4)
        .max()
        .expect("160M has params");
    let m160 = Autotune {
        fixed_vram_bytes: 20 * n, // 4N storage + 4N fp32 master + 4N fp32 grad + 8N AdamW m/v
        act_bytes_per_mb: 128 * MIB,
        host_ram_bytes: 8 * n,
        payload_bytes: 4 * n / 64,
        max_tensor_bytes,
    };
    let ram_mb = host_ram_mb().max(1);
    let limits = cuda_device_limits(probe.vram_mb, probe.max_alloc_mb, ram_mb);
    let v = m160.verdict(&limits, DEFAULT_MAX_MICROBATCH);
    eprintln!(
        "preset_160m_eligible_on_cuda_discrete: adapter={} vram_mb={} ram_mb={} => {v:?}",
        probe.adapter, limits.vram_mb, limits.ram_mb
    );
    assert!(
        v.eligible,
        "160M must be ELIGIBLE on the 4090's discrete 24 GB budget: {:?}",
        v.reasons
    );
    assert!(v.micro_batch >= 1);
    // The device footprint (~2.9 GiB fixed) is well under 24 GB, and there is no UMA joint pool.
    assert!(!limits.unified);
    assert_eq!(limits.shared_mb, 0);
}
