// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

// The probe reads platform device memory via the same `daemon_train::autotune` FFI the worker's
// `Command::Probe` path runs; no unsafe in this bin (the FFI lives in the lib behind safe wrappers).
#![forbid(unsafe_code)]

//! `daemon-train-probe` — a minimal, telemetry-free device-limits readout for fleet validation (C2).
//!
//! Prints the per-platform [`daemon_train::autotune::DeviceLimits`] the worker's `Probe`/assess path
//! computes (Windows DXGI/D3D12, macOS Metal, or the CPU fallback) — deployable as a single
//! cross-built binary to a bare fleet box (Windows cmd.exe, macOS, RunPod) to record the
//! three-platform probe matrix end-to-end. It links **none** of the worker's stdio/crash-reporting
//! stack (the always-on `sentry-rust-minidump` native-minidump path does not link under MinGW — see
//! swarm-ledger-p2-c2), so this is the linkable Windows validation artifact. The actual VRAM/UMA
//! decision logic is the shared `autotune` code, so the numbers are identical to a live `Probe`.

fn main() {
    println!("daemon-train-probe — device limits (swarm P2 C2)");
    println!("target_os = {}", std::env::consts::OS);

    #[cfg(windows)]
    {
        match daemon_train::autotune::probe_windows_device_limits() {
            Some(dl) => println!("windows DXGI/D3D12 device_limits = {dl:#?}"),
            None => println!("windows DXGI probe: no usable (non-WARP) adapter found"),
        }
    }
    #[cfg(target_os = "macos")]
    {
        match daemon_train::autotune::probe_macos_device_limits() {
            Some(dl) => println!("macos Metal device_limits = {dl:#?}"),
            None => println!("macos Metal probe: no MTLDevice available"),
        }
    }
    #[cfg(all(not(windows), not(target_os = "macos")))]
    {
        println!(
            "linux/other: the worker sources DeviceLimits from amdgpu sysfs + wgpu (feature `wgpu`); \
             run the real `daemon-train-worker` with DAEMON_TRAIN_PROBE=1 for the sysfs path."
        );
    }

    // CUDA lane runtime smoke (feature `cuda`, C2 RunPod adjudication evidence): bring up the real
    // burn-cuda backend (host libcuda via cudarc dynamic loading) and run one tensor op, which
    // exercises the NVRTC JIT path end-to-end. Failure panics loudly — this bin is validation-only.
    #[cfg(feature = "cuda")]
    {
        use burn::backend::cuda::{Cuda, CudaDevice};
        use burn::tensor::Tensor;
        let device = CudaDevice::new(0);
        let t = Tensor::<Cuda<f32, i32>, 1>::from_floats([1.0, 2.0, 3.0], &device);
        let sum = (t.clone() + t).sum().into_scalar();
        println!("cuda lane: device 0 up, (t + t).sum() = {sum} (expected 12)");
    }
}
