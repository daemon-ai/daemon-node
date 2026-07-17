// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `compute@2` on the **CUDA tier** — a true-backend probe of the deferred-error contract, and the
//! evidence that the fence-visibility gap is now closed on the CUDA backend.
//!
//! `burn-cubecl` implements `BackendIr`, so [`ComputeRunner`] instantiates over the real `Cuda`
//! backend UNCHANGED — the same codec, handle-liveness, RESERVED refusals, and deferred-error latch
//! as tier-1/wgpu, now over the real NVRTC-JIT CUDA queue.
//!
//! **Background (the gap this lane closed).** cubecl-cuda's allocation path
//! (`CudaServer::initialize_memory`) reserved device memory with `command.reserve(size).unwrap()`,
//! panicking on a fallible reservation. cubecl catches that per-task panic on its device-stream
//! thread and drops it, so no error was recorded on the stream's error queue and a subsequent
//! `RunnerClient::sync` (the `fence()`) returned `Ok` — device allocation faults were
//! readback-visible only, though [`ComputeRunner::fence`] is REQUIRED to surface deferred device
//! faults (architecture §3.3, ABI §15, decisions D8). The build under test pins a vendored
//! `cubecl-cuda` (`vendor/cubecl-cuda`, wired via `[patch.crates-io]`) whose alloc path records the
//! reservation failure on the stream error queue the way the `launch`/`write` paths already do, so
//! the fence drains it as the typed [`ComputeError::Device`] (trap twin `ComputeFault`).
//!
//! **What the cases below validate on real hardware (RTX 4090):**
//! - the injectable latch still surfaces at the fence, typed, exactly once (baseline);
//! - a **host-side pool-cap rejection** (a single buffer larger than any pool's `max_alloc_size`,
//!   refused before any `cuMemAlloc`) is now **fence-visible** typed, not just readback-visible;
//! - a **genuine driver `CUDA_ERROR_OUT_OF_MEMORY`** — provoked by allocations that are each
//!   individually pool-acceptable but sum past free VRAM, so the driver's own `cuMemAlloc` fails
//!   (the previous 2×-VRAM single buffer never engaged the driver) — is **fence-visible AND
//!   readback-visible** typed, enqueue stays infallible, and the host + CUDA context survive: the
//!   same runner and a fresh runner keep serving. This is the non-sticky driver-fault class.
//! - the genuinely **sticky** class (an illegal-address kernel that poisons the context) is
//!   **unreachable through the pinned op set** — every servable `burn_ir::OperationIr` variant is a
//!   bounds-checked tensor op, and `OperationIr::Custom` (the only escape to a hand-written kernel)
//!   is refused pre-dispatch by the host — documented and asserted below rather than claimed.
//!
//! Opt-in (`--features cuda`), self-skipping without a usable CUDA device + staged driver-matched
//! NVRTC (`DAEMON_CUDA_RUNTIME_DIR`; tier-2 discipline: GPU lanes are never the default gate).
#![cfg(feature = "cuda")]

use burn::tensor::{DType, Shape, Tensor, TensorData};
use burn_ir::{
    CustomOpIr, FullOpIr, NumericOperationIr, OperationIr, ScalarIr, TensorId, TensorIr,
    TensorStatus,
};
use daemon_vhc_host::compute::ComputeError;
use daemon_vhc_host::probe::{cuda_nvrtc_ready, probe_cuda};
use daemon_vhc_host::{cuda_adapter_available, ComputeRunner};

type CudaReal = burn::backend::Cuda;

macro_rules! require_cuda {
    () => {
        if !cuda_adapter_available() {
            eprintln!(
                "SKIP {}: no usable CUDA device (run in .#cuda-train on a CUDA box — TDD §8.1 tier-2)",
                module_path!()
            );
            return;
        }
        if !cuda_nvrtc_ready() {
            eprintln!(
                "SKIP {}: CUDA device present but NVRTC runtime not staged \
                 (DAEMON_CUDA_RUNTIME_DIR + CUDA_PATH)",
                module_path!()
            );
            return;
        }
    };
}

fn ser<T: serde::Serialize>(v: &T) -> Vec<u8> {
    let mut buf = Vec::new();
    ciborium::into_writer(v, &mut buf).expect("value encodes");
    buf
}

fn read_ir(id: u64, len: usize) -> Vec<u8> {
    ser(&TensorIr {
        id: TensorId::new(id),
        shape: Shape::from(vec![len]),
        status: TensorStatus::ReadOnly,
        dtype: DType::F32,
    })
}

/// A `Full` creation op whose single output buffer holds `elems` f32 values, named `out_id`. Used
/// to force device allocations of a controlled size.
fn full_op(out_id: u64, elems: usize) -> Vec<u8> {
    ser(&OperationIr::NumericFloat(
        DType::F32,
        NumericOperationIr::Full(FullOpIr {
            out: TensorIr {
                id: TensorId::new(out_id),
                shape: Shape::from(vec![elems]),
                status: TensorStatus::NotInit,
                dtype: DType::F32,
            },
            value: ScalarIr::Float(1.0),
        }),
    ))
}

/// The CUDA twin of `compute_wgpu.rs`: import → fence → readback round-trips over the real
/// asynchronous CUDA queue, and the injectable latch surfaces at the fence, typed, exactly once
/// (the seam a real queue-drain error lands in).
#[test]
fn cuda_tier_round_trips_and_defers_errors_to_the_fence() {
    require_cuda!();
    let device = Default::default();
    let mut runner = ComputeRunner::<CudaReal>::new(device);

    let data = TensorData::new(vec![1.0f32, -2.0, 3.5, 0.25], [4usize]);
    runner.import_tensor(1, &ser(&data)).expect("import");
    runner.fence().expect("clean fence after import");

    // A native-lane sanity op on the same backend (exercises the NVRTC JIT path end-to-end).
    let native = {
        let t = Tensor::<CudaReal, 1>::from_data(data.clone(), &Default::default());
        t.add_scalar(0.5f32)
            .into_data()
            .to_vec::<f32>()
            .expect("f32")
    };
    assert_eq!(native, vec![1.5f32, -1.5, 4.0, 0.75]);

    let exported = runner.read_tensor(&read_ir(1, 4)).expect("async readback");
    let round: TensorData = ciborium::from_reader(exported.as_slice()).expect("decodes");
    assert_eq!(
        round.to_vec::<f32>().expect("f32"),
        data.to_vec::<f32>().expect("f32"),
        "import → device → export round-trips on the CUDA tier"
    );

    runner.inject_device_fault("cuda-tier injected fault");
    let err = runner.fence().unwrap_err();
    assert!(matches!(err, ComputeError::Device(_)), "got {err:?}");
    assert_eq!(err.trap_code().slug(), "ComputeFault");
    runner.fence().expect("the fault surfaced exactly once");
}

/// **Fence-visibility of a host-side pool-cap rejection** (the narrower fault class, now closed by
/// the vendored cubecl alloc-error-queueing patch). An output sized ~2× total VRAM cannot fit any
/// pool's `max_alloc_size`, so cubecl refuses it (`IoError::BufferTooBig`) before any `cuMemAlloc`
/// — the driver is never engaged. Obligations:
///
/// 1. enqueue is infallible — `submit_op` returns `Ok` even though the allocation is refused (§3.3);
/// 2. the fault is **fence-visible**, typed — `fence()` (burn-router `RunnerClient::sync`) drains
///    the queued stream error and returns [`ComputeError::Device`] / `ComputeFault`. This is the
///    behavior the patch adds; the pre-patch backend returned `Ok` here (readback-visible only).
#[test]
fn oversized_alloc_reject_is_fence_visible_typed() {
    require_cuda!();
    let probe = probe_cuda().expect("device probed (require_cuda passed)");
    let mut runner = ComputeRunner::<CudaReal>::new(Default::default());

    // Prove the queue is healthy first.
    let data = TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [4usize]);
    runner.import_tensor(1, &ser(&data)).expect("import");
    runner.fence().expect("clean fence before the fault");

    // ~2x the device's total VRAM in one contiguous f32 buffer: no pool's `max_alloc_size` accepts
    // it, so the host-side allocator rejects it before any `cuMemAlloc`.
    let vram_bytes = probe.vram_mb * 1024 * 1024;
    let huge_elems = usize::try_from(vram_bytes * 2 / 4).expect("fits usize on x86_64");
    eprintln!(
        "host-side pool-cap rejection: requesting {} MiB against {} MiB of VRAM",
        huge_elems as u64 * 4 / (1024 * 1024),
        probe.vram_mb
    );

    // (1) Enqueue is infallible even though the allocation is refused.
    runner
        .submit_op(&full_op(100, huge_elems))
        .expect("enqueue is infallible — the fault must NOT surface at submit_op");

    // (2) With the alloc-error-queueing patch the fault is fence-visible, typed. Fence FIRST (before
    // any readback) so the drained stream error is unambiguously what `sync` reports.
    let err = runner
        .fence()
        .expect_err("the pool-cap rejection must be fence-visible with the cubecl patch");
    let ComputeError::Device(reason) = &err else {
        panic!("expected the typed deferred Device fault at the fence, got {err:?}");
    };
    eprintln!("the pool-cap rejection surfaced at the fence: {reason}");
    assert_eq!(err.trap_code().slug(), "ComputeFault");

    // The host + runner survive: a fresh runner on the same device constructs and works.
    let mut second = ComputeRunner::<CudaReal>::new(Default::default());
    second.import_tensor(1, &ser(&data)).expect("import");
    second.fence().expect("a fresh runner is clean");
    let exported = second.read_tensor(&read_ir(1, 4)).expect("readback");
    let round: TensorData = ciborium::from_reader(exported.as_slice()).expect("decodes");
    assert_eq!(
        round.to_vec::<f32>().expect("f32"),
        data.to_vec::<f32>().expect("f32"),
        "the host absorbed the allocation rejection and the device serves new work"
    );
}

/// **A genuine driver `CUDA_ERROR_OUT_OF_MEMORY`, fence-visible AND readback-visible, typed, with
/// host + context survival.** The prior 2×-VRAM single buffer was refused host-side by the pool cap
/// and never engaged the driver. Here every buffer is individually pool-acceptable (well under
/// `max_alloc_size`) but their sum far exceeds free VRAM, so the pool must grow and the driver's own
/// `cuMemAlloc` returns `CUDA_ERROR_OUT_OF_MEMORY` — which the GPU storage layer maps to `IoError`
/// and, with the patch, the alloc path queues on the stream so the fence/readback drains it. This is
/// the reachable *non-sticky* driver-fault class: the context is not poisoned, so recovery is clean.
///
/// Obligations: (1) enqueue stays infallible across the whole faulting window; (2) the fault is
/// fence-visible, typed; (3) it is readback-visible, typed (never a host panic/abort); (4) the same
/// runner AND a fresh runner keep serving afterward — the driver was never wedged.
#[test]
fn driver_oom_from_summed_pool_acceptable_allocs_is_fence_and_readback_visible_and_survivable() {
    require_cuda!();
    let probe = probe_cuda().expect("device probed (require_cuda passed)");
    let mut runner = ComputeRunner::<CudaReal>::new(Default::default());

    // Prove the queue is healthy first.
    let data = TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [4usize]);
    runner.import_tensor(1, &ser(&data)).expect("import");
    runner.fence().expect("clean fence before the fault");

    // 2 GiB per buffer — comfortably under any pool's `max_alloc_size` (~total VRAM on a discrete
    // card), so each is individually acceptable; the SUM is sized well past total VRAM so the driver
    // itself must run out (no host spill on a discrete 4090). Each output is a distinct live handle
    // (never dropped), so the pool cannot reuse and must keep growing.
    let per_buf_mb: u64 = 2048;
    let per_buf_elems = usize::try_from(per_buf_mb * 1024 * 1024 / 4).expect("fits usize");
    let n = usize::try_from(probe.vram_mb / per_buf_mb + 8).expect("fits"); // ~ VRAM/2GiB + 8 → past VRAM
    eprintln!(
        "driver-OOM stimulus: {n} × {per_buf_mb} MiB pool-acceptable allocs = {} MiB vs {} MiB VRAM",
        n as u64 * per_buf_mb,
        probe.vram_mb
    );

    // (1) Every enqueue is infallible — the device fault must never surface at submit_op.
    for i in 0..n {
        runner
            .submit_op(&full_op(1000 + i as u64, per_buf_elems))
            .expect("enqueue stays infallible through the faulting window");
    }

    // (2) The driver OOM is fence-visible, typed (the fix). Fence first so the drained stream error
    // is unambiguously the reported fault.
    let ferr = runner.fence().expect_err(
        "a genuine driver CUDA_ERROR_OUT_OF_MEMORY must be fence-visible with the patch",
    );
    let ComputeError::Device(reason) = &ferr else {
        panic!("expected the typed deferred Device fault at the fence, got {ferr:?}");
    };
    eprintln!("the driver OOM surfaced at the fence: {reason}");
    assert_eq!(ferr.trap_code().slug(), "ComputeFault");

    // (3) Readback of an affected (unbacked) handle also surfaces the typed Device fault — never a
    // host panic/abort. The LAST handle was submitted well after VRAM was exhausted, so its
    // allocation is the one that failed (early handles fit and are backed).
    let last_handle = 1000 + n as u64 - 1;
    let rerr = runner
        .read_tensor(&read_ir(last_handle, per_buf_elems))
        .expect_err("readback of an unbacked handle must surface the typed Device fault");
    let ComputeError::Device(reason) = &rerr else {
        panic!("expected the typed deferred Device fault at readback, got {rerr:?}");
    };
    eprintln!("the driver OOM surfaced at readback: {reason}");
    assert_eq!(rerr.trap_code().slug(), "ComputeFault");

    // (4) Survival — the driver OOM is non-sticky: the same runner serves fresh work...
    runner
        .import_tensor(2, &ser(&data))
        .expect("fresh import after the fault");
    runner
        .fence()
        .expect("clean fence after the fault — same runner recovered");
    let exported = runner
        .read_tensor(&read_ir(2, 4))
        .expect("readback after the fault — the CUDA context survived");
    let round: TensorData = ciborium::from_reader(exported.as_slice()).expect("decodes");
    assert_eq!(
        round.to_vec::<f32>().expect("f32"),
        data.to_vec::<f32>().expect("f32")
    );

    // ...and a fresh runner on the same device also constructs and works.
    let mut second = ComputeRunner::<CudaReal>::new(Default::default());
    second.import_tensor(1, &ser(&data)).expect("import");
    second.fence().expect("a fresh runner is clean");
    let exported = second.read_tensor(&read_ir(1, 4)).expect("readback");
    let round: TensorData = ciborium::from_reader(exported.as_slice()).expect("decodes");
    assert_eq!(
        round.to_vec::<f32>().expect("f32"),
        data.to_vec::<f32>().expect("f32"),
        "the device recovered from a genuine driver OOM"
    );
}

/// **The genuinely sticky CUDA error class is unreachable through the pinned `compute@2` op set.**
/// A sticky error (e.g. an illegal memory access that poisons the CUDA context so every subsequent
/// op fails until the context is recreated) can only come from a hand-written kernel that
/// dereferences out of bounds. The `compute@2` wire is `CBOR(burn_ir::OperationIr)` at the pinned
/// Burn version, and every *servable* variant is a bounds-checked burn tensor op — there is no
/// raw-pointer or out-of-bounds primitive to express. The one escape to an arbitrary kernel,
/// `OperationIr::Custom`, is refused by the host **before dispatch** (ABI §15 RESERVED). This test
/// records that refusal as the evidence of unreachability; the reachable driver fault (OOM, above)
/// is non-sticky and its context-survival is validated there.
#[test]
fn sticky_context_error_is_unreachable_through_the_pinned_op_set() {
    require_cuda!();
    let mut runner = ComputeRunner::<CudaReal>::new(Default::default());

    // Custom is the ONLY route to a hand-authored kernel (which could produce a sticky illegal
    // address). The host refuses it pre-dispatch, so the sticky-context class cannot be reached.
    let custom = ser(&OperationIr::Custom(CustomOpIr {
        id: "sticky_probe_illegal_address".to_string(),
        inputs: vec![],
        outputs: vec![],
    }));
    let err = runner
        .submit_op(&custom)
        .expect_err("Custom ops are refused before dispatch — no hand-written kernel can run");
    assert!(
        matches!(err, ComputeError::CustomOpUnsupported(_)),
        "the only kernel-authoring escape must be closed; got {err:?}"
    );
    eprintln!("sticky-error class is unreachable — Custom op refused pre-dispatch: {err}");

    // The refusal is a programming-error trap, not a device fault: the runner is untouched and
    // continues to serve.
    let data = TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [4usize]);
    runner
        .import_tensor(1, &ser(&data))
        .expect("import after the refusal");
    runner
        .fence()
        .expect("the refusal never touched the device");
    let exported = runner.read_tensor(&read_ir(1, 4)).expect("readback");
    let round: TensorData = ciborium::from_reader(exported.as_slice()).expect("decodes");
    assert_eq!(
        round.to_vec::<f32>().expect("f32"),
        data.to_vec::<f32>().expect("f32")
    );
}
