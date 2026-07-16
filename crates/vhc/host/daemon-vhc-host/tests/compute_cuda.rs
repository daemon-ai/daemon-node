// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `compute@2` on the **CUDA tier** — a true-backend probe of the C1 deferred-error gap
//! ("a genuine device-side fault path was only ever simulated via the injectable latch;
//! wgpu/AMD covered the async-queue shape, but real CUDA sticky-error semantics are
//! unexercised"). `burn-cubecl` implements `BackendIr`, so [`ComputeRunner`] instantiates over
//! the real `Cuda` backend UNCHANGED — the same codec, handle-liveness, RESERVED refusals, and
//! deferred-error latch as tier-1/wgpu, now over the real NVRTC-JIT CUDA queue.
//!
//! The non-injected fault lane: a `Full` creation op whose single output buffer is ~2× the
//! device's physical VRAM. **Observed semantics (RTX 4090, burn 0.21 / cubecl 0.10, run
//! 2026-07-16; wording corrected same day after an adversarial audit of the logs):** the
//! request is rejected HOST-SIDE by cubecl's memory pools (`IoError::BufferTooBig` — no pool's
//! `max_alloc_size` accepts it) before any `cuMemAlloc`; the driver is never engaged and no
//! `CUDA_ERROR_OUT_OF_MEMORY` occurs. The rejection panics the task on cubecl's device-stream
//! thread (`DSD-0-0`: `initialize_memory` unwraps the fallible reserve), NOT the submitting
//! thread — so enqueue stays infallible (§3.3 holds) and the dispatch-seam `catch_unwind`
//! never fires. cubecl catches that panic per-task, so the stream and runner keep serving; the
//! fault is scoped to the one unbacked handle and surfaces at **readback of the affected
//! tensor** as the typed [`ComputeError::Device`] (trap twin `ComputeFault`) — a generic
//! "runner panicked" fault for that handle, not an allocation diagnostic — never a host
//! panic/abort. **Reported gap:** `fence()` (burn-router `RunnerClient::sync`) returns `Ok`
//! after the fault — cubecl's alloc path panics instead of pushing the error into the stream's
//! error queue the way its `launch` path does, so no error state exists for `sync` to drain;
//! the deferred fault is readback-visible but not fence-visible, unlike the injectable-latch
//! model where both report. The assertions below accept a fence-side error if a future
//! burn/cubecl starts reporting one, but require the readback-side typed surfacing. The host +
//! runner survive to serve subsequent ops. Residual (C1's escalated gap stays open): genuine
//! driver-reported `CUDA_ERROR_OUT_OF_MEMORY` and sticky-error semantics (an illegal-address
//! kernel, which the pinned op set cannot express) remain unvalidated.
//!
//! Open follow-up (out of scope here, needs remote-hardware iteration): make cubecl-cuda's
//! alloc path record errors like its launch path (push into the stream's error queue) so
//! `fence()`/`sync` can drain the typed fault instead of returning `Ok`.
//!
//! Opt-in (`--features cuda`), self-skipping without a usable CUDA device + staged
//! driver-matched NVRTC (`DAEMON_CUDA_RUNTIME_DIR`; tier-2 discipline: GPU lanes are never the
//! default gate). Runnable lane: `.#cuda-train` on a CUDA box.
#![cfg(feature = "cuda")]

use burn::tensor::{DType, Shape, Tensor, TensorData};
use burn_ir::{
    FullOpIr, NumericOperationIr, OperationIr, ScalarIr, TensorId, TensorIr, TensorStatus,
};
use daemon_vhc_host::autotune::{cuda_nvrtc_ready, probe_cuda};
use daemon_vhc_host::compute::ComputeError;
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
                 (DAEMON_CUDA_RUNTIME_DIR + CUDA_PATH — swarm-ledger-p3-g D6)",
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

/// **The non-injected fault lane** (probing the C1 escalated gap — narrowed, not closed): an op
/// whose single output buffer cannot physically fit the device is rejected host-side by cubecl's
/// memory pools (`IoError::BufferTooBig`; the driver is never engaged — see the module doc). The
/// conformance obligations (see the module doc for the observed fence-visibility gap):
///
/// 1. enqueue is infallible — `submit_op` returns `Ok` even though the allocation is refused
///    (§3.3);
/// 2. the fault is DEFERRED and surfaces typed — `ComputeError::Device` (trap twin
///    `ComputeFault`) at the readback of the affected tensor (and at the fence too, if the
///    backend reports it there) — never a host panic, never a process abort;
/// 3. enqueue stays infallible in the faulted window;
/// 4. the host AND the runner survive: the fault is scoped to the unbacked handle, so the same
///    runner serves a fresh import → fence → readback afterwards, and a second runner on the
///    same device also works — the driver was never engaged, let alone wedged.
#[test]
fn oversized_alloc_reject_defers_to_readback_typed_and_the_host_survives() {
    require_cuda!();
    let probe = probe_cuda().expect("device probed (require_cuda passed)");
    let device = Default::default();
    let mut runner = ComputeRunner::<CudaReal>::new(device);

    // Prove the queue is healthy first.
    let data = TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [4usize]);
    runner.import_tensor(1, &ser(&data)).expect("import");
    runner.fence().expect("clean fence before the fault");

    // An output sized ~2x the device's total VRAM (f32 elements). One contiguous buffer, so no
    // cubecl memory pool's `max_alloc_size` accepts it: the host-side allocator rejects it
    // (`IoError::BufferTooBig`) before any `cuMemAlloc` — a real (if host-side) allocation
    // failure, not an injected one; the driver is never engaged.
    let vram_bytes = probe.vram_mb * 1024 * 1024;
    let huge_elems = usize::try_from(vram_bytes * 2 / 4).expect("fits usize on x86_64");
    eprintln!(
        "forcing a host-side pool-cap rejection: requesting {} MiB against {} MiB of VRAM",
        huge_elems as u64 * 4 / (1024 * 1024),
        probe.vram_mb
    );
    let huge_out = TensorIr {
        id: TensorId::new(100),
        shape: Shape::from(vec![huge_elems]),
        status: TensorStatus::NotInit,
        dtype: DType::F32,
    };
    let huge_op = OperationIr::NumericFloat(
        DType::F32,
        NumericOperationIr::Full(FullOpIr {
            out: huge_out,
            value: ScalarIr::Float(1.0),
        }),
    );
    // (1) Enqueue is infallible even though the allocation is refused.
    runner
        .submit_op(&ser(&huge_op))
        .expect("enqueue is infallible — the fault must NOT surface at submit_op");

    // (3) The faulted window: another enqueue still succeeds.
    let data2 = TensorData::new(vec![9.0f32], [1usize]);
    runner
        .import_tensor(2, &ser(&data2))
        .expect("enqueue stays infallible in the faulted window");

    // (2) The fault never crashes the host at the fence. Observed on real CUDA: the fault is
    // NOT fence-visible (burn-router sync returns Ok — cubecl's alloc path panics instead of
    // queueing the error for sync to drain; the reported gap); a future backend reporting it
    // here as the typed Device fault is also conformant.
    match runner.fence() {
        Ok(()) => eprintln!("fence after the alloc rejection: Ok (fault is readback-visible only — the observed cubecl 0.10 shape)"),
        Err(err) => {
            let ComputeError::Device(reason) = &err else {
                panic!("expected the typed deferred Device fault at the fence, got {err:?}");
            };
            eprintln!("the alloc-rejection fault surfaced at the fence: {reason}");
            assert_eq!(err.trap_code().slug(), "ComputeFault");
        }
    }

    // (2) The readback of the affected tensor MUST surface the typed deferred Device fault —
    // never a host panic/abort. (It is a generic "runner panicked" fault scoped to this handle,
    // not an allocation diagnostic.)
    let err = runner.read_tensor(&read_ir(100, huge_elems)).unwrap_err();
    let ComputeError::Device(reason) = &err else {
        panic!("expected the typed deferred Device fault at readback, got {err:?}");
    };
    eprintln!("the alloc-rejection fault surfaced at readback: {reason}");
    assert_eq!(err.trap_code().slug(), "ComputeFault");

    // (4) Survival: the same runner serves fresh work (the fault is scoped to the unbacked
    // handle — cubecl caught the per-task panic and the stream keeps serving)...
    runner.import_tensor(3, &ser(&data)).expect("fresh import");
    runner.fence().expect("clean fence after the fault");
    let exported = runner
        .read_tensor(&read_ir(3, 4))
        .expect("readback after the fault — the CUDA context survived");
    let round: TensorData = ciborium::from_reader(exported.as_slice()).expect("decodes");
    assert_eq!(
        round.to_vec::<f32>().expect("f32"),
        data.to_vec::<f32>().expect("f32")
    );

    // ...and a second runner on the same device also constructs and works: the driver (never
    // engaged by the rejected request) is sane.
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
