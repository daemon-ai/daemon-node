// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The `compute@2` host runner (Phase C, track C1) — the Burn-shaped compute world.
//!
//! **The wire is `CBOR(burn_ir::OperationIr)` at the pinned Burn version** (`burn = 0.21.0`;
//! [`daemon_vhc_abi::COMPUTE_BURN_VERSION`]) — decisions D8, ABI §15. The Burn-over-`HostBackend`
//! spike (the Phase-C entry gate) established the load-bearing finding this module productionizes:
//! **`burn-router` + `burn-ir` already *are* the `compute@` boundary.** `Runner<B>` +
//! `OperationIr`/`TensorId`/`TensorIr`/`TensorData` are exactly a handle-based op-lowering backend —
//! opaque `u64` handles, one rank/dtype-erased serializable op enum, guest-side metadata and
//! refcount+`Drop`, one blocking readback with a typed error — so the governed artifact is the
//! **pinned IR schema + the runner dispatch match table** (which *is* the op inventory), never a
//! hand-curated op list.
//!
//! This module is the host side of the wasm import shim ([`daemon_vhc_abi::COMPUTE_V2_SYMBOLS`]:
//! `submit_op`/`fence`/`export`/`import`). It:
//!
//! - decodes each `CBOR(OperationIr)` op-blob and dispatches it through `burn-router`'s
//!   [`Runner`] over a real backend (tier-1: [`HostReal`] = ndarray CPU; wgpu/cuda ride the same
//!   [`burn_ir::BackendIr`] seam behind the host's feature lanes);
//! - **refuses the RESERVED variants cleanly** (ABI §15): `OperationIr::Custom` (custom/fused
//!   kernels stay in C2's host-side custom-op registry) and quantization/`QFloat`
//!   (`Float(Quantize/Dequantize)` do not lower today) — a typed refusal, never a panic;
//! - maps **stale/unknown tensor handles to a typed [`TrapCode::StaleHandle`] /
//!   [`TrapCode::InvalidHandle`]** rather than the host crash the upstream runner would produce
//!   (the Phase-C obligation from the spike: "the runner panics on an unknown id"). It does so by
//!   tracking handle liveness host-side and validating operands *before* dispatch, so the runner is
//!   never handed an id it does not hold.
//!
//! Handles: `TensorId`s are **guest-minted, guest reference-counted, released via
//! `OperationIr::Drop`** (ABI §15). They travel *inside* the op-blob and are a guest-owned id space
//! — like staging/timer IDs, **not** [§7.2](daemon_vhc_abi::pack_handle) host-arena handles. Bulk
//! `TensorData` upload/readback rides `BufferHandle`/`read_into` (§3.4), never inline in the
//! op-stream; this module is the codec + dispatch, and the driver's `export`/`import` shim moves the
//! bulk bytes over the buffer layer.

use std::collections::BTreeSet;
use std::panic::AssertUnwindSafe;

use burn_backend::TensorData;
use burn_ir::{BackendIr, FloatOperationIr, OperationIr, TensorId, TensorIr, TensorStatus};
use burn_ndarray::{NdArray, NdArrayDevice};
use burn_router::{Runner, RunnerClient};

use crate::trap::TrapCode;

/// The tier-1 real backend the host executes `compute@2` ops on: ndarray CPU (f32/i64/i8).
///
/// `burn-ndarray` is always compiled (the root `burn` dep pins `std`+`ndarray`+`autodiff`), so this
/// alias is unconditional. The wgpu/cuda tiers ([`ComputeRunner`] is generic over
/// [`burn_ir::BackendIr`]) ride the same seam behind the host's `wgpu`/`cuda` feature lanes.
pub type HostReal = NdArray<f32, i64, i8>;

/// A typed `compute@2` fault (decisions D8, ABI §7.6/§15). Enqueue is infallible in the
/// command-queue model (architecture §3.3); these surface at `submit_op` argument validation, at a
/// fence, or at a readback.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ComputeError {
    /// A tensor handle that WAS registered but is no longer live (dropped, or from a dead
    /// instance) — maps to [`TrapCode::StaleHandle`] (ABI §7.6). The upstream runner panics on
    /// this; C1's obligation is the typed trap.
    #[error("stale tensor handle {0}: dropped or from a dead instance")]
    StaleHandle(u64),
    /// A tensor handle that was NEVER registered — maps to [`TrapCode::InvalidHandle`].
    #[error("invalid tensor handle {0}: never registered")]
    InvalidHandle(u64),
    /// A RESERVED op variant refused cleanly until specified (ABI §15): custom ops (C2 registry)
    /// or quantization/`QFloat`.
    #[error("reserved compute operation refused: {0}")]
    Reserved(&'static str),
    /// The op-blob / tensor-data was not decodable canonical CBOR of the pinned IR (a malformed
    /// guest-supplied structure, ABI §7.6 `BadEvent`; also the clean refusal for the feature-gated
    /// `Distributed` variant, whose absence makes it undecodable).
    #[error("compute op decode failed: {0}")]
    Decode(String),
    /// A deferred device execution error surfaced at fence/readback (CUDA-style, architecture
    /// §3.3). Tier-1 ndarray is synchronous so this is rare; the wgpu/cuda tiers exercise the
    /// deferred timing.
    #[error("device execution error: {0}")]
    Device(String),
}

impl ComputeError {
    /// The host trap code this fault surfaces as at a readback / fence (ABI §7.6). Stale/unknown
    /// handles are the load-bearing mapping (§15); reserved/undecodable ops fail closed as
    /// `BadEnum`/`BadEvent`; a device error surfaces as a `BudgetOps`-free typed compute fault.
    #[must_use]
    pub fn trap_code(&self) -> TrapCode {
        match self {
            Self::StaleHandle(_) => TrapCode::StaleHandle,
            Self::InvalidHandle(_) => TrapCode::InvalidHandle,
            // A reserved op variant is an op the host does not serve — fail closed as an unknown
            // enum value (the guest sent an op outside the pinned/served set).
            Self::Reserved(_) => TrapCode::BadEnum,
            Self::Decode(_) => TrapCode::BadEvent,
            // A deferred device error at readback is a compute fault, not a resource breach; the
            // async completion path maps it to a `comp-error` instead (driver shim).
            Self::Device(_) => TrapCode::ComputeFault,
        }
    }
}

/// Classify an [`OperationIr`] as a RESERVED variant refused until specified (ABI §15), returning a
/// human-readable reason, or `None` for a servable op. This is the one governance point beyond the
/// pinned IR schema: it names exactly the variants the router runner does not lower today.
#[must_use]
pub fn reserved_reason(op: &OperationIr) -> Option<&'static str> {
    match op {
        // Custom/fused kernels cross the boundary as registered names, not as generic IR — they
        // stay in C2's host-side custom-op registry (architecture §3.2). The runner panics
        // ("Can't execute custom operation here") by design; C1 refuses cleanly instead.
        OperationIr::Custom(_) => Some(
            "OperationIr::Custom — custom/fused kernels are the C2 host-side custom-op registry \
             (architecture §3.2; ABI §15 RESERVED)",
        ),
        // Quantization / QFloat: `Quantize`/`Dequantize` are `todo!()` in the router runner —
        // quantized tensors do not lower today (ABI §15 RESERVED).
        OperationIr::Float(_, FloatOperationIr::Quantize(_)) => Some(
            "OperationIr::Float(Quantize) — quantization/QFloat does not lower (ABI §15 RESERVED)",
        ),
        OperationIr::Float(_, FloatOperationIr::Dequantize(_)) => Some(
            "OperationIr::Float(Dequantize) — quantization/QFloat does not lower (ABI §15 RESERVED)",
        ),
        _ => None,
    }
}

/// The host-side runner for one role-instance's `compute@2` command queue.
///
/// Wraps `burn-router`'s [`Runner`] over a real backend `B` and owns the host-side handle-liveness
/// bookkeeping (`live`/`seen`) that turns the runner's would-be panic on an unknown `TensorId` into
/// a typed [`TrapCode::StaleHandle`]/[`TrapCode::InvalidHandle`] (ABI §15). Instance-class: dropped
/// with the instance; a fresh incarnation gets a fresh runner (stale handles from the dead one are
/// simply absent from `live`).
pub struct ComputeRunner<B: BackendIr> {
    runner: Runner<B>,
    /// `TensorId`s currently registered host-side (a live handle).
    live: BTreeSet<u64>,
    /// `TensorId`s ever registered — distinguishes a dropped handle (`StaleHandle`) from one that
    /// was never valid (`InvalidHandle`).
    seen: BTreeSet<u64>,
}

impl ComputeRunner<HostReal> {
    /// A tier-1 ndarray-CPU runner ([`HostReal`]).
    #[must_use]
    pub fn ndarray_cpu() -> Self {
        Self::new(NdArrayDevice::Cpu)
    }
}

impl<B: BackendIr> ComputeRunner<B> {
    /// A runner over `device` of the real backend `B` (the generic tier: ndarray tier-1;
    /// wgpu/cuda behind features).
    #[must_use]
    pub fn new(device: B::Device) -> Self {
        Self {
            runner: Runner::new(device),
            live: BTreeSet::new(),
            seen: BTreeSet::new(),
        }
    }

    /// Whether `id` is a currently-live tensor handle (test/introspection).
    #[must_use]
    pub fn is_live(&self, id: u64) -> bool {
        self.live.contains(&id)
    }

    /// The typed fault for a non-live operand: `StaleHandle` if it was ever registered, else
    /// `InvalidHandle` (ABI §7.6).
    fn absent_handle_fault(&self, id: u64) -> ComputeError {
        if self.seen.contains(&id) {
            ComputeError::StaleHandle(id)
        } else {
            ComputeError::InvalidHandle(id)
        }
    }

    /// `compute@2::submit_op` — decode + dispatch one `CBOR(OperationIr)` op-blob (command-queue
    /// enqueue; ABI §3.3). RESERVED variants are refused; every read operand is validated live
    /// *before* dispatch so the runner never panics on an unknown id; output handles become live.
    ///
    /// # Errors
    ///
    /// [`ComputeError::Decode`] (malformed op-blob / undecodable variant),
    /// [`ComputeError::Reserved`] (custom/QFloat), [`ComputeError::StaleHandle`] /
    /// [`ComputeError::InvalidHandle`] (a non-live read operand), or [`ComputeError::Device`] (a
    /// residual runner fault, defensively caught rather than crashing the host).
    pub fn submit_op(&mut self, op_cbor: &[u8]) -> Result<(), ComputeError> {
        let op: OperationIr =
            ciborium::from_reader(op_cbor).map_err(|e| ComputeError::Decode(e.to_string()))?;

        if let Some(reason) = reserved_reason(&op) {
            return Err(ComputeError::Reserved(reason));
        }

        // `Drop` retires a handle: it must be one we hold (else it is a stale/invalid free), and it
        // leaves `live` without ever reaching the runner as a required input.
        if let OperationIr::Drop(t) = &op {
            let id = t.id.value();
            if !self.live.contains(&id) {
                return Err(self.absent_handle_fault(id));
            }
            self.live.remove(&id);
            // Register the Drop so the runner frees its buffer; it holds the id, so this is safe.
            self.dispatch(op)?;
            return Ok(());
        }

        // Validate every read operand (skip freshly-created `NotInit` outputs) before dispatch.
        for t in op.inputs() {
            if t.status != TensorStatus::NotInit && !self.live.contains(&t.id.value()) {
                return Err(self.absent_handle_fault(t.id.value()));
            }
        }

        // Collect output ids before moving `op` into the runner.
        let outputs: Vec<u64> = op.outputs().map(|t| t.id.value()).collect();
        self.dispatch(op)?;
        for id in outputs {
            self.live.insert(id);
            self.seen.insert(id);
        }
        Ok(())
    }

    /// Dispatch an already-validated op through the runner, defensively catching any residual
    /// panic (an id our liveness tracking somehow missed) and surfacing it as a typed fault rather
    /// than a host crash (ABI §7.6 / §15: "never a host crash").
    fn dispatch(&self, op: OperationIr) -> Result<(), ComputeError> {
        let runner = &self.runner;
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let res = std::panic::catch_unwind(AssertUnwindSafe(|| runner.register_op(op)));
        std::panic::set_hook(prev);
        res.map_err(|_| {
            ComputeError::Device("runner panicked dispatching an op (unknown handle?)".to_string())
        })
    }

    /// `compute@2::import` — register host-owned `CBOR(TensorData)` under the guest-minted `id`
    /// (staging → device). The tensor becomes live. The bulk bytes ride a `BufferHandle` at the
    /// shim; this is the codec + registration.
    ///
    /// # Errors
    ///
    /// [`ComputeError::Decode`] when the data is not decodable canonical CBOR of `TensorData`.
    pub fn import_tensor(&mut self, id: u64, data_cbor: &[u8]) -> Result<(), ComputeError> {
        let data: TensorData =
            ciborium::from_reader(data_cbor).map_err(|e| ComputeError::Decode(e.to_string()))?;
        self.runner.register_tensor_data_id(TensorId::new(id), data);
        self.live.insert(id);
        self.seen.insert(id);
        Ok(())
    }

    /// `compute@2::export` / the readback path — read a live tensor named by `CBOR(TensorIr)` back
    /// to `CBOR(TensorData)` (device → staging). Blocking (architecture §3.2 second blocking
    /// point); deferred device errors surface here (§3.3).
    ///
    /// # Errors
    ///
    /// [`ComputeError::Decode`] (bad `TensorIr`), [`ComputeError::StaleHandle`] /
    /// [`ComputeError::InvalidHandle`] (the tensor is not live), or [`ComputeError::Device`] (a
    /// deferred device execution error).
    pub fn read_tensor(&self, ir_cbor: &[u8]) -> Result<Vec<u8>, ComputeError> {
        let ir: TensorIr =
            ciborium::from_reader(ir_cbor).map_err(|e| ComputeError::Decode(e.to_string()))?;
        let id = ir.id.value();
        if !self.live.contains(&id) {
            return Err(self.absent_handle_fault(id));
        }
        let runner = &self.runner;
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let res =
            std::panic::catch_unwind(AssertUnwindSafe(|| block_on(runner.read_tensor_async(ir))));
        std::panic::set_hook(prev);
        let data = match res {
            Ok(Ok(data)) => data,
            Ok(Err(e)) => return Err(ComputeError::Device(format!("{e}"))),
            Err(_) => {
                return Err(ComputeError::Device(
                    "runner panicked reading a tensor (unknown handle?)".to_string(),
                ))
            }
        };
        let mut buf = Vec::new();
        ciborium::into_writer(&data, &mut buf).map_err(|e| ComputeError::Device(e.to_string()))?;
        Ok(buf)
    }

    /// `compute@2::fence` — drain the command queue to a consistent point (architecture §3.3). For
    /// the synchronous tier-1 backend this is a no-op success; on wgpu/cuda it awaits the device
    /// and surfaces a deferred error here. The driver delivers `Event::Fence(id)` after a
    /// successful fence.
    ///
    /// # Errors
    ///
    /// [`ComputeError::Device`] when the device reports a deferred execution error.
    pub fn fence(&self) -> Result<(), ComputeError> {
        RunnerClient::sync(&self.runner).map_err(|e| ComputeError::Device(format!("{e}")))
    }

    /// Seed the host backend's RNG (host-side `Float/Random`, journaled for determinism by the
    /// driver).
    pub fn seed(&self, seed: u64) {
        self.runner.seed(seed);
    }

    /// Force-reclaim every live handle (trap/teardown, ABI §7.3): a fresh incarnation gets a fresh
    /// runner, so this only clears the bookkeeping.
    pub fn clear(&mut self) {
        self.live.clear();
    }
}

/// Minimal blocking executor for the runner's readback future. A synchronous backend's
/// `read_tensor_async` future is always immediately `Ready`, so this never spins in practice;
/// device backends complete when the queue drains.
fn block_on<F: core::future::Future>(fut: F) -> F::Output {
    use core::task::{Context, Poll, Waker};
    // `Waker::noop` is safe (stable since 1.85; the crate MSRV is 1.93), so this executor needs no
    // `unsafe` — the crate keeps its `#![deny(unsafe_code)]` guarantee.
    let waker = Waker::noop();
    let mut cx = Context::from_waker(waker);
    let mut fut = Box::pin(fut);
    loop {
        match fut.as_mut().poll(&mut cx) {
            Poll::Ready(v) => return v,
            Poll::Pending => core::hint::spin_loop(),
        }
    }
}
