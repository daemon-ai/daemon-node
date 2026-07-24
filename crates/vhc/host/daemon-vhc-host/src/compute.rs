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
/// command-queue model (architecture §3.3) — a *device* failure never surfaces at `submit_op`;
/// these arise at `submit_op` **argument validation** (stale/invalid handles, reserved variants,
/// undecodable blobs — programming errors, which trap immediately per §7.6), or as **deferred
/// device errors** at a fence/readback.
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
    /// `OperationIr::Custom{id}` — named custom ops do not dispatch through the generic IR wire
    /// (they are C2's host-side custom-op registry, `v2::CustomOpRegistry`; actual Custom-IR
    /// dispatch is deferred until specified). Carries the C2 refusal vocabulary
    /// ([`daemon_vhc_abi::AbiRefusalCode::CustomOpUnsupported`]) so both halves of the Phase-C
    /// seam speak one code.
    #[error("CustomOpUnsupported: custom op `{0}` does not dispatch through the compute@2 IR wire (C2 registry seam, ABI §15)")]
    CustomOpUnsupported(String),
    /// A RESERVED op variant refused cleanly until specified (ABI §15): quantization/`QFloat`.
    #[error("reserved compute operation refused: {0}")]
    Reserved(&'static str),
    /// The op-blob / tensor-data was not decodable canonical CBOR of the pinned IR (a malformed
    /// guest-supplied structure, ABI §7.6 `BadEvent`; also the clean refusal for the feature-gated
    /// `Distributed` variant, whose absence makes it undecodable).
    #[error("compute op decode failed: {0}")]
    Decode(String),
    /// A deferred device execution error surfaced at fence/readback (CUDA-style, architecture
    /// §3.3): the tier-1 trap twin is [`TrapCode::ComputeFault`]; the completion twin is
    /// [`daemon_vhc_abi::COMP_ERR_DEVICE`]. Synchronous ndarray defers dispatch faults through
    /// [`ComputeRunner`]'s pending-error latch; the wgpu/cuda tiers ride the same seam with real
    /// async timing.
    #[error("device execution error: {0}")]
    Device(String),
}

impl ComputeError {
    /// The host trap code this fault surfaces as (ABI §7.6). Stale/unknown handles are the
    /// load-bearing mapping (§15); reserved/custom/undecodable ops fail closed as
    /// `BadEnum`/`BadEvent` (the custom-op trap detail carries the C2
    /// `CustomOpUnsupported` vocabulary code via `Display`); a deferred device error at a fence
    /// is the typed compute fault.
    #[must_use]
    pub fn trap_code(&self) -> TrapCode {
        match self {
            Self::StaleHandle(_) => TrapCode::StaleHandle,
            Self::InvalidHandle(_) => TrapCode::InvalidHandle,
            // An op variant the host does not serve — fail closed as an unknown enum value.
            Self::CustomOpUnsupported(_) | Self::Reserved(_) => TrapCode::BadEnum,
            Self::Decode(_) => TrapCode::BadEvent,
            Self::Device(_) => TrapCode::ComputeFault,
        }
    }

    /// The admission-vocabulary refusal code this fault corresponds to, where one exists: the
    /// C2 seam's [`daemon_vhc_abi::AbiRefusalCode::CustomOpUnsupported`] for a submitted
    /// `OperationIr::Custom` (the manifest-level twin is `CustomOpRegistry::admit`).
    #[must_use]
    pub fn refusal_code(&self) -> Option<daemon_vhc_abi::AbiRefusalCode> {
        match self {
            Self::CustomOpUnsupported(_) => {
                Some(daemon_vhc_abi::AbiRefusalCode::CustomOpUnsupported)
            }
            _ => None,
        }
    }
}

/// Classify an [`OperationIr`] the generic IR wire does not serve (ABI §15), or `None` for a
/// servable op. This is the one governance point beyond the pinned IR schema: it names exactly
/// the variants the router runner does not lower today.
///
/// - `Custom` → [`ComputeError::CustomOpUnsupported`] carrying the op's versioned name: named
///   custom ops resolve through C2's `v2::CustomOpRegistry` at admission; actual Custom-IR
///   dispatch stays deferred (the runner panics "Can't execute custom operation here" by design —
///   C1 refuses cleanly instead, with the C2 vocabulary code).
/// - Quantization/`QFloat` (`Float(Quantize/Dequantize)`) → [`ComputeError::Reserved`]: `todo!()`
///   in the router runner — quantized tensors do not lower today.
#[must_use]
pub fn unservable_op(op: &OperationIr) -> Option<ComputeError> {
    match op {
        OperationIr::Custom(c) => Some(ComputeError::CustomOpUnsupported(c.id.clone())),
        OperationIr::Float(_, FloatOperationIr::Quantize(_)) => Some(ComputeError::Reserved(
            "OperationIr::Float(Quantize) — quantization/QFloat does not lower (ABI §15 RESERVED)",
        )),
        OperationIr::Float(_, FloatOperationIr::Dequantize(_)) => Some(ComputeError::Reserved(
            "OperationIr::Float(Dequantize) — quantization/QFloat does not lower (ABI §15 RESERVED)",
        )),
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
    /// The **deferred-error latch** (CUDA-style semantics, architecture §3.3): a device execution
    /// fault parked here at enqueue surfaces at the NEXT fence or readback, never at `submit_op`.
    /// Synchronous ndarray populates it from a caught dispatch fault; async device tiers
    /// (wgpu/cuda) populate it from their queue-drain results at the same seam; tests inject
    /// through [`Self::inject_device_fault`] (the testkit simulated-providers pattern) — so the
    /// deferred *timing* is exercised even on the CPU tier (the true-GPU timing gap is reported,
    /// not silent).
    pending_device_error: Option<String>,
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
            pending_device_error: None,
        }
    }

    /// Park a device fault to surface at the next fence/readback (deferred-error semantics,
    /// architecture §3.3). The injectable-fault seam for the deferred-error conformance tests and
    /// the async device tiers.
    pub fn inject_device_fault(&mut self, reason: impl Into<String>) {
        self.pending_device_error.get_or_insert(reason.into());
    }

    /// Take the parked device fault, if any (it surfaces exactly once).
    fn take_device_fault(&mut self) -> Result<(), ComputeError> {
        match self.pending_device_error.take() {
            Some(reason) => Err(ComputeError::Device(reason)),
            None => Ok(()),
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

    /// `compute@2::submit_op` — decode + enqueue one `CBOR(OperationIr)` op-blob (command-queue
    /// enqueue; ABI §3.3). Validation faults (undecodable blob, unservable variant, non-live read
    /// operand) are **programming errors and trap at the call** (§7.6); a **device** fault never
    /// surfaces here — it parks in the deferred-error latch and surfaces at the next
    /// fence/readback (§3.3). Output handles become live.
    ///
    /// # Errors
    ///
    /// [`ComputeError::Decode`] (malformed op-blob / undecodable variant),
    /// [`ComputeError::CustomOpUnsupported`] / [`ComputeError::Reserved`] (custom / QFloat), or
    /// [`ComputeError::StaleHandle`] / [`ComputeError::InvalidHandle`] (a non-live read operand).
    pub fn submit_op(&mut self, op_cbor: &[u8]) -> Result<(), ComputeError> {
        let op: OperationIr =
            ciborium::from_reader(op_cbor).map_err(|e| ComputeError::Decode(e.to_string()))?;

        if let Some(err) = unservable_op(&op) {
            return Err(err);
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
            self.dispatch(op);
            return Ok(());
        }

        // Validate every read operand (skip freshly-created `NotInit` outputs) before dispatch.
        for t in op.inputs() {
            if t.status != TensorStatus::NotInit && !self.live.contains(&t.id.value()) {
                return Err(self.absent_handle_fault(t.id.value()));
            }
        }

        // Collect output ids before moving `op` into the runner. Outputs become live even when
        // the device queue is poisoned (a latched fault): the guest keeps enqueueing against a
        // consistent handle space and the fault surfaces, once, at the fence/readback.
        let outputs: Vec<u64> = op.outputs().map(|t| t.id.value()).collect();
        self.dispatch(op);
        for id in outputs {
            self.live.insert(id);
            self.seen.insert(id);
        }
        Ok(())
    }

    /// Dispatch an already-validated op through the runner. A residual runner fault (an id the
    /// liveness tracking somehow missed, an internal backend failure) is caught and **parked in
    /// the deferred-error latch** — never a host crash (ABI §7.6/§15), never a `submit_op` error
    /// (enqueue is infallible, §3.3). A poisoned queue (already-latched fault) skips dispatch:
    /// real device queues stop executing after a fault, and the first fault is the one reported.
    fn dispatch(&mut self, op: OperationIr) {
        if self.pending_device_error.is_some() {
            return;
        }
        let runner = &self.runner;
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let res = std::panic::catch_unwind(AssertUnwindSafe(|| runner.register_op(op)));
        std::panic::set_hook(prev);
        if res.is_err() {
            self.pending_device_error
                .get_or_insert("runner fault dispatching an op".to_string());
        }
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
    /// point); **deferred device errors surface here** (§3.3) — a latched fault is taken before
    /// any read.
    ///
    /// # Errors
    ///
    /// [`ComputeError::Decode`] (bad `TensorIr`), [`ComputeError::StaleHandle`] /
    /// [`ComputeError::InvalidHandle`] (the tensor is not live), or [`ComputeError::Device`] (a
    /// deferred device execution error — latched or raised by the read itself).
    pub fn read_tensor(&mut self, ir_cbor: &[u8]) -> Result<Vec<u8>, ComputeError> {
        let ir: TensorIr =
            ciborium::from_reader(ir_cbor).map_err(|e| ComputeError::Decode(e.to_string()))?;
        let id = ir.id.value();
        if !self.live.contains(&id) {
            return Err(self.absent_handle_fault(id));
        }
        self.take_device_fault()?;
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

    /// `compute@2::fence` — drain the command queue to a consistent point (architecture §3.3):
    /// **deferred device errors surface here** (a latched fault is taken first, then the device
    /// sync). For the synchronous tier-1 backend the sync is immediate; on wgpu/cuda it awaits
    /// the device. The driver delivers `Event::Fence(id)` only after a successful fence.
    ///
    /// # Errors
    ///
    /// [`ComputeError::Device`] when the device reports a deferred execution error.
    pub fn fence(&mut self) -> Result<(), ComputeError> {
        self.take_device_fault()?;
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

// ==== the execution-backend selection layer =====================================================
//
// The driver constructs ONE of these per run instance from `EngineConfig.backend` — the seam
// that replaces the former unconditional ndarray construction. Three invariants live here:
//
// 1. **No silent fallback.** A selected backend whose feature is compiled but whose device is
//    unavailable at construction is a typed error the caller surfaces
//    (`RunError::BackendUnavailable` pre-spawn; a `ComputeFault` trap on the guest thread) —
//    never a quiet CPU run.
// 2. **Thread affinity.** cubecl-cuda derives its stream + memory-pool registry from the
//    CALLING thread (`StreamId::current()` is a thread-local) and backends are `Send` but not
//    `Sync`; driving one from multiple OS threads silently splits pool bookkeeping and dies
//    under memory pressure. The driver constructs this runner ON the per-instance guest thread
//    and every `compute@2` import is a synchronous host call on that same thread, so affinity
//    holds by construction; the recorded [`std::thread::ThreadId`] turns any future violation
//    into a loud debug assertion instead of a latent pool split.
// 3. **One device-compute instance per process.** cubecl memory pools never shrink — the peak
//    working set is permanent for the process — so a second concurrent device-backed instance
//    would compete for a pool sized by the first. The process-wide slot refuses a second LIVE
//    device instance typed; sequential instances are permitted (the slot releases on drop) with
//    the caveat that reclamation is only real after a process restart, which the node's
//    worker-respawn discipline owns.

/// Whether the selected backend can run on this host right now: the runtime-probe half of the
/// selection ladder (feature-compiled is necessary but not sufficient). CPU/ndarray lanes are
/// always available; the device lanes require a usable adapter/device (and staged NVRTC for
/// CUDA — the two-leg readiness gate). The probes are memoized process-wide, so this is cheap
/// on every call.
///
/// # Errors
///
/// A human-readable reason the backend cannot serve (the `BackendUnavailable` detail).
pub fn backend_available(kind: crate::runtime::BackendKind) -> Result<(), String> {
    match kind {
        crate::runtime::BackendKind::Cpu => Ok(()),
        #[cfg(feature = "burn-ndarray")]
        crate::runtime::BackendKind::BurnNdarray => Ok(()),
        #[cfg(feature = "wgpu")]
        crate::runtime::BackendKind::Wgpu => {
            if crate::probe::probe_wgpu().is_some() {
                Ok(())
            } else {
                Err("no usable wgpu adapter on this host".to_string())
            }
        }
        #[cfg(feature = "cuda")]
        crate::runtime::BackendKind::Cuda => {
            if crate::probe::probe_cuda().is_none() {
                return Err("no usable CUDA device on this host".to_string());
            }
            if !crate::probe::cuda_nvrtc_ready() {
                return Err(
                    "CUDA device present but the NVRTC runtime is not staged (two-leg gate: \
                     loadable libnvrtc AND the cudart JIT include tree)"
                        .to_string(),
                );
            }
            Ok(())
        }
    }
}

/// The process-wide device-compute slot: at most ONE live device-backed (wgpu/cuda) compute
/// instance per process. CPU/ndarray instances are unbounded (tests run many concurrently).
static DEVICE_COMPUTE_SLOT: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Holding this guard IS the device-compute slot; dropping it releases the slot.
#[derive(Debug)]
pub struct DeviceComputeGuard(());

impl DeviceComputeGuard {
    /// Acquire the process's single device-compute slot, or report it occupied.
    ///
    /// # Errors
    ///
    /// The slot is held by a live device-backed compute instance in this process.
    pub fn acquire() -> Result<Self, String> {
        use std::sync::atomic::Ordering;
        if DEVICE_COMPUTE_SLOT
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            Ok(Self(()))
        } else {
            Err(
                "a device-backed compute instance is already live in this process (one \
                 device-compute instance per process; device memory pools never shrink)"
                    .to_string(),
            )
        }
    }

    /// Whether the slot is currently held (the cheap pre-spawn peek; the acquire on the guest
    /// thread stays authoritative).
    #[must_use]
    pub fn is_held() -> bool {
        DEVICE_COMPUTE_SLOT.load(std::sync::atomic::Ordering::Acquire)
    }
}

impl Drop for DeviceComputeGuard {
    fn drop(&mut self) {
        DEVICE_COMPUTE_SLOT.store(false, std::sync::atomic::Ordering::Release);
    }
}

/// The per-backend runner arm behind [`HostCompute`]. Feature-gated exactly like
/// [`crate::runtime::BackendKind`]; dispatch is a plain match, monomorphic per arm.
enum SelectedRunner {
    /// The ndarray CPU arm — serves both `BackendKind::Cpu` and `BackendKind::BurnNdarray`
    /// (one real implementation: the burn-ndarray backend).
    Ndarray(ComputeRunner<HostReal>),
    #[cfg(feature = "wgpu")]
    Wgpu(ComputeRunner<burn::backend::Wgpu>),
    #[cfg(feature = "cuda")]
    Cuda(ComputeRunner<burn::backend::Cuda>),
}

/// The driver-facing `compute@2` runner for one run instance: the [`ComputeRunner`] surface
/// over the backend `EngineConfig.backend` selected, plus the thread-affinity record and (for
/// device arms) the per-process device-compute slot.
pub struct HostCompute {
    /// The guest thread that constructed (and exclusively drives) this runner — the pinned
    /// device thread. See the module invariants above.
    thread: std::thread::ThreadId,
    runner: SelectedRunner,
    /// Held for the device arms; releases with the instance.
    _device_slot: Option<DeviceComputeGuard>,
}

impl HostCompute {
    /// Construct the selected backend's runner ON THE CALLING THREAD (the driver calls this on
    /// the per-instance guest thread — the pinned device thread). Device arms acquire the
    /// process device-compute slot and bring the device up inside `catch_unwind` (cubecl panics
    /// on a missing adapter; that panic must surface typed, never abort the host).
    ///
    /// # Errors
    ///
    /// A human-readable reason (unavailable device, occupied device slot, bring-up panic) —
    /// the caller maps it to the typed refusal surface.
    pub fn build(cfg: &crate::runtime::EngineConfig) -> Result<Self, String> {
        backend_available(cfg.backend)?;
        let (runner, slot) = match cfg.backend {
            crate::runtime::BackendKind::Cpu => {
                (SelectedRunner::Ndarray(ComputeRunner::ndarray_cpu()), None)
            }
            #[cfg(feature = "burn-ndarray")]
            crate::runtime::BackendKind::BurnNdarray => {
                (SelectedRunner::Ndarray(ComputeRunner::ndarray_cpu()), None)
            }
            #[cfg(feature = "wgpu")]
            crate::runtime::BackendKind::Wgpu => {
                let slot = DeviceComputeGuard::acquire()?;
                let device = match cfg.gpu_index {
                    Some(i) => {
                        let d = burn::backend::wgpu::WgpuDevice::DiscreteGpu(i as usize);
                        // The probe registered only `DefaultDevice` under the selected graphics
                        // API (Dx12 on Windows); a node-directed discrete placement must register
                        // THAT device under the same API too, else the router's lazy bring-up
                        // falls back to cubecl's `AutoGraphicsApi` (Vulkan on Windows). Idempotent
                        // + panic-safe. (The `None`/default path is already brought up by the
                        // mandatory `backend_available` → `probe_wgpu` that precedes this.)
                        crate::probe::ensure_wgpu_registered(&d);
                        d
                    }
                    None => burn::backend::wgpu::WgpuDevice::DefaultDevice,
                };
                let runner = catch_bringup(|| ComputeRunner::<burn::backend::Wgpu>::new(device))
                    .map_err(|e| format!("wgpu device bring-up: {e}"))?;
                (SelectedRunner::Wgpu(runner), Some(slot))
            }
            #[cfg(feature = "cuda")]
            crate::runtime::BackendKind::Cuda => {
                let slot = DeviceComputeGuard::acquire()?;
                let device =
                    burn::backend::cuda::CudaDevice::new(cfg.gpu_index.unwrap_or(0) as usize);
                let runner = catch_bringup(|| ComputeRunner::<burn::backend::Cuda>::new(device))
                    .map_err(|e| format!("CUDA device bring-up: {e}"))?;
                (SelectedRunner::Cuda(runner), Some(slot))
            }
        };
        Ok(Self {
            thread: std::thread::current().id(),
            runner,
            _device_slot: slot,
        })
    }

    /// The pinned-thread check (invariant 2 above): every call must arrive on the constructing
    /// guest thread. A violation is a host programming error — loud in debug builds, where the
    /// whole test matrix runs; never a user-facing failure mode.
    #[inline]
    fn check_thread(&self) {
        debug_assert_eq!(
            std::thread::current().id(),
            self.thread,
            "HostCompute driven off its pinned guest thread (backend thread-affinity violation)"
        );
    }

    /// See [`ComputeRunner::submit_op`].
    ///
    /// # Errors
    ///
    /// As [`ComputeRunner::submit_op`].
    pub fn submit_op(&mut self, op_cbor: &[u8]) -> Result<(), ComputeError> {
        self.check_thread();
        match &mut self.runner {
            SelectedRunner::Ndarray(r) => r.submit_op(op_cbor),
            #[cfg(feature = "wgpu")]
            SelectedRunner::Wgpu(r) => r.submit_op(op_cbor),
            #[cfg(feature = "cuda")]
            SelectedRunner::Cuda(r) => r.submit_op(op_cbor),
        }
    }

    /// See [`ComputeRunner::fence`]. Backend-specific deferred-error caveat (ABI §15 hardware
    /// findings): a successful fence does NOT prove device health — validated on real CUDA
    /// hardware, faults may surface only at readback — so callers treat readback as the
    /// authoritative fault surface and this fence as a best-effort early drain.
    ///
    /// # Errors
    ///
    /// As [`ComputeRunner::fence`].
    pub fn fence(&mut self) -> Result<(), ComputeError> {
        self.check_thread();
        match &mut self.runner {
            SelectedRunner::Ndarray(r) => r.fence(),
            #[cfg(feature = "wgpu")]
            SelectedRunner::Wgpu(r) => r.fence(),
            #[cfg(feature = "cuda")]
            SelectedRunner::Cuda(r) => r.fence(),
        }
    }

    /// See [`ComputeRunner::read_tensor`] — the authoritative deferred-fault surface.
    ///
    /// # Errors
    ///
    /// As [`ComputeRunner::read_tensor`].
    pub fn read_tensor(&mut self, ir_cbor: &[u8]) -> Result<Vec<u8>, ComputeError> {
        self.check_thread();
        match &mut self.runner {
            SelectedRunner::Ndarray(r) => r.read_tensor(ir_cbor),
            #[cfg(feature = "wgpu")]
            SelectedRunner::Wgpu(r) => r.read_tensor(ir_cbor),
            #[cfg(feature = "cuda")]
            SelectedRunner::Cuda(r) => r.read_tensor(ir_cbor),
        }
    }

    /// See [`ComputeRunner::import_tensor`].
    ///
    /// # Errors
    ///
    /// As [`ComputeRunner::import_tensor`].
    pub fn import_tensor(&mut self, id: u64, data_cbor: &[u8]) -> Result<(), ComputeError> {
        self.check_thread();
        match &mut self.runner {
            SelectedRunner::Ndarray(r) => r.import_tensor(id, data_cbor),
            #[cfg(feature = "wgpu")]
            SelectedRunner::Wgpu(r) => r.import_tensor(id, data_cbor),
            #[cfg(feature = "cuda")]
            SelectedRunner::Cuda(r) => r.import_tensor(id, data_cbor),
        }
    }

    /// See [`ComputeRunner::seed`].
    pub fn seed(&self, seed: u64) {
        self.check_thread();
        match &self.runner {
            SelectedRunner::Ndarray(r) => r.seed(seed),
            #[cfg(feature = "wgpu")]
            SelectedRunner::Wgpu(r) => r.seed(seed),
            #[cfg(feature = "cuda")]
            SelectedRunner::Cuda(r) => r.seed(seed),
        }
    }

    /// See [`ComputeRunner::inject_device_fault`] (the deferred-error test seam).
    pub fn inject_device_fault(&mut self, reason: impl Into<String>) {
        self.check_thread();
        match &mut self.runner {
            SelectedRunner::Ndarray(r) => r.inject_device_fault(reason),
            #[cfg(feature = "wgpu")]
            SelectedRunner::Wgpu(r) => r.inject_device_fault(reason),
            #[cfg(feature = "cuda")]
            SelectedRunner::Cuda(r) => r.inject_device_fault(reason),
        }
    }
}

/// Run a device bring-up closure under `catch_unwind` with the panic hook silenced (the probe
/// idiom this crate already uses): cubecl panics on a missing/failed adapter, and that panic
/// must surface as a typed error on the guest thread — never a host abort.
#[cfg(any(feature = "wgpu", feature = "cuda"))]
fn catch_bringup<T>(f: impl FnOnce() -> T + std::panic::UnwindSafe) -> Result<T, String> {
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let res = std::panic::catch_unwind(f);
    std::panic::set_hook(prev);
    res.map_err(|payload| {
        payload
            .downcast_ref::<&str>()
            .map(|s| (*s).to_string())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "device bring-up panicked".to_string())
    })
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Serializes the tests that OBSERVE or HOLD the process-global device-compute slot: the
    /// parallel test harness otherwise races `is_held()` in one test against a live guard in
    /// another (a real, occasional det-lane failure — the slot is process state, not test state).
    static DEVICE_SLOT_TESTS: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// The CPU arm builds unconditionally, dispatches on the constructing thread, and holds no
    /// device slot (many CPU instances may coexist — the whole tier-1 test matrix relies on it).
    #[test]
    fn cpu_host_compute_builds_and_serves_without_a_device_slot() {
        let _slot_tests = DEVICE_SLOT_TESTS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let cfg = crate::runtime::EngineConfig::default();
        let mut a = HostCompute::build(&cfg).expect("cpu arm always available");
        let mut b = HostCompute::build(&cfg).expect("a second CPU instance is unbounded");
        assert!(
            !DeviceComputeGuard::is_held(),
            "CPU arms take no device slot"
        );

        // The runner surface round-trips through the selection layer (import → fence → read).
        let data = TensorData::new(vec![1.0f32, 2.0, 3.0], [3usize]);
        let mut cbor = Vec::new();
        ciborium::into_writer(&data, &mut cbor).expect("tensor data encodes");
        a.import_tensor(1, &cbor).expect("import");
        a.fence().expect("clean fence");
        let ir = TensorIr {
            id: TensorId::new(1),
            shape: burn_backend::Shape::from(vec![3usize]),
            status: TensorStatus::ReadOnly,
            dtype: burn_backend::DType::F32,
        };
        let mut ir_cbor = Vec::new();
        ciborium::into_writer(&ir, &mut ir_cbor).expect("ir encodes");
        let out = a.read_tensor(&ir_cbor).expect("readback");
        let round: TensorData = ciborium::from_reader(out.as_slice()).expect("decodes");
        assert_eq!(round.to_vec::<f32>().expect("f32"), vec![1.0f32, 2.0, 3.0]);

        // The deferred-error latch rides the selection layer: injected fault surfaces at the
        // next fence, typed, exactly once.
        b.inject_device_fault("selection-layer injected fault");
        let err = b.fence().expect_err("latched fault surfaces at the fence");
        assert!(matches!(err, ComputeError::Device(_)));
        b.fence().expect("the fault surfaced exactly once");
    }

    /// The process device-compute slot: one live holder; a second acquire refuses typed; the
    /// slot releases on drop. (One test owns the whole lifecycle — the slot is process-global,
    /// so splitting these assertions across tests would race the parallel harness.)
    #[test]
    fn device_compute_slot_is_single_holder_and_releases_on_drop() {
        let _slot_tests = DEVICE_SLOT_TESTS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let first = DeviceComputeGuard::acquire().expect("free slot acquires");
        assert!(DeviceComputeGuard::is_held());
        let second = DeviceComputeGuard::acquire();
        assert!(second.is_err(), "a second live device instance must refuse");
        drop(first);
        assert!(!DeviceComputeGuard::is_held());
        let again = DeviceComputeGuard::acquire().expect("released slot re-acquires");
        drop(again);
    }

    /// The CPU/ndarray rungs are always available (the availability probe is the runtime half
    /// of the selection ladder; the device rungs are exercised on hardware lanes).
    #[test]
    fn cpu_rungs_are_always_available() {
        assert!(backend_available(crate::runtime::BackendKind::Cpu).is_ok());
        #[cfg(feature = "burn-ndarray")]
        assert!(backend_available(crate::runtime::BackendKind::BurnNdarray).is_ok());
    }
}
