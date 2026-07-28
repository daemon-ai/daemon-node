// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-vhc-sdk-compute` — the guest-side Burn compute tier (Phase C, track C1).
//!
//! **`HostBackend` is an ordinary Burn backend** (`BackendRouter` over the `compute@2` import
//! shim): module authors write ordinary Burn models wrapped in [`AutodiffHostBackend`]
//! (`Autodiff<HostBackend>`), and every tensor op lowers to a `CBOR(burn_ir::OperationIr)` blob
//! enqueued through `compute@2::submit_op` — opaque `u64` `TensorId` handles, guest-cached
//! shape/dtype metadata, guest reference-counting with `OperationIr::Drop` on release
//! (architecture §3.2; decisions D8). The autodiff tape (graph + backward closures) is **pure
//! guest-side bookkeeping over handles**: a full backward pass enqueues ops with zero
//! intermediate readbacks (the spike-proven claim) — no autodiff-specific ABI import exists.
//!
//! ## Reading results back — the event-loop discipline
//!
//! There is **no synchronous in-guest readback**: `Tensor::into_data()`/`into_scalar()` (Burn's
//! sync readback path) is unsupported over `HostBackend` and panics with a clear message —
//! extraction is the explicit, budgeted, journaled `compute@2` path (architecture §3.2/§3.4):
//!
//! 1. [`export_tensor`] (device → sealed buffer): returns an `OpId`; the completion
//!    (`Event::Completion(op, Ok(BufferHandle))`) carries the tensor's `CBOR(TensorData)`;
//! 2. `read_buffer(handle)` (sdk-v2) crosses the bytes into linear memory (budgeted);
//! 3. [`decode_tensor_data`] decodes them.
//!
//! Fences ([`fence`], `Event::Fence`) are the compute-queue consistency points (§3.3): enqueue a
//! window of steps, return to the event loop, read back at the fence.

#![cfg(target_arch = "wasm32")]

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use burn_backend::{DType, DTypeUsage, DTypeUsageSet, ExecutionError, Shape, TensorData};
use burn_ir::{OperationIr, TensorId, TensorIr};
use burn_ndarray::NdArrayDevice;
use burn_router::{BackendRouter, MultiBackendBridge, RouterTensor, RunnerChannel, RunnerClient};
use burn_std::future::DynFut;

/// The guest-facing compute backend: an ordinary `burn` `Backend` whose ops cross the `compute@2`
/// boundary as CBOR op-blobs on opaque `u64` handles.
pub type HostBackend = BackendRouter<ComputeChannel>;

/// `Autodiff<HostBackend>` — the guest-side tape over the same enqueue/handle primitives
/// (decisions D8: `backward@1`/`grad@1` retire under `compute@2`).
pub type AutodiffHostBackend = burn::backend::Autodiff<HostBackend>;

/// The device token guests pass to Burn APIs. `compute@2` has exactly one device per
/// role-instance (the accelerator the instance is bound to, decisions D6 "one accelerator per
/// role-instance"), so this is an opaque single-value token — `NdArrayDevice::Cpu` reused purely
/// for its `DeviceOps` impl; **no ndarray compute ever runs guest-side**.
pub type ComputeDevice = NdArrayDevice;

/// The default (only) device.
#[must_use]
pub fn device() -> ComputeDevice {
    NdArrayDevice::Cpu
}

fn ser<T: serde::Serialize>(v: &T) -> Vec<u8> {
    let mut buf = Vec::new();
    ciborium::into_writer(v, &mut buf).expect("burn-ir values encode as CBOR");
    buf
}

struct ClientInner {
    /// The guest-owned `TensorId` mint: the guest is the single source of truth for the handle
    /// space (no host round-trip to allocate, no shape round-trip to infer — metadata is
    /// guest-authoritative, decisions D8 rule 3).
    next_id: AtomicU64,
}

/// The guest-side `RunnerClient`: marshals every op across the `compute@2` boundary.
#[derive(Clone)]
pub struct ComputeClient {
    inner: Arc<ClientInner>,
}

impl ComputeClient {
    fn mint(&self) -> u64 {
        self.inner.next_id.fetch_add(1, Ordering::Relaxed)
    }
}

impl RunnerClient for ComputeClient {
    type Device = ComputeDevice;

    fn register_op(&self, op: OperationIr) {
        daemon_vhc_sdk::compute_submit_op(&ser(&op));
    }

    fn read_tensor_async(&self, _tensor: TensorIr) -> DynFut<Result<TensorData, ExecutionError>> {
        // No synchronous in-guest readback exists (module docs): extraction is the explicit
        // export → Completion(BufferHandle) → read_buffer path. Burn's sync into_data() would
        // block the event loop inside an op — fail loudly instead (a GuestPanic trap).
        panic!(
            "HostBackend has no in-guest readback: use daemon_vhc_sdk_compute::export_tensor + \
             the Event::Completion(BufferHandle) path (architecture §3.2/§3.4)"
        );
    }

    fn sync(&self) -> Result<(), ExecutionError> {
        // The compute@2 consistency primitive is the explicit, event-delivered fence (§3.3);
        // Burn's implicit sync is a no-op over the command queue (enqueue order is preserved
        // host-side).
        Ok(())
    }

    fn create_empty_handle(&self) -> TensorId {
        TensorId::new(self.mint())
    }

    fn register_tensor_data(&self, data: TensorData) -> RouterTensor<Self> {
        // Guest-authored tensor data crosses by sealed buffer (§3.4), never inline in the
        // op-stream: seal → import under a guest-minted id → release the guest's buffer hold.
        // Registration is synchronous host-side, so the tensor is usable immediately; the
        // import's Ok(()) completion is event-loop noise the module may ignore.
        let id = self.mint();
        let shape = data.shape.clone();
        let dtype = data.dtype;
        let bytes = ser(&data);
        let buffer = daemon_vhc_sdk::create_from(&bytes);
        let _op = daemon_vhc_sdk::compute_import(buffer, id);
        daemon_vhc_sdk::buffer_release(buffer);
        RouterTensor::new(TensorId::new(id), shape, dtype, self.clone())
    }

    fn device(&self) -> Self::Device {
        device()
    }

    fn seed(&self, _seed: u64) {
        // Host-side RNG is seeded by the host from the run's execution identity (deterministic
        // per incarnation); the guest has no seeding surface. Guest-side reproducibility comes
        // from sys@2::rng_seed-driven data, not device RNG.
    }

    fn dtype_usage(&self, _dtype: DType) -> DTypeUsageSet {
        DTypeUsage::general()
    }
}

/// Single-device: cross-backend transfer is unreachable (one accelerator per role-instance).
pub struct NoBridge;

impl MultiBackendBridge for NoBridge {
    type TensorHandle = ();
    type Device = ComputeDevice;

    fn change_backend_float(_: (), _: Shape, _: &ComputeDevice) {
        unreachable!("compute@2 is single-device per role-instance")
    }
    fn change_backend_int(_: (), _: Shape, _: &ComputeDevice) {
        unreachable!("compute@2 is single-device per role-instance")
    }
    fn change_backend_bool(_: (), _: Shape, _: &ComputeDevice) {
        unreachable!("compute@2 is single-device per role-instance")
    }
}

/// The channel binding [`HostBackend`] to the [`ComputeClient`].
#[derive(Clone)]
pub struct ComputeChannel;

impl RunnerChannel for ComputeChannel {
    type Device = ComputeDevice;
    type Bridge = NoBridge;
    type Client = ComputeClient;
    type FloatElem = f32;
    type IntElem = i64;
    // Bool tensors ride the `burn-ir` wire as u32 storage, never native bool: WGSL forbids `bool`
    // in the `storage` address space (not host-shareable), so a `DType::Bool(Native)` in the IR
    // makes every WGSL host lane (DX12, Metal) refuse the cmp/mask kernels at shader validation
    // (tracel-ai/cubecl#1274 — codegen will not paper over it; burn's own wgpu aliases use
    // u32/u8). u32 is legal on every wgpu lane and matches burn's `Wgpu` alias. The values are
    // 0/1 either way — no arithmetic changes.
    type BoolElem = u32;

    fn name(_device: &ComputeDevice) -> String {
        "compute@2(HostBackend)".to_string()
    }

    fn init_client(_device: &ComputeDevice) -> ComputeClient {
        ComputeClient {
            inner: Arc::new(ClientInner {
                next_id: AtomicU64::new(1),
            }),
        }
    }

    fn get_tensor_handle(_tensor: &TensorIr, _client: &ComputeClient) {
        unreachable!("compute@2 is single-device per role-instance")
    }

    fn register_tensor(
        _client: &ComputeClient,
        _handle: (),
        _shape: Shape,
        _dtype: DType,
    ) -> RouterTensor<ComputeClient> {
        unreachable!("compute@2 is single-device per role-instance")
    }
}

/// Insert a compute-queue fence marker (§3.3): `Event::Fence(fence_id)` delivers when the device
/// passes it; a deferred device error surfaces at this call as a typed `ComputeFault` trap.
pub fn fence(fence_id: u64) {
    daemon_vhc_sdk::compute_fence(fence_id);
}

/// Export a float tensor to a sealed buffer (device → staging, §3.4): returns the `OpId`; the
/// `Event::Completion(op, Ok(BufferHandle))` carries the tensor's `CBOR(TensorData)`. Consumes
/// the tensor (its handle refcount transfers to the export; the host tensor stays live until an
/// `OperationIr::Drop` retires it).
#[must_use]
pub fn export_tensor<const D: usize>(tensor: burn::tensor::Tensor<HostBackend, D>) -> u64 {
    let primitive = tensor.into_primitive().tensor();
    let ir = primitive.into_ir();
    daemon_vhc_sdk::compute_export(&ser(&ir))
}

/// Decode an exported tensor's `CBOR(TensorData)` bytes (from `read_buffer` over the completion's
/// `BufferHandle`).
#[must_use]
pub fn decode_tensor_data(bytes: &[u8]) -> TensorData {
    ciborium::from_reader(bytes).expect("exported TensorData decodes")
}

/// Build a float tensor from raw `f32` data guest-side (crosses by sealed buffer — see
/// [`ComputeClient::register_tensor_data`]).
#[must_use]
pub fn tensor_from_floats<const D: usize>(
    values: Vec<f32>,
    shape: [usize; D],
) -> burn::tensor::Tensor<HostBackend, D> {
    burn::tensor::Tensor::from_data(TensorData::new(values, shape), &device())
}

/// Import a sealed buffer's `CBOR(TensorData)` (e.g. a peer's exported tensor received over a
/// stream, §3.4) as a device float tensor: registers it under a guest-minted `TensorId` via
/// `compute@2::import` and wraps the handle as an ordinary Burn tensor. The caller keeps its
/// buffer hold (release it after this returns). `D` must match the encoded rank.
#[must_use]
pub fn import_buffer_as_tensor<const D: usize>(
    buffer: u64,
    data: &TensorData,
) -> burn::tensor::Tensor<HostBackend, D> {
    let client = burn_router::get_client::<ComputeChannel>(&device());
    let id = client.mint();
    let _op = daemon_vhc_sdk::compute_import(buffer, id);
    let router = RouterTensor::new(TensorId::new(id), data.shape.clone(), data.dtype, client);
    burn::tensor::Tensor::from_primitive(burn::tensor::TensorPrimitive::Float(router))
}

// -- the wasm32 `custom` getrandom backend (see Cargo.toml) ---------------------------------------

/// The `getrandom` custom-backend definition the Burn dependency tree links against on
/// wasm32-unknown-unknown (selected by the guests-workspace rustc shim's
/// `--cfg getrandom_backend="custom"`). **Deterministic by design**: a sandboxed module has no
/// ambient entropy (architecture §3.2 — randomness is the seeded `sys@2::rng_seed`), so this
/// fills a fixed byte pattern. Conforming modules never reach it (no `compute@2` RNG surface
/// exists; `Float/Random` ops execute host-side); it exists so the wasm links and so any stray
/// in-guest sampling stays replay-deterministic instead of trapping.
///
/// # Safety
///
/// Called only by `getrandom` with a valid `dest..dest+len` allocation (its documented contract).
#[no_mangle]
unsafe extern "Rust" fn __getrandom_v03_custom(
    dest: *mut u8,
    len: usize,
) -> Result<(), getrandom::Error> {
    // SAFETY (whole fn): getrandom guarantees `dest..dest+len` is valid for writes.
    for i in 0..len {
        dest.add(i).write(0xD5);
    }
    Ok(())
}
