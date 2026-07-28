// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `compute@2` conformance suite (Phase C, track C1; ABI §15 / decisions D8).
//!
//! The normative Phase-C acceptance: **the pinned op set round-trips** — IR encode → host decode →
//! `burn-router` runner dispatch → result — plus the RESERVED-variant refusals and the typed
//! stale/invalid handle faults. This suite productionizes the Burn-over-`HostBackend` spike's
//! tier-1 gate against the real [`daemon_vhc_host::ComputeRunner`].
//!
//! The `HostBackend` here (`BackendRouter<ConformanceChannel>`) is the guest-shaped backend: every
//! tensor op lowers to a `CBOR(burn_ir::OperationIr)` op-blob and is dispatched, through the exact
//! `ComputeRunner` codec + dispatch the wasm import shim uses, onto a host-side `Runner<NdArray>`.
//! Bit-exactness against a native `NdArray` run proves the indirection is semantics-preserving —
//! the strongest possible pass (both sides run identical ndarray kernels). It doubles as evidence
//! for the guest-side `HostBackend` + autodiff-tape shape (deliverable 5): `Autodiff<HostBackend>`
//! is an ordinary Burn backend over the handle boundary.

use std::sync::{Arc, Mutex};

use burn::backend::Autodiff;
use burn::tensor::{Tensor, TensorData};
use burn_backend::TensorData as BackendTensorData;
use burn_backend::{DType, DTypeUsage, DTypeUsageSet, ExecutionError, Shape};
use burn_ir::{CustomOpIr, OperationIr, TensorId, TensorIr, TensorStatus};
use burn_ndarray::NdArrayDevice;
use burn_router::{BackendRouter, MultiBackendBridge, RouterTensor, RunnerChannel, RunnerClient};
use burn_std::future::DynFut;

use daemon_vhc_host::compute::{ComputeError, HostReal};
use daemon_vhc_host::ComputeRunner;

/// The tier-1 real backend behind the boundary (matches [`HostReal`]).
type Real = HostReal;
/// The guest-facing backend: every op is a CBOR op-blob dispatched through [`ComputeRunner`].
type HostBackend = BackendRouter<ConformanceChannel>;

fn ser<T: serde::Serialize>(v: &T) -> Vec<u8> {
    let mut buf = Vec::new();
    ciborium::into_writer(v, &mut buf).expect("IR/TensorData must encode");
    buf
}

/// The boundary: routes every op through the production `ComputeRunner` (the codec + dispatch the
/// wasm shim uses), behind a `Mutex` (the runner's `submit_op` is `&mut self`).
#[derive(Clone)]
struct Boundary {
    runner: Arc<Mutex<ComputeRunner<Real>>>,
}

struct AbiClientInner {
    boundary: Boundary,
    device: NdArrayDevice,
    next_id: std::sync::atomic::AtomicU64,
}

#[derive(Clone)]
struct AbiClient {
    inner: Arc<AbiClientInner>,
}

impl AbiClient {
    fn mint(&self) -> u64 {
        self.inner
            .next_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }
}

impl RunnerClient for AbiClient {
    type Device = NdArrayDevice;

    fn register_op(&self, op: OperationIr) {
        // The guest enqueues; the production runner decodes + dispatches. A servable model never
        // emits a RESERVED op, so a fault here is a genuine test failure.
        self.inner
            .boundary
            .runner
            .lock()
            .unwrap()
            .submit_op(&ser(&op))
            .expect("servable op dispatches");
    }

    fn read_tensor_async(
        &self,
        tensor: TensorIr,
    ) -> DynFut<Result<BackendTensorData, ExecutionError>> {
        let bytes = ser(&tensor);
        let out = match self
            .inner
            .boundary
            .runner
            .lock()
            .unwrap()
            .read_tensor(&bytes)
        {
            Ok(data_bytes) => {
                Ok(ciborium::from_reader(&data_bytes[..]).expect("TensorData decodes"))
            }
            Err(e) => Err(ExecutionError::WithContext {
                reason: e.to_string(),
            }),
        };
        Box::pin(core::future::ready(out))
    }

    fn sync(&self) -> Result<(), ExecutionError> {
        self.inner
            .boundary
            .runner
            .lock()
            .unwrap()
            .fence()
            .map_err(|e| ExecutionError::WithContext {
                reason: e.to_string(),
            })
    }

    fn create_empty_handle(&self) -> TensorId {
        TensorId::new(self.mint())
    }

    fn register_tensor_data(&self, data: BackendTensorData) -> RouterTensor<Self> {
        let id = self.mint();
        let shape = data.shape.clone();
        let dtype = data.dtype;
        self.inner
            .boundary
            .runner
            .lock()
            .unwrap()
            .import_tensor(id, &ser(&data))
            .expect("tensor data imports");
        RouterTensor::new(TensorId::new(id), shape, dtype, self.clone())
    }

    fn device(&self) -> Self::Device {
        self.inner.device
    }

    fn seed(&self, seed: u64) {
        self.inner.boundary.runner.lock().unwrap().seed(seed);
    }

    fn dtype_usage(&self, _dtype: DType) -> DTypeUsageSet {
        DTypeUsage::general()
    }
}

struct NoBridge;

impl MultiBackendBridge for NoBridge {
    type TensorHandle = ();
    type Device = NdArrayDevice;
    fn change_backend_float(_: (), _: Shape, _: &NdArrayDevice) {
        unreachable!("single-device conformance never transfers backends")
    }
    fn change_backend_int(_: (), _: Shape, _: &NdArrayDevice) {
        unreachable!("single-device conformance never transfers backends")
    }
    fn change_backend_bool(_: (), _: Shape, _: &NdArrayDevice) {
        unreachable!("single-device conformance never transfers backends")
    }
}

#[derive(Clone)]
struct ConformanceChannel;

impl RunnerChannel for ConformanceChannel {
    type Device = NdArrayDevice;
    type Bridge = NoBridge;
    type Client = AbiClient;
    type FloatElem = f32;
    // Mirrors the guest SDK channel (`daemon-vhc-sdk-compute`): i32 indices (i64 kernels are
    // DXC-only on DX12) and u32 bool storage — WGSL forbids native bool in the storage address
    // space (cubecl#1274).
    type IntElem = i32;
    type BoolElem = u32;

    fn name(_device: &NdArrayDevice) -> String {
        "compute@2-conformance(ndarray)".to_string()
    }

    fn init_client(device: &NdArrayDevice) -> AbiClient {
        AbiClient {
            inner: Arc::new(AbiClientInner {
                boundary: Boundary {
                    runner: Arc::new(Mutex::new(ComputeRunner::ndarray_cpu())),
                },
                device: *device,
                next_id: std::sync::atomic::AtomicU64::new(1),
            }),
        }
    }

    fn get_tensor_handle(_tensor: &TensorIr, _client: &AbiClient) {
        unreachable!("single-device conformance never extracts a cross-backend handle")
    }

    fn register_tensor(
        _client: &AbiClient,
        _handle: (),
        _shape: Shape,
        _dtype: DType,
    ) -> RouterTensor<AbiClient> {
        unreachable!("single-device conformance never registers a cross-backend handle")
    }
}

/// The fixed inputs both backends run, so the comparison is a bit-exact equality (not tolerance).
fn inputs() -> (Vec<f32>, [usize; 2], Vec<f32>, [usize; 2]) {
    let a = vec![0.5, -1.0, 2.0, 3.0, -0.25, 1.5]; // [2,3]
    let w = vec![1.0, 0.0, -1.0, 2.0, 0.5, -0.5]; // [3,2]
    (a, [2, 3], w, [3, 2])
}

/// A small but multi-op forward: matmul → +scalar → relu → sum-to-scalar. Exercises Init, matmul,
/// AddScalar, Relu (via clamp/mask), and a reduction round-trip — the readback is the sum scalar.
fn forward<B: burn::tensor::backend::Backend>(device: &B::Device) -> f32
where
    B::FloatElem: Into<f32>,
{
    let (a, a_shape, w, w_shape) = inputs();
    let a = Tensor::<B, 2>::from_data(TensorData::new(a, a_shape), device);
    let w = Tensor::<B, 2>::from_data(TensorData::new(w, w_shape), device);
    let y = burn::tensor::activation::relu(a.matmul(w).add_scalar(0.75_f32));
    let s = y.sum();
    s.into_scalar().into()
}

#[test]
fn pinned_op_set_round_trips_bit_exact_against_native() {
    // The conformance gate: every op the forward emits round-trips (IR encode → host decode →
    // runner dispatch → readback) and reproduces the native ndarray result BIT-EXACTLY, because
    // both sides run identical kernels — only the compute@2 path routes each op through CBOR +
    // the ComputeRunner (ABI §15).
    let host = forward::<HostBackend>(&NdArrayDevice::Cpu);
    let native = forward::<Real>(&NdArrayDevice::Cpu);
    assert_eq!(
        host.to_bits(),
        native.to_bits(),
        "compute@2 round-trip must be bit-exact (host {host} vs native {native})"
    );
}

#[test]
fn autodiff_tape_is_guest_side_over_the_handle_boundary() {
    // Deliverable 5 evidence: Autodiff<HostBackend> is an ordinary Burn backend over the handle
    // boundary — the tape is guest-side bookkeeping, the backward pass enqueues ops on handles, and
    // the gradient reads back bit-exactly against native Autodiff<NdArray>. (No autodiff-specific
    // ABI import — backward@1/grad@1 retire under compute@2.)
    fn grad_sum<B: burn::tensor::backend::AutodiffBackend>(device: &B::Device) -> f32
    where
        B::FloatElem: Into<f32>,
    {
        let (a, a_shape, w, w_shape) = inputs();
        let a = Tensor::<B, 2>::from_data(TensorData::new(a, a_shape), device).require_grad();
        let w = Tensor::<B, 2>::from_data(TensorData::new(w, w_shape), device);
        let loss = burn::tensor::activation::relu(a.clone().matmul(w).add_scalar(0.75_f32)).sum();
        let grads = loss.backward();
        let ga = a.grad(&grads).expect("grad exists");
        ga.sum().into_scalar().into()
    }
    let host = grad_sum::<Autodiff<HostBackend>>(&NdArrayDevice::Cpu);
    let native = grad_sum::<Autodiff<Real>>(&NdArrayDevice::Cpu);
    assert_eq!(
        host.to_bits(),
        native.to_bits(),
        "backward must be bit-exact"
    );
}

#[test]
fn custom_op_ir_refuses_with_the_c2_vocabulary_code() {
    // ABI §15: OperationIr::Custom does not dispatch through the generic IR wire — named custom
    // ops are C2's host-side registry (v2::CustomOpRegistry) and actual Custom-IR dispatch stays
    // deferred. The refusal is clean and typed, carrying C2's `CustomOpUnsupported` vocabulary
    // code — never the panic the upstream runner produces. (The QFloat arms —
    // Float(Quantize)/Float(Dequantize) — are the identical match mechanism in unservable_op;
    // both stay RESERVED until specified.)
    let mut runner = ComputeRunner::ndarray_cpu();
    let custom = OperationIr::Custom(CustomOpIr {
        id: "flash_attn@1".to_string(),
        inputs: vec![],
        outputs: vec![],
    });
    let mut buf = Vec::new();
    ciborium::into_writer(&custom, &mut buf).unwrap();
    let err = runner.submit_op(&buf).unwrap_err();
    match &err {
        ComputeError::CustomOpUnsupported(name) => assert_eq!(name, "flash_attn@1"),
        other => panic!("expected CustomOpUnsupported, got {other:?}"),
    }
    assert_eq!(
        err.refusal_code(),
        Some(daemon_vhc_abi::AbiRefusalCode::CustomOpUnsupported),
        "the runtime refusal speaks the C2 admission vocabulary"
    );
    assert_eq!(err.trap_code().slug(), "BadEnum");
    assert!(err.to_string().contains("CustomOpUnsupported"));
}

#[test]
fn injected_device_fault_defers_to_fence_and_readback() {
    // Deferred-error semantics (architecture §3.3, the ndarray-tier injectable-fault seam): a
    // device fault parked at enqueue time NEVER surfaces at submit_op — the next fence (or
    // readback) reports it, exactly once, as a typed Device error (trap twin: ComputeFault;
    // completion twin: COMP_ERR_DEVICE).
    let mut runner = ComputeRunner::ndarray_cpu();
    let data = BackendTensorData::new(vec![1.0f32, 2.0], [2usize]);
    runner.import_tensor(7, &ser(&data)).unwrap();
    runner.fence().expect("clean fence before the fault");

    runner.inject_device_fault("simulated async device fault");
    // Enqueue against the poisoned queue still succeeds (enqueue is infallible, §3.3).
    let drop_op = OperationIr::Drop(TensorIr {
        id: TensorId::new(7),
        shape: Shape::from(vec![2usize]),
        status: TensorStatus::ReadWrite,
        dtype: DType::F32,
    });
    runner
        .submit_op(&ser(&drop_op))
        .expect("enqueue infallible");
    // The fault surfaces at the fence — once.
    let err = runner.fence().unwrap_err();
    assert!(
        matches!(&err, ComputeError::Device(reason) if reason.contains("simulated")),
        "got {err:?}"
    );
    assert_eq!(err.trap_code().slug(), "ComputeFault");
    runner.fence().expect("the fault surfaced exactly once");

    // And the readback path surfaces a latched fault the same way.
    let mut runner = ComputeRunner::ndarray_cpu();
    runner.import_tensor(9, &ser(&data)).unwrap();
    runner.inject_device_fault("fault before readback");
    let ir = TensorIr {
        id: TensorId::new(9),
        shape: Shape::from(vec![2usize]),
        status: TensorStatus::ReadOnly,
        dtype: DType::F32,
    };
    let err = runner.read_tensor(&ser(&ir)).unwrap_err();
    assert!(matches!(err, ComputeError::Device(_)), "got {err:?}");
    runner
        .read_tensor(&ser(&ir))
        .expect("readback clean after the fault surfaced");
}

#[test]
fn malformed_op_blob_fails_closed() {
    // A non-decodable op-blob is a clean typed Decode refusal (BadEvent), never a panic (ABI §5.2
    // fail-closed). This also covers the feature-gated `Distributed` variant, undecodable here.
    let mut runner = ComputeRunner::ndarray_cpu();
    let err = runner.submit_op(&[0xff, 0x00, 0x13, 0x37]).unwrap_err();
    assert!(matches!(err, ComputeError::Decode(_)), "got {err:?}");
    assert_eq!(err.trap_code().slug(), "BadEvent");
}

#[test]
fn unknown_and_stale_tensor_handles_are_typed_faults() {
    // The Phase-C obligation (ABI §15): a stale/unknown handle is a typed StaleHandle/InvalidHandle
    // trap, NEVER the host crash the upstream runner would produce.
    let mut runner = ComputeRunner::ndarray_cpu();
    let ir = TensorIr {
        id: TensorId::new(9_999_999),
        shape: Shape::from(vec![2usize, 2usize]),
        status: TensorStatus::ReadOnly,
        dtype: DType::F32,
    };
    let mut buf = Vec::new();
    ciborium::into_writer(&ir, &mut buf).unwrap();
    // Never registered → InvalidHandle.
    match runner.read_tensor(&buf).unwrap_err() {
        ComputeError::InvalidHandle(id) => assert_eq!(id, 9_999_999),
        other => panic!("expected InvalidHandle, got {other:?}"),
    }

    // Register a tensor, drop it, then read it back → StaleHandle (was valid, now gone).
    let mut runner = ComputeRunner::ndarray_cpu();
    let data = BackendTensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [2usize, 2usize]);
    runner.import_tensor(42, &ser(&data)).unwrap();
    assert!(runner.is_live(42));
    let drop_op = OperationIr::Drop(TensorIr {
        id: TensorId::new(42),
        shape: Shape::from(vec![2usize, 2usize]),
        status: TensorStatus::ReadWrite,
        dtype: DType::F32,
    });
    runner.submit_op(&ser(&drop_op)).unwrap();
    assert!(!runner.is_live(42));
    let read_ir = TensorIr {
        id: TensorId::new(42),
        shape: Shape::from(vec![2usize, 2usize]),
        status: TensorStatus::ReadOnly,
        dtype: DType::F32,
    };
    match runner.read_tensor(&ser(&read_ir)).unwrap_err() {
        ComputeError::StaleHandle(id) => assert_eq!(id, 42),
        other => panic!("expected StaleHandle after Drop, got {other:?}"),
    }
    assert_eq!(
        ComputeError::StaleHandle(42).trap_code().slug(),
        "StaleHandle"
    );
    assert_eq!(
        ComputeError::InvalidHandle(1).trap_code().slug(),
        "InvalidHandle"
    );
}
