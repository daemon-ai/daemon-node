// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// The Phase-C **compute-replay** tier (refactor §7: "compute replay … becomes a gated harness —
// the second of the three replay tiers"; architecture §3.6 "Compute replay (tolerance-equivalent):
// re-running the kernels behind a journal on other hardware reproduces the trajectory only within
// the native lane's tolerance class").
//
// The harness has two halves:
//   1. RECORD — an ordinary `Autodiff<HostBackend>` forward+backward is run over a *recording*
//      `burn-router` client that captures the exact `compute@2` op stream (ordered imports +
//      `CBOR(burn_ir::OperationIr)` blobs) into an `OpJournal`, plus the exported tensor's
//      `TensorIr`. This is the same op stream a real guest submits over `submit_op` — the journal
//      is the compute-plane analogue of the §8.7 event journal.
//   2. REPLAY — the journal is re-executed against the PRODUCTION `daemon_vhc_host::ComputeRunner`
//      over a chosen `burn_ir::BackendIr`. The runner is the exact codec + dispatch the wasm shim
//      uses; kernels ARE re-executed here (unlike §8.7 input replay, which replays recorded
//      results), on whatever backend the replay picks.
//
// Two tiers, exactly as the deliverable prescribes:
//   * ndarray↔ndarray DEGENERATE (tier-1, always run): the same journal re-executed on ndarray
//     twice is bit-exact (tolerance 0), and equals a native `Autodiff<NdArray>` run — so the
//     harness itself is always exercised even where no second backend exists.
//   * wgpu cross-backend (hardware-gated, `--features wgpu`): the ndarray journal re-executed on
//     wgpu reproduces the trajectory within the native lane's tolerance class (NOT bit-exact).
#![allow(clippy::disallowed_methods)]

use std::cell::RefCell;

use burn::backend::Autodiff;
use burn::tensor::{Tensor, TensorData};
use burn_backend::{
    DType, DTypeUsage, DTypeUsageSet, ExecutionError, Shape, TensorData as BackendTensorData,
};
use burn_ir::{BackendIr, OperationIr, TensorId, TensorIr};
use burn_ndarray::NdArrayDevice;
use burn_router::{BackendRouter, MultiBackendBridge, RouterTensor, RunnerChannel, RunnerClient};
use burn_std::future::DynFut;
use daemon_vhc_host::{ComputeRunner, HostReal};

// -- the op-journal (the compute-plane journal replayed across backends) ------------------------

/// One recorded compute-queue step, in submission order.
#[derive(Clone)]
enum ComputeStep {
    /// `compute@2::import` — guest-supplied `CBOR(TensorData)` registered under a guest-minted id.
    Import { id: u64, data: Vec<u8> },
    /// `compute@2::submit_op` — one `CBOR(burn_ir::OperationIr)` op-blob.
    Op(Vec<u8>),
}

// The recorder is thread-local: `burn-router`'s cached client for a device is a stateless handle
// that reads this, and every tensor op for one recording runs on one test thread, so parallel
// tests never cross-contaminate (each thread has its own recorder).
thread_local! {
    static REC: RefCell<(u64, Vec<ComputeStep>)> = const { RefCell::new((1, Vec::new())) };
}

fn rec_reset() {
    REC.with(|r| *r.borrow_mut() = (1, Vec::new()));
}
fn rec_mint() -> u64 {
    REC.with(|r| {
        let mut b = r.borrow_mut();
        let id = b.0;
        b.0 += 1;
        id
    })
}
fn rec_push(step: ComputeStep) {
    REC.with(|r| r.borrow_mut().1.push(step));
}
fn rec_take() -> Vec<ComputeStep> {
    REC.with(|r| std::mem::take(&mut r.borrow_mut().1))
}

fn ser<T: serde::Serialize>(v: &T) -> Vec<u8> {
    let mut buf = Vec::new();
    ciborium::into_writer(v, &mut buf).expect("IR/TensorData encodes");
    buf
}

// -- the recording burn-router client (captures the op stream, never executes) ------------------

#[derive(Clone)]
struct RecClient;

impl RunnerClient for RecClient {
    type Device = NdArrayDevice;

    fn register_op(&self, op: OperationIr) {
        rec_push(ComputeStep::Op(ser(&op)));
    }

    fn read_tensor_async(
        &self,
        _tensor: TensorIr,
    ) -> DynFut<Result<BackendTensorData, ExecutionError>> {
        // Recording only enqueues; the exported tensor is captured by its `TensorIr`, never read.
        panic!("the recording client never reads back — capture the TensorIr and replay instead");
    }

    fn sync(&self) -> Result<(), ExecutionError> {
        Ok(())
    }

    fn create_empty_handle(&self) -> TensorId {
        TensorId::new(rec_mint())
    }

    fn register_tensor_data(&self, data: BackendTensorData) -> RouterTensor<Self> {
        let id = rec_mint();
        let shape = data.shape.clone();
        let dtype = data.dtype;
        rec_push(ComputeStep::Import {
            id,
            data: ser(&data),
        });
        RouterTensor::new(TensorId::new(id), shape, dtype, self.clone())
    }

    fn device(&self) -> Self::Device {
        NdArrayDevice::Cpu
    }

    fn seed(&self, _seed: u64) {}

    fn dtype_usage(&self, _dtype: DType) -> DTypeUsageSet {
        DTypeUsage::general()
    }
}

struct NoBridge;

impl MultiBackendBridge for NoBridge {
    type TensorHandle = ();
    type Device = NdArrayDevice;
    fn change_backend_float(_: (), _: Shape, _: &NdArrayDevice) {
        unreachable!("single-device recording never transfers backends")
    }
    fn change_backend_int(_: (), _: Shape, _: &NdArrayDevice) {
        unreachable!("single-device recording never transfers backends")
    }
    fn change_backend_bool(_: (), _: Shape, _: &NdArrayDevice) {
        unreachable!("single-device recording never transfers backends")
    }
}

#[derive(Clone)]
struct RecChannel;

impl RunnerChannel for RecChannel {
    type Device = NdArrayDevice;
    type Bridge = NoBridge;
    type Client = RecClient;
    type FloatElem = f32;
    type IntElem = i64;
    // Mirrors the guest SDK channel (`daemon-vhc-sdk-compute`): bool rides the wire as u32
    // storage — WGSL forbids native bool in the storage address space (cubecl#1274).
    type BoolElem = u32;

    fn name(_device: &NdArrayDevice) -> String {
        "compute@2-recording".to_string()
    }

    fn init_client(_device: &NdArrayDevice) -> RecClient {
        RecClient
    }

    fn get_tensor_handle(_tensor: &TensorIr, _client: &RecClient) {
        unreachable!("single-device recording never extracts a cross-backend handle")
    }

    fn register_tensor(
        _client: &RecClient,
        _handle: (),
        _shape: Shape,
        _dtype: DType,
    ) -> RouterTensor<RecClient> {
        unreachable!("single-device recording never registers a cross-backend handle")
    }
}

type RecBackend = BackendRouter<RecChannel>;

/// The fixed inputs (bit-exact against `compute_conformance.rs`'s, so a replay's correctness is a
/// bit-exact equality vs native, not a tolerance).
fn inputs() -> (Vec<f32>, [usize; 2], Vec<f32>, [usize; 2]) {
    let a = vec![0.5, -1.0, 2.0, 3.0, -0.25, 1.5]; // [2,3]
    let w = vec![1.0, 0.0, -1.0, 2.0, 0.5, -0.5]; // [3,2]
    (a, [2, 3], w, [3, 2])
}

/// Record the op-journal of `relu(a·w + 0.75).sum()` forward+backward, exporting `∂loss/∂a`.
/// Returns `(journal, exported TensorIr CBOR)`.
fn record_grad_journal() -> (Vec<ComputeStep>, Vec<u8>) {
    rec_reset();
    let dev = NdArrayDevice::Cpu;
    let (a_d, a_s, w_d, w_s) = inputs();
    let a = Tensor::<Autodiff<RecBackend>, 2>::from_data(TensorData::new(a_d, a_s), &dev)
        .require_grad();
    let w = Tensor::<Autodiff<RecBackend>, 2>::from_data(TensorData::new(w_d, w_s), &dev);
    let loss = burn::tensor::activation::relu(a.clone().matmul(w).add_scalar(0.75_f32)).sum();
    let grads = loss.backward();
    let ga = a.grad(&grads).expect("∂loss/∂a exists"); // Tensor<RecBackend, 2>
                                                       // Consume the gradient into its TensorIr (no Drop enqueued): this is the export handle the
                                                       // replay reads back. Steps are drained BEFORE a/w/loss drop, so no Drop ops enter the journal
                                                       // and the export id stays live at replay.
    let ir: TensorIr = ga.into_primitive().tensor().into_ir();
    let export = ser(&ir);
    let steps = rec_take();
    (steps, export)
}

/// Replay a recorded op-journal against the PRODUCTION `ComputeRunner` over backend `B`, returning
/// the exported tensor's `f32` values (kernels re-executed on `B`).
fn replay_on<B: BackendIr>(steps: &[ComputeStep], export_ir: &[u8], device: B::Device) -> Vec<f32> {
    let mut runner = ComputeRunner::<B>::new(device);
    for step in steps {
        match step {
            ComputeStep::Import { id, data } => {
                runner.import_tensor(*id, data).expect("import replays");
            }
            ComputeStep::Op(op) => runner.submit_op(op).expect("op replays"),
        }
    }
    runner.fence().expect("fence drains clean");
    let data_cbor = runner.read_tensor(export_ir).expect("export reads back");
    let data: BackendTensorData =
        ciborium::from_reader(&data_cbor[..]).expect("exported TensorData decodes");
    data.convert::<f32>().to_vec::<f32>().expect("f32 values")
}

/// The native oracle: the identical forward+backward on `Autodiff<NdArray>`, no router boundary.
fn native_grad() -> Vec<f32> {
    type B = Autodiff<HostReal>;
    let dev = NdArrayDevice::Cpu;
    let (a_d, a_s, w_d, w_s) = inputs();
    let a = Tensor::<B, 2>::from_data(TensorData::new(a_d, a_s), &dev).require_grad();
    let w = Tensor::<B, 2>::from_data(TensorData::new(w_d, w_s), &dev);
    let loss = burn::tensor::activation::relu(a.clone().matmul(w).add_scalar(0.75_f32)).sum();
    let grads = loss.backward();
    a.grad(&grads)
        .expect("grad")
        .into_data()
        .to_vec::<f32>()
        .expect("f32 grads")
}

#[test]
fn compute_replay_degenerate_ndarray_is_bit_exact() {
    // The always-on tier-1 lane: record once, replay the SAME journal on ndarray twice. The
    // degenerate (same-backend) case is bit-exact — tolerance 0 — so the harness itself is always
    // exercised even where a second backend is unavailable. And each replay reconstructs the true
    // gradient (bit-exact vs a native, boundary-free Autodiff<NdArray> run), proving the journal
    // actually reproduces the computation rather than merely matching itself.
    let (journal, export_ir) = record_grad_journal();
    assert!(
        journal
            .iter()
            .any(|s| matches!(s, ComputeStep::Import { .. }))
            && journal.iter().any(|s| matches!(s, ComputeStep::Op(_))),
        "the journal captured both imports and ops"
    );

    let a = replay_on::<HostReal>(&journal, &export_ir, NdArrayDevice::Cpu);
    let b = replay_on::<HostReal>(&journal, &export_ir, NdArrayDevice::Cpu);
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b.iter()) {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "ndarray↔ndarray replay must be bit-exact (degenerate tolerance class)"
        );
    }

    let native = native_grad();
    assert_eq!(a.len(), native.len(), "gradient element count");
    for (x, n) in a.iter().zip(native.iter()) {
        assert_eq!(
            x.to_bits(),
            n.to_bits(),
            "replay must reconstruct the native gradient bit-exactly"
        );
    }
}

/// Record the trainer's mask/compare path — `arange` → `greater` → `mask_fill` → `gather`, the
/// exact shape `tiny-llama`'s `causal_mask` + `target_logprobs` submit — and return the journal
/// plus the gathered tensor's export `TensorIr`.
fn record_mask_journal() -> (Vec<ComputeStep>, Vec<u8>) {
    use burn::tensor::Int;

    rec_reset();
    let dev = NdArrayDevice::Cpu;
    let s = 4usize;
    // The causal mask, built on device from one arange row (guest `model.rs::causal_mask`).
    let pos = Tensor::<RecBackend, 1, Int>::arange(0..s as i64, &dev);
    let cols = pos.clone().reshape([1, s]).expand([s, s]);
    let rows = pos.reshape([s, 1]).expand([s, s]);
    let mask = Tensor::<RecBackend, 2>::zeros([s, s], &dev).mask_fill(cols.greater(rows), -1.0e30);
    // The loss's indexed selection (guest `model.rs::target_logprobs`): a gather over
    // guest-authored ids.
    let ids =
        Tensor::<RecBackend, 1, Int>::from_data(TensorData::new(vec![2i64, 0, 3, 1], [s]), &dev)
            .reshape([s, 1]);
    let picked = mask.gather(1, ids);
    let ir: TensorIr = picked.into_primitive().tensor().into_ir();
    let export = ser(&ir);
    (rec_take(), export)
}

#[test]
fn the_mask_path_rides_the_wire_without_native_bool_storage() {
    // The D1 regression net (backend-lane audit 2026-07-28): the guest channel's `BoolElem` chose
    // native bool, so the comparison/mask kernels carried `DType::Bool(Native)` across the
    // `compute@2` wire and every WGSL host lane (DX12, Metal) refused `var<storage> array<bool>`
    // at shader validation (WGSL: bool is not host-shareable; cubecl#1274). The channel now pins
    // u32 bool storage; this walks every dtype the mask path actually puts on the wire.
    use burn_backend::BoolStore;

    let (journal, export_ir) = record_mask_journal();

    let mut bool_seen = false;
    let mut check = |dtype: DType, what: &str| {
        if let DType::Bool(store) = dtype {
            bool_seen = true;
            assert!(
                !matches!(store, BoolStore::Native),
                "{what} carries native-bool storage, which WGSL host lanes refuse \
                 (bool is not host-shareable in the storage address space)"
            );
        }
    };
    for step in &journal {
        match step {
            ComputeStep::Import { data, .. } => {
                let data: BackendTensorData =
                    ciborium::from_reader(&data[..]).expect("imported TensorData decodes");
                check(data.dtype, "an imported tensor");
            }
            ComputeStep::Op(op) => {
                let op: OperationIr = ciborium::from_reader(&op[..]).expect("op-blob decodes");
                for node in op.nodes() {
                    check(node.dtype, "an op tensor");
                }
            }
        }
    }
    assert!(
        bool_seen,
        "the mask workload must put a bool tensor on the wire, or this test guards nothing"
    );

    // And the u32-bool journal still executes on the production runner: the gathered values are
    // the mask's, bit-exact (row i, col j: 0.0 on/below the diagonal, -1.0e30 above).
    let vals = replay_on::<HostReal>(&journal, &export_ir, NdArrayDevice::Cpu);
    assert_eq!(vals, vec![-1.0e30, 0.0, -1.0e30, 0.0]);
}

/// The wgpu cross-backend tier — hardware-gated (the same convention as `parity`'s wgpu tier:
/// the `.#vulkan` shell / a GPU runner exercises it). The ndarray-recorded journal re-executed on
/// wgpu reproduces the trajectory within the native lane's TOLERANCE CLASS — NOT bit-exact
/// (heterogeneous-hardware arithmetic differs; architecture §3.6/§10).
#[cfg(feature = "wgpu")]
#[test]
fn compute_replay_cross_backend_wgpu_within_tolerance() {
    use burn::backend::wgpu::{Wgpu, WgpuDevice};

    if !daemon_vhc_host::wgpu_adapter_available() {
        eprintln!("SKIP compute_replay(wgpu): no usable wgpu adapter on this runner");
        return;
    }

    let (journal, export_ir) = record_grad_journal();
    // The tier-1 backend (ndarray) is the reference; wgpu is the second backend behind the journal.
    let reference = replay_on::<HostReal>(&journal, &export_ir, NdArrayDevice::Cpu);
    let crossed = replay_on::<Wgpu<f32, i32>>(&journal, &export_ir, WgpuDevice::default());

    assert_eq!(reference.len(), crossed.len(), "gradient element count");
    // Native-lane tolerance class (matmul/elementwise on f32): a loose rtol/atol band — the claim
    // is tolerance-equivalence across backends, never bit-identity (architecture §3.6).
    let (rtol, atol) = (1e-4_f32, 1e-5_f32);
    for (r, c) in reference.iter().zip(crossed.iter()) {
        let tol = atol + rtol * r.abs();
        assert!(
            (r - c).abs() <= tol,
            "cross-backend replay outside tolerance: ndarray {r} vs wgpu {c} (tol {tol})"
        );
    }
}
