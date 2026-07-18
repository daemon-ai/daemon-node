// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `test-compute-v2` — the Phase-C compute acceptance guest (track C1; refactor §7, ABI §15).
//!
//! An ordinary Burn model over the **guest-side `HostBackend`** (`daemon-vhc-sdk-compute`),
//! running as a real wasm32 module against the real driver: every tensor op crosses the
//! `compute@2` boundary as a `CBOR(burn_ir::OperationIr)` blob on opaque `u64` handles, the
//! autodiff tape walks guest-side with zero intermediate readbacks, and extraction rides the
//! fence → export → `Completion(BufferHandle)` → `read_into` path (architecture §3.2–§3.4).
//!
//! Config: `[scenario, depth]`.
//!
//! | scenario | behavior |
//! |---|---|
//! | 0 | forward+backward (`relu(a·w + 0.75).sum()`), fence, export ∂loss/∂a, publish its `CBOR(TensorData)` — the host compares bit-exact vs native `Autodiff<NdArray>` |
//! | 1 | export a never-registered `TensorId` — expects the typed `InvalidHandle` trap |
//! | 2 | submit `depth + 1` ops without a fence — expects the `GrantViolation` queue-depth trap |
//! | 3 | one op under injected device fault, then `fence` — expects the `ComputeFault` trap |
//! | 4 | one op under injected device fault, then `export` — expects `Err(DeviceError)` in the completion; publishes `b"device-error"` |

use burn::tensor::Tensor;
use burn_ir::{TensorId, TensorIr, TensorStatus};
use daemon_vhc_sdk::{GuestModule, ModuleDecl};
use daemon_vhc_sdk_compute::{
    device, export_tensor, fence, tensor_from_floats, AutodiffHostBackend, HostBackend,
};

const EV_FENCE: u64 = 5;
const EV_COMPLETION: u64 = 6;
const EV_STOP: u64 = 4;

struct ComputeGuest {
    scenario: u8,
    depth: u8,
}

/// The deterministic inputs (identical to the host conformance suite's, so the comparison is a
/// bit-exact equality).
fn inputs() -> (Vec<f32>, [usize; 2], Vec<f32>, [usize; 2]) {
    let a = vec![0.5, -1.0, 2.0, 3.0, -0.25, 1.5]; // [2,3]
    let w = vec![1.0, 0.0, -1.0, 2.0, 0.5, -0.5]; // [3,2]
    (a, [2, 3], w, [3, 2])
}

impl GuestModule for ComputeGuest {
    fn decl() -> ModuleDecl {
        ModuleDecl {
            name: "test-compute-v2",
            version: env!("CARGO_PKG_VERSION"),
            abi_minor: 2, // compute@2 imports force the Phase-C minor (ABI §1.3 step 5)
            channels: vec![0],
            host_state_bytes: 1 << 20,
            host_scratch_bytes: 1 << 20,
            device_state_bytes: 1 << 20,
            device_scratch_bytes: 1 << 20,
        }
    }

    fn init(config: &[u8], _grants: &[u8]) -> Result<Self, u32> {
        if config.is_empty() {
            return Err(16);
        }
        Ok(Self {
            scenario: config[0],
            depth: config.get(1).copied().unwrap_or(4),
        })
    }

    fn run(&mut self) -> u32 {
        match self.scenario {
            0 => forward_backward_export(),
            1 => invalid_handle_export(),
            2 => queue_depth_breach(self.depth),
            3 => fault_surfaces_at_fence(),
            4 => fault_surfaces_at_export(),
            _ => 16,
        }
    }
}

daemon_vhc_sdk::main!(ComputeGuest);

/// Scenario 0 — the acceptance path: an ordinary Burn forward+backward over `HostBackend`, the
/// gradient extracted through fence → export → completion → read.
///
/// **Drop discipline:** every tensor a Burn guest still holds when it consumes `Stop` would
/// enqueue its `OperationIr::Drop` through `submit_op` *after* `Stop` — a `PhaseViolation` trap
/// (ABI §4.4). So everything except the exported gradient is built (and dropped) in an inner
/// scope BEFORE the event loop, and the gradient itself is consumed by the export.
fn forward_backward_export() -> u32 {
    let mut ga: Option<Tensor<HostBackend, 2>> = Some({
        let (a_data, a_shape, w_data, w_shape) = inputs();
        let dev = device();
        let a = Tensor::<AutodiffHostBackend, 2>::from_data(
            burn::tensor::TensorData::new(a_data, a_shape),
            &dev,
        )
        .require_grad();
        let w = Tensor::<AutodiffHostBackend, 2>::from_data(
            burn::tensor::TensorData::new(w_data, w_shape),
            &dev,
        );
        // Forward + backward: pure enqueueing on handles — the tape is guest-side (decisions D8).
        let loss = burn::tensor::activation::relu(a.clone().matmul(w).add_scalar(0.75_f32)).sum();
        let grads = loss.backward();
        a.grad(&grads).expect("grad exists")
        // a / grads / loss remnants drop HERE — their Drop ops enqueue pre-Stop (legal).
    });

    // Consistency point, then extraction, through the event loop (§3.3).
    fence(1);
    let mut export_op = 0u64;
    let mut buf: Vec<u8> = Vec::with_capacity(256);
    loop {
        let ev = daemon_vhc_sdk::next_event(&mut buf);
        match ev.tag {
            EV_FENCE if ev.uint(1) == 1 => {
                export_op = export_tensor(ga.take().expect("fence delivers once"));
            }
            EV_COMPLETION if export_op != 0 && ev.uint(1) == export_op => {
                let Some(ciborium::value::Value::Array(result)) = ev.items.get(2) else {
                    return 17;
                };
                let ok = result
                    .first()
                    .and_then(|v| v.as_integer())
                    .is_some_and(|n| i128::from(n) == 0);
                if !ok {
                    daemon_vhc_sdk::publish(0, b"export-failed");
                    return 18;
                }
                let handle = result
                    .get(1)
                    .and_then(|v| v.as_integer())
                    .map(|n| u64::try_from(i128::from(n)).unwrap_or(0))
                    .unwrap_or(0);
                // The exported CBOR(TensorData), crossed into linear memory (budgeted §3.4).
                let bytes = daemon_vhc_sdk::read_buffer(handle);
                daemon_vhc_sdk::buffer_release(handle);
                daemon_vhc_sdk::publish(0, &bytes);
            }
            EV_STOP => return 0,
            _ => {}
        }
    }
}

/// Scenario 1 — a never-registered handle at export is the typed `InvalidHandle` trap (ABI §15).
fn invalid_handle_export() -> u32 {
    let ir = TensorIr {
        id: TensorId::new(9_999_999),
        shape: burn::tensor::Shape::from(vec![2usize, 2usize]),
        status: TensorStatus::ReadOnly,
        dtype: burn::tensor::DType::F32,
    };
    let mut bytes = Vec::new();
    ciborium::into_writer(&ir, &mut bytes).expect("ir encodes");
    let _op = daemon_vhc_sdk::compute_export(&bytes); // traps InvalidHandle
    19 // unreachable when the host conforms
}

/// Scenario 2 — `depth + 1` enqueues without a fence breach the queue-depth grant (§3.3).
fn queue_depth_breach(depth: u8) -> u32 {
    let t = tensor_from_floats(vec![1.0, 2.0, 3.0, 4.0], [2, 2]);
    // Keep every intermediate alive: a guest-side drop would enqueue OperationIr::Drop ops and
    // muddy the count — exactly `depth + 1` submit_ops happen here.
    let mut keep: Vec<Tensor<HostBackend, 2>> = Vec::new();
    for i in 0..=u64::from(depth) {
        #[allow(clippy::cast_precision_loss)]
        keep.push(t.clone().add_scalar(i as f32)); // traps GrantViolation at i == depth
    }
    20 // unreachable when the host conforms
}

/// Scenario 3 — an injected deferred device fault surfaces at the NEXT fence as `ComputeFault`.
fn fault_surfaces_at_fence() -> u32 {
    let t = tensor_from_floats(vec![1.0, 2.0], [2]);
    let kept = t.add_scalar(1.0_f32); // the first submit_op; the host latches the fault after it
    fence(7); // traps ComputeFault
    drop(kept);
    21 // unreachable when the host conforms
}

/// Scenario 4 — the same fault surfaces at export as an `Err(DeviceError)` completion.
fn fault_surfaces_at_export() -> u32 {
    let t = tensor_from_floats(vec![1.0, 2.0], [2]);
    let kept = t.add_scalar(1.0_f32);
    let op = export_tensor(kept);
    let mut buf: Vec<u8> = Vec::with_capacity(128);
    loop {
        let ev = daemon_vhc_sdk::next_event(&mut buf);
        match ev.tag {
            EV_COMPLETION if ev.uint(1) == op => {
                let Some(ciborium::value::Value::Array(result)) = ev.items.get(2) else {
                    return 22;
                };
                let failed = result
                    .first()
                    .and_then(|v| v.as_integer())
                    .is_some_and(|n| i128::from(n) == 1);
                daemon_vhc_sdk::publish(
                    0,
                    if failed {
                        b"device-error"
                    } else {
                        b"unexpected"
                    },
                );
            }
            EV_STOP => return 0,
            _ => {}
        }
    }
}
