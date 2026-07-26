// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `toy-mlp` — the **Phase-C model-agnostic acceptance** (refactor §7: "a non-LLaMA toy authored
//! with zero host changes"; architecture §3.2 "modules write ordinary Burn models").
//!
//! A small, deliberately **non-LLaMA** model — a two-layer MLP (`relu(x·W1)·W2`, sum-of-squares
//! loss) trained by hand-rolled SGD — authored as an ordinary Burn model over
//! `daemon-vhc-sdk-compute`'s `Autodiff<HostBackend>`. Its ENTIRE dependency surface is
//! `daemon-vhc-sdk-compute` + `daemon-vhc-sdk`: no LLaMA silhouette, no SDK model code, and —
//! the point of the lane — **no host edit**. It runs against the same `compute@2` runner, driver,
//! and journal the `tiny-llama` reference does, proving the compute ABI is model-agnostic (the op
//! stream is `CBOR(burn_ir::OperationIr)` blobs the host dispatches by shape, never by model).
//!
//! Config (raw bytes): `[steps]` (u8, default 3) — SGD steps to run before extracting the model.
//!
//! Behavior (scenario-free; one path):
//! 1. Build `x`,`y` (inputs) and `W1`,`W2` (parameters) from deterministic constants.
//! 2. Run `steps` of SGD: forward → sum-squared loss → backward → `W ← W − lr·∂loss/∂W`. The
//!    whole loop is **pure enqueueing on handles** — the autodiff tape walks guest-side with zero
//!    intermediate readbacks (decisions D8), exactly like the LLaMA reference.
//! 3. `fence`, then `export` the trained `W1` (device → sealed buffer), and on the completion read
//!    its `CBOR(TensorData)` into linear memory and publish it — the host compares it bit-exact vs
//!    a native `Autodiff<NdArray>` run of the identical loop (the model-agnostic-lowering proof,
//!    the MLP twin of `test-compute-v2`'s single-op gradient check).
//!
//! **Drop discipline** (ABI §4.4, as in `test-compute-v2`): every tensor still held when `Stop`
//! arrives would enqueue its `OperationIr::Drop` *after* `Stop` — a `PhaseViolation`. So the whole
//! model is built (and every non-exported tensor dropped) in an inner scope BEFORE the event loop,
//! and the exported weight is consumed by `export`.

use burn::tensor::{Tensor, TensorData};
use daemon_vhc_sdk::{GuestModule, ModuleDecl};
use daemon_vhc_sdk_compute::{device, export_tensor, fence, AutodiffHostBackend, HostBackend};

const EV_FENCE: u64 = 5;
const EV_COMPLETION: u64 = 6;
const EV_STOP: u64 = 4;

/// Model dimensions (kept in lockstep with the host oracle in `tests/toy_mlp.rs`).
const IN: usize = 3;
const HID: usize = 4;
const OUT: usize = 2;
const BATCH: usize = 2;
/// SGD learning rate (a plain constant — no schedule).
const LR: f32 = 0.1;

struct ToyMlp {
    steps: u8,
}

/// The deterministic inputs + initial parameters (identical to the host oracle's).
fn params() -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    // x: [BATCH, IN], y: [BATCH, OUT], w1: [IN, HID], w2: [HID, OUT].
    let x = vec![0.5, -1.0, 2.0, 0.25, 1.5, -0.5];
    let y = vec![1.0, -0.5, 0.0, 2.0];
    let w1 = vec![
        0.1, -0.2, 0.3, 0.05, //
        0.4, -0.1, 0.2, -0.3, //
        -0.25, 0.15, 0.35, 0.1,
    ];
    let w2 = vec![
        0.2, -0.4, //
        0.1, 0.3, //
        -0.15, 0.25, //
        0.05, -0.35,
    ];
    (x, y, w1, w2)
}

impl GuestModule for ToyMlp {
    fn decl() -> ModuleDecl {
        ModuleDecl {
            name: "toy-mlp",
            version: env!("CARGO_PKG_VERSION"),
            // compute@2 imports force the Phase-C minor (ABI §1.3 step 5), exactly as the LLaMA
            // reference and `test-compute-v2` — the point being nothing else differs.
            abi_minor: 2,
            channels: vec![0],
            host_state_bytes: 1 << 20,
            host_scratch_bytes: 1 << 20,
            device_state_bytes: 1 << 20,
            device_scratch_bytes: 1 << 20,
        }
    }

    /// This module's Logical Resource Plan. Its algorithm holds nothing device-resident, so the
    /// canonical trivial plan IS its plan — it carries the module's own linear-memory floor and
    /// declares no device tensor, no operation family and no bounded transfer. It is emitted here
    /// rather than written down beside the module because a plan authoring could publish without
    /// the module having produced it is a second source, however small its contents.
    fn resource_plan(
        config: &[u8],
        _capability_grants: &[u8],
    ) -> Result<daemon_vhc_sdk::LogicalResourcePlan, u32> {
        Ok(daemon_vhc_sdk::trivial_resource_plan(
            &Self::decl_for_config(config),
        ))
    }

    fn init(config: &[u8], _grants: &[u8]) -> Result<Self, u32> {
        Ok(Self {
            steps: config.first().copied().unwrap_or(3),
        })
    }

    fn run(&mut self) -> u32 {
        train_and_publish(self.steps)
    }
}

daemon_vhc_sdk::main!(ToyMlp);

/// Train the MLP for `steps` SGD steps and publish the trained `W1` through the fence → export →
/// completion → read path.
fn train_and_publish(steps: u8) -> u32 {
    // Build + train inside a scope so every AD graph tensor drops (its Drop op enqueues pre-Stop,
    // legal) — only the extracted `W1` value survives to be exported.
    let mut export_me: Option<Tensor<HostBackend, 2>> = Some({
        let dev = device();
        let (x_d, y_d, w1_d, w2_d) = params();
        let x =
            Tensor::<AutodiffHostBackend, 2>::from_data(TensorData::new(x_d, [BATCH, IN]), &dev);
        let y =
            Tensor::<AutodiffHostBackend, 2>::from_data(TensorData::new(y_d, [BATCH, OUT]), &dev);
        let mut w1 =
            Tensor::<AutodiffHostBackend, 2>::from_data(TensorData::new(w1_d, [IN, HID]), &dev)
                .require_grad();
        let mut w2 =
            Tensor::<AutodiffHostBackend, 2>::from_data(TensorData::new(w2_d, [HID, OUT]), &dev)
                .require_grad();

        for _ in 0..steps {
            // Forward: relu(x·W1)·W2 — an ordinary Burn expression, every op an enqueued
            // CBOR(OperationIr) blob on opaque handles.
            let hidden = burn::tensor::activation::relu(x.clone().matmul(w1.clone()));
            let logits = hidden.matmul(w2.clone());
            let diff = logits.sub(y.clone());
            // Sum-of-squares loss (a scalar); backward is pure enqueueing (zero readbacks).
            let loss = diff.clone().mul(diff).sum();
            let grads = loss.backward();
            let g1 = w1.grad(&grads).expect("∂loss/∂W1 exists");
            let g2 = w2.grad(&grads).expect("∂loss/∂W2 exists");
            // SGD update on-device: detach (`inner`), step, re-require_grad for the next step.
            w1 = Tensor::from_inner(w1.inner().sub(g1.mul_scalar(LR))).require_grad();
            w2 = Tensor::from_inner(w2.inner().sub(g2.mul_scalar(LR))).require_grad();
        }
        // The trained W1 as a plain (inner-backend) tensor; the AD graph (x/y/w2 wrappers) drops
        // at this scope's close, before the fence + event loop below.
        w1.inner()
    });

    fence(1);
    let mut export_op = 0u64;
    let mut buf: Vec<u8> = Vec::with_capacity(256);
    loop {
        let ev = daemon_vhc_sdk::next_event(&mut buf);
        match ev.tag {
            EV_FENCE if ev.uint(1) == 1 => {
                export_op = export_tensor(export_me.take().expect("fence delivers once"));
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
                let bytes = daemon_vhc_sdk::read_buffer(handle);
                daemon_vhc_sdk::buffer_release(handle);
                daemon_vhc_sdk::publish(0, &bytes);
            }
            EV_STOP => return 0,
            _ => {}
        }
    }
}
