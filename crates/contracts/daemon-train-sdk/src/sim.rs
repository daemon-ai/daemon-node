// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The `sim` backend: an in-crate CPU reference implementation of `tabi@1` (ABI §10.4).
//!
//! It swaps the extern block for a native fp32 store so experiments and profiles are unit-testable
//! with `cargo test --features sim` — no GPU, no wasm host. The **det lane** delegates to the shared
//! `det-core` kernels (so "sim ≡ host" is one implementation, ABI §5.6/§10.1); the native lane runs
//! a tiny reverse-mode tape (enough for a dense MLP: matmul / bias-add / relu / cross-entropy),
//! which is *semantics*-reference, not performance-reference.
//!
//! State is thread-local so `cargo test`'s parallel threads never share a store. Handles carry a tag
//! in their top byte (params vs persistents vs step tensors vs det tensors vs containers vs
//! batches), mirroring the host arena's lane/class tagging (ABI §3.3/§3.4).

use crate::abi::RawHandle;
use crate::{Batch, UpdateBuilder};
use std::cell::RefCell;

// -- handle tags --------------------------------------------------------------------------------

const TAG_SHIFT: u32 = 56;
const IDX_MASK: u64 = (1 << TAG_SHIFT) - 1;

const TAG_NODE: u64 = 1; // native step tensor (tape node)
const TAG_DET: u64 = 2; // det step tensor
const TAG_PARAM: u64 = 3;
const TAG_PERSIST: u64 = 4;
const TAG_DETPERSIST: u64 = 5;
const TAG_UPD: u64 = 6;
const TAG_BATCH: u64 = 7;

fn enc(tag: u64, idx: usize) -> RawHandle {
    (tag << TAG_SHIFT) | ((idx as u64) + 1)
}

fn dec(h: RawHandle) -> (u64, usize) {
    (h >> TAG_SHIFT, ((h & IDX_MASK) - 1) as usize)
}

fn numel(dims: &[u32]) -> usize {
    dims.iter().map(|&d| d as usize).product()
}

// -- store types --------------------------------------------------------------------------------

#[derive(Clone)]
struct ParamSlot {
    #[allow(dead_code)]
    name: String,
    shape: Vec<usize>,
    master: Vec<f32>,
    storage: Vec<f32>,
    grad: Vec<f32>,
    round_base: Vec<f32>,
}

#[derive(Clone)]
struct StateSlot {
    #[allow(dead_code)]
    name: String,
    #[allow(dead_code)]
    shape: Vec<usize>,
    #[allow(dead_code)]
    class: u32,
    data: Vec<f32>,
}

#[derive(Clone)]
enum Op {
    Const,
    MatMul {
        a: RawHandle,
        b: RawHandle,
        m: usize,
        k: usize,
        n: usize,
    },
    Add {
        a: RawHandle,
        b: RawHandle,
    },
    AddBias {
        a: RawHandle,
        b: RawHandle,
        rows: usize,
        cols: usize,
    },
    Sub {
        a: RawHandle,
        b: RawHandle,
    },
    Mul {
        a: RawHandle,
        b: RawHandle,
    },
    MulS {
        x: RawHandle,
        s: f32,
    },
    Relu {
        x: RawHandle,
    },
    CrossEntropy {
        logits: RawHandle,
        rows: usize,
        cols: usize,
        targets: Vec<i64>,
        softmax: Vec<f32>,
    },
}

struct Node {
    value: Vec<f32>,
    shape: Vec<usize>,
    op: Op,
    grad: Vec<f32>,
}

struct DetVal {
    data: Vec<f32>,
    shape: Vec<usize>,
}

enum Section {
    Bytes(Vec<u8>),
    Tensor { data: Vec<f32>, shape: Vec<usize> },
}

struct Container {
    sections: Vec<Section>,
}

struct BatchVal {
    tokens: Vec<u32>,
    batch: u32,
    seq: u32,
}

/// The thread-local CPU store.
pub struct Store {
    run_seed: u64,
    params: Vec<ParamSlot>,
    persistents: Vec<StateSlot>,
    det_persistents: Vec<StateSlot>,
    nodes: Vec<Node>,
    dets: Vec<DetVal>,
    containers: Vec<Container>,
    staged: Vec<usize>,
    batches: Vec<BatchVal>,
    metrics: Vec<(String, f32)>,
}

thread_local! {
    static STORE: RefCell<Store> = RefCell::new(Store::new(0));
}

/// Run `f` against the thread-local store (used by the [`crate::abi`] sim dispatch).
pub(crate) fn with<R>(f: impl FnOnce(&mut Store) -> R) -> R {
    STORE.with(|s| f(&mut s.borrow_mut()))
}

// -- public driver API (tests / harnesses) ------------------------------------------------------

/// Reset the store to a fresh run keyed by `run_seed` (host warmup, ABI §2.3).
pub fn reset(run_seed: u64) {
    STORE.with(|s| *s.borrow_mut() = Store::new(run_seed));
}

/// Take the round-base master snapshot every param exposes via `param_round_base`/`det_param` —
/// the host does this at the ingest barrier (ABI §2.3/§5.9).
pub fn snapshot_round_base() {
    with(|s| {
        for p in &mut s.params {
            p.round_base = p.master.clone();
        }
    });
}

/// Stage a built update for the next `da_ingest_updates` (self-inclusive; the host would stage the
/// committed set in record order, ABI §5.11 — the sim stages exactly what the test provides).
pub fn stage(update: &UpdateBuilder) {
    let (_, idx) = dec(update.handle());
    with(|s| s.staged.push(idx));
}

/// Clear the staged set.
pub fn clear_staged() {
    with(|s| s.staged.clear());
}

/// Provide a host micro-batch and return its [`Batch`] handle.
#[must_use]
pub fn make_batch(tokens: Vec<u32>, batch: u32, seq: u32) -> Batch {
    let h = with(|s| {
        s.batches.push(BatchVal { tokens, batch, seq });
        enc(TAG_BATCH, s.batches.len() - 1)
    });
    Batch::from_handle(h)
}

/// The current fp32 canonical master of a registered param (inspection).
#[must_use]
pub fn param_master(name: &str) -> Option<Vec<f32>> {
    with(|s| {
        s.params
            .iter()
            .find(|p| p.name == name)
            .map(|p| p.master.clone())
    })
}

/// The metrics reported via `metric@1` this run.
#[must_use]
pub fn metrics() -> Vec<(String, f32)> {
    with(|s| s.metrics.clone())
}

// -- deterministic init ------------------------------------------------------------------------

fn fnv1a(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut x = z;
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

struct InitRng(u64);
impl InitRng {
    fn unit(&mut self) -> f32 {
        self.0 = splitmix(self.0);
        ((self.0 >> 40) as f32) / ((1u64 << 24) as f32)
    }
    fn normal(&mut self) -> f32 {
        // Box-Muller; two units → one standard normal.
        let u1 = self.unit().max(1.0e-7);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (core::f32::consts::TAU * u2).cos()
    }
}

fn init_values(run_seed: u64, name: &str, n: usize, init: u32, p0: f64, p1: f64) -> Vec<f32> {
    let mut rng = InitRng(splitmix(run_seed ^ fnv1a(name)));
    let (p0, p1) = (p0 as f32, p1 as f32);
    (0..n)
        .map(|_| match init {
            0 => 0.0,                                     // Zeros
            1 => 1.0,                                     // Ones
            2 => p0 + (p1 - p0) * rng.unit(),             // Uniform(p0,p1)
            3 => p0 + p1 * rng.normal(),                  // Normal(mean=p0,std=p1)
            4 => p0 + p1 * rng.normal().clamp(-2.0, 2.0), // TruncNormal
            _ => 0.0,
        })
        .collect()
}

// -- linear algebra helpers ---------------------------------------------------------------------

fn mm(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0_f32;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            out[i * n + j] = acc;
        }
    }
    out
}

fn transpose(a: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            out[j * rows + i] = a[i * cols + j];
        }
    }
    out
}

fn add_into(dst: &mut [f32], src: &[f32]) {
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d += s;
    }
}

// -- the ABI surface (called via crate::abi's sim branch) ---------------------------------------

impl Store {
    fn new(run_seed: u64) -> Self {
        Self {
            run_seed,
            params: Vec::new(),
            persistents: Vec::new(),
            det_persistents: Vec::new(),
            nodes: Vec::new(),
            dets: Vec::new(),
            containers: Vec::new(),
            staged: Vec::new(),
            batches: Vec::new(),
            metrics: Vec::new(),
        }
    }

    fn push_node(&mut self, value: Vec<f32>, shape: Vec<usize>, op: Op) -> RawHandle {
        let grad = vec![0.0_f32; value.len()];
        self.nodes.push(Node {
            value,
            shape,
            op,
            grad,
        });
        enc(TAG_NODE, self.nodes.len() - 1)
    }

    fn push_det(&mut self, data: Vec<f32>, shape: Vec<usize>) -> RawHandle {
        self.dets.push(DetVal { data, shape });
        enc(TAG_DET, self.dets.len() - 1)
    }

    fn native_value(&self, h: RawHandle) -> Vec<f32> {
        let (tag, idx) = dec(h);
        match tag {
            TAG_PARAM => self.params[idx].storage.clone(),
            TAG_PERSIST => self.persistents[idx].data.clone(),
            TAG_NODE => self.nodes[idx].value.clone(),
            _ => panic!("sim: handle {h:#x} is not a native tensor"),
        }
    }

    fn native_shape(&self, h: RawHandle) -> Vec<usize> {
        let (tag, idx) = dec(h);
        match tag {
            TAG_PARAM => self.params[idx].shape.clone(),
            TAG_PERSIST => self.persistents[idx].shape.clone(),
            TAG_NODE => self.nodes[idx].shape.clone(),
            _ => panic!("sim: handle {h:#x} is not a native tensor"),
        }
    }

    fn det_value(&self, h: RawHandle) -> Vec<f32> {
        let (tag, idx) = dec(h);
        match tag {
            TAG_DET => self.dets[idx].data.clone(),
            TAG_DETPERSIST => self.det_persistents[idx].data.clone(),
            _ => panic!("sim: handle {h:#x} is not a det tensor"),
        }
    }

    fn add_native_grad(&mut self, h: RawHandle, g: &[f32]) {
        let (tag, idx) = dec(h);
        match tag {
            TAG_PARAM => add_into(&mut self.params[idx].grad, g),
            TAG_NODE => add_into(&mut self.nodes[idx].grad, g),
            _ => {} // persistents / consts do not accumulate grad
        }
    }

    // registration ------------------------------------------------------------------------------

    pub(crate) fn param(
        &mut self,
        name: &str,
        dims: &[u32],
        _dtype: u32,
        init: u32,
        p0: f64,
        p1: f64,
    ) -> RawHandle {
        let n = numel(dims);
        let master = init_values(self.run_seed, name, n, init, p0, p1);
        self.params.push(ParamSlot {
            name: name.to_string(),
            shape: dims.iter().map(|&d| d as usize).collect(),
            storage: master.clone(),
            grad: vec![0.0; n],
            round_base: master.clone(),
            master,
        });
        enc(TAG_PARAM, self.params.len() - 1)
    }

    pub(crate) fn persistent(
        &mut self,
        name: &str,
        dims: &[u32],
        _dtype: u32,
        class: u32,
    ) -> RawHandle {
        let n = numel(dims);
        self.persistents.push(StateSlot {
            name: name.to_string(),
            shape: dims.iter().map(|&d| d as usize).collect(),
            class,
            data: vec![0.0; n],
        });
        enc(TAG_PERSIST, self.persistents.len() - 1)
    }

    pub(crate) fn det_persistent(&mut self, name: &str, dims: &[u32], class: u32) -> RawHandle {
        let n = numel(dims);
        self.det_persistents.push(StateSlot {
            name: name.to_string(),
            shape: dims.iter().map(|&d| d as usize).collect(),
            class,
            data: vec![0.0; n],
        });
        enc(TAG_DETPERSIST, self.det_persistents.len() - 1)
    }

    pub(crate) fn drop_handle(&mut self, _h: RawHandle) {
        // The autodiff graph retains what backward needs (Burn-tape semantics, ABI §3.3), so a
        // step-handle drop is a no-op for the sim's correctness; eager-free budgeting is a HOST
        // property (tested in daemon-train), not a numeric-reference one.
    }

    // state / autodiff --------------------------------------------------------------------------

    pub(crate) fn param_round_base(&mut self, p: RawHandle) -> RawHandle {
        let (_, idx) = dec(p);
        let v = self.params[idx].round_base.clone();
        let shape = self.params[idx].shape.clone();
        self.push_node(v, shape, Op::Const)
    }

    pub(crate) fn grad(&mut self, p: RawHandle) -> RawHandle {
        let (_, idx) = dec(p);
        let v = self.params[idx].grad.clone();
        let shape = self.params[idx].shape.clone();
        self.push_node(v, shape, Op::Const)
    }

    pub(crate) fn zero_grads(&mut self) {
        for p in &mut self.params {
            for g in &mut p.grad {
                *g = 0.0;
            }
        }
    }

    pub(crate) fn assign(&mut self, dst: RawHandle, src: RawHandle) {
        let v = self.native_value(src);
        let (tag, idx) = dec(dst);
        match tag {
            TAG_PARAM => {
                self.params[idx].storage = v.clone();
                self.params[idx].master = v;
            }
            TAG_PERSIST => self.persistents[idx].data = v,
            _ => panic!("sim: assign target must be a param/persistent"),
        }
    }

    pub(crate) fn zeros(&mut self, dims: &[u32], _dtype: u32) -> RawHandle {
        self.full(dims, _dtype, 0.0)
    }

    pub(crate) fn full(&mut self, dims: &[u32], _dtype: u32, value: f64) -> RawHandle {
        let n = numel(dims);
        self.push_node(
            vec![value as f32; n],
            dims.iter().map(|&d| d as usize).collect(),
            Op::Const,
        )
    }

    pub(crate) fn matmul(&mut self, a: RawHandle, b: RawHandle) -> RawHandle {
        let ash = self.native_shape(a);
        let bsh = self.native_shape(b);
        assert_eq!(ash.len(), 2, "sim matmul is 2-D");
        assert_eq!(bsh.len(), 2, "sim matmul is 2-D");
        let (m, k, n) = (ash[0], ash[1], bsh[1]);
        assert_eq!(k, bsh[0], "sim matmul inner dims must agree");
        let av = self.native_value(a);
        let bv = self.native_value(b);
        let value = mm(&av, &bv, m, k, n);
        self.push_node(value, vec![m, n], Op::MatMul { a, b, m, k, n })
    }

    pub(crate) fn add(&mut self, a: RawHandle, b: RawHandle) -> RawHandle {
        let ash = self.native_shape(a);
        let bsh = self.native_shape(b);
        let av = self.native_value(a);
        let bv = self.native_value(b);
        if ash == bsh {
            let value: Vec<f32> = av.iter().zip(bv.iter()).map(|(&x, &y)| x + y).collect();
            self.push_node(value, ash, Op::Add { a, b })
        } else {
            // trailing-dim broadcast: bias `[cols]` onto `[.., cols]`.
            let cols = *ash.last().expect("add lhs has rank >= 1");
            assert_eq!(bsh, vec![cols], "sim add broadcast: bias must be [cols]");
            let rows = av.len() / cols;
            let mut value = av.clone();
            for i in 0..rows {
                for j in 0..cols {
                    value[i * cols + j] += bv[j];
                }
            }
            self.push_node(value, ash, Op::AddBias { a, b, rows, cols })
        }
    }

    pub(crate) fn sub(&mut self, a: RawHandle, b: RawHandle) -> RawHandle {
        let shape = self.native_shape(a);
        let av = self.native_value(a);
        let bv = self.native_value(b);
        let value: Vec<f32> = av.iter().zip(bv.iter()).map(|(&x, &y)| x - y).collect();
        self.push_node(value, shape, Op::Sub { a, b })
    }

    pub(crate) fn mul(&mut self, a: RawHandle, b: RawHandle) -> RawHandle {
        let shape = self.native_shape(a);
        let av = self.native_value(a);
        let bv = self.native_value(b);
        let value: Vec<f32> = av.iter().zip(bv.iter()).map(|(&x, &y)| x * y).collect();
        self.push_node(value, shape, Op::Mul { a, b })
    }

    pub(crate) fn mul_s(&mut self, x: RawHandle, v: f64) -> RawHandle {
        let shape = self.native_shape(x);
        let s = v as f32;
        let value: Vec<f32> = self.native_value(x).iter().map(|&e| e * s).collect();
        self.push_node(value, shape, Op::MulS { x, s })
    }

    pub(crate) fn relu(&mut self, x: RawHandle) -> RawHandle {
        let shape = self.native_shape(x);
        let value: Vec<f32> = self.native_value(x).iter().map(|&e| e.max(0.0)).collect();
        self.push_node(value, shape, Op::Relu { x })
    }

    pub(crate) fn cross_entropy(
        &mut self,
        logits: RawHandle,
        targets: RawHandle,
        ignore_index: i64,
    ) -> RawHandle {
        let sh = self.native_shape(logits);
        assert_eq!(sh.len(), 2, "sim cross_entropy expects [rows, classes]");
        let (rows, cols) = (sh[0], sh[1]);
        let lv = self.native_value(logits);
        let tv: Vec<i64> = self
            .native_value(targets)
            .iter()
            .map(|&t| t as i64)
            .collect();

        let mut softmax = vec![0.0_f32; rows * cols];
        let mut loss = 0.0_f32;
        let mut counted = 0.0_f32;
        for i in 0..rows {
            let row = &lv[i * cols..(i + 1) * cols];
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut denom = 0.0_f32;
            for j in 0..cols {
                let e = (row[j] - max).exp();
                softmax[i * cols + j] = e;
                denom += e;
            }
            for j in 0..cols {
                softmax[i * cols + j] /= denom;
            }
            let t = tv[i];
            if t != ignore_index {
                let p = softmax[i * cols + t as usize].max(1.0e-12);
                loss -= p.ln();
                counted += 1.0;
            }
        }
        let mean = if counted > 0.0 { loss / counted } else { 0.0 };
        self.push_node(
            vec![mean],
            Vec::new(),
            Op::CrossEntropy {
                logits,
                rows,
                cols,
                targets: tv,
                softmax,
            },
        )
    }

    pub(crate) fn backward(&mut self, loss: RawHandle) {
        let (tag, li) = dec(loss);
        assert_eq!(tag, TAG_NODE, "sim backward: loss must be a step tensor");
        // Zero node grads (params accumulate across micro-batches; nodes are per-pass).
        for n in &mut self.nodes {
            for g in &mut n.grad {
                *g = 0.0;
            }
        }
        self.nodes[li].grad = vec![1.0; self.nodes[li].value.len().max(1)];

        for k in (0..self.nodes.len()).rev() {
            let g = self.nodes[k].grad.clone();
            if g.iter().all(|&x| x == 0.0) {
                continue;
            }
            let op = self.nodes[k].op.clone();
            match op {
                Op::Const => {}
                Op::MatMul { a, b, m, k: kk, n } => {
                    let av = self.native_value(a);
                    let bv = self.native_value(b);
                    let g_a = mm(&g, &transpose(&bv, kk, n), m, n, kk); // [m,n]·[n,k]
                    let g_b = mm(&transpose(&av, m, kk), &g, kk, m, n); // [k,m]·[m,n]
                    self.add_native_grad(a, &g_a);
                    self.add_native_grad(b, &g_b);
                }
                Op::Add { a, b } => {
                    self.add_native_grad(a, &g);
                    self.add_native_grad(b, &g);
                }
                Op::AddBias { a, b, rows, cols } => {
                    self.add_native_grad(a, &g);
                    let mut gb = vec![0.0_f32; cols];
                    for i in 0..rows {
                        for j in 0..cols {
                            gb[j] += g[i * cols + j];
                        }
                    }
                    self.add_native_grad(b, &gb);
                }
                Op::Sub { a, b } => {
                    self.add_native_grad(a, &g);
                    let neg: Vec<f32> = g.iter().map(|&x| -x).collect();
                    self.add_native_grad(b, &neg);
                }
                Op::Mul { a, b } => {
                    let av = self.native_value(a);
                    let bv = self.native_value(b);
                    let ga: Vec<f32> = g.iter().zip(bv.iter()).map(|(&x, &y)| x * y).collect();
                    let gb: Vec<f32> = g.iter().zip(av.iter()).map(|(&x, &y)| x * y).collect();
                    self.add_native_grad(a, &ga);
                    self.add_native_grad(b, &gb);
                }
                Op::MulS { x, s } => {
                    let gx: Vec<f32> = g.iter().map(|&v| v * s).collect();
                    self.add_native_grad(x, &gx);
                }
                Op::Relu { x } => {
                    let xv = self.native_value(x);
                    let gx: Vec<f32> = g
                        .iter()
                        .zip(xv.iter())
                        .map(|(&gv, &xe)| if xe > 0.0 { gv } else { 0.0 })
                        .collect();
                    self.add_native_grad(x, &gx);
                }
                Op::CrossEntropy {
                    logits,
                    rows,
                    cols,
                    targets,
                    softmax,
                } => {
                    let upstream = g[0];
                    let counted = targets.iter().filter(|&&t| t >= 0).count().max(1) as f32;
                    let mut gl = vec![0.0_f32; rows * cols];
                    for i in 0..rows {
                        let t = targets[i];
                        if t < 0 {
                            continue;
                        }
                        for j in 0..cols {
                            let mut d = softmax[i * cols + j];
                            if j == t as usize {
                                d -= 1.0;
                            }
                            gl[i * cols + j] = upstream * d / counted;
                        }
                    }
                    self.add_native_grad(logits, &gl);
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn adamw_step(
        &mut self,
        p: RawHandle,
        g: RawHandle,
        m: RawHandle,
        v: RawHandle,
        step: u32,
        lr: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
        wd: f64,
    ) {
        let grad = self.native_value(g);
        let (_, pi) = dec(p);
        let (_, mi) = dec(m);
        let (_, vi) = dec(v);
        let t = step.max(1) as i32;
        let bc1 = 1.0 - beta1.powi(t);
        let bc2 = 1.0 - beta2.powi(t);
        let n = self.params[pi].master.len();
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            let gi = grad[i] as f64;
            let mut mi_v = self.persistents[mi].data[i] as f64;
            let mut vi_v = self.persistents[vi].data[i] as f64;
            mi_v = beta1 * mi_v + (1.0 - beta1) * gi;
            vi_v = beta2 * vi_v + (1.0 - beta2) * gi * gi;
            self.persistents[mi].data[i] = mi_v as f32;
            self.persistents[vi].data[i] = vi_v as f32;
            let mhat = mi_v / bc1;
            let vhat = vi_v / bc2;
            let mut w = self.params[pi].master[i] as f64;
            w -= lr * wd * w; // decoupled weight decay (AdamW)
            w -= lr * mhat / (vhat.sqrt() + eps);
            self.params[pi].master[i] = w as f32;
        }
        self.params[pi].storage = self.params[pi].master.clone();
    }

    // batch -------------------------------------------------------------------------------------

    pub(crate) fn batch_tokens(&mut self, b: RawHandle) -> RawHandle {
        let (_, idx) = dec(b);
        let bv = &self.batches[idx];
        let (batch, seq) = (bv.batch as usize, bv.seq as usize);
        let value: Vec<f32> = bv.tokens.iter().map(|&t| t as f32).collect();
        self.push_node(value, vec![batch, seq], Op::Const)
    }

    pub(crate) fn batch_size(&mut self, b: RawHandle) -> u32 {
        let (_, idx) = dec(b);
        self.batches[idx].batch
    }

    pub(crate) fn batch_seq_len(&mut self, b: RawHandle) -> u32 {
        let (_, idx) = dec(b);
        self.batches[idx].seq
    }

    // readout -----------------------------------------------------------------------------------

    pub(crate) fn scalar(&mut self, x: RawHandle) -> f64 {
        let (tag, idx) = dec(x);
        let v = match tag {
            TAG_DET => &self.dets[idx].data,
            TAG_DETPERSIST => &self.det_persistents[idx].data,
            TAG_NODE => &self.nodes[idx].value,
            _ => panic!("sim: scalar on non-tensor handle"),
        };
        assert_eq!(v.len(), 1, "sim scalar: numel must be 1");
        f64::from(v[0])
    }

    pub(crate) fn metric(&mut self, name: &str, x: RawHandle) {
        let val = self.scalar_relaxed(x);
        self.metrics.push((name.to_string(), val));
    }

    fn scalar_relaxed(&self, x: RawHandle) -> f32 {
        let (tag, idx) = dec(x);
        let v = match tag {
            TAG_DET => &self.dets[idx].data,
            TAG_DETPERSIST => &self.det_persistents[idx].data,
            TAG_NODE => &self.nodes[idx].value,
            TAG_PARAM => &self.params[idx].storage,
            TAG_PERSIST => &self.persistents[idx].data,
            _ => return 0.0,
        };
        v.first().copied().unwrap_or(0.0)
    }

    pub(crate) fn log(&mut self, _level: u32, _msg: &str) {}

    pub(crate) fn abi_minor(&mut self) -> u32 {
        crate::DA_ABI_MINOR
    }

    // update container --------------------------------------------------------------------------

    pub(crate) fn upd_new(&mut self) -> RawHandle {
        self.containers.push(Container {
            sections: Vec::new(),
        });
        enc(TAG_UPD, self.containers.len() - 1)
    }

    pub(crate) fn upd_push_bytes(&mut self, u: RawHandle, data: &[u8]) {
        let (_, idx) = dec(u);
        self.containers[idx]
            .sections
            .push(Section::Bytes(data.to_vec()));
    }

    pub(crate) fn upd_push_tensor(&mut self, u: RawHandle, x: RawHandle) {
        let data = self.native_value(x);
        let shape = self.native_shape(x);
        let (_, idx) = dec(u);
        self.containers[idx]
            .sections
            .push(Section::Tensor { data, shape });
    }

    fn staged_section(&self, i: u32, s: u32) -> &Section {
        let ci = self.staged[i as usize];
        &self.containers[ci].sections[s as usize]
    }

    pub(crate) fn upd_sections(&mut self, i: u32) -> u32 {
        let ci = self.staged[i as usize];
        self.containers[ci].sections.len() as u32
    }

    pub(crate) fn upd_kind(&mut self, i: u32, s: u32) -> u32 {
        match self.staged_section(i, s) {
            Section::Bytes(_) => 0,
            Section::Tensor { .. } => 1,
        }
    }

    pub(crate) fn upd_bytes_len(&mut self, i: u32, s: u32) -> u32 {
        match self.staged_section(i, s) {
            Section::Bytes(b) => b.len() as u32,
            Section::Tensor { .. } => 0,
        }
    }

    pub(crate) fn upd_read_bytes(&mut self, i: u32, s: u32, dst: &mut [u8]) -> u32 {
        match self.staged_section(i, s) {
            Section::Bytes(b) => {
                let n = b.len().min(dst.len());
                dst[..n].copy_from_slice(&b[..n]);
                n as u32
            }
            Section::Tensor { .. } => 0,
        }
    }

    pub(crate) fn upd_tensor(&mut self, i: u32, s: u32) -> RawHandle {
        let (data, shape) = match self.staged_section(i, s) {
            Section::Tensor { data, shape } => (data.clone(), shape.clone()),
            Section::Bytes(_) => panic!("sim: upd_tensor on a bytes section"),
        };
        self.push_det(data, shape)
    }

    // det lane ----------------------------------------------------------------------------------

    pub(crate) fn det_zeros(&mut self, dims: &[u32]) -> RawHandle {
        let n = numel(dims);
        self.push_det(vec![0.0; n], dims.iter().map(|&d| d as usize).collect())
    }

    pub(crate) fn det_sum(&mut self, handles: &[RawHandle]) -> RawHandle {
        let vecs: Vec<Vec<f32>> = handles.iter().map(|&h| self.det_value(h)).collect();
        let refs: Vec<&[f32]> = vecs.iter().map(Vec::as_slice).collect();
        let out = det_core::det_sum(&refs).expect("det_sum shapes must agree");
        let shape = handles
            .first()
            .map(|&h| self.det_shape(h))
            .unwrap_or_default();
        self.push_det(out, shape)
    }

    fn det_shape(&self, h: RawHandle) -> Vec<usize> {
        let (tag, idx) = dec(h);
        match tag {
            TAG_DET => self.dets[idx].shape.clone(),
            TAG_DETPERSIST => self.det_persistents[idx].shape.clone(),
            _ => Vec::new(),
        }
    }

    pub(crate) fn det_scale(&mut self, x: RawHandle, alpha: f64) -> RawHandle {
        let out = det_core::det_scale(&self.det_value(x), alpha);
        let shape = self.det_shape(x);
        self.push_det(out, shape)
    }

    pub(crate) fn det_l2norm(&mut self, x: RawHandle) -> f64 {
        f64::from(det_core::det_l2norm(&self.det_value(x)))
    }

    pub(crate) fn det_sign(&mut self, x: RawHandle) -> RawHandle {
        let out = det_core::det_sign(&self.det_value(x));
        let shape = self.det_shape(x);
        self.push_det(out, shape)
    }

    pub(crate) fn det_add(&mut self, a: RawHandle, b: RawHandle) -> RawHandle {
        let out =
            det_core::det_add(&self.det_value(a), &self.det_value(b)).expect("det_add shapes");
        let shape = self.det_shape(a);
        self.push_det(out, shape)
    }

    pub(crate) fn det_sub(&mut self, a: RawHandle, b: RawHandle) -> RawHandle {
        let out =
            det_core::det_sub(&self.det_value(a), &self.det_value(b)).expect("det_sub shapes");
        let shape = self.det_shape(a);
        self.push_det(out, shape)
    }

    pub(crate) fn det_mul(&mut self, a: RawHandle, b: RawHandle) -> RawHandle {
        let out =
            det_core::det_mul(&self.det_value(a), &self.det_value(b)).expect("det_mul shapes");
        let shape = self.det_shape(a);
        self.push_det(out, shape)
    }

    pub(crate) fn det_absmax_unpack(
        &mut self,
        packed: RawHandle,
        chunk: u32,
        bits: u32,
    ) -> RawHandle {
        let bytes: Vec<u8> = self.det_value(packed).iter().map(|&f| f as u8).collect();
        let out = det_core::det_absmax_unpack(&bytes, chunk as usize, bits)
            .expect("det_absmax_unpack layout");
        let n = out.len();
        self.push_det(out, vec![n])
    }

    pub(crate) fn det_chunk_scatter_add(
        &mut self,
        acc: RawHandle,
        vals: RawHandle,
        idx: RawHandle,
        chunk: u32,
    ) {
        let vals_v = self.det_value(vals);
        let idx_v: Vec<u32> = self.det_value(idx).iter().map(|&f| f as u32).collect();
        let (_, ai) = dec(acc);
        det_core::det_chunk_scatter_add(&mut self.dets[ai].data, &vals_v, &idx_v, chunk as usize)
            .expect("det_chunk_scatter_add layout");
    }

    pub(crate) fn det_chunk_scatter(
        &mut self,
        vals: RawHandle,
        idx: RawHandle,
        chunk: u32,
        dims: &[u32],
    ) -> RawHandle {
        let vals_v = self.det_value(vals);
        let idx_v: Vec<u32> = self.det_value(idx).iter().map(|&f| f as u32).collect();
        let out = det_core::det_chunk_scatter(&vals_v, &idx_v, chunk as usize, numel(dims))
            .expect("det_chunk_scatter layout");
        self.push_det(out, dims.iter().map(|&d| d as usize).collect())
    }

    pub(crate) fn det_assign(&mut self, dst: RawHandle, src: RawHandle) {
        let v = self.det_value(src);
        let (tag, idx) = dec(dst);
        match tag {
            TAG_DETPERSIST => self.det_persistents[idx].data = v,
            TAG_DET => self.dets[idx].data = v,
            _ => panic!("sim: det_assign target must be a det persistent"),
        }
    }

    pub(crate) fn det_param(&mut self, p: RawHandle) -> RawHandle {
        let (_, idx) = dec(p);
        let v = self.params[idx].round_base.clone();
        let shape = self.params[idx].shape.clone();
        self.push_det(v, shape)
    }

    pub(crate) fn det_reset_param_to_base(&mut self, p: RawHandle) {
        let (_, idx) = dec(p);
        self.params[idx].master = self.params[idx].round_base.clone();
        self.params[idx].storage = self.params[idx].master.clone();
    }

    pub(crate) fn det_axpy_param(&mut self, p: RawHandle, x: RawHandle, alpha: f64) {
        let xv = self.det_value(x);
        let (_, idx) = dec(p);
        det_core::det_axpy(&mut self.params[idx].master, alpha, &xv)
            .expect("det_axpy_param shapes must agree");
        self.params[idx].storage = self.params[idx].master.clone();
    }
}
