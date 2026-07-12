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
    Embedding {
        w: RawHandle,
        ids: Vec<usize>,
        vocab: usize,
        d: usize,
    },
    Rmsnorm {
        x: RawHandle,
        w: RawHandle,
        rows: usize,
        d: usize,
        inv_rms: Vec<f32>,
    },
    Silu {
        x: RawHandle,
    },
    Softmax {
        x: RawHandle,
        outer: usize,
        dimlen: usize,
        inner: usize,
        probs: Vec<f32>,
    },
    Rope {
        x: RawHandle,
        rows: usize,
        seq: usize,
        hd: usize,
        pos_start: usize,
        theta: f32,
        interleaved: bool,
    },
    FlashAttn {
        q: RawHandle,
        k: RawHandle,
        v: RawHandle,
        bh: usize,
        s: usize,
        d: usize,
        scale: f32,
        probs: Vec<f32>,
    },
    Reshape {
        x: RawHandle,
    },
    Transpose {
        x: RawHandle,
        d0: usize,
        d1: usize,
        shape_in: Vec<usize>,
    },
    Slice {
        x: RawHandle,
        dim: usize,
        start: usize,
        shape_in: Vec<usize>,
        shape_out: Vec<usize>,
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

/// The current accumulated fp32 gradient of a registered param (inspection; HOST-9 / autodiff
/// checks). Reflects everything `backward@1` has accumulated since the last `zero_grads@1`.
#[must_use]
pub fn param_grad(name: &str) -> Option<Vec<f32>> {
    with(|s| {
        s.params
            .iter()
            .find(|p| p.name == name)
            .map(|p| p.grad.clone())
    })
}

/// The metrics reported via `metric@1` this run.
#[must_use]
pub fn metrics() -> Vec<(String, f32)> {
    with(|s| s.metrics.clone())
}

/// The element/byte length of section `s` of a built update container (test inspection: profile
/// compression-ratio assertions). Packed `U8` payloads are stored one byte per element, so a tensor
/// section's element count equals its wire byte count.
#[must_use]
pub fn section_len(ub: &UpdateBuilder, s: usize) -> usize {
    let (_, idx) = dec(ub.handle());
    with(|st| match &st.containers[idx].sections[s] {
        Section::Bytes(b) => b.len(),
        Section::Tensor { data, .. } => data.len(),
    })
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

fn numel_usz(shape: &[usize]) -> usize {
    shape.iter().product()
}

fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Swap axes `d0`/`d1` of a row-major tensor (used by transpose forward + backward).
fn permute_axes(data: &[f32], shape_in: &[usize], d0: usize, d1: usize) -> Vec<f32> {
    let mut shape_out = shape_in.to_vec();
    shape_out.swap(d0, d1);
    let sin = row_major_strides(shape_in);
    let sout = row_major_strides(&shape_out);
    let mut out = vec![0.0_f32; data.len()];
    let rank = shape_in.len();
    let mut coord = vec![0usize; rank];
    for (flat, &v) in data.iter().enumerate() {
        // decode flat → coord in the INPUT layout
        let mut rem = flat;
        for r in 0..rank {
            coord[r] = rem / sin[r];
            rem %= sin[r];
        }
        // swap coords, encode into OUTPUT layout
        coord.swap(d0, d1);
        let mut out_flat = 0usize;
        for r in 0..rank {
            out_flat += coord[r] * sout[r];
        }
        out[out_flat] = v;
        coord.swap(d0, d1); // restore for reuse
    }
    out
}

/// Copy the `start..end` sub-range along `dim` of a row-major tensor.
fn slice_dim(data: &[f32], shape_in: &[usize], dim: usize, start: usize, end: usize) -> Vec<f32> {
    let mut shape_out = shape_in.to_vec();
    shape_out[dim] = end - start;
    let sin = row_major_strides(shape_in);
    let sout = row_major_strides(&shape_out);
    let rank = shape_in.len();
    let mut out = vec![0.0_f32; numel_usz(&shape_out)];
    let mut coord = vec![0usize; rank];
    for (flat, o) in out.iter_mut().enumerate() {
        let mut rem = flat;
        for r in 0..rank {
            coord[r] = rem / sout[r];
            rem %= sout[r];
        }
        let mut in_flat = 0usize;
        for r in 0..rank {
            let c = if r == dim { coord[r] + start } else { coord[r] };
            in_flat += c * sin[r];
        }
        *o = data[in_flat];
    }
    out
}

/// Scatter `grad` (the slice-output grad) back into an input-shaped zero buffer.
fn unslice_dim(grad: &[f32], shape_in: &[usize], dim: usize, start: usize, end: usize) -> Vec<f32> {
    let mut shape_out = shape_in.to_vec();
    shape_out[dim] = end - start;
    let sin = row_major_strides(shape_in);
    let sout = row_major_strides(&shape_out);
    let rank = shape_in.len();
    let mut full = vec![0.0_f32; numel_usz(shape_in)];
    let mut coord = vec![0usize; rank];
    for (flat, &gv) in grad.iter().enumerate() {
        let mut rem = flat;
        for r in 0..rank {
            coord[r] = rem / sout[r];
            rem %= sout[r];
        }
        let mut in_flat = 0usize;
        for r in 0..rank {
            let c = if r == dim { coord[r] + start } else { coord[r] };
            in_flat += c * sin[r];
        }
        full[in_flat] = gv;
    }
    full
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

    // -- Wave-2 NN / shape forward (autodiff-recorded) ------------------------------------------

    pub(crate) fn embedding(&mut self, w: RawHandle, ids: RawHandle) -> RawHandle {
        let wsh = self.native_shape(w);
        assert_eq!(wsh.len(), 2, "sim embedding weight is [vocab, d]");
        let (vocab, d) = (wsh[0], wsh[1]);
        let wv = self.native_value(w);
        let idsh = self.native_shape(ids);
        let ids_usize: Vec<usize> = self.native_value(ids).iter().map(|&f| f as usize).collect();
        let mut value = Vec::with_capacity(ids_usize.len() * d);
        for &id in &ids_usize {
            let base = id * d;
            value.extend_from_slice(&wv[base..base + d]);
        }
        let mut shape = idsh;
        shape.push(d);
        self.push_node(
            value,
            shape,
            Op::Embedding {
                w,
                ids: ids_usize,
                vocab,
                d,
            },
        )
    }

    pub(crate) fn rmsnorm(&mut self, x: RawHandle, w: RawHandle, eps: f64) -> RawHandle {
        let shape = self.native_shape(x);
        let d = *shape.last().expect("rmsnorm input has rank >= 1");
        let rows = numel_usz(&shape) / d;
        let xv = self.native_value(x);
        let wv = self.native_value(w);
        let eps = eps as f32;
        let mut value = vec![0.0_f32; xv.len()];
        let mut inv_rms = vec![0.0_f32; rows];
        for r in 0..rows {
            let row = &xv[r * d..(r + 1) * d];
            let ms = row.iter().map(|&v| v * v).sum::<f32>() / d as f32;
            let inv = 1.0 / (ms + eps).sqrt();
            inv_rms[r] = inv;
            for i in 0..d {
                value[r * d + i] = row[i] * inv * wv[i];
            }
        }
        self.push_node(
            value,
            shape,
            Op::Rmsnorm {
                x,
                w,
                rows,
                d,
                inv_rms,
            },
        )
    }

    pub(crate) fn silu(&mut self, x: RawHandle) -> RawHandle {
        let shape = self.native_shape(x);
        let value: Vec<f32> = self
            .native_value(x)
            .iter()
            .map(|&v| v / (1.0 + (-v).exp()))
            .collect();
        self.push_node(value, shape, Op::Silu { x })
    }

    pub(crate) fn softmax(&mut self, x: RawHandle, dim: u32) -> RawHandle {
        let shape = self.native_shape(x);
        let dim = dim as usize;
        let dimlen = shape[dim];
        let inner: usize = shape[dim + 1..].iter().product();
        let outer: usize = shape[..dim].iter().product();
        let xv = self.native_value(x);
        let mut probs = vec![0.0_f32; xv.len()];
        for o in 0..outer {
            for i in 0..inner {
                let base = o * dimlen * inner + i;
                let mut max = f32::NEG_INFINITY;
                for t in 0..dimlen {
                    max = max.max(xv[base + t * inner]);
                }
                let mut denom = 0.0_f32;
                for t in 0..dimlen {
                    let e = (xv[base + t * inner] - max).exp();
                    probs[base + t * inner] = e;
                    denom += e;
                }
                for t in 0..dimlen {
                    probs[base + t * inner] /= denom;
                }
            }
        }
        self.push_node(
            probs.clone(),
            shape,
            Op::Softmax {
                x,
                outer,
                dimlen,
                inner,
                probs,
            },
        )
    }

    pub(crate) fn rope(
        &mut self,
        x: RawHandle,
        pos_start: u32,
        theta: f64,
        interleaved: u32,
    ) -> RawHandle {
        let shape = self.native_shape(x);
        let rank = shape.len();
        let (seq, hd) = (shape[rank - 2], shape[rank - 1]);
        let rows = numel_usz(&shape) / hd;
        let xv = self.native_value(x);
        let interleaved = interleaved != 0;
        let theta = theta as f32;
        let mut value = xv.clone();
        for r in 0..rows {
            let pos = (pos_start as usize + (r % seq)) as f32;
            for j in 0..hd / 2 {
                let freq = 1.0 / theta.powf(2.0 * j as f32 / hd as f32);
                let angle = pos * freq;
                let (c, s) = (angle.cos(), angle.sin());
                let (ia, ib) = if interleaved {
                    (2 * j, 2 * j + 1)
                } else {
                    (j, j + hd / 2)
                };
                let (a, b) = (xv[r * hd + ia], xv[r * hd + ib]);
                value[r * hd + ia] = a * c - b * s;
                value[r * hd + ib] = a * s + b * c;
            }
        }
        self.push_node(
            value,
            shape,
            Op::Rope {
                x,
                rows,
                seq,
                hd,
                pos_start: pos_start as usize,
                theta,
                interleaved,
            },
        )
    }

    pub(crate) fn flash_attn(
        &mut self,
        q: RawHandle,
        k: RawHandle,
        v: RawHandle,
        causal: u32,
        scale: f64,
    ) -> RawHandle {
        let shape = self.native_shape(q);
        assert_eq!(shape.len(), 4, "sim flash_attn expects [b, h, s, d]");
        let (b, h, s, d) = (shape[0], shape[1], shape[2], shape[3]);
        let bh = b * h;
        let scale = scale as f32;
        let causal = causal != 0;
        let qv = self.native_value(q);
        let kv = self.native_value(k);
        let vv = self.native_value(v);
        let mut out = vec![0.0_f32; qv.len()];
        let mut probs = vec![0.0_f32; bh * s * s];
        for g in 0..bh {
            let base = g * s * d;
            let pbase = g * s * s;
            for i in 0..s {
                // scores over j, with causal mask.
                let jmax = if causal { i + 1 } else { s };
                let mut scores = vec![f32::NEG_INFINITY; s];
                let mut maxv = f32::NEG_INFINITY;
                for j in 0..jmax {
                    let mut dot = 0.0_f32;
                    for e in 0..d {
                        dot += qv[base + i * d + e] * kv[base + j * d + e];
                    }
                    let sc = dot * scale;
                    scores[j] = sc;
                    maxv = maxv.max(sc);
                }
                let mut denom = 0.0_f32;
                for j in 0..jmax {
                    let e = (scores[j] - maxv).exp();
                    probs[pbase + i * s + j] = e;
                    denom += e;
                }
                for j in 0..jmax {
                    probs[pbase + i * s + j] /= denom;
                }
                for e in 0..d {
                    let mut acc = 0.0_f32;
                    for j in 0..jmax {
                        acc += probs[pbase + i * s + j] * vv[base + j * d + e];
                    }
                    out[base + i * d + e] = acc;
                }
            }
        }
        self.push_node(
            out,
            shape,
            Op::FlashAttn {
                q,
                k,
                v,
                bh,
                s,
                d,
                scale,
                probs,
            },
        )
    }

    pub(crate) fn reshape(&mut self, x: RawHandle, dims: &[u32]) -> RawHandle {
        let value = self.native_value(x);
        self.push_node(
            value,
            dims.iter().map(|&d| d as usize).collect(),
            Op::Reshape { x },
        )
    }

    pub(crate) fn transpose(&mut self, x: RawHandle, d0: u32, d1: u32) -> RawHandle {
        let shape_in = self.native_shape(x);
        let (d0, d1) = (d0 as usize, d1 as usize);
        let xv = self.native_value(x);
        let mut shape_out = shape_in.clone();
        shape_out.swap(d0, d1);
        let value = permute_axes(&xv, &shape_in, d0, d1);
        self.push_node(
            value,
            shape_out,
            Op::Transpose {
                x,
                d0,
                d1,
                shape_in,
            },
        )
    }

    pub(crate) fn slice(&mut self, x: RawHandle, dim: u32, start: u32, end: u32) -> RawHandle {
        let shape_in = self.native_shape(x);
        let (dim, start, end) = (dim as usize, start as usize, end as usize);
        let mut shape_out = shape_in.clone();
        shape_out[dim] = end - start;
        let xv = self.native_value(x);
        let value = slice_dim(&xv, &shape_in, dim, start, end);
        self.push_node(
            value,
            shape_out.clone(),
            Op::Slice {
                x,
                dim,
                start,
                shape_in,
                shape_out,
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
                Op::Embedding { w, ids, vocab, d } => {
                    let mut gw = vec![0.0_f32; vocab * d];
                    for (r, &id) in ids.iter().enumerate() {
                        for i in 0..d {
                            gw[id * d + i] += g[r * d + i];
                        }
                    }
                    self.add_native_grad(w, &gw);
                }
                Op::Rmsnorm {
                    x,
                    w,
                    rows,
                    d,
                    inv_rms,
                } => {
                    let xv = self.native_value(x);
                    let wv = self.native_value(w);
                    let mut gx = vec![0.0_f32; rows * d];
                    let mut gw = vec![0.0_f32; d];
                    for r in 0..rows {
                        let inv = inv_rms[r];
                        let xrow = &xv[r * d..(r + 1) * d];
                        let grow = &g[r * d..(r + 1) * d];
                        // Σ_i g_i · w_i · x_i
                        let mut dot = 0.0_f32;
                        for i in 0..d {
                            dot += grow[i] * wv[i] * xrow[i];
                        }
                        let coef = inv * inv * inv / d as f32 * dot;
                        for i in 0..d {
                            gx[r * d + i] = inv * wv[i] * grow[i] - coef * xrow[i];
                            gw[i] += grow[i] * xrow[i] * inv;
                        }
                    }
                    self.add_native_grad(x, &gx);
                    self.add_native_grad(w, &gw);
                }
                Op::Silu { x } => {
                    let xv = self.native_value(x);
                    let gx: Vec<f32> = g
                        .iter()
                        .zip(xv.iter())
                        .map(|(&gv, &xe)| {
                            let sig = 1.0 / (1.0 + (-xe).exp());
                            gv * (sig * (1.0 + xe * (1.0 - sig)))
                        })
                        .collect();
                    self.add_native_grad(x, &gx);
                }
                Op::Softmax {
                    x,
                    outer,
                    dimlen,
                    inner,
                    probs,
                } => {
                    let mut gx = vec![0.0_f32; g.len()];
                    for o in 0..outer {
                        for i in 0..inner {
                            let base = o * dimlen * inner + i;
                            let mut dot = 0.0_f32;
                            for t in 0..dimlen {
                                dot += g[base + t * inner] * probs[base + t * inner];
                            }
                            for t in 0..dimlen {
                                let idx = base + t * inner;
                                gx[idx] = probs[idx] * (g[idx] - dot);
                            }
                        }
                    }
                    self.add_native_grad(x, &gx);
                }
                Op::Rope {
                    x,
                    rows,
                    seq,
                    hd,
                    pos_start,
                    theta,
                    interleaved,
                } => {
                    // Rotation is orthogonal: the input grad is the transpose (inverse) rotation.
                    let mut gx = vec![0.0_f32; g.len()];
                    for r in 0..rows {
                        let pos = (pos_start + (r % seq)) as f32;
                        for j in 0..hd / 2 {
                            let freq = 1.0 / theta.powf(2.0 * j as f32 / hd as f32);
                            let angle = pos * freq;
                            let (c, s) = (angle.cos(), angle.sin());
                            let (ia, ib) = if interleaved {
                                (2 * j, 2 * j + 1)
                            } else {
                                (j, j + hd / 2)
                            };
                            let (ga, gb) = (g[r * hd + ia], g[r * hd + ib]);
                            gx[r * hd + ia] = ga * c + gb * s;
                            gx[r * hd + ib] = -ga * s + gb * c;
                        }
                    }
                    self.add_native_grad(x, &gx);
                }
                Op::FlashAttn {
                    q,
                    k,
                    v,
                    bh,
                    s,
                    d,
                    scale,
                    probs,
                } => {
                    let qv = self.native_value(q);
                    let kv = self.native_value(k);
                    let vv = self.native_value(v);
                    let mut gq = vec![0.0_f32; qv.len()];
                    let mut gk = vec![0.0_f32; kv.len()];
                    let mut gv = vec![0.0_f32; vv.len()];
                    for grp in 0..bh {
                        let base = grp * s * d;
                        let pbase = grp * s * s;
                        for i in 0..s {
                            // dP[i][j] = Σ_e dO[i][e]·V[j][e]; dV[j][e] += P[i][j]·dO[i][e]
                            let mut dp = vec![0.0_f32; s];
                            for j in 0..s {
                                let p = probs[pbase + i * s + j];
                                if p == 0.0 {
                                    continue;
                                }
                                let mut dpj = 0.0_f32;
                                for e in 0..d {
                                    let go = g[base + i * d + e];
                                    dpj += go * vv[base + j * d + e];
                                    gv[base + j * d + e] += p * go;
                                }
                                dp[j] = dpj;
                            }
                            // dS[i][j] = P[i][j]·(dP[i][j] − Σ_j' P[i][j']·dP[i][j'])
                            let mut sum = 0.0_f32;
                            for j in 0..s {
                                sum += probs[pbase + i * s + j] * dp[j];
                            }
                            for j in 0..s {
                                let p = probs[pbase + i * s + j];
                                if p == 0.0 {
                                    continue;
                                }
                                let ds = p * (dp[j] - sum) * scale;
                                for e in 0..d {
                                    gq[base + i * d + e] += ds * kv[base + j * d + e];
                                    gk[base + j * d + e] += ds * qv[base + i * d + e];
                                }
                            }
                        }
                    }
                    self.add_native_grad(q, &gq);
                    self.add_native_grad(k, &gk);
                    self.add_native_grad(v, &gv);
                }
                Op::Reshape { x } => {
                    self.add_native_grad(x, &g);
                }
                Op::Transpose {
                    x,
                    d0,
                    d1,
                    shape_in,
                } => {
                    // g is in the OUTPUT (swapped) layout; swapping back yields the input layout.
                    let mut shape_out = shape_in.clone();
                    shape_out.swap(d0, d1);
                    let gx = permute_axes(&g, &shape_out, d0, d1);
                    self.add_native_grad(x, &gx);
                }
                Op::Slice {
                    x,
                    dim,
                    start,
                    shape_in,
                    shape_out,
                } => {
                    let end = start + shape_out[dim];
                    let gx = unslice_dim(&g, &shape_in, dim, start, end);
                    self.add_native_grad(x, &gx);
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

    // -- Wave-2 compression natives (native lane; no autodiff — local payload math) -------------

    pub(crate) fn topk_chunk(
        &mut self,
        x: RawHandle,
        chunk: u32,
        k: u32,
    ) -> (RawHandle, RawHandle) {
        let xv = self.native_value(x);
        let (vals, idx) =
            det_core::topk_chunk(&xv, chunk as usize, k as usize).expect("topk_chunk layout");
        let numel = xv.len();
        let n_chunks = numel / chunk.max(1) as usize;
        let shape = vec![n_chunks, k as usize];
        let vh = self.push_node(vals, shape.clone(), Op::Const);
        let ivals: Vec<f32> = idx.iter().map(|&i| i as f32).collect();
        let ih = self.push_node(ivals, shape, Op::Const);
        (vh, ih)
    }

    pub(crate) fn chunk_scatter(
        &mut self,
        vals: RawHandle,
        idx: RawHandle,
        chunk: u32,
        dims: &[u32],
    ) -> RawHandle {
        let valsv = self.native_value(vals);
        let idxv: Vec<u32> = self.native_value(idx).iter().map(|&f| f as u32).collect();
        let out = det_core::det_chunk_scatter(&valsv, &idxv, chunk as usize, numel(dims))
            .expect("chunk_scatter layout");
        self.push_node(out, dims.iter().map(|&d| d as usize).collect(), Op::Const)
    }

    pub(crate) fn absmax_pack(&mut self, x: RawHandle, chunk: u32, bits: u32) -> RawHandle {
        let xv = self.native_value(x);
        let packed = det_core::absmax_pack(&xv, chunk as usize, bits).expect("absmax_pack layout");
        let n = packed.len();
        let vals: Vec<f32> = packed.iter().map(|&b| f32::from(b)).collect();
        self.push_node(vals, vec![n], Op::Const)
    }

    pub(crate) fn absmax_unpack(
        &mut self,
        packed: RawHandle,
        chunk: u32,
        bits: u32,
        _dtype: u32,
    ) -> RawHandle {
        let bytes: Vec<u8> = self.native_value(packed).iter().map(|&f| f as u8).collect();
        let out = det_core::det_absmax_unpack(&bytes, chunk as usize, bits)
            .expect("absmax_unpack layout");
        let n = out.len();
        self.push_node(out, vec![n], Op::Const)
    }

    pub(crate) fn dct2(&mut self, x: RawHandle, tile: u32) -> RawHandle {
        let shape = self.native_shape(x);
        let out = det_core::dct2(&self.native_value(x), tile as usize).expect("dct2 layout");
        self.push_node(out, shape, Op::Const)
    }

    pub(crate) fn idct2(&mut self, x: RawHandle, tile: u32) -> RawHandle {
        let shape = self.native_shape(x);
        let out = det_core::idct2(&self.native_value(x), tile as usize).expect("idct2 layout");
        self.push_node(out, shape, Op::Const)
    }

    pub(crate) fn det_idct2(&mut self, x: RawHandle, tile: u32) -> RawHandle {
        let shape = self.det_shape(x);
        let out = det_core::idct2(&self.det_value(x), tile as usize).expect("det_idct2 layout");
        self.push_det(out, shape)
    }
}
