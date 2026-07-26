// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The module's **Logical Resource Plan** — what this training algorithm needs, in logical units.
//!
//! This is the whole of what the module says about resources. It names shapes, dtypes, lifetimes
//! and overlap groups; it names nothing physical. There is no backend here, no allocator, no
//! driver, no measured constant and no cost model, because a module that carried one would be
//! describing a machine it cannot see — and would have to be rebuilt and re-certified every time
//! that machine's driver changed, for reasons having nothing to do with training.
//!
//! ## How to read the terms
//!
//! Every size is symbolic over the admitted configuration, and one dimension is left **free**:
//! `micro_batch`. The module states the range it is willing to run at; the host prices that range
//! against the machine it actually has and hands back the value it chose. That is the only
//! quantity in this file whose value the module does not know at derivation time.
//!
//! Tensors are grouped by **class**, not listed one per layer: a 24-block decoder's blocks are
//! identical, so `n_layers` is a leading shape dimension rather than 24 copies of one declaration.
//! A grouped term covers every tensor in its class — the obligation is that no config-scaled
//! device tensor is missing, not that each has its own line.
//!
//! ## Why the peak is a maximum and not a sum
//!
//! The three linear-memory walks — the init expansion, the round's update walk, the round's ingest
//! walk — never overlap in time. The module used to declare their sum, reasoning that wasm linear
//! memory never shrinks. That conflates two quantities: the **page count** never shrinks, but the
//! **heap** inside those pages is an ordinary allocator and freed storage is reusable. What the
//! never-shrinking page count really means is that the peak, once reached, is permanent — so the
//! declaration must cover the peak, and the honest way to cover the allocator's inability to reuse
//! a freed block of the wrong size class is a **named fragmentation allowance**, not a sum that
//! hides one. The overlap groups below say which terms can be live together; the allowance says
//! what the size-class mismatch between those groups costs.

use daemon_vhc_proto::resource_plan::{
    Dimension, Domain, Dtype, Expr, Lifetime, LinearLifetime, LinearMemoryTerm,
    LogicalResourcePlan, OperationDecl, Retention, SelectionScope, TensorDecl, TransferDecl,
    TransferKind,
};

use super::model::ModelCfg;

/// The free dimension: the micro-batch the host selects out of the range the module offers.
pub const DIM_MICRO_BATCH: &str = "micro_batch";

/// The transient overlap group of one forward/backward/optimizer step.
const LIVE_STEP: &str = "training_step";
/// The transient overlap group of the round's state walks (export, update, ingest).
const LIVE_STATE_WALK: &str = "state_walk";
/// The transient overlap group of the one-off init expansion, before any round opens.
const LIVE_INIT: &str = "init_expansion";
/// The transient overlap group of a round's staged corpus window.
const LIVE_ROUND_STAGING: &str = "round_staging";

fn konst(v: u64) -> Expr {
    Expr::Const(v)
}

fn micro_batch() -> Expr {
    Expr::Dimension(DIM_MICRO_BATCH.to_string())
}

fn mul(a: Expr, b: Expr) -> Expr {
    Expr::Mul(Box::new(a), Box::new(b))
}

fn tensor(name: &str, shape: Vec<Expr>, dtype: Dtype, lifetime: Lifetime) -> TensorDecl {
    TensorDecl {
        name: name.to_string(),
        shape,
        dtype,
        layout: Vec::new(),
        lifetime,
    }
}

fn linear(name: &str, lifetime: LinearLifetime, bytes: Expr) -> LinearMemoryTerm {
    LinearMemoryTerm {
        name: name.to_string(),
        lifetime,
        bytes,
    }
}

/// The derived quantities every term is a function of. Fixed by the admitted configuration —
/// only the micro-batch is left free.
struct Geometry {
    params: u64,
    d_model: u64,
    qdim: u64,
    hidden: u64,
    vocab: u64,
    layers: u64,
    heads: u64,
    /// Positions predicted per sequence: `seq_len - 1`.
    span: u64,
    /// The state-walk window in elements.
    window_elems: u64,
    /// The round's inner-step count.
    steps_per_round: u64,
    /// Bytes of bookkeeping that scale with the parameter layout and the fold geometry.
    bookkeeping_bytes: u64,
    /// The module's config-independent linear-memory floor.
    baseline_bytes: u64,
}

impl Geometry {
    /// One micro-batch's predicted positions: `micro_batch × (seq_len - 1)`.
    fn rows(&self) -> Expr {
        mul(micro_batch(), konst(self.span))
    }
}

/// The device-resident tensors, by class.
fn tensors(g: &Geometry) -> Vec<TensorDecl> {
    let f32t = Dtype::F32;
    let mut out = vec![
        // -- persistent: the model and its optimizer state -------------------------------------
        //
        // The canonical parameters, the two AdamW moments, and the gradient buffer the backward
        // pass fills. The moments are optimizer retention because a restore without them forks
        // the committed trajectory even when the ingest-side digest still matches.
        tensor(
            "model_parameters",
            vec![konst(g.params)],
            f32t,
            Lifetime::Persistent(Retention::Run),
        ),
        tensor(
            "optimizer_first_moment",
            vec![konst(g.params)],
            f32t,
            Lifetime::Persistent(Retention::Optimizer),
        ),
        tensor(
            "optimizer_second_moment",
            vec![konst(g.params)],
            f32t,
            Lifetime::Persistent(Retention::Optimizer),
        ),
        // -- transient: one training step --------------------------------------------------------
        //
        // The gradient buffer is live from the backward pass to the optimizer step; it is a step
        // term rather than a persistent one because nothing reads it across steps.
        tensor(
            "step_gradients",
            vec![konst(g.params)],
            f32t,
            Lifetime::Transient(LIVE_STEP.to_string()),
        ),
        // The input and target position ids the forward pass indexes with.
        tensor(
            "step_input_ids",
            vec![g.rows()],
            Dtype::I64,
            Lifetime::Transient(LIVE_STEP.to_string()),
        ),
        tensor(
            "step_target_ids",
            vec![g.rows()],
            Dtype::I64,
            Lifetime::Transient(LIVE_STEP.to_string()),
        ),
        // The residual stream entering each block, plus the final normalized output. The autodiff
        // tape retains one per block, which is why `n_layers` is a shape dimension and not a
        // divisor.
        tensor(
            "step_residual_stream",
            vec![konst(g.layers + 1), g.rows(), konst(g.d_model)],
            f32t,
            Lifetime::Transient(LIVE_STEP.to_string()),
        ),
        // Query, key and value projections per block (three of them), and the attention output
        // before the output projection.
        tensor(
            "step_attention_projections",
            vec![konst(g.layers), konst(3), g.rows(), konst(g.qdim)],
            f32t,
            Lifetime::Transient(LIVE_STEP.to_string()),
        ),
        tensor(
            "step_attention_output",
            vec![konst(g.layers), g.rows(), konst(g.qdim)],
            f32t,
            Lifetime::Transient(LIVE_STEP.to_string()),
        ),
        // Dense causal attention: the score plane per head per block, retained for the backward
        // pass. This is the term that scales quadratically in the sequence, and naming it is the
        // point — an unnamed quadratic term is the residency defect this contract exists to make
        // impossible.
        tensor(
            "step_attention_scores",
            vec![
                konst(g.layers),
                mul(micro_batch(), konst(g.heads)),
                konst(g.span),
                konst(g.span),
            ],
            f32t,
            Lifetime::Transient(LIVE_STEP.to_string()),
        ),
        // The additive causal mask: one plane, shared across blocks and heads, built on device.
        tensor(
            "step_causal_mask",
            vec![konst(g.span), konst(g.span)],
            f32t,
            Lifetime::Transient(LIVE_STEP.to_string()),
        ),
        // The SwiGLU gate and up projections per block.
        tensor(
            "step_ffn_projections",
            vec![konst(g.layers), konst(2), g.rows(), konst(g.hidden)],
            f32t,
            Lifetime::Transient(LIVE_STEP.to_string()),
        ),
        // The tied-embedding logit plane and its log-softmax — the two terms that scale with the
        // vocabulary. The loss reads one value per row out of the second by index; it does not
        // build a third plane to do so.
        tensor(
            "step_logits",
            vec![g.rows(), konst(g.vocab)],
            f32t,
            Lifetime::Transient(LIVE_STEP.to_string()),
        ),
        tensor(
            "step_log_softmax",
            vec![g.rows(), konst(g.vocab)],
            f32t,
            Lifetime::Transient(LIVE_STEP.to_string()),
        ),
        tensor(
            "step_target_log_probabilities",
            vec![g.rows(), konst(1)],
            f32t,
            Lifetime::Transient(LIVE_STEP.to_string()),
        ),
        // -- transient: the round's state walks --------------------------------------------------
        //
        // Between rounds the step terms are dead and these are live: the windows the export,
        // update and ingest walks move through, bounded by the run-pinned window size rather than
        // by the family they traverse.
        tensor(
            "state_walk_window",
            vec![konst(g.window_elems)],
            f32t,
            Lifetime::Transient(LIVE_STATE_WALK.to_string()),
        ),
    ];
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

/// The operation families the plan admits, and how many of each can be in flight.
fn operations() -> Vec<OperationDecl> {
    let one = |name: &str, family: &str| OperationDecl {
        name: name.to_string(),
        family: family.to_string(),
        inputs: Vec::new(),
        outputs: Vec::new(),
        workspace_class: Some(family.to_string()),
        max_in_flight: Expr::Const(1),
    };
    let mut out = vec![
        one("attention_scores", "matmul"),
        one("attention_softmax", "softmax"),
        one("elementwise", "elementwise"),
        one("embedding_select", "gather"),
        one("logit_projection", "matmul"),
        one("loss_reduction", "reduction"),
        one("normalization", "reduction"),
        one("optimizer_step", "elementwise"),
        one("rotary_embedding", "elementwise"),
        one("target_gather", "gather"),
    ];
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

/// The bounded transfer windows. Every readback and every ingest the module performs is bounded
/// by a declared window plus framing — never by the object the window is part of.
fn transfers(g: &Geometry) -> Vec<TransferDecl> {
    let window_bytes = g.window_elems.saturating_mul(4);
    let mut out = vec![
        TransferDecl {
            name: "corpus_window_ingest".to_string(),
            kind: TransferKind::Ingest,
            window_bytes: Expr::Const(g.corpus_window_bytes()),
            max_in_flight: Expr::Const(g.steps_per_round.max(1)),
        },
        TransferDecl {
            name: "state_window_export".to_string(),
            kind: TransferKind::Export,
            window_bytes: Expr::Const(window_bytes),
            max_in_flight: Expr::Const(4),
        },
        TransferDecl {
            name: "state_window_readback".to_string(),
            kind: TransferKind::Readback,
            window_bytes: Expr::Const(window_bytes),
            max_in_flight: Expr::Const(4),
        },
    ];
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

impl Geometry {
    /// A round's staged corpus window, in bytes: the whole round's tokens, because a live round
    /// plans and stages its window before it trains the first step.
    fn corpus_window_bytes(&self) -> u64 {
        // Four bytes per token bounds both the widest manifest token width and the decoded `u32`.
        self.span
            .saturating_add(1)
            .saturating_mul(self.steps_per_round.max(1))
            .saturating_mul(4)
    }
}

/// The module's own linear-memory terms — the second of the two caps, and the only one the module
/// states directly.
fn linear_memory(g: &Geometry) -> Vec<LinearMemoryTerm> {
    let window_bytes = g.window_elems.saturating_mul(4);
    let mut out = vec![
        linear(
            "module_baseline",
            LinearLifetime::Persistent,
            konst(g.baseline_bytes),
        ),
        linear(
            "layout_bookkeeping",
            LinearLifetime::Persistent,
            konst(g.bookkeeping_bytes),
        ),
        // The seed expansion streams a window, its little-endian image, and the zeroed
        // error-feedback window it seals beside it. One-off, before any round.
        linear(
            "init_expansion_windows",
            LinearLifetime::Transient(LIVE_INIT.to_string()),
            konst(window_bytes.saturating_mul(3)),
        ),
        // The update and ingest walks, at their in-flight window counts.
        linear(
            "update_walk_windows",
            LinearLifetime::Transient(LIVE_STATE_WALK.to_string()),
            konst(window_bytes.saturating_mul(4).saturating_mul(4)),
        ),
        linear(
            "ingest_walk_windows",
            LinearLifetime::Transient(LIVE_STATE_WALK.to_string()),
            konst(window_bytes.saturating_mul(2).saturating_mul(4)),
        ),
        // The round's staged corpus window: fetched bytes and their decoded tokens, all resident
        // across the inner loop.
        linear(
            "round_staged_corpus",
            LinearLifetime::Transient(LIVE_ROUND_STAGING.to_string()),
            konst(g.corpus_window_bytes().saturating_mul(2)),
        ),
        // The position id rows one micro-batch's forward pass builds host-side.
        linear(
            "step_position_ids",
            LinearLifetime::Transient(LIVE_STEP.to_string()),
            mul(g.rows(), konst(16)),
        ),
    ];
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

/// The maximal sets of transient groups that can be live at once. Distinct sets are alternative
/// peak candidates; no listed set is a strict subset of another.
fn transient_live_sets() -> Vec<Vec<String>> {
    let mut out = vec![
        // Before any round: the expansion alone.
        vec![LIVE_INIT.to_string()],
        // Inside a round's inner loop: the staged window and one step.
        {
            let mut s = vec![LIVE_ROUND_STAGING.to_string(), LIVE_STEP.to_string()];
            s.sort();
            s
        },
        // At a round boundary: the staged window and the state walks. The step terms are dead —
        // the barrier defers a round's export behind its last step, and defers the next round's
        // open behind an in-flight fold.
        {
            let mut s = vec![LIVE_ROUND_STAGING.to_string(), LIVE_STATE_WALK.to_string()];
            s.sort();
            s
        },
    ];
    out.sort();
    out
}

/// The allocator's inability to reuse a freed block of the wrong size class, declared rather than
/// hidden inside a sum.
///
/// The overlap groups above hand off between phases whose block sizes differ by construction —
/// one window against four, against two-plus-per-peer rows, against a whole round's token window
/// — so a block freed by one phase frequently cannot satisfy the next phase's request. A quarter
/// of the transient peak is the declared allowance for that mismatch. It is a term the host and
/// the owner can see and refuse, which is the property a hidden sum did not have.
fn fragmentation_headroom(g: &Geometry) -> Expr {
    let transient_total: Vec<Expr> = linear_memory(g)
        .into_iter()
        .filter(|t| matches!(t.lifetime, LinearLifetime::Transient(_)))
        .map(|t| t.bytes)
        .collect();
    Expr::CeilDiv(Box::new(Expr::Add(transient_total)), 4)
}

/// Derive the module's Logical Resource Plan for one admitted configuration.
///
/// `micro_batch_max` is the largest micro-batch the run's authoring is willing to have selected;
/// the module offers every value from one up to it and lets the host choose.
#[must_use]
pub fn derive(
    cfg: &ModelCfg,
    micro_batch_max: u32,
    steps_per_round: u32,
    window_bytes: u64,
    bookkeeping_bytes: u64,
    baseline_bytes: u64,
) -> LogicalResourcePlan {
    let params: u64 = cfg.param_numels().iter().map(|&n| n as u64).sum();
    let g = Geometry {
        params,
        d_model: u64::from(cfg.d_model),
        qdim: u64::from(cfg.n_heads) * u64::from(cfg.head_dim),
        hidden: u64::from(cfg.ffn_mult) * u64::from(cfg.d_model),
        vocab: u64::from(cfg.vocab),
        layers: u64::from(cfg.n_layers),
        heads: u64::from(cfg.n_heads),
        span: u64::from(cfg.seq_len).saturating_sub(1).max(1),
        window_elems: (window_bytes / 4).max(1),
        steps_per_round: u64::from(steps_per_round),
        bookkeeping_bytes,
        baseline_bytes,
    };
    LogicalResourcePlan {
        // The reference trainer is uniform: every participant trains the same micro-batch, and
        // the outer update weights contributions equally. Heterogeneous selection would need a
        // normalization contract this module does not define, so it does not offer the scope.
        selection_scope: SelectionScope::UniformRun,
        equivalence_contract_hash: None,
        dimensions: vec![Dimension {
            name: DIM_MICRO_BATCH.to_string(),
            domain: Domain::UintRange {
                lo: 1,
                hi: u64::from(micro_batch_max.max(1)),
            },
        }],
        tensors: tensors(&g),
        operations: operations(),
        transfers: transfers(&g),
        linear_memory: linear_memory(&g),
        transient_live_sets: transient_live_sets(),
        linear_fragmentation_headroom: fragmentation_headroom(&g),
    }
}
