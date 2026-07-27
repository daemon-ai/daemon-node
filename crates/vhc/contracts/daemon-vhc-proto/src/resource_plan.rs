// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The **Logical Resource Plan** — a module's backend-neutral, parametric statement of what its
//! algorithm needs, in logical units (`docs/specs/vhc-architecture-spec.md` §9.6 `[RC-2]`,
//! `[RC-3]`, `[RC-4]`, `[RC-11]`, `[RC-12]`).
//!
//! A plan describes **demand**. It never describes a machine: no backend name, no allocator pool
//! size, no driver revision, no kernel workspace figure, no staging multiplier, no headroom for a
//! physical implementation. Those belong to the host's Backend Execution Profile, which is why
//! the profile type lives in a host crate and this one does not. Composing the two is the host
//! planner's job (`[RC-4]`); this module owns the plan's schema, its validation, and the
//! **planner semantics** — the derived logical byte sizes and the maximal-live-set peak
//! arithmetic — because those are properties of the plan, not discretion the profile may exercise.
//!
//! ## Shape
//!
//! Schema 1 is a closed, bounded canonical-CBOR document. Every list is unique and sorted, every
//! name resolves, every arithmetic expression is a term in a small total grammar over the plan's
//! own declared dimensions, and there are no unknown members or operators. Sizes are not authored:
//! a tensor's logical byte size is *derived* from its shape and its dtype spelling, so two
//! authors of the same tensor cannot disagree about how big it is.
//!
//! ## Bounded by construction
//!
//! Deriving a plan is capability-free, compute-free, allocation-free and execution-free, so its
//! execution budget is analytic rather than empirical: a fixed base plus a bounded cost per plan
//! node and per byte of encoded output, under an absolute ceiling
//! ([`plan_derivation_fuel`], [`PLAN_DERIVATION_FUEL_CEILING`]). A plan that could only be
//! produced by walking a materialized tensor graph is a mis-shaped plan, not a plan that needs a
//! larger budget.

use std::collections::{BTreeMap, BTreeSet};

use ciborium::value::Value;

use crate::bytes::Hash;
use crate::canonical::to_canonical_vec;
use crate::error::VhcProtoError;
use crate::hash::blake3_hash;

/// The schema this build authors and accepts.
pub const LOGICAL_RESOURCE_PLAN_SCHEMA: u64 = 1;

/// The hard ceiling on a plan's canonical encoding, in bytes (`[RC-12]`). A returned span longer
/// than this is rejected **before** it is read out of guest memory.
pub const LOGICAL_RESOURCE_PLAN_BYTES_MAX: usize = 1_048_576;

/// The hard ceiling on a plan's node count: dimensions, tensors, operations, transfers,
/// linear-memory terms and every nested expression node, summed (`[RC-12]`).
pub const LOGICAL_RESOURCE_PLAN_NODES_MAX: usize = 4_096;

/// The hard ceiling on declared dimensions (`[RC-12]`).
pub const LOGICAL_RESOURCE_PLAN_DIMENSIONS_MAX: usize = 256;

/// The maximum nesting depth of an expression (`[RC-12]`).
pub const LOGICAL_RESOURCE_PLAN_EXPR_DEPTH_MAX: usize = 64;

/// Every identifier in a plan is 1–64 UTF-8 bytes (`[RC-12]`).
pub const LOGICAL_RESOURCE_PLAN_IDENT_BYTES_MAX: usize = 64;

/// The fixed base of the derived plan-derivation budget (`[RC-4]`).
pub const PLAN_DERIVATION_FUEL_BASE: u64 = 4_000_000;

/// The bounded per-plan-node cost of the derived plan-derivation budget (`[RC-4]`).
pub const PLAN_DERIVATION_FUEL_PER_NODE: u64 = 16_384;

/// The bounded per-output-byte cost of the derived plan-derivation budget (`[RC-4]`).
pub const PLAN_DERIVATION_FUEL_PER_OUTPUT_BYTE: u64 = 64;

/// The absolute ceiling on the derived plan-derivation budget (`[RC-4]`). It is what the host
/// actually arms before calling the export, because the node and byte counts are not knowable
/// until the plan exists; the *derived* limit below is what the produced plan must then fit under.
pub const PLAN_DERIVATION_FUEL_CEILING: u64 = 256_000_000;

/// The derived derivation budget for a plan of `nodes` nodes encoding to `output_bytes`
/// (`[RC-4]`): a fixed base plus a bounded cost per node and per output byte, under an absolute
/// ceiling. This is a *limit computed from the plan*, never a constant raised until derivation
/// happens to fit.
#[must_use]
pub fn plan_derivation_fuel(nodes: u64, output_bytes: u64) -> u64 {
    PLAN_DERIVATION_FUEL_BASE
        .saturating_add(nodes.saturating_mul(PLAN_DERIVATION_FUEL_PER_NODE))
        .saturating_add(output_bytes.saturating_mul(PLAN_DERIVATION_FUEL_PER_OUTPUT_BYTE))
        .min(PLAN_DERIVATION_FUEL_CEILING)
}

/// Why a plan was refused. The two variants map onto the two typed ABI refusals: a malformed,
/// non-canonical, unresolved or physically-contaminated plan is `LogicalResourcePlanInvalid`; a
/// plan over a byte, node, depth or derived-fuel bound is `LogicalResourcePlanExceedsPolicy`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanRefusal {
    /// The plan is not a well-formed schema-1 document, or its semantics do not check out.
    Invalid(String),
    /// The plan is well-formed but breaches a declared bound.
    ExceedsPolicy(String),
}

impl PlanRefusal {
    /// The stable slug the host reports and the journal records.
    #[must_use]
    pub fn slug(&self) -> &'static str {
        match self {
            Self::Invalid(_) => "LogicalResourcePlanInvalid",
            Self::ExceedsPolicy(_) => "LogicalResourcePlanExceedsPolicy",
        }
    }

    /// The human-readable reason.
    #[must_use]
    pub fn detail(&self) -> &str {
        match self {
            Self::Invalid(d) | Self::ExceedsPolicy(d) => d,
        }
    }
}

impl std::fmt::Display for PlanRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.slug(), self.detail())
    }
}

impl std::error::Error for PlanRefusal {}

impl From<PlanRefusal> for VhcProtoError {
    fn from(value: PlanRefusal) -> Self {
        Self::Validation(value.to_string())
    }
}

fn invalid<T>(detail: impl Into<String>) -> Result<T, PlanRefusal> {
    Err(PlanRefusal::Invalid(detail.into()))
}

fn exceeds<T>(detail: impl Into<String>) -> Result<T, PlanRefusal> {
    Err(PlanRefusal::ExceedsPolicy(detail.into()))
}

// -- the closed dtype spelling -------------------------------------------------------------------

/// The closed dtype vocabulary. The spelling fixes the width: `bool1` is one bit, and every other
/// spelling carries its bit width as a numeric suffix. Nothing else is admissible, so a logical
/// byte size can be derived rather than authored.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Dtype {
    /// One bit per element.
    Bool1,
    /// Unsigned 8-bit.
    U8,
    /// Signed 8-bit.
    I8,
    /// Unsigned 16-bit.
    U16,
    /// Signed 16-bit.
    I16,
    /// IEEE-754 half.
    F16,
    /// bfloat16.
    Bf16,
    /// Unsigned 32-bit.
    U32,
    /// Signed 32-bit.
    I32,
    /// IEEE-754 single.
    F32,
    /// Unsigned 64-bit.
    U64,
    /// Signed 64-bit.
    I64,
    /// IEEE-754 double.
    F64,
}

impl Dtype {
    /// The element width in bits, fixed by the spelling.
    #[must_use]
    pub fn bits(self) -> u64 {
        match self {
            Self::Bool1 => 1,
            Self::U8 | Self::I8 => 8,
            Self::U16 | Self::I16 | Self::F16 | Self::Bf16 => 16,
            Self::U32 | Self::I32 | Self::F32 => 32,
            Self::U64 | Self::I64 | Self::F64 => 64,
        }
    }

    /// The canonical spelling.
    #[must_use]
    pub fn spelling(self) -> &'static str {
        match self {
            Self::Bool1 => "bool1",
            Self::U8 => "u8",
            Self::I8 => "i8",
            Self::U16 => "u16",
            Self::I16 => "i16",
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
            Self::U32 => "u32",
            Self::I32 => "i32",
            Self::F32 => "f32",
            Self::U64 => "u64",
            Self::I64 => "i64",
            Self::F64 => "f64",
        }
    }

    fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "bool1" => Self::Bool1,
            "u8" => Self::U8,
            "i8" => Self::I8,
            "u16" => Self::U16,
            "i16" => Self::I16,
            "f16" => Self::F16,
            "bf16" => Self::Bf16,
            "u32" => Self::U32,
            "i32" => Self::I32,
            "f32" => Self::F32,
            "u64" => Self::U64,
            "i64" => Self::I64,
            "f64" => Self::F64,
            _ => return None,
        })
    }
}

/// What a persistent tensor is retained for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Retention {
    /// Live for the whole run.
    Run,
    /// Retained across the forward pass.
    Forward,
    /// Retained across the backward pass.
    Backward,
    /// Retained for autodiff.
    Autodiff,
    /// Retained by the optimizer.
    Optimizer,
    /// Retained for checkpointing.
    Checkpoint,
}

impl Retention {
    /// The canonical spelling.
    #[must_use]
    pub fn spelling(self) -> &'static str {
        match self {
            Self::Run => "run",
            Self::Forward => "forward",
            Self::Backward => "backward",
            Self::Autodiff => "autodiff",
            Self::Optimizer => "optimizer",
            Self::Checkpoint => "checkpoint",
        }
    }

    fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "run" => Self::Run,
            "forward" => Self::Forward,
            "backward" => Self::Backward,
            "autodiff" => Self::Autodiff,
            "optimizer" => Self::Optimizer,
            "checkpoint" => Self::Checkpoint,
            _ => return None,
        })
    }
}

/// A transfer's direction across the capability boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum TransferKind {
    /// Bytes entering the module.
    Ingest,
    /// Bytes the module seals out.
    Export,
    /// A ranged read of a host or device resource.
    Readback,
}

impl TransferKind {
    /// The canonical spelling.
    #[must_use]
    pub fn spelling(self) -> &'static str {
        match self {
            Self::Ingest => "ingest",
            Self::Export => "export",
            Self::Readback => "readback",
        }
    }

    fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "ingest" => Self::Ingest,
            "export" => Self::Export,
            "readback" => Self::Readback,
            _ => return None,
        })
    }
}

/// How broadly one selected logical configuration must apply (`[RC-11]`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum SelectionScope {
    /// One value, frozen at authoring, that every participant consumes verbatim. The default.
    UniformRun,
    /// Each admitting host selects locally, inside the frozen choice set and under the module's
    /// declared normalization/equivalence contract.
    PerParticipant,
}

impl SelectionScope {
    /// The canonical spelling.
    #[must_use]
    pub fn spelling(self) -> &'static str {
        match self {
            Self::UniformRun => "uniform-run",
            Self::PerParticipant => "per-participant",
        }
    }

    /// Parse a canonical spelling.
    #[must_use]
    pub fn parse_spelling(s: &str) -> Option<Self> {
        Some(match s {
            "uniform-run" => Self::UniformRun,
            "per-participant" => Self::PerParticipant,
            _ => return None,
        })
    }
}

/// The domain of a declared dimension: either a numeric range or a closed set of spellings.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Domain {
    /// An inclusive `lo..=hi` unsigned range. Referenced numerically through
    /// [`Expr::Dimension`].
    UintRange {
        /// Inclusive lower bound.
        lo: u64,
        /// Inclusive upper bound.
        hi: u64,
    },
    /// A closed set of spellings, referenced only through [`Expr::Select`].
    Enum(Vec<String>),
}

/// One declared dimension of the plan's parametric freedom.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Dimension {
    /// The dimension's identifier.
    pub name: String,
    /// Its domain.
    pub domain: Domain,
}

/// The plan's arithmetic grammar over its own dimensions. Every operator is monotone
/// non-decreasing in its dimension arguments, which is what makes the domain-minimum evaluation
/// a sound lower bound.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Expr {
    /// A literal.
    Const(u64),
    /// The value bound to a uint-range dimension.
    Dimension(String),
    /// One arm per value of an enum dimension.
    Select {
        /// The enum dimension being selected on.
        dimension: String,
        /// Exactly one arm per enum value.
        arms: BTreeMap<String, Expr>,
    },
    /// Two or more terms, summed.
    Add(Vec<Expr>),
    /// A product of two terms.
    Mul(Box<Expr>, Box<Expr>),
    /// One or more terms, maximized.
    Max(Vec<Expr>),
    /// A ceiling division by a positive literal.
    CeilDiv(Box<Expr>, u64),
}

/// A tensor's lifetime.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Lifetime {
    /// Live beyond a single transient window, under a stated retention.
    Persistent(Retention),
    /// Live only within the named transient-lifetime id.
    Transient(String),
}

/// A declared logical tensor. Its byte size is derived, never authored.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorDecl {
    /// The tensor's identifier.
    pub name: String,
    /// Ordered dimensions. An empty shape is one element.
    pub shape: Vec<Expr>,
    /// The element type.
    pub dtype: Dtype,
    /// Sorted layout constraints. These add no physical padding — alignment belongs to the
    /// Backend Execution Profile.
    pub layout: Vec<String>,
    /// When the tensor is live.
    pub lifetime: Lifetime,
}

/// A declared operation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OperationDecl {
    /// The operation's identifier.
    pub name: String,
    /// Its logical operation family — what the profile prices.
    pub family: String,
    /// Input tensor names.
    pub inputs: Vec<String>,
    /// Output tensor names.
    pub outputs: Vec<String>,
    /// The logical workspace class the profile prices, if any.
    pub workspace_class: Option<String>,
    /// The maximum number of simultaneous in-flight instances.
    pub max_in_flight: Expr,
}

/// A declared transfer, bounded by a window rather than by the object it is part of (`[RC-8]`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransferDecl {
    /// The transfer's identifier.
    pub name: String,
    /// Its direction.
    pub kind: TransferKind,
    /// The declared window size in bytes.
    pub window_bytes: Expr,
    /// The maximum number of simultaneous in-flight transfers.
    pub max_in_flight: Expr,
}

/// A linear-memory term's lifetime.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LinearLifetime {
    /// Live across phases.
    Persistent,
    /// Live only within the named transient-lifetime id.
    Transient(String),
}

/// One named, derived term of the guest's own linear-memory footprint (`[RC-1]`(2), `[RC-3]`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinearMemoryTerm {
    /// The term's identifier.
    pub name: String,
    /// When it is live.
    pub lifetime: LinearLifetime,
    /// Its size.
    pub bytes: Expr,
}

/// A module's backend-neutral, parametric statement of logical resource demand.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogicalResourcePlan {
    /// The broadest selection scope the algorithm's semantics permit.
    pub selection_scope: SelectionScope,
    /// blake3 of the module's normalization/equivalence contract. Required for
    /// [`SelectionScope::PerParticipant`], forbidden for [`SelectionScope::UniformRun`].
    pub equivalence_contract_hash: Option<Hash>,
    /// Declared dimensions, sorted by name.
    pub dimensions: Vec<Dimension>,
    /// Declared tensors, sorted by name.
    pub tensors: Vec<TensorDecl>,
    /// Declared operations, sorted by name.
    pub operations: Vec<OperationDecl>,
    /// Declared transfers, sorted by name.
    pub transfers: Vec<TransferDecl>,
    /// Declared linear-memory terms, sorted by name.
    pub linear_memory: Vec<LinearMemoryTerm>,
    /// The maximal sets of transient-lifetime ids that can be concurrently live. Ids within a set
    /// are simultaneous; distinct sets are alternative peak candidates.
    pub transient_live_sets: Vec<Vec<String>>,
    /// The allocator fragmentation and headroom allowance for linear memory (`[RC-3]`).
    pub linear_fragmentation_headroom: Expr,
}

/// A value bound to one dimension.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DimensionValue {
    /// The value of a uint-range dimension.
    Uint(u64),
    /// The chosen arm of an enum dimension.
    Enum(String),
}

/// A complete assignment of every declared dimension — what an Execution Grant carries.
pub type Binding = BTreeMap<String, DimensionValue>;

/// The plan's evaluated logical footprint under one binding. Every figure is **logical**: the
/// physical claim is what the host planner produces by composing these with a certified Backend
/// Execution Profile.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PlanFootprint {
    /// Sum of persistent tensor logical bytes.
    pub device_persistent_bytes: u64,
    /// Maximal-live-set peak over transient tensors.
    pub device_transient_peak_bytes: u64,
    /// `device_persistent_bytes + device_transient_peak_bytes`.
    pub device_peak_bytes: u64,
    /// Sum of persistent linear-memory terms.
    pub linear_persistent_bytes: u64,
    /// Maximal-live-set peak over transient linear-memory terms.
    pub linear_transient_peak_bytes: u64,
    /// The evaluated fragmentation/headroom allowance.
    pub linear_fragmentation_headroom_bytes: u64,
    /// `persistent + transient peak + fragmentation headroom` — exactly `[RC-3]`.
    pub linear_peak_bytes: u64,
    /// The largest single logical tensor. The planner composes this with the profile's
    /// alignment, workspace, pooling and staging behavior to obtain the physical
    /// maximum-individual-allocation figure; the plan never states a physical one.
    pub largest_logical_tensor_bytes: u64,
    /// The largest declared transfer window.
    pub largest_transfer_window_bytes: u64,
}

// -- identifiers and the physical-content prohibition --------------------------------------------

/// Identifier fragments that name a physical backend, allocator, driver or measurement. A plan
/// carrying one of these is describing a machine, which is exactly what `[RC-4]`'s governing
/// invariant forbids. The list is deliberately narrow and literal: it catches the mistake of
/// naming a backend, not every word that could conceivably relate to hardware.
const FORBIDDEN_PHYSICAL_FRAGMENTS: &[&str] = &[
    "vulkan",
    "metal",
    "cuda",
    "rocm",
    "hip",
    "dx12",
    "d3d12",
    "directx",
    "opencl",
    "wgpu",
    "sycl",
    "oneapi",
    "driver",
    "vram",
    "adapter",
    "allocator",
    "mempool",
    "vendor",
    "nvidia",
    "amdgpu",
    "radeon",
    "geforce",
    "apple_silicon",
    "cudnn",
    "cublas",
    "rocblas",
    "shader",
    "kernel_binary",
    "pcie",
    "numa",
];

fn check_ident(what: &str, name: &str) -> Result<(), PlanRefusal> {
    if name.is_empty() {
        return invalid(format!("{what} identifier is empty"));
    }
    if name.len() > LOGICAL_RESOURCE_PLAN_IDENT_BYTES_MAX {
        return exceeds(format!(
            "{what} identifier `{name}` is {} bytes (max {LOGICAL_RESOURCE_PLAN_IDENT_BYTES_MAX})",
            name.len()
        ));
    }
    let lowered = name.to_ascii_lowercase();
    for fragment in FORBIDDEN_PHYSICAL_FRAGMENTS {
        if lowered.contains(fragment) {
            return invalid(format!(
                "{what} identifier `{name}` names physical backend content (`{fragment}`); a \
                 Logical Resource Plan states logical demand only"
            ));
        }
    }
    Ok(())
}

fn check_sorted_unique(what: &str, names: &[&str]) -> Result<(), PlanRefusal> {
    for pair in names.windows(2) {
        if pair[0] == pair[1] {
            return invalid(format!("{what} contains the duplicate name `{}`", pair[0]));
        }
        if pair[0] > pair[1] {
            return invalid(format!(
                "{what} is not sorted by UTF-8 byte order (`{}` precedes `{}`)",
                pair[0], pair[1]
            ));
        }
    }
    Ok(())
}

// -- expression walking --------------------------------------------------------------------------

impl Expr {
    /// The number of expression nodes, this one included.
    #[must_use]
    pub fn node_count(&self) -> usize {
        match self {
            Self::Const(_) | Self::Dimension(_) => 1,
            Self::Select { arms, .. } => 1 + arms.values().map(Self::node_count).sum::<usize>(),
            Self::Add(terms) | Self::Max(terms) => {
                1 + terms.iter().map(Self::node_count).sum::<usize>()
            }
            Self::Mul(a, b) => 1 + a.node_count() + b.node_count(),
            Self::CeilDiv(inner, _) => 1 + inner.node_count(),
        }
    }

    fn depth(&self) -> usize {
        match self {
            Self::Const(_) | Self::Dimension(_) => 1,
            Self::Select { arms, .. } => {
                1 + arms.values().map(Self::depth).max().unwrap_or_default()
            }
            Self::Add(terms) | Self::Max(terms) => {
                1 + terms.iter().map(Self::depth).max().unwrap_or_default()
            }
            Self::Mul(a, b) => 1 + a.depth().max(b.depth()),
            Self::CeilDiv(inner, _) => 1 + inner.depth(),
        }
    }

    /// Evaluate under a complete binding, in checked `u64` arithmetic. Overflow is refusal.
    pub fn evaluate(&self, binding: &Binding) -> Result<u64, PlanRefusal> {
        match self {
            Self::Const(v) => Ok(*v),
            Self::Dimension(name) => match binding.get(name) {
                Some(DimensionValue::Uint(v)) => Ok(*v),
                Some(DimensionValue::Enum(_)) => invalid(format!(
                    "dimension `{name}` is bound to an enum value but referenced numerically"
                )),
                None => invalid(format!("dimension `{name}` has no bound value")),
            },
            Self::Select { dimension, arms } => match binding.get(dimension) {
                Some(DimensionValue::Enum(chosen)) => match arms.get(chosen) {
                    Some(arm) => arm.evaluate(binding),
                    None => invalid(format!(
                        "select on `{dimension}` has no arm for the bound value `{chosen}`"
                    )),
                },
                Some(DimensionValue::Uint(_)) => invalid(format!(
                    "select names `{dimension}`, which is bound to a numeric value"
                )),
                None => invalid(format!("dimension `{dimension}` has no bound value")),
            },
            Self::Add(terms) => terms.iter().try_fold(0u64, |acc, t| {
                acc.checked_add(t.evaluate(binding)?)
                    .ok_or_else(|| PlanRefusal::Invalid("checked u64 overflow in add".into()))
            }),
            Self::Mul(a, b) => a
                .evaluate(binding)?
                .checked_mul(b.evaluate(binding)?)
                .ok_or_else(|| PlanRefusal::Invalid("checked u64 overflow in mul".into())),
            Self::Max(terms) => {
                let mut best = 0u64;
                for term in terms {
                    best = best.max(term.evaluate(binding)?);
                }
                Ok(best)
            }
            Self::CeilDiv(inner, divisor) => {
                if *divisor == 0 {
                    return invalid("ceil-div by zero");
                }
                Ok(inner.evaluate(binding)?.div_ceil(*divisor))
            }
        }
    }

    /// The minimum value this expression can take over every admissible binding. Sound because
    /// every operator is monotone non-decreasing in its dimension arguments and `select` is
    /// minimized over its arms.
    fn evaluate_min(&self, dims: &BTreeMap<&str, &Domain>) -> Result<u64, PlanRefusal> {
        match self {
            Self::Const(v) => Ok(*v),
            Self::Dimension(name) => match dims.get(name.as_str()) {
                Some(Domain::UintRange { lo, .. }) => Ok(*lo),
                Some(Domain::Enum(_)) => invalid(format!(
                    "dimension `{name}` is an enum and may be referenced only through `select`"
                )),
                None => invalid(format!("expression references unknown dimension `{name}`")),
            },
            Self::Select { dimension, arms } => {
                let mut best = u64::MAX;
                for arm in arms.values() {
                    best = best.min(arm.evaluate_min(dims)?);
                }
                let _ = dimension;
                Ok(best)
            }
            Self::Add(terms) => terms.iter().try_fold(0u64, |acc, t| {
                acc.checked_add(t.evaluate_min(dims)?)
                    .ok_or_else(|| PlanRefusal::Invalid("checked u64 overflow in add".into()))
            }),
            Self::Mul(a, b) => a
                .evaluate_min(dims)?
                .checked_mul(b.evaluate_min(dims)?)
                .ok_or_else(|| PlanRefusal::Invalid("checked u64 overflow in mul".into())),
            Self::Max(terms) => {
                let mut best = 0u64;
                for term in terms {
                    best = best.max(term.evaluate_min(dims)?);
                }
                Ok(best)
            }
            Self::CeilDiv(inner, divisor) => {
                if *divisor == 0 {
                    return invalid("ceil-div by zero");
                }
                Ok(inner.evaluate_min(dims)?.div_ceil(*divisor))
            }
        }
    }

    /// Structural validation against the plan's declared dimensions: references resolve, an enum
    /// dimension is reached only through `select`, `select` has exactly one arm per enum value,
    /// arities hold, and the divisor is positive.
    fn validate(&self, dims: &BTreeMap<&str, &Domain>) -> Result<(), PlanRefusal> {
        if self.depth() > LOGICAL_RESOURCE_PLAN_EXPR_DEPTH_MAX {
            return exceeds(format!(
                "expression depth {} exceeds {LOGICAL_RESOURCE_PLAN_EXPR_DEPTH_MAX}",
                self.depth()
            ));
        }
        self.validate_inner(dims)
    }

    fn validate_inner(&self, dims: &BTreeMap<&str, &Domain>) -> Result<(), PlanRefusal> {
        match self {
            Self::Const(_) => Ok(()),
            Self::Dimension(name) => match dims.get(name.as_str()) {
                Some(Domain::UintRange { .. }) => Ok(()),
                Some(Domain::Enum(_)) => invalid(format!(
                    "dimension `{name}` is an enum and may be referenced only through `select`"
                )),
                None => invalid(format!("expression references unknown dimension `{name}`")),
            },
            Self::Select { dimension, arms } => {
                let Some(domain) = dims.get(dimension.as_str()) else {
                    return invalid(format!("select references unknown dimension `{dimension}`"));
                };
                let Domain::Enum(values) = domain else {
                    return invalid(format!(
                        "select names `{dimension}`, which is a uint-range dimension"
                    ));
                };
                let declared: BTreeSet<&str> = values.iter().map(String::as_str).collect();
                let present: BTreeSet<&str> = arms.keys().map(String::as_str).collect();
                if declared != present {
                    return invalid(format!(
                        "select on `{dimension}` must have exactly one arm for every enum value \
                         and no other arm"
                    ));
                }
                for arm in arms.values() {
                    arm.validate_inner(dims)?;
                }
                Ok(())
            }
            Self::Add(terms) => {
                if terms.len() < 2 {
                    return invalid("add takes two or more terms");
                }
                for term in terms {
                    term.validate_inner(dims)?;
                }
                Ok(())
            }
            Self::Max(terms) => {
                if terms.is_empty() {
                    return invalid("max takes one or more terms");
                }
                for term in terms {
                    term.validate_inner(dims)?;
                }
                Ok(())
            }
            Self::Mul(a, b) => {
                a.validate_inner(dims)?;
                b.validate_inner(dims)
            }
            Self::CeilDiv(inner, divisor) => {
                if *divisor == 0 {
                    return invalid("ceil-div divisor must be greater than zero");
                }
                inner.validate_inner(dims)
            }
        }
    }
}

// -- the plan --------------------------------------------------------------------------------------

/// The linear-memory floor of a wasm32 Rust `cdylib` — the bytes a module's image needs before any
/// declared state exists (the linker's initial memory, the data segments, the first heap pages).
///
/// Measured, not guessed: every guest in the workspace fails to instantiate at a 1 MiB cap and
/// comes up at 2 MiB independent of its size, so this is the toolchain's floor rather than any
/// module's, and the declared figure doubles the measured minimum. It lives beside the plan format
/// because the canonical trivial plan is exactly this floor and nothing else, and because a floor
/// spelled once cannot drift from a floor spelled twice.
pub const WASM_GUEST_LINEAR_FLOOR_BYTES: u64 = 4 << 20;

impl LogicalResourcePlan {
    /// The **canonical trivial plan** for a module whose algorithm has no device demand at all.
    ///
    /// It is not the empty plan: a compute-free module still has a wasm heap, so the plan carries
    /// that one linear-memory term and nothing else — no device tensor, no operation family, no
    /// bounded transfer, and therefore no transient group and no fragmentation allowance.
    ///
    /// It lives here, beside the format, so that every module that needs it emits **the same
    /// bytes from the same construction**. Written out module by module it would be a value
    /// maintained in as many places as there are compute-free roles, drifting the first time the
    /// schema or the encoding moves — and a plan is the one object in this model whose whole
    /// purpose is to be the single derivation the host prices.
    #[must_use]
    pub fn trivial(linear_floor_bytes: u64) -> Self {
        Self {
            selection_scope: SelectionScope::UniformRun,
            equivalence_contract_hash: None,
            dimensions: Vec::new(),
            tensors: Vec::new(),
            operations: Vec::new(),
            transfers: Vec::new(),
            linear_memory: vec![LinearMemoryTerm {
                name: "module_linear_floor".to_string(),
                lifetime: LinearLifetime::Persistent,
                bytes: Expr::Const(linear_floor_bytes),
            }],
            transient_live_sets: Vec::new(),
            linear_fragmentation_headroom: Expr::Const(0),
        }
    }

    /// The plan's node count: dimensions, tensors, operations, transfers, linear-memory terms and
    /// every nested expression node.
    #[must_use]
    pub fn node_count(&self) -> usize {
        let expr_nodes: usize = self
            .tensors
            .iter()
            .flat_map(|t| t.shape.iter())
            .map(Expr::node_count)
            .chain(self.operations.iter().map(|o| o.max_in_flight.node_count()))
            .chain(self.transfers.iter().map(|t| t.window_bytes.node_count()))
            .chain(self.transfers.iter().map(|t| t.max_in_flight.node_count()))
            .chain(self.linear_memory.iter().map(|l| l.bytes.node_count()))
            .sum::<usize>()
            + self.linear_fragmentation_headroom.node_count();
        self.dimensions.len()
            + self.tensors.len()
            + self.operations.len()
            + self.transfers.len()
            + self.linear_memory.len()
            + expr_nodes
    }

    /// Every transient-lifetime id named by a tensor or a linear-memory term.
    fn transient_ids(&self) -> BTreeSet<&str> {
        self.tensors
            .iter()
            .filter_map(|t| match &t.lifetime {
                Lifetime::Transient(id) => Some(id.as_str()),
                Lifetime::Persistent(_) => None,
            })
            .chain(self.linear_memory.iter().filter_map(|l| match &l.lifetime {
                LinearLifetime::Transient(id) => Some(id.as_str()),
                LinearLifetime::Persistent => None,
            }))
            .collect()
    }

    /// Full schema-1 validation. Structure, bounds, name resolution, live-set well-formedness,
    /// scope/equivalence coupling, and the physical-content prohibition.
    pub fn validate(&self) -> Result<(), PlanRefusal> {
        if self.dimensions.len() > LOGICAL_RESOURCE_PLAN_DIMENSIONS_MAX {
            return exceeds(format!(
                "{} dimensions exceeds {LOGICAL_RESOURCE_PLAN_DIMENSIONS_MAX}",
                self.dimensions.len()
            ));
        }
        let nodes = self.node_count();
        if nodes > LOGICAL_RESOURCE_PLAN_NODES_MAX {
            return exceeds(format!(
                "{nodes} plan nodes exceeds {LOGICAL_RESOURCE_PLAN_NODES_MAX}"
            ));
        }

        match (self.selection_scope, self.equivalence_contract_hash) {
            (SelectionScope::PerParticipant, None) => {
                return invalid(
                    "per-participant selection requires a normalization/equivalence contract hash \
                     ([RC-11])",
                )
            }
            (SelectionScope::UniformRun, Some(_)) => {
                return invalid(
                    "uniform-run selection must not carry an equivalence contract hash ([RC-11])",
                )
            }
            _ => {}
        }

        // Dimensions.
        let dim_names: Vec<&str> = self.dimensions.iter().map(|d| d.name.as_str()).collect();
        check_sorted_unique("dimensions", &dim_names)?;
        for dim in &self.dimensions {
            check_ident("dimension", &dim.name)?;
            match &dim.domain {
                Domain::UintRange { lo, hi } => {
                    if lo > hi {
                        return invalid(format!(
                            "dimension `{}` has an empty uint-range domain {lo}..={hi}",
                            dim.name
                        ));
                    }
                }
                Domain::Enum(values) => {
                    if values.is_empty() {
                        return invalid(format!("dimension `{}` has an empty enum", dim.name));
                    }
                    let refs: Vec<&str> = values.iter().map(String::as_str).collect();
                    check_sorted_unique(&format!("dimension `{}` enum", dim.name), &refs)?;
                    for value in values {
                        check_ident("enum value", value)?;
                    }
                }
            }
        }
        let dims: BTreeMap<&str, &Domain> = self
            .dimensions
            .iter()
            .map(|d| (d.name.as_str(), &d.domain))
            .collect();

        // Tensors.
        let tensor_names: Vec<&str> = self.tensors.iter().map(|t| t.name.as_str()).collect();
        check_sorted_unique("tensors", &tensor_names)?;
        for tensor in &self.tensors {
            check_ident("tensor", &tensor.name)?;
            for extent in &tensor.shape {
                extent.validate(&dims)?;
            }
            let layout: Vec<&str> = tensor.layout.iter().map(String::as_str).collect();
            check_sorted_unique(&format!("tensor `{}` layout", tensor.name), &layout)?;
            for constraint in &tensor.layout {
                check_ident("layout constraint", constraint)?;
            }
            if let Lifetime::Transient(id) = &tensor.lifetime {
                check_ident("transient-lifetime id", id)?;
            }
        }
        let tensor_set: BTreeSet<&str> = tensor_names.iter().copied().collect();

        // Operations.
        let op_names: Vec<&str> = self.operations.iter().map(|o| o.name.as_str()).collect();
        check_sorted_unique("operations", &op_names)?;
        for op in &self.operations {
            check_ident("operation", &op.name)?;
            check_ident("operation family", &op.family)?;
            if let Some(class) = &op.workspace_class {
                check_ident("workspace class", class)?;
            }
            for referenced in op.inputs.iter().chain(op.outputs.iter()) {
                if !tensor_set.contains(referenced.as_str()) {
                    return invalid(format!(
                        "operation `{}` references unknown tensor `{referenced}`",
                        op.name
                    ));
                }
            }
            op.max_in_flight.validate(&dims)?;
            if op.max_in_flight.evaluate_min(&dims)? == 0 {
                return invalid(format!(
                    "operation `{}` has a max_in_flight that can evaluate to zero",
                    op.name
                ));
            }
        }

        // Transfers.
        let transfer_names: Vec<&str> = self.transfers.iter().map(|t| t.name.as_str()).collect();
        check_sorted_unique("transfers", &transfer_names)?;
        for transfer in &self.transfers {
            check_ident("transfer", &transfer.name)?;
            transfer.window_bytes.validate(&dims)?;
            transfer.max_in_flight.validate(&dims)?;
            if transfer.window_bytes.evaluate_min(&dims)? == 0 {
                return invalid(format!(
                    "transfer `{}` has a window_bytes that can evaluate to zero",
                    transfer.name
                ));
            }
            if transfer.max_in_flight.evaluate_min(&dims)? == 0 {
                return invalid(format!(
                    "transfer `{}` has a max_in_flight that can evaluate to zero",
                    transfer.name
                ));
            }
        }

        // Linear-memory terms.
        let linear_names: Vec<&str> = self.linear_memory.iter().map(|l| l.name.as_str()).collect();
        check_sorted_unique("linear_memory", &linear_names)?;
        for term in &self.linear_memory {
            check_ident("linear-memory term", &term.name)?;
            term.bytes.validate(&dims)?;
            if let LinearLifetime::Transient(id) = &term.lifetime {
                check_ident("transient-lifetime id", id)?;
            }
        }
        self.linear_fragmentation_headroom.validate(&dims)?;

        self.validate_live_sets()
    }

    fn validate_live_sets(&self) -> Result<(), PlanRefusal> {
        let declared = self.transient_ids();
        if declared.is_empty() != self.transient_live_sets.is_empty() {
            return invalid(
                "transient_live_sets is empty exactly when the plan declares no transient terms",
            );
        }
        let outer: Vec<&str> = Vec::new();
        let _ = outer;
        let mut seen: BTreeSet<&str> = BTreeSet::new();
        for set in &self.transient_live_sets {
            if set.is_empty() {
                return invalid("a transient live set must name at least one id");
            }
            let refs: Vec<&str> = set.iter().map(String::as_str).collect();
            check_sorted_unique("a transient live set", &refs)?;
            for id in &refs {
                if !declared.contains(id) {
                    return invalid(format!(
                        "transient live set names `{id}`, which no term declares"
                    ));
                }
                seen.insert(id);
            }
        }
        if seen != declared {
            let missing: Vec<&str> = declared.difference(&seen).copied().collect();
            return invalid(format!(
                "every transient id must occur in at least one live set; missing {missing:?}"
            ));
        }
        // The outer list is unique and lexicographically sorted.
        for pair in self.transient_live_sets.windows(2) {
            if pair[0] == pair[1] {
                return invalid("transient_live_sets contains a duplicate set");
            }
            if pair[0] > pair[1] {
                return invalid("transient_live_sets is not lexicographically sorted");
            }
        }
        // No listed set is a strict subset of another — the sets are maximal by construction.
        for (i, a) in self.transient_live_sets.iter().enumerate() {
            let sa: BTreeSet<&str> = a.iter().map(String::as_str).collect();
            for (j, b) in self.transient_live_sets.iter().enumerate() {
                if i == j {
                    continue;
                }
                let sb: BTreeSet<&str> = b.iter().map(String::as_str).collect();
                if sa.is_subset(&sb) && sa != sb {
                    return invalid(format!(
                        "transient live set {a:?} is a strict subset of {b:?}; only maximal sets \
                         are listed"
                    ));
                }
            }
        }
        Ok(())
    }

    /// The logical byte size of one tensor under a binding: `ceil(product(shape) × bits / 8)`,
    /// with an empty shape counting one element and every multiplication checked.
    pub fn tensor_bytes(tensor: &TensorDecl, binding: &Binding) -> Result<u64, PlanRefusal> {
        let mut elements: u64 = 1;
        for extent in &tensor.shape {
            elements = elements
                .checked_mul(extent.evaluate(binding)?)
                .ok_or_else(|| {
                    PlanRefusal::Invalid(format!(
                        "checked u64 overflow computing the element count of `{}`",
                        tensor.name
                    ))
                })?;
        }
        let bits = elements.checked_mul(tensor.dtype.bits()).ok_or_else(|| {
            PlanRefusal::Invalid(format!(
                "checked u64 overflow computing the bit size of `{}`",
                tensor.name
            ))
        })?;
        Ok(bits.div_ceil(8))
    }

    /// Check a binding names exactly the plan's dimensions, each within its declared domain.
    pub fn check_binding(&self, binding: &Binding) -> Result<(), PlanRefusal> {
        if binding.len() != self.dimensions.len() {
            return invalid(format!(
                "binding names {} dimensions; the plan declares {}",
                binding.len(),
                self.dimensions.len()
            ));
        }
        for dim in &self.dimensions {
            let Some(value) = binding.get(&dim.name) else {
                return invalid(format!("binding has no value for dimension `{}`", dim.name));
            };
            match (&dim.domain, value) {
                (Domain::UintRange { lo, hi }, DimensionValue::Uint(v)) => {
                    if v < lo || v > hi {
                        return invalid(format!(
                            "dimension `{}` value {v} is outside its declared range {lo}..={hi}",
                            dim.name
                        ));
                    }
                }
                (Domain::Enum(values), DimensionValue::Enum(chosen)) => {
                    if !values.iter().any(|v| v == chosen) {
                        return invalid(format!(
                            "dimension `{}` value `{chosen}` is not one of its declared arms",
                            dim.name
                        ));
                    }
                }
                (Domain::UintRange { .. }, DimensionValue::Enum(_)) => {
                    return invalid(format!(
                        "dimension `{}` is a uint range but the binding gives a spelling",
                        dim.name
                    ))
                }
                (Domain::Enum(_), DimensionValue::Uint(_)) => {
                    return invalid(format!(
                        "dimension `{}` is an enum but the binding gives a number",
                        dim.name
                    ))
                }
            }
        }
        Ok(())
    }

    /// Evaluate the plan's logical footprint under one binding — the maximal-live-set peak
    /// arithmetic of `[RC-3]`, applied per resource domain. These are planner semantics, not
    /// profile discretion.
    pub fn footprint(&self, binding: &Binding) -> Result<PlanFootprint, PlanRefusal> {
        self.check_binding(binding)?;

        let mut out = PlanFootprint::default();
        let add = |a: u64, b: u64| -> Result<u64, PlanRefusal> {
            a.checked_add(b)
                .ok_or_else(|| PlanRefusal::Invalid("checked u64 overflow summing terms".into()))
        };

        // Device-logical: persistent floor + transient peak.
        let mut transient_device: BTreeMap<&str, u64> = BTreeMap::new();
        for tensor in &self.tensors {
            let bytes = Self::tensor_bytes(tensor, binding)?;
            out.largest_logical_tensor_bytes = out.largest_logical_tensor_bytes.max(bytes);
            match &tensor.lifetime {
                Lifetime::Persistent(_) => {
                    out.device_persistent_bytes = add(out.device_persistent_bytes, bytes)?;
                }
                Lifetime::Transient(id) => {
                    // All terms sharing an id are summed.
                    let slot = transient_device.entry(id.as_str()).or_default();
                    *slot = add(*slot, bytes)?;
                }
            }
        }

        let mut transient_linear: BTreeMap<&str, u64> = BTreeMap::new();
        for term in &self.linear_memory {
            let bytes = term.bytes.evaluate(binding)?;
            match &term.lifetime {
                LinearLifetime::Persistent => {
                    out.linear_persistent_bytes = add(out.linear_persistent_bytes, bytes)?;
                }
                LinearLifetime::Transient(id) => {
                    let slot = transient_linear.entry(id.as_str()).or_default();
                    *slot = add(*slot, bytes)?;
                }
            }
        }

        for set in &self.transient_live_sets {
            let mut device_sum = 0u64;
            let mut linear_sum = 0u64;
            for id in set {
                device_sum = add(
                    device_sum,
                    transient_device.get(id.as_str()).copied().unwrap_or(0),
                )?;
                linear_sum = add(
                    linear_sum,
                    transient_linear.get(id.as_str()).copied().unwrap_or(0),
                )?;
            }
            out.device_transient_peak_bytes = out.device_transient_peak_bytes.max(device_sum);
            out.linear_transient_peak_bytes = out.linear_transient_peak_bytes.max(linear_sum);
        }

        out.linear_fragmentation_headroom_bytes =
            self.linear_fragmentation_headroom.evaluate(binding)?;

        out.device_peak_bytes = add(out.device_persistent_bytes, out.device_transient_peak_bytes)?;
        out.linear_peak_bytes = add(
            add(out.linear_persistent_bytes, out.linear_transient_peak_bytes)?,
            out.linear_fragmentation_headroom_bytes,
        )?;

        for transfer in &self.transfers {
            out.largest_transfer_window_bytes = out
                .largest_transfer_window_bytes
                .max(transfer.window_bytes.evaluate(binding)?);
        }

        Ok(out)
    }

    /// The plan's canonical CBOR bytes.
    pub fn to_canonical_bytes(&self) -> Result<Vec<u8>, PlanRefusal> {
        let value = self.to_value();
        to_canonical_vec(&value).map_err(|e| PlanRefusal::Invalid(format!("plan encoding: {e}")))
    }

    /// blake3 of the plan's canonical bytes — the `logical_resource_plan_hash` of the admitted
    /// tuple (`[DI-9]`). Never compare this to a historical `claim_hash`: that member carried the
    /// hash of a physical-tier claim, which is a different object with different semantics.
    pub fn plan_hash(&self) -> Result<Hash, PlanRefusal> {
        Ok(blake3_hash(&self.to_canonical_bytes()?))
    }

    /// Decode and fully validate a plan from bytes, enforcing the byte ceiling, canonicality and
    /// every schema-1 rule. This is the host's admission-side entry point.
    pub fn decode_canonical(bytes: &[u8]) -> Result<Self, PlanRefusal> {
        if bytes.len() > LOGICAL_RESOURCE_PLAN_BYTES_MAX {
            return exceeds(format!(
                "{} plan bytes exceeds {LOGICAL_RESOURCE_PLAN_BYTES_MAX}",
                bytes.len()
            ));
        }
        let value: Value = ciborium::de::from_reader(bytes)
            .map_err(|e| PlanRefusal::Invalid(format!("plan is not well-formed CBOR: {e}")))?;
        let plan = Self::from_value(&value)?;
        plan.validate()?;
        // Canonicality: the bytes must be exactly what re-encoding the decoded value produces.
        let reencoded = plan.to_canonical_bytes()?;
        if reencoded != bytes {
            return invalid("plan bytes are not canonical CBOR for their own content");
        }
        let fuel = plan_derivation_fuel(plan.node_count() as u64, bytes.len() as u64);
        if fuel > PLAN_DERIVATION_FUEL_CEILING {
            return exceeds(format!(
                "derived derivation budget {fuel} exceeds the absolute ceiling \
                 {PLAN_DERIVATION_FUEL_CEILING}"
            ));
        }
        Ok(plan)
    }

    // -- encoding -------------------------------------------------------------------------------

    fn to_value(&self) -> Value {
        let mut members = vec![
            (
                Value::Text("schema".into()),
                Value::Integer(LOGICAL_RESOURCE_PLAN_SCHEMA.into()),
            ),
            (
                Value::Text("selection_scope".into()),
                Value::Text(self.selection_scope.spelling().into()),
            ),
            (
                Value::Text("equivalence_contract_hash".into()),
                match self.equivalence_contract_hash {
                    Some(h) => Value::Bytes(h.0.to_vec()),
                    None => Value::Null,
                },
            ),
            (
                Value::Text("dimensions".into()),
                Value::Array(self.dimensions.iter().map(dimension_value).collect()),
            ),
            (
                Value::Text("tensors".into()),
                Value::Array(self.tensors.iter().map(tensor_value).collect()),
            ),
            (
                Value::Text("operations".into()),
                Value::Array(self.operations.iter().map(operation_value).collect()),
            ),
            (
                Value::Text("transfers".into()),
                Value::Array(self.transfers.iter().map(transfer_value).collect()),
            ),
            (
                Value::Text("linear_memory".into()),
                Value::Array(self.linear_memory.iter().map(linear_value).collect()),
            ),
            (
                Value::Text("transient_live_sets".into()),
                Value::Array(
                    self.transient_live_sets
                        .iter()
                        .map(|set| {
                            Value::Array(set.iter().map(|id| Value::Text(id.clone())).collect())
                        })
                        .collect(),
                ),
            ),
            (
                Value::Text("linear_fragmentation_headroom".into()),
                expr_value(&self.linear_fragmentation_headroom),
            ),
        ];
        members.sort_by(|a, b| match (&a.0, &b.0) {
            (Value::Text(x), Value::Text(y)) => x.cmp(y),
            _ => std::cmp::Ordering::Equal,
        });
        Value::Map(members)
    }

    fn from_value(value: &Value) -> Result<Self, PlanRefusal> {
        let map = as_map(value, "logical-resource-plan")?;
        let known = [
            "schema",
            "selection_scope",
            "equivalence_contract_hash",
            "dimensions",
            "tensors",
            "operations",
            "transfers",
            "linear_memory",
            "transient_live_sets",
            "linear_fragmentation_headroom",
        ];
        reject_unknown_members(&map, &known, "logical-resource-plan")?;

        let schema = member_uint(&map, "schema")?;
        if schema != LOGICAL_RESOURCE_PLAN_SCHEMA {
            return invalid(format!(
                "unknown logical-resource-plan schema {schema} (this build understands \
                 {LOGICAL_RESOURCE_PLAN_SCHEMA})"
            ));
        }
        let selection_scope = SelectionScope::parse_spelling(member_text(&map, "selection_scope")?)
            .ok_or_else(|| PlanRefusal::Invalid("unknown selection_scope".into()))?;
        let equivalence_contract_hash = match member(&map, "equivalence_contract_hash")? {
            Value::Null => None,
            Value::Bytes(b) => Some(Hash(<[u8; 32]>::try_from(b.as_slice()).map_err(|_| {
                PlanRefusal::Invalid("equivalence_contract_hash must be 32 bytes".into())
            })?)),
            _ => return invalid("equivalence_contract_hash must be null or a 32-byte string"),
        };

        let dimensions = as_array(member(&map, "dimensions")?, "dimensions")?
            .iter()
            .map(dimension_from_value)
            .collect::<Result<Vec<_>, _>>()?;
        let tensors = as_array(member(&map, "tensors")?, "tensors")?
            .iter()
            .map(tensor_from_value)
            .collect::<Result<Vec<_>, _>>()?;
        let operations = as_array(member(&map, "operations")?, "operations")?
            .iter()
            .map(operation_from_value)
            .collect::<Result<Vec<_>, _>>()?;
        let transfers = as_array(member(&map, "transfers")?, "transfers")?
            .iter()
            .map(transfer_from_value)
            .collect::<Result<Vec<_>, _>>()?;
        let linear_memory = as_array(member(&map, "linear_memory")?, "linear_memory")?
            .iter()
            .map(linear_from_value)
            .collect::<Result<Vec<_>, _>>()?;
        let transient_live_sets = as_array(member(&map, "transient_live_sets")?, "live sets")?
            .iter()
            .map(|set| {
                as_array(set, "a transient live set")?
                    .iter()
                    .map(|id| Ok(as_text(id, "a transient-lifetime id")?.to_string()))
                    .collect::<Result<Vec<_>, PlanRefusal>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        let linear_fragmentation_headroom =
            expr_from_value(member(&map, "linear_fragmentation_headroom")?)?;

        Ok(Self {
            selection_scope,
            equivalence_contract_hash,
            dimensions,
            tensors,
            operations,
            transfers,
            linear_memory,
            transient_live_sets,
            linear_fragmentation_headroom,
        })
    }
}

// -- value helpers ---------------------------------------------------------------------------------

type MemberMap<'a> = BTreeMap<&'a str, &'a Value>;

fn as_map<'a>(value: &'a Value, what: &str) -> Result<MemberMap<'a>, PlanRefusal> {
    let Value::Map(entries) = value else {
        return invalid(format!("{what} must be a CBOR map"));
    };
    let mut out = MemberMap::new();
    for (k, v) in entries {
        let Value::Text(key) = k else {
            return invalid(format!("{what} has a non-text member key"));
        };
        if out.insert(key.as_str(), v).is_some() {
            return invalid(format!("{what} has the duplicate member `{key}`"));
        }
    }
    Ok(out)
}

fn reject_unknown_members(
    map: &MemberMap<'_>,
    known: &[&str],
    what: &str,
) -> Result<(), PlanRefusal> {
    for key in map.keys() {
        if !known.contains(key) {
            return invalid(format!("{what} has the unknown member `{key}`"));
        }
    }
    for key in known {
        if !map.contains_key(key) {
            return invalid(format!("{what} is missing the member `{key}`"));
        }
    }
    Ok(())
}

fn member<'a>(map: &MemberMap<'a>, key: &str) -> Result<&'a Value, PlanRefusal> {
    map.get(key)
        .copied()
        .ok_or_else(|| PlanRefusal::Invalid(format!("missing member `{key}`")))
}

fn as_array<'a>(value: &'a Value, what: &str) -> Result<&'a Vec<Value>, PlanRefusal> {
    match value {
        Value::Array(items) => Ok(items),
        _ => invalid(format!("{what} must be a CBOR array")),
    }
}

fn as_text<'a>(value: &'a Value, what: &str) -> Result<&'a str, PlanRefusal> {
    match value {
        Value::Text(s) => Ok(s.as_str()),
        _ => invalid(format!("{what} must be a text string")),
    }
}

fn as_uint(value: &Value, what: &str) -> Result<u64, PlanRefusal> {
    match value {
        Value::Integer(i) => u64::try_from(i128::from(*i))
            .map_err(|_| PlanRefusal::Invalid(format!("{what} must be an unsigned integer"))),
        _ => invalid(format!("{what} must be an unsigned integer")),
    }
}

fn member_uint(map: &MemberMap<'_>, key: &str) -> Result<u64, PlanRefusal> {
    as_uint(member(map, key)?, key)
}

fn member_text<'a>(map: &MemberMap<'a>, key: &str) -> Result<&'a str, PlanRefusal> {
    as_text(member(map, key)?, key)
}

fn dimension_value(dim: &Dimension) -> Value {
    let domain = match &dim.domain {
        Domain::UintRange { lo, hi } => Value::Array(vec![
            Value::Text("uint-range".into()),
            Value::Integer((*lo).into()),
            Value::Integer((*hi).into()),
        ]),
        Domain::Enum(values) => {
            let mut items = vec![Value::Text("enum".into())];
            items.extend(values.iter().map(|v| Value::Text(v.clone())));
            Value::Array(items)
        }
    };
    Value::Map(vec![
        (Value::Text("domain".into()), domain),
        (Value::Text("name".into()), Value::Text(dim.name.clone())),
    ])
}

fn dimension_from_value(value: &Value) -> Result<Dimension, PlanRefusal> {
    let map = as_map(value, "dimension")?;
    reject_unknown_members(&map, &["name", "domain"], "dimension")?;
    let name = member_text(&map, "name")?.to_string();
    let items = as_array(member(&map, "domain")?, "dimension domain")?;
    let tag = items
        .first()
        .map(|v| as_text(v, "dimension domain tag"))
        .transpose()?
        .unwrap_or_default();
    let domain = match tag {
        "uint-range" => {
            if items.len() != 3 {
                return invalid("uint-range domain takes exactly a low and a high bound");
            }
            Domain::UintRange {
                lo: as_uint(&items[1], "uint-range low bound")?,
                hi: as_uint(&items[2], "uint-range high bound")?,
            }
        }
        "enum" => {
            if items.len() < 2 {
                return invalid("enum domain takes at least one value");
            }
            Domain::Enum(
                items[1..]
                    .iter()
                    .map(|v| Ok(as_text(v, "enum value")?.to_string()))
                    .collect::<Result<Vec<_>, PlanRefusal>>()?,
            )
        }
        other => return invalid(format!("unknown dimension domain kind `{other}`")),
    };
    Ok(Dimension { name, domain })
}

fn lifetime_value(lifetime: &Lifetime) -> Value {
    match lifetime {
        Lifetime::Persistent(retention) => Value::Array(vec![
            Value::Text("persistent".into()),
            Value::Text(retention.spelling().into()),
        ]),
        Lifetime::Transient(id) => Value::Array(vec![
            Value::Text("transient".into()),
            Value::Text(id.clone()),
        ]),
    }
}

fn tensor_value(tensor: &TensorDecl) -> Value {
    Value::Map(vec![
        (
            Value::Text("dtype".into()),
            Value::Text(tensor.dtype.spelling().into()),
        ),
        (
            Value::Text("layout".into()),
            Value::Array(
                tensor
                    .layout
                    .iter()
                    .map(|c| Value::Text(c.clone()))
                    .collect(),
            ),
        ),
        (
            Value::Text("lifetime".into()),
            lifetime_value(&tensor.lifetime),
        ),
        (Value::Text("name".into()), Value::Text(tensor.name.clone())),
        (
            Value::Text("shape".into()),
            Value::Array(tensor.shape.iter().map(expr_value).collect()),
        ),
    ])
}

fn tensor_from_value(value: &Value) -> Result<TensorDecl, PlanRefusal> {
    let map = as_map(value, "tensor-decl")?;
    reject_unknown_members(
        &map,
        &["name", "shape", "dtype", "layout", "lifetime"],
        "tensor-decl",
    )?;
    let lifetime_items = as_array(member(&map, "lifetime")?, "tensor lifetime")?;
    let lifetime = match lifetime_items
        .first()
        .map(|v| as_text(v, "lifetime kind"))
        .transpose()?
        .unwrap_or_default()
    {
        "persistent" => {
            if lifetime_items.len() != 2 {
                return invalid("a persistent lifetime takes exactly one retention");
            }
            Lifetime::Persistent(
                Retention::parse(as_text(&lifetime_items[1], "retention")?)
                    .ok_or_else(|| PlanRefusal::Invalid("unknown retention".into()))?,
            )
        }
        "transient" => {
            if lifetime_items.len() != 2 {
                return invalid("a transient lifetime takes exactly one lifetime id");
            }
            Lifetime::Transient(as_text(&lifetime_items[1], "transient-lifetime id")?.to_string())
        }
        other => return invalid(format!("unknown tensor lifetime kind `{other}`")),
    };
    Ok(TensorDecl {
        name: member_text(&map, "name")?.to_string(),
        shape: as_array(member(&map, "shape")?, "tensor shape")?
            .iter()
            .map(expr_from_value)
            .collect::<Result<Vec<_>, _>>()?,
        dtype: Dtype::parse(member_text(&map, "dtype")?)
            .ok_or_else(|| PlanRefusal::Invalid("unknown dtype spelling".into()))?,
        layout: as_array(member(&map, "layout")?, "tensor layout")?
            .iter()
            .map(|v| Ok(as_text(v, "layout constraint")?.to_string()))
            .collect::<Result<Vec<_>, PlanRefusal>>()?,
        lifetime,
    })
}

fn operation_value(op: &OperationDecl) -> Value {
    Value::Map(vec![
        (Value::Text("family".into()), Value::Text(op.family.clone())),
        (
            Value::Text("inputs".into()),
            Value::Array(op.inputs.iter().map(|n| Value::Text(n.clone())).collect()),
        ),
        (
            Value::Text("max_in_flight".into()),
            expr_value(&op.max_in_flight),
        ),
        (Value::Text("name".into()), Value::Text(op.name.clone())),
        (
            Value::Text("outputs".into()),
            Value::Array(op.outputs.iter().map(|n| Value::Text(n.clone())).collect()),
        ),
        (
            Value::Text("workspace_class".into()),
            match &op.workspace_class {
                Some(class) => Value::Text(class.clone()),
                None => Value::Null,
            },
        ),
    ])
}

fn operation_from_value(value: &Value) -> Result<OperationDecl, PlanRefusal> {
    let map = as_map(value, "operation-decl")?;
    reject_unknown_members(
        &map,
        &[
            "name",
            "family",
            "inputs",
            "outputs",
            "workspace_class",
            "max_in_flight",
        ],
        "operation-decl",
    )?;
    Ok(OperationDecl {
        name: member_text(&map, "name")?.to_string(),
        family: member_text(&map, "family")?.to_string(),
        inputs: as_array(member(&map, "inputs")?, "operation inputs")?
            .iter()
            .map(|v| Ok(as_text(v, "operation input")?.to_string()))
            .collect::<Result<Vec<_>, PlanRefusal>>()?,
        outputs: as_array(member(&map, "outputs")?, "operation outputs")?
            .iter()
            .map(|v| Ok(as_text(v, "operation output")?.to_string()))
            .collect::<Result<Vec<_>, PlanRefusal>>()?,
        workspace_class: match member(&map, "workspace_class")? {
            Value::Null => None,
            other => Some(as_text(other, "workspace class")?.to_string()),
        },
        max_in_flight: expr_from_value(member(&map, "max_in_flight")?)?,
    })
}

fn transfer_value(transfer: &TransferDecl) -> Value {
    Value::Map(vec![
        (
            Value::Text("kind".into()),
            Value::Text(transfer.kind.spelling().into()),
        ),
        (
            Value::Text("max_in_flight".into()),
            expr_value(&transfer.max_in_flight),
        ),
        (
            Value::Text("name".into()),
            Value::Text(transfer.name.clone()),
        ),
        (
            Value::Text("window_bytes".into()),
            expr_value(&transfer.window_bytes),
        ),
    ])
}

fn transfer_from_value(value: &Value) -> Result<TransferDecl, PlanRefusal> {
    let map = as_map(value, "transfer-decl")?;
    reject_unknown_members(
        &map,
        &["name", "kind", "window_bytes", "max_in_flight"],
        "transfer-decl",
    )?;
    Ok(TransferDecl {
        name: member_text(&map, "name")?.to_string(),
        kind: TransferKind::parse(member_text(&map, "kind")?)
            .ok_or_else(|| PlanRefusal::Invalid("unknown transfer kind".into()))?,
        window_bytes: expr_from_value(member(&map, "window_bytes")?)?,
        max_in_flight: expr_from_value(member(&map, "max_in_flight")?)?,
    })
}

fn linear_value(term: &LinearMemoryTerm) -> Value {
    let lifetime = match &term.lifetime {
        LinearLifetime::Persistent => Value::Array(vec![Value::Text("persistent".into())]),
        LinearLifetime::Transient(id) => Value::Array(vec![
            Value::Text("transient".into()),
            Value::Text(id.clone()),
        ]),
    };
    Value::Map(vec![
        (Value::Text("bytes".into()), expr_value(&term.bytes)),
        (Value::Text("lifetime".into()), lifetime),
        (Value::Text("name".into()), Value::Text(term.name.clone())),
    ])
}

fn linear_from_value(value: &Value) -> Result<LinearMemoryTerm, PlanRefusal> {
    let map = as_map(value, "linear-memory-term")?;
    reject_unknown_members(&map, &["name", "lifetime", "bytes"], "linear-memory-term")?;
    let items = as_array(member(&map, "lifetime")?, "linear-memory lifetime")?;
    let lifetime = match items
        .first()
        .map(|v| as_text(v, "lifetime kind"))
        .transpose()?
        .unwrap_or_default()
    {
        "persistent" => {
            if items.len() != 1 {
                return invalid("a persistent linear-memory lifetime takes no argument");
            }
            LinearLifetime::Persistent
        }
        "transient" => {
            if items.len() != 2 {
                return invalid("a transient linear-memory lifetime takes exactly one id");
            }
            LinearLifetime::Transient(as_text(&items[1], "transient-lifetime id")?.to_string())
        }
        other => return invalid(format!("unknown linear-memory lifetime kind `{other}`")),
    };
    Ok(LinearMemoryTerm {
        name: member_text(&map, "name")?.to_string(),
        lifetime,
        bytes: expr_from_value(member(&map, "bytes")?)?,
    })
}

fn expr_value(expr: &Expr) -> Value {
    match expr {
        Expr::Const(v) => Value::Array(vec![
            Value::Text("const".into()),
            Value::Integer((*v).into()),
        ]),
        Expr::Dimension(name) => Value::Array(vec![
            Value::Text("dimension".into()),
            Value::Text(name.clone()),
        ]),
        Expr::Select { dimension, arms } => Value::Array(vec![
            Value::Text("select".into()),
            Value::Text(dimension.clone()),
            Value::Map(
                arms.iter()
                    .map(|(k, v)| (Value::Text(k.clone()), expr_value(v)))
                    .collect(),
            ),
        ]),
        Expr::Add(terms) => {
            let mut items = vec![Value::Text("add".into())];
            items.extend(terms.iter().map(expr_value));
            Value::Array(items)
        }
        Expr::Mul(a, b) => Value::Array(vec![
            Value::Text("mul".into()),
            expr_value(a),
            expr_value(b),
        ]),
        Expr::Max(terms) => {
            let mut items = vec![Value::Text("max".into())];
            items.extend(terms.iter().map(expr_value));
            Value::Array(items)
        }
        Expr::CeilDiv(inner, divisor) => Value::Array(vec![
            Value::Text("ceil-div".into()),
            expr_value(inner),
            Value::Integer((*divisor).into()),
        ]),
    }
}

fn expr_from_value(value: &Value) -> Result<Expr, PlanRefusal> {
    let items = as_array(value, "expr")?;
    let op = items
        .first()
        .map(|v| as_text(v, "expr operator"))
        .transpose()?
        .unwrap_or_default();
    match op {
        "const" => {
            if items.len() != 2 {
                return invalid("const takes exactly one literal");
            }
            Ok(Expr::Const(as_uint(&items[1], "const literal")?))
        }
        "dimension" => {
            if items.len() != 2 {
                return invalid("dimension takes exactly one name");
            }
            Ok(Expr::Dimension(
                as_text(&items[1], "dimension name")?.to_string(),
            ))
        }
        "select" => {
            if items.len() != 3 {
                return invalid("select takes a dimension name and an arm map");
            }
            let arm_map = as_map(&items[2], "select arms")?;
            if arm_map.is_empty() {
                return invalid("select takes at least one arm");
            }
            let mut arms = BTreeMap::new();
            for (key, val) in arm_map {
                arms.insert(key.to_string(), expr_from_value(val)?);
            }
            Ok(Expr::Select {
                dimension: as_text(&items[1], "select dimension")?.to_string(),
                arms,
            })
        }
        "add" => {
            if items.len() < 3 {
                return invalid("add takes two or more terms");
            }
            Ok(Expr::Add(
                items[1..]
                    .iter()
                    .map(expr_from_value)
                    .collect::<Result<Vec<_>, _>>()?,
            ))
        }
        "mul" => {
            if items.len() != 3 {
                return invalid("mul takes exactly two terms");
            }
            Ok(Expr::Mul(
                Box::new(expr_from_value(&items[1])?),
                Box::new(expr_from_value(&items[2])?),
            ))
        }
        "max" => {
            if items.len() < 2 {
                return invalid("max takes one or more terms");
            }
            Ok(Expr::Max(
                items[1..]
                    .iter()
                    .map(expr_from_value)
                    .collect::<Result<Vec<_>, _>>()?,
            ))
        }
        "ceil-div" => {
            if items.len() != 3 {
                return invalid("ceil-div takes a term and a positive divisor");
            }
            let divisor = as_uint(&items[2], "ceil-div divisor")?;
            if divisor == 0 {
                return invalid("ceil-div divisor must be greater than zero");
            }
            Ok(Expr::CeilDiv(
                Box::new(expr_from_value(&items[1])?),
                divisor,
            ))
        }
        other => invalid(format!("unknown expr operator `{other}`")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dim_uint(name: &str, lo: u64, hi: u64) -> Dimension {
        Dimension {
            name: name.into(),
            domain: Domain::UintRange { lo, hi },
        }
    }

    /// A small but complete plan: one free numeric dimension, one enum dimension, persistent and
    /// transient tensors in two alternative live sets, a linear term and a transfer window.
    fn sample() -> LogicalResourcePlan {
        LogicalResourcePlan {
            selection_scope: SelectionScope::UniformRun,
            equivalence_contract_hash: None,
            dimensions: vec![
                dim_uint("micro_batch", 1, 8),
                Dimension {
                    name: "precision".into(),
                    domain: Domain::Enum(vec!["full".into(), "half".into()]),
                },
            ],
            tensors: vec![
                TensorDecl {
                    name: "activations".into(),
                    shape: vec![Expr::Dimension("micro_batch".into()), Expr::Const(1024)],
                    dtype: Dtype::F32,
                    layout: vec![],
                    lifetime: Lifetime::Transient("forward".into()),
                },
                TensorDecl {
                    name: "grads".into(),
                    shape: vec![Expr::Const(2048)],
                    dtype: Dtype::F32,
                    layout: vec![],
                    lifetime: Lifetime::Transient("backward".into()),
                },
                TensorDecl {
                    name: "params".into(),
                    shape: vec![Expr::Const(4096)],
                    dtype: Dtype::F32,
                    layout: vec!["row-major".into()],
                    lifetime: Lifetime::Persistent(Retention::Run),
                },
            ],
            operations: vec![OperationDecl {
                name: "matmul".into(),
                family: "gemm".into(),
                inputs: vec!["params".into()],
                outputs: vec!["activations".into()],
                workspace_class: Some("reduction".into()),
                max_in_flight: Expr::Const(2),
            }],
            transfers: vec![TransferDecl {
                name: "corpus_window".into(),
                kind: TransferKind::Ingest,
                window_bytes: Expr::Const(65536),
                max_in_flight: Expr::Const(1),
            }],
            linear_memory: vec![
                LinearMemoryTerm {
                    name: "index".into(),
                    lifetime: LinearLifetime::Persistent,
                    bytes: Expr::Const(32_768),
                },
                LinearMemoryTerm {
                    name: "window".into(),
                    lifetime: LinearLifetime::Transient("forward".into()),
                    bytes: Expr::Select {
                        dimension: "precision".into(),
                        arms: BTreeMap::from([
                            ("full".into(), Expr::Const(8192)),
                            ("half".into(), Expr::Const(4096)),
                        ]),
                    },
                },
            ],
            transient_live_sets: vec![vec!["backward".into()], vec!["forward".into()]],
            linear_fragmentation_headroom: Expr::Const(1024),
        }
    }

    fn binding(micro_batch: u64, precision: &str) -> Binding {
        Binding::from([
            ("micro_batch".to_string(), DimensionValue::Uint(micro_batch)),
            (
                "precision".to_string(),
                DimensionValue::Enum(precision.into()),
            ),
        ])
    }

    #[test]
    fn the_sample_plan_validates_and_round_trips_canonically() {
        let plan = sample();
        plan.validate().expect("valid");
        let bytes = plan.to_canonical_bytes().unwrap();
        let decoded = LogicalResourcePlan::decode_canonical(&bytes).unwrap();
        assert_eq!(decoded, plan);
        assert_eq!(decoded.to_canonical_bytes().unwrap(), bytes);
        assert_eq!(decoded.plan_hash().unwrap(), plan.plan_hash().unwrap());
    }

    /// Logical bytes are derived from shape and dtype, not authored — including the `bool1`
    /// bit-packing edge and the empty (scalar) shape.
    #[test]
    fn logical_tensor_bytes_are_derived_from_shape_and_dtype() {
        let b = binding(4, "full");
        let scalar = TensorDecl {
            name: "scalar".into(),
            shape: vec![],
            dtype: Dtype::F64,
            layout: vec![],
            lifetime: Lifetime::Persistent(Retention::Run),
        };
        assert_eq!(LogicalResourcePlan::tensor_bytes(&scalar, &b).unwrap(), 8);

        let mask = TensorDecl {
            name: "mask".into(),
            shape: vec![Expr::Const(9)],
            dtype: Dtype::Bool1,
            layout: vec![],
            lifetime: Lifetime::Persistent(Retention::Run),
        };
        assert_eq!(
            LogicalResourcePlan::tensor_bytes(&mask, &b).unwrap(),
            2,
            "nine bits round up to two bytes"
        );

        let acts = &sample().tensors[0];
        assert_eq!(
            LogicalResourcePlan::tensor_bytes(acts, &b).unwrap(),
            4 * 1024 * 4
        );
    }

    /// The peak is the maximal concurrently-live set, never the sum over time — `[RC-3]`.
    #[test]
    fn the_transient_peak_is_the_maximal_live_set_and_never_the_sum() {
        let plan = sample();
        let f = plan.footprint(&binding(4, "full")).unwrap();
        assert_eq!(f.device_persistent_bytes, 4096 * 4);
        // forward = activations (4×1024×4 = 16384); backward = grads (2048×4 = 8192).
        assert_eq!(f.device_transient_peak_bytes, 16_384);
        assert_ne!(
            f.device_transient_peak_bytes,
            16_384 + 8_192,
            "sequential transients do not accumulate"
        );
        assert_eq!(f.device_peak_bytes, 16_384 + 16_384);
        assert_eq!(f.linear_persistent_bytes, 32_768);
        assert_eq!(f.linear_transient_peak_bytes, 8_192);
        assert_eq!(f.linear_fragmentation_headroom_bytes, 1_024);
        assert_eq!(f.linear_peak_bytes, 32_768 + 8_192 + 1_024);
        assert_eq!(f.largest_logical_tensor_bytes, 16_384);
        assert_eq!(f.largest_transfer_window_bytes, 65_536);
    }

    /// An enum dimension reaches the arithmetic only through `select`, and the arm the grant
    /// chooses is the one that prices.
    #[test]
    fn select_prices_the_arm_the_binding_chooses() {
        let plan = sample();
        assert_eq!(
            plan.footprint(&binding(1, "half"))
                .unwrap()
                .linear_transient_peak_bytes,
            4_096
        );
    }

    #[test]
    fn plan_derivation_is_deterministic_from_identical_inputs() {
        let a = sample().to_canonical_bytes().unwrap();
        let b = sample().to_canonical_bytes().unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn checked_arithmetic_refuses_overflow() {
        let mut plan = sample();
        plan.tensors[2].shape = vec![Expr::Const(u64::MAX), Expr::Const(2)];
        let err = plan.footprint(&binding(1, "full")).unwrap_err();
        assert!(matches!(err, PlanRefusal::Invalid(_)));
        assert!(err.detail().contains("overflow"));
    }

    #[test]
    fn malformed_references_are_refused() {
        let mut plan = sample();
        plan.operations[0].inputs = vec!["nope".into()];
        assert!(plan.validate().unwrap_err().detail().contains("nope"));

        let mut plan = sample();
        plan.tensors[0].shape = vec![Expr::Dimension("absent".into())];
        assert!(plan.validate().unwrap_err().detail().contains("absent"));

        // An enum dimension referenced numerically.
        let mut plan = sample();
        plan.tensors[0].shape = vec![Expr::Dimension("precision".into())];
        assert!(plan
            .validate()
            .unwrap_err()
            .detail()
            .contains("only through `select`"));
    }

    #[test]
    fn select_must_cover_exactly_the_enum() {
        let mut plan = sample();
        let LinearMemoryTerm { bytes, .. } = &mut plan.linear_memory[1];
        *bytes = Expr::Select {
            dimension: "precision".into(),
            arms: BTreeMap::from([("full".into(), Expr::Const(1))]),
        };
        assert!(plan
            .validate()
            .unwrap_err()
            .detail()
            .contains("exactly one arm"));
    }

    #[test]
    fn lists_must_be_sorted_and_unique() {
        let mut plan = sample();
        plan.tensors.swap(0, 1);
        assert!(plan.validate().unwrap_err().detail().contains("not sorted"));
    }

    #[test]
    fn live_sets_must_be_maximal_complete_and_sorted() {
        let mut plan = sample();
        plan.transient_live_sets = vec![vec!["forward".into()]];
        assert!(plan.validate().unwrap_err().detail().contains("backward"));

        let mut plan = sample();
        plan.transient_live_sets = vec![
            vec!["backward".into()],
            vec!["backward".into(), "forward".into()],
            vec!["forward".into()],
        ];
        assert!(plan
            .validate()
            .unwrap_err()
            .detail()
            .contains("strict subset"));

        let mut plan = sample();
        plan.transient_live_sets = vec![vec!["forward".into()], vec!["backward".into()]];
        assert!(plan
            .validate()
            .unwrap_err()
            .detail()
            .contains("lexicographically sorted"));
    }

    #[test]
    fn the_scope_and_the_equivalence_contract_are_coupled() {
        let mut plan = sample();
        plan.selection_scope = SelectionScope::PerParticipant;
        assert!(plan
            .validate()
            .unwrap_err()
            .detail()
            .contains("equivalence contract"));
        plan.equivalence_contract_hash = Some(Hash([9u8; 32]));
        plan.validate().expect("per-participant with a contract");

        plan.selection_scope = SelectionScope::UniformRun;
        assert!(plan
            .validate()
            .unwrap_err()
            .detail()
            .contains("must not carry"));
    }

    /// The plan states logical demand. Naming a backend is not a smaller plan, it is a different
    /// kind of statement, and it is refused.
    #[test]
    fn physical_backend_content_is_refused() {
        let mut plan = sample();
        plan.operations[0].family = "vulkan_gemm".into();
        assert!(plan
            .validate()
            .unwrap_err()
            .detail()
            .contains("physical backend content"));

        let mut plan = sample();
        plan.tensors[2].layout = vec!["driver_padded".into()];
        assert!(plan
            .validate()
            .unwrap_err()
            .detail()
            .contains("physical backend content"));
    }

    #[test]
    fn unknown_members_and_operators_are_refused() {
        let plan = sample();
        let mut value = plan.to_value();
        if let Value::Map(members) = &mut value {
            members.push((Value::Text("extra".into()), Value::Integer(1.into())));
        }
        let bytes = to_canonical_vec(&value).unwrap();
        assert!(LogicalResourcePlan::decode_canonical(&bytes)
            .unwrap_err()
            .detail()
            .contains("unknown member"));

        let bogus = Value::Array(vec![Value::Text("pow".into()), Value::Integer(2.into())]);
        assert!(expr_from_value(&bogus)
            .unwrap_err()
            .detail()
            .contains("unknown expr operator"));
    }

    #[test]
    fn non_canonical_bytes_are_refused() {
        let plan = sample();
        // ciborium's own encoder preserves insertion order and does not shorten heads, so an
        // unsorted member list produces well-formed but non-canonical bytes.
        let Value::Map(members) = plan.to_value() else {
            unreachable!()
        };
        let mut reversed = members;
        reversed.reverse();
        let mut bytes = Vec::new();
        ciborium::ser::into_writer(&Value::Map(reversed), &mut bytes).unwrap();
        assert!(LogicalResourcePlan::decode_canonical(&bytes)
            .unwrap_err()
            .detail()
            .contains("not canonical"));
    }

    #[test]
    fn the_byte_ceiling_is_checked_before_anything_else() {
        let oversized = vec![0u8; LOGICAL_RESOURCE_PLAN_BYTES_MAX + 1];
        let err = LogicalResourcePlan::decode_canonical(&oversized).unwrap_err();
        assert_eq!(err.slug(), "LogicalResourcePlanExceedsPolicy");
    }

    #[test]
    fn the_derivation_budget_is_derived_and_capped() {
        assert_eq!(plan_derivation_fuel(0, 0), PLAN_DERIVATION_FUEL_BASE);
        assert!(plan_derivation_fuel(10, 100) > PLAN_DERIVATION_FUEL_BASE);
        assert_eq!(
            plan_derivation_fuel(u64::MAX, u64::MAX),
            PLAN_DERIVATION_FUEL_CEILING,
            "the ceiling is absolute"
        );
    }

    #[test]
    fn a_binding_must_match_the_declared_domains() {
        let plan = sample();
        assert!(plan
            .check_binding(&binding(99, "full"))
            .unwrap_err()
            .detail()
            .contains("outside its declared range"));
        assert!(plan
            .check_binding(&binding(1, "quarter"))
            .unwrap_err()
            .detail()
            .contains("not one of its declared arms"));
        let short = Binding::from([("micro_batch".to_string(), DimensionValue::Uint(1))]);
        assert!(plan.check_binding(&short).is_err());
    }
}
