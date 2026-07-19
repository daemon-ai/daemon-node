> **Provenance:** the archived hardware-validation findings that informed the compute-backend
> selection layer. The prototype workspace this report describes has been retired; the
> run instructions and branch/worktree references below are historical.

# Burn-over-`HostBackend` — Phase-C architecture-gate SPIKE: findings

**Status:** SPIKE (validated knowledge + throwaway prototype). Nothing here is production code or
lands on `vhc-integration`.
**Gate:** the explicit Phase-C entry gate named in `daemon-vhc-refactor.md` §7 and
`daemon-vhc-architecture.md` §3.2 ("A nontrivial Burn model running over `HostBackend` is an
explicit architecture gate, to be prototyped before this surface is treated as settled").
**Tier reached:** **Tier 1 complete (bit-exact)** + **Tier 2 complete** (op-inventory, fused
`ModuleOps`, handle lifetime, stale-handle fault). **Tier 3 (real wasm32/wasmtime) deliberately not
attempted** — reasoning in §7.

**Branch:** `vhc/spike-burn-hostbackend` (worktree `/home/j/experiments/daemon-worktree/vhc-spike-burn`),
off `vhc-integration`.
**Commits:** `1018dde` (prototype), plus this report.
**Prototype:** detached mini-workspace at `crates/vhc/spike/` (path-deps nothing in the tree; not a
member of the daemon-node workspace — its globs are `crates/vhc/{contracts,sdk,host}/*`).

## How to run

```bash
cd /home/j/experiments/daemon-worktree/vhc-spike-burn/crates/vhc/spike
CARGO_BUILD_JOBS=5 nix develop /home/j/experiments/daemon-worktree/vhc-spike-burn \
  --command cargo test --offline -j5 -- --nocapture --test-threads=1
```

- `tests/parity.rs::hostbackend_matches_native_forward_and_backward` — Tier-1 equality + op report.
- `tests/parity.rs::tape_walk_is_handles_in_handles_out_no_intermediate_readback` — the §3.3 claim.
- `tests/tier2.rs::nn_module_ops_lower_and_run_over_boundary` — fused `ModuleOps` path.
- `src/lib.rs` `stale_handle::reading_unregistered_handle_faults` — §7 stale-handle behavior.

---

## 0. TL;DR / recommendation

**Recommendation: PROCEED to Phase C codegen — with amendments (§6).** The core question is answered
**yes**: an ordinary `Autodiff<HostBackend>` transformer block runs correctly with every tensor op
as an indirect, handle-based call on opaque `u64` handles into a host-side real Burn backend, and
the forward output, scalar loss, and **all** weight gradients match a native `Autodiff<NdArray>` run
**bit-exactly**.

The single most important finding reshapes Phase C's cost and risk:

> **Burn 0.21 already ships the `compute@` boundary.** `burn-router` (`BackendRouter<R>`,
> `RunnerClient`, `Runner<B>`) + `burn-ir` (`OperationIr`, `TensorId`, `TensorIr`, `TensorData`) are
> *exactly* a handle-based op-lowering backend: opaque `TensorId` handles, rank/dtype-erased ops as a
> single serializable enum, guest-side metadata, guest-side refcount+drop, a single blocking
> readback with a typed error. `OperationIr`/`TensorIr`/`TensorData` all already derive
> `Serialize`/`Deserialize`.

So Phase C's "codegen the surface from Burn's `Backend`/`AutodiffBackend` traits" (refactor §7) should
be **reframed**: the governed artifact is not ~200 hand-lowered trait methods, it is the **pinned
`burn_ir::OperationIr` grammar + the `burn-router` runner dispatch**. The "generator" shrinks to
(a) the wasm import shim that moves CBOR op-blobs + `u64` handles, and (b) a conformance test that the
pinned op set round-trips. This is a large de-risking of Phase C.

The honesty note in architecture §3.2 ("Burn's `Backend` trait is a Rust API, not a stable wire ABI")
is **confirmed and sharpened**: the ops *are* serializable, but the wire format is literally
"CBOR of `burn_ir::OperationIr` **at one pinned Burn version**". A Burn bump is an ABI event.

---

## 1. What was built (fidelity)

`HostBackend = BackendRouter<AbiChannel>`. Every tensor primitive is a `RouterTensor<AbiClient>` =
an opaque `TensorId(u64)` + guest-cached `shape`/`dtype` + an `Arc<AtomicU32>` refcount. The model is
authored in **ordinary Burn** (`burn::tensor`, `burn::nn`, `Autodiff<…>`) with zero boundary
awareness.

- **The boundary** (`trait ComputeBoundary`) is object-safe and carries **only CBOR byte buffers and
  `u64` handles** — no Burn generic and no native tensor type crosses. Each op is
  `ciborium`-serialized and dispatched through a `dyn` call: the faithful analogue of a *synchronous
  wasm host-import* (which is what a `compute@` import is).
- **The host** (`NdArrayHost`) wraps `burn_router::Runner<NdArray<f32,i64,i8>>`, owns the
  `TensorId → real ndarray tensor` handle table, deserializes each op and executes it on the real
  backend. Guests never see an ndarray tensor.
- **The model** is a genuinely nontrivial pre-norm **transformer block**: multi-head self-attention
  (4 heads) + MLP (64→256→64) + two LayerNorms + two residuals, `batch=2, seq=16, d_model=64`. Run
  **forward and backward** under `Autodiff<HostBackend>`.

**Fidelity caveats (honest):** the boundary is in-process synchronous `dyn`+CBOR, not wasm32 under
wasmtime (Tier 3). Handles are minted from a plain guest-side `u64` counter, not the ABI's
`(kind,generation,index)` layout (§7.2) — but `TensorId` *is* just a `u64`, so the generational scheme
is drop-in. Weights are built from a host-independent seeded `Vec<f32>` (not `Config::init`) so both
backends get byte-identical parameters — that is what makes the comparison an equality check.

---

## 2. Tier-1 result (the gate)

```
output max|Δ| = 0e0   (2048 elems)
loss  host = 0.105156444  native = 0.105156444   |Δ| = 0e0
grad w_q/w_k/w_v/w_o/w_ff1/w_ff2  max|Δ| = 0e0   (up to 16384 elems)
```

**Bit-exact**, not merely tolerance-class — because both paths run the identical ndarray kernels; the
only difference is that one routes every op through the serialized handle boundary. This is the
strongest possible pass: the indirection is provably semantics-preserving for forward *and* backward.

**Tape-walk claim (architecture §3.3) — empirically validated:**

```
forward:  67 ops enqueued, readbacks = 0
backward: +126 ops (193 total), readbacks = 0
explicit grad extraction: readbacks = 1   ← the first and only readback
```

The autodiff tape is guest-side bookkeeping; the backward pass is **pure enqueueing of ops on opaque
handles** with **zero intermediate readbacks**. The real gradient tensors live host-side and are read
back only on explicit extraction. This is precisely "handles-in / handles-out".

---

## 3. Per-lowering-rule findings (refactor §7 list)

| # | Lowering rule | Verdict | Finding |
|---|---|---|---|
| 1 | **Pinned Burn version per ABI major** | ✅ clean, **load-bearing** | Wire = `CBOR(burn_ir::OperationIr)`; the enum derives `Serialize`/`Deserialize` already. But variant set/discriminants/fields are Burn-version-specific (0.21 added `ArgTopK`, `Cum{Sum,Prod,Min,Max}`, `Gather/ScatterNd`, `Cross`, module `Attention`, …). **Pin exact `burn = 0.21.0`** (daemon-node `Cargo.lock`, checksum `39474be…`). A Burn bump ⇒ ABI compute-major event. |
| 2 | **Associated types + rank/dtype genericity across a C boundary** | ✅ clean (thanks to a Burn property) | The primitive assoc types (`Float/Int/BoolTensorPrimitive`) are **rank-erased** (`type FloatTensorPrimitive: TensorMetadata + 'static` — no `const D`). One opaque handle serves all ranks and all three kinds; rank/dtype are **runtime data in the IR**, never type params. **No generics cross.** This is *the* enabling property. Hard floor: a Burn that reintroduces rank-in-type breaks the single-handle model. |
| 3 | **Tensor metadata & shape ownership** | ✅ **guest-side** | `RouterTensor` caches `shape`+`dtype`; `TensorMetadata::shape()/dtype()` answer **synchronously with no host call**. Output shapes are computed **guest-side** by the IR builders (`BinaryOpIr::create`, …). Proven: 0 readbacks across a fwd+bwd despite many `.dims()` calls. ⇒ **the compute ABI needs no `get_shape(handle)` import on the hot path.** Consequence: guest shape-inference (burn-ir's `*Ir::create`) is part of the pinned contract — the host must honor guest-computed output shapes. |
| 4 | **Handle lifetime & stale-handle behavior** | ✅ clean / ⚠️ needs one rule | Instance-class handles, **guest-side reference-counted** (`Arc<AtomicU32>`); last drop enqueues `OperationIr::Drop(TensorIr)` → host removes from the table (25 `Drop`s in one block). Matches ABI §7.1/§7.3. In-place/aliasing is expressed by **`TensorStatus` (`ReadWrite`/`ReadOnly`/`NotInit`) carried in every operand `TensorIr`** — must be preserved on the wire. **Stale handle:** the runner **panics** on an unknown id; the ABI must catch this at the readback/fence and surface a typed `StaleHandle`/`InvalidHandle` trap (§7.6), never a host crash (probe test confirms the fault). |
| 5 | **Error model (deferred, CUDA-style at fence/readback)** | ✅ clean | `register_op` returns nothing (infallible enqueue); `read_tensor_async` → `Result<TensorData, ExecutionError>`, `sync` → `Result<(), ExecutionError>`. Errors surface **only at readback/fence**, exactly as §3.3 mandates. Amendment: `ExecutionError` is stringly-typed (`WithContext`/`Generic`) — the ABI must define a stable mapping into the §7.5 `comp-error` codes / §7.6 trap slugs. |
| 6 | **Which autodiff state lives guest vs host** | ✅ clean, validated | The tape (graph + backward closures) lives **guest-side** in `burn-autodiff`, expressed over inner-backend `RouterTensor` handles; the actual gradient tensors live **host-side** by handle. Backward = 126 enqueued ops, 0 readbacks. ⇒ **no autodiff-specific ABI import is needed**; `backward@1`/`grad@1` (v1) can retire under `compute@2`. Autodiff is purely a guest SDK concern on the same enqueue/handle primitives. |

---

## 4. Tier-2 op-inventory (what a codegen/conformance suite consumes)

**Empirical surface, raw transformer block (one fwd+bwd), by `OperationIr` category:**

```
BaseFloat/{Reshape×45, SwapDims×26, Ones×5, MaskFill×2}   Drop×25   Init×13
Float/{Matmul×24, Exp×2, Recip×2, Sqrt×2, PowfScalar×1}
NumericFloat/{Add×14, Mul×24, MulScalar×8, SumDim×19, MeanDim×4, Div×5, Sub×3,
              AddScalar×2, LowerEqualElem×2, MaxDim×1, Mean×1, DivScalar×1}
```

**Fused `ModuleOps` path (nn::Linear + LayerNorm block):** `Module/Linear` lowers as a single fused
op, and its **backward decomposes into distinct fused ops** the codegen must also cover:
`Module/{LinearXBackward, LinearWeightBackward, LinearBiasBackward}`. `Float/Random` appears (nn
init) — RNG runs **host-side** (a `Distribution` + seed in the op), which the journal must record for
determinism.

**Full static surface (from `burn-router/src/runner.rs`, the complete dispatch match):** the lowerable
surface is the whole `OperationIr` tree — `BaseOperationIr` (reshape/swap/permute/flip/expand/unfold/
slice/gather/scatter/select/mask/cat/cast/empty/ones/zeros/…), `NumericOperationIr` (arith, reductions,
cmp, argmax/min, topk, clamp, cumsum/prod, …) for float **and** int, `FloatOperationIr` (exp/log/
trig/matmul/erf/…), `IntOperationIr` (bitwise, matmul), `BoolOperationIr`, and `ModuleOperationIr`
(embedding, linear, conv1/2/3d, pools, attention, interpolate, ctc, rfft…) — **each already
enumerated and dispatched by the router runner**. This match table *is* the op inventory; a codegen
generator would consume it rather than the trait method list.

**Special-case rules surfaced (Tier-2 answers):**
- **Scalar returns:** there is no scalar-tensor readback op; a scalar is a rank-0/[1] tensor read via
  the normal `read_tensor` → `TensorData` path. No special ABI variant needed.
- **Bool/int tensors:** first-class — same `TensorId` handle, `dtype`-tagged; `NumericInt`/`Bool`
  op families. Attention masks etc. flow as bool tensors.
- **Reshape/slice/metadata ops:** ordinary ops (`BaseFloat/Reshape`, `Slice`, `SwapDims`); dominate
  the op count (~30% here) but are cheap on the wire.
- **In-place semantics:** **not** separate ops — encoded by `TensorStatus` on each operand `TensorIr`
  (`ReadWrite` ⇒ host may mutate/free the buffer in place).
- **Multi-output ops:** e.g. `MaxDimWithIndices`, `MaxPool2dWithIndices`, `deform_conv2d_backward`
  register several output handles from one op — the ABI must allow N output handles per op.

**Gaps / not-yet-lowerable (declare in the ABI):**
- **Quantization / `QFloat`:** `FloatOperationIr::Quantize/Dequantize` and QFloat readback are
  `todo!()` in the router runner — **quantized tensors do not lower today.** Defer or reserve.
- **`OperationIr::Custom`:** the runner **panics** ("Can't execute custom operation here") — matches
  architecture §3.2's separate host-side custom-op registry; custom/fused kernels are *not* part of
  the generic surface by design.
- **`OperationIr::Distributed`:** feature-gated (`distributed`), `todo!()` — out of scope.

---

## 5. Perf smell (NOT the gate)

For one fwd+bwd of the tiny block: **231 ops**, **~35 KB** total CBOR op-stream (**avg 153 B/op**),
**13 data uploads (~208 KB)**, **8 readbacks (~205 KB)**. Order-of-magnitude read:

- **Op-stream is small** (hundreds of bytes/op, one indirect call/op). A real training step (thousands
  of ops) ⇒ low-MB op-stream and low-thousands of boundary crossings per step — non-trivial but tiny
  next to the kernel time on real tensors, since **tensors never cross** (only the IR does). This is
  exactly the architecture's premise (§3.4: sandbox constrains authority, not throughput).
- **Bulk bytes are in uploads/readbacks, and are rare.** The spike inlined `TensorData` as CBOR;
  production must route these through `BufferHandle`/`export`/`import` (§3.4), not inline them.
- **Cheap wins if ever needed:** batch op submission (drain the queue in chunks rather than one
  import/op), and a non-CBOR compact codec for the op-stream. Neither is a gate concern.

---

## 6. Specific ABI-spec amendments Phase C needs

Against `daemon-vhc-abi-spec.md` (its compute section is currently reserved):

1. **Define `compute@2` payload as `CBOR(burn_ir::OperationIr)` at a pinned Burn version.** Add a
   normative constant pinning `burn = 0.21.0` (checksum `39474be…`). State the compatibility rule:
   an additive Burn change *might* be a compute-world minor, but variant **insertion** (which shifts
   discriminants) is a compute-**major**. Recommend adopting `burn-ir` as the compute contract rather
   than a bespoke hand-curated op list — reframing refactor §7's "codegen" as "pin + shim + conformance".
2. **Handle model:** one opaque `u64` `TensorId` per tensor, kind/dtype-tagged in the op, instance-class,
   **guest reference-counted**, released via `OperationIr::Drop`. Preserve **`TensorStatus`** on the
   wire (in-place hint). Map unknown-handle host faults → typed `StaleHandle`/`InvalidHandle` trap.
   Allow **N output handles per op**.
3. **Metadata is guest-authoritative** (shape/dtype/rank): no host metadata import; pin burn-ir's
   shape-inference (`*Ir::create`) as part of the contract — the host MUST produce outputs of the
   guest-computed shape.
4. **Error model:** enqueue infallible; errors deferred to `read_back`/fence; define the mapping from
   Burn's `ExecutionError` + runner panics into the §7.5 `comp-error` codes / §7.6 trap slugs.
5. **Autodiff:** no autodiff ABI surface; retire `backward@1`/`grad@1` for v2 (guest-side tape on the
   same enqueue/handle primitives).
6. **Explicitly reserve/defer:** `QFloat`/quantization (runner `todo!()`), `OperationIr::Custom`
   (host custom-op registry, architecture §3.2), `Distributed` (feature-gated). Declare the
   **rank-erased-primitive Burn** (≳0.14) as a hard ABI floor.
7. **Bulk data:** readback/upload `TensorData` must ride `BufferHandle` (§3.4), not inline in the
   op-stream.

---

## 7. Tier 3 (real wasm32/wasmtime) — deliberately not attempted, with reasoning

A full wasmtime host-function harness was **not** built, and this is a defensible stop, not a blocked
tier:

- The property Tier 3 would add — "the op stream survives real linear-memory constraints" — is
  **already answered**: the spike proves the entire op stream + handles + readback are representable as
  **CBOR byte buffers + `u64` handles**, which is precisely what crosses a wasm import (guest marshals
  bytes into linear memory, host reads them; handles are integers). No native tensor and no Rust
  generic ever needs to cross — the hard part.
- The remaining Tier-3 work (a `wasmtime::Linker` with `submit_op`/`read_tensor`/`create_from`/
  `read_into` host functions over a shared linear-memory arena) is **plumbing already specified** by
  ABI §3.4/§7.4 and is disproportionate to the marginal knowledge for a gate decision.
- **Recommendation:** fold Tier 3 into Phase C proper (or Phase B's testkit), where the wasm import
  shim is being built anyway, rather than duplicating it as spike scaffolding.

---

## 8. Honest limits of this spike

- **Real backend was ndarray CPU only.** CUDA/wgpu are the same `BackendIr` seam (the `Runner<B>` is
  generic over `BackendIr`, which cuda/wgpu implement), but their **deferred-error timing** (async
  device failures at fence) is not exercised here — only the *shape* of the error path is.
- **Bit-exact here is an artifact** of both sides running ndarray; cross-hardware remains a tolerance
  class per architecture §3.6/§10. The spike validates *lowering correctness*, not cross-backend
  numerics (which is explicitly a non-goal).
- **No wasm sandbox, budgets, or journal** — the spike models only the compute op boundary, not the
  surrounding ABI machinery (that is Phases A/B, already landing).
- The handle allocator is a plain counter, not the generational `(kind,gen,index)` scheme (§7.2);
  compatible but not exercised for ABA/stale-generation behavior.
