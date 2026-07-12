# daemon swarm training — tensor ABI & experiment SDK (interface specification)

**Status:** interface specification draft (companion to
[`swarm-training-spec.md`](swarm-training-spec.md); that document owns the architecture, this one
owns the contract)
**Scope:** the complete host↔guest interface between the `daemon-train` worker and run-defined
**experiment modules** (`tensor-abi@1`, import namespace `tabi@1`), plus the guest SDK
(`daemon-train-sdk`) that makes the ABI pleasant to author against. Everything an experiment can
observe or effect is defined here; anything not defined here does not exist for a module.

Referenced sections of the architecture spec: §4.3 (seam rule), §5.1 (experiment module), §5.2
(host vocabulary), §5.3 (SDK profiles), §5.6 (determinism), §6.1 (envelope), §6.4 (round
protocol — the host obligations behind §5.11 staging and the §2.3 barrier), §6.5 (admission),
§10.2 (worker protocol), §10.5 (preemption), §12 (security), §16 (versioning).

---

## 1. Design tenets (why the interface looks like this)

- **T1 — Tensor data never crosses the boundary.** Guests hold `u64` handles; bytes live in host
  (GPU/CPU) memory. The wasm linear memory carries logic, config, and payload *headers* only.
  This is simultaneously the performance model (zero-copy GPU residency), the budget model (a
  64 MiB guest cannot hoard VRAM), and the security model (§12).
- **T2 — The host calls the guest; never the reverse.** All control flow enters through the
  guest exports (§4) at host-chosen moments. There are no callbacks, no reentrancy, no guest
  threads, no yields. Consequences: GPU-governor preemption is possible between any two calls
  (architecture §10.5), fuel accounting is per-call, and handle arenas can be step-scoped.
- **T3 — Guest memory is cache, never state.** Every durable value lives host-side (params,
  persistent tensors, det-lane persistents), keyed by registration order and name. The host may
  drop and re-instantiate the module at any entry-point boundary (crash recovery, preemption,
  upgrade); re-running `da_build` deterministically re-derives every handle. A module that
  hides state in linear memory across rounds is *incorrect* (and cannot corrupt the swarm — it
  only desyncs itself, which digests catch, architecture §5.6).
- **T4 — Errors are traps.** Host functions never return status codes. Misuse (bad handle,
  shape mismatch, phase violation) traps immediately with a typed code (§3.6); the worker maps
  traps to the `Module` error class (architecture §13). Guest code therefore contains no
  error-handling ceremony — the SDK surface is plain values.
- **T5 — Mode-blind guests.** The same module bytes serve `meta`, `trace`, and `execute`
  (architecture §5.1). No import reveals the mode; a module cannot (and must not) behave
  differently per mode. Fidelity loss from value-dependent control flow is detected and
  reported, not prevented (§2.4).
- **T6 — Determinism is layered, not global.** Guest wasm is fully deterministic (§7). Native
  ops are fast and vendor-variant. The `det` lane (§5.9) is bit-exact everywhere. The ABI makes
  the lane of every handle explicit so the agree-path cannot accidentally touch vendor-variant
  numerics — lane confusion is a trap, not a silent desync.

---

## 2. Module shape & execution model

### 2.1 Module requirements

An experiment module is a **core wasm module** (not a component), target
`wasm32-unknown-unknown`, validated against the table below — at build/freeze time by the
authoring tooling (`daemon-cli swarm module check`, §10.6) and **re-validated independently by
every peer at assess time** (architecture §6.5). The registry/coordinator stores and lists the
module but neither parses nor executes wasm (architecture §11.1); nothing downstream trusts the
freeze-time result:

| Requirement | Value |
|---|---|
| imports | only from module namespace `tabi@1` (§5) — anything else fails validation |
| exports | the `da_*` set (§4) + `memory` |
| wasm features allowed | MVP + `multi-value` + `sign-ext` + `bulk-memory` + fixed-width `simd128` + `tail-call` |
| wasm features forbidden | threads/atomics, relaxed-simd, reference types beyond MVP tables, GC, exceptions, memory64, multi-memory, component model |
| size | ≤ 32 MiB binary (envelope artifact; typical SDK module ≈ 1–4 MiB) |

The **required capability set** published in the envelope (architecture §6.1) is derived
mechanically: it is the module's static import list. No execution is needed to know what a
module requires. The envelope's copy is a *pre-screen convenience* (peers can rule a run out
before fetching the module); the authoritative check is each peer re-deriving
`imports(module) ⊆ worker.capabilities` from the verified module bytes at assess
(architecture §6.5).

### 2.2 Runtime configuration (wasmtime host profile)

Fixed host-side settings, part of the ABI contract because they affect observable semantics:

- `cranelift_nan_canonicalization(true)` — closes the NaN-payload nondeterminism hole; with
  threads off and relaxed-simd off, guest execution is fully deterministic (wasmtime's own
  deterministic-execution guidance).
- `consume_fuel(true)` — **fuel is the deterministic budget** (same module + same inputs traps
  at the same instruction, everywhere). Budgets in §8.
- `epoch_interruption(true)` — wall-clock backstop (watchdog against pure-guest spins) and the
  **preemption lever**: governor pause/abort (architecture §10.5) epoch-interrupts an in-flight
  entry point at op granularity, then drops the instance (T3 makes that always safe). Never
  part of consensus, because epochs are explicitly non-deterministic.
- Pooling allocator + `InstancePre` + `memory_init_cow(true)` — instantiation is µs-scale, so
  T3's "re-instantiate at any boundary" is cheap in practice.
- Module compilation is cached keyed by `(module blake3, wasmtime version, tabi minor)`;
  compile-once per run per host.

### 2.3 Lifecycle mapping (architecture §6.2 phases → ABI calls)

```
Warmup            instantiate → da_abi() → da_build(config)
                  → host materializes params (checkpoint load, else deterministic init)
                  → [meta already ran at assess time; execute-mode OOM probe fixes micro-batch]

RoundTrain (per   for each micro-batch j of M in inner step s:
 inner step s;        da_step(batch, inner_step=s, mb_index=j,  # forward + backward
 s of H per round,            mb_count=M, step_seqs=Q)          #   (accumulate; Q = sequences
 H from the                                                     #    this inner step, §4)
 module manifest  da_inner_update(inner_step=s)                # guest applies inner optimizer
 §6.2)            [governor may pause/abort between (or inside) any two calls, §8]

Round end         u = da_make_update(round)                    # compress → update container
                  → host seals container → payload bytes → transport (commitment + gossip)
                  → transport overlap: peers prefetch committed payloads while others still
                    train (architecture §6.4 step 2)

BARRIER: ingest   da_ingest_updates(round, count=N)            # host staged exactly the round
                                                               #   record's committed set (§5.11)
                  → guest: decode (det lane, streaming) → aggregate → outer step
                  → host: snapshot round base (§5.9), digest state, emit Digest
                  → only then may round r+1's first da_step run (architecture §6.4 I2;
                    the reserved "pipelined" round mode shifts this by one round and must be
                    declared in the manifest, §6.2)

Cooldown          host checkpoints from canonical state (no guest involvement)
Any boundary      host may drop the instance; Warmup path re-runs da_build (T3)
```

`da_step`/`da_inner_update` split (rather than one `train_step`) is deliberate: the host owns
gradient-accumulation cadence and micro-batch sizing (OOM probe), so the boundary between
"accumulate" and "apply" must be host-visible; and every guest call is a preemption point. The
round's inner-step count H comes from the module manifest via the envelope (`[data]
.steps_per_round`, architecture §6.1) — the host paces the calls; the guest never counts
rounds. `inner_step` is **run-monotonic** (it never resets per round), so guest LR schedules
are plain functions of it.

### 2.4 Host modes (one module, three interpreters)

| | `execute` | `meta` | `trace` |
|---|---|---|---|
| tensors | real, GPU (`Autodiff<B>`) | shape/dtype symbols only — no allocation | symbols + op-graph capture |
| `scalar@1` | real value (GPU sync) | returns `0.0` **and sets the `value_dependent` flag** | `0.0`, flag, node recorded |
| `metric@1`, `log@1` | async readback / rate-limited log | no-op (sizes recorded) | recorded as nodes |
| `upd_push_*` | real serialization | section sizes accumulated → payload estimate | recorded |
| fuel | budget enforced (§8) | measured → becomes the execute op-budget seed | measured |
| output | training effects | `MetaReport` (§6.4) | `TraceGraph` (§6.5) |

A `meta` (or `trace`) run covers the full lifecycle once: `da_build` → one `da_step` at the
envelope's representative batch shape → `da_inner_update` → `da_make_update` → then
`da_ingest_updates` **twice**, with 1 and with `min_peers` phantom updates synthesized from the
module's own meta `da_make_update` section shapes — two points fit the linear per-peer model
(`ingest_fuel_base` + n × `ingest_fuel_per_peer`, §6.4) that execute-mode budgets scale by the
actual staged count (§8; a run at `max_peers` must not trap a budget seeded at `min_peers`).
That single pass yields every §6.4 estimate.

Value-dependent control flow (branching on `scalar@1` results) executes fine but makes
`meta`/`trace` under-approximate the real run; the flag propagates into eligibility
(requirements fall back to author-declared bounds — architecture §5.1, open question 9).

---

## 3. ABI conventions

### 3.1 Type legend (used throughout §4–§5)

| notation | core wasm | meaning |
|---|---|---|
| `T` | `i64` | tensor handle, native (GPU) lane |
| `D` | `i64` | tensor handle, det lane (CPU fp32) (§5.9) |
| `U` | `i64` | update-container handle (§5.11) |
| `B` | `i64` | batch handle (§5.10) |
| `u` | `i32` | unsigned 32-bit (dims, counts, enums, bools 0/1) |
| `x64` | `i64` | unsigned/signed 64-bit scalar (round numbers, ignore_index) |
| `f` | `f64` | floating scalar (hyperparameters, readouts) |
| `str` | `i32, i32` | (ptr, len) UTF-8 in guest memory, host-read |
| `dims` | `i32, i32` | (ptr, rank) — array of `u32` extents in guest memory, rank ≤ 8 |
| `vec<T>` | `i32, i32` | (ptr, count) — array of `u64` handles in guest memory |
| `bytes` | `i32, i32` | (ptr, len) guest memory span |
| multi-return | wasm multi-value | e.g. `-> (T, T)` |

All integers little-endian; pointers are `u32` offsets into the single exported `memory`. The
host validates every span against the memory bounds (out-of-range ⇒ trap `MemOob`).

### 3.2 Dtypes

`dtype` enum (`u`): `0=F32, 1=BF16, 2=F16, 3=I64, 4=I32, 5=U32, 6=U8, 7=Bool`. Quantized/packed
data is carried as `U8` tensors with documented layouts (§6.6) — packing is an op property, not
a dtype. The det lane is implicitly `F32` (plus `U8`/`U32` for staged payload sections).

### 3.3 Handles

- Handles are opaque `u64`, nonzero; `0` is never valid.
- **Stable handles** — params, persistents, det persistents: equal to their 1-based registration
  index within their class, assigned in `da_build` registration order (= the canonical state
  dict order, architecture §5.1/§9). Deterministic across re-instantiation (T3).
- **Step handles** — everything else: allocated from a per-entry-point generational arena; all
  step handles are invalidated wholesale when the entry point returns. Using a stale handle
  traps (`StaleHandle`). Holding a step handle in a guest global across calls is therefore
  structurally impossible to misuse silently.
- **Eager free:** `drop@1` (§5.1) releases a step handle (and its backing tensor) immediately —
  the SDK implements Rust `Drop` on its tensor types over it, so intermediates free at scope
  exit exactly like tch/Burn RAII. This is what keeps params-scale ingest streaming (peak ≈ one
  accumulator + one decoded tensor, §5.9) instead of accumulating `count` dense decodes until
  return. Dropping a stable handle traps (`InvalidHandle`); the wholesale-free-at-return rule
  remains as the backstop for guests that never drop.
- Live step-handle count is budgeted (§8).

### 3.4 Lanes

Every tensor handle carries a lane tag: **native** (`T`, GPU, vendor-variant numerics, fast) or
**det** (`D`, CPU fp32, bit-exact everywhere, ingest-only, streaming working set §5.9). Ops
accept exactly one lane
(§5); mixing traps (`LaneMismatch`). The only lane crossings:

- **into det:** `upd_tensor@1` stages received payload sections as `D` (bytes are canonical, so
  det by construction), plus `det_persistent@1` state, `det_zeros@1`, and exactly one
  state-carrying read: `det_param@1`, a view of a param's **round-base master snapshot** (§5.9)
  — deliberately *never* the live master, which carries the round's vendor-variant local drift.
  **There is no bridge for arbitrary native tensors** — no step tensor, activation, gradient,
  or live param can enter the det lane. Every det value therefore derives from canonical
  payload bytes, det persistents, `f64` immediates, and the digested round-base state. Lane
  discipline is thus a proof shape: the agree-path cannot observe vendor-variant numerics by
  type rule rather than by convention.
- **out of det:** `det_reset_param_to_base@1` + `det_axpy_param@1` write the fp32 canonical
  masters (§5.9) — the only det→param doorway, and the only mutation the agree-path performs.

Local (native) results reach peers only through the update container as bytes (§5.11); the
host-retained `param_round_base@1` view (§5.1) gives payload production its θ-at-round-start
baseline without any guest-side snapshot bookkeeping.

### 3.5 Phase legality

Imports are legal only inside specific guest entry points (else trap `PhaseViolation`):

| import group | `da_build` | `da_step` | `da_inner_update` | `da_make_update` | `da_ingest_updates` |
|---|---|---|---|---|---|
| `param`, `persistent`, `det_persistent` (registration) | ✓ | | | | |
| creation / shape / math / NN (§5.2–5.6) | | ✓ | ✓ | ✓ | |
| `assign`, `detach` | | ✓ | ✓ | ✓ | |
| `backward`, `grad`, `zero_grads` | | ✓ | ✓ | | |
| optimizer steps (§5.7) | | | ✓ | | |
| compression (§5.8) | | | ✓ | ✓ | |
| `param_round_base` | | | | ✓ | |
| `upd_new`, `upd_push_*` | | | | ✓ | |
| `upd_sections/…/upd_tensor` | | | | | ✓ |
| det ops incl. `det_assign`, `det_chunk_scatter_add`, `det_axpy_param` (§5.9) | | | | | ✓ |
| batch accessors (§5.10) | | ✓ | | | |
| `drop` (step handles only, §3.3) | | ✓ | ✓ | ✓ | ✓ |
| `scalar`, `metric`, `log`, introspection | | ✓ | ✓ | ✓ | ✓ (det lane only for `scalar`, §7) |

The matrix is normative; it encodes the seam between local math (native lane, rounds) and
consensus math (det lane, ingest) at the type-system level of the ABI.

### 3.6 Traps (the complete taxonomy)

Every host-raised trap carries one code; the worker surfaces it in the `Module` error
(architecture §10.2, §13) as `{code, import, entry_point, detail}`.

`InvalidHandle`, `StaleHandle`, `LaneMismatch`, `PhaseViolation`, `ShapeMismatch`,
`DtypeMismatch`, `RankOverflow` (>8), `MemOob`, `AllocFail` (guest `da_alloc` returned
0/misaligned), `PayloadOverflow` (§5.11 cap), `BudgetFuel`, `BudgetEpoch`, `BudgetMemory`,
`BudgetHandles`, `BudgetOps`, `GuestPanic` (guest executed `unreachable`), `NameCollision`
(duplicate param/persistent name), `NotScalar` (`scalar@1` on numel≠1), `BadEnum`.

Guest-originated traps (`GuestPanic`) and host-originated traps are indistinguishable in
consequence: the entry-point invocation is aborted, step handles are freed, and the worker
decides retry vs `Module` error. No trap ever kills the worker (architecture §13).

---

## 4. Guest exports (complete)

```rust
// Memory & glue -------------------------------------------------------------
export memory: Memory;                       // single linear memory
fn da_alloc(size: u, align: u) -> u;         // host requests guest buffers through this
fn da_free(ptr: u, size: u, align: u);       // paired release (host calls after copying out)

// Identity ------------------------------------------------------------------
fn da_abi() -> u;                            // (major << 16) | minor the module was built for;
                                             //   host refuses major ≠ 1, minor > host minor
fn da_manifest(cfg_ptr: u, cfg_len: u)       // (ptr, len) canonical CBOR: name/version/sdk +
    -> (u, u);                               //   the cadence & round-mode block (§6.2) the
                                             //   envelope freeze copies out; pure function of
                                             //   the config bytes (H may be a config knob)
fn da_defaults() -> (u, u);                  // (ptr, len) canonical CBOR map: the defaults
                                             //   figment layer for [experiment.config] (§6.1
                                             //   of the architecture spec); may be empty map

// Lifecycle (§2.3) ----------------------------------------------------------
fn da_build(cfg_ptr: u, cfg_len: u);         // register params/persistents from config (§6.3)
fn da_step(batch: B, inner_step: u,          // one micro-batch: forward + backward.
           mb_index: u, mb_count: u,         //   mb_index ∈ [0, mb_count) within this inner
           step_seqs: u);                    //   step; step_seqs = Σ sequences across its
                                             //   micro-batches — so the guest can scale each
                                             //   micro-batch loss by size(batch)/step_seqs and
                                             //   accumulated grads equal the exact step mean,
                                             //   independent of the host's OOM-probed slicing
fn da_inner_update(inner_step: u);           // apply inner optimizer at accumulation boundary
fn da_make_update(round: x64) -> U;          // compress local progress into update container
fn da_ingest_updates(round: x64, count: u);  // decode + aggregate + outer step (det lane)
```

Rules:

- `da_manifest`/`da_defaults` are pure: callable before `da_build`, must not touch tensor
  imports (validated in meta mode at build/assess time). `da_defaults` takes no config (it *is*
  a config layer); `da_manifest` is a pure function of the frozen config bytes.
- Round/step counters are host-passed (never guest-tracked) so re-instantiation is transparent
  (T3).
- The host calls `da_free` for every buffer it obtained via `da_alloc` and every
  `(ptr,len)` return it has finished reading; the SDK's glue implements both over the guest
  allocator.
- A module that exports extra symbols is valid (ignored); missing any `da_*` export fails
  validation.

---

## 5. Host imports — the `tabi@1` vocabulary (complete)

Import module namespace: `"tabi@1"`. Field names carry per-op versions: `"matmul@1"`. Growth is
additive (new fields / new versions side-by-side); removal or semantic change requires namespace
`tabi@2` (architecture §16). The vocabulary below **is** `tensor-abi@1` in its entirety —
**108 imports**. Anything absent (I/O, clocks, randomness beyond §5.2's seeded forms, digests,
wall time) is deliberately inexpressible (T1/T2; architecture §5.2).

Broadcasting: binary elementwise ops broadcast trailing dimensions NumPy-style; shape conflicts
trap. Autodiff: every native op in §5.2–§5.6 is recorded on the tape in execute mode;
`backward@1` differentiates through the whole streamed graph (Burn's tape autodiff,
architecture §3).

### 5.1 State, memory & autodiff

| import | signature | semantics |
|---|---|---|
| `param@1` | `(name: str, dims, dtype: u, init: u, p0: f, p1: f) -> T` | register a trainable weight. `init`: `0=zeros, 1=ones, 2=uniform(p0,p1), 3=normal(p0=mean,p1=std), 4=trunc_normal`. Init seed is host-derived from `(run_id, name)` — author code carries no seeds. Storage dtype as declared; host additionally maintains the **fp32 canonical master** (§5.9). Registration order = canonical state dict. |
| `persistent@1` | `(name: str, dims, dtype: u, class: u) -> T` | auxiliary state surviving across rounds, zero-initialized. `class` (architecture §5.1): `0=local` — droppable, never swarm-checkpointed, never digested (inner moments, error feedback; peers rebuild in ≤H steps, architecture §9); `1=replicated` — **consensus state**: included in the round digest and carried fp32-exact in epoch checkpoints. Native-lane replicated state is rare (consensus inputs normally live det-side) but legal. |
| `det_persistent@1` | `(name: str, dims, class: u) -> D` | det-lane fp32 persistent, same `class` enum. Outer momentum / EMA that feeds the outer step **must** be `1=replicated` — a rejoiner rebuilding it from zero would compute a different outer step than incumbents every round thereafter (permanent desync); the digest+checkpoint coverage is what makes rejoin exact. `0=local` is for det-side scratch that is legitimately peer-divergent. |
| `drop@1` | `(h: T or D or U)` | eagerly free a **step** handle and its backing tensor (§3.3); the handle becomes stale. Stable handles (params, persistents) trap `InvalidHandle`. SDK: `impl Drop` on tensor types — authors get streaming memory behavior for free. |
| `param_round_base@1` | `(p: T) -> T` | native-lane read-only view of `p`'s fp32 master **as of the start of the round being trained** (host snapshots masters at the ingest barrier, after each `da_ingest_updates` returns — §2.3; after build/checkpoint-load for the first round). The DiLoCo-family pseudo-gradient baseline θ⁽ᵗ⁾, host-retained so it needs no guest bookkeeping and survives re-instantiation (T3). Prior art: OpenDiLoCo's CPU-offloaded baseline params (architecture Appendix A.8). |
| `backward@1` | `(loss: T)` | `loss` must be scalar-shaped (numel 1). Runs reverse pass; **accumulates** into per-param fp32 grad buffers (host-owned). Multiple calls per step accumulate (micro-batching falls out). |
| `grad@1` | `(p: T) -> T` | read-only view of `p`'s accumulated fp32 gradient. Traps if `p` is not a param. |
| `zero_grads@1` | `()` | clear all grad accumulators. Guest-called (typically in `da_inner_update` after applying). |
| `assign@1` | `(dst: T, src: T)` | overwrite a param/persistent's storage with `src` (shape/dtype must match; casts are explicit). The only native-lane mutation path besides fused optimizer steps. |
| `detach@1` | `(x: T) -> T` | cut the tape. |

### 5.2 Creation

| import | signature | semantics |
|---|---|---|
| `zeros@1` / `ones@1` | `(dims, dtype: u) -> T` | |
| `full@1` | `(dims, dtype: u, value: f) -> T` | |
| `arange@1` | `(len: u) -> T` | `I32`, `0..len` |
| `dropout@1` | `(x: T, p: f, salt: u) -> T` | deterministic mask from host-derived seed `(run_id, round, inner_step, salt)`. Identical across replays on one peer; **not** cross-peer meaningful (local math only). |

There is no unseeded RNG in the ABI: param init and dropout are the only stochastic surfaces,
both host-seeded (§7).

### 5.3 Shape & layout

| import | signature |
|---|---|
| `reshape@1` | `(x: T, dims) -> T` |
| `transpose@1` | `(x: T, d0: u, d1: u) -> T` |
| `slice@1` | `(x: T, dim: u, start: u, end: u) -> T` |
| `concat@1` | `(xs: vec<T>, dim: u) -> T` |
| `cast@1` | `(x: T, dtype: u) -> T` |
| `gather@1` | `(x: T, dim: u, idx: T) -> T` |
| `shape_rank@1` | `(x: T) -> u` |
| `shape_dim@1` | `(x: T, d: u) -> u` |
| `numel@1` | `(x: T) -> x64` |
| `dtype_of@1` | `(x: T) -> u` |

(The SDK tracks shapes guest-side; the query ops exist for assertions and generic code.)

### 5.4 Elementwise

Binary (broadcasting), each `(a: T, b: T) -> T`:
`add@1, sub@1, mul@1, div@1, pow@1, maximum@1, minimum@1`.
Scalar right-hand variants, each `(x: T, v: f) -> T`:
`add_s@1, sub_s@1, mul_s@1, div_s@1, pow_s@1`.
Unary, each `(x: T) -> T`:
`neg@1, abs@1, sign@1, exp@1, log@1, sqrt@1, rsqrt@1, tanh@1, sigmoid@1, erf@1, relu@1, silu@1,
gelu@1`.
And `clamp@1 (x: T, lo: f, hi: f) -> T`.

### 5.5 Reductions

| import | signature | note |
|---|---|---|
| `sum_all@1` / `mean_all@1` / `max_all@1` / `min_all@1` | `(x: T) -> T` | rank-0 result |
| `sum_dim@1` / `mean_dim@1` / `max_dim@1` | `(x: T, dim: u, keepdim: u) -> T` | |
| `l2_norm@1` | `(x: T) -> T` | rank-0; fused (clipping, metrics) |

### 5.6 NN fused

| import | signature | semantics |
|---|---|---|
| `matmul@1` | `(a: T, b: T) -> T` | batched; trailing-2-dim contraction |
| `embedding@1` | `(w: T, ids: T) -> T` | `w` param `[vocab, d]`; `ids` `U32/I32/I64` |
| `rmsnorm@1` | `(x: T, w: T, eps: f) -> T` | |
| `layernorm@1` | `(x: T, w: T, b: T, eps: f) -> T` | |
| `rope@1` | `(x: T, pos_start: u, theta: f, interleaved: u) -> T` | apply per q/k call |
| `flash_attn@1` | `(q: T, k: T, v: T, causal: u, scale: f) -> T` | `[b, h, s, d]`; GQA via head-count ratio k,v vs q |
| `softmax@1` | `(x: T, dim: u) -> T` | |
| `cross_entropy@1` | `(logits: T, targets: T, ignore_index: x64) -> T` | mean over non-ignored; rank-0 |

Coarse-grained composites are the v1 posture (fewer boundary crossings, fusion below the ABI);
the vocabulary can later carry finer ops side-by-side (architecture §18, open q. 8).

### 5.7 Optimizer steps (fused, mutating; legal only in `da_inner_update`)

| import | signature |
|---|---|
| `adamw_step@1` | `(p: T, g: T, m: T, v: T, step: u, lr: f, beta1: f, beta2: f, eps: f, wd: f)` |
| `nadamw_step@1` | `(p: T, g: T, m: T, v: T, step: u, lr: f, beta1: f, beta2: f, eps: f, wd: f, momentum_decay: f)` |
| `sgdm_step@1` | `(p: T, g: T, mom: T, lr: f, momentum: f, nesterov: u, wd: f)` |
| `signum_step@1` | `(p: T, g: T, lr: f, wd: f)` |

`p` param; `g` its grad view (or any same-shaped native tensor — composed update rules may
preprocess); `m/v/mom` persistents. Fused steps update the fp32 master and requantize to the
storage dtype (round-to-nearest-even), same as `det_axpy_param` (§5.9) — one write discipline
everywhere. LR schedules are guest `f64` math (deterministic wasm); no schedule ops exist.
Composed update rules out of §5.4 ops + `assign@1` are always available; fused steps are the
performance path (architecture §5.2).

### 5.8 Compression (native lane; legal in `da_inner_update` + `da_make_update`)

| import | signature | semantics |
|---|---|---|
| `topk_chunk@1` | `(x: T, chunk: u, k: u) -> (T, T)` | view `x` flattened as `[numel/chunk, chunk]` (`numel % chunk == 0`, pad guest-side); per-row top-k by magnitude. Returns `(values, indices: U32)` shaped `[n_chunks, k]`. Composes with `dct2` for the demo path (`chunk = tile²`). |
| `chunk_scatter@1` | `(vals: T, idx: T, chunk: u, dims) -> T` | inverse: dense from per-chunk sparse (zeros elsewhere) — error-feedback residuals |
| `absmax_pack@1` | `(x: T, chunk: u, bits: u) -> T` | `bits ∈ {1,2,4,8}`; per-chunk absmax codebook, layout §6.6; result `U8` |
| `absmax_unpack@1` | `(packed: T, chunk: u, bits: u, dtype: u) -> T` | |
| `dct2@1` / `idct2@1` | `(x: T, tile: u) -> T` | orthonormal 2-D DCT on `tile×tile` tiles, `tile ∈ {8,16,32,64,128}` (precomputed-matmul kernels, architecture §15.2) |

These six + the elementwise set express the sparse-loco, diloco-int8, and demo wire math
(architecture §5.3); 8-bit optimizer state is `absmax_pack/unpack` at `bits=8` composed around
optimizer steps (which is why this group is also legal in `da_inner_update`).

### 5.9 The det lane (CPU fp32, bit-exact; legal only in `da_ingest_updates`)

Model: every param has its declared storage dtype **plus one host-owned fp32 canonical
master**. Digests (architecture §5.6) and checkpoints (architecture §9) read the master, and
the master and storage never drift: every write path (optimizer steps §5.7, `assign@1`,
`det_axpy_param@1`) writes master-first, then requantizes to storage. Residency is a host
detail with a fixed contract: GPU-resident while native ops run (it is the mixed-precision
master the §5.7 steps update — already budgeted in the architecture §5.1 VRAM table), and
**materialized CPU-side at the ingest boundary**, where the det doorway ops rewrite it
fp32-exactly from the round base + canonical aggregate (~4 bytes/param of host RAM in flight,
~4.8 GB at 1.2B — the price of exact consensus).

Cross-peer master agreement after ingest is a *module property* the lane rules make easy and
digests verify: DiLoCo-family profiles rebase (`det_reset_param_to_base` discards the round's
vendor-variant local drift, then `det_axpy_param` applies the canonical aggregate); per-step
profiles (demo, H=1 rounds) never advance params in `da_inner_update` at all — the only param
writes are the det-lane sign-SGD + decoupled decay at ingest (`det_sign`, `det_param`,
`det_axpy_param`), so masters stay canonical by construction. A module that lets native writes
leak into post-ingest masters without a rebase only desyncs itself — digests catch it (T3,
architecture §5.6).

| import | signature | semantics |
|---|---|---|
| `det_zeros@1` | `(dims) -> D` | |
| `det_add@1` / `det_sub@1` / `det_mul@1` | `(a: D, b: D) -> D` | elementwise, fp32, fixed order |
| `det_scale@1` | `(x: D, alpha: f) -> D` | |
| `det_sign@1` | `(x: D) -> D` | |
| `det_sum@1` | `(xs: vec<D>) -> D` | summation in exactly array order — the reduce the profiles use post-clip |
| `det_l2norm@1` | `(x: D) -> f` | fixed-order accumulation; safe to branch on (det inputs ⇒ identical everywhere) |
| `det_absmax_unpack@1` | `(packed: D, chunk: u, bits: u) -> D` | decode staged payload sections |
| `det_chunk_scatter@1` | `(vals: D, idx: D, chunk: u, dims) -> D` | dense from per-chunk sparse (allocating form) |
| `det_chunk_scatter_add@1` | `(acc: D, vals: D, idx: D, chunk: u)` | **in-place** `acc[chunk, idx] += vals` in fixed order — the streaming-ingest hot path: decode one payload → scatter-add into one accumulator → `drop` the decode, so peak det memory is O(1 dense tensor) regardless of peer count (§3.3). `acc` is any writable `D` (typically `det_zeros`) |
| `det_idct2@1` | `(x: D, tile: u) -> D` | demo-profile decode |
| `det_assign@1` | `(dst: D, src: D)` | overwrite a det persistent (outer-momentum / EMA updates); `dst` must be a `det_persistent` |
| `det_param@1` | `(p: T) -> D` | read-only det view of `p`'s **round-base master snapshot** (the same snapshot `param_round_base@1` exposes natively — canonical by construction, digest-attested). Never the live master: a DiLoCo-family peer's live master carries local drift at ingest entry, which must not enter the lane. Uses: decoupled weight decay, outer math on θ |
| `det_reset_param_to_base@1` | `(p: T)` | `master(p) ← round-base snapshot` (§5.1 `param_round_base`). The DiLoCo-family outer step rebases before applying the aggregate; per-step profiles (demo) simply don't call it |
| `det_axpy_param@1` | `(p: T, x: D, alpha: f)` | `master(p) += alpha · x`, then requantize to storage — **the outer step**, and (with the reset above) the only det→param doorway (§3.4) |

Robust aggregation (median-norm clip, trimmed means, …) is deliberately *not* fused: guest
`f64` math over `det_l2norm` results is deterministic, so clipping logic is ordinary profile
code (architecture §12 — hardened SDK defaults, author-replaceable). There is no det matmul, no
det autodiff: the lane exists for decode → combine → apply at payload scale, nothing else.

**Memory discipline:** decoded payloads are params-scale (a dense fp32 at 1.2B is ~4.8 GB), so
the profiles ingest **streaming** — per staged update: decode → clip-scale →
`det_chunk_scatter_add` into one accumulator → `drop` the intermediates — keeping the det-lane
peak at ~2 dense fp32 tensors (accumulator + in-flight decode) regardless of `count`. Holding
`count` dense decodes to feed one `det_sum` call is legal but budget-hostile; `det_sum` remains
for small tensors and norm vectors. Determinism is unaffected: the staged order is fixed
(§5.11) and each in-place accumulation is a fixed-order fp32 sequence.

### 5.10 Batch access (legal only in `da_step`)

| import | signature | semantics |
|---|---|---|
| `batch_tokens@1` | `(b: B) -> T` | `U32 [batch, seq_len]` token ids, host data pipeline (architecture §8) |
| `batch_size@1` | `(b: B) -> u` | |
| `batch_seq_len@1` | `(b: B) -> u` | |

Targets are guest-composed (`slice` shift). v1 carries tokens only; document masks /
position arrays are a `tabi` minor addition when a preset needs them (§13).

### 5.11 Update container

Sealed frame is host-defined (§6.6: version, section table, blake3); section *contents* are
experiment-defined — the swarm never parses them (architecture §4.3, §7.3). The envelope's
`update_mb_max` caps the sealed size; `upd_push_*` traps `PayloadOverflow` at the source.

Build side (in `da_make_update`):

| import | signature |
|---|---|
| `upd_new@1` | `() -> U` |
| `upd_push_bytes@1` | `(upd: U, data: bytes)` |
| `upd_push_tensor@1` | `(upd: U, x: T)` — serialized as `(dtype, dims, LE data)`; packed `U8` tensors pass through verbatim |

Ingest side (in `da_ingest_updates`): the host has staged `count` peer updates — **exactly the
set committed by the signed `RoundRecord`'s root (architecture §6.4; the host verifies the set
object against that root before staging), in the set's total order (ascending node public-key
bytes — the ed25519 node identity, never the iroh id)**, each
hash-verified against its set entry. A peer that cannot assemble the full set does not call
ingest at all — it stalls or resyncs per the architecture §6.4 recovery ladder; subset ingest
never happens. This staging guarantee is a host obligation and a precondition of the
determinism contract (§7).

| import | signature |
|---|---|
| `upd_sections@1` | `(i: u) -> u` |
| `upd_kind@1` | `(i: u, s: u) -> u` — `0=bytes, 1=tensor` |
| `upd_bytes_len@1` | `(i: u, s: u) -> u` |
| `upd_read_bytes@1` | `(i: u, s: u, dst: bytes) -> u` — copies into guest memory, returns len |
| `upd_tensor@1` | `(i: u, s: u) -> D` — staged as det lane (§3.4) |

A peer's own update is included in the staged set (self-inclusive aggregation, matching the
profiles' math).

### 5.12 Readouts & telemetry

| import | signature | semantics |
|---|---|---|
| `scalar@1` | `(x: T or D) -> f` | numel-1 readout. Native lane: GPU sync — legal in `da_step`/`da_inner_update`/`da_make_update` only. Det lane: legal in `da_ingest_updates` (deterministic). Meta mode: `0.0` + `value_dependent` flag (§2.4). |
| `metric@1` | `(name: str, x: T or D)` | non-blocking readback → worker `RoundProgress` events (architecture §10.2); loss/grad-norm reporting without a sync. Accepts either lane (det-lane in ingest) |
| `log@1` | `(level: u, msg: str)` | host-rate-limited tracing; never consensus-relevant |
| `abi_minor@1` | `() -> u` | host's implemented minor within major 1 |

---

## 6. Data formats

### 6.1 Experiment config

The envelope's `[experiment.config]` TOML table (architecture §6.1) is canonicalized at
envelope-freeze time to **deterministic CBOR (RFC 8949 §4.2 core deterministic encoding)**;
those bytes are what `da_build` receives, what the envelope hash covers, and what the SDK
deserializes (serde). TOML value domain only (no CBOR exotica); float canonicalization per RFC
8949; key order lexicographic byte-wise.

### 6.2 `da_manifest` / `da_defaults`

Canonical CBOR. Manifest:

```cddl
manifest = {
  "name": tstr, "version": tstr, "sdk": tstr,
  "steps_per_round": uint,              ; H — inner-step cadence the host must pace (§2.3)
  "round_modes": [+ ("barrier" / "pipelined")],  ; apply orderings this experiment's math
                                        ;   tolerates (architecture §6.4); v1 hosts run barrier
  "min_round_interval_ms": uint,        ; 0 = any; eligibility gate against the coordinator's
                                        ;   cadence class (architecture §5.3.3 — demo declares
                                        ;   seconds-scale here and pipelined support)
}
```

The cadence block is *module-derived coordination data*: envelope freeze evaluates
`da_manifest` against the frozen config bytes (H may be a config knob) and copies
`steps_per_round` into `[data]` (architecture §6.1) exactly as it copies the capability set;
peers re-evaluate and verify manifest == envelope at assess. Defaults: a map in the same value
domain as `[experiment.config]`; the envelope authoring tool layers it lowest (module defaults
← TOML ← env ← CLI, architecture §6.1).

### 6.3 Registration effects of `da_build`

Ordered lists (params, persistents, det persistents) of `(name, dims, dtype, init/class)`. The
param list **is** the canonical state dict (checkpoint tensor order, digest coverage, chunking
for P2P sharing). Names are unique per class (`NameCollision`), ≤ 128 bytes, UTF-8.

### 6.4 `MetaReport` (assess/authoring output; architecture §6.5)

Canonical CBOR:

```cddl
meta-report = {
  "abi": uint,                       ; module's da_abi()
  "params": [* [tstr, [* uint], uint]],      ; name, dims, dtype
  "persistent": [* [tstr, [* uint], uint, uint]],   ; name, dims, dtype, class (0/1, §5.1)
  "det_persistent": [* [tstr, [* uint], uint]],     ; name, dims, class
  "param_bytes": uint, "master_bytes": uint, "grad_bytes": uint,
  "act_bytes_est": uint,             ; peak live activation estimate (shape propagation)
  "payload_bytes_est": uint,         ; from meta upd_push_* sizes
  "ingest_bytes_est": uint,          ; peak staged+working set under streaming discipline (§5.9)
  "host_ram_bytes_est": uint,        ; CPU side: masters + round base + offloaded persistents
                                     ;   + staging (feeds [requirements].ram_gb_min)
  "fuel": { * tstr => uint },        ; per entry point, measured in the meta pass (§2.4)
  "ingest_fuel_per_peer": uint,      ; linear model from the two-point ingest measurement
  "op_calls": { * tstr => uint },    ; per entry point — these seed the §8 execute budgets
  "ingest_op_calls_per_peer": uint,
  "ops_used": [* tstr],              ; static import scan (exact, no execution needed)
  "value_dependent": bool,
}
```

For `da_ingest_updates`, `fuel["da_ingest_updates"]`/`op_calls[…]` hold the *base* (count = 0
intercept) of the linear fit; execute budgets scale by the staged count (§8).

### 6.5 `TraceGraph` (audit export; architecture §14)

Canonical CBOR list of nodes `(op, version, inputs: [handle-ids], attrs, out-shape)` in call
order, per entry point — `daemon-cli swarm trace <run>` renders it. It is an *export* of what
executed, not an input format: nothing consumes it at runtime (the graph-IR alternative stays
rejected, architecture §18).

### 6.6 Packed-tensor & sealed-payload layouts

- `absmax_pack` layout (`U8` tensor): per chunk — `f16` absmax codebook scalar, then `bits`-wide
  codes packed LSB-first, chunk-major, zero-padded to byte. Documented so third-party tooling
  can decode payloads offline; guests never parse it (they call `*_unpack`).
- Sealed update container: `header {magic "DAUP", version u8, section_count u16}` then per
  section `{kind u8, len u64, dtype u8 + rank u8 + dims (tensor kind only)}`, then section
  bodies, then trailing blake3. The host writes and verifies this frame; `update_mb_max` is
  enforced on the sealed size at push time (sender) and before staging (receiver, architecture
  §7.3).

---

## 7. Determinism contract (ABI-level restatement of architecture §5.6)

1. **Guest execution is bit-deterministic.** Core wasm, no threads, no relaxed-simd, NaN
   canonicalization on, fuel (not epochs) as the semantic budget. Same module + same inputs ⇒
   same import call stream, everywhere, on every vendor.
2. **Native-lane ops are contract-free across peers.** `T`-lane results may differ per
   GPU/vendor/driver. They feed local training and locally-produced payload bytes only.
3. **Det-lane ops are bit-exact across peers.** fp32, fixed evaluation order, CPU kernels
   (architecture §15.2). `det_l2norm`/det-lane `scalar` are the only readouts guest logic may
   branch on inside `da_ingest_updates` — and the phase-legality matrix (§3.5) plus lane rules
   (§3.4) make violating this a trap, not a bug hunt.
4. **Ingest inputs are identical by host obligation and module discipline:** the staged set is
   the signed `RoundRecord`'s, in record order, self-inclusive (§5.11); config bytes are
   canonical CBOR (§6.1); round/step numbers are host-passed arguments; and the det-readable
   state — round-base master snapshots (`det_param`) and **`replicated` det persistents** — is
   cross-peer identical inductively: each round's base is the previous ingest's output, itself
   written det-only (§5.9's agreement discipline), and replicated persistents enter the round
   digest and ride epoch checkpoints fp32-exact (§5.1), so the induction survives joins,
   rejoins, and resyncs — there is no state a correct peer must possess that the checkpoint +
   retained records cannot reconstruct (architecture §6.4 I1). `local` state is outside the
   induction by definition.
5. Therefore: `da_ingest_updates` is a deterministic function of `(module, config, committed
   payload set, round, canonical round-base state)` — every peer computes the identical outer
   step and the identical post-ingest fp32 masters, which the host digests to *detect* any
   residual divergence (defense in depth, not a correctness dependency).

Init and dropout seeds are host-derived (§5.1, §5.2), so "same checkpoint + same batches +
same config" also replays identically *within* one peer — the property the conformance suite
(§11) and `just swarm-dev` loops rely on.

---

## 8. Budgets & sandbox profile (self-protection, not domain limits)

| budget | default | trap | notes |
|---|---|---|---|
| linear memory | 64 MiB | `BudgetMemory` | logic + config + payload headers only (T1) |
| fuel per entry point (except ingest) | `max(2^26, 8 × MetaReport.fuel[entry])` | `BudgetFuel` | deterministic (wasmtime `consume_fuel`); meta runs get a fixed generous constant |
| fuel for `da_ingest_updates` | `max(2^26, 8 × (fuel[ingest] + count × ingest_fuel_per_peer))` | `BudgetFuel` | **scales with the staged count** — a `max_peers` round must not trap a budget measured at `min_peers` (§2.4, §6.4) |
| epoch deadline | 5 s pure-guest compute per call | `BudgetEpoch` | wall-clock watchdog only; host-op (GPU) time is governed by the worker watchdog instead (architecture §10.2) |
| live step handles | 2^20 | `BudgetHandles` | generational arena (§3.3); `drop@1` frees slots eagerly |
| host-op calls per entry point | `max(2^22, 8 × MetaReport.op_calls[entry])`; ingest scales per-peer like fuel | `BudgetOps` | caps pathological op streams independently of fuel |
| sealed payload | envelope `update_mb_max` | `PayloadOverflow` | enforced at push (sender) and pre-stage (receiver) |
| tables / globals / module size | pooling-allocator caps; 32 MiB binary | validation error | §2.1 |

Host-side (not guest-visible) costs the worker accounts for per run — **VRAM**: params
(storage dtype) + fp32 masters while resident (§5.9) + fp32 grads + native persistents +
activations; **host RAM**: CPU-materialized masters + the round-base snapshot + offloaded
`local` persistents + staged payloads (≤ `count × update_mb_max`) + det-lane working set
(~2 dense fp32 tensors under §5.9's streaming discipline). All are computable from the
`MetaReport` (`host_ram_bytes_est` aggregates the RAM side) — which is exactly what `AssessRun`
compares against probed VRAM *and* RAM (architecture §5.1 "host-RAM planning", §6.5).

---

## 9. Versioning & capability negotiation

- **Namespace = major.** `tabi@1` is this document. A breaking change (signature, semantic,
  trap-condition, layout §6.6) ships namespace `tabi@2`; hosts may link both and modules import
  one.
- **Field = op version.** Additive evolution adds fields (`rope@2` beside `rope@1`). An op's
  documented semantics never change under its name@version.
- **Capability set = static import list.** Freeze-time derivation of the envelope's
  `capabilities` and every peer's assess-time re-derivation both use `imports(module)` — no
  execution, and no party needs to trust another's scan (architecture §6.1, §6.5; the
  registry stores, it does not parse — §2.1). The worker's advertised set is its compiled
  host table (architecture §10.2 `Probe`).
- `abi_minor@1` exists for SDK diagnostics only; admission never consults the guest.
- Freeze process: `tabi@1` is frozen at the P1 exit gate (architecture §17) — after that,
  additive-only. Conformance fixtures (§11) are the enforcement mechanism; a fixture, once
  published, never changes for a given `op@version`.

---

## 10. The guest SDK (`daemon-train-sdk`)

Crate in `crates/contracts/` (architecture §10.1): `wasm32-unknown-unknown` primary target,
**zero heavy deps** (serde + a CBOR codec only), `forbid(unsafe_code)` except the extern block.
Layout:

```
daemon-train-sdk/
  src/abi.rs         # the extern "C" import block for tabi@1 + da_* export glue (macro target)
  src/tensor.rs      # Tensor (native), DetTensor — shape/dtype tracked guest-side
  src/nn.rs          # Embedding, Linear, RmsNorm, RotarySelfAttention, SwiGlu, TransformerBlock
  src/optim.rs       # AdamW, NAdamW, Sgdm, Signum (fused-op wrappers) + pure-f64 LR schedules
  src/update.rs      # UpdateBuilder, UpdatesView (container ops)
  src/profiles/      # sparse_loco.rs, diloco.rs, demo.rs   (architecture §5.3 — normative math)
  src/experiment.rs  # Experiment trait + experiment! macro
  src/sim.rs         # feature "sim": native host simulator (see below)
```

### 10.1 Core types

```rust
pub struct Tensor { h: Handle, shape: Shape, dtype: Dtype }   // native lane
pub struct DetTensor { h: Handle, shape: Shape }              // det lane — separate type ⇒
                                                              //   lane errors are compile errors
pub struct Param(Tensor);        // + fn grad(&self) -> Tensor, fn round_base(&self) -> Tensor
pub struct Persistent(Tensor);   // ::local(..) / ::replicated(..) constructors (§5.1 classes)
pub struct DetPersistent(DetTensor);
pub struct Batch { .. }          // tokens(), size(), seq_len()
pub struct StepCtx { pub inner_step: u32, pub mb_index: u32,  // da_step's accumulation args;
                     pub mb_count: u32, pub step_seqs: u32 }  //   loss_scale() = size/step_seqs
pub struct Config(CborValue);    // serde-deserializable view of [experiment.config]
```

`Tensor`/`DetTensor` implement `Drop` over `drop@1` for step-scoped handles (stable handles
skip it), so intermediates free at scope exit — tch/Burn RAII ergonomics, and the reason
streaming ingest (§5.9) is the *natural* way to write a profile, not an optimization.

The SDK mirrors every shape/dtype rule from §5 guest-side and panics (→ `GuestPanic`) with a
readable message *before* the host would trap — same outcome, better DX. Operator overloads
(`&a + &b`, `x.matmul(&w)`) map 1:1 onto imports; nothing in the SDK computes tensor math
itself.

### 10.2 The `Experiment` trait and entry-point glue

```rust
pub trait Experiment: Sized {
    fn manifest(cfg: &Config) -> Manifest;   // cadence + round modes (§6.2); a profile
                                             //   provides it (SparseLoco::manifest reads h
                                             //   from its config section)
    fn build(cfg: &Config) -> Self;
    fn step(&mut self, batch: &Batch, ctx: &StepCtx);
    fn inner_update(&mut self, inner_step: u32);
    fn make_update(&mut self, round: u64) -> UpdateBuilder;
    fn ingest(&mut self, round: u64, updates: &UpdatesView);
}

daemon_train_sdk::experiment!(SmolLm);   // generates: da_alloc/da_free (global allocator),
                                         //   da_abi (SDK version), da_manifest (name/version
                                         //   from Cargo metadata + Experiment::manifest),
                                         //   da_defaults (from #[derive(ExperimentConfig)]
                                         //   Default impl), da_build/da_step/… trampolines
                                         //   holding the singleton in a guest static
```

The singleton static is legitimate under T3: it holds *handles and config*, both of which
`da_build` re-derives deterministically after any re-instantiation.

### 10.3 Profiles as libraries (architecture §5.3, unchanged)

```rust
pub struct SparseLoco { cfg: SparseLocoCfg, ef: Vec<Persistent> }       // EF buffers, native lane
impl SparseLoco {
    pub fn manifest(cfg: &SparseLocoCfg) -> Manifest;
    //   steps_per_round = cfg.h; round_modes = [barrier, pipelined]; interval = any
    pub fn make_update(&mut self, params: &[Param]) -> UpdateBuilder;
    //   Δ = p.round_base() − p  → acc = β·ef + Δ → topk_chunk → absmax_pack(2-bit) → push;
    //   ef ← acc − chunk_scatter(sent)                       (all native lane, local math)
    pub fn ingest(&mut self, params: &[Param], u: &UpdatesView);
    //   pass 1 (per update): det_absmax_unpack values → det_l2norm  (norms only; drop decodes)
    //   pass 2 (per update, streaming): decode → det_scale(clip/median) →
    //     det_chunk_scatter_add into one accumulator → intermediates drop at scope exit
    //   then: det_reset_param_to_base(p) → det_axpy_param(p, acc, −α/R)
}                             // (all det lane, canonical inputs, O(1)-tensor peak memory §5.9)
```

`diloco` (dense/int8 + outer Nesterov via a **replicated** `det_persistent` momentum, §5.1) and
`demo` (per-step DCT path; manifest: `steps_per_round = 1`, requires `pipelined` or a
seconds-scale coordinator — architecture §5.3.3) have the same shape. An experiment composes:
its `manifest`/`make_update`/`ingest` typically one-line delegate to a profile; ablations are
config fields; novel algorithms are new guest code over §5.8/§5.9 primitives.

### 10.4 The `sim` feature (native testing without GPU or wasm)

`daemon-train-sdk` with `feature = "sim"` swaps the extern block for an in-crate reference
implementation (ndarray, CPU, fp32; det ops shared with `daemon-train`'s real det kernels via a
common `det-core` micro-crate). Purpose: unit-test experiments and profiles with `cargo test`,
generate/verify conformance fixtures (§11), and property-test lane/phase legality — all in the
default workspace gate, no GPU lanes required. The sim is *semantics-reference, not
performance-reference*.

### 10.5 First-party presets (architecture §5.1)

`daemon-train` ships preset experiment modules built from the SDK (LLaMA-family decoder:
RMSNorm + SwiGLU + RoPE + GQA, wired to each profile), parameterized entirely by
`[experiment.config]`. The presets are the SDK's reference consumers and the P1 dogfood
(architecture §17); `llama-burn` remains the numerics golden reference (architecture §15.3).

### 10.6 Authoring workflow

```
cargo new my-exp --lib && cargo add daemon-train-sdk
# implement Experiment + experiment!(MyExp)
cargo build --target wasm32-unknown-unknown --release
daemon-cli swarm module check target/.../my_exp.wasm      # validate §2.1 + meta run + MetaReport
daemon-cli swarm module sign …                            # author signature (architecture §12)
daemon-cli swarm create --module my_exp.wasm …            # envelope authoring (architecture §6.1)
```

`module check` prints the derived capability set, VRAM/payload estimates, and the
`value_dependent` flag — the author sees exactly what admission will see.

---

## 11. Conformance & testing

- **Fixture suite** (in `daemon-train`, shared with the sim): for every `op@version`, golden
  input/output tensors. Native-lane ops carry a tolerance class per backend
  (`exact | ulp(n) | rel(1e-5)`); det-lane ops and the container/config codecs are byte-exact,
  no tolerance. Fixtures are append-only per §9.
- **Cross-lane replay test:** run a preset for N steps in sim and on each GPU lane; assert det
  digests equal (masters identical) while native losses agree within tolerance — the §7
  contract, mechanized.
- **ABI fuzzing:** arbitrary import-call streams against the host (proptest) must only ever
  trap with §3.6 codes — never UB, never a worker crash. Phase/lane/handle rules are the
  grammar; the corpus must reach every trap code at least once (taxonomy completeness).
- **Codec conformance (consensus-critical bytes):** the canonical-CBOR encoder (§6.1/§6.2 —
  the bytes that are hashed, signed, and fed to `da_build`/`da_manifest`) against the RFC 8949
  §4.2 test vectors + adversarial key orders/floats; DAUP container framing (§6.6) round-trip +
  truncation/corruption fixtures; `MetaReport` schema validation (§6.4).
- **Re-instantiation replay (T3):** drop the instance mid-round, re-instantiate, `da_build`,
  and assert every stable handle, registration list, and round-base view is identical — the
  crash/preemption/upgrade path (architecture §10.5), mechanized.
- **Budget scaling:** meta at `min_peers`, execute-mode ingest at `max_peers` synthetic
  updates ⇒ must complete without `BudgetFuel`/`BudgetOps` (the §8 per-peer scaling, pinned by
  test so it cannot regress to a fixed multiplier).
- **Determinism CI:** same module + fuel budget twice ⇒ identical trap-or-complete at identical
  fuel remaining (guards against accidental nondeterminism in host op implementations leaking
  into guest-observable behavior). Dropout RNG (open q. 3) gets pinned fixtures the moment an
  algorithm is chosen.
- Golden profile tests (vs paper reference implementations) stay as specified in architecture
  §15.3.

---

## 12. Rejected alternatives (interface-level)

| Alternative | Reason |
|---|---|
| **wasi-nn** | Inference-only by design (Phase 2; "concerned initially with inference, not training") and explicitly *"not designed to provide support for individual ML operations (a 'model builder' API)"* — the model-builder API is precisely what experiments need. Its graph-loader shape (pass ONNX/GGML bytes to a backend) is the rejected static-IR path (architecture §18). |
| **Component model / WIT** | Attractive typing, but: adds the canonical-ABI indirection to every µs-scale op call on a hot boundary; guest-toolchain maturity outside Rust is uneven; and our interface is deliberately narrow (handles + scalars + small spans), which core wasm expresses exactly. Revisit at `tabi@2` — the vocabulary (§5) translates to WIT mechanically if adopted. |
| **Guest-owned tensor memory** (tensors in linear memory) | Defeats zero-copy GPU residency, explodes the memory budget, and turns the sandbox boundary into a bandwidth bottleneck — violates T1. |
| **Reentrant/callback ABI** (host→guest→host) | Breaks per-call fuel accounting, step-scoped arenas, and preemption points (T2) for zero expressiveness gain — the host already drives the lifecycle. |
| **Error codes instead of traps** | Doubles every signature, invites unchecked-result bugs in guest code, and models unrecoverable misuse as recoverable — contradicts T4 and the worker's existing typed-error protocol. |
| **Separate ABIs for model vs optimizer vs comm** | One vocabulary with phase legality (§3.5) gives the same separation without three version axes (architecture §5.2's "no second composition language", applied to the interface itself). |
| **Guest-visible mode flag** | Any module that branches on mode invalidates meta/trace fidelity by construction; keeping guests mode-blind makes the three interpreters trustworthy (T5). |
| **Digest / randomness / clock imports** | Digests are host consensus machinery (architecture §5.6); free randomness and clocks destroy replay determinism (§7). Seeded forms (`param` init, `dropout`) cover the legitimate uses. |

---

## 13. Open questions (tracked here, non-blocking for P1)

1. **Batch schema growth** — document-boundary masks and explicit position ids as
   `batch_*@1` additions when a preset needs them; decide layout with the first consumer.
2. **Multi-loss / multi-tape** — `backward@1` differentiates one scalar per call; auxiliary
   losses currently sum guest-side first. Fine for v1; revisit if a preset needs separate
   grad scaling per loss term.
3. **Dropout RNG algorithm** — counter-based (Philox-style) vs host-native; must be pinned
   before any run relies on cross-peer-reproducible *local* replays (per-peer determinism
   already holds either way).
4. **`f16` scalar readback rounding** — `scalar@1` on `F16/BF16` tensors: round-trip via f64 is
   exact, but document the widening rule explicitly in the fixture suite.
5. **Fuel-cost stability across wasmtime upgrades** — fuel per instruction can change with
   compiler versions; budgets derive from meta runs on the *same* host version (§8), but the
   conformance "determinism CI" should pin expectations per wasmtime release.
6. **Het-dtype params in one state dict** (e.g. fp32 norms + bf16 matrices) — allowed today
   (per-param dtype); confirm checkpoint tooling (§9 of the architecture spec) round-trips
   mixed dtypes before P2.

---

## 14. Worked example (abridged; the P1 preset is the full version)

```rust
use daemon_train_sdk::{prelude::*, profiles::sparse_loco::SparseLoco};

#[derive(ExperimentConfig)]           // serde + da_defaults derivation
struct Cfg { d_model: u32, n_layers: u32, n_heads: u32, n_kv_heads: u32,
             seq_len: u32, vocab: u32, inner: AdamWCfg, comm: SparseLocoCfg }

struct SmolLm { cfg: Cfg, tok: Embedding, blocks: Vec<TransformerBlock>,
                norm: RmsNorm, opt: AdamW, comm: SparseLoco }

impl Experiment for SmolLm {
    fn manifest(cfg: &Config) -> Manifest {
        SparseLoco::manifest(&cfg.parse::<Cfg>().comm)    // steps_per_round = comm.h, …
    }
    fn build(cfg: &Config) -> Self {
        let cfg: Cfg = cfg.parse();                       // canonical CBOR → struct
        let tok = Embedding::new("tok", cfg.vocab, cfg.d_model);
        let blocks = (0..cfg.n_layers)
            .map(|i| TransformerBlock::llama(&format!("l{i}"), &cfg)).collect();
        let norm = RmsNorm::new("norm", cfg.d_model);
        let opt = AdamW::new(&cfg.inner);                 // registers m/v persistents (local)
        let comm = SparseLoco::new(&cfg.comm);            // registers EF buffers (local)
        Self { cfg, tok, blocks, norm, opt, comm }
    }
    fn step(&mut self, batch: &Batch, ctx: &StepCtx) {
        let ids = batch.tokens();
        let (x, targets) = (ids.slice(1, 0, batch.seq_len() - 1),
                            ids.slice(1, 1, batch.seq_len()));
        let mut h = self.tok.forward(&x);
        for b in &self.blocks { h = b.forward(&h); }      // rmsnorm/flash_attn/rope/swiglu ops
        let loss = self.norm.forward(&h).logits(&self.tok).cross_entropy(&targets);
        metric("loss", &loss);                            // report the unscaled mean
        loss.mul_s(ctx.loss_scale())                      // × size/step_seqs: accumulated grads
            .backward();                                  //   equal the exact step mean (§4);
                                                          //   fp32 grads accumulate host-side
    }
    fn inner_update(&mut self, s: u32) { self.opt.step_all(s); zero_grads(); }
    fn make_update(&mut self, _r: u64) -> UpdateBuilder { self.comm.make_update(params()) }
    fn ingest(&mut self, _r: u64, u: &UpdatesView) { self.comm.ingest(params(), u); }
}

daemon_train_sdk::experiment!(SmolLm);
```

Static import scan of the compiled module yields exactly the envelope capability set
(architecture §6.1):
`tabi@1::{param,persistent,param_round_base,embedding,rmsnorm,flash_attn,rope,matmul,
cross_entropy,backward,grad,zero_grads,adamw_step,topk_chunk,absmax_pack,chunk_scatter,
det_absmax_unpack,det_chunk_scatter_add,det_l2norm,det_scale,det_reset_param_to_base,
det_axpy_param,upd_new,upd_push_bytes,upd_push_tensor,upd_sections,upd_kind,upd_read_bytes,
upd_tensor,slice,sub,add,mul_s,assign,drop,scalar,metric,…}` — all `@1`.
