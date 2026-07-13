# Swarm P1 — lane M1 ledger (model/data lane, Wave 2)

Lane **M1** of the *Swarm P1 + Transport* program (program ledger:
[`swarm-p1-ledger.md`](swarm-p1-ledger.md)). M1 → M2 owns: the 160M preset in
`daemon-train-sdk/src/models.rs`, the `guests/*` binding, the pre-tokenized data path
(`daemon-swarm-run/src/data.rs`, **additive**), the `xtask` corpus subcommand, the safetensors
checkpoint converter, and the parity/golden harness in `tests/`. This ledger is written first
(`mirror(M1): ledger`); the "Exported seams" section is frozen at **Merge 2**.

## Base + branch

- **Repo:** `daemon-node` (standalone submodule checkout).
- **Worktree:** `/home/j/experiments/daemon-worktree/swarm-proto`, branch `swarm/m1`.
- **Base commit:** `bd2cb5b` (`mirror(merge-1): freeze Wave-1 interfaces`) = Merge 1 HEAD on
  `integrations/swarm-p1`.
- Never modifies: the main checkouts, the FROZEN files (root `Cargo.toml`, `deny.toml`,
  `flake.nix`), or other lanes' directories. G2 (`daemon-train/src/{backend,burn_backend,
  wasm_backend,meta}.rs` + worker `backend` module) is live in the same wave — M1 stays out of
  those files and coordinates any new host dispatch as a `.patch` section below (none needed — see
  "tabi@1: no new op").

## Frozen surfaces M1 respects (Merge-1 inventory)

- `TrainerBackend` (`checkpoint_save`/`checkpoint_load` are frozen — the safetensors converter lands
  **alongside** as an additive library, never as an edit to those methods).
- `tabi@1` = **66 imports** (`daemon_train_sdk::TABI_IMPORTS` + `daemon-train/tests/abi_surface.rs` +
  `phase.rs`). Additive growth is allowed only until Merge 3. **M1 adds no op** (see below).
- `Manifest`/`ShardDesc` format in `data.rs` (extended **additively** — new `#[serde(default)]`
  optional fields only; old manifests stay valid).
- The SDK `Experiment` trait + `experiment!` macro (guest stays a one-line `experiment!(TinyLlama)`).
- `TinyLlamaCfg` — the `n_kv_heads == n_heads` assert at `models.rs` `build()` is **kept** (GQA
  deferred, see below).

## tabi@1: no new op needed (coordination artifact: NONE)

The 160M LLaMA decoder is expressible with the frozen 66-op vocabulary:

- **Causal masking is already inside `flash_attn@1`.** ABI §5.6: `flash_attn@1 (q,k,v, causal: u,
  scale: f)`; the host + sim honor the `causal` flag (sim `flash_attn` masks `j > i`,
  `sim.rs` ~L946). The preset already calls `q.flash_attn(&k,&v, true, scale)` (`models.rs`
  `step`). No separate mask op.
- **GQA is an ABI capability, not an op** ("GQA via head-count ratio k,v vs q", ABI §5.6). At the
  P1 gate `n_kv_heads == n_heads` suffices, so the `models.rs` assert is **kept** and GQA is
  recorded as deferred. No `tabi` growth for GQA-repeat.

⇒ No host-side `Linker`/`phase.rs` patch is required from the Merge-2 owner. The additive `tabi@1`
window stays untouched by M1.

## Decisions (the 160M preset)

- **Architecture:** `d_model 768, n_layers 12, n_heads 12, n_kv_heads 12, head_dim 64,
  seq_len 1024, ffn_mult 4` (SwiGLU hidden = 3072), `rope_theta 10000`, `rmsnorm_eps 1e-5`, tied
  input/output embedding. Inner AdamW: `lr 4e-4, betas [0.9,0.95], eps 1e-8, wd 0.1` (TDD §5).
- **Tokenizer / vocab:** GPT-2 BPE, **vocab 50257** (TinyStories' native GPT-Neo vocabulary; it is
  a GPT-2 BPE). `50257 < 65536` ⇒ **`u16` shards** (`TokenWidth::U16`). A Llama tokenizer (32000)
  was the alternative; GPT-2 is the pragmatic TinyStories match and keeps shards `u16`.
- **Exact parameter count:** see "Numbers" below (reported exact; ~152M, within bounds of the
  "160M" spec row).
- **Comm profile:** `sparse_loco`, `h = 30` (TDD §5 golden H). `chunk` is capped by the params'
  2-adic valuation (`gcd` of all param element counts = 768 = 2^8·3, so the largest power-of-two
  chunk dividing every param without guest-side padding is **256**); density kept at the golden
  1/64 via `topk = 4`, `bits = 2`, `ef_decay 0.95`, `outer_alpha 1.0`, `clip true`. The real-model
  golden uses `chunk 4096` (which does not divide the embedding: `50257·768` has 2-adic valuation
  8), so `256`/`4` preserves the 1/64 ratio without padding. M2 tunes the numerics; this is a
  preset config, not the numeric gate.
- **GQA:** deferred (assert kept). Recorded so the additive window is not spent on it.

## Planned slices (commits)

1. `mirror(M1): ledger` — this file.
2. `feat(train-sdk): 160M llama preset + canonical param layout (green)` — `models.rs`
   `TinyLlamaCfg::llama_160m()`, `param_count`, `canonical_param_layout`, unit tests.
3. `feat(swarm-run): additive manifest tokenizer/dataset metadata (green)` — `data.rs` optional
   provenance fields + back-compat test.
4. `feat(train-safetensors): state-dict <-> safetensors converter (green)` — new crate.
5. `feat(xtask): tokenize-corpus subcommand (green)` — HF-pull + tokenize + shard writer.
6. `feat(swarm-run): vendor TinyStories fixture + RUN-3 on real manifest (green)` — fixture + tests.
7. `test(train-sdk): HOST-11 numeric goldens + 160M meta reconcile (green)` — golden suite.

## Final — HEAD `ec8af90` (base `bd2cb5b`)

### Commit list (oldest → newest)

| Commit | Subject |
|---|---|
| `845bbe0` | `mirror(M1): ledger` |
| `83504c1` | `feat(train-sdk): 160M llama preset + canonical param layout (green)` |
| `2d10bc8` | `feat(swarm-run): additive manifest tokenizer/dataset metadata (green)` |
| `1511478` | `feat(train-safetensors): state-dict <-> safetensors converter (green)` |
| `d92a7fc` | `feat(xtask): tokenize-corpus subcommand (green)` |
| `3b5b9b6` | `feat(swarm-run): vendor TinyStories fixture + Corpus::from_parts + RUN-3 (green)` |
| `17df76a` | `test(train-sdk): HOST-11 numeric goldens + 160M preset smoke/meta (green)` |
| `ec8af90` | `feat(train-sdk): 160m + tiny example config (toml + canonical cbor) (green)` |

### Exported seams (FREEZE at Merge 2)

1. **160M config schema + example envelope fragment.** `TinyLlamaCfg::llama_160m()` (models.rs) —
   `d_model 768, n_layers 12, n_heads 12, n_kv_heads 12, head_dim 64, vocab 50257, seq_len 1024,
   ffn_mult 4, rope_theta 1e4, rmsnorm_eps 1e-5`, inner AdamW `lr 4e-4/β[0.9,0.95]/eps 1e-8/wd 0.1`,
   profile `sparse_loco { h 30, chunk 256, topk 4, bits 2, ef_decay 0.95, outer_alpha 1.0, clip }`.
   Plus `TinyLlamaCfg::canonical_param_layout() -> Vec<(String, Vec<u32>)>` and `param_count() -> u64`
   (the single source of truth for registration order, shared by the safetensors converter + M2's
   burn reference). Example envelope fragments + canonical byte form:
   `crates/contracts/daemon-train-sdk/presets/{llama-160m,tiny-llama}.{toml,cbor}` (the `.cbor` is
   what `da_build` receives; `preset_cbor_fixtures_parse` asserts it round-trips to the constructor).
2. **Manifest metadata extensions.** `daemon_swarm_run::data::Manifest` grew four optional
   `#[serde(default, skip_serializing_if=Option::is_none)]` fields: `tokenizer`,
   `tokenizer_revision`, `dataset`, `dataset_revision`. **Back-compat is a test invariant**
   (`manifest_provenance_is_additive_and_back_compatible`): a pre-Wave-2 manifest parses, and a
   provenance-less manifest serializes byte-identically to the old shape. Also added (additive):
   `Corpus::from_parts(Manifest, Vec<Vec<u8>>)` — the real corpus constructor (blake3-verifies each
   shard, §8) that the runtime/B3 use for fetched-or-vendored shards (vs `Corpus::synthetic`).
3. **`xtask tokenize-corpus` CLI surface.** Args:
   `--dataset <hf-id>? --dataset-file <name>? --revision <sha|tag=main> --text <path>?
    --tokenizer <hf-id|path> --tokenizer-revision <sha|tag>? --out-dir <dir>
    --shard-tokens <u64=1048576> --seq-len <u32=1024> --token-width <u16|u32=u16> --max-tokens <u64>?`.
   Egress via **`hf-hub`** (revision-pinned; no raw `reqwest::Client` — the clippy `disallowed_types`
   egress ban is workspace-global and respected, so no `#[allow]` was needed); `tokenizers` crate for
   BPE. Writes fixed-width LE shards + `manifest.json` (with provenance). `--text`/`--tokenizer <path>`
   give a fully offline path.
4. **safetensors converter surface.** New crate **`daemon-train-safetensors`**
   (`crates/coprocessor/daemon-train-safetensors`). API: `StateDict { tensors: Vec<(String, Vec<usize>,
   Vec<f32>)> }` with `push` / `from_named` / `names` / `to_safetensors() -> Vec<u8>` /
   `from_safetensors(&[u8]) -> StateDict`, and `blake3_hex(&[u8]) -> String`. fp32 only (P1 masters +
   fp32-exact replicated persistents, §9). Canonical **registration order is preserved** via a single
   `__metadata__["order"]` key (safetensors sorts tensors by `(dtype,name)` on write); a single
   metadata key keeps the bytes **deterministic** (spec §9 needs byte-identical checkpointer uploads
   for the hash-match gate — `serialization_is_deterministic` guards it). Sits **alongside** the
   frozen `TrainerBackend::checkpoint_save/load` (never edits them). A caller assembles a `StateDict`
   from a live instance via `Instance::params()` (names+shapes) + `Instance::param_master(name)`
   (fp32 master). **Integration-owner action at Merge 2:** to let M2/B3 consume it via
   `{ workspace = true }`, add a `[workspace.dependencies]` path entry
   `daemon-train-safetensors = { path = "crates/coprocessor/daemon-train-safetensors" }` (a root
   `Cargo.toml` edit — not a lane action). In-lane the crate builds standalone (picked up by the
   `crates/*/*` glob) and is fully self-tested.

### Numbers (reported exact)

- **160M param count: `151,862,784`** (≈152M; the "160M" spec row is a label — within bounds).
  Breakdown: tok `50257·768 = 38,597,376`; per layer `9,438,720` × 12 = `113,264,640`; final norm
  `768`. Tied embedding counted once.
- **Meta-mode footprint (fp32, from `Instance::meta` — `preset_160m_reduced_meta_report` proves the
  `4·params` arithmetic; the full pass is the `#[ignore]`d `preset_160m_full_smoke_and_reconcile`):**
  `param_bytes = master_bytes = grad_bytes = 4·N = 607,451,136 B (~0.566 GiB)`; fp32 Adam m+v
  `= 8·N = ~1.13 GiB`; steady-state VRAM (params+master+grad+adam, excl. activations)
  `≈ 2.26 GiB`; coarse `WasmBackend` VRAM estimate (`master·3`) `≈ 1.70 GiB`; host RAM
  (`master·2`, masters + round-base) `≈ 1.13 GiB`.
- **Spec §5.1 reconciliation (within bounds, with a recorded correction):** the spec table assumes
  **bf16 weights** (160M row: weights 0.3 GB, grads 0.6 GB, Adam+master 1.9 GB, ~total+act ~4.5 GB,
  host ~2 GB, fits 8 GB card). The P1 preset stores **fp32** (det-lane exactness), so per-tensor
  weight/master bytes are 2× the bf16 row (0.57 GiB vs 0.3 GB); grads match (~0.6 GB); the fp32
  steady state (~2.3 GiB) + activations (seq **1024**, not the table's 2048) still fits an 8 GB card
  — the spec's operative conclusion holds. **Spec-amendment candidate for Merge 2:** either annotate
  the §5.1 table as bf16-weights-specific (and add an fp32-storage note), or the preset should adopt
  bf16 storage in a later wave (a G-lane numerics change, not M1's — would touch backend dtype +
  goldens). The reconciliation is asserted analytically in `models.rs`
  (`llama_160m_footprint_reconciles_with_spec_table`, always-on) and against the real meta report in
  the `#[ignore]`d full test.
- **Guest wasm: `tiny_llama.wasm = 143,697 bytes (~140 KiB)`**, unchanged shape (still a one-line
  `experiment!(TinyLlama)`; the 160M preset is pure config). **Guest memory at 160M config is
  comfortable** (T1: params live host-side): the guest holds only handles + config — ~550 stable
  registrations (110 params + 110+110 AdamW m/v + 110 sparse_loco EF persistents), each a small
  handle/shape record, plus the `446`-byte `llama-160m.cbor` config; well under the 64 MiB linear-
  memory cap. The 160M module loads + `da_build`s + steps + `make_update`s on the CPU host in
  `preset_160m_reduced_*` (fast) and the `#[ignore]`d full test.
- **Fixture provenance (REAL corpus — egress succeeded):** `xtask tokenize-corpus` was run once
  against real TinyStories. Vendored at
  `crates/swarm/daemon-swarm-run/tests/fixtures/tinystories/` = 4 × `u16` shards (262,144 tokens
  each = **1,048,576 tokens, ~2.1 MB total**, seq_len 1024 ⇒ 1024 sequences) + `manifest.json`.
  Dataset **`roneneldan/TinyStories`** file `TinyStories-valid.txt` @
  `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`; tokenizer **`gpt2`** (GPT-2 BPE, 50257) @
  `607a30d783dfa663caf39e06633721c8d4cfcd7e`. Exact regeneration command is in the docstring of
  `tests/tinystories_fixture.rs`. (Deterministic: `main` and the pinned SHA produced identical
  shard blake3s.)

### tabi@1 / coordination

**No new op was added.** Causal masking is inside `flash_attn@1` (`causal` flag) and GQA is an ABI
head-count-ratio capability, not an op — the 160M preset is fully expressible in the frozen 66-op
vocabulary, so `TABI_IMPORTS` / `phase.rs` / the host `Linker` / `abi_surface.rs` are untouched and
**no host-side `.patch` for the Merge-2 owner is required**. GQA is deferred (the
`n_kv_heads == n_heads` assert is kept). The additive `tabi@1` window is left entirely unspent by M1.

### Test counts (all green)

- `daemon-train-sdk --features sim`: lib 9, host11_golden 4 (RMSNorm/RoPE/SwiGLU/attention vs an
  independent from-definition oracle + hand anchors), accumulation 1, profiles 6, tiny_llama 2,
  toy_experiment 3.
- `daemon-swarm-run`: lib 33 (incl. manifest back-compat), tinystories_fixture 5 (RUN-3
  `manifest_batchid_maps` + `interval_slices_into_h_steps` on the real fixture, provenance, real-shard
  Corpus decode, tamper rejection).
- `daemon-train-safetensors`: 6 (bit-exact + order-preserving round-trip, determinism, special
  floats, shape-mismatch + garbage rejection, real canonical-layout round-trip).
- `daemon-train` (workspace, CpuBackend): `preset_160m` 2 always-on + 1 `#[ignore]` (full 160M).
- Workspace + clippy(`-D warnings`) + fmt + `typos docs/specs` + `build-guests`: green **except** the
  documented pre-existing `daemon-conformance` detached-delegation flake (a *different* trio member
  fails per full run; each passes in isolation — verified `detached_fanout_materializes_distinct_children`
  green alone). M1 never touches `daemon-conformance`.

### Deviations / watch items for Merge-2 (and M2)

- **Repo-root config files added (additive; not lane-frozen, but the integration owner should note):**
  `.gitleaks.toml` (allowlists the two **public** HF commit SHAs + the swarm test-fixtures path —
  the 40-hex SHAs otherwise trip gitleaks' `generic-api-key` near the `tokenizer_revision` key) and a
  `[type.swarm-corpus-shard] check-file=false` block in `typos.toml` for `*.bin` (the vendored
  binary token shards spell dictionary-word fragments; `extend-exclude` is only honored on directory
  walks, not the hook's explicit-path invocation — hence the type override). Both are minimal and
  scoped.
- **`safetensors` version:** the new crate depends on `safetensors = "0.4"` (resolves 0.4.5, already
  in the lock; `Cargo.lock` gained only the new crate's own node — no new third-party crate, so
  `cargo deny` stays green). If the integration owner prefers a single safetensors version tree, note
  the lock also carries 0.3.3/0.7.0 transitively (`bans.multiple-versions = "warn"`).
- **160M numeric parity is M2's job**, not M1's: the `sparse_loco` `chunk 256 / topk 4` choice (vs the
  golden real-model `chunk 4096`) is a *preset config* dictated by the params' 2-adic valuation
  (no guest padding), not a tuned optimum — M2 should treat it as a knob when it builds the llama-burn
  reference and measures loss/throughput. `canonical_param_layout()` is the safetensors↔burn name map
  M2 needs.
- **Full-160M CPU execute pass is minutes + GBs** (spec Risk 3 — meta is a real execute pass): the
  full test is `#[ignore]`d; run it in the wgpu lane / a beefy box, not per-PR CI. The reduced
  variant + analytical reconciliation are the always-on guards.
- **safetensors consumption edge** for M2/B3 needs the `[workspace.dependencies]` path entry at Merge
  2 (see seam 4).
