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

## Exported seams (FROZEN at Merge 2 — draft until then)

_(Filled in as slices land; see "Final" section at the bottom.)_

1. **160M config schema + example envelope fragment** — `TinyLlamaCfg::llama_160m()`;
   `presets/llama-160m.{toml,cbor}`.
2. **Manifest metadata extensions** — the additive `data.rs` fields.
3. **`xtask tokenize-corpus` CLI surface** — arg set.
4. **safetensors converter surface** — the `daemon-train-safetensors` crate API.

## Numbers (reported exact — updated as slices land)

_(see Final section.)_

## Deviations / watch items for Merge-2

_(see Final section.)_
