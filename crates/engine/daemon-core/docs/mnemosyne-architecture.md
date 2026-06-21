# Mnemosyne: Complete Architecture Map

A full visual map of the [`Mnemosyne/`](Mnemosyne/) codebase — a zero-dependency,
SQLite-backed AI memory layer implementing **BEAM** (Bilevel Episodic-Associative Memory).
~26K LOC of Python in `mnemosyne/core/` plus interfaces (MCP, CLI), integrations, and a sync
server. Produced in the same style as [hermes-agent-architecture.md](hermes-agent-architecture.md).

> Mnemosyne is a *universal, Hermes-first memory layer* usable by any agent (Claude Code,
> Cursor, Codex, OpenWebUI, OpenClaw, or custom). One `pip install`, one SQLite database, no
> external services. Default data root: `~/.hermes/mnemosyne/data/`.

---

## 1. System overview (10,000 ft)

Mnemosyne is a **library + interfaces** (not an agent). Any host reaches one engine —
`BeamMemory` — through the thin `Mnemosyne` facade, and everything persists into a single
SQLite file with `sqlite-vec` + FTS5.

```mermaid
graph TD
  subgraph hosts [Hosts / Surfaces]
    MCP[MCP server<br/>24 tools - stdio/SSE]
    CLI[cli.py<br/>store/recall/sleep/...]
    SDK[Python SDK<br/>import mnemosyne]
    INT[Integrations<br/>OpenWebUI / OpenClaw / Obsidian / VSCode]
    HERMES[hermes_memory_provider<br/>Hermes plugin]
  end

  subgraph api [API layer]
    FACADE[Mnemosyne<br/>core/memory.py]
    ENGINE[BeamMemory<br/>core/beam.py 8.3K LOC]
  end

  subgraph tiers [BEAM tiers]
    WM[working_memory<br/>hot context]
    EM[episodic_memory<br/>long-term]
    SP[scratchpad<br/>agent workspace]
  end

  subgraph index [Index layer - all in SQLite]
    VEC[sqlite-vec<br/>vec_working / vec_episodes]
    FTS[FTS5<br/>fts_working / fts_episodes]
    MIB[MIB binary vectors<br/>48-byte BLOB]
  end

  subgraph knowledge [Knowledge layer]
    TRIP[TripleStore]
    ANN[AnnotationStore]
    CAN[CanonicalStore]
    GRAPH[EpisodicGraph]
    VER[VeracityConsolidator]
  end

  hosts --> FACADE --> ENGINE
  ENGINE --> tiers
  tiers --> index
  ENGINE --> knowledge
  tiers --> DB[(single SQLite .db<br/>WAL)]
  index --> DB
  knowledge --> DB
```

---

## 2. Public API & object model

```mermaid
graph LR
  INIT[mnemosyne/__init__.py<br/>lazy exports]
  INIT --> FNS[remember / recall / get_context /<br/>forget / update / get / get_stats / sleep]
  FNS --> FACADE[Mnemosyne<br/>core/memory.py 987 LOC]
  FACADE --> ENGINE[BeamMemory<br/>core/beam.py]
  FACADE -.legacy dual-write.-> LEGACY[memories table]
  BANKS[banks.py<br/>per-bank .db isolation] --> FACADE
```

- **`Mnemosyne`** ([core/memory.py](Mnemosyne/mnemosyne/core/memory.py)) is a facade over
  `BeamMemory`, plus legacy dual-write to a flat `memories` table.
- **`BeamMemory`** ([core/beam.py](Mnemosyne/mnemosyne/core/beam.py), 8,326 lines ~ 32% of
  core) holds all BEAM logic and wires the knowledge subgraphs at init.
- **Banks** ([core/banks.py](Mnemosyne/mnemosyne/core/banks.py)) give each named memory bank
  its own directory + `mnemosyne.db`.
- `core/orchestrator.py` is a stub placeholder (not wired into production recall).

Main operations: `remember`, `recall`, `get_context` (prompt injection), `forget`/`update`/
`invalidate`, `sleep` (consolidation), `scratchpad_*`, `reindex_vectors`, `export`/`import`.

---

## 3. BEAM memory tiers

> **BEAM = Bilevel Episodic-Associative Memory.** Three SQLite tables: `working_memory` (hot,
> auto-injected), `episodic_memory` (long-term, vector + FTS5), `scratchpad` (ephemeral).

```mermaid
flowchart TB
  subgraph write [Write path: remember]
    R[remember] --> WM[working_memory]
    R --> EMB[embed -> memory_embeddings + vec_working]
    R --> ENR[enrichment: typed_memory, entities,<br/>extraction, graph, veracity]
  end

  subgraph consolidate [sleep - consolidation]
    WM -->|TTL 168h, batch| SUM[LLM or AAAK summary]
    SUM --> EM[episodic_memory]
    WM -->|mark consolidated_at<br/>originals retained| WM
  end

  subgraph read [Read path]
    Q[recall] --> WM
    Q --> EM
    WM --> CTX[get_context<br/>auto-inject before LLM]
  end

  SP[scratchpad<br/>session-scoped] -.-> SP
```

- **Working memory** — hot/recent; searched by FTS5 + sqlite-vec with recency fallback;
  feeds `get_context()` prompt injection.
- **Episodic memory** — long-term; sleep summaries + durable facts; carries the 48-byte MIB
  `binary_vector` and age-based tier degradation (T1 fresh, T2 >30d, T3 >180d).
- **Scratchpad** — ephemeral agent reasoning, no hybrid recall.
- **Sleep** is additive: consolidated WM rows are *marked*, not deleted; `pinned` rows skip
  sleep.

---

## 4. Storage layer (SQLite)

Single-file DB per bank, thread-local connection, `WAL`, `busy_timeout=5000`,
`sqlite_vec.load()` for vector virtual tables. Schema is created/evolved inline in
`init_beam()` via `CREATE TABLE IF NOT EXISTS` + `_add_column_if_missing`.

```mermaid
graph TD
  subgraph relational [Relational tables]
    WMt[working_memory]
    EMt[episodic_memory]
    SPt[scratchpad]
    EV[memory_events - sync log]
    MEMB[memory_embeddings - float32 fallback]
    VAL[memory_validations]
    FCT[facts + consolidation_log]
    MEMORIA[memoria_facts / timelines /<br/>instructions / preferences / kg]
  end
  subgraph virtual [Virtual / index tables]
    VW[vec_working]
    VE[vec_episodes]
    FW[fts_working]
    FE[fts_episodes]
    FF[fts_facts]
  end
  subgraph subgraphs [Knowledge schemas]
    TR[triples]
    ANt[annotations]
    CANt[canonical_facts]
    GI[gists / graph_edges]
    CONF[consolidated_facts / conflicts]
    SH[harmonic_beliefs / memory_resonance_log]
    QC[query_cache - separate db]
  end
  EMt -. FTS triggers .-> FE
  WMt -. FTS triggers .-> FW
  EMt -. binary_vector BLOB .-> MIBNOTE[MIB 48 bytes]
```

Rich columns on memory rows include lifecycle (`valid_until`, `superseded_by`, `scope`,
`consolidated_at`, `pinned`), identity (`author_id`, `channel_id`, `validator`), trust
(`veracity`, `trust_tier`, `memory_type`, `recall_count`), and temporal (`event_date`,
`temporal_tags`).

### Hybrid search flow

```mermaid
flowchart LR
  Q[query text] --> EQ[embed_query]
  Q --> FTS[FTS5 MATCH<br/>BM25 rank]
  EQ --> VEC[sqlite-vec KNN]
  EQ --> FALL[memory_embeddings<br/>numpy cosine fallback]
  FTS --> CAND[candidate ids]
  VEC --> CAND
  FALL --> CAND
  CAND --> FILT[filter: valid_until, superseded_by,<br/>session/scope, author/channel]
  FILT --> SCORE[hybrid score + recency decay + bonuses]
  SCORE --> OUT[ranked results]
```

---

## 5. Vectors & embeddings

```mermaid
graph LR
  TXT[text] --> EMB[embeddings.py]
  EMB -->|local ONNX| FE[fastembed<br/>BAAI/bge-small-en-v1.5 384-dim]
  EMB -->|MNEMOSYNE_EMBEDDINGS_VIA_API| API[OpenAI-compatible endpoint]
  EMB -->|MNEMOSYNE_NO_EMBEDDINGS| KW[keyword-only mode]
  FE --> MIB[binary_vectors.py<br/>MIB sign binarization]
  MIB --> PACK[np.packbits<br/>384 float32 -> 48 bytes]
  PACK --> BLOB[episodic_memory.binary_vector]
  BLOB --> HAM[Hamming distance<br/>XOR + popcount]
```

- **Default embeddings:** `BAAI/bge-small-en-v1.5` (384-dim) via local fastembed; alternatives
  via OpenAI-compatible API; can be disabled for keyword-only recall.
- **MIB (Maximally Informative Binarization)** ([core/binary_vectors.py](Mnemosyne/mnemosyne/core/binary_vectors.py)):
  sign-binarize each dimension (bit = 1 if > 0), `packbits` to 48 bytes — 8x smaller than
  float32. Recall scores a small `binary_bonus` from Hamming distance.

---

## 6. Retrieval & ranking pipelines

Three selectable pipelines share the BEAM tiers:

```mermaid
graph TD
  Q[recall query] --> MODE{recall mode}
  MODE -->|default| LIN[Linear hybrid<br/>50% vec + 30% FTS + 20% importance]
  MODE -->|MNEMOSYNE_ENHANCED_RECALL=1| ENH[Enhanced stack]
  MODE -->|MNEMOSYNE_POLYPHONIC_RECALL=1| POLY[Polyphonic RRF]

  ENH --> EI[query_intent classify]
  EI --> SYN[synonym expansion]
  SYN --> QCa[query_cache check]
  QCa --> base[base hybrid recall]
  base --> WEI[Weibull rescore]
  WEI --> MMRr[MMR diversity rerank]
  MMRr --> EXP[associative graph expansion]

  POLY --> V1[vector voice 0.35]
  POLY --> V2[graph voice 0.25]
  POLY --> V3[fact voice 0.25]
  POLY --> V4[temporal voice 0.15]
  V1 --> RRF[Reciprocal Rank Fusion k=60]
  V2 --> RRF
  V3 --> RRF
  V4 --> RRF
  RRF --> DIV[diversity rerank + token budget]
```

- **Default hybrid score** (episodic): `sim*0.5 + fts*0.3 + importance*0.2`, then
  `* (0.7 + 0.3*decay)` recency, plus graph/fact/binary bonuses, gated by a lexical relevance
  floor and multiplied by a veracity weight (stated 1.0 ... unknown 0.8).
- **MMR** ([core/mmr.py](Mnemosyne/mnemosyne/core/mmr.py)): `λ*relevance - (1-λ)*max_jaccard`.
- **Polyphonic** ([core/polyphonic_recall.py](Mnemosyne/mnemosyne/core/polyphonic_recall.py)):
  4 weighted voices fused by RRF.
- **SHMR** ([core/shmr.py](Mnemosyne/mnemosyne/core/shmr.py)): background "Self-Harmonizing
  Memory Reasoning" — clusters and converges beliefs; separate from main recall.
- Supporting: `query_intent.py` (regex intent -> weight bias), `query_cache.py` (5-tier
  semantic cache), `synonyms.py`, `recall_diagnostics.py` (per-path counters).

---

## 7. Knowledge layer (graph, triples, veracity)

```mermaid
flowchart TB
  TXT[raw text / conversation] --> REG[regex: episodic_graph + MEMORIA]
  TXT --> LLMx[LLM: extraction.py / extraction client]
  TXT --> ENT[entities.py regex<br/>mentions, fuzzy match]

  REG --> FACTS[facts table]
  REG --> GIST[gists]
  LLMx --> ANNf[annotations kind=fact]
  LLMx --> MEMt[memoria_* tables]
  ENT --> MEN[annotations kind=mentions]

  FACTS --> VC[VeracityConsolidator<br/>SHA-256 fact_id, Bayesian confidence]
  VC --> CF[consolidated_facts]
  VC --> CONFr[conflicts]

  TADD[TripleStore.add] --> TR[triples<br/>valid_from/valid_until chains]
  CADD[CanonicalStore.remember] --> CANf[canonical_facts<br/>owner-scoped version chains]
  TEMP[temporal_parser] --> ED[event_date on rows]
```

- **TripleStore** ([triples.py](Mnemosyne/mnemosyne/core/triples.py)): single-current-truth
  temporal facts; a new `(subject, predicate, object)` closes prior open triples; `query(as_of=)`
  for historical truth.
- **AnnotationStore** ([annotations.py](Mnemosyne/mnemosyne/core/annotations.py)): append-only
  multi-valued per-memory tags (post-E6 split migration).
- **CanonicalStore** ([canonical.py](Mnemosyne/mnemosyne/core/canonical.py)): owner-scoped
  identity cards with version history.
- **EpisodicGraph** ([episodic_graph.py](Mnemosyne/mnemosyne/core/episodic_graph.py)): gists +
  SPO facts + `graph_edges`; proactive linking of co-occurring entities.
- **VeracityConsolidator** ([veracity_consolidation.py](Mnemosyne/mnemosyne/core/veracity_consolidation.py)):
  compounds mention counts into confidence, detects `(S,P)` contradictions.

---

## 8. Extraction & LLM

```mermaid
graph TD
  R[remember] --> MEMORIAx[MEMORIA regex extraction<br/>always-on, zero LLM]
  R -->|extract=True| LLMpath[LLM structured extraction]
  LLMpath --> CHAIN
  subgraph CHAIN [LLM fallback chain]
    H0[0. host-provided backend<br/>Hermes integration]
    H1[1. remote OpenAI-compatible API]
    H2[2. local GGUF<br/>MiniCPM5-1B Q4_K_M ~656MB]
    H3[3. skip - graceful degradation]
    H0 --> H1 --> H2 --> H3
  end
  CHAIN --> OUT[facts / annotations / memoria]
  SLEEP[sleep summarization] --> CHAIN
  SLEEP -.on failure.-> AAAK[AAAK lossless shorthand]
```

- Extraction backends are pluggable ([llm_backends.py](Mnemosyne/mnemosyne/core/llm_backends.py));
  hosts register via `set_host_llm_backend()`.
- Local path ([local_llm.py](Mnemosyne/mnemosyne/core/local_llm.py)) uses `llama-cpp-python`
  (fallback `ctransformers`); cloud path defaults to `google/gemini-2.5-flash` via OpenRouter.

---

## 9. Memory dynamics

```mermaid
graph LR
  ING[ingest] --> TYPE[typed_memory<br/>13 regex types]
  ING --> IMP[importance 0..1<br/>= 20% of score]
  REC[recall] --> DECAY[recency decay<br/>exp -age/halflife 168h]
  REC --> WEIB[Weibull per-type decay]
  REC --> BUMP[bump recall_count / last_recalled]
  AGE[age] --> TIER[EM tier degradation<br/>T1 / T2 30d / T3 180d]
  SLEEPp[sleep] --> CONS[consolidate WM -> EM]
  PIN[pinned=1] -.skip.-> SLEEPp
  PAT[patterns.py<br/>MemoryCompressor + PatternDetector]
```

Retention is usage-driven (recall bumps), age-tiered (degradation), type-aware (Weibull), and
trust-weighted (veracity multipliers).

---

## 10. Supporting subsystems

```mermaid
graph TD
  subgraph repl [Replication / streaming]
    SYNC[sync.py - SyncEngine<br/>event-log replication + AES]
    SS[sync_server.py - HTTP]
    STREAM[streaming.py - MemoryStream / DeltaSync]
  end
  subgraph ops [Operational]
    PLUG[plugins.py - PluginManager<br/>logging/metrics/filter hooks]
    BANKS2[banks.py - bank isolation]
    SAN[content_sanitizer.py]
    TOK[token_counter.py]
    COST[cost_log.py]
  end
  subgraph importers [importers/]
    IMP2[mem0 / hindsight / cognee /<br/>letta / zep / honcho / supermemory]
  end
```

---

## 11. Interfaces

```mermaid
graph TD
  subgraph mcp [MCP - 24 tools]
    SRV[mcp_server.py<br/>stdio + SSE]
    TOOLS[mcp_tools.py<br/>remember/recall/sleep/triple/canonical/<br/>scratchpad/graph/export/diagnose...]
  end
  subgraph cli [CLI]
    C[cli.py<br/>store/recall/context/stats/sleep/<br/>scratchpad/export/import/mcp/diagnose/banks]
  end
  subgraph integ [Integrations]
    OW[OpenWebUI tool + auto-save]
    OC[OpenClaw bridge]
    OB[Obsidian plugin - TS]
    VS[VSCode extension - TS]
    HMP[hermes_memory_provider<br/>Hermes plugin: llm adapter, sync, audit]
  end
  subgraph ops2 [Resilience]
    DR[dr/recovery.py<br/>gzip backups + restore]
    MIG[migrations/e6_triplestore_split.py]
  end
  SRV --> TOOLS --> ENGINE2[BeamMemory]
  C --> ENGINE2
  integ --> ENGINE2
```

---

## 12. Directory reference

- **`mnemosyne/__init__.py`** — lazy public API (`remember`, `recall`, `get_context`, ...).
- **`mnemosyne/core/`** (~26K LOC, ~35 modules) — the engine:
  - `beam.py` (8.3K) engine, `memory.py` facade, `banks.py` isolation.
  - Index/vectors: `embeddings.py`, `binary_vectors.py`.
  - Recall: `mmr.py`, `shmr.py`, `polyphonic_recall.py`, `query_intent.py`, `query_cache.py`,
    `synonyms.py`, `recall_diagnostics.py`.
  - Knowledge: `triples.py`, `annotations.py`, `canonical.py`, `episodic_graph.py`,
    `entities.py`, `temporal_parser.py`, `veracity_consolidation.py`, `llm_conflict_detector.py`.
  - Extraction/LLM: `extraction.py`, `llm_backends.py`, `local_llm.py`, `aaak.py`.
  - Dynamics: `weibull.py`, `typed_memory.py`, `patterns.py`.
  - Support: `sync.py`, `sync_server.py`, `streaming.py`, `plugins.py`, `content_sanitizer.py`,
    `token_counter.py`, `cost_log.py`, `chat_normalize.py`.
- **`mnemosyne/mcp_server.py` / `mcp_tools.py`** — MCP server (24 tools).
- **`mnemosyne/cli.py`** — command-line interface.
- **`mnemosyne/integrations/`** — OpenWebUI, OpenClaw, memory browser.
- **`mnemosyne/extraction/`** — cloud extraction client + prompts + diagnostics.
- **`mnemosyne/dr/`** — disaster-recovery backups.
- **`mnemosyne/migrations/`** — packaged migrations (E6 triplestore split).
- **`hermes_memory_provider/`** — Hermes plugin (LLM adapter, sync adapter, audit).
- **`integrations/`** — Obsidian + VSCode (TypeScript) + Hermes plugin manifest.
- **`tools/`** — BEAM benchmark + diagnostics scripts (ICLR 2026 BEAM benchmark).
- **`deploy/`** — sync server deployment (Caddy, fly.io, docker-compose).
- **`scripts/`**, **`docs/`**, **`examples/`**, **`tests/`** — tooling, docs, examples, tests.

---

## 13. The one-sentence summary

Mnemosyne is a **single-SQLite-file memory engine** (`BeamMemory`) exposing a tiny facade
(`remember`/`recall`/`get_context`/`sleep`) over **three BEAM tiers** (working / episodic /
scratchpad), a **hybrid recall stack** (sqlite-vec + FTS5 + importance, with optional
polyphonic-RRF and enhanced pipelines), **MIB 48-byte binary vectors**, and a co-located
**temporal knowledge layer** (triples / annotations / canonical / episodic graph / veracity) —
all dependency-free, reachable from any agent via **MCP (24 tools), CLI, SDK, or
host-plugin**, with optional local-LLM extraction, decay/consolidation dynamics, and
event-log sync.
