//! `daemon-context-lcm` — a native Rust port of **hermes-lcm**, exposed to `daemon-core` as the
//! default [`ContextEngine`](daemon_core::context::ContextEngine) (§10 in-session context manager).
//!
//! LCM = **Living Context Model**: it manages the *in-session* context body — measuring budget
//! pressure each turn, compacting the conversation when it grows past the model's threshold, and
//! (eventually) maintaining a summary DAG over a SQLite store with FTS5 search and the `lcm_*`
//! drill-down tools. See `crates/engine/daemon-core/docs/daemon-context-lcm-port-spec.md` for the
//! full architecture spec with the authoritative Python `file:line` references each module ports.
//!
//! Implemented (spec milestones **M1-M8**, the full port): the SQLite store (lossless `messages`
//! transcript + summary DAG with `source_ids` lineage + FTS5 indexes + lifecycle frontier — §4/§5),
//! `tiktoken` token counting (§6.1), the 3-level escalation summarizer over a host-injected aux
//! [`Provider`](daemon_core::provider::Provider) with a per-route fallback chain (§7), the compaction
//! engine that summarizes the region outside a fresh tail into the DAG and reassembles
//! `[system] + [summary] + [fresh tail]` (§6), full per-turn transcript ingest with rehydration
//! reconcile, the [`search`] stack (§11), and the seven `lcm_*` drill-down [`tools`] (§10).
//!
//! **M5 ingest [`protection`]** (§8/§9): opt-in sensitive redaction, the always-on (persistent-bank)
//! base64/data-URI storage guard, loop/heartbeat quarantine, large-payload [`externalize`]ation, and
//! pre-compaction [`extraction`] of durable decisions — wired at the ingest/compaction boundary, all
//! opt-in except the storage guard. **M7 filters/routing/presets** (§12.3-12.6): message + session
//! filter [`patterns`], the [`model_routing`] shim, and inert [`presets`] surfaced via `lcm_status`.
//!
//! Tools are *not* dispatched through the §10 seam: the `lcm_*` tools register through the §12
//! [`ToolRegistry`](daemon_core::tools) via a host adapter that resolves the calling session's
//! engine and delegates to [`LcmContextEngine::call_tool`]; [`ContextEngine::tools`] returns only
//! their advisory names.

#![forbid(unsafe_code)]

pub mod compaction;
pub mod config;
pub mod error;
pub mod escalation;
pub mod externalize;
pub mod extraction;
pub mod ingest;
pub mod model_routing;
pub mod patterns;
pub mod presets;
pub mod protection;
pub mod provider;
pub mod search;
pub mod store;
pub mod tokens;
pub mod tools;

pub use config::LcmConfig;
pub use error::{Error, Result};
pub use provider::{command_specs, LcmContextEngine};
pub use search::{MessageResult, NodeResult, SortMode};
pub use store::{MessageFilter, MessageRow, NewMessage, NewNode, NodeHit, SourceType, SummaryNode};
