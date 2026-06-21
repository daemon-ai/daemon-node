//! `daemon-context-lcm` — a native Rust port of **hermes-lcm**, exposed to `daemon-core` as the
//! default [`ContextEngine`](daemon_core::context::ContextEngine) (§10 in-session context manager).
//!
//! LCM = **Living Context Model**: it manages the *in-session* context body — measuring budget
//! pressure each turn, compacting the conversation when it grows past the model's threshold, and
//! (eventually) maintaining a summary DAG over a SQLite store with FTS5 search and the `lcm_*`
//! drill-down tools. See `crates/engine/daemon-core/docs/daemon-context-lcm-port-spec.md` for the
//! full architecture spec with the authoritative Python `file:line` references each module ports.
//!
//! Implemented (spec milestones **M1-M4**): the SQLite store (lossless `messages` transcript +
//! summary DAG with `source_ids` lineage + FTS5 indexes + lifecycle frontier — §4/§5), `tiktoken`
//! token counting (§6.1), the 3-level escalation summarizer over a host-injected aux
//! [`Provider`](daemon_core::provider::Provider) (§7), and the compaction engine that summarizes the
//! region outside a fresh tail into the DAG and reassembles `[system] + [summary] + [fresh tail]`
//! (§6). Still to come: ingest protection (M5), the seven `lcm_*` drill-down tools + search (M6),
//! and routing/presets (M7).
//!
//! Tools are *not* dispatched through the §10 seam: when the `lcm_*` tools land they register through
//! the §12 [`ToolRegistry`](daemon_core::tools) holding an `Arc<LcmContextEngine>`; [`ContextEngine::tools`]
//! returns only their advisory names.

#![forbid(unsafe_code)]

pub mod compaction;
pub mod config;
pub mod error;
pub mod escalation;
pub mod ingest;
pub mod provider;
pub mod store;
pub mod tokens;

pub use config::LcmConfig;
pub use error::{Error, Result};
pub use provider::LcmContextEngine;
pub use store::{MessageRow, NewMessage, NewNode, SourceType, SummaryNode};
