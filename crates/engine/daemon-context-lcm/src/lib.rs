//! `daemon-context-lcm` — a native Rust port of **hermes-lcm**, exposed to `daemon-core` as the
//! default [`ContextEngine`](daemon_core::context::ContextEngine) (§10 in-session context manager).
//!
//! LCM = **Living Context Model**: it manages the *in-session* context body — measuring budget
//! pressure each turn, compacting the conversation when it grows past the model's threshold, and
//! (eventually) maintaining a summary DAG over a SQLite store with FTS5 search and the `lcm_*`
//! drill-down tools. See `crates/engine/daemon-core/docs/daemon-context-lcm-port-spec.md` for the
//! full architecture spec with the authoritative Python `file:line` references each module ports.
//!
//! This is the **skeleton** milestone: the seam is implemented end-to-end (the engine wires
//! [`LcmContextEngine`] as its `ContextEngine`), the SQLite store + config layers exist, and
//! compaction is a faithful-but-minimal drop-oldest delegating to the in-core
//! [`BudgetedContextEngine`](daemon_core::context::BudgetedContextEngine). The deep port (summary
//! DAG, 3-level escalation, externalization, `lcm_*` tools) grows from here per the spec phases.
//!
//! Tools are *not* dispatched through the §10 seam: when the `lcm_*` tools land they register through
//! the §12 [`ToolRegistry`](daemon_core::tools) holding an `Arc<LcmContextEngine>`; [`ContextEngine::tools`]
//! returns only their advisory names.

#![forbid(unsafe_code)]

pub mod compaction;
pub mod config;
pub mod error;
pub mod provider;
pub mod store;

pub use config::LcmConfig;
pub use error::{Error, Result};
pub use provider::LcmContextEngine;
pub use store::SummaryNode;
