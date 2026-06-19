//! `daemon-core` — the agent engine (the "brain").
//!
//! Owns turn/policy logic and the [`Tool`] trait that tools implement. Speaks the §17 host protocol
//! (`daemon-protocol`); it is intentionally unaware of `daemon-supervision` — the host adapts the
//! management protocol on its behalf.
//!
//! See `crates/engine/daemon-core/docs/` for the engine spec family.

#![forbid(unsafe_code)]

use async_trait::async_trait;

/// A capability the engine can invoke during a turn.
///
/// Tool crates (`daemon-tool-*`) implement this; `daemon-tool-orchestrate` is the veneer over the
/// fleet runtime. Signature is a placeholder for the scaffold.
#[async_trait]
pub trait Tool: Send + Sync {
    /// Stable tool name as exposed to the engine.
    fn name(&self) -> &str;
}
