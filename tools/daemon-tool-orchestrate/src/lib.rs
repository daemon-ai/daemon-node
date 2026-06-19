//! `daemon-tool-orchestrate` — the agent veneer over the fleet runtime.
//!
//! Exposes orchestration as a `daemon_core::Tool` so the engine can spawn/steer children by policy;
//! the actual fleet mechanism lives in `daemon-orchestration`. This is the explicit DOWN edge of the
//! orchestration flow.

#![forbid(unsafe_code)]

// TODO: Tool surface that drives daemon-orchestration.
