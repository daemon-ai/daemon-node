//! `daemon-orchestration` — the fleet runtime (not an engine).
//!
//! Holds fleet state and runs the downward management-protocol client + child placement that the host
//! exposes for an orchestrator node. The engine decides *what/when* (policy) and calls in via
//! `daemon-tool-orchestrate`; this crate owns the *how* (child lifecycle, event/completion plumbing,
//! upward request handling). Depends on `daemon-host` + `daemon-supervision`.
//!
//! See `docs/research/daemon-orchestration-synthesis.md`.

#![forbid(unsafe_code)]

// TODO: fleet runtime — child placement, management-client loop, completion/escalation plumbing.
