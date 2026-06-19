//! `daemon-host` — the durable substrate that runs a unit.
//!
//! Tiles the logical `ManagedUnit` tree onto real runtimes: activation, resident-service supervision,
//! credential authority, workspace provisioning, live-resource ownership, and the §17 ⇄ management
//! protocol translation. For a leaf it is single-faced (management up, §17 down to one engine); for an
//! orchestrator node it is two-faced (management server up, management client down to children's
//! hosts across cuts).
//!
//! See `docs/specs/daemon-host-spec.md`.

#![forbid(unsafe_code)]

// TODO: assemble activation + supervision + provisioning + credentials behind the §17/management adapter.
