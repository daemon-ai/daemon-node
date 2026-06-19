//! `daemon-common` — shared primitives across the workspace.
//!
//! Stable identifiers (`SessionId`, `UnitId`, `JobId`), `Budget`, `FenceToken`, error scaffolding,
//! wire-version, and CDDL helpers. Pure types only; no runtime. This is the root of the crate DAG —
//! it depends on nothing internal.
//!
//! See `docs/daemon-workspace-layout.md` and `docs/specs/`.

#![forbid(unsafe_code)]

// TODO: define SessionId / UnitId / JobId / Budget / FenceToken and shared error types.
