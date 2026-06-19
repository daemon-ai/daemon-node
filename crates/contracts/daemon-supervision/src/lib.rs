//! `daemon-supervision` — the generic management protocol.
//!
//! `ManageEvent` / `ManageRequest` / `Ack`, restart strategies (OneForOne / OneForAll / RestForOne),
//! backoff and meltdown policy, and the `ManagedUnit` interface that makes the unit tree recursive.
//! Engine crates do **not** depend on this (host adapts §17 ⇄ management). Depends only on
//! `daemon-common`.
//!
//! See `docs/specs/daemon-supervision-spec.md`.

#![forbid(unsafe_code)]

// TODO: define ManagedUnit, ManageEvent/ManageRequest/Ack, restart strategies + meltdown policy.
