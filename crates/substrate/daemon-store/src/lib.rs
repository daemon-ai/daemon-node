//! `daemon-store` — durable persistence primitives for the activation core.
//!
//! `SessionStore` trait: snapshots, completion inbox, wake outbox, recovery scans, leases/fencing.
//! The `sqlite` feature selects a concrete backend; the default is an in-memory store for tests.
//! Depends only on `daemon-common`.
//!
//! See `docs/specs/daemon-lifecycle-persistence.md`.

#![forbid(unsafe_code)]

// TODO: define SessionStore trait + in-memory backend (sqlite behind `sqlite` feature).
