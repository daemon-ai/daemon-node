//! `daemon-conformance` — the substrate conformance harness.
//!
//! Property/integration suites that pin the activation invariants (single-activation, fencing, crash
//! recovery, exactly-once completion) using `daemon-stub-engine` against `daemon-host`/`daemon-activation`.
//! This is the executable acceptance gate for the build-first milestone.

#![forbid(unsafe_code)]

// TODO: conformance suites for the durable activation core.
