//! `daemon-activation` — the durable activation / virtual-entity core.
//!
//! Single-activation guarantee, passivation/rehydration, lease + fencing, recovery scan driving the
//! wake outbox. This is the correctness-critical layer with no upstream reference implementation —
//! the build-first milestone. The `elfo` feature gates an optional elfo-backed mailbox experiment.
//! Depends on `daemon-store` + `daemon-common`.

#![forbid(unsafe_code)]

// TODO: implement activation manager (single-activation, lease/fence, passivate/rehydrate, recovery).
