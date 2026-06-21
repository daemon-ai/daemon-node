//! `daemon-pytool` — the wire protocol for the out-of-process **Python tool worker**.
//!
//! Python tools run in a separate process, exactly like the MeTTa symbolic coprocessor
//! (`daemon-metta`) and the local-inference worker (`daemon-infer`) isolate their crash-prone /
//! `!Send` engines: the daemon spawns a worker, speaks this [`protocol`] over a length-framed
//! [`daemon_provision::CutChannel`], and registers a proxy [`daemon_core::Tool`] per discovered
//! Python tool (the supervised client lives in `daemon-pytool-client`). Nothing Python-specific
//! links into the daemon — only these `serde` wire types.
//!
//! Unlike the metta/infer workers (which are Rust binaries), the real Python worker is the
//! `daemon_pytool` Python package shipped under `python/`. The body codec is therefore **JSON**
//! (not CBOR), so the Python SDK stays stdlib-only (`json` + `struct`): the
//! [`CutChannel`](daemon_provision::CutChannel) owns the `u32`-LE length prefix, this module owns
//! the body [`encode`](protocol::encode)/[`decode`](protocol::decode).
//!
//! The crate also ships a hermetic Rust [`fake-pytool-worker`](../fake_pytool_worker/index.html)
//! binary that speaks the protocol, so the client's tests (and `tests/daemon-conformance`) exercise
//! the supervised path without a system Python.

#![forbid(unsafe_code)]

pub mod protocol;

pub use protocol::{
    Command, Concurrency, ErrorClass, Event, ResultDetail, ToolManifest, PROTOCOL_VERSION,
};
