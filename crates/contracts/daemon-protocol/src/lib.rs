//! `daemon-protocol` — the §17 host wire protocol and message envelopes.
//!
//! Defines the engine⇄host frames, request/response envelopes, and (de)serialization (CBOR via
//! `ciborium`). Depends only on `daemon-common`.

#![forbid(unsafe_code)]

// TODO: define §17 host-protocol frames + envelopes.
