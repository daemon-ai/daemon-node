//! `daemon-ffi` - C ABI shell for embedding the durable daemon system.
//!
//! This crate will expose `daemon-host` over the management protocol plus the Section 17 host
//! protocol using opaque handles and CBOR byte buffers. It intentionally defines no domain model of
//! its own; payload contracts stay in `daemon-protocol` and `daemon-supervision`.
//!
//! See `docs/specs/daemon-ffi-spec.md`.

#![deny(unsafe_op_in_unsafe_fn)]

// TODO: add the opaque host/session handles and CBOR message pump once the host management surface is
// implemented.
