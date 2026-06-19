//! `daemon-core-ffi` - C ABI shell for embedding the engine.
//!
//! This crate will expose the `daemon-core` brain over the Section 17 host protocol using opaque
//! handles and CBOR byte buffers. It intentionally defines no domain model of its own; the payload
//! contract stays in `daemon-protocol`.
//!
//! See `docs/specs/daemon-ffi-spec.md`.

#![deny(unsafe_op_in_unsafe_fn)]

// TODO: add the opaque runtime/session handles and CBOR message pump once `daemon-protocol` frames
// are implemented.
