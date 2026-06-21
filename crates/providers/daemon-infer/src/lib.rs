//! `daemon-infer` — the supervised local-inference worker (library + binary).
//!
//! Local engines can wedge a process on a GPU OOM/segfault/hang, so local inference runs in this
//! separate, supervised worker rather than in-process. The worker hosts an engine-agnostic
//! [`InferenceBackend`] seam with two impls behind features — llama.cpp (`llama-cpp-4`) and
//! mistral.rs (`mistralrs`) — selected at runtime by `--engine`, and speaks the length-framed
//! [`protocol`] over stdio. The daemon's `LocalProvider` (in `daemon-providers`) spawns + supervises
//! it and maps its frames onto the engine's streaming/recovery contract.
//!
//! The default build compiles only the [`StubBackend`] (no engine, no cmake): `daemon-providers`
//! depends on this crate with `default-features = false` purely for [`protocol`], and
//! `cargo test --workspace` builds the inert stub binary.

#![forbid(unsafe_code)]

pub mod backend;
pub mod backends;
pub mod grammar;
pub mod protocol;
pub mod tooling;

pub use backend::{BackendChunk, BackendError, GenerateRequest, InferenceBackend, StubBackend};
