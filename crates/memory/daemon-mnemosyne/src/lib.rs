//! `daemon-mnemosyne` — a native Rust port of the Mnemosyne BEAM memory engine, exposed to
//! `daemon-core` as the default [`MemoryProvider`](daemon_core::memory::MemoryProvider).
//!
//! BEAM = **Bilevel Episodic-Associative Memory**: a single SQLite file per *bank* holding three
//! tiers (`working_memory`, `episodic_memory`, `scratchpad`), a hybrid recall stack (FTS5 + vector
//! similarity + importance), MIB 48-byte binary vectors, and a temporal knowledge layer
//! (triples / annotations / canonical / episodic graph / veracity). See
//! `docs/research/hermes/mnemosyne-rust-port-spec.md` for the full architecture spec with the
//! authoritative Python `file:line` references each module ports.
//!
//! Default build is light (no ONNX, no C vector extension): embeddings are keyword-only and vectors
//! score with an in-Rust cosine fallback. The `embeddings`, `vec-ext`, and `sync` features add the
//! heavier capabilities.

// The only `unsafe` in the crate is the sqlite-vec auto-extension registration (one transmute,
// behind `vec-ext`); the default build is fully safe.
#![cfg_attr(not(feature = "vec-ext"), forbid(unsafe_code))]

pub mod aaak;
pub mod binary_vectors;
pub mod config;
pub mod dynamics;
pub mod embeddings;
pub mod engine;
pub mod error;
pub mod knowledge;
pub mod provider;
pub mod recall;
pub mod sanitize;
pub mod store;
pub mod tokens;
pub mod util;

pub use config::MnemosyneConfig;
pub use engine::{Engine, MemoryRow, RememberArgs, Tier};
pub use error::{Error, Result};
pub use provider::MnemosyneProvider;
