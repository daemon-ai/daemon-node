//! `daemon-models` — unified local-inference model search, acquisition, caching, and catalog.
//!
//! This crate is the one owner of the model lifecycle the daemon exposes to clients (the
//! `daemon-api` `ModelApi` surface) and consumes internally (the local provider's
//! resolve-before-load). It unifies what was asymmetric across the local engines: `mistral.rs`
//! auto-downloads from Hugging Face while `llama-cpp-4` only takes a local path. Here, the **daemon**
//! owns acquisition for *both* — it downloads into the shared Hugging Face cache via `hf-hub` (the
//! same cache the `mistralrs` engine reads), then the engine sidecar loads from the warmed cache
//! offline.
//!
//! Modules:
//! - [`cache`] — shared HF cache-dir + token resolution (ported from `mistral.rs`), plus the
//!   offline sidecar environment.
//! - [`hf`] — the read surface: repo [`hf::search`] (step 1) and per-repo [`hf::files`] (step 2).
//! - [`gguf`] — GGUF filename heuristics (quant label, split shards) + a magic-byte preflight.
//! - [`acquire`] — `hf-hub`-backed downloads with per-job progress / pause / resume / cancel.
//! - [`resolve`] — turning a [`daemon_common::ModelRef`] into a concrete download plan per engine.
//! - [`registry`] — the installed-model catalog (atomic JSON manifest).
//! - [`manager`] — the [`manager::ModelManager`] facade the node + provider wiring call.

#![forbid(unsafe_code)]

pub mod acquire;
pub mod cache;
pub mod error;
pub mod gguf;
pub mod hardware;
pub mod hf;
pub mod inspect;
pub mod manager;
pub mod quantize;
pub mod recommend;
pub mod registry;
pub mod resolve;

pub use acquire::{DownloadPlan, DownloadProgressCb, Downloader, PlanFile, ResolvedArtifact};
pub use cache::CacheConfig;
pub use error::{ModelError, Result};
pub use hardware::HardwareProbe;
pub use hf::HfClient;
pub use inspect::inspect;
pub use manager::{ActiveModels, ManagerConfig, ModelManager};
pub use quantize::{QuantizeRequest, Quantizer};
pub use registry::{model_id, Registry};
