//! Retrieval & ranking: the shared scoring math plus the selectable pipelines (linear hybrid is the
//! default; enhanced/polyphonic/SHMR are opt-in).

pub mod diagnostics;
pub mod mmr;
pub mod polyphonic;
pub mod query_cache;
pub mod query_intent;
pub mod scoring;
pub mod shmr;
pub mod synonyms;
