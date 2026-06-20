//! AnnotationStore — port of `annotations.py` (the E6 triplestore split).
//!
//! Append-only, multi-valued per-memory tags with a `(memory_id, kind, value)` unique index and
//! `INSERT OR IGNORE` writes (`annotations.py` L128-L264). Scaffold.

/// The annotation kinds (`annotations.py` `ANNOTATION_KINDS` L77-L82).
pub const ANNOTATION_KINDS: &[&str] = &["mentions", "fact", "occurred_on", "has_source"];

/// Minimum fact length kept by the read-time filter (`annotations.py` L89).
pub const MIN_FACT_LENGTH: usize = 10;
