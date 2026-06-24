//! Memory compression + pattern detection — port of `patterns.py` (P3).
//!
//! This module is intentionally not on the default path yet. It marks the future home for
//! `MemoryCompressor` (dict/RLE/semantic) and `PatternDetector` (temporal/content/sequence) once the
//! P3 dynamics pass is ported.

/// The currently shipped state of the memory-pattern dynamics port.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PatternDynamicsStatus {
    /// The module is reserved, but no runtime pattern/compression pass is wired.
    Reserved,
}

/// Report the explicit port status for docs/tests without implying runtime behavior.
pub const fn status() -> PatternDynamicsStatus {
    PatternDynamicsStatus::Reserved
}
