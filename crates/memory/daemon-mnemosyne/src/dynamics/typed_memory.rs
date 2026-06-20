//! Typed-memory classification — port of `typed_memory.py`.
//!
//! 13 regex-classified types (`typed_memory.py` L37-L52) feeding the Weibull decay map. Scaffold:
//! the type enum is complete; `classify` returns the documented default (`<5 words -> Fact`, else
//! `Context`) until the 69-pattern table (L67-L168) is ported.

/// The 13 memory types plus `Unknown` (`typed_memory.py` L37-L52).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryType {
    /// A stable fact.
    Fact,
    /// A user preference.
    Preference,
    /// A decision.
    Decision,
    /// A time-critical commitment.
    Commitment,
    /// A goal.
    Goal,
    /// A dated event.
    Event,
    /// An agent instruction.
    Instruction,
    /// A relationship between entities.
    Relationship,
    /// Conversational context.
    Context,
    /// A learning / insight.
    Learning,
    /// An observation.
    Observation,
    /// An error.
    Error,
    /// An artifact reference (commit/branch/...).
    Artifact,
    /// Unclassified.
    Unknown,
}

impl MemoryType {
    /// The lowercase string used as the DB `memory_type` value and Weibull key.
    pub fn as_str(self) -> &'static str {
        match self {
            MemoryType::Fact => "fact",
            MemoryType::Preference => "preference",
            MemoryType::Decision => "decision",
            MemoryType::Commitment => "commitment",
            MemoryType::Goal => "goal",
            MemoryType::Event => "event",
            MemoryType::Instruction => "instruction",
            MemoryType::Relationship => "relationship",
            MemoryType::Context => "context",
            MemoryType::Learning => "learning",
            MemoryType::Observation => "observation",
            MemoryType::Error => "error",
            MemoryType::Artifact => "artifact",
            MemoryType::Unknown => "unknown",
        }
    }
}

/// Classify content into a [`MemoryType`] (`typed_memory.py` `classify_memory` L191-L249).
/// Scaffold: default heuristic only (`<5 words -> Fact@0.3`, else `Context@0.3`, L242-L247).
pub fn classify(content: &str) -> MemoryType {
    // TODO: port TYPE_PATTERNS (69 regexes) + CONFIDENCE_BOOSTERS scoring.
    if content.split_whitespace().count() < 5 {
        MemoryType::Fact
    } else {
        MemoryType::Context
    }
}
