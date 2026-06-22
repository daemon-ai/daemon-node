//! Typed-memory classification — port of `typed_memory.py`.
//!
//! 13 regex-classified types (`typed_memory.py` L37-L52) feeding the Weibull decay map. The full
//! 69-pattern `TYPE_PATTERNS` table (L67-L168), `CONFIDENCE_BOOSTERS` (L174-L188), and the
//! `classify_memory` scoring/tie-break (L191-L249) are ported verbatim.

use regex::Regex;
use std::sync::OnceLock;

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

    /// The enum's positional index (`list(MemoryType).index(...)`, `typed_memory.py` L232). This
    /// drives the tie-break `confidence * (1 + 0.1 * index)`, so later types win ties.
    fn type_index(self) -> usize {
        match self {
            MemoryType::Fact => 0,
            MemoryType::Preference => 1,
            MemoryType::Decision => 2,
            MemoryType::Commitment => 3,
            MemoryType::Goal => 4,
            MemoryType::Event => 5,
            MemoryType::Instruction => 6,
            MemoryType::Relationship => 7,
            MemoryType::Context => 8,
            MemoryType::Learning => 9,
            MemoryType::Observation => 10,
            MemoryType::Error => 11,
            MemoryType::Artifact => 12,
            MemoryType::Unknown => 13,
        }
    }

    /// Confidence-booster keywords for a type (`CONFIDENCE_BOOSTERS`, `typed_memory.py` L174-L188).
    fn boosters(self) -> &'static [&'static str] {
        match self {
            MemoryType::Fact => {
                &["verified", "confirmed", "official", "documented", "according to", "data shows"]
            }
            MemoryType::Preference => {
                &["always", "never", "absolutely", "definitely", "strongly"]
            }
            MemoryType::Decision => &["final", "official", "approved", "agreed", "consensus"],
            MemoryType::Commitment => &["promise", "guarantee", "committed", "deadline", "sla"],
            MemoryType::Goal => &["target", "objective", "kpi", "okr", "success metric"],
            MemoryType::Event => &["specifically", "exactly", "precisely", "at", "on"],
            MemoryType::Instruction => &["mandatory", "required", "critical", "important"],
            MemoryType::Relationship => &["directly", "reports to", "managed by", "owned by"],
            MemoryType::Context => &["currently", "right now", "active", "in progress"],
            MemoryType::Learning => &["key lesson", "important finding", "critical insight"],
            MemoryType::Observation => &["consistently", "repeatedly", "over time", "pattern"],
            MemoryType::Error => &["critical", "severe", "blocking", "p0", "p1"],
            MemoryType::Artifact => &["official", "canonical", "source of truth", "reference"],
            MemoryType::Unknown => &[],
        }
    }
}

/// The 69-entry `TYPE_PATTERNS` table (`typed_memory.py` L67-L168): `(regex, type, base_confidence)`.
/// Priority is dropped (the engine only persists the type). Patterns are compiled once,
/// case-insensitively, and matched against the lowercased content.
#[allow(clippy::type_complexity)]
fn type_patterns() -> &'static [(Regex, MemoryType, f64)] {
    static PATTERNS: OnceLock<Vec<(Regex, MemoryType, f64)>> = OnceLock::new();
    PATTERNS.get_or_init(|| {
        let raw: &[(&str, MemoryType, f64)] = &[
            // FACT
            (r"\b(is|are|was|were)\s+(a|an|the)\s+\w+", MemoryType::Fact, 0.6),
            (r"\b(has|have|had)\s+\d+", MemoryType::Fact, 0.7),
            (r"\b(contains|consists?|comprises?)\b", MemoryType::Fact, 0.8),
            (r"\b(version|v)\s*\d+\.?\d*", MemoryType::Fact, 0.9),
            (r"\b(API|endpoint|URL|database|DB)\s+(is|at|points?\s+to)", MemoryType::Fact, 0.8),
            (r"\b(created|modified|updated)\s+(on|at)\s+\d{4}", MemoryType::Fact, 0.8),
            // PREFERENCE
            (r"\b(prefer|likes?|enjoys?|loves?|hates?|dislikes?)\b", MemoryType::Preference, 0.8),
            (r"\b(want|wants|wanted)\s+(to|the|a|an)\b", MemoryType::Preference, 0.6),
            (r"\b(rather|instead|alternative)\b", MemoryType::Preference, 0.5),
            (r"\b(dark\s+mode|light\s+mode|theme|color\s+scheme)\b", MemoryType::Preference, 0.9),
            (r"\b(usually|typically|normally|generally)\b", MemoryType::Preference, 0.6),
            // DECISION
            (r"\b(decided|chose|selected|picked|opted)\b", MemoryType::Decision, 0.9),
            (r"\b(going\s+with|settled\s+on|locked\s+in)\b", MemoryType::Decision, 0.8),
            (r"\b(choose|select|pick)\s+(between|from|among)\b", MemoryType::Decision, 0.7),
            (r"\b(final\s+decision|final\s+call|final\s+choice)\b", MemoryType::Decision, 0.9),
            (r"\b(will\s+use|using|adopt|adopting)\s+(the|a|an)?\s*\w+", MemoryType::Decision, 0.7),
            // COMMITMENT
            (
                r"\b(will|shall|must|need\s+to)\s+\w+\s+(by|before|until)\b",
                MemoryType::Commitment,
                0.8,
            ),
            (r"\b(deadline|due\s+date|due|milestone)\b", MemoryType::Commitment, 0.9),
            (r"\b(promise|committed|pledged|obligated)\b", MemoryType::Commitment, 0.9),
            (r"\b(deliver|ship|release|deploy)\s+(by|before|on)\b", MemoryType::Commitment, 0.8),
            (
                r"\b(EOD|COB|end\s+of\s+day|close\s+of\s+business)\b",
                MemoryType::Commitment,
                0.7,
            ),
            (r"\b(tomorrow|next\s+week|Monday|Friday)\s+(by|at)\b", MemoryType::Commitment, 0.6),
            // GOAL
            (r"\b(goal|objective|target|aim|purpose)\b", MemoryType::Goal, 0.9),
            (r"\b(achieve|reach|hit|attain|accomplish)\s+\d+", MemoryType::Goal, 0.8),
            (r"\b(KPI|metric|OKR|success\s+criteria)\b", MemoryType::Goal, 0.9),
            (r"\b(roadmap|plan|strategy)\s+(for|to)\b", MemoryType::Goal, 0.7),
            (
                r"\b(reach|get\s+to|grow\s+to)\s+\d+[KkMm]?\s+(users|customers|revenue)\b",
                MemoryType::Goal,
                0.8,
            ),
            // EVENT
            (r"\b(meeting|call|discussion|conversation)\s+(with|about)\b", MemoryType::Event, 0.7),
            (r"\b(happened|occurred|took\s+place|went\s+down)\b", MemoryType::Event, 0.8),
            (r"\b(yesterday|last\s+week|last\s+month|earlier\s+today)\b", MemoryType::Event, 0.6),
            (r"\b(scheduled|planned|booked|set\s+up)\s+(for|at)\b", MemoryType::Event, 0.7),
            (r"\b(incident|outage|bug|issue)\s+#?\d+", MemoryType::Event, 0.8),
            (r"\b( launched|released|shipped|deployed)\s+(on|at)\b", MemoryType::Event, 0.8),
            // INSTRUCTION
            (r"\b(always|never|must|should|shall|do\s+not|don't)\b", MemoryType::Instruction, 0.7),
            (r"\b(rule|policy|guideline|procedure|protocol)\b", MemoryType::Instruction, 0.9),
            (r"\b(how\s+to|steps?\s+to|guide\s+to|tutorial)\b", MemoryType::Instruction, 0.8),
            (r"\b(remember\s+to|make\s+sure|ensure|verify)\b", MemoryType::Instruction, 0.6),
            (r"\b(first|then|next|finally)\s*,?\s*\w+", MemoryType::Instruction, 0.5),
            (r"\b(if\s+.+\s+then\s+.+)", MemoryType::Instruction, 0.7),
            // RELATIONSHIP
            (r"\b(manages?|reports?\s+to|supervises?|leads?)\b", MemoryType::Relationship, 0.9),
            (r"\b(owns?|belongs?\s+to|part\s+of|member\s+of)\b", MemoryType::Relationship, 0.8),
            (
                r"\b(works?\s+with|collaborates?\s+with|partners?\s+with)\b",
                MemoryType::Relationship,
                0.8,
            ),
            (r"\b(depends?\s+on|requires?|needs?)\b", MemoryType::Relationship, 0.7),
            (r"\b(related\s+to|connected\s+to|associated\s+with)\b", MemoryType::Relationship, 0.6),
            (
                r"\b(is\s+a|is\s+an)\s+(type\s+of|kind\s+of|form\s+of)\b",
                MemoryType::Relationship,
                0.7,
            ),
            // CONTEXT
            (r"\b(currently|right\s+now|at\s+the\s+moment|presently)\b", MemoryType::Context, 0.7),
            (r"\b(working\s+on|focusing\s+on|dealing\s+with)\b", MemoryType::Context, 0.8),
            (r"\b(status|state|phase|stage)\s+(is|of)\b", MemoryType::Context, 0.7),
            (r"\b(in\s+progress|ongoing|active|pending|blocked)\b", MemoryType::Context, 0.8),
            (r"\b(environment|setup|configuration|settings?)\b", MemoryType::Context, 0.6),
            (r"\b(today|this\s+week|this\s+sprint|this\s+quarter)\b", MemoryType::Context, 0.5),
            // LEARNING
            (r"\b(learned|realized|discovered|found\s+out)\b", MemoryType::Learning, 0.8),
            (r"\b(lesson|takeaway|insight|finding)\b", MemoryType::Learning, 0.9),
            (r"\b(turns?\s+out|surprisingly|interestingly)\b", MemoryType::Learning, 0.7),
            (r"\b(should\s+have|could\s+have|would\s+have)\b", MemoryType::Learning, 0.6),
            (
                r"\b(best\s+practice|lessons?\s+learned|post[-\s]?mortem)\b",
                MemoryType::Learning,
                0.9,
            ),
            // OBSERVATION
            (r"\b(noticed|observed|saw|seems?)\b", MemoryType::Observation, 0.7),
            (r"\b(pattern|trend|correlation|tends?\s+to)\b", MemoryType::Observation, 0.9),
            (r"\b(often|frequently|sometimes|rarely|usually)\s+\w+", MemoryType::Observation, 0.6),
            (r"\b(appears?|looks?\s+like|seems?\s+like)\b", MemoryType::Observation, 0.6),
            (
                r"\b(increasing|decreasing|growing|shrinking|stable)\b",
                MemoryType::Observation,
                0.7,
            ),
            (r"\b(every\s+time|whenever|each\s+time)\b", MemoryType::Observation, 0.8),
            // ERROR
            (r"\b(error|bug|issue|problem|failure|crash)\b", MemoryType::Error, 0.7),
            (r"\b(broke|broken|failed|failing|doesn't\s+work)\b", MemoryType::Error, 0.8),
            (
                r"\b(do\s+not|never|avoid|watch\s+out|be\s+careful)\s+\w+\s+(error|bug|issue)\b",
                MemoryType::Error,
                0.9,
            ),
            (r"\b(deprecated|obsolete|legacy|outdated)\b", MemoryType::Error, 0.8),
            (r"\b(exception|timeout|crash|hang|freeze)\b", MemoryType::Error, 0.8),
            (r"\b(workaround|hotfix|patch|kludge)\b", MemoryType::Error, 0.7),
            // ARTIFACT
            (r"\b(document|doc|spreadsheet|sheet|slide)\b", MemoryType::Artifact, 0.6),
            (r"\b(file|folder|directory|path)\s+(name|called|at)\b", MemoryType::Artifact, 0.7),
            (r"\b(PR|pull\s+request|issue|ticket|ticket)\s+#?\d+", MemoryType::Artifact, 0.9),
            (r"\b(commit|branch|tag|release)\s+[a-f0-9]{7,40}\b", MemoryType::Artifact, 0.9),
            (r"\b(repo|repository|project|codebase)\s+(at|on|in)\b", MemoryType::Artifact, 0.7),
            (r"\b(link|URL|href|reference)\s+(to|for)\b", MemoryType::Artifact, 0.6),
            (r"\b(README|CHANGELOG|LICENSE|CONTRIBUTING)\b", MemoryType::Artifact, 0.9),
        ];
        raw.iter()
            .map(|(pat, ty, conf)| {
                let re = Regex::new(&format!("(?i){pat}"))
                    .expect("typed_memory TYPE_PATTERNS regex must compile");
                (re, *ty, *conf)
            })
            .collect()
    })
}

/// Classify content into a [`MemoryType`] (`typed_memory.py` `classify_memory` L191-L249).
///
/// Scores every matching pattern (base confidence `+0.1` if the match is >20 chars / `+0.05` if
/// >10, `+0.05` per booster keyword, capped at 1.0), tie-breaks with `confidence * (1 + 0.1 *
/// type_index)`, and falls back to `Fact` (<5 words) or `Context` when nothing matches.
pub fn classify(content: &str) -> MemoryType {
    classify_scored(content).0
}

/// Like [`classify`] but also returns the winning confidence (used by parity tests).
pub fn classify_scored(content: &str) -> (MemoryType, f64) {
    if content.trim().is_empty() {
        return (MemoryType::Unknown, 0.0);
    }
    let lower = content.to_lowercase();
    let mut best: Option<(MemoryType, f64)> = None;
    let mut best_score = 0.0_f64;
    for (re, mem_type, base) in type_patterns() {
        if let Some(m) = re.find(&lower) {
            let mut confidence = *base;
            let match_len = m.as_str().chars().count();
            if match_len > 20 {
                confidence += 0.1;
            } else if match_len > 10 {
                confidence += 0.05;
            }
            for booster in mem_type.boosters() {
                if lower.contains(booster) {
                    confidence += 0.05;
                }
            }
            confidence = confidence.min(1.0);
            let score = confidence * (1.0 + 0.1 * mem_type.type_index() as f64);
            if score > best_score {
                best_score = score;
                best = Some((*mem_type, confidence));
            }
        }
    }
    match best {
        Some(hit) => hit,
        None => {
            if content.split_whitespace().count() < 5 {
                (MemoryType::Fact, 0.3)
            } else {
                (MemoryType::Context, 0.3)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The labeled corpus from `typed_memory.py.__main__` (L325-L338): each line must classify to
    /// the documented type, exercising every one of the 13 types.
    #[test]
    fn labeled_corpus_matches_python() {
        let cases: &[(&str, MemoryType)] = &[
            ("The API endpoint is at https://api.example.com/v2", MemoryType::Fact),
            ("I prefer dark mode for all my applications", MemoryType::Preference),
            ("We decided to go with PostgreSQL instead of MongoDB", MemoryType::Decision),
            ("I will deliver the report by Friday EOD", MemoryType::Commitment),
            ("Our goal is to reach 10K users by Q4", MemoryType::Goal),
            ("We had a meeting with the CEO yesterday at 2pm", MemoryType::Event),
            ("Always validate user input before processing", MemoryType::Instruction),
            ("Alice manages Bob and reports to Charlie", MemoryType::Relationship),
            ("Currently working on the authentication module", MemoryType::Context),
            ("Key lesson: users need simpler onboarding", MemoryType::Learning),
            ("I noticed traffic peaks every Friday afternoon", MemoryType::Observation),
            ("Critical bug: null pointer exception in login flow", MemoryType::Error),
            ("See the Q3 budget spreadsheet for details", MemoryType::Artifact),
        ];
        for (content, want) in cases {
            assert_eq!(classify(content), *want, "misclassified: {content:?}");
        }
    }

    #[test]
    fn empty_is_unknown() {
        assert_eq!(classify("   "), MemoryType::Unknown);
    }

    #[test]
    fn no_match_defaults_by_length() {
        assert_eq!(classify("blue green"), MemoryType::Fact); // <5 words
        assert_eq!(
            classify("the quick brown fox jumped over"),
            MemoryType::Context // >=5 words, no pattern
        );
    }

    #[test]
    fn booster_raises_confidence() {
        // "critical" is an ERROR booster; the bare pattern is 0.7, boosted to 0.75.
        let (ty, conf) = classify_scored("there is a critical bug here");
        assert_eq!(ty, MemoryType::Error);
        assert!((conf - 0.75).abs() < 1e-9, "got {conf}");
    }
}
