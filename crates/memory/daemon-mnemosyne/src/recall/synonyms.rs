//! Query normalization + synonym expansion — port of `synonyms.py` (plus beam's recall-time map).
//!
//! Two synonym tables are ported here, mirroring the Python split:
//! - [`SYNONYM_GROUPS`] (40 concept groups, `synonyms.py` L15-L56) drives `normalize_query`
//!   (`synonyms.py` L90-L113) and `expand_query` (L116-L143) for the enhanced-recall FTS expansion.
//! - [`recall_synonyms`] ports beam's small conservative `_RECALL_SYNONYMS` map (`beam.py`
//!   L1477-L1489), used by the lexical `+0.75` synonym partial in `Engine::lexical_relevance`.

use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

/// The 40 concept groups (`synonyms.py` `SYNONYM_GROUPS` L15-L56): canonical form first, then its
/// synonyms.
#[allow(clippy::type_complexity)]
pub const SYNONYM_GROUPS: &[(&str, &[&str])] = &[
    ("database", &["db", "datastore", "data_store"]),
    (
        "password",
        &["pass", "pwd", "passwd", "credential", "secret", "token"],
    ),
    ("config", &["configuration", "settings", "cfg", "setup"]),
    (
        "error",
        &[
            "bug",
            "issue",
            "fault",
            "failure",
            "crash",
            "exception",
            "traceback",
        ],
    ),
    (
        "fix",
        &["repair", "resolve", "solve", "patch", "correct", "address"],
    ),
    (
        "deploy",
        &["deployment", "release", "ship", "push", "rollout"],
    ),
    (
        "server",
        &["host", "machine", "vm", "instance", "node", "vps"],
    ),
    ("api", &["endpoint", "interface", "service"]),
    ("key", &["token", "credential", "secret", "api_key"]),
    ("user", &["account", "profile", "identity", "person"]),
    (
        "model",
        &["llm", "ai", "provider", "gpt", "claude", "gemini"],
    ),
    (
        "speed",
        &["fast", "quick", "performance", "latency", "throughput"],
    ),
    ("memory", &["recall", "remember", "storage", "retention"]),
    ("search", &["find", "lookup", "query", "retrieve", "locate"]),
    ("file", &["document", "doc", "text", "note"]),
    ("code", &["script", "program", "source", "implementation"]),
    ("test", &["verify", "check", "validate", "probe", "examine"]),
    ("backup", &["snapshot", "copy", "save", "archive"]),
    ("install", &["setup", "configure", "bootstrap", "init"]),
    ("update", &["upgrade", "refresh", "renew", "sync"]),
    (
        "delete",
        &["remove", "destroy", "purge", "clean", "wipe", "erase"],
    ),
    ("list", &["show", "display", "enumerate", "catalog"]),
    ("time", &["date", "when", "timestamp", "schedule"]),
    ("url", &["link", "address", "uri", "path"]),
    ("health", &["status", "check", "pulse", "alive", "up"]),
    ("service", &["daemon", "process", "systemd", "worker"]),
    ("port", &["socket", "bind", "listen"]),
    (
        "network",
        &["internet", "connection", "connectivity", "dns"],
    ),
    ("ssh", &["terminal", "shell", "remote", "connect"]),
    (
        "git",
        &["commit", "push", "pull", "repo", "repository", "branch"],
    ),
    ("log", &["output", "stdout", "stderr", "trace", "debug"]),
    ("cron", &["schedule", "job", "task", "timer", "periodic"]),
    ("email", &["mail", "message", "inbox", "smtp"]),
    ("image", &["picture", "photo", "screenshot", "graphic"]),
    ("browser", &["web", "page", "site", "navigate", "chrome"]),
    ("monitor", &["watch", "observe", "track", "survey"]),
    ("alert", &["notify", "notification", "warning", "ping"]),
    ("migrate", &["transfer", "move", "relocate", "port"]),
    ("compare", &["diff", "versus", "vs", "contrast"]),
    ("save", &["store", "persist", "preserve", "keep"]),
];

/// Stop words dropped during normalization (`synonyms.py` `STOP_WORDS` L59-L74).
const STOP_WORDS: &[&str] = &[
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "can",
    "shall",
    "must",
    "i",
    "you",
    "he",
    "she",
    "it",
    "we",
    "they",
    "me",
    "him",
    "her",
    "us",
    "them",
    "my",
    "your",
    "his",
    "its",
    "our",
    "their",
    "mine",
    "yours",
    "hers",
    "ours",
    "theirs",
    "what",
    "which",
    "who",
    "whom",
    "where",
    "when",
    "why",
    "how",
    "this",
    "that",
    "these",
    "those",
    "of",
    "in",
    "to",
    "for",
    "on",
    "with",
    "at",
    "by",
    "from",
    "as",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "between",
    "under",
    "and",
    "but",
    "or",
    "nor",
    "not",
    "so",
    "than",
    "too",
    "very",
    "just",
    "about",
    "also",
    "really",
    "actually",
    "basically",
    "simply",
    "if",
    "then",
    "else",
    "while",
    "because",
    "though",
    "although",
];

fn stop_words() -> &'static HashSet<&'static str> {
    static SET: OnceLock<HashSet<&'static str>> = OnceLock::new();
    SET.get_or_init(|| STOP_WORDS.iter().copied().collect())
}

/// `word -> canonical` reverse map (`synonyms.py` `_build_reverse_map` L77-L87): canonical maps to
/// itself, every synonym maps to its canonical.
fn word_to_canonical() -> &'static HashMap<&'static str, &'static str> {
    static MAP: OnceLock<HashMap<&'static str, &'static str>> = OnceLock::new();
    MAP.get_or_init(|| {
        let mut m = HashMap::new();
        for (canonical, syns) in SYNONYM_GROUPS {
            m.insert(*canonical, *canonical);
            for syn in *syns {
                m.insert(*syn, *canonical);
            }
        }
        m
    })
}

/// Normalize a query for exact-match caching (`synonyms.py` `normalize_query` L90-L113): lowercase,
/// drop stop words, map synonyms to canonical, then sort + dedup the words.
pub fn normalize_query(query: &str) -> String {
    let map = word_to_canonical();
    let lower = query.to_lowercase();
    let mut words: Vec<&str> = lower
        .split_whitespace()
        .filter(|w| !stop_words().contains(w))
        .map(|w| map.get(w).copied().unwrap_or(w))
        .collect();
    words.sort_unstable();
    words.dedup();
    words.join(" ")
}

/// Expand a query with synonym groups for broader FTS5 matching (`synonyms.py` `expand_query`
/// L116-L143). Each word that has a synonym group becomes `(word|canonical|syn1|...)`; stop words
/// and unknown words pass through unchanged.
pub fn expand_query(query: &str) -> String {
    let map = word_to_canonical();
    let groups = synonym_groups_map();
    let lower = query.to_lowercase();
    let mut parts: Vec<String> = Vec::new();
    for word in lower.split_whitespace() {
        if stop_words().contains(word) {
            parts.push(word.to_string());
            continue;
        }
        match map.get(word) {
            Some(canonical) if groups.contains_key(canonical) => {
                let mut group: Vec<&str> = vec![*canonical];
                group.extend_from_slice(groups[canonical]);
                if !group.contains(&word) {
                    group.insert(0, word);
                }
                parts.push(format!("({})", group.join("|")));
            }
            _ => parts.push(word.to_string()),
        }
    }
    parts.join(" ")
}

fn synonym_groups_map() -> &'static HashMap<&'static str, &'static [&'static str]> {
    static MAP: OnceLock<HashMap<&'static str, &'static [&'static str]>> = OnceLock::new();
    MAP.get_or_init(|| SYNONYM_GROUPS.iter().map(|(c, s)| (*c, *s)).collect())
}

/// All synonyms for a word, including its canonical form (`synonyms.py` `get_synonyms` L146-L152).
pub fn get_synonyms(word: &str) -> Vec<&'static str> {
    let word_lc = word.to_lowercase();
    let groups = synonym_groups_map();
    if let Some(canonical) = word_to_canonical().get(word_lc.as_str()) {
        if let Some(syns) = groups.get(canonical) {
            let mut out = vec![*canonical];
            out.extend_from_slice(syns);
            return out;
        }
    }
    Vec::new()
}

/// Beam's small, conservative recall-time synonym map (`beam.py` `_RECALL_SYNONYMS` L1477-L1489).
/// This is **distinct** from [`SYNONYM_GROUPS`]: it is one-directional (only the listed keys expand)
/// and tuned for the lexical `+0.75` partial in `Engine::lexical_relevance`.
#[allow(clippy::type_complexity)]
const RECALL_SYNONYMS: &[(&str, &[&str])] = &[
    ("branding", &["brand", "positioning", "identity", "wording"]),
    (
        "preference",
        &[
            "prefer", "prefers", "want", "wants", "reject", "rejects", "avoid", "grounded",
        ],
    ),
    ("professional", &["software", "builder"]),
    ("url", &["link", "profile"]),
    ("current", &["now", "live", "latest"]),
    ("feeling", &["feel", "feels"]),
    ("imposter", &["self-doubt", "doubt", "insecure"]),
];

/// The recall-time synonyms for a query token (`beam.py` `_RECALL_SYNONYMS.get(token, ())`), empty
/// when the token has no entry.
pub fn recall_synonyms(token: &str) -> &'static [&'static str] {
    static MAP: OnceLock<HashMap<&'static str, &'static [&'static str]>> = OnceLock::new();
    let map = MAP.get_or_init(|| RECALL_SYNONYMS.iter().map(|(k, v)| (*k, *v)).collect());
    map.get(token).copied().unwrap_or(&[])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_to_canonical_sorted() {
        // "db" -> "database"; stop words dropped; sorted + deduped.
        assert_eq!(
            normalize_query("what is the db password"),
            "database password"
        );
    }

    #[test]
    fn expands_synonym_groups() {
        let e = expand_query("database password");
        assert!(e.starts_with("(database|"));
        assert!(e.contains("db"));
        assert!(e.contains("(password|"));
    }

    #[test]
    fn unknown_words_pass_through() {
        assert_eq!(expand_query("frobnicate widget"), "frobnicate widget");
    }

    #[test]
    fn recall_synonyms_are_one_directional() {
        assert!(recall_synonyms("preference").contains(&"want"));
        // Not a key -> empty.
        assert!(recall_synonyms("want").is_empty());
    }
}
