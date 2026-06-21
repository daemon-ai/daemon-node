//! Message + session filter patterns (`daemon-context-lcm-port-spec.md` §12.3).
//!
//! Two filter families:
//! - **Message ignore** ([`MessagePatterns`]): linear-time `regex` patterns matched against a
//!   message's text; a matching turn is dropped before the store. Python guarded the optional
//!   `regex` package with a 50 ms per-match timeout; the Rust `regex` crate is linear-time, so the
//!   timeout machinery is unnecessary — compile once, drop invalid patterns with a warning (§14.7).
//! - **Session globs** ([`SessionGlobs`]): `*` → one colon segment (`[^:]*`), `**` → across colons
//!   (`.*`), anchored, matched against the keys `session_id`, `platform`, `platform:session_id`.
//!   Hand-rolled (not `globset`) because the colon semantics differ.

use regex::Regex;

/// Compiled message-ignore patterns (drop a matching message before it reaches the store).
#[derive(Debug, Default)]
pub struct MessagePatterns {
    patterns: Vec<(String, Regex)>,
}

impl MessagePatterns {
    /// Compile `patterns`, dropping any that fail to parse (with a warning). An empty input yields a
    /// matcher that never matches.
    pub fn compile(patterns: &[String]) -> Self {
        let mut compiled = Vec::new();
        for raw in patterns {
            match Regex::new(raw) {
                Ok(re) => compiled.push((raw.clone(), re)),
                Err(e) => {
                    tracing::warn!(pattern = %raw, error = %e, "lcm: dropping invalid message-ignore pattern");
                }
            }
        }
        Self { patterns: compiled }
    }

    /// Whether any pattern is configured.
    pub fn is_empty(&self) -> bool {
        self.patterns.is_empty()
    }

    /// The first pattern that matches `text` (the "match key" for diagnostics), or `None`.
    pub fn matched(&self, text: &str) -> Option<&str> {
        self.patterns
            .iter()
            .find(|(_, re)| re.is_match(text))
            .map(|(raw, _)| raw.as_str())
    }

    /// Whether `text` matches any ignore pattern.
    pub fn is_match(&self, text: &str) -> bool {
        self.matched(text).is_some()
    }
}

/// Compiled session globs (match a session against `session_ignored` / `session_stateless` rules).
#[derive(Debug, Default)]
pub struct SessionGlobs {
    globs: Vec<(String, Regex)>,
}

impl SessionGlobs {
    /// Compile session globs, dropping any that fail to parse (with a warning).
    pub fn compile(patterns: &[String]) -> Self {
        let mut globs = Vec::new();
        for raw in patterns {
            let pattern = glob_to_regex(raw);
            match Regex::new(&pattern) {
                Ok(re) => globs.push((raw.clone(), re)),
                Err(e) => {
                    tracing::warn!(glob = %raw, error = %e, "lcm: dropping invalid session glob");
                }
            }
        }
        Self { globs }
    }

    /// Whether any glob is configured.
    pub fn is_empty(&self) -> bool {
        self.globs.is_empty()
    }

    /// Whether any glob matches any of the session match `keys`.
    pub fn matches(&self, keys: &[String]) -> bool {
        self.globs
            .iter()
            .any(|(_, re)| keys.iter().any(|k| re.is_match(k)))
    }
}

/// Build the session match keys (`build_session_match_keys`, §12.3): always `session_id`, plus
/// `platform` and `platform:session_id` when a platform is known.
pub fn build_session_match_keys(platform: &str, session_id: &str) -> Vec<String> {
    let mut keys = vec![session_id.to_string()];
    if !platform.is_empty() {
        keys.push(platform.to_string());
        keys.push(format!("{platform}:{session_id}"));
    }
    keys
}

/// Translate a session glob to an anchored regex: `**` → `.*` (across colons), `*` → `[^:]*` (one
/// colon segment); every other run is matched literally.
fn glob_to_regex(glob: &str) -> String {
    let chars: Vec<char> = glob.chars().collect();
    let mut out = String::from("^");
    let mut literal = String::new();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '*' {
            if !literal.is_empty() {
                out.push_str(&regex::escape(&literal));
                literal.clear();
            }
            if i + 1 < chars.len() && chars[i + 1] == '*' {
                out.push_str(".*");
                i += 2;
            } else {
                out.push_str("[^:]*");
                i += 1;
            }
        } else {
            literal.push(chars[i]);
            i += 1;
        }
    }
    if !literal.is_empty() {
        out.push_str(&regex::escape(&literal));
    }
    out.push('$');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_patterns_drop_invalid_and_match() {
        let pats = MessagePatterns::compile(&[
            r"^/debug\b".to_string(),
            r"(unbalanced".to_string(), // invalid -> dropped
            r"(?i)heartbeat".to_string(),
        ]);
        assert_eq!(pats.matched("/debug now").unwrap(), r"^/debug\b");
        assert!(pats.is_match("HEARTBEAT ping"));
        assert!(!pats.is_match("a normal message"));
    }

    #[test]
    fn single_star_is_one_colon_segment() {
        let globs = SessionGlobs::compile(&["slack:*".to_string()]);
        let keys = build_session_match_keys("slack", "C123");
        assert!(globs.matches(&keys), "slack:* matches slack:C123");
        // A nested colon is NOT crossed by a single star.
        let nested = build_session_match_keys("slack", "team:C123");
        assert!(
            !globs.matches(&nested),
            "single-star stops at the colon: slack:team:C123 not matched"
        );
    }

    #[test]
    fn double_star_crosses_colons() {
        let globs = SessionGlobs::compile(&["slack:**".to_string()]);
        let nested = build_session_match_keys("slack", "team:C123");
        assert!(globs.matches(&nested), "slack:** crosses colons");
    }

    #[test]
    fn glob_matches_bare_session_id_key() {
        let globs = SessionGlobs::compile(&["scratch-*".to_string()]);
        let keys = build_session_match_keys("", "scratch-42");
        assert!(globs.matches(&keys));
        assert!(!globs.matches(&build_session_match_keys("", "prod-1")));
    }
}
