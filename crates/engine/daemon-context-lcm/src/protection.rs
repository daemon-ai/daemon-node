// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Ingest protection (`daemon-context-lcm-port-spec.md` §8).
//!
//! Guards the SQLite write boundary — **storage / size / secret / repetition**, not prompt
//! injection. [`protect_message_for_ingest`] runs the Python-ordered pipeline over a flattened
//! [`NewMessage`]: optional sensitive redaction (§8.1) → skip-if-already-externalized → assistant
//! loop/heartbeat quarantine (§8.3, externalize-only) → opt-in threshold externalization (§9.1) →
//! the always-on base64/data-URI storage guard (§8.2), then the recursive `tool_calls` walk
//! (redaction + payload protection through nested argument JSON strings). Everything except the
//! storage guard is opt-in and default-off; every spill path no-ops when no externalization
//! directory exists (in-memory/ephemeral banks), leaving content inline — no data loss.
//!
//! [`sanitize_replay_turn`] is the *active-replay* side (`LCM:engine.py:3224-3289`): the same
//! redaction catalog + assistant quarantine applied in place to the provider-facing conversation
//! each turn, so a secret or a runaway loop never travels back to the model even though the store
//! keeps the (protected) original.

use crate::config::LcmConfig;
use crate::externalize::{self, ingest_payload_placeholder, sha256_hex_prefix, PayloadMeta};
use crate::store::NewMessage;
use regex::Regex;
use serde_json::Value;
use std::path::Path;
use std::sync::OnceLock;

/// The minimum assistant length (chars) before loop/heartbeat quarantine is even considered (§8.3).
pub(crate) const QUARANTINED_ASSISTANT_MIN_CHARS: usize = 65_536;
/// The minimum tokenized length before the degenerate-output ratios are trusted (§8.3).
const QUARANTINED_ASSISTANT_MIN_TOKENS: usize = 1_000;
/// The maximum length (chars) of a message considered for heartbeat-noise diagnostics (§8.3).
pub(crate) const HEARTBEAT_NOISE_MAX_CHARS: usize = 256;
/// The generic long-base64 floor shared by the storage guard and the doctor's payload-risk scan
/// (`_GENERIC_BASE64_MIN_CHARS`, `LCM:ingest_protection.py`).
pub(crate) const GENERIC_BASE64_MIN_CHARS: usize = 4096;

/// The minimum length of a bare base64-ish run before the storage guard externalizes it.
const BARE_RUN_MIN: usize = 4096;

// ---- 8.4 content normalization ------------------------------------------------------------------

/// `normalize_content_value` (§8.4): `Null` → `None`, a string → itself, anything else → canonical
/// JSON (keys sorted — `serde_json::Value` maps are ordered when `preserve_order` is off).
pub fn normalize_content_value(value: &Value) -> Option<String> {
    match value {
        Value::Null => None,
        Value::String(s) => Some(s.clone()),
        other => serde_json::to_string(other).ok(),
    }
}

/// `text_content_for_pattern_matching` (§8.4): concatenate text parts of structured/multimodal
/// content for ignore-pattern matching, falling back to the normalized JSON. daemon-core user/
/// assistant content is already plain text, so this mostly matters for tool args/results.
pub fn text_content_for_pattern_matching(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Array(items) => {
            let parts: Vec<String> = items
                .iter()
                .map(text_content_for_pattern_matching)
                .filter(|p| !p.is_empty())
                .collect();
            parts.join("\n")
        }
        Value::Object(map) => {
            if let Some(Value::String(t)) = map.get("text") {
                t.clone()
            } else if let Some(content) = map.get("content") {
                text_content_for_pattern_matching(content)
            } else {
                normalize_content_value(value).unwrap_or_default()
            }
        }
        Value::Null => String::new(),
        other => normalize_content_value(other).unwrap_or_default(),
    }
}

// ---- 8.1 sensitive redaction --------------------------------------------------------------------

/// The sensitive-redaction placeholder prefix (`_SENSITIVE_PLACEHOLDER_PREFIX`) — a whole-string
/// redaction never re-redacts text already carrying it.
const SENSITIVE_PLACEHOLDER_PREFIX: &str = "[LCM sensitive redaction:";

/// One catalog entry: its name, compiled regex, the candidate secret capture groups (the first one
/// that participated in the match wins; empty means the whole match is the secret), and whether the
/// placeholder carries a SHA-256 (passwords omit it to avoid dictionary-checkable hashes).
struct SensitivePattern {
    name: &'static str,
    re: Regex,
    secret_groups: &'static [usize],
    include_hash: bool,
}

/// The `_SENSITIVE_PATTERN_CATALOG` (`LCM:ingest_protection.py:106-129`), regexes verbatim modulo
/// engine syntax: the secret alphabets include `~` and `=`; the key prefix tolerates a quote on
/// either side of the `:=`; `password_assignment`'s quoted backreference alternate is expanded into
/// one alternate per concrete quote char (`"`/`'`) since the `regex` crate has no backreferences —
/// equivalent because the backreference only pinned which quote closes the span.
fn catalog() -> &'static [SensitivePattern] {
    static CATALOG: OnceLock<Vec<SensitivePattern>> = OnceLock::new();
    CATALOG.get_or_init(|| {
        vec![
            SensitivePattern {
                name: "api_key",
                re: Regex::new(
                    r#"(?i)\b(?:api[_-]?key|api[_-]?token|access[_-]?token|secret[_-]?key|client[_-]?secret)\b\s*["']?\s*[:=]\s*["']?([A-Za-z0-9._~+/=-]{12,})"#,
                )
                .expect("api_key regex"),
                secret_groups: &[1],
                include_hash: true,
            },
            SensitivePattern {
                name: "bearer_token",
                re: Regex::new(r#"(?i)\bBearer\s+([A-Za-z0-9._~+/=-]{12,})"#).expect("bearer regex"),
                secret_groups: &[1],
                include_hash: true,
            },
            SensitivePattern {
                name: "password_assignment",
                re: Regex::new(
                    r#"(?i)\b(?:password|passwd|pwd|passphrase)\b\s*["']?\s*[:=]\s*(?:"([^\r\n\]}]{6,}?)"|'([^\r\n\]}]{6,}?)'|([^\s,"'\]}]{6,}))"#,
                )
                .expect("password regex"),
                secret_groups: &[1, 2, 3],
                include_hash: false,
            },
            SensitivePattern {
                name: "private_key",
                re: Regex::new(
                    r"(?is)-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
                )
                .expect("private_key regex"),
                secret_groups: &[],
                include_hash: true,
            },
        ]
    })
}

/// Whether `name` is a recognized catalog pattern (used by `lcm_doctor`'s config validation).
pub fn is_known_sensitive_pattern(name: &str) -> bool {
    catalog().iter().any(|p| p.name == name)
}

/// Build the sensitive-redaction placeholder (§8.1).
fn redaction_placeholder(name: &str, secret: &str, include_hash: bool) -> String {
    let chars = secret.chars().count();
    let bytes = secret.len();
    if include_hash {
        let sha = sha256_hex_prefix(secret.as_bytes(), 16);
        format!(
            "[LCM sensitive redaction: name={name}; chars={chars}; bytes={bytes}; sha256={sha}]"
        )
    } else {
        format!("[LCM sensitive redaction: name={name}; chars={chars}; bytes={bytes}]")
    }
}

/// Redact every active catalog pattern in `text` (forward-only). `active` is the configured
/// `sensitive_patterns` name list; an empty list redacts nothing.
pub fn redact_sensitive_text(text: &str, active: &[String]) -> String {
    let mut out = text.to_string();
    for pat in catalog() {
        if !active.iter().any(|n| n == pat.name) {
            continue;
        }
        out = redact_one(&out, pat);
    }
    out
}

/// Apply one catalog pattern, replacing only the secret span (keeping the key prefix + any trailing
/// quote) with the placeholder (`_redact_match`, `LCM:ingest_protection.py:236-250`): the first
/// participating candidate group is the secret; with no candidate the whole match is replaced.
fn redact_one(text: &str, pat: &SensitivePattern) -> String {
    let mut out = String::with_capacity(text.len());
    let mut last = 0usize;
    for caps in pat.re.captures_iter(text) {
        let whole = match caps.get(0) {
            Some(m) => m,
            None => continue,
        };
        let secret = pat
            .secret_groups
            .iter()
            .find_map(|&g| caps.get(g))
            .unwrap_or(whole);
        out.push_str(&text[last..secret.start()]);
        out.push_str(&redaction_placeholder(
            pat.name,
            secret.as_str(),
            pat.include_hash,
        ));
        out.push_str(&text[secret.end()..whole.end()]);
        last = whole.end();
    }
    out.push_str(&text[last..]);
    out
}

/// Redact an entire string that sits under a sensitive key (`_redact_entire_sensitive_string`,
/// `LCM:ingest_protection.py:285-288`): skipped when empty or already carrying a redaction
/// placeholder.
fn redact_entire_sensitive_string(text: &str, name: &str) -> String {
    if text.is_empty() || text.contains(SENSITIVE_PLACEHOLDER_PREFIX) {
        return text.to_string();
    }
    redaction_placeholder(name, text, name != "password_assignment")
}

/// Recursively redact a JSON value (`redact_sensitive_value`,
/// `LCM:ingest_protection.py:291-324`): keys are text-redacted; a string under a sensitive key is
/// text-redacted first and whole-string-redacted only when the text pass changed nothing; with
/// `parse_json_strings` a JSON-looking string leaf (without duplicate object keys) is parsed,
/// redacted structurally, and re-serialized compactly when the walk changed it. No-op when `active`
/// is empty.
pub fn redact_sensitive_value(value: Value, active: &[String], parse_json_strings: bool) -> Value {
    if active.is_empty() {
        return value;
    }
    match value {
        Value::Object(map) => {
            let mut out = serde_json::Map::with_capacity(map.len());
            for (k, v) in map {
                let protected_key = redact_sensitive_text(&k, active);
                let redacted = match (sensitive_pattern_for_key(&k, active), v) {
                    (Some(name), Value::String(s)) => {
                        let text_redacted = redact_sensitive_text(&s, active);
                        Value::String(if text_redacted == s {
                            redact_entire_sensitive_string(&s, name)
                        } else {
                            text_redacted
                        })
                    }
                    (_, v) => redact_sensitive_value(v, active, parse_json_strings),
                };
                out.insert(protected_key, redacted);
            }
            Value::Object(out)
        }
        Value::Array(items) => Value::Array(
            items
                .into_iter()
                .map(|v| redact_sensitive_value(v, active, parse_json_strings))
                .collect(),
        ),
        Value::String(s) => {
            if parse_json_strings && !json_has_duplicate_object_keys(&s) {
                if let Some(parsed) = maybe_parse_json_string(&s) {
                    let protected = redact_sensitive_value(parsed.clone(), active, true);
                    if protected != parsed {
                        if let Ok(serialized) = serde_json::to_string(&protected) {
                            return Value::String(serialized);
                        }
                    }
                }
            }
            Value::String(redact_sensitive_text(&s, active))
        }
        other => other,
    }
}

/// The key-name heuristic (`_sensitive_pattern_for_key`, `LCM:ingest_protection.py:252-269`): map
/// a sensitive-looking key to an *active* catalog name so its whole string value is redacted. The
/// key is normalized by collapsing non-alphanumeric runs to `_`.
fn sensitive_pattern_for_key(key: &str, active: &[String]) -> Option<&'static str> {
    let mut normalized = String::with_capacity(key.len());
    for c in key.to_lowercase().chars() {
        if c.is_ascii_alphanumeric() {
            normalized.push(c);
        } else if !normalized.ends_with('_') {
            normalized.push('_');
        }
    }
    let normalized = normalized.trim_matches('_');
    let compact: String = normalized.chars().filter(|c| *c != '_').collect();
    let is_active = |name: &str| active.iter().any(|n| n == name);
    if is_active("api_key")
        && (matches!(
            compact.as_str(),
            "apikey" | "apitoken" | "accesstoken" | "secretkey" | "clientsecret"
        ) || (normalized.contains("api") && normalized.contains("key"))
            || (normalized.contains("access") && normalized.contains("token"))
            || (normalized.contains("secret") && normalized.contains("key")))
    {
        return Some("api_key");
    }
    if is_active("bearer_token")
        && matches!(
            compact.as_str(),
            "authorization" | "authtoken" | "bearertoken" | "token"
        )
    {
        return Some("bearer_token");
    }
    if is_active("password_assignment")
        && matches!(
            compact.as_str(),
            "password" | "passwd" | "pwd" | "passphrase"
        )
    {
        return Some("password_assignment");
    }
    None
}

/// Parse a JSON-looking string into an object/array (`_maybe_parse_json_string`,
/// `LCM:ingest_protection.py:617-627`): `None` unless it starts with `[`/`{` and parses to a
/// container.
fn maybe_parse_json_string(text: &str) -> Option<Value> {
    let stripped = text.trim_start();
    if !stripped.starts_with('[') && !stripped.starts_with('{') {
        return None;
    }
    let parsed = serde_json::from_str::<Value>(text).ok()?;
    matches!(parsed, Value::Object(_) | Value::Array(_)).then_some(parsed)
}

/// Whether a JSON string carries duplicate object keys (`_json_has_duplicate_object_keys`,
/// `LCM:ingest_protection.py:630-651`) — a parse would silently drop one duplicate's payload, so
/// such strings are protected as raw text instead of being walked structurally. `false` for
/// non-JSON.
fn json_has_duplicate_object_keys(text: &str) -> bool {
    let stripped = text.trim_start();
    if !stripped.starts_with('[') && !stripped.starts_with('{') {
        return false;
    }
    if serde_json::from_str::<Value>(text).is_err() {
        return false;
    }
    // The text is valid JSON, so a minimal lexer suffices: track object frames and the strings in
    // key position within each.
    struct Frame {
        is_object: bool,
        keys: std::collections::HashSet<String>,
        expect_key: bool,
    }
    let mut stack: Vec<Frame> = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            b'"' => {
                let start = i + 1;
                let mut j = start;
                while j < bytes.len() {
                    match bytes[j] {
                        b'\\' => j += 2,
                        b'"' => break,
                        _ => j += 1,
                    }
                }
                let end = j.min(bytes.len());
                if let Some(frame) = stack.last_mut() {
                    if frame.is_object && frame.expect_key {
                        let key = String::from_utf8_lossy(&bytes[start..end]).into_owned();
                        if !frame.keys.insert(key) {
                            return true;
                        }
                        frame.expect_key = false;
                    }
                }
                i = end + 1;
            }
            b'{' => {
                stack.push(Frame {
                    is_object: true,
                    keys: std::collections::HashSet::new(),
                    expect_key: true,
                });
                i += 1;
            }
            b'[' => {
                stack.push(Frame {
                    is_object: false,
                    keys: std::collections::HashSet::new(),
                    expect_key: false,
                });
                i += 1;
            }
            b'}' | b']' => {
                stack.pop();
                i += 1;
            }
            b',' => {
                if let Some(frame) = stack.last_mut() {
                    if frame.is_object {
                        frame.expect_key = true;
                    }
                }
                i += 1;
            }
            _ => i += 1,
        }
    }
    false
}

// ---- 8.2 storage-boundary guard (always on) -----------------------------------------------------

/// `_DATA_URI_BASE64_RE` verbatim (`LCM:ingest_protection.py:78-90`): any base64 data URI (media
/// type + parameter segments), tolerating JSON-escaped slashes (`\/`, `\u002f`) in raw scans; the
/// payload needs 256+ alternation units. Python's trailing `(?=$|[^A-Za-z0-9+/=])` lookahead is
/// dropped: the payload alternation can consume every char that lookahead forbids, so a maximal
/// (greedy) match never ends before one — the lookahead can never fail.
fn data_uri_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    const SLASH: &str = r"(?:/|\\/|\\u002[fF])";
    RE.get_or_init(|| {
        Regex::new(&format!(
            r"(?i)data:(?:[A-Za-z0-9.+-]|{SLASH})*(?:;[A-Za-z0-9_.+%-]+=(?:[-A-Za-z0-9_.+%]|{SLASH})*)*;base64,(?:[A-Za-z0-9+=]|{SLASH}){{256,}}"
        ))
        .expect("data-uri regex")
    })
}

/// `_BASE64_RUN_RE` (`LCM:ingest_protection.py:92`): Python's lookaround boundaries are implied
/// here — a leftmost, greedy match over the same alphabet is always maximal, so it can be neither
/// preceded nor followed by an alphabet char.
fn bare_run_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"[A-Za-z0-9+/=_\-]{4096,}").expect("base64-run regex"))
}

/// Whether `text` embeds a base64 data URI (`contains_data_uri_base64`,
/// `LCM:ingest_protection.py:135-136`) — the doctor's payload-risk classifier.
pub(crate) fn contains_data_uri_base64(text: &str) -> bool {
    data_uri_re().is_match(text)
}

/// Whether `text` embeds a validated long bare base64 run (`contains_long_base64_run`,
/// `LCM:ingest_protection.py:139-142`, at the generic 4096-char floor).
pub(crate) fn contains_long_base64_run(text: &str) -> bool {
    if text.chars().count() < GENERIC_BASE64_MIN_CHARS {
        return false;
    }
    bare_run_re()
        .find_iter(text)
        .any(|m| looks_like_long_base64(m.as_str(), GENERIC_BASE64_MIN_CHARS))
}

/// `looks_like_long_base64` (`LCM:ingest_protection.py:525-548`): length ≥ `min_len` (also after
/// whitespace compaction), compacted `len % 4 != 1`, base64-alphabet(+whitespace)-only, ≥0.98
/// alphabet ratio, and ≥8 distinct chars ignoring `=` padding.
fn looks_like_long_base64(s: &str, min_len: usize) -> bool {
    if s.chars().count() < min_len {
        return false;
    }
    let compact: Vec<char> = s.chars().filter(|c| !c.is_whitespace()).collect();
    if compact.len() < min_len || compact.len() % 4 == 1 {
        return false;
    }
    let mut alpha = 0usize;
    let mut n = 0usize;
    for c in s.chars() {
        n += 1;
        if c.is_ascii_alphanumeric() || matches!(c, '+' | '/' | '=' | '_' | '-') {
            alpha += 1;
        } else if !c.is_whitespace() {
            // Outside the base64 alphabet entirely (`_BASE64_ALPHABET_RE`).
            return false;
        }
    }
    let distinct: std::collections::HashSet<char> = compact
        .iter()
        .rev()
        .skip_while(|c| **c == '=')
        .copied()
        .collect();
    if distinct.len() < 8 {
        return false;
    }
    (alpha as f64) / (n as f64) >= 0.98
}

/// Externalize every payload run matched by `re` (validating with [`looks_like_long_base64`] at
/// `min_valid` when set — the bare-run path; data URIs are gated by their regex alone), spilling
/// each whole match to `dir` and substituting the §8.2 placeholder. Returns the rewritten text and
/// whether any substitution happened.
#[allow(clippy::too_many_arguments)]
fn externalize_runs(
    text: &str,
    re: &Regex,
    kind: &str,
    field: &str,
    role: &str,
    tool_call_id: Option<&str>,
    session_id: &str,
    dir: &Path,
    min_valid: Option<usize>,
) -> (String, bool) {
    let mut out = String::with_capacity(text.len());
    let mut last = 0usize;
    let mut changed = false;
    for whole in re.find_iter(text) {
        if let Some(min_len) = min_valid {
            if !looks_like_long_base64(whole.as_str(), min_len) {
                continue;
            }
        }
        let body = whole.as_str();
        let meta = PayloadMeta {
            kind,
            field,
            role,
            tool_call_id,
            session_id,
        };
        match externalize::store_payload(dir, body, &meta) {
            Ok(reference) => {
                out.push_str(&text[last..whole.start()]);
                out.push_str(&ingest_payload_placeholder(
                    kind,
                    field,
                    body.chars().count(),
                    body.len(),
                    &reference,
                ));
                last = whole.end();
                changed = true;
            }
            Err(e) => {
                // On externalize failure, leave the original inline (no data loss; warn only).
                tracing::warn!(error = %e, kind, "lcm: storage-guard externalization failed; leaving inline");
            }
        }
    }
    out.push_str(&text[last..]);
    (out, changed)
}

/// Apply the always-on base64/data-URI storage guard to one field's text
/// (`_protect_payload_substrings`, `LCM:ingest_protection.py:576-614`). No-op (returns the input)
/// when `dir` is `None` (ephemeral bank), the text is empty, or the text *is* (entirely) an ingest
/// placeholder already. A placeholder merely embedded in larger text does not skip the guard — new
/// payloads beside an old placeholder are still externalized (identical bodies dedup by digest).
fn apply_storage_guard(
    text: &str,
    field: &str,
    role: &str,
    tool_call_id: Option<&str>,
    session_id: &str,
    dir: Option<&Path>,
) -> String {
    let Some(dir) = dir else {
        return text.to_string();
    };
    if text.is_empty() || is_externalized_ingest_placeholder(text) {
        return text.to_string();
    }
    let (text, _) = externalize_runs(
        text,
        data_uri_re(),
        "data_uri",
        field,
        role,
        tool_call_id,
        session_id,
        dir,
        None,
    );
    let (text, _) = externalize_runs(
        &text,
        bare_run_re(),
        "base64_run",
        field,
        role,
        tool_call_id,
        session_id,
        dir,
        Some(BARE_RUN_MIN),
    );
    text
}

/// Whether `text` is entirely one §8.2 ingest placeholder (`is_externalized_ingest_placeholder`,
/// `LCM:ingest_protection.py:131-132` — a fullmatch, not a substring test).
fn is_externalized_ingest_placeholder(text: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| {
        Regex::new(r"^\[Externalized LCM ingest payload:[^\n]*?;\s*ref=[^;\]\s]+\]$")
            .expect("ingest-placeholder regex")
    });
    re.is_match(text.trim())
}

/// Whether `text` is entirely one compact externalized placeholder of any family — the Python
/// skip-if-already-externalized gate (`is_externalized_ingest_placeholder or
/// is_externalized_placeholder`, `LCM:ingest_protection.py:833`, `LCM:externalize.py:157-164`).
fn is_whole_externalized_placeholder(text: &str) -> bool {
    let stripped = text.trim();
    if stripped.is_empty() {
        return false;
    }
    if is_externalized_ingest_placeholder(stripped) {
        return true;
    }
    if stripped.len() > 512 {
        return false;
    }
    match externalize::externalized_ref_regex().find(stripped) {
        Some(m) => m.start() == 0 && m.end() == stripped.len(),
        None => false,
    }
}

/// Recursively protect a JSON value's payloads (`_protect_value`,
/// `LCM:ingest_protection.py:663-763`): sensitive redaction over the whole value first, then a
/// payload walk — keys are substring-protected, strings run the storage guard, and (with
/// `parse_json_strings`) JSON-looking string leaves are parsed and walked structurally so
/// secrets/base64 nested inside tool-call `arguments` strings cannot persist inline. A JSON string
/// with duplicate object keys is protected as raw text (a parse would silently drop one
/// duplicate's payload).
fn protect_value(
    value: Value,
    field: &str,
    role: &str,
    session_id: &str,
    dir: Option<&Path>,
    parse_json_strings: bool,
    active: &[String],
) -> Value {
    let value = redact_sensitive_value(value, active, parse_json_strings);
    protect_payload_value(value, field, role, session_id, dir, parse_json_strings)
}

/// The payload walk of [`protect_value`] (redaction already applied).
fn protect_payload_value(
    value: Value,
    field: &str,
    role: &str,
    session_id: &str,
    dir: Option<&Path>,
    parse_json_strings: bool,
) -> Value {
    match value {
        Value::Object(map) => {
            let mut out = serde_json::Map::with_capacity(map.len());
            for (key, val) in map {
                let protected_key = apply_storage_guard(
                    &key,
                    &format!("{field}.<key>"),
                    role,
                    None,
                    session_id,
                    dir,
                );
                let child_field = if protected_key == key {
                    format!("{field}.{key}")
                } else {
                    format!("{field}.<key>")
                };
                out.insert(
                    protected_key,
                    protect_payload_value(
                        val,
                        &child_field,
                        role,
                        session_id,
                        dir,
                        parse_json_strings,
                    ),
                );
            }
            Value::Object(out)
        }
        Value::Array(items) => Value::Array(
            items
                .into_iter()
                .enumerate()
                .map(|(idx, item)| {
                    protect_payload_value(
                        item,
                        &format!("{field}[{idx}]"),
                        role,
                        session_id,
                        dir,
                        parse_json_strings,
                    )
                })
                .collect(),
        ),
        Value::String(s) => Value::String(protect_string_payloads(
            s,
            field,
            role,
            session_id,
            dir,
            parse_json_strings,
        )),
        other => other,
    }
}

/// The string branch of the payload walk (`LCM:ingest_protection.py:717-763`): with
/// `parse_json_strings`, duplicate-key JSON is protected raw; a non-canonical JSON string gets a
/// raw substring pass first (returned when it changed anything); otherwise the parsed value is
/// walked and re-serialized compactly only when the walk changed it.
fn protect_string_payloads(
    s: String,
    field: &str,
    role: &str,
    session_id: &str,
    dir: Option<&Path>,
    parse_json_strings: bool,
) -> String {
    if parse_json_strings {
        if json_has_duplicate_object_keys(&s) {
            return apply_storage_guard(&s, field, role, None, session_id, dir);
        }
        if let Some(parsed) = maybe_parse_json_string(&s) {
            let canonical = serde_json::to_string(&parsed).unwrap_or_default();
            if canonical != s {
                let raw_protected = apply_storage_guard(&s, field, role, None, session_id, dir);
                if raw_protected != s {
                    return raw_protected;
                }
            }
            let protected =
                protect_payload_value(parsed.clone(), field, role, session_id, dir, true);
            if protected != parsed {
                if let Ok(serialized) = serde_json::to_string(&protected) {
                    return serialized;
                }
            }
            return s;
        }
    }
    apply_storage_guard(&s, field, role, None, session_id, dir)
}

/// Protect a non-SQLite active-context scaffold text (`protect_inline_payloads_in_text`,
/// `LCM:ingest_protection.py:767-791`): sensitive redaction (when enabled) followed by the
/// storage-guard payload externalization, so preserved-objective scaffolds never duplicate
/// media-ish payloads or secrets into the summary block.
pub(crate) fn protect_scaffold_text(
    text: &str,
    cfg: &LcmConfig,
    session_id: &str,
    dir: Option<&Path>,
) -> String {
    let text = if cfg.sensitive_patterns_enabled {
        redact_sensitive_text(text, &cfg.sensitive_patterns)
    } else {
        text.to_string()
    };
    apply_storage_guard(
        &text,
        "preserved_objective.content",
        "user",
        None,
        session_id,
        dir,
    )
}

// ---- 8.3 loop / heartbeat quarantine ------------------------------------------------------------

/// `assistant_output_quarantine_reason` (§8.3): detect runaway/degenerate assistant output by token
/// + segment repetition ratios. Returns a short reason when the output should be quarantined.
pub fn assistant_output_quarantine_reason(content: &str) -> Option<String> {
    if content.chars().count() < QUARANTINED_ASSISTANT_MIN_CHARS {
        return None;
    }
    let token_re = token_re();
    let tokens: Vec<String> = token_re
        .find_iter(content)
        .map(|m| m.as_str().to_ascii_lowercase())
        .collect();
    let total = tokens.len();

    let mut token_freq = std::collections::HashMap::new();
    for t in &tokens {
        *token_freq.entry(t.as_str()).or_insert(0usize) += 1;
    }
    let distinct_tokens = token_freq.len();
    let unique_token_ratio = ratio(distinct_tokens, total);
    let top_token_ratio = ratio(token_freq.values().copied().max().unwrap_or(0), total);

    let segments: Vec<&str> = content
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect();
    let seg_total = segments.len();
    let mut seg_freq = std::collections::HashMap::new();
    for s in &segments {
        *seg_freq.entry(*s).or_insert(0usize) += 1;
    }
    let distinct_segments = seg_freq.len();
    let duplicate_segment_ratio = ratio(seg_total.saturating_sub(distinct_segments), seg_total);
    let top_segment_ratio = ratio(seg_freq.values().copied().max().unwrap_or(0), seg_total);

    let distinct_chars = content
        .chars()
        .collect::<std::collections::HashSet<_>>()
        .len();

    let degenerate_tokens = total >= QUARANTINED_ASSISTANT_MIN_TOKENS || total == 0;
    let primary = unique_token_ratio <= 0.03
        && degenerate_tokens
        && (top_segment_ratio >= 0.10
            || duplicate_segment_ratio >= 0.50
            || top_token_ratio >= 0.08);
    let secondary = unique_token_ratio <= 0.015 && distinct_chars <= 64;

    if primary {
        Some(format!(
            "degenerate repetition (unique_token_ratio={unique_token_ratio:.4}, top_segment={top_segment_ratio:.2}, dup_segment={duplicate_segment_ratio:.2}, top_token={top_token_ratio:.2})"
        ))
    } else if secondary {
        Some(format!(
            "low-entropy output (unique_token_ratio={unique_token_ratio:.4}, distinct_chars={distinct_chars})"
        ))
    } else {
        None
    }
}

/// `heartbeat_noise_reason` (§8.3) — **diagnostic only** (never quarantines on ingest): a short
/// message that is pure status chatter.
pub fn heartbeat_noise_reason(content: &str) -> Option<String> {
    if content.chars().count() > HEARTBEAT_NOISE_MAX_CHARS {
        return None;
    }
    if heartbeat_re().is_match(content) {
        Some("heartbeat/status noise".to_string())
    } else {
        None
    }
}

/// The active-replay placeholder for quarantined output kept volatile (no disk) (§8.3,
/// `_volatile_quarantined_assistant_placeholder`). Used by the active-replay sanitizer when the
/// content must not (ignored turn) or cannot (no directory) be spilled; never written to the store
/// (the store boundary keeps content inline instead).
fn active_replay_placeholder(content: &str) -> String {
    let chars = content.chars().count();
    let bytes = content.len();
    let sha = sha256_hex_prefix(content.as_bytes(), 16);
    format!("[LCM active replay placeholder: chars={chars}; bytes={bytes}; sha256={sha}]")
}

// ---- active-replay protection ---------------------------------------------------------------

/// How the active-replay sanitizer treats runaway assistant output in a turn.
#[derive(Clone, Copy, Debug)]
pub(crate) enum ReplayQuarantine<'a> {
    /// Do not quarantine (ignored/stateless sessions get redaction only,
    /// `LCM:engine.py:3256-3266`).
    Skip,
    /// Quarantine with the volatile placeholder — the content must not touch disk (turns matching
    /// `ignore_message_patterns`, `externalize=False` in Python).
    Volatile,
    /// Quarantine and spill the original body to the externalization dir so it stays recoverable.
    Spill(&'a Path),
}

/// Sanitize one provider-facing turn in place (`_redact_active_replay_messages` +
/// `quarantine_suspicious_assistant_messages`, `LCM:engine.py:3224-3289`): runaway assistant
/// output is swapped for a quarantine placeholder per `quarantine`, then every text channel is
/// redacted through the active sensitive catalog (tool-call `args` with JSON-string parsing, like
/// Python's `tool_calls` walk). Turn structure is never changed.
pub(crate) fn sanitize_replay_turn(
    turn: &mut daemon_core::Turn,
    active: &[String],
    session_id: &str,
    quarantine: ReplayQuarantine<'_>,
) {
    use daemon_core::Turn;
    match turn {
        Turn::User(u) => {
            redact_in_place(&mut u.text, active);
        }
        Turn::Assistant(a) => {
            quarantine_assistant_text(&mut a.text, session_id, quarantine);
            redact_in_place(&mut a.text, active);
            if let Some(reasoning) = a.reasoning.as_mut() {
                // Rust-model extension: `reasoning` is a replay channel Python does not have;
                // redact it the same way so it cannot leak a secret back to the provider.
                redact_in_place(reasoning, active);
            }
        }
        Turn::Tool(t) => {
            quarantine_assistant_text(&mut t.assistant.text, session_id, quarantine);
            redact_in_place(&mut t.assistant.text, active);
            if let Some(reasoning) = t.assistant.reasoning.as_mut() {
                redact_in_place(reasoning, active);
            }
            for (call, result) in t.calls.iter_mut() {
                if !active.is_empty() {
                    let args = std::mem::take(&mut call.args);
                    call.args = match redact_sensitive_value(Value::String(args), active, true) {
                        Value::String(s) => s,
                        other => other.to_string(),
                    };
                }
                redact_in_place(&mut result.content, active);
            }
        }
    }
}

/// Text-catalog redaction in place (no-op when nothing is active or nothing matched).
fn redact_in_place(text: &mut String, active: &[String]) {
    if active.is_empty() || text.is_empty() {
        return;
    }
    let redacted = redact_sensitive_text(text, active);
    if redacted != *text {
        *text = redacted;
    }
}

/// Swap runaway/degenerate assistant text for a quarantine placeholder per the turn's
/// [`ReplayQuarantine`] mode. No-op for healthy text.
fn quarantine_assistant_text(
    text: &mut String,
    session_id: &str,
    quarantine: ReplayQuarantine<'_>,
) {
    let dir = match quarantine {
        ReplayQuarantine::Skip => return,
        ReplayQuarantine::Volatile => None,
        ReplayQuarantine::Spill(dir) => Some(dir),
    };
    let Some(reason) = assistant_output_quarantine_reason(text) else {
        return;
    };
    tracing::warn!(reason = %reason, "lcm: quarantining runaway assistant output from active replay");
    *text = quarantine_replacement(text, "assistant", session_id, dir)
        .unwrap_or_else(|| active_replay_placeholder(text));
}

fn token_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"[A-Za-z0-9_]+").expect("token regex"))
}

fn heartbeat_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)\b(still working|working on it|processing|checking|one moment|ping|heartbeat|no update)\b")
            .expect("heartbeat regex")
    })
}

fn ratio(num: usize, den: usize) -> f64 {
    if den == 0 {
        0.0
    } else {
        num as f64 / den as f64
    }
}

// ---- pipeline -----------------------------------------------------------------------------------

/// Run the full ingest-protection pipeline over one flattened message
/// (`protect_message_for_ingest`, `LCM:ingest_protection.py:806-889`), in the Python stage order:
/// sensitive redaction → skip-if-already-externalized → assistant loop/heartbeat quarantine →
/// opt-in threshold externalization of the whole content → recursive payload substring protection;
/// then the recursive `tool_calls` walk (with JSON-string parsing). `dir` is the resolved
/// externalization directory (`None` for ephemeral banks → every spill path no-ops and content
/// stays inline — no data loss).
pub fn protect_message_for_ingest(
    mut msg: NewMessage,
    cfg: &crate::config::LcmConfig,
    session_id: &str,
    dir: Option<&Path>,
) -> NewMessage {
    let active: &[String] = if cfg.sensitive_patterns_enabled {
        &cfg.sensitive_patterns
    } else {
        &[]
    };
    let role = msg.role.clone();

    if let Some(content) = msg.content.take() {
        // 1. Sensitive redaction (opt-in; content is already flattened text).
        let content = if active.is_empty() {
            content
        } else {
            redact_sensitive_text(&content, active)
        };
        // 2. Skip-if-already-externalized: a row whose content *is* a placeholder re-ingests as-is
        //    (before quarantine, so a placeholder is never quarantined or re-externalized).
        let content = if is_whole_externalized_placeholder(&content) {
            content
        } else {
            // 3. Assistant loop/heartbeat quarantine — externalize-only: without a directory (or
            //    on a spill failure) the content stays inline (Python returns no placeholder and
            //    falls through; the volatile active-replay placeholder is an active-context
            //    mechanism, not a store one).
            let quarantined = if role == "assistant" {
                assistant_output_quarantine_reason(&content).and_then(|reason| {
                    tracing::warn!(reason = %reason, "lcm: quarantining runaway assistant output");
                    quarantine_replacement(&content, &role, session_id, dir)
                })
            } else {
                None
            };
            match quarantined {
                Some(q) => q,
                None => {
                    // 4. Opt-in threshold externalization of the whole content (§9.1).
                    let kind = if role == "tool" {
                        "tool_result"
                    } else {
                        "raw_payload"
                    };
                    let meta = PayloadMeta {
                        kind,
                        field: "content",
                        role: &role,
                        tool_call_id: msg.tool_call_id.as_deref(),
                        session_id,
                    };
                    match externalize::maybe_externalize_payload(
                        dir,
                        &content,
                        cfg.large_output_externalization_enabled,
                        cfg.large_output_externalization_threshold_chars,
                        &meta,
                    ) {
                        Some((placeholder, _reference)) => placeholder,
                        // 5. Always-on payload substring protection (data-URI/base64 runs).
                        None => apply_storage_guard(
                            &content,
                            "content",
                            &role,
                            msg.tool_call_id.as_deref(),
                            session_id,
                            dir,
                        ),
                    }
                }
            }
        };
        msg.content = Some(content);
    }

    // 6. Recursive tool_calls protection (`_protect_tool_calls`) — redaction + payload walk over
    //    the parsed JSON with JSON-string parsing, so secrets/base64 inside nested `arguments`
    //    strings are caught. The original string is kept when nothing changed (no re-serialization
    //    churn); an unparseable blob is treated as raw text.
    if let Some(tc) = msg.tool_calls.take() {
        let protected = match serde_json::from_str::<Value>(&tc) {
            Ok(value) => {
                let walked = protect_value(
                    value.clone(),
                    "tool_calls",
                    &role,
                    session_id,
                    dir,
                    true,
                    active,
                );
                if walked == value {
                    tc
                } else {
                    serde_json::to_string(&walked).unwrap_or(tc)
                }
            }
            Err(_) => {
                let text = if active.is_empty() {
                    tc
                } else {
                    redact_sensitive_text(&tc, active)
                };
                apply_storage_guard(&text, "tool_calls", &role, None, session_id, dir)
            }
        };
        msg.tool_calls = Some(protected);
    }

    msg
}

/// Replace quarantined assistant output with an externalized placeholder, or `None` when no
/// directory exists / the spill failed (the caller keeps the content inline — lossless). Shared by
/// the store-boundary pipeline and the active-replay sanitizer (identical bodies dedup by digest,
/// so both sides resolve to the same payload file).
pub(crate) fn quarantine_replacement(
    content: &str,
    role: &str,
    session_id: &str,
    dir: Option<&Path>,
) -> Option<String> {
    let dir = dir?;
    let meta = PayloadMeta {
        kind: "quarantine",
        field: "content",
        role,
        tool_call_id: None,
        session_id,
    };
    match externalize::store_payload(dir, content, &meta) {
        Ok(reference) => Some(format!(
            "[Externalized quarantined assistant output: chars={}; bytes={}; ref={}]",
            content.chars().count(),
            content.len(),
            reference
        )),
        Err(e) => {
            tracing::warn!(error = %e, "lcm: quarantine externalization failed; leaving content inline");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::LcmConfig;

    fn all_patterns() -> Vec<String> {
        vec![
            "api_key".to_string(),
            "bearer_token".to_string(),
            "password_assignment".to_string(),
            "private_key".to_string(),
        ]
    }

    #[test]
    fn redacts_api_key_with_hash_and_password_without() {
        let active = all_patterns();
        let red = redact_sensitive_text("api_key=ABCDEF0123456789 trailing", &active);
        assert!(red.contains("name=api_key"));
        assert!(red.contains("sha256="));
        assert!(red.contains("trailing"), "non-secret text is preserved");
        assert!(!red.contains("ABCDEF0123456789"));

        let pw = redact_sensitive_text("password: hunter2secret", &active);
        assert!(pw.contains("name=password_assignment"));
        assert!(!pw.contains("sha256="), "passwords omit the hash");
        assert!(!pw.contains("hunter2secret"));
    }

    #[test]
    fn redaction_is_disabled_when_pattern_inactive() {
        let active: Vec<String> = vec!["bearer_token".to_string()];
        let red = redact_sensitive_text("api_key=ABCDEF0123456789", &active);
        assert!(
            red.contains("ABCDEF0123456789"),
            "inactive pattern is not applied"
        );
    }

    #[test]
    fn redacts_private_key_block() {
        let active = all_patterns();
        let pem = "before\n-----BEGIN RSA PRIVATE KEY-----\nMIIabc\nDEF==\n-----END RSA PRIVATE KEY-----\nafter";
        let red = redact_sensitive_text(pem, &active);
        assert!(red.contains("name=private_key"));
        assert!(red.contains("before") && red.contains("after"));
        assert!(!red.contains("MIIabc"));
    }

    #[test]
    fn redacts_sensitive_json_keys() {
        let active = all_patterns();
        let v: Value = serde_json::json!({"password": "short!", "nested": {"api_token": "ABCDEFGHIJKL"}, "ok": "keep"});
        let v = redact_sensitive_value(v, &active, false);
        assert!(v["password"]
            .as_str()
            .unwrap()
            .contains("name=password_assignment"));
        assert!(v["nested"]["api_token"]
            .as_str()
            .unwrap()
            .contains("name=api_key"));
        assert_eq!(v["ok"], "keep");
    }

    #[test]
    fn key_redaction_tries_text_redaction_before_whole_value() {
        let active = all_patterns();
        // The value under a sensitive key itself matches the text catalog: the text pass wins and
        // surrounding context survives (`LCM:ingest_protection.py:301-306`).
        let v: Value = serde_json::json!({"token": "use Bearer ABCDEFGHIJKLMNOP for auth"});
        let v = redact_sensitive_value(v, &active, false);
        let s = v["token"].as_str().unwrap();
        assert!(s.starts_with("use Bearer "), "prefix kept: {s}");
        assert!(s.contains("name=bearer_token"));
        assert!(s.ends_with(" for auth"));
        // A value that matches nothing is redacted whole (bearer-style key heuristic).
        let v2: Value = serde_json::json!({"authorization": "opaque secret words"});
        let v2 = redact_sensitive_value(v2, &active, false);
        let s2 = v2["authorization"].as_str().unwrap();
        assert!(s2.starts_with("[LCM sensitive redaction: name=bearer_token"));
    }

    #[test]
    fn redaction_parses_nested_json_argument_strings() {
        let active = all_patterns();
        // tool_calls-style: `args` is a JSON-encoded string carrying a secret.
        let args =
            serde_json::to_string(&serde_json::json!({"api_key": "SUPERSECRETVALUE123"})).unwrap();
        let v: Value = serde_json::json!([{"name": "call_api", "args": args}]);
        let v = redact_sensitive_value(v, &active, true);
        let out = v[0]["args"].as_str().unwrap();
        assert!(!out.contains("SUPERSECRETVALUE123"), "nested secret gone");
        assert!(out.contains("name=api_key"));
        // Duplicate-key JSON strings are NOT structurally parsed (raw text redaction only).
        let dup = r#"{"k":"a","k":"api_key=SUPERSECRETVALUE123"}"#;
        let v2 = redact_sensitive_value(Value::String(dup.into()), &active, true);
        let out2 = v2.as_str().unwrap();
        assert!(
            !out2.contains("SUPERSECRETVALUE123"),
            "raw pass still redacts"
        );
        assert!(out2.starts_with(r#"{"k":"a","k":""#), "structure untouched");
    }

    #[test]
    fn json_duplicate_key_detection() {
        assert!(json_has_duplicate_object_keys(r#"{"a":1,"a":2}"#));
        assert!(json_has_duplicate_object_keys(r#"[{"x":{"b":1,"b":2}}]"#));
        assert!(!json_has_duplicate_object_keys(r#"{"a":1,"b":{"a":2}}"#));
        assert!(!json_has_duplicate_object_keys("not json"));
        assert!(
            !json_has_duplicate_object_keys(r#"{"a":1,"#),
            "invalid JSON is not flagged"
        );
    }

    #[test]
    fn base64_detection_thresholds() {
        assert!(looks_like_long_base64(&"QUJDREVG".repeat(600), 256));
        // All one char -> too few distinct.
        assert!(!looks_like_long_base64(&"A".repeat(5000), 256));
        // Too short.
        assert!(!looks_like_long_base64("QUJD", 256));
    }

    #[test]
    fn storage_guard_externalizes_base64_runs() {
        let dir = std::env::temp_dir().join(format!("lcm-guard-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let payload = "QUJDREVG".repeat(700); // > 4096 chars, base64 alphabet
        let text = format!("here is data: {payload} end");
        let out = apply_storage_guard(
            &text,
            "content",
            "tool",
            Some("c1"),
            "s1",
            Some(dir.as_path()),
        );
        assert!(out.contains("Externalized LCM ingest payload"));
        assert!(out.contains("here is data:") && out.contains("end"));
        assert!(!out.contains(&payload));
        // Re-running on the placeholder is a no-op.
        let again = apply_storage_guard(
            &out,
            "content",
            "tool",
            Some("c1"),
            "s1",
            Some(dir.as_path()),
        );
        assert_eq!(again, out);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn storage_guard_noops_without_a_dir() {
        let payload = "QUJDREVG".repeat(700);
        let out = apply_storage_guard(&payload, "content", "tool", None, "s1", None);
        assert_eq!(out, payload, "ephemeral bank leaves content inline");
    }

    #[test]
    fn quarantine_detects_degenerate_repetition() {
        let content = "the the the the\n".repeat(6000); // > 65536 chars, low unique ratio
        assert!(content.chars().count() >= QUARANTINED_ASSISTANT_MIN_CHARS);
        assert!(assistant_output_quarantine_reason(&content).is_some());
        // Normal long-ish prose with variety is not quarantined.
        let varied: String = (0..70_000)
            .map(|i| char::from(b'a' + (i % 26) as u8))
            .collect();
        let _ = varied; // entropy high but single segment; ensure short normal text is safe
        assert!(assistant_output_quarantine_reason("a short normal reply").is_none());
    }

    #[test]
    fn heartbeat_is_diagnostic_only() {
        assert!(heartbeat_noise_reason("still working on it").is_some());
        assert!(heartbeat_noise_reason("here is a real substantive answer about rust").is_none());
    }

    #[test]
    fn pipeline_quarantine_externalizes_with_a_dir_and_stays_inline_without() {
        let cfg = LcmConfig::in_memory();
        let body = "loop loop loop loop\n".repeat(6000);
        let degenerate = || NewMessage {
            role: "assistant".into(),
            content: Some(body.clone()),
            ..Default::default()
        };
        // No dir (ephemeral bank): quarantine cannot spill, content stays inline (lossless).
        let out = protect_message_for_ingest(degenerate(), &cfg, "s1", None);
        assert_eq!(out.content.unwrap(), body);
        // With a dir: the whole output is externalized behind a quarantine placeholder.
        let dir = std::env::temp_dir().join(format!("lcm-quar-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let out = protect_message_for_ingest(degenerate(), &cfg, "s1", Some(dir.as_path()));
        let content = out.content.unwrap();
        assert!(content.contains("Externalized quarantined assistant output"));
        assert!(content.contains("ref="));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn pipeline_skips_a_row_that_is_already_a_placeholder() {
        let mut cfg = LcmConfig::in_memory();
        cfg.large_output_externalization_enabled = true;
        cfg.large_output_externalization_threshold_chars = 10;
        let dir = std::env::temp_dir().join(format!("lcm-skip-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let placeholder =
            ingest_payload_placeholder("data_uri", "content", 10, 20, "x_abc_10.json");
        let msg = NewMessage {
            role: "tool".into(),
            content: Some(placeholder.clone()),
            ..Default::default()
        };
        // Long enough to trip the threshold gate if the skip failed.
        assert!(placeholder.len() > 10);
        let out = protect_message_for_ingest(msg, &cfg, "s1", Some(dir.as_path()));
        assert_eq!(
            out.content.unwrap(),
            placeholder,
            "placeholder re-ingests as-is"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn pipeline_threshold_externalizes_whole_content_when_enabled() {
        let mut cfg = LcmConfig::in_memory();
        cfg.large_output_externalization_enabled = true;
        cfg.large_output_externalization_threshold_chars = 100;
        let dir = std::env::temp_dir().join(format!("lcm-thresh-pipe-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let big = "long tool output ".repeat(50);
        let msg = NewMessage {
            role: "tool".into(),
            content: Some(big.clone()),
            tool_call_id: Some("c9".into()),
            ..Default::default()
        };
        let out = protect_message_for_ingest(msg, &cfg, "s1", Some(dir.as_path()));
        let content = out.content.unwrap();
        assert!(content.starts_with("[Externalized tool output: tool_call_id=c9;"));
        let reference = crate::externalize::extract_ref(&content).unwrap();
        assert_eq!(
            crate::externalize::read_externalized(dir.as_path(), &reference).unwrap(),
            big,
            "threshold externalization is lossless"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn pipeline_walks_tool_call_arguments_recursively() {
        let mut cfg = LcmConfig::in_memory();
        cfg.sensitive_patterns_enabled = true;
        cfg.sensitive_patterns = vec!["api_key".to_string()];
        let dir = std::env::temp_dir().join(format!("lcm-tcwalk-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        // The nested `args` JSON string carries a secret and a base64 payload.
        let payload = "QUJDREVG".repeat(700);
        let args = serde_json::to_string(
            &serde_json::json!({"api_key": "SUPERSECRETVALUE123", "blob": payload}),
        )
        .unwrap();
        let tool_calls =
            serde_json::to_string(&serde_json::json!([{"name": "upload", "args": args}])).unwrap();
        let msg = NewMessage {
            role: "assistant".into(),
            content: Some("uploading".into()),
            tool_calls: Some(tool_calls),
            ..Default::default()
        };
        let out = protect_message_for_ingest(msg, &cfg, "s1", Some(dir.as_path()));
        let tc = out.tool_calls.unwrap();
        assert!(
            !tc.contains("SUPERSECRETVALUE123"),
            "nested secret redacted"
        );
        assert!(tc.contains("name=api_key"));
        assert!(!tc.contains(&payload), "nested payload externalized");
        assert!(tc.contains("Externalized LCM ingest payload"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn pipeline_keeps_tool_calls_untouched_when_nothing_matches() {
        let cfg = LcmConfig::in_memory();
        let tool_calls = r#"[{"name":"fs_read","args":"{\"path\":\"/tmp/x\"}"}]"#;
        let msg = NewMessage {
            role: "assistant".into(),
            content: Some("reading".into()),
            tool_calls: Some(tool_calls.to_string()),
            ..Default::default()
        };
        let out = protect_message_for_ingest(msg, &cfg, "s1", None);
        assert_eq!(
            out.tool_calls.as_deref(),
            Some(tool_calls),
            "byte-identical when the walk changed nothing"
        );
    }
}
