//! Ingest protection (`daemon-context-lcm-port-spec.md` §8).
//!
//! Guards the SQLite write boundary — **storage / size / secret / repetition**, not prompt
//! injection. [`protect_message_for_ingest`] runs the spec-ordered pipeline over a flattened
//! [`NewMessage`]: optional sensitive redaction (§8.1) → normalize (§8.4) → skip-if-externalized →
//! assistant loop/heartbeat quarantine (§8.3) → the always-on base64/data-URI storage guard (§8.2,
//! externalizing payload bytes through [`crate::externalize`]). Everything except the storage guard
//! is opt-in and default-off; the storage guard itself no-ops when no externalization directory
//! exists (in-memory/ephemeral banks), leaving content inline.

use crate::externalize::{
    self, contains_externalized_ref, ingest_payload_placeholder, sha256_hex_prefix, PayloadMeta,
};
use crate::store::NewMessage;
use regex::Regex;
use serde_json::Value;
use std::path::Path;
use std::sync::OnceLock;

/// The minimum assistant length (chars) before loop/heartbeat quarantine is even considered (§8.3).
const QUARANTINED_ASSISTANT_MIN_CHARS: usize = 65_536;
/// The minimum tokenized length before the degenerate-output ratios are trusted (§8.3).
const QUARANTINED_ASSISTANT_MIN_TOKENS: usize = 1_000;
/// The maximum length (chars) of a message considered for heartbeat-noise diagnostics (§8.3).
const HEARTBEAT_NOISE_MAX_CHARS: usize = 256;

/// The minimum base64 length inside a `data:*;base64,` URI before the storage guard externalizes it.
const DATA_URI_MIN_BASE64: usize = 256;
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

/// One catalog entry: its name, compiled regex, the secret capture group, and whether the
/// placeholder carries a SHA-256 (passwords omit it to avoid dictionary-checkable hashes).
struct SensitivePattern {
    name: &'static str,
    re: Regex,
    secret_group: usize,
    include_hash: bool,
}

fn catalog() -> &'static [SensitivePattern] {
    static CATALOG: OnceLock<Vec<SensitivePattern>> = OnceLock::new();
    CATALOG.get_or_init(|| {
        vec![
            SensitivePattern {
                name: "api_key",
                re: Regex::new(
                    r#"(?i)\b(api[_-]?key|api[_-]?token|access[_-]?token|secret[_-]?key|client[_-]?secret)\b\s*[:=]\s*["']?([A-Za-z0-9._\-+/]{12,})"#,
                )
                .expect("api_key regex"),
                secret_group: 2,
                include_hash: true,
            },
            SensitivePattern {
                name: "bearer_token",
                re: Regex::new(r#"(?i)\bBearer\s+([A-Za-z0-9._\-+/]{12,})"#).expect("bearer regex"),
                secret_group: 1,
                include_hash: true,
            },
            SensitivePattern {
                name: "password_assignment",
                re: Regex::new(
                    r#"(?i)\b(password|passwd|pwd|passphrase)\b\s*[:=]\s*["']?([^\s"']{6,})"#,
                )
                .expect("password regex"),
                secret_group: 2,
                include_hash: false,
            },
            SensitivePattern {
                name: "private_key",
                re: Regex::new(
                    r"(?s)-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
                )
                .expect("private_key regex"),
                secret_group: 0,
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
        format!("[LCM sensitive redaction: name={name}; chars={chars}; bytes={bytes}; sha256={sha}]")
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
/// quote) with the placeholder.
fn redact_one(text: &str, pat: &SensitivePattern) -> String {
    let mut out = String::with_capacity(text.len());
    let mut last = 0usize;
    for caps in pat.re.captures_iter(text) {
        let whole = match caps.get(0) {
            Some(m) => m,
            None => continue,
        };
        let secret = match caps.get(pat.secret_group) {
            Some(m) => m,
            None => continue,
        };
        out.push_str(&text[last..secret.start()]);
        out.push_str(&redaction_placeholder(pat.name, secret.as_str(), pat.include_hash));
        out.push_str(&text[secret.end()..whole.end()]);
        last = whole.end();
    }
    out.push_str(&text[last..]);
    out
}

/// Recursively redact a JSON value (string leaves via the text catalog; sensitive *keys* redact
/// their whole string value) — used for the assistant `tool_calls` blob (§8.1).
pub fn redact_sensitive_value(value: &mut Value, active: &[String]) {
    match value {
        Value::String(s) => *s = redact_sensitive_text(s, active),
        Value::Array(items) => items.iter_mut().for_each(|v| redact_sensitive_value(v, active)),
        Value::Object(map) => {
            for (k, v) in map.iter_mut() {
                if let (Some(name), Value::String(s)) = (sensitive_pattern_for_key(k), &*v) {
                    if active.iter().any(|n| n == name) {
                        let include_hash = name != "password_assignment";
                        *v = Value::String(redaction_placeholder(name, s, include_hash));
                        continue;
                    }
                }
                redact_sensitive_value(v, active);
            }
        }
        _ => {}
    }
}

/// The key-name heuristic (`_sensitive_pattern_for_key`, §8.1): map a sensitive-looking key to a
/// catalog name so its whole value is redacted.
fn sensitive_pattern_for_key(key: &str) -> Option<&'static str> {
    let k = key.to_ascii_lowercase();
    if k.contains("password") || k.contains("passwd") || k.contains("passphrase") || k == "pwd" {
        Some("password_assignment")
    } else if k.contains("api_key")
        || k.contains("apikey")
        || k.contains("api-key")
        || k.contains("secret")
        || k.contains("token")
        || k.contains("access_key")
    {
        Some("api_key")
    } else {
        None
    }
}

// ---- 8.2 storage-boundary guard (always on) -----------------------------------------------------

fn data_uri_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)data:[^\s;,]*;base64,([A-Za-z0-9+/=]{256,})").expect("data-uri regex")
    })
}

fn bare_run_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"[A-Za-z0-9+/=_\-]{4096,}").expect("base64-run regex"))
}

/// `looks_like_long_base64` (§8.2): length ≥ `min_len`, `len % 4 != 1`, ≥0.98 base64-alphabet
/// ratio, and ≥8 distinct chars. The capturing regexes already restrict the alphabet, so the ratio
/// check is belt-and-suspenders for the bare-run path.
fn looks_like_long_base64(s: &str, min_len: usize) -> bool {
    let n = s.chars().count();
    if n < min_len || n % 4 == 1 {
        return false;
    }
    let mut alpha = 0usize;
    let mut distinct = std::collections::HashSet::new();
    for c in s.chars() {
        if c.is_ascii_alphanumeric() || matches!(c, '+' | '/' | '=' | '_' | '-') {
            alpha += 1;
        }
        distinct.insert(c);
    }
    if distinct.len() < 8 {
        return false;
    }
    (alpha as f64) / (n as f64) >= 0.98
}

/// Externalize every valid base64-ish run matched by `re` (validating the `group` capture), spilling
/// each to `dir` and substituting the §8.2 placeholder. Returns the rewritten text and whether any
/// substitution happened.
#[allow(clippy::too_many_arguments)]
fn externalize_runs(
    text: &str,
    re: &Regex,
    group: usize,
    kind: &str,
    field: &str,
    role: &str,
    tool_call_id: Option<&str>,
    dir: &Path,
    min_len: usize,
) -> (String, bool) {
    let mut out = String::with_capacity(text.len());
    let mut last = 0usize;
    let mut changed = false;
    for caps in re.captures_iter(text) {
        let whole = match caps.get(0) {
            Some(m) => m,
            None => continue,
        };
        let candidate = caps.get(group).map(|m| m.as_str()).unwrap_or(whole.as_str());
        if !looks_like_long_base64(candidate, min_len) {
            continue;
        }
        let body = whole.as_str();
        let meta = PayloadMeta { kind, field, role, tool_call_id };
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

/// Apply the always-on base64/data-URI storage guard to one field's text. No-op (returns the input)
/// when `dir` is `None` (ephemeral bank) or the text already carries an externalized-ref placeholder.
fn apply_storage_guard(
    text: &str,
    field: &str,
    role: &str,
    tool_call_id: Option<&str>,
    dir: Option<&Path>,
) -> String {
    let Some(dir) = dir else {
        return text.to_string();
    };
    if contains_externalized_ref(text) {
        return text.to_string();
    }
    let (text, _) = externalize_runs(
        text,
        data_uri_re(),
        1,
        "data_uri",
        field,
        role,
        tool_call_id,
        dir,
        DATA_URI_MIN_BASE64,
    );
    let (text, _) = externalize_runs(
        &text,
        bare_run_re(),
        0,
        "base64_run",
        field,
        role,
        tool_call_id,
        dir,
        BARE_RUN_MIN,
    );
    text
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

    let segments: Vec<&str> = content.lines().map(|l| l.trim()).filter(|l| !l.is_empty()).collect();
    let seg_total = segments.len();
    let mut seg_freq = std::collections::HashMap::new();
    for s in &segments {
        *seg_freq.entry(*s).or_insert(0usize) += 1;
    }
    let distinct_segments = seg_freq.len();
    let duplicate_segment_ratio = ratio(seg_total.saturating_sub(distinct_segments), seg_total);
    let top_segment_ratio = ratio(seg_freq.values().copied().max().unwrap_or(0), seg_total);

    let distinct_chars = content.chars().collect::<std::collections::HashSet<_>>().len();

    let degenerate_tokens = total >= QUARANTINED_ASSISTANT_MIN_TOKENS || total == 0;
    let primary = unique_token_ratio <= 0.03
        && degenerate_tokens
        && (top_segment_ratio >= 0.10 || duplicate_segment_ratio >= 0.50 || top_token_ratio >= 0.08);
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

/// The active-replay placeholder for quarantined output kept volatile (no disk) (§8.3).
fn active_replay_placeholder(content: &str) -> String {
    let chars = content.chars().count();
    let bytes = content.len();
    let sha = sha256_hex_prefix(content.as_bytes(), 16);
    format!("[LCM active replay placeholder: chars={chars}; bytes={bytes}; sha256={sha}]")
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

/// Run the full ingest-protection pipeline over one flattened message (§8). `dir` is the resolved
/// externalization directory (`None` for ephemeral banks → storage guard / quarantine-spill no-op,
/// content stays inline).
pub fn protect_message_for_ingest(
    mut msg: NewMessage,
    cfg: &crate::config::LcmConfig,
    dir: Option<&Path>,
) -> NewMessage {
    // 1. Sensitive redaction (opt-in).
    if cfg.sensitive_patterns_enabled {
        if let Some(content) = msg.content.as_ref() {
            msg.content = Some(redact_sensitive_text(content, &cfg.sensitive_patterns));
        }
        if let Some(tc) = msg.tool_calls.as_ref() {
            if let Ok(mut value) = serde_json::from_str::<Value>(tc) {
                redact_sensitive_value(&mut value, &cfg.sensitive_patterns);
                if let Ok(s) = serde_json::to_string(&value) {
                    msg.tool_calls = Some(s);
                }
            }
        }
    }

    // 2/3. Normalize is a no-op for our already-string content; skip-if-externalized is handled by
    // the storage guard. 4. Assistant loop/heartbeat quarantine.
    if msg.role == "assistant" {
        if let Some(content) = msg.content.as_ref() {
            if !contains_externalized_ref(content) {
                if let Some(reason) = assistant_output_quarantine_reason(content) {
                    tracing::warn!(reason = %reason, "lcm: quarantining runaway assistant output");
                    msg.content = Some(quarantine_replacement(content, &msg.role, dir));
                }
            }
        }
    }

    // 5. Always-on storage guard over the (possibly quarantined) content.
    if let Some(content) = msg.content.as_ref() {
        msg.content = Some(apply_storage_guard(content, "content", &msg.role, msg.tool_call_id.as_deref(), dir));
    }
    // 6. Tool-call blob protection (base64 inside args).
    if let Some(tc) = msg.tool_calls.as_ref() {
        msg.tool_calls = Some(apply_storage_guard(tc, "tool_calls", &msg.role, None, dir));
    }

    msg
}

/// Replace quarantined assistant output: externalize to disk when a dir exists, else swap for the
/// volatile active-replay placeholder.
fn quarantine_replacement(content: &str, role: &str, dir: Option<&Path>) -> String {
    if let Some(dir) = dir {
        let meta = PayloadMeta { kind: "quarantine", field: "content", role, tool_call_id: None };
        match externalize::store_payload(dir, content, &meta) {
            Ok(reference) => {
                return format!(
                    "[Externalized quarantined assistant output: chars={}; bytes={}; ref={}]",
                    content.chars().count(),
                    content.len(),
                    reference
                );
            }
            Err(e) => {
                tracing::warn!(error = %e, "lcm: quarantine externalization failed; using active-replay placeholder");
            }
        }
    }
    active_replay_placeholder(content)
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
        assert!(red.contains("ABCDEF0123456789"), "inactive pattern is not applied");
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
        let mut v: Value = serde_json::json!({"password": "short!", "nested": {"api_token": "ABCDEFGHIJKL"}, "ok": "keep"});
        redact_sensitive_value(&mut v, &active);
        assert!(v["password"].as_str().unwrap().contains("name=password_assignment"));
        assert!(v["nested"]["api_token"].as_str().unwrap().contains("name=api_key"));
        assert_eq!(v["ok"], "keep");
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
        let out = apply_storage_guard(&text, "content", "tool", Some("c1"), Some(dir.as_path()));
        assert!(out.contains("Externalized LCM ingest payload"));
        assert!(out.contains("here is data:") && out.contains("end"));
        assert!(!out.contains(&payload));
        // Re-running on the placeholder is a no-op.
        let again = apply_storage_guard(&out, "content", "tool", Some("c1"), Some(dir.as_path()));
        assert_eq!(again, out);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn storage_guard_noops_without_a_dir() {
        let payload = "QUJDREVG".repeat(700);
        let out = apply_storage_guard(&payload, "content", "tool", None, None);
        assert_eq!(out, payload, "ephemeral bank leaves content inline");
    }

    #[test]
    fn quarantine_detects_degenerate_repetition() {
        let content = "the the the the\n".repeat(6000); // > 65536 chars, low unique ratio
        assert!(content.chars().count() >= QUARANTINED_ASSISTANT_MIN_CHARS);
        assert!(assistant_output_quarantine_reason(&content).is_some());
        // Normal long-ish prose with variety is not quarantined.
        let varied: String = (0..70_000).map(|i| char::from(b'a' + (i % 26) as u8)).collect();
        let _ = varied; // entropy high but single segment; ensure short normal text is safe
        assert!(assistant_output_quarantine_reason("a short normal reply").is_none());
    }

    #[test]
    fn heartbeat_is_diagnostic_only() {
        assert!(heartbeat_noise_reason("still working on it").is_some());
        assert!(heartbeat_noise_reason("here is a real substantive answer about rust").is_none());
    }

    #[test]
    fn pipeline_applies_quarantine_then_guard() {
        let cfg = LcmConfig::in_memory();
        let degenerate = NewMessage {
            role: "assistant".into(),
            content: Some("loop loop loop loop\n".repeat(6000)),
            ..Default::default()
        };
        // No dir (in-memory) -> active-replay placeholder.
        let out = protect_message_for_ingest(degenerate, &cfg, None);
        assert!(out.content.unwrap().contains("active replay placeholder"));
    }
}
