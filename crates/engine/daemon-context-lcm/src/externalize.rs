// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Large-payload externalization (`daemon-context-lcm-port-spec.md` §9.1).
//!
//! LCM keeps payload bytes (oversized base64/media, huge tool outputs) **out** of `lcm.db`, FTS, the
//! WAL, and backups by spilling them to a side-channel directory under the data root and leaving a
//! compact placeholder (carrying a recovery `ref`) in the row. The always-on storage guard (§8.2)
//! and the opt-in threshold path (`maybe_externalize_payload`) both route through [`store_payload`];
//! [`lcm_expand`](crate::tools)/[`lcm_describe`](crate::tools) recover the bytes via
//! [`read_externalized`]. When no externalization directory exists (in-memory/ephemeral banks) the
//! callers no-op and leave content inline — no data loss, the store just carries the bytes.

use sha2::{Digest, Sha256};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use regex::Regex;
use serde_json::json;

/// The first `n` hex chars of the SHA-256 of `bytes` (digests: redaction uses 16, externalization 12).
pub(crate) fn sha256_hex_prefix(bytes: &[u8], n: usize) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    let mut s = String::with_capacity(n);
    for b in digest.iter() {
        if s.len() >= n {
            break;
        }
        s.push_str(&format!("{b:02x}"));
    }
    s.truncate(n);
    s
}

/// Recover the `ref=<id>` token from any externalized/GC'd placeholder (§9.1 `_EXTERNALIZED_REF_RE`).
///
/// Faithful broadening of the Python regex: the body between the family prefix and `; ref=` is
/// matched non-greedily as `[^\]]*?` so the §8.2 ingest-payload placeholder
/// (`[Externalized LCM ingest payload: …]`) and the quarantine placeholder are captured by the same
/// pattern as the §9.1 `tool output`/`payload`/`GC'd` families.
pub fn externalized_ref_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?s)\[(?:Externalized|GC'd externalized)[^\]]*?[;:]\s*ref=([^;\]\s]+)\]")
            .expect("externalized-ref regex is valid")
    })
}

/// The §8.2 ingest-payload placeholder family only (`_INGEST_PLACEHOLDER_RE`,
/// `LCM:ingest_protection.py:104`) — `lcm_expand` prefers ingest refs over the legacy families
/// (`extract_ingest_externalized_refs`, `LCM:tools.py:350-351`).
fn ingest_placeholder_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?s)\[Externalized LCM ingest payload:[^\]]*?;\s*ref=([^;\]\s]+)\]")
            .expect("ingest-placeholder ref regex is valid")
    })
}

/// Whether `text` already carries an externalized-payload placeholder (so the storage guard skips it).
pub fn contains_externalized_ref(text: &str) -> bool {
    externalized_ref_regex().is_match(text)
}

/// Extract the first recovery `ref` from a placeholder, if present.
pub fn extract_ref(text: &str) -> Option<String> {
    externalized_ref_regex()
        .captures(text)
        .and_then(|c| c.get(1))
        .map(|m| m.as_str().to_string())
}

/// All `ref`s of the §8.2 ingest-placeholder family, in text order, deduplicated
/// (`extract_ingest_externalized_refs`, `LCM:ingest_protection.py:145-153`).
pub fn extract_ingest_refs(text: &str) -> Vec<String> {
    let mut refs: Vec<String> = Vec::new();
    for cap in ingest_placeholder_regex().captures_iter(text) {
        let reference = cap[1].trim().to_string();
        if !reference.is_empty() && !refs.contains(&reference) {
            refs.push(reference);
        }
    }
    refs
}

/// All recovery `ref`s from every recognized placeholder family, basename-validated and
/// deduplicated (`extract_all_externalized_payload_refs`, `LCM:ingest_protection.py:160-168`).
pub fn extract_all_refs(text: &str) -> Vec<String> {
    let mut refs: Vec<String> = Vec::new();
    for cap in externalized_ref_regex().captures_iter(text) {
        let reference = cap[1].trim().to_string();
        if is_basename_ref(&reference) && !refs.contains(&reference) {
            refs.push(reference);
        }
    }
    refs
}

/// Metadata recorded alongside an externalized payload (and reflected in its placeholder).
#[derive(Clone, Debug)]
pub struct PayloadMeta<'a> {
    /// The payload family (`data_uri`, `base64_run`, `tool_result`, `raw_payload`, `quarantine`).
    pub kind: &'a str,
    /// The originating field (`content`, `tool_calls`).
    pub field: &'a str,
    /// The originating role.
    pub role: &'a str,
    /// For a tool result: the call id this payload answered.
    pub tool_call_id: Option<&'a str>,
    /// The owning session — payload metadata/content recovery is session-scoped
    /// (`_get_externalized_payload`, `LCM:tools.py:84-97`). Empty means unscoped (legacy records).
    pub session_id: &'a str,
}

/// Persist `body` to `dir` and return its recovery `ref` (the file name). The directory is created
/// `0700` and the file written `O_CREAT|O_EXCL` `0600` with JSON indent 2 (§9.1). Identical bodies
/// dedup by a `*_{digest12}_*.json` scan **within the same session** — payload records are
/// session-scoped (Python's dedup filter includes `session_id`,
/// `LCM:externalize.py:372-386`), so the same body in another session gets its own file rather
/// than pinning the payload to the first writer's session.
pub fn store_payload(dir: &Path, body: &str, meta: &PayloadMeta<'_>) -> io::Result<String> {
    ensure_dir(dir)?;
    let digest = sha256_hex_prefix(body.as_bytes(), 12);
    if let Some((existing, _)) = find_payload_for_content(
        dir,
        body,
        Some(meta.kind),
        meta.tool_call_id.unwrap_or(""),
        meta.session_id,
    ) {
        return Ok(existing);
    }
    let chars = body.chars().count();
    let bytes = body.len();
    let kind = sanitize_kind(meta.kind);
    // The session token keeps file names distinct across sessions for the same body (the
    // exclusive-create dedup fallback must not alias another session's record).
    let file_name = if meta.session_id.is_empty() {
        format!("{kind}_{digest}_{chars}.json")
    } else {
        format!(
            "{kind}_{digest}_{chars}_{}.json",
            sha256_hex_prefix(meta.session_id.as_bytes(), 8)
        )
    };
    let path = dir.join(&file_name);
    let payload = json!({
        "content": body,
        "kind": meta.kind,
        "field": meta.field,
        "role": meta.role,
        "tool_call_id": meta.tool_call_id,
        "session_id": meta.session_id,
        "chars": chars,
        "bytes": bytes,
        "digest": digest,
        "created_at": now_secs(),
    });
    let serialized = serde_json::to_string_pretty(&payload)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    match create_exclusive(&path) {
        Ok(()) => {
            fs::write(&path, serialized.as_bytes())?;
            Ok(file_name)
        }
        // A concurrent writer produced the same name (same digest+size): treat as a dedup hit.
        Err(e) if e.kind() == io::ErrorKind::AlreadyExists => Ok(file_name),
        Err(e) => Err(e),
    }
}

/// Whether a recovery `ref` is a bare file name. A ref carrying path components (separators, `..`,
/// a trailing slash) is **rejected**, never coerced to its basename — coercion would make a
/// traversal-shaped ref silently resolve to a legitimate file (`load_externalized_payload`,
/// `LCM:externalize.py:168-169`; `_is_basename_ref`, `LCM:ingest_protection.py:157-158`).
fn is_basename_ref(reference: &str) -> bool {
    !reference.is_empty()
        && !reference.contains('/')
        && !reference.contains('\\')
        && Path::new(reference)
            .file_name()
            .is_some_and(|n| n == std::ffi::OsStr::new(reference))
}

/// Read back an externalized payload's original body by its recovery `ref` (a bare file name; a
/// ref with path components is rejected).
pub fn read_externalized(dir: &Path, reference: &str) -> Option<String> {
    if !is_basename_ref(reference) {
        return None;
    }
    let path = dir.join(reference);
    let raw = fs::read_to_string(path).ok()?;
    let value: serde_json::Value = serde_json::from_str(&raw).ok()?;
    value
        .get("content")
        .and_then(|c| c.as_str())
        .map(|s| s.to_string())
}

/// Read back an externalized payload's full record (metadata + content) by its `ref`, for
/// `lcm_describe(externalized_ref=…)` (which strips the content for its metadata-only view). A ref
/// with path components is rejected.
pub fn read_payload_record(dir: &Path, reference: &str) -> Option<serde_json::Value> {
    if !is_basename_ref(reference) {
        return None;
    }
    let path = dir.join(reference);
    let raw = fs::read_to_string(path).ok()?;
    serde_json::from_str(&raw).ok()
}

/// Map a payload record to the Python summary shape (`_externalized_summary`,
/// `LCM:externalize.py:97-108`): `ref`/`kind`/`tool_call_id`/`role`/`session_id`/`field_path`/
/// `content_chars`/`content_bytes`/`created_at`, tolerating both this port's record keys
/// (`field`/`chars`/`bytes`) and the Python ones (`field_path`/`content_chars`/`content_bytes`).
pub fn payload_summary(reference: &str, record: &serde_json::Value) -> serde_json::Value {
    let content = record.get("content").and_then(|c| c.as_str()).unwrap_or("");
    let get_str = |keys: &[&str]| -> String {
        keys.iter()
            .find_map(|k| record.get(*k).and_then(|v| v.as_str()))
            .unwrap_or("")
            .to_string()
    };
    let get_num = |keys: &[&str], default: u64| -> u64 {
        keys.iter()
            .find_map(|k| record.get(*k).and_then(|v| v.as_u64()))
            .unwrap_or(default)
    };
    json!({
        "ref": reference,
        "kind": if get_str(&["kind"]).is_empty() { "tool_result".to_string() } else { get_str(&["kind"]) },
        "tool_call_id": get_str(&["tool_call_id"]),
        "role": get_str(&["role"]),
        "session_id": get_str(&["session_id"]),
        "field_path": get_str(&["field_path", "field"]),
        "content_chars": get_num(&["content_chars", "chars"], content.chars().count() as u64),
        "content_bytes": get_num(&["content_bytes", "bytes"], content.len() as u64),
        "created_at": record.get("created_at").cloned().unwrap_or(serde_json::Value::Null),
    })
}

/// Load an externalized payload as its Python summary shape **plus** `content`
/// (`load_externalized_payload`, `LCM:externalize.py:167-182`). `None` for a missing/invalid ref.
pub fn load_payload(dir: &Path, reference: &str) -> Option<serde_json::Value> {
    let record = read_payload_record(dir, reference)?;
    let mut summary = payload_summary(reference, &record);
    if let serde_json::Value::Object(ref mut map) = summary {
        map.insert(
            "content".to_string(),
            serde_json::Value::String(
                record
                    .get("content")
                    .and_then(|c| c.as_str())
                    .unwrap_or("")
                    .to_string(),
            ),
        );
    }
    Some(summary)
}

/// Whether a payload record's `kind` is a §8.2 ingest spill. Python writes every ingest spill
/// record with the umbrella kind `ingest_payload`; the Rust `store_payload` records keep the
/// family (`data_uri` / `base64_run` / `payload`) instead, so restoration accepts the union.
/// Quarantine and tool-result/GC spills stay excluded (they are recovery surfaces, not identity
/// content).
fn is_ingest_spill_kind(kind: &str) -> bool {
    matches!(
        kind,
        "ingest_payload" | "data_uri" | "base64_run" | "payload"
    )
}

/// Replace §8.2 ingest placeholders with their stored payload content, for identity matching only
/// (`restore_ingest_payload_placeholders`, `LCM:ingest_protection.py:496-522`). A missing,
/// non-ingest, or session-mismatched payload leaves the placeholder untouched so callers never
/// fabricate content or hide a recovery problem.
pub fn restore_ingest_placeholders(dir: &Path, text: &str, session_id: &str) -> String {
    if !text.contains("[Externalized LCM ingest payload:") {
        return text.to_string();
    }
    ingest_placeholder_regex()
        .replace_all(text, |caps: &regex::Captures<'_>| {
            let reference = caps[1].trim();
            let Some(payload) = load_payload(dir, reference) else {
                return caps[0].to_string();
            };
            let kind = payload.get("kind").and_then(|k| k.as_str()).unwrap_or("");
            if !is_ingest_spill_kind(kind) {
                return caps[0].to_string();
            }
            let payload_session = payload
                .get("session_id")
                .and_then(|s| s.as_str())
                .unwrap_or("");
            if !session_id.is_empty()
                && !payload_session.is_empty()
                && payload_session != session_id
            {
                return caps[0].to_string();
            }
            match payload.get("content").and_then(|c| c.as_str()) {
                Some(content) => content.to_string(),
                None => caps[0].to_string(),
            }
        })
        .into_owned()
}

/// The opt-in threshold-externalization gate (§9.1 `maybe_externalize_payload`): when `enabled` and
/// `body` exceeds `threshold_chars`, spill it and return `(placeholder, ref)`; otherwise `None`
/// (leave inline). No-op when `dir` is `None` (ephemeral bank).
pub fn maybe_externalize_payload(
    dir: Option<&Path>,
    body: &str,
    enabled: bool,
    threshold_chars: usize,
    meta: &PayloadMeta<'_>,
) -> Option<(String, String)> {
    if !enabled || body.chars().count() < threshold_chars {
        return None;
    }
    let dir = dir?;
    if contains_externalized_ref(body) {
        return None;
    }
    match store_payload(dir, body, meta) {
        Ok(reference) => {
            let placeholder =
                payload_placeholder(meta, body.chars().count(), body.len(), &reference);
            Some((placeholder, reference))
        }
        Err(e) => {
            tracing::warn!(error = %e, kind = meta.kind, "lcm: payload externalization failed; leaving inline");
            None
        }
    }
}

/// The §8.2 storage-guard placeholder for an externalized ingest payload (base64/data-URI run).
pub fn ingest_payload_placeholder(
    kind: &str,
    field: &str,
    chars: usize,
    bytes: usize,
    reference: &str,
) -> String {
    format!(
        "[Externalized LCM ingest payload: kind={kind}; field={field}; chars={chars}; bytes={bytes}; ref={reference}]"
    )
}

/// The §9.1 generic / tool-output threshold placeholder (tool output when a `tool_call_id` is set).
pub fn payload_placeholder(
    meta: &PayloadMeta<'_>,
    chars: usize,
    bytes: usize,
    reference: &str,
) -> String {
    match meta.tool_call_id {
        Some(id) => format!(
            "[Externalized tool output: tool_call_id={id}; chars={chars}; bytes={bytes}; ref={reference}]"
        ),
        None => format!(
            "[Externalized payload: kind={kind}; role={role}; chars={chars}; bytes={bytes}; ref={reference}]",
            kind = meta.kind,
            role = meta.role,
        ),
    }
}

/// The §9.1 transcript-GC placeholder rewritten over an already-externalized, already-summarized row.
pub fn gc_placeholder(is_tool_output: bool, reference: &str) -> String {
    if is_tool_output {
        format!("[GC'd externalized tool output: ref={reference}]")
    } else {
        format!("[GC'd externalized payload: ref={reference}]")
    }
}

/// Scan `dir` for a payload record whose content equals `body` (the digest-prefix candidate scan of
/// `find_externalized_payload_for_message`, `LCM:externalize.py:225-266`): candidates match on
/// `kind` (when given), `tool_call_id`, and session — a record for the caller's session wins;
/// otherwise the first record without a session claim is the fallback.
pub fn find_payload_for_content(
    dir: &Path,
    body: &str,
    kind: Option<&str>,
    tool_call_id: &str,
    session_id: &str,
) -> Option<(String, serde_json::Value)> {
    let digest = sha256_hex_prefix(body.as_bytes(), 12);
    let needle = format!("_{digest}_");
    let mut names: Vec<String> = fs::read_dir(dir)
        .ok()?
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            (name.ends_with(".json") && name.contains(&needle)).then_some(name)
        })
        .collect();
    names.sort();
    let mut fallback: Option<(String, serde_json::Value)> = None;
    for name in names {
        let Some(record) = read_payload_record(dir, &name) else {
            continue;
        };
        let record_kind = record
            .get("kind")
            .and_then(|k| k.as_str())
            .unwrap_or("tool_result");
        if kind.is_some_and(|k| record_kind != k) {
            continue;
        }
        let record_call = record
            .get("tool_call_id")
            .and_then(|c| c.as_str())
            .unwrap_or("");
        if record_call != tool_call_id {
            continue;
        }
        if record.get("content").and_then(|c| c.as_str()) != Some(body) {
            continue;
        }
        let record_session = record
            .get("session_id")
            .and_then(|s| s.as_str())
            .unwrap_or("");
        if !session_id.is_empty() {
            if record_session == session_id {
                return Some((name, record));
            }
            continue;
        }
        if fallback.is_none() {
            fallback = Some((name, record));
        }
    }
    fallback
}

/// Restrict a `kind` to a safe file-name token.
fn sanitize_kind(kind: &str) -> String {
    let s: String = kind
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    if s.is_empty() {
        "payload".to_string()
    } else {
        s
    }
}

fn now_secs() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// Create `dir` (recursively) and tighten its mode to `0700` on Unix (best-effort).
fn ensure_dir(dir: &Path) -> io::Result<()> {
    fs::create_dir_all(dir)?;
    set_mode(dir, 0o700);
    Ok(())
}

/// Create `path` exclusively (fails if it exists) with mode `0600` on Unix.
fn create_exclusive(path: &Path) -> io::Result<()> {
    let mut opts = fs::OpenOptions::new();
    opts.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    opts.open(path).map(|_| ())
}

#[cfg(unix)]
fn set_mode(path: &Path, mode: u32) {
    use std::os::unix::fs::PermissionsExt;
    let _ = fs::set_permissions(path, fs::Permissions::from_mode(mode));
}

#[cfg(not(unix))]
fn set_mode(_path: &Path, _mode: u32) {}

/// The default `<data_root>/lcm-large-outputs` directory name (kept here for symmetry with the
/// config resolver, which owns the actual path policy).
#[allow(dead_code)]
pub(crate) fn default_subdir() -> PathBuf {
    PathBuf::from("lcm-large-outputs")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("lcm-ext-{}-{}", tag, std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn round_trips_a_payload() {
        let dir = tmp("rt");
        let meta = PayloadMeta {
            kind: "base64_run",
            field: "content",
            role: "tool",
            tool_call_id: Some("c1"),
            session_id: "s1",
        };
        let body = "QUJD".repeat(2000);
        let reference = store_payload(&dir, &body, &meta).unwrap();
        let back = read_externalized(&dir, &reference).unwrap();
        assert_eq!(back, body);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn dedups_identical_bodies() {
        let dir = tmp("dedup");
        let meta = PayloadMeta {
            kind: "payload",
            field: "content",
            role: "assistant",
            tool_call_id: None,
            session_id: "s1",
        };
        let body = "x".repeat(5000);
        let r1 = store_payload(&dir, &body, &meta).unwrap();
        let r2 = store_payload(&dir, &body, &meta).unwrap();
        assert_eq!(r1, r2, "same digest reuses the same file");
        let count = fs::read_dir(&dir).unwrap().count();
        assert_eq!(count, 1, "no duplicate file written");
        // A different session never aliases another session's record (payload metadata is
        // session-scoped, `LCM:externalize.py:259-263`).
        let other = PayloadMeta {
            session_id: "s2",
            ..meta.clone()
        };
        let r3 = store_payload(&dir, &body, &other).unwrap();
        assert_ne!(r1, r3, "same body in another session gets its own file");
        assert_eq!(fs::read_dir(&dir).unwrap().count(), 2);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn payload_summary_maps_record_keys_to_python_shape() {
        let dir = tmp("summary");
        let meta = PayloadMeta {
            kind: "tool_result",
            field: "content",
            role: "tool",
            tool_call_id: Some("c9"),
            session_id: "sX",
        };
        let body = "b".repeat(64);
        let reference = store_payload(&dir, &body, &meta).unwrap();
        let loaded = load_payload(&dir, &reference).unwrap();
        assert_eq!(loaded["ref"], reference.as_str());
        assert_eq!(loaded["kind"], "tool_result");
        assert_eq!(loaded["tool_call_id"], "c9");
        assert_eq!(loaded["session_id"], "sX");
        assert_eq!(loaded["field_path"], "content");
        assert_eq!(loaded["content_chars"], 64);
        assert_eq!(loaded["content_bytes"], 64);
        assert_eq!(loaded["content"], body.as_str());
        // Content-equality lookup honors session scoping.
        assert!(find_payload_for_content(&dir, &body, Some("tool_result"), "c9", "sX").is_some());
        assert!(
            find_payload_for_content(&dir, &body, Some("tool_result"), "c9", "other").is_none(),
            "a session-claimed record never matches another session"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn ref_regex_captures_every_family() {
        let p_ingest =
            ingest_payload_placeholder("data_uri", "content", 10, 20, "data_uri_abc_10.json");
        let p_tool = payload_placeholder(
            &PayloadMeta {
                kind: "tool_output",
                field: "content",
                role: "tool",
                tool_call_id: Some("c1"),
                session_id: "s1",
            },
            10,
            20,
            "tool_output_def_10.json",
        );
        let p_payload = payload_placeholder(
            &PayloadMeta {
                kind: "payload",
                field: "content",
                role: "assistant",
                tool_call_id: None,
                session_id: "s1",
            },
            10,
            20,
            "payload_ghi_10.json",
        );
        let p_gc = gc_placeholder(true, "tool_output_def_10.json");
        assert_eq!(
            extract_ref(&p_ingest).as_deref(),
            Some("data_uri_abc_10.json")
        );
        assert_eq!(
            extract_ref(&p_tool).as_deref(),
            Some("tool_output_def_10.json")
        );
        assert_eq!(
            extract_ref(&p_payload).as_deref(),
            Some("payload_ghi_10.json")
        );
        assert_eq!(
            extract_ref(&p_gc).as_deref(),
            Some("tool_output_def_10.json")
        );
        assert!(contains_externalized_ref(&p_ingest));
        assert!(!contains_externalized_ref("just some text"));
    }

    #[test]
    fn refs_with_path_components_are_rejected_not_coerced() {
        let dir = tmp("traversal");
        let meta = PayloadMeta {
            kind: "payload",
            field: "content",
            role: "assistant",
            tool_call_id: None,
            session_id: "s1",
        };
        let body = "y".repeat(5000);
        let reference = store_payload(&dir, &body, &meta).unwrap();
        assert!(
            read_externalized(&dir, &reference).is_some(),
            "bare ref reads"
        );
        // A traversal-shaped ref must NOT resolve to the legitimate file via basename coercion.
        for bad in [
            format!("../{reference}"),
            format!("sub/{reference}"),
            format!("..\\{reference}"),
            "..".to_string(),
            String::new(),
        ] {
            assert!(
                read_externalized(&dir, &bad).is_none(),
                "rejected ref: {bad:?}"
            );
            assert!(read_payload_record(&dir, &bad).is_none());
        }
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn threshold_gate_respects_enabled_and_size() {
        let dir = tmp("thresh");
        let meta = PayloadMeta {
            kind: "tool_output",
            field: "content",
            role: "tool",
            tool_call_id: Some("c1"),
            session_id: "s1",
        };
        // Disabled -> None even when large.
        assert!(
            maybe_externalize_payload(Some(&dir), &"a".repeat(100), false, 10, &meta).is_none()
        );
        // Enabled but under threshold -> None.
        assert!(maybe_externalize_payload(Some(&dir), "short", true, 10_000, &meta).is_none());
        // Enabled + over threshold -> Some.
        let big = "a".repeat(50);
        let (placeholder, reference) =
            maybe_externalize_payload(Some(&dir), &big, true, 10, &meta).unwrap();
        assert!(placeholder.contains("ref="));
        assert_eq!(read_externalized(&dir, &reference).unwrap(), big);
        let _ = fs::remove_dir_all(&dir);
    }
}
