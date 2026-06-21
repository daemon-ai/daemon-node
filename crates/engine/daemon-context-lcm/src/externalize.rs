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

/// Metadata recorded alongside an externalized payload (and reflected in its placeholder).
#[derive(Clone, Debug)]
pub struct PayloadMeta<'a> {
    /// The payload family (`data_uri`, `base64_run`, `tool_output`, `payload`, `quarantine`).
    pub kind: &'a str,
    /// The originating field (`content`, `tool_calls`).
    pub field: &'a str,
    /// The originating role.
    pub role: &'a str,
    /// For a tool result: the call id this payload answered.
    pub tool_call_id: Option<&'a str>,
}

/// Persist `body` to `dir` and return its recovery `ref` (the file name). The directory is created
/// `0700` and the file written `O_CREAT|O_EXCL` `0600` with JSON indent 2 (§9.1). Identical bodies
/// dedup by a `*_{digest12}_*.json` scan, so re-ingesting the same payload reuses one file.
pub fn store_payload(dir: &Path, body: &str, meta: &PayloadMeta<'_>) -> io::Result<String> {
    ensure_dir(dir)?;
    let digest = sha256_hex_prefix(body.as_bytes(), 12);
    if let Some(existing) = find_by_digest(dir, &digest) {
        return Ok(existing);
    }
    let chars = body.chars().count();
    let bytes = body.len();
    let kind = sanitize_kind(meta.kind);
    let file_name = format!("{kind}_{digest}_{chars}.json");
    let path = dir.join(&file_name);
    let payload = json!({
        "content": body,
        "kind": meta.kind,
        "field": meta.field,
        "role": meta.role,
        "tool_call_id": meta.tool_call_id,
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

/// Read back an externalized payload's original body by its recovery `ref` (file name). Path
/// components are stripped to a bare file name to keep the read inside `dir`.
pub fn read_externalized(dir: &Path, reference: &str) -> Option<String> {
    let name = Path::new(reference).file_name()?;
    let path = dir.join(name);
    let raw = fs::read_to_string(path).ok()?;
    let value: serde_json::Value = serde_json::from_str(&raw).ok()?;
    value
        .get("content")
        .and_then(|c| c.as_str())
        .map(|s| s.to_string())
}

/// Read back an externalized payload's full record (metadata + content) by its `ref`, for
/// `lcm_describe(externalized_ref=…)` (which strips the content for its metadata-only view).
pub fn read_payload_record(dir: &Path, reference: &str) -> Option<serde_json::Value> {
    let name = Path::new(reference).file_name()?;
    let path = dir.join(name);
    let raw = fs::read_to_string(path).ok()?;
    serde_json::from_str(&raw).ok()
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
            let placeholder = payload_placeholder(meta, body.chars().count(), body.len(), &reference);
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
pub fn payload_placeholder(meta: &PayloadMeta<'_>, chars: usize, bytes: usize, reference: &str) -> String {
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

/// Find an existing payload file for `digest` (the dedup scan `*_{digest12}_*.json`).
fn find_by_digest(dir: &Path, digest: &str) -> Option<String> {
    let needle = format!("_{digest}_");
    let entries = fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.ends_with(".json") && name.contains(&needle) {
            return Some(name.to_string());
        }
    }
    None
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
        };
        let body = "x".repeat(5000);
        let r1 = store_payload(&dir, &body, &meta).unwrap();
        let r2 = store_payload(&dir, &body, &meta).unwrap();
        assert_eq!(r1, r2, "same digest reuses the same file");
        let count = fs::read_dir(&dir).unwrap().count();
        assert_eq!(count, 1, "no duplicate file written");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn ref_regex_captures_every_family() {
        let p_ingest = ingest_payload_placeholder("data_uri", "content", 10, 20, "data_uri_abc_10.json");
        let p_tool = payload_placeholder(
            &PayloadMeta { kind: "tool_output", field: "content", role: "tool", tool_call_id: Some("c1") },
            10,
            20,
            "tool_output_def_10.json",
        );
        let p_payload = payload_placeholder(
            &PayloadMeta { kind: "payload", field: "content", role: "assistant", tool_call_id: None },
            10,
            20,
            "payload_ghi_10.json",
        );
        let p_gc = gc_placeholder(true, "tool_output_def_10.json");
        assert_eq!(extract_ref(&p_ingest).as_deref(), Some("data_uri_abc_10.json"));
        assert_eq!(extract_ref(&p_tool).as_deref(), Some("tool_output_def_10.json"));
        assert_eq!(extract_ref(&p_payload).as_deref(), Some("payload_ghi_10.json"));
        assert_eq!(extract_ref(&p_gc).as_deref(), Some("tool_output_def_10.json"));
        assert!(contains_externalized_ref(&p_ingest));
        assert!(!contains_externalized_ref("just some text"));
    }

    #[test]
    fn threshold_gate_respects_enabled_and_size() {
        let dir = tmp("thresh");
        let meta = PayloadMeta { kind: "tool_output", field: "content", role: "tool", tool_call_id: Some("c1") };
        // Disabled -> None even when large.
        assert!(maybe_externalize_payload(Some(&dir), &"a".repeat(100), false, 10, &meta).is_none());
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
