// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Install-from-URL parsing (wire v48): turn a pasted Hugging Face link into repo coordinates,
//! node-side and strictly validated — the client never parses URLs.
//!
//! Accepted forms:
//! - a bare repo id: `org/name`
//! - a model page: `https://huggingface.co/org/name`
//! - a tree view: `https://huggingface.co/org/name/tree/<revision>[/subpath…]` (the subpath is
//!   ignored — the repo + revision is what an install needs)
//! - a direct artifact: `https://huggingface.co/org/name/resolve/<revision>/<path…>` (also
//!   `blob/`; a trailing `?download=true` is tolerated)
//!
//! Everything else — other hosts, non-https schemes, userinfo, traversal (`..`), empty or
//! percent-mangled segments — is rejected with an actionable [`ModelError::Invalid`].

use crate::error::{ModelError, Result};

/// The Hub hosts a pasted link may name.
const ALLOWED_HOSTS: &[&str] = &["huggingface.co", "www.huggingface.co", "hf.co"];

/// The repo coordinates a pasted URL resolved to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParsedHfUrl {
    /// The `org/name` repo id.
    pub repo: String,
    /// The git revision the URL pinned (`"main"` when unspecified).
    pub revision: String,
    /// The repo-relative artifact path, when the URL named a single file (`resolve/`/`blob/`).
    pub file: Option<String>,
}

/// Whether one path segment is acceptable inside a repo id or artifact path.
fn valid_segment(seg: &str) -> bool {
    !seg.is_empty()
        && seg != "."
        && seg != ".."
        && !seg.contains('\\')
        && !seg.chars().any(|c| c.is_control())
}

/// Validate an `org/name` repo id (two non-empty, traversal-free segments).
fn validate_repo(org: &str, name: &str) -> Result<String> {
    if !valid_segment(org) || !valid_segment(name) {
        return Err(ModelError::Invalid(format!(
            "not a valid Hugging Face repo id: {org:?}/{name:?}"
        )));
    }
    Ok(format!("{org}/{name}"))
}

/// Parse + validate a pasted Hugging Face URL (or bare `org/name` repo id).
pub fn parse_hf_url(input: &str) -> Result<ParsedHfUrl> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err(ModelError::Invalid("the URL is empty".into()));
    }

    // A bare repo id: exactly `org/name`, no scheme.
    if !trimmed.contains("://") {
        let mut parts = trimmed.split('/');
        let (org, name) = match (parts.next(), parts.next(), parts.next()) {
            (Some(org), Some(name), None) => (org, name),
            _ => {
                return Err(ModelError::Invalid(format!(
                    "{trimmed:?} is not an org/name repo id or a huggingface.co URL"
                )))
            }
        };
        return Ok(ParsedHfUrl {
            repo: validate_repo(org, name)?,
            revision: "main".to_string(),
            file: None,
        });
    }

    let url = url::Url::parse(trimmed)
        .map_err(|e| ModelError::Invalid(format!("not a valid URL: {e}")))?;
    if url.scheme() != "https" {
        return Err(ModelError::Invalid(format!(
            "only https URLs are accepted (got {})",
            url.scheme()
        )));
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err(ModelError::Invalid(
            "URLs with embedded credentials are not accepted".into(),
        ));
    }
    let host = url
        .host_str()
        .ok_or_else(|| ModelError::Invalid("the URL has no host".into()))?;
    if !ALLOWED_HOSTS.contains(&host.to_ascii_lowercase().as_str()) {
        return Err(ModelError::Invalid(format!(
            "{host} is not a Hugging Face host — paste a huggingface.co link"
        )));
    }
    if url.port().is_some() {
        return Err(ModelError::Invalid(
            "URLs with an explicit port are not accepted".into(),
        ));
    }

    // `Url::path_segments` yields percent-DECODED-unsafe raw segments; decode each explicitly so
    // a `%2e%2e` traversal cannot slip through as an opaque token.
    let segments: Vec<String> = url
        .path_segments()
        .map(|segs| {
            segs.filter(|s| !s.is_empty())
                .map(|s| {
                    percent_decode(s).ok_or_else(|| {
                        ModelError::Invalid(format!("undecodable path segment: {s:?}"))
                    })
                })
                .collect::<Result<Vec<_>>>()
        })
        .transpose()?
        .unwrap_or_default();

    for seg in &segments {
        if !valid_segment(seg) {
            return Err(ModelError::Invalid(format!(
                "the URL path contains an invalid segment: {seg:?}"
            )));
        }
    }

    match segments.as_slice() {
        // Model page: /org/name
        [org, name] => Ok(ParsedHfUrl {
            repo: validate_repo(org, name)?,
            revision: "main".to_string(),
            file: None,
        }),
        // Tree view: /org/name/tree/<revision>[/subpath…] — install needs repo + revision only.
        [org, name, kind, revision, ..] if kind == "tree" => Ok(ParsedHfUrl {
            repo: validate_repo(org, name)?,
            revision: revision.clone(),
            file: None,
        }),
        // Direct artifact: /org/name/(resolve|blob)/<revision>/<path…>
        [org, name, kind, revision, path @ ..]
            if (kind == "resolve" || kind == "blob") && !path.is_empty() =>
        {
            Ok(ParsedHfUrl {
                repo: validate_repo(org, name)?,
                revision: revision.clone(),
                file: Some(path.join("/")),
            })
        }
        _ => Err(ModelError::Invalid(format!(
            "unrecognized Hugging Face URL form: {}",
            url.path()
        ))),
    }
}

/// Strict percent-decoding of one path segment: rejects malformed escapes and non-UTF-8 results.
fn percent_decode(seg: &str) -> Option<String> {
    let bytes = seg.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            let hi = bytes.get(i + 1).and_then(|b| (*b as char).to_digit(16))?;
            let lo = bytes.get(i + 2).and_then(|b| (*b as char).to_digit(16))?;
            out.push(((hi << 4) | lo) as u8);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(out).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok(input: &str) -> ParsedHfUrl {
        parse_hf_url(input).expect(input)
    }

    fn rejected(input: &str) {
        assert!(
            matches!(parse_hf_url(input), Err(ModelError::Invalid(_))),
            "{input:?} must be rejected"
        );
    }

    #[test]
    fn accepts_the_documented_forms() {
        assert_eq!(
            ok("bartowski/SmolLM2-135M-Instruct-GGUF"),
            ParsedHfUrl {
                repo: "bartowski/SmolLM2-135M-Instruct-GGUF".into(),
                revision: "main".into(),
                file: None,
            }
        );
        assert_eq!(
            ok("https://huggingface.co/org/name"),
            ParsedHfUrl {
                repo: "org/name".into(),
                revision: "main".into(),
                file: None,
            }
        );
        // Tree pins the revision; a deeper subpath is ignored.
        assert_eq!(
            ok("https://huggingface.co/org/name/tree/v2/sub/dir").revision,
            "v2"
        );
        assert_eq!(
            ok("https://huggingface.co/org/name/tree/v2/sub/dir").file,
            None
        );
        // Resolve names the artifact (query string tolerated); blob is equivalent.
        let direct = ok("https://huggingface.co/org/name/resolve/main/m-Q4_K_M.gguf?download=true");
        assert_eq!(direct.file.as_deref(), Some("m-Q4_K_M.gguf"));
        assert_eq!(direct.revision, "main");
        let nested = ok("https://hf.co/org/name/blob/main/sub/dir/m.gguf");
        assert_eq!(nested.file.as_deref(), Some("sub/dir/m.gguf"));
    }

    #[test]
    fn rejects_unsafe_or_foreign_urls() {
        rejected("");
        rejected("not-a-repo");
        rejected("a/b/c");
        rejected("http://huggingface.co/org/name"); // non-https
        rejected("https://evil.example/org/name"); // foreign host
        rejected("https://user:pw@huggingface.co/org/name"); // userinfo
        rejected("https://huggingface.co:8443/org/name"); // explicit port
        rejected("https://huggingface.co/org/../name"); // traversal
        rejected("https://huggingface.co/org/name/resolve/main/%2e%2e/escape.gguf"); // encoded traversal
        rejected("https://huggingface.co/org/name/resolve/main"); // resolve without a file
        rejected("https://huggingface.co/org"); // too few segments
        rejected("https://huggingface.co/org/name/resolve/main/a%zz.gguf"); // malformed escape
    }
}
