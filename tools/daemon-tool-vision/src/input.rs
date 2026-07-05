// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Image-source classification and `data:` URL decoding.
//!
//! `vision_analyze` accepts three source shapes (hermes `vision_tools.py` parity): an http(s) URL
//! (downloaded, SSRF-guarded), a `data:` URL (decoded inline), and anything else as a
//! workspace-contained local path (a leading `file://` is stripped, mirroring hermes' resolution).
//! Classification is purely syntactic — validation (egress policy, containment, MIME sniff) happens
//! on the resolved bytes downstream.

use std::path::PathBuf;

use base64::Engine as _;

use crate::error::VisionError;

/// Where an image comes from, classified from the raw `image_url` argument.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ImageSource {
    /// A remote http(s) URL — downloaded through the SSRF-guarded fetch path.
    Http(String),
    /// A `data:` URL carrying the image inline (the full original string).
    DataUrl(String),
    /// A local path, read through the workspace-contained execution environment.
    Local(PathBuf),
}

/// Classify the raw `image_url` argument into its [`ImageSource`] shape.
pub fn classify_source(raw: &str) -> ImageSource {
    let trimmed = raw.trim();
    let lower = trimmed.to_ascii_lowercase();
    if lower.starts_with("http://") || lower.starts_with("https://") {
        ImageSource::Http(trimmed.to_string())
    } else if lower.starts_with("data:") {
        ImageSource::DataUrl(trimmed.to_string())
    } else {
        // A `file://` URI resolves as the local path it names (hermes strips the scheme the same
        // way); everything else is taken as a path verbatim.
        let path = trimmed.strip_prefix("file://").unwrap_or(trimmed);
        ImageSource::Local(PathBuf::from(path))
    }
}

/// Decode a `data:` URL into its payload bytes. The declared mediatype is deliberately ignored —
/// the magic-byte sniff on the decoded bytes is authoritative (a mislabelled `data:text/plain`
/// carrying PNG bytes is accepted; a `data:image/png` carrying HTML is rejected downstream).
pub fn parse_data_url(raw: &str) -> Result<Vec<u8>, VisionError> {
    let rest = raw
        .trim()
        .get("data:".len()..)
        .ok_or_else(|| VisionError::BadInput("malformed data: URL".to_string()))?;
    let (meta, payload) = rest.split_once(',').ok_or_else(|| {
        VisionError::BadInput("malformed data: URL (missing the ',' separator)".to_string())
    })?;
    if meta.to_ascii_lowercase().ends_with(";base64") {
        base64::engine::general_purpose::STANDARD
            .decode(payload.trim())
            .map_err(|e| VisionError::BadInput(format!("invalid base64 in data: URL: {e}")))
    } else {
        // A non-base64 data URL (plain/percent-encoded text): pass the raw bytes through — a real
        // image is always base64-encoded, so the downstream sniff rejects these as non-images.
        Ok(payload.as_bytes().to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_http_data_and_local_sources() {
        assert_eq!(
            classify_source("https://example.com/a.png"),
            ImageSource::Http("https://example.com/a.png".to_string())
        );
        assert_eq!(
            classify_source("HTTP://EXAMPLE.COM/a.png"),
            ImageSource::Http("HTTP://EXAMPLE.COM/a.png".to_string())
        );
        assert_eq!(
            classify_source("data:image/png;base64,QUJD"),
            ImageSource::DataUrl("data:image/png;base64,QUJD".to_string())
        );
        assert_eq!(
            classify_source("shots/a.png"),
            ImageSource::Local(PathBuf::from("shots/a.png"))
        );
        // `file://` strips to the local path it names.
        assert_eq!(
            classify_source("file:///ws/a.png"),
            ImageSource::Local(PathBuf::from("/ws/a.png"))
        );
        // Leading/trailing whitespace is tolerated.
        assert_eq!(
            classify_source("  https://example.com/x "),
            ImageSource::Http("https://example.com/x".to_string())
        );
    }

    #[test]
    fn parses_base64_data_url_payload() {
        let bytes = b"\x89PNG\r\n\x1a\nrest";
        let encoded = base64::engine::general_purpose::STANDARD.encode(bytes);
        let url = format!("data:image/png;base64,{encoded}");
        assert_eq!(parse_data_url(&url).unwrap(), bytes.to_vec());
    }

    #[test]
    fn passes_non_base64_payload_bytes_through() {
        // Not an image — the downstream sniff rejects it; the parse itself succeeds.
        assert_eq!(
            parse_data_url("data:text/plain,hi").unwrap(),
            b"hi".to_vec()
        );
    }

    #[test]
    fn rejects_malformed_data_urls() {
        assert!(matches!(
            parse_data_url("data:image/png;base64"),
            Err(VisionError::BadInput(_))
        ));
        assert!(matches!(
            parse_data_url("data:image/png;base64,!!!not-base64!!!"),
            Err(VisionError::BadInput(_))
        ));
    }
}
