// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Magic-byte MIME sniffing for the supported raster formats (hermes
//! `vision_tools.py::_detect_image_mime_type` parity, minus SVG — a text format the aux providers
//! do not accept as an image part). Deliberately hand-rolled over a sniffing crate: five prefix
//! checks need no dependency, and the sniffed type always wins over any caller-declared MIME so a
//! mislabelled `data:` URL or extension cannot smuggle a non-image payload to the provider.

/// How many leading bytes [`sniff_image_mime`] needs to classify every supported format.
pub(crate) const SNIFF_HEADER_LEN: usize = 16;

/// Classify `header` (the file's leading bytes) as a supported image MIME type, or `None` when the
/// bytes are not a recognizable PNG/JPEG/GIF/BMP/WebP image.
pub fn sniff_image_mime(header: &[u8]) -> Option<&'static str> {
    if header.starts_with(b"\x89PNG\r\n\x1a\n") {
        return Some("image/png");
    }
    if header.starts_with(b"\xff\xd8\xff") {
        return Some("image/jpeg");
    }
    if header.starts_with(b"GIF87a") || header.starts_with(b"GIF89a") {
        return Some("image/gif");
    }
    if header.starts_with(b"BM") {
        return Some("image/bmp");
    }
    if header.len() >= 12 && &header[..4] == b"RIFF" && &header[8..12] == b"WEBP" {
        return Some("image/webp");
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sniffs_every_supported_format() {
        assert_eq!(
            sniff_image_mime(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"),
            Some("image/png")
        );
        assert_eq!(
            sniff_image_mime(b"\xff\xd8\xff\xe0\x00\x10JFIF"),
            Some("image/jpeg")
        );
        assert_eq!(sniff_image_mime(b"GIF87a\x01\x00"), Some("image/gif"));
        assert_eq!(sniff_image_mime(b"GIF89a\x01\x00"), Some("image/gif"));
        assert_eq!(sniff_image_mime(b"BM\x36\x00\x00\x00"), Some("image/bmp"));
        assert_eq!(
            sniff_image_mime(b"RIFF\x24\x00\x00\x00WEBPVP8 "),
            Some("image/webp")
        );
    }

    #[test]
    fn rejects_non_images_and_short_headers() {
        assert_eq!(sniff_image_mime(b"<!doctype html><html>"), None);
        assert_eq!(sniff_image_mime(b"%PDF-1.7"), None);
        assert_eq!(sniff_image_mime(b"plain text"), None);
        assert_eq!(sniff_image_mime(b""), None);
        // Truncated magic: PNG needs all 8 signature bytes.
        assert_eq!(sniff_image_mime(b"\x89PN"), None);
        // RIFF container that is not WEBP (e.g. WAV audio).
        assert_eq!(sniff_image_mime(b"RIFF\x24\x00\x00\x00WAVEfmt "), None);
    }
}
