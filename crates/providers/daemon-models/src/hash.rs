// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Artifact content hashing for provenance pinning (Phase 3 / Cluster E).
//!
//! A model artifact is pinned to the sha256 of its bytes: the pin is recorded at install (verified
//! against the Hub-declared git-LFS `oid` when present) and re-verified **before load**, so a
//! tampered / corrupted artifact is refused instead of loaded. The hash is a streaming read (a
//! multi-GB GGUF never lands in memory); callers on an async runtime wrap this in `spawn_blocking`.

use std::io::Read;
use std::path::Path;

use sha2::{Digest, Sha256};

/// The buffered read size for streaming a large artifact through the hasher.
const CHUNK: usize = 1 << 20; // 1 MiB

/// Lowercase-hex sha256 of the file at `path`, read in bounded chunks (never fully buffered).
pub fn sha256_file(path: &Path) -> std::io::Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; CHUNK];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hashes_a_file_streaming() {
        let path =
            std::env::temp_dir().join(format!("daemon-models-hash-{}.bin", std::process::id()));
        // 3 MiB + a tail so more than one chunk boundary is exercised.
        let bytes = vec![0x5Au8; (CHUNK * 3) + 7];
        std::fs::write(&path, &bytes).unwrap();

        let got = sha256_file(&path).unwrap();
        // Cross-check against a one-shot hash of the same bytes.
        let mut h = Sha256::new();
        h.update(&bytes);
        let want: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(got, want);
        assert_eq!(got.len(), 64, "sha256 hex is 64 chars");

        let _ = std::fs::remove_file(&path);
    }
}
