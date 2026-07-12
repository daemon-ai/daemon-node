// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Artifact fetch: scheme-dispatch resolution + blake3 verification (spec §8, §12).
//!
//! Everything a run references externally goes through the envelope's artifact map: a name →
//! `(url, blake3)` table the *host* fetches, verifies, and caches (the module has no I/O and
//! addresses artifacts by name). This wave wires **`file://` only**, blake3-verified on read; the
//! resolver dispatches on an [`ArtifactScheme`] enum so `r2` / `hf` / `https` slot in later
//! **without egress this wave** — `reqwest` is clippy-banned outside `daemon-egress`, so no HTTP
//! client is constructed here (those schemes return [`SwarmNetError::SchemeUnsupported`]).

use std::path::PathBuf;

use daemon_swarm_proto::blake3_hash;

use crate::seam::ContentHash;
use crate::SwarmNetError;

/// The transport scheme of an artifact URL.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArtifactScheme {
    /// `file://` — a local absolute path (wired this wave).
    File,
    /// `r2://` — a presigned R2/S3 object (reserved; awaits the egress plane).
    R2,
    /// `hf://` — a Hugging Face repo artifact, revision-pinned (reserved; awaits the egress plane).
    Hf,
    /// `https://` — a plain static host (reserved; awaits the egress plane).
    Https,
}

impl ArtifactScheme {
    /// Split `url` into its scheme + the remainder after `scheme://`.
    fn parse(url: &str) -> Result<(Self, &str), SwarmNetError> {
        let (scheme, rest) = url
            .split_once("://")
            .ok_or_else(|| SwarmNetError::BadUrl(format!("missing scheme separator: {url}")))?;
        let scheme = match scheme {
            "file" => Self::File,
            "r2" => Self::R2,
            "hf" => Self::Hf,
            "https" => Self::Https,
            other => {
                return Err(SwarmNetError::BadUrl(format!("unknown scheme: {other}")));
            }
        };
        Ok((scheme, rest))
    }
}

/// One entry of the envelope artifact map: a URL plus the blake3 it must hash to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArtifactRef {
    /// The source URL (`file://…` this wave).
    pub url: String,
    /// The blake3 the fetched bytes must match (content addressing, §8/§12).
    pub blake3: ContentHash,
}

impl ArtifactRef {
    /// Construct an artifact reference.
    pub fn new(url: impl Into<String>, blake3: ContentHash) -> Self {
        Self {
            url: url.into(),
            blake3,
        }
    }
}

/// Resolves artifact references, verifying each against its blake3.
#[derive(Clone, Copy, Debug, Default)]
pub struct ArtifactResolver;

impl ArtifactResolver {
    /// A resolver over the wired schemes (`file://` only this wave).
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Fetch `artifact` and verify its blake3. A mismatch is a typed
    /// [`SwarmNetError::HashMismatch`] (tamper/corruption reject, §12).
    pub async fn fetch(&self, artifact: &ArtifactRef) -> Result<Vec<u8>, SwarmNetError> {
        let bytes = self.fetch_raw(&artifact.url).await?;
        let actual = blake3_hash(&bytes);
        if actual != artifact.blake3 {
            return Err(SwarmNetError::HashMismatch {
                expected: artifact.blake3.to_hex(),
                actual: actual.to_hex(),
            });
        }
        Ok(bytes)
    }

    /// Fetch the raw bytes for `url`, dispatching on scheme (no verification).
    async fn fetch_raw(&self, url: &str) -> Result<Vec<u8>, SwarmNetError> {
        let (scheme, rest) = ArtifactScheme::parse(url)?;
        match scheme {
            ArtifactScheme::File => read_file_uri(rest).await,
            ArtifactScheme::R2 | ArtifactScheme::Hf | ArtifactScheme::Https => {
                Err(SwarmNetError::SchemeUnsupported(format!(
                    "{scheme:?} awaits the daemon-egress plane; only file:// is wired this wave"
                )))
            }
        }
    }
}

/// Resolve the local path of a `file://` URI's remainder (`<host>/<abs-path>`), accepting an empty
/// or `localhost` host per RFC 8089.
fn file_uri_path(rest: &str) -> Result<PathBuf, SwarmNetError> {
    // `file:///abs/path` -> rest = "/abs/path"; `file://localhost/abs` -> rest = "localhost/abs".
    let path = if let Some(stripped) = rest.strip_prefix('/') {
        // Empty host: rest began with the leading slash of the absolute path.
        format!("/{stripped}")
    } else if let Some((host, path)) = rest.split_once('/') {
        if !host.is_empty() && host != "localhost" {
            return Err(SwarmNetError::BadUrl(format!(
                "file:// host must be empty or localhost, got {host:?}"
            )));
        }
        format!("/{path}")
    } else {
        return Err(SwarmNetError::BadUrl(format!(
            "file:// url has no path: file://{rest}"
        )));
    };
    Ok(PathBuf::from(path))
}

/// Read a `file://` artifact's bytes.
///
/// `file://` artifact URLs come from the run **envelope's** artifact map — authored and signed
/// (§4.3/§8), not attacker-influenced relative paths — and the bytes are blake3-verified by
/// [`ArtifactResolver::fetch`] immediately after read. `ContainedRoot`'s relative-containment model
/// does not apply to an absolute, operator-pinned path, so this is the one sanctioned raw-fs read in
/// the crate. (Network schemes route through `daemon-egress`, never raw fs.)
#[allow(clippy::disallowed_methods)]
async fn read_file_uri(rest: &str) -> Result<Vec<u8>, SwarmNetError> {
    let path = file_uri_path(rest)?;
    tokio::fs::read(&path)
        .await
        .map_err(|e| SwarmNetError::Fetch(format!("read {}: {e}", path.display())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::temp_root;
    use daemon_core::ContainedRoot;
    use std::path::Path;

    /// The canonical blake3 test vector for the empty input (pinned golden, NET-2).
    const BLAKE3_EMPTY_HEX: &str =
        "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262";

    #[test]
    fn blake3_empty_golden() {
        assert_eq!(blake3_hash(b"").to_hex(), BLAKE3_EMPTY_HEX);
    }

    #[test]
    fn scheme_parsing() {
        assert_eq!(
            ArtifactScheme::parse("file:///a/b").unwrap(),
            (ArtifactScheme::File, "/a/b")
        );
        assert_eq!(
            ArtifactScheme::parse("hf://repo@rev/f").unwrap().0,
            ArtifactScheme::Hf
        );
        assert!(matches!(
            ArtifactScheme::parse("no-scheme"),
            Err(SwarmNetError::BadUrl(_))
        ));
    }

    /// Write a file into a temp dir via `ContainedRoot` and return its absolute path + a `file://`
    /// URL for it.
    async fn write_artifact(dir: &Path, name: &str, bytes: &[u8]) -> (PathBuf, String) {
        let root = ContainedRoot::open(dir).unwrap();
        root.write(Path::new(name), bytes).await.unwrap();
        let abs = dir.join(name);
        let url = format!("file://{}", abs.display());
        (abs, url)
    }

    #[tokio::test]
    async fn fetch_file_verifies_blake3() {
        let dir = temp_root("artifact-ok");
        let (_abs, url) = write_artifact(dir.path(), "module.wasm", b"wasm-bytes").await;
        let art = ArtifactRef::new(url, blake3_hash(b"wasm-bytes"));

        let got = ArtifactResolver::new().fetch(&art).await.unwrap();
        assert_eq!(got, b"wasm-bytes");
    }

    #[tokio::test]
    async fn fetch_file_rejects_tamper() {
        let dir = temp_root("artifact-tamper");
        let (_abs, url) = write_artifact(dir.path(), "module.wasm", b"tampered").await;
        // The artifact map claims a different blake3 than the file actually has.
        let art = ArtifactRef::new(url, blake3_hash(b"expected-original"));

        let err = ArtifactResolver::new().fetch(&art).await.unwrap_err();
        assert!(
            matches!(err, SwarmNetError::HashMismatch { .. }),
            "got {err:?}"
        );
    }

    #[tokio::test]
    async fn missing_file_is_fetch_error() {
        let art = ArtifactRef::new("file:///no/such/daemon-swarm/artifact", blake3_hash(b""));
        let err = ArtifactResolver::new().fetch(&art).await.unwrap_err();
        assert!(matches!(err, SwarmNetError::Fetch(_)), "got {err:?}");
    }

    #[tokio::test]
    async fn network_schemes_unsupported_without_egress() {
        for url in [
            "r2://bucket/obj",
            "hf://org/repo@abcdef/file",
            "https://host/x",
        ] {
            let art = ArtifactRef::new(url, blake3_hash(b""));
            let err = ArtifactResolver::new().fetch(&art).await.unwrap_err();
            assert!(
                matches!(err, SwarmNetError::SchemeUnsupported(_)),
                "{url}: got {err:?}"
            );
        }
    }
}
