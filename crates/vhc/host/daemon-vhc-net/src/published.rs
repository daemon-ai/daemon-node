// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! [`PublishedArtifact`] — the single definition of where a run's genesis-pinned objects live.
//!
//! A run's artifact plane has two sides that must agree on a key, byte for byte:
//!
//! - the **publisher** (`xtask publish-module` / `publish-corpus`) uploads each object to a
//!   content-addressed, run-relative path;
//! - the **genesis author** (`daemon_vhc_testkit::ceremony`, `daemon_vhc_testkit::live_genesis`)
//!   commits that same path as the artifact map's `url` — which is what the run-time
//!   [`PinnedArtifactStore`](crate::PinnedArtifactStore) presigns and GETs for the module-driven
//!   `data.fetch` seat.
//!
//! While each side spelled the scheme out for itself, they diverged: the ceremony authoring wrote
//! `corpus/<hash>` where the publisher writes `corpus/<hash>.cbor` / `.json` / `.bin`, so every
//! genesis-pinned corpus fetch presigned a key nothing had published, got a hard 404, and the
//! trainer guest — which treats its corpus manifest as fatal-if-absent, correctly: it IS the run's
//! data identity — trapped at init. Modules escaped only because both sides happened to spell
//! `.wasm`. So the scheme is spelled ONCE, here, and both sides derive from the object's KIND and
//! content id.
//!
//! The layout itself is unchanged and remains the deployed one (spec §8; ABI §12.7 [CC-7]):
//! `modules/<blake3>.wasm` for a module, and for the chunk-addressed corpus
//! `corpus/<manifest blake3>.cbor`, `corpus/<tokenizer blake3>.json`, `corpus/<shard fold>.bin`.
//!
//! Paths are **run-relative**: the presign surface prefixes `runs/<run>/` through the one §11.3
//! layout function ([`r2_object_key`](crate::r2_object_key)). Authoring cannot embed the run
//! prefix anyway — the run id is the blake3 of the envelope being authored.

use crate::seam::ContentHash;

/// The artifact-map url scheme for an object in the run's own payload store.
const R2_SCHEME: &str = "r2://";

/// The run-relative prefix modules publish under.
const MODULE_PREFIX: &str = "modules";

/// The run-relative prefix every corpus object publishes under.
const CORPUS_PREFIX: &str = "corpus";

/// One object the run's publisher uploads and the run's genesis pins, named by the content id the
/// guest's `data.fetch` addresses it with.
///
/// The variant IS the key scheme: it picks the prefix and the suffix, so an authoring path and a
/// publishing path that agree on the kind cannot disagree on the key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PublishedArtifact {
    /// A wasm module (`publish-module`) — `modules/<blake3>.wasm`.
    Module(ContentHash),
    /// The canonical-CBOR corpus manifest, the run's `corpus_manifest` pin — `corpus/<blake3>.cbor`.
    CorpusManifest(ContentHash),
    /// The corpus tokenizer artifact — `corpus/<blake3>.json`.
    CorpusTokenizer(ContentHash),
    /// One chunk-addressed corpus shard, keyed by its domain-separated chunk **fold** (which never
    /// equals `blake3(bytes)`) — `corpus/<fold>.bin`.
    CorpusShard(ContentHash),
}

impl PublishedArtifact {
    /// The content id the module names and the genesis artifact map records as `blake3`: the plain
    /// hash of a whole object, or a shard's fold identity.
    #[must_use]
    pub fn content_id(&self) -> ContentHash {
        match *self {
            Self::Module(h)
            | Self::CorpusManifest(h)
            | Self::CorpusTokenizer(h)
            | Self::CorpusShard(h) => h,
        }
    }

    /// The run-relative object path the publisher writes and the presign surface resolves.
    #[must_use]
    pub fn object_path(&self) -> String {
        let (prefix, ext) = match self {
            Self::Module(_) => (MODULE_PREFIX, "wasm"),
            Self::CorpusManifest(_) => (CORPUS_PREFIX, "cbor"),
            Self::CorpusTokenizer(_) => (CORPUS_PREFIX, "json"),
            Self::CorpusShard(_) => (CORPUS_PREFIX, "bin"),
        };
        format!("{prefix}/{}.{ext}", self.content_id().to_hex())
    }

    /// The artifact-map url a genesis commits for this object.
    #[must_use]
    pub fn url(&self) -> String {
        format!("{R2_SCHEME}{}", self.object_path())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use daemon_vhc_proto::blake3_hash;

    /// The four published shapes, spelled out once so a silent scheme change has to face a golden.
    #[test]
    fn every_kind_renders_its_published_key() {
        let h = blake3_hash(b"an object");
        let hex = h.to_hex();
        for (artifact, expect) in [
            (PublishedArtifact::Module(h), format!("modules/{hex}.wasm")),
            (
                PublishedArtifact::CorpusManifest(h),
                format!("corpus/{hex}.cbor"),
            ),
            (
                PublishedArtifact::CorpusTokenizer(h),
                format!("corpus/{hex}.json"),
            ),
            (
                PublishedArtifact::CorpusShard(h),
                format!("corpus/{hex}.bin"),
            ),
        ] {
            assert_eq!(artifact.object_path(), expect);
            assert_eq!(artifact.url(), format!("r2://{expect}"));
            assert_eq!(artifact.content_id(), h);
        }
    }

    /// A url is a resolvable `r2://` artifact reference: the path after the scheme is exactly what
    /// the presign surface takes as an artifact path (no leading slash, no run prefix).
    #[test]
    fn a_url_carries_the_presignable_path_verbatim() {
        let artifact = PublishedArtifact::CorpusManifest(blake3_hash(b"manifest"));
        let path = artifact
            .url()
            .strip_prefix("r2://")
            .expect("published urls are r2:// urls")
            .to_string();
        assert_eq!(path, artifact.object_path());
        assert!(!path.starts_with('/'), "the path is run-relative: {path}");
        assert_eq!(
            crate::r2_object_key(
                &crate::RunId::new("run-x"),
                &crate::PresignRequest::artifact(crate::PresignOp::Get, &path),
            )
            .expect("artifact requests carry a path"),
            format!("runs/run-x/{path}"),
        );
    }

    /// Every kind's key is distinct even at the SAME content id — the suffix is load-bearing, not
    /// decoration (this is the property whose absence made two planes point at one another's key).
    #[test]
    fn kinds_at_one_content_id_do_not_collide() {
        let h = blake3_hash(b"same bytes, four seats");
        let keys = [
            PublishedArtifact::Module(h).object_path(),
            PublishedArtifact::CorpusManifest(h).object_path(),
            PublishedArtifact::CorpusTokenizer(h).object_path(),
            PublishedArtifact::CorpusShard(h).object_path(),
        ];
        let unique: std::collections::BTreeSet<&String> = keys.iter().collect();
        assert_eq!(unique.len(), keys.len(), "{keys:?}");
    }
}
