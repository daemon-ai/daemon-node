// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Encrypted, content-addressed sidecars for large read-back values (ABI companion §8.5).
//!
//! A `read_back` value whose plaintext exceeds [`daemon_vhc_abi::READBACK_INLINE_MAX`] is stored as a
//! **sidecar**: a separate file named by the blake3 of its plaintext, referenced from the record as a
//! [`SidecarRef`](super::record::SidecarRef). Sidecars can hold private model state, so they are
//! encrypted at rest under this concrete profile (§8.5):
//!
//! * **AEAD: XChaCha20-Poly1305.**
//! * **Key scope: one key per journal**, held node-locally (`daemon-credentials`) and NEVER written
//!   to the journal or any sidecar. This crate does **not** invent key storage: the key enters via a
//!   pluggable [`KeyProvider`] (the node supplies it at construction — [`StaticKey`] is the trivial
//!   "here is the key" provider).
//! * **Nonce (exact):** the 24-byte nonce is `LE64(ord) || LE64(instantiation_counter) || LE64(0)`
//!   where `ord` is the referencing record's journal ordinal. Ordinals are journal-global + monotone
//!   and the key is journal-scoped, so a `(key, nonce)` pair is never reused; the instantiation
//!   counter is belt-and-braces against ordinal reuse after an unnoticed truncation.
//! * **File layout:** `magic "DVHCSC01" || u32-LE len || sidecar-header (canonical CBOR) ||
//!   ciphertext || 16-byte Poly1305 tag`, with the header as **AAD**. On read: verify the AEAD tag
//!   (header as AAD), decrypt, then verify the plaintext blake3 against `hash`. The execution
//!   identity in the header makes ownership explicit and prevents cross-journal splicing.

// Sanctioned raw-fs home (see journal/mod.rs): host-owned sidecar dir + atomic temp-write/rename +
// dir fsync durability the ContainedRoot API does not model. No spawn / env mutation here.
#![allow(clippy::disallowed_methods)]

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use chacha20poly1305::aead::{Aead, KeyInit, Payload};
use chacha20poly1305::{Key, XChaCha20Poly1305, XNonce};
use ciborium::value::Value;

use daemon_vhc_abi::JOURNAL_SIDECAR_MAGIC;
use daemon_vhc_proto::{blake3_hash, to_canonical_vec, Hash};

use super::record::{ExecIdentity, SidecarRef};
use super::{fsync_dir, JournalError};

/// Supplies the per-journal XChaCha20-Poly1305 key (§8.5).
///
/// The key is a **node-local secret** (`daemon-credentials`), one per run-instance journal, generated
/// fresh at journal creation and never persisted into the journal or a sidecar. This trait is the
/// seam: the substrate takes the key as an input, it never stores or mints one.
pub trait KeyProvider {
    /// The 32-byte journal key.
    fn journal_key(&self) -> [u8; 32];
}

/// The trivial [`KeyProvider`]: the node hands the key in directly (from `daemon-credentials`).
#[derive(Clone)]
pub struct StaticKey([u8; 32]);

impl StaticKey {
    /// Wrap a caller-provided 32-byte journal key.
    #[must_use]
    pub fn new(key: [u8; 32]) -> Self {
        Self(key)
    }
}

impl KeyProvider for StaticKey {
    fn journal_key(&self) -> [u8; 32] {
        self.0
    }
}

/// The §8.5 nonce: `LE64(ord) || LE64(instantiation_counter) || LE64(0)` (24 bytes).
fn nonce(ord: u64, instantiation_counter: u64) -> [u8; 24] {
    let mut n = [0u8; 24];
    n[0..8].copy_from_slice(&ord.to_le_bytes());
    n[8..16].copy_from_slice(&instantiation_counter.to_le_bytes());
    // n[16..24] is LE64(0) — already zero.
    n
}

/// The `sidecar-header` (ABI §8.5 / journal.cddl), the AEAD AAD.
fn header_cbor(
    id: &ExecIdentity,
    ord: u64,
    hash: &Hash,
    size: u64,
) -> Result<Vec<u8>, JournalError> {
    let map = Value::Map(vec![
        (
            Value::Text("run_id".into()),
            Value::Bytes(id.run_id.as_bytes().to_vec()),
        ),
        (Value::Text("epoch".into()), Value::Integer(id.epoch.into())),
        (Value::Text("role".into()), Value::Text(id.role.clone())),
        (
            Value::Text("instance".into()),
            Value::Integer(id.instance.into()),
        ),
        (
            Value::Text("module".into()),
            Value::Bytes(id.module.as_bytes().to_vec()),
        ),
        (Value::Text("ord".into()), Value::Integer(ord.into())),
        (
            Value::Text("hash".into()),
            Value::Bytes(hash.as_bytes().to_vec()),
        ),
        (Value::Text("size".into()), Value::Integer(size.into())),
    ]);
    to_canonical_vec(&map).map_err(|e| JournalError::Codec(format!("encode sidecar header: {e}")))
}

/// An encrypted, content-addressed sidecar store rooted at a `sidecars/` directory (§8.5).
pub struct SidecarStore<K: KeyProvider> {
    dir: PathBuf,
    id: ExecIdentity,
    key: K,
}

impl<K: KeyProvider> SidecarStore<K> {
    /// Open a sidecar store over `dir`, owned by execution identity `id`, keyed by `key`.
    ///
    /// # Errors
    /// [`JournalError::Io`] if the directory cannot be created.
    pub fn open(dir: impl AsRef<Path>, id: ExecIdentity, key: K) -> Result<Self, JournalError> {
        let dir = dir.as_ref().to_path_buf();
        fs::create_dir_all(&dir)?;
        Ok(Self { dir, id, key })
    }

    fn path_for(&self, hash: &Hash) -> PathBuf {
        self.dir.join(format!("{}.dvhcsc", hash.to_hex()))
    }

    /// Encrypt + store a plaintext value, returning its [`SidecarRef`] (§8.5). Content-addressed: a
    /// value already present (same plaintext hash) is not rewritten.
    ///
    /// # Errors
    /// [`JournalError::Codec`]/[`JournalError::Sidecar`]/[`JournalError::Io`] on encode/encrypt/write
    /// failure.
    pub fn put(
        &self,
        ord: u64,
        instantiation_counter: u64,
        seg: u64,
        plaintext: &[u8],
    ) -> Result<SidecarRef, JournalError> {
        let hash = blake3_hash(plaintext);
        let size = plaintext.len() as u64;
        let sref = SidecarRef { hash, size, seg };
        let path = self.path_for(&hash);
        if path.exists() {
            return Ok(sref); // content-addressed: identical plaintext -> identical file.
        }
        let header = header_cbor(&self.id, ord, &hash, size)?;
        let cipher = XChaCha20Poly1305::new(Key::from_slice(&self.key.journal_key()));
        let n = nonce(ord, instantiation_counter);
        let ciphertext = cipher
            .encrypt(
                XNonce::from_slice(&n),
                Payload {
                    msg: plaintext,
                    aad: &header,
                },
            )
            .map_err(|_| JournalError::Sidecar("XChaCha20-Poly1305 encrypt failed".into()))?;

        let mut out = Vec::with_capacity(8 + 4 + header.len() + ciphertext.len());
        out.extend_from_slice(JOURNAL_SIDECAR_MAGIC);
        let hlen = u32::try_from(header.len())
            .map_err(|_| JournalError::Codec("sidecar header exceeds u32".into()))?;
        out.extend_from_slice(&hlen.to_le_bytes());
        out.extend_from_slice(&header);
        out.extend_from_slice(&ciphertext); // ciphertext || 16-byte Poly1305 tag (AEAD output).

        // Write to a temp file then atomically rename so a reader never sees a partial sidecar.
        let tmp = path.with_extension("dvhcsc.tmp");
        {
            let mut f = fs::File::create(&tmp)?;
            io::Write::write_all(&mut f, &out)?;
            f.sync_all()?;
        }
        fs::rename(&tmp, &path)?;
        fsync_dir(&self.dir)?;
        Ok(sref)
    }

    /// Whether a sidecar for `sref` is present on disk.
    #[must_use]
    pub fn contains(&self, sref: &SidecarRef) -> bool {
        self.path_for(&sref.hash).exists()
    }

    /// Fetch + decrypt a sidecar (§8.5): verify the AEAD tag (header as AAD), decrypt with the
    /// `(ord, instantiation_counter)` nonce, then verify the plaintext blake3 against `sref.hash`.
    ///
    /// Returns [`SidecarMissing`] if the file is absent (the replay layer maps this to the typed
    /// `ReplayMissingPayload` outcome, §8.7 — never a silent divergence).
    ///
    /// # Errors
    /// [`SidecarError::Missing`] if absent; [`SidecarError::Verify`] on any AEAD / content-address /
    /// ownership failure; [`SidecarError::Io`] on read failure.
    pub fn get(
        &self,
        sref: &SidecarRef,
        ord: u64,
        instantiation_counter: u64,
    ) -> Result<Vec<u8>, SidecarError> {
        let path = self.path_for(&sref.hash);
        let bytes = match fs::read(&path) {
            Ok(b) => b,
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                return Err(SidecarError::Missing { hash: sref.hash })
            }
            Err(e) => return Err(SidecarError::Io(e)),
        };
        let mut off = 0usize;
        if bytes.len() < 12 || &bytes[0..8] != JOURNAL_SIDECAR_MAGIC.as_slice() {
            return Err(SidecarError::Verify("bad sidecar magic".into()));
        }
        off += 8;
        let hlen = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) as usize;
        off += 4;
        if off + hlen > bytes.len() {
            return Err(SidecarError::Verify("truncated sidecar header".into()));
        }
        let header = &bytes[off..off + hlen];
        off += hlen;
        let ciphertext = &bytes[off..];

        // Recompute the header we expect for this owner+ord and verify the stored header matches it,
        // so a sidecar spliced from another journal (different execution identity) is rejected before
        // decryption — and its AAD would fail anyway (belt and braces).
        let expected = header_cbor(&self.id, ord, &sref.hash, sref.size)
            .map_err(|e| SidecarError::Verify(format!("rebuild header: {e}")))?;
        if header != expected.as_slice() {
            return Err(SidecarError::Verify(
                "sidecar header does not match the owning execution identity + ordinal".into(),
            ));
        }

        let cipher = XChaCha20Poly1305::new(Key::from_slice(&self.key.journal_key()));
        let n = nonce(ord, instantiation_counter);
        let plaintext = cipher
            .decrypt(
                XNonce::from_slice(&n),
                Payload {
                    msg: ciphertext,
                    aad: header,
                },
            )
            .map_err(|_| SidecarError::Verify("XChaCha20-Poly1305 auth failed".into()))?;

        if blake3_hash(&plaintext) != sref.hash {
            return Err(SidecarError::Verify(
                "sidecar plaintext blake3 does not match its content address".into(),
            ));
        }
        Ok(plaintext)
    }
}

/// Why a sidecar fetch failed (§8.5, §8.7).
#[derive(Debug, thiserror::Error)]
pub enum SidecarError {
    /// The referenced sidecar file is absent — the replay layer maps this to `ReplayMissingPayload`
    /// (§8.7), identifying the hash; the run is reported incomplete, never a pass.
    #[error("sidecar missing: {}", hash.to_hex())]
    Missing {
        /// The content address that could not be fetched.
        hash: Hash,
    },
    /// AEAD / content-address / ownership verification failed.
    #[error("sidecar verification failed: {0}")]
    Verify(String),
    /// A read error.
    #[error("sidecar io error: {0}")]
    Io(#[from] io::Error),
}
