// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The virtual corpus (architecture §6): deterministic token windows by `(peer, cursor)`.
//!
//! Windowing/batching is worker-module policy (architecture §3.2) — this only provides the raw
//! by-hash-equivalent token stream the `data@` world would fetch. Tokens are a deterministic
//! function of `(seed, peer, cursor, position)`, so every replay draws the identical window.

/// A deterministic virtual corpus: a seeded token stream over a fixed vocabulary.
#[derive(Debug, Clone)]
pub struct VirtualCorpus {
    seed: u64,
    vocab: u32,
}

impl VirtualCorpus {
    /// A corpus over `vocab` token ids, seeded by `seed`.
    #[must_use]
    pub fn new(seed: u64, vocab: u32) -> Self {
        Self {
            seed,
            vocab: vocab.max(1),
        }
    }

    /// A `len`-token window at `cursor` for `peer` — deterministic (replay draws the same window).
    #[must_use]
    pub fn window(&self, peer: usize, cursor: u64, len: usize) -> Vec<u32> {
        (0..len as u64)
            .map(|i| {
                let mut h = self
                    .seed
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add(peer as u64);
                h ^= cursor.wrapping_add(i).wrapping_mul(0xD1B5_4A32_D192_ED03);
                h = (h ^ (h >> 33)).wrapping_mul(0xFF51_AFD7_ED55_8CCD);
                (h % u64::from(self.vocab)) as u32
            })
            .collect()
    }
}
