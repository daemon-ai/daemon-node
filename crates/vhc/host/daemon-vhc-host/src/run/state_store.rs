// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The **host state store** (ABI §12.14 [SF-4]/[SF-7]; architecture: the chunk-addressed
//! det-state contract): the per-instance, chunk-addressed home of canonical det-lane state the
//! guest writes through `vhc@2::state_open`/`state_emit`/`state_seal` and reads back through
//! `data@2::fetch` ([SF-R1] — a self-sealed fold is registered by construction).
//!
//! ## Custody model
//!
//! Chunks are **content-addressed** (`blake3(chunk)` → bytes) and the host hashes them itself at
//! `state_emit` — the bytes never leave host custody between emit and fetch, so the emit is the
//! verification point (the corpus posture "the pump is the sole verifier", with the pump doing
//! its verifying at the write). Fetch assembly walks the sealed fold's ordered `(hash, len)`
//! list and re-hashes the served span as a custody cross-check.
//!
//! ## Per-parameter chunking ⇒ per-chunk lengths
//!
//! Det-state families are chunked **per parameter** (a parameter never spans a chunk boundary;
//! its last chunk may be short — [SF-1]), so a family's interior chunks are NOT a uniform grid
//! and the corpus `covering_span` arithmetic does not apply. The store records each emitted
//! chunk's length; range→chunk resolution walks the actual offsets. (The host does not know the
//! parameter layout — framing is deliberately coarse, `0 < len ≤ chunk_size` per emit and
//! `Σ len == byte_len` at seal; exact per-parameter alignment is a fold-identity concern the
//! peer digest / `expected_root` cross-checks catch, not a host trap.)
//!
//! ## Durability rule (torn folds)
//!
//! Only `state_seal` mints a durable, fetchable, retained artifact. An opened-but-unsealed
//! stream's chunks are garbage-collected when the stream population clears (instance teardown /
//! trap force-reclaim), and the store itself is instance-scoped — a crash mid-fold leaves
//! nothing a restart can observe ([SF-4] crash rule).
//!
//! ## Retention ([SF-7], design §8.2)
//!
//! Sealed folds are retained per family under `state_retain_roots` (`0` = unbounded by this
//! grant; default [`daemon_vhc_proto::STATE_RETAIN_ROOTS_DEFAULT`]), plus every **pinned** fold
//! (checkpoint-referenced / init artifacts — the pin API the checkpoint wave drives). Eviction
//! is per-artifact, oldest-seal first; chunks are refcounted, so identical chunks shared across
//! rounds/folds are stored once and survive until their last holder goes.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use daemon_vhc_proto::family_fold;

/// The top bit set on guest-created state stream ids — the same never-collides-with-host-ids
/// namespace discipline as guest staging ids (ABI §10.2), with its own per-instance counter.
pub const STATE_STREAM_ID_TOP_BIT: u64 = 1 << 63;

/// A typed state-store refusal — the linker maps each onto its trap
/// (`crate::trap::TrapCode`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StateStoreError {
    /// The state plane is not provisioned for this run (no genesis state contract ⇒
    /// `state_chunk_size == 0`).
    NotProvisioned,
    /// `state-streams-max`: the concurrent-open-streams grant is exhausted.
    StreamsExhausted {
        /// The admitted bound.
        max: u64,
    },
    /// The stream id names no open stream (never issued, already sealed, or abandoned).
    UnknownStream,
    /// A misframed emit ([SF-4]): `len == 0`, `len > chunk_size`, a per-emit grant breach is
    /// NOT this (that is [`StateStoreError::WriteBudget`]), or the emit would exceed the
    /// stream's declared `byte_len`.
    MisframedEmit {
        /// What was wrong, human-readable.
        detail: String,
    },
    /// `state-write-budget`: the per-emit byte ceiling or the token bucket refused the write.
    WriteBudget {
        /// What was exhausted, human-readable.
        detail: String,
    },
    /// An incomplete seal ([SF-4]): bytes emitted ≠ the declared `byte_len`.
    IncompleteSeal {
        /// Bytes emitted so far.
        emitted: u64,
        /// The declared stream length.
        declared: u64,
    },
    /// `state-store-bytes`: sealing this fold would exceed the live retained-byte grant even
    /// after retention eviction; the seal is refused and rolled back (nothing was retained).
    StoreBytes {
        /// Live retained bytes after the attempted seal.
        retained: u64,
        /// The admitted ceiling.
        max: u64,
    },
    /// `state_open` declared a zero-length family (nothing to fold).
    EmptyFamily,
}

impl std::fmt::Display for StateStoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotProvisioned => write!(
                f,
                "the state plane is not provisioned for this run (no genesis state contract)"
            ),
            Self::StreamsExhausted { max } => {
                write!(f, "state-streams-max grant exhausted ({max} open streams)")
            }
            Self::UnknownStream => write!(f, "the stream id names no open state stream"),
            Self::MisframedEmit { detail } => write!(f, "misframed state_emit: {detail}"),
            Self::WriteBudget { detail } => write!(f, "state-write-budget: {detail}"),
            Self::IncompleteSeal { emitted, declared } => write!(
                f,
                "incomplete seal: {emitted} of the declared {declared} bytes were emitted"
            ),
            Self::StoreBytes { retained, max } => write!(
                f,
                "state-store-bytes grant exceeded: {retained} live retained bytes > {max} \
                 (after retention eviction)"
            ),
            Self::EmptyFamily => write!(f, "state_open declared a zero-length family"),
        }
    }
}

/// The admitted state-plane bounds one run instance enforces (`0` = unbounded by this grant,
/// the ABI §2.3 convention — except `chunk_size`, where `0` = the state plane is not
/// provisioned at all).
#[derive(Debug, Clone, Copy)]
pub struct StateStoreConfig {
    /// The run-pinned `state_chunk_size` (genesis state contract; `0` = state plane disabled).
    pub chunk_size: u64,
    /// `state-streams-max`: concurrent open write streams.
    pub streams_max: u64,
    /// `state-write-budget.max_bytes`: the per-emit byte ceiling.
    pub emit_max_bytes: u64,
    /// `state-write-budget.rate_per_min`: the token-bucket write rate (raw bytes per minute;
    /// bucket capacity is one minute's worth). Live-pump enforcement only — replay is not the
    /// budget gate (the recording already enforced it), the same posture as the epoch watchdog.
    pub write_rate_per_min: u64,
    /// `state-store-bytes`: the live retained-byte ceiling across sealed families.
    pub store_bytes_max: u64,
    /// `state_retain_roots`: sealed roots retained per family (`0` = unbounded).
    pub retain_roots: u64,
}

impl Default for StateStoreConfig {
    fn default() -> Self {
        Self {
            chunk_size: 0,
            streams_max: 0,
            emit_max_bytes: 0,
            write_rate_per_min: 0,
            store_bytes_max: 0,
            retain_roots: daemon_vhc_proto::STATE_RETAIN_ROOTS_DEFAULT,
        }
    }
}

/// One content-addressed chunk object plus its holder counts.
struct ChunkEntry {
    bytes: Arc<Vec<u8>>,
    /// Total holders: open streams + sealed folds (per occurrence).
    refs: u64,
    /// Sealed-fold holders only — the retained-byte meter counts a chunk while this is > 0.
    sealed_refs: u64,
}

/// One open (unsealed) write stream: emitted chunks in order, nothing durable yet.
struct OpenStream {
    tag: String,
    byte_len: u64,
    emitted: u64,
    /// `(chunk blake3, chunk len)` in emit order — lengths are load-bearing (module docs).
    chunks: Vec<([u8; 32], u32)>,
}

/// One sealed family artifact: the fold identity's full geometry, fetchable by construction
/// ([SF-R1]).
#[derive(Debug, Clone)]
pub struct SealedFold {
    /// The family tag the stream was opened with.
    pub tag: String,
    /// The family byte length.
    pub byte_len: u64,
    /// `(chunk blake3, chunk len)` in fold order.
    pub chunks: Vec<([u8; 32], u32)>,
}

/// Introspection snapshot for tests / the fleet-preflight disk line item.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StateStoreStats {
    /// Sealed (retained) folds.
    pub sealed_folds: usize,
    /// Open (unsealed) streams.
    pub open_streams: usize,
    /// Live retained bytes (unique chunk bytes held by at least one sealed fold).
    pub retained_bytes: u64,
    /// Unique chunk objects live (sealed or stream-staged).
    pub chunk_objects: usize,
}

/// The per-instance host state store (see module docs). Lives in the pump state beside the
/// buffer/op/stream tables; every method runs under the pump lock.
pub struct StateStore {
    cfg: StateStoreConfig,
    chunks: HashMap<[u8; 32], ChunkEntry>,
    open: HashMap<u64, OpenStream>,
    next_stream: u64,
    sealed: HashMap<[u8; 32], SealedFold>,
    /// Seal order per family tag (eviction is oldest-seal first within a family).
    family_order: HashMap<String, VecDeque<[u8; 32]>>,
    /// Pinned folds (checkpoint-referenced / init artifacts): exempt from retention eviction.
    pinned: std::collections::HashSet<[u8; 32]>,
    /// Live retained bytes (unique chunks with `sealed_refs > 0`).
    retained_bytes: u64,
    /// The write token bucket: available bytes + the last refill instant (logical pump ms).
    bucket_bytes: u64,
    bucket_at_ms: u64,
}

impl StateStore {
    /// A fresh store under the admitted bounds. The token bucket opens full (one minute's
    /// worth) at logical time zero.
    #[must_use]
    pub fn new(cfg: StateStoreConfig) -> Self {
        Self {
            cfg,
            chunks: HashMap::new(),
            open: HashMap::new(),
            next_stream: 1,
            sealed: HashMap::new(),
            family_order: HashMap::new(),
            pinned: std::collections::HashSet::new(),
            retained_bytes: 0,
            bucket_bytes: cfg.write_rate_per_min,
            bucket_at_ms: 0,
        }
    }

    /// Open a family write stream ([SF-4] `state_open`): returns the counter-deterministic
    /// stream id (top-bit namespace — a pure function of guest call order, `dc` class).
    ///
    /// # Errors
    /// [`StateStoreError::NotProvisioned`] / [`StateStoreError::EmptyFamily`] /
    /// [`StateStoreError::StreamsExhausted`].
    pub fn open(&mut self, tag: &str, byte_len: u64) -> Result<u64, StateStoreError> {
        if self.cfg.chunk_size == 0 {
            return Err(StateStoreError::NotProvisioned);
        }
        if byte_len == 0 {
            return Err(StateStoreError::EmptyFamily);
        }
        if self.cfg.streams_max != 0 && self.open.len() as u64 >= self.cfg.streams_max {
            return Err(StateStoreError::StreamsExhausted {
                max: self.cfg.streams_max,
            });
        }
        let id = STATE_STREAM_ID_TOP_BIT | self.next_stream;
        self.next_stream += 1;
        self.open.insert(
            id,
            OpenStream {
                tag: tag.to_string(),
                byte_len,
                emitted: 0,
                chunks: Vec::new(),
            },
        );
        Ok(id)
    }

    /// Emit one chunk ([SF-4] `state_emit`): coarse framing (`0 < len ≤ chunk_size`, never past
    /// the declared `byte_len`), the write budget, then copy + hash + store content-addressed.
    /// Returns the chunk ordinal (0-based emit index — under the pinned schedule, the family
    /// chunk ordinal).
    ///
    /// `now_ms` is the pump's logical clock (token-bucket refill); pass `0` where no budget is
    /// configured (replay, tests).
    ///
    /// # Errors
    /// [`StateStoreError::UnknownStream`] / [`StateStoreError::MisframedEmit`] /
    /// [`StateStoreError::WriteBudget`].
    pub fn emit(&mut self, stream: u64, bytes: &[u8], now_ms: u64) -> Result<u64, StateStoreError> {
        let chunk_size = self.cfg.chunk_size;
        let emit_max = self.cfg.emit_max_bytes;
        let len = bytes.len() as u64;
        {
            let st = self
                .open
                .get(&stream)
                .ok_or(StateStoreError::UnknownStream)?;
            // Coarse framing ([SF-4] as ratified): the host checks integrity-and-bounds shape
            // only — exact per-parameter tails are a fold-identity concern (module docs).
            if len == 0 {
                return Err(StateStoreError::MisframedEmit {
                    detail: "an empty chunk".into(),
                });
            }
            if len > chunk_size {
                return Err(StateStoreError::MisframedEmit {
                    detail: format!("chunk of {len} bytes > state chunk_size {chunk_size}"),
                });
            }
            if st.emitted + len > st.byte_len {
                return Err(StateStoreError::MisframedEmit {
                    detail: format!(
                        "emit runs past the declared byte_len ({} + {len} > {})",
                        st.emitted, st.byte_len
                    ),
                });
            }
        }
        // The write budget ([SF-7] `state-write-budget`): per-emit ceiling + token bucket.
        if emit_max != 0 && len > emit_max {
            return Err(StateStoreError::WriteBudget {
                detail: format!("emit of {len} bytes > per-emit ceiling {emit_max}"),
            });
        }
        if self.cfg.write_rate_per_min != 0 {
            self.refill_bucket(now_ms);
            if len > self.bucket_bytes {
                return Err(StateStoreError::WriteBudget {
                    detail: format!(
                        "write token bucket exhausted ({} of {} bytes/min available, {len} \
                         requested)",
                        self.bucket_bytes, self.cfg.write_rate_per_min
                    ),
                });
            }
            self.bucket_bytes -= len;
        }
        // Copy out of guest memory happened at the import (the caller hands owned bytes a
        // slice of); hash + store content-addressed.
        let hash = *blake3::hash(bytes).as_bytes();
        self.hold_chunk(hash, bytes);
        let st = self.open.get_mut(&stream).expect("stream checked open");
        st.emitted += len;
        st.chunks.push((hash, bytes.len() as u32));
        Ok(st.chunks.len() as u64 - 1)
    }

    /// Seal a stream ([SF-4] `state_seal`): completeness check, the domain-separated family
    /// fold over the accumulated chunk hashes, registration as a retained + fetchable artifact
    /// ([SF-R1]), retention eviction ([SF-7]), and the `state-store-bytes` gate. On a
    /// store-bytes refusal the seal **rolls back** — nothing stays retained, the stream is
    /// closed (its chunks are released), and the caller traps typed.
    ///
    /// Returns the 32-byte family fold.
    ///
    /// # Errors
    /// [`StateStoreError::UnknownStream`] / [`StateStoreError::IncompleteSeal`] /
    /// [`StateStoreError::StoreBytes`].
    pub fn seal(&mut self, stream: u64) -> Result<[u8; 32], StateStoreError> {
        {
            let st = self
                .open
                .get(&stream)
                .ok_or(StateStoreError::UnknownStream)?;
            if st.emitted != st.byte_len {
                return Err(StateStoreError::IncompleteSeal {
                    emitted: st.emitted,
                    declared: st.byte_len,
                });
            }
        }
        let st = self.open.remove(&stream).expect("stream checked open");
        let hashes: Vec<daemon_vhc_proto::Hash> = st
            .chunks
            .iter()
            .map(|(h, _)| daemon_vhc_proto::Hash(*h))
            .collect();
        let fold = family_fold(self.cfg.chunk_size, st.byte_len, &hashes).0;
        if self.sealed.contains_key(&fold) {
            // Re-sealing identical content (the same round re-derived): the artifact already
            // exists; release the stream's duplicate holds and return the same identity —
            // content addressing makes this a no-op, not an error.
            for (h, _) in &st.chunks {
                self.release_chunk(*h);
            }
            return Ok(fold);
        }
        // Register: the stream's holds transfer to the fold; sealed_refs meter retained bytes.
        for (h, _) in &st.chunks {
            let e = self.chunks.get_mut(h).expect("stream chunk held");
            e.sealed_refs += 1;
            if e.sealed_refs == 1 {
                self.retained_bytes += e.bytes.len() as u64;
            }
        }
        self.sealed.insert(
            fold,
            SealedFold {
                tag: st.tag.clone(),
                byte_len: st.byte_len,
                chunks: st.chunks,
            },
        );
        self.family_order
            .entry(st.tag.clone())
            .or_default()
            .push_back(fold);
        // Retention ([SF-7]): oldest unpinned folds beyond `state_retain_roots`, per family.
        self.evict_family(&st.tag);
        // The store-bytes gate: if retention could not bring us under the ceiling, the seal
        // itself is refused — rolled back so nothing durable remains.
        if self.cfg.store_bytes_max != 0 && self.retained_bytes > self.cfg.store_bytes_max {
            let retained = self.retained_bytes;
            self.evict_fold(fold);
            return Err(StateStoreError::StoreBytes {
                retained,
                max: self.cfg.store_bytes_max,
            });
        }
        Ok(fold)
    }

    /// Whether `hash` names a sealed (retained) fold — the [SF-R1] "registered by construction"
    /// membership test the fetch path consults before the grant check.
    #[must_use]
    pub fn sealed(&self, hash: &[u8; 32]) -> Option<&SealedFold> {
        self.sealed.get(hash)
    }

    /// The `(chunk hash, chunk bytes)` list of a sealed fold in fold order — what a checkpoint
    /// publisher (or the golden harness standing in for the payload plane) uploads content-addressed
    /// so a restoring instance's chunk-keyed [SF-R2] fetch resolves. `None` if the fold is not
    /// sealed here.
    #[must_use]
    pub fn sealed_chunks(&self, fold: &[u8; 32]) -> Option<Vec<(daemon_vhc_proto::Hash, Vec<u8>)>> {
        let sealed = self.sealed.get(fold)?;
        let mut out = Vec::with_capacity(sealed.chunks.len());
        for (h, _len) in &sealed.chunks {
            let entry = self.chunks.get(h)?;
            out.push((daemon_vhc_proto::Hash(*h), (*entry.bytes).clone()));
        }
        Some(out)
    }

    /// Reconstruct the [`daemon_vhc_proto::det_state::FamilyRef`] a by-reference checkpoint
    /// section carries (design §7.2, [SF-6]) from the store's own record of a self-sealed fold.
    /// `None` when the fold is not sealed here. The host is the authority for its self-sealed
    /// folds' geometry: the ordered chunk hashes were recorded on emit and `chunk_size` is the
    /// run-pinned state-contract value, so the reconstructed ref re-derives exactly `fold` (it
    /// validates by construction — `seal` mints the identity with the same `family_fold` inputs).
    /// This is what lets the DRAIN path carry by-ref sections without the guest re-listing chunk
    /// hashes it already emitted: the guest names the sealed fold, the host fills the geometry.
    #[must_use]
    pub fn sealed_family_ref(
        &self,
        fold: &[u8; 32],
    ) -> Option<daemon_vhc_proto::det_state::FamilyRef> {
        let sealed = self.sealed.get(fold)?;
        Some(daemon_vhc_proto::det_state::FamilyRef {
            fold: daemon_vhc_proto::Hash(*fold),
            byte_len: sealed.byte_len,
            chunk_size: self.cfg.chunk_size,
            chunk_hashes: sealed
                .chunks
                .iter()
                .map(|(h, _)| daemon_vhc_proto::Hash(*h))
                .collect(),
        })
    }

    /// Assemble the byte range `[off, end)` of a sealed fold from its content-addressed chunks,
    /// re-hashing each contributing chunk (custody cross-check). `None` when the fold is
    /// unknown; `Err` describes an out-of-bounds range or a custody violation (impossible
    /// unless memory corrupted — surfaced loudly, never silently).
    pub fn read_range(
        &self,
        fold: &[u8; 32],
        off: u64,
        end: u64,
    ) -> Option<Result<Vec<u8>, String>> {
        let sealed = self.sealed.get(fold)?;
        if off > sealed.byte_len || end > sealed.byte_len || off > end {
            return Some(Err(format!(
                "range [{off}, {end}) out of bounds (sealed family is {} bytes)",
                sealed.byte_len
            )));
        }
        let mut out = Vec::with_capacity((end - off) as usize);
        let mut cursor = 0u64;
        for (hash, len) in &sealed.chunks {
            let chunk_start = cursor;
            let chunk_end = cursor + u64::from(*len);
            cursor = chunk_end;
            if chunk_end <= off {
                continue;
            }
            if chunk_start >= end {
                break;
            }
            let Some(entry) = self.chunks.get(hash) else {
                return Some(Err("sealed fold references an evicted chunk".into()));
            };
            if blake3::hash(&entry.bytes).as_bytes() != hash {
                return Some(Err("custody violation: chunk bytes do not re-hash".into()));
            }
            let lo = off.saturating_sub(chunk_start) as usize;
            let hi = (end.min(chunk_end) - chunk_start) as usize;
            out.extend_from_slice(&entry.bytes[lo..hi]);
        }
        Some(Ok(out))
    }

    /// Pin a fold out of retention eviction (checkpoint-referenced / init artifacts, design
    /// §8.2 — driven by the checkpoint wave; exposed now so retention is complete). Unknown
    /// folds pin harmlessly (the pin holds if the fold seals later — idempotent set semantics).
    pub fn pin(&mut self, fold: [u8; 32]) {
        self.pinned.insert(fold);
    }

    /// Remove a pin (the checkpoint slot moved on); the fold re-enters ordinary retention at
    /// the NEXT seal of its family (eviction runs at seals, never spontaneously).
    pub fn unpin(&mut self, fold: &[u8; 32]) {
        self.pinned.remove(fold);
    }

    /// Force-reclaim every open (unsealed) stream — the torn-fold GC ([SF-4] crash rule),
    /// run at instance teardown beside the buffer/op/stream force-reclaims. Sealed artifacts
    /// are instance-scoped anyway (the store dies with the pump); this exists so a live
    /// teardown path observably drops staged-but-unsealed chunks.
    pub fn clear_open(&mut self) {
        let streams: Vec<u64> = self.open.keys().copied().collect();
        for s in streams {
            let st = self.open.remove(&s).expect("keyed");
            for (h, _) in &st.chunks {
                self.release_chunk(*h);
            }
        }
    }

    /// Introspection (tests, preflight disk/RAM line items).
    #[must_use]
    pub fn stats(&self) -> StateStoreStats {
        StateStoreStats {
            sealed_folds: self.sealed.len(),
            open_streams: self.open.len(),
            retained_bytes: self.retained_bytes,
            chunk_objects: self.chunks.len(),
        }
    }

    // -- internals --------------------------------------------------------------------------

    fn refill_bucket(&mut self, now_ms: u64) {
        let rate = self.cfg.write_rate_per_min;
        let elapsed = now_ms.saturating_sub(self.bucket_at_ms);
        self.bucket_at_ms = now_ms;
        let refill = rate.saturating_mul(elapsed) / 60_000;
        self.bucket_bytes = self.bucket_bytes.saturating_add(refill).min(rate);
    }

    fn hold_chunk(&mut self, hash: [u8; 32], bytes: &[u8]) {
        match self.chunks.entry(hash) {
            std::collections::hash_map::Entry::Occupied(mut e) => e.get_mut().refs += 1,
            std::collections::hash_map::Entry::Vacant(e) => {
                e.insert(ChunkEntry {
                    bytes: Arc::new(bytes.to_vec()),
                    refs: 1,
                    sealed_refs: 0,
                });
            }
        }
    }

    fn release_chunk(&mut self, hash: [u8; 32]) {
        if let Some(e) = self.chunks.get_mut(&hash) {
            e.refs -= 1;
            if e.refs == 0 {
                self.chunks.remove(&hash);
            }
        }
    }

    /// Evict oldest unpinned folds of `tag` beyond `state_retain_roots` (`0` = unbounded).
    fn evict_family(&mut self, tag: &str) {
        if self.cfg.retain_roots == 0 {
            return;
        }
        loop {
            let order = self.family_order.entry(tag.to_string()).or_default();
            let unpinned = order.iter().filter(|f| !self.pinned.contains(*f)).count();
            if unpinned as u64 <= self.cfg.retain_roots {
                return;
            }
            let Some(victim) = order.iter().find(|f| !self.pinned.contains(*f)).copied() else {
                return;
            };
            self.evict_fold(victim);
        }
    }

    /// Remove one sealed fold: retire its sealed refs (retained-byte meter) and its holds.
    fn evict_fold(&mut self, fold: [u8; 32]) {
        let Some(sealed) = self.sealed.remove(&fold) else {
            return;
        };
        if let Some(order) = self.family_order.get_mut(&sealed.tag) {
            order.retain(|f| *f != fold);
        }
        for (h, _) in &sealed.chunks {
            if let Some(e) = self.chunks.get_mut(h) {
                e.sealed_refs -= 1;
                if e.sealed_refs == 0 {
                    self.retained_bytes -= e.bytes.len() as u64;
                }
            }
            self.release_chunk(*h);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(chunk_size: u64) -> StateStoreConfig {
        StateStoreConfig {
            chunk_size,
            ..StateStoreConfig::default()
        }
    }

    fn expected_fold(chunk_size: u64, byte_len: u64, chunks: &[&[u8]]) -> [u8; 32] {
        let hashes: Vec<daemon_vhc_proto::Hash> = chunks
            .iter()
            .map(|c| daemon_vhc_proto::Hash(*blake3::hash(c).as_bytes()))
            .collect();
        family_fold(chunk_size, byte_len, &hashes).0
    }

    #[test]
    fn open_emit_seal_round_trips_and_matches_the_proto_fold() {
        let mut s = StateStore::new(cfg(8));
        let id = s.open("master", 12).unwrap();
        assert_eq!(id & STATE_STREAM_ID_TOP_BIT, STATE_STREAM_ID_TOP_BIT);
        assert_eq!(s.emit(id, b"AAAAAAAA", 0).unwrap(), 0);
        assert_eq!(s.emit(id, b"BBBB", 0).unwrap(), 1, "short tail chunk");
        let fold = s.seal(id).unwrap();
        assert_eq!(fold, expected_fold(8, 12, &[b"AAAAAAAA", b"BBBB"]));
        // Sealed ⇒ fetchable by construction [SF-R1]: ranges assemble across the boundary.
        let bytes = s.read_range(&fold, 6, 10).unwrap().unwrap();
        assert_eq!(bytes, b"AABB");
        assert_eq!(
            s.read_range(&fold, 0, 12).unwrap().unwrap(),
            b"AAAAAAAABBBB"
        );
        // Bounds are typed refusals, not panics.
        assert!(s.read_range(&fold, 4, 13).unwrap().is_err());
        // The stream is closed: a second seal is UnknownStream.
        assert_eq!(s.seal(id).unwrap_err(), StateStoreError::UnknownStream);
        assert_eq!(s.stats().retained_bytes, 12);
    }

    #[test]
    fn sealed_family_ref_reconstructs_a_validating_by_ref_section() {
        // The DRAIN by-ref carriage ([SF-6]): the host reconstructs the FamilyRef a checkpoint
        // section references from its own record of the self-sealed fold — no guest re-listing.
        let mut s = StateStore::new(cfg(8));
        let id = s.open("master", 12).unwrap();
        s.emit(id, b"AAAAAAAA", 0).unwrap();
        s.emit(id, b"BBBB", 0).unwrap();
        let fold = s.seal(id).unwrap();

        let fref = s
            .sealed_family_ref(&fold)
            .expect("sealed fold reconstructs");
        // The reconstructed ref IS self-consistent: its fold is the fold of its own chunk list.
        fref.validate().expect("reconstructed FamilyRef validates");
        assert_eq!(fref.fold.0, fold);
        assert_eq!(fref.byte_len, 12);
        assert_eq!(fref.chunk_size, 8);
        assert_eq!(fref.chunk_hashes.len(), 2, "8-byte chunk + 4-byte tail");
        assert_eq!(
            fref.chunk_hashes[0],
            daemon_vhc_proto::Hash(*blake3::hash(b"AAAAAAAA").as_bytes())
        );
        // An unknown fold has no reconstruction.
        assert!(s.sealed_family_ref(&[0u8; 32]).is_none());
    }

    #[test]
    fn degenerate_single_window_geometry_is_the_same_code_path() {
        // chunk_size ≥ byte_len ⇒ one chunk, one window — the 64-dim acceptance shape.
        let mut s = StateStore::new(cfg(64));
        let id = s.open("master", 5).unwrap();
        assert_eq!(s.emit(id, b"hello", 0).unwrap(), 0);
        let fold = s.seal(id).unwrap();
        assert_eq!(fold, expected_fold(64, 5, &[b"hello"]));
        assert_eq!(s.read_range(&fold, 1, 4).unwrap().unwrap(), b"ell");
    }

    #[test]
    fn framing_refusals_are_typed() {
        let mut s = StateStore::new(cfg(8));
        assert_eq!(
            s.open("master", 0).unwrap_err(),
            StateStoreError::EmptyFamily
        );
        let id = s.open("master", 12).unwrap();
        assert!(matches!(
            s.emit(id, b"", 0).unwrap_err(),
            StateStoreError::MisframedEmit { .. }
        ));
        assert!(matches!(
            s.emit(id, b"AAAAAAAAA", 0).unwrap_err(), // 9 > chunk_size 8
            StateStoreError::MisframedEmit { .. }
        ));
        s.emit(id, b"AAAAAAAA", 0).unwrap();
        assert!(matches!(
            s.emit(id, b"BBBBB", 0).unwrap_err(), // 8 + 5 > 12
            StateStoreError::MisframedEmit { .. }
        ));
        // Incomplete seal is typed with both figures.
        assert_eq!(
            s.seal(id).unwrap_err(),
            StateStoreError::IncompleteSeal {
                emitted: 8,
                declared: 12
            }
        );
        assert_eq!(
            s.emit(1234, b"x", 0).unwrap_err(),
            StateStoreError::UnknownStream
        );
        // The failed seal did NOT close the stream (it may be completed and retried).
        s.emit(id, b"BBBB", 0).unwrap();
        s.seal(id).unwrap();
    }

    #[test]
    fn unprovisioned_state_plane_refuses_open() {
        let mut s = StateStore::new(cfg(0));
        assert_eq!(
            s.open("master", 8).unwrap_err(),
            StateStoreError::NotProvisioned
        );
    }

    #[test]
    fn streams_max_and_write_budget_grants_refuse_typed() {
        let mut s = StateStore::new(StateStoreConfig {
            chunk_size: 8,
            streams_max: 1,
            emit_max_bytes: 4,
            write_rate_per_min: 6,
            ..StateStoreConfig::default()
        });
        let id = s.open("master", 8).unwrap();
        assert!(matches!(
            s.open("ef", 8).unwrap_err(),
            StateStoreError::StreamsExhausted { max: 1 }
        ));
        // Per-emit ceiling (4) refuses a 5-byte emit even though chunk_size (8) allows it.
        assert!(matches!(
            s.emit(id, b"AAAAA", 0).unwrap_err(),
            StateStoreError::WriteBudget { .. }
        ));
        // Token bucket: capacity 6/min; 4 + 4 exceeds it at t=0…
        s.emit(id, b"AAAA", 0).unwrap();
        assert!(matches!(
            s.emit(id, b"BBBB", 0).unwrap_err(),
            StateStoreError::WriteBudget { .. }
        ));
        // …but refills with logical time (60 s ⇒ full bucket again).
        s.emit(id, b"BBBB", 60_000).unwrap();
        s.seal(id).unwrap();
    }

    #[test]
    fn retention_evicts_oldest_unpinned_and_dedups_shared_chunks() {
        let mut s = StateStore::new(StateStoreConfig {
            chunk_size: 8,
            retain_roots: 2,
            ..StateStoreConfig::default()
        });
        let mut folds = Vec::new();
        // Three rounds of "master": round r = [shared 8-byte chunk][round-distinct tail].
        for r in 0..3u8 {
            let id = s.open("master", 12).unwrap();
            s.emit(id, b"SHAREDCK", 0).unwrap();
            s.emit(id, &[r; 4], 0).unwrap();
            folds.push(s.seal(id).unwrap());
        }
        // retain_roots = 2: round 0's fold evicted, rounds 1/2 retained.
        assert!(s.sealed(&folds[0]).is_none(), "oldest evicted");
        assert!(s.sealed(&folds[1]).is_some() && s.sealed(&folds[2]).is_some());
        // Content-addressed dedup: the shared chunk is stored once — retained bytes are
        // 8 (shared) + 4 + 4, not 2 × 12.
        assert_eq!(s.stats().retained_bytes, 16);
        assert_eq!(s.stats().chunk_objects, 3);
        // An evicted fold no longer serves ranges.
        assert!(s.read_range(&folds[0], 0, 4).is_none());

        // Pinned folds are exempt AND do not count against `state_retain_roots` (design §8.2:
        // retained roots PLUS checkpoint-pinned): pin round 1, then seal rounds 3 and 4 — the
        // unpinned population {2, 3} fits after round 3; round 4 evicts round 2 (oldest
        // unpinned) while the pinned round 1 survives everything.
        s.pin(folds[1]);
        let mut extra = Vec::new();
        for r in [9u8, 10] {
            let id = s.open("master", 12).unwrap();
            s.emit(id, b"SHAREDCK", 0).unwrap();
            s.emit(id, &[r; 4], 0).unwrap();
            extra.push(s.seal(id).unwrap());
        }
        assert!(s.sealed(&folds[1]).is_some(), "pinned fold survives");
        assert!(s.sealed(&folds[2]).is_none(), "oldest unpinned evicted");
        assert!(s.sealed(&extra[0]).is_some() && s.sealed(&extra[1]).is_some());
    }

    #[test]
    fn store_bytes_gate_rolls_back_the_refused_seal() {
        let mut s = StateStore::new(StateStoreConfig {
            chunk_size: 8,
            store_bytes_max: 20,
            retain_roots: 0, // unbounded retention: the ceiling must do the refusing
            ..StateStoreConfig::default()
        });
        let id = s.open("master", 12).unwrap();
        s.emit(id, b"AAAAAAAA", 0).unwrap();
        s.emit(id, b"BBBB", 0).unwrap();
        let f1 = s.seal(id).unwrap();
        // A second, distinct 12-byte fold takes retained past 20 → refused + rolled back.
        let id = s.open("master", 12).unwrap();
        s.emit(id, b"CCCCCCCC", 0).unwrap();
        s.emit(id, b"DDDD", 0).unwrap();
        assert!(matches!(
            s.seal(id).unwrap_err(),
            StateStoreError::StoreBytes { max: 20, .. }
        ));
        assert_eq!(s.stats().sealed_folds, 1, "refused seal left nothing");
        assert_eq!(s.stats().retained_bytes, 12);
        assert!(s.sealed(&f1).is_some(), "prior artifact untouched");
        // The refused stream is closed (rolled back, not resumable): UnknownStream.
        assert_eq!(
            s.emit(id, b"x", 0).unwrap_err(),
            StateStoreError::UnknownStream
        );
    }

    #[test]
    fn torn_folds_are_garbage_collected_and_never_durable() {
        let mut s = StateStore::new(cfg(8));
        let id = s.open("master", 12).unwrap();
        s.emit(id, b"AAAAAAAA", 0).unwrap();
        // Crash before seal: force-reclaim (teardown path) drops the staged chunks.
        assert_eq!(s.stats().open_streams, 1);
        assert_eq!(s.stats().chunk_objects, 1);
        s.clear_open();
        assert_eq!(s.stats().open_streams, 0);
        assert_eq!(s.stats().chunk_objects, 0, "staged chunks GCed");
        assert_eq!(s.stats().retained_bytes, 0, "nothing durable");
        assert_eq!(s.stats().sealed_folds, 0);
    }

    #[test]
    fn resealing_identical_content_is_a_dedup_no_op() {
        let mut s = StateStore::new(cfg(8));
        for _ in 0..2 {
            let id = s.open("master", 8).unwrap();
            s.emit(id, b"AAAAAAAA", 0).unwrap();
            s.seal(id).unwrap();
        }
        assert_eq!(s.stats().sealed_folds, 1);
        assert_eq!(s.stats().retained_bytes, 8);
        assert_eq!(s.stats().chunk_objects, 1);
    }

    #[test]
    fn stream_ids_are_counter_deterministic() {
        let mut a = StateStore::new(cfg(8));
        let mut b = StateStore::new(cfg(8));
        let ids_a = [a.open("m", 1).unwrap(), a.open("e", 1).unwrap()];
        let ids_b = [b.open("m", 1).unwrap(), b.open("e", 1).unwrap()];
        assert_eq!(ids_a, ids_b, "ids are a pure function of call order");
        assert_eq!(ids_a[0], STATE_STREAM_ID_TOP_BIT | 1);
        assert_eq!(ids_a[1], STATE_STREAM_ID_TOP_BIT | 2);
    }
}
