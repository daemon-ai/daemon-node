// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Direct peer streams under credit-based flow control (architecture §3.3/§3.4) — Phase B
//! (track B1).
//!
//! A [`StreamTable`] owns one run instance's kind-9 `StreamHandle` population. Streams are
//! instance-class resources (ABI §7.1) minted at **completion arrival** — a `stream_open` /
//! `stream_accept` op completes with the handle — so, like completion-minted buffers, indices are
//! monotone and never reused: every handle value is a pure function of the journaled completion
//! order (§7.1 replay determinism).
//!
//! **Credit is pump-enforced host mechanism** (§3.3: "a stream grants writable credit, replenished
//! via completions, so a fast producer cannot force unbounded host buffering"): each stream
//! carries a send-credit window in bytes. `stream_write` consumes credit at issue; a write
//! exceeding the available window is **held** — its `OpId` is outstanding but no transport request
//! is emitted — until the receiver's reads replenish credit ([`StreamTable::grant`], driven by the
//! embedder as the remote side consumes). The guest never sees a credit number: held writes
//! completing later IS the credit signal, exactly the wire the architecture fixes
//! (`stream_write(stream, buffer) -> OpId → Completion(())`).

use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;

use daemon_vhc_abi::{
    handle_generation, handle_index, handle_kind, pack_handle, HANDLE_KIND_STREAM,
    HANDLE_MAX_GENERATION,
};

/// One live stream: its writable credit window plus writes held for credit.
struct StreamState {
    /// Remaining writable credit (bytes).
    credit: u64,
    /// Writes held until credit arrives: `(op, bytes)` in guest issue order (FIFO fairness).
    held: VecDeque<(u64, Arc<Vec<u8>>)>,
}

/// The per-instance stream table (see module docs).
pub struct StreamTable {
    live: BTreeMap<u32, StreamState>,
    next_index: u32,
    generation_seed: u32,
}

impl StreamTable {
    /// A fresh table, generation-seeded from the journaled instantiation counter (ABI §7.1).
    #[must_use]
    pub fn new(instantiation_counter: u64) -> Self {
        Self {
            live: BTreeMap::new(),
            next_index: 1,
            generation_seed: (u32::try_from(instantiation_counter).unwrap_or(u32::MAX)
                & HANDLE_MAX_GENERATION)
                .wrapping_add(1),
        }
    }

    /// Mint a stream at completion arrival with its initial writable credit (receiver-granted).
    /// Indices are monotone/never-reused — the handle is a pure function of mint order.
    pub fn open(&mut self, initial_credit: u64) -> u64 {
        let index = self.next_index;
        self.next_index += 1;
        self.live.insert(
            index,
            StreamState {
                credit: initial_credit,
                held: VecDeque::new(),
            },
        );
        pack_handle(HANDLE_KIND_STREAM, self.generation_seed, index)
    }

    fn index_of(&self, stream: u64) -> Option<u32> {
        if handle_kind(stream) != HANDLE_KIND_STREAM
            || handle_generation(stream) != self.generation_seed
        {
            return None;
        }
        let index = handle_index(stream);
        self.live.contains_key(&index).then_some(index)
    }

    /// Whether `stream` is live.
    #[must_use]
    pub fn is_live(&self, stream: u64) -> bool {
        self.index_of(stream).is_some()
    }

    /// The stream's remaining writable credit (test/introspection).
    #[must_use]
    pub fn credit(&self, stream: u64) -> Option<u64> {
        self.index_of(stream).map(|i| self.live[&i].credit)
    }

    /// Consume credit for a write of `bytes`, or hold it. Returns `true` when the write may be
    /// emitted to the transport now, `false` when it was held for credit. `None` = unknown/stale
    /// stream.
    pub fn write(&mut self, stream: u64, op: u64, bytes: Arc<Vec<u8>>) -> Option<bool> {
        let index = self.index_of(stream)?;
        let st = self.live.get_mut(&index).expect("indexed");
        let len = bytes.len() as u64;
        // FIFO fairness: a write behind held writes queues behind them even if it would fit.
        if st.held.is_empty() && st.credit >= len {
            st.credit -= len;
            Some(true)
        } else {
            st.held.push_back((op, bytes));
            Some(false)
        }
    }

    /// Replenish `stream`'s writable credit (the receiver consumed bytes), releasing held writes
    /// whose sizes now fit, FIFO. Returns the released `(op, bytes)` list, in order.
    pub fn grant(&mut self, stream: u64, credit: u64) -> Vec<(u64, Arc<Vec<u8>>)> {
        let Some(index) = self.index_of(stream) else {
            return Vec::new();
        };
        let st = self.live.get_mut(&index).expect("indexed");
        st.credit = st.credit.saturating_add(credit);
        let mut released = Vec::new();
        while let Some((_, bytes)) = st.held.front() {
            let len = bytes.len() as u64;
            if st.credit < len {
                break;
            }
            st.credit -= len;
            released.push(st.held.pop_front().expect("front checked"));
        }
        released
    }

    /// Drop a held write (its op was cancelled). Returns whether it was found+removed.
    pub fn cancel_held(&mut self, op: u64) -> bool {
        for st in self.live.values_mut() {
            if let Some(pos) = st.held.iter().position(|(o, _)| *o == op) {
                st.held.remove(pos);
                return true;
            }
        }
        false
    }

    /// Force-reclaim every stream (trap/teardown, ABI §7.3).
    pub fn clear(&mut self) {
        self.live.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arc(b: &[u8]) -> Arc<Vec<u8>> {
        Arc::new(b.to_vec())
    }

    #[test]
    fn open_mints_monotone_kind9_handles() {
        let mut t = StreamTable::new(0);
        let s1 = t.open(16);
        let s2 = t.open(16);
        assert_eq!(handle_kind(s1), HANDLE_KIND_STREAM);
        assert_eq!((handle_index(s1), handle_index(s2)), (1, 2));
        assert!(t.is_live(s1));
        assert_eq!(t.credit(s1), Some(16));
        assert!(!t.is_live(pack_handle(HANDLE_KIND_STREAM, 1, 99)));
    }

    #[test]
    fn writes_consume_credit_and_hold_beyond_the_window() {
        let mut t = StreamTable::new(0);
        let s = t.open(10);
        assert_eq!(t.write(s, 101, arc(b"1234")), Some(true), "4 <= 10");
        assert_eq!(t.credit(s), Some(6));
        assert_eq!(t.write(s, 102, arc(b"1234567")), Some(false), "7 > 6: held");
        // FIFO fairness: a small write behind a held one queues too.
        assert_eq!(t.write(s, 103, arc(b"a")), Some(false));
        assert_eq!(t.credit(s), Some(6), "held writes consume nothing yet");
        // Replenish: releases held writes in order while credit lasts.
        let released = t.grant(s, 2); // credit 8: the 7-byte write fits, then 1 byte for "a"
        assert_eq!(
            released.iter().map(|(op, _)| *op).collect::<Vec<_>>(),
            vec![102, 103],
            "FIFO release under replenished credit"
        );
        assert_eq!(t.credit(s), Some(0));
    }

    #[test]
    fn cancel_removes_a_held_write() {
        let mut t = StreamTable::new(0);
        let s = t.open(0);
        assert_eq!(t.write(s, 7, arc(b"blocked")), Some(false));
        assert!(t.cancel_held(7));
        assert!(!t.cancel_held(7), "already removed");
        assert!(t.grant(s, 100).is_empty(), "nothing held after cancel");
    }
}
