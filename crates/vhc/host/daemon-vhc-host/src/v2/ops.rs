// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The outstanding-operation table (ABI §3.3/§7.5) — Phase B (track B1).
//!
//! Every non-immediate capability call returns an **`OpId`** (handle kind 10, instance-class) and
//! completes through `Event::Completion(op, result)`. This table owns one run instance's
//! outstanding population: the deterministic handle mint (generation-seeded like every §7.1
//! arena), the per-op request the embedder services (the async-runtime bridge: "all actual waiting
//! lives in the host's async runtime", architecture §3.3), and the `max_outstanding` grant bound
//! (ABI §2.3 `grant-bound.max_outstanding`).
//!
//! **Cancellation is deterministic bookkeeping** (recordless by design): `cancel(op)` succeeds —
//! removing the op and reporting `Cancelled` through its completion — iff the op is still
//! outstanding (no completion enqueued yet). Whether a cancel raced the service is therefore fully
//! captured by the journaled completion result (`Cancelled` vs anything else), so replay derives
//! the cancel return from the recorded event stream instead of a dedicated record (the same
//! rationale as the `NeedCapacity` recordlessness, ABI §8.3).

use std::sync::Arc;

use daemon_vhc_abi::{
    handle_generation, handle_index, handle_kind, pack_handle, HANDLE_KIND_OP_ID,
    HANDLE_MAX_GENERATION,
};

use crate::trap::TrapCode;

/// What an outstanding operation asked of the embedder — the request the async runtime services
/// via `PumpHandle::take_op_requests` / `complete_op`.
#[derive(Debug, Clone)]
pub enum OpRequest {
    /// `net.payload_put(buffer)` — store these sealed bytes on the run's payload plane. The
    /// completion hash is computed by the PUMP over exactly these bytes (hashing an outgoing
    /// buffer is a host-side op, architecture §3.4), never trusted from the embedder.
    PayloadPut {
        /// The sealed buffer bytes (the op's own refcount hold).
        bytes: Arc<Vec<u8>>,
    },
    /// `net.payload_get(hash)` — fetch the content-addressed bytes. The pump hash-verifies the
    /// serviced bytes BEFORE the completion is delivered (§3.4 "verification unchanged").
    PayloadGet {
        /// The requested blake3.
        hash: [u8; 32],
    },
}

/// One outstanding op slot.
struct OpSlot {
    generation: u32,
    live: Option<OpRequest>,
}

/// The per-instance outstanding-operation table (see module docs).
pub struct OpTable {
    slots: Vec<OpSlot>,
    free: Vec<u32>,
    generation_seed: u32,
    outstanding: u64,
    max_outstanding: u64,
}

impl OpTable {
    /// A fresh table under the admitted `max_outstanding` bound (`0` = unbounded by this grant),
    /// generation-seeded from the journaled instantiation counter (ABI §7.1).
    #[must_use]
    pub fn new(instantiation_counter: u64, max_outstanding: u64) -> Self {
        Self {
            slots: Vec::new(),
            free: Vec::new(),
            generation_seed: (u32::try_from(instantiation_counter).unwrap_or(u32::MAX)
                & HANDLE_MAX_GENERATION)
                .wrapping_add(1),
            outstanding: 0,
            max_outstanding,
        }
    }

    /// Outstanding (issued, not yet completed/cancelled) operations.
    #[must_use]
    pub fn outstanding(&self) -> u64 {
        self.outstanding
    }

    /// Issue a fresh `OpId` for `request`.
    ///
    /// # Errors
    ///
    /// [`TrapCode::GrantViolation`] when the `max_outstanding` grant bound is exhausted (a typed,
    /// attributable refusal of the CALL — the outstanding set is unchanged).
    pub fn begin(&mut self, request: OpRequest) -> Result<u64, TrapCode> {
        if self.max_outstanding != 0 && self.outstanding + 1 > self.max_outstanding {
            return Err(TrapCode::GrantViolation);
        }
        self.outstanding += 1;
        if let Some(index) = self.free.pop() {
            let slot = &mut self.slots[index as usize];
            slot.live = Some(request);
            Ok(pack_handle(HANDLE_KIND_OP_ID, slot.generation, index + 1))
        } else {
            let index = self.slots.len() as u32;
            self.slots.push(OpSlot {
                generation: self.generation_seed,
                live: Some(request),
            });
            Ok(pack_handle(
                HANDLE_KIND_OP_ID,
                self.generation_seed,
                index + 1,
            ))
        }
    }

    fn slot_of(&self, op: u64) -> Option<usize> {
        if handle_kind(op) != HANDLE_KIND_OP_ID {
            return None;
        }
        let idx1 = handle_index(op);
        if idx1 == 0 {
            return None;
        }
        let index = (idx1 - 1) as usize;
        let slot = self.slots.get(index)?;
        (slot.live.is_some() && slot.generation == handle_generation(op)).then_some(index)
    }

    /// Whether `op` is still outstanding.
    #[must_use]
    pub fn is_outstanding(&self, op: u64) -> bool {
        self.slot_of(op).is_some()
    }

    /// Retire `op` (completion enqueued, or cancel accepted), returning its request. `None` when
    /// the op is not outstanding — the raced-cancel case a late `complete_op` must tolerate.
    pub fn finish(&mut self, op: u64) -> Option<OpRequest> {
        let index = self.slot_of(op)?;
        let slot = &mut self.slots[index];
        let request = slot.live.take();
        self.outstanding -= 1;
        if slot.generation >= HANDLE_MAX_GENERATION {
            // Permanent retirement at the generation ceiling (no ABA, ABI §7.1).
        } else {
            slot.generation += 1;
            self.free.push(index as u32);
        }
        request
    }

    /// Force-reclaim every outstanding op (trap/teardown, ABI §7.3).
    pub fn clear(&mut self) {
        self.free.clear();
        for (index, slot) in self.slots.iter_mut().enumerate() {
            if slot.live.take().is_some() && slot.generation < HANDLE_MAX_GENERATION {
                slot.generation += 1;
            }
            if slot.generation < HANDLE_MAX_GENERATION {
                self.free.push(index as u32);
            }
        }
        self.outstanding = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn begin_finish_round_trips_with_kind10_handles() {
        let mut t = OpTable::new(0, 4);
        let op = t.begin(OpRequest::PayloadGet { hash: [7u8; 32] }).unwrap();
        assert_eq!(handle_kind(op), HANDLE_KIND_OP_ID);
        assert!(t.is_outstanding(op));
        assert_eq!(t.outstanding(), 1);
        let Some(OpRequest::PayloadGet { hash }) = t.finish(op) else {
            panic!("request returned at finish");
        };
        assert_eq!(hash, [7u8; 32]);
        assert!(!t.is_outstanding(op));
        assert_eq!(
            t.finish(op),
            None,
            "double finish is the raced-cancel no-op"
        );
    }

    #[test]
    fn max_outstanding_is_a_typed_grant_refusal() {
        let mut t = OpTable::new(0, 1);
        let op1 = t
            .begin(OpRequest::PayloadPut {
                bytes: Arc::new(vec![1]),
            })
            .unwrap();
        assert_eq!(
            t.begin(OpRequest::PayloadGet { hash: [0u8; 32] }),
            Err(TrapCode::GrantViolation),
            "max_outstanding = 1"
        );
        t.finish(op1);
        t.begin(OpRequest::PayloadGet { hash: [0u8; 32] })
            .expect("capacity returns after finish");
    }

    #[test]
    fn generations_advance_on_reuse_and_stale_ops_are_not_outstanding() {
        let mut t = OpTable::new(2, 0);
        let op = t.begin(OpRequest::PayloadGet { hash: [1u8; 32] }).unwrap();
        assert_eq!(handle_generation(op), 3, "seeded from counter 2");
        t.finish(op);
        let op2 = t.begin(OpRequest::PayloadGet { hash: [2u8; 32] }).unwrap();
        assert_ne!(op, op2);
        assert!(!t.is_outstanding(op), "retired handle is stale");
        assert!(t.is_outstanding(op2));
    }

    impl PartialEq for OpRequest {
        fn eq(&self, other: &Self) -> bool {
            match (self, other) {
                (Self::PayloadPut { bytes: a }, Self::PayloadPut { bytes: b }) => a == b,
                (Self::PayloadGet { hash: a }, Self::PayloadGet { hash: b }) => a == b,
                _ => false,
            }
        }
    }
}
