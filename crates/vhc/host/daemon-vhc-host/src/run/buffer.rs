// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The buffer layer (architecture §3.4; ABI §7.4) — Phase B (track B1).
//!
//! A [`BufferTable`] owns the **`BufferHandle`** population of one run instance: opaque,
//! host-owned, **sealed** (immutable after creation) byte regions that cross capability worlds by
//! handle, never through wasm linear memory. Buffers are kind-8, instance-class resources
//! (ABI §7.1/§7.2):
//!
//! - **Refcounted** host objects: the table holds one reference; in-flight operations (a
//!   `payload_put` the embedder is servicing) clone the [`Arc`] and keep the bytes alive past a
//!   guest release — the guest's *quota* is freed at release, the *bytes* live until the last
//!   holder drops.
//! - **Quota'd against grants**: live handles and live bytes are metered against the admitted
//!   `buffer-req` quotas (`max_live_handles` / `max_live_bytes`, ABI §2.3); breach is the typed,
//!   attributable [`TrapCode::BudgetHandles`] / [`TrapCode::BudgetMemory`].
//! - **Generation-deterministic** (ABI §7.1): generations seed from the journaled instantiation
//!   counter and advance by one on slot reuse — the v1 `StepArena` discipline — so replay
//!   reproduces every handle value bit-exactly. A slot whose generation would wrap past the
//!   24-bit ceiling is **permanently retired** (no ABA reuse, ABI §7.1).
//! - **Force-reclaimed on trap** through this per-instance table: [`BufferTable::clear`] drops
//!   every live buffer wholesale; a fresh instantiation seeds a new generation, so any retained
//!   handle from the dead instance decodes to a wrong generation and traps `StaleHandle`.
//!
//! The two budgeted linear-memory crossing paths (`create_from` — sealed at creation — and
//! `read_into`) are driver imports over this table; the table itself never touches guest memory.
//!
//! ## Two index partitions (replay determinism, ABI §7.1)
//!
//! Buffer handles are minted from two disjoint index domains:
//!
//! - **Guest partition** (indices `1 .. HOST_INDEX_BASE`): `create_from` allocations — a reusing,
//!   generation-bumping arena whose state is a pure function of the guest's own call order.
//! - **Host partition** (indices `≥ HOST_INDEX_BASE`): buffers minted at **completion arrival**
//!   (`payload_get` results). Arrival interleaves nondeterministically with guest calls, so these
//!   take monotone, **never-reused** indices at the fixed seed generation — the handle value is a
//!   pure function of the *journaled completion order*, and the guest arena's state never depends
//!   on embedder timing. Replay re-derives both partitions bit-exactly (§7.1 "handle generations
//!   derive from journaled instantiation counters, never host randomness").

use std::collections::BTreeMap;
use std::sync::Arc;

use daemon_vhc_abi::{
    handle_generation, handle_index, handle_kind, pack_handle, HANDLE_KIND_BUFFER,
    HANDLE_MAX_GENERATION,
};

use crate::trap::TrapCode;

/// The first index of the host (completion-minted) partition. Guest-arena indices stay below.
pub const HOST_INDEX_BASE: u32 = 0x8000_0000;

/// One buffer slot: the current generation plus the sealed bytes (`None` = free).
struct BufferSlot {
    generation: u32,
    data: Option<Arc<Vec<u8>>>,
}

/// The per-instance `BufferHandle` table (see module docs).
pub struct BufferTable {
    slots: Vec<BufferSlot>,
    free: Vec<u32>,
    /// The §7.1 generation seed: fresh slots start at `instantiation counter + 1` (counter 0 ⇒
    /// generation 1, the v1 discipline; generation 0 is reserved for registered-class handles).
    generation_seed: u32,
    /// The host partition: completion-minted buffers, monotone never-reused indices.
    host_slots: BTreeMap<u32, Arc<Vec<u8>>>,
    next_host_index: u32,
    live_handles: u64,
    live_bytes: u64,
    max_live_handles: u64,
    max_live_bytes: u64,
}

impl BufferTable {
    /// A fresh table under the admitted `buffer-req` quotas, generation-seeded from the journaled
    /// instantiation counter (ABI §7.1). A quota of `0` means "unbounded by this grant" — still
    /// bounded by the lane ceiling at admission (ABI §2.3).
    #[must_use]
    pub fn new(instantiation_counter: u64, max_live_handles: u64, max_live_bytes: u64) -> Self {
        Self {
            slots: Vec::new(),
            free: Vec::new(),
            generation_seed: (u32::try_from(instantiation_counter).unwrap_or(u32::MAX)
                & HANDLE_MAX_GENERATION)
                .wrapping_add(1),
            host_slots: BTreeMap::new(),
            next_host_index: HOST_INDEX_BASE,
            live_handles: 0,
            live_bytes: 0,
            max_live_handles,
            max_live_bytes,
        }
    }

    /// Live (guest-held) buffer handles.
    #[must_use]
    pub fn live_handles(&self) -> u64 {
        self.live_handles
    }

    /// Live (guest-held) buffer bytes — the quota-metered figure.
    #[must_use]
    pub fn live_bytes(&self) -> u64 {
        self.live_bytes
    }

    fn charge(&mut self, len: u64) -> Result<(), TrapCode> {
        if self.max_live_handles != 0 && self.live_handles + 1 > self.max_live_handles {
            return Err(TrapCode::BudgetHandles);
        }
        if self.max_live_bytes != 0 && self.live_bytes + len > self.max_live_bytes {
            return Err(TrapCode::BudgetMemory);
        }
        self.live_handles += 1;
        self.live_bytes += len;
        Ok(())
    }

    /// Seal `bytes` into a fresh **guest-partition** buffer (`create_from`) and return its kind-8
    /// handle. Quota-checked BEFORE the slot is taken, so a refused create leaves the table
    /// unchanged.
    ///
    /// # Errors
    ///
    /// [`TrapCode::BudgetHandles`] / [`TrapCode::BudgetMemory`] on a `buffer-req` quota breach
    /// (typed, attributable to the module — ABI §9.1).
    pub fn create(&mut self, bytes: Arc<Vec<u8>>) -> Result<u64, TrapCode> {
        self.charge(bytes.len() as u64)?;
        if let Some(index) = self.free.pop() {
            let slot = &mut self.slots[index as usize];
            slot.data = Some(bytes);
            Ok(pack_handle(HANDLE_KIND_BUFFER, slot.generation, index + 1))
        } else {
            let index = self.slots.len() as u32;
            debug_assert!(
                index + 1 < HOST_INDEX_BASE,
                "guest arena below the host partition"
            );
            self.slots.push(BufferSlot {
                generation: self.generation_seed,
                data: Some(bytes),
            });
            Ok(pack_handle(
                HANDLE_KIND_BUFFER,
                self.generation_seed,
                index + 1,
            ))
        }
    }

    /// Seal `bytes` into a fresh **host-partition** buffer — the completion-arrival mint
    /// (`payload_get` results): a monotone, never-reused index at the seed generation, so the
    /// handle value is a pure function of the journaled completion order (module docs). `None`
    /// on a quota breach — completion-time creation is not a guest call, so quota pressure is
    /// "deny new buffers" (the completion reports `GrantExhausted`, the declared first
    /// degradation step of ABI §9.1), never a trap.
    pub fn create_host(&mut self, bytes: Arc<Vec<u8>>) -> Option<u64> {
        self.charge(bytes.len() as u64).ok()?;
        let index = self.next_host_index;
        self.next_host_index += 1;
        self.host_slots.insert(index, bytes);
        Some(pack_handle(HANDLE_KIND_BUFFER, self.generation_seed, index))
    }

    /// Which partition + slot a handle names, checked live + generation-valid.
    fn locate(&self, handle: u64) -> Result<Location, TrapCode> {
        if handle_kind(handle) != HANDLE_KIND_BUFFER {
            return Err(TrapCode::InvalidHandle);
        }
        let idx1 = handle_index(handle);
        if idx1 == 0 {
            return Err(TrapCode::InvalidHandle);
        }
        if idx1 >= HOST_INDEX_BASE {
            if handle_generation(handle) != self.generation_seed {
                return Err(TrapCode::StaleHandle);
            }
            if idx1 >= self.next_host_index {
                return Err(TrapCode::InvalidHandle);
            }
            if !self.host_slots.contains_key(&idx1) {
                return Err(TrapCode::StaleHandle);
            }
            return Ok(Location::Host(idx1));
        }
        let index = (idx1 - 1) as usize;
        let slot = self.slots.get(index).ok_or(TrapCode::InvalidHandle)?;
        if slot.data.is_none() || slot.generation != handle_generation(handle) {
            return Err(TrapCode::StaleHandle);
        }
        Ok(Location::Guest(index))
    }

    /// Resolve a live buffer handle to its sealed bytes (a cheap [`Arc`] clone — the refcount).
    ///
    /// # Errors
    ///
    /// [`TrapCode::InvalidHandle`] for a non-buffer/out-of-range handle;
    /// [`TrapCode::StaleHandle`] for a released or wrong-generation one.
    pub fn resolve(&self, handle: u64) -> Result<Arc<Vec<u8>>, TrapCode> {
        match self.locate(handle)? {
            Location::Guest(index) => {
                Ok(self.slots[index].data.clone().expect("slot checked live"))
            }
            Location::Host(index) => Ok(self.host_slots[&index].clone()),
        }
    }

    /// Release the guest's hold on a buffer, freeing its quota. A guest-partition slot's
    /// generation advances and refrees (permanent retirement at the ceiling — no ABA, ABI §7.1);
    /// a host-partition index is simply removed (never reminted). Returns the freed byte count.
    /// The bytes themselves live until the last in-flight [`Arc`] holder drops.
    ///
    /// # Errors
    ///
    /// As [`BufferTable::resolve`].
    pub fn release(&mut self, handle: u64) -> Result<u64, TrapCode> {
        let len = match self.locate(handle)? {
            Location::Guest(index) => {
                let slot = &mut self.slots[index];
                let len = slot.data.take().expect("slot checked live").len() as u64;
                if slot.generation >= HANDLE_MAX_GENERATION {
                    // Permanent retirement: never back on the free list (ABI §7.1 wrap rule).
                } else {
                    slot.generation += 1;
                    self.free.push(index as u32);
                }
                len
            }
            Location::Host(index) => self
                .host_slots
                .remove(&index)
                .expect("located host slot live")
                .len() as u64,
        };
        self.live_handles -= 1;
        self.live_bytes -= len;
        Ok(len)
    }

    /// Force-reclaim every live buffer (trap/teardown path, architecture §3.4): quotas zero, every
    /// slot freed, generations advanced so any retained handle traps `StaleHandle`.
    pub fn clear(&mut self) {
        self.free.clear();
        for (index, slot) in self.slots.iter_mut().enumerate() {
            if slot.data.take().is_some() && slot.generation < HANDLE_MAX_GENERATION {
                slot.generation += 1;
            }
            if slot.generation < HANDLE_MAX_GENERATION {
                self.free.push(index as u32);
            }
        }
        self.host_slots.clear();
        self.live_handles = 0;
        self.live_bytes = 0;
    }
}

/// A located live buffer: guest-arena slot index or host-partition index.
enum Location {
    Guest(usize),
    Host(u32),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arc(bytes: &[u8]) -> Arc<Vec<u8>> {
        Arc::new(bytes.to_vec())
    }

    #[test]
    fn create_resolve_release_round_trips_and_meters_quota() {
        let mut t = BufferTable::new(0, 4, 64);
        let h = t.create(arc(b"sealed-bytes")).unwrap();
        assert_eq!(daemon_vhc_abi::handle_kind(h), HANDLE_KIND_BUFFER);
        assert_eq!((t.live_handles(), t.live_bytes()), (1, 12));
        assert_eq!(t.resolve(h).unwrap().as_slice(), b"sealed-bytes");
        assert_eq!(t.release(h).unwrap(), 12);
        assert_eq!((t.live_handles(), t.live_bytes()), (0, 0));
        assert_eq!(t.resolve(h), Err(TrapCode::StaleHandle));
        assert_eq!(t.release(h), Err(TrapCode::StaleHandle));
    }

    #[test]
    fn quotas_trap_typed_and_leave_the_table_unchanged() {
        let mut t = BufferTable::new(0, 2, 10);
        let _h1 = t.create(arc(b"aaaa")).unwrap();
        // Byte quota: 4 + 8 > 10.
        assert_eq!(t.create(arc(b"bbbbbbbb")), Err(TrapCode::BudgetMemory));
        assert_eq!(
            (t.live_handles(), t.live_bytes()),
            (1, 4),
            "refused create is a no-op"
        );
        let _h2 = t.create(arc(b"bb")).unwrap();
        // Handle quota: a third live handle exceeds max_live_handles = 2.
        assert_eq!(t.create(arc(b"c")), Err(TrapCode::BudgetHandles));
    }

    #[test]
    fn generations_seed_from_the_instantiation_counter_and_advance_on_reuse() {
        // Counter 0 ⇒ generation 1 (the v1 fresh-slot discipline); counter 3 ⇒ generation 4.
        let mut t0 = BufferTable::new(0, 0, 0);
        let h0 = t0.create(arc(b"x")).unwrap();
        assert_eq!(daemon_vhc_abi::handle_generation(h0), 1);
        let mut t3 = BufferTable::new(3, 0, 0);
        let h3 = t3.create(arc(b"x")).unwrap();
        assert_eq!(daemon_vhc_abi::handle_generation(h3), 4);
        // Reuse advances by one: the old handle is stale, the new one distinct.
        t3.release(h3).unwrap();
        let h3b = t3.create(arc(b"y")).unwrap();
        assert_eq!(daemon_vhc_abi::handle_generation(h3b), 5);
        assert_ne!(h3, h3b);
        assert_eq!(t3.resolve(h3), Err(TrapCode::StaleHandle));
        assert_eq!(t3.resolve(h3b).unwrap().as_slice(), b"y");
    }

    #[test]
    fn refcount_survives_guest_release_for_inflight_holders() {
        // An in-flight op's Arc keeps the bytes alive past the guest's release; the QUOTA frees.
        let mut t = BufferTable::new(0, 0, 0);
        let h = t.create(arc(b"in-flight")).unwrap();
        let held = t.resolve(h).unwrap(); // the op's clone
        t.release(h).unwrap();
        assert_eq!(t.live_bytes(), 0, "quota freed at guest release");
        assert_eq!(held.as_slice(), b"in-flight", "bytes alive for the holder");
    }

    #[test]
    fn clear_force_reclaims_everything() {
        let mut t = BufferTable::new(0, 0, 0);
        let h1 = t.create(arc(b"a")).unwrap();
        let h2 = t.create(arc(b"bb")).unwrap();
        t.clear();
        assert_eq!((t.live_handles(), t.live_bytes()), (0, 0));
        assert_eq!(t.resolve(h1), Err(TrapCode::StaleHandle));
        assert_eq!(t.resolve(h2), Err(TrapCode::StaleHandle));
        // The table remains usable after a force-reclaim (trap-restart path).
        let h3 = t.create(arc(b"c")).unwrap();
        assert_eq!(t.resolve(h3).unwrap().as_slice(), b"c");
    }

    #[test]
    fn generation_ceiling_permanently_retires_the_slot() {
        let mut t = BufferTable::new(0, 0, 0);
        // Force a slot to the ceiling by seeding at the max (counter masked to 24 bits).
        t.generation_seed = HANDLE_MAX_GENERATION;
        let h = t.create(arc(b"z")).unwrap();
        assert_eq!(daemon_vhc_abi::handle_generation(h), HANDLE_MAX_GENERATION);
        t.release(h).unwrap();
        // The slot never returns: a fresh create takes a NEW index (index 2, 1-based).
        let h2 = t.create(arc(b"w")).unwrap();
        assert_eq!(
            daemon_vhc_abi::handle_index(h2),
            2,
            "retired slot is never refreed"
        );
    }

    #[test]
    fn host_partition_mints_monotone_never_reused_indices() {
        // Completion-minted buffers: index sequence is a pure function of mint ORDER (the
        // journaled completion order) — never interleaved with the guest arena's reuse.
        let mut t = BufferTable::new(0, 0, 0);
        let g1 = t.create(arc(b"guest")).unwrap();
        let h1 = t.create_host(arc(b"net-1")).unwrap();
        let h2 = t.create_host(arc(b"net-2")).unwrap();
        assert_eq!(daemon_vhc_abi::handle_index(h1), HOST_INDEX_BASE);
        assert_eq!(daemon_vhc_abi::handle_index(h2), HOST_INDEX_BASE + 1);
        assert!(daemon_vhc_abi::handle_index(g1) < HOST_INDEX_BASE);
        assert_eq!(t.resolve(h1).unwrap().as_slice(), b"net-1");
        // Release + re-mint: the index is NEVER reused; the released handle is stale.
        assert_eq!(t.release(h1).unwrap(), 5);
        let h3 = t.create_host(arc(b"net-3")).unwrap();
        assert_eq!(daemon_vhc_abi::handle_index(h3), HOST_INDEX_BASE + 2);
        assert_eq!(t.resolve(h1), Err(TrapCode::StaleHandle));
        // Host buffers meter the same quotas.
        assert_eq!(t.live_handles(), 3, "guest + net-2 + net-3");
    }

    #[test]
    fn host_partition_quota_pressure_denies_not_traps() {
        // Completion-time creation under quota pressure is "deny new buffers" (None), never a
        // trap — the completion path reports GrantExhausted (ABI §9.1 degradation step 0).
        let mut t = BufferTable::new(0, 1, 0);
        let _g = t.create(arc(b"held")).unwrap();
        assert!(t.create_host(arc(b"denied")).is_none());
        assert_eq!(t.live_handles(), 1, "denied mint is a no-op");
    }

    #[test]
    fn zero_and_foreign_handles_are_invalid_not_stale() {
        let t = BufferTable::new(0, 0, 0);
        assert_eq!(t.resolve(0), Err(TrapCode::InvalidHandle));
        // A kind-10 (OpId) handle is not a buffer.
        let op = pack_handle(daemon_vhc_abi::HANDLE_KIND_OP_ID, 1, 1);
        assert_eq!(t.resolve(op), Err(TrapCode::InvalidHandle));
        // An out-of-range buffer handle is invalid, not stale.
        let bogus = pack_handle(HANDLE_KIND_BUFFER, 1, 999);
        assert_eq!(t.resolve(bogus), Err(TrapCode::InvalidHandle));
    }
}
