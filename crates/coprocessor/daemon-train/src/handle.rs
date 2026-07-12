// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The handle model (ABI §3.3/§3.4): opaque nonzero `u64`, lane- and class-tagged.
//!
//! - **Stable handles** (params, persistents, det persistents) are a deterministic function of their
//!   1-based registration index within their class — identical across re-instantiation (T3).
//! - **Step handles** come from a per-entry-point **generational** arena; they are invalidated
//!   wholesale when the entry point returns, and a stale or cross-lane use is a typed trap
//!   ([`TrapCode::StaleHandle`] / [`TrapCode::LaneMismatch`]), never a silent read.
//!
//! `0` is never a valid handle. The encoding packs `kind` (top byte) + `generation` (24 bits) +
//! `index` (low 32 bits); stable handles carry generation `0`.

use crate::backend::TensorId;
use crate::trap::TrapCode;

/// Tensor lane (ABI §3.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lane {
    /// Native (GPU/CPU, vendor-variant numerics).
    Native,
    /// Det (CPU fp32, bit-exact everywhere).
    Det,
}

const K_STEP_NATIVE: u64 = 1;
const K_STEP_DET: u64 = 2;
const K_PARAM: u64 = 3;
const K_PERSIST: u64 = 4;
const K_DETPERSIST: u64 = 5;
const K_UPDATE: u64 = 6;
const K_BATCH: u64 = 7;

const KIND_SHIFT: u64 = 56;
const GEN_SHIFT: u64 = 32;
const GEN_MASK: u64 = 0xFF_FFFF;
const IDX_MASK: u64 = 0xFFFF_FFFF;

fn encode(kind: u64, gen: u32, index: u32) -> u64 {
    (kind << KIND_SHIFT) | ((u64::from(gen) & GEN_MASK) << GEN_SHIFT) | u64::from(index)
}

fn kind_of(h: u64) -> u64 {
    h >> KIND_SHIFT
}
fn gen_of(h: u64) -> u32 {
    ((h >> GEN_SHIFT) & GEN_MASK) as u32
}
fn index_of(h: u64) -> u32 {
    (h & IDX_MASK) as u32
}

/// The class/lane a handle decodes to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HandleClass {
    /// A step tensor in the given lane.
    Step(Lane),
    /// A stable param (native lane).
    Param,
    /// A stable native persistent.
    Persistent,
    /// A stable det persistent.
    DetPersistent,
    /// An update container.
    Update,
    /// A batch.
    Batch,
}

impl HandleClass {
    /// The lane this class lives in, if any.
    #[must_use]
    pub fn lane(self) -> Option<Lane> {
        match self {
            Self::Step(l) => Some(l),
            Self::Param | Self::Persistent => Some(Lane::Native),
            Self::DetPersistent => Some(Lane::Det),
            Self::Update | Self::Batch => None,
        }
    }
}

/// Decode a raw handle's class, or `None` if it is `0` / an unknown kind.
#[must_use]
pub fn classify(h: u64) -> Option<HandleClass> {
    if h == 0 {
        return None;
    }
    Some(match kind_of(h) {
        K_STEP_NATIVE => HandleClass::Step(Lane::Native),
        K_STEP_DET => HandleClass::Step(Lane::Det),
        K_PARAM => HandleClass::Param,
        K_PERSIST => HandleClass::Persistent,
        K_DETPERSIST => HandleClass::DetPersistent,
        K_UPDATE => HandleClass::Update,
        K_BATCH => HandleClass::Batch,
        _ => return None,
    })
}

/// The stable handle for a param at 1-based registration `index` (ABI §3.3).
#[must_use]
pub fn param_handle(index: u32) -> u64 {
    encode(K_PARAM, 0, index)
}
/// The stable handle for a native persistent at 1-based registration `index`.
#[must_use]
pub fn persistent_handle(index: u32) -> u64 {
    encode(K_PERSIST, 0, index)
}
/// The stable handle for a det persistent at 1-based registration `index`.
#[must_use]
pub fn det_persistent_handle(index: u32) -> u64 {
    encode(K_DETPERSIST, 0, index)
}
/// The handle for an update container at 1-based `index`.
#[must_use]
pub fn update_handle(index: u32) -> u64 {
    encode(K_UPDATE, 0, index)
}
/// The handle for a batch at 1-based `index`.
#[must_use]
pub fn batch_handle(index: u32) -> u64 {
    encode(K_BATCH, 0, index)
}

/// The 1-based registration index a stable/container/batch handle decodes to.
#[must_use]
pub fn stable_index(h: u64) -> u32 {
    index_of(h)
}

struct Slot {
    gen: u32,
    tensor: Option<TensorId>,
    shape: Vec<u32>,
}

/// A per-entry-point generational arena for step handles of one lane (ABI §3.3).
pub struct StepArena {
    lane: Lane,
    slots: Vec<Slot>,
    free: Vec<u32>,
    live: usize,
}

impl StepArena {
    /// A fresh arena for `lane`.
    #[must_use]
    pub fn new(lane: Lane) -> Self {
        Self {
            lane,
            slots: Vec::new(),
            free: Vec::new(),
            live: 0,
        }
    }

    /// The number of live step handles (budgeted, ABI §8).
    #[must_use]
    pub fn live(&self) -> usize {
        self.live
    }

    /// Allocate a step handle backing `tensor` with `shape`.
    pub fn alloc(&mut self, tensor: TensorId, shape: Vec<u32>) -> u64 {
        self.live += 1;
        let kind = match self.lane {
            Lane::Native => K_STEP_NATIVE,
            Lane::Det => K_STEP_DET,
        };
        if let Some(index) = self.free.pop() {
            let slot = &mut self.slots[index as usize];
            slot.tensor = Some(tensor);
            slot.shape = shape;
            encode(kind, slot.gen, index + 1)
        } else {
            let index = self.slots.len() as u32;
            self.slots.push(Slot {
                gen: 1,
                tensor: Some(tensor),
                shape,
            });
            encode(kind, 1, index + 1)
        }
    }

    fn slot_of(&self, h: u64) -> Result<usize, TrapCode> {
        let idx1 = index_of(h);
        if idx1 == 0 {
            return Err(TrapCode::InvalidHandle);
        }
        let index = (idx1 - 1) as usize;
        let slot = self.slots.get(index).ok_or(TrapCode::InvalidHandle)?;
        if slot.tensor.is_none() || slot.gen != gen_of(h) {
            return Err(TrapCode::StaleHandle);
        }
        Ok(index)
    }

    /// Resolve a step handle to its backing tensor + shape (checks liveness + generation).
    ///
    /// # Errors
    ///
    /// [`TrapCode::InvalidHandle`] for an out-of-range handle; [`TrapCode::StaleHandle`] for a freed
    /// or wrong-generation one.
    pub fn resolve(&self, h: u64) -> Result<(TensorId, &[u32]), TrapCode> {
        let index = self.slot_of(h)?;
        let slot = &self.slots[index];
        Ok((slot.tensor.unwrap(), &slot.shape))
    }

    /// Eagerly free a step handle (`drop@1`), returning its backing tensor to free.
    ///
    /// # Errors
    ///
    /// As [`Self::resolve`].
    pub fn free(&mut self, h: u64) -> Result<TensorId, TrapCode> {
        let index = self.slot_of(h)?;
        let slot = &mut self.slots[index];
        let tensor = slot.tensor.take().unwrap();
        slot.gen = slot.gen.wrapping_add(1);
        self.free.push(index as u32);
        self.live -= 1;
        Ok(tensor)
    }

    /// Invalidate every live step handle wholesale (entry-point return) — bumps generations so any
    /// retained handle traps `StaleHandle`. Returns the tensors to free backend-side.
    pub fn clear(&mut self) -> Vec<TensorId> {
        let mut freed = Vec::new();
        self.free.clear();
        for (index, slot) in self.slots.iter_mut().enumerate() {
            if let Some(t) = slot.tensor.take() {
                freed.push(t);
                slot.gen = slot.gen.wrapping_add(1);
            }
            self.free.push(index as u32);
        }
        self.live = 0;
        freed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_handles_are_deterministic_and_classified() {
        assert_eq!(param_handle(1), param_handle(1));
        assert_ne!(param_handle(1), param_handle(2));
        assert_eq!(classify(param_handle(3)), Some(HandleClass::Param));
        assert_eq!(
            classify(det_persistent_handle(1)),
            Some(HandleClass::DetPersistent)
        );
        assert_eq!(stable_index(param_handle(7)), 7);
        assert_eq!(classify(0), None);
    }

    #[test]
    fn lane_of_classes() {
        assert_eq!(
            classify(param_handle(1)).unwrap().lane(),
            Some(Lane::Native)
        );
        assert_eq!(
            classify(det_persistent_handle(1)).unwrap().lane(),
            Some(Lane::Det)
        );
        assert_eq!(classify(update_handle(1)).unwrap().lane(), None);
    }

    #[test]
    fn arena_alloc_resolve_free() {
        let mut a = StepArena::new(Lane::Native);
        let h = a.alloc(10, vec![2, 3]);
        assert_eq!(classify(h), Some(HandleClass::Step(Lane::Native)));
        assert_eq!(a.live(), 1);
        let (t, shape) = a.resolve(h).unwrap();
        assert_eq!(t, 10);
        assert_eq!(shape, &[2, 3]);
        assert_eq!(a.free(h).unwrap(), 10);
        assert_eq!(a.live(), 0);
    }

    #[test]
    fn stale_handle_traps_after_free() {
        let mut a = StepArena::new(Lane::Det);
        let h = a.alloc(1, vec![4]);
        a.free(h).unwrap();
        assert_eq!(a.resolve(h), Err(TrapCode::StaleHandle));
        assert_eq!(a.free(h), Err(TrapCode::StaleHandle));
    }

    #[test]
    fn reused_slot_bumps_generation() {
        let mut a = StepArena::new(Lane::Native);
        let h1 = a.alloc(1, vec![1]);
        a.free(h1).unwrap();
        let h2 = a.alloc(2, vec![1]); // reuses the slot with a bumped generation
        assert_ne!(h1, h2);
        assert_eq!(a.resolve(h1), Err(TrapCode::StaleHandle));
        assert_eq!(a.resolve(h2).unwrap().0, 2);
    }

    #[test]
    fn clear_invalidates_all_live_handles() {
        let mut a = StepArena::new(Lane::Native);
        let h1 = a.alloc(1, vec![1]);
        let h2 = a.alloc(2, vec![1]);
        let freed = a.clear();
        assert_eq!(freed.len(), 2);
        assert_eq!(a.live(), 0);
        assert_eq!(a.resolve(h1), Err(TrapCode::StaleHandle));
        assert_eq!(a.resolve(h2), Err(TrapCode::StaleHandle));
    }

    #[test]
    fn out_of_range_is_invalid_not_stale() {
        let a = StepArena::new(Lane::Native);
        let bogus = encode(K_STEP_NATIVE, 1, 999);
        assert_eq!(a.resolve(bogus), Err(TrapCode::InvalidHandle));
    }
}
