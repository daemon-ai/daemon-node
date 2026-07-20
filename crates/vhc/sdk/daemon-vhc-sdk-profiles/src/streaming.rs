// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The **streamed det-lane fold walks**: the resident profile math re-expressed as
//! completion-driven multi-slice state machines over chunk-addressed state windows, per the
//! chunk-addressed det-state contract (ABI spec §12.14) and its pinned slice-decomposition
//! schedule ([`daemon_vhc_sdk_consensus::fold_walk`]).
//!
//! # Why walks instead of calls
//!
//! The resident `ingest`/`make_update` are single synchronous calls over ALL parameters — at
//! ceremony-model scale that is gigabytes of guest linear memory and a single event slice whose
//! fuel/op budgets cannot honestly cover the work. The streamed walks hold at most the
//! configured in-flight window set resident: each window read completes as an event, each slice
//! folds the maximal contiguous run of completed windows and issues the next reads, and the
//! digest carry / family seal land in the walk's final slice. Per-slice work is bounded by
//! construction (windows in flight × window bytes), so the honest fuel claim is per-window.
//!
//! # Bit-identity with the resident path (the load-bearing property)
//!
//! Windows are validated to be **profile-chunk aligned** (`window_size` a positive multiple of
//! the compression chunk's byte width, every parameter's numel a multiple of the chunk), and the
//! schedule partitions each parameter ascending with a parameter never spanning a window. So:
//!
//! - every compression row (`[n_chunks, k]` payload row) belongs to exactly one window, and the
//!   per-parameter payload sections slice contiguously by row;
//! - per window, the fold performs the identical operation sequence the resident path performs
//!   per parameter (record-ordered scatter-adds into a zeroed accumulator, rebase copy, single
//!   axpy — or Δ, error-feedback accumulate, per-row top-k, per-row absmax pack), window-sliced;
//! - the one non-window-local step, the median-norm clip, is computed once per (peer, parameter)
//!   over the resident compressed payload **value sections** at walk construction — exactly the
//!   resident pre-pass;
//! - folds execute ascending and contiguous regardless of completion arrival order (the pinned
//!   schedule invariant), so every f32 op executes with identical operands in identical order.
//!
//! The windowed ≡ resident parity suites (`tests/streaming_parity.rs`) prove the emitted
//! masters, payload sections, error-feedback state, and digests bit-identical to the resident
//! oracle across geometries (including the degenerate single-window tier and a ceremony-shaped
//! scaled layout), window sizes, in-flight bounds, and arrival permutations.
//!
//! # The ABI seam (deliberately not wired here)
//!
//! The walks are ABI-agnostic library code, natively testable: window inputs arrive as f32
//! slices and outputs leave as f32 windows. The guest driver owns the wiring — `data@2::fetch`
//! of sealed round r−1 family windows (and ranged `read_into` of device-export buffers) on the
//! issue side, `vhc@2::state_emit` of each emitted window (window ordinal ≡ family chunk
//! ordinal, so one emitted window is exactly one state chunk) and `state_seal` in the sealing
//! slice on the emit side. [`f32s_to_le_bytes`]/[`le_bytes_to_f32s`] are the byte seam (state
//! families are f32-le by contract).
//!
//! The digest carry is threaded through the ingest walk: each emitted master window advances it
//! as it is emitted, in ascending window order — the sealed carry reproduces the resident round
//! state digest bit-for-bit (the digest-carry equivalence the shared vectors pin). `sparse_loco`
//! declares no replicated det state, so the sealed carry finalizes directly to the round digest;
//! a profile with replicated families would keep feeding the returned carry in canonical order.

use std::collections::BTreeMap;

use daemon_vhc_sdk_consensus::digest::DigestCarry;
use daemon_vhc_sdk_consensus::fold_walk::{windows, FoldWalk, Window};

use crate::{
    bytes_of, f32_of_bytes, indices_of, median_of, scale_add, tensor_data, Section, SparseLocoCfg,
};

/// The byte width of one state element (families are f32-le by the det-state contract).
const ELEM_BYTES: u64 = 4;

/// Encode f32 windows to their state-family byte image (f32-le, the one wire form) into `out`
/// (cleared first) — the `state_emit` seam.
pub fn f32s_to_le_bytes(vals: &[f32], out: &mut Vec<u8>) {
    out.clear();
    out.reserve(vals.len() * 4);
    for v in vals {
        out.extend_from_slice(&v.to_le_bytes());
    }
}

/// Decode a state-family byte window (f32-le) to f32 values — the fetch-completion seam.
///
/// # Errors
/// A `String` when `bytes` is not a multiple of 4 (a torn window read — typed, never silent).
pub fn le_bytes_to_f32s(bytes: &[u8]) -> Result<Vec<f32>, String> {
    if !bytes.len().is_multiple_of(4) {
        return Err(format!(
            "state window of {} bytes is not an f32-le image",
            bytes.len()
        ));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// One slice of walk output, mirroring the pinned schedule's slice actions: the windows folded
/// in this slice (ascending, contiguous with everything already folded — for the ingest walk
/// these are the master windows to `state_emit`; for the update walk the new error-feedback
/// windows), the window reads to issue next, and whether this slice sealed the walk.
#[derive(Debug)]
pub struct WalkSlice {
    /// The folded windows with their emitted f32 state, in fold order.
    pub emitted: Vec<(Window, Vec<f32>)>,
    /// The window reads this slice issues (start their fetches, then deliver each completion
    /// back to the walk).
    pub issue: Vec<Window>,
    /// Whether the walk sealed in this slice (fires exactly once, with the last fold).
    pub sealed: bool,
}

/// The shared completion-driven core both walks drive: the pinned schedule plus the bounded
/// stash of completed-but-not-yet-foldable window inputs (out-of-order arrivals; at most
/// `in_flight` entries by construction).
struct WalkCore<T> {
    walk: FoldWalk,
    schedule: Vec<Window>,
    pending: BTreeMap<u64, T>,
    started: bool,
}

impl<T> WalkCore<T> {
    fn new(numels: &[usize], window_size: u64, in_flight: u64) -> Self {
        let numels_u64: Vec<u64> = numels.iter().map(|&n| n as u64).collect();
        let schedule = windows(&numels_u64, window_size);
        let walk = FoldWalk::new(schedule.len() as u64, in_flight);
        Self {
            walk,
            schedule,
            pending: BTreeMap::new(),
            started: false,
        }
    }

    fn start(&mut self) -> Result<(Vec<Window>, bool), String> {
        if self.started {
            return Err("walk already started".into());
        }
        self.started = true;
        let actions = self.walk.start();
        Ok((self.issue_windows(&actions.issue), actions.seal))
    }

    /// Deliver one completed window read; returns the (window, input) pairs now foldable in
    /// ascending order, the next issues, and the seal flag.
    #[allow(clippy::type_complexity)]
    fn ready(
        &mut self,
        ordinal: u64,
        input: T,
    ) -> Result<(Vec<(Window, T)>, Vec<Window>, bool), String> {
        if !self.started {
            return Err("walk not started".into());
        }
        // Let the schedule refuse first (duplicate / never-issued / already-folded), so a bad
        // completion can never clobber the stashed input of a pending window.
        let actions = self
            .walk
            .on_completion(ordinal)
            .map_err(|e| e.to_string())?;
        self.pending.insert(ordinal, input);
        let mut folds = Vec::with_capacity(actions.fold.len());
        for f in &actions.fold {
            let input = self
                .pending
                .remove(f)
                .ok_or_else(|| format!("window {f} foldable without a stashed input"))?;
            folds.push((
                self.schedule[usize::try_from(*f).expect("ordinal fits usize")],
                input,
            ));
        }
        Ok((folds, self.issue_windows(&actions.issue), actions.seal))
    }

    fn issue_windows(&self, ordinals: &[u64]) -> Vec<Window> {
        ordinals
            .iter()
            .map(|&o| self.schedule[usize::try_from(o).expect("ordinal fits usize")])
            .collect()
    }

    fn is_sealed(&self) -> bool {
        self.walk.is_sealed()
    }
}

/// Validate the windowed-fold geometry: the profile-chunk constraint (`chunk` divides every
/// numel — the det kernels refuse a non-multiple layout only at first use) and the window
/// alignment rule (`window_size` a positive multiple of the chunk's byte width, so no
/// compression row ever spans a window) — the genesis `state_chunk_size` rule, applied at the
/// engine boundary.
fn validate_geometry(numels: &[usize], chunk: usize, window_size: u64) -> Result<(), String> {
    if chunk == 0 {
        return Err("profile chunk must be > 0".into());
    }
    if numels.is_empty() {
        return Err("windowed fold needs a non-empty parameter layout".into());
    }
    for (i, &numel) in numels.iter().enumerate() {
        if numel == 0 || !numel.is_multiple_of(chunk) {
            return Err(format!(
                "profile chunk {chunk} does not divide parameter {i}'s numel {numel}"
            ));
        }
    }
    let width = chunk as u64 * ELEM_BYTES;
    if window_size == 0 || !window_size.is_multiple_of(width) {
        return Err(format!(
            "window size {window_size} must be a non-zero multiple of the profile chunk byte \
             width {width}"
        ));
    }
    Ok(())
}

/// A window's element geometry under the per-parameter chunking rule: `(element offset within
/// the parameter, element count)` — both chunk-aligned by [`validate_geometry`].
fn window_elems(w: &Window) -> (usize, usize) {
    (
        usize::try_from(w.param_off / ELEM_BYTES).expect("offset fits usize"),
        usize::try_from(w.len / ELEM_BYTES).expect("window fits usize"),
    )
}

/// Refuse a completion whose input cannot be folded BEFORE the schedule consumes it — a
/// mis-sized window must leave the walk state (fold cursor, digest carry, outstanding set)
/// untouched, so the driver can re-deliver the corrected read.
fn check_window_input(
    schedule: &[Window],
    ordinal: u64,
    name: &str,
    len: usize,
) -> Result<(), String> {
    let w = schedule
        .get(usize::try_from(ordinal).map_err(|_| format!("window {ordinal} out of range"))?)
        .ok_or_else(|| format!("window {ordinal} out of range ({} windows)", schedule.len()))?;
    let (_, elems) = window_elems(w);
    if len != elems {
        return Err(format!(
            "window {ordinal}: {name} has {len} elements, window needs {elems}"
        ));
    }
    Ok(())
}

// ================================================================================================
// sparse_loco ingest, streamed
// ================================================================================================

/// One peer's decoded per-parameter payload: the packed value bytes, the chunk-local indices,
/// and the clip scale (1.0 when clipping is off or the norm is at/below the median) — decoded
/// and validated once at walk construction, exactly the resident pre-pass.
struct PeerParam {
    packed: Vec<u8>,
    idx: Vec<u32>,
    scale: f64,
}

/// The streamed `sparse_loco` det-lane ingest: the resident [`crate::SparseLoco::ingest`] as a
/// completion-driven walk over round-base (= sealed round r−1 master) windows, emitting round r
/// master windows and threading the digest carry through the emission order.
///
/// Drive: [`Self::start`] once (issue the returned window reads), then
/// [`Self::on_window_ready`] per completed read with the fetched round-base window; emit each
/// returned master window in order; after the sealing slice, [`Self::seal`] returns the carry.
pub struct SparseLocoIngestWalk {
    core: WalkCore<Vec<f32>>,
    cfg: SparseLocoCfg,
    /// Decoded payloads, `[payload][param]` (record order preserved).
    peers: Vec<Vec<PeerParam>>,
    /// The resident divisor: `payloads.len().max(1)`.
    count: usize,
    carry: DigestCarry,
    byte_buf: Vec<u8>,
}

impl SparseLocoIngestWalk {
    /// Build the walk: validate the geometry, decode + validate every payload's sections up
    /// front (typed refusals at construction, not mid-walk), and run the median-norm clip
    /// pre-pass per (peer, parameter) over the resident compressed value sections.
    ///
    /// `payloads` are the record-ordered committed containers (decoded); `carry` is the fresh
    /// round digest carry (seeded with the round seed at the pinned block size).
    ///
    /// # Errors
    /// A `String` on a geometry violation or a malformed payload.
    pub fn new(
        cfg: &SparseLocoCfg,
        numels: &[usize],
        window_size: u64,
        in_flight: u64,
        payloads: &[Vec<Section>],
        carry: DigestCarry,
    ) -> Result<Self, String> {
        let chunk = cfg.chunk as usize;
        let k = cfg.topk as usize;
        validate_geometry(numels, chunk, window_size)?;

        // Decode every (payload, parameter) section pair once; compute the clip pre-pass with
        // the resident math (full-parameter unpack → l2 norm → median → scale).
        let mut peers: Vec<Vec<PeerParam>> = payloads
            .iter()
            .map(|pl| {
                numels
                    .iter()
                    .enumerate()
                    .map(|(i, &numel)| {
                        let n_chunks = numel / chunk;
                        let packed = bytes_of(tensor_data(pl, 2 * i)?);
                        let idx = indices_of(tensor_data(pl, 2 * i + 1)?);
                        if idx.len() != n_chunks * k {
                            return Err(format!(
                                "param {i}: index section has {} entries, layout needs {}",
                                idx.len(),
                                n_chunks * k
                            ));
                        }
                        if packed.len() != n_chunks * packed_row_stride(k, cfg.bits) {
                            return Err(format!(
                                "param {i}: packed section has {} bytes, layout needs {}",
                                packed.len(),
                                n_chunks * packed_row_stride(k, cfg.bits)
                            ));
                        }
                        Ok(PeerParam {
                            packed,
                            idx,
                            scale: 1.0,
                        })
                    })
                    .collect::<Result<Vec<_>, String>>()
            })
            .collect::<Result<Vec<_>, String>>()?;

        if cfg.clip {
            for i in 0..numels.len() {
                let norms: Vec<f64> = peers
                    .iter()
                    .map(|peer| {
                        let vals = daemon_vhc_det::det_absmax_unpack(&peer[i].packed, k, cfg.bits)
                            .map_err(|e| format!("unpack: {e:?}"))?;
                        Ok(f64::from(daemon_vhc_det::det_l2norm(&vals)))
                    })
                    .collect::<Result<_, String>>()?;
                let median = median_of(&norms);
                for (peer, &norm) in peers.iter_mut().zip(norms.iter()) {
                    peer[i].scale = if norm > median && norm > 0.0 {
                        median / norm
                    } else {
                        1.0
                    };
                }
            }
        }

        Ok(Self {
            core: WalkCore::new(numels, window_size, in_flight),
            cfg: cfg.clone(),
            peers,
            count: payloads.len().max(1),
            carry,
            byte_buf: Vec::new(),
        })
    }

    /// The pinned window schedule (window ordinal ≡ family chunk ordinal) — the map the driver
    /// uses to turn issued windows into fetches and emitted windows into state chunks.
    #[must_use]
    pub fn schedule(&self) -> &[Window] {
        &self.core.schedule
    }

    /// The opening slice: the first window reads to issue (a zero-window layout seals
    /// immediately).
    ///
    /// # Errors
    /// A `String` when the walk was already started.
    pub fn start(&mut self) -> Result<WalkSlice, String> {
        let (issue, sealed) = self.core.start()?;
        Ok(WalkSlice {
            emitted: Vec::new(),
            issue,
            sealed,
        })
    }

    /// One round-base window read completed: fold the maximal contiguous run now available
    /// (each fold scatter-adds the record-ordered payload rows of the window, rebases, applies
    /// the outer step, and advances the digest carry with the emitted master window), refill
    /// the read window, and seal with the last fold.
    ///
    /// # Errors
    /// A `String` on a completion the schedule refuses (not outstanding), a mis-sized window,
    /// or a det-kernel refusal.
    pub fn on_window_ready(
        &mut self,
        ordinal: u64,
        round_base: &[f32],
    ) -> Result<WalkSlice, String> {
        check_window_input(&self.core.schedule, ordinal, "round-base", round_base.len())?;
        let (folds, issue, sealed) = self.core.ready(ordinal, round_base.to_vec())?;
        let mut emitted = Vec::with_capacity(folds.len());
        for (window, base) in folds {
            let master = self.fold_window(&window, &base)?;
            f32s_to_le_bytes(&master, &mut self.byte_buf);
            self.carry.update(&self.byte_buf);
            emitted.push((window, master));
        }
        Ok(WalkSlice {
            emitted,
            issue,
            sealed,
        })
    }

    /// Finish a sealed walk, returning the digest carry advanced over the full master family in
    /// canonical order. `sparse_loco` declares no replicated det state, so
    /// `seal().finalize()` IS the round state digest; a profile with replicated families would
    /// keep feeding the carry with its replicated chunks in canonical order first.
    ///
    /// # Errors
    /// A `String` when the walk has not sealed (windows are still outstanding).
    pub fn seal(self) -> Result<DigestCarry, String> {
        if !self.core.is_sealed() {
            return Err("ingest walk sealed before every window folded".into());
        }
        Ok(self.carry)
    }

    /// The window-local fold: the resident per-parameter ingest sequence sliced to the window's
    /// chunk rows — record-ordered scatter-adds of the clip-scaled decoded rows into a zeroed
    /// accumulator, rebase copy, single axpy.
    fn fold_window(&self, window: &Window, round_base: &[f32]) -> Result<Vec<f32>, String> {
        let chunk = self.cfg.chunk as usize;
        let k = self.cfg.topk as usize;
        let (off_elems, elems) = window_elems(window);
        let row0 = off_elems / chunk;
        let rows = elems / chunk;
        let stride = packed_row_stride(k, self.cfg.bits);
        let param = window.param as usize;

        let mut acc = vec![0.0f32; elems];
        for peer in &self.peers {
            let pp = &peer[param];
            let packed_rows = &pp.packed[row0 * stride..(row0 + rows) * stride];
            let vals = daemon_vhc_det::det_absmax_unpack(packed_rows, k, self.cfg.bits)
                .map_err(|e| format!("unpack: {e:?}"))?;
            let idx_rows = &pp.idx[row0 * k..(row0 + rows) * k];
            let scaled = daemon_vhc_det::det_scale(&vals, pp.scale);
            daemon_vhc_det::det_chunk_scatter_add(&mut acc, &scaled, idx_rows, chunk)
                .map_err(|e| format!("scatter_add: {e:?}"))?;
        }
        // θ⁽ᵗ⁺¹⁾ = θ⁽ᵗ⁾ − α·(1/R)·Σ Δ̂, window-sliced (rebase copy, then the canonical axpy).
        let mut master = round_base.to_vec();
        #[allow(clippy::cast_precision_loss)]
        daemon_vhc_det::det_axpy(&mut master, -self.cfg.outer_alpha / self.count as f64, &acc)
            .map_err(|e| format!("axpy: {e:?}"))?;
        Ok(master)
    }
}

/// The absmax-packed byte stride of one `[k]` value row at `bits` (the frozen §6.6 record
/// layout: a 2-byte f16 codebook scalar + `k` codes packed LSB-first, zero-padded to a byte).
fn packed_row_stride(k: usize, bits: u32) -> usize {
    2 + (k * bits as usize).div_ceil(8)
}

// ================================================================================================
// sparse_loco make_update, streamed
// ================================================================================================

/// The three window inputs of one update-walk window, all window-sliced from their families:
/// the trained θ (device-export buffer range), the round base (sealed round r−1 master window),
/// and the error-feedback window (sealed ef family).
#[derive(Debug, Clone)]
pub struct UpdateWindowInputs {
    /// The trained parameter window θ.
    pub theta: Vec<f32>,
    /// The det-lane canonical round-base window θ⁽ᵗ⁾.
    pub round_base: Vec<f32>,
    /// The error-feedback residual window.
    pub ef: Vec<f32>,
}

/// The streamed `sparse_loco` native-lane update: the resident
/// [`crate::SparseLoco::make_update`] as a completion-driven walk over (θ, round-base, ef)
/// windows, emitting the new error-feedback windows as it folds and assembling the payload
/// section fragments; [`Self::seal`] yields the resident-identical section list.
///
/// The emitted `WalkSlice` windows are the NEW ef family (replica-local, digest-invisible) —
/// the driver `state_emit`s them; the payload sections stay resident (they are the small
/// compressed wire form) and leave through the sealed container as today.
pub struct SparseLocoUpdateWalk {
    core: WalkCore<UpdateWindowInputs>,
    cfg: SparseLocoCfg,
    numels: Vec<usize>,
    /// Per-parameter section fragments, appended in fold order (ascending windows make this a
    /// straight append per parameter).
    packed: Vec<Vec<u8>>,
    indices: Vec<Vec<u32>>,
}

impl SparseLocoUpdateWalk {
    /// Build the walk over the parameter layout at `window_size` bytes with at most
    /// `in_flight` outstanding window reads.
    ///
    /// # Errors
    /// A `String` on a geometry violation (chunk/window alignment).
    pub fn new(
        cfg: &SparseLocoCfg,
        numels: &[usize],
        window_size: u64,
        in_flight: u64,
    ) -> Result<Self, String> {
        validate_geometry(numels, cfg.chunk as usize, window_size)?;
        if cfg.topk as usize > cfg.chunk as usize {
            return Err(format!(
                "topk {} exceeds the profile chunk {}",
                cfg.topk, cfg.chunk
            ));
        }
        Ok(Self {
            core: WalkCore::new(numels, window_size, in_flight),
            cfg: cfg.clone(),
            numels: numels.to_vec(),
            packed: vec![Vec::new(); numels.len()],
            indices: vec![Vec::new(); numels.len()],
        })
    }

    /// The pinned window schedule (shared with the ingest walk — ingest and update differ only
    /// in per-window math and emitted families).
    #[must_use]
    pub fn schedule(&self) -> &[Window] {
        &self.core.schedule
    }

    /// The opening slice: the first window reads to issue.
    ///
    /// # Errors
    /// A `String` when the walk was already started.
    pub fn start(&mut self) -> Result<WalkSlice, String> {
        let (issue, sealed) = self.core.start()?;
        Ok(WalkSlice {
            emitted: Vec::new(),
            issue,
            sealed,
        })
    }

    /// One window's inputs are complete: fold the maximal contiguous run now available (each
    /// fold runs Δ → error-feedback accumulate → per-row top-k → per-row absmax pack, appends
    /// the section fragments, and emits the new ef window), refill the read window, and seal
    /// with the last fold.
    ///
    /// # Errors
    /// A `String` on a refused completion, mis-sized windows, or a det-kernel refusal.
    pub fn on_window_ready(
        &mut self,
        ordinal: u64,
        inputs: UpdateWindowInputs,
    ) -> Result<WalkSlice, String> {
        for (name, v) in [
            ("theta", &inputs.theta),
            ("round-base", &inputs.round_base),
            ("ef", &inputs.ef),
        ] {
            check_window_input(&self.core.schedule, ordinal, name, v.len())?;
        }
        let (folds, issue, sealed) = self.core.ready(ordinal, inputs)?;
        let mut emitted = Vec::with_capacity(folds.len());
        for (window, inputs) in folds {
            let ef_new = self.fold_window(&window, &inputs)?;
            emitted.push((window, ef_new));
        }
        Ok(WalkSlice {
            emitted,
            issue,
            sealed,
        })
    }

    /// Finish a sealed walk: the assembled payload sections, bit-identical to the resident
    /// `make_update` output over the same inputs (per param `i`: section `2i` packed values on
    /// their f32-per-byte tensor ride, section `2i+1` chunk-local indices, `[n_chunks, k]`).
    ///
    /// # Errors
    /// A `String` when the walk has not sealed (windows are still outstanding).
    pub fn seal(self) -> Result<Vec<Section>, String> {
        if !self.core.is_sealed() {
            return Err("update walk sealed before every window folded".into());
        }
        let k = self.cfg.topk;
        let chunk = self.cfg.chunk as usize;
        let mut sections = Vec::with_capacity(self.numels.len() * 2);
        for (i, (packed, idx)) in self.packed.iter().zip(self.indices.iter()).enumerate() {
            let n_chunks = (self.numels[i] / chunk) as u32;
            sections.push(Section::Tensor {
                data: f32_of_bytes(packed),
                shape: vec![packed.len() as u32],
            });
            #[allow(clippy::cast_precision_loss)]
            sections.push(Section::Tensor {
                data: idx.iter().map(|&v| v as f32).collect(),
                shape: vec![n_chunks, k],
            });
        }
        Ok(sections)
    }

    /// The window-local update math: the resident per-parameter sequence sliced to the window's
    /// chunk rows — `Δ = θ⁽ᵗ⁾ − θ`, `acc = β·ef + Δ`, per-row top-k, per-row absmax pack,
    /// `ef ← acc − scatter(dequant(sent))`.
    fn fold_window(
        &mut self,
        window: &Window,
        inputs: &UpdateWindowInputs,
    ) -> Result<Vec<f32>, String> {
        let chunk = self.cfg.chunk as usize;
        let k = self.cfg.topk as usize;
        let (_, elems) = window_elems(window);
        let delta: Vec<f32> = inputs
            .round_base
            .iter()
            .zip(inputs.theta.iter())
            .map(|(&b, &t)| b - t)
            .collect();
        let acc = scale_add(&inputs.ef, self.cfg.ef_decay, &delta);
        let (vals, idx) =
            daemon_vhc_det::topk_chunk(&acc, chunk, k).map_err(|e| format!("topk: {e:?}"))?;
        let packed = daemon_vhc_det::absmax_pack(&vals, k, self.cfg.bits)
            .map_err(|e| format!("pack: {e:?}"))?;
        // ef ← acc − chunk_scatter(dequant(sent)), window-sliced (rows are window-local).
        let sent_vals = daemon_vhc_det::det_absmax_unpack(&packed, k, self.cfg.bits)
            .map_err(|e| format!("unpack: {e:?}"))?;
        let sent = daemon_vhc_det::det_chunk_scatter(&sent_vals, &idx, chunk, elems)
            .map_err(|e| format!("scatter: {e:?}"))?;
        let ef_new: Vec<f32> = acc.iter().zip(sent.iter()).map(|(&a, &s)| a - s).collect();

        let param = window.param as usize;
        self.packed[param].extend_from_slice(&packed);
        self.indices[param].extend_from_slice(&idx);
        Ok(ef_new)
    }
}
