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
//! - every compression row (`[n_chunks, k]` payload row) belongs to exactly one window, so a
//!   window's rows are exactly one committed-payload chunk pair (the container is chunk-addressed
//!   on this same schedule — one value chunk and one index chunk per window);
//! - per window, the fold performs the identical operation sequence the resident path performs
//!   per parameter (record-ordered scatter-adds into a zeroed accumulator, rebase copy, single
//!   axpy — or Δ, error-feedback accumulate, per-row top-k, per-row absmax pack), window-sliced;
//! - the one non-window-local step, the median-norm clip, needs a whole parameter's value rows per
//!   peer, so it is folded through a streamed norm carry ([`daemon_vhc_det::det_sumsq_into`]) in a
//!   first pass over the value chunks alone — bit-identical to the resident whole-section
//!   pre-pass, because the accumulation order is the same;
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
//! slices or as raw chunk bytes, and outputs leave as f32 windows and chunk bytes. The guest
//! driver owns the wiring — `data@2::fetch` of sealed round r−1 family windows (and ranged
//! `read_into` of device-export buffers) and `net@2::payload_get` of committed-payload chunks on
//! the issue side, `vhc@2::state_emit` of each emitted window (window ordinal ≡ family chunk
//! ordinal, so one emitted window is exactly one state chunk), `state_seal` in the sealing slice,
//! and `net@2::payload_put` of each payload chunk on the emit side.
//! [`f32s_to_le_bytes`]/[`le_bytes_to_f32s`] are the state byte seam (state families are f32-le by
//! contract); the payload chunk seam is the container contract
//! ([`daemon_vhc_proto::committed_payload`]).
//!
//! The digest carry is threaded through the ingest walk: each emitted master window advances it
//! as it is emitted, in ascending window order — the sealed carry reproduces the resident round
//! state digest bit-for-bit (the digest-carry equivalence the shared vectors pin). `sparse_loco`
//! declares no replicated det state, so the sealed carry finalizes directly to the round digest;
//! a profile with replicated families would keep feeding the returned carry in canonical order.

use std::collections::BTreeMap;

use daemon_vhc_sdk_consensus::digest::DigestCarry;
use daemon_vhc_sdk_consensus::fold_walk::{windows, FoldWalk, Window};

use crate::payload::{PayloadLayout, PayloadSpan};
use crate::{median_of, scale_add, SparseLocoCfg};

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

/// One slice of update-walk output, mirroring the pinned schedule's slice actions: the windows
/// folded in this slice (ascending, contiguous with everything already folded — the new
/// error-feedback windows to `state_emit`), the committed-payload chunk pair each fold produced
/// (to externalize on the payload plane), the window reads to issue next, and whether this slice
/// sealed the walk.
#[derive(Debug)]
pub struct WalkSlice {
    /// The folded windows with their emitted f32 state, in fold order.
    pub emitted: Vec<(Window, Vec<f32>)>,
    /// The committed-payload bytes of each folded window, in fold order — appended to the open
    /// payload buffer and then forgotten.
    pub payload: Vec<(Window, PayloadWindowBytes)>,
    /// The window reads this slice issues (start their fetches, then deliver each completion
    /// back to the walk).
    pub issue: Vec<Window>,
    /// Whether the walk sealed in this slice (fires exactly once, with the last fold).
    pub sealed: bool,
}

/// One fold window's committed-payload bytes: its absmax-packed value rows followed by its packed
/// chunk-local index rows — exactly the span [`PayloadLayout`] places at that window's offset.
///
/// The producing driver APPENDS these to its open payload buffer stream and drops them; a whole
/// container is never assembled in linear memory (the emit-side mirror of [SF-R3]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PayloadWindowBytes(pub Vec<u8>);

impl PayloadWindowBytes {
    /// The bytes to append.
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        &self.0
    }
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

/// Which of a fold window's inputs a chunk-addressed ingest walk is waiting on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum IngestPart {
    /// The round-base (sealed round r−1 master) state window.
    RoundBase,
    /// A peer's packed VALUE chunk for the window (`u32` indexes the record-ordered peer).
    Values(u32),
    /// A peer's packed INDEX chunk for the window.
    Indices(u32),
}

/// One read the ingest walk asks the driver to start.
///
/// `span` is the byte range to `read_into` out of that peer's committed-payload BUFFER for the
/// payload parts ([SF-R3]: the payload never leaves the host buffer whole), and `None` for
/// [`IngestPart::RoundBase`], whose bytes come from the round-base family fold at the window's
/// absolute family offset — the driver already knows both.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IngestFetch {
    /// Which input this read carries.
    pub part: IngestPart,
    /// The fold window it belongs to.
    pub window: Window,
    /// The payload byte range to read (`None` for the round-base state window).
    pub span: Option<PayloadSpan>,
}

/// One slice of ingest-walk output: the master windows folded in this slice (ascending, contiguous
/// with everything already folded — `state_emit` them in order), the reads to issue next, and
/// whether this slice sealed the fold.
#[derive(Debug)]
pub struct IngestSlice {
    /// The folded windows with their emitted master state, in fold order.
    pub emitted: Vec<(Window, Vec<f32>)>,
    /// The reads this slice issues (start them, then deliver each completion back to the walk).
    pub issue: Vec<IngestFetch>,
    /// Whether the fold sealed in this slice (fires exactly once, with the last fold).
    pub sealed: bool,
}

/// A fold window's partially-arrived inputs (the pieces stash the walk fills before the schedule
/// may consume the window). The vectors are peer-indexed in record order.
#[derive(Debug, Default)]
struct WindowParts {
    round_base: Option<Vec<f32>>,
    values: BTreeMap<u32, Vec<f32>>,
    indices: BTreeMap<u32, Vec<u32>>,
}

/// The walk's two phases over a chunk-addressed committed set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IngestPhase {
    /// Streaming every peer's VALUE chunks to fold the contribution norms the median-norm clip
    /// needs (a parameter's whole value section, so it cannot be window-local). Skipped entirely
    /// when the profile does not clip.
    Norms,
    /// Streaming round-base + value + index chunks and folding the master windows.
    Fold,
}

/// The streamed `sparse_loco` det-lane ingest over **buffer-resident** committed payloads
/// ([SF-R3]): the resident [`crate::SparseLoco::ingest`] as a completion-driven walk that never
/// holds more than the in-flight window set — one round-base window plus one window's worth of
/// value/index rows per peer — emitting round r master windows and threading the digest carry
/// through the emission order.
///
/// Each peer's committed payload stays where `payload_get` put it: a host buffer the host already
/// blake3-verified against the record-listed hash before it delivered the completion (architecture
/// §3.4). The walk asks for byte RANGES of those buffers ([`IngestFetch::span`], computed by
/// [`PayloadLayout`]) and the driver serves them with ranged `read_into`, so no whole payload —
/// and no whole `Vec<u32>` index section — ever enters linear memory.
///
/// # Two phases, because the clip is not window-local
///
/// The median-norm clip scales a peer's contribution by `median / ‖contribution‖` over a whole
/// **parameter**, so the first window of a parameter cannot fold before that parameter's last
/// value row has been seen. The resident path solved this with a pre-pass over resident sections;
/// with the sections chunk-addressed the walk streams the pre-pass instead: phase [`Norms`] folds
/// every peer's value chunks through [`daemon_vhc_det::det_sumsq_into`] (bit-identical to the
/// resident whole-section [`daemon_vhc_det::det_l2norm`] — same index order), then phase [`Fold`]
/// re-reads the value chunks beside the index and round-base windows and folds. The value chunks
/// are therefore fetched twice; they are ~16 % of the container (the indices are the bulk), which
/// is why values and indices are separate objects. The alternative — trusting a producer-declared
/// norm — would let a peer evade its own clip, so the norms are always recomputed from the rows.
///
/// [`Norms`]: IngestPhase::Norms
/// [`Fold`]: IngestPhase::Fold
///
/// Drive: [`Self::start`] once (issue the returned reads), then [`Self::on_part_ready`] per
/// completed read; emit each returned master window in order; after the sealing slice,
/// [`Self::seal`] returns the carry.
pub struct SparseLocoIngestWalk {
    cfg: SparseLocoCfg,
    schedule: Vec<Window>,
    in_flight: u64,
    /// The container layout every peer's payload is read through (the run's own geometry).
    layout: PayloadLayout,
    /// How many committed peers the walk folds (record order is the peer index).
    peers: usize,
    /// The resident divisor: `payloads.len().max(1)`.
    count: usize,
    phase: IngestPhase,
    /// The fold cursor of the phase in flight.
    walk: FoldWalk,
    /// Windows whose inputs are still arriving (at most `in_flight` entries by construction).
    parts: BTreeMap<u64, WindowParts>,
    /// `[peer][param]` contribution-norm carry (sum of squares), folded in phase `Norms`.
    sumsq: Vec<Vec<f32>>,
    /// `[peer][param]` clip scale (1.0 when clipping is off or the norm is at/below the median).
    scale: Vec<Vec<f64>>,
    started: bool,
    carry: DigestCarry,
    byte_buf: Vec<u8>,
}

impl SparseLocoIngestWalk {
    /// Build the walk over `peers` record-ordered committed payloads, each of which the driver
    /// holds as a host buffer (hash-verified by the host before delivery).
    ///
    /// `carry` is the fresh round digest carry (seeded with the round seed at the pinned block
    /// size). The container geometry is the run's own ([`PayloadLayout`]); a peer whose header
    /// disagrees is refused by the driver at [`PayloadLayout::check_header`] before the walk starts.
    ///
    /// # Errors
    /// A `String` on a geometry violation.
    pub fn new(
        cfg: &SparseLocoCfg,
        numels: &[usize],
        window_size: u64,
        in_flight: u64,
        peers: usize,
        carry: DigestCarry,
    ) -> Result<Self, String> {
        let chunk = cfg.chunk as usize;
        validate_geometry(numels, chunk, window_size)?;
        let layout = PayloadLayout::new(cfg, numels, window_size)?;
        let schedule = layout.schedule().to_vec();
        // The norm pre-pass exists only to scale peers against each other; with no committed peers
        // (the resident `count = max(1)` degenerate) there is nothing to scale and no value rows
        // to read, so the walk opens straight into the fold.
        let phase = if cfg.clip && peers > 0 {
            IngestPhase::Norms
        } else {
            IngestPhase::Fold
        };
        Ok(Self {
            cfg: cfg.clone(),
            in_flight,
            walk: FoldWalk::new(schedule.len() as u64, in_flight),
            schedule,
            layout,
            count: peers.max(1),
            sumsq: vec![vec![0.0f32; numels.len()]; peers],
            scale: vec![vec![1.0f64; numels.len()]; peers],
            peers,
            phase,
            parts: BTreeMap::new(),
            started: false,
            carry,
            byte_buf: Vec::new(),
        })
    }

    /// The container layout the walk reads every peer's payload through — the driver uses it to
    /// cross-check each fetched payload's header before folding it.
    #[must_use]
    pub fn layout(&self) -> &PayloadLayout {
        &self.layout
    }

    /// The pinned window schedule (window ordinal ≡ master chunk ordinal ≡ payload chunk-pair
    /// ordinal) — the one map the driver turns issued windows into reads with.
    #[must_use]
    pub fn schedule(&self) -> &[Window] {
        &self.schedule
    }

    /// The opening slice: the first reads to issue (a zero-window layout seals immediately).
    ///
    /// # Errors
    /// A `String` when the walk was already started.
    pub fn start(&mut self) -> Result<IngestSlice, String> {
        if self.started {
            return Err("walk already started".into());
        }
        self.started = true;
        let actions = self.walk.start();
        let issue = self.fetches_for(&actions.issue);
        // A layout with no windows (structurally impossible — `validate_geometry` refuses an empty
        // layout) would seal in the norm phase; keep the phase advance honest anyway.
        if actions.seal && self.phase == IngestPhase::Norms {
            return self.advance_to_fold();
        }
        Ok(IngestSlice {
            emitted: Vec::new(),
            issue,
            sealed: actions.seal,
        })
    }

    /// One read completed: stash it, and once a window holds every input the phase needs, fold the
    /// maximal contiguous run now available, refill the read window, and seal with the last fold.
    ///
    /// In the norm phase a fold accumulates contribution norms and emits nothing; in the fold phase
    /// each fold scatter-adds the record-ordered payload rows of the window, rebases, applies the
    /// outer step, and advances the digest carry with the emitted master window. The transition
    /// between phases is internal: the slice that seals the norm phase carries the fold phase's
    /// opening reads.
    ///
    /// # Errors
    /// A `String` on a completion the schedule refuses (not outstanding), a mis-sized window or
    /// chunk, or a det-kernel refusal.
    pub fn on_part_ready(
        &mut self,
        part: IngestPart,
        ordinal: u64,
        bytes: &[u8],
    ) -> Result<IngestSlice, String> {
        if !self.started {
            return Err("walk not started".into());
        }
        // Decode + length-check the piece BEFORE it touches walk state, so a torn read leaves the
        // fold cursor and the digest carry untouched and the driver can re-deliver.
        match part {
            IngestPart::RoundBase => {
                let vals = le_bytes_to_f32s(bytes)?;
                check_window_input(&self.schedule, ordinal, "round-base", vals.len())?;
                self.parts.entry(ordinal).or_default().round_base = Some(vals);
            }
            IngestPart::Values(peer) => {
                let vals = self
                    .layout
                    .decode_values(ordinal, bytes)
                    .map_err(|e| format!("peer {peer}: {e}"))?;
                self.parts
                    .entry(ordinal)
                    .or_default()
                    .values
                    .insert(peer, vals);
            }
            IngestPart::Indices(peer) => {
                let idx = self
                    .layout
                    .decode_indices(ordinal, bytes)
                    .map_err(|e| format!("peer {peer}: {e}"))?;
                self.parts
                    .entry(ordinal)
                    .or_default()
                    .indices
                    .insert(peer, idx);
            }
        }
        if !self.window_complete(ordinal) {
            return Ok(IngestSlice {
                emitted: Vec::new(),
                issue: Vec::new(),
                sealed: false,
            });
        }
        // Let the schedule refuse first (duplicate / never-issued / already-folded).
        let actions = self
            .walk
            .on_completion(ordinal)
            .map_err(|e| e.to_string())?;
        let mut emitted = Vec::with_capacity(actions.fold.len());
        for &folded in &actions.fold {
            let window = self.schedule[usize::try_from(folded).expect("ordinal fits usize")];
            let parts = self
                .parts
                .remove(&folded)
                .ok_or_else(|| format!("window {folded} foldable without stashed inputs"))?;
            match self.phase {
                IngestPhase::Norms => self.fold_norms(&window, &parts),
                IngestPhase::Fold => {
                    let master = self.fold_window(&window, &parts)?;
                    f32s_to_le_bytes(&master, &mut self.byte_buf);
                    self.carry.update(&self.byte_buf);
                    emitted.push((window, master));
                }
            }
        }
        let mut issue = self.fetches_for(&actions.issue);
        if actions.seal && self.phase == IngestPhase::Norms {
            let opening = self.advance_to_fold()?;
            issue = opening.issue;
            return Ok(IngestSlice {
                emitted,
                issue,
                sealed: opening.sealed,
            });
        }
        Ok(IngestSlice {
            emitted,
            issue,
            sealed: actions.seal,
        })
    }

    /// Finish a sealed walk, returning the digest carry advanced over the full master family in
    /// canonical order. `sparse_loco` declares no replicated det state, so `seal().finalize()` IS
    /// the round state digest; a profile with replicated families would keep feeding the carry with
    /// its replicated chunks in canonical order first.
    ///
    /// # Errors
    /// A `String` when the walk has not sealed (windows are still outstanding).
    pub fn seal(self) -> Result<DigestCarry, String> {
        if self.phase != IngestPhase::Fold || !self.walk.is_sealed() {
            return Err("ingest walk sealed before every window folded".into());
        }
        Ok(self.carry)
    }

    /// Close the norm phase: turn each parameter's per-peer norm carries into the median-norm clip
    /// scales (the resident pre-pass's arithmetic, over carries instead of whole sections), then
    /// open the fold phase and return its first reads.
    fn advance_to_fold(&mut self) -> Result<IngestSlice, String> {
        for param in 0..self.scale.first().map_or(0, Vec::len) {
            let norms: Vec<f64> = self
                .sumsq
                .iter()
                .map(|peer| f64::from(peer[param].sqrt()))
                .collect();
            let median = median_of(&norms);
            for (peer, &norm) in self.scale.iter_mut().zip(norms.iter()) {
                peer[param] = if norm > median && norm > 0.0 {
                    median / norm
                } else {
                    1.0
                };
            }
        }
        self.phase = IngestPhase::Fold;
        self.walk = FoldWalk::new(self.schedule.len() as u64, self.in_flight);
        self.parts.clear();
        let actions = self.walk.start();
        Ok(IngestSlice {
            emitted: Vec::new(),
            issue: self.fetches_for(&actions.issue),
            sealed: actions.seal,
        })
    }

    /// The reads one set of issued window ordinals needs in the phase in flight: value chunks only
    /// while folding norms; round base + value + index chunks while folding masters.
    fn fetches_for(&self, ordinals: &[u64]) -> Vec<IngestFetch> {
        let mut out = Vec::with_capacity(ordinals.len() * (1 + 2 * self.peers));
        for &ordinal in ordinals {
            let window = self.schedule[usize::try_from(ordinal).expect("ordinal fits usize")];
            if self.phase == IngestPhase::Fold {
                out.push(IngestFetch {
                    part: IngestPart::RoundBase,
                    window,
                    span: None,
                });
            }
            for p in 0..self.peers {
                let peer = u32::try_from(p).expect("peer index fits u32");
                out.push(IngestFetch {
                    part: IngestPart::Values(peer),
                    window,
                    span: Some(self.layout.values_span(ordinal)),
                });
                if self.phase == IngestPhase::Fold {
                    out.push(IngestFetch {
                        part: IngestPart::Indices(peer),
                        window,
                        span: Some(self.layout.indices_span(ordinal)),
                    });
                }
            }
        }
        out
    }

    /// Whether a window holds every input the phase in flight folds with.
    fn window_complete(&self, ordinal: u64) -> bool {
        let Some(parts) = self.parts.get(&ordinal) else {
            return false;
        };
        let peers = self.peers;
        match self.phase {
            IngestPhase::Norms => parts.values.len() == peers,
            IngestPhase::Fold => {
                parts.round_base.is_some()
                    && parts.values.len() == peers
                    && parts.indices.len() == peers
            }
        }
    }

    /// The norm-phase fold: accumulate each peer's window value rows into its (peer, parameter)
    /// contribution-norm carry, in ascending row order — the resident whole-section `det_l2norm`
    /// bit-for-bit.
    fn fold_norms(&mut self, window: &Window, parts: &WindowParts) {
        let param = window.param as usize;
        for (&peer, vals) in &parts.values {
            daemon_vhc_det::det_sumsq_into(&mut self.sumsq[peer as usize][param], vals);
        }
    }

    /// The window-local fold: the resident per-parameter ingest sequence sliced to the window's
    /// chunk rows — record-ordered scatter-adds of the clip-scaled decoded rows into a zeroed
    /// accumulator, rebase copy, single axpy.
    fn fold_window(&self, window: &Window, parts: &WindowParts) -> Result<Vec<f32>, String> {
        let chunk = self.cfg.chunk as usize;
        let (_, elems) = window_elems(window);
        let param = window.param as usize;
        let round_base = parts
            .round_base
            .as_ref()
            .ok_or("a folded window holds its round base")?;

        let mut acc = vec![0.0f32; elems];
        for peer in 0..self.peers {
            let key = u32::try_from(peer).expect("peer index fits u32");
            let vals = parts
                .values
                .get(&key)
                .ok_or_else(|| format!("window fold missing peer {peer} values"))?;
            let idx = parts
                .indices
                .get(&key)
                .ok_or_else(|| format!("window fold missing peer {peer} indices"))?;
            let scaled = daemon_vhc_det::det_scale(vals, self.scale[peer][param]);
            daemon_vhc_det::det_chunk_scatter_add(&mut acc, &scaled, idx, chunk)
                .map_err(|e| format!("scatter_add: {e:?}"))?;
        }
        // θ⁽ᵗ⁺¹⁾ = θ⁽ᵗ⁾ − α·(1/R)·Σ Δ̂, window-sliced (rebase copy, then the canonical axpy).
        let mut master = round_base.clone();
        #[allow(clippy::cast_precision_loss)]
        daemon_vhc_det::det_axpy(&mut master, -self.cfg.outer_alpha / self.count as f64, &acc)
            .map_err(|e| format!("axpy: {e:?}"))?;
        Ok(master)
    }
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
/// windows, emitting the new error-feedback windows AND the committed payload's chunk pair as it
/// folds; [`Self::seal`] yields the container's index document.
///
/// The emitted `WalkSlice.emitted` windows are the NEW ef family (replica-local,
/// digest-invisible) — the driver `state_emit`s them. `WalkSlice.payload` carries the window's
/// committed-payload bytes, which the driver APPENDS to its open payload buffer and drops: the
/// producing side never holds the whole container either (its residency used to be the mirror image
/// of the consuming side's — one `Vec` per parameter, assembled across the whole walk and encoded
/// whole at seal). The walk keeps only the running byte count.
pub struct SparseLocoUpdateWalk {
    core: WalkCore<UpdateWindowInputs>,
    cfg: SparseLocoCfg,
    /// The container layout the appended sections tile (the header comes from here too).
    layout: PayloadLayout,
    /// Bytes appended so far (the seal cross-check against the layout's total).
    appended: u64,
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
            layout: PayloadLayout::new(cfg, numels, window_size)?,
            cfg: cfg.clone(),
            appended: 0,
        })
    }

    /// The committed container's fixed header — the FIRST bytes the driver appends to its payload
    /// buffer, before any window folds (its geometry is known at construction).
    #[must_use]
    pub fn payload_header(&self) -> Vec<u8> {
        self.layout.header()
    }

    /// The container layout the appended sections tile.
    #[must_use]
    pub fn layout(&self) -> &PayloadLayout {
        &self.layout
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
            payload: Vec::new(),
            issue,
            sealed,
        })
    }

    /// One window's inputs are complete: fold the maximal contiguous run now available (each
    /// fold runs Δ → error-feedback accumulate → per-row top-k → per-row absmax pack, records the
    /// window's payload chunk pair, and emits the new ef window), refill the read window, and seal
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
        let mut payload = Vec::with_capacity(folds.len());
        for (window, inputs) in folds {
            let (ef_new, chunks) = self.fold_window(&window, &inputs)?;
            emitted.push((window, ef_new));
            payload.push((window, chunks));
        }
        Ok(WalkSlice {
            emitted,
            payload,
            issue,
            sealed,
        })
    }

    /// Finish a sealed walk: the container's total byte length, cross-checked against what the walk
    /// actually appended (header + every window's sections). The driver seals its payload buffer
    /// and `payload_put`s it — the blob's blake3 is the commitment, exactly as before.
    ///
    /// # Errors
    /// A `String` when the walk has not sealed (windows are still outstanding) or the appended
    /// bytes do not tile the layout (a mis-framed producer, caught before anything is published).
    pub fn seal(self) -> Result<u64, String> {
        if !self.core.is_sealed() {
            return Err("update walk sealed before every window folded".into());
        }
        let want = self.layout.total_len();
        let got = crate::payload::HEADER_BYTES + self.appended;
        if got != want {
            return Err(format!(
                "the appended committed payload is {got} bytes, the layout says {want}"
            ));
        }
        Ok(want)
    }

    /// The window-local update math: the resident per-parameter sequence sliced to the window's
    /// chunk rows — `Δ = θ⁽ᵗ⁾ − θ`, `acc = β·ef + Δ`, per-row top-k, per-row absmax pack,
    /// `ef ← acc − scatter(dequant(sent))` — plus the window's payload chunk pair.
    fn fold_window(
        &mut self,
        window: &Window,
        inputs: &UpdateWindowInputs,
    ) -> Result<(Vec<f32>, PayloadWindowBytes), String> {
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

        // The window's committed-payload section pair: the packed value rows verbatim + the same
        // indices at their own bit width — the span the layout places at this window's offset.
        let packed_idx = daemon_vhc_det::pack_chunk_indices(&idx, chunk)
            .map_err(|e| format!("pack indices: {e:?}"))?;
        let (_, vlen) = self.layout.values_span(window.ordinal);
        let (_, ilen) = self.layout.indices_span(window.ordinal);
        if packed.len() as u64 != vlen || packed_idx.len() as u64 != ilen {
            return Err(format!(
                "window {} folded {} value + {} index bytes, the layout reserves {vlen} + {ilen}",
                window.ordinal,
                packed.len(),
                packed_idx.len()
            ));
        }
        let mut bytes = packed;
        bytes.extend_from_slice(&packed_idx);
        self.appended += bytes.len() as u64;
        Ok((ef_new, PayloadWindowBytes(bytes)))
    }
}
