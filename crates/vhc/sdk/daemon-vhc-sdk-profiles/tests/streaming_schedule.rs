// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// Schedule conformance of the streamed fold walks against the SHARED slice-decomposition
// vectors (`daemon-vhc-sdk-consensus/tests/fixtures/fold-walk-vectors.json`): the vectors pin
// the schedule contract on the raw `FoldWalk` state machine in its home crate; this suite pins
// that the PROFILE WALKS (ingest and make_update, which wrap the schedule around real det
// math) reproduce the identical window enumeration and fold/issue/seal order under the same
// arrival permutations — the engine preserves the pinned contract, it does not re-derive it.

use daemon_vhc_sdk_consensus::digest::DigestCarry;
use daemon_vhc_sdk_profiles::payload::PayloadLayout;
use daemon_vhc_sdk_profiles::streaming::{
    f32s_to_le_bytes, IngestFetch, IngestPart, SparseLocoIngestWalk, SparseLocoUpdateWalk,
    UpdateWindowInputs, WalkSlice,
};
use daemon_vhc_sdk_profiles::{ParamView, SparseLoco, SparseLocoCfg};

use daemon_vhc_proto::bytes::Seed;

/// The SHARED schedule vectors, embedded from their home crate's fixture tree (the sibling is a
/// normal dependency, so the path is workspace-guaranteed).
const VECTORS: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../daemon-vhc-sdk-consensus/tests/fixtures/fold-walk-vectors.json"
));

#[derive(serde::Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}

#[derive(serde::Deserialize)]
struct Case {
    name: String,
    numels: Vec<usize>,
    window_size: u64,
    in_flight: u64,
    windows: Vec<(u64, u32, u64, u64)>,
    arrivals: Vec<u64>,
    slices: Vec<Slice>,
}

#[derive(serde::Deserialize)]
struct Slice {
    fold: Vec<u64>,
    issue: Vec<u64>,
    seal: bool,
}

/// Deterministic per-parameter f32 values (seeded LCG, the shared-vector generator family).
fn param_values(seed: u64, numel: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(numel);
    let mut s = seed;
    for _ in 0..numel {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        #[allow(clippy::cast_precision_loss)]
        out.push(((s >> 40) as f32) / 16_777_216.0);
    }
    out
}

/// The widest profile chunk compatible with a vector case's geometry: divides every numel AND
/// the window's element count (so `window_size` is a multiple of `chunk × 4`).
fn compatible_chunk(numels: &[usize], window_size: u64) -> u32 {
    let window_elems = usize::try_from(window_size / 4).unwrap();
    let cap = *numels.iter().min().unwrap();
    (1..=cap)
        .rev()
        .find(|&c| numels.iter().all(|n| n % c == 0) && window_elems % c == 0)
        .map(|c| u32::try_from(c).unwrap())
        .unwrap()
}

fn cfg_for(numels: &[usize], window_size: u64) -> SparseLocoCfg {
    let chunk = compatible_chunk(numels, window_size);
    SparseLocoCfg {
        h: 1,
        ef_decay: 0.95,
        chunk,
        topk: chunk.min(2),
        bits: 2,
        outer_alpha: 1.0,
        clip: true,
    }
}

/// The observed slice actions of a walk step, as ordinals.
fn observe(step: &WalkSlice) -> (Vec<u64>, Vec<u64>, bool) {
    (
        step.emitted.iter().map(|(w, _)| w.ordinal).collect(),
        step.issue.iter().map(|w| w.ordinal).collect(),
        step.sealed,
    )
}

fn assert_slice(case: &str, slice_no: usize, got: &(Vec<u64>, Vec<u64>, bool), want: &Slice) {
    assert_eq!(got.0, want.fold, "{case}: slice {slice_no} fold order");
    assert_eq!(got.1, want.issue, "{case}: slice {slice_no} issues");
    assert_eq!(got.2, want.seal, "{case}: slice {slice_no} seal");
}

/// Record-ordered payloads for a case's geometry, produced by the resident profile (two peers
/// with distinct trajectories).
fn payloads_for(
    cfg: &SparseLocoCfg,
    numels: &[usize],
) -> Vec<Vec<daemon_vhc_sdk_profiles::Section>> {
    (0..2u64)
        .map(|peer| {
            let base: Vec<Vec<f32>> = numels
                .iter()
                .enumerate()
                .map(|(i, &n)| param_values(1000 + i as u64, n))
                .collect();
            let theta: Vec<Vec<f32>> = numels
                .iter()
                .enumerate()
                .map(|(i, &n)| param_values(2000 + peer * 100 + i as u64, n))
                .collect();
            let mut profile = SparseLoco::new(cfg.clone(), numels);
            let views: Vec<ParamView<'_>> = base
                .iter()
                .zip(theta.iter())
                .map(|(b, t)| ParamView {
                    theta: t,
                    round_base: b,
                })
                .collect();
            profile.make_update(&views)
        })
        .collect()
}

#[test]
fn profile_walks_reproduce_the_shared_schedule_vectors() {
    let fixture: Fixture = serde_json::from_str(VECTORS).unwrap();
    assert!(!fixture.cases.is_empty());

    for case in &fixture.cases {
        let cfg = cfg_for(&case.numels, case.window_size);
        let payloads = payloads_for(&cfg, &case.numels);
        let bases: Vec<Vec<f32>> = case
            .numels
            .iter()
            .enumerate()
            .map(|(i, &n)| param_values(1000 + i as u64, n))
            .collect();
        let thetas: Vec<Vec<f32>> = case
            .numels
            .iter()
            .enumerate()
            .map(|(i, &n)| param_values(3000 + i as u64, n))
            .collect();
        let efs: Vec<Vec<f32>> = case.numels.iter().map(|&n| vec![0.0; n]).collect();

        // -- ingest walk ------------------------------------------------------------------
        //
        // A committed payload stays in its host buffer, so a window's inputs arrive as several ranged
        // reads (its round base plus a value and an index section per peer) and the walk runs the clip
        // pre-pass over the same schedule before the emitting fold phase. Both phases must walk the
        // pinned window order; the vectors' fold/seal decomposition is asserted against the
        // EMITTING phase (the pre-pass emits nothing by construction).
        let layout = PayloadLayout::new(&cfg, &case.numels, case.window_size).unwrap();
        let buffers: Vec<Vec<u8>> = payloads
            .iter()
            .map(|s| layout.encode_sections(s).unwrap())
            .collect();
        let mut ingest = SparseLocoIngestWalk::new(
            &cfg,
            &case.numels,
            case.window_size,
            case.in_flight,
            buffers.len(),
            DigestCarry::new(&Seed([7; 32]), 64),
        )
        .unwrap();

        // The pinned window enumeration IS the walk's schedule.
        let schedule: Vec<(u64, u32, u64, u64)> = ingest
            .schedule()
            .iter()
            .map(|w| (w.ordinal, w.param, w.param_off, w.len))
            .collect();
        assert_eq!(schedule, case.windows, "{}: window enumeration", case.name);

        // Deliver one window's parts as a group, in the pinned arrival order, so each window
        // produces exactly one slice — the shape the vectors pin.
        let mut byte_buf = Vec::new();
        let deliver = |walk: &mut SparseLocoIngestWalk,
                       pending: &mut Vec<IngestFetch>,
                       ordinal: u64,
                       byte_buf: &mut Vec<u8>|
         -> (Vec<u64>, Vec<u64>, bool) {
            let mut fold = Vec::new();
            let mut issue = Vec::new();
            let mut seal = false;
            let (mine, rest): (Vec<IngestFetch>, Vec<IngestFetch>) = pending
                .iter()
                .copied()
                .partition(|f| f.window.ordinal == ordinal);
            *pending = rest;
            assert!(!mine.is_empty(), "the pinned arrival must be outstanding");
            for fetch in mine {
                let bytes = match fetch.span {
                    Some((off, len)) => {
                        let peer = match fetch.part {
                            IngestPart::Values(p) | IngestPart::Indices(p) => p as usize,
                            IngestPart::RoundBase => unreachable!(),
                        };
                        buffers[peer][off as usize..(off + len) as usize].to_vec()
                    }
                    None => {
                        let w = fetch.window;
                        let (off, elems) = (
                            usize::try_from(w.param_off / 4).unwrap(),
                            usize::try_from(w.len / 4).unwrap(),
                        );
                        f32s_to_le_bytes(&bases[w.param as usize][off..off + elems], byte_buf);
                        byte_buf.clone()
                    }
                };
                let step = walk
                    .on_part_ready(fetch.part, fetch.window.ordinal, &bytes)
                    .unwrap();
                fold.extend(step.emitted.iter().map(|(w, _)| w.ordinal));
                for f in &step.issue {
                    if !issue.contains(&f.window.ordinal) {
                        issue.push(f.window.ordinal);
                    }
                    pending.push(*f);
                }
                seal |= step.sealed;
            }
            (fold, issue, seal)
        };

        // Phase 1 (the clip pre-pass): the same window order, no emissions.
        let mut pending: Vec<IngestFetch> = Vec::new();
        let opening = ingest.start().unwrap();
        let mut opening_windows: Vec<u64> = Vec::new();
        for f in &opening.issue {
            if !opening_windows.contains(&f.window.ordinal) {
                opening_windows.push(f.window.ordinal);
            }
            pending.push(*f);
        }
        assert_eq!(
            opening_windows, case.slices[0].issue,
            "{}: pre-pass opening issues the pinned windows",
            case.name
        );
        // The pre-pass runs the same arrival order; its last window's slice carries the emitting
        // phase's opening issues (the phase transition is internal to the walk).
        let mut phase_two_opening: Vec<u64> = Vec::new();
        for (i, &arrival) in case.arrivals.iter().enumerate() {
            let (fold, issue, seal) = deliver(&mut ingest, &mut pending, arrival, &mut byte_buf);
            assert!(
                fold.is_empty() && !seal,
                "{}: the clip pre-pass emits nothing and never seals the walk",
                case.name
            );
            if i + 1 < case.arrivals.len() {
                assert_eq!(
                    issue,
                    case.slices[i + 1].issue,
                    "{}: pre-pass slice {} issues",
                    case.name,
                    i + 1
                );
            } else {
                phase_two_opening = issue;
            }
        }
        assert_eq!(
            phase_two_opening, case.slices[0].issue,
            "{}: the emitting phase opens on the pinned windows",
            case.name
        );

        // Phase 2 (the emitting fold): the pinned fold/issue/seal decomposition, verbatim.
        for (i, &arrival) in case.arrivals.iter().enumerate() {
            let step = deliver(&mut ingest, &mut pending, arrival, &mut byte_buf);
            assert_slice(&case.name, i + 1, &step, &case.slices[i + 1]);
        }
        ingest.seal().unwrap();

        // -- update walk (same schedule contract, different per-window math) ---------------
        let mut update =
            SparseLocoUpdateWalk::new(&cfg, &case.numels, case.window_size, case.in_flight)
                .unwrap();
        let opening = observe(&update.start().unwrap());
        assert_slice(&case.name, 0, &opening, &case.slices[0]);
        for (i, &arrival) in case.arrivals.iter().enumerate() {
            let w = update.schedule()[usize::try_from(arrival).unwrap()];
            let (off, elems) = (
                usize::try_from(w.param_off / 4).unwrap(),
                usize::try_from(w.len / 4).unwrap(),
            );
            let p = w.param as usize;
            let inputs = UpdateWindowInputs {
                theta: thetas[p][off..off + elems].to_vec(),
                round_base: bases[p][off..off + elems].to_vec(),
                ef: efs[p][off..off + elems].to_vec(),
            };
            let step = observe(&update.on_window_ready(arrival, inputs).unwrap());
            assert_slice(&case.name, i + 1, &step, &case.slices[i + 1]);
        }
        update.seal().unwrap();
    }
}
