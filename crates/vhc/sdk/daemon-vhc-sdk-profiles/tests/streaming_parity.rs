// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// The windowed ≡ resident parity suite: the streamed fold walks reproduce the resident
// `sparse_loco` path BIT-FOR-BIT — emitted masters, payload sections, error-feedback state,
// and round state digests — across window geometries (degenerate single-window, one-chunk
// windows, short parameter tails, a ceremony-shaped scaled layout), in-flight bounds, and
// completion arrival permutations. The resident implementation is the oracle (itself pinned by
// the crate's golden suite), so a mismatch is a fold-engine bug, never tolerance: the det lane
// is CPU fp32 with fixed evaluation order on both sides, and the walk executes the identical
// operation sequence window-sliced.
//
// The digest side doubles as the digest-carry equivalence obligation at the engine level: the
// carry threaded through the ingest walk must finalize to
// `digest_state(seed, 64, u32::MAX, post-ingest master image)` — the exact resident formula
// over the exact resident bytes (the existing pinned goldens transfer by this identity).

use daemon_vhc_proto::bytes::{Seed, StateDigest};
use daemon_vhc_sdk_consensus::digest::{digest_state, DigestCarry};
use daemon_vhc_sdk_profiles::streaming::{
    f32s_to_le_bytes, le_bytes_to_f32s, SparseLocoIngestWalk, SparseLocoUpdateWalk,
    UpdateWindowInputs,
};
use daemon_vhc_sdk_profiles::{IngestParam, ParamView, Section, SparseLoco, SparseLocoCfg};

// ================================================================================================
// deterministic inputs + drivers
// ================================================================================================

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

fn family(seed: u64, numels: &[usize]) -> Vec<Vec<f32>> {
    numels
        .iter()
        .enumerate()
        .map(|(i, &n)| param_values(seed + i as u64, n))
        .collect()
}

fn bits_of(fam: &[Vec<f32>]) -> Vec<Vec<u32>> {
    fam.iter()
        .map(|p| p.iter().map(|v| v.to_bits()).collect())
        .collect()
}

/// The family's f32-le byte image (registration order) — the digest coverage.
fn family_bytes(fam: &[Vec<f32>]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut buf = Vec::new();
    for p in fam {
        f32s_to_le_bytes(p, &mut buf);
        out.extend_from_slice(&buf);
    }
    out
}

/// How a driver picks the next completion among the outstanding reads.
#[derive(Clone, Copy)]
enum Arrival {
    /// Oldest issue first (in-order network).
    Fifo,
    /// Newest issue first (maximally reversed within the in-flight window).
    Lifo,
    /// Seeded pseudo-random pick (adversarial reordering).
    Shuffled(u64),
}

fn pick(outstanding: &mut Vec<u64>, arrival: Arrival, lcg: &mut u64) -> u64 {
    let i = match arrival {
        Arrival::Fifo => 0,
        Arrival::Lifo => outstanding.len() - 1,
        Arrival::Shuffled(_) => {
            *lcg = lcg
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            usize::try_from(*lcg >> 33).unwrap() % outstanding.len()
        }
    };
    outstanding.remove(i)
}

fn arrival_seed(arrival: Arrival) -> u64 {
    match arrival {
        Arrival::Shuffled(s) => s,
        _ => 0,
    }
}

/// Drive the streamed ingest walk to completion, asserting the schedule invariants along the
/// way (ascending contiguous folds; the seal exactly once) — returns the emitted master family
/// and the finalized round state digest.
// One flat driver signature keeps every parity call site self-describing (test-only helper).
#[allow(clippy::too_many_arguments)]
fn drive_ingest(
    cfg: &SparseLocoCfg,
    numels: &[usize],
    window_size: u64,
    in_flight: u64,
    payloads: &[Vec<Section>],
    seed: &Seed,
    bases: &[Vec<f32>],
    arrival: Arrival,
) -> (Vec<Vec<f32>>, StateDigest) {
    let mut walk = SparseLocoIngestWalk::new(
        cfg,
        numels,
        window_size,
        in_flight,
        payloads,
        DigestCarry::new(seed, 64),
    )
    .unwrap();
    let schedule = walk.schedule().to_vec();
    let mut masters: Vec<Vec<f32>> = numels.iter().map(|&n| vec![0.0f32; n]).collect();
    let mut outstanding: Vec<u64> = Vec::new();
    let mut lcg = arrival_seed(arrival);
    let mut next_fold = 0u64;
    let mut seals = 0u32;

    let opening = walk.start().unwrap();
    assert!(opening.emitted.is_empty());
    outstanding.extend(opening.issue.iter().map(|w| w.ordinal));
    assert!(
        outstanding.len() as u64 <= in_flight.max(1),
        "bounded issue"
    );
    seals += u32::from(opening.sealed);

    while !outstanding.is_empty() {
        let ordinal = pick(&mut outstanding, arrival, &mut lcg);
        let w = schedule[usize::try_from(ordinal).unwrap()];
        let (off, elems) = (
            usize::try_from(w.param_off / 4).unwrap(),
            usize::try_from(w.len / 4).unwrap(),
        );
        let base_win = bases[w.param as usize][off..off + elems].to_vec();
        let step = walk.on_window_ready(ordinal, &base_win).unwrap();
        for (win, master) in &step.emitted {
            assert_eq!(win.ordinal, next_fold, "folds ascending and contiguous");
            next_fold += 1;
            let (o, e) = (
                usize::try_from(win.param_off / 4).unwrap(),
                usize::try_from(win.len / 4).unwrap(),
            );
            masters[win.param as usize][o..o + e].copy_from_slice(master);
        }
        outstanding.extend(step.issue.iter().map(|w| w.ordinal));
        seals += u32::from(step.sealed);
    }
    assert_eq!(next_fold, schedule.len() as u64, "every window folded");
    assert_eq!(seals, 1, "the seal fires exactly once");
    let digest = walk.seal().unwrap().finalize();
    (masters, digest)
}

/// Drive the streamed update walk to completion — returns the assembled payload sections and
/// the emitted new error-feedback family.
// One flat driver signature keeps every parity call site self-describing (test-only helper).
#[allow(clippy::too_many_arguments)]
fn drive_update(
    cfg: &SparseLocoCfg,
    numels: &[usize],
    window_size: u64,
    in_flight: u64,
    thetas: &[Vec<f32>],
    bases: &[Vec<f32>],
    efs: &[Vec<f32>],
    arrival: Arrival,
) -> (Vec<Section>, Vec<Vec<f32>>) {
    let mut walk = SparseLocoUpdateWalk::new(cfg, numels, window_size, in_flight).unwrap();
    let schedule = walk.schedule().to_vec();
    let mut ef_new: Vec<Vec<f32>> = numels.iter().map(|&n| vec![0.0f32; n]).collect();
    let mut outstanding: Vec<u64> = Vec::new();
    let mut lcg = arrival_seed(arrival);
    let mut seals = 0u32;

    let opening = walk.start().unwrap();
    outstanding.extend(opening.issue.iter().map(|w| w.ordinal));
    seals += u32::from(opening.sealed);

    while !outstanding.is_empty() {
        let ordinal = pick(&mut outstanding, arrival, &mut lcg);
        let w = schedule[usize::try_from(ordinal).unwrap()];
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
        let step = walk.on_window_ready(ordinal, inputs).unwrap();
        for (win, ef) in &step.emitted {
            let (o, e) = (
                usize::try_from(win.param_off / 4).unwrap(),
                usize::try_from(win.len / 4).unwrap(),
            );
            ef_new[win.param as usize][o..o + e].copy_from_slice(ef);
        }
        outstanding.extend(step.issue.iter().map(|w| w.ordinal));
        seals += u32::from(step.sealed);
    }
    assert_eq!(seals, 1);
    (walk.seal().unwrap(), ef_new)
}

/// The resident ingest oracle over the same inputs.
fn resident_ingest(
    cfg: &SparseLocoCfg,
    numels: &[usize],
    payloads: &[Vec<Section>],
    bases: &[Vec<f32>],
) -> Vec<Vec<f32>> {
    let mut profile = SparseLoco::new(cfg.clone(), numels);
    let mut masters: Vec<Vec<f32>> = bases.to_vec();
    let mut params: Vec<IngestParam<'_>> = masters
        .iter_mut()
        .zip(bases.iter())
        .map(|(m, b)| IngestParam {
            master: m,
            round_base: b,
        })
        .collect();
    profile.ingest(&mut params, payloads).unwrap();
    masters
}

// ================================================================================================
// geometry matrix
// ================================================================================================

struct Geometry {
    name: &'static str,
    numels: &'static [usize],
    cfg: SparseLocoCfg,
    /// Window sizes to sweep (bytes; each a multiple of `chunk × 4`).
    window_sizes: &'static [u64],
}

fn geometries() -> Vec<Geometry> {
    vec![
        // The ceremony-shaped scaled layout of the shared digest-carry vectors: the frozen
        // model's registration order (token embedding; attn-norm, wq, wk, wv, wo, ffn-norm,
        // w1, w3, w2; final norm) with d_model-wide norms at profile chunk 16 — windows of 3
        // chunks (short tails), 1 chunk (the smallest legal), and whole-family (degenerate).
        Geometry {
            name: "ceremony-shaped scaled",
            numels: &[1024, 16, 256, 256, 256, 256, 16, 768, 768, 768, 16],
            cfg: SparseLocoCfg {
                h: 1,
                ef_decay: 0.95,
                chunk: 16,
                topk: 4,
                bits: 2,
                outer_alpha: 1.0,
                clip: true,
            },
            window_sizes: &[192, 64, 1 << 20],
        },
        // The acceptance-tier degenerate geometry: one 64-dim parameter, one window ≥ the
        // family — the same code path, single-window walk.
        Geometry {
            name: "64-dim degenerate",
            numels: &[64],
            cfg: SparseLocoCfg {
                h: 1,
                ef_decay: 0.9,
                chunk: 64,
                topk: 8,
                bits: 2,
                outer_alpha: 0.7,
                clip: true,
            },
            window_sizes: &[4096],
        },
        // Short parameter tails (72 f32 at 64-byte windows: 4 whole + one 32-byte tail) with
        // clipping OFF — the unclipped scale path.
        Geometry {
            name: "short tails, clip off",
            numels: &[72, 16],
            cfg: SparseLocoCfg {
                h: 1,
                ef_decay: 0.95,
                chunk: 4,
                topk: 2,
                bits: 2,
                outer_alpha: 1.0,
                clip: false,
            },
            window_sizes: &[64],
        },
    ]
}

/// Record-ordered payloads from R resident peers over distinct trajectories (peers share the
/// round base; each trains its own θ).
fn peer_payloads(
    cfg: &SparseLocoCfg,
    numels: &[usize],
    bases: &[Vec<f32>],
    theta_seed: u64,
    peers: u64,
) -> Vec<Vec<Section>> {
    (0..peers)
        .map(|peer| {
            let theta = family(theta_seed + peer * 1000, numels);
            let mut profile = SparseLoco::new(cfg.clone(), numels);
            let views: Vec<ParamView<'_>> = bases
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

// ================================================================================================
// parity suites
// ================================================================================================

/// Streamed ingest ≡ resident ingest, and the threaded digest carry ≡ the resident digest
/// formula — across the geometry matrix, window sizes, in-flight bounds, and arrival
/// permutations.
#[test]
fn windowed_ingest_is_bit_identical_to_resident() {
    for g in geometries() {
        let bases = family(50, g.numels);
        let payloads = peer_payloads(&g.cfg, g.numels, &bases, 9000, 4);
        let want = resident_ingest(&g.cfg, g.numels, &payloads, &bases);
        let seed = Seed([0xA5; 32]);
        let want_digest = digest_state(&seed, 64, u32::MAX, &family_bytes(&want));

        for &window_size in g.window_sizes {
            for in_flight in [1u64, 3, 8] {
                for arrival in [Arrival::Fifo, Arrival::Lifo, Arrival::Shuffled(0xD1CE)] {
                    let (got, got_digest) = drive_ingest(
                        &g.cfg,
                        g.numels,
                        window_size,
                        in_flight,
                        &payloads,
                        &seed,
                        &bases,
                        arrival,
                    );
                    assert_eq!(
                        bits_of(&got),
                        bits_of(&want),
                        "{}: masters (window {window_size}, in-flight {in_flight})",
                        g.name
                    );
                    assert_eq!(
                        got_digest, want_digest,
                        "{}: digest (window {window_size}, in-flight {in_flight})",
                        g.name
                    );
                }
            }
        }
    }
}

/// Streamed make_update ≡ resident make_update: payload sections AND the rewritten
/// error-feedback family, bit-for-bit.
#[test]
fn windowed_update_is_bit_identical_to_resident() {
    for g in geometries() {
        let bases = family(50, g.numels);
        let theta = family(70, g.numels);
        // Non-zero starting ef: run the resident profile one round first, then compare round 2
        // (parity must hold with real residuals, not just the zero init).
        let mut resident = SparseLoco::new(g.cfg.clone(), g.numels);
        let views: Vec<ParamView<'_>> = bases
            .iter()
            .zip(theta.iter())
            .map(|(b, t)| ParamView {
                theta: t,
                round_base: b,
            })
            .collect();
        let _ = resident.make_update(&views);
        let ef_start: Vec<Vec<f32>> = resident.ef_state().to_vec();

        let theta2 = family(90, g.numels);
        let views2: Vec<ParamView<'_>> = bases
            .iter()
            .zip(theta2.iter())
            .map(|(b, t)| ParamView {
                theta: t,
                round_base: b,
            })
            .collect();
        let want_sections = resident.make_update(&views2);
        let want_ef: Vec<Vec<f32>> = resident.ef_state().to_vec();

        for &window_size in g.window_sizes {
            for in_flight in [1u64, 3, 8] {
                for arrival in [Arrival::Fifo, Arrival::Lifo, Arrival::Shuffled(0xBEEF)] {
                    let (got_sections, got_ef) = drive_update(
                        &g.cfg,
                        g.numels,
                        window_size,
                        in_flight,
                        &theta2,
                        &bases,
                        &ef_start,
                        arrival,
                    );
                    assert_eq!(
                        got_sections, want_sections,
                        "{}: sections (window {window_size}, in-flight {in_flight})",
                        g.name
                    );
                    assert_eq!(
                        bits_of(&got_ef),
                        bits_of(&want_ef),
                        "{}: ef (window {window_size}, in-flight {in_flight})",
                        g.name
                    );
                }
            }
        }
    }
}

/// A two-round trajectory on the ceremony-shaped layout: the windowed walks thread their own
/// error-feedback and master families across rounds (sealed master(r) IS the round base of
/// r+1) and stay bit-identical to the resident trajectory — payloads, masters, ef, digests.
#[test]
fn windowed_trajectory_tracks_resident_across_rounds() {
    let g = &geometries()[0];
    let (window_size, in_flight) = (192u64, 3u64);
    let peers = 3usize;

    // Resident side: one profile per peer (ef persists inside).
    let mut resident: Vec<SparseLoco> = (0..peers)
        .map(|_| SparseLoco::new(g.cfg.clone(), g.numels))
        .collect();
    // Windowed side: explicit ef families per peer (host-side family in production).
    let mut windowed_ef: Vec<Vec<Vec<f32>>> = (0..peers)
        .map(|_| g.numels.iter().map(|&n| vec![0.0f32; n]).collect())
        .collect();

    let mut base_resident = family(50, g.numels);
    let mut base_windowed = base_resident.clone();

    for round in 0..2u64 {
        let mut payloads_resident = Vec::with_capacity(peers);
        let mut payloads_windowed = Vec::with_capacity(peers);
        for (peer, profile) in resident.iter_mut().enumerate() {
            let theta = family(10_000 + round * 100 + peer as u64 * 7, g.numels);
            let views: Vec<ParamView<'_>> = base_resident
                .iter()
                .zip(theta.iter())
                .map(|(b, t)| ParamView {
                    theta: t,
                    round_base: b,
                })
                .collect();
            payloads_resident.push(profile.make_update(&views));

            let (sections, ef_new) = drive_update(
                &g.cfg,
                g.numels,
                window_size,
                in_flight,
                &theta,
                &base_windowed,
                &windowed_ef[peer],
                Arrival::Shuffled(round * 31 + peer as u64),
            );
            assert_eq!(
                sections, payloads_resident[peer],
                "round {round} peer {peer}: payload sections"
            );
            assert_eq!(
                bits_of(&ef_new),
                bits_of(profile.ef_state()),
                "round {round} peer {peer}: ef"
            );
            windowed_ef[peer] = ef_new;
            payloads_windowed.push(sections);
        }

        let masters_resident =
            resident_ingest(&g.cfg, g.numels, &payloads_resident, &base_resident);
        let seed = Seed([round as u8 + 1; 32]);
        let want_digest = digest_state(&seed, 64, u32::MAX, &family_bytes(&masters_resident));
        let (masters_windowed, got_digest) = drive_ingest(
            &g.cfg,
            g.numels,
            window_size,
            in_flight,
            &payloads_windowed,
            &seed,
            &base_windowed,
            Arrival::Lifo,
        );
        assert_eq!(
            bits_of(&masters_windowed),
            bits_of(&masters_resident),
            "round {round}: masters"
        );
        assert_eq!(got_digest, want_digest, "round {round}: digest");

        // The sealed master of round r IS the round base of round r+1 — one artifact, two
        // roles, on both sides.
        base_resident = masters_resident;
        base_windowed = masters_windowed;
    }
}

/// Zero committed payloads (the resident `count = max(1)` degenerate): the walk still folds,
/// emits base-preserving masters, and stays digest-identical.
#[test]
fn windowed_ingest_with_no_payloads_matches_resident() {
    let g = &geometries()[1];
    let bases = family(50, g.numels);
    let want = resident_ingest(&g.cfg, g.numels, &[], &bases);
    let seed = Seed([3; 32]);
    let want_digest = digest_state(&seed, 64, u32::MAX, &family_bytes(&want));
    let (got, got_digest) =
        drive_ingest(&g.cfg, g.numels, 4096, 2, &[], &seed, &bases, Arrival::Fifo);
    assert_eq!(bits_of(&got), bits_of(&want));
    assert_eq!(got_digest, want_digest);
}

// ================================================================================================
// typed refusals + the byte seam
// ================================================================================================

#[test]
fn geometry_violations_are_typed_construction_refusals() {
    let cfg = SparseLocoCfg {
        h: 1,
        ef_decay: 0.95,
        chunk: 16,
        topk: 4,
        bits: 2,
        outer_alpha: 1.0,
        clip: true,
    };
    // The profile chunk must divide every numel (the det kernels refuse only at first use;
    // the walk refuses at construction).
    assert!(SparseLocoUpdateWalk::new(&cfg, &[100], 64, 2).is_err());
    // The window must be a positive multiple of the chunk byte width.
    assert!(SparseLocoUpdateWalk::new(&cfg, &[64], 60, 2).is_err());
    assert!(SparseLocoUpdateWalk::new(&cfg, &[64], 0, 2).is_err());
    // An empty layout is degenerate.
    assert!(SparseLocoUpdateWalk::new(&cfg, &[], 64, 2).is_err());
    // The same rules govern the ingest walk.
    assert!(SparseLocoIngestWalk::new(
        &cfg,
        &[100],
        64,
        2,
        &[],
        DigestCarry::new(&Seed([1; 32]), 64)
    )
    .is_err());
}

#[test]
fn walk_protocol_violations_are_typed() {
    let cfg = SparseLocoCfg {
        h: 1,
        ef_decay: 0.95,
        chunk: 16,
        topk: 4,
        bits: 2,
        outer_alpha: 1.0,
        clip: false,
    };
    let numels = [64usize, 32];
    let mut walk = SparseLocoIngestWalk::new(
        &cfg,
        &numels,
        64,
        2,
        &[],
        DigestCarry::new(&Seed([1; 32]), 64),
    )
    .unwrap();
    // Completion before start is refused.
    assert!(walk.on_window_ready(0, &[0.0; 16]).is_err());
    let opening = walk.start().unwrap();
    assert_eq!(opening.issue.len(), 2);
    // A second start is refused.
    assert!(walk.start().is_err());
    // A never-issued ordinal is refused.
    assert!(walk.on_window_ready(3, &[0.0; 16]).is_err());
    // A mis-sized window is refused.
    assert!(walk.on_window_ready(0, &[0.0; 4]).is_err());
    // Sealing with outstanding windows is refused.
    let base = vec![0.0f32; 16];
    let _ = walk.on_window_ready(0, &base).unwrap();
    // A duplicate completion is refused.
    assert!(walk.on_window_ready(0, &base).is_err());
}

#[test]
fn seal_before_completion_is_refused() {
    let cfg = SparseLocoCfg {
        h: 1,
        ef_decay: 0.95,
        chunk: 16,
        topk: 4,
        bits: 2,
        outer_alpha: 1.0,
        clip: false,
    };
    let mut walk = SparseLocoUpdateWalk::new(&cfg, &[64], 64, 2).unwrap();
    let _ = walk.start().unwrap();
    assert!(walk.seal().is_err());
}

#[test]
fn byte_seam_round_trips_and_refuses_torn_windows() {
    let vals: Vec<f32> = vec![1.5, -2.25, 0.0, f32::MIN_POSITIVE];
    let mut buf = Vec::new();
    f32s_to_le_bytes(&vals, &mut buf);
    assert_eq!(buf.len(), 16);
    let back = le_bytes_to_f32s(&buf).unwrap();
    assert_eq!(
        back.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        vals.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    // A torn (non-multiple-of-4) window read is a typed refusal.
    assert!(le_bytes_to_f32s(&buf[..7]).is_err());
}
