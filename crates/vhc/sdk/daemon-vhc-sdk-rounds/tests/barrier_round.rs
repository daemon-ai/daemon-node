// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// The A2 choreography-move pins (refactor §5 A2 item 3): the round logic relocated from the v1
// engine keeps its exact semantics as library code —
//
// - the Staged bridging oracle: RECORD-LISTED order + per-item blake3 verification, all-or-
//   nothing mint (Phase D re-types it as Committed<T> byte-identically against these pins);
// - the train-loop order: per inner step every micro-window through train_step then one
//   inner_update; make_update once; the commitment hash = blake3(payload);
// - the barrier: strictly ascending round ingest; an unfetchable head stalls;
// - the straggle ladder: Fetching → Stalled heartbeats → budget exhaustion leaves; catch-up
//   ingests late as CaughtUp; the resync watermark never re-ingests.

use std::collections::BTreeMap;

use daemon_vhc_proto::merkle::commit_set;
use daemon_vhc_proto::{blake3_hash, Hash, PeerId, Seed};
use daemon_vhc_sdk_consensus::messages::{BatchWindow, RecordEntry, RoundOpen, RoundRecord};
use daemon_vhc_sdk_rounds::{
    Authorized, BarrierRound, Committed, MintError, Outbound, RoundCfg, RoundExperiment, StepCtx,
};

fn peer(b: u8) -> PeerId {
    PeerId([b; 32])
}

fn entry(p: PeerId, bytes: &[u8]) -> RecordEntry {
    RecordEntry {
        peer: p,
        hash: blake3_hash(bytes),
        size: bytes.len() as u64,
    }
}

/// A recording experiment: logs every choreography call in order.
#[derive(Default)]
struct Recorder {
    calls: Vec<String>,
    ingested: Vec<(u64, Vec<PeerId>)>,
}

impl RoundExperiment for Recorder {
    fn train_step(&mut self, ctx: &StepCtx) {
        self.calls.push(format!(
            "step r{} h{} mb{}/{} [{},{})",
            ctx.round, ctx.inner_step, ctx.mb_index, ctx.mb_count, ctx.micro.start, ctx.micro.end
        ));
    }
    fn inner_update(&mut self, inner_step: u32) {
        self.calls.push(format!("inner h{inner_step}"));
    }
    fn make_update(&mut self, round: u64) -> Vec<u8> {
        self.calls.push(format!("make r{round}"));
        format!("update-r{round}").into_bytes()
    }
    fn ingest(&mut self, round: u64, committed: &Committed) -> [u8; 16] {
        self.calls.push(format!("ingest r{round}"));
        self.ingested
            .push((round, committed.items().iter().map(|i| i.peer).collect()));
        let mut d = [0u8; 16];
        d[..8].copy_from_slice(&round.to_le_bytes());
        d
    }
}

fn cfg(me: PeerId, roster: Vec<PeerId>) -> RoundCfg {
    RoundCfg {
        peer: me,
        roster,
        steps_per_round: 2,
        micro_batch: 2,
        stall_rounds_max: 2,
    }
}

fn round_open(round: u64) -> RoundOpen {
    RoundOpen {
        round,
        seed: Seed([round as u8; 32]),
        roster_digest: Hash([0; 32]),
        batch: BatchWindow {
            start: 0,
            end: 16, // 2 peers × (2 steps × 2 sequences × 2 micro) — divisible everywhere
        },
        deadline_unix_s: 0,
    }
}

fn round_record(round: u64, entries: &[RecordEntry]) -> RoundRecord {
    let set: Vec<(PeerId, Hash)> = entries.iter().map(|e| (e.peer, e.hash)).collect();
    RoundRecord {
        round,
        set: commit_set(&set).commitment(),
        drops: Vec::new(),
        next_seed: Seed([0; 32]),
        set_locator: daemon_vhc_sdk_consensus::messages::Locator::StoreKey(String::new()),
        inline: None,
    }
}

// -- the Committed mint (the re-typed Staged bridging oracle) ----------------------------------------

#[test]
fn committed_mint_pins_record_listed_order_verification_and_byte_identity() {
    let (a, b, c) = (peer(3), peer(1), peer(2));
    // Deliberately NOT pubkey-sorted: the mint must follow the RECORD-LISTED order, nothing else.
    let entries = vec![entry(a, b"pay-a"), entry(b, b"pay-b"), entry(c, b"pay-c")];
    let mut source: BTreeMap<(u64, PeerId), Vec<u8>> = BTreeMap::new();
    source.insert((7, a), b"pay-a".to_vec());
    source.insert((7, b), b"pay-b".to_vec());
    source.insert((7, c), b"pay-c".to_vec());
    let auth = Authorized::from_authoritative_channel(0);

    let committed = Committed::mint(&auth, 7, &entries, &mut source).expect("mint");
    let order: Vec<PeerId> = committed.items().iter().map(|i| i.peer).collect();
    assert_eq!(
        order,
        vec![a, b, c],
        "record-listed order, not map/pubkey order"
    );

    // Byte-identity across mints (the Phase-D re-typing contract).
    assert_eq!(
        committed,
        Committed::mint(&auth, 7, &entries, &mut source).expect("mint again")
    );

    // A missing payload refuses (the stall ladder's input) — all-or-nothing.
    source.remove(&(7, b));
    assert_eq!(
        Committed::mint(&auth, 7, &entries, &mut source).unwrap_err(),
        MintError::Missing { peer: b }
    );

    // Tampered bytes refuse with the offending peer named.
    source.insert((7, b), b"pay-B".to_vec());
    assert_eq!(
        Committed::mint(&auth, 7, &entries, &mut source).unwrap_err(),
        MintError::HashMismatch { peer: b }
    );
}

// -- the host-staged repr (the bridge's payload form) -------------------------------------------------

#[test]
fn host_staged_mint_keeps_order_and_all_or_nothing_with_delegated_verification() {
    use daemon_vhc_sdk_rounds::HostStaged;
    let (a, b) = (peer(2), peer(1));
    // Hashes are the RECORD's (host-verified at staging, ABI §4.3); the repr is a staging token.
    let entries = vec![entry(a, b"pay-a"), entry(b, b"pay-b")];
    let mut source: BTreeMap<(u64, PeerId), HostStaged> = BTreeMap::new();
    source.insert((3, a), HostStaged(11));
    source.insert((3, b), HostStaged(12));
    let auth = Authorized::from_authoritative_channel(0);

    let committed = Committed::mint(&auth, 3, &entries, &mut source).expect("mint");
    let order: Vec<(PeerId, u64)> = committed
        .items()
        .iter()
        .map(|i| (i.peer, i.bytes.0))
        .collect();
    assert_eq!(
        order,
        vec![(a, 11), (b, 12)],
        "record-listed order over tokens"
    );

    // All-or-nothing stands: a missing token refuses exactly like missing bytes.
    source.remove(&(3, b));
    assert_eq!(
        Committed::mint(&auth, 3, &entries, &mut source).unwrap_err(),
        MintError::Missing { peer: b }
    );
}

// -- the train-loop order ---------------------------------------------------------------------------

#[test]
fn train_loop_order_is_micro_steps_then_inner_update_then_one_make_update() {
    let me = peer(9);
    let mut driver = BarrierRound::new(Recorder::default(), cfg(me, vec![me]));
    let mut source: BTreeMap<(u64, PeerId), Vec<u8>> = BTreeMap::new();

    let out = driver.on_round_open(&round_open(0), &mut source);

    // Solo roster: the whole window [0,16) → 2 steps of 8 sequences → 4 micro-windows of 2 each,
    // exactly the engine's slice_interval arithmetic.
    let calls = &driver.experiment().calls;
    assert_eq!(
        calls,
        &vec![
            "step r0 h0 mb0/4 [0,2)".to_string(),
            "step r0 h0 mb1/4 [2,4)".to_string(),
            "step r0 h0 mb2/4 [4,6)".to_string(),
            "step r0 h0 mb3/4 [6,8)".to_string(),
            "inner h0".to_string(),
            "step r0 h1 mb0/4 [8,10)".to_string(),
            "step r0 h1 mb1/4 [10,12)".to_string(),
            "step r0 h1 mb2/4 [12,14)".to_string(),
            "step r0 h1 mb3/4 [14,16)".to_string(),
            "inner h1".to_string(),
            "make r0".to_string(),
        ],
        "the v1 train-loop order, relocated verbatim"
    );

    // One Commit action; its hash is blake3 of the sealed payload.
    let [Outbound::Commit {
        commitment,
        payload,
    }] = &out[..]
    else {
        panic!("exactly one Commit, got {out:?}");
    };
    assert_eq!(commitment.round, 0);
    assert_eq!(commitment.payload, blake3_hash(payload));
    assert_eq!(commitment.size, payload.len() as u64);
}

// -- the barrier + straggle ladder --------------------------------------------------------------------

#[test]
fn barrier_stalls_heartbeats_catches_up_and_respects_ascending_order() {
    let me = peer(9);
    let other = peer(5);
    let mut driver = BarrierRound::new(Recorder::default(), cfg(me, vec![me, other]));
    let mut source: BTreeMap<(u64, PeerId), Vec<u8>> = BTreeMap::new();

    // Round 1's record arrives with `other`'s payload unfetchable → Straggle{Fetching}.
    let e1 = vec![entry(me, b"m1"), entry(other, b"o1")];
    source.insert((1, me), b"m1".to_vec());
    let out = driver.on_round_record(&round_record(1, &e1), e1.clone(), &mut source);
    assert_eq!(
        out,
        vec![Outbound::Straggle {
            round: 1,
            fetching: true
        }]
    );

    // Round 2's record is fully fetchable — but MUST NOT ingest ahead of stalled round 1.
    let e2 = vec![entry(me, b"m2"), entry(other, b"o2")];
    source.insert((2, me), b"m2".to_vec());
    source.insert((2, other), b"o2".to_vec());
    let out = driver.on_round_record(&round_record(2, &e2), e2, &mut source);
    assert_eq!(
        out,
        vec![Outbound::Straggle {
            round: 2,
            fetching: true
        }],
        "strictly ascending ingest: round 2 waits behind stalled round 1"
    );
    assert!(driver.experiment().ingested.is_empty());

    // A RoundOpen while stalled: skip training, heartbeat Stalled (budget 1 of 2).
    let out = driver.on_round_open(&round_open(3), &mut source);
    assert_eq!(
        out,
        vec![Outbound::Straggle {
            round: 3,
            fetching: false
        }]
    );
    assert!(driver
        .experiment()
        .calls
        .iter()
        .all(|c| !c.starts_with("step r3")));

    // The missing payload becomes fetchable → the next event catches up BOTH rounds, in order.
    source.insert((1, other), b"o1".to_vec());
    let out = driver.on_round_open(&round_open(4), &mut source);
    assert_eq!(
        out[..2],
        [
            Outbound::CaughtUp {
                round: 1,
                digest: {
                    let mut d = [0u8; 16];
                    d[..8].copy_from_slice(&1u64.to_le_bytes());
                    d
                }
            },
            Outbound::CaughtUp {
                round: 2,
                digest: {
                    let mut d = [0u8; 16];
                    d[..8].copy_from_slice(&2u64.to_le_bytes());
                    d
                }
            },
        ]
    );
    // Ingest order was 1 then 2, each in record-listed order.
    assert_eq!(
        driver.experiment().ingested,
        vec![(1, vec![me, other]), (2, vec![me, other])]
    );
    // Ladder reset: round 4 itself then trains + commits (the tail of the same call).
    assert!(
        matches!(out.last(), Some(Outbound::Commit { commitment, .. }) if commitment.round == 4)
    );
}

#[test]
fn stall_budget_exhaustion_leaves_for_the_epoch() {
    let me = peer(9);
    let other = peer(5);
    let mut driver = BarrierRound::new(Recorder::default(), cfg(me, vec![me, other]));
    let mut source: BTreeMap<(u64, PeerId), Vec<u8>> = BTreeMap::new();

    let e1 = vec![entry(me, b"m1"), entry(other, b"o1")];
    source.insert((1, me), b"m1".to_vec());
    driver.on_round_record(&round_record(1, &e1), e1, &mut source);

    // stall_rounds_max = 2: two stalled opens heartbeat; the third leaves.
    for r in [2u64, 3] {
        let out = driver.on_round_open(&round_open(r), &mut source);
        assert_eq!(
            out,
            vec![Outbound::Straggle {
                round: r,
                fetching: false
            }]
        );
    }
    let out = driver.on_round_open(&round_open(4), &mut source);
    assert_eq!(out.last(), Some(&Outbound::Left { round: 1 }));
}

#[test]
fn resync_watermark_never_reingests() {
    let me = peer(9);
    let mut driver = BarrierRound::new(Recorder::default(), cfg(me, vec![me]));
    let mut source: BTreeMap<(u64, PeerId), Vec<u8>> = BTreeMap::new();

    let e1 = vec![entry(me, b"m1")];
    source.insert((1, me), b"m1".to_vec());
    let out = driver.on_round_record(&round_record(1, &e1), e1.clone(), &mut source);
    assert!(matches!(out[0], Outbound::RoundComplete { round: 1, .. }));

    // A buffered/late replay of the same record at/below the watermark is a no-op (a double
    // outer-step would diverge the digest — the engine's resync-composability guard).
    let out = driver.on_round_record(&round_record(1, &e1), e1, &mut source);
    assert!(out.is_empty());
    assert_eq!(driver.experiment().ingested.len(), 1);
}
