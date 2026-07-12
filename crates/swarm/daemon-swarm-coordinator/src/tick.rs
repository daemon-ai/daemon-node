// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The pure coordinator `tick` (spec §6.2, §6.4; TDD PROTO-1/2/3/5/7/9/10/14).
//!
//! `tick(state, input) -> (state', outputs)` is a total, I/O-free function: it never reads a clock,
//! never signs, never touches the network. Time enters as [`Input::Clock`]; signed evidence as
//! [`Input::Message`]; operator intents as [`Input::Control`]. Identical `(state, input)` always
//! yields identical `(state', outputs)` — the replay-oracle foundation (I1, PROTO-20). The commit
//! rule ([`crate::commit`]) consumes only signed evidence (I6).

use daemon_swarm_proto::envelope::StopCondition;
use daemon_swarm_proto::messages::{
    Attestation, BatchWindow, Commitment, Digest, Heartbeat, Join, Locator, RoundOpen, RoundRecord,
    SignedMessage, StorageReceipt, Straggle, SwarmMessage,
};
use daemon_swarm_proto::sign::Signed;
use daemon_swarm_proto::{
    blake3_hash, commit_set, global_batch_at, select_committee, Hash, PeerId, Seed,
    SwarmProtoVersion,
};

use crate::admission::{admit, JoinCandidate};
use crate::commit::{all_committed, all_evidenced, committed_entries};
use crate::io::{ControlAction, ControlRequest, Input, Notice, Output, Rejection};
use crate::state::{ClientState, CoordinatorState, Member, Phase, RoundState};

/// Advance the coordinator by one input. Pure: no I/O, no clock read, no signing.
#[must_use]
pub fn tick(mut state: CoordinatorState, input: Input) -> (CoordinatorState, Vec<Output>) {
    let mut out = Vec::new();
    match input {
        Input::Clock(now) => on_clock(&mut state, &mut out, now),
        Input::Message(sm) => on_message(&mut state, &mut out, sm),
        Input::Control(req) => on_control(&mut state, &mut out, req),
    }
    (state, out)
}

// ----- clock -----

fn on_clock(state: &mut CoordinatorState, out: &mut Vec<Output>, now: u64) {
    if now > state.now_s {
        state.now_s = now;
    }
    if state.phase.is_halted() {
        out.push(Output::Reject(Rejection::Halted(state.phase)));
        return;
    }
    drive_time(state, out);
}

/// Time-driven phase transitions (§6.2 timeouts, Appendix A.1 `check_timeout`).
fn drive_time(state: &mut CoordinatorState, out: &mut Vec<Output>) {
    let now = state.now_s;
    let cfg = &state.config;
    match state.phase {
        Phase::WaitingForMembers => {
            if state.healthy_count() >= cfg.min_peers {
                change_phase(state, out, Phase::Warmup);
            }
        }
        Phase::Warmup => {
            if state.healthy_count() < cfg.min_peers {
                change_phase(state, out, Phase::WaitingForMembers);
            } else if now >= state.phase_start_s + cfg.warmup_s {
                state.epoch_start_round = state.round;
                open_round(state, out);
            }
        }
        Phase::RoundTrain => {
            let committable =
                current_slot(state).is_some_and(|rs| all_committed(rs, &state.roster));
            if committable || now >= state.phase_start_s + cfg.round_train_max_s {
                change_phase(state, out, Phase::RoundWitness);
                maybe_finalize(state, out);
            }
        }
        Phase::RoundWitness => {
            let evidenced = current_slot(state).is_some_and(|rs| all_evidenced(rs, &state.roster));
            if evidenced || now >= state.phase_start_s + cfg.round_witness_s {
                finalize_round(state, out);
            }
        }
        Phase::Cooldown => {
            if now >= state.phase_start_s + cfg.cooldown_s {
                exit_cooldown(state, out);
            }
        }
        Phase::Uninitialized | Phase::Finished | Phase::Paused => {}
    }
}

// ----- messages -----

fn on_message(state: &mut CoordinatorState, out: &mut Vec<Output>, sm: SignedMessage) {
    if sm.version != state.config.proto_version {
        out.push(Output::Reject(Rejection::VersionMismatch {
            expected: state.config.proto_version,
            got: sm.version,
        }));
        return;
    }
    if sm.verify().is_err() {
        out.push(Output::Reject(Rejection::BadSignature));
        return;
    }
    if state.phase.is_halted() {
        out.push(Output::Reject(Rejection::Halted(state.phase)));
        return;
    }
    let signer = sm.signer;
    let version = sm.version;
    match sm.payload {
        SwarmMessage::Join(j) => on_join(state, out, signer, version, j),
        SwarmMessage::Commitment(c) => on_commitment(state, out, signer, c),
        SwarmMessage::Attestation(a) => on_attestation(state, out, signer, a),
        SwarmMessage::StorageReceipt(sr) => on_receipt(state, out, sr),
        SwarmMessage::Digest(d) => on_digest(state, out, signer, d),
        SwarmMessage::Straggle(s) => on_straggle(state, signer, s),
        SwarmMessage::Heartbeat(h) => on_heartbeat(state, signer, h),
        SwarmMessage::RoundOpen(_) | SwarmMessage::RoundRecord(_) => {
            out.push(Output::Reject(Rejection::UnexpectedMessage));
        }
    }
}

fn on_join(
    state: &mut CoordinatorState,
    out: &mut Vec<Output>,
    signer: PeerId,
    version: SwarmProtoVersion,
    j: Join,
) {
    let cand = JoinCandidate {
        peer: signer,
        version,
        join: &j,
        asserted_hash: None,
    };
    match admit(
        &state.config,
        state.phase,
        &state.roster,
        &state.pending,
        &cand,
    ) {
        Err(reason) => out.push(Output::Reject(Rejection::Admission(reason))),
        Ok(()) => {
            let m = Member::joining(signer, j.iroh_id, j.class, state.epoch);
            if state.phase == Phase::WaitingForMembers {
                upsert_member(state, m);
            } else {
                state.pending.push(m);
            }
            out.push(Output::Note(Notice::Admitted(signer)));
        }
    }
}

fn on_commitment(
    state: &mut CoordinatorState,
    out: &mut Vec<Output>,
    signer: PeerId,
    c: Commitment,
) {
    if !state.phase.is_round_active() {
        out.push(Output::Reject(Rejection::UnexpectedMessage));
        return;
    }
    if c.round != state.round {
        out.push(Output::Reject(Rejection::StaleRound {
            current: state.round,
            got: c.round,
        }));
        return;
    }
    if !state.is_healthy_member(&signer) {
        out.push(Output::Reject(Rejection::UnknownPeer));
        return;
    }
    let round = state.round;
    if let Some(rs) = state.rounds.get_mut(round) {
        rs.commitments.insert(signer, c);
    }
    maybe_advance(state, out);
}

fn on_attestation(
    state: &mut CoordinatorState,
    out: &mut Vec<Output>,
    signer: PeerId,
    a: Attestation,
) {
    if !state.phase.is_round_active() {
        out.push(Output::Reject(Rejection::UnexpectedMessage));
        return;
    }
    if a.round != state.round {
        out.push(Output::Reject(Rejection::StaleRound {
            current: state.round,
            got: a.round,
        }));
        return;
    }
    let round = state.round;
    let is_witness = state
        .rounds
        .get(round)
        .is_some_and(|rs| rs.witnesses.contains(&signer));
    if !is_witness {
        out.push(Output::Reject(Rejection::NotWitness));
        return;
    }
    if let Some(rs) = state.rounds.get_mut(round) {
        rs.attestations.insert(signer, a);
    }
    maybe_advance(state, out);
}

fn on_receipt(state: &mut CoordinatorState, out: &mut Vec<Output>, sr: StorageReceipt) {
    if !state.phase.is_round_active() {
        out.push(Output::Reject(Rejection::UnexpectedMessage));
        return;
    }
    if sr.round != state.round {
        out.push(Output::Reject(Rejection::StaleRound {
            current: state.round,
            got: sr.round,
        }));
        return;
    }
    let round = state.round;
    if let Some(rs) = state.rounds.get_mut(round) {
        for e in sr.verified {
            if !rs
                .receipts
                .iter()
                .any(|x| x.peer == e.peer && x.hash == e.hash)
            {
                rs.receipts.push(e);
            }
        }
    }
    maybe_advance(state, out);
}

fn on_digest(state: &mut CoordinatorState, out: &mut Vec<Output>, signer: PeerId, d: Digest) {
    if !state.phase.is_round_active() {
        out.push(Output::Reject(Rejection::UnexpectedMessage));
        return;
    }
    let mut mismatch: Option<(PeerId, PeerId)> = None;
    let held = match state.rounds.get_mut(d.round) {
        None => false,
        Some(rs) => {
            for (p, existing) in &rs.digests {
                if *existing != d.digest {
                    mismatch = Some((*p, signer));
                    break;
                }
            }
            rs.digests.insert(signer, d.digest);
            if mismatch.is_some() {
                rs.desync = true;
            }
            true
        }
    };
    if !held {
        out.push(Output::Reject(Rejection::StaleRound {
            current: state.round,
            got: d.round,
        }));
        return;
    }
    if let Some(peers) = mismatch {
        out.push(Output::Note(Notice::DigestMismatch {
            round: d.round,
            peers,
        }));
    }
}

fn on_straggle(state: &mut CoordinatorState, signer: PeerId, s: Straggle) {
    if let Some(m) = state.member_mut(&signer) {
        m.last_straggle_round = Some(s.round);
    }
}

fn on_heartbeat(state: &mut CoordinatorState, signer: PeerId, h: Heartbeat) {
    if h.round > state.max_reported_round {
        state.max_reported_round = h.round;
    }
    if let Some(m) = state.member_mut(&signer) {
        if h.round > m.last_seen_round {
            m.last_seen_round = h.round;
        }
    }
}

// ----- control -----

fn on_control(state: &mut CoordinatorState, out: &mut Vec<Output>, req: Signed<ControlRequest>) {
    if req.verify().is_err() {
        out.push(Output::Reject(Rejection::BadSignature));
        return;
    }
    if req.body.run_id != state.config.run_id {
        out.push(Output::Reject(Rejection::RunIdMismatch));
        return;
    }
    if !state.config.authorized.contains(&req.signer) {
        out.push(Output::Reject(Rejection::Unauthorized));
        return;
    }
    match req.body.action {
        ControlAction::Pause => {
            if state.phase.is_halted() {
                out.push(Output::Reject(Rejection::Halted(state.phase)));
            } else {
                state.paused_from = Some(state.phase);
                change_phase(state, out, Phase::Paused);
            }
        }
        ControlAction::Resume => {
            if state.phase == Phase::Paused {
                state.paused_from = None;
                change_phase(state, out, Phase::WaitingForMembers);
            } else {
                out.push(Output::Reject(Rejection::Halted(state.phase)));
            }
        }
    }
}

// ----- round lifecycle -----

/// Early-advance after new evidence: `RoundTrain → RoundWitness → commit` when the conditions are
/// already met (the "all submitted" fast path, Appendix A.3).
fn maybe_advance(state: &mut CoordinatorState, out: &mut Vec<Output>) {
    if state.phase == Phase::RoundTrain {
        let committable = current_slot(state).is_some_and(|rs| all_committed(rs, &state.roster));
        if committable {
            change_phase(state, out, Phase::RoundWitness);
        }
    }
    maybe_finalize(state, out);
}

fn maybe_finalize(state: &mut CoordinatorState, out: &mut Vec<Output>) {
    if state.phase == Phase::RoundWitness {
        let evidenced = current_slot(state).is_some_and(|rs| all_evidenced(rs, &state.roster));
        if evidenced {
            finalize_round(state, out);
        }
    }
}

/// Open `state.round` for training: install the ring slot, publish `RoundOpen`, enter `RoundTrain`.
fn open_round(state: &mut CoordinatorState, out: &mut Vec<Output>) {
    let peers = state.healthy_peer_ids();
    let committee = select_committee(&peers, &state.seed, state.config.witness_target);
    let gb = global_batch_at(state.config.global_batch, state.round);
    let batch = BatchWindow {
        start: state.data_index,
        end: state.data_index + gb,
    };
    let rs = RoundState::opened(
        state.round,
        state.seed,
        state.data_index,
        batch,
        committee.witnesses,
    );
    state.rounds.install(rs);

    let from = state.phase;
    state.phase = Phase::RoundTrain;
    state.phase_start_s = state.now_s;
    out.push(Output::Note(Notice::PhaseChanged {
        from,
        to: Phase::RoundTrain,
    }));

    let ro = RoundOpen {
        round: state.round,
        seed: state.seed,
        roster_digest: roster_digest(&peers),
        batch,
        deadline_unix_s: state.now_s + state.config.round_train_max_s,
    };
    out.push(Output::publish(SwarmMessage::RoundOpen(ro)));
}

/// Freeze the round record from signed evidence, account absences/drops, and decide the next phase.
fn finalize_round(state: &mut CoordinatorState, out: &mut Vec<Output>) {
    let round = state.round;
    let Some(slot) = state.rounds.get(round).cloned() else {
        return;
    };
    let committed = committed_entries(&slot, &state.roster);
    let present: std::collections::BTreeSet<PeerId> = committed.iter().map(|e| e.peer).collect();

    let k = state.config.k_absences;
    let stall = u64::from(state.config.stall_rounds_max);
    let mut drops: Vec<PeerId> = Vec::new();
    for m in state.roster.iter_mut().filter(|m| m.is_healthy()) {
        if present.contains(&m.peer) {
            m.absences = 0;
            continue;
        }
        let straggling = m
            .last_straggle_round
            .is_some_and(|r| round >= r && round - r <= stall);
        if straggling {
            continue;
        }
        m.absences += 1;
        if k > 0 && m.absences >= k {
            m.state = ClientState::Dropped;
            drops.push(m.peer);
        }
    }
    for p in &drops {
        out.push(Output::Note(Notice::Dropped(*p)));
    }

    let pairs: Vec<(PeerId, Hash)> = committed.iter().map(|e| (e.peer, e.hash)).collect();
    let set = commit_set(&pairs).commitment();
    let ns = next_seed(&slot.seed, round);
    let record = RoundRecord {
        round,
        set,
        drops,
        next_seed: ns,
        set_locator: Locator::StoreKey(format!(
            "runs/{}/rounds/{round}/record-set.cbor",
            state.config.run_id
        )),
        inline: Some(committed),
    };
    if let Some(s) = state.rounds.get_mut(round) {
        s.record = Some(record.clone());
    }
    out.push(Output::publish(SwarmMessage::RoundRecord(record)));

    let gb = global_batch_at(state.config.global_batch, round);
    state.rounds_done += 1;
    state.tokens_done = state
        .tokens_done
        .saturating_add(gb.saturating_mul(state.config.seq_len));

    // Advance the cursor/seed for the next round (harmless if the run is finishing).
    let rounds_this_epoch = (round + 1).saturating_sub(state.epoch_start_round);
    state.data_index = state.data_index.saturating_add(gb);
    state.round = round + 1;
    state.seed = ns;

    if stop_reached(state) {
        change_phase(state, out, Phase::Cooldown);
        return;
    }
    let epoch_boundary =
        state.config.epoch_rounds > 0 && rounds_this_epoch >= state.config.epoch_rounds;
    let floor_breach = state.healthy_count() < state.config.min_peers;
    if epoch_boundary || floor_breach {
        change_phase(state, out, Phase::Cooldown);
        return;
    }
    open_round(state, out);
}

fn exit_cooldown(state: &mut CoordinatorState, out: &mut Vec<Output>) {
    if stop_reached(state) {
        change_phase(state, out, Phase::Finished);
        out.push(Output::Note(Notice::Finished));
        return;
    }
    state.epoch += 1;
    let pending = std::mem::take(&mut state.pending);
    for mut m in pending {
        m.joined_epoch = state.epoch;
        upsert_member(state, m);
    }
    change_phase(state, out, Phase::WaitingForMembers);
}

// ----- helpers -----

fn change_phase(state: &mut CoordinatorState, out: &mut Vec<Output>, to: Phase) {
    let from = state.phase;
    state.phase = to;
    state.phase_start_s = state.now_s;
    out.push(Output::Note(Notice::PhaseChanged { from, to }));
}

fn upsert_member(state: &mut CoordinatorState, m: Member) {
    if let Some(existing) = state.member_mut(&m.peer) {
        *existing = m;
    } else {
        state.roster.push(m);
    }
}

fn current_slot(state: &CoordinatorState) -> Option<&RoundState> {
    state.rounds.get(state.round)
}

fn stop_reached(state: &CoordinatorState) -> bool {
    match state.config.stop {
        StopCondition::Tokens(t) => state.tokens_done >= t,
        StopCondition::Rounds(r) => state.rounds_done >= r,
    }
}

fn next_seed(seed: &Seed, round: u64) -> Seed {
    let mut buf = [0u8; Seed::LEN + 8];
    buf[..Seed::LEN].copy_from_slice(seed.as_bytes());
    buf[Seed::LEN..].copy_from_slice(&round.to_le_bytes());
    Seed(*blake3_hash(&buf).as_bytes())
}

fn roster_digest(peers: &[PeerId]) -> Hash {
    let mut buf = Vec::with_capacity(peers.len() * PeerId::LEN);
    for p in peers {
        buf.extend_from_slice(p.as_bytes());
    }
    blake3_hash(&buf)
}
