// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The pure coordinator `tick` (spec §6.2, §6.4; TDD PROTO-1/2/3/5/7/9/10/14).
//!
//! `tick(state, input) -> (state', outputs)` is a total, I/O-free function: it never reads a clock,
//! never signs, never touches the network. Time enters as [`Input::Clock`]; signed evidence as
//! [`Input::Message`]; operator intents as [`Input::Control`]. Identical `(state, input)` always
//! yields identical `(state', outputs)` — the replay-oracle foundation (I1, PROTO-20). The commit
//! rule ([`crate::commit`]) consumes only signed evidence (I6).

use crate::messages::{
    Attestation, BatchWindow, Commitment, Digest, Finished, Heartbeat, Join, Locator, RoundOpen,
    RoundRecord, SignedMessage, StorageReceipt, Straggle, VhcMessage,
};
use crate::{global_batch_at, select_committee};
use daemon_vhc_proto::envelope::StopCondition;
use daemon_vhc_proto::sign::Signed;
use daemon_vhc_proto::{blake3_hash, commit_set, Hash, PeerId, Seed, VhcProtoVersion};

use crate::authority::Authorized;
use crate::coordinator::admission::{admit, JoinCandidate};
use crate::coordinator::commit::{all_committed, all_evidenced, committed_entries};
use crate::coordinator::io::{
    AdmissionReject, ControlAction, ControlRequest, Input, Notice, Output, Rejection,
};
use crate::coordinator::state::{ClientState, CoordinatorState, Member, Phase, RoundState};

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

/// Advance the coordinator by a message whose delivery authority is attested by D1's
/// [`Authorized`] token — the reconciled D2 seam (formerly the pre-D1 `SingleKey` stub).
///
/// Rationale (architecture §4.2/§4.3): the wasm `coordinator-quorum` guest receives worker
/// messages as **host-verified** `Frame` events — the host authenticated the §12 signed-frame
/// envelope above the sandbox on a **declared authoritative channel** and delivers only the
/// opaque payload + the (authenticated) sender, never a re-checkable ed25519 signature. That is
/// exactly the provenance [`Authorized::from_authoritative_channel`] encodes (D1's host-delivery
/// bridge path); the in-guest signature path obtains the same token from `Authority::authorize`.
/// Either way, the caller cannot reach the dispatch below without having gone through D1's
/// authority vocabulary — the trust decision lives in the token's mint, not here.
///
/// The decision logic is **identical** to [`tick`]'s message path after signature verification
/// (both funnel through the private `dispatch_payload`), so a native reference fed validly-signed
/// frames and this guest fed the same host-authenticated payloads produce byte-identical outputs —
/// the dual-compilation identity property (refactor §8/D2 acceptance). The token itself carries no
/// decision-relevant data (the channel is delivery provenance), so it cannot perturb identity.
#[must_use]
pub fn tick_authenticated(
    mut state: CoordinatorState,
    signer: PeerId,
    version: VhcProtoVersion,
    payload: VhcMessage,
    authorized: Authorized,
) -> (CoordinatorState, Vec<Output>) {
    // The token is the proof-of-provenance; its channel is not a dispatch input (delivery routing
    // is host mechanism, ABI §6.2). Consume it explicitly so the requirement is visible.
    let _ = authorized.channel();
    let mut out = Vec::new();
    if version != state.config.proto_version {
        out.push(Output::Reject(Rejection::VersionMismatch {
            expected: state.config.proto_version,
            got: version,
        }));
    } else if state.phase.is_halted() {
        out.push(Output::Reject(Rejection::Halted(state.phase)));
    } else {
        dispatch_payload(&mut state, &mut out, signer, version, payload);
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
            // Membership decay must not freeze while waiting (reliability spec §7, the C2
            // floor-breach wedge): absence accounting otherwise lives only inside
            // `finalize_round`, so once the floor breaches there are no rounds, hence no decay,
            // hence a dead member holds its seat forever and a rejoiner meets `RosterFull`.
            decay_stale_members(state, out);
            if state.healthy_count() >= state.config.min_peers {
                enter_warmup(state, out);
            }
        }
        Phase::Warmup => {
            if state.healthy_count() < cfg.min_peers {
                change_phase(state, out, Phase::WaitingForMembers);
            } else if all_warmup_ready(state) || now >= state.phase_start_s + cfg.warmup_s {
                // Exit warmup early once every admitted member signals readiness (additive),
                // else on the warmup timeout (the always-available path, back-compat).
                open_epoch_first_round(state, out);
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
    dispatch_payload(state, out, sm.signer, sm.version, sm.payload);
}

/// Dispatch an authenticated, phase-legal message to its per-type handler (the shared tail of the
/// [`tick`] `Input::Message` path and the [`tick_authenticated`] seam). Both callers have already
/// established version match, signature/authenticity, and non-halted phase — this is the pure
/// decision logic, so the two entry paths make byte-identical decisions on identical payloads.
fn dispatch_payload(
    state: &mut CoordinatorState,
    out: &mut Vec<Output>,
    signer: PeerId,
    version: VhcProtoVersion,
    payload: VhcMessage,
) {
    // Liveness stamp: any authenticated frame proves the signer was alive at this logical time —
    // the staleness input decay-while-waiting reads (a member never heard through a whole
    // round-scaled absence window while the run WAITS is a zombie holding a seat). Deterministic:
    // `now_s` is state, and the stamp is a pure function of (state, input).
    let now = state.now_s;
    if let Some(m) = state.member_mut(&signer) {
        m.last_seen_s = now;
    }
    match payload {
        VhcMessage::Join(j) => on_join(state, out, signer, version, j),
        VhcMessage::Commitment(c) => on_commitment(state, out, signer, c),
        VhcMessage::Attestation(a) => on_attestation(state, out, signer, a),
        VhcMessage::StorageReceipt(sr) => on_receipt(state, out, sr),
        VhcMessage::Digest(d) => on_digest(state, out, signer, d),
        VhcMessage::Straggle(s) => on_straggle(state, signer, s),
        VhcMessage::Heartbeat(h) => on_heartbeat(state, out, signer, h),
        VhcMessage::CheckpointAttestation(ca) => on_checkpoint_attestation(state, out, &ca),
        VhcMessage::RoundOpen(_) | VhcMessage::RoundRecord(_) | VhcMessage::Finished(_) => {
            out.push(Output::Reject(Rejection::UnexpectedMessage));
        }
    }
}

/// Record a checkpoint attestation into the consensus ledger (Phase E cold join, architecture
/// §5.3). The attestation is **self-authenticating**: its inner domain-separated signature by the
/// bound `signer` is verified here (independent of the outer frame's signer, which may be a
/// relayer re-broadcasting another peer's attestation verbatim — that is why the outer `signer`
/// is deliberately not an input). Verified + newly-recorded attestations emit
/// [`Notice::CheckpointAttested`] with the checkpoint's fresh digest-tier count (the K-gate
/// input); a duplicate `(checkpoint, tier, signer)` re-send is idempotently silent; an invalid
/// tier or signature is a typed rejection.
fn on_checkpoint_attestation(
    state: &mut CoordinatorState,
    out: &mut Vec<Output>,
    ca: &crate::messages::CheckpointAttestation,
) {
    let Ok(att) = crate::attestation::SignedAttestation::from_wire(ca) else {
        out.push(Output::Reject(Rejection::UnexpectedMessage)); // unknown tier tag
        return;
    };
    let (checkpoint, tier, signer) = (att.body.checkpoint, att.body.tier, att.body.signer);
    match state.attestations.record(att) {
        Ok(true) => {
            let digest_count = state
                .attestations
                .count(&checkpoint, crate::attestation::AttestationTier::Digest);
            out.push(Output::Note(Notice::CheckpointAttested {
                checkpoint,
                tier: tier.tag(),
                signer,
                digest_count,
            }));
        }
        Ok(false) => {} // idempotent duplicate
        Err(_) => out.push(Output::Reject(Rejection::BadSignature)),
    }
}

fn on_join(
    state: &mut CoordinatorState,
    out: &mut Vec<Output>,
    signer: PeerId,
    version: VhcProtoVersion,
    j: Join,
) {
    let cand = JoinCandidate {
        peer: signer,
        version,
        join: &j,
        // The frozen `Join` now carries an additive `envelope_hash`; forward it so the
        // `EnvelopeHashMismatch` admission check is reachable from the wire (a peer that assessed a
        // different envelope is rejected). Legacy joins omit it (`None`) → check skipped (back-compat).
        asserted_hash: j.envelope_hash.as_ref(),
    };
    match admit(
        &state.config,
        state.phase,
        &state.roster,
        &state.pending,
        &cand,
    ) {
        // An active member re-joining is a SESSION REPLACEMENT, not a protocol error: the peer's
        // run identity persists across worker incarnations (one lifecycle, two identities), so a
        // trainer whose session churned (coordinator outage, REPLACE re-admission, late-join
        // restore) re-announces the same Join it always sends until it observes a round. Refusing
        // it as a duplicate starved exactly that rejoiner of the replay-forward below — it folded
        // nothing, waited for a round open that is never re-flooded, and died on the witness
        // timeout, forever (the c15b post-reconstruction livelock). Roster and pending stay
        // untouched (the membership is already correct); the rejoiner only needs catch-up.
        Err(AdmissionReject::DuplicatePeer) => {
            out.push(Output::Note(Notice::Admitted(signer)));
            replay_forward(state, out);
        }
        Err(reason) => out.push(Output::Reject(Rejection::Admission(reason))),
        Ok(()) => {
            let mut m = Member::joining(signer, j.iroh_id, j.class, state.epoch);
            // The join itself is the first liveness evidence (decay-while-waiting's input).
            m.last_seen_s = state.now_s;
            // The epoch roster forms through the whole pre-round window (`WaitingForMembers` +
            // `Warmup`): a peer that joins before the first round opens is a live member. A join
            // that arrives once a round is ACTIVE is staged `pending` (the roster is frozen for the
            // epoch, §6.2) and materializes at the next epoch boundary. Admitting during `Warmup`
            // lets an initial roster larger than `min_peers` form even when the coordinator's
            // synthetic clock advances one tick per delivered join (so `min_peers` is reached before
            // the last bootstrap join lands) — the driver need not pin `min_peers` to the exact
            // bootstrap count.
            if matches!(state.phase, Phase::WaitingForMembers | Phase::Warmup) {
                upsert_member(state, m);
            } else {
                state.pending.push(m);
            }
            out.push(Output::Note(Notice::Admitted(signer)));
            replay_forward(state, out);
        }
    }
}

/// Replay-forward for a (re)joiner (architecture: "a rejoiner replays forward from the freshest
/// reachable checkpoint"): re-publish the retained ring's committed records, ascending, then the
/// standing `RoundOpen` of the currently-active round (if any).
///
/// The records let a restorer whose resync watermark lags the live round fold the gap instead of
/// ending `StaleRestore` (ABI §4.5 code 3) — rounds committed while it was detached are otherwise
/// never re-delivered on the live-only control plane, and a restore that outlasts a round leaves
/// every retry exactly as stale (the C1 relay churn drill's livelock). The standing open closes
/// the other half of the same gap: an open is a one-shot flood, so a peer whose session was born
/// after it (post-crash reconstruction, REPLACE re-admission) folds every record and then waits
/// forever for round N+1's open — it must be re-delivered, verbatim, on the join that proves the
/// peer is listening again (the c15b livelock). Bounded by the ring ([`NUM_STORED_ROUNDS`], the
/// same bound payload retention is validated against at authoring); idempotent fleet-wide — every
/// peer at/below its own watermark skips a re-emitted record by the resync guard, and a trainer
/// that already planned the open round skips the re-delivered open by its open watermark.
/// Deterministic: records and the retained open are frozen state, the emission is a pure function
/// of (state, join), and replay re-derives it (I1).
fn replay_forward(state: &CoordinatorState, out: &mut Vec<Output>) {
    let mut retained: Vec<&RoundRecord> = state
        .rounds
        .slots
        .iter()
        .filter_map(|rs| rs.record.as_ref())
        .collect();
    retained.sort_by_key(|r| r.round);
    for record in retained {
        out.push(Output::publish(VhcMessage::RoundRecord(record.clone())));
    }
    if state.phase.is_round_active() {
        if let Some(ro) = state
            .rounds
            .get(state.round)
            .and_then(|rs| rs.open.as_ref())
        {
            out.push(Output::publish(VhcMessage::RoundOpen(ro.clone())));
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

fn on_heartbeat(state: &mut CoordinatorState, out: &mut Vec<Output>, signer: PeerId, h: Heartbeat) {
    if h.round > state.max_reported_round {
        state.max_reported_round = h.round;
    }
    if let Some(m) = state.member_mut(&signer) {
        if h.round > m.last_seen_round {
            m.last_seen_round = h.round;
        }
        if h.ready == Some(true) {
            m.warmup_ready = true;
        }
    }
    // A readiness heartbeat can open round 0 without waiting for the warmup timeout (§6.2).
    maybe_exit_warmup(state, out);
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

/// Enter `Warmup`, clearing any stale per-member readiness from a previous epoch (§6.2).
fn enter_warmup(state: &mut CoordinatorState, out: &mut Vec<Output>) {
    for m in &mut state.roster {
        m.warmup_ready = false;
    }
    change_phase(state, out, Phase::Warmup);
}

/// Whether every healthy member has signalled model-readiness this warmup (the early-exit gate).
fn all_warmup_ready(state: &CoordinatorState) -> bool {
    let mut any = false;
    for m in state.roster.iter().filter(|m| m.is_healthy()) {
        any = true;
        if !m.warmup_ready {
            return false;
        }
    }
    any
}

/// Exit `Warmup` early (no timeout) once every admitted member is ready — the event-driven path a
/// readiness heartbeat triggers (§6.2/§6.5, additive).
fn maybe_exit_warmup(state: &mut CoordinatorState, out: &mut Vec<Output>) {
    if state.phase == Phase::Warmup
        && state.healthy_count() >= state.config.min_peers
        && all_warmup_ready(state)
    {
        open_epoch_first_round(state, out);
    }
}

/// Mark the epoch's first training round and open it.
fn open_epoch_first_round(state: &mut CoordinatorState, out: &mut Vec<Output>) {
    state.epoch_start_round = state.round;
    open_round(state, out);
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
    // Retain the open verbatim in the slot: it is re-published to rejoiners (§6.5 replay-forward)
    // and must be bit-identical to the original flood, not rebuilt from post-churn state.
    if let Some(s) = state.rounds.get_mut(state.round) {
        s.open = Some(ro.clone());
    }
    out.push(Output::publish(VhcMessage::RoundOpen(ro)));
}

/// Decay zombie roster entries WITHOUT a round (reliability spec §7, `WaitingForMembers` only):
/// a healthy member not heard for the full round-scaled absence equivalent — the time
/// `k_absences` rounds would have taken at their authored maximum
/// (`k × (round_train_max_s + round_witness_s)`) — is dropped, freeing its seat for an announced
/// rejoiner. Staleness is floored at the CURRENT phase's start, so entering the phase restarts
/// every member's window (a restored/reconstructed roster is never mass-dropped on tick one) and
/// a member must stay silent through the whole waiting window to decay.
///
/// Consensus safety: this only SHRINKS the healthy set — it never admits anyone the Join path
/// would not, and a decayed member that is actually alive rejoins through the ordinary
/// previously-`Dropped` lane (its node-side keeper recycles the silent session and re-announces).
fn decay_stale_members(state: &mut CoordinatorState, out: &mut Vec<Output>) {
    let k = state.config.k_absences;
    if k == 0 {
        return; // absence-dropping disabled by authored policy — waiting decay follows it
    }
    let round_span = state.config.round_train_max_s + state.config.round_witness_s;
    let window = u64::from(k).saturating_mul(round_span);
    if window == 0 {
        return;
    }
    let (now, phase_start) = (state.now_s, state.phase_start_s);
    let mut drops: Vec<PeerId> = Vec::new();
    for m in state.roster.iter_mut().filter(|m| m.is_healthy()) {
        let seen = m.last_seen_s.max(phase_start);
        if now.saturating_sub(seen) >= window {
            m.state = ClientState::Dropped;
            drops.push(m.peer);
        }
    }
    for p in drops {
        out.push(Output::Note(Notice::Dropped(p)));
    }
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
    out.push(Output::publish(VhcMessage::RoundRecord(record)));

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
        // The completion is a PUBLISHED decision, not only an advisory note: the trainers exit
        // on it (they do not know the stop condition) and the host classifies the run Completed.
        // A note alone dies in the guest wrapper and the finished run idles forever (the c15d
        // closure wedge).
        out.push(Output::publish(VhcMessage::Finished(Finished {
            rounds: state.rounds_done,
        })));
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
