// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Conversation rewind (conversation-rewind spec §2/§4): resolve a `RewindAnchor` to a retained
//! turn count, truncate the sealed-off tail, reset the derived/transient snapshot fields, bump the
//! epoch to fence late arrivals, and emit `AgentEvent::Rewound`. Split out of `engine.rs` as a
//! self-contained `impl Engine` block; behavior-preserving verbatim move.

use super::*;

impl Engine {
    /// Rewind the conversation to `anchor` (conversation-rewind spec §2/§4): truncate the turns after
    /// the anchor, reconstruct the engine state for that point (clear transient/in-flight state), bump
    /// the [`Epoch`] to fence late arrivals from the abandoned turn, and emit [`AgentEvent::Rewound`].
    ///
    /// No per-turn snapshots are stored, so the snapshot is *reconstructed* by truncating the live
    /// conversation in place (the same vec compaction mutates) and resetting the derived/transient
    /// fields that only make sense relative to the dropped tail. Returns the [`RewindOutcome`] (the
    /// retained turn count + the new epoch) so the caller can drive the durable journal seal.
    ///
    /// Must be called when no turn is live (the actor interrupts first). Anchors that resolve outside
    /// the live conversation (out of range, or compacted away below the live floor) are rejected.
    pub fn rewind_to(
        &mut self,
        anchor: &RewindAnchor,
        request_id: ReqId,
        events: &EventSink,
    ) -> Result<RewindOutcome, RewindError> {
        let span = tracing::info_span!(
            "engine.rewind",
            session = %self.snapshot.session_id,
            request_id = request_id.0,
            anchor = ?anchor
        );
        let _guard = span.enter();
        let len = self.snapshot.conversation.turns.len();
        // The snapshot surgery is shared with the durable (dormant, non-resident) rewind path.
        let outcome = match rewind_snapshot(&mut self.snapshot, anchor) {
            Ok(outcome) => outcome,
            Err(err) => {
                tracing::warn!(error = ?err, turns_before = len, "engine.rewind.rejected");
                return Err(err);
            }
        };

        // Clear the ENGINE-transient in-flight state the snapshot surgery cannot see.
        self.pending.clear();
        self.next_trigger = None;
        self.next_origin = None;
        // A rewind to the conversation root is a full context clear — the daemon's `/new` analog.
        // Notify the §10 context engine so a stateful engine resets its per-session state in step
        // with the emptied conversation (LCM: retained-DAG prune + ingest-cursor/counter reset).
        // Partial rewinds are not resets: the engine re-measures the shortened body next turn.
        if outcome.retained_turns == 0 {
            self.context.on_session_reset(&self.snapshot.session_id);
        }

        let epoch = outcome.epoch;
        let to_cursor = outcome.retained_turns as u64;
        events.emit(|seq| AgentEvent::Rewound {
            seq,
            request_id,
            to_cursor,
            epoch: epoch.0,
        });
        tracing::info!(
            turns_before = len,
            turns_after = outcome.retained_turns,
            dropped_call_ids = outcome.dropped_call_ids.len(),
            new_epoch = epoch.0,
            to_cursor,
            "engine.rewind.applied"
        );

        Ok(outcome)
    }
}

/// The snapshot-level rewind surgery (conversation-rewind spec §2/§4), shared by the resident
/// path ([`Engine::rewind_to`]) and the durable path (the host rewinds a DORMANT session's
/// persisted [`Snapshot`] directly — no engine exists to truncate): resolve the anchor, truncate
/// the sealed-off tail, reset every snapshot-derived field that described the dropped suffix, and
/// bump the [`Epoch`] to fence late arrivals. Purely snapshot-local — engine-transient state
/// (in-flight batch, armed trigger, the §10 context engine's root-reset hook) is the resident
/// caller's responsibility, and the durable caller has none of it by construction (the session is
/// dormant; a root rewind's LCM prune simply happens lazily at the next hydration's re-measure).
pub fn rewind_snapshot(
    snapshot: &mut Snapshot,
    anchor: &RewindAnchor,
) -> Result<RewindOutcome, RewindError> {
    let len = snapshot.conversation.turns.len();
    let retained = resolve_anchor(&snapshot.conversation.turns, anchor, len)?;

    // Collect the tool call-ids in the sealed-off tail (oldest first) so the host can roll the
    // workspace back via the §12 checkpoints captured before those tools ran.
    let dropped_call_ids: Vec<String> = snapshot.conversation.turns[retained..]
        .iter()
        .filter_map(|t| match t {
            Turn::Tool(tt) => Some(tt),
            _ => None,
        })
        .flat_map(|tt| tt.calls.iter().map(|(call, _)| call.call_id.clone()))
        .collect();

    // Reconstruct the snapshot for the rewound point: drop the sealed-off tail and reset every
    // derived field that only made sense relative to the now-abandoned suffix.
    snapshot.conversation.turns.truncate(retained);
    snapshot.waiting_for.clear();
    snapshot.pending_approvals.clear();
    // Cadence counters are relative to the dropped tail; reset so the rewound point starts clean.
    snapshot.iters_since_skill = 0;
    snapshot.turns_since_memory = 0;

    // Bump the incarnation epoch so any in-flight commit/event from the interrupted turn that
    // arrives late is fenced and dropped (mirrors the suspension epoch bump).
    snapshot.epoch = snapshot.epoch.next();

    Ok(RewindOutcome {
        retained_turns: retained,
        epoch: snapshot.epoch,
        dropped_call_ids,
    })
}

/// Resolve a [`RewindAnchor`] to the number of turns to retain (`[0, retained)` survive). `len` is
/// the live conversation turn count.
fn resolve_anchor(turns: &[Turn], anchor: &RewindAnchor, len: usize) -> Result<usize, RewindError> {
    match anchor {
        // Seal off the user turn at `ordinal` and everything after: keep `[0, ordinal)`.
        RewindAnchor::UserTurn { ordinal } => {
            let o = usize::try_from(*ordinal).map_err(|_| RewindError::OutOfRange)?;
            if o >= len {
                return Err(RewindError::OutOfRange);
            }
            if !matches!(turns.get(o), Some(Turn::User(_))) {
                return Err(RewindError::NotAUserTurn);
            }
            Ok(o)
        }
        // Keep the user turn at `ordinal`, seal off its reply: keep `[0, ordinal]`.
        RewindAnchor::ReplyAfter { ordinal } => {
            let o = usize::try_from(*ordinal).map_err(|_| RewindError::OutOfRange)?;
            if o >= len {
                return Err(RewindError::OutOfRange);
            }
            if !matches!(turns.get(o), Some(Turn::User(_))) {
                return Err(RewindError::NotAUserTurn);
            }
            Ok(o + 1)
        }
        // A durable journal cursor maps 1:1 onto a retained turn count in this engine's
        // coordinate space (each kept turn is one journal-addressable position). A cursor past
        // the live conversation is out of range.
        RewindAnchor::Cursor { seq } => {
            let keep = usize::try_from(*seq).map_err(|_| RewindError::OutOfRange)?;
            if keep > len {
                return Err(RewindError::OutOfRange);
            }
            Ok(keep)
        }
    }
}
