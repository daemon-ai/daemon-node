// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The compaction engine (`daemon-context-lcm-port-spec.md` §6).
//!
//! Per `compact()`: select the region between the leading anchor and the fresh tail, take the
//! **oldest leaf chunk** within the working chunk budget (`leaf_chunk_tokens`, grown by the
//! opt-in dynamic doubling ladder under backlog pressure, widened by the overflow deficit when
//! the conversation is far over budget — the force-overflow analog), summarize it into a **D0**
//! node under the 3-attempt **leaf-rescue ladder** (shrink 75% → 50% → drop-last on retry-worthy
//! aux failures, `LCM:engine.py:801-849`), advance the lifecycle frontier, opportunistically
//! condense sibling nodes up the DAG (fanin >= 4 -> D1/D2/D3, subject to the opt-in
//! cache-friendly gate), and reassemble the body as `[system] + [synthetic summary turn over the
//! DAG frontier] + [uncompacted remainder + fresh tail]`, bounded by the opt-in assembly cap
//! (`max_assembly_tokens` / `reserve_tokens_floor`). A focus topic auto-derived from recent user
//! turns and the configured `custom_instructions` are threaded into every summarization prompt;
//! the newest real user objective that fell outside the tail is preserved as a scaffold section
//! inside the summary turn (`LCM:engine.py:3978-4015`). Operating on whole [`Turn`]s keeps
//! tool-call/result pairs intact by construction (§6.7).
//!
//! The opt-in escape hatches (`LCM:engine.py:860-1160`, all default off): **dynamic leaf
//! chunking** grows the working chunk and allows up to 4 leaf passes per call; **deferred
//! maintenance** persists raw-backlog debt in the lifecycle row and spends bounded catch-up
//! passes on later turns; **critical budget pressure** bypasses the polite gates once prompt
//! pressure crosses the configured fraction of the context window.

use crate::config::LcmConfig;
use crate::escalation::{
    summarize_with_escalation, truncate_with_ellipsis, Level, SummaryCircuitBreaker, SummaryRequest,
};
use crate::externalize::{extract_ref, gc_placeholder, read_payload_record};
use crate::extraction;
use crate::ingest::render_turns;
use crate::protection::{protect_scaffold_text, redact_sensitive_text};
use crate::store::{NewNode, SourceType, Store};
use crate::tokens::Tokenizer;
use daemon_core::conversation::AssistantMsg;
use daemon_core::{Conversation, Provider, Turn};
use std::sync::Arc;
use std::time::Duration;

/// The leading marker on a synthetic summary turn (so re-compaction skips it as scaffold — §6.4).
pub(crate) const SUMMARY_SENTINEL: &str = "[LCM context summary]";

/// The scaffold prefix on a preserved newest-user-objective section
/// (`_PRESERVED_OBJECTIVE_CONTEXT_PREFIX`, `LCM:engine.py:227`).
pub(crate) const PRESERVED_OBJECTIVE_PREFIX: &str =
    "[Current user objective preserved from compacted history]";

/// The scaffold prefix on a preserved todo-list message (`_PRESERVED_TODO_CONTEXT_PREFIX`,
/// `LCM:engine.py:226`) — skipped when hunting for the newest real user objective.
const PRESERVED_TODO_PREFIX: &str =
    "[Your active task list was preserved across context compression]";

/// The separator between scaffold sections inside the synthetic summary turn (the Python
/// `summary_parts` joiner, `LCM:engine.py:4127`).
const SUMMARY_PART_SEPARATOR: &str = "\n\n---\n\n";

/// The LCM tooling note (the `_append_lcm_note_to_content` analog, `LCM:engine.py:3939-3945`) —
/// contributed as a **static guidance slot from session start** via
/// [`ContextEngine::guidance_block`](daemon_core::ContextEngine::guidance_block) instead of being
/// appended to `Conversation.system` on the first compaction: a mid-session system mutation would
/// bust the provider's cached prefix. Worded in the conditional so it is accurate before any
/// compaction has run.
pub(crate) const LCM_SYSTEM_NOTE: &str = "[Note: This conversation uses Lossless Context \
Management (LCM). As it grows, earlier turns may be compacted into hierarchical summaries. Use \
lcm_grep to search history, lcm_describe to inspect the DAG, and lcm_expand to recover original \
details from any summary.]";

/// Auto-focus bounds (`LCM:engine.py:89-91`): up to 3 recent user turns, 260 chars each, 700 total.
const AUTO_FOCUS_MAX_TURNS: usize = 3;
const AUTO_FOCUS_TURN_MAX_CHARS: usize = 260;
const AUTO_FOCUS_MAX_CHARS: usize = 700;

/// The leaf-rescue attempt cap (`_summarize_leaf_chunk_with_rescue`, `LCM:engine.py:807`).
const MAX_LEAF_RESCUE_ATTEMPTS: usize = 3;

/// The leaf-pass cap of a normal turn when dynamic leaf chunking is enabled
/// (`base_max_leaf_passes`, `LCM:engine.py:923`).
const DYNAMIC_BASE_MAX_LEAF_PASSES: usize = 4;

/// The lifecycle debt kind for deferred raw-backlog maintenance (`LCM:engine.py:1697`).
const RAW_BACKLOG_DEBT_KIND: &str = "raw_backlog";

/// What one [`run_compaction`] call did — the port of Python's `_last_compression_status` values
/// the compress path can produce (`compacted` / `overflow_recovery` / `noop`).
pub(crate) enum CompressionStatus {
    /// At least one leaf pass wrote a new summary node.
    Compacted,
    /// Forced overflow recovery reassembled the context under the cap without a new summary node
    /// (`_finalize_forced_overflow_result`, `LCM:engine.py:4178-4198`).
    OverflowRecovery,
    /// Nothing changed, with the Python `noop_reason`.
    Noop(String),
}

/// The result of one [`run_compaction`] call.
pub(crate) struct CompactionOutcome {
    /// The (re)assembled conversation.
    pub conv: Conversation,
    /// The rebuilt per-turn ingest index aligned with `conv.turns`.
    pub index: Vec<Vec<i64>>,
    /// What this call did (drives `lcm_status`'s `last_compression_*` fields).
    pub status: CompressionStatus,
}

/// Run one compaction call (`compress`, `LCM:engine.py:851-1160`): up to `max_leaf_passes` leaf
/// passes over the region between the leading anchor and the fresh tail, then condensation and
/// reassembly.
///
/// `turn_store_ids[i]` holds the `messages.store_id`s already persisted for `conv.turns[i]` (by the
/// caller's `before_turn` ingest), so a D0 node's `source_ids` come from the index rather than a
/// re-ingest (avoiding duplicate rows). `budget` is the compaction target the engine was asked to
/// hit; the leaf chunk widens beyond the working chunk when the measured usage exceeds it by more
/// (the §6.3 force-overflow analog, so one pass can plausibly get back under). `context_length`
/// (the model window from `on_model`) sizes the reserve-based assembly cap and the
/// critical-budget-pressure bypass.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_compaction(
    store: &Store,
    tok: &Tokenizer,
    cfg: &LcmConfig,
    aux_chain: &[Arc<dyn Provider>],
    breakers: &mut [SummaryCircuitBreaker],
    session_id: &str,
    turn_store_ids: Vec<Vec<i64>>,
    conv: Conversation,
    budget: usize,
    context_length: Option<usize>,
    now: f64,
) -> CompactionOutcome {
    let used_tokens = tok.count_conversation(&conv);
    let Conversation { system, mut turns } = conv;
    let mut index = turn_store_ids;
    let noop = |system: daemon_core::SystemPrompt,
                turns: Vec<Turn>,
                index: Vec<Vec<i64>>,
                reason: &str| CompactionOutcome {
        conv: Conversation { system, turns },
        index,
        status: CompressionStatus::Noop(reason.to_string()),
    };
    if turns.is_empty() {
        return noop(system, turns, index, "empty message list");
    }

    // Assembly cap + force overflow (`_should_force_overflow_recovery`, `LCM:engine.py:4217-4232`):
    // with a cap configured (`max_assembly_tokens` / `reserve_tokens_floor`), a conversation at or
    // over it compacts the whole eligible region and reassembles under the cap.
    let assembly_cap = cfg.effective_assembly_token_cap(context_length);
    let force_overflow = assembly_cap.is_some_and(|cap| used_tokens >= cap);
    // Critical budget pressure (`_critical_budget_pressure_reached`, `LCM:engine.py:1643-1656`) —
    // measured once against the pre-compaction usage, like Python's compress() entry.
    let critical_budget_pressure =
        critical_budget_pressure_reached(cfg, context_length, used_tokens);
    // Deferred maintenance (`LCM:engine.py:914-926`): a debt-carrying conversation may spend extra
    // bounded catch-up passes this turn.
    let deferred_maintenance_active = !force_overflow
        && should_run_deferred_maintenance(
            store,
            tok,
            cfg,
            session_id,
            &turns,
            critical_budget_pressure,
        );
    if deferred_maintenance_active {
        let _ = store.record_maintenance_attempt(session_id, now);
    }
    let base_max_leaf_passes = if cfg.dynamic_leaf_chunk_enabled {
        DYNAMIC_BASE_MAX_LEAF_PASSES
    } else {
        1
    };
    let max_leaf_passes = if deferred_maintenance_active {
        cfg.deferred_maintenance_max_passes.max(1)
    } else {
        base_max_leaf_passes
    };

    // Auto-derive the focus topic from recent real user turns (`_derive_auto_focus_topic`,
    // `LCM:engine.py:4340-4399`) so summarization prioritizes current user intent.
    let focus_topic = derive_auto_focus_topic(&turns, cfg).unwrap_or_default();

    let overflow_deficit = used_tokens.saturating_sub(budget);
    let mut estimated_active_tokens = used_tokens;
    let mut leaf_passes = 0usize;
    let mut compacted_any = false;
    let mut noop_reason = "no eligible raw backlog outside fresh tail";

    while leaf_passes < max_leaf_passes {
        // Region = turns[anchor .. fresh_tail_start). The anchor skips any leading synthetic
        // summary turns from a previous compaction so we never re-summarize a summary (§6.4).
        let anchor = leading_scaffold_count(&turns);
        let fresh_tail_start = turns.len().saturating_sub(cfg.fresh_tail_count);
        if fresh_tail_start <= anchor {
            noop_reason = "no eligible raw backlog outside fresh tail";
            break;
        }
        let region = &turns[anchor..fresh_tail_start];
        let raw_tokens_outside_tail: usize = region.iter().map(|t| tok.count_turn(t)).sum();
        if raw_tokens_outside_tail == 0 {
            noop_reason = "no eligible raw backlog outside fresh tail";
            break;
        }

        // Leaf floor (`LCM:engine.py:978-996`): a backlog below the working chunk defers to a
        // later, fuller turn. Bypassed by force overflow, by a budget-driven overflow deficit
        // (the daemon calls `compact` only when over budget — the §6.3 analog), and by a
        // debt-carrying turn under critical pressure.
        let working_leaf_chunk = working_leaf_chunk_tokens(cfg, raw_tokens_outside_tail);
        if raw_tokens_outside_tail < working_leaf_chunk
            && !force_overflow
            && overflow_deficit == 0
            && !(deferred_maintenance_active && critical_budget_pressure)
        {
            noop_reason = "raw backlog outside fresh tail is below leaf chunk threshold";
            break;
        }

        // Leaf chunk selection (`LCM:engine.py:978-997`): dynamic chunking summarizes the oldest
        // prefix within the working chunk budget (widened to twice the overflow deficit when the
        // conversation is far over `budget` — the §6.3 analog); static mode and force overflow
        // take the whole region (`to_compact = candidate_raw`).
        let initial_len = if force_overflow || !cfg.dynamic_leaf_chunk_enabled {
            region.len()
        } else {
            let chunk_budget = working_leaf_chunk.max(overflow_deficit.saturating_mul(2));
            select_oldest_leaf_chunk_len(region, tok, chunk_budget)
        };
        if initial_len == 0 {
            noop_reason = "no eligible leaf chunk selected";
            break;
        }

        // §9.2 pre-compaction extraction (opt-in): distill durable decisions to the daily markdown
        // before this chunk is summarized. Best-effort — it never blocks or fails compaction.
        // Routed through the primary aux provider (`LCM:engine.py:1004`).
        if cfg.extraction_enabled {
            if let Some(primary) = aux_chain.first() {
                let _ = extraction::run_extraction(
                    primary.as_ref(),
                    &render_turns(&region[..initial_len]),
                    cfg.extraction_dir().as_deref(),
                    Duration::from_millis(cfg.summary_timeout_ms),
                    now,
                )
                .await;
            }
        }

        // Summarize the leaf chunk under the 3-attempt rescue ladder
        // (`_summarize_leaf_chunk_with_rescue`, `LCM:engine.py:801-849`): budget = max(2000, 20%
        // of source) capped at 12000 (§6.5); on a retry-worthy aux failure (timeout /
        // context-length) shrink the chunk 75% → 50% → drop-last and retry. Where Python raises
        // after exhausting attempts, the port accepts the final L3 truncation — `compact()`
        // cannot fail, and the raw rows stay losslessly recoverable either way.
        let mut chunk_len = initial_len;
        let mut attempt = 0usize;
        let (source_tokens, summary) = loop {
            attempt += 1;
            let chunk = &region[..chunk_len];
            let source_tokens: usize = chunk.iter().map(|t| tok.count_turn(t)).sum();
            let text = render_turns(chunk);
            let leaf_budget = (source_tokens * 20 / 100).clamp(2000, 12000);
            let out = summarize_with_escalation(
                aux_chain,
                tok,
                breakers,
                cfg.l2_budget_ratio,
                cfg.l3_truncate_tokens,
                Duration::from_millis(cfg.summary_timeout_ms),
                SummaryRequest {
                    text: &text,
                    source_tokens,
                    token_budget: leaf_budget,
                    depth: 0,
                    focus_topic: &focus_topic,
                    custom_instructions: &cfg.custom_instructions,
                },
            )
            .await;
            let rescue_worthy = out.level == Level::L3 && out.retry_worthy_failure;
            if !rescue_worthy || attempt >= MAX_LEAF_RESCUE_ATTEMPTS || chunk_len <= 1 {
                break (source_tokens, out.text);
            }
            let next = next_leaf_rescue_chunk_len(region, chunk_len, source_tokens, tok, cfg);
            if next == 0 || next >= chunk_len {
                break (source_tokens, out.text);
            }
            tracing::warn!(
                attempt,
                max_attempts = MAX_LEAF_RESCUE_ATTEMPTS,
                from_turns = chunk_len,
                to_turns = next,
                "lcm: leaf summarization retrying with a smaller oldest chunk after a retry-worthy failure"
            );
            chunk_len = next;
        };

        // The chunk's `store_id`s come from the precomputed ingest index (persisted by
        // `before_turn`); the D0 time window is the real MIN/MAX message timestamp over them
        // (`LCM:engine.py:1014`).
        let store_ids: Vec<i64> = index
            .get(anchor..anchor + chunk_len)
            .map(|groups| groups.iter().flatten().copied().collect())
            .unwrap_or_default();
        let (earliest_at, latest_at) = store.get_time_bounds(&store_ids).unwrap_or((None, None));

        let summary_tokens = tok.count_text(&summary);
        let d0 = NewNode {
            session_id: session_id.to_string(),
            depth: 0,
            summary: summary.clone(),
            token_count: summary_tokens as i64,
            source_token_count: source_tokens as i64,
            source_ids: store_ids.clone(),
            source_type: SourceType::Messages,
            created_at: now,
            earliest_at,
            latest_at,
            expand_hint: expand_hint(&summary),
        };
        if let Err(e) = store.add_node(&d0) {
            tracing::warn!(error = %e, "lcm: failed to persist D0 node; stopping compaction");
            noop_reason = "no eligible leaf chunk selected";
            break;
        }
        if let Some(max_id) = store_ids.iter().copied().max() {
            let _ = store.advance_frontier(session_id, max_id, now);
            // §9.1 transcript GC (opt-in): now that this chunk is summarized, rewrite its
            // already-externalized payload rows to a compact GC placeholder (store_id-preserving).
            if cfg.large_output_transcript_gc_enabled {
                gc_externalized_payloads(
                    store,
                    tok,
                    cfg.externalization_dir().as_deref(),
                    session_id,
                    max_id,
                );
            }
        }

        // Drop the compacted chunk from the working view (the scaffold prefix stays for anchor
        // consistency; final assembly replaces it with the fresh summary turn).
        turns.drain(anchor..anchor + chunk_len);
        if anchor < index.len() {
            index.drain(anchor..(anchor + chunk_len).min(index.len()));
        }
        compacted_any = true;
        leaf_passes += 1;
        estimated_active_tokens = estimated_active_tokens
            .saturating_sub(source_tokens)
            .saturating_add(summary_tokens);

        // Multi-pass continuation (`LCM:engine.py:1042-1061`): only dynamic chunking loops; a
        // normal pass stops once the estimate is back under the target, or when the remaining
        // backlog dropped below the working threshold (a debt-carrying turn under critical
        // pressure keeps going up to its pass cap).
        if !cfg.dynamic_leaf_chunk_enabled {
            break;
        }
        if !force_overflow {
            if !deferred_maintenance_active && budget > 0 && estimated_active_tokens < budget {
                break;
            }
            let anchor = leading_scaffold_count(&turns);
            let tail_start = turns.len().saturating_sub(cfg.fresh_tail_count);
            if tail_start <= anchor {
                break;
            }
            let remaining_raw: usize = turns[anchor..tail_start]
                .iter()
                .map(|t| tok.count_turn(t))
                .sum();
            if remaining_raw < working_leaf_chunk_tokens(cfg, remaining_raw)
                && !(deferred_maintenance_active && critical_budget_pressure)
            {
                break;
            }
        }
    }

    // Deferred-maintenance debt bookkeeping (`_refresh_raw_backlog_debt`,
    // `LCM:engine.py:1063-1067/1126-1129`): runs on both the no-op and the compacted path so the
    // lifecycle row reflects the backlog that remains *after* this call.
    refresh_raw_backlog_debt(
        store,
        tok,
        cfg,
        session_id,
        &turns,
        critical_budget_pressure,
        now,
    );

    if !compacted_any {
        // Forced overflow recovery (`LCM:engine.py:1069-1079` + `_assemble_overflow_recovery_
        // context`, `LCM:engine.py:4293-4326`): even without a new summary node, reassemble under
        // the cap — the stale leading summary scaffold is regenerated from the frontier and the
        // over-cap tail is dropped. `overflow_recovery` when anything changed, else a no-op.
        if force_overflow {
            let anchor = leading_scaffold_count(&turns);
            let original = turns.clone();
            let kept_turns: Vec<Turn> = turns.into_iter().skip(anchor).collect();
            let kept_index: Vec<Vec<i64>> = index.into_iter().skip(anchor).collect();
            let (mut new_turns, mut new_index) = assemble_capped(
                store,
                tok,
                session_id,
                &system,
                None,
                kept_turns,
                kept_index,
                assembly_cap,
            );
            // Nothing fit under the cap: fall back to the newest turn alone rather than an empty
            // context (`LCM:engine.py:4322-4325`).
            if new_turns.is_empty() {
                if let Some(last) = original.last() {
                    new_turns = vec![last.clone()];
                    new_index = vec![Vec::new()];
                }
            }
            let status = if new_turns != original {
                CompressionStatus::OverflowRecovery
            } else {
                CompressionStatus::Noop(
                    "forced overflow recovery found no droppable active-context messages"
                        .to_string(),
                )
            };
            return CompactionOutcome {
                conv: Conversation {
                    system,
                    turns: new_turns,
                },
                index: new_index,
                status,
            };
        }
        return noop(system, turns, index, noop_reason);
    }

    // Condensation: climb depths while a level has >= fanin uncondensed siblings (§6.6), carrying
    // the same focus topic + custom instructions into the condensation prompts, subject to the
    // opt-in cache-friendly follow-on gate.
    condense(
        store,
        tok,
        cfg,
        aux_chain,
        breakers,
        session_id,
        &focus_topic,
        CondensationGate {
            leaf_compacted_this_turn: true,
            force_overflow,
            critical_budget_pressure,
        },
        now,
    )
    .await;

    // Preserve the newest real user objective that fell outside the tail as a scaffold section
    // (`_latest_user_context_anchor`, `LCM:engine.py:3978-4015`).
    let fresh_tail_start = turns.len().saturating_sub(cfg.fresh_tail_count);
    let ext_dir = cfg.externalization_dir();
    let objective = latest_user_context_anchor(
        &turns,
        fresh_tail_start,
        cfg,
        session_id,
        ext_dir.as_deref(),
    );

    // NOTE: the system prompt is deliberately untouched here. The LCM tooling note
    // (`LCM_SYSTEM_NOTE`) is a static guidance slot composed at session start
    // (`LcmContextEngine::guidance_block`) — mutating `Conversation.system` mid-session would
    // bust the provider's cached prefix.

    // Assemble: [system] + [summary turn over the DAG frontier] + [uncompacted remainder + fresh
    // tail], rebuilding the ingest index in lockstep (the synthetic summary turn holds no
    // store_ids; the kept turns keep theirs). With an assembly cap active, the tail is selected
    // newest-first under the cap and the summary sections get the remaining budget
    // (`_assemble_context`, `LCM:engine.py:4017-4158`).
    let anchor = leading_scaffold_count(&turns);
    let kept_turns: Vec<Turn> = turns.into_iter().skip(anchor).collect();
    let kept_index: Vec<Vec<i64>> = index.into_iter().skip(anchor).collect();
    let (new_turns, new_index) = assemble_capped(
        store,
        tok,
        session_id,
        &system,
        objective,
        kept_turns,
        kept_index,
        assembly_cap,
    );

    CompactionOutcome {
        conv: Conversation {
            system,
            turns: new_turns,
        },
        index: new_index,
        status: CompressionStatus::Compacted,
    }
}

/// The working leaf-chunk threshold (`_working_leaf_chunk_tokens`, `LCM:engine.py:735-743`): the
/// base `leaf_chunk_tokens`, doubled while the backlog exceeds twice the working value, capped at
/// `dynamic_leaf_chunk_max` (identity when dynamic chunking is disabled).
pub(crate) fn working_leaf_chunk_tokens(cfg: &LcmConfig, raw_tokens_outside_tail: usize) -> usize {
    let base = cfg.leaf_chunk_tokens.max(1);
    if !cfg.dynamic_leaf_chunk_enabled {
        return base;
    }
    let ceiling = base.max(cfg.dynamic_leaf_chunk_max);
    let mut working = base;
    while working < ceiling && raw_tokens_outside_tail > working * 2 {
        working = (working * 2).min(ceiling);
    }
    working
}

/// The token total of the raw backlog between the leading anchor and the fresh tail
/// (`_raw_backlog_tokens`, `LCM:engine.py:1607-1611`).
fn raw_backlog_tokens(turns: &[Turn], tok: &Tokenizer, cfg: &LcmConfig) -> usize {
    let anchor = leading_scaffold_count(turns);
    let tail_start = turns.len().saturating_sub(cfg.fresh_tail_count);
    if tail_start <= anchor {
        return 0;
    }
    turns[anchor..tail_start]
        .iter()
        .map(|t| tok.count_turn(t))
        .sum()
}

/// The backlog size that counts as actionable debt (`_raw_backlog_threshold`,
/// `LCM:engine.py:1613-1616`).
fn raw_backlog_threshold(cfg: &LcmConfig, raw_tokens: usize) -> usize {
    if cfg.dynamic_leaf_chunk_enabled {
        working_leaf_chunk_tokens(cfg, raw_tokens)
    } else {
        cfg.leaf_chunk_tokens.max(1)
    }
}

/// Whether the session's lifecycle row carries live raw-backlog debt (`_has_raw_backlog_debt`,
/// `LCM:engine.py:1618-1622`).
fn has_raw_backlog_debt(store: &Store, cfg: &LcmConfig, session_id: &str) -> bool {
    if !cfg.deferred_maintenance_enabled || session_id.is_empty() {
        return false;
    }
    store
        .get_lifecycle(session_id)
        .ok()
        .flatten()
        .is_some_and(|row| {
            row.debt_kind.as_deref() == Some(RAW_BACKLOG_DEBT_KIND) && row.debt_size_estimate > 0
        })
}

/// Whether prompt pressure crossed the configured critical fraction of the context window
/// (`_critical_budget_pressure_reached`, `LCM:engine.py:1643-1656`; disabled at ratio `0`).
pub(crate) fn critical_budget_pressure_reached(
    cfg: &LcmConfig,
    context_length: Option<usize>,
    observed_tokens: usize,
) -> bool {
    let threshold = cfg.critical_budget_pressure_ratio;
    if threshold <= 0.0 || observed_tokens == 0 {
        return false;
    }
    let Some(window) = context_length.filter(|w| *w > 0) else {
        return false;
    };
    observed_tokens as f64 / window as f64 >= threshold
}

/// Whether a debt-carrying conversation should spend catch-up passes this turn
/// (`_should_run_deferred_maintenance`, `LCM:engine.py:1658-1674`): live debt, a non-empty
/// backlog, and either an actionable backlog size or critical pressure.
pub(crate) fn should_run_deferred_maintenance(
    store: &Store,
    tok: &Tokenizer,
    cfg: &LcmConfig,
    session_id: &str,
    turns: &[Turn],
    critical_budget_pressure: bool,
) -> bool {
    if !has_raw_backlog_debt(store, cfg, session_id) {
        return false;
    }
    let raw_tokens = raw_backlog_tokens(turns, tok, cfg);
    if raw_tokens == 0 {
        return false;
    }
    if raw_tokens >= raw_backlog_threshold(cfg, raw_tokens) {
        return true;
    }
    critical_budget_pressure
}

/// Re-derive the lifecycle debt from the current backlog (`_refresh_raw_backlog_debt`,
/// `LCM:engine.py:1676-1702`): record actionable backlog (or keep existing debt alive under
/// critical pressure), clear it once the backlog is gone or below threshold.
pub(crate) fn refresh_raw_backlog_debt(
    store: &Store,
    tok: &Tokenizer,
    cfg: &LcmConfig,
    session_id: &str,
    turns: &[Turn],
    critical_budget_pressure: bool,
    now: f64,
) {
    if !cfg.deferred_maintenance_enabled || session_id.is_empty() {
        return;
    }
    let raw_tokens = raw_backlog_tokens(turns, tok, cfg);
    let threshold = if raw_tokens > 0 {
        raw_backlog_threshold(cfg, raw_tokens)
    } else {
        0
    };
    let keep_under_critical_pressure =
        raw_tokens > 0 && has_raw_backlog_debt(store, cfg, session_id) && critical_budget_pressure;
    if raw_tokens > 0 && (raw_tokens >= threshold || keep_under_critical_pressure) {
        let _ = store.record_debt(session_id, RAW_BACKLOG_DEBT_KIND, raw_tokens as i64, now);
        return;
    }
    if has_raw_backlog_debt(store, cfg, session_id) {
        let _ = store.clear_debt(session_id, now);
    }
}

/// The oldest prefix of `region` whose token total stays within `budget` — always at least one
/// turn once anything is selected (`_select_oldest_leaf_chunk`, `LCM:engine.py:745-758`). Returns
/// the number of turns selected.
fn select_oldest_leaf_chunk_len(region: &[Turn], tok: &Tokenizer, budget: usize) -> usize {
    let mut used = 0usize;
    let mut selected = 0usize;
    for turn in region {
        let cost = tok.count_turn(turn);
        if used + cost > budget && selected > 0 {
            break;
        }
        selected += 1;
        used += cost;
    }
    selected
}

/// The next, smaller rescue chunk after a retry-worthy failure (`_next_leaf_rescue_chunk`,
/// `LCM:engine.py:778-799`): shrink toward 75% then 50% of the failing chunk's tokens (floored at
/// `leaf_chunk_tokens`), falling back to dropping the newest turn. Returns the new turn count
/// (`0` when the chunk cannot shrink).
fn next_leaf_rescue_chunk_len(
    region: &[Turn],
    current_len: usize,
    current_source_tokens: usize,
    tok: &Tokenizer,
    cfg: &LcmConfig,
) -> usize {
    if current_len <= 1 {
        return 0;
    }
    let floor_tokens = cfg.leaf_chunk_tokens.max(1);
    let shrink_targets = [
        floor_tokens.max(current_source_tokens * 3 / 4),
        floor_tokens.max(current_source_tokens / 2),
    ];
    for target in shrink_targets {
        if target >= current_source_tokens {
            continue;
        }
        let smaller = select_oldest_leaf_chunk_len(&region[..current_len], tok, target);
        if smaller > 0 && smaller < current_len {
            return smaller;
        }
    }
    current_len - 1
}

/// Infer a compact focus hint from the most recent real user turns (`_derive_auto_focus_topic`,
/// `LCM:engine.py:4340-4399`): walk backwards collecting up to 3 user turns (skipping synthetic
/// context summaries), redact each through the configured sensitive catalog, bound each line at
/// 260 chars and the whole brief at 700.
pub(crate) fn derive_auto_focus_topic(turns: &[Turn], cfg: &LcmConfig) -> Option<String> {
    let mut candidates: Vec<String> = Vec::new();
    for turn in turns.iter().rev() {
        let Turn::User(user) = turn else {
            continue;
        };
        if is_context_summary_content(&user.text) {
            continue;
        }
        let text = if cfg.sensitive_patterns_enabled {
            redact_sensitive_text(&user.text, &cfg.sensitive_patterns)
        } else {
            user.text.clone()
        };
        let text = text.split_whitespace().collect::<Vec<_>>().join(" ");
        if text.is_empty() {
            continue;
        }
        candidates.push(truncate_with_ellipsis(&text, AUTO_FOCUS_TURN_MAX_CHARS));
        if candidates.len() >= AUTO_FOCUS_MAX_TURNS {
            break;
        }
    }
    if candidates.is_empty() {
        return None;
    }
    candidates.reverse();
    let focus = format!(
        "Recent user focus:\n{}",
        candidates
            .iter()
            .map(|c| format!("- {c}"))
            .collect::<Vec<_>>()
            .join("\n")
    );
    Some(truncate_with_ellipsis(&focus, AUTO_FOCUS_MAX_CHARS))
}

/// Whether text is a synthetic context summary rather than real user intent
/// (`_is_context_summary_content`, `LCM:engine.py:4401-4415`, plus this port's own scaffold
/// sentinel).
fn is_context_summary_content(text: &str) -> bool {
    text.contains("CONTEXT COMPACTION")
        || text.contains("CONTEXT SUMMARY")
        || text.contains("Earlier turns have been compacted")
        || text.contains("Earlier turns were compacted")
        || text.trim_start().starts_with(SUMMARY_SENTINEL)
}

/// The scaffolded newest real user objective omitted from the tail (`_latest_user_context_anchor`,
/// `LCM:engine.py:3978-4015`): tool-heavy turns can push the operative user request outside the
/// fresh tail; the returned section is emitted inside the summary turn so restart reconciliation
/// ignores it instead of ingesting a duplicate user message. A previously emitted
/// preserved-objective section is carried forward when no newer real user turn exists.
fn latest_user_context_anchor(
    turns: &[Turn],
    fresh_tail_start: usize,
    cfg: &LcmConfig,
    session_id: &str,
    ext_dir: Option<&std::path::Path>,
) -> Option<String> {
    for (i, turn) in turns.iter().enumerate().rev() {
        match turn {
            Turn::Assistant(a) if a.text.starts_with(SUMMARY_SENTINEL) => {
                // A previous compaction's scaffold: carry its preserved objective forward (there is
                // no newer real user turn or we would have returned already).
                if let Some(section) = preserved_objective_in_scaffold(&a.text) {
                    return Some(section);
                }
            }
            Turn::User(u) => {
                if u.text.trim_start().starts_with(PRESERVED_TODO_PREFIX) {
                    continue;
                }
                if i >= fresh_tail_start {
                    // The newest real user turn is inside the kept tail — nothing to preserve.
                    return None;
                }
                let content = protect_scaffold_text(&u.text, cfg, session_id, ext_dir);
                return Some(format!("{PRESERVED_OBJECTIVE_PREFIX}\n{content}"));
            }
            _ => {}
        }
    }
    None
}

/// Extract a previously emitted preserved-objective section from a scaffold summary turn's body
/// (the carry-forward path of `_latest_user_context_anchor`).
fn preserved_objective_in_scaffold(text: &str) -> Option<String> {
    let start = text.find(PRESERVED_OBJECTIVE_PREFIX)?;
    let rest = &text[start..];
    let end = rest.find(SUMMARY_PART_SEPARATOR).unwrap_or(rest.len());
    Some(rest[..end].trim_end().to_string())
}

/// Rewrite summarized, unpinned tool-result rows that still carry an inline externalized-payload
/// placeholder to a compact GC placeholder (§9.1 transcript GC —
/// `_maybe_gc_compacted_tool_results`, `LCM:engine.py:3437-3492`). Guards, in order:
///
/// - only `role='tool' AND pinned=0` rows are candidates (the store query);
/// - the payload record must be recoverable on disk *before* the inline copy is dropped (the
///   Python `load_externalized_payload(ref) is not None` data-loss guard);
/// - the record must be a whole-body tool-result spill (`kind == "tool_result"`,
///   `LCM:engine.py:3466`): a base64/data-URI run placeholder covers only a slice of the row, so
///   GC'ing it would drop the surrounding prose;
/// - the rewrite re-checks role/pinned/idempotence under the store lock and re-estimates the
///   row's cached tokens (`gc_externalized_tool_result`, `LCM:store.py:381-405`).
///
/// Best-effort: a failed row is left as-is. No-op without an externalization dir (nothing can be
/// verified recoverable).
fn gc_externalized_payloads(
    store: &Store,
    tok: &Tokenizer,
    ext_dir: Option<&std::path::Path>,
    session_id: &str,
    max_store_id: i64,
) {
    let Some(dir) = ext_dir else {
        return;
    };
    let rows = match store.messages_to_gc(session_id, max_store_id) {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!(error = %e, "lcm: transcript-GC candidate query failed");
            return;
        }
    };
    for row in rows {
        let Some(content) = row.content.as_deref() else {
            continue;
        };
        let Some(reference) = extract_ref(content) else {
            continue;
        };
        let Some(record) = read_payload_record(dir, &reference) else {
            continue;
        };
        if record.get("kind").and_then(|k| k.as_str()) != Some("tool_result") {
            continue;
        }
        let placeholder = gc_placeholder(true, &reference);
        let placeholder_tokens = tok.count_text(&placeholder) as i64;
        match store.gc_externalized_tool_result(row.store_id, &placeholder, placeholder_tokens) {
            Ok(_) => {}
            Err(e) => {
                tracing::warn!(error = %e, store_id = row.store_id, "lcm: transcript-GC rewrite failed");
            }
        }
    }
}

/// Count leading synthetic-summary turns (the replayed scaffold to skip — §6.4).
pub(crate) fn leading_scaffold_count(turns: &[Turn]) -> usize {
    turns
        .iter()
        .take_while(|t| matches!(t, Turn::Assistant(a) if a.text.starts_with(SUMMARY_SENTINEL)))
        .count()
}

/// Whether a normal leaf compaction pass can actually make progress, with the Python no-op reason
/// (`_leaf_compaction_candidate_status`, `LCM:engine.py:696-733`): there is raw backlog between
/// the leading scaffold and the protected fresh tail, and its token total meets the working
/// leaf-chunk floor (the dynamic ladder when enabled). `before_turn` uses this so over-threshold
/// pressure is not advertised while `compact()` would immediately no-op (all pressure inside the
/// fresh tail, or a backlog below the chunk floor); error-driven overflow recovery bypasses the
/// gate (`force_overflow` in Python).
pub(crate) fn leaf_compaction_candidate_status(
    turns: &[Turn],
    tok: &Tokenizer,
    cfg: &LcmConfig,
) -> (bool, &'static str) {
    if turns.is_empty() {
        return (false, "empty message list");
    }
    let anchor = leading_scaffold_count(turns);
    let fresh_tail_start = turns.len().saturating_sub(cfg.fresh_tail_count);
    if fresh_tail_start <= anchor {
        return (false, "no eligible raw backlog outside fresh tail");
    }
    let raw_tokens: usize = turns[anchor..fresh_tail_start]
        .iter()
        .map(|t| tok.count_turn(t))
        .sum();
    if raw_tokens < working_leaf_chunk_tokens(cfg, raw_tokens) {
        return (
            false,
            "raw backlog outside fresh tail is below leaf chunk threshold",
        );
    }
    (true, "eligible raw backlog outside fresh tail")
}

/// The compress-path flags governing the cache-friendly follow-on condensation gate
/// (`_should_allow_follow_on_condensation`, `LCM:engine.py:3817-3840`).
#[derive(Clone, Copy)]
struct CondensationGate {
    /// Whether a leaf pass just ran this call (the gate only applies right after one).
    leaf_compacted_this_turn: bool,
    /// Overflow recovery always condenses (getting under the cap outranks cache stability).
    force_overflow: bool,
    /// Critical pressure bypasses the polite gate.
    critical_budget_pressure: bool,
}

impl CondensationGate {
    /// `Ok(())` allows the group; `Err(reason)` suppresses it (the Python suppression-reason
    /// strings, surfaced in the debug log).
    fn allows(&self, cfg: &LcmConfig, uncondensed_count: usize) -> Result<(), &'static str> {
        if !self.leaf_compacted_this_turn
            || !cfg.cache_friendly_condensation_enabled
            || self.force_overflow
            || self.critical_budget_pressure
        {
            return Ok(());
        }
        let fanin = cfg.condensation_fanin.max(1);
        let debt_threshold = fanin * cfg.cache_friendly_min_debt_groups.max(1);
        if uncondensed_count >= debt_threshold {
            return Ok(());
        }
        if uncondensed_count == fanin {
            Err("cache_friendly_single_group")
        } else {
            Err("cache_friendly_low_debt")
        }
    }
}

/// Climb the DAG, condensing each depth that has accumulated >= `fanin` uncondensed siblings into a
/// single node at the next depth (`_maybe_condense`, `LCM:engine.py:3842-3934`). One group per
/// depth per call; depths with too few siblings are skipped, not terminal (a deeper level can
/// still condense). In cache-friendly mode a follow-on condensation right after a leaf pass needs
/// `fanin * cache_friendly_min_debt_groups` accumulated siblings, and at most one group condenses
/// per call.
#[allow(clippy::too_many_arguments)]
async fn condense(
    store: &Store,
    tok: &Tokenizer,
    cfg: &LcmConfig,
    aux_chain: &[Arc<dyn Provider>],
    breakers: &mut [SummaryCircuitBreaker],
    session_id: &str,
    focus_topic: &str,
    gate: CondensationGate,
    now: f64,
) {
    let fanin = cfg.condensation_fanin.max(1);
    // `incremental_max_depth`: 0 disables condensation; -1 (unlimited) derives the upper bound
    // from the deepest existing node + 1 so condensation can always create the next depth.
    let upper = match cfg.incremental_max_depth {
        0 => return,
        d if d < 0 => store.max_depth(session_id).unwrap_or(-1).max(0) + 1,
        d => d,
    };
    for depth in 0..upper {
        // Unbounded fetch: the cache-friendly debt gate compares the *total* uncondensed count at
        // this depth against `fanin * min_debt_groups`, not just the first group.
        let uncondensed = match store.get_uncondensed_at_depth(session_id, depth, i64::MAX) {
            Ok(f) => f,
            Err(e) => {
                tracing::warn!(error = %e, depth, "lcm: condensation feeder query failed");
                break;
            }
        };
        if uncondensed.len() < fanin {
            continue;
        }
        if let Err(reason) = gate.allows(cfg, uncondensed.len()) {
            tracing::debug!(
                depth,
                uncondensed = uncondensed.len(),
                reason,
                "lcm: cache-friendly gate suppressed follow-on condensation"
            );
            continue;
        }
        let feeders = &uncondensed[..fanin];
        let child_ids: Vec<i64> = feeders.iter().map(|n| n.node_id).collect();
        let source_tokens: usize = feeders.iter().map(|n| n.token_count.max(0) as usize).sum();
        let text = feeders
            .iter()
            .map(|n| n.summary.as_str())
            .collect::<Vec<_>>()
            .join(SUMMARY_PART_SEPARATOR);
        // Condense budget = max(1000, 40% of source) (§6.6).
        let budget = (source_tokens * 40 / 100).max(1000);
        let out = summarize_with_escalation(
            aux_chain,
            tok,
            breakers,
            cfg.l2_budget_ratio,
            cfg.l3_truncate_tokens,
            Duration::from_millis(cfg.summary_timeout_ms),
            SummaryRequest {
                text: &text,
                source_tokens: source_tokens.max(1),
                token_budget: budget,
                depth: depth + 1,
                focus_topic,
                custom_instructions: &cfg.custom_instructions,
            },
        )
        .await;
        let summary = out.text;
        let earliest = feeders
            .iter()
            .filter_map(|n| n.earliest_at)
            .fold(None, min_opt);
        let latest = feeders
            .iter()
            .filter_map(|n| n.latest_at)
            .fold(None, max_opt);
        let node = NewNode {
            session_id: session_id.to_string(),
            depth: depth + 1,
            summary: summary.clone(),
            token_count: tok.count_text(&summary) as i64,
            source_token_count: source_tokens as i64,
            source_ids: child_ids,
            source_type: SourceType::Nodes,
            created_at: now,
            earliest_at: earliest,
            latest_at: latest,
            expand_hint: expand_hint(&summary),
        };
        if let Err(e) = store.add_node(&node) {
            tracing::warn!(error = %e, depth, "lcm: failed to persist condensation node");
            break;
        }
        // Cache-friendly mode condenses at most one group per compress call
        // (`LCM:engine.py:3929-3930`) — the next group waits for a later turn.
        if gate.leaf_compacted_this_turn && cfg.cache_friendly_condensation_enabled {
            break;
        }
    }
}

/// Assemble the post-compaction body — `[summary turn over the DAG frontier] + [tail]` — under an
/// optional assembly cap (`_assemble_context`, `LCM:engine.py:4017-4158`). Without a cap this is
/// the plain summary turn + full tail. With one:
///
/// - the tail is selected newest-first while it fits (`LCM:engine.py:4060-4080`): an over-budget
///   assistant/tool turn is skippable (derived context), but a skipped gap ends selection at the
///   next fitting turn, and an over-budget *user* turn (prompt-bearing) stops immediately;
/// - the summary sections get the remaining budget (`summary_budget`,
///   `LCM:engine.py:4081/4114-4125`): the preserved objective first, then the frontier sections,
///   each skipped (not terminal) when the cumulative body would exceed the budget.
///
/// The port counts each summary candidate as the full rendered body (sentinel included), which is
/// strictly conservative, so Python's post-hoc over-cap objective strip
/// (`LCM:engine.py:4137-4156`) is unreachable and not ported.
#[allow(clippy::too_many_arguments)]
fn assemble_capped(
    store: &Store,
    tok: &Tokenizer,
    session_id: &str,
    system: &daemon_core::SystemPrompt,
    preserved_objective: Option<String>,
    kept_turns: Vec<Turn>,
    kept_index: Vec<Vec<i64>>,
    cap: Option<usize>,
) -> (Vec<Turn>, Vec<Vec<i64>>) {
    let used = if system.text.is_empty() {
        0
    } else {
        tok.count_text(&system.text) + crate::tokens::PER_MESSAGE_OVERHEAD
    };

    // Tail selection (newest-first under the cap; identity without one).
    let mut pairs: Vec<(Turn, Vec<i64>)> = {
        let mut index = kept_index;
        index.resize(kept_turns.len(), Vec::new());
        kept_turns.into_iter().zip(index).collect()
    };
    let mut tail_token_total = 0usize;
    if let Some(cap) = cap {
        let mut kept_rev: Vec<(Turn, Vec<i64>)> = Vec::new();
        let mut skipped_tail_gap = false;
        for (turn, ids) in pairs.into_iter().rev() {
            let turn_tokens = tok.count_turn(&turn);
            if used + tail_token_total + turn_tokens > cap {
                if is_budget_droppable_tail_turn(&turn) {
                    skipped_tail_gap = true;
                    continue;
                }
                break;
            }
            if skipped_tail_gap {
                break;
            }
            kept_rev.push((turn, ids));
            tail_token_total += turn_tokens;
        }
        kept_rev.reverse();
        pairs = kept_rev;
    }

    // Summary sections under the remaining budget: the preserved objective first, then the
    // frontier (highest depth first), each section skipped when the cumulative body would not fit.
    let summary_budget = cap.map(|c| c.saturating_sub(used + tail_token_total));
    let frontier = store
        .get_uncondensed_frontier(session_id)
        .unwrap_or_default();
    let mut parts: Vec<String> = Vec::new();
    if let Some(objective) = preserved_objective {
        parts.push(objective);
    }
    for node in &frontier {
        parts.push(format!(
            "[{label} Summary (d{depth}, node {id})]\n{summary}\n[Expand for details: {hint}]",
            label = depth_label(node.depth),
            depth = node.depth,
            id = node.node_id,
            summary = node.summary,
            hint = node.expand_hint,
        ));
    }
    let mut selected: Vec<String> = Vec::new();
    for part in parts {
        if let Some(budget) = summary_budget {
            let mut candidate = selected.clone();
            candidate.push(part.clone());
            let body = summary_turn_body(&candidate);
            if tok.count_text(&body) + crate::tokens::PER_MESSAGE_OVERHEAD > budget {
                continue;
            }
        }
        selected.push(part);
    }

    let mut new_turns = Vec::with_capacity(1 + pairs.len());
    let mut new_index = Vec::with_capacity(1 + pairs.len());
    if !selected.is_empty() {
        new_turns.push(Turn::Assistant(AssistantMsg::text(summary_turn_body(
            &selected,
        ))));
        new_index.push(Vec::new());
    }
    for (turn, ids) in pairs {
        new_turns.push(turn);
        new_index.push(ids);
    }
    (new_turns, new_index)
}

/// The rendered summary-turn body: the scaffold sentinel plus the `---`-joined sections.
fn summary_turn_body(parts: &[String]) -> String {
    format!(
        "{SUMMARY_SENTINEL}\n\n{}",
        parts.join(SUMMARY_PART_SEPARATOR)
    )
}

/// Whether an over-budget tail turn may be evicted during capped assembly
/// (`_is_budget_droppable_tail_message`, `LCM:engine.py:4160-4176`): assistant/tool turns are
/// derived context (droppable unless they carry a preserved todo/objective scaffold); user turns
/// are prompt-bearing and stop tail selection.
fn is_budget_droppable_tail_turn(turn: &Turn) -> bool {
    let text = match turn {
        Turn::User(_) => return false,
        Turn::Assistant(a) => &a.text,
        Turn::Tool(t) => &t.assistant.text,
    };
    !text.contains(PRESERVED_TODO_PREFIX) && !text.contains(PRESERVED_OBJECTIVE_PREFIX)
}

/// A human label for a DAG depth used in the assembled summary block (`LCM:engine.py:4103-4107`).
fn depth_label(depth: i64) -> String {
    match depth {
        0 => "Recent".to_string(),
        1 => "Session Arc".to_string(),
        2 => "Durable".to_string(),
        d => format!("Depth-{d}"),
    }
}

/// Extract the text after the last `Expand for details about:` marker (first line only) — the
/// stored expand hint (`_extract_expand_hint`, `LCM:engine.py:4417-4426`).
fn expand_hint(summary: &str) -> String {
    const MARKER: &str = "Expand for details about:";
    match summary.rfind(MARKER) {
        Some(idx) => summary[idx + MARKER.len()..]
            .trim_start()
            .lines()
            .next()
            .unwrap_or("")
            .trim()
            .to_string(),
        None => String::new(),
    }
}

fn min_opt(acc: Option<f64>, v: f64) -> Option<f64> {
    Some(acc.map_or(v, |a| a.min(v)))
}

fn max_opt(acc: Option<f64>, v: f64) -> Option<f64> {
    Some(acc.map_or(v, |a| a.max(v)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use daemon_protocol::UserMsg;

    fn cfg() -> LcmConfig {
        LcmConfig::in_memory()
    }

    fn user(text: &str) -> Turn {
        Turn::User(UserMsg::new(text))
    }

    fn assistant(text: &str) -> Turn {
        Turn::Assistant(AssistantMsg::text(text))
    }

    #[test]
    fn oldest_leaf_chunk_respects_the_token_budget() {
        let tok = Tokenizer::heuristic();
        // Each turn ~ 40 chars -> ~10 tokens + 4 overhead = 14.
        let region: Vec<Turn> = (0..10).map(|i| user(&format!("{i} ").repeat(20))).collect();
        let per_turn = tok.count_turn(&region[0]);
        let n = select_oldest_leaf_chunk_len(&region, &tok, per_turn * 3);
        assert_eq!(n, 3, "three turns fit the budget");
        // A budget below one turn still selects the first turn.
        assert_eq!(select_oldest_leaf_chunk_len(&region, &tok, 1), 1);
    }

    #[test]
    fn rescue_ladder_shrinks_75_then_50_then_drop_last() {
        let tok = Tokenizer::heuristic();
        let region: Vec<Turn> = (0..8).map(|i| user(&format!("{i} ").repeat(50))).collect();
        let mut c = cfg();
        c.leaf_chunk_tokens = 1; // floor below every target so the percentage targets apply
        let total: usize = region.iter().map(|t| tok.count_turn(t)).sum();
        let after_75 = next_leaf_rescue_chunk_len(&region, 8, total, &tok, &c);
        assert!((5..8).contains(&after_75), "75% target shrank: {after_75}");
        // A single-turn chunk cannot shrink.
        assert_eq!(next_leaf_rescue_chunk_len(&region, 1, 100, &tok, &c), 0);
    }

    #[test]
    fn auto_focus_collects_recent_user_turns_newest_last() {
        let turns = vec![
            user("first question"),
            assistant("answer"),
            user("second question"),
            assistant("answer"),
            user("third question"),
            assistant("answer"),
            user("fourth question"),
        ];
        let focus = derive_auto_focus_topic(&turns, &cfg()).unwrap();
        assert!(focus.starts_with("Recent user focus:\n"));
        // The three most recent user turns, oldest first.
        assert_eq!(
            focus,
            "Recent user focus:\n- second question\n- third question\n- fourth question"
        );
    }

    #[test]
    fn auto_focus_skips_synthetic_summaries_and_redacts() {
        let mut c = cfg();
        c.sensitive_patterns_enabled = true;
        let turns = vec![
            user("please use api_key=SECRETSECRET12345 for the call"), // gitleaks:allow (fixture)
            user("Earlier turns have been compacted into summaries"),
        ];
        let focus = derive_auto_focus_topic(&turns, &c).unwrap();
        assert!(!focus.contains("SECRETSECRET12345"), "secret redacted");
        assert!(focus.contains("name=api_key"));
        assert!(
            !focus.contains("compacted into summaries"),
            "synthetic summary content skipped"
        );
    }

    #[test]
    fn auto_focus_bounds_each_line_and_the_brief() {
        let long = "word ".repeat(200);
        let turns = vec![user(&long), user(&long), user(&long), user(&long)];
        let focus = derive_auto_focus_topic(&turns, &cfg()).unwrap();
        assert!(focus.chars().count() <= AUTO_FOCUS_MAX_CHARS);
        assert!(focus.ends_with('…'));
    }

    #[test]
    fn anchor_preserves_newest_user_objective_outside_the_tail() {
        let turns = vec![
            user("the real objective"),
            assistant("working on it"),
            assistant("still working"),
            assistant("tool output noise"),
        ];
        // Tail = last 2 turns; the newest user turn (index 0) is outside it.
        let anchor = latest_user_context_anchor(&turns, 2, &cfg(), "s1", None).unwrap();
        assert!(anchor.starts_with(PRESERVED_OBJECTIVE_PREFIX));
        assert!(anchor.contains("the real objective"));
    }

    #[test]
    fn anchor_is_none_when_the_newest_user_turn_is_in_the_tail() {
        let turns = vec![
            user("old question"),
            assistant("answer"),
            user("current question"),
            assistant("answering"),
        ];
        assert!(latest_user_context_anchor(&turns, 2, &cfg(), "s1", None).is_none());
    }

    #[test]
    fn anchor_carries_a_previous_scaffold_objective_forward() {
        let scaffold_body = format!(
            "{SUMMARY_SENTINEL}\n\n{PRESERVED_OBJECTIVE_PREFIX}\nkeep porting the crate{SUMMARY_PART_SEPARATOR}[Recent Summary (d0, node 1)]\nstuff\n[Expand for details: hint]"
        );
        let turns = vec![
            assistant(&scaffold_body),
            assistant("tool chatter"),
            assistant("more chatter"),
        ];
        let anchor = latest_user_context_anchor(&turns, 2, &cfg(), "s1", None).unwrap();
        assert_eq!(
            anchor,
            format!("{PRESERVED_OBJECTIVE_PREFIX}\nkeep porting the crate")
        );
    }

    #[test]
    fn anchor_skips_preserved_todo_messages() {
        let turns = vec![
            user("the actual objective"),
            user(&format!("{PRESERVED_TODO_PREFIX}\n- [ ] a task")),
            assistant("chatter"),
            assistant("chatter"),
        ];
        let anchor = latest_user_context_anchor(&turns, 2, &cfg(), "s1", None).unwrap();
        assert!(anchor.contains("the actual objective"));
    }

    #[test]
    fn expand_hint_takes_text_after_the_marker_first_line_only() {
        let s = "A summary.\nExpand for details about: the auth refactor\nand more prose";
        assert_eq!(expand_hint(s), "the auth refactor");
        assert_eq!(expand_hint("no marker here"), "");
        // The last occurrence wins (rfind), matching the Python.
        let twice = "Expand for details about: first\nExpand for details about: second";
        assert_eq!(expand_hint(twice), "second");
    }

    #[test]
    fn working_leaf_chunk_doubles_with_backlog_pressure_up_to_the_ceiling() {
        let mut c = cfg();
        c.leaf_chunk_tokens = 10_000;
        c.dynamic_leaf_chunk_max = 40_000;
        // Disabled: identity regardless of backlog.
        c.dynamic_leaf_chunk_enabled = false;
        assert_eq!(working_leaf_chunk_tokens(&c, 1_000_000), 10_000);
        c.dynamic_leaf_chunk_enabled = true;
        // Backlog <= 2x base: stays at base.
        assert_eq!(working_leaf_chunk_tokens(&c, 20_000), 10_000);
        // One doubling.
        assert_eq!(working_leaf_chunk_tokens(&c, 25_000), 20_000);
        // Ladder caps at dynamic_leaf_chunk_max.
        assert_eq!(working_leaf_chunk_tokens(&c, 1_000_000), 40_000);
    }

    #[test]
    fn condensation_gate_suppresses_only_polite_follow_on_groups() {
        let mut c = cfg();
        c.condensation_fanin = 4;
        c.cache_friendly_min_debt_groups = 2;
        let gate = |leaf, force, critical| CondensationGate {
            leaf_compacted_this_turn: leaf,
            force_overflow: force,
            critical_budget_pressure: critical,
        };
        // Disabled feature: always allowed.
        c.cache_friendly_condensation_enabled = false;
        assert!(gate(true, false, false).allows(&c, 4).is_ok());
        c.cache_friendly_condensation_enabled = true;
        // Not a follow-on (no leaf pass this turn): allowed.
        assert!(gate(false, false, false).allows(&c, 4).is_ok());
        // Follow-on with exactly one group: suppressed as single-group.
        assert_eq!(
            gate(true, false, false).allows(&c, 4),
            Err("cache_friendly_single_group")
        );
        // Between one group and the debt threshold: low debt.
        assert_eq!(
            gate(true, false, false).allows(&c, 6),
            Err("cache_friendly_low_debt")
        );
        // At the debt threshold (fanin * min_debt_groups): allowed.
        assert!(gate(true, false, false).allows(&c, 8).is_ok());
        // Overflow/critical pressure bypass the gate.
        assert!(gate(true, true, false).allows(&c, 4).is_ok());
        assert!(gate(true, false, true).allows(&c, 4).is_ok());
    }

    #[test]
    fn critical_budget_pressure_needs_a_ratio_and_a_window() {
        let mut c = cfg();
        // Disabled at ratio 0 (the default).
        assert!(!critical_budget_pressure_reached(&c, Some(1000), 999));
        c.critical_budget_pressure_ratio = 0.9;
        // No window / no usage: not critical.
        assert!(!critical_budget_pressure_reached(&c, None, 999));
        assert!(!critical_budget_pressure_reached(&c, Some(1000), 0));
        // Under / at the fraction.
        assert!(!critical_budget_pressure_reached(&c, Some(1000), 899));
        assert!(critical_budget_pressure_reached(&c, Some(1000), 900));
    }

    #[test]
    fn effective_assembly_cap_prefers_the_tighter_of_max_and_reserve() {
        let mut c = cfg();
        assert_eq!(c.effective_assembly_token_cap(Some(10_000)), None);
        c.max_assembly_tokens = 6_000;
        assert_eq!(c.effective_assembly_token_cap(Some(10_000)), Some(6_000));
        c.reserve_tokens_floor = 5_000;
        // window - reserve = 5000 < max_assembly_tokens.
        assert_eq!(c.effective_assembly_token_cap(Some(10_000)), Some(5_000));
        // Without a window the reserve leg is inert.
        assert_eq!(c.effective_assembly_token_cap(None), Some(6_000));
    }

    #[test]
    fn capped_assembly_keeps_the_newest_fitting_tail_and_stops_at_a_user_turn() {
        let store = Store::open_in_memory().unwrap();
        let tok = Tokenizer::heuristic();
        let system = daemon_core::SystemPrompt::new("");
        let filler = "w ".repeat(200); // ~100 tokens + overhead per turn
        let turns = vec![
            assistant(&filler),
            user(&filler),
            assistant(&filler),
            assistant("tiny"),
        ];
        let per_big = tok.count_turn(&turns[0]);
        let per_tiny = tok.count_turn(&turns[3]);
        // Cap fits [tiny + one big]: newest-first selection keeps turns 2..4 and stops at the
        // over-budget *user* turn (index 1) without skipping past it.
        let cap = per_big + per_tiny + 1;
        let (out, index) = assemble_capped(
            &store,
            &tok,
            "s1",
            &system,
            None,
            turns.clone(),
            vec![Vec::new(); 4],
            Some(cap),
        );
        assert_eq!(out.len(), 2);
        assert_eq!(out[0], turns[2]);
        assert_eq!(out[1], turns[3]);
        assert_eq!(index.len(), 2);

        // An over-budget *assistant* turn is droppable: with a cap fitting only the tiny turn the
        // big assistant turns are skipped, and selection ends at the first fitting turn after a
        // gap (keeping just the newest tiny turn).
        let (out, _) = assemble_capped(
            &store,
            &tok,
            "s1",
            &system,
            None,
            vec![assistant("tiny"), assistant(&filler), assistant("tiny2")],
            vec![Vec::new(); 3],
            Some(per_tiny + 2),
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0], assistant("tiny2"));

        // Without a cap assembly is the identity (plus the summary turn when a frontier exists).
        let (out, _) = assemble_capped(
            &store,
            &tok,
            "s1",
            &system,
            None,
            turns.clone(),
            vec![Vec::new(); 4],
            None,
        );
        assert_eq!(out, turns);
    }

    #[test]
    fn capped_assembly_trims_summary_sections_to_the_remaining_budget() {
        let store = Store::open_in_memory().unwrap();
        let tok = Tokenizer::heuristic();
        let system = daemon_core::SystemPrompt::new("");
        for (i, size) in [("small", 40usize), ("large", 4000)].iter().enumerate() {
            store
                .add_node(&NewNode {
                    session_id: "s1".into(),
                    depth: 1 - i as i64, // d1 first (frontier orders highest depth first)
                    summary: format!("{} {}", size.0, "s ".repeat(size.1)),
                    token_count: 10,
                    source_token_count: 100,
                    source_ids: vec![],
                    source_type: SourceType::Messages,
                    created_at: 1.0,
                    earliest_at: None,
                    latest_at: None,
                    expand_hint: String::new(),
                })
                .unwrap();
        }
        let tail = vec![assistant("tiny")];
        // Budget fits the d1 (small) section but not the d0 (large) one: the oversized section is
        // skipped, not terminal.
        let (out, _) = assemble_capped(
            &store,
            &tok,
            "s1",
            &system,
            None,
            tail.clone(),
            vec![Vec::new()],
            Some(200),
        );
        assert_eq!(out.len(), 2, "summary turn + tail");
        let Turn::Assistant(a) = &out[0] else {
            panic!("expected the summary turn first");
        };
        assert!(a.text.contains("small"), "small section kept");
        assert!(!a.text.contains("large"), "large section trimmed");

        // A cap too small for any section drops the summary turn entirely.
        let (out, _) = assemble_capped(
            &store,
            &tok,
            "s1",
            &system,
            None,
            tail.clone(),
            vec![Vec::new()],
            Some(tok.count_turn(&tail[0]) + 1),
        );
        assert_eq!(out, tail);
    }

    #[test]
    fn budget_droppable_tail_turns_exclude_user_and_preserved_scaffolds() {
        assert!(!is_budget_droppable_tail_turn(&user("a prompt")));
        assert!(is_budget_droppable_tail_turn(&assistant("derived chatter")));
        assert!(!is_budget_droppable_tail_turn(&assistant(&format!(
            "{PRESERVED_TODO_PREFIX}\n- [ ] task"
        ))));
        assert!(!is_budget_droppable_tail_turn(&assistant(&format!(
            "{PRESERVED_OBJECTIVE_PREFIX}\nthe objective"
        ))));
    }

    #[test]
    fn raw_backlog_debt_records_and_clears_with_the_backlog() {
        let store = Store::open_in_memory().unwrap();
        let tok = Tokenizer::heuristic();
        let mut c = cfg();
        c.deferred_maintenance_enabled = true;
        c.fresh_tail_count = 2;
        c.leaf_chunk_tokens = 10; // tiny threshold so the backlog is actionable
        store.bind_session("s1", "s1", 1.0).unwrap();
        let backlog: Vec<Turn> = (0..6)
            .map(|i| user(&format!("turn {i} {}", "x ".repeat(30))))
            .collect();

        // Actionable backlog outside the tail records debt.
        refresh_raw_backlog_debt(&store, &tok, &c, "s1", &backlog, false, 2.0);
        assert!(has_raw_backlog_debt(&store, &c, "s1"));
        assert!(should_run_deferred_maintenance(
            &store, &tok, &c, "s1", &backlog, false
        ));

        // Once everything fits the fresh tail the debt clears.
        let tail_only: Vec<Turn> = backlog[..2].to_vec();
        refresh_raw_backlog_debt(&store, &tok, &c, "s1", &tail_only, false, 3.0);
        assert!(!has_raw_backlog_debt(&store, &c, "s1"));
        assert!(!should_run_deferred_maintenance(
            &store, &tok, &c, "s1", &backlog, false
        ));

        // Disabled feature: no writes at all.
        c.deferred_maintenance_enabled = false;
        refresh_raw_backlog_debt(&store, &tok, &c, "s1", &backlog, false, 4.0);
        assert!(!has_raw_backlog_debt(&store, &c, "s1"));
    }
}
