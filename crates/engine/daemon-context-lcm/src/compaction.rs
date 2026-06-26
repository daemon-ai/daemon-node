//! The compaction engine (`daemon-context-lcm-port-spec.md` §6).
//!
//! Per `compact()`: select the region between the leading anchor and the fresh tail, summarize the
//! oldest leaf chunk into a **D0** node (advancing the lifecycle frontier), opportunistically
//! condense sibling nodes up the DAG (fanin >= 4 -> D1/D2/D3), and reassemble the body as
//! `[system] + [synthetic summary turn over the DAG frontier] + [fresh tail]`. Operating on whole
//! [`Turn`]s keeps tool-call/result pairs intact by construction (§6.7).
//!
//! Milestone scope: the always-on default path only — single leaf chunk, single condensation group
//! per depth, no dynamic chunking / deferred-debt / critical-pressure (all opt-in, default off).

use crate::config::LcmConfig;
use crate::escalation::{summarize_with_escalation, SummaryCircuitBreaker, SummaryRequest};
use crate::externalize::{extract_ref, gc_placeholder};
use crate::extraction;
use crate::ingest::render_turns;
use crate::store::{NewNode, SourceType, Store};
use crate::tokens::Tokenizer;
use daemon_core::conversation::AssistantMsg;
use daemon_core::{Conversation, Provider, Turn};
use std::sync::Arc;
use std::time::Duration;

/// The leading marker on a synthetic summary turn (so re-compaction skips it as scaffold — §6.4).
pub(crate) const SUMMARY_SENTINEL: &str = "[LCM context summary]";

/// Run one compaction pass. Returns the new conversation, whether anything was actually compacted
/// (`false` => nothing eligible outside the fresh tail; the caller treats it as a no-op), and the
/// rebuilt per-turn ingest index aligned with the returned conversation.
///
/// `turn_store_ids[i]` holds the `messages.store_id`s already persisted for `conv.turns[i]` (by the
/// caller's `before_turn` ingest), so the D0 node's `source_ids` come from the index rather than a
/// re-ingest (avoiding duplicate rows).
#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_compaction(
    store: &Store,
    tok: &Tokenizer,
    cfg: &LcmConfig,
    aux_chain: &[Arc<dyn Provider>],
    breakers: &mut [SummaryCircuitBreaker],
    session_id: &str,
    first_compaction: bool,
    turn_store_ids: Vec<Vec<i64>>,
    conv: Conversation,
    now: f64,
) -> (Conversation, bool, Vec<Vec<i64>>) {
    let Conversation { system, turns } = conv;

    // Region = turns[anchor .. fresh_tail_start). The anchor skips any leading synthetic summary
    // turns from a previous compaction so we never re-summarize a summary (§6.4).
    let anchor = leading_scaffold_count(&turns);
    let fresh_tail_start = turns.len().saturating_sub(cfg.fresh_tail_count);
    if fresh_tail_start <= anchor {
        // Everything outside the fresh tail is already summarized scaffold — nothing to do.
        return (Conversation { system, turns }, false, turn_store_ids);
    }
    let region = &turns[anchor..fresh_tail_start];
    let source_tokens: usize = region.iter().map(|t| tok.count_turn(t)).sum();
    if source_tokens == 0 {
        return (Conversation { system, turns }, false, turn_store_ids);
    }

    // The region's `store_id`s come from the precomputed ingest index (persisted by `before_turn`).
    let store_ids: Vec<i64> = turn_store_ids
        .get(anchor..fresh_tail_start)
        .map(|groups| groups.iter().flatten().copied().collect())
        .unwrap_or_default();

    // Summarize the leaf chunk: budget = max(2000, 20% of source) capped at 12000 (§6.5).
    let leaf_budget = (source_tokens * 20 / 100).clamp(2000, 12000);
    let text = render_turns(region);

    // §9.2 pre-compaction extraction (opt-in): distill durable decisions to the daily markdown
    // before this region is summarized. Best-effort — it never blocks or fails compaction. Routed
    // through the primary aux provider.
    if cfg.extraction_enabled {
        if let Some(primary) = aux_chain.first() {
            let _ = extraction::run_extraction(
                primary.as_ref(),
                &text,
                cfg.extraction_dir().as_deref(),
                Duration::from_millis(cfg.summary_timeout_ms),
                now,
            )
            .await;
        }
    }

    let (summary, _level) = summarize_with_escalation(
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
            focus_topic: "",
            custom_instructions: "",
        },
    )
    .await;

    let d0 = NewNode {
        session_id: session_id.to_string(),
        depth: 0,
        summary: summary.clone(),
        token_count: tok.count_text(&summary) as i64,
        source_token_count: source_tokens as i64,
        source_ids: store_ids.clone(),
        source_type: SourceType::Messages,
        created_at: now,
        earliest_at: Some(now),
        latest_at: Some(now),
        expand_hint: expand_hint(&summary),
    };
    if let Err(e) = store.add_node(&d0) {
        tracing::warn!(error = %e, "lcm: failed to persist D0 node; skipping compaction");
        return (Conversation { system, turns }, false, turn_store_ids);
    }
    if let Some(max_id) = store_ids.iter().copied().max() {
        let _ = store.advance_frontier(session_id, max_id, now);
        // §9.1 transcript GC (opt-in): now that this region is summarized, rewrite its
        // already-externalized payload rows to a compact GC placeholder (store_id-preserving).
        if cfg.large_output_transcript_gc_enabled {
            gc_externalized_payloads(store, session_id, max_id);
        }
    }

    // Condensation: climb depths while a level has >= fanin uncondensed siblings (§6.6).
    condense(store, tok, cfg, aux_chain, breakers, session_id, now).await;

    // Assemble: [system] + [summary turn over the DAG frontier] + [fresh tail], rebuilding the
    // ingest index in lockstep (the synthetic summary turn holds no store_ids; the tail keeps its).
    let summary_turn = assemble_summary_turn(store, session_id, first_compaction);
    let tail_index: Vec<Vec<i64>> = turn_store_ids.into_iter().skip(fresh_tail_start).collect();
    let mut new_turns = Vec::with_capacity(1 + turns.len() - fresh_tail_start);
    let mut new_index = Vec::with_capacity(1 + tail_index.len());
    if let Some(t) = summary_turn {
        new_turns.push(t);
        new_index.push(Vec::new());
    }
    new_turns.extend(turns.into_iter().skip(fresh_tail_start));
    new_index.extend(tail_index);

    (
        Conversation {
            system,
            turns: new_turns,
        },
        true,
        new_index,
    )
}

/// Rewrite summarized rows that still carry an inline externalized-payload placeholder to a compact
/// GC placeholder (§9.1 transcript GC). The bytes already live on disk under the externalization
/// dir; this only shrinks the `messages` row (and its FTS shadow) — recovery via the `ref` is
/// unchanged. Best-effort: a failed row is left as-is.
fn gc_externalized_payloads(store: &Store, session_id: &str, max_store_id: i64) {
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
        let is_tool_output = row.tool_call_id.is_some();
        let placeholder = gc_placeholder(is_tool_output, &reference);
        if let Err(e) = store.update_message_content(row.store_id, &placeholder) {
            tracing::warn!(error = %e, store_id = row.store_id, "lcm: transcript-GC rewrite failed");
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

/// Climb the DAG, condensing each depth that has accumulated >= `fanin` uncondensed siblings into a
/// single node at the next depth. One group per depth per call (the cache-friendly default).
#[allow(clippy::too_many_arguments)]
async fn condense(
    store: &Store,
    tok: &Tokenizer,
    cfg: &LcmConfig,
    aux_chain: &[Arc<dyn Provider>],
    breakers: &mut [SummaryCircuitBreaker],
    session_id: &str,
    now: f64,
) {
    let fanin = cfg.condensation_fanin.max(1);
    let mut depth = 0i64;
    loop {
        // Respect the max-depth knob (0 disables condensation; -1 is unlimited).
        if cfg.incremental_max_depth == 0 {
            break;
        }
        if cfg.incremental_max_depth > 0 && depth >= cfg.incremental_max_depth {
            break;
        }
        let feeders = match store.get_uncondensed_at_depth(session_id, depth, fanin as i64) {
            Ok(f) => f,
            Err(e) => {
                tracing::warn!(error = %e, depth, "lcm: condensation feeder query failed");
                break;
            }
        };
        if feeders.len() < fanin {
            break;
        }
        let child_ids: Vec<i64> = feeders.iter().map(|n| n.node_id).collect();
        let source_tokens: usize = feeders.iter().map(|n| n.token_count.max(0) as usize).sum();
        let text = feeders
            .iter()
            .map(|n| n.summary.as_str())
            .collect::<Vec<_>>()
            .join("\n\n");
        // Condense budget = max(1000, 40% of source) (§6.6).
        let budget = (source_tokens * 40 / 100).max(1000);
        let (summary, _level) = summarize_with_escalation(
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
                focus_topic: "",
                custom_instructions: "",
            },
        )
        .await;
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
        depth += 1;
    }
}

/// Build the synthetic summary turn from the DAG frontier (highest depth first), or `None` if the
/// frontier is empty. On the first compaction a one-line "recoverable via tools" note is prepended.
fn assemble_summary_turn(store: &Store, session_id: &str, first_compaction: bool) -> Option<Turn> {
    let frontier = store
        .get_uncondensed_frontier(session_id)
        .unwrap_or_default();
    if frontier.is_empty() {
        return None;
    }
    let mut body = String::from(SUMMARY_SENTINEL);
    if first_compaction {
        body.push_str(
            "\n(Older turns were compacted into the summaries below; full detail is recoverable via the lcm_* tools.)",
        );
    }
    for node in &frontier {
        body.push_str("\n\n");
        body.push_str(depth_label(node.depth));
        body.push_str(":\n");
        body.push_str(&node.summary);
    }
    Some(Turn::Assistant(AssistantMsg::text(body)))
}

/// A human label for a DAG depth used in the assembled summary block.
fn depth_label(depth: i64) -> &'static str {
    match depth {
        0 => "Recent summary",
        1 => "Session arc",
        2 => "Durable narrative",
        _ => "Deep history",
    }
}

/// Derive a short expand hint from a summary's trailing "Expand for details about: ..." line, if
/// the model included one; else empty.
fn expand_hint(summary: &str) -> String {
    summary
        .lines()
        .rev()
        .find(|l| l.to_ascii_lowercase().contains("expand for details about"))
        .map(|l| l.trim().to_string())
        .unwrap_or_default()
}

fn min_opt(acc: Option<f64>, v: f64) -> Option<f64> {
    Some(acc.map_or(v, |a| a.min(v)))
}

fn max_opt(acc: Option<f64>, v: f64) -> Option<f64> {
    Some(acc.map_or(v, |a| a.max(v)))
}
