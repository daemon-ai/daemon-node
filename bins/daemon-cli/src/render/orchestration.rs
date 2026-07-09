// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Orchestration-surface responses: fleet roster, the unit tree (and node lines), unit events,
//! the signed journal, the log page, delivery targets/sessions, and the journal verifying key.

use daemon_api::{ApiResponse, JournalRecordPayload};
use daemon_common::UnitId;

pub(super) fn try_render(resp: ApiResponse) -> Option<ApiResponse> {
    match resp {
        ApiResponse::Fleet(f) => {
            println!(
                "fleet: children={} usage(in={} out={} api_calls={})",
                f.children.len(),
                f.usage.input_tokens,
                f.usage.output_tokens,
                f.usage.api_calls
            );
            for c in f.children {
                println!("  - {c}");
            }
        }
        ApiResponse::Tree(t) => {
            println!(
                "tree: root={} nodes={}",
                t.root
                    .as_ref()
                    .map(|r| r.to_string())
                    .unwrap_or_else(|| "-".into()),
                t.nodes.len()
            );
            render_tree(&t);
        }
        ApiResponse::Unit(unit) => match unit {
            Some(n) => render_unit_node(&n),
            None => println!("unit: not found"),
        },
        ApiResponse::UnitEvents(events) => {
            println!("unit events: {}", events.len());
            for e in events {
                println!("  - {e:?}");
            }
        }
        ApiResponse::Journal(page) => {
            println!(
                "history: {} entr(ies) next_cursor={} head_cursor={}",
                page.entries.len(),
                page.next_cursor,
                page.head_cursor
            );
            for r in page.entries {
                let mark = if r.verified { "verified" } else { "unsealed" };
                match r.payload {
                    JournalRecordPayload::Management { detail } => println!(
                        "  - [{}] cur={} seg={} seq={} {} mgmt: {}",
                        mark, r.cursor, r.segment, r.seq, r.kind, detail
                    ),
                    JournalRecordPayload::Block { block } => println!(
                        "  - [{}] cur={} seg={} seq={} {} block: {:?}",
                        mark, r.cursor, r.segment, r.seq, r.kind, block
                    ),
                    // wire v37 (W2-E): the richer chat-message history entry.
                    JournalRecordPayload::Chat { message } => {
                        let author = match &message.author {
                            Some(daemon_api::Participant::Contact(c)) => {
                                c.display_name.clone().unwrap_or_else(|| c.id.clone())
                            }
                            Some(daemon_api::Participant::Agent { member, .. }) => member.clone(),
                            None => "-".into(),
                        };
                        let mut flags = Vec::new();
                        if message.delivered() {
                            flags.push("delivered");
                        }
                        if message.edited() {
                            flags.push("edited");
                        }
                        if !message.attachments.is_empty() {
                            flags.push("attachments");
                        }
                        let flags = if flags.is_empty() {
                            String::new()
                        } else {
                            format!(" [{}]", flags.join(","))
                        };
                        println!(
                            "  - [{}] cur={} seg={} seq={} {} chat: <{}> {}{}",
                            mark, r.cursor, r.segment, r.seq, r.kind, author, message.text, flags
                        );
                    }
                }
            }
        }
        ApiResponse::LogPage(page) => {
            println!(
                "log: {} entr(ies) next_seq={} head_seq={}",
                page.entries.len(),
                page.next_seq,
                page.head_seq
            );
            for e in page.entries {
                println!(
                    "  - seq={} {:?} {} {:?}",
                    e.seq, e.direction, e.origin.transport.0, e.payload
                );
            }
        }
        ApiResponse::DeliveryTargets(targets) => {
            println!("delivery_targets: {}", targets.len());
            for t in targets {
                println!("  - {} {} {:?}", t.transport.0, t.route.0, t.kind);
            }
        }
        ApiResponse::DeliverySessions(page) => {
            println!("delivery_sessions: {}", page.items.len());
            for s in page.items {
                println!("  - {s}");
            }
            if let Some(next) = page.next {
                println!("  next={next}");
            }
        }
        ApiResponse::VerifyingKey(key) => match key {
            Some(hex) => println!("verifying_key: {hex}"),
            None => println!("verifying_key: none (node exposes no journal signer)"),
        },
        other => return Some(other),
    }
    None
}

/// Render the orchestration tree with depth: a DFS from `root`, indenting each level, so the
/// GUI/TUI's nested fleets-of-fleets read as a tree rather than a flat roster. Any node not
/// reachable from the root (there should be none) is printed flat afterwards, so nothing is hidden.
fn render_tree(t: &daemon_api::TreeReport) {
    use std::collections::{HashMap, HashSet};
    let index: HashMap<&UnitId, &daemon_api::UnitNode> =
        t.nodes.iter().map(|n| (&n.id, n)).collect();
    let mut seen: HashSet<UnitId> = HashSet::new();
    if let Some(root) = &t.root {
        render_tree_node(root, &index, 0, &mut seen);
    }
    for n in &t.nodes {
        if !seen.contains(&n.id) {
            render_unit_node_at(n, 0);
            seen.insert(n.id.clone());
        }
    }
}

/// Render `id`'s node indented at `depth`, then recurse into its children (cycle-guarded).
fn render_tree_node(
    id: &UnitId,
    index: &std::collections::HashMap<&UnitId, &daemon_api::UnitNode>,
    depth: usize,
    seen: &mut std::collections::HashSet<UnitId>,
) {
    if !seen.insert(id.clone()) {
        return;
    }
    match index.get(id) {
        Some(n) => {
            render_unit_node_at(n, depth);
            for child in &n.children {
                render_tree_node(child, index, depth + 1, seen);
            }
        }
        None => println!("{}- {} (not projected)", "  ".repeat(depth + 1), id),
    }
}

/// Render one tree node line (id, kind, state, work, usage), indented by tree `depth`.
fn render_unit_node(n: &daemon_api::UnitNode) {
    render_unit_node_at(n, 0);
}

/// Render one tree node line indented by `depth` levels.
fn render_unit_node_at(n: &daemon_api::UnitNode, depth: usize) {
    println!(
        "{}- {} kind={:?} state={:?} work={} usage(in={} out={} api_calls={}) children={}",
        "  ".repeat(depth + 1),
        n.id,
        n.kind,
        n.state,
        n.work.as_deref().unwrap_or("-"),
        n.usage.input_tokens,
        n.usage.output_tokens,
        n.usage.api_calls,
        n.children.len()
    );
}
