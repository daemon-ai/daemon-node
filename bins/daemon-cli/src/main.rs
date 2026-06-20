//! `daemon-cli` — the operator surface over the node's [`daemon_api`] interface.
//!
//! A thin client: every subcommand marshals one [`daemon_api::ApiRequest`] over the Unix-socket
//! transport ([`daemon_host::ApiClient`]) and renders the [`daemon_api::ApiResponse`]. It reaches
//! the *same* surface the in-process caller and the C FFI reach — only the transport differs.

#![forbid(unsafe_code)]

use std::path::PathBuf;

use clap::{Parser, Subcommand};
use daemon_api::{ApiRequest, ApiResponse, JournalRecordPayload};
use daemon_common::{ReqId, SessionId, UnitId};
use daemon_host::ApiClient;
use daemon_protocol::{AgentCommand, UserMsg};

/// Operate a running `daemon` host node over its api socket.
#[derive(Parser)]
#[command(name = "daemon-cli", version, about)]
struct Cli {
    /// Path to the node's api socket (defaults to `$DAEMON_API_SOCKET` or `$TMPDIR/daemon-api.sock`).
    #[arg(long, global = true)]
    socket: Option<PathBuf>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Print resident-service health and durable stats together.
    Status,
    /// Resident-service tree health.
    Health,
    /// Durable queue depths and session/active counts.
    Stats,
    /// List durable sessions and their states.
    Sessions,
    /// Create-if-absent and wake a durable session.
    Assign {
        /// The session id.
        id: String,
    },
    /// Cancel in-flight work for a session.
    Cancel {
        /// The session id.
        id: String,
    },
    /// The orchestration fleet roster + folded usage.
    Fleet,
    /// The orchestration tree the GUI/TUI drives (unit structure, state, work, usage).
    Tree,
    /// One unit's node view.
    Unit {
        /// The unit id.
        id: String,
    },
    /// Drain recent management events for a unit (drill-down).
    UnitEvents {
        /// The unit id.
        id: String,
        /// Maximum events to drain (0 = all buffered).
        #[arg(long, default_value_t = 0)]
        max: u32,
    },
    /// Drain the rich §17 outbound stream for a unit (transcript-fidelity drill-down).
    UnitOutbound {
        /// The unit id.
        id: String,
        /// Maximum items to drain (0 = all buffered).
        #[arg(long, default_value_t = 0)]
        max: u32,
    },
    /// Pause a unit's scheduling (orchestrator sub-fleets).
    Pause {
        /// The unit id.
        id: String,
    },
    /// Resume a unit's scheduling.
    Resume {
        /// The unit id.
        id: String,
    },
    /// Scale a unit (sub-fleet) to N members.
    Scale {
        /// The unit id.
        id: String,
        /// The target member count.
        n: u32,
    },
    /// Open/continue an interactive session by starting a turn.
    Submit {
        /// The session id.
        id: String,
        /// The user message text.
        #[arg(default_value = "hello")]
        text: String,
    },
    /// Drain outbound events/requests from an interactive session.
    Poll {
        /// The session id.
        id: String,
        /// Maximum items to drain (0 = all available).
        #[arg(long, default_value_t = 0)]
        max: u32,
    },
    /// Read a session's durable verifiable history (non-destructive scroll-back).
    History {
        /// The session id.
        id: String,
        /// Return entries with cursor strictly greater than this (0 from the start).
        #[arg(long, default_value_t = 0)]
        after: u64,
        /// Maximum entries to return (0 = all available).
        #[arg(long, default_value_t = 0)]
        max: u32,
    },
    /// Read any unit's durable verifiable history (non-destructive scroll-back).
    UnitHistory {
        /// The unit id.
        id: String,
        /// Return entries with cursor strictly greater than this (0 from the start).
        #[arg(long, default_value_t = 0)]
        after: u64,
        /// Maximum entries to return (0 = all available).
        #[arg(long, default_value_t = 0)]
        max: u32,
    },
    /// Print the node's journal verifying key (hex dCBOR) for offline audit.
    VerifyingKey,
}

fn default_socket() -> PathBuf {
    if let Some(p) = std::env::var_os("DAEMON_API_SOCKET") {
        return PathBuf::from(p);
    }
    let dir = std::env::var_os("TMPDIR").unwrap_or_else(|| "/tmp".into());
    PathBuf::from(dir).join("daemon-api.sock")
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let socket = cli.socket.clone().unwrap_or_else(default_socket);
    let client = ApiClient::new(socket);

    match cli.command {
        Command::Status => {
            render(client.call(ApiRequest::Health).await?);
            render(client.call(ApiRequest::Stats).await?);
        }
        Command::Health => render(client.call(ApiRequest::Health).await?),
        Command::Stats => render(client.call(ApiRequest::Stats).await?),
        Command::Sessions => render(client.call(ApiRequest::Sessions).await?),
        Command::Assign { id } => render(
            client
                .call(ApiRequest::Assign {
                    session: SessionId::new(id),
                })
                .await?,
        ),
        Command::Cancel { id } => render(
            client
                .call(ApiRequest::Cancel {
                    session: SessionId::new(id),
                })
                .await?,
        ),
        Command::Fleet => render(client.call(ApiRequest::Fleet).await?),
        Command::Tree => render(client.call(ApiRequest::Tree).await?),
        Command::Unit { id } => render(
            client
                .call(ApiRequest::Unit {
                    unit: UnitId::new(id),
                })
                .await?,
        ),
        Command::UnitOutbound { id, max } => render(
            client
                .call(ApiRequest::UnitOutbound {
                    unit: UnitId::new(id),
                    max,
                })
                .await?,
        ),
        Command::UnitEvents { id, max } => render(
            client
                .call(ApiRequest::UnitEvents {
                    unit: UnitId::new(id),
                    max,
                })
                .await?,
        ),
        Command::Pause { id } => render(
            client
                .call(ApiRequest::Pause {
                    unit: UnitId::new(id),
                })
                .await?,
        ),
        Command::Resume { id } => render(
            client
                .call(ApiRequest::Resume {
                    unit: UnitId::new(id),
                })
                .await?,
        ),
        Command::Scale { id, n } => render(
            client
                .call(ApiRequest::Scale {
                    unit: UnitId::new(id),
                    n,
                })
                .await?,
        ),
        Command::Submit { id, text } => render(
            client
                .call(ApiRequest::Submit {
                    session: SessionId::new(id),
                    command: AgentCommand::StartTurn {
                        input: UserMsg::new(text),
                        request_id: ReqId(1),
                    },
                })
                .await?,
        ),
        Command::Poll { id, max } => render(
            client
                .call(ApiRequest::Poll {
                    session: SessionId::new(id),
                    max,
                })
                .await?,
        ),
        Command::History { id, after, max } => render(
            client
                .call(ApiRequest::SessionHistory {
                    session: SessionId::new(id),
                    after_cursor: after,
                    max,
                })
                .await?,
        ),
        Command::UnitHistory { id, after, max } => render(
            client
                .call(ApiRequest::UnitHistory {
                    unit: UnitId::new(id),
                    after_cursor: after,
                    max,
                })
                .await?,
        ),
        Command::VerifyingKey => render(client.call(ApiRequest::VerifyingKey).await?),
    }
    Ok(())
}

/// Render an api response in a compact, operator-readable form.
fn render(resp: ApiResponse) {
    match resp {
        ApiResponse::Ok => println!("ok"),
        ApiResponse::Health(h) => {
            println!("health: all_ok={}", h.all_ok);
            for s in h.services {
                let detail = s.detail.map(|d| format!(" ({d})")).unwrap_or_default();
                println!("  - {} ok={} restarts={}{}", s.name, s.ok, s.restarts, detail);
            }
        }
        ApiResponse::Stats(s) => println!(
            "stats: jobs={} wakes={} sessions={} active={}",
            s.pending_jobs, s.pending_wakes, s.sessions, s.active
        ),
        ApiResponse::Sessions(list) => {
            println!("sessions: {}", list.len());
            for info in list {
                println!("  - {} {:?}", info.session, info.state);
            }
        }
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
        ApiResponse::Drained(items) => {
            println!("drained: {} item(s)", items.len());
            for item in items {
                println!("  - {item:?}");
            }
        }
        ApiResponse::Tree(t) => {
            println!(
                "tree: root={} nodes={}",
                t.root.as_ref().map(|r| r.to_string()).unwrap_or_else(|| "-".into()),
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
                }
            }
        }
        ApiResponse::VerifyingKey(key) => match key {
            Some(hex) => println!("verifying_key: {hex}"),
            None => println!("verifying_key: none (node exposes no journal signer)"),
        },
        ApiResponse::Error(e) => println!("error: {e}"),
    }
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
