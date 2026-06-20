//! `daemon-cli` — the operator surface over the node's [`daemon_api`] interface.
//!
//! A thin client: every subcommand marshals one [`daemon_api::ApiRequest`] over the Unix-socket
//! transport ([`daemon_host::ApiClient`]) and renders the [`daemon_api::ApiResponse`]. It reaches
//! the *same* surface the in-process caller and the C FFI reach — only the transport differs.

#![forbid(unsafe_code)]

use std::path::PathBuf;

use clap::{Parser, Subcommand};
use daemon_api::{ApiRequest, ApiResponse};
use daemon_common::{ReqId, SessionId};
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
        ApiResponse::Error(e) => println!("error: {e}"),
    }
}
