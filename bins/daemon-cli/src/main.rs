//! `daemon-cli` — the operator surface over the node's [`daemon_api`] interface.
//!
//! A thin client: every subcommand marshals one [`daemon_api::ApiRequest`] over the Unix-socket
//! transport ([`daemon_host::ApiClient`]) and renders the [`daemon_api::ApiResponse`]. It reaches
//! the *same* surface the in-process caller and the C FFI reach — only the transport differs.

#![forbid(unsafe_code)]

use std::path::PathBuf;

use clap::{Parser, Subcommand};
use daemon_api::{ApiRequest, ApiResponse, JournalRecordPayload};
use daemon_common::{
    DownloadId, ModelEngine, ModelId, ModelRef, ModelSource, ReqId, SearchQuery, SearchSort,
    SessionId, UnitId,
};
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
    /// Telemetry dump: folded usage/cost + event count + health + queue depths.
    Telemetry,
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
    /// Manage local-inference models (search/download/cache/catalog).
    Model {
        #[command(subcommand)]
        cmd: ModelCmd,
    },
}

/// Parse an engine selector, defaulting to llama.
fn parse_engine(s: &str) -> anyhow::Result<ModelEngine> {
    ModelEngine::parse(s)
        .ok_or_else(|| anyhow::anyhow!("unknown engine {s:?} (expected llama|mistralrs)"))
}

#[derive(Subcommand)]
enum ModelCmd {
    /// Search Hugging Face for models loadable by the given engine (step 1).
    Search {
        /// The free-text query.
        query: String,
        /// The target engine (llama|mistralrs).
        #[arg(long, default_value = "llama")]
        engine: String,
        /// Results per page.
        #[arg(long, default_value_t = 25)]
        limit: u32,
        /// The 0-based page.
        #[arg(long, default_value_t = 0)]
        page: u32,
    },
    /// List a repo's loadable files for the given engine (step 2).
    Files {
        /// The `org/name` repo id.
        repo: String,
        /// The git revision (defaults to `main`).
        #[arg(long)]
        revision: Option<String>,
        /// The target engine (llama|mistralrs).
        #[arg(long, default_value = "llama")]
        engine: String,
    },
    /// Download a model into the shared cache.
    Pull {
        /// The `org/name` repo id.
        repo: String,
        /// The GGUF file to fetch (required for llama; the first shard pulls the whole split set).
        #[arg(long)]
        file: Option<String>,
        /// The git revision (defaults to `main`).
        #[arg(long)]
        revision: Option<String>,
        /// The target engine (llama|mistralrs).
        #[arg(long, default_value = "llama")]
        engine: String,
    },
    /// List in-flight + finished download jobs.
    Downloads,
    /// Cancel a download job.
    Cancel {
        /// The download id (numeric).
        id: u64,
    },
    /// Pause a download job.
    Pause {
        /// The download id (numeric).
        id: u64,
    },
    /// Resume a paused/failed download job.
    Resume {
        /// The download id (numeric).
        id: u64,
    },
    /// List installed (cataloged) models.
    Ls,
    /// Delete an installed model (catalog record + cached artifact).
    Rm {
        /// The catalog id.
        id: String,
    },
    /// Activate a cataloged model so new worker spawns load it.
    Activate {
        /// The catalog id.
        id: String,
        /// The profile to activate it for (defaults to the node's default local profile).
        #[arg(long)]
        profile: Option<String>,
    },
    /// Recommend a quantization for a repo on the detected hardware.
    Recommend {
        /// The `org/name` repo id.
        repo: String,
        /// The target engine (llama|mistralrs).
        #[arg(long, default_value = "llama")]
        engine: String,
        /// The git revision (defaults to `main`).
        #[arg(long)]
        revision: Option<String>,
        /// Override the memory budget, in GiB (defaults to auto-detected VRAM/RAM).
        #[arg(long)]
        vram: Option<f64>,
    },
    /// Quantize a repo's GGUF to a target precision offline (via the llama-enabled worker).
    Quantize {
        /// The `org/name` repo id.
        repo: String,
        /// The target quant label (e.g. Q4_K_M).
        #[arg(long)]
        ftype: String,
        /// The source GGUF file (defaults to the highest-precision one in the repo).
        #[arg(long)]
        source: Option<String>,
        /// The git revision (defaults to `main`).
        #[arg(long)]
        revision: Option<String>,
    },
    /// List in-flight + finished quantization jobs.
    Quantizes,
    /// Inspect a cataloged model's GGUF metadata.
    Inspect {
        /// The catalog id.
        id: String,
    },
    /// Quickstart: recommend a quant, download it, and activate it — one command to get running.
    Up {
        /// The `org/name` repo id.
        repo: String,
        /// The target engine (llama|mistralrs).
        #[arg(long, default_value = "llama")]
        engine: String,
        /// Override the memory budget, in GiB (defaults to auto-detected VRAM/RAM).
        #[arg(long)]
        vram: Option<f64>,
        /// The profile to activate for (defaults to the node's default local profile).
        #[arg(long)]
        profile: Option<String>,
    },
}

/// Build a [`ModelRef`] from CLI repo/file/revision/engine args.
fn build_model_ref(
    engine: ModelEngine,
    repo: String,
    file: Option<String>,
    revision: Option<String>,
) -> ModelRef {
    let revision = revision.unwrap_or_else(|| "main".to_string());
    ModelRef::new(
        engine,
        ModelSource::Hf {
            repo,
            file,
            revision,
        },
    )
}

/// GiB → bytes.
fn gib_to_bytes(gib: f64) -> u64 {
    (gib * 1024.0 * 1024.0 * 1024.0) as u64
}

/// Dispatch a `model` subcommand against the node.
async fn run_model(client: &ApiClient, cmd: ModelCmd) -> anyhow::Result<()> {
    // The quickstart orchestrates several calls (recommend → pull → activate), so it is handled
    // out-of-band rather than as a single request mapping.
    if let ModelCmd::Up {
        repo,
        engine,
        vram,
        profile,
    } = cmd
    {
        return quickstart_up(client, repo, parse_engine(&engine)?, vram, profile).await;
    }

    let req = match cmd {
        ModelCmd::Search {
            query,
            engine,
            limit,
            page,
        } => ApiRequest::ModelSearch {
            query: SearchQuery {
                text: query,
                engine: parse_engine(&engine)?,
                sort: SearchSort::default(),
                page,
                limit,
            },
        },
        ModelCmd::Files {
            repo,
            revision,
            engine,
        } => ApiRequest::ModelFiles {
            repo,
            revision,
            engine: parse_engine(&engine)?,
        },
        ModelCmd::Pull {
            repo,
            file,
            revision,
            engine,
        } => ApiRequest::ModelDownload {
            model: build_model_ref(parse_engine(&engine)?, repo, file, revision),
        },
        ModelCmd::Downloads => ApiRequest::ModelDownloads,
        ModelCmd::Cancel { id } => ApiRequest::ModelCancel { id: DownloadId(id) },
        ModelCmd::Pause { id } => ApiRequest::ModelPause { id: DownloadId(id) },
        ModelCmd::Resume { id } => ApiRequest::ModelResume { id: DownloadId(id) },
        ModelCmd::Ls => ApiRequest::ModelCatalog,
        ModelCmd::Rm { id } => ApiRequest::ModelDelete {
            id: ModelId::new(id),
        },
        ModelCmd::Activate { id, profile } => ApiRequest::ModelActivate {
            id: ModelId::new(id),
            profile,
        },
        ModelCmd::Recommend {
            repo,
            engine,
            revision,
            vram,
        } => ApiRequest::ModelRecommend {
            repo,
            revision,
            engine: parse_engine(&engine)?,
            budget_bytes: vram.map(gib_to_bytes),
        },
        ModelCmd::Quantize {
            repo,
            ftype,
            source,
            revision,
        } => ApiRequest::ModelQuantize {
            repo,
            revision,
            target_quant: ftype,
            source_file: source,
        },
        ModelCmd::Quantizes => ApiRequest::ModelQuantizes,
        ModelCmd::Inspect { id } => ApiRequest::ModelInspect {
            id: ModelId::new(id),
        },
        ModelCmd::Up { .. } => unreachable!("handled above"),
    };
    render(client.call(req).await?);
    Ok(())
}

/// The `model up` quickstart: recommend a quant for the hardware, download it, then activate it.
async fn quickstart_up(
    client: &ApiClient,
    repo: String,
    engine: ModelEngine,
    vram: Option<f64>,
    profile: Option<String>,
) -> anyhow::Result<()> {
    // 1. Recommend a quant for the detected (or overridden) budget.
    let rec = match client
        .call(ApiRequest::ModelRecommend {
            repo: repo.clone(),
            revision: None,
            engine,
            budget_bytes: vram.map(gib_to_bytes),
        })
        .await?
    {
        ApiResponse::ModelRecommend(rec) => rec,
        ApiResponse::Error(e) => anyhow::bail!("recommend failed: {e}"),
        other => anyhow::bail!("unexpected response to recommend: {other:?}"),
    };
    println!(
        "recommend: {} {} (fits={}) — {}",
        rec.repo, rec.quant, rec.fits, rec.reason
    );

    // 2. Build the model ref and start the download (llama needs the recommended file).
    let model = ModelRef::new(
        engine,
        ModelSource::Hf {
            repo: repo.clone(),
            file: rec.file.clone(),
            revision: "main".to_string(),
        },
    );
    if matches!(engine, ModelEngine::Llama) && rec.file.is_none() {
        anyhow::bail!(
            "no downloadable GGUF recommended for {repo}: {}",
            rec.reason
        );
    }
    let job = match client
        .call(ApiRequest::ModelDownload {
            model: model.clone(),
        })
        .await?
    {
        ApiResponse::ModelDownloadStarted(id) => id,
        ApiResponse::Error(e) => anyhow::bail!("download failed to start: {e}"),
        other => anyhow::bail!("unexpected response to download: {other:?}"),
    };
    println!("download: started {job}; waiting for completion…");

    // 3. Poll until the job reaches a terminal state.
    loop {
        tokio::time::sleep(std::time::Duration::from_millis(700)).await;
        let jobs = match client.call(ApiRequest::ModelDownloads).await? {
            ApiResponse::ModelDownloads(jobs) => jobs,
            other => anyhow::bail!("unexpected response to downloads: {other:?}"),
        };
        let Some(status) = jobs.into_iter().find(|j| j.id == job) else {
            continue;
        };
        use daemon_common::DownloadState::*;
        match status.state {
            Completed => {
                println!("download: complete");
                break;
            }
            Failed | Cancelled => {
                anyhow::bail!(
                    "download {job} ended {:?}: {}",
                    status.state,
                    status.error.unwrap_or_default()
                );
            }
            _ => {
                let pct = if status.total_bytes > 0 {
                    (status.downloaded_bytes as f64 / status.total_bytes as f64 * 100.0) as u64
                } else {
                    0
                };
                println!(
                    "  … {pct}% ({}/{} bytes)",
                    status.downloaded_bytes, status.total_bytes
                );
            }
        }
    }

    // 4. Find the cataloged record for this model and activate it.
    let catalog = match client.call(ApiRequest::ModelCatalog).await? {
        ApiResponse::ModelCatalog(models) => models,
        other => anyhow::bail!("unexpected response to catalog: {other:?}"),
    };
    let record = catalog
        .into_iter()
        .find(|m| m.model == model)
        .ok_or_else(|| anyhow::anyhow!("downloaded model not found in catalog"))?;
    println!("activate: {} ({})", record.id, record.display_name);
    render(
        client
            .call(ApiRequest::ModelActivate {
                id: record.id,
                profile,
            })
            .await?,
    );
    Ok(())
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
        Command::Telemetry => render(client.call(ApiRequest::Telemetry).await?),
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
                    origin: None,
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
        Command::Model { cmd } => run_model(&client, cmd).await?,
    }
    Ok(())
}

/// Render an api response in a compact, operator-readable form.
fn render(resp: ApiResponse) {
    match resp {
        ApiResponse::Ok => println!("ok"),
        ApiResponse::Routed { session } => println!("routed: session={session}"),
        ApiResponse::Health(h) => {
            println!("health: all_ok={}", h.all_ok);
            for s in h.services {
                let detail = s.detail.map(|d| format!(" ({d})")).unwrap_or_default();
                println!(
                    "  - {} ok={} restarts={}{}",
                    s.name, s.ok, s.restarts, detail
                );
            }
        }
        ApiResponse::Stats(s) => println!(
            "stats: jobs={} wakes={} sessions={} active={} usage(in={} out={} cache_r={} cache_w={} reason={} cost=${:.4})",
            s.pending_jobs,
            s.pending_wakes,
            s.sessions,
            s.active,
            s.usage.input_tokens,
            s.usage.output_tokens,
            s.usage.cache_read_tokens,
            s.usage.cache_write_tokens,
            s.usage.reasoning_tokens,
            s.usage.cost_micros as f64 / 1_000_000.0,
        ),
        ApiResponse::Telemetry(d) => println!(
            "telemetry: healthy={} events={} jobs={} wakes={} sessions={} active={} usage(in={} out={} cache_r={} cache_w={} reason={} api_calls={} cost=${:.4})",
            d.healthy,
            d.events,
            d.pending_jobs,
            d.pending_wakes,
            d.sessions,
            d.active,
            d.usage.input_tokens,
            d.usage.output_tokens,
            d.usage.cache_read_tokens,
            d.usage.cache_write_tokens,
            d.usage.reasoning_tokens,
            d.usage.api_calls,
            d.usage.cost_micros as f64 / 1_000_000.0,
        ),
        ApiResponse::Sessions(list) => {
            println!("sessions: {}", list.len());
            for info in list {
                println!("  - {} {:?}", info.session, info.state);
            }
        }
        ApiResponse::Approvals(list) => {
            println!("pending approvals: {}", list.len());
            for info in list {
                let path = info.path.map(|p| format!(" path={p}")).unwrap_or_default();
                println!(
                    "  - {} req={}{} :: {}",
                    info.session, info.request_id, path, info.prompt
                );
            }
        }
        ApiResponse::Checkpoints(list) => {
            println!("checkpoints: {}", list.len());
            for info in list {
                println!(
                    "  - {} session={} tool={} created={}",
                    info.id, info.session, info.tool, info.created_unix
                );
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
        ApiResponse::VerifyingKey(key) => match key {
            Some(hex) => println!("verifying_key: {hex}"),
            None => println!("verifying_key: none (node exposes no journal signer)"),
        },
        ApiResponse::ModelSearch(page) => {
            println!(
                "search: page={} results={} has_more={}",
                page.page,
                page.results.len(),
                page.has_more
            );
            for h in page.results {
                let gated = if h.gated { " [gated]" } else { "" };
                println!(
                    "  - {} downloads={} likes={} {}{}",
                    h.repo,
                    h.downloads,
                    h.likes,
                    h.pipeline_tag.as_deref().unwrap_or("-"),
                    gated
                );
            }
        }
        ApiResponse::ModelFiles(files) => {
            println!("files: {}", files.len());
            for f in files {
                let quant = f.quant.map(|q| format!(" quant={q}")).unwrap_or_default();
                let split = if f.is_split { " split" } else { "" };
                println!("  - {} ({} bytes){}{}", f.path, f.size_bytes, quant, split);
            }
        }
        ApiResponse::ModelDownloadStarted(id) => println!("download started: {id}"),
        ApiResponse::ModelDownloads(jobs) => {
            println!("downloads: {}", jobs.len());
            for j in jobs {
                let pct = if j.total_bytes > 0 {
                    (j.downloaded_bytes as f64 / j.total_bytes as f64 * 100.0) as u64
                } else {
                    0
                };
                let err = j.error.map(|e| format!(" error={e}")).unwrap_or_default();
                println!(
                    "  - {} {:?} {}% ({}/{} bytes, files {}/{}){}",
                    j.id,
                    j.state,
                    pct,
                    j.downloaded_bytes,
                    j.total_bytes,
                    j.files_done,
                    j.files_total,
                    err
                );
            }
        }
        ApiResponse::ModelCatalog(models) => {
            println!("installed: {}", models.len());
            for m in models {
                let quant = m.quant.map(|q| format!(" quant={q}")).unwrap_or_default();
                let arch = m.arch.map(|a| format!(" arch={a}")).unwrap_or_default();
                let ctx = m
                    .context_length
                    .map(|c| format!(" ctx={c}"))
                    .unwrap_or_default();
                println!(
                    "  - {} {} ({} bytes){}{}{} -> {}",
                    m.id,
                    m.display_name,
                    m.size_bytes,
                    quant,
                    arch,
                    ctx,
                    m.local_path.display()
                );
            }
        }
        ApiResponse::ModelRecommend(rec) => {
            println!(
                "recommend: {} engine={} quant={} fits={} budget={} bytes",
                rec.repo, rec.engine, rec.quant, rec.fits, rec.budget_bytes
            );
            if let Some(f) = &rec.file {
                println!("  file: {f}");
            }
            if let Some(s) = rec.size_bytes {
                println!("  size: {s} bytes");
            }
            println!("  reason: {}", rec.reason);
            for c in rec.candidates {
                let size = c
                    .size_bytes
                    .map(|s| format!("{s} bytes"))
                    .unwrap_or_else(|| "?".into());
                let file = c.file.map(|f| format!(" {f}")).unwrap_or_default();
                println!("    - {} ({}) fits={}{}", c.quant, size, c.fits, file);
            }
        }
        ApiResponse::ModelQuantizeStarted(id) => println!("quantize started: {id}"),
        ApiResponse::ModelQuantizes(jobs) => {
            println!("quantizations: {}", jobs.len());
            for j in jobs {
                let out = j
                    .output_path
                    .map(|p| format!(" -> {}", p.display()))
                    .unwrap_or_default();
                let err = j.error.map(|e| format!(" error={e}")).unwrap_or_default();
                println!(
                    "  - {} {} -> {} {:?}{}{}",
                    j.id, j.source_file, j.target_quant, j.state, out, err
                );
            }
        }
        ApiResponse::ModelInspect(info) => {
            println!("inspect:");
            println!(
                "  architecture: {}",
                info.architecture.as_deref().unwrap_or("-")
            );
            println!("  name: {}", info.name.as_deref().unwrap_or("-"));
            println!("  file_type: {}", info.file_type.as_deref().unwrap_or("-"));
            println!(
                "  context_length: {}",
                info.context_length
                    .map(|c| c.to_string())
                    .unwrap_or_else(|| "-".into())
            );
            println!(
                "  block_count: {}",
                info.block_count
                    .map(|c| c.to_string())
                    .unwrap_or_else(|| "-".into())
            );
            println!(
                "  parameters: {}",
                info.parameter_count
                    .map(|c| c.to_string())
                    .unwrap_or_else(|| "-".into())
            );
            println!("  size_bytes: {}", info.size_bytes);
        }
        ApiResponse::Profiles(profiles) => {
            println!("profiles: {}", profiles.len());
            for p in profiles {
                let active = if p.is_active { " *" } else { "" };
                println!("  - {} [{:?}] {}{}", p.id, p.provider, p.model, active);
            }
        }
        ApiResponse::Profile(spec) => match spec {
            Some(s) => {
                println!("profile: {}", s.id);
                println!("  provider: {:?}", s.provider);
                println!("  model: {}", s.model);
                if let Some(base) = &s.base_url {
                    println!("  base_url: {base}");
                }
                println!("  credential_ref: {}", s.credential_profile());
            }
            None => println!("profile: none"),
        },
        ApiResponse::ConfigSchema(schema) => {
            println!("config schema: {} field(s)", schema.fields.len());
            for f in schema.fields {
                let opts = if f.options.is_empty() {
                    String::new()
                } else {
                    format!(" [{}]", f.options.join("|"))
                };
                println!("  - {} ({}){}: {}", f.key, f.kind, opts, f.description);
            }
        }
        ApiResponse::Credentials(creds) => {
            println!("credentials: {}", creds.len());
            for c in creds {
                let state = if c.present { c.hint.clone() } else { "(none)".to_string() };
                println!("  - {} {}", c.profile, state);
            }
        }
        ApiResponse::Models(models) => {
            println!("models: {}", models.len());
            for m in models {
                let ctx = m
                    .context_length
                    .map(|c| format!(" ctx={c}"))
                    .unwrap_or_default();
                let kind = if m.local { " [local]" } else { "" };
                println!("  - {} [{:?}]{}{}", m.id, m.provider, ctx, kind);
            }
        }
        ApiResponse::ModelCurrent(model) => match model {
            Some(m) => {
                let ctx = m
                    .context_length
                    .map(|c| format!(" ctx={c}"))
                    .unwrap_or_default();
                println!("current model: {} [{:?}]{}", m.id, m.provider, ctx);
            }
            None => println!("current model: none"),
        },
        ApiResponse::Distribution(d) => {
            println!("distribution: {} (wire v{})", d.profile.id, d.wire_version.0);
            println!("  provider: {:?}", d.profile.provider);
            println!("  model: {}", d.profile.model);
            println!("  credential_ref: {}", d.profile.credential_profile());
            println!("  skills: {}", d.skills.len());
            for s in &d.skills {
                println!("    - {}", s.name);
            }
            if let Some(seq) = d.head_seq {
                println!("  head revision: {seq}");
            }
        }
        ApiResponse::ProfileId(id) => println!("imported profile: {id}"),
        ApiResponse::Revisions(revs) => {
            println!("revisions: {}", revs.len());
            for r in revs {
                let author = match &r.author {
                    daemon_api::Author::Operator => "operator".to_string(),
                    daemon_api::Author::Agent(label) => format!("agent:{label}"),
                };
                println!("  - #{} [{}] {} (parent {:?})", r.seq, author, r.reason, r.parent);
            }
        }
        ApiResponse::SkillBundle(b) => {
            let cat = b.category.as_deref().unwrap_or("general");
            println!("skill: {} [{}] ({} file(s))", b.name, cat, b.files.len());
            for path in b.files.keys() {
                println!("  - {path}");
            }
        }
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
