// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon` — the host binary that assembles an engine, its host, tools, and orchestration.
//!
//! It is the role-by-config node (workspace-layout §6):
//! - default **host** role: build the policy inputs (store, credentials, provider registry, engine
//!   tunables) and hand them to [`daemon_node::assemble`] — the single host-composition root shared
//!   with the conformance harness — then serve the one [`daemon_api`] surface over a Unix socket.
//! - **placed-child** role (`daemon internal placed-child`): the far side of a placement cut, driving
//!   an engine whose durable state is brokered back to the parent's store.
//! - **transport-server** role (`daemon internal transport-server <addr>`): host a unit + authoritative
//!   store reached over a socket ([`daemon_transport::RemoteHost`]).

#![forbid(unsafe_code)]

mod config;

use clap::Parser as _;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use daemon_api::{
    BudgetSpec, ContextEngineSel, EngineTunables, MemoryProviderSel, ModelDescriptor, ProfileSpec,
    ProviderDescriptor, ProviderKindWire, ProviderSelector,
};
use daemon_common::{
    CredMode, CredScope, JournalStreamId, ModelEngine, ModelRef, ModelSource, ProfileRef,
    SessionId, UnitId,
};
use daemon_context_lcm::{LcmConfig, LcmContextEngine};
use daemon_core::{
    CommandProviderHandle, ContextEngine, ContextEngineBuilder, CredentialBuilder,
    CredentialProvider, EmbeddingProvider, EngineProfile, FileMemory, MemoryBuilder,
    MemoryProvider, MockProvider, Provider, ProviderRegistry, ScriptStep, ScriptedProvider,
    SystemPrompt, Tool, ToolCall, ToolDef, ToolOutcome, ToolProvider, ToolRegistry, TurnCx,
    UnconfiguredProvider,
};
use daemon_credentials::{CapabilitySigner, CredentialAuthority};
// The unix-socket api transport is unix-only (tokio has no AF_UNIX on windows); a windows node
// serves the networked TLS/WS/HTTP surfaces instead (see `run_as_host`).
#[cfg(unix)]
use daemon_host::serve_api_unix;
use daemon_host::{
    run_placed_child, run_placed_child_journaled, BrokeredCredentialProvider, CloudCatalog,
    CommandRegistry, CoreEngineFactory, CredentialAuditDrain, CredentialBroker, CredentialStore,
    EngineUnit, FileCredentialStore, FileProfileStore, FileRevisionLog, HostConfig, JournalFeeder,
    JournalSink, MemCredentialStore, MemProfileStore, MultiProfileStoreBroker, OriginMatcher,
    OwnerBroker, PooledStoreCredentialSource, ProfileStore, RoutingRegistry, ScopePattern,
    SessionBinding, TransportPattern,
};
use daemon_infer::protocol::{Engine, ModelParams};
use daemon_metta::protocol::Bounds as MettaBounds;
use daemon_metta_client::{MettaConfig as MettaClientConfig, MettaCoprocessor};
use daemon_mnemosyne::{MnemosyneConfig, MnemosyneProvider};
use daemon_models::{ActiveModels, ManagerConfig, ModelManager};
use daemon_node::{assemble, AssembledNode, NodeAssembly, ProviderResolver};
use daemon_protocol::{IsolationPolicy, TransportId};
use daemon_providers::{
    discovery_vendor_ids, genai_listed_models, genai_models_for_id, GenAiEmbedder, GenAiProvider,
    LocalEmbedder, SwitchableLocalProvider, WorkerConfig, DAEMON_CLOUD_BASE,
};
use daemon_provision::CutChannel;
use daemon_pytool_client::{PyToolConfig, PyToolProvider};
use daemon_store::{InMemoryStore, SessionStore};
use daemon_supervision::ManagedUnit;
use daemon_tool_clarify::ClarifyTool;
use daemon_tool_metta::MettaTool;
use daemon_tool_todo::TodoTool;
use daemon_tool_vision::{VisionAnalyzeTool, VisionToolConfig};
use daemon_tool_web::{
    FirecrawlFetch, LocalFetch, SecretSource, TavilySearch, WebExtractTool, WebFetchBackend,
    WebSearchTool,
};
use daemon_transport::RemoteHost;

use config::{
    ContextEngineKind, EmbedKind, MemoryProviderKind, NodeConfig, ProviderKind, RoutingConfig,
    StoreBackend, VisionKind,
};

/// `daemon` — the role-by-config host node. With no subcommand it runs the host role, loading the
/// layered [`NodeConfig`] (defaults <- TOML <- env <- these CLI overrides).
#[derive(clap::Parser)]
#[command(name = "daemon", version = daemon_common::VERSION, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
    #[command(flatten)]
    overrides: ConfigOverrides,
}

#[derive(clap::Subcommand)]
enum Command {
    /// Operate a Matrix account (SSO login + session persistence).
    Matrix {
        #[command(subcommand)]
        cmd: MatrixCmd,
    },
    /// Inspect the layered configuration.
    Config {
        #[command(subcommand)]
        cmd: ConfigCmd,
    },
    /// Internal process roles used by the node itself (not for direct use).
    #[command(hide = true, subcommand)]
    Internal(InternalCmd),
}

#[derive(clap::Subcommand)]
enum ConfigCmd {
    /// Print the generated Markdown configuration reference (every key: TOML path, env var, type,
    /// default) — the source for `docs/config-reference.md` and its drift gate.
    Reference,
}

#[derive(clap::Subcommand)]
enum MatrixCmd {
    /// SSO-login one account and persist its session under a credential-ref (the same key the
    /// profile's `bound_accounts` declares). Writes to the durable `FileCredentialStore`.
    Login {
        /// The homeserver base URL.
        #[arg(long)]
        homeserver: String,
        /// The credential-ref key the resulting session is stored under.
        #[arg(long, alias = "credential_ref")]
        credential_ref: String,
    },
}

#[derive(clap::Subcommand)]
enum InternalCmd {
    /// The far side of a placement cut, driving an engine over stdio brokered by the parent.
    PlacedChild,
    /// Host a unit + authoritative store reached over a socket at `<addr>`.
    TransportServer {
        /// The bind address (e.g. a Unix socket path).
        addr: String,
    },
}

/// Top-level config overrides — the highest-precedence layer. Only flags that are set serialize
/// (unset flags never clobber env/TOML/defaults). Field names match the [`NodeConfig`] serde keys.
#[derive(clap::Args, serde::Serialize)]
struct ConfigOverrides {
    /// The Unix socket the node serves its api on.
    #[arg(long = "socket")]
    #[serde(rename = "socket_path", skip_serializing_if = "Option::is_none")]
    socket_path: Option<std::path::PathBuf>,
    /// The host data directory rooting the profile-scoped subsystem databases.
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    data_dir: Option<std::path::PathBuf>,
    /// The provider/credential profile name.
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    profile: Option<String>,
    /// The model provider (mock|scripted|genai|daemon_api|llama|mistralrs).
    #[arg(long = "model-provider")]
    #[serde(rename = "model_provider", skip_serializing_if = "Option::is_none")]
    model_provider: Option<String>,
    /// The model name sent to a real provider (or the model path / HF id for a local provider).
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<String>,
    /// The launch credential key (the daemon-api / provider bearer).
    #[arg(long = "credential-key")]
    #[serde(rename = "credential_key", skip_serializing_if = "Option::is_none")]
    credential_key: Option<String>,
    /// The durable store backend (memory|sqlite).
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    store: Option<String>,
    /// The in-process HTTP/WS surface bind address (enables it).
    #[arg(long = "http-addr")]
    #[serde(rename = "http_addr", skip_serializing_if = "Option::is_none")]
    http_addr: Option<String>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // clap owns `--version`/`-h` (exiting before any setup keeps their output clean).
    let cli = Cli::parse();

    // Stderr-only structured logging (stdout is the cut transport in the child role), plus the
    // OpenTelemetry OTLP export layer when built with `--features otel` and an endpoint is set. The
    // guard flushes the exporter on process exit, so it is held for the whole of `main`.
    let _telemetry = daemon_telemetry::init_telemetry();
    // With a live exporter, switch on GenAI span content capture in the engine (off otherwise, so a
    // default build attaches no session content to any span).
    #[cfg(feature = "otel")]
    if _telemetry.is_exporting() {
        daemon_core::set_genai_capture(true);
    }

    match cli.command {
        Some(Command::Internal(InternalCmd::PlacedChild)) => {
            run_as_placed_child().await;
            Ok(())
        }
        Some(Command::Internal(InternalCmd::TransportServer { addr })) => {
            run_as_transport_server(addr).await
        }
        Some(Command::Config {
            cmd: ConfigCmd::Reference,
        }) => {
            print!("{}", config::config_reference());
            Ok(())
        }
        Some(Command::Matrix {
            cmd:
                MatrixCmd::Login {
                    homeserver,
                    credential_ref,
                },
        }) => run_matrix_login(&homeserver, &credential_ref).await,
        None => {
            // defaults <- TOML <- env <- CLI overrides (later wins).
            let fig = NodeConfig::base_figment()
                .merge(figment::providers::Serialized::defaults(cli.overrides));
            run_as_host(NodeConfig::from_figment(fig)?).await
        }
    }
}

/// The `daemon matrix login` subcommand: SSO-login one account and persist its session under the
/// given credential-ref. Writes to the durable `FileCredentialStore` the host reads at startup.
async fn run_matrix_login(homeserver: &str, credential_ref: &str) -> anyhow::Result<()> {
    let cfg = NodeConfig::load()?;
    // Login implies persistence: write to the same on-disk credential store the host reads,
    // through the same single creation helper the host boot uses (private on create).
    ensure_data_dir(&cfg.data_dir)?;
    let credential_store: Arc<dyn CredentialStore> = Arc::new(FileCredentialStore::open(
        cfg.data_dir.join("credentials.json"),
    )?);
    daemon_matrix::login(
        credential_store,
        homeserver,
        &cfg.matrix.store_root,
        credential_ref,
    )
    .await
}

/// Build the provider registry for the explicitly-selected `provider_kind` (there is no silent
/// default — a host launch resolves the kind through [`NodeConfig::validate_for_host`] first).
/// `Mock`/`Scripted` are opt-in only and keep the deterministic fleet wiring (a completing default
/// plus the delegating-orchestrator / completing-child demo profiles); a real provider becomes the
/// registry default for every profile (the engine threads the credential lease secret onto each
/// request as the bearer).
/// The price sheet for a cloud `model`, looked up in the built-in catalog (the static fallback that
/// also backs the GUI model picker). `None` for an unknown / local model — cost is then left
/// uncomputed (`cost_micros == 0`). Cache read/write rates are derived from the base input rate.
fn pricing_for(model: &str) -> Option<daemon_common::Pricing> {
    ModelDescriptor::builtin_cloud_catalog()
        .into_iter()
        .find(|m| m.id == model)
        .and_then(|m| {
            match (
                m.input_price_micros_per_mtok,
                m.output_price_micros_per_mtok,
            ) {
                (Some(input), Some(output)) => Some(daemon_common::Pricing::from_io(input, output)),
                _ => None,
            }
        })
}

/// The binary's live networked-model discovery hook (the `daemon-host` is provider-agnostic and
/// never links `genai`). Asks `genai` for the models of every adapter whose key resolves, unions
/// them with the static catalog (which also carries the pricing/context overlay), and namespaces
/// live ids so they round-trip through adapter inference.
struct GenAiCloudCatalog;

/// Overlay a genai-listed cloud model id with the static catalog's pricing/context (genai supplies
/// neither), tagged as the `GenAi` provider. Shared by `list()` and per-vendor `provider_models`.
fn genai_model_descriptor(id: String) -> ModelDescriptor {
    let overlay = ModelDescriptor::builtin_cloud_catalog();
    let known = overlay.iter().find(|m| m.id == id);
    ModelDescriptor {
        context_length: known
            .and_then(|m| m.context_length)
            .or_else(|| ModelDescriptor::known_context_length(&id)),
        input_price_micros_per_mtok: known.and_then(|m| m.input_price_micros_per_mtok),
        output_price_micros_per_mtok: known.and_then(|m| m.output_price_micros_per_mtok),
        display_name: None,
        id,
        provider: ProviderSelector::GenAi,
        local: false,
    }
}

/// A Daemon Cloud gateway model row (OpenAI-compatible `GET /models`; ids are `author/slug`), best
/// effort: name/context/pricing when the gateway supplies them, else `None`. Deserialized tolerantly.
#[derive(serde::Deserialize)]
struct GatewayModel {
    id: String,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    context_length: Option<u32>,
}

/// Fetch Daemon Cloud gateway models keyless via `GET {base}/models` (unauth MVP). Tolerates both the
/// OpenAI `{ "data": [..] }` envelope and a bare array; a non-200 (incl. the 500 "Registry not
/// published") or a transport error yields an empty list (never an error to the picker). Ids stay
/// `author/slug` so they feed `ProfileSpec.model` verbatim.
async fn daemon_cloud_gateway_models(base: &str) -> Vec<ModelDescriptor> {
    let url = format!("{}models", NodeConfig::ensure_trailing_slash(base));
    let resp = match reqwest::Client::new().get(&url).send().await {
        Ok(r) if r.status().is_success() => r,
        Ok(r) => {
            tracing::debug!(status = %r.status(), "daemon cloud gateway /models non-success");
            return Vec::new();
        }
        Err(e) => {
            tracing::debug!(error = %e, "daemon cloud gateway /models unreachable");
            return Vec::new();
        }
    };
    let body: serde_json::Value = match resp.json().await {
        Ok(v) => v,
        Err(e) => {
            tracing::debug!(error = %e, "daemon cloud gateway /models parse");
            return Vec::new();
        }
    };
    let raw = body.get("data").cloned().unwrap_or(body);
    let models: Vec<GatewayModel> = serde_json::from_value(raw).unwrap_or_default();
    models
        .into_iter()
        .map(|m| ModelDescriptor {
            id: m.id,
            provider: ProviderSelector::DaemonApi,
            display_name: m.name,
            context_length: m.context_length,
            input_price_micros_per_mtok: None,
            output_price_micros_per_mtok: None,
            local: false,
        })
        .collect()
}

#[async_trait::async_trait]
impl CloudCatalog for GenAiCloudCatalog {
    async fn list(&self) -> Vec<ModelDescriptor> {
        // Start from the static catalog: the pricing/context overlay + the no-key fallback list.
        let mut out = ModelDescriptor::builtin_cloud_catalog();
        let mut seen: std::collections::HashSet<String> =
            out.iter().map(|m| m.id.clone()).collect();
        for id in genai_listed_models().await {
            if !seen.insert(id.clone()) {
                continue;
            }
            out.push(genai_model_descriptor(id));
        }
        out
    }

    async fn providers(&self) -> Vec<ProviderDescriptor> {
        let mut out = vec![
            // Local inference engines (models come from the ModelManager catalog, node-owned).
            ProviderDescriptor {
                id: "llama_cpp".into(),
                display_name: "llama.cpp (local)".into(),
                kind: ProviderKindWire::Local,
                wire_selector: ProviderSelector::LlamaCpp,
                requires_key: false,
                supports_model_discovery: true,
                default_base_url: None,
            },
            ProviderDescriptor {
                id: "mistral_rs".into(),
                display_name: "mistral.rs (local)".into(),
                kind: ProviderKindWire::Local,
                wire_selector: ProviderSelector::MistralRs,
                requires_key: false,
                supports_model_discovery: true,
                default_base_url: None,
            },
        ];
        // One row per genai cloud vendor (all bind `ProviderSelector::GenAi`; the vendor dimension is
        // carried by `id`). Listing their models needs a key.
        for (id, display_name) in discovery_vendor_ids() {
            out.push(ProviderDescriptor {
                id,
                display_name,
                kind: ProviderKindWire::Cloud,
                wire_selector: ProviderSelector::GenAi,
                requires_key: true,
                supports_model_discovery: true,
                default_base_url: None,
            });
        }
        // Daemon Cloud (OpenRouter clone). Needs a key to RUN TURNS (its
        // `/api/v1/chat/completions` is bearer-authed), so `requires_key` is true — but model
        // LISTING stays keyless (the public gateway `/models` is unauth; `provider_models` never
        // gates on this flag). Carries the gateway base so the app never hardcodes it.
        out.push(ProviderDescriptor {
            id: "daemon_cloud".into(),
            display_name: "Daemon Cloud".into(),
            kind: ProviderKindWire::DaemonCloud,
            wire_selector: ProviderSelector::DaemonApi,
            requires_key: true,
            supports_model_discovery: true,
            default_base_url: Some(DAEMON_CLOUD_BASE.to_string()),
        });
        out
    }

    async fn provider_models(
        &self,
        provider_id: &str,
        key: Option<String>,
    ) -> Vec<ModelDescriptor> {
        match provider_id {
            // Local engines are served by the host from the ModelManager catalog, not here.
            "llama_cpp" | "mistral_rs" => Vec::new(),
            // Daemon Cloud: keyless gateway listing (author/slug).
            "daemon_cloud" => daemon_cloud_gateway_models(DAEMON_CLOUD_BASE).await,
            // A genai cloud vendor: credential-aware live listing, overlaid with static pricing.
            vendor => genai_models_for_id(vendor, key.as_deref())
                .await
                .into_iter()
                .map(genai_model_descriptor)
                .collect(),
        }
    }
}

/// Parse the [`ProviderKind::Scripted`] replay script from `DAEMON_MOCK_SCRIPT`: a JSON array whose
/// entries are either `{"call": "<tool>", "args": "<payload>"}` (emit a tool call) or
/// `{"final": "<text>"}` (the completing text). Returns the ordered call steps plus the final text
/// (the last `final` wins; default "done"). A malformed/empty script yields an immediately-completing
/// provider, so a misconfigured launch degrades to a trivial turn rather than panicking.
fn parse_mock_script(raw: Option<&str>) -> (Vec<ScriptStep>, String) {
    #[derive(serde::Deserialize)]
    struct Entry {
        call: Option<String>,
        #[serde(default)]
        args: String,
        #[serde(rename = "final")]
        final_text: Option<String>,
    }
    let mut steps = Vec::new();
    let mut final_text = String::from("done");
    if let Some(text) = raw {
        match serde_json::from_str::<Vec<Entry>>(text) {
            Ok(entries) => {
                for e in entries {
                    if let Some(name) = e.call {
                        steps.push(ScriptStep::Call { name, args: e.args });
                    } else if let Some(t) = e.final_text {
                        final_text = t;
                    }
                }
            }
            Err(err) => tracing::warn!(%err, "DAEMON_MOCK_SCRIPT is not a valid step array"),
        }
    }
    (steps, final_text)
}

/// Build the scripted provider builder shared by the launch registry + per-profile resolver.
fn scripted_builder(cfg: &NodeConfig) -> daemon_core::ProviderBuilder {
    let (steps, final_text) = parse_mock_script(cfg.mock_script.as_deref());
    Arc::new(move || {
        Arc::new(ScriptedProvider::new(steps.clone(), final_text.clone())) as Arc<dyn Provider>
    })
}

fn build_providers(
    cfg: &NodeConfig,
    provider_kind: Option<ProviderKind>,
    manager: &Arc<ModelManager>,
    active: &ActiveModels,
) -> ProviderRegistry {
    let mut providers = ProviderRegistry::new();
    let Some(provider_kind) = provider_kind else {
        // Unconfigured boot: the default provider fails every turn with a clear, actionable error
        // (never a silent mock). Discovery + profile creation still work; a configured profile
        // resolves its own provider via `provider_builder_for`.
        providers.set_default(Arc::new(|| {
            Arc::new(UnconfiguredProvider::new()) as Arc<dyn Provider>
        }));
        return providers;
    };
    match provider_kind {
        ProviderKind::Scripted => {
            // Hermetic tool-using turn: the default provider replays the configured script, so a
            // side-effecting tool call parks an approval under the node's Ask policy (the HITL e2e).
            providers.set_default(scripted_builder(cfg));
            providers.register(
                "child",
                Arc::new(|| Arc::new(MockProvider::completing("child done")) as Arc<dyn Provider>),
            );
        }
        ProviderKind::Mock => {
            providers.set_default(Arc::new(|| {
                Arc::new(MockProvider::completing("session done")) as Arc<dyn Provider>
            }));
            providers.register(
                "orchestrator",
                Arc::new(|| {
                    Arc::new(MockProvider::delegating("orchestrate", "fleet done"))
                        as Arc<dyn Provider>
                }),
            );
            providers.register(
                "child",
                Arc::new(|| Arc::new(MockProvider::completing("child done")) as Arc<dyn Provider>),
            );
        }
        ProviderKind::GenAi => {
            // One genai-backed default; the adapter is inferred from the model name (`for_model`).
            let (base, model) = (cfg.base_url.clone(), cfg.model.clone());
            let pricing = pricing_for(&model);
            providers.set_default(Arc::new(move || {
                let mut p = GenAiProvider::for_model(model.clone());
                if let Some(base) = &base {
                    p = p.with_endpoint(base.clone());
                }
                if let Some(pricing) = pricing {
                    p = p.with_pricing(pricing);
                }
                Arc::new(p) as Arc<dyn Provider>
            }));
        }
        ProviderKind::DaemonApi => {
            // Daemon Cloud (OpenRouter clone) via `GenAiProvider::daemon_cloud`: genai's OpenAI
            // adapter pinned at the daemon base (default `https://api.daemon.ai/api/v1/`, override
            // `DAEMON_BASE_URL`). NEVER `for_model(...)` — that would infer the Anthropic-native wire
            // for `claude-*` ids against an OpenAI-compatible gateway. The bearer flows per-call.
            let (base, model) = (cfg.daemon_api_base(), cfg.model.clone());
            let pricing = pricing_for(&model);
            providers.set_default(Arc::new(move || {
                let mut p = GenAiProvider::daemon_cloud(model.clone()).with_endpoint(base.clone());
                if let Some(pricing) = pricing {
                    p = p.with_pricing(pricing);
                }
                Arc::new(p) as Arc<dyn Provider>
            }));
        }
        ProviderKind::LlamaCpp | ProviderKind::MistralRs => {
            let engine = match provider_kind {
                ProviderKind::MistralRs => Engine::MistralRs,
                _ => Engine::Llama,
            };
            if cfg.model.is_empty() {
                tracing::warn!(
                    "local provider selected but no model set; set DAEMON_MODEL to the GGUF path / HF id"
                );
            }
            // One supervised worker, shared (cloned) across every profile/engine: a single local
            // model lives in VRAM once and serializes generations behind the provider's mutex. The
            // switchable wrapper resolves the profile's *active* model through `daemon-models`
            // (download-on-first-use into the shared cache) and hot-swaps the worker on activation.
            let provider: Arc<dyn Provider> = Arc::new(SwitchableLocalProvider::new(
                local_worker_config(cfg, engine),
                manager.clone(),
                active.clone(),
                cfg.profile.clone(),
            ));
            providers.set_default(Arc::new(move || provider.clone()));
        }
    }
    providers
}

/// Build a single provider client for a [`ProfileSpec`] — the per-session resolution seam handed to
/// [`daemon_node::assemble`] so an interactive session's provider/model/base-URL come from the
/// active profile bundle (the GUI-settable surface), not the fixed launch config. Mirrors
/// [`build_providers`]' per-kind construction. The API key still flows per-call through the
/// credential broker.
fn provider_builder_for(
    spec: &ProfileSpec,
    cfg: &NodeConfig,
    provider_kind: Option<ProviderKind>,
    manager: &Arc<ModelManager>,
    active: &ActiveModels,
) -> daemon_core::ProviderBuilder {
    // Scripted launch overrides the per-profile provider: the wire ProviderSelector stays Mock
    // (no contract change), but every interactive session runs the scripted tool-calling provider
    // so the HITL e2e drives real parked approvals/clarify.
    if provider_kind == Some(ProviderKind::Scripted) {
        return scripted_builder(cfg);
    }
    match spec.provider {
        ProviderSelector::Mock => {
            Arc::new(|| Arc::new(MockProvider::completing("session done")) as Arc<dyn Provider>)
        }
        // The single genai-backed path; the adapter is inferred from the model name (`for_model`).
        // A networked selector with no model yet is UNCONFIGURED (never a silent mock): the turn
        // fails clearly until the GUI picks a provider + model.
        ProviderSelector::GenAi => {
            if spec.model.trim().is_empty() {
                return Arc::new(|| Arc::new(UnconfiguredProvider::new()) as Arc<dyn Provider>);
            }
            let (base, model) = (spec.base_url.clone(), spec.model.clone());
            let pricing = pricing_for(&model);
            Arc::new(move || {
                let mut p = GenAiProvider::for_model(model.clone());
                if let Some(base) = &base {
                    p = p.with_endpoint(base.clone());
                }
                if let Some(pricing) = pricing {
                    p = p.with_pricing(pricing);
                }
                Arc::new(p) as Arc<dyn Provider>
            })
        }
        // Daemon Cloud: genai's OpenAI adapter pinned at the profile's base URL (default
        // `https://api.daemon.ai/api/v1/`) via `GenAiProvider::daemon_cloud`. NEVER `for_model(...)`
        // — the gateway is OpenAI-compatible, so an inferred Anthropic-native wire for `claude-*` ids
        // would be wrong. An empty model is UNCONFIGURED (clear error, never mock).
        ProviderSelector::DaemonApi => {
            if spec.model.trim().is_empty() {
                return Arc::new(|| Arc::new(UnconfiguredProvider::new()) as Arc<dyn Provider>);
            }
            let base = spec.base_url.clone();
            let model = spec.model.clone();
            let pricing = pricing_for(&model);
            Arc::new(move || {
                let mut p = match &base {
                    Some(base) => GenAiProvider::daemon_cloud(model.clone())
                        .with_endpoint(NodeConfig::ensure_trailing_slash(base)),
                    None => GenAiProvider::daemon_cloud(model.clone()),
                };
                if let Some(pricing) = pricing {
                    p = p.with_pricing(pricing);
                }
                Arc::new(p) as Arc<dyn Provider>
            })
        }
        ProviderSelector::LlamaCpp | ProviderSelector::MistralRs => {
            let engine = match spec.provider {
                ProviderSelector::MistralRs => Engine::MistralRs,
                _ => Engine::Llama,
            };
            let provider: Arc<dyn Provider> = Arc::new(SwitchableLocalProvider::new(
                local_worker_config(cfg, engine),
                manager.clone(),
                active.clone(),
                spec.id.clone(),
            ));
            Arc::new(move || provider.clone())
        }
    }
}

/// Seed the launch [`NodeConfig`] as the node's default [`ProfileSpec`] so existing env/TOML
/// launches keep working: the active profile mirrors the configured provider/model/base-URL/engine
/// tunables, and a GUI can then clone/edit/select alternatives over the `ProfileApi`.
fn default_profile_spec(
    cfg: &NodeConfig,
    provider_kind: Option<ProviderKind>,
    cloud_seed: bool,
) -> ProfileSpec {
    // D2: when a Daemon Cloud attach key was seeded (`DAEMON_CLOUD_API_KEY[_FILE]`), the first-boot
    // default profile must deterministically select the `daemon_api` ("Daemon Cloud") provider at
    // the gateway base, regardless of any `DAEMON_MODEL_PROVIDER`. The model (below) still comes
    // from `DAEMON_MODEL` (`cfg.model`), which stays empty ("pick a model at first turn") when
    // unset — D2 wires provider + credential + base, not a default model.
    let provider = if cloud_seed {
        ProviderSelector::DaemonApi
    } else {
        match provider_kind {
            // Unconfigured boot: seed a Daemon Cloud selector with an EMPTY model so the profile does
            // not silently chat — `provider_builder_for` resolves it to `UnconfiguredProvider` until the
            // GUI picks a provider + model. New-profile default surface is Daemon Cloud (base prefilled).
            None => ProviderSelector::DaemonApi,
            // Scripted is binary-internal: the durable profile records Mock so the wire stays unchanged
            // (provider_builder_for swaps in the scripted provider for the live session).
            Some(ProviderKind::Mock | ProviderKind::Scripted) => ProviderSelector::Mock,
            Some(ProviderKind::GenAi) => ProviderSelector::GenAi,
            Some(ProviderKind::DaemonApi) => ProviderSelector::DaemonApi,
            Some(ProviderKind::LlamaCpp) => ProviderSelector::LlamaCpp,
            Some(ProviderKind::MistralRs) => ProviderSelector::MistralRs,
        }
    };
    let context_engine = match cfg.context_engine {
        ContextEngineKind::Lcm => ContextEngineSel::Lcm,
        ContextEngineKind::Budgeted => ContextEngineSel::Budgeted,
    };
    let memory_provider = match cfg.memory_provider {
        MemoryProviderKind::Mnemosyne => MemoryProviderSel::Mnemosyne,
        MemoryProviderKind::File => MemoryProviderSel::File,
        MemoryProviderKind::None => MemoryProviderSel::None,
    };
    ProfileSpec {
        id: cfg.profile.clone(),
        provider,
        model: cfg.model.clone(),
        base_url: cfg.base_url.clone(),
        system_prompt: String::new(),
        tool_allowlist: None,
        budget: BudgetSpec::default(),
        tunables: EngineTunables {
            model_retry_attempts: Some(cfg.engine.model_retry_attempts),
            context_budget_tokens: cfg.engine.context_budget_tokens,
            max_iterations: Some(cfg.engine.max_iterations),
            tool_result_budget: Some(cfg.engine.tool_result_budget),
        },
        context_engine,
        memory_provider,
        credential_ref: None,
        fallback_credential_ref: None,
        bound_accounts: Vec::new(),
        // The launch-config seed always runs the native engine; foreign (ACP) profiles are
        // created explicitly over the ProfileApi.
        engine: daemon_api::EngineSelector::Core,
    }
}

/// The default §10 context wiring: an optional per-session context-engine *builder* and the
/// `lcm_*` drill-down tools registered on every role registry. Mirrors [`MemoryWiring`] — the
/// `Budgeted` fallback yields neither.
struct ContextWiring {
    builder: Option<ContextEngineBuilder>,
    tools: Vec<Arc<dyn Tool>>,
    /// The node-scoped `/lcm` command provider (resolves the per-session engine via the bank cache),
    /// folded into the node command registry. `None` for the `Budgeted` fallback.
    command_provider: Option<CommandProviderHandle>,
}

impl ContextWiring {
    fn off() -> Self {
        Self {
            builder: None,
            tools: Vec::new(),
            command_provider: None,
        }
    }
}

/// Build the default §10 context wiring the config selected. `Lcm` opens a shared [`LcmBanks`] cache
/// so the per-session context builder and the registered `lcm_*` tools resolve the *same*
/// [`LcmContextEngine`] instance for a given session (shared compaction state + store), exactly as
/// [`MnemosyneBanks`] does for memory. `Budgeted` leaves the engine on the in-core
/// [`BudgetedContextEngine`](daemon_core::BudgetedContextEngine) fallback with no tools.
fn build_context(cfg: &NodeConfig, aux: Arc<dyn Provider>) -> ContextWiring {
    match cfg.context_engine {
        ContextEngineKind::Budgeted => ContextWiring::off(),
        ContextEngineKind::Lcm => {
            let persist = cfg.persist_providers();
            // The base config carries the *root*; the bank cache re-roots `data_dir` to
            // `<data_root>/<profile>/` per resolved profile (in-memory banks ignore the path).
            let mut lcm_cfg = if persist {
                LcmConfig {
                    data_dir: cfg.data_root(),
                    bank: "default".to_string(),
                    ..LcmConfig::default()
                }
            } else {
                LcmConfig::in_memory()
            };
            // Inject the `[lcm]` tunables (the context crate reads no env itself).
            cfg.lcm.apply(&mut lcm_cfg);
            let banks = Arc::new(LcmBanks::new(
                lcm_cfg,
                persist,
                cfg.data_root(),
                ProfileRef::new(cfg.profile.clone()),
                aux,
            ));
            // The `lcm_*` tool defs are session-independent; enumerate once.
            let tools: Vec<Arc<dyn Tool>> = daemon_context_lcm::tools::tool_defs()
                .into_iter()
                .map(|def| {
                    Arc::new(LcmTool {
                        banks: banks.clone(),
                        def,
                    }) as Arc<dyn Tool>
                })
                .collect();
            let builder: ContextEngineBuilder = {
                let banks = banks.clone();
                Arc::new(move |profile: Option<&ProfileRef>, id: &SessionId| {
                    match banks.get_or_open(profile, id) {
                        Some(lcm) => lcm as Arc<dyn ContextEngine>,
                        None => {
                            tracing::warn!(session = %id,
                                "failed to open LCM context engine for session; using budgeted fallback");
                            Arc::new(daemon_core::BudgetedContextEngine::default())
                                as Arc<dyn ContextEngine>
                        }
                    }
                })
            };
            let command_provider = Some(Arc::new(LcmCommandProvider {
                banks: banks.clone(),
            }) as CommandProviderHandle);
            ContextWiring {
                builder: Some(builder),
                tools,
                command_provider,
            }
        }
    }
}

/// A node-scoped `/lcm` command provider: resolves the calling session's [`LcmContextEngine`] from
/// the shared [`LcmBanks`] cache (so the command observes that session's live compaction state +
/// durable transcript) and delegates to the engine's own [`CommandProvider`] handler. The catalog
/// metadata is the session-independent [`daemon_context_lcm::command_specs`].
struct LcmCommandProvider {
    banks: Arc<LcmBanks>,
}

#[async_trait::async_trait]
impl daemon_core::CommandProvider for LcmCommandProvider {
    fn name(&self) -> &str {
        "lcm"
    }

    fn commands(&self) -> Vec<daemon_core::CommandSpec> {
        daemon_context_lcm::command_specs()
    }

    async fn run_command(
        &self,
        invocation: &daemon_core::CommandInvocation,
        cx: &daemon_core::CommandCx<'_>,
    ) -> Result<daemon_core::CommandOutput, daemon_core::CommandError> {
        let session = cx
            .session
            .as_ref()
            .ok_or(daemon_core::CommandError::MissingSession)?;
        let lcm = self
            .banks
            .get_or_open(None, session)
            .ok_or_else(|| daemon_core::CommandError::Failed("lcm bank unavailable".into()))?;
        lcm.run_command(invocation, cx).await
    }
}

/// A shared, agent-wide LCM bank cache: one per-session [`LcmContextEngine`] over the same profile
/// `lcm.db` (or a per-session in-memory bank when ephemeral). The cache lets the §10 context builder
/// and the `lcm_*` tools resolve the *same* instance for a session, so the tools observe that
/// session's live compaction state and durable transcript.
struct LcmBanks {
    cfg: LcmConfig,
    /// Whether banks are durable (re-rooted per profile under `data_root`) or in-memory.
    persist: bool,
    /// The `<data_dir>` root profile homes hang off; a durable bank opens under `data_root/<profile>`.
    data_root: std::path::PathBuf,
    /// The profile a `None` (legacy single-profile) resolution scopes to — the node launch profile.
    default_profile: ProfileRef,
    aux: Arc<dyn Provider>,
    sessions: Mutex<HashMap<(ProfileRef, SessionId), Arc<LcmContextEngine>>>,
}

impl LcmBanks {
    fn new(
        cfg: LcmConfig,
        persist: bool,
        data_root: std::path::PathBuf,
        default_profile: ProfileRef,
        aux: Arc<dyn Provider>,
    ) -> Self {
        Self {
            cfg,
            persist,
            data_root,
            default_profile,
            aux,
            sessions: Mutex::new(HashMap::new()),
        }
    }

    /// Open (once) and cache the engine for `(profile, session)`. A durable bank is rooted at
    /// `<data_root>/<profile>/`, so two sessions routed to two profiles never share compaction
    /// state or transcript; `profile = None` scopes to the node default (legacy single-profile home).
    fn get_or_open(
        &self,
        profile: Option<&ProfileRef>,
        session: &SessionId,
    ) -> Option<Arc<LcmContextEngine>> {
        let profile = profile
            .cloned()
            .unwrap_or_else(|| self.default_profile.clone());
        let key = (profile.clone(), session.clone());
        let mut sessions = self.sessions.lock().expect("lcm banks poisoned");
        if let Some(existing) = sessions.get(&key) {
            return Some(existing.clone());
        }
        let mut cfg = self.cfg.clone();
        if self.persist {
            cfg.data_dir = self.data_root.join(profile.as_str());
        }
        match LcmContextEngine::open_for_session(cfg, session, self.aux.clone()) {
            Ok(lcm) => {
                let lcm = Arc::new(lcm);
                sessions.insert(key, lcm.clone());
                Some(lcm)
            }
            Err(e) => {
                tracing::warn!(error = %e, profile = %profile, session = %session,
                    "failed to open LCM bank");
                None
            }
        }
    }
}

/// A §12 [`Tool`] adapter that dispatches an `lcm_*` call to the calling session's LCM engine,
/// resolved from the shared [`LcmBanks`] by `cx.session_id` at run time (so the tool and the §10
/// context hooks always operate on the same per-session instance).
struct LcmTool {
    banks: Arc<LcmBanks>,
    def: ToolDef,
}

#[async_trait::async_trait]
impl Tool for LcmTool {
    fn name(&self) -> &str {
        &self.def.name
    }

    fn schema(&self) -> &str {
        &self.def.schema
    }

    async fn run(&self, call: &ToolCall, cx: &TurnCx<'_>) -> ToolOutcome {
        let args = serde_json::from_str(&call.args).unwrap_or(serde_json::Value::Null);
        let result = match self.banks.get_or_open(cx.profile.as_ref(), &cx.session_id) {
            Some(lcm) => lcm.call_tool(&self.def.name, args).await,
            None => {
                serde_json::json!({"status": "error", "error": "lcm bank unavailable"}).to_string()
            }
        };
        ToolOutcome::text(call.call_id.clone(), true, result)
    }
}

/// The default §11 memory wiring: an optional per-session memory *builder*, an optional set of
/// shared (session-independent) providers, and the tools registered into every role registry. The
/// builder/shared split mirrors [`EngineProfile`](daemon_core::EngineProfile) — Mnemosyne is
/// session-scoped (a builder), while `FileMemory` is a frozen, stateless snapshot (shared).
struct MemoryWiring {
    builder: Option<MemoryBuilder>,
    shared: Vec<Arc<dyn MemoryProvider>>,
    tools: Vec<Arc<dyn Tool>>,
    /// The node-scoped `/memory` command provider (resolves the per-session provider via the bank
    /// cache), folded into the node command registry. `None` when memory is off / file-backed.
    command_provider: Option<CommandProviderHandle>,
}

impl MemoryWiring {
    fn off() -> Self {
        Self {
            builder: None,
            shared: Vec::new(),
            tools: Vec::new(),
            command_provider: None,
        }
    }
}

/// A node-scoped `/memory` command provider: resolves the calling session's [`MnemosyneProvider`]
/// from the shared [`MnemosyneBanks`] cache and delegates to the provider's own [`CommandProvider`]
/// handler. The catalog metadata is the session-independent [`daemon_mnemosyne::command_specs`].
struct MemoryCommandProvider {
    banks: Arc<MnemosyneBanks>,
}

#[async_trait::async_trait]
impl daemon_core::CommandProvider for MemoryCommandProvider {
    fn name(&self) -> &str {
        "mnemosyne"
    }

    fn commands(&self) -> Vec<daemon_core::CommandSpec> {
        daemon_mnemosyne::command_specs()
    }

    async fn run_command(
        &self,
        invocation: &daemon_core::CommandInvocation,
        cx: &daemon_core::CommandCx<'_>,
    ) -> Result<daemon_core::CommandOutput, daemon_core::CommandError> {
        let session = cx
            .session
            .as_ref()
            .ok_or(daemon_core::CommandError::MissingSession)?;
        let provider = self.banks.get_or_open(None, session).ok_or_else(|| {
            daemon_core::CommandError::Failed("mnemosyne bank unavailable".into())
        })?;
        provider.run_command(invocation, cx).await
    }
}

/// Build the default §11 memory wiring the config selected. Mnemosyne is the default: a shared
/// [`MnemosyneBanks`] cache opens one per-session provider over the agent-wide bank (correct
/// `session_id` row scoping), and both the per-session memory builder and the registered
/// `mnemosyne_*` tools resolve the *same* per-session instance from it. On any open failure memory
/// is simply left off (the node still runs).
fn build_memory(
    cfg: &NodeConfig,
    embedder: Option<Arc<dyn EmbeddingProvider>>,
    llm: Option<Arc<dyn Provider>>,
) -> MemoryWiring {
    match cfg.memory_provider {
        MemoryProviderKind::None => MemoryWiring::off(),
        MemoryProviderKind::File => match &cfg.memory_file {
            Some(path) => MemoryWiring {
                builder: None,
                shared: vec![Arc::new(FileMemory::load(path)) as Arc<dyn MemoryProvider>],
                tools: Vec::new(),
                command_provider: None,
            },
            None => {
                tracing::warn!("memory_provider=file but DAEMON_MEMORY_FILE is unset; memory off");
                MemoryWiring::off()
            }
        },
        MemoryProviderKind::Mnemosyne => {
            // The base config carries the *root*; the bank cache re-roots `data_dir` to
            // `<data_root>/<profile>/` per resolved profile (in-memory banks ignore the path).
            let mut base = if cfg.persist_providers() {
                MnemosyneConfig {
                    data_dir: cfg.data_root(),
                    ..MnemosyneConfig::default()
                }
            } else {
                MnemosyneConfig::default()
            };
            // Inject the `[mnemosyne]` recall + identity knobs (the memory crate reads no env itself).
            base.recall_mode = cfg.mnemosyne.recall_mode;
            base.llm_conflict_detection = cfg.mnemosyne.llm_conflict_detection;
            base.author_id = cfg.mnemosyne.author_id.clone();
            base.author_type = cfg.mnemosyne.author_type.clone();
            base.channel_id = cfg.mnemosyne.channel_id.clone();
            let banks = Arc::new(MnemosyneBanks::new(
                base,
                cfg.persist_providers(),
                cfg.data_root(),
                ProfileRef::new(cfg.profile.clone()),
                embedder,
                llm,
            ));
            // The `mnemosyne_*` tool defs are session-independent; enumerate them once from a probe
            // instance (its bank is shared, so this is cheap and discarded).
            let tool_defs = match banks.probe_tool_defs() {
                Some(defs) => defs,
                None => {
                    tracing::warn!("failed to open Mnemosyne memory; memory off");
                    return MemoryWiring::off();
                }
            };
            let tools: Vec<Arc<dyn Tool>> = tool_defs
                .into_iter()
                .map(|def| {
                    Arc::new(MemoryProviderTool {
                        banks: banks.clone(),
                        def,
                    }) as Arc<dyn Tool>
                })
                .collect();
            let builder: MemoryBuilder = {
                let banks = banks.clone();
                Arc::new(move |profile: Option<&ProfileRef>, id: &SessionId| {
                    match banks.get_or_open(profile, id) {
                        Some(p) => vec![p as Arc<dyn MemoryProvider>],
                        None => Vec::new(),
                    }
                })
            };
            let command_provider = Some(Arc::new(MemoryCommandProvider {
                banks: banks.clone(),
            }) as CommandProviderHandle);
            MemoryWiring {
                builder: Some(builder),
                shared: Vec::new(),
                tools,
                command_provider,
            }
        }
    }
}

/// Build the optional `metta` symbolic-coprocessor tool the config selected. Disabled by default;
/// when enabled it wires a supervised [`MettaCoprocessor`] over the configured worker binary + state
/// dir + bounds, exposed through the single `metta` [`Tool`]. The worker (and `hyperon`) is a
/// separately-built process spawned lazily on first use — nothing engine-heavy links into the daemon.
fn build_metta_tool(cfg: &NodeConfig) -> Option<Arc<dyn Tool>> {
    if !cfg.metta.enable {
        return None;
    }
    // Persisting nodes default the worker state dir to `<profile_home>/metta`; ephemeral nodes (and
    // an unset dir) keep the worker in-memory so the coprocessor matches the store's durability.
    let state_dir = cfg.metta.state_dir.clone().or_else(|| {
        cfg.persist_providers()
            .then(|| cfg.profile_home().join("metta"))
    });
    let mut client_cfg = MettaClientConfig::new(cfg.metta.worker_bin.clone());
    client_cfg.state_dir = state_dir;
    client_cfg.max_restarts = cfg.metta.max_restarts;
    client_cfg.restart_window = cfg.metta.restart_window;
    let copro = Arc::new(MettaCoprocessor::new(client_cfg));
    let default_bounds = MettaBounds {
        max_steps: cfg.metta.max_steps,
        timeout_ms: cfg.metta.timeout_ms,
        max_results: cfg.metta.max_results,
    };
    let tool = MettaTool::new(copro).with_default_bounds(default_bounds);
    tracing::info!(
        worker = %cfg.metta.worker_bin.display(),
        "metta symbolic coprocessor tool enabled"
    );
    Some(Arc::new(tool) as Arc<dyn Tool>)
}

/// Adapts the host's [`CredentialStore`] to the web tools' [`SecretSource`] seam so the heavy
/// substrate type never enters the tool crate. Reads happen at call time, so a key set later via
/// `CredentialApi` takes effect immediately.
struct CredentialSecrets(Arc<dyn CredentialStore>);

impl SecretSource for CredentialSecrets {
    fn secret(&self, key: &str) -> Option<String> {
        self.0.get(key)
    }
}

/// Build the `web_search` + `web_extract` tools when enabled. `web_search` resolves through Tavily
/// (keyed); `web_extract` tries Firecrawl (keyed) then, when enabled, the dependency-light local
/// readability fallback. Both return their content marked untrusted (the §12 pipeline fences it).
fn build_web_tools(cfg: &NodeConfig, credentials: Arc<dyn CredentialStore>) -> Vec<Arc<dyn Tool>> {
    if !cfg.web.enable {
        return Vec::new();
    }
    let secrets: Arc<dyn SecretSource> = Arc::new(CredentialSecrets(credentials));
    let search = TavilySearch::new(secrets.clone()).with_key_id(cfg.web.tavily_key_id.clone());
    let mut fetchers: Vec<Arc<dyn WebFetchBackend>> = vec![Arc::new(
        FirecrawlFetch::new(secrets).with_key_id(cfg.web.firecrawl_key_id.clone()),
    )];
    if cfg.web.local_fallback {
        fetchers.push(Arc::new(LocalFetch::new()));
    }
    tracing::info!(
        local_fallback = cfg.web.local_fallback,
        "web_search + web_extract tools enabled"
    );
    vec![
        Arc::new(WebSearchTool::new(Arc::new(search))) as Arc<dyn Tool>,
        Arc::new(WebExtractTool::new(fetchers)) as Arc<dyn Tool>,
    ]
}

/// Resolve the aux provider the `vision_analyze` tool describes images through (`[vision]`, `off`
/// by default). `main` resolves the launch profile's builder — the same resolution as `lcm_aux` —
/// so the tool rides whatever provider the node chats with (which must itself accept image input);
/// `genai` builds a dedicated vision-capable model. `None` keeps the tool unregistered.
fn build_vision_provider(
    cfg: &NodeConfig,
    providers: &ProviderRegistry,
    cred_profile: &ProfileRef,
) -> Option<Arc<dyn Provider>> {
    match cfg.vision.kind {
        VisionKind::Off => None,
        VisionKind::Main => providers.builder_for(cred_profile).map(|b| b()),
        VisionKind::Genai => {
            if cfg.vision.model.is_empty() {
                tracing::warn!(
                    "vision provider=genai but [vision].model is unset; vision_analyze off"
                );
                return None;
            }
            let mut provider = GenAiProvider::for_model(cfg.vision.model.clone());
            if let Some(base) = &cfg.vision.base_url {
                provider = provider.with_endpoint(base.clone());
            }
            Some(Arc::new(provider) as Arc<dyn Provider>)
        }
    }
}

/// Build the `vision_analyze` tool when a vision aux provider resolves ([`build_vision_provider`]).
/// The optional `[vision].credential_key` bearer threads into each aux call's `Request::auth`;
/// absent, the provider's environment credential applies (the `lcm_aux` behavior).
fn build_vision_tool(
    cfg: &NodeConfig,
    providers: &ProviderRegistry,
    cred_profile: &ProfileRef,
) -> Option<Arc<dyn Tool>> {
    let aux = build_vision_provider(cfg, providers, cred_profile)?;
    let tool_cfg = VisionToolConfig {
        auth: cfg.vision.credential_key.clone(),
        call_timeout: cfg.vision.timeout,
        max_download_bytes: cfg.vision.max_download_mb * 1024 * 1024,
        max_base64_bytes: cfg.vision.max_base64_mb * 1024 * 1024,
    };
    tracing::info!(provider = ?cfg.vision.kind, "vision_analyze tool enabled");
    Some(Arc::new(VisionAnalyzeTool::new(aux, tool_cfg)) as Arc<dyn Tool>)
}

/// Discover + register Python tools from the `daemon_pytool` worker when enabled. Like the metta
/// coprocessor the worker runs out-of-process over a length-framed cut; one `PyToolHost` backs a
/// proxy [`Tool`] per discovered Python tool and respawns the worker lazily after a crash. Discovery
/// failures degrade gracefully: a warning is logged and no Python tools are registered.
async fn build_python_tools(cfg: &NodeConfig) -> Vec<Arc<dyn Tool>> {
    let py = &cfg.python;
    if !py.enable {
        return Vec::new();
    }

    // Either spawn a standalone worker binary, or `interpreter -m <module>`.
    let (program, mut args) = match &py.worker_bin {
        Some(bin) => (bin.clone(), Vec::new()),
        None => (
            py.interpreter.clone(),
            vec!["-m".to_string(), py.worker_module.clone()],
        ),
    };
    if let Some(dir) = &py.tools_dir {
        args.push("--tools-dir".to_string());
        args.push(dir.display().to_string());
    }

    let mut client_cfg = PyToolConfig::new(program, args);
    // Make the shipped SDK package importable for `-m <module>` (set absolute; an operator needing a
    // richer environment uses `worker_bin` or a venv interpreter).
    if let Some(pkg) = &py.package_path {
        client_cfg
            .env
            .push(("PYTHONPATH".to_string(), pkg.display().to_string()));
    }
    client_cfg.op_timeout = py.op_timeout;
    client_cfg.spawn_timeout = py.spawn_timeout;
    client_cfg.max_restarts = py.max_restarts;
    client_cfg.restart_window = py.restart_window;

    // Discover through the shared `ToolProvider` seam (the same boundary a future MCP provider uses).
    let provider = PyToolProvider::new(client_cfg);
    match provider.discover().await {
        Ok(tools) => {
            tracing::info!(
                count = tools.len(),
                interpreter = %py.interpreter.display(),
                "python tools discovered and registered"
            );
            tools
        }
        Err(err) => {
            tracing::warn!(error = %err, "python tool worker failed to start; no python tools registered");
            Vec::new()
        }
    }
}

/// Discover + register tools from every enabled MCP server (`[[mcp.servers]]`). Each server is
/// surfaced through the shared [`ToolProvider`](daemon_core::ToolProvider) seam (the same boundary as
/// the Python worker); its `mcp__{server}__{tool}` proxies join the registry by name and the
/// connection is (re)established lazily. Discovery failures degrade gracefully: a warning is logged
/// and that server contributes no tools, so one unreachable server never blocks node startup.
async fn build_mcp_tools(cfg: &NodeConfig) -> Vec<Arc<dyn Tool>> {
    use config::{McpServerEntry, McpTransportEntry};
    use daemon_mcp_client::{McpClientProvider, McpServerConfig, McpTransport};

    let mut out: Vec<Arc<dyn Tool>> = Vec::new();
    for entry in &cfg.mcp.servers {
        let McpServerEntry {
            name,
            enable,
            transport,
            op_timeout,
        } = entry;
        if !enable {
            continue;
        }
        let transport = match transport {
            McpTransportEntry::Stdio { command, args, env } => McpTransport::Stdio {
                command: command.clone(),
                args: args.clone(),
                env: env.clone(),
            },
            McpTransportEntry::Http { url } => McpTransport::Http { url: url.clone() },
        };
        let provider = McpClientProvider::new(McpServerConfig {
            name: name.clone(),
            transport,
            op_timeout: *op_timeout,
        });
        match provider.discover().await {
            Ok(tools) => {
                tracing::info!(server = %name, count = tools.len(), "mcp tools discovered and registered");
                out.extend(tools);
            }
            Err(err) => {
                tracing::warn!(server = %name, error = %err, "mcp server discovery failed; no tools registered");
            }
        }
    }
    out
}

/// Build the `browser` tool when enabled and compiled in (the `browser` feature). The supervised
/// Chromium is launched lazily on first use.
#[cfg(feature = "browser")]
fn build_browser_tool(cfg: &NodeConfig) -> Option<Arc<dyn Tool>> {
    use daemon_tool_browser::{BrowserSettings, BrowserSupervisor, BrowserTool};

    if !cfg.browser.enable {
        return None;
    }
    let screenshot_dir = cfg
        .browser
        .screenshot_dir
        .clone()
        .unwrap_or_else(|| cfg.profile_home().join("browser").join("screenshots"));
    let settings = BrowserSettings {
        chrome_path: cfg.browser.chrome_path.clone(),
        headless: cfg.browser.headless,
        screenshot_dir,
        launch_timeout: cfg.browser.launch_timeout,
        auto_dismiss_dialogs: cfg.browser.auto_dismiss_dialogs,
    };
    let supervisor = Arc::new(BrowserSupervisor::new(settings));
    let mut tool = BrowserTool::new(supervisor);
    if cfg.browser.approve_navigation {
        tool = tool.with_navigation_approval();
    }
    tracing::info!(
        headless = cfg.browser.headless,
        approve_navigation = cfg.browser.approve_navigation,
        "browser tool enabled"
    );
    Some(Arc::new(tool) as Arc<dyn Tool>)
}

/// The `browser` feature is off: the tool is never registered (and chromiumoxide is not compiled).
#[cfg(not(feature = "browser"))]
fn build_browser_tool(cfg: &NodeConfig) -> Option<Arc<dyn Tool>> {
    if cfg.browser.enable {
        tracing::warn!(
            "browser tool is enabled in config but the daemon was built without the `browser` \
             feature; the tool will not be registered"
        );
    }
    None
}

/// A shared, agent-wide Mnemosyne bank cache: one per-session [`MnemosyneProvider`] over the same
/// bank database (or a per-session in-memory bank when the node is ephemeral). Memory is scoped at
/// the row level by `session_id`, so all sessions share global/long-term rows while keeping their own
/// session-local working memory. The cache lets the §11 memory builder and the `mnemosyne_*` tools
/// resolve the *same* instance for a given session.
struct MnemosyneBanks {
    base: MnemosyneConfig,
    persist: bool,
    /// The `<data_dir>` root profile homes hang off; a durable bank opens under `data_root/<profile>`.
    data_root: std::path::PathBuf,
    /// The profile a `None` (legacy single-profile) resolution scopes to — the node launch profile.
    default_profile: ProfileRef,
    embedder: Option<Arc<dyn EmbeddingProvider>>,
    llm: Option<Arc<dyn Provider>>,
    sessions: Mutex<HashMap<(ProfileRef, SessionId), Arc<MnemosyneProvider>>>,
}

impl MnemosyneBanks {
    fn new(
        base: MnemosyneConfig,
        persist: bool,
        data_root: std::path::PathBuf,
        default_profile: ProfileRef,
        embedder: Option<Arc<dyn EmbeddingProvider>>,
        llm: Option<Arc<dyn Provider>>,
    ) -> Self {
        Self {
            base,
            persist,
            data_root,
            default_profile,
            embedder,
            llm,
            sessions: Mutex::new(HashMap::new()),
        }
    }

    /// Open (once) and cache the provider for `(profile, session)`. A durable bank is rooted at
    /// `<data_root>/<profile>/`, so two sessions routed to two profiles get isolated memory banks on
    /// disk; `profile = None` scopes to the node default (legacy single-profile home). Returns `None`
    /// if the bank cannot be opened.
    fn get_or_open(
        &self,
        profile: Option<&ProfileRef>,
        session: &SessionId,
    ) -> Option<Arc<MnemosyneProvider>> {
        let profile = profile
            .cloned()
            .unwrap_or_else(|| self.default_profile.clone());
        let key = (profile.clone(), session.clone());
        let mut sessions = self.sessions.lock().expect("mnemosyne banks poisoned");
        if let Some(existing) = sessions.get(&key) {
            return Some(existing.clone());
        }
        let mut cfg = self.base.clone();
        cfg.session_id = session.as_str().to_string();
        let provider = if self.persist {
            cfg.data_dir = self.data_root.join(profile.as_str());
            MnemosyneProvider::open_with_backends(cfg, self.embedder.clone(), self.llm.clone())
        } else {
            // Ephemeral node: a private in-memory bank per (profile, session) (no cross-session
            // sharing, which is acceptable when the session store itself is non-durable).
            crate::ephemeral_mnemosyne(cfg, self.embedder.clone(), self.llm.clone())
        };
        match provider {
            Ok(p) => {
                let p = Arc::new(p);
                sessions.insert(key, p.clone());
                Some(p)
            }
            Err(e) => {
                tracing::warn!(error = %e, profile = %profile, session = %session,
                    "failed to open Mnemosyne bank");
                None
            }
        }
    }

    /// Enumerate the `mnemosyne_*` tool defs from a throwaway probe instance (session-independent).
    fn probe_tool_defs(&self) -> Option<Vec<ToolDef>> {
        let probe = self.get_or_open(None, &SessionId::new("__probe__"))?;
        let defs = probe.tools();
        self.sessions
            .lock()
            .expect("mnemosyne banks poisoned")
            .remove(&(self.default_profile.clone(), SessionId::new("__probe__")));
        Some(defs)
    }
}

/// Open an in-memory Mnemosyne provider for an ephemeral node, with an optional embedder.
fn ephemeral_mnemosyne(
    cfg: MnemosyneConfig,
    embedder: Option<Arc<dyn EmbeddingProvider>>,
    llm: Option<Arc<dyn Provider>>,
) -> daemon_mnemosyne::Result<MnemosyneProvider> {
    use daemon_mnemosyne::Engine;
    let engine = Arc::new(Engine::open_in_memory(cfg)?);
    Ok(MnemosyneProvider::with_backends(engine, embedder, llm))
}

/// A §12 [`Tool`] adapter that dispatches a `mnemosyne_*` call to the calling session's bank,
/// resolved from the shared [`MnemosyneBanks`] by `cx.session_id` at run time (so the tool and the
/// §11 memory hook always operate on the same per-session instance). This keeps memory tools out of
/// the §11 seam (which is about context, not dispatch) while still exposing them to the model.
struct MemoryProviderTool {
    banks: Arc<MnemosyneBanks>,
    def: ToolDef,
}

#[async_trait::async_trait]
impl Tool for MemoryProviderTool {
    fn name(&self) -> &str {
        &self.def.name
    }

    fn schema(&self) -> &str {
        &self.def.schema
    }

    async fn run(&self, call: &ToolCall, cx: &TurnCx<'_>) -> ToolOutcome {
        let args = serde_json::from_str(&call.args).unwrap_or(serde_json::Value::Null);
        let result = match self.banks.get_or_open(cx.profile.as_ref(), &cx.session_id) {
            Some(provider) => provider.call_tool(&self.def.name, args).await,
            None => serde_json::json!({"status": "error", "error": "memory bank unavailable"})
                .to_string(),
        };
        ToolOutcome::text(call.call_id.clone(), true, result)
    }
}

/// Parse the configured `model` string into a [`ModelRef`] for a local engine: an existing local
/// path becomes a [`ModelSource::Local`]; otherwise it is a Hugging Face id (`org/name` for
/// mistral.rs; `org/name/file.gguf` for llama, which names the GGUF to fetch).
fn parse_model_ref(kind: ProviderKind, s: &str) -> Option<ModelRef> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    let engine = match kind {
        ProviderKind::MistralRs => ModelEngine::MistralRs,
        _ => ModelEngine::Llama,
    };
    let path = std::path::Path::new(s);
    if path.exists() {
        return Some(ModelRef::new(
            engine,
            ModelSource::Local {
                path: path.to_path_buf(),
            },
        ));
    }
    match engine {
        ModelEngine::Llama => {
            if s.to_ascii_lowercase().ends_with(".gguf") {
                // `org/name/<file path>` → repo `org/name`, file the remainder.
                let segs: Vec<&str> = s.splitn(3, '/').collect();
                match segs.as_slice() {
                    [org, name, file] => Some(ModelRef::new(
                        engine,
                        ModelSource::hf_file(format!("{org}/{name}"), *file),
                    )),
                    _ => None,
                }
            } else {
                // A bare repo: llama still needs a file at resolve time (surfaced as an error then).
                Some(ModelRef::new(engine, ModelSource::hf(s)))
            }
        }
        ModelEngine::MistralRs => Some(ModelRef::new(engine, ModelSource::hf(s))),
    }
}

/// Build the [`WorkerConfig`] for a local provider from the node config's `[local]` tuning.
fn local_worker_config(cfg: &NodeConfig, engine: Engine) -> WorkerConfig {
    let local = &cfg.infer;
    let mut wc = WorkerConfig::new(local.worker_bin.clone(), engine, cfg.model.clone());
    wc.params = ModelParams {
        n_gpu_layers: local.n_gpu_layers,
        n_ctx: local.n_ctx,
        n_threads: local.n_threads,
        flash_attn: local.flash_attn,
        isq: local.isq.clone(),
        embeddings: false,
        // The paired vision projector is per-model: the switchable provider fills it from the
        // resolved catalog record at load, never from static config.
        mmproj: None,
    };
    wc.max_tokens = local.max_tokens;
    wc.load_timeout = local.load_timeout;
    wc.ttft_timeout = local.ttft_timeout;
    wc.inter_token_timeout = local.inter_token_timeout;
    wc.max_restarts = local.max_restarts;
    wc.restart_window = local.restart_window;
    wc
}

/// Parse the configured embedding `model` string into a [`ModelRef`] for a local engine (mirrors
/// [`parse_model_ref`] but keyed by [`ModelEngine`] directly): an existing local path becomes a
/// [`ModelSource::Local`], `org/name/file.gguf` a fetched GGUF (llama), else a bare HF repo.
fn parse_embed_model_ref(engine: ModelEngine, s: &str) -> Option<ModelRef> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    let path = std::path::Path::new(s);
    if path.exists() {
        return Some(ModelRef::new(
            engine,
            ModelSource::Local {
                path: path.to_path_buf(),
            },
        ));
    }
    match engine {
        ModelEngine::Llama => {
            if s.to_ascii_lowercase().ends_with(".gguf") {
                let segs: Vec<&str> = s.splitn(3, '/').collect();
                match segs.as_slice() {
                    [org, name, file] => Some(ModelRef::new(
                        engine,
                        ModelSource::hf_file(format!("{org}/{name}"), *file),
                    )),
                    _ => None,
                }
            } else {
                Some(ModelRef::new(engine, ModelSource::hf(s)))
            }
        }
        ModelEngine::MistralRs => Some(ModelRef::new(engine, ModelSource::hf(s))),
    }
}

/// Build the optional embedding provider the config selected (`None` keeps recall keyword-only).
///
/// Remote (`genai`) embedders apply the configured credential as the bearer secret and accept a
/// base-URL override (any OpenAI-compatible endpoint). Local embedders resolve their model through
/// the shared [`ModelManager`] (downloading into the warmed cache on first use) and run in a
/// dedicated embedding-mode `daemon-infer` worker. Any failure leaves embeddings off (recall still
/// works, keyword-only).
async fn build_embedder(
    cfg: &NodeConfig,
    manager: &Arc<ModelManager>,
) -> Option<Arc<dyn EmbeddingProvider>> {
    match cfg.embed.kind {
        EmbedKind::Off => None,
        EmbedKind::Genai => {
            if cfg.embed.model.is_empty() {
                tracing::warn!(
                    "embed provider=genai but DAEMON_EMBED_MODEL is unset; embeddings off"
                );
                return None;
            }
            let mut embedder = GenAiEmbedder::openai(cfg.embed.model.clone())
                .with_auth(cfg.credential_key.clone());
            if let Some(base) = &cfg.embed.base_url {
                embedder = embedder.with_endpoint(base.clone());
            }
            if cfg.embed.dims > 0 {
                embedder = embedder.with_dimensions(cfg.embed.dims);
            }
            Some(Arc::new(embedder) as Arc<dyn EmbeddingProvider>)
        }
        EmbedKind::Local => {
            if cfg.embed.model.is_empty() {
                tracing::warn!(
                    "embed provider=local but DAEMON_EMBED_MODEL is unset; embeddings off"
                );
                return None;
            }
            let engine = match cfg.embed.engine.to_ascii_lowercase().as_str() {
                "mistralrs" | "mistral-rs" | "mistral.rs" => ModelEngine::MistralRs,
                _ => ModelEngine::Llama,
            };
            let model_ref = match parse_embed_model_ref(engine, &cfg.embed.model) {
                Some(r) => r,
                None => {
                    tracing::warn!(
                        model = %cfg.embed.model,
                        "could not parse DAEMON_EMBED_MODEL; embeddings off"
                    );
                    return None;
                }
            };
            let artifact = match manager.resolve(&model_ref).await {
                Ok(a) => a,
                Err(e) => {
                    tracing::warn!(error = %e, "failed to resolve embedding model; embeddings off");
                    return None;
                }
            };
            let infer_engine = match engine {
                ModelEngine::MistralRs => Engine::MistralRs,
                ModelEngine::Llama => Engine::Llama,
            };
            let mut wc = WorkerConfig::new(
                cfg.infer.worker_bin.clone(),
                infer_engine,
                artifact.local_path.to_string_lossy().into_owned(),
            );
            wc.params = ModelParams {
                n_gpu_layers: cfg.infer.n_gpu_layers,
                n_ctx: cfg.infer.n_ctx,
                n_threads: cfg.infer.n_threads,
                flash_attn: cfg.infer.flash_attn,
                isq: None,
                embeddings: true,
                mmproj: None,
            };
            // Load from the warmed cache offline (the daemon owns acquisition).
            wc.env.extend(manager.cache().sidecar_env());
            wc.load_timeout = cfg.infer.load_timeout;
            wc.max_restarts = cfg.infer.max_restarts;
            wc.restart_window = cfg.infer.restart_window;
            Some(Arc::new(LocalEmbedder::new(
                wc,
                cfg.embed.dims,
                cfg.embed.model.clone(),
            )) as Arc<dyn EmbeddingProvider>)
        }
    }
}

/// Build the durable store backend the config selected.
fn build_store(backend: &StoreBackend) -> anyhow::Result<Arc<dyn SessionStore>> {
    match backend {
        StoreBackend::Memory => Ok(Arc::new(InMemoryStore::new())),
        StoreBackend::Sqlite { path } => {
            let store = daemon_store::SqliteStore::open(path)
                .map_err(|e| anyhow::anyhow!("opening sqlite store at {}: {e}", path.display()))?;
            Ok(Arc::new(store))
        }
    }
}

/// Environment keys for the first-admin bootstrap (#2). Read directly here (one-shot startup reads)
/// rather than threaded through [`NodeConfig`], keeping this concern out of `config.rs`.
const ADMIN_USERNAME_ENV: &str = "DAEMON_ADMIN_USERNAME";
const ADMIN_PASSWORD_ENV: &str = "DAEMON_ADMIN_PASSWORD";
const ADMIN_PASSWORD_FILE_ENV: &str = "DAEMON_ADMIN_PASSWORD_FILE";

/// Environment keys for the Daemon Cloud attach-credential bootstrap (D2). Read directly here
/// (one-shot startup reads), mirroring the first-admin (#2) pattern above rather than threading a
/// non-config secret through [`NodeConfig`]. The `__`-nesting convention is deliberately NOT used:
/// these are direct-read secret vars, not figment config paths (a `DAEMON_BOOTSTRAP__…` name would
/// read as the config path `bootstrap.…`, which `NodeConfig` does not own and figment would ignore).
const CLOUD_API_KEY_ENV: &str = "DAEMON_CLOUD_API_KEY";
const CLOUD_API_KEY_FILE_ENV: &str = "DAEMON_CLOUD_API_KEY_FILE";

/// Resolve the first-admin seeding policy from the environment: **env-first** — if
/// `DAEMON_ADMIN_USERNAME` is set, require a password from `DAEMON_ADMIN_PASSWORD` or
/// `DAEMON_ADMIN_PASSWORD_FILE` and refuse an empty/whitespace one (never seed `admin`/`<blank>`);
/// otherwise auto-generate. Factored out so it is unit-testable without touching the store.
fn resolve_admin_seed() -> anyhow::Result<daemon_auth::AdminSeed> {
    let username = std::env::var(ADMIN_USERNAME_ENV)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    let Some(username) = username else {
        return Ok(daemon_auth::AdminSeed::Generate);
    };
    let password = match std::env::var(ADMIN_PASSWORD_ENV).ok().filter(|s| !s.is_empty()) {
        Some(p) => p,
        None => match std::env::var(ADMIN_PASSWORD_FILE_ENV)
            .ok()
            .filter(|s| !s.is_empty())
        {
            Some(path) => std::fs::read_to_string(&path).map_err(|e| {
                anyhow::anyhow!("reading {ADMIN_PASSWORD_FILE_ENV} ({path}): {e}")
            })?,
            None => anyhow::bail!(
                "{ADMIN_USERNAME_ENV} is set without a password; set {ADMIN_PASSWORD_ENV} or {ADMIN_PASSWORD_FILE_ENV}"
            ),
        },
    };
    let password = password.trim().to_string();
    if password.is_empty() {
        anyhow::bail!("first-admin password is empty/whitespace; refusing to seed an admin");
    }
    Ok(daemon_auth::AdminSeed::Explicit { username, password })
}

/// Resolve the Daemon Cloud attach key for the D2 credential-store seed: **env-first** —
/// `DAEMON_CLOUD_API_KEY`, else the (trimmed) contents of the file named by
/// `DAEMON_CLOUD_API_KEY_FILE`. `Ok(None)` when neither source is set (no cloud credential is
/// seeded — a keyless boot is a supported state; a turn against the unconfigured profile fails
/// clearly). A source that IS set but blank/whitespace is a deliberate misconfiguration and is
/// refused (never seed an empty bearer). Factored out so it is unit-testable without a store.
fn resolve_cloud_api_key() -> anyhow::Result<Option<String>> {
    let direct = std::env::var(CLOUD_API_KEY_ENV)
        .ok()
        .filter(|s| !s.is_empty());
    let raw = match direct {
        Some(key) => key,
        None => match std::env::var(CLOUD_API_KEY_FILE_ENV)
            .ok()
            .filter(|s| !s.is_empty())
        {
            Some(path) => std::fs::read_to_string(&path)
                .map_err(|e| anyhow::anyhow!("reading {CLOUD_API_KEY_FILE_ENV} ({path}): {e}"))?,
            None => return Ok(None),
        },
    };
    let key = raw.trim().to_string();
    if key.is_empty() {
        anyhow::bail!(
            "{CLOUD_API_KEY_ENV}/{CLOUD_API_KEY_FILE_ENV} is set but empty/whitespace; \
             refusing to seed an empty Daemon Cloud credential"
        );
    }
    Ok(Some(key))
}

/// Idempotently seed the first admin (delegates the empty-table guard + create to
/// [`daemon_auth::AuthStore::seed_first_admin_if_empty`]). For the auto-generated path, emit the
/// password EXACTLY ONCE — to stderr and to a `0600` file under the data dir. This one-time
/// emission is the sole deliberate secret-print exception; it is never routed through `tracing`
/// (so it stays out of structured logs/journald) and never enters the audit journal.
fn seed_first_admin_if_empty(
    auth_store: &daemon_auth::AuthStore,
    cfg: &NodeConfig,
) -> anyhow::Result<()> {
    let seed = resolve_admin_seed()?;
    let Some(created) = auth_store.seed_first_admin_if_empty(seed)? else {
        return Ok(()); // users already exist — idempotent no-op
    };
    match created.generated_password {
        // Operator-supplied identity: they already know the password; log the id only (no secret).
        None => tracing::info!(username = %created.username, "seeded first admin from environment"),
        // Auto-generated: emit once to a 0600 file + stderr.
        Some(password) => emit_generated_admin(cfg, &created.username, &password),
    }
    Ok(())
}

/// Write the auto-generated first-admin credentials to a `0600` file under the data dir and print
/// them once to stderr. Best-effort on the file (a write failure warns but never blocks startup —
/// stderr still carries the secret). Called only on a fresh node with no configured admin.
fn emit_generated_admin(cfg: &NodeConfig, username: &str, password: &str) {
    use std::io::Write as _;

    let path = cfg.data_dir.join("first-admin-credentials.txt");
    let contents = format!(
        "daemon first-admin bootstrap (delete this file once you have saved the password)\n\
         username: {username}\npassword: {password}\n"
    );
    let mut opts = std::fs::OpenOptions::new();
    opts.create(true).write(true).truncate(true);
    // Owner-only on unix; windows has no mode bits (the file inherits the data dir's ACL).
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        opts.mode(0o600);
    }
    match opts.open(&path) {
        Ok(mut f) => {
            if let Err(e) = f.write_all(contents.as_bytes()) {
                tracing::warn!(error = %e, path = %path.display(), "writing first-admin credentials file");
            }
        }
        Err(e) => {
            tracing::warn!(error = %e, path = %path.display(), "creating first-admin credentials file");
        }
    }
    // The one deliberate secret print: stderr, once, on first boot only.
    eprintln!(
        "\n==== daemon first-admin bootstrap (generated) ====\n  \
         username: {username}\n  password: {password}\n  \
         (also written 0600 to {})\n  \
         Save it now and log in over TLS/SCRAM — it will not be shown again.\n\
         ==================================================\n",
        path.display()
    );
}

/// Register the shutdown-signal listeners and return the future that resolves (to the signal
/// name, for the shutdown log line) when one arrives: SIGINT (`ctrl_c`) everywhere, plus SIGTERM
/// on unix — container runtimes (`docker stop`, Fly Machines, systemd) send SIGTERM first, so it
/// must trip the same graceful shutdown instead of running into the stop timeout + SIGKILL.
/// SIGTERM registration happens at the *call* (the top of `run_as_host`), not at the await: a
/// stop that lands while the node is still assembling is queued on the signal stream instead of
/// hitting the default disposition (killed, unclean exit).
#[cfg(unix)]
fn shutdown_signal(
) -> anyhow::Result<impl std::future::Future<Output = anyhow::Result<&'static str>>> {
    use tokio::signal::unix::{signal, SignalKind};
    let mut sigterm = signal(SignalKind::terminate())?;
    Ok(async move {
        tokio::select! {
            result = tokio::signal::ctrl_c() => {
                result?;
                Ok("SIGINT")
            }
            _ = sigterm.recv() => Ok("SIGTERM"),
        }
    })
}

/// Non-unix fallback: only `ctrl_c` is portable (registered lazily on first poll, as before).
#[cfg(not(unix))]
fn shutdown_signal(
) -> anyhow::Result<impl std::future::Future<Output = anyhow::Result<&'static str>>> {
    Ok(async {
        tokio::signal::ctrl_c().await?;
        Ok("ctrl_c")
    })
}

/// Ensure the data directory exists before anything under it is opened — the sqlite store,
/// credentials.json, the auth db, blobs, workspaces, revisions, profiles, and checkpoints all
/// hang off it, and none of their opens creates parent directories. Creation is recursive, and
/// on unix every directory *created here* is private (0700 — the tree holds `auth.sqlite` and
/// journal seeds); a directory that already exists is left completely untouched (recursive
/// create skips existing components and never chmods), so operator-managed setups keep their
/// permissions. A failure is an early, path-naming boot error.
fn ensure_data_dir(dir: &std::path::Path) -> anyhow::Result<()> {
    let mut builder = std::fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder
        .create(dir)
        .map_err(|e| anyhow::anyhow!("creating data dir {}: {e}", dir.display()))
}

/// Assemble and run the default host node, serving the unified surface over a Unix socket until
/// a shutdown signal (SIGINT/`ctrl_c`, or SIGTERM on unix) trips a graceful shutdown. The wiring
/// itself lives in [`daemon_node::assemble`]; this role only builds the policy inputs (store,
/// credentials, provider registry, engine tunables).
async fn run_as_host(cfg: NodeConfig) -> anyhow::Result<()> {
    // Listen for shutdown signals from the very start (see `shutdown_signal` on why this is
    // registered before any listener is bound rather than where it is awaited below).
    let shutdown = shutdown_signal()?;
    // Boot resolution (no silent mock default): an unset provider boots UNCONFIGURED (`None` — the
    // node installs `UnconfiguredProvider` and serves; a turn then fails clearly). An explicitly-set
    // networked provider (`genai`/`daemon_api`) still requires a model (else a deliberate misconfig
    // aborts). A credential is NOT required at boot — it arrives per-profile via `CredentialSet`. The
    // resolved `Option<ProviderKind>` is threaded into provider construction below.
    let provider_kind = cfg.resolve_for_host()?;
    // The ONE data-dir creation point for the host role: everything below (store, credential
    // store, auth db, blobs, workspaces, ...) opens paths under it and assumes it exists.
    ensure_data_dir(&cfg.data_dir)?;
    let store = build_store(&cfg.store_backend())?;

    // The persisted credential store backing the `CredentialApi` surface and the owner authority.
    // Durable nodes persist secrets under the data root; the ephemeral default keeps them in memory.
    let credential_store: Arc<dyn CredentialStore> = if cfg.persist_providers() {
        Arc::new(FileCredentialStore::open(
            cfg.data_dir.join("credentials.json"),
        )?)
    } else {
        Arc::new(MemCredentialStore::new())
    };
    // Seed the launch-configured key so existing launches keep authenticating until a GUI sets one.
    // Skip when empty (mock/local need no credential): we no longer mint a placeholder secret.
    if !cfg.credential_key.is_empty() {
        credential_store
            .set(&cfg.profile, &cfg.credential_key)
            .map_err(|e| anyhow::anyhow!("seeding credential: {e}"))?;
    }
    // D2: first-boot Daemon Cloud attach-credential seed. When the provisioner injects
    // `DAEMON_CLOUD_API_KEY[_FILE]`, idempotently seed the credential-store entry for this node's
    // profile so a hosted node routes inference through the metered gateway with no GUI setup.
    // `CredentialStore::set` is create-or-update, so a rotated secret + restart re-seeds (the §9.4
    // rotation flow) and the same secret is a no-op — never a duplicate. The seed is
    // ENV-AUTHORITATIVE: while the attach key is present it wins over a same-profile GUI-set key on
    // every boot (the control plane owns the attach credential; BYOK belongs on another profile).
    // Unset never scrubs an existing credential (mirrors the first-admin seed). The key value is
    // NEVER logged.
    let cloud_api_key = resolve_cloud_api_key()?;
    if let Some(key) = &cloud_api_key {
        credential_store
            .set(&cfg.profile, key)
            .map_err(|e| anyhow::anyhow!("seeding Daemon Cloud credential: {e}"))?;
        tracing::info!(
            profile = %cfg.profile,
            "seeded Daemon Cloud credential from environment (D2)"
        );
    }

    // Credentials: a PER-PROFILE owner broker over the credential store, brokered into *every*
    // engine, uniformly across the durable, interactive, and fleet-child construction paths
    // (host-spec §6). Per-profile provisioning means a GUI `CredentialSet` on any profile reaches
    // that profile's sessions, so onboarding never depends on a launch-configured profile name.
    let owner_broker = build_multi_profile_broker(&cfg.credential_key, credential_store.clone());
    let owner: Arc<dyn CredentialBroker> = owner_broker.clone();
    let cred_profile = ProfileRef::new(cfg.profile.clone());
    let credentials: CredentialBuilder = {
        let owner = owner.clone();
        Arc::new(move || {
            Arc::new(BrokeredCredentialProvider::new(owner.clone(), None))
                as Arc<dyn CredentialProvider>
        })
    };

    // B4 audit-to-journal: drain the owner authority's credential audit (request/grant/use/revoke/
    // rotate) into the verifiable journal on the production path — keyed to a per-node credential
    // stream and sealed under the node's seed-derived signer (the same seal key the placed-child and
    // transport paths use), so "who acquired which credential when" rides the tamper-evident chain.
    let cred_audit_signer = Arc::new(
        cfg.journal_seed
            .map(|seed| daemon_telemetry::TraceSigner::from_seed(&seed))
            .unwrap_or_else(daemon_telemetry::TraceSigner::generate),
    );
    let cred_audit_drain = daemon_host::spawn_credential_audit_drain(
        owner_broker.clone() as Arc<dyn CredentialAuditDrain>,
        store.clone(),
        cred_audit_signer,
        JournalStreamId::unit(&UnitId::new("node-credentials")),
        std::time::Duration::from_secs(5),
    );

    // Model management: the daemon owns search + acquisition + caching + catalog for the local
    // engines (unified across llama.cpp + mistral.rs). Built unconditionally so the `ModelApi`
    // surface works even on a remote-only node (the GUI can browse/download regardless).
    let manager = Arc::new(
        ModelManager::new(ManagerConfig {
            cache_dir: cfg.models.cache_dir.clone(),
            // HOME-less environments (containers/microvms): when neither `[models].cache_dir` nor
            // the `HF_*`/XDG/`HOME` precedence resolves, cache under the daemon's own data dir
            // (standard hub layout) instead of depending on a home directory existing.
            fallback_cache_dir: Some(cfg.data_dir.join("huggingface").join("hub")),
            registry_path: cfg.models.registry_path.clone(),
            endpoint: cfg.models.endpoint.clone(),
            // Offline quantization runs out-of-process via the llama-enabled inference worker; reuse
            // the configured worker binary (it has the `quantize` subcommand when built with llama).
            quantize_worker_bin: Some(cfg.infer.worker_bin.clone()),
        })
        .await?,
    );
    let active = manager.active_handle();
    // Seed the configured model as the active selection for a local provider, so resolve-before-load
    // downloads it into the shared cache on first use.
    if let Some(kind @ (ProviderKind::LlamaCpp | ProviderKind::MistralRs)) = provider_kind {
        if let Some(model_ref) = parse_model_ref(kind, &cfg.model) {
            active.set(cfg.profile.clone(), model_ref);
        }
    }

    // Provider selection seam: Mock is the zero-config default; a real networked provider drops in
    // via `set_default(...)` without touching the engine or the construction sites. The API key
    // flows per-call through the credential broker (the lease secret -> `Request.auth`), so a real
    // provider builder needs only the base URL + model.
    let providers = build_providers(&cfg, provider_kind, &manager, &active);

    // The default context engine (§10, LCM) and memory providers (§11, Mnemosyne) wired into every
    // engine this node builds, with their `mnemosyne_*` tools registered on the shared registry. Both
    // are per-session builders (LCM keeps per-session compaction state; Mnemosyne scopes by
    // `session_id`), so concurrent sessions never share mutable provider state.
    // LCM summarizes through the same default provider the agent uses: resolve the profile's builder
    // (falling back to the unconfigured provider — never a silent mock; with no provider the context
    // layer's summarization simply errors and LCM stays on its deterministic baseline).
    let lcm_aux: Arc<dyn Provider> = providers
        .builder_for(&cred_profile)
        .map(|b| b())
        .unwrap_or_else(|| Arc::new(UnconfiguredProvider::new()) as Arc<dyn Provider>);
    let context = build_context(&cfg, lcm_aux);
    // The optional embedding backend (Mnemosyne vector recall), reusing the shared `ModelManager`
    // for local-model acquisition. `Off` by default — recall stays keyword-only.
    let embedder = build_embedder(&cfg, &manager).await;
    // Mnemosyne's optional LLM backend for structured extraction + sleep summarization, resolved the
    // same way as `lcm_aux` (the profile's builder, falling back to the unconfigured provider — never
    // a silent mock). With no provider the knowledge layer stays on its deterministic regex/AAAK
    // baselines.
    let mnemosyne_llm: Arc<dyn Provider> = providers
        .builder_for(&cred_profile)
        .map(|b| b())
        .unwrap_or_else(|| Arc::new(UnconfiguredProvider::new()) as Arc<dyn Provider>);
    let memory = build_memory(&cfg, embedder, Some(mnemosyne_llm));
    // Both context (`lcm_*`) and memory (`mnemosyne_*`) tools register on every role registry.
    let mut extra_tools: Vec<Arc<dyn Tool>> = memory
        .tools
        .iter()
        .cloned()
        .chain(context.tools.iter().cloned())
        .collect();

    // The optional MeTTa symbolic-coprocessor tool (opt-in, like the HTTP/MCP surfaces). When
    // enabled it joins the lcm/mnemosyne tools on every role registry; its supervised worker is
    // spawned lazily on first use.
    if let Some(metta_tool) = build_metta_tool(&cfg) {
        extra_tools.push(metta_tool);
    }

    // Core chat tools registered on every role: the per-session `todo` planner and the `clarify`
    // human-in-the-loop ask (both dependency-light, so they are always on).
    extra_tools.push(Arc::new(TodoTool::new()) as Arc<dyn Tool>);
    extra_tools.push(Arc::new(ClarifyTool::new()) as Arc<dyn Tool>);

    // The skills subsystem (opt-out via `[skills].enable = false`): the `skill_*` tools join every
    // role registry, and the progressive-disclosure index is folded into the stable system-prompt
    // tier (`prompt_sources`). The background `skill_review` curator activates only when the engine's
    // `skill_review_interval` is non-zero (see `[engine]`/`DAEMON_SKILL_REVIEW_INTERVAL`).
    // The append-only revision log backing profile + skill versioning. Durable nodes persist it
    // under the data dir (next to `profiles/` and `skills/`); ephemeral nodes run without history.
    let revisions: Option<Arc<dyn daemon_common::RevisionLog>> = if cfg.persist_providers() {
        match FileRevisionLog::open(cfg.data_dir.join("revisions")) {
            Ok(log) => Some(Arc::new(log) as Arc<dyn daemon_common::RevisionLog>),
            Err(e) => {
                tracing::warn!(error = %e, "opening revision log failed; versioning disabled");
                None
            }
        }
    } else {
        None
    };

    let prompt_sources: Vec<Arc<dyn daemon_core::StablePromptSource>> = Vec::new();
    // The skills subsystem is now resolved **per profile (per agent)**, not built once over the
    // launch agent and shared: a `SkillsProvider` yields an `Arc<SkillStore>` rooted at each agent's
    // `<data_dir>/<id>/skills`, and the engine path resolves each session's own `skill_*` tools +
    // index through `skills_resolver` keyed on the session's profile id. The provider is also bound
    // into the node's api surface for skill versioning / distribution / curation.
    let mut skills_provider: Option<Arc<daemon_skills::SkillsProvider>> = None;
    let mut skills_resolver: Option<daemon_node::SkillsResolver> = None;
    if cfg.skills.enable {
        // `[skills].dir` (when set) is the legacy single-dir override shared by every profile; the
        // default is per-profile (`<data_dir>/<id>/skills`), so two agents never share a library.
        let mut provider = match &cfg.skills.dir {
            Some(dir) => daemon_skills::SkillsProvider::fixed(dir.clone()),
            None => daemon_skills::SkillsProvider::per_profile(cfg.data_root()),
        };
        // Version skill writes (incl. the agent's own background-review edits) when versioning is on.
        if let Some(revisions) = &revisions {
            provider = provider.with_revisions(revisions.clone());
        }
        // The per-profile `.usage.json` usage + lifecycle sidecar, co-located with each agent's
        // skills dir (the curator's system of record).
        provider = provider.with_usage(Arc::new(|root: &std::path::Path| {
            Arc::new(daemon_skills::FileSkillUsageLog::open(root))
                as Arc<dyn daemon_common::SkillUsageLog>
        }));
        let provider = Arc::new(provider);
        // The engine-path resolver: each session's engine gets its own profile's tools + index.
        let resolver_provider = provider.clone();
        skills_resolver = Some(Arc::new(move |pref: &ProfileRef| {
            let store = resolver_provider.for_profile(pref.as_str());
            daemon_node::ResolvedSkills {
                tools: daemon_tool_skill::skill_tools(store.clone()),
                index: Arc::new(daemon_skills::SkillsPromptSource::new(store).enabled(true))
                    as Arc<dyn daemon_core::StablePromptSource>,
            }
        }) as daemon_node::SkillsResolver);
        skills_provider = Some(provider);
    }

    // The optional web tools (`web_search`/`web_extract`, opt-in). Keys are read live from the
    // credential store, so a GUI-set Tavily/Firecrawl key applies without a restart.
    extra_tools.extend(build_web_tools(&cfg, credential_store.clone()));

    // The optional `vision_analyze` tool (`[vision]`, off by default): images are described by a
    // vision-capable aux provider — the launch profile's own provider (`main`, the `lcm_aux`
    // resolution) or a dedicated genai model (`genai`).
    if let Some(vision_tool) = build_vision_tool(&cfg, &providers, &cred_profile) {
        extra_tools.push(vision_tool);
    }

    // The optional Python tools (opt-in, `daemon_pytool` worker). Discovered up-front so each Python
    // tool joins the registry by name; the worker process itself is (re)spawned lazily on first call.
    extra_tools.extend(build_python_tools(&cfg).await);

    // The optional MCP tools (opt-in per `[[mcp.servers]]`). Discovered up-front through the same
    // `ToolProvider` seam; each server's `mcp__{server}__{tool}` proxies join the registry by name
    // and the connection is (re)established lazily. Gated by `ProfileSpec.tool_allowlist` like any
    // other tool.
    extra_tools.extend(build_mcp_tools(&cfg).await);

    // The optional `browser` tool — only available when the daemon is built with the `browser`
    // feature (which compiles chromiumoxide); a no-op otherwise.
    if let Some(browser_tool) = build_browser_tool(&cfg) {
        extra_tools.push(browser_tool);
    }

    let host_config = HostConfig {
        partition: cfg.partition,
        dispatch_interval: cfg.dispatch_interval,
        scan_interval: cfg.scan_interval,
        ..HostConfig::default()
    };

    // The profile store backing the `ProfileApi` surface + per-session engine resolution. It
    // persists alongside the durable subsystem databases when the node is durable (sqlite), else it
    // is in-memory (the ephemeral default). The launch config is seeded as the active default only
    // on a store with no profiles at all (first boot) — once a GUI/operator replaces and deletes
    // the seeded placeholder, a reboot never resurrects it.
    let profile_store: Arc<dyn ProfileStore> = if cfg.persist_providers() {
        Arc::new(FileProfileStore::open(cfg.data_dir.join("profiles"))?)
    } else {
        Arc::new(MemProfileStore::new())
    };
    profile_store
        .seed(default_profile_spec(
            &cfg,
            provider_kind,
            cloud_api_key.is_some(),
        ))
        .map_err(|e| anyhow::anyhow!("seeding default profile: {e}"))?;
    // The per-session provider resolution seam: maps the active profile bundle onto a provider
    // client (so a GUI can switch model/provider live).
    let provider_resolver: ProviderResolver = {
        let cfg = cfg.clone();
        let manager = manager.clone();
        let active = active.clone();
        Arc::new(move |spec: &ProfileSpec| {
            provider_builder_for(spec, &cfg, provider_kind, &manager, &active)
        })
    };

    let routing = build_routing_registry(&cfg.routing);
    // The §12 tool-checkpoint store: a workspace checkpoint is recorded before each mutating tool
    // runs (rewindable via the `Checkpoint{List,Rewind}` control ops). The ledger lives under the
    // data dir so rewind points survive a restart.
    let checkpoints: Option<Arc<dyn daemon_core::CheckpointStore>> = Some(Arc::new(
        daemon_core::LocalCheckpointStore::new(cfg.data_dir.join("checkpoints")),
    )
        as Arc<dyn daemon_core::CheckpointStore>);

    // Register the interactive-auth families this node exposes over the wire `AuthApi` (the
    // client-driven SSO/OAuth2 login seam). The Matrix SSO factory is registered whenever the matrix
    // transport is enabled, so a decoupled GUI can drive `auth_begin`/`auth_complete` to mint and bind
    // an account's session — keyed by the same per-account store root the transport's `serve` uses.
    let auth_factories: Vec<Arc<dyn daemon_host::AuthFlowFactory>> = if cfg.matrix.enabled {
        vec![Arc::new(daemon_matrix::MatrixAuthFlowFactory::new(
            cfg.matrix.store_root.clone(),
        ))]
    } else {
        vec![]
    };

    // Materialize the workspace root eagerly (the way FileBlobStore::open creates its own dir):
    // nothing else creates it, so after a state wipe every `fs_list` on the advertised `workspace`
    // root would fail with read_dir ENOENT. Per-session sandboxes stay lazy.
    let ws_root = cfg.workspace_root();
    std::fs::create_dir_all(&ws_root)
        .map_err(|e| anyhow::anyhow!("creating workspace root {}: {e}", ws_root.display()))?;

    let AssembledNode {
        node,
        handle,
        signer,
        ..
    } = assemble(NodeAssembly {
        store: store.clone(),
        partition: cfg.partition,
        host_config,
        providers,
        credentials: Some(credentials),
        profile: cred_profile,
        engine_config: cfg.engine,
        journal_seed: cfg.journal_seed,
        nesting_depth: cfg.nesting_depth,
        context: None,
        context_builder: context.builder,
        memory: memory.shared,
        memory_builder: memory.builder,
        extra_tools,
        models: Some(manager.clone()),
        profiles: Some(profile_store),
        provider_resolver: Some(provider_resolver),
        credential_store: Some(credential_store),
        cloud_catalog: Some(Arc::new(GenAiCloudCatalog)),
        prompt_sources,
        revisions,
        skills: skills_provider,
        skills_resolver,
        routing,
        checkpoints,
        auth_factories,
        workspace_root: Some(cfg.workspace_root()),
        blob_root: Some(cfg.blob_root()),
    });
    // Build the daemon-authoritative command catalog (`command_list`/`command_invoke`): the built-in
    // node-op commands unified with the provider commands the context engine (`/lcm`) and memory
    // provider (`/memory`) contribute, each resolving its per-session bank at invocation time. Bound
    // post-assembly because these providers wrap node-owned bank caches.
    {
        let mut command_registry = CommandRegistry::with_builtins();
        if let Some(provider) = context.command_provider {
            command_registry.register_provider(provider);
        }
        if let Some(provider) = memory.command_provider {
            command_registry.register_provider(provider);
        }
        node.set_commands(Arc::new(command_registry));
    }
    // Load any durable chat→session routing pins (§5.9, I5) into the live registry so resolve-first
    // overrides survive restarts; rides the same hot-reload seam profile/auth changes use.
    node.load_routing_pins().await;
    tracing::info!("daemon host node started");

    // The identity store backing the authenticator (created if absent), shared by every
    // auth-required transport (the Unix socket when local trust is disabled, and TCP/TLS always).
    let auth_db = cfg.auth_db();
    let auth_store = Arc::new(
        daemon_auth::AuthStore::open(&auth_db)
            .map_err(|e| anyhow::anyhow!("opening auth store {}: {e}", auth_db.display()))?,
    );

    // #2 first-admin bootstrap: seed exactly one admin when the users table is empty (idempotent).
    // The default `local_trust=system` operator is already admin without SASL; this human seed is
    // for TLS/networked SCRAM login and to make the admin `AccessControl` API usable.
    seed_first_admin_if_empty(&auth_store, &cfg)?;

    // #3 bind the identity store + a shared auth-audit sink onto BOTH the node and the transport
    // authenticator (the SAME `Arc`s), so admin `AccessControl` ops resolve (instead of
    // `Unsupported`) and login/denial/admin events chain together on the verifiable `node-auth`
    // journal stream. `NodeApiImpl` is `Clone`; deref-clone-rewrap (see conformance positive_e2e),
    // reassigning `node` before any listener spawn. The audit shares the node's durable `store` +
    // journal `signer` so its records land on — and verify against — the node's own journal.
    let auth_audit = daemon_host::AuthAudit::shared(store.clone(), signer.clone());
    let node = Arc::new(
        (*node)
            .clone()
            .with_auth_store(auth_store.clone())
            .with_auth_audit(auth_audit.clone()),
    );
    // The store handle is cloned in (not moved): the web front's `/healthz` readiness probe below
    // keeps its own reference for the auth check.
    let authenticator =
        Arc::new(daemon_host::Authenticator::new(auth_store.clone()).with_audit(auth_audit));
    // B5: `[api].local_trust` defaults to `system` — the Unix socket / FFI / in-process HTTP run as
    // the deliberate full-trust principal. Disable it to require SCRAM on the Unix socket and fully
    // gate HTTP. TCP/TLS always requires authentication regardless of this flag.
    let local_trust = cfg.api.local_trust.is_some();

    // Bind the api socket (fresh) and serve the unified surface over it. A managed/user launch may
    // target a nested runtime path (e.g. under $XDG_RUNTIME_DIR) whose parent dir does not exist
    // yet, so ensure the parent exists (and clear any stale socket) before binding.
    #[cfg(unix)]
    let server = {
        prepare_api_socket(&cfg.socket_path)?;
        let listener = tokio::net::UnixListener::bind(&cfg.socket_path)?;
        if local_trust {
            tracing::info!(socket = %cfg.socket_path.display(), "serving daemon-api over unix socket (local trust: system)");
            tokio::spawn(serve_api_unix(listener, node.clone()))
        } else {
            tracing::info!(socket = %cfg.socket_path.display(), "serving daemon-api over unix socket (SCRAM required)");
            tokio::spawn(daemon_host::serve_api_unix_authenticated(
                listener,
                node.clone(),
                authenticator.clone(),
            ))
        }
    };
    // No unix-socket surface on windows: require at least one networked listener so a launch that
    // would serve nothing fails loudly at boot instead of idling unreachable.
    #[cfg(not(unix))]
    if cfg.api.tls_addr.is_none()
        && cfg.api.ws_addr.is_none()
        && cfg.web.addr.is_none()
        && cfg.http_addr.is_none()
    {
        anyhow::bail!(
            "no api surface configured: the unix-socket transport is unavailable on this \
             platform — set [api].ws_addr, [api].tls_addr, [web].addr, or http_addr"
        );
    }

    // The networked TLS/TCP api transport (opt-in via `[api].tls_addr`). TCP always requires
    // authentication (never local-trusted).
    let tls_server = match (&cfg.api.tls_addr, &cfg.api.tls_cert, &cfg.api.tls_key) {
        (Some(addr), Some(cert), Some(key)) => {
            let tls_cfg = daemon_host::ApiTlsConfig {
                cert_path: cert.clone(),
                key_path: key.clone(),
                require_client_cert: cfg.api.require_client_cert,
                client_ca_path: cfg.api.tls_client_ca.clone(),
            };
            let server_config = daemon_host::build_server_config(&tls_cfg)
                .map_err(|e| anyhow::anyhow!("building TLS server config: {e}"))?;
            let tls_listener = tokio::net::TcpListener::bind(addr).await?;
            tracing::info!(
                %addr,
                require_client_cert = cfg.api.require_client_cert,
                "serving daemon-api over TLS/TCP (authentication required)"
            );
            Some(tokio::spawn(daemon_host::serve_api_tls_tcp(
                tls_listener,
                server_config,
                node.clone(),
                authenticator.clone(),
            )))
        }
        (Some(_), _, _) => {
            anyhow::bail!("[api].tls_addr is set but [api].tls_cert / [api].tls_key are missing")
        }
        _ => None,
    };

    // The plain-WebSocket mux carrier (opt-in via `[api].ws_addr`) for browser (Qt WASM) clients:
    // the same CBOR mux, one binary message per frame, subprotocol `daemon-mux`, authentication
    // ALWAYS required (never local-trusted). Browser origins are gated by
    // `[api].ws_allowed_origins`; wss:// terminates at a reverse proxy for now.
    let ws_server = match &cfg.api.ws_addr {
        Some(addr) => {
            let ws_listener = tokio::net::TcpListener::bind(addr).await?;
            tracing::info!(
                %addr,
                allowed_origins = ?cfg.api.ws_allowed_origins,
                "serving daemon-api over WebSocket (subprotocol daemon-mux, authentication required)"
            );
            Some(tokio::spawn(daemon_host::serve_mux_ws(
                ws_listener,
                node.clone(),
                authenticator.clone(),
                cfg.api.ws_allowed_origins.clone(),
            )))
        }
        None => None,
    };

    // The single-origin web front (opt-in via `[web].addr`): ONE listener serving the Qt WASM app
    // bundle (`[web].root`, scanned once at startup) as static files AND the same mux-over-
    // WebSocket carrier on `/ws` — the browser loads the GUI from the daemon and connects back to
    // the same origin, so same-origin upgrades need zero origin config (`[api].ws_allowed_origins`
    // adds extra cross-origin allowance). Static files are public; `/ws` still requires SASL.
    let web_server = match &cfg.web.addr {
        Some(addr) => {
            let root = cfg.web.root.as_ref().ok_or_else(|| {
                anyhow::anyhow!(
                    "[web].addr is set but [web].root is missing (set it to the wasm app bundle directory)"
                )
            })?;
            let site = daemon_host::WebRoot::scan(root).map_err(|e| {
                anyhow::anyhow!(
                    "[web].root {} is not a servable directory: {e}",
                    root.display()
                )
            })?;
            let web_listener = tokio::net::TcpListener::bind(addr).await?;
            // D3: the unauthenticated `GET /healthz` readiness probe on this same listener. The
            // checks are wired here (the binary owns the handles); evaluation is bounded + TTL-
            // cached inside the web front, so infrastructure polling costs at most one pass per
            // window. Depth achieved (honest accounting):
            // - `store`: a live query round-trip via `SessionStore::stats()` — proves the durable
            //   handle + connection lock complete a read (a wedged store hangs into the probe
            //   timeout => 503). Its signature is infallible (the sqlite impl folds SQL errors to
            //   zeros), so corruption/read errors do NOT degrade it; a deeper check needs a
            //   fallible ping (e.g. a `PRAGMA`-backed `SessionStore` health op).
            // - `auth`: `AuthStore::user_count()` — a genuinely fallible COUNT on the (tiny) users
            //   table; proves auth.sqlite answers queries on the data volume.
            // - `journal`: a boot-time fact — the seed-derived journal signer is constructed
            //   before any listener binds (boot fails otherwise), and journal appends ride the
            //   same durable store the `store` check exercises, so there is nothing cheaper to
            //   probe per poll; this flips to a real check if journal init ever becomes lazy.
            let health = {
                let store = store.clone();
                let auth_store = auth_store.clone();
                daemon_host::WebHealth::new()
                    .with_check("store", move || {
                        let store = store.clone();
                        async move {
                            let _ = store.stats().await;
                            Ok(())
                        }
                    })
                    .with_check("auth", move || {
                        let auth_store = auth_store.clone();
                        async move {
                            auth_store
                                .user_count()
                                .map(|_| ())
                                .map_err(|e| e.to_string())
                        }
                    })
                    .with_check("journal", || async { Ok(()) })
            };
            tracing::info!(
                %addr,
                root = %root.display(),
                files = site.len(),
                "serving web app bundle + daemon-api WebSocket at /ws + /healthz readiness (single origin, authentication required)"
            );
            Some(tokio::spawn(daemon_host::serve_web(
                web_listener,
                site,
                node.clone(),
                authenticator.clone(),
                cfg.api.ws_allowed_origins.clone(),
                health,
            )))
        }
        None => None,
    };

    // Optionally bind the in-process HTTP/WS surface (the `daemon-http` adapter), toggled on by a
    // configured bind address (like the MCP surface). It shares the same `Arc<dyn NodeApi>`, so it is
    // just another transport over the one canonical interface — JSON dispatch plus SSE/WS streaming
    // over the merged session event log.
    // Build the transport-adapter registry and drive it from the node (registry-driven lifecycle,
    // daemon-messaging-adapter-spec.md §12.1). Each enabled adapter is registered; `set_adapters`
    // installs the registry on the assembled node and `spawn_adapters` runs every adapter's `serve`
    // with the node as its `api`. The same registry then backs `transport_adapters` /
    // `transport_instances` enumeration and the generic `conv_*`/`member_*` management forwarding.
    //
    // Rooms drives the same `Arc<dyn NodeApi>` as an in-process client; its "homeserver" is the daemon
    // itself (no external accounts/credentials), so it consumes only the durable store. Matrix
    // additionally consumes the host's in-process `AccountProvisioning` seam (the node itself).
    let mut adapter_registry = daemon_host::AdapterRegistry::new();
    if cfg.rooms.enabled {
        tracing::info!("registering internal rooms transport (daemon-rooms)");
        adapter_registry = adapter_registry.with_adapter(daemon_rooms::RoomsAdapter::new(
            store,
            signer.clone(),
            cfg.rooms.clone(),
        ));
    }
    if cfg.matrix.enabled {
        tracing::info!("registering matrix transport (daemon-matrix)");
        let provisioning: Arc<dyn daemon_host::AccountProvisioning> = node.clone();
        adapter_registry = adapter_registry.with_adapter(daemon_matrix::MatrixAdapter::new(
            provisioning,
            cfg.matrix.clone(),
        ));
    }
    node.set_adapters(adapter_registry);
    let adapter_tasks = node.spawn_adapters();

    let http_server = match &cfg.http_addr {
        Some(addr) => {
            let http_listener = tokio::net::TcpListener::bind(addr).await?;
            tracing::info!(%addr, "serving daemon-api over http (json dispatch + sse/ws subscribe)");
            let api: Arc<dyn daemon_api::NodeApi> = node;
            Some(tokio::spawn(async move {
                if let Err(e) = daemon_http::serve_http(http_listener, api, local_trust).await {
                    tracing::warn!(error = %e, "http surface ended");
                }
            }))
        }
        None => None,
    };

    let signal = shutdown.await?;
    tracing::info!(signal, "shutdown signal received; shutting down");
    #[cfg(unix)]
    server.abort();
    if let Some(tls_server) = tls_server {
        tls_server.abort();
    }
    if let Some(ws_server) = ws_server {
        ws_server.abort();
    }
    if let Some(web_server) = web_server {
        web_server.abort();
    }
    if let Some(http_server) = http_server {
        http_server.abort();
    }
    for task in &adapter_tasks {
        task.abort();
    }
    cred_audit_drain.abort();
    handle.shutdown().await;
    // Clear the socket file this run bound (windows never bound one).
    #[cfg(unix)]
    let _ = std::fs::remove_file(&cfg.socket_path);
    Ok(())
}

/// Build the host routing registry (daemon-event-io-spec §5.9) from the resolved `[routing]` config,
/// mapping the declarative rules onto `SessionBinding`s / instance bindings / the node default.
/// Returns `None` when the table is empty, so the node installs no agent-selection routing (routed
/// submits then use `PerThread` naming + the active default profile — the legacy behavior).
fn build_routing_registry(cfg: &RoutingConfig) -> Option<RoutingRegistry> {
    if cfg.is_empty() {
        return None;
    }
    let mut reg = RoutingRegistry::new();
    if let Some(profile) = &cfg.default_profile {
        reg = reg.with_default_profile(ProfileRef::new(profile.clone()));
    }
    for ip in &cfg.instance_profiles {
        reg = reg.bind_instance(
            TransportId::new(ip.transport.clone()),
            ProfileRef::new(ip.profile.clone()),
        );
    }
    for rule in &cfg.routes {
        let transport = match (&rule.transport, &rule.transport_family) {
            (Some(t), _) => TransportPattern::Exact(TransportId::new(t.clone())),
            (None, Some(f)) => TransportPattern::Family(f.clone()),
            (None, None) => TransportPattern::Any,
        };
        let scope = match rule.scope.to_ascii_lowercase().as_str() {
            "dm" => ScopePattern::Dm,
            "group" => ScopePattern::Group {
                chat_glob: rule.chat_glob.clone(),
            },
            "api" => ScopePattern::Api,
            "internal" => ScopePattern::Internal,
            _ => ScopePattern::Any,
        };
        let isolation = match rule.isolation.to_ascii_lowercase().as_str() {
            "per_user" => IsolationPolicy::PerUser,
            "per_chat" => IsolationPolicy::PerChat,
            "shared" => IsolationPolicy::Shared,
            _ => IsolationPolicy::PerThread,
        };
        let mut binding = SessionBinding::new(OriginMatcher { transport, scope }, isolation);
        if let Some(profile) = &rule.profile {
            binding = binding.with_profile(ProfileRef::new(profile.clone()));
        }
        reg = reg.with_binding(binding);
    }
    Some(reg)
}

/// Build the owner end of the credential brokering chain over the persisted credential store
/// (host-spec §7). The authority hands over the stored provider key (the GUI-set secret, falling
/// back to the configured key) as the request bearer for `profile`, so a real provider call carries
/// the live credential. Bearer mode + non-minting: the stored key is the secret, no STS dance.
///
/// Returns both the broker (injected into every engine) and the underlying [`CredentialAuthority`]
/// so the host can drain its audit log into the verifiable journal (B4 audit-to-journal).
fn build_owner_broker(
    profile: &str,
    fallback_key: &str,
    store: Arc<dyn CredentialStore>,
) -> (Arc<dyn CredentialBroker>, Arc<CredentialAuthority>) {
    let signer = Arc::new(CapabilitySigner::generate());
    // The pooled source selects/rotates among the profile's key pool (multi-key) on a rotatable
    // failure, falling back to the launch-configured key when the pool is empty.
    let source = Arc::new(PooledStoreCredentialSource::new(
        store,
        profile,
        fallback_key,
    ));
    let scope = CredScope::new([profile], ["chat", "embed"], Some(1_000));
    let authority = Arc::new(CredentialAuthority::new(
        scope,
        CredMode::Bearer,
        60_000,
        signer,
        source,
    ));
    let broker = Arc::new(OwnerBroker::new(authority.clone())) as Arc<dyn CredentialBroker>;
    (broker, authority)
}

/// Build the per-profile owner broker for the host path: a `CredentialSet` on any profile reaches
/// that profile's sessions (each profile gets a lazily-built authority over a pooled store source,
/// all sharing one node signer). Returned as the concrete type so the caller can coerce it to both
/// the engine-facing `CredentialBroker` and the audit-drain `CredentialAuditDrain`. Bearer mode +
/// the launch-configured `fallback_key` for any profile with no stored key (zero-config bootstrap).
fn build_multi_profile_broker(
    fallback_key: &str,
    store: Arc<dyn CredentialStore>,
) -> Arc<MultiProfileStoreBroker> {
    let signer = Arc::new(CapabilitySigner::generate());
    Arc::new(MultiProfileStoreBroker::new(
        store,
        signer,
        fallback_key,
        ["chat", "embed"],
        Some(1_000),
        CredMode::Bearer,
        60_000,
    ))
}

/// Run as a transport server: host a completing engine unit + an authoritative store, reachable as
/// a `ManagedUnit` over a socket (with the cross-node lease/fence handshake). The engine is built
/// through a *dressed* [`EngineProfile`] (engine tunables + a local owner-broker credential seam,
/// since a transport node is its own authority over its own store) and journals its transcript per
/// turn under a seed-derived signer, so its construction matches the host path.
async fn run_as_transport_server(addr: String) -> anyhow::Result<()> {
    let cfg = NodeConfig::load()?;
    // Fail fast on the transport path too: no silent mock default — a launch must configure a
    // provider (Mock is reachable only via explicit `DAEMON_MODEL_PROVIDER=mock`).
    let _provider_kind = cfg.resolve_for_host()?;
    let store: Arc<dyn SessionStore> = Arc::new(InMemoryStore::new());

    // A transport node owns its store, so it mints its own credentials (the host path's owner
    // broker) rather than brokering from a parent — the engine is therefore not credential-less.
    let cred_store: Arc<dyn CredentialStore> = Arc::new(MemCredentialStore::new());
    if !cfg.credential_key.is_empty() {
        cred_store
            .set(&cfg.profile, &cfg.credential_key)
            .map_err(|e| anyhow::anyhow!("seeding credential: {e}"))?;
    }
    let (owner, _cred_authority) =
        build_owner_broker(&cfg.profile, &cfg.credential_key, cred_store);
    let credentials: CredentialBuilder = {
        let owner = owner.clone();
        Arc::new(move || {
            Arc::new(BrokeredCredentialProvider::new(owner.clone(), None))
                as Arc<dyn CredentialProvider>
        })
    };
    // The configured provider (genai / daemon_api / local / explicit mock), built through the same
    // seam as the host + placed-child paths — not a hardcoded `MockProvider`.
    let provider = build_placed_child_provider(&cfg).await;
    let profile = EngineProfile::new(
        provider,
        Arc::new(ToolRegistry::new()),
        SystemPrompt::new("transport-hosted unit"),
    )
    .with_config(cfg.engine)
    .with_credentials(credentials, ProfileRef::new(cfg.profile.clone()));

    // The unit journals per turn into the local store, keyed by its UnitId, sealed under the
    // config-seeded signer (or an ephemeral key when no seed is configured).
    let unit_id = UnitId::new("u1");
    let signer = Arc::new(
        cfg.journal_seed
            .map(|seed| daemon_telemetry::TraceSigner::from_seed(&seed))
            .unwrap_or_else(daemon_telemetry::TraceSigner::generate),
    );
    let sink = JournalSink::new(store.clone(), signer, JournalStreamId::unit(&unit_id));
    let feeder = Arc::new(JournalFeeder::new(Arc::new(sink)));

    let unit: Arc<dyn ManagedUnit> = Arc::new(EngineUnit::spawn_journaled(
        unit_id.clone(),
        profile.fresh(SessionId::new(unit_id.as_str())),
        Some(feeder),
    ));
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!(%addr, "transport server listening");
    Arc::new(RemoteHost::new(store, unit))
        .serve(listener)
        .await?;
    Ok(())
}

/// The configured provider builder for a placed child (B2), resolved through the same
/// [`build_providers`] seam the in-process node uses so the child runs the real model rather than a
/// mock. Builds the `ModelManager` (needed by the local engines for resolve-before-load; harmless for
/// remote/mock) and seeds the active local model exactly as the node assembly does. Falls back to a
/// completing mock if the model manager cannot be initialized, so the placement path never dies on a
/// model-subsystem error.
async fn build_placed_child_provider(cfg: &NodeConfig) -> daemon_core::ProviderBuilder {
    let unconfigured = || -> daemon_core::ProviderBuilder {
        Arc::new(|| Arc::new(UnconfiguredProvider::new()) as Arc<dyn Provider>)
    };
    // The placed child inherits the parent env; resolve the configured kind. An unset provider, an
    // explicit misconfig, or a model-subsystem error yields the UNCONFIGURED provider (clear
    // turn-time error, never a silent mock) so placement never dies but also never fabricates output.
    let provider_kind = match cfg.resolve_for_host() {
        Ok(Some(kind)) => kind,
        Ok(None) => return unconfigured(),
        Err(e) => {
            tracing::warn!(error = %e, "placed child: provider misconfigured; using unconfigured provider");
            return unconfigured();
        }
    };
    let manager = match ModelManager::new(ManagerConfig {
        cache_dir: cfg.models.cache_dir.clone(),
        // Same HOME-less data-dir fallback as the host role (the child inherits the parent env).
        fallback_cache_dir: Some(cfg.data_dir.join("huggingface").join("hub")),
        registry_path: cfg.models.registry_path.clone(),
        endpoint: cfg.models.endpoint.clone(),
        quantize_worker_bin: Some(cfg.infer.worker_bin.clone()),
    })
    .await
    {
        Ok(manager) => Arc::new(manager),
        Err(e) => {
            tracing::warn!(error = %e, "placed child: model manager init failed; using unconfigured provider");
            return unconfigured();
        }
    };
    let active = manager.active_handle();
    if matches!(
        provider_kind,
        ProviderKind::LlamaCpp | ProviderKind::MistralRs
    ) {
        if let Some(model_ref) = parse_model_ref(provider_kind, &cfg.model) {
            active.set(cfg.profile.clone(), model_ref);
        }
    }
    let providers = build_providers(cfg, Some(provider_kind), &manager, &active);
    providers
        .builder_for(&ProfileRef::new(cfg.profile.clone()))
        .unwrap_or_else(unconfigured)
}

/// The tool registry for a placed child (B2): the always-on, dependency-light core chat tools the
/// node registers on every role (the `todo` planner + the `clarify` HITL ask). The heavier optional
/// subsystems (skills / memory / MCP / Python / web / browser) and their per-session resolvers stay
/// with the parent node; the placed child gets the real provider and the core toolset.
fn build_placed_child_tools(_cfg: &NodeConfig) -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(TodoTool::new()) as Arc<dyn Tool>);
    registry.register(Arc::new(ClarifyTool::new()) as Arc<dyn Tool>);
    registry
}

/// Run as the far side of a placement cut: a completing engine driven over the brokered store. The
/// engine is built from a *dressed* [`EngineProfile`] (engine tunables applied, via
/// [`CoreEngineFactory::from_profile`]) so it shares the host's construction seam rather than a
/// bespoke literal. When the node's journal seed is configured (passed down via `DAEMON_JOURNAL_SEED`
/// by the spawning parent), the child journals its durable transcript **through the parent's brokered
/// store**, sealed under the node's seed-derived signer so the chain verifies under the node's
/// published verifying key. Credentials are brokered over the *same* multiplexed cut: the engine
/// acquires each turn's lease from the parent (falling back to its embedded pool only when the
/// parent serves no authority).
async fn run_as_placed_child() {
    let cfg = match NodeConfig::load() {
        Ok(cfg) => cfg,
        Err(e) => {
            tracing::error!(error = %e, "placed child failed to load config");
            return;
        }
    };
    // B2: the placed child runs the node's **configured** provider (genai / local / mock), built
    // through the same `build_providers` seam as the in-process node, plus the always-on core tools —
    // not a hardcoded `MockProvider`. Credentials are brokered over the multiplexed cut (wired by
    // `run_placed_child*`), so a real provider resolves its API key from the parent's authority.
    let provider = build_placed_child_provider(&cfg).await;
    let registry = build_placed_child_tools(&cfg);
    let profile = EngineProfile::new(
        provider,
        Arc::new(registry),
        SystemPrompt::new("placed child"),
    )
    .with_config(cfg.engine);
    let factory = CoreEngineFactory::from_profile(profile);
    let channel = CutChannel::from_stdio();
    let cred_profile = ProfileRef::new(cfg.profile.clone());

    match cfg.journal_seed {
        Some(seed) => {
            let signer = Arc::new(daemon_telemetry::TraceSigner::from_seed(&seed));
            run_placed_child_journaled(channel, factory, cfg.partition, signer, cred_profile).await;
        }
        None => run_placed_child(channel, factory, cfg.partition, cred_profile).await,
    }
}

/// Prepare the api socket path for a fresh bind: create its parent directory if missing (a
/// managed/user launch may target a nested runtime path, e.g. under `$XDG_RUNTIME_DIR`), refuse to
/// displace a LIVE daemon already serving the path, and clear a stale (dead) socket file left by a
/// previous run. Kept separate from [`run_as_host`] so it is unit-testable without standing up the
/// full host.
#[cfg(unix)]
fn prepare_api_socket(path: &std::path::Path) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| anyhow::anyhow!("creating socket dir {}: {e}", parent.display()))?;
        }
    }
    if !path.exists() {
        return Ok(());
    }
    // Probe before unlink: blindly removing the file would orphan a live daemon (it keeps serving
    // its already-bound listener while every new connect goes to the usurper — the leaked-daemon
    // incident). A Unix-socket connect is local and immediate: success proves a live listener;
    // refused / not-a-socket means the path is stale debris and safe to clear.
    match std::os::unix::net::UnixStream::connect(path) {
        Ok(_) => anyhow::bail!(
            "api socket already bound by a live daemon: {}",
            path.display()
        ),
        Err(_) => std::fs::remove_file(path)
            .map_err(|e| anyhow::anyhow!("removing stale socket {}: {e}", path.display()))?,
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use daemon_auth::AdminSeed;
    use std::sync::Mutex;

    /// The api socket bind path may point at a nested runtime dir that does not exist yet
    /// (managed/user launch); `prepare_api_socket` must create the parent so `bind` succeeds.
    #[cfg(unix)]
    #[test]
    fn prepare_api_socket_creates_missing_parent_dir() {
        let base = std::env::temp_dir().join(format!(
            "daemon-sockprep-{}-{:p}",
            std::process::id(),
            &0u8 as *const u8
        ));
        let _ = std::fs::remove_dir_all(&base);
        let sock = base.join("nested/run/api.sock");
        assert!(!sock.parent().unwrap().exists());
        prepare_api_socket(&sock).expect("prepare_api_socket");
        assert!(sock.parent().unwrap().is_dir());
        // The prepared path is now bindable (and re-preparing over a STALE socket — bound once,
        // listener gone — still clears it and binds fresh).
        let listener = std::os::unix::net::UnixListener::bind(&sock).expect("bind nested socket");
        drop(listener);
        prepare_api_socket(&sock).expect("re-prepare clears stale socket");
        let listener = std::os::unix::net::UnixListener::bind(&sock).expect("rebind after clear");
        drop(listener);
        let _ = std::fs::remove_dir_all(&base);
    }

    /// A LIVE listener on the api socket path must fail the prepare (and thus the launch) instead
    /// of being unlinked out from under its daemon — the orphaned-daemon incident.
    #[cfg(unix)]
    #[test]
    fn prepare_api_socket_refuses_live_listener() {
        let base = std::env::temp_dir().join(format!(
            "daemon-socklive-{}-{:p}",
            std::process::id(),
            &0u8 as *const u8
        ));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).expect("create test dir");
        let sock = base.join("api.sock");
        let listener = std::os::unix::net::UnixListener::bind(&sock).expect("bind live listener");

        let err = prepare_api_socket(&sock).expect_err("a live listener must refuse the prepare");
        assert!(
            err.to_string().contains("already bound by a live daemon"),
            "unexpected error: {err}"
        );
        assert!(sock.exists(), "the live socket must be left in place");

        drop(listener);
        let _ = std::fs::remove_dir_all(&base);
    }

    /// Serializes the env-mutating bootstrap tests (they share the process environment).
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// Clear all three bootstrap env keys so each case starts from a known state.
    fn clear_admin_env() {
        std::env::remove_var(ADMIN_USERNAME_ENV);
        std::env::remove_var(ADMIN_PASSWORD_ENV);
        std::env::remove_var(ADMIN_PASSWORD_FILE_ENV);
    }

    #[test]
    fn no_env_selects_auto_generate() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_admin_env();
        assert!(matches!(resolve_admin_seed().unwrap(), AdminSeed::Generate));
    }

    #[test]
    fn username_plus_password_is_explicit() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_admin_env();
        std::env::set_var(ADMIN_USERNAME_ENV, "root");
        std::env::set_var(ADMIN_PASSWORD_ENV, "s3cret-pw");
        match resolve_admin_seed().unwrap() {
            AdminSeed::Explicit { username, password } => {
                assert_eq!(username, "root");
                assert_eq!(password, "s3cret-pw");
            }
            AdminSeed::Generate => panic!("expected explicit seed from env"),
        }
        clear_admin_env();
    }

    #[test]
    fn username_without_password_is_refused() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_admin_env();
        std::env::set_var(ADMIN_USERNAME_ENV, "root");
        assert!(
            resolve_admin_seed().is_err(),
            "username without any password source must be refused"
        );
        clear_admin_env();
    }

    #[test]
    fn empty_or_whitespace_password_is_refused() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_admin_env();
        std::env::set_var(ADMIN_USERNAME_ENV, "root");
        std::env::set_var(ADMIN_PASSWORD_ENV, "   ");
        assert!(
            resolve_admin_seed().is_err(),
            "whitespace-only password must be refused (never seed admin/<blank>)"
        );
        clear_admin_env();
    }

    #[test]
    fn password_file_is_read_and_trimmed() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_admin_env();
        let dir = std::env::temp_dir().join(format!("daemon-admin-pw-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("pw.txt");
        std::fs::write(&path, "  file-pw\n").unwrap();
        std::env::set_var(ADMIN_USERNAME_ENV, "root");
        std::env::set_var(ADMIN_PASSWORD_FILE_ENV, &path);
        match resolve_admin_seed().unwrap() {
            AdminSeed::Explicit { username, password } => {
                assert_eq!(username, "root");
                assert_eq!(password, "file-pw", "file contents are trimmed");
            }
            AdminSeed::Generate => panic!("expected explicit seed from password file"),
        }
        clear_admin_env();
        let _ = std::fs::remove_file(&path);
    }

    // --- Daemon Cloud attach-credential bootstrap (D2) ------------------------------------------

    /// Clear both D2 env keys so each case starts from a known state.
    fn clear_cloud_env() {
        std::env::remove_var(CLOUD_API_KEY_ENV);
        std::env::remove_var(CLOUD_API_KEY_FILE_ENV);
    }

    #[test]
    fn no_cloud_env_seeds_nothing() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_cloud_env();
        assert!(
            resolve_cloud_api_key().unwrap().is_none(),
            "no cloud env must seed no credential (keyless boot is supported)"
        );
    }

    #[test]
    fn cloud_key_env_is_read() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_cloud_env();
        std::env::set_var(CLOUD_API_KEY_ENV, "sk-daemon-cloud-attach");
        assert_eq!(
            resolve_cloud_api_key().unwrap().as_deref(),
            Some("sk-daemon-cloud-attach")
        );
        clear_cloud_env();
    }

    #[test]
    fn cloud_key_file_is_read_and_trimmed() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_cloud_env();
        let dir = std::env::temp_dir().join(format!("daemon-cloud-key-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("attach.txt");
        std::fs::write(&path, "  sk-from-file\n").unwrap();
        std::env::set_var(CLOUD_API_KEY_FILE_ENV, &path);
        assert_eq!(
            resolve_cloud_api_key().unwrap().as_deref(),
            Some("sk-from-file"),
            "file contents are trimmed"
        );
        clear_cloud_env();
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn cloud_key_env_wins_over_file() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_cloud_env();
        let dir = std::env::temp_dir().join(format!("daemon-cloud-prec-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("attach.txt");
        std::fs::write(&path, "sk-from-file").unwrap();
        std::env::set_var(CLOUD_API_KEY_ENV, "sk-from-env");
        std::env::set_var(CLOUD_API_KEY_FILE_ENV, &path);
        assert_eq!(
            resolve_cloud_api_key().unwrap().as_deref(),
            Some("sk-from-env"),
            "the direct env var wins over the file"
        );
        clear_cloud_env();
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn blank_cloud_key_is_refused() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_cloud_env();
        std::env::set_var(CLOUD_API_KEY_ENV, "   ");
        assert!(
            resolve_cloud_api_key().is_err(),
            "a set-but-whitespace key must be refused (never seed an empty bearer)"
        );
        clear_cloud_env();
    }

    // --- provider + model discovery (Track 2) ---------------------------------------------------

    /// `ProviderCatalog` enumerates the local engines + every genai cloud vendor + Daemon Cloud, and
    /// is independent of the launch config (an unconfigured node still lists providers).
    #[tokio::test]
    async fn provider_catalog_lists_local_all_genai_vendors_and_daemon_cloud() {
        let catalog = GenAiCloudCatalog;
        let providers = CloudCatalog::providers(&catalog).await;
        let ids: std::collections::HashSet<&str> =
            providers.iter().map(|p| p.id.as_str()).collect();

        // Local engines (node-owned model lists).
        assert!(ids.contains("llama_cpp"), "missing llama_cpp: {ids:?}");
        assert!(ids.contains("mistral_rs"), "missing mistral_rs: {ids:?}");
        // Every genai cloud vendor in the discovery set.
        for (vendor_id, _) in discovery_vendor_ids() {
            assert!(
                ids.contains(vendor_id.as_str()),
                "missing genai vendor {vendor_id}: {ids:?}"
            );
        }
        // Daemon Cloud carries the gateway base so the app never hardcodes it, needs a key to RUN
        // TURNS (requires_key = true; model LISTING stays keyless — see the keyless-gateway test),
        // and binds the DaemonApi selector.
        let daemon_cloud = providers
            .iter()
            .find(|p| p.id == "daemon_cloud")
            .expect("daemon_cloud present");
        assert_eq!(daemon_cloud.kind, ProviderKindWire::DaemonCloud);
        assert_eq!(daemon_cloud.wire_selector, ProviderSelector::DaemonApi);
        assert!(
            daemon_cloud.requires_key,
            "Daemon Cloud needs a key to run turns (lists keyless)"
        );
        assert_eq!(
            daemon_cloud.default_base_url.as_deref(),
            Some(DAEMON_CLOUD_BASE)
        );

        // Genai vendors require a key to run turns; local engines do not.
        let anthropic = providers.iter().find(|p| p.id == "anthropic").unwrap();
        assert!(anthropic.requires_key, "genai vendors need a key");
        assert_eq!(anthropic.wire_selector, ProviderSelector::GenAi);
    }

    /// `ProviderModels(daemon_cloud)` lists the gateway's `author/slug` models keyless via
    /// `GET {base}/models`, against a mock upstream (OpenAI `{ "data": [..] }` envelope).
    #[tokio::test]
    async fn daemon_cloud_models_listed_keyless_via_mock_gateway() {
        // A one-shot mock gateway: accept a connection, ignore the request, return the model list.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            let (mut sock, _) = listener.accept().await.unwrap();
            let mut buf = [0u8; 1024];
            let _ = sock.read(&mut buf).await;
            let body = r#"{"data":[{"id":"anthropic/claude-sonnet-4-5","name":"Claude Sonnet 4.5","context_length":200000}]}"#;
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = sock.write_all(resp.as_bytes()).await;
            let _ = sock.flush().await;
        });

        let base = format!("http://{addr}/");
        let models = daemon_cloud_gateway_models(&base).await;
        server.await.unwrap();

        assert_eq!(models.len(), 1, "one gateway model: {models:?}");
        let m = &models[0];
        assert_eq!(m.id, "anthropic/claude-sonnet-4-5", "id stays author/slug");
        assert_eq!(m.provider, ProviderSelector::DaemonApi);
        assert_eq!(m.display_name.as_deref(), Some("Claude Sonnet 4.5"));
        assert_eq!(m.context_length, Some(200_000));
    }

    /// A gateway that reports the 500 "Registry not published" (or is unreachable) yields an empty
    /// list, never an error to the picker.
    #[tokio::test]
    async fn daemon_cloud_models_empty_on_gateway_error() {
        // An unroutable/closed port: the GET fails and the picker sees an empty list.
        let models = daemon_cloud_gateway_models("http://127.0.0.1:1/").await;
        assert!(models.is_empty(), "gateway error => empty list: {models:?}");
    }
}
