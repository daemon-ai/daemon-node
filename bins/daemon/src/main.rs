//! `daemon` — the host binary that assembles an engine, its host, tools, and orchestration.
//!
//! It is the role-by-config node (workspace-layout §6):
//! - default **host** role: build the policy inputs (store, credentials, provider registry, engine
//!   tunables) and hand them to [`daemon_node::assemble`] — the single host-composition root shared
//!   with the conformance harness — then serve the one [`daemon_api`] surface over a Unix socket.
//! - **placed-child** role (`DAEMON_PLACED_CHILD`): the far side of a placement cut, driving an
//!   engine whose durable state is brokered back to the parent's store.
//! - **transport-server** role (`DAEMON_TRANSPORT_SERVER=<addr>`): host a unit + authoritative
//!   store reached over a socket ([`daemon_transport::RemoteHost`]).

#![forbid(unsafe_code)]

mod config;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use daemon_common::{
    CredMode, CredScope, JournalStreamId, ModelEngine, ModelRef, ModelSource, ProfileRef,
    SessionId, UnitId,
};
use daemon_context_lcm::{LcmConfig, LcmContextEngine};
use daemon_core::{
    ContextEngine, ContextEngineBuilder, CredentialBuilder, CredentialProvider, EmbeddingProvider,
    EngineProfile, FileMemory, MemoryBuilder, MemoryProvider, MockProvider, Provider,
    ProviderRegistry, SystemPrompt, Tool, ToolCall, ToolDef, ToolOutcome, ToolProvider,
    ToolRegistry, TurnCx,
};
use daemon_credentials::{CapabilitySigner, CredentialAuthority};
use daemon_host::{
    run_placed_child, run_placed_child_journaled, serve_api_unix, BrokeredCredentialProvider,
    CloudCatalog, CoreEngineFactory, CredentialBroker, CredentialStore, EngineUnit,
    FileCredentialStore, FileProfileStore, FileRevisionLog, HostConfig, JournalFeeder, JournalSink,
    MemCredentialStore, MemProfileStore, OriginMatcher, OwnerBroker, PooledStoreCredentialSource,
    ProfileStore, RoutingRegistry, ScopePattern, SessionBinding, TransportPattern,
};
use daemon_protocol::{IsolationPolicy, TransportId};
use daemon_api::{
    BudgetSpec, ContextEngineSel, EngineTunables, MemoryProviderSel, ModelDescriptor, ProfileSpec,
    ProviderSelector,
};
use daemon_infer::protocol::{Engine, ModelParams};
use daemon_metta::protocol::Bounds as MettaBounds;
use daemon_metta_client::{MettaConfig as MettaClientConfig, MettaCoprocessor};
use daemon_pytool_client::{PyToolConfig, PyToolProvider};
use daemon_mnemosyne::{MnemosyneConfig, MnemosyneProvider};
use daemon_tool_metta::MettaTool;
use daemon_tool_clarify::ClarifyTool;
use daemon_tool_todo::TodoTool;
use daemon_tool_web::{
    FirecrawlFetch, LocalFetch, SecretSource, TavilySearch, WebExtractTool, WebFetchBackend,
    WebSearchTool,
};
use daemon_models::{ActiveModels, ManagerConfig, ModelManager};
use daemon_node::{assemble, AssembledNode, NodeAssembly, ProviderResolver};
use daemon_providers::{
    genai_listed_models, GenAiEmbedder, GenAiProvider, LocalEmbedder, SwitchableLocalProvider,
    WorkerConfig,
};
use daemon_provision::CutChannel;
use daemon_store::{InMemoryStore, SessionStore};
use daemon_supervision::ManagedUnit;
use daemon_transport::RemoteHost;

use config::{
    ContextEngineKind, EmbedKind, MemoryProviderKind, NodeConfig, ProviderKind, RoutingConfig,
    StoreBackend,
};

/// The environment variable that selects the placed-child role.
const PLACED_CHILD_ENV: &str = "DAEMON_PLACED_CHILD";
/// The environment variable that selects the transport-server role (its value is the bind address).
const TRANSPORT_SERVER_ENV: &str = "DAEMON_TRANSPORT_SERVER";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Stderr-only structured logging (stdout is the cut transport in the child role).
    daemon_telemetry::init_subscriber();

    if std::env::var_os(PLACED_CHILD_ENV).is_some() {
        run_as_placed_child().await;
        return Ok(());
    }

    if let Some(addr) = std::env::var_os(TRANSPORT_SERVER_ENV) {
        run_as_transport_server(addr.to_string_lossy().into_owned()).await?;
        return Ok(());
    }

    // One-shot operator subcommand: `daemon matrix login --homeserver <url> --credential-ref <key>`
    // performs the interactive SSO flow and writes the resulting session into the credential store
    // (daemon-matrix-transport-spec §6.1). Everything else falls through to the host role.
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("matrix")
        && args.get(2).map(String::as_str) == Some("login")
    {
        return run_matrix_login(&args[3..]).await;
    }

    run_as_host(NodeConfig::load()?).await
}

/// The `daemon matrix login` subcommand: SSO-login one account and persist its session under the
/// given credential-ref (the same key the profile's `bound_accounts` declares). Writes to the
/// durable `FileCredentialStore` the host reads at startup.
async fn run_matrix_login(args: &[String]) -> anyhow::Result<()> {
    let mut homeserver: Option<String> = None;
    let mut credential_ref: Option<String> = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--homeserver" => homeserver = it.next().cloned(),
            "--credential-ref" | "--credential_ref" => credential_ref = it.next().cloned(),
            other => anyhow::bail!(
                "unknown `matrix login` arg {other:?} (use --homeserver <url> --credential-ref <key>)"
            ),
        }
    }
    let homeserver =
        homeserver.ok_or_else(|| anyhow::anyhow!("`matrix login` requires --homeserver <url>"))?;
    let credential_ref = credential_ref
        .ok_or_else(|| anyhow::anyhow!("`matrix login` requires --credential-ref <key>"))?;

    let cfg = NodeConfig::load()?;
    // Login implies persistence: write to the same on-disk credential store the host reads.
    std::fs::create_dir_all(&cfg.data_dir).map_err(|e| {
        anyhow::anyhow!("creating data dir {}: {e}", cfg.data_dir.display())
    })?;
    let credential_store: Arc<dyn CredentialStore> =
        Arc::new(FileCredentialStore::open(cfg.data_dir.join("credentials.json"))?);
    daemon_matrix::login(
        credential_store,
        &homeserver,
        &cfg.matrix.store_root,
        &credential_ref,
    )
    .await
}

/// Build the provider registry the config selected. `Mock` keeps the deterministic fleet wiring
/// (a completing default plus the delegating-orchestrator / completing-child demo profiles); a real
/// provider becomes the registry default for every profile (the engine threads the credential lease
/// secret onto each request as the bearer).
/// The price sheet for a cloud `model`, looked up in the built-in catalog (the static fallback that
/// also backs the GUI model picker). `None` for an unknown / local model — cost is then left
/// uncomputed (`cost_micros == 0`). Cache read/write rates are derived from the base input rate.
fn pricing_for(model: &str) -> Option<daemon_common::Pricing> {
    ModelDescriptor::builtin_cloud_catalog()
        .into_iter()
        .find(|m| m.id == model)
        .and_then(|m| match (m.input_price_micros_per_mtok, m.output_price_micros_per_mtok) {
            (Some(input), Some(output)) => Some(daemon_common::Pricing::from_io(input, output)),
            _ => None,
        })
}

/// The binary's live networked-model discovery hook (the `daemon-host` is provider-agnostic and
/// never links `genai`). Asks `genai` for the models of every adapter whose key resolves, unions
/// them with the static catalog (which also carries the pricing/context overlay), and namespaces
/// live ids so they round-trip through adapter inference.
struct GenAiCloudCatalog;

#[async_trait::async_trait]
impl CloudCatalog for GenAiCloudCatalog {
    async fn list(&self) -> Vec<ModelDescriptor> {
        // Start from the static catalog: the pricing/context overlay + the no-key fallback list.
        let mut out = ModelDescriptor::builtin_cloud_catalog();
        let mut seen: std::collections::HashSet<String> =
            out.iter().map(|m| m.id.clone()).collect();
        // Overlay pricing/context for any live id that the static table also knows (by id).
        let overlay = ModelDescriptor::builtin_cloud_catalog();
        for id in genai_listed_models().await {
            if !seen.insert(id.clone()) {
                continue;
            }
            let known = overlay.iter().find(|m| m.id == id);
            out.push(ModelDescriptor {
                context_length: known
                    .and_then(|m| m.context_length)
                    .or_else(|| ModelDescriptor::known_context_length(&id)),
                input_price_micros_per_mtok: known.and_then(|m| m.input_price_micros_per_mtok),
                output_price_micros_per_mtok: known.and_then(|m| m.output_price_micros_per_mtok),
                id,
                provider: ProviderSelector::GenAi,
                local: false,
            });
        }
        out
    }
}

fn build_providers(
    cfg: &NodeConfig,
    manager: &Arc<ModelManager>,
    active: &ActiveModels,
) -> ProviderRegistry {
    let mut providers = ProviderRegistry::new();
    match cfg.provider_kind {
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
        ProviderKind::LlamaCpp | ProviderKind::MistralRs => {
            let engine = match cfg.provider_kind {
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
    manager: &Arc<ModelManager>,
    active: &ActiveModels,
) -> daemon_core::ProviderBuilder {
    match spec.provider {
        ProviderSelector::Mock => {
            Arc::new(|| Arc::new(MockProvider::completing("session done")) as Arc<dyn Provider>)
        }
        // The single genai-backed path; the adapter is inferred from the model name (`for_model`).
        ProviderSelector::GenAi => {
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
fn default_profile_spec(cfg: &NodeConfig) -> ProfileSpec {
    let provider = match cfg.provider_kind {
        ProviderKind::Mock => ProviderSelector::Mock,
        ProviderKind::GenAi => ProviderSelector::GenAi,
        ProviderKind::LlamaCpp => ProviderSelector::LlamaCpp,
        ProviderKind::MistralRs => ProviderSelector::MistralRs,
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
    }
}

/// The default §10 context wiring: an optional per-session context-engine *builder* and the
/// `lcm_*` drill-down tools registered on every role registry. Mirrors [`MemoryWiring`] — the
/// `Budgeted` fallback yields neither.
struct ContextWiring {
    builder: Option<ContextEngineBuilder>,
    tools: Vec<Arc<dyn Tool>>,
}

impl ContextWiring {
    fn off() -> Self {
        Self {
            builder: None,
            tools: Vec::new(),
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
            let lcm_cfg = if persist {
                LcmConfig {
                    data_dir: cfg.data_root(),
                    bank: "default".to_string(),
                    ..LcmConfig::default()
                }
            } else {
                LcmConfig::in_memory()
            };
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
            ContextWiring {
                builder: Some(builder),
                tools,
            }
        }
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
        let profile = profile.cloned().unwrap_or_else(|| self.default_profile.clone());
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
            None => serde_json::json!({"status": "error", "error": "lcm bank unavailable"})
                .to_string(),
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
}

impl MemoryWiring {
    fn off() -> Self {
        Self {
            builder: None,
            shared: Vec::new(),
            tools: Vec::new(),
        }
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
            },
            None => {
                tracing::warn!("memory_provider=file but DAEMON_MEMORY_FILE is unset; memory off");
                MemoryWiring::off()
            }
        },
        MemoryProviderKind::Mnemosyne => {
            // The base config carries the *root*; the bank cache re-roots `data_dir` to
            // `<data_root>/<profile>/` per resolved profile (in-memory banks ignore the path).
            let base = if cfg.persist_providers() {
                MnemosyneConfig {
                    data_dir: cfg.data_root(),
                    ..MnemosyneConfig::default()
                }
            } else {
                MnemosyneConfig::default()
            };
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
                Arc::new(
                    move |profile: Option<&ProfileRef>, id: &SessionId| {
                        match banks.get_or_open(profile, id) {
                            Some(p) => vec![p as Arc<dyn MemoryProvider>],
                            None => Vec::new(),
                        }
                    },
                )
            };
            MemoryWiring {
                builder: Some(builder),
                shared: Vec::new(),
                tools,
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
        let profile = profile.cloned().unwrap_or_else(|| self.default_profile.clone());
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
    let local = &cfg.local;
    let mut wc = WorkerConfig::new(local.worker_bin.clone(), engine, cfg.model.clone());
    wc.params = ModelParams {
        n_gpu_layers: local.n_gpu_layers,
        n_ctx: local.n_ctx,
        n_threads: local.n_threads,
        flash_attn: local.flash_attn,
        isq: local.isq.clone(),
        embeddings: false,
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
            let mut embedder =
                GenAiEmbedder::openai(cfg.embed.model.clone()).with_auth(cfg.credential_key.clone());
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
                cfg.local.worker_bin.clone(),
                infer_engine,
                artifact.local_path.to_string_lossy().into_owned(),
            );
            wc.params = ModelParams {
                n_gpu_layers: cfg.local.n_gpu_layers,
                n_ctx: cfg.local.n_ctx,
                n_threads: cfg.local.n_threads,
                flash_attn: cfg.local.flash_attn,
                isq: None,
                embeddings: true,
            };
            // Load from the warmed cache offline (the daemon owns acquisition).
            wc.env.extend(manager.cache().sidecar_env());
            wc.load_timeout = cfg.local.load_timeout;
            wc.max_restarts = cfg.local.max_restarts;
            wc.restart_window = cfg.local.restart_window;
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

/// Assemble and run the default host node, serving the unified surface over a Unix socket until
/// `ctrl_c` trips a graceful shutdown. The wiring itself lives in [`daemon_node::assemble`]; this
/// role only builds the policy inputs (store, credentials, provider registry, engine tunables).
async fn run_as_host(cfg: NodeConfig) -> anyhow::Result<()> {
    let store = build_store(&cfg.store)?;

    // The persisted credential store backing the `CredentialApi` surface and the owner authority.
    // Durable nodes persist secrets under the data root; the ephemeral default keeps them in memory.
    let credential_store: Arc<dyn CredentialStore> = if cfg.persist_providers() {
        Arc::new(FileCredentialStore::open(cfg.data_dir.join("credentials.json"))?)
    } else {
        Arc::new(MemCredentialStore::new())
    };
    // Seed the launch-configured key so existing launches keep authenticating until a GUI sets one.
    credential_store
        .set(&cfg.profile, &cfg.credential_key)
        .map_err(|e| anyhow::anyhow!("seeding credential: {e}"))?;

    // Credentials: an owner authority brokered into *every* engine, uniformly across the durable,
    // interactive, and fleet-child construction paths (host-spec §6).
    let owner = build_owner_broker(&cfg.profile, &cfg.credential_key, credential_store.clone());
    let cred_profile = ProfileRef::new(cfg.profile.clone());
    let credentials: CredentialBuilder = {
        let owner = owner.clone();
        Arc::new(move || {
            Arc::new(BrokeredCredentialProvider::new(owner.clone(), None))
                as Arc<dyn CredentialProvider>
        })
    };

    // Model management: the daemon owns search + acquisition + caching + catalog for the local
    // engines (unified across llama.cpp + mistral.rs). Built unconditionally so the `ModelApi`
    // surface works even on a remote-only node (the GUI can browse/download regardless).
    let manager = Arc::new(
        ModelManager::new(ManagerConfig {
            cache_dir: cfg.models.cache_dir.clone(),
            registry_path: cfg.models.registry_path.clone(),
            endpoint: cfg.models.endpoint.clone(),
            // Offline quantization runs out-of-process via the llama-enabled inference worker; reuse
            // the configured worker binary (it has the `quantize` subcommand when built with llama).
            quantize_worker_bin: Some(cfg.local.worker_bin.clone()),
        })
        .await?,
    );
    let active = manager.active_handle();
    // Seed the configured model as the active selection for a local provider, so resolve-before-load
    // downloads it into the shared cache on first use.
    if matches!(
        cfg.provider_kind,
        ProviderKind::LlamaCpp | ProviderKind::MistralRs
    ) {
        if let Some(model_ref) = parse_model_ref(cfg.provider_kind, &cfg.model) {
            active.set(cfg.profile.clone(), model_ref).await;
        }
    }

    // Provider selection seam: Mock is the zero-config default; a real networked provider drops in
    // via `set_default(...)` without touching the engine or the construction sites. The API key
    // flows per-call through the credential broker (the lease secret -> `Request.auth`), so a real
    // provider builder needs only the base URL + model.
    let providers = build_providers(&cfg, &manager, &active);

    // The default context engine (§10, LCM) and memory providers (§11, Mnemosyne) wired into every
    // engine this node builds, with their `mnemosyne_*` tools registered on the shared registry. Both
    // are per-session builders (LCM keeps per-session compaction state; Mnemosyne scopes by
    // `session_id`), so concurrent sessions never share mutable provider state.
    // LCM summarizes through the same default provider the agent uses: resolve the profile's builder
    // (falling back to a mock) and hand the context engine an aux provider instance.
    let lcm_aux: Arc<dyn Provider> = providers
        .builder_for(&cred_profile)
        .map(|b| b())
        .unwrap_or_else(|| Arc::new(MockProvider::completing("")) as Arc<dyn Provider>);
    let context = build_context(&cfg, lcm_aux);
    // The optional embedding backend (Mnemosyne vector recall), reusing the shared `ModelManager`
    // for local-model acquisition. `Off` by default — recall stays keyword-only.
    let embedder = build_embedder(&cfg, &manager).await;
    // Mnemosyne's optional LLM backend for structured extraction + sleep summarization, resolved the
    // same way as `lcm_aux` (the profile's builder, falling back to a mock). With no provider the
    // knowledge layer stays on its deterministic regex/AAAK baselines.
    let mnemosyne_llm: Arc<dyn Provider> = providers
        .builder_for(&cred_profile)
        .map(|b| b())
        .unwrap_or_else(|| Arc::new(MockProvider::completing("")) as Arc<dyn Provider>);
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

    let mut prompt_sources: Vec<Arc<dyn daemon_core::StablePromptSource>> = Vec::new();
    // Held out of the skills block so the node's api surface can bind it for skill versioning +
    // the skill payload of a profile distribution.
    let mut skills_store: Option<Arc<daemon_skills::SkillStore>> = None;
    if cfg.skills.enable {
        let skills_dir = cfg
            .skills
            .dir
            .clone()
            .unwrap_or_else(|| cfg.profile_home().join("skills"));
        let mut store = daemon_skills::SkillStore::new(skills_dir);
        // Version skill writes (incl. the agent's own background-review edits) when versioning is on.
        if let Some(revisions) = &revisions {
            store = store.with_revisions(revisions.clone());
        }
        let skill_store = Arc::new(store);
        // Seed the curated, tool-agnostic bundled skills into the profile on first run (skipping any
        // the user already has) — parity with hermes' bundled-skills sync. Best-effort: a seed
        // failure must not block node startup.
        match skill_store.seed_bundled() {
            Ok(seeded) if !seeded.is_empty() => {
                tracing::info!(skills = ?seeded, "seeded bundled skills into profile")
            }
            Ok(_) => {}
            Err(e) => tracing::warn!(error = %e, "seeding bundled skills failed"),
        }
        extra_tools.extend(daemon_tool_skill::skill_tools(skill_store.clone()));
        prompt_sources.push(Arc::new(
            daemon_skills::SkillsPromptSource::new(skill_store.clone()).enabled(true),
        ) as Arc<dyn daemon_core::StablePromptSource>);
        skills_store = Some(skill_store);
    }

    // The optional web tools (`web_search`/`web_extract`, opt-in). Keys are read live from the
    // credential store, so a GUI-set Tavily/Firecrawl key applies without a restart.
    extra_tools.extend(build_web_tools(&cfg, credential_store.clone()));

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
    // is in-memory (the ephemeral default). The launch config is seeded as the active default.
    let profile_store: Arc<dyn ProfileStore> = if cfg.persist_providers() {
        Arc::new(FileProfileStore::open(cfg.data_dir.join("profiles"))?)
    } else {
        Arc::new(MemProfileStore::new())
    };
    profile_store
        .seed(default_profile_spec(&cfg))
        .map_err(|e| anyhow::anyhow!("seeding default profile: {e}"))?;
    // The per-session provider resolution seam: maps the active profile bundle onto a provider
    // client (so a GUI can switch model/provider live).
    let provider_resolver: ProviderResolver = {
        let cfg = cfg.clone();
        let manager = manager.clone();
        let active = active.clone();
        Arc::new(move |spec: &ProfileSpec| provider_builder_for(spec, &cfg, &manager, &active))
    };

    let routing = build_routing_registry(&cfg.routing);
    // The §12 tool-checkpoint store: a workspace checkpoint is recorded before each mutating tool
    // runs (rewindable via the `Checkpoint{List,Rewind}` control ops). The ledger lives under the
    // data dir so rewind points survive a restart.
    let checkpoints: Option<Arc<dyn daemon_core::CheckpointStore>> =
        Some(Arc::new(daemon_core::LocalCheckpointStore::new(
            cfg.data_dir.join("checkpoints"),
        )) as Arc<dyn daemon_core::CheckpointStore>);

    let AssembledNode { node, handle, .. } = assemble(NodeAssembly {
        store,
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
        skills: skills_store,
        routing,
        checkpoints,
    });
    tracing::info!("daemon host node started");

    // Bind the api socket (fresh) and serve the unified surface over it.
    let _ = std::fs::remove_file(&cfg.socket_path);
    let listener = tokio::net::UnixListener::bind(&cfg.socket_path)?;
    tracing::info!(socket = %cfg.socket_path.display(), "serving daemon-api over unix socket");
    let server = tokio::spawn(serve_api_unix(listener, node.clone()));

    // Optionally bind the in-process HTTP/WS surface (the `daemon-http` adapter), toggled on by a
    // configured bind address (like the MCP surface). It shares the same `Arc<dyn NodeApi>`, so it is
    // just another transport over the one canonical interface — JSON dispatch plus SSE/WS streaming
    // over the merged session event log.
    // Optionally spawn the Matrix chat transport (the `daemon-matrix` adapter), toggled on by
    // `[matrix].enabled`. It drives the same `Arc<dyn NodeApi>` as a client and consumes the host's
    // in-process `AccountProvisioning` seam (enumerate bound accounts + resolve/write-back session
    // blobs); both `node` coercions come off the one concrete `NodeApiImpl`.
    let matrix_server = if cfg.matrix.enabled {
        tracing::info!("spawning matrix transport (daemon-matrix)");
        let api: Arc<dyn daemon_api::NodeApi> = node.clone();
        let provisioning: Arc<dyn daemon_host::AccountProvisioning> = node.clone();
        let mcfg = cfg.matrix.clone();
        Some(tokio::spawn(daemon_matrix::serve(api, provisioning, mcfg)))
    } else {
        None
    };

    let http_server = match &cfg.http_addr {
        Some(addr) => {
            let http_listener = tokio::net::TcpListener::bind(addr).await?;
            tracing::info!(%addr, "serving daemon-api over http (json dispatch + sse/ws subscribe)");
            let api: Arc<dyn daemon_api::NodeApi> = node;
            Some(tokio::spawn(async move {
                if let Err(e) = daemon_http::serve_http(http_listener, api).await {
                    tracing::warn!(error = %e, "http surface ended");
                }
            }))
        }
        None => None,
    };

    tokio::signal::ctrl_c().await?;
    tracing::info!("ctrl_c received; shutting down");
    server.abort();
    if let Some(http_server) = http_server {
        http_server.abort();
    }
    if let Some(matrix_server) = matrix_server {
        matrix_server.abort();
    }
    handle.shutdown().await;
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
fn build_owner_broker(
    profile: &str,
    fallback_key: &str,
    store: Arc<dyn CredentialStore>,
) -> Arc<dyn CredentialBroker> {
    let signer = Arc::new(CapabilitySigner::generate());
    // The pooled source selects/rotates among the profile's key pool (multi-key) on a rotatable
    // failure, falling back to the launch-configured key when the pool is empty.
    let source = Arc::new(PooledStoreCredentialSource::new(store, profile, fallback_key));
    let scope = CredScope::new([profile], ["chat", "embed"], Some(1_000));
    let authority = Arc::new(CredentialAuthority::new(
        scope,
        CredMode::Bearer,
        60_000,
        signer,
        source,
    ));
    Arc::new(OwnerBroker::new(authority))
}

/// Run as a transport server: host a completing engine unit + an authoritative store, reachable as
/// a `ManagedUnit` over a socket (with the cross-node lease/fence handshake). The engine is built
/// through a *dressed* [`EngineProfile`] (engine tunables + a local owner-broker credential seam,
/// since a transport node is its own authority over its own store) and journals its transcript per
/// turn under a seed-derived signer, so its construction matches the host path.
async fn run_as_transport_server(addr: String) -> anyhow::Result<()> {
    let cfg = NodeConfig::load()?;
    let store: Arc<dyn SessionStore> = Arc::new(InMemoryStore::new());

    // A transport node owns its store, so it mints its own credentials (the host path's owner
    // broker) rather than brokering from a parent — the engine is therefore not credential-less.
    let cred_store: Arc<dyn CredentialStore> = Arc::new(MemCredentialStore::new());
    cred_store
        .set(&cfg.profile, &cfg.credential_key)
        .map_err(|e| anyhow::anyhow!("seeding credential: {e}"))?;
    let owner = build_owner_broker(&cfg.profile, &cfg.credential_key, cred_store);
    let credentials: CredentialBuilder = {
        let owner = owner.clone();
        Arc::new(move || {
            Arc::new(BrokeredCredentialProvider::new(owner.clone(), None))
                as Arc<dyn CredentialProvider>
        })
    };
    let profile = EngineProfile::new(
        Arc::new(|| Arc::new(MockProvider::completing("transport done")) as Arc<dyn Provider>),
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

/// Run as the far side of a placement cut: a completing engine driven over the brokered store. The
/// engine is built from a *dressed* [`EngineProfile`] (engine tunables applied, via
/// [`CoreEngineFactory::from_profile`]) so it shares the host's construction seam rather than a
/// bespoke literal. When the node's journal seed is configured (passed down via `DAEMON_JOURNAL_SEED`
/// by the spawning parent), the child journals its durable transcript **through the parent's brokered
/// store**, sealed under the node's seed-derived signer so the chain verifies under the node's
/// published verifying key. Credentials stay on the embedded L1 pool — brokering them over the cut
/// is a separate channel, deferred.
async fn run_as_placed_child() {
    let cfg = match NodeConfig::load() {
        Ok(cfg) => cfg,
        Err(e) => {
            tracing::error!(error = %e, "placed child failed to load config");
            return;
        }
    };
    let profile = EngineProfile::new(
        Arc::new(|| Arc::new(MockProvider::completing("placed child done")) as Arc<dyn Provider>),
        Arc::new(ToolRegistry::new()),
        SystemPrompt::new("placed child"),
    )
    .with_config(cfg.engine);
    let factory = CoreEngineFactory::from_profile(profile);
    let channel = CutChannel::from_stdio();

    match cfg.journal_seed {
        Some(seed) => {
            let signer = Arc::new(daemon_telemetry::TraceSigner::from_seed(&seed));
            run_placed_child_journaled(channel, factory, cfg.partition, signer).await;
        }
        None => run_placed_child(channel, Arc::new(factory), cfg.partition).await,
    }
}
