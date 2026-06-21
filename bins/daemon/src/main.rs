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
    ProviderRegistry, SystemPrompt, Tool, ToolCall, ToolDef, ToolOutcome, ToolRegistry, TurnCx,
};
use daemon_credentials::{CapabilitySigner, CredentialAuthority, StubCredentialSource};
use daemon_host::{
    run_placed_child, run_placed_child_journaled, serve_api_unix, BrokeredCredentialProvider,
    CoreEngineFactory, CredentialBroker, EngineUnit, HostConfig, JournalFeeder, JournalSink,
    OwnerBroker,
};
use daemon_infer::protocol::{Engine, ModelParams};
use daemon_metta::protocol::Bounds as MettaBounds;
use daemon_metta_client::{MettaConfig as MettaClientConfig, MettaCoprocessor};
use daemon_mnemosyne::{MnemosyneConfig, MnemosyneProvider};
use daemon_tool_metta::MettaTool;
use daemon_models::{ActiveModels, ManagerConfig, ModelManager};
use daemon_node::{assemble, AssembledNode, NodeAssembly};
use daemon_providers::{
    GenAiEmbedder, GenAiProvider, LocalEmbedder, SwitchableLocalProvider, WorkerConfig,
};
use daemon_provision::CutChannel;
use daemon_store::{InMemoryStore, SessionStore};
use daemon_supervision::ManagedUnit;
use daemon_transport::RemoteHost;

use config::{
    ContextEngineKind, EmbedKind, MemoryProviderKind, NodeConfig, ProviderKind, StoreBackend,
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

    run_as_host(NodeConfig::load()?).await
}

/// Build the provider registry the config selected. `Mock` keeps the deterministic fleet wiring
/// (a completing default plus the delegating-orchestrator / completing-child demo profiles); a real
/// provider becomes the registry default for every profile (the engine threads the credential lease
/// secret onto each request as the bearer).
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
        ProviderKind::OpenAi => {
            let (base, model) = (cfg.base_url.clone(), cfg.model.clone());
            providers.set_default(Arc::new(move || {
                let mut p = GenAiProvider::openai(model.clone());
                if let Some(base) = &base {
                    p = p.with_endpoint(base.clone());
                }
                Arc::new(p) as Arc<dyn Provider>
            }));
        }
        ProviderKind::Anthropic => {
            let (base, model) = (cfg.base_url.clone(), cfg.model.clone());
            providers.set_default(Arc::new(move || {
                let mut p = GenAiProvider::anthropic(model.clone());
                if let Some(base) = &base {
                    p = p.with_endpoint(base.clone());
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

/// Build the default §10 context-engine *builder* the config selected. `Lcm` returns a per-session
/// builder ([`ContextEngineBuilder`]) that opens a fresh [`LcmContextEngine`] bound to each session
/// (so its per-session compaction state is never shared across concurrent sessions) over the shared
/// profile-scoped `lcm.db`; `Budgeted` returns `None`, leaving the engine on the in-core
/// [`BudgetedContextEngine`](daemon_core::BudgetedContextEngine) fallback.
fn build_context_engine(
    cfg: &NodeConfig,
    aux: Arc<dyn Provider>,
) -> Option<ContextEngineBuilder> {
    match cfg.context_engine {
        ContextEngineKind::Budgeted => None,
        ContextEngineKind::Lcm => {
            let lcm_cfg = if cfg.persist_providers() {
                LcmConfig {
                    data_dir: cfg.profile_home(),
                    bank: "default".to_string(),
                    ..LcmConfig::default()
                }
            } else {
                LcmConfig::in_memory()
            };
            Some(Arc::new(move |id: &SessionId| {
                match LcmContextEngine::open_for_session(lcm_cfg.clone(), id, aux.clone()) {
                    Ok(lcm) => Arc::new(lcm) as Arc<dyn ContextEngine>,
                    Err(e) => {
                        tracing::warn!(error = %e, session = %id,
                            "failed to open LCM context engine for session; using budgeted fallback");
                        Arc::new(daemon_core::BudgetedContextEngine::default())
                            as Arc<dyn ContextEngine>
                    }
                }
            }) as ContextEngineBuilder)
        }
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
            let base = if cfg.persist_providers() {
                MnemosyneConfig {
                    data_dir: cfg.profile_home(),
                    ..MnemosyneConfig::default()
                }
            } else {
                MnemosyneConfig::default()
            };
            let banks = Arc::new(MnemosyneBanks::new(base, cfg.persist_providers(), embedder));
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
                Arc::new(move |id: &SessionId| match banks.get_or_open(id) {
                    Some(p) => vec![p as Arc<dyn MemoryProvider>],
                    None => Vec::new(),
                })
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

/// A shared, agent-wide Mnemosyne bank cache: one per-session [`MnemosyneProvider`] over the same
/// bank database (or a per-session in-memory bank when the node is ephemeral). Memory is scoped at
/// the row level by `session_id`, so all sessions share global/long-term rows while keeping their own
/// session-local working memory. The cache lets the §11 memory builder and the `mnemosyne_*` tools
/// resolve the *same* instance for a given session.
struct MnemosyneBanks {
    base: MnemosyneConfig,
    persist: bool,
    embedder: Option<Arc<dyn EmbeddingProvider>>,
    sessions: Mutex<HashMap<SessionId, Arc<MnemosyneProvider>>>,
}

impl MnemosyneBanks {
    fn new(
        base: MnemosyneConfig,
        persist: bool,
        embedder: Option<Arc<dyn EmbeddingProvider>>,
    ) -> Self {
        Self {
            base,
            persist,
            embedder,
            sessions: Mutex::new(HashMap::new()),
        }
    }

    /// Open (once) and cache the provider for `session`, configured with that session id over the
    /// shared bank. Returns `None` if the bank cannot be opened.
    fn get_or_open(&self, session: &SessionId) -> Option<Arc<MnemosyneProvider>> {
        let mut sessions = self.sessions.lock().expect("mnemosyne banks poisoned");
        if let Some(existing) = sessions.get(session) {
            return Some(existing.clone());
        }
        let mut cfg = self.base.clone();
        cfg.session_id = session.as_str().to_string();
        let provider = if self.persist {
            match &self.embedder {
                Some(embedder) => MnemosyneProvider::open_with_embedder(cfg, embedder.clone()),
                None => MnemosyneProvider::open(cfg),
            }
        } else {
            // Ephemeral node: a private in-memory bank per session (no cross-session sharing, which
            // is acceptable when the session store itself is non-durable).
            crate::ephemeral_mnemosyne(cfg, self.embedder.clone())
        };
        match provider {
            Ok(p) => {
                let p = Arc::new(p);
                sessions.insert(session.clone(), p.clone());
                Some(p)
            }
            Err(e) => {
                tracing::warn!(error = %e, session = %session, "failed to open Mnemosyne bank");
                None
            }
        }
    }

    /// Enumerate the `mnemosyne_*` tool defs from a throwaway probe instance (session-independent).
    fn probe_tool_defs(&self) -> Option<Vec<ToolDef>> {
        let probe = self.get_or_open(&SessionId::new("__probe__"))?;
        let defs = probe.tools();
        self.sessions
            .lock()
            .expect("mnemosyne banks poisoned")
            .remove(&SessionId::new("__probe__"));
        Some(defs)
    }
}

/// Open an in-memory Mnemosyne provider for an ephemeral node, with an optional embedder.
fn ephemeral_mnemosyne(
    cfg: MnemosyneConfig,
    embedder: Option<Arc<dyn EmbeddingProvider>>,
) -> daemon_mnemosyne::Result<MnemosyneProvider> {
    use daemon_mnemosyne::Engine;
    let engine = Arc::new(Engine::open_in_memory(cfg)?);
    Ok(match embedder {
        Some(embedder) => MnemosyneProvider::with_embedder(engine, embedder),
        None => MnemosyneProvider::new(engine),
    })
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
        let result = match self.banks.get_or_open(&cx.session_id) {
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

    // Credentials: an owner authority brokered into *every* engine, uniformly across the durable,
    // interactive, and fleet-child construction paths (host-spec §6).
    let owner = build_owner_broker(&cfg.profile, &cfg.credential_key);
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
    let context_builder = build_context_engine(&cfg, lcm_aux);
    // The optional embedding backend (Mnemosyne vector recall), reusing the shared `ModelManager`
    // for local-model acquisition. `Off` by default — recall stays keyword-only.
    let embedder = build_embedder(&cfg, &manager).await;
    let memory = build_memory(&cfg, embedder);

    // The optional MeTTa symbolic-coprocessor tool (opt-in, like the HTTP/MCP surfaces). When
    // enabled it is appended to the role tool registry alongside the memory tools; its supervised
    // worker is spawned lazily on first use.
    let mut extra_tools = memory.tools;
    if let Some(metta_tool) = build_metta_tool(&cfg) {
        extra_tools.push(metta_tool);
    }

    let host_config = HostConfig {
        partition: cfg.partition,
        dispatch_interval: cfg.dispatch_interval,
        scan_interval: cfg.scan_interval,
        ..HostConfig::default()
    };

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
        context_builder,
        memory: memory.shared,
        memory_builder: memory.builder,
        extra_tools,
        models: Some(manager.clone()),
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
    handle.shutdown().await;
    let _ = std::fs::remove_file(&cfg.socket_path);
    Ok(())
}

/// Build the owner end of the credential brokering chain over a stub source (host-spec §7), minting
/// the configured key for the configured profile.
fn build_owner_broker(profile: &str, key: &str) -> Arc<dyn CredentialBroker> {
    let signer = Arc::new(CapabilitySigner::generate());
    let source = Arc::new(StubCredentialSource::minting(profile, key));
    let scope = CredScope::new([profile], ["chat", "embed"], Some(1_000));
    let authority = Arc::new(CredentialAuthority::new(
        scope,
        CredMode::Native,
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
    let owner = build_owner_broker(&cfg.profile, &cfg.credential_key);
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
