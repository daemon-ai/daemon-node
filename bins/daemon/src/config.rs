//! The node's layered configuration: an optional TOML file, overlaid by environment variables
//! (env wins). This is the *composition-layer* config (partition, socket, store backend, resident
//! cadence, provider/credential selection) — distinct from the engine tunables
//! ([`daemon_core::Config`]), which the host fills from here and injects via the `EngineProfile`.

use anyhow::Context;
use daemon_common::PartitionId;
use serde::Deserialize;
use std::path::PathBuf;
use std::time::Duration;

/// Path to an optional TOML config file.
const CONFIG_ENV: &str = "DAEMON_CONFIG";
/// Overrides the api socket path.
const API_SOCKET_ENV: &str = "DAEMON_API_SOCKET";
/// Enables the optional in-process HTTP/WS surface (the `daemon-http` adapter) and sets its bind
/// address (e.g. `127.0.0.1:8787`). Absent => the HTTP surface is off (toggle-on-launch, like MCP).
const HTTP_ADDR_ENV: &str = "DAEMON_HTTP_ADDR";
/// Selects the durable store backend: `memory` (default) or `sqlite`.
const STORE_ENV: &str = "DAEMON_STORE";
/// The SQLite database path (when the backend is `sqlite`).
const STORE_PATH_ENV: &str = "DAEMON_STORE_PATH";
/// The host data directory rooting the profile-scoped subsystem databases (the §10/§11 LCM +
/// Mnemosyne stores live under `<data_dir>/<profile>/`, mirroring hermes' per-profile home).
const DATA_DIR_ENV: &str = "DAEMON_DATA_DIR";
/// Overrides the owned partition id (a `u64`).
const PARTITION_ENV: &str = "DAEMON_PARTITION";
/// Overrides the model provider/credential profile name.
const PROFILE_ENV: &str = "DAEMON_PROFILE";
/// Selects the default context engine (§10): `lcm` (default) or `budgeted`/`none`.
const CONTEXT_ENGINE_ENV: &str = "DAEMON_CONTEXT_ENGINE";
/// Selects the default memory provider (§11): `mnemosyne` (default), `file`, or `none`.
const MEMORY_PROVIDER_ENV: &str = "DAEMON_MEMORY_PROVIDER";
/// The snapshot file the `file` memory provider serves as its frozen memory (when selected).
const MEMORY_FILE_ENV: &str = "DAEMON_MEMORY_FILE";
/// Selects the model provider implementation: `mock` (default), `openai`, or `anthropic`.
const MODEL_PROVIDER_ENV: &str = "DAEMON_MODEL_PROVIDER";
/// Overrides the provider API base URL (defaults per provider).
const BASE_URL_ENV: &str = "DAEMON_BASE_URL";
/// Overrides the model name sent to a real provider.
const MODEL_ENV: &str = "DAEMON_MODEL";
/// Overrides the (stub) credential key the owner authority mints.
const CREDENTIAL_KEY_ENV: &str = "DAEMON_CREDENTIAL_KEY";
/// Overrides the engine's `model_retry_attempts` tunable.
const MODEL_RETRY_ATTEMPTS_ENV: &str = "DAEMON_MODEL_RETRY_ATTEMPTS";
/// Overrides the engine's `context_budget_tokens` tunable.
const CONTEXT_BUDGET_TOKENS_ENV: &str = "DAEMON_CONTEXT_BUDGET_TOKENS";
/// Overrides the engine's `max_iterations` (per-turn ReAct round cap) tunable.
const MAX_ITERATIONS_ENV: &str = "DAEMON_MAX_ITERATIONS";
/// Overrides the engine's `tool_result_budget` (per-tool result-byte cap) tunable.
const TOOL_RESULT_BUDGET_ENV: &str = "DAEMON_TOOL_RESULT_BUDGET";
/// The 32-byte verifiable-journal signer seed, hex-encoded (64 hex chars).
const JOURNAL_SEED_ENV: &str = "DAEMON_JOURNAL_SEED";
/// How many orchestrator levels the top fleet materializes before its leaves (fleets-of-fleets).
const NESTING_DEPTH_ENV: &str = "DAEMON_NESTING_DEPTH";

// --- Local-inference worker (`daemon-infer`) tuning (DAEMON_INFER_*) -------------------------
/// Path to the `daemon-infer` worker binary (default: a `daemon-infer` next to the daemon binary).
const INFER_BIN_ENV: &str = "DAEMON_INFER_BIN";
/// llama.cpp: number of layers offloaded to the GPU (`0` = CPU only).
const INFER_N_GPU_LAYERS_ENV: &str = "DAEMON_INFER_N_GPU_LAYERS";
/// The context window to allocate (`0` = the model's training default).
const INFER_N_CTX_ENV: &str = "DAEMON_INFER_N_CTX";
/// Threads used for generation/prompt processing.
const INFER_N_THREADS_ENV: &str = "DAEMON_INFER_N_THREADS";
/// Enable Flash Attention where the backend supports it (`1`/`true`).
const INFER_FLASH_ATTN_ENV: &str = "DAEMON_INFER_FLASH_ATTN";
/// mistral.rs in-situ quantization spec (e.g. `Q4K`).
const INFER_ISQ_ENV: &str = "DAEMON_INFER_ISQ";
/// The output-token cap per generation (`0` = the worker default).
const INFER_MAX_TOKENS_ENV: &str = "DAEMON_INFER_MAX_TOKENS";
/// How long to wait for the model to load (ms).
const INFER_LOAD_TIMEOUT_MS_ENV: &str = "DAEMON_INFER_LOAD_TIMEOUT_MS";
/// Watchdog: max wait for the first token of a generation (ms).
const INFER_TTFT_TIMEOUT_MS_ENV: &str = "DAEMON_INFER_TTFT_TIMEOUT_MS";
/// Watchdog: max wait between tokens once streaming (ms).
const INFER_INTER_TOKEN_TIMEOUT_MS_ENV: &str = "DAEMON_INFER_INTER_TOKEN_TIMEOUT_MS";
/// Crash-loop meltdown: max worker restarts within the restart window.
const INFER_MAX_RESTARTS_ENV: &str = "DAEMON_INFER_MAX_RESTARTS";
/// The sliding window (ms) over which restarts are counted for meltdown.
const INFER_RESTART_WINDOW_MS_ENV: &str = "DAEMON_INFER_RESTART_WINDOW_MS";

// --- Embeddings (`daemon-mnemosyne` vector recall) tuning (DAEMON_EMBED_*) --------------------
/// Selects the embedding backend: `off` (default, keyword-only), `genai` (remote, OpenAI-compatible),
/// or `local` (a `daemon-infer` embedding worker).
const EMBED_PROVIDER_ENV: &str = "DAEMON_EMBED_PROVIDER";
/// The embedding model: a `genai` model name (remote) or a model spec / GGUF path (local).
const EMBED_MODEL_ENV: &str = "DAEMON_EMBED_MODEL";
/// The embedding dimensionality (for store/index validation; `0` = unknown).
const EMBED_DIMS_ENV: &str = "DAEMON_EMBED_DIMS";
/// Remote embeddings: the OpenAI-compatible API base URL override (`None` = the provider default).
const EMBED_BASE_URL_ENV: &str = "DAEMON_EMBED_BASE_URL";
/// Local embeddings: the inference engine (`llama` default, or `mistralrs`).
const EMBED_ENGINE_ENV: &str = "DAEMON_EMBED_ENGINE";

// --- Model management (`daemon-models`) tuning (DAEMON_MODELS_*) ------------------------------
/// The shared Hugging Face hub cache directory (default: the `HF_*`/XDG precedence).
const MODELS_CACHE_DIR_ENV: &str = "DAEMON_MODELS_CACHE_DIR";
/// The installed-model catalog manifest path (default: `<hub>/daemon-catalog.json`).
const MODELS_REGISTRY_ENV: &str = "DAEMON_MODELS_REGISTRY";
/// The Hugging Face Hub endpoint override (default: `https://huggingface.co`; mainly for tests).
const MODELS_ENDPOINT_ENV: &str = "DAEMON_MODELS_ENDPOINT";

/// Which model provider implementation the node uses (selected by config).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProviderKind {
    /// The deterministic in-tree provider (zero-config default; no network/keys).
    Mock,
    /// The networked OpenAI Chat Completions provider.
    OpenAi,
    /// The networked Anthropic Messages provider.
    Anthropic,
    /// A local llama.cpp model via the supervised `daemon-infer` worker.
    LlamaCpp,
    /// A local mistral.rs model via the supervised `daemon-infer` worker.
    MistralRs,
}

/// Which default context engine (§10) the node wires into every engine it builds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContextEngineKind {
    /// The native LCM port (`daemon-context-lcm`) — the default.
    Lcm,
    /// The in-core [`BudgetedContextEngine`](daemon_core::BudgetedContextEngine) (drop-oldest); also
    /// selected by `none`/`default`. Leaves the engine on its built-in fallback (no extra crate).
    Budgeted,
}

/// Which default memory provider (§11) the node wires into every engine it builds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryProviderKind {
    /// The native Mnemosyne port (`daemon-mnemosyne`) — the default.
    Mnemosyne,
    /// The in-core [`FileMemory`](daemon_core::FileMemory) over a frozen snapshot file.
    File,
    /// No memory provider (memory off).
    None,
}

/// Tuning for the local-inference [`daemon-infer`] worker (used only for the local provider kinds).
#[derive(Clone, Debug)]
pub struct LocalConfig {
    /// Path to the `daemon-infer` worker binary.
    pub worker_bin: PathBuf,
    /// llama.cpp: number of layers offloaded to the GPU (`0` = CPU only).
    pub n_gpu_layers: u32,
    /// The context window to allocate (`0` = the model's training default).
    pub n_ctx: u32,
    /// Threads used for generation/prompt processing (`None` = engine default).
    pub n_threads: Option<u32>,
    /// Enable Flash Attention where supported.
    pub flash_attn: bool,
    /// mistral.rs in-situ quantization spec (e.g. `Q4K`); `None` = load as-is.
    pub isq: Option<String>,
    /// The output-token cap per generation (`0` = the worker default).
    pub max_tokens: u32,
    /// How long to wait for `Event::Ready` after load.
    pub load_timeout: Duration,
    /// Watchdog: max wait for the first token of a generation.
    pub ttft_timeout: Duration,
    /// Watchdog: max wait between tokens once streaming.
    pub inter_token_timeout: Duration,
    /// Crash-loop meltdown: max restarts within [`LocalConfig::restart_window`].
    pub max_restarts: u32,
    /// The sliding window over which restarts are counted for meltdown.
    pub restart_window: Duration,
}

impl Default for LocalConfig {
    fn default() -> Self {
        Self {
            worker_bin: default_worker_bin(),
            n_gpu_layers: 0,
            n_ctx: 0,
            n_threads: None,
            flash_attn: false,
            isq: None,
            max_tokens: 0,
            load_timeout: Duration::from_secs(120),
            ttft_timeout: Duration::from_secs(60),
            inter_token_timeout: Duration::from_secs(30),
            max_restarts: 3,
            restart_window: Duration::from_secs(60),
        }
    }
}

/// The default worker binary: a `daemon-infer` next to the running daemon executable, falling back
/// to a bare `daemon-infer` (resolved on `PATH`).
fn default_worker_bin() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|dir| dir.join("daemon-infer")))
        .unwrap_or_else(|| PathBuf::from("daemon-infer"))
}

/// Which embedding backend Mnemosyne uses for vector recall (selected by config).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmbedKind {
    /// No embeddings — recall is keyword-only (the zero-config default).
    Off,
    /// A remote, OpenAI-compatible embeddings API via `genai`.
    Genai,
    /// A local embedding model via a supervised `daemon-infer` worker.
    Local,
}

/// Tuning for the embeddings backend (`DAEMON_EMBED_*`). `kind = Off` keeps recall keyword-only.
#[derive(Clone, Debug)]
pub struct EmbedConfig {
    /// Which backend to use (off|genai|local).
    pub kind: EmbedKind,
    /// The embedding model: a `genai` model name (remote) or a model spec / GGUF path (local).
    pub model: String,
    /// The embedding dimensionality (`0` = unknown).
    pub dims: usize,
    /// Remote: the OpenAI-compatible API base URL override (`None` = provider default).
    pub base_url: Option<String>,
    /// Local: the inference engine identifier (`llama` default, or `mistralrs`).
    pub engine: String,
}

impl Default for EmbedConfig {
    fn default() -> Self {
        Self {
            kind: EmbedKind::Off,
            model: String::new(),
            dims: 0,
            base_url: None,
            engine: "llama".to_string(),
        }
    }
}

/// Tuning for the `daemon-models` model-management facade (shared cache + catalog + Hub endpoint).
#[derive(Clone, Debug, Default)]
pub struct ModelsConfig {
    /// The shared Hugging Face hub cache directory; `None` follows the `HF_*`/XDG precedence.
    pub cache_dir: Option<PathBuf>,
    /// The catalog manifest path; `None` places it next to the cache.
    pub registry_path: Option<PathBuf>,
    /// The Hugging Face Hub endpoint; `None` uses the default.
    pub endpoint: Option<String>,
}

/// The durable store backend selected by config.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StoreBackend {
    /// The in-memory backend (non-durable; the default).
    Memory,
    /// The SQLite backend at a database file path.
    Sqlite {
        /// Path to the SQLite database file.
        path: PathBuf,
    },
}

/// The resolved node configuration (TOML overlaid by env).
#[derive(Clone, Debug)]
pub struct NodeConfig {
    /// The partition this node owns.
    pub partition: PartitionId,
    /// The Unix socket the node serves its [`daemon_api`](daemon_api) surface on.
    pub socket_path: PathBuf,
    /// The optional in-process HTTP/WS surface bind address (the `daemon-http` adapter). `None`
    /// leaves it off; `Some(addr)` binds an axum server alongside the Unix socket (toggle-on-launch,
    /// like the MCP surface), sharing the same `Arc<dyn NodeApi>`.
    pub http_addr: Option<String>,
    /// The durable store backend.
    pub store: StoreBackend,
    /// The host data directory rooting the profile-scoped subsystem databases (§10/§11). The LCM and
    /// Mnemosyne stores live under [`NodeConfig::profile_home`]; see [`NodeConfig::persist_providers`].
    pub data_dir: PathBuf,
    /// How often the wake/job dispatchers poll the durable outboxes.
    pub dispatch_interval: Duration,
    /// How often the recovery scanner re-checks for resumable sessions.
    pub scan_interval: Duration,
    /// The model provider + credential profile name (selects the registered provider builder).
    pub profile: String,
    /// The default context engine (§10) wired into every engine (`lcm` default).
    pub context_engine: ContextEngineKind,
    /// The default memory provider (§11) wired into every engine (`mnemosyne` default).
    pub memory_provider: MemoryProviderKind,
    /// The snapshot file the `file` memory provider serves (when `memory_provider = file`).
    pub memory_file: Option<PathBuf>,
    /// Which model provider implementation to use (mock|openai|anthropic).
    pub provider_kind: ProviderKind,
    /// An optional provider API base-URL override. `None` uses the provider's default endpoint (the
    /// usual case); `Some` points the client elsewhere (a gateway/proxy, or a test mock server).
    pub base_url: Option<String>,
    /// The model name sent to a real provider, or the model path / HF id for a local provider
    /// (resolved with a per-provider default; empty for mock and local kinds — set explicitly).
    pub model: String,
    /// Local-inference worker tuning (meaningful only for the [`ProviderKind::LlamaCpp`] /
    /// [`ProviderKind::MistralRs`] kinds).
    pub local: LocalConfig,
    /// Model-management (search/download/cache/catalog) tuning.
    pub models: ModelsConfig,
    /// Embeddings backend tuning (Mnemosyne vector recall; `Off` by default).
    pub embed: EmbedConfig,
    /// The (stub) credential key the owner authority mints for that profile.
    pub credential_key: String,
    /// The engine tunables (§20) injected into every engine via the `EngineProfile`.
    pub engine: daemon_core::Config,
    /// The 32-byte seed for the node's verifiable-journal signer (a stable verifying key across
    /// restarts). `None` => an ephemeral key generated each boot.
    pub journal_seed: Option<[u8; 32]>,
    /// How many orchestrator levels the top fleet materializes before its engine leaves. `0` (the
    /// default) is a flat fleet; `n` nests the management tree `n` deep (fleets-of-fleets).
    pub nesting_depth: usize,
}

/// The TOML file shape — every field optional, so a partial file is valid and env fills the rest.
#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct FileConfig {
    partition: Option<u64>,
    socket_path: Option<PathBuf>,
    http_addr: Option<String>,
    store: Option<String>,
    store_path: Option<PathBuf>,
    data_dir: Option<PathBuf>,
    dispatch_interval_ms: Option<u64>,
    scan_interval_ms: Option<u64>,
    profile: Option<String>,
    context_engine: Option<String>,
    memory_provider: Option<String>,
    memory_file: Option<PathBuf>,
    model_provider: Option<String>,
    base_url: Option<String>,
    model: Option<String>,
    credential_key: Option<String>,
    model_retry_attempts: Option<u8>,
    context_budget_tokens: Option<u32>,
    max_iterations: Option<u32>,
    tool_result_budget: Option<usize>,
    journal_seed: Option<String>,
    nesting_depth: Option<usize>,
    local: Option<FileLocalConfig>,
    models: Option<FileModelsConfig>,
    embed: Option<FileEmbedConfig>,
}

/// The `[embed]` TOML table — embeddings tuning (every field optional; env wins).
#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct FileEmbedConfig {
    provider: Option<String>,
    model: Option<String>,
    dims: Option<usize>,
    base_url: Option<String>,
    engine: Option<String>,
}

/// The `[models]` TOML table — model-management tuning (every field optional; env wins).
#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct FileModelsConfig {
    cache_dir: Option<PathBuf>,
    registry_path: Option<PathBuf>,
    endpoint: Option<String>,
}

/// The `[local]` TOML table — local-inference worker tuning (every field optional; env wins).
#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct FileLocalConfig {
    worker_bin: Option<PathBuf>,
    n_gpu_layers: Option<u32>,
    n_ctx: Option<u32>,
    n_threads: Option<u32>,
    flash_attn: Option<bool>,
    isq: Option<String>,
    max_tokens: Option<u32>,
    load_timeout_ms: Option<u64>,
    ttft_timeout_ms: Option<u64>,
    inter_token_timeout_ms: Option<u64>,
    max_restarts: Option<u32>,
    restart_window_ms: Option<u64>,
}

fn env_string(key: &str) -> Option<String> {
    std::env::var_os(key).map(|v| v.to_string_lossy().into_owned())
}

/// Parse a permissive boolean (`1`/`true`/`yes`/`on`, case-insensitive => true).
fn parse_bool(s: &str) -> bool {
    matches!(
        s.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

/// Overlay a millisecond [`Duration`] field from an optional TOML value then an env override.
fn resolve_duration_ms(
    field: &mut Duration,
    file: Option<u64>,
    env_key: &str,
) -> anyhow::Result<()> {
    if let Some(ms) = file {
        *field = Duration::from_millis(ms);
    }
    if let Some(s) = env_string(env_key) {
        let ms = s
            .parse::<u64>()
            .with_context(|| format!("{env_key} must be a u64 (milliseconds)"))?;
        *field = Duration::from_millis(ms);
    }
    Ok(())
}

/// Parse a 64-char hex string into a 32-byte journal signer seed.
fn parse_seed(hex: &str) -> anyhow::Result<[u8; 32]> {
    let hex = hex.trim();
    anyhow::ensure!(
        hex.len() == 64,
        "DAEMON_JOURNAL_SEED must be 64 hex chars (32 bytes), got {}",
        hex.len()
    );
    let mut seed = [0u8; 32];
    for (i, byte) in seed.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16)
            .context("DAEMON_JOURNAL_SEED must be valid hex")?;
    }
    Ok(seed)
}

fn default_socket() -> PathBuf {
    let dir = std::env::var_os("TMPDIR").unwrap_or_else(|| "/tmp".into());
    PathBuf::from(dir).join("daemon-api.sock")
}

/// The default host data directory: `$DAEMON_DATA_DIR` is resolved first by the caller; this fallback
/// prefers `$XDG_DATA_HOME/daemon`, then `$HOME/.local/share/daemon`, else a temp-dir `daemon` home.
fn default_data_dir() -> PathBuf {
    if let Some(xdg) = std::env::var_os("XDG_DATA_HOME").filter(|v| !v.is_empty()) {
        return PathBuf::from(xdg).join("daemon");
    }
    if let Some(home) = std::env::var_os("HOME").filter(|v| !v.is_empty()) {
        return PathBuf::from(home).join(".local/share/daemon");
    }
    std::env::temp_dir().join("daemon")
}

impl NodeConfig {
    /// The profile-scoped data home (`<data_dir>/<profile>/`) rooting this node's §10/§11 subsystem
    /// databases. Mirrors hermes' per-profile layout so different profiles never share a memory bank.
    pub fn profile_home(&self) -> PathBuf {
        self.data_dir.join(&self.profile)
    }

    /// Whether the §10/§11 providers persist to disk. Durability follows the store backend: an
    /// in-memory session store (the zero-config default) keeps memory/context ephemeral too, so the
    /// default node is fully ephemeral and coherent; the SQLite backend persists under [`Self::profile_home`].
    pub fn persist_providers(&self) -> bool {
        matches!(self.store, StoreBackend::Sqlite { .. })
    }

    /// Load the layered config: read the optional TOML file at `$DAEMON_CONFIG`, then overlay env.
    pub fn load() -> anyhow::Result<Self> {
        let file = match std::env::var_os(CONFIG_ENV) {
            Some(path) => {
                let text = std::fs::read_to_string(&path)
                    .with_context(|| format!("reading config file {}", path.to_string_lossy()))?;
                toml::from_str::<FileConfig>(&text).context("parsing TOML config")?
            }
            None => FileConfig::default(),
        };

        let partition = match env_string(PARTITION_ENV) {
            Some(s) => PartitionId(s.parse().context("DAEMON_PARTITION must be a u64")?),
            None => file
                .partition
                .map(PartitionId)
                .unwrap_or(PartitionId::DEFAULT),
        };

        let store = Self::resolve_store(&file)?;
        let data_dir = env_string(DATA_DIR_ENV)
            .map(PathBuf::from)
            .or_else(|| file.data_dir.clone())
            .unwrap_or_else(default_data_dir);
        // Resolve engine tunables before the `String`/`PathBuf` fields below partially move `file`.
        let engine = Self::resolve_engine(&file)?;

        let socket_path = env_string(API_SOCKET_ENV)
            .map(PathBuf::from)
            .or(file.socket_path)
            .unwrap_or_else(default_socket);

        let http_addr = env_string(HTTP_ADDR_ENV).or(file.http_addr);

        let dispatch_interval = Duration::from_millis(file.dispatch_interval_ms.unwrap_or(2));
        let scan_interval = Duration::from_millis(file.scan_interval_ms.unwrap_or(10));

        let profile = env_string(PROFILE_ENV)
            .or(file.profile)
            .unwrap_or_else(|| "openai".to_string());

        let context_engine = match env_string(CONTEXT_ENGINE_ENV)
            .or(file.context_engine)
            .unwrap_or_else(|| "lcm".to_string())
            .to_ascii_lowercase()
            .as_str()
        {
            "lcm" => ContextEngineKind::Lcm,
            "budgeted" | "none" | "default" => ContextEngineKind::Budgeted,
            other => anyhow::bail!("unknown context engine {other:?} (expected lcm|budgeted|none)"),
        };

        let memory_provider = match env_string(MEMORY_PROVIDER_ENV)
            .or(file.memory_provider)
            .unwrap_or_else(|| "mnemosyne".to_string())
            .to_ascii_lowercase()
            .as_str()
        {
            "mnemosyne" => MemoryProviderKind::Mnemosyne,
            "file" => MemoryProviderKind::File,
            "none" | "off" => MemoryProviderKind::None,
            other => {
                anyhow::bail!("unknown memory provider {other:?} (expected mnemosyne|file|none)")
            }
        };

        let memory_file = env_string(MEMORY_FILE_ENV)
            .map(PathBuf::from)
            .or(file.memory_file);

        let provider_kind = match env_string(MODEL_PROVIDER_ENV)
            .or(file.model_provider)
            .unwrap_or_else(|| "mock".to_string())
            .to_ascii_lowercase()
            .as_str()
        {
            "mock" => ProviderKind::Mock,
            "openai" => ProviderKind::OpenAi,
            "anthropic" => ProviderKind::Anthropic,
            "llama" | "llamacpp" | "llama-cpp" => ProviderKind::LlamaCpp,
            "mistralrs" | "mistral-rs" | "mistral.rs" => ProviderKind::MistralRs,
            other => anyhow::bail!(
                "unknown model provider {other:?} (expected mock|openai|anthropic|llama|mistralrs)"
            ),
        };
        // No default: `None` lets the provider client use its own default endpoint. An override is
        // only meaningful for a gateway/proxy or the in-process wire tests.
        let base_url = env_string(BASE_URL_ENV).or(file.base_url);
        let model = env_string(MODEL_ENV)
            .or(file.model)
            .unwrap_or_else(|| match provider_kind {
                ProviderKind::OpenAi => "gpt-4o-mini".to_string(),
                ProviderKind::Anthropic => "claude-3-5-sonnet-latest".to_string(),
                // Mock and the local kinds have no sensible default model — set DAEMON_MODEL.
                ProviderKind::Mock | ProviderKind::LlamaCpp | ProviderKind::MistralRs => {
                    String::new()
                }
            });

        let credential_key = env_string(CREDENTIAL_KEY_ENV)
            .or(file.credential_key)
            .unwrap_or_else(|| "sk-configured".to_string());

        let journal_seed = match env_string(JOURNAL_SEED_ENV).or(file.journal_seed) {
            Some(hex) => Some(parse_seed(&hex)?),
            None => None,
        };

        let nesting_depth = match env_string(NESTING_DEPTH_ENV) {
            Some(s) => s.parse().context("DAEMON_NESTING_DEPTH must be a usize")?,
            None => file.nesting_depth.unwrap_or(0),
        };

        let local = Self::resolve_local(file.local.unwrap_or_default())?;
        let models = Self::resolve_models(file.models.unwrap_or_default());
        let embed = Self::resolve_embed(file.embed.unwrap_or_default())?;

        Ok(Self {
            partition,
            socket_path,
            http_addr,
            store,
            data_dir,
            dispatch_interval,
            scan_interval,
            profile,
            context_engine,
            memory_provider,
            memory_file,
            provider_kind,
            base_url,
            model,
            local,
            models,
            embed,
            credential_key,
            engine,
            journal_seed,
            nesting_depth,
        })
    }

    /// Resolve embeddings tuning (env overriding the `[embed]` TOML table).
    fn resolve_embed(file: FileEmbedConfig) -> anyhow::Result<EmbedConfig> {
        let kind = match env_string(EMBED_PROVIDER_ENV)
            .or(file.provider)
            .unwrap_or_else(|| "off".to_string())
            .to_ascii_lowercase()
            .as_str()
        {
            "off" | "none" => EmbedKind::Off,
            "genai" | "remote" | "openai" => EmbedKind::Genai,
            "local" | "daemon-infer" => EmbedKind::Local,
            other => {
                anyhow::bail!("unknown embed provider {other:?} (expected off|genai|local)")
            }
        };
        let model = env_string(EMBED_MODEL_ENV)
            .or(file.model)
            .unwrap_or_default();
        let dims = match env_string(EMBED_DIMS_ENV) {
            Some(s) => s.parse().context("DAEMON_EMBED_DIMS must be a usize")?,
            None => file.dims.unwrap_or(0),
        };
        let base_url = env_string(EMBED_BASE_URL_ENV).or(file.base_url);
        let engine = env_string(EMBED_ENGINE_ENV)
            .or(file.engine)
            .unwrap_or_else(|| "llama".to_string());
        Ok(EmbedConfig {
            kind,
            model,
            dims,
            base_url,
            engine,
        })
    }

    /// Resolve model-management tuning (env overriding the `[models]` TOML table).
    fn resolve_models(file: FileModelsConfig) -> ModelsConfig {
        let cache_dir = env_string(MODELS_CACHE_DIR_ENV)
            .map(PathBuf::from)
            .or(file.cache_dir);
        let registry_path = env_string(MODELS_REGISTRY_ENV)
            .map(PathBuf::from)
            .or(file.registry_path);
        let endpoint = env_string(MODELS_ENDPOINT_ENV).or(file.endpoint);
        ModelsConfig {
            cache_dir,
            registry_path,
            endpoint,
        }
    }

    /// Resolve the local-inference worker tuning (env overriding the `[local]` TOML table overriding
    /// [`LocalConfig`] defaults).
    fn resolve_local(file: FileLocalConfig) -> anyhow::Result<LocalConfig> {
        let mut local = LocalConfig::default();
        if let Some(path) = file.worker_bin {
            local.worker_bin = path;
        }
        if let Some(s) = env_string(INFER_BIN_ENV) {
            local.worker_bin = PathBuf::from(s);
        }
        if let Some(n) = file.n_gpu_layers {
            local.n_gpu_layers = n;
        }
        if let Some(s) = env_string(INFER_N_GPU_LAYERS_ENV) {
            local.n_gpu_layers = s
                .parse()
                .context("DAEMON_INFER_N_GPU_LAYERS must be a u32")?;
        }
        if let Some(n) = file.n_ctx {
            local.n_ctx = n;
        }
        if let Some(s) = env_string(INFER_N_CTX_ENV) {
            local.n_ctx = s.parse().context("DAEMON_INFER_N_CTX must be a u32")?;
        }
        if let Some(n) = file.n_threads {
            local.n_threads = Some(n);
        }
        if let Some(s) = env_string(INFER_N_THREADS_ENV) {
            local.n_threads = Some(s.parse().context("DAEMON_INFER_N_THREADS must be a u32")?);
        }
        if let Some(b) = file.flash_attn {
            local.flash_attn = b;
        }
        if let Some(s) = env_string(INFER_FLASH_ATTN_ENV) {
            local.flash_attn = parse_bool(&s);
        }
        if let Some(isq) = file.isq {
            local.isq = Some(isq);
        }
        if let Some(s) = env_string(INFER_ISQ_ENV) {
            local.isq = Some(s);
        }
        if let Some(n) = file.max_tokens {
            local.max_tokens = n;
        }
        if let Some(s) = env_string(INFER_MAX_TOKENS_ENV) {
            local.max_tokens = s.parse().context("DAEMON_INFER_MAX_TOKENS must be a u32")?;
        }
        resolve_duration_ms(
            &mut local.load_timeout,
            file.load_timeout_ms,
            INFER_LOAD_TIMEOUT_MS_ENV,
        )?;
        resolve_duration_ms(
            &mut local.ttft_timeout,
            file.ttft_timeout_ms,
            INFER_TTFT_TIMEOUT_MS_ENV,
        )?;
        resolve_duration_ms(
            &mut local.inter_token_timeout,
            file.inter_token_timeout_ms,
            INFER_INTER_TOKEN_TIMEOUT_MS_ENV,
        )?;
        if let Some(n) = file.max_restarts {
            local.max_restarts = n;
        }
        if let Some(s) = env_string(INFER_MAX_RESTARTS_ENV) {
            local.max_restarts = s
                .parse()
                .context("DAEMON_INFER_MAX_RESTARTS must be a u32")?;
        }
        resolve_duration_ms(
            &mut local.restart_window,
            file.restart_window_ms,
            INFER_RESTART_WINDOW_MS_ENV,
        )?;
        Ok(local)
    }

    /// Resolve the engine tunables (§20), env overriding TOML overriding [`daemon_core::Config`]
    /// defaults.
    fn resolve_engine(file: &FileConfig) -> anyhow::Result<daemon_core::Config> {
        let mut engine = daemon_core::Config::default();
        if let Some(n) = file.model_retry_attempts {
            engine.model_retry_attempts = n;
        }
        if let Some(s) = env_string(MODEL_RETRY_ATTEMPTS_ENV) {
            engine.model_retry_attempts = s
                .parse()
                .context("DAEMON_MODEL_RETRY_ATTEMPTS must be a u8")?;
        }
        if let Some(n) = file.context_budget_tokens {
            engine.context_budget_tokens = Some(n);
        }
        if let Some(s) = env_string(CONTEXT_BUDGET_TOKENS_ENV) {
            engine.context_budget_tokens = Some(
                s.parse()
                    .context("DAEMON_CONTEXT_BUDGET_TOKENS must be a u32")?,
            );
        }
        if let Some(n) = file.max_iterations {
            engine.max_iterations = n;
        }
        if let Some(s) = env_string(MAX_ITERATIONS_ENV) {
            engine.max_iterations = s.parse().context("DAEMON_MAX_ITERATIONS must be a u32")?;
        }
        if let Some(n) = file.tool_result_budget {
            engine.tool_result_budget = n;
        }
        if let Some(s) = env_string(TOOL_RESULT_BUDGET_ENV) {
            engine.tool_result_budget = s
                .parse()
                .context("DAEMON_TOOL_RESULT_BUDGET must be a usize")?;
        }
        Ok(engine)
    }

    fn resolve_store(file: &FileConfig) -> anyhow::Result<StoreBackend> {
        let kind = env_string(STORE_ENV)
            .or_else(|| file.store.clone())
            .unwrap_or_else(|| "memory".to_string());
        match kind.as_str() {
            "memory" => Ok(StoreBackend::Memory),
            "sqlite" => {
                let path = env_string(STORE_PATH_ENV)
                    .map(PathBuf::from)
                    .or_else(|| file.store_path.clone())
                    .unwrap_or_else(|| {
                        let dir = std::env::var_os("TMPDIR").unwrap_or_else(|| "/tmp".into());
                        PathBuf::from(dir).join("daemon-store.sqlite")
                    });
                Ok(StoreBackend::Sqlite { path })
            }
            other => anyhow::bail!("unknown store backend {other:?} (expected memory|sqlite)"),
        }
    }
}
