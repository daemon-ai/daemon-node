// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The node's layered configuration: composed by [`figment`] from four sources, later ones winning:
//! defaults ([`NodeConfig::default`]) <- an optional TOML file (`$DAEMON_CONFIG`) <- environment
//! (`DAEMON_*`) <- CLI overrides. This is the *composition-layer* config (partition, socket, store
//! backend, resident cadence, provider/credential selection) — distinct from the engine tunables
//! ([`daemon_core::Config`], nested here as `[engine]`), which the host injects via the `EngineProfile`.
//!
//! Naming is mechanical and needs no hand-maintained list: `NodeConfig` (and its nested structs) is
//! the single source of truth, and every environment variable is `DAEMON_` + the serde path,
//! uppercased, with `__` between struct levels (e.g. `python.op_timeout_ms` <- `DAEMON_PYTHON__OP_TIMEOUT_MS`,
//! `engine.model_retry_attempts` <- `DAEMON_ENGINE__MODEL_RETRY_ATTEMPTS`). `Env::prefixed("DAEMON_").split("__")`
//! performs the whole mapping. Unknown keys are ignored (env carries sibling `DAEMON_*` vars the
//! config does not own — admin creds, the config-file path, test knobs), so `deny_unknown_fields` is
//! intentionally not used on the merged extract.

use anyhow::Context;
use daemon_common::PartitionId;
use figment::providers::{Env, Format, Serialized, Toml};
use figment::Figment;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;

/// The environment variable naming the optional TOML config file (read directly to pick the file
/// provider; the `Env` layer ignores the resulting `config` key).
const CONFIG_ENV: &str = "DAEMON_CONFIG";

// --- serde helpers ---------------------------------------------------------------------------

/// (De)serialize a [`Duration`] as whole milliseconds (`u64`) — the `*_ms` TOML/env convention.
mod duration_ms {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::Duration;

    pub fn serialize<S: Serializer>(d: &Duration, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_u64(u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Duration, D::Error> {
        Ok(Duration::from_millis(u64::deserialize(d)?))
    }
}

/// (De)serialize the optional 32-byte verifiable-journal seed as a 64-char hex string.
mod journal_seed_hex {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(v: &Option<[u8; 32]>, s: S) -> Result<S::Ok, S::Error> {
        match v {
            Some(bytes) => s.serialize_some(&super::hex_encode(bytes)),
            None => s.serialize_none(),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<[u8; 32]>, D::Error> {
        match Option::<String>::deserialize(d)? {
            Some(hex) => Ok(Some(
                super::parse_seed(&hex).map_err(serde::de::Error::custom)?,
            )),
            None => Ok(None),
        }
    }
}

/// (De)serialize a list of `(key, value)` pairs as a map (the `env = { K = "v" }` TOML shape).
mod kv_map {
    use serde::ser::SerializeMap;
    use serde::{Deserialize, Deserializer, Serializer};
    use std::collections::BTreeMap;

    pub fn serialize<S: Serializer>(v: &[(String, String)], s: S) -> Result<S::Ok, S::Error> {
        let mut m = s.serialize_map(Some(v.len()))?;
        for (k, val) in v {
            m.serialize_entry(k, val)?;
        }
        m.end()
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<(String, String)>, D::Error> {
        Ok(BTreeMap::<String, String>::deserialize(d)?
            .into_iter()
            .collect())
    }
}

/// Parse a 64-char hex string into a 32-byte journal signer seed.
fn parse_seed(hex: &str) -> anyhow::Result<[u8; 32]> {
    let hex = hex.trim();
    anyhow::ensure!(
        hex.len() == 64,
        "journal_seed must be 64 hex chars (32 bytes), got {}",
        hex.len()
    );
    let mut seed = [0u8; 32];
    for (i, byte) in seed.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16)
            .context("journal_seed must be valid hex")?;
    }
    Ok(seed)
}

/// Hex-encode 32 bytes (lower-case) for the journal-seed round-trip.
fn hex_encode(bytes: &[u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    for b in bytes {
        use std::fmt::Write as _;
        let _ = write!(s, "{b:02x}");
    }
    s
}

fn default_socket() -> PathBuf {
    let dir = std::env::var_os("TMPDIR").unwrap_or_else(|| "/tmp".into());
    PathBuf::from(dir).join("daemon-api.sock")
}

/// The default host data directory: `$XDG_DATA_HOME/daemon`, then `$HOME/.local/share/daemon`, else
/// a temp-dir `daemon` home. (`DAEMON_DATA_DIR` overrides this via the normal env layer.)
fn default_data_dir() -> PathBuf {
    if let Some(xdg) = std::env::var_os("XDG_DATA_HOME").filter(|v| !v.is_empty()) {
        return PathBuf::from(xdg).join("daemon");
    }
    if let Some(home) = std::env::var_os("HOME").filter(|v| !v.is_empty()) {
        return PathBuf::from(home).join(".local/share/daemon");
    }
    std::env::temp_dir().join("daemon")
}

fn default_profile() -> String {
    "openai".to_string()
}

fn default_true() -> bool {
    true
}

fn default_mcp_op_timeout() -> Duration {
    Duration::from_millis(60_000)
}

// --- selector enums (declarative string mapping via serde aliases) ---------------------------

/// Which model provider implementation the node uses. Canonical values are lower-case; the legacy
/// per-family names are accepted as aliases (they all map to the single genai-backed client).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderKind {
    /// The deterministic in-tree provider (zero-config; no network/keys).
    Mock,
    /// The deterministic in-tree scripted provider replaying a fixed script (`mock_script`).
    #[serde(alias = "script")]
    Scripted,
    /// Any networked provider served by `genai`; the adapter is inferred from the model name.
    #[serde(
        rename = "genai",
        alias = "openai",
        alias = "anthropic",
        alias = "gemini",
        alias = "google",
        alias = "groq",
        alias = "deepseek",
        alias = "deep_seek",
        alias = "xai",
        alias = "grok",
        alias = "openrouter",
        alias = "open_router",
        alias = "cohere"
    )]
    GenAi,
    /// The daemon-api OpenRouter-clone gateway (OpenAI-compatible).
    #[serde(rename = "daemon_api", alias = "daemonapi")]
    DaemonApi,
    /// A local llama.cpp model via the supervised `daemon-infer` worker.
    #[serde(rename = "llama", alias = "llamacpp", alias = "llama_cpp")]
    LlamaCpp,
    /// A local mistral.rs model via the supervised `daemon-infer` worker.
    #[serde(rename = "mistralrs", alias = "mistral_rs")]
    MistralRs,
}

/// Which default context engine (§10) the node wires into every engine it builds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContextEngineKind {
    /// The native LCM port (`daemon-context-lcm`) — the default.
    Lcm,
    /// The in-core drop-oldest budgeted engine; also selected by `none`/`default`.
    #[serde(alias = "none", alias = "default")]
    Budgeted,
}

/// Which default memory provider (§11) the node wires into every engine it builds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryProviderKind {
    /// The native Mnemosyne port (`daemon-mnemosyne`) — the default.
    Mnemosyne,
    /// The in-core `FileMemory` over a frozen snapshot file.
    File,
    /// No memory provider (memory off).
    #[serde(alias = "off")]
    None,
}

/// Which embedding backend Mnemosyne uses for vector recall.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbedKind {
    /// No embeddings — recall is keyword-only (the zero-config default).
    #[serde(alias = "none")]
    Off,
    /// A remote, OpenAI-compatible embeddings API via `genai`.
    #[serde(rename = "genai", alias = "remote", alias = "openai")]
    Genai,
    /// A local embedding model via a supervised `daemon-infer` worker.
    #[serde(alias = "daemon-infer", alias = "daemon_infer")]
    Local,
}

/// The durable store backend selector (the config surface; `store` + `store_path`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StoreKind {
    /// The in-memory backend (non-durable; the default).
    Memory,
    /// The SQLite backend at `store_path` (default `$TMPDIR/daemon-store.sqlite`).
    Sqlite,
}

/// The resolved durable store backend (runtime view assembled by [`NodeConfig::store_backend`]).
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

// --- worker / tool sub-configs ----------------------------------------------------------------

/// Tuning for the local-inference [`daemon-infer`] worker (used only for the local provider kinds).
/// Exposed as the `[infer]` TOML table / `DAEMON_INFER__*` env.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
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
    #[serde(with = "daemon_common::flex_bool")]
    pub flash_attn: bool,
    /// mistral.rs in-situ quantization spec (e.g. `Q4K`); `None` = load as-is.
    pub isq: Option<String>,
    /// The output-token cap per generation (`0` = the worker default).
    pub max_tokens: u32,
    /// How long to wait for `Event::Ready` after load.
    #[serde(rename = "load_timeout_ms", with = "duration_ms")]
    pub load_timeout: Duration,
    /// Watchdog: max wait for the first token of a generation.
    #[serde(rename = "ttft_timeout_ms", with = "duration_ms")]
    pub ttft_timeout: Duration,
    /// Watchdog: max wait between tokens once streaming.
    #[serde(rename = "inter_token_timeout_ms", with = "duration_ms")]
    pub inter_token_timeout: Duration,
    /// Crash-loop meltdown: max restarts within [`LocalConfig::restart_window`].
    pub max_restarts: u32,
    /// The sliding window over which restarts are counted for meltdown.
    #[serde(rename = "restart_window_ms", with = "duration_ms")]
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

/// Tuning for the MeTTa symbolic coprocessor (`daemon-metta`). `[metta]` / `DAEMON_METTA__*`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct MettaConfig {
    /// Whether to register the `metta` tool (spawning the supervised worker on first use).
    #[serde(with = "daemon_common::flex_bool")]
    pub enable: bool,
    /// Path to the `daemon-metta` worker binary.
    pub worker_bin: PathBuf,
    /// The worker's durable state directory (`None` => ephemeral / in-memory).
    pub state_dir: Option<PathBuf>,
    /// Default bounded-eval step cap.
    pub max_steps: u64,
    /// Default bounded-eval timeout (ms).
    pub timeout_ms: u64,
    /// Default bounded-eval result cap.
    pub max_results: u64,
    /// Crash-loop meltdown: max restarts within [`MettaConfig::restart_window`].
    pub max_restarts: u32,
    /// The sliding window over which restarts are counted for meltdown.
    #[serde(rename = "restart_window_ms", with = "duration_ms")]
    pub restart_window: Duration,
}

impl Default for MettaConfig {
    fn default() -> Self {
        Self {
            enable: false,
            worker_bin: default_metta_bin(),
            state_dir: None,
            max_steps: 1_000,
            timeout_ms: 1_000,
            max_results: 100,
            max_restarts: 3,
            restart_window: Duration::from_secs(60),
        }
    }
}

/// The default worker binary: a `daemon-metta` next to the running daemon executable, else a bare
/// `daemon-metta` (resolved on `PATH`).
fn default_metta_bin() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|dir| dir.join("daemon-metta")))
        .unwrap_or_else(|| PathBuf::from("daemon-metta"))
}

/// Tuning for the Python tools worker (`daemon-pytool`). `[python]` / `DAEMON_PYTHON__*`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct PythonToolsConfig {
    /// Whether to discover + register Python tools.
    #[serde(with = "daemon_common::flex_bool")]
    pub enable: bool,
    /// The Python interpreter to spawn the worker with (when `worker_bin` is unset).
    pub interpreter: PathBuf,
    /// The worker module run as `python -m <module>`.
    pub worker_module: String,
    /// A standalone worker executable; spawned directly instead of `interpreter -m module` when set.
    pub worker_bin: Option<PathBuf>,
    /// A directory of user tool modules (imported for their `@tool` registrations).
    pub tools_dir: Option<PathBuf>,
    /// A path prepended to the worker's `PYTHONPATH` so `-m <module>` resolves the shipped package.
    pub package_path: Option<PathBuf>,
    /// How long to wait for a tool call / discovery reply (the transport-fault watchdog).
    #[serde(rename = "op_timeout_ms", with = "duration_ms")]
    pub op_timeout: Duration,
    /// How long to wait for the worker's `Ready` after spawning.
    #[serde(rename = "spawn_timeout_ms", with = "duration_ms")]
    pub spawn_timeout: Duration,
    /// Crash-loop meltdown: max restarts within [`PythonToolsConfig::restart_window`].
    pub max_restarts: u32,
    /// The sliding window over which restarts are counted for meltdown.
    #[serde(rename = "restart_window_ms", with = "duration_ms")]
    pub restart_window: Duration,
}

impl Default for PythonToolsConfig {
    fn default() -> Self {
        Self {
            enable: false,
            interpreter: PathBuf::from("python3"),
            worker_module: "daemon_pytool".to_string(),
            worker_bin: None,
            tools_dir: None,
            package_path: None,
            op_timeout: Duration::from_secs(60),
            spawn_timeout: Duration::from_secs(30),
            max_restarts: 3,
            restart_window: Duration::from_secs(60),
        }
    }
}

/// How a [`McpServerEntry`] reaches its server (internally tagged by `transport`).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "transport", rename_all = "snake_case")]
pub enum McpTransportEntry {
    /// Spawn a local server binary and speak MCP over its stdio.
    Stdio {
        /// The program to exec.
        command: String,
        /// Arguments passed to the program.
        #[serde(default)]
        args: Vec<String>,
        /// Extra environment variables set on the child.
        #[serde(default, with = "kv_map")]
        env: Vec<(String, String)>,
    },
    /// Connect to a remote server over streamable HTTP.
    Http {
        /// The base MCP endpoint.
        url: String,
    },
}

/// One configured MCP server (`[[mcp.servers]]`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct McpServerEntry {
    /// A short, stable name used for tool namespacing + diagnostics.
    pub name: String,
    /// Whether to connect to + register this server's tools.
    #[serde(default = "default_true", with = "daemon_common::flex_bool")]
    pub enable: bool,
    /// How to reach the server (the flattened `transport` discriminator + its fields).
    #[serde(flatten)]
    pub transport: McpTransportEntry,
    /// Per-operation timeout (discovery / a tool call).
    #[serde(
        rename = "op_timeout_ms",
        default = "default_mcp_op_timeout",
        with = "duration_ms"
    )]
    pub op_timeout: Duration,
}

/// MCP-client tuning: the external servers the daemon connects to. `[mcp]` (`[[mcp.servers]]`).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct McpConfig {
    /// The configured servers (each contributes `mcp__{name}__{tool}` tools when reachable).
    pub servers: Vec<McpServerEntry>,
}

/// Tuning for the web tools (`daemon-tool-web`). `[web]` / `DAEMON_WEB__*`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct WebConfig {
    /// Whether to register the `web_search` + `web_extract` tools.
    #[serde(with = "daemon_common::flex_bool")]
    pub enable: bool,
    /// Include the dependency-light local `reqwest`+readability `web_extract` fallback.
    #[serde(with = "daemon_common::flex_bool")]
    pub local_fallback: bool,
    /// The credential-profile id the Tavily search key is read from.
    pub tavily_key_id: String,
    /// The credential-profile id the Firecrawl scraper key is read from.
    pub firecrawl_key_id: String,
}

impl Default for WebConfig {
    fn default() -> Self {
        Self {
            enable: false,
            local_fallback: true,
            tavily_key_id: "tavily".to_string(),
            firecrawl_key_id: "firecrawl".to_string(),
        }
    }
}

/// Tuning for the `browser` tool (`daemon-tool-browser`). `[browser]` / `DAEMON_BROWSER__*`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct BrowserConfig {
    /// Whether to register the `browser` tool (launching Chromium lazily on first use).
    #[serde(with = "daemon_common::flex_bool")]
    pub enable: bool,
    /// An explicit Chromium/Chrome executable path (`None` => chromiumoxide auto-detection).
    pub chrome_path: Option<PathBuf>,
    /// Run headless (the default; `false` shows a window — local debugging only).
    #[serde(with = "daemon_common::flex_bool")]
    pub headless: bool,
    /// The screenshot output directory (`None` => `<profile_home>/browser/screenshots`).
    pub screenshot_dir: Option<PathBuf>,
    /// Require interactive host approval before each navigation.
    #[serde(with = "daemon_common::flex_bool")]
    pub approve_navigation: bool,
    /// The browser launch timeout.
    #[serde(rename = "launch_timeout_ms", with = "duration_ms")]
    pub launch_timeout: Duration,
    /// Auto-dismiss JS dialogs so a modal cannot wedge the session.
    #[serde(with = "daemon_common::flex_bool")]
    pub auto_dismiss_dialogs: bool,
}

impl Default for BrowserConfig {
    fn default() -> Self {
        Self {
            enable: false,
            chrome_path: None,
            headless: true,
            screenshot_dir: None,
            approve_navigation: false,
            launch_timeout: Duration::from_secs(20),
            auto_dismiss_dialogs: true,
        }
    }
}

/// Tuning for the skills subsystem. `[skills]` / `DAEMON_SKILLS__*`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct SkillsConfig {
    /// Whether the skills subsystem is active.
    #[serde(with = "daemon_common::flex_bool")]
    pub enable: bool,
    /// The skills root directory (`None` => `<profile_home>/skills`).
    pub dir: Option<PathBuf>,
}

impl Default for SkillsConfig {
    fn default() -> Self {
        Self {
            enable: true,
            dir: None,
        }
    }
}

/// Tuning for the embeddings backend. `[embed]` / `DAEMON_EMBED__*`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct EmbedConfig {
    /// Which backend to use (off|genai|local). TOML/env key `provider`.
    #[serde(rename = "provider")]
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

/// Tuning for the `daemon-models` model-management facade. `[models]` / `DAEMON_MODELS__*`.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelsConfig {
    /// The shared Hugging Face hub cache directory; `None` follows the `HF_*`/XDG precedence.
    pub cache_dir: Option<PathBuf>,
    /// The catalog manifest path; `None` places it next to the cache.
    pub registry_path: Option<PathBuf>,
    /// The Hugging Face Hub endpoint; `None` uses the default.
    pub endpoint: Option<String>,
}

/// The `[api]` transport surface: the networked TLS/TCP listener + identity-store path.
/// `[api]` / `DAEMON_API__*`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ApiConfig {
    /// TLS/TCP bind address; `None` keeps the node Unix-socket-only (the prior behavior).
    pub tls_addr: Option<String>,
    /// PEM server certificate chain (required when `tls_addr` is set).
    pub tls_cert: Option<PathBuf>,
    /// PEM server private key (required when `tls_addr` is set).
    pub tls_key: Option<PathBuf>,
    /// Require + verify a client certificate (mTLS) on the TLS transport.
    #[serde(with = "daemon_common::flex_bool")]
    pub require_client_cert: bool,
    /// PEM CA bundle trusted to sign client certs (required with `require_client_cert`).
    pub tls_client_ca: Option<PathBuf>,
    /// SQLite identity store path (`None` => `<data_dir>/auth.sqlite`; see [`NodeConfig::auth_db`]).
    pub auth_db: Option<PathBuf>,
    /// The local-trust principal for the Unix socket / FFI / in-process HTTP. `Some(name)` (default
    /// `"system"`) binds a full-trust local context; `""`/`off`/`none` (normalized to `None`) makes
    /// the Unix socket require SCRAM and fully gates HTTP. TCP/TLS always requires authentication.
    pub local_trust: Option<String>,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            tls_addr: None,
            tls_cert: None,
            tls_key: None,
            require_client_cert: false,
            tls_client_ca: None,
            auth_db: None,
            local_trust: Some("system".to_string()),
        }
    }
}

/// A single declarative routing rule (§5.9) — `[[routing.route]]`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct RouteRule {
    /// Match this exact instance-qualified transport id (e.g. `matrix/@bot:hs.org`). Mutually
    /// exclusive with [`RouteRule::transport_family`].
    pub transport: Option<String>,
    /// Match any instance of this transport family (the `family/...` prefix before the first `/`).
    pub transport_family: Option<String>,
    /// Scope kind to match: `dm` | `group` | `api` | `internal` | `any` (default `any`).
    pub scope: String,
    /// For `group` scope, the chat-handle `*`-glob (default `*`).
    pub chat_glob: String,
    /// The profile override for matched origins (precedence step 1); `None` falls through.
    pub profile: Option<String>,
    /// Isolation policy for naming: `per_user` | `per_chat` | `per_thread` | `shared` (default
    /// `per_thread`).
    pub isolation: String,
}

impl Default for RouteRule {
    fn default() -> Self {
        Self {
            transport: None,
            transport_family: None,
            scope: "any".to_string(),
            chat_glob: "*".to_string(),
            profile: None,
            isolation: "per_thread".to_string(),
        }
    }
}

/// Binds a transport instance to a default profile — `[[routing.instance_profile]]` (§5.9 step 2).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InstanceProfile {
    /// The instance-qualified transport id.
    pub transport: String,
    /// The profile all of that instance's origins run under unless a route overrides.
    pub profile: String,
}

/// The `[routing]` config: the general host routing table (§5.9). Empty by default.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct RoutingConfig {
    /// The node default profile for routed submits with no matching route/instance (step 3).
    pub default_profile: Option<String>,
    /// Per-instance default profiles (step 2). TOML key `instance_profile`.
    #[serde(rename = "instance_profile")]
    pub instance_profiles: Vec<InstanceProfile>,
    /// Ordered routing rules (first match wins). TOML key `route`.
    #[serde(rename = "route")]
    pub routes: Vec<RouteRule>,
}

impl RoutingConfig {
    /// Whether the table carries no routing information at all.
    pub fn is_empty(&self) -> bool {
        self.default_profile.is_none()
            && self.instance_profiles.is_empty()
            && self.routes.is_empty()
    }
}

/// LCM context-engine tuning (`[lcm]` / `DAEMON_LCM__*`). Injected into the per-node `LcmConfig`
/// template so the context crate itself reads no environment (`data_dir` is set from the profile
/// home separately). Only the two historically env-tunable knobs are surfaced; the remaining
/// Appendix-A compaction constants stay compile-time defaults.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct LcmOpts {
    /// The fraction of the model context window at which compaction triggers (`0.0 < v <= 1.0`).
    pub context_threshold: f64,
    /// The number of most-recent turns always kept verbatim (the fresh tail).
    pub fresh_tail_count: usize,
}

impl Default for LcmOpts {
    fn default() -> Self {
        // Mirrors `daemon_context_lcm::LcmConfig` Appendix-A defaults.
        Self {
            context_threshold: 0.35,
            fresh_tail_count: 32,
        }
    }
}

/// Mnemosyne recall + multi-agent identity knobs (`[mnemosyne]` / `DAEMON_MNEMOSYNE__*`). Injected
/// into the per-node `MnemosyneConfig` template so the memory crate itself reads no environment.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct MnemosyneOpts {
    /// The recall pipeline: `base` (default), `enhanced`, or `polyphonic`.
    pub recall_mode: daemon_mnemosyne::RecallMode,
    /// Enable the opt-in tier-2 LLM conflict detector during sleep.
    #[serde(with = "daemon_common::flex_bool")]
    pub llm_conflict_detection: bool,
    /// Multi-agent identity: the original writer id (stamps rows + widens recall scope).
    pub author_id: Option<String>,
    /// Multi-agent identity: author type (`human`/`agent`/`system`).
    pub author_type: Option<String>,
    /// Multi-agent identity: channel/group id (recall filters on it only when set).
    pub channel_id: Option<String>,
}

// --- the node configuration -------------------------------------------------------------------

/// The node configuration: the single source of truth, deserialized by [`figment`] from
/// defaults <- TOML <- env <- CLI. Field names are the TOML/serde keys; env keys are `DAEMON_` +
/// the (uppercased, `__`-nested) path. Paths that default relative to `data_dir`/`profile_home`
/// are `Option` here and resolved by accessor methods (or [`NodeConfig::finalize`]).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct NodeConfig {
    /// The partition this node owns.
    pub partition: PartitionId,
    /// The Unix socket the node serves its `daemon_api` surface on (env `DAEMON_SOCKET_PATH`).
    pub socket_path: PathBuf,
    /// The optional in-process HTTP/WS surface bind address (`None` leaves it off).
    pub http_addr: Option<String>,
    /// The durable store backend selector (`memory`|`sqlite`).
    pub store: StoreKind,
    /// The SQLite database path when `store = sqlite` (`None` => `$TMPDIR/daemon-store.sqlite`).
    pub store_path: Option<PathBuf>,
    /// The host data directory rooting the profile-scoped subsystem databases (§10/§11).
    pub data_dir: PathBuf,
    /// The parent directory of per-session workspace sandboxes (`None` => `<data_dir>/workspaces`).
    pub workspace_root: Option<PathBuf>,
    /// The content-store (blob CAS) root (`None` => `<data_dir>/blobs`).
    pub blob_root: Option<PathBuf>,
    /// How often the wake/job dispatchers poll the durable outboxes.
    #[serde(rename = "dispatch_interval_ms", with = "duration_ms")]
    pub dispatch_interval: Duration,
    /// How often the recovery scanner re-checks for resumable sessions.
    #[serde(rename = "scan_interval_ms", with = "duration_ms")]
    pub scan_interval: Duration,
    /// The model provider + credential profile name (selects the registered provider builder).
    pub profile: String,
    /// The default context engine (§10) wired into every engine (`lcm` default).
    pub context_engine: ContextEngineKind,
    /// The default memory provider (§11) wired into every engine (`mnemosyne` default).
    pub memory_provider: MemoryProviderKind,
    /// The snapshot file the `file` memory provider serves (when `memory_provider = file`).
    pub memory_file: Option<PathBuf>,
    /// Which model provider implementation to use. `None` = unset (a host launch fails fast).
    /// TOML/env key `model_provider` (`DAEMON_MODEL_PROVIDER`).
    #[serde(rename = "model_provider")]
    pub provider_kind: Option<ProviderKind>,
    /// The scripted provider's replay script (raw JSON); `None` otherwise.
    pub mock_script: Option<String>,
    /// An optional provider API base-URL override (`None` uses the provider's default endpoint).
    pub base_url: Option<String>,
    /// The model name sent to a real provider, or the model path / HF id for a local provider.
    pub model: String,
    /// Local-inference worker tuning (`[infer]`; meaningful only for the local provider kinds).
    pub infer: LocalConfig,
    /// Model-management (search/download/cache/catalog) tuning.
    pub models: ModelsConfig,
    /// Embeddings backend tuning (Mnemosyne vector recall; `Off` by default).
    pub embed: EmbedConfig,
    /// MeTTa symbolic-coprocessor tuning (`enable = false` by default).
    pub metta: MettaConfig,
    /// Python-tools tuning (`enable = false` by default).
    pub python: PythonToolsConfig,
    /// MCP-client tuning (no servers by default).
    pub mcp: McpConfig,
    /// Web-tool tuning (`enable = false` by default).
    pub web: WebConfig,
    /// Browser-tool tuning (`enable = false` by default).
    pub browser: BrowserConfig,
    /// Skills-subsystem tuning (`enable = true` by default).
    pub skills: SkillsConfig,
    /// LCM context-engine tuning (injected into the context-engine template).
    pub lcm: LcmOpts,
    /// Mnemosyne recall + multi-agent identity knobs (injected into the memory provider template).
    pub mnemosyne: MnemosyneOpts,
    /// The credential key the owner authority mints (the daemon-api / provider bearer). Empty means
    /// "no launch credential"; a networked provider then fails fast in [`NodeConfig::validate_for_host`].
    pub credential_key: String,
    /// The engine tunables (§20) injected into every engine (`[engine]` / `DAEMON_ENGINE__*`).
    pub engine: daemon_core::Config,
    /// The 32-byte seed for the node's verifiable-journal signer (hex; `None` => ephemeral per boot).
    #[serde(with = "journal_seed_hex")]
    pub journal_seed: Option<[u8; 32]>,
    /// How many orchestrator levels the top fleet materializes before its engine leaves (`0` flat).
    pub nesting_depth: usize,
    /// The general host routing table (§5.9). Empty by default.
    pub routing: RoutingConfig,
    /// The Matrix chat transport config (`enabled = false` by default).
    pub matrix: daemon_matrix::MatrixConfig,
    /// The internal Rooms loopback transport config (`enabled = false` by default).
    pub rooms: daemon_rooms::RoomsConfig,
    /// The `[api]` transport surface: the networked TLS/TCP listener + identity-store path.
    pub api: ApiConfig,
}

impl Default for NodeConfig {
    fn default() -> Self {
        Self {
            partition: PartitionId::DEFAULT,
            socket_path: default_socket(),
            http_addr: None,
            store: StoreKind::Memory,
            store_path: None,
            data_dir: default_data_dir(),
            workspace_root: None,
            blob_root: None,
            dispatch_interval: Duration::from_millis(2),
            scan_interval: Duration::from_millis(10),
            profile: default_profile(),
            context_engine: ContextEngineKind::Lcm,
            memory_provider: MemoryProviderKind::Mnemosyne,
            memory_file: None,
            provider_kind: None,
            mock_script: None,
            base_url: None,
            model: String::new(),
            infer: LocalConfig::default(),
            models: ModelsConfig::default(),
            embed: EmbedConfig::default(),
            metta: MettaConfig::default(),
            python: PythonToolsConfig::default(),
            mcp: McpConfig::default(),
            web: WebConfig::default(),
            browser: BrowserConfig::default(),
            skills: SkillsConfig::default(),
            lcm: LcmOpts::default(),
            mnemosyne: MnemosyneOpts::default(),
            credential_key: String::new(),
            engine: daemon_core::Config::default(),
            journal_seed: None,
            nesting_depth: 0,
            routing: RoutingConfig::default(),
            matrix: daemon_matrix::MatrixConfig::default(),
            rooms: daemon_rooms::RoomsConfig::default(),
            api: ApiConfig::default(),
        }
    }
}

impl NodeConfig {
    /// The daemon-api gateway default base URL (trailing slash is load-bearing for genai's relative
    /// `Url::join`).
    pub const DAEMON_API_DEFAULT_BASE: &'static str = "https://api.daemon.ai/api/v1/";

    /// Load the layered config: defaults <- optional TOML (`$DAEMON_CONFIG`) <- `DAEMON_*` env.
    pub fn load() -> anyhow::Result<Self> {
        Self::from_figment(Self::base_figment())
    }

    /// The base figment (defaults <- TOML <- env), before any CLI overrides are layered on top.
    pub fn base_figment() -> Figment {
        let mut fig = Figment::from(Serialized::defaults(NodeConfig::default()));
        if let Some(path) = std::env::var_os(CONFIG_ENV) {
            fig = fig.merge(Toml::file(PathBuf::from(path)));
        }
        fig.merge(Env::prefixed("DAEMON_").split("__"))
    }

    /// Extract + finalize a [`NodeConfig`] from a fully-layered figment (the CLI layer, if any, is
    /// merged by the caller before this).
    pub fn from_figment(fig: Figment) -> anyhow::Result<Self> {
        let mut cfg: NodeConfig = fig.extract().context("loading node configuration")?;
        cfg.finalize()?;
        Ok(cfg)
    }

    /// Apply the data_dir-relative + normalization rules that cannot be pure serde defaults, and
    /// validate cross-field invariants.
    fn finalize(&mut self) -> anyhow::Result<()> {
        // The Matrix per-account store root is resolved against the data dir (an absolute override
        // is preserved: `Path::join` with an absolute right-hand side replaces the left).
        let store_root = self.data_dir.join(&self.matrix.store_root);
        self.matrix.store_root = store_root;

        // Local trust: an explicit empty / `off` / `none` / `false` disables the synthetic principal.
        if let Some(v) = &self.api.local_trust {
            let normalized = v.trim();
            self.api.local_trust = match normalized.to_ascii_lowercase().as_str() {
                "" | "off" | "none" | "false" => None,
                _ => Some(normalized.to_string()),
            };
        }

        // A route selects an instance id XOR a transport family, never both.
        for route in &self.routing.routes {
            anyhow::ensure!(
                !(route.transport.is_some() && route.transport_family.is_some()),
                "[[routing.route]] sets both `transport` and `transport_family` (pick one)"
            );
        }
        Ok(())
    }

    /// The resolved durable store backend (combining `store` + `store_path`).
    pub fn store_backend(&self) -> StoreBackend {
        match self.store {
            StoreKind::Memory => StoreBackend::Memory,
            StoreKind::Sqlite => {
                let path = self.store_path.clone().unwrap_or_else(|| {
                    let dir = std::env::var_os("TMPDIR").unwrap_or_else(|| "/tmp".into());
                    PathBuf::from(dir).join("daemon-store.sqlite")
                });
                StoreBackend::Sqlite { path }
            }
        }
    }

    /// The parent directory of per-session workspace sandboxes (`<data_dir>/workspaces` by default).
    pub fn workspace_root(&self) -> PathBuf {
        self.workspace_root
            .clone()
            .unwrap_or_else(|| self.data_dir.join("workspaces"))
    }

    /// The content-store (blob CAS) root (`<data_dir>/blobs` by default).
    pub fn blob_root(&self) -> PathBuf {
        self.blob_root
            .clone()
            .unwrap_or_else(|| self.data_dir.join("blobs"))
    }

    /// The SQLite identity-store path backing authentication (`<data_dir>/auth.sqlite` by default).
    pub fn auth_db(&self) -> PathBuf {
        self.api
            .auth_db
            .clone()
            .unwrap_or_else(|| self.data_dir.join("auth.sqlite"))
    }

    /// The profile-scoped data home (`<data_dir>/<profile>/`) rooting this node's subsystem databases.
    pub fn profile_home(&self) -> PathBuf {
        self.profile_home_for(&self.profile)
    }

    /// The data home for an arbitrary `profile` (`<data_dir>/<profile>/`).
    pub fn profile_home_for(&self, profile: &str) -> PathBuf {
        self.data_dir.join(profile)
    }

    /// The data-dir root that profile homes hang off (`<data_dir>`).
    pub fn data_root(&self) -> PathBuf {
        self.data_dir.clone()
    }

    /// Whether the §10/§11 providers persist to disk (follows the store backend).
    pub fn persist_providers(&self) -> bool {
        matches!(self.store, StoreKind::Sqlite)
    }

    /// Normalize a base URL so it ends with `/` (genai appends a relative adapter suffix to it).
    pub(crate) fn ensure_trailing_slash(base: &str) -> String {
        if base.ends_with('/') {
            base.to_string()
        } else {
            format!("{base}/")
        }
    }

    /// The resolved daemon-api base: the `base_url` override (slash-normalized) when set, else the
    /// [`NodeConfig::DAEMON_API_DEFAULT_BASE`].
    pub fn daemon_api_base(&self) -> String {
        Self::ensure_trailing_slash(
            self.base_url
                .as_deref()
                .unwrap_or(Self::DAEMON_API_DEFAULT_BASE),
        )
    }

    /// The pure boot-resolution core (unit-tested without touching process env): a host now boots
    /// with **no** provider configured (`None` => the node installs [`UnconfiguredProvider`] and
    /// serves; a turn against an unconfigured profile fails clearly, never a silent mock). An
    /// *explicitly-set-but-incomplete* networked provider (`genai`/`daemon_api` with no model) is a
    /// deliberate misconfiguration and still fails fast. A credential is **not** required at boot —
    /// it arrives per-profile over the API via `CredentialSet`.
    fn resolve_provider(
        kind: Option<ProviderKind>,
        model: &str,
    ) -> anyhow::Result<Option<ProviderKind>> {
        let Some(kind) = kind else {
            // Unset: boot unconfigured (no default provider).
            return Ok(None);
        };
        match kind {
            ProviderKind::GenAi | ProviderKind::DaemonApi => {
                anyhow::ensure!(
                    !model.trim().is_empty(),
                    "model provider {kind:?} is set but has no model: set DAEMON_MODEL \
                     (or unset DAEMON_MODEL_PROVIDER to boot unconfigured)"
                );
            }
            ProviderKind::Mock
            | ProviderKind::Scripted
            | ProviderKind::LlamaCpp
            | ProviderKind::MistralRs => {}
        }
        Ok(Some(kind))
    }

    /// Boot-time provider resolution for the **host** role: `None` when no provider is configured
    /// (boot unconfigured), `Some(kind)` when one is set and complete, `Err` when one is set but
    /// incomplete (explicit misconfiguration). Never requires a credential.
    pub fn resolve_for_host(&self) -> anyhow::Result<Option<ProviderKind>> {
        Self::resolve_provider(self.provider_kind, &self.model)
    }
}

/// Generate the config reference (Markdown) from [`NodeConfig::default`] — the single source of
/// truth. Serializing the defaults to JSON and walking it enumerates *every* serializable key
/// (including `Option` fields, which serialize as `null`), so the reference can never omit a field.
/// Each row derives its env var mechanically: `DAEMON_` + the dotted serde path, uppercased, with
/// `__` between nesting levels.
pub fn config_reference() -> String {
    // Normalize the machine-dependent path defaults (home/tmp/exe-relative) to symbolic values so
    // the generated reference is reproducible across machines (the drift gate diffs it verbatim).
    let defaults = NodeConfig {
        data_dir: PathBuf::from("$XDG_DATA_HOME/daemon"),
        socket_path: PathBuf::from("$TMPDIR/daemon-api.sock"),
        infer: LocalConfig {
            worker_bin: PathBuf::from("daemon-infer (next to the daemon binary)"),
            ..LocalConfig::default()
        },
        metta: MettaConfig {
            worker_bin: PathBuf::from("daemon-metta (next to the daemon binary)"),
            ..MettaConfig::default()
        },
        ..NodeConfig::default()
    };
    let value =
        serde_json::to_value(&defaults).expect("NodeConfig::default must serialize to JSON");
    let mut rows: Vec<(String, String, String, String)> = Vec::new();
    walk_reference("", &value, &mut rows);
    rows.sort_by(|a, b| a.0.cmp(&b.0));

    let mut out = String::new();
    out.push_str("# daemon configuration reference\n\n");
    out.push_str(
        "This file is generated from `NodeConfig` (the single source of truth) by \
         `daemon config reference`. Do not edit by hand; run the generator and commit the result \
         (the `check-config-docs` gate diffs it).\n\n",
    );
    out.push_str(
        "Configuration is layered by [figment](https://docs.rs/figment), later sources winning: \
         built-in defaults, then an optional TOML file (`$DAEMON_CONFIG`), then environment \
         variables, then CLI flags. Every environment variable is `DAEMON_` + the TOML path \
         uppercased with `__` between table levels (e.g. `python.op_timeout_ms` \u{2190} \
         `DAEMON_PYTHON__OP_TIMEOUT_MS`).\n\n",
    );
    out.push_str("| TOML path | Environment variable | Type | Default |\n");
    out.push_str("|-----------|----------------------|------|---------|\n");
    for (path, env, ty, default) in &rows {
        out.push_str(&format!("| `{path}` | `{env}` | {ty} | {default} |\n"));
    }
    out
}

/// Recursively flatten a serialized-default JSON value into `(toml_path, env_var, type, default)`
/// reference rows. Objects recurse (nested tables); every other value is a documented leaf.
fn walk_reference(
    prefix: &str,
    value: &serde_json::Value,
    rows: &mut Vec<(String, String, String, String)>,
) {
    if let serde_json::Value::Object(map) = value {
        for (key, child) in map {
            let path = if prefix.is_empty() {
                key.clone()
            } else {
                format!("{prefix}.{key}")
            };
            walk_reference(&path, child, rows);
        }
        return;
    }
    let env = format!("DAEMON_{}", prefix.replace('.', "__").to_uppercase());
    let (ty, default) = match value {
        serde_json::Value::Null => ("optional".to_string(), "_(unset)_".to_string()),
        serde_json::Value::Bool(b) => ("bool".to_string(), format!("`{b}`")),
        serde_json::Value::Number(n) => ("number".to_string(), format!("`{n}`")),
        serde_json::Value::String(s) if s.is_empty() => {
            ("string".to_string(), "`\"\"`".to_string())
        }
        serde_json::Value::String(s) => ("string".to_string(), format!("`{s}`")),
        serde_json::Value::Array(_) => ("array".to_string(), "`[]`".to_string()),
        serde_json::Value::Object(_) => unreachable!("objects recurse above"),
    };
    rows.push((prefix.to_string(), env, ty, default));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unset_provider_boots_unconfigured() {
        // A bare launch (no DAEMON_MODEL_PROVIDER) now BOOTS with no default provider; the node
        // installs UnconfiguredProvider and a turn fails clearly (never a silent mock).
        assert_eq!(
            NodeConfig::resolve_provider(None, "any")
                .expect("an unset provider must boot unconfigured, not fail"),
            None,
        );
    }

    #[test]
    fn explicit_networked_provider_without_a_model_fails_fast() {
        // Explicitly selecting a networked provider without a model is a deliberate misconfig.
        for kind in [ProviderKind::GenAi, ProviderKind::DaemonApi] {
            let err = NodeConfig::resolve_provider(Some(kind), "   ")
                .expect_err("explicit networked provider without a model must fail fast");
            assert!(err.to_string().contains("DAEMON_MODEL"), "{err}");
        }
    }

    #[test]
    fn networked_provider_boots_without_a_credential() {
        // Credentials are provisioned per-profile over the API (CredentialSet), not at boot.
        for kind in [ProviderKind::GenAi, ProviderKind::DaemonApi] {
            assert_eq!(
                NodeConfig::resolve_provider(Some(kind), "author/slug")
                    .unwrap_or_else(|e| panic!("{kind:?} should boot keyless: {e}")),
                Some(kind),
            );
        }
    }

    #[test]
    fn mock_scripted_and_local_need_neither_model_nor_key() {
        for kind in [
            ProviderKind::Mock,
            ProviderKind::Scripted,
            ProviderKind::LlamaCpp,
            ProviderKind::MistralRs,
        ] {
            assert_eq!(
                NodeConfig::resolve_provider(Some(kind), "")
                    .unwrap_or_else(|e| panic!("{kind:?} should resolve keyless/modelless: {e}")),
                Some(kind),
            );
        }
    }

    /// Drift gate: the committed `docs/config-reference.md` must match the generator exactly. Adding
    /// or renaming a `NodeConfig` field without regenerating fails here — docs cannot silently drift.
    /// Regenerate with `daemon config reference > docs/config-reference.md`.
    #[test]
    fn config_reference_is_committed_and_current() {
        let committed = include_str!("../../../docs/config-reference.md");
        assert_eq!(
            config_reference(),
            committed,
            "docs/config-reference.md is stale; regenerate: `daemon config reference > docs/config-reference.md`"
        );
    }

    #[test]
    fn ensure_trailing_slash_normalizes() {
        assert_eq!(
            NodeConfig::ensure_trailing_slash("https://api.daemon.ai/api/v1"),
            "https://api.daemon.ai/api/v1/"
        );
        assert_eq!(
            NodeConfig::ensure_trailing_slash("http://127.0.0.1:8787/api/v1/"),
            "http://127.0.0.1:8787/api/v1/"
        );
    }

    #[test]
    fn defaults_extract_cleanly() {
        let cfg =
            NodeConfig::from_figment(Figment::from(Serialized::defaults(NodeConfig::default())))
                .expect("defaults must extract");
        assert_eq!(cfg.profile, "openai");
        assert_eq!(cfg.context_engine, ContextEngineKind::Lcm);
        assert_eq!(cfg.memory_provider, MemoryProviderKind::Mnemosyne);
        assert!(cfg.provider_kind.is_none());
        assert!(matches!(cfg.store_backend(), StoreBackend::Memory));
    }

    // `figment::Jail::expect_with` dictates a closure returning `Result<(), figment::Error>`; the
    // large `Err` variant is figment's API, not ours.
    #[allow(clippy::result_large_err)]
    #[test]
    fn env_layers_over_defaults_with_nesting() {
        figment::Jail::expect_with(|jail| {
            jail.set_env("DAEMON_MODEL_PROVIDER", "openai"); // legacy alias -> GenAi
            jail.set_env("DAEMON_MODEL", "claude-opus-4-8");
            jail.set_env("DAEMON_PYTHON__ENABLE", "1"); // bool coercion + nested table
            jail.set_env("DAEMON_PYTHON__OP_TIMEOUT_MS", "5000"); // duration ms + nesting
            jail.set_env("DAEMON_ENGINE__MODEL_RETRY_ATTEMPTS", "4"); // engine table
            jail.set_env("DAEMON_STORE", "sqlite");
            let cfg = NodeConfig::from_figment(NodeConfig::base_figment())
                .unwrap_or_else(|e| panic!("env layer must extract: {e:#}"));
            assert_eq!(cfg.provider_kind, Some(ProviderKind::GenAi));
            assert_eq!(cfg.model, "claude-opus-4-8");
            assert!(cfg.python.enable);
            assert_eq!(cfg.python.op_timeout, Duration::from_millis(5000));
            assert_eq!(cfg.engine.model_retry_attempts, 4);
            assert!(matches!(cfg.store_backend(), StoreBackend::Sqlite { .. }));
            Ok(())
        });
    }

    #[allow(clippy::result_large_err)] // figment's `Jail` closure Result type; not ours to shrink.
    #[test]
    fn routing_rejects_transport_and_family_together() {
        figment::Jail::expect_with(|jail| {
            jail.create_file(
                "cfg.toml",
                "[[routing.route]]\ntransport = \"matrix/@a:hs\"\ntransport_family = \"matrix\"\n",
            )?;
            jail.set_env("DAEMON_CONFIG", "cfg.toml");
            let err = NodeConfig::from_figment(NodeConfig::base_figment())
                .expect_err("both transport + family must fail");
            assert!(err.to_string().contains("pick one"), "{err}");
            Ok(())
        });
    }
}
