// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

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
const WORKSPACE_ROOT_ENV: &str = "DAEMON_WORKSPACE_ROOT";
const BLOB_ROOT_ENV: &str = "DAEMON_BLOB_ROOT";
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
/// Selects the model provider implementation: `mock` (default), `scripted`, `openai`, or `anthropic`.
const MODEL_PROVIDER_ENV: &str = "DAEMON_MODEL_PROVIDER";
/// The scripted provider's replay script (a JSON array of `{call,args}`/`{final}` steps). Only read
/// when `DAEMON_MODEL_PROVIDER=scripted`.
const MOCK_SCRIPT_ENV: &str = "DAEMON_MOCK_SCRIPT";
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
/// Overrides the engine's `tool_search_threshold_bytes` (deferrable-schema collapse threshold; 0 off).
const TOOL_SEARCH_THRESHOLD_ENV: &str = "DAEMON_TOOL_SEARCH_THRESHOLD_BYTES";
/// Overrides the engine's `skill_review_interval` (post-turn skill-review nudge cadence; 0 disables).
const SKILL_REVIEW_INTERVAL_ENV: &str = "DAEMON_SKILL_REVIEW_INTERVAL";
/// Overrides the engine's `memory_review_interval` (post-turn memory-review nudge cadence; 0 disables).
const MEMORY_REVIEW_INTERVAL_ENV: &str = "DAEMON_MEMORY_REVIEW_INTERVAL";
/// Toggles the skills subsystem (index + `skill_*` tools + background curation).
const SKILLS_ENABLE_ENV: &str = "DAEMON_SKILLS_ENABLE";
/// Overrides the skills directory (defaults to `<profile_home>/skills`).
const SKILLS_DIR_ENV: &str = "DAEMON_SKILLS_DIR";
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

// --- MeTTa symbolic coprocessor (`daemon-metta`) tuning (DAEMON_METTA_*) ----------------------
/// Enable the `metta` symbolic-coprocessor tool (default: off; opt-in like the HTTP/MCP surfaces).
const METTA_ENABLE_ENV: &str = "DAEMON_METTA_ENABLE";
/// Path to the `daemon-metta` worker binary (default: a `daemon-metta` next to the daemon binary).
const METTA_BIN_ENV: &str = "DAEMON_METTA_BIN";
/// The worker's durable state directory (default: `<profile_home>/metta` when persisting, else
/// ephemeral / in-memory).
const METTA_STATE_DIR_ENV: &str = "DAEMON_METTA_STATE_DIR";
/// Default bounded-eval step cap.
const METTA_MAX_STEPS_ENV: &str = "DAEMON_METTA_MAX_STEPS";
/// Default bounded-eval wall-clock timeout (ms).
const METTA_TIMEOUT_MS_ENV: &str = "DAEMON_METTA_TIMEOUT_MS";
/// Default bounded-eval result cap.
const METTA_MAX_RESULTS_ENV: &str = "DAEMON_METTA_MAX_RESULTS";
/// Crash-loop meltdown: max worker restarts within the restart window.
const METTA_MAX_RESTARTS_ENV: &str = "DAEMON_METTA_MAX_RESTARTS";
/// The sliding window (ms) over which restarts are counted for meltdown.
const METTA_RESTART_WINDOW_MS_ENV: &str = "DAEMON_METTA_RESTART_WINDOW_MS";

// --- Web tools (`daemon-tool-web`) tuning (DAEMON_WEB_*) --------------------------------------
/// Register the `web_search`/`web_extract` tools (`false` by default — opt-in, like `metta`).
const WEB_ENABLE_ENV: &str = "DAEMON_WEB_ENABLE";
/// Include the dependency-light local `reqwest`+readability `web_extract` fallback (default `true`).
const WEB_LOCAL_FALLBACK_ENV: &str = "DAEMON_WEB_LOCAL_FALLBACK";
/// The credential-profile id the Tavily search key is read from (default `tavily`).
const WEB_TAVILY_KEY_ENV: &str = "DAEMON_WEB_TAVILY_KEY_ID";
/// The credential-profile id the Firecrawl scraper key is read from (default `firecrawl`).
const WEB_FIRECRAWL_KEY_ENV: &str = "DAEMON_WEB_FIRECRAWL_KEY_ID";

// --- Matrix transport (`daemon-matrix`) tuning (DAEMON_MATRIX_*) -------------------------------
/// Spawn the Matrix chat transport (`false` by default — opt-in, like `web`/`mcp`). Accounts come
/// from `bound_accounts` + the credential store, never from this config.
const MATRIX_ENABLE_ENV: &str = "DAEMON_MATRIX_ENABLE";
/// The per-account store root, relative to `data_dir` (default `matrix`).
const MATRIX_STORE_ROOT_ENV: &str = "DAEMON_MATRIX_STORE_ROOT";

// --- Rooms transport (`daemon-rooms`) tuning (DAEMON_ROOMS_*) ----------------------------------
/// Spawn the internal Rooms loopback transport (`false` by default — opt-in, like `matrix`). Rooms +
/// membership are durable in the store, never in this config.
const ROOMS_ENABLE_ENV: &str = "DAEMON_ROOMS_ENABLE";
/// The per-Room cascade turn budget (echo-storm cap; `0` = unbounded). Default mirrors `RoomsConfig`.
const ROOMS_MAX_TURNS_ENV: &str = "DAEMON_ROOMS_MAX_TURNS";

// --- Python tools (`daemon-pytool`) tuning (DAEMON_PYTHON_*) -----------------------------------
/// Register Python tools discovered from the `daemon_pytool` worker (`false` by default — opt-in,
/// like `metta`/`web`).
const PYTHON_ENABLE_ENV: &str = "DAEMON_PYTHON_ENABLE";
/// The Python interpreter to spawn the worker with (default `python3`, resolved on `PATH`).
const PYTHON_INTERPRETER_ENV: &str = "DAEMON_PYTHON_INTERPRETER";
/// The worker module run as `python -m <module>` (default `daemon_pytool`).
const PYTHON_WORKER_MODULE_ENV: &str = "DAEMON_PYTHON_WORKER_MODULE";
/// A standalone worker executable; when set it is spawned directly instead of `interpreter -m module`.
const PYTHON_WORKER_BIN_ENV: &str = "DAEMON_PYTHON_WORKER_BIN";
/// A directory of user tool modules (each top-level `*.py` is imported for its `@tool` registrations).
const PYTHON_TOOLS_DIR_ENV: &str = "DAEMON_PYTHON_TOOLS_DIR";
/// A path prepended to the worker's `PYTHONPATH` so `-m <module>` resolves the shipped SDK package.
const PYTHON_PACKAGE_PATH_ENV: &str = "DAEMON_PYTHON_PACKAGE_PATH";
/// How long to wait for a tool call / discovery reply before declaring a transport fault (ms).
const PYTHON_OP_TIMEOUT_MS_ENV: &str = "DAEMON_PYTHON_OP_TIMEOUT_MS";
/// How long to wait for the worker's `Ready` after spawning (ms).
const PYTHON_SPAWN_TIMEOUT_MS_ENV: &str = "DAEMON_PYTHON_SPAWN_TIMEOUT_MS";
/// Crash-loop meltdown: max worker restarts within the restart window.
const PYTHON_MAX_RESTARTS_ENV: &str = "DAEMON_PYTHON_MAX_RESTARTS";
/// The sliding window (ms) over which restarts are counted for meltdown.
const PYTHON_RESTART_WINDOW_MS_ENV: &str = "DAEMON_PYTHON_RESTART_WINDOW_MS";

// --- Browser tool (`daemon-tool-browser`, `browser` feature) tuning (DAEMON_BROWSER_*) --------
/// Register the `browser` tool (`false` by default; also requires the `browser` build feature).
const BROWSER_ENABLE_ENV: &str = "DAEMON_BROWSER_ENABLE";
/// An explicit Chromium/Chrome executable path (`None` => chromiumoxide auto-detection).
const BROWSER_CHROME_PATH_ENV: &str = "DAEMON_BROWSER_CHROME_PATH";
/// Run the browser headless (default `true`).
const BROWSER_HEADLESS_ENV: &str = "DAEMON_BROWSER_HEADLESS";
/// The directory screenshots are written to (`None` => `<profile_home>/browser/screenshots`).
const BROWSER_SCREENSHOT_DIR_ENV: &str = "DAEMON_BROWSER_SCREENSHOT_DIR";
/// Require interactive host approval before each navigation (default `false`).
const BROWSER_APPROVE_NAV_ENV: &str = "DAEMON_BROWSER_APPROVE_NAVIGATION";
/// The browser launch timeout in milliseconds (default `20000`).
const BROWSER_LAUNCH_TIMEOUT_MS_ENV: &str = "DAEMON_BROWSER_LAUNCH_TIMEOUT_MS";
/// Auto-dismiss JS dialogs so a modal cannot wedge the session (default `true`).
const BROWSER_DISMISS_DIALOGS_ENV: &str = "DAEMON_BROWSER_DISMISS_DIALOGS";

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
    /// The deterministic in-tree [`ScriptedProvider`](daemon_core::ScriptedProvider) replaying a
    /// fixed tool-call/final script from `DAEMON_MOCK_SCRIPT` (env-only, binary-internal). Used to
    /// drive a hermetic tool-using turn (so the host parks approvals/clarify) in the HITL e2e — the
    /// wire `ProviderSelector` stays `Mock`, so no contract change.
    Scripted,
    /// Any networked provider served by `genai`; the adapter is inferred from the (optionally
    /// namespaced) `DAEMON_MODEL` name. Replaces the former per-family launch kinds.
    GenAi,
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

/// Tuning for the MeTTa symbolic coprocessor (`daemon-metta`). `enable = false` keeps the `metta`
/// tool unregistered (the default), exactly like the opt-in HTTP/MCP surfaces.
#[derive(Clone, Debug)]
pub struct MettaConfig {
    /// Whether to register the `metta` tool (spawning the supervised worker on first use).
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

/// Tuning for the Python tools worker (`daemon-pytool`). `enable = false` keeps Python tools
/// unregistered (the default), exactly like the opt-in `metta`/`web` surfaces. When enabled the host
/// spawns `interpreter -m worker_module [--tools-dir <dir>]` (or `worker_bin` directly), discovers
/// its tools, and registers a proxy `Tool` for each; the worker is spawned lazily on first use.
#[derive(Clone, Debug)]
pub struct PythonToolsConfig {
    /// Whether to discover + register Python tools.
    pub enable: bool,
    /// The Python interpreter to spawn the worker with (when [`PythonToolsConfig::worker_bin`] is
    /// unset).
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
    pub op_timeout: Duration,
    /// How long to wait for the worker's `Ready` after spawning.
    pub spawn_timeout: Duration,
    /// Crash-loop meltdown: max restarts within [`PythonToolsConfig::restart_window`].
    pub max_restarts: u32,
    /// The sliding window over which restarts are counted for meltdown.
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

/// MCP-client tuning: the external Model-Context-Protocol servers the daemon connects to and
/// surfaces tools from (through the same [`ToolProvider`](daemon_core::ToolProvider) seam as the
/// Python worker). Empty by default — every server is opt-in via `[[mcp.servers]]`.
#[derive(Clone, Debug, Default)]
pub struct McpConfig {
    /// The configured servers (each contributes `mcp__{name}__{tool}` tools when reachable).
    pub servers: Vec<McpServerEntry>,
}

/// One configured MCP server.
#[derive(Clone, Debug)]
pub struct McpServerEntry {
    /// A short, stable name used for tool namespacing + diagnostics.
    pub name: String,
    /// Whether to connect to + register this server's tools.
    pub enable: bool,
    /// How to reach the server.
    pub transport: McpTransportEntry,
    /// Per-operation timeout (discovery / a tool call).
    pub op_timeout: Duration,
}

/// How a [`McpServerEntry`] reaches its server.
#[derive(Clone, Debug)]
pub enum McpTransportEntry {
    /// Spawn a local server binary and speak MCP over its stdio.
    Stdio {
        /// The program to exec.
        command: String,
        /// Arguments passed to the program.
        args: Vec<String>,
        /// Extra environment variables set on the child.
        env: Vec<(String, String)>,
    },
    /// Connect to a remote server over streamable HTTP.
    Http {
        /// The base MCP endpoint.
        url: String,
    },
}

/// Tuning for the web tools (`daemon-tool-web`). `enable = false` keeps `web_search`/`web_extract`
/// unregistered (the default). The Tavily/Firecrawl keys are read live from the `CredentialStore`
/// under [`WebConfig::tavily_key_id`]/[`WebConfig::firecrawl_key_id`], so a GUI-set key applies
/// without a restart; an unkeyed `web_extract` falls back to the local readability path.
#[derive(Clone, Debug)]
pub struct WebConfig {
    /// Whether to register the `web_search` + `web_extract` tools.
    pub enable: bool,
    /// Include the dependency-light local `reqwest`+readability `web_extract` fallback.
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

/// Tuning for the `browser` tool (`daemon-tool-browser`). `enable = false` keeps the tool
/// unregistered (the default); registration also requires the daemon to be built with the `browser`
/// feature (which compiles the heavy chromiumoxide CDP bindings).
#[derive(Clone, Debug)]
pub struct BrowserConfig {
    /// Whether to register the `browser` tool (launching Chromium lazily on first use).
    pub enable: bool,
    /// An explicit Chromium/Chrome executable path (`None` => chromiumoxide auto-detection).
    pub chrome_path: Option<PathBuf>,
    /// Run headless (the default; `false` shows a window — local debugging only).
    pub headless: bool,
    /// The screenshot output directory (`None` => `<profile_home>/browser/screenshots`).
    pub screenshot_dir: Option<PathBuf>,
    /// Require interactive host approval before each navigation.
    pub approve_navigation: bool,
    /// The browser launch timeout.
    pub launch_timeout: Duration,
    /// Auto-dismiss JS dialogs so a modal cannot wedge the session.
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

/// Tuning for the skills subsystem (`daemon-skills` + `daemon-tool-skill`). `enable = true` registers
/// the `skills_list`/`skill_view`/`skill_manage` tools, injects the progressive-disclosure index into
/// every engine's stable system-prompt tier, and (when the engine's review nudge intervals are
/// non-zero) lets the post-turn trigger spawn the `skill_review` background curator.
#[derive(Clone, Debug)]
pub struct SkillsConfig {
    /// Whether the skills subsystem is active.
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

/// A single declarative routing rule — the config surface of a `SessionBinding`
/// (daemon-event-io-spec §5.9). Matches an inbound origin and selects session naming + an optional
/// per-rule profile override.
#[derive(Clone, Debug)]
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

/// Binds a transport instance to a default profile — the account->profile baseline (§5.9 step 2).
#[derive(Clone, Debug)]
pub struct InstanceProfile {
    /// The instance-qualified transport id.
    pub transport: String,
    /// The profile all of that instance's origins run under unless a route overrides.
    pub profile: String,
}

/// The resolved `[routing]` config: the general host routing table (§5.9), consumed into a
/// `daemon_host::RoutingRegistry` at assembly. Empty by default — routed submits then use
/// `PerThread` naming + the node's active default profile (the legacy single-profile behavior).
#[derive(Clone, Debug, Default)]
pub struct RoutingConfig {
    /// The node default profile for routed submits with no matching route/instance (step 3).
    pub default_profile: Option<String>,
    /// Per-instance default profiles (step 2).
    pub instance_profiles: Vec<InstanceProfile>,
    /// Ordered routing rules (first match wins).
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
    /// The parent directory of per-session workspace sandboxes and the `FsRootId::Workspace` browse
    /// root for the filesystem surface (daemon-fs-surface-spec.md). Every engine roots under it
    /// (`<workspace_root>/<session_id>`, or an operator-bound directory) instead of `$TMP`. Defaults
    /// to `<data_dir>/workspaces`.
    pub workspace_root: PathBuf,
    /// The node content store (content-addressed blob CAS, daemon-content-transfer-spec.md) root,
    /// backing the `blob_*` ops + `fs_write_from_blob`. Defaults to `<data_dir>/blobs`.
    pub blob_root: PathBuf,
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
    /// Which model provider implementation to use (mock|scripted|genai|llama|mistralrs).
    pub provider_kind: ProviderKind,
    /// The [`ProviderKind::Scripted`] replay script (raw `DAEMON_MOCK_SCRIPT` JSON); `None` otherwise.
    pub mock_script: Option<String>,
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
    /// MeTTa symbolic-coprocessor tuning (`enable = false` by default — the `metta` tool is opt-in).
    pub metta: MettaConfig,
    /// Python-tools tuning (`enable = false` by default — the `daemon_pytool` worker is opt-in).
    pub python: PythonToolsConfig,
    /// MCP-client tuning (no servers by default — each `[[mcp.servers]]` entry is opt-in).
    pub mcp: McpConfig,
    /// Web-tool tuning (`enable = false` by default — `web_search`/`web_extract` are opt-in).
    pub web: WebConfig,
    /// Browser-tool tuning (`enable = false` by default — also requires the `browser` build feature).
    pub browser: BrowserConfig,
    /// Skills-subsystem tuning (`enable = true` by default — the index + `skill_*` tools).
    pub skills: SkillsConfig,
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
    /// The general host routing table (§5.9) consumed into the routing registry at assembly. Empty
    /// by default (no agent-selection routing — routed submits use the active default profile).
    pub routing: RoutingConfig,
    /// The Matrix chat transport (`daemon-matrix`) config (`enabled = false` by default — opt-in).
    /// Accounts/sessions come from `bound_accounts` + the credential store, not from here.
    pub matrix: daemon_matrix::MatrixConfig,
    /// The internal Rooms loopback transport (`daemon-rooms`) config (`enabled = false` by default —
    /// opt-in). Rooms + membership are durable in the store, not in this config.
    pub rooms: daemon_rooms::RoomsConfig,
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
    workspace_root: Option<PathBuf>,
    blob_root: Option<PathBuf>,
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
    tool_search_threshold_bytes: Option<usize>,
    skill_review_interval: Option<u32>,
    memory_review_interval: Option<u32>,
    journal_seed: Option<String>,
    nesting_depth: Option<usize>,
    local: Option<FileLocalConfig>,
    models: Option<FileModelsConfig>,
    embed: Option<FileEmbedConfig>,
    metta: Option<FileMettaConfig>,
    python: Option<FilePythonConfig>,
    mcp: Option<FileMcpConfig>,
    web: Option<FileWebConfig>,
    browser: Option<FileBrowserConfig>,
    skills: Option<FileSkillsConfig>,
    routing: Option<FileRoutingConfig>,
    matrix: Option<FileMatrixConfig>,
    rooms: Option<FileRoomsConfig>,
}

/// The `[matrix]` TOML table — the Matrix transport surface (every field optional). Carries **no**
/// account secrets: accounts/sessions live in `bound_accounts` + the credential store.
#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct FileMatrixConfig {
    enabled: Option<bool>,
    store_root: Option<String>,
    route: Vec<FileMatrixRoute>,
}

/// A `[[matrix.route]]` entry — which rooms to engage + how addressing is classified.
#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct FileMatrixRoute {
    account: Option<String>,
    room_glob: Option<String>,
    dm_only: Option<bool>,
    mention_gating: Option<bool>,
}

/// The `[rooms]` TOML table — the internal Rooms loopback transport surface (every field optional).
/// Rooms + membership themselves are durable in the store, not declared here.
#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct FileRoomsConfig {
    enabled: Option<bool>,
    max_turns: Option<u32>,
}

/// The `[routing]` TOML table — the general host routing table (§5.9; every field optional).
#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct FileRoutingConfig {
    default_profile: Option<String>,
    instance_profile: Vec<FileInstanceProfile>,
    route: Vec<FileRouteRule>,
}

/// A `[[routing.instance_profile]]` entry — bind a transport instance to a default profile.
#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct FileInstanceProfile {
    transport: Option<String>,
    profile: Option<String>,
}

/// A `[[routing.route]]` entry — one declarative routing rule.
#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct FileRouteRule {
    transport: Option<String>,
    transport_family: Option<String>,
    scope: Option<String>,
    chat_glob: Option<String>,
    profile: Option<String>,
    isolation: Option<String>,
}

/// The `[skills]` TOML table — skills-subsystem tuning (every field optional; env wins).
#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct FileSkillsConfig {
    enable: Option<bool>,
    dir: Option<PathBuf>,
}

/// The `[python]` TOML table — Python-tools tuning (every field optional; env wins).
#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct FilePythonConfig {
    enable: Option<bool>,
    interpreter: Option<PathBuf>,
    worker_module: Option<String>,
    worker_bin: Option<PathBuf>,
    tools_dir: Option<PathBuf>,
    package_path: Option<PathBuf>,
    op_timeout_ms: Option<u64>,
    spawn_timeout_ms: Option<u64>,
    max_restarts: Option<u32>,
    restart_window_ms: Option<u64>,
}

/// The `[mcp]` TOML table — MCP-client tuning (the `[[mcp.servers]]` array; every field optional).
///
/// Each `[[mcp.servers]]` entry registers an external MCP server whose tools are surfaced to the
/// agent under namespaced names `mcp__{name}__{tool}` (untrusted-fenced; gateable by
/// `ProfileSpec.tool_allowlist`). A server that fails to start logs a warning and contributes no
/// tools. No servers are configured by default. Example:
///
/// ```toml
/// # A local stdio server the daemon spawns and owns (MCP over the child's stdio):
/// [[mcp.servers]]
/// name = "github"
/// transport = "stdio"             # default; may be omitted
/// command = "npx"
/// args = ["-y", "@modelcontextprotocol/server-github"]
/// env = { GITHUB_TOKEN = "ghp_..." }
/// op_timeout_ms = 60000           # per-call/discovery timeout (default 60s)
///
/// # A remote server the daemon connects to over streamable HTTP:
/// [[mcp.servers]]
/// name = "docs"
/// transport = "http"
/// url = "http://127.0.0.1:8000/mcp"
///
/// # A temporarily disabled server (kept in config, contributes no tools):
/// [[mcp.servers]]
/// name = "scratch"
/// enable = false
/// command = "my-mcp-server"
/// ```
#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct FileMcpConfig {
    servers: Vec<FileMcpServer>,
}

/// A `[[mcp.servers]]` entry — one external MCP server.
#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct FileMcpServer {
    name: Option<String>,
    enable: Option<bool>,
    /// `"stdio"` (default) or `"http"`.
    transport: Option<String>,
    command: Option<String>,
    args: Option<Vec<String>>,
    env: Option<std::collections::HashMap<String, String>>,
    url: Option<String>,
    op_timeout_ms: Option<u64>,
}

/// The `[metta]` TOML table — symbolic-coprocessor tuning (every field optional; env wins).
#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct FileMettaConfig {
    enable: Option<bool>,
    worker_bin: Option<PathBuf>,
    state_dir: Option<PathBuf>,
    max_steps: Option<u64>,
    timeout_ms: Option<u64>,
    max_results: Option<u64>,
    max_restarts: Option<u32>,
    restart_window_ms: Option<u64>,
}

/// The `[web]` TOML table — web-tool tuning (every field optional; env wins).
#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct FileWebConfig {
    enable: Option<bool>,
    local_fallback: Option<bool>,
    tavily_key_id: Option<String>,
    firecrawl_key_id: Option<String>,
}

/// The `[browser]` TOML table — browser-tool tuning (every field optional; env wins).
#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct FileBrowserConfig {
    enable: Option<bool>,
    chrome_path: Option<PathBuf>,
    headless: Option<bool>,
    screenshot_dir: Option<PathBuf>,
    approve_navigation: Option<bool>,
    launch_timeout_ms: Option<u64>,
    auto_dismiss_dialogs: Option<bool>,
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
        self.profile_home_for(&self.profile)
    }

    /// The data home for an arbitrary `profile` (`<data_dir>/<profile>/`). The §5.9 routed/identity
    /// profile keys the LCM/Mnemosyne bank caches here, so a session routed to a non-default profile
    /// gets its own isolated subsystem stores rather than sharing the launch profile's home.
    pub fn profile_home_for(&self, profile: &str) -> PathBuf {
        self.data_dir.join(profile)
    }

    /// The data-dir root that profile homes hang off (`<data_dir>`); the bank caches join the
    /// resolved profile id onto this to compute each profile's home.
    pub fn data_root(&self) -> PathBuf {
        self.data_dir.clone()
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

        // The workspace root: the parent of per-session sandboxes + the filesystem-surface
        // browse root. Defaults under the data dir so a fresh node has a stable, persistent place
        // for agent workspaces (instead of `$TMP/daemon-ws-*`).
        let workspace_root = env_string(WORKSPACE_ROOT_ENV)
            .map(PathBuf::from)
            .or_else(|| file.workspace_root.clone())
            .unwrap_or_else(|| data_dir.join("workspaces"));

        // The content store root (blob CAS); defaults under the data dir alongside workspaces.
        let blob_root = env_string(BLOB_ROOT_ENV)
            .map(PathBuf::from)
            .or_else(|| file.blob_root.clone())
            .unwrap_or_else(|| data_dir.join("blobs"));

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
            // Binary-internal scripted provider for the hermetic HITL e2e (DAEMON_MOCK_SCRIPT).
            "scripted" | "script" => ProviderKind::Scripted,
            // All networked providers are genai-backed; the adapter is inferred from DAEMON_MODEL.
            // The legacy per-family names remain accepted for launch back-compat and all map here.
            "genai" | "openai" | "anthropic" | "gemini" | "google" | "groq" | "deepseek"
            | "deep_seek" | "deep-seek" | "xai" | "grok" | "openrouter" | "open_router"
            | "open-router" | "cohere" => ProviderKind::GenAi,
            "llama" | "llamacpp" | "llama-cpp" => ProviderKind::LlamaCpp,
            "mistralrs" | "mistral-rs" | "mistral.rs" => ProviderKind::MistralRs,
            other => anyhow::bail!(
                "unknown model provider {other:?} (expected mock|scripted|genai|llama|mistralrs)"
            ),
        };
        // The scripted provider's replay script (env-only; JSON array of {call,args}/{final} steps).
        let mock_script = env_string(MOCK_SCRIPT_ENV);
        // No default: `None` lets the provider client use its own default endpoint. An override is
        // only meaningful for a gateway/proxy or the in-process wire tests.
        let base_url = env_string(BASE_URL_ENV).or(file.base_url);
        // The genai adapter is inferred from the model name, so there is no per-provider default
        // model: a networked launch must set DAEMON_MODEL (e.g. `claude-opus-4-8`, `groq::…`).
        let model = env_string(MODEL_ENV).or(file.model).unwrap_or_default();

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
        let metta = Self::resolve_metta(file.metta.unwrap_or_default())?;
        let python = Self::resolve_python(file.python.unwrap_or_default())?;
        let mcp = Self::resolve_mcp(file.mcp.unwrap_or_default())?;
        let web = Self::resolve_web(file.web.unwrap_or_default());
        let browser = Self::resolve_browser(file.browser.unwrap_or_default())?;
        let skills = Self::resolve_skills(file.skills.unwrap_or_default())?;
        let routing = Self::resolve_routing(file.routing.unwrap_or_default())?;
        let matrix = Self::resolve_matrix(file.matrix.unwrap_or_default(), &data_dir);
        let rooms = Self::resolve_rooms(file.rooms.unwrap_or_default());

        Ok(Self {
            partition,
            socket_path,
            http_addr,
            store,
            data_dir,
            workspace_root,
            blob_root,
            dispatch_interval,
            scan_interval,
            profile,
            context_engine,
            memory_provider,
            memory_file,
            provider_kind,
            mock_script,
            base_url,
            model,
            local,
            models,
            embed,
            metta,
            python,
            mcp,
            web,
            browser,
            skills,
            credential_key,
            engine,
            journal_seed,
            nesting_depth,
            routing,
            matrix,
            rooms,
        })
    }

    /// Resolve the `[rooms]` table (env overriding TOML overriding defaults). Rooms carry no secrets
    /// and no per-room declarations here — the durable Rooms + membership live in the store.
    fn resolve_rooms(file: FileRoomsConfig) -> daemon_rooms::RoomsConfig {
        let defaults = daemon_rooms::RoomsConfig::default();
        let mut enabled = file.enabled.unwrap_or(defaults.enabled);
        if let Some(s) = env_string(ROOMS_ENABLE_ENV) {
            enabled = parse_bool(&s);
        }
        let max_turns = env_string(ROOMS_MAX_TURNS_ENV)
            .and_then(|s| s.parse().ok())
            .or(file.max_turns)
            .unwrap_or(defaults.max_turns);
        daemon_rooms::RoomsConfig { enabled, max_turns }
    }

    /// Resolve the `[matrix]` table (env overriding TOML overriding defaults). `store_root` is made
    /// absolute under `data_dir`; the route table maps directly onto the adapter's `MatrixRoute`.
    fn resolve_matrix(
        file: FileMatrixConfig,
        data_dir: &std::path::Path,
    ) -> daemon_matrix::MatrixConfig {
        let mut enabled = file.enabled.unwrap_or(false);
        if let Some(s) = env_string(MATRIX_ENABLE_ENV) {
            enabled = parse_bool(&s);
        }
        let store_root_rel = env_string(MATRIX_STORE_ROOT_ENV)
            .or(file.store_root)
            .unwrap_or_else(|| "matrix".to_string());
        let routes = file
            .route
            .into_iter()
            .map(|r| daemon_matrix::MatrixRoute {
                account: r.account,
                room_glob: r.room_glob,
                dm_only: r.dm_only.unwrap_or(false),
                mention_gating: r.mention_gating.unwrap_or(true),
            })
            .collect();
        daemon_matrix::MatrixConfig {
            enabled,
            store_root: data_dir.join(store_root_rel),
            routes,
        }
    }

    /// Resolve the `[routing]` table into the general host routing config (§5.9). File-only (no env);
    /// validates the `transport`/`transport_family` exclusivity and the required instance-profile
    /// fields, leaving string->type mapping (patterns/isolation) to the assembly step in `main`.
    fn resolve_routing(file: FileRoutingConfig) -> anyhow::Result<RoutingConfig> {
        let mut instance_profiles = Vec::new();
        for ip in file.instance_profile {
            let transport = ip.transport.ok_or_else(|| {
                anyhow::anyhow!("[[routing.instance_profile]] requires `transport`")
            })?;
            let profile = ip.profile.ok_or_else(|| {
                anyhow::anyhow!("[[routing.instance_profile]] requires `profile`")
            })?;
            instance_profiles.push(InstanceProfile { transport, profile });
        }
        let mut routes = Vec::new();
        for r in file.route {
            if r.transport.is_some() && r.transport_family.is_some() {
                anyhow::bail!(
                    "[[routing.route]] sets both `transport` and `transport_family` (pick one)"
                );
            }
            routes.push(RouteRule {
                transport: r.transport,
                transport_family: r.transport_family,
                scope: r.scope.unwrap_or_else(|| "any".to_string()),
                chat_glob: r.chat_glob.unwrap_or_else(|| "*".to_string()),
                profile: r.profile,
                isolation: r.isolation.unwrap_or_else(|| "per_thread".to_string()),
            });
        }
        Ok(RoutingConfig {
            default_profile: file.default_profile,
            instance_profiles,
            routes,
        })
    }

    /// Resolve skills-subsystem tuning (env overriding the `[skills]` TOML table overriding defaults).
    fn resolve_skills(file: FileSkillsConfig) -> anyhow::Result<SkillsConfig> {
        let mut skills = SkillsConfig::default();
        if let Some(b) = file.enable {
            skills.enable = b;
        }
        if let Some(s) = env_string(SKILLS_ENABLE_ENV) {
            skills.enable = parse_bool(&s);
        }
        if let Some(d) = file.dir {
            skills.dir = Some(d);
        }
        if let Some(s) = env_string(SKILLS_DIR_ENV) {
            skills.dir = Some(PathBuf::from(s));
        }
        Ok(skills)
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

    /// Resolve MeTTa coprocessor tuning (env overriding the `[metta]` TOML table overriding defaults).
    fn resolve_metta(file: FileMettaConfig) -> anyhow::Result<MettaConfig> {
        let mut metta = MettaConfig::default();
        if let Some(b) = file.enable {
            metta.enable = b;
        }
        if let Some(s) = env_string(METTA_ENABLE_ENV) {
            metta.enable = parse_bool(&s);
        }
        if let Some(path) = file.worker_bin {
            metta.worker_bin = path;
        }
        if let Some(s) = env_string(METTA_BIN_ENV) {
            metta.worker_bin = PathBuf::from(s);
        }
        metta.state_dir = env_string(METTA_STATE_DIR_ENV)
            .map(PathBuf::from)
            .or(file.state_dir);
        if let Some(n) = file.max_steps {
            metta.max_steps = n;
        }
        if let Some(s) = env_string(METTA_MAX_STEPS_ENV) {
            metta.max_steps = s.parse().context("DAEMON_METTA_MAX_STEPS must be a u64")?;
        }
        if let Some(n) = file.timeout_ms {
            metta.timeout_ms = n;
        }
        if let Some(s) = env_string(METTA_TIMEOUT_MS_ENV) {
            metta.timeout_ms = s.parse().context("DAEMON_METTA_TIMEOUT_MS must be a u64")?;
        }
        if let Some(n) = file.max_results {
            metta.max_results = n;
        }
        if let Some(s) = env_string(METTA_MAX_RESULTS_ENV) {
            metta.max_results = s
                .parse()
                .context("DAEMON_METTA_MAX_RESULTS must be a u64")?;
        }
        if let Some(n) = file.max_restarts {
            metta.max_restarts = n;
        }
        if let Some(s) = env_string(METTA_MAX_RESTARTS_ENV) {
            metta.max_restarts = s
                .parse()
                .context("DAEMON_METTA_MAX_RESTARTS must be a u32")?;
        }
        resolve_duration_ms(
            &mut metta.restart_window,
            file.restart_window_ms,
            METTA_RESTART_WINDOW_MS_ENV,
        )?;
        Ok(metta)
    }

    /// Resolve the `[mcp]` table into the MCP-client config (file-only; per-server). Each entry must
    /// name a transport; `stdio` requires `command`, `http` requires `url`.
    fn resolve_mcp(file: FileMcpConfig) -> anyhow::Result<McpConfig> {
        let mut servers = Vec::new();
        for s in file.servers {
            let name = s
                .name
                .ok_or_else(|| anyhow::anyhow!("[[mcp.servers]] requires `name`"))?;
            let enable = s.enable.unwrap_or(true);
            let transport = match s.transport.as_deref().unwrap_or("stdio") {
                "stdio" => {
                    let command = s.command.ok_or_else(|| {
                        anyhow::anyhow!("[[mcp.servers]] `{name}` (stdio) requires `command`")
                    })?;
                    let env = s
                        .env
                        .map(|m| m.into_iter().collect::<Vec<_>>())
                        .unwrap_or_default();
                    McpTransportEntry::Stdio {
                        command,
                        args: s.args.unwrap_or_default(),
                        env,
                    }
                }
                "http" => {
                    let url = s.url.ok_or_else(|| {
                        anyhow::anyhow!("[[mcp.servers]] `{name}` (http) requires `url`")
                    })?;
                    McpTransportEntry::Http { url }
                }
                other => anyhow::bail!(
                    "[[mcp.servers]] `{name}` has unknown transport `{other}` (use `stdio` or `http`)"
                ),
            };
            let op_timeout = Duration::from_millis(s.op_timeout_ms.unwrap_or(60_000));
            servers.push(McpServerEntry {
                name,
                enable,
                transport,
                op_timeout,
            });
        }
        Ok(McpConfig { servers })
    }

    /// Resolve Python-tools tuning (env overriding the `[python]` TOML table overriding defaults).
    fn resolve_python(file: FilePythonConfig) -> anyhow::Result<PythonToolsConfig> {
        let mut python = PythonToolsConfig::default();
        if let Some(b) = file.enable {
            python.enable = b;
        }
        if let Some(s) = env_string(PYTHON_ENABLE_ENV) {
            python.enable = parse_bool(&s);
        }
        if let Some(p) = file.interpreter {
            python.interpreter = p;
        }
        if let Some(s) = env_string(PYTHON_INTERPRETER_ENV) {
            python.interpreter = PathBuf::from(s);
        }
        if let Some(m) = file.worker_module {
            python.worker_module = m;
        }
        if let Some(s) = env_string(PYTHON_WORKER_MODULE_ENV) {
            python.worker_module = s;
        }
        python.worker_bin = env_string(PYTHON_WORKER_BIN_ENV)
            .map(PathBuf::from)
            .or(file.worker_bin);
        python.tools_dir = env_string(PYTHON_TOOLS_DIR_ENV)
            .map(PathBuf::from)
            .or(file.tools_dir);
        python.package_path = env_string(PYTHON_PACKAGE_PATH_ENV)
            .map(PathBuf::from)
            .or(file.package_path);
        resolve_duration_ms(
            &mut python.op_timeout,
            file.op_timeout_ms,
            PYTHON_OP_TIMEOUT_MS_ENV,
        )?;
        resolve_duration_ms(
            &mut python.spawn_timeout,
            file.spawn_timeout_ms,
            PYTHON_SPAWN_TIMEOUT_MS_ENV,
        )?;
        if let Some(n) = file.max_restarts {
            python.max_restarts = n;
        }
        if let Some(s) = env_string(PYTHON_MAX_RESTARTS_ENV) {
            python.max_restarts = s
                .parse()
                .context("DAEMON_PYTHON_MAX_RESTARTS must be a u32")?;
        }
        resolve_duration_ms(
            &mut python.restart_window,
            file.restart_window_ms,
            PYTHON_RESTART_WINDOW_MS_ENV,
        )?;
        Ok(python)
    }

    /// Resolve web-tool tuning (env overriding the `[web]` TOML table overriding defaults).
    fn resolve_web(file: FileWebConfig) -> WebConfig {
        let mut web = WebConfig::default();
        if let Some(b) = file.enable {
            web.enable = b;
        }
        if let Some(s) = env_string(WEB_ENABLE_ENV) {
            web.enable = parse_bool(&s);
        }
        if let Some(b) = file.local_fallback {
            web.local_fallback = b;
        }
        if let Some(s) = env_string(WEB_LOCAL_FALLBACK_ENV) {
            web.local_fallback = parse_bool(&s);
        }
        if let Some(s) = file.tavily_key_id {
            web.tavily_key_id = s;
        }
        if let Some(s) = env_string(WEB_TAVILY_KEY_ENV) {
            web.tavily_key_id = s;
        }
        if let Some(s) = file.firecrawl_key_id {
            web.firecrawl_key_id = s;
        }
        if let Some(s) = env_string(WEB_FIRECRAWL_KEY_ENV) {
            web.firecrawl_key_id = s;
        }
        web
    }

    /// Resolve browser-tool tuning (env overriding the `[browser]` TOML table overriding defaults).
    fn resolve_browser(file: FileBrowserConfig) -> anyhow::Result<BrowserConfig> {
        let mut browser = BrowserConfig::default();
        if let Some(b) = file.enable {
            browser.enable = b;
        }
        if let Some(s) = env_string(BROWSER_ENABLE_ENV) {
            browser.enable = parse_bool(&s);
        }
        browser.chrome_path = env_string(BROWSER_CHROME_PATH_ENV)
            .map(PathBuf::from)
            .or(file.chrome_path);
        if let Some(b) = file.headless {
            browser.headless = b;
        }
        if let Some(s) = env_string(BROWSER_HEADLESS_ENV) {
            browser.headless = parse_bool(&s);
        }
        browser.screenshot_dir = env_string(BROWSER_SCREENSHOT_DIR_ENV)
            .map(PathBuf::from)
            .or(file.screenshot_dir);
        if let Some(b) = file.approve_navigation {
            browser.approve_navigation = b;
        }
        if let Some(s) = env_string(BROWSER_APPROVE_NAV_ENV) {
            browser.approve_navigation = parse_bool(&s);
        }
        if let Some(b) = file.auto_dismiss_dialogs {
            browser.auto_dismiss_dialogs = b;
        }
        if let Some(s) = env_string(BROWSER_DISMISS_DIALOGS_ENV) {
            browser.auto_dismiss_dialogs = parse_bool(&s);
        }
        resolve_duration_ms(
            &mut browser.launch_timeout,
            file.launch_timeout_ms,
            BROWSER_LAUNCH_TIMEOUT_MS_ENV,
        )?;
        Ok(browser)
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
        if let Some(n) = file.tool_search_threshold_bytes {
            engine.tool_search_threshold_bytes = n;
        }
        if let Some(s) = env_string(TOOL_SEARCH_THRESHOLD_ENV) {
            engine.tool_search_threshold_bytes = s
                .parse()
                .context("DAEMON_TOOL_SEARCH_THRESHOLD_BYTES must be a usize")?;
        }
        if let Some(n) = file.skill_review_interval {
            engine.skill_review_interval = n;
        }
        if let Some(s) = env_string(SKILL_REVIEW_INTERVAL_ENV) {
            engine.skill_review_interval = s
                .parse()
                .context("DAEMON_SKILL_REVIEW_INTERVAL must be a u32")?;
        }
        if let Some(n) = file.memory_review_interval {
            engine.memory_review_interval = n;
        }
        if let Some(s) = env_string(MEMORY_REVIEW_INTERVAL_ENV) {
            engine.memory_review_interval = s
                .parse()
                .context("DAEMON_MEMORY_REVIEW_INTERVAL must be a u32")?;
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
