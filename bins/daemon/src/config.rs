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
/// Selects the durable store backend: `memory` (default) or `sqlite`.
const STORE_ENV: &str = "DAEMON_STORE";
/// The SQLite database path (when the backend is `sqlite`).
const STORE_PATH_ENV: &str = "DAEMON_STORE_PATH";
/// Overrides the owned partition id (a `u64`).
const PARTITION_ENV: &str = "DAEMON_PARTITION";
/// Overrides the model provider/credential profile name.
const PROFILE_ENV: &str = "DAEMON_PROFILE";
/// Overrides the (stub) credential key the owner authority mints.
const CREDENTIAL_KEY_ENV: &str = "DAEMON_CREDENTIAL_KEY";

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
    /// The durable store backend.
    pub store: StoreBackend,
    /// How often the wake/job dispatchers poll the durable outboxes.
    pub dispatch_interval: Duration,
    /// How often the recovery scanner re-checks for resumable sessions.
    pub scan_interval: Duration,
    /// The model provider + credential profile name (selects the registered provider builder).
    pub profile: String,
    /// The (stub) credential key the owner authority mints for that profile.
    pub credential_key: String,
}

/// The TOML file shape — every field optional, so a partial file is valid and env fills the rest.
#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct FileConfig {
    partition: Option<u64>,
    socket_path: Option<PathBuf>,
    store: Option<String>,
    store_path: Option<PathBuf>,
    dispatch_interval_ms: Option<u64>,
    scan_interval_ms: Option<u64>,
    profile: Option<String>,
    credential_key: Option<String>,
}

fn env_string(key: &str) -> Option<String> {
    std::env::var_os(key).map(|v| v.to_string_lossy().into_owned())
}

fn default_socket() -> PathBuf {
    let dir = std::env::var_os("TMPDIR").unwrap_or_else(|| "/tmp".into());
    PathBuf::from(dir).join("daemon-api.sock")
}

impl NodeConfig {
    /// Load the layered config: read the optional TOML file at `$DAEMON_CONFIG`, then overlay env.
    pub fn load() -> anyhow::Result<Self> {
        let file = match std::env::var_os(CONFIG_ENV) {
            Some(path) => {
                let text = std::fs::read_to_string(&path).with_context(|| {
                    format!("reading config file {}", path.to_string_lossy())
                })?;
                toml::from_str::<FileConfig>(&text).context("parsing TOML config")?
            }
            None => FileConfig::default(),
        };

        let partition = match env_string(PARTITION_ENV) {
            Some(s) => PartitionId(s.parse().context("DAEMON_PARTITION must be a u64")?),
            None => file.partition.map(PartitionId).unwrap_or(PartitionId::DEFAULT),
        };

        let store = Self::resolve_store(&file)?;

        let socket_path = env_string(API_SOCKET_ENV)
            .map(PathBuf::from)
            .or(file.socket_path)
            .unwrap_or_else(default_socket);

        let dispatch_interval =
            Duration::from_millis(file.dispatch_interval_ms.unwrap_or(2));
        let scan_interval = Duration::from_millis(file.scan_interval_ms.unwrap_or(10));

        let profile = env_string(PROFILE_ENV)
            .or(file.profile)
            .unwrap_or_else(|| "openai".to_string());
        let credential_key = env_string(CREDENTIAL_KEY_ENV)
            .or(file.credential_key)
            .unwrap_or_else(|| "sk-configured".to_string());

        Ok(Self {
            partition,
            socket_path,
            store,
            dispatch_interval,
            scan_interval,
            profile,
            credential_key,
        })
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
