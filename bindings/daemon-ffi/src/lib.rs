//! `daemon-ffi` — the C ABI adapter over the **durable host** (the node management + §17 surface).
//!
//! Where [`daemon-core-ffi`](../daemon_core_ffi/index.html) embeds the L1 *brain* (one completing
//! engine per session, no durability), this embeds the L2 *durable system*: it boots an in-process
//! [`daemon-node`](daemon_node) — durable store, resident-service supervision, activation, the
//! orchestration fleet, and the full [`NodeApi`] control plane — behind an opaque handle, then lets
//! a non-Rust host drive **every** node operation across one CBOR call (daemon-ffi-spec §2.2).
//!
//! Shape (daemon-ffi-spec §3):
//! - opaque [`daemon_host_t`] owns the Tokio runtime + the assembled [`NodeApiImpl`] + its
//!   resident-service [`SupervisorHandle`];
//! - `daemon_host_new`/`daemon_host_new_with_config` assemble the node (config marshalled in as
//!   CBOR), `daemon_host_free` drives a graceful shutdown;
//! - one generic [`daemon_host_call`] marshals a CBOR [`ApiRequest`] through the same
//!   [`daemon_api::dispatch`] every transport (Unix socket / HTTP) uses and returns a CBOR
//!   [`ApiResponse`]. The whole surface — `Submit`/`Poll`/`Respond`, fleet/tree, fs/cron, etc. —
//!   rides this one call; the event drain is just `Poll`/`Subscribe`/`UnitOutbound`/`Tree` requests.
//! - every entry point is `catch_unwind`-guarded (panics never cross the boundary; relies on the
//!   workspace `panic = "unwind"`), reporting failure via a `daemon_status` code + a thread-local
//!   last-error message.

#![deny(unsafe_op_in_unsafe_fn)]
#![allow(non_camel_case_types)]

use std::cell::RefCell;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::sync::Arc;

use daemon_api::{dispatch, from_cbor, to_cbor, ApiRequest, ApiResponse, ProviderSelector};
use daemon_common::{CredScope, PartitionId, ProfileRef};
use daemon_core::{
    Config, CredentialBuilder, CredentialProvider, EmbeddedCredentialPool, MockProvider, Provider,
    ProviderBuilder, ProviderRegistry,
};
use daemon_host::{HostConfig, NodeApiImpl, SupervisorHandle};
use daemon_node::{assemble, AssembledNode, NodeAssembly};
use daemon_providers::GenAiProvider;
use daemon_store::{InMemoryStore, SessionStore, SqliteStore};
use serde::{Deserialize, Serialize};
use tokio::runtime::Runtime;

/// The ABI version of the handle/function shell (semver-disciplined, distinct from the payload
/// `wire_version`).
const ABI_VERSION: u32 = 1;

// ---------------------------------------------------------------------------
// status codes + thread-local last error
// ---------------------------------------------------------------------------

/// `daemon_status`: the success/failure code every entry point returns.
pub const DAEMON_OK: i32 = 0;
/// A recoverable error occurred; details via `daemon_last_error`.
pub const DAEMON_ERROR: i32 = 1;
/// A panic was caught at the boundary (should not happen in normal operation).
pub const DAEMON_PANIC: i32 = 3;
/// A null handle or invalid argument was passed.
pub const DAEMON_INVALID: i32 = 4;

thread_local! {
    static LAST_ERROR: RefCell<String> = const { RefCell::new(String::new()) };
}

fn set_last_error(msg: impl Into<String>) {
    LAST_ERROR.with(|e| *e.borrow_mut() = msg.into());
}

// ---------------------------------------------------------------------------
// construction config
// ---------------------------------------------------------------------------

/// Construction config for a durable host, marshalled in as CBOR by `daemon_host_new_with_config`.
///
/// Like [`daemon_core_ffi`]'s `CoreFfiConfig`, this is a *construction* input, not a protocol/domain
/// message (daemon-ffi-spec §2.1): it bundles existing contract types so a C embedder can boot a
/// real durable node without implementing any host-side port. Every field defaults, so a
/// partial/absent blob degrades to an in-memory node on the deterministic mock provider
/// (`daemon_host_new`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct HostFfiConfig {
    /// Path to a durable sqlite store. `None` => an ephemeral in-memory store (sessions do not
    /// survive `daemon_host_free`).
    pub store_path: Option<String>,
    /// The partition this node owns.
    pub partition: u64,
    /// Which model provider every engine this node builds resolves. `Mock` (the default) keeps the
    /// deterministic in-tree provider; `GenAi` builds a real networked provider.
    pub provider: ProviderSelector,
    /// The (optionally namespaced) model name, e.g. `claude-sonnet-4` or `gpt-4o`.
    pub model: String,
    /// Override the provider base URL (custom gateway / proxy). `None` => the adapter default.
    pub base_url: Option<String>,
    /// The API key brokered onto every engine (it lands on `Request.auth`). `None` keeps the
    /// engines on their embedded L1 pool (a networked provider then resolves its env-var key).
    pub api_key: Option<String>,
    /// The node's session + credential profile name (defaults to `default`).
    pub profile: String,
    /// Output-token cap per generation (provider-side). `None` => the provider's per-model default.
    pub max_output_tokens: Option<u32>,
    /// Engine tunables (§20) every engine this node builds runs under. `None` => [`Config::default`].
    pub engine: Option<Config>,
    /// The workspace root binding the `fs_*` surface and rooting session sandboxes. `None` leaves
    /// the filesystem surface unbound (a temp-sandbox per session).
    pub workspace_root: Option<String>,
    /// The blob CAS root binding the `blob_*` + `fs_write_from_blob` surface. `None` leaves the
    /// content surface unbound.
    pub blob_root: Option<String>,
}

impl Default for HostFfiConfig {
    fn default() -> Self {
        Self {
            store_path: None,
            partition: 0,
            provider: ProviderSelector::Mock,
            model: String::new(),
            base_url: None,
            api_key: None,
            profile: "default".to_string(),
            max_output_tokens: None,
            engine: None,
            workspace_root: None,
            blob_root: None,
        }
    }
}

/// Build the per-engine model-provider factory the config selects: a real networked
/// [`GenAiProvider`] for `GenAi` (adapter inferred from the model name, with optional endpoint /
/// output-token overrides), else the deterministic [`MockProvider`]. Local-inference kinds need a
/// `ModelManager` + on-disk worker binary not assembled here, so they also fall back to the mock.
fn provider_builder(cfg: &HostFfiConfig) -> ProviderBuilder {
    match cfg.provider {
        ProviderSelector::GenAi => {
            let model = cfg.model.clone();
            let base_url = cfg.base_url.clone();
            let max_output = cfg.max_output_tokens;
            Arc::new(move || {
                let mut p = GenAiProvider::for_model(model.clone());
                if let Some(base) = &base_url {
                    p = p.with_endpoint(base.clone());
                }
                if let Some(max) = max_output {
                    p = p.with_max_tokens(max);
                }
                Arc::new(p) as Arc<dyn Provider>
            })
        }
        _ => Arc::new(|| Arc::new(MockProvider::completing("ffi node done")) as Arc<dyn Provider>),
    }
}

/// Assemble the durable node the config describes. Must run inside a Tokio runtime context (the
/// assembly starts resident services + the fleet via `tokio::spawn`).
fn build_node(cfg: &HostFfiConfig) -> Result<AssembledNode, String> {
    let store: Arc<dyn SessionStore> = match &cfg.store_path {
        Some(path) => Arc::new(
            SqliteStore::open(path).map_err(|e| format!("opening sqlite store at {path}: {e}"))?,
        ),
        None => Arc::new(InMemoryStore::new()),
    };

    // One builder serves the default + the orchestrator/child role profiles, so delegated fleet
    // children run the same provider as the session engine.
    let builder = provider_builder(cfg);
    let mut providers = ProviderRegistry::new();
    providers.set_default(builder.clone());
    providers.register("orchestrator", builder.clone());
    providers.register("child", builder);

    // A real key reaches `Request.auth` through an embedded L1 pool scoped to the node profile — no
    // brokered cut needed for an in-process embed. Without a key the engines keep their default
    // single-key pool.
    let credentials: Option<CredentialBuilder> = cfg.api_key.as_ref().map(|key| {
        let profile = cfg.profile.clone();
        let pool = Arc::new(EmbeddedCredentialPool::new(
            profile.clone(),
            CredScope::new([profile.as_str()], ["chat"], None),
            [("ffi".to_string(), key.clone())],
            60_000,
            30_000,
        ));
        Arc::new(move || pool.clone() as Arc<dyn CredentialProvider>) as CredentialBuilder
    });

    let host_config = HostConfig {
        partition: PartitionId(cfg.partition),
        ..HostConfig::default()
    };

    Ok(assemble(NodeAssembly {
        store,
        partition: PartitionId(cfg.partition),
        host_config,
        providers,
        credentials,
        profile: ProfileRef::new(&cfg.profile),
        engine_config: cfg.engine.unwrap_or_default(),
        journal_seed: None,
        nesting_depth: 0,
        context: None,
        context_builder: None,
        memory: Vec::new(),
        memory_builder: None,
        extra_tools: Vec::new(),
        models: None,
        profiles: None,
        provider_resolver: None,
        credential_store: None,
        cloud_catalog: None,
        prompt_sources: vec![],
        revisions: None,
        skills: None,
        skills_resolver: None,
        routing: None,
        checkpoints: None,
        auth_factories: vec![],
        workspace_root: cfg.workspace_root.clone().map(PathBuf::from),
        blob_root: cfg.blob_root.clone().map(PathBuf::from),
    }))
}

// ---------------------------------------------------------------------------
// opaque handle
// ---------------------------------------------------------------------------

/// Opaque durable-host handle: owns the Tokio runtime, the assembled node surface, and its started
/// resident-service supervisor (taken on `daemon_host_free` to drive a graceful shutdown).
pub struct daemon_host_t {
    rt: Runtime,
    node: Arc<NodeApiImpl>,
    handle: Option<SupervisorHandle>,
}

// ---------------------------------------------------------------------------
// C ABI
// ---------------------------------------------------------------------------

/// The ABI version of this shell.
#[no_mangle]
pub extern "C" fn daemon_abi_version() -> u32 {
    ABI_VERSION
}

/// Boot a durable host with the zero-config default (in-memory store, deterministic mock provider).
/// Returns null on failure (see `daemon_last_error`). Use `daemon_host_new_with_config` for a real
/// store/provider.
#[no_mangle]
pub extern "C" fn daemon_host_new() -> *mut daemon_host_t {
    build_host(HostFfiConfig::default())
}

/// Boot a durable host from a CBOR-encoded [`HostFfiConfig`] `(cfg, len)`. Returns null on failure
/// (see `daemon_last_error`).
///
/// # Safety
/// `cfg` must point to `len` readable bytes (a CBOR `HostFfiConfig`); `len` may be `0` for defaults.
#[no_mangle]
pub unsafe extern "C" fn daemon_host_new_with_config(
    cfg: *const u8,
    len: usize,
) -> *mut daemon_host_t {
    if cfg.is_null() && len != 0 {
        set_last_error("null config pointer to daemon_host_new_with_config");
        return std::ptr::null_mut();
    }
    let parsed = catch_unwind(AssertUnwindSafe(|| {
        if len == 0 {
            return Ok::<_, String>(HostFfiConfig::default());
        }
        let bytes = unsafe { std::slice::from_raw_parts(cfg, len) };
        from_cbor::<HostFfiConfig>(bytes).map_err(|e| e.to_string())
    }));
    match parsed {
        Ok(Ok(config)) => build_host(config),
        Ok(Err(msg)) => {
            set_last_error(msg);
            std::ptr::null_mut()
        }
        Err(_) => {
            set_last_error("panic decoding host config");
            std::ptr::null_mut()
        }
    }
}

/// Shared construction: build the Tokio runtime, assemble the node inside its context, and box it
/// into an opaque handle (null on failure).
fn build_host(config: HostFfiConfig) -> *mut daemon_host_t {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| e.to_string())?;
        // `assemble` starts resident services + the fleet via `tokio::spawn`, so it must run inside
        // the runtime context.
        let assembled = {
            let _enter = rt.enter();
            build_node(&config)?
        };
        let AssembledNode { node, handle, .. } = assembled;
        Ok::<_, String>(Box::new(daemon_host_t {
            rt,
            node,
            handle: Some(handle),
        }))
    }));
    match result {
        Ok(Ok(b)) => Box::into_raw(b),
        Ok(Err(msg)) => {
            set_last_error(msg);
            std::ptr::null_mut()
        }
        Err(_) => {
            set_last_error("panic assembling host");
            std::ptr::null_mut()
        }
    }
}

/// Free a host handle created by `daemon_host_new`/`daemon_host_new_with_config`, gracefully
/// shutting its resident services down first.
///
/// # Safety
/// `h` must be a pointer returned by a `daemon_host_new*` and not already freed.
#[no_mangle]
pub unsafe extern "C" fn daemon_host_free(h: *mut daemon_host_t) {
    if h.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| {
        let mut host = unsafe { Box::from_raw(h) };
        if let Some(handle) = host.handle.take() {
            host.rt.block_on(handle.shutdown());
        }
        drop(host);
    }));
}

/// Dispatch a CBOR-encoded [`ApiRequest`] against the node and return the CBOR-encoded
/// [`ApiResponse`]. On `DAEMON_OK`, `*out_resp` points to a library-owned buffer of `*out_len`
/// bytes that the caller releases with `daemon_buf_free` (callee-allocates / callee-frees,
/// daemon-ffi-spec §3.1).
///
/// This one call carries the entire node surface — `Submit`/`Poll`/`Respond`, fleet/tree, fs/cron,
/// model/profile/credential/auth — exactly as the Unix-socket transport routes it.
///
/// # Safety
/// `h` must be valid; `req` must point to `req_len` readable bytes; `out_resp` and `out_len` must be
/// writable.
#[no_mangle]
pub unsafe extern "C" fn daemon_host_call(
    h: *mut daemon_host_t,
    req: *const u8,
    req_len: usize,
    out_resp: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    if h.is_null() || out_resp.is_null() || out_len.is_null() || (req.is_null() && req_len != 0) {
        set_last_error("null argument to daemon_host_call");
        return DAEMON_INVALID;
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        let host = unsafe { &*h };
        let bytes = unsafe { std::slice::from_raw_parts(req, req_len) };
        let request: ApiRequest = from_cbor(bytes).map_err(|e| e.to_string())?;
        let response: ApiResponse = host.rt.block_on(dispatch(host.node.as_ref(), request));
        Ok::<_, String>(to_cbor(&response))
    }));
    match result {
        Ok(Ok(bytes)) => {
            // A boxed slice has capacity == len, so `daemon_buf_free` can reconstruct it exactly.
            let boxed = bytes.into_boxed_slice();
            let len = boxed.len();
            let ptr = Box::into_raw(boxed) as *mut u8;
            unsafe {
                *out_resp = ptr;
                *out_len = len;
            }
            DAEMON_OK
        }
        Ok(Err(msg)) => {
            set_last_error(msg);
            DAEMON_ERROR
        }
        Err(_) => {
            set_last_error("panic in daemon_host_call");
            DAEMON_PANIC
        }
    }
}

/// Copy the thread-local last-error message (UTF-8, not NUL-terminated) into `buf`, writing its full
/// length into `out_len`. Returns `DAEMON_OK`.
///
/// # Safety
/// `buf` must point to `cap` writable bytes; `out_len` must be writable.
#[no_mangle]
pub unsafe extern "C" fn daemon_last_error(buf: *mut u8, cap: usize, out_len: *mut usize) -> i32 {
    if out_len.is_null() || (buf.is_null() && cap != 0) {
        return DAEMON_INVALID;
    }
    LAST_ERROR.with(|e| {
        let msg = e.borrow();
        let bytes = msg.as_bytes();
        unsafe { *out_len = bytes.len() };
        let n = bytes.len().min(cap);
        if n > 0 {
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf, n) };
        }
    });
    DAEMON_OK
}

/// Free a library-allocated buffer `(ptr, len)` handed out by `daemon_host_call`.
///
/// # Safety
/// `(ptr, len)` must be a buffer previously returned by `daemon_host_call`, not already freed.
#[no_mangle]
pub unsafe extern "C" fn daemon_buf_free(ptr: *mut u8, len: usize) {
    if ptr.is_null() || len == 0 {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| {
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
        drop(unsafe { Box::from_raw(slice as *mut [u8]) });
    }));
}

#[cfg(test)]
mod config_tests {
    //! The construction-config surface: a [`HostFfiConfig`] survives the CBOR round-trip the FFI
    //! entry point decodes, and `provider_builder` selects a real (streaming) provider for `GenAi`
    //! while the default config stays on the mock node.

    use super::*;

    #[test]
    fn host_ffi_config_round_trips_through_cbor() {
        let cfg = HostFfiConfig {
            store_path: Some("/tmp/daemon.db".into()),
            partition: 7,
            provider: ProviderSelector::GenAi,
            model: "gpt-4o".into(),
            base_url: Some("https://gateway.example/v1".into()),
            api_key: Some("sk-test".into()),
            profile: "work".into(),
            max_output_tokens: Some(8192),
            engine: Some(Config::default()),
            workspace_root: Some("/tmp/ws".into()),
            blob_root: Some("/tmp/blobs".into()),
        };
        let bytes = to_cbor(&cfg);
        let decoded: HostFfiConfig = from_cbor(&bytes).expect("decode HostFfiConfig");
        assert_eq!(cfg, decoded);
    }

    #[test]
    fn empty_blob_decodes_to_default_via_serde_default() {
        let decoded: HostFfiConfig = from_cbor(&[0xA0]).expect("decode empty map");
        assert_eq!(decoded, HostFfiConfig::default());
        assert_eq!(decoded.provider, ProviderSelector::Mock);
        assert!(decoded.store_path.is_none());
    }

    #[test]
    fn genai_config_builds_a_real_streaming_provider() {
        let cfg = HostFfiConfig {
            provider: ProviderSelector::GenAi,
            model: "gpt-4o".into(),
            ..HostFfiConfig::default()
        };
        assert!(
            provider_builder(&cfg)().capabilities().supports_streaming,
            "GenAi config must build the real (streaming) provider"
        );
    }

    #[test]
    fn default_config_builds_the_mock_provider() {
        assert!(
            !provider_builder(&HostFfiConfig::default())()
                .capabilities()
                .supports_streaming,
            "default config must stay on the non-streaming mock node"
        );
    }
}

#[cfg(test)]
mod wire_fixtures {
    //! Pins the exact CBOR the C harness (`harness/harness.c`) embeds for its node calls. If the
    //! `daemon-api`/`daemon-protocol` serde encoding ever changes, this fails — the signal to
    //! regenerate the harness fixtures so the cross-language gate stays honest.

    use super::*;
    use daemon_common::{ReqId, SessionId};
    use daemon_protocol::{AgentCommand, UserMsg};

    /// CBOR for `ApiRequest::Submit { session: "ffi-host", command: StartTurn { input: { text:
    /// "hi", attachments: [] }, request_id: 1 }, origin: null, profile: null }`.
    const SUBMIT_START_TURN: &[u8] = &[
        0xA1, // map(1)
        0x66, b'S', b'u', b'b', b'm', b'i', b't', // "Submit"
        0xA4, // map(4)
        0x67, b's', b'e', b's', b's', b'i', b'o', b'n', // "session"
        0x68, b'f', b'f', b'i', b'-', b'h', b'o', b's', b't', // "ffi-host"
        0x67, b'c', b'o', b'm', b'm', b'a', b'n', b'd', // "command"
        0xA1, // map(1)
        0x69, b'S', b't', b'a', b'r', b't', b'T', b'u', b'r', b'n', // "StartTurn"
        0xA2, // map(2)
        0x65, b'i', b'n', b'p', b'u', b't', // "input"
        0xA2, // map(2)
        0x64, b't', b'e', b'x', b't', // "text"
        0x62, b'h', b'i', // "hi"
        0x6B, b'a', b't', b't', b'a', b'c', b'h', b'm', b'e', b'n', b't', b's', // "attachments"
        0x80, // array(0)
        0x6A, b'r', b'e', b'q', b'u', b'e', b's', b't', b'_', b'i', b'd', // "request_id"
        0x01, // 1
        0x66, b'o', b'r', b'i', b'g', b'i', b'n', // "origin"
        0xF6, // null
        0x67, b'p', b'r', b'o', b'f', b'i', b'l', b'e', // "profile"
        0xF6, // null
    ];

    /// CBOR for `ApiRequest::Poll { session: "ffi-host", max: 16 }`.
    const POLL: &[u8] = &[
        0xA1, // map(1)
        0x64, b'P', b'o', b'l', b'l', // "Poll"
        0xA2, // map(2)
        0x67, b's', b'e', b's', b's', b'i', b'o', b'n', // "session"
        0x68, b'f', b'f', b'i', b'-', b'h', b'o', b's', b't', // "ffi-host"
        0x63, b'm', b'a', b'x', // "max"
        0x10, // 16
    ];

    #[test]
    fn submit_start_turn_matches_canonical_cbor() {
        let req = ApiRequest::Submit {
            session: SessionId::new("ffi-host"),
            command: AgentCommand::StartTurn {
                input: UserMsg::new("hi"),
                request_id: ReqId(1),
            },
            origin: None,
            profile: None,
        };
        assert_eq!(
            to_cbor(&req),
            SUBMIT_START_TURN,
            "the C harness Submit fixture is stale; regenerate harness/harness.c"
        );
    }

    #[test]
    fn poll_matches_canonical_cbor() {
        let req = ApiRequest::Poll {
            session: SessionId::new("ffi-host"),
            max: 16,
        };
        assert_eq!(
            to_cbor(&req),
            POLL,
            "the C harness Poll fixture is stale; regenerate harness/harness.c"
        );
    }
}

#[cfg(test)]
mod dispatch_tests {
    //! The durable node end-to-end over the in-process assembly: boot a host, drive a `StartTurn`
    //! and drain its `TurnFinished` through the same `dispatch` the C ABI calls — proving the
    //! generic CBOR call carries the §17 session surface on top of a real durable node.

    use super::*;
    use daemon_api::Outbound;
    use daemon_common::{ReqId, SessionId};
    use daemon_protocol::{AgentCommand, AgentEvent, UserMsg};
    use std::time::Duration;

    #[test]
    fn boots_a_node_and_drains_turn_finished() {
        let host = build_host(HostFfiConfig::default());
        assert!(!host.is_null(), "host must boot");
        let host = unsafe { &mut *host };
        let session = SessionId::new("ffi-it");

        // Submit a StartTurn through the node dispatch.
        let submit = ApiRequest::Submit {
            session: session.clone(),
            command: AgentCommand::StartTurn {
                input: UserMsg::new("hi"),
                request_id: ReqId(1),
            },
            origin: None,
            profile: None,
        };
        let resp = host.rt.block_on(dispatch(host.node.as_ref(), submit));
        assert!(
            !matches!(resp, ApiResponse::Error(_)),
            "submit should be accepted: {resp:?}"
        );

        // Poll the drain until the mock provider's terminal event arrives.
        let mut finished = false;
        for _ in 0..200 {
            let poll = ApiRequest::Poll {
                session: session.clone(),
                max: 16,
            };
            if let ApiResponse::Drained(items) =
                host.rt.block_on(dispatch(host.node.as_ref(), poll))
            {
                if items
                    .iter()
                    .any(|o| matches!(o, Outbound::Event(AgentEvent::TurnFinished { .. })))
                {
                    finished = true;
                    break;
                }
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        assert!(finished, "must drain TurnFinished over the node dispatch");

        // Graceful shutdown.
        if let Some(handle) = host.handle.take() {
            host.rt.block_on(handle.shutdown());
        }
        drop(unsafe { Box::from_raw(host as *mut daemon_host_t) });
    }
}
