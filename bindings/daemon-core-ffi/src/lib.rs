//! `daemon-core-ffi` — the C ABI adapter over the §17 **session sub-surface**.
//!
//! This is one more transport over the one node interface ([`daemon_api`]): where the Unix socket
//! moves framed CBOR over a stream, this moves the *same* CBOR mirror across opaque handles and
//! caller buffers, routed through the *same* [`daemon_api::dispatch_session`]. Only the byte
//! transport differs (daemon-ffi-spec §1).
//!
//! Shape (daemon-ffi-spec §3):
//! - opaque [`daemon_runtime_t`] owns the Tokio runtime + a [`CoreSessionApi`] (the L1 brain: a
//!   completing engine per session, no durable host);
//! - opaque [`daemon_session_t`] binds a runtime to one session id;
//! - `submit`/`poll`/`respond` marshal CBOR [`daemon_protocol`] values through `dispatch_session`;
//! - every entry point is `catch_unwind`-guarded (panics never cross the boundary; relies on the
//!   workspace `panic = "unwind"`), reporting failure via a `daemon_status` code + a thread-local
//!   last-error message.

#![deny(unsafe_op_in_unsafe_fn)]
#![allow(non_camel_case_types)]

use std::any::Any;
use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use daemon_api::{
    dispatch_session, from_cbor, to_cbor, ApiError, ApiRequest, ApiResponse, LogPageView,
    LogStream, Outbound, SessionApi,
};
use daemon_common::{ReqId, SessionId};
use daemon_core::{
    spawn_agent_session, AgentHandle, Config, Engine, EngineProfile, MockProvider, Provider,
    SystemPrompt, ToolRegistry,
};
use daemon_protocol::{
    AgentCommand, Direction, Disposition, HostRequest, HostRequestHandler, HostResponse,
    HostResponseBody, Origin, OriginScope, SessionLogEntry, SessionPayload, TransportId,
};
use dashmap::DashMap;
use futures::stream::{self, StreamExt};
use tokio::runtime::{Handle, Runtime};
use tokio::sync::{broadcast, oneshot};
use tokio::task::JoinHandle;
use tokio_stream::wrappers::BroadcastStream;

/// The session's own attribution for engine-emitted (outbound) merged-log entries.
fn engine_origin() -> Origin {
    Origin {
        transport: TransportId::new("engine"),
        scope: OriginScope::Internal,
    }
}

/// The attribution stamped on inbound items entering through the FFI session surface.
fn ffi_origin() -> Origin {
    Origin {
        transport: TransportId::new("ffi"),
        scope: OriginScope::Internal,
    }
}

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
/// No item was available to drain (poll on an idle session).
pub const DAEMON_EMPTY: i32 = 2;
/// A panic was caught at the boundary (should not happen in normal operation).
pub const DAEMON_PANIC: i32 = 3;
/// A null handle or invalid argument was passed.
pub const DAEMON_INVALID: i32 = 4;
/// The caller buffer was too small for the next item.
pub const DAEMON_BUFFER_TOO_SMALL: i32 = 5;

thread_local! {
    static LAST_ERROR: RefCell<String> = const { RefCell::new(String::new()) };
}

fn set_last_error(msg: impl Into<String>) {
    LAST_ERROR.with(|e| *e.borrow_mut() = msg.into());
}

/// Map a `catch_unwind` + fallible body result onto a `daemon_status`.
fn finish(result: Result<Result<(), String>, Box<dyn Any + Send>>) -> i32 {
    match result {
        Ok(Ok(())) => DAEMON_OK,
        Ok(Err(msg)) => {
            set_last_error(msg);
            DAEMON_ERROR
        }
        Err(_) => {
            set_last_error("panic caught at FFI boundary");
            DAEMON_PANIC
        }
    }
}

// ---------------------------------------------------------------------------
// opaque handles
// ---------------------------------------------------------------------------

/// Opaque runtime handle: owns the Tokio runtime and the session-surface implementation.
pub struct daemon_runtime_t {
    rt: Runtime,
    api: Arc<CoreSessionApi>,
}

/// Opaque session handle: a runtime handle + the session-surface impl + a bound session id.
pub struct daemon_session_t {
    handle: Handle,
    api: Arc<CoreSessionApi>,
    session: SessionId,
}

// ---------------------------------------------------------------------------
// C ABI
// ---------------------------------------------------------------------------

/// The ABI version of this shell.
#[no_mangle]
pub extern "C" fn daemon_abi_version() -> u32 {
    ABI_VERSION
}

/// Create a runtime handle. Returns null on failure (see `daemon_last_error`).
#[no_mangle]
pub extern "C" fn daemon_runtime_new() -> *mut daemon_runtime_t {
    let result = catch_unwind(|| {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| e.to_string())?;
        Ok::<_, String>(Box::new(daemon_runtime_t {
            rt,
            api: Arc::new(CoreSessionApi::new()),
        }))
    });
    match result {
        Ok(Ok(b)) => Box::into_raw(b),
        Ok(Err(msg)) => {
            set_last_error(msg);
            std::ptr::null_mut()
        }
        Err(_) => {
            set_last_error("panic constructing runtime");
            std::ptr::null_mut()
        }
    }
}

/// Free a runtime handle created by `daemon_runtime_new`.
///
/// # Safety
/// `rt` must be a pointer returned by `daemon_runtime_new` and not already freed.
#[no_mangle]
pub unsafe extern "C" fn daemon_runtime_free(rt: *mut daemon_runtime_t) {
    if rt.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| {
        drop(unsafe { Box::from_raw(rt) });
    }));
}

/// Open a session bound to `rt`, identified by the UTF-8 name `(name, name_len)`. Returns null on
/// failure.
///
/// # Safety
/// `rt` must be valid; `name` must point to `name_len` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn daemon_session_open(
    rt: *mut daemon_runtime_t,
    name: *const u8,
    name_len: usize,
) -> *mut daemon_session_t {
    if rt.is_null() || (name.is_null() && name_len != 0) {
        set_last_error("null argument to daemon_session_open");
        return std::ptr::null_mut();
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        let runtime = unsafe { &*rt };
        let bytes = unsafe { std::slice::from_raw_parts(name, name_len) };
        let id = std::str::from_utf8(bytes).map_err(|e| e.to_string())?;
        Ok::<_, String>(Box::new(daemon_session_t {
            handle: runtime.rt.handle().clone(),
            api: runtime.api.clone(),
            session: SessionId::new(id),
        }))
    }));
    match result {
        Ok(Ok(b)) => Box::into_raw(b),
        Ok(Err(msg)) => {
            set_last_error(msg);
            std::ptr::null_mut()
        }
        Err(_) => {
            set_last_error("panic opening session");
            std::ptr::null_mut()
        }
    }
}

/// Free a session handle created by `daemon_session_open`.
///
/// # Safety
/// `s` must be a pointer returned by `daemon_session_open` and not already freed.
#[no_mangle]
pub unsafe extern "C" fn daemon_session_free(s: *mut daemon_session_t) {
    if s.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| {
        drop(unsafe { Box::from_raw(s) });
    }));
}

/// Submit a CBOR-encoded `AgentCommand` to the session.
///
/// # Safety
/// `s` must be valid; `cmd` must point to `len` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn daemon_session_submit(
    s: *mut daemon_session_t,
    cmd: *const u8,
    len: usize,
) -> i32 {
    if s.is_null() || (cmd.is_null() && len != 0) {
        set_last_error("null argument to daemon_session_submit");
        return DAEMON_INVALID;
    }
    finish(catch_unwind(AssertUnwindSafe(|| {
        let session = unsafe { &*s };
        let bytes = unsafe { std::slice::from_raw_parts(cmd, len) };
        let command: AgentCommand = from_cbor(bytes).map_err(|e| e.to_string())?;
        let req = ApiRequest::Submit {
            session: session.session.clone(),
            command,
            origin: None,
        };
        match session
            .handle
            .block_on(dispatch_session(session.api.as_ref(), req))
        {
            ApiResponse::Error(e) => Err(e.to_string()),
            _ => Ok(()),
        }
    })))
}

/// Drain the next outbound item (CBOR-encoded [`daemon_api::Outbound`]) into the caller buffer.
/// Returns `DAEMON_EMPTY` when idle, `DAEMON_BUFFER_TOO_SMALL` if `cap` is too small (and writes the
/// needed length into `out_len`).
///
/// # Safety
/// `s` must be valid; `out_buf` must point to `cap` writable bytes; `out_len` must be writable.
#[no_mangle]
pub unsafe extern "C" fn daemon_session_poll(
    s: *mut daemon_session_t,
    out_buf: *mut u8,
    cap: usize,
    out_len: *mut usize,
) -> i32 {
    if s.is_null() || out_len.is_null() || (out_buf.is_null() && cap != 0) {
        set_last_error("null argument to daemon_session_poll");
        return DAEMON_INVALID;
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        let session = unsafe { &*s };
        let req = ApiRequest::Poll {
            session: session.session.clone(),
            max: 1,
        };
        match session
            .handle
            .block_on(dispatch_session(session.api.as_ref(), req))
        {
            ApiResponse::Drained(mut items) => match items.pop() {
                None => Ok(None),
                Some(item) => Ok(Some(to_cbor(&item))),
            },
            ApiResponse::Error(e) => Err(e.to_string()),
            other => Err(format!("unexpected poll response: {other:?}")),
        }
    }));

    match result {
        Ok(Ok(None)) => DAEMON_EMPTY,
        Ok(Ok(Some(bytes))) => {
            unsafe { *out_len = bytes.len() };
            if bytes.len() > cap {
                return DAEMON_BUFFER_TOO_SMALL;
            }
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), out_buf, bytes.len()) };
            DAEMON_OK
        }
        Ok(Err(msg)) => {
            set_last_error(msg);
            DAEMON_ERROR
        }
        Err(_) => {
            set_last_error("panic polling session");
            DAEMON_PANIC
        }
    }
}

/// Answer a parked host request with a CBOR-encoded `HostResponse` (its `request_id` correlates).
///
/// # Safety
/// `s` must be valid; `resp` must point to `len` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn daemon_session_respond(
    s: *mut daemon_session_t,
    resp: *const u8,
    len: usize,
) -> i32 {
    if s.is_null() || (resp.is_null() && len != 0) {
        set_last_error("null argument to daemon_session_respond");
        return DAEMON_INVALID;
    }
    finish(catch_unwind(AssertUnwindSafe(|| {
        let session = unsafe { &*s };
        let bytes = unsafe { std::slice::from_raw_parts(resp, len) };
        let response: HostResponse = from_cbor(bytes).map_err(|e| e.to_string())?;
        let req = ApiRequest::Respond {
            session: session.session.clone(),
            response,
        };
        match session
            .handle
            .block_on(dispatch_session(session.api.as_ref(), req))
        {
            ApiResponse::Error(e) => Err(e.to_string()),
            _ => Ok(()),
        }
    })))
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

/// Free a library-allocated byte buffer `(ptr, len)`. Provided for the callee-allocates ownership
/// convention (daemon-ffi-spec §3.1); the poll path uses caller buffers and does not require it.
///
/// # Safety
/// `(ptr, len)` must be a buffer previously handed out by this library, not already freed.
#[no_mangle]
pub unsafe extern "C" fn daemon_buf_free(ptr: *mut u8, len: usize) {
    if ptr.is_null() || len == 0 {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| {
        drop(unsafe { Vec::from_raw_parts(ptr, len, len) });
    }));
}

// ---------------------------------------------------------------------------
// CoreSessionApi — the L1 brain behind the session sub-surface
// ---------------------------------------------------------------------------

type Drain = Arc<Mutex<VecDeque<Outbound>>>;
type Pending = Arc<Mutex<HashMap<ReqId, oneshot::Sender<HostResponse>>>>;
type Merged = Arc<Mutex<MergedLog>>;

/// The non-destructive, `seq`-stamped merged session event log (inbound + outbound) backing the
/// long-poll cursor read ([`SessionApi::log_after`]) and the live push ([`SessionApi::subscribe`]).
/// A focused mirror of the durable host's log, kept self-contained so the brain FFI stays free of
/// `daemon-host`.
struct MergedLog {
    next_seq: u64,
    entries: Vec<SessionLogEntry>,
    tx: broadcast::Sender<SessionLogEntry>,
}

impl MergedLog {
    fn new() -> Self {
        let (tx, _rx) = broadcast::channel(256);
        // Seq starts at 1 so the `after_seq` cursor convention (exclusive lower bound; 0 = "from the
        // start") can address the very first entry.
        Self {
            next_seq: 1,
            entries: Vec::new(),
            tx,
        }
    }

    fn append(
        &mut self,
        direction: Direction,
        origin: Origin,
        disposition: Disposition,
        payload: SessionPayload,
    ) {
        let seq = self.next_seq;
        self.next_seq += 1;
        let entry = SessionLogEntry {
            seq,
            direction,
            origin,
            disposition,
            payload,
        };
        self.entries.push(entry.clone());
        let _ = self.tx.send(entry);
    }

    fn page(&self, after_seq: u64, max: u32) -> LogPageView {
        let head_seq = self.next_seq.saturating_sub(1);
        let mut entries = Vec::new();
        for e in self.entries.iter().filter(|e| e.seq > after_seq) {
            if max != 0 && entries.len() >= max as usize {
                break;
            }
            entries.push(e.clone());
        }
        let next_seq = entries.last().map(|e| e.seq).unwrap_or(after_seq);
        LogPageView {
            entries,
            next_seq,
            head_seq,
        }
    }

    fn subscribe(&self, after_seq: u64) -> LogStream {
        let backlog: Vec<SessionLogEntry> = self
            .entries
            .iter()
            .filter(|e| e.seq > after_seq)
            .cloned()
            .collect();
        let rx = self.tx.subscribe();
        let live = BroadcastStream::new(rx).filter_map(|r| async move { r.ok() });
        stream::iter(backlog).chain(live).boxed()
    }
}

struct LiveSession {
    handle: AgentHandle,
    drain: Drain,
    pending: Pending,
    log: Merged,
    pump: JoinHandle<()>,
}

impl Drop for LiveSession {
    fn drop(&mut self) {
        self.pump.abort();
    }
}

/// A standalone [`SessionApi`] over the §17 actor: one completing engine per session, with the same
/// poll/drain + parked-request model the durable node uses — kept self-contained so the brain FFI
/// stays free of `daemon-host`.
struct CoreSessionApi {
    sessions: DashMap<SessionId, LiveSession>,
}

impl CoreSessionApi {
    fn new() -> Self {
        Self {
            sessions: DashMap::new(),
        }
    }

    fn build_engine(id: SessionId) -> Engine {
        // Construct through a `with_config`-dressed `EngineProfile` so the FFI's engine matches the
        // rest of the system's construction seam (explicit tunables rather than a silent default).
        //
        // This embed path is deliberately self-contained (it cannot depend on `daemon-host`/
        // `daemon-node`). Injecting a real provider, brokered credentials, or a verifiable-journal
        // sink would need new C-ABI surface to pass those in; that is deferred to the embedding /
        // ACP phase. Until then it uses the Mock provider, the engine's embedded L1 credential pool,
        // and no journal.
        EngineProfile::new(
            Arc::new(|| {
                Arc::new(MockProvider::completing("ffi session done")) as Arc<dyn Provider>
            }),
            Arc::new(ToolRegistry::new()),
            SystemPrompt::new("daemon-core-ffi session"),
        )
        .with_config(Config::default())
        .fresh(id)
    }

    fn ensure(&self, session: &SessionId) -> AgentHandle {
        if let Some(s) = self.sessions.get(session) {
            return s.handle.clone();
        }
        let drain: Drain = Arc::new(Mutex::new(VecDeque::new()));
        let pending: Pending = Arc::new(Mutex::new(HashMap::new()));
        let log: Merged = Arc::new(Mutex::new(MergedLog::new()));
        let host = Arc::new(ParkingHandler {
            drain: drain.clone(),
            pending: pending.clone(),
            log: log.clone(),
        });
        let handle = spawn_agent_session(Self::build_engine(session.clone()), host);

        let mut rx = handle.subscribe();
        let pump_drain = drain.clone();
        let pump_log = log.clone();
        let pump = tokio::spawn(async move {
            loop {
                match rx.recv().await {
                    Ok(ev) => {
                        pump_log.lock().unwrap().append(
                            Direction::Outbound,
                            engine_origin(),
                            Disposition::Context,
                            SessionPayload::Event(ev.clone()),
                        );
                        pump_drain.lock().unwrap().push_back(Outbound::Event(ev));
                    }
                    Err(broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
        });

        self.sessions.insert(
            session.clone(),
            LiveSession {
                handle: handle.clone(),
                drain,
                pending,
                log,
                pump,
            },
        );
        handle
    }

    /// Append an inbound entry to a session's merged log (no-op if the session is gone), attributed
    /// to `origin` so per-event provenance is preserved on the log.
    fn record_inbound(
        &self,
        session: &SessionId,
        origin: Origin,
        disposition: Disposition,
        payload: SessionPayload,
    ) {
        if let Some(s) = self.sessions.get(session) {
            s.log
                .lock()
                .unwrap()
                .append(Direction::Inbound, origin, disposition, payload);
        }
    }
}

#[async_trait]
impl SessionApi for CoreSessionApi {
    async fn submit(&self, session: SessionId, command: AgentCommand) -> Result<(), ApiError> {
        // The bare FFI surface carries no external attribution: default to the host-internal `ffi`
        // origin. Callers that have an `Origin` use `submit_from`.
        self.submit_from(session, ffi_origin(), command).await
    }

    async fn submit_from(
        &self,
        session: SessionId,
        origin: Origin,
        command: AgentCommand,
    ) -> Result<(), ApiError> {
        match command {
            AgentCommand::StartTurn { input, request_id } => {
                let handle = self.ensure(&session);
                self.record_inbound(
                    &session,
                    origin,
                    Disposition::Context,
                    SessionPayload::Command(AgentCommand::StartTurn {
                        input: input.clone(),
                        request_id,
                    }),
                );
                tokio::spawn(async move {
                    let _ = handle.start_turn(input).await;
                });
                Ok(())
            }
            AgentCommand::Interrupt { reason } => {
                let handle = self
                    .sessions
                    .get(&session)
                    .map(|s| s.handle.clone())
                    .ok_or_else(|| ApiError::UnknownSession(session.to_string()))?;
                self.record_inbound(
                    &session,
                    origin,
                    Disposition::Transport,
                    SessionPayload::Command(AgentCommand::Interrupt {
                        reason: reason.clone(),
                    }),
                );
                handle.interrupt(reason).await;
                Ok(())
            }
            AgentCommand::Shutdown => {
                self.record_inbound(
                    &session,
                    origin,
                    Disposition::Transport,
                    SessionPayload::Command(AgentCommand::Shutdown),
                );
                if let Some((_, s)) = self.sessions.remove(&session) {
                    s.handle.shutdown().await;
                }
                Ok(())
            }
            AgentCommand::Steer { text, request_id } => {
                let handle = self.ensure(&session);
                self.record_inbound(
                    &session,
                    origin,
                    Disposition::Context,
                    SessionPayload::Command(AgentCommand::Steer {
                        text: text.clone(),
                        request_id,
                    }),
                );
                handle.steer(request_id, text).await;
                Ok(())
            }
            AgentCommand::Snapshot { request_id } => {
                let handle = self
                    .sessions
                    .get(&session)
                    .map(|s| s.handle.clone())
                    .ok_or_else(|| ApiError::UnknownSession(session.to_string()))?;
                self.record_inbound(
                    &session,
                    origin,
                    Disposition::Transport,
                    SessionPayload::Command(AgentCommand::Snapshot { request_id }),
                );
                handle.snapshot(request_id).await;
                Ok(())
            }
            _ => Err(ApiError::Unsupported("unknown agent command".into())),
        }
    }

    async fn poll(&self, session: SessionId, max: u32) -> Result<Vec<Outbound>, ApiError> {
        let s = self
            .sessions
            .get(&session)
            .ok_or_else(|| ApiError::UnknownSession(session.to_string()))?;
        let mut q = s.drain.lock().unwrap();
        let take = if max == 0 {
            q.len()
        } else {
            (max as usize).min(q.len())
        };
        Ok(q.drain(..take).collect())
    }

    async fn respond(&self, session: SessionId, response: HostResponse) -> Result<(), ApiError> {
        let s = self
            .sessions
            .get(&session)
            .ok_or_else(|| ApiError::UnknownSession(session.to_string()))?;
        let tx = s.pending.lock().unwrap().remove(&response.request_id);
        match tx {
            Some(tx) => {
                s.log.lock().unwrap().append(
                    Direction::Inbound,
                    ffi_origin(),
                    Disposition::Context,
                    SessionPayload::Response(response.clone()),
                );
                let _ = tx.send(response);
                Ok(())
            }
            None => Err(ApiError::Other("no parked request for that id".into())),
        }
    }

    async fn log_after(
        &self,
        session: SessionId,
        after_seq: u64,
        max: u32,
    ) -> Result<LogPageView, ApiError> {
        match self.sessions.get(&session) {
            Some(s) => Ok(s.log.lock().unwrap().page(after_seq, max)),
            None => Ok(LogPageView::default()),
        }
    }

    async fn subscribe(&self, session: SessionId, after_seq: u64) -> Result<LogStream, ApiError> {
        match self.sessions.get(&session) {
            Some(s) => Ok(s.log.lock().unwrap().subscribe(after_seq)),
            None => Ok(stream::empty().boxed()),
        }
    }

    async fn record_meta(
        &self,
        session: SessionId,
        origin: Origin,
        kind: String,
        body: Vec<u8>,
    ) -> Result<(), ApiError> {
        // Observability-only: lands on the merged log + broadcast as `Transport`, never the engine
        // or journal. No-op if the session is gone.
        self.record_inbound(
            &session,
            origin,
            Disposition::Transport,
            SessionPayload::Meta { kind, body },
        );
        Ok(())
    }
}

/// Parks each blocking §17 request onto the drain queue + a pending table; the engine's
/// `oneshot` completes when `daemon_session_respond` arrives (daemon-ffi-spec §3.3).
struct ParkingHandler {
    drain: Drain,
    pending: Pending,
    log: Merged,
}

#[async_trait]
impl HostRequestHandler for ParkingHandler {
    async fn request(&self, req: HostRequest) -> HostResponse {
        let (tx, rx) = oneshot::channel();
        let request_id = req.request_id;
        self.pending.lock().unwrap().insert(request_id, tx);
        self.log.lock().unwrap().append(
            Direction::Outbound,
            engine_origin(),
            Disposition::Context,
            SessionPayload::Request(req.clone()),
        );
        self.drain.lock().unwrap().push_back(Outbound::Request(req));
        match rx.await {
            Ok(resp) => resp,
            Err(_) => HostResponse {
                request_id,
                body: HostResponseBody::Approved(false),
            },
        }
    }
}

#[cfg(test)]
mod fixture_tests {
    //! Pins the exact CBOR the C harness (`harness/harness.c`) embeds for its `StartTurn`. If the
    //! protocol's serde encoding ever changes, this fails — the signal to regenerate the harness
    //! fixture so the cross-language gate stays honest.

    use super::*;

    /// CBOR for `AgentCommand::StartTurn { input: { text: "hi" }, request_id: 1 }`
    /// (externally-tagged: `{"StartTurn": {"input": {"text": "hi"}, "request_id": 1}}`).
    const START_TURN_HI: &[u8] = &[
        0xA1, // map(1)
        0x69, b'S', b't', b'a', b'r', b't', b'T', b'u', b'r', b'n', // "StartTurn"
        0xA2, // map(2)
        0x65, b'i', b'n', b'p', b'u', b't', // "input"
        0xA1, // map(1)
        0x64, b't', b'e', b'x', b't', // "text"
        0x62, b'h', b'i', // "hi"
        0x6A, b'r', b'e', b'q', b'u', b'e', b's', b't', b'_', b'i', b'd', // "request_id"
        0x01, // 1
    ];

    #[test]
    fn start_turn_fixture_matches_canonical_cbor() {
        let cmd = AgentCommand::StartTurn {
            input: daemon_protocol::UserMsg::new("hi"),
            request_id: ReqId(1),
        };
        assert_eq!(
            to_cbor(&cmd),
            START_TURN_HI,
            "the C harness fixture is stale; regenerate harness/harness.c"
        );
    }

    /// CBOR for `AgentCommand::Snapshot { request_id: 2 }`
    /// (externally-tagged: `{"Snapshot": {"request_id": 2}}`).
    const SNAPSHOT_2: &[u8] = &[
        0xA1, // map(1)
        0x68, b'S', b'n', b'a', b'p', b's', b'h', b'o', b't', // "Snapshot"
        0xA1, // map(1)
        0x6A, b'r', b'e', b'q', b'u', b'e', b's', b't', b'_', b'i', b'd', // "request_id"
        0x02, // 2
    ];

    #[test]
    fn snapshot_fixture_matches_canonical_cbor() {
        let cmd = AgentCommand::Snapshot {
            request_id: ReqId(2),
        };
        assert_eq!(
            to_cbor(&cmd),
            SNAPSHOT_2,
            "the C harness fixture is stale; regenerate harness/harness.c"
        );
    }

    /// CBOR for `AgentCommand::Steer { text: "go", request_id: 3 }`
    /// (externally-tagged: `{"Steer": {"text": "go", "request_id": 3}}`).
    const STEER_GO: &[u8] = &[
        0xA1, // map(1)
        0x65, b'S', b't', b'e', b'e', b'r', // "Steer"
        0xA2, // map(2)
        0x64, b't', b'e', b'x', b't', // "text"
        0x62, b'g', b'o', // "go"
        0x6A, b'r', b'e', b'q', b'u', b'e', b's', b't', b'_', b'i', b'd', // "request_id"
        0x03, // 3
    ];

    #[test]
    fn steer_fixture_matches_canonical_cbor() {
        let cmd = AgentCommand::Steer {
            text: "go".into(),
            request_id: ReqId(3),
        };
        assert_eq!(
            to_cbor(&cmd),
            STEER_GO,
            "the C harness fixture is stale; regenerate harness/harness.c"
        );
    }
}

#[cfg(test)]
mod merged_log_tests {
    //! The non-destructive merged log over the self-contained FFI [`CoreSessionApi`]: a submitted
    //! `StartTurn` is recorded as an inbound entry under the unified `seq`, ahead of the engine's
    //! outbound replies, and `log_after` is a non-destructive cursor (re-reads from the same cursor
    //! return the same entries — unlike the destructive `poll` drain).

    use super::*;
    use daemon_protocol::UserMsg;
    use std::time::Duration;

    #[tokio::test]
    async fn log_after_records_inbound_then_outbound_non_destructively() {
        let api = CoreSessionApi::new();
        let session = SessionId::new("merged");

        api.submit(
            session.clone(),
            AgentCommand::StartTurn {
                input: UserMsg::new("hi"),
                request_id: ReqId(1),
            },
        )
        .await
        .expect("submit");

        // Wait until the background turn has produced its terminal outbound event.
        let mut page = LogPageView::default();
        for _ in 0..50 {
            page = api.log_after(session.clone(), 0, 0).await.unwrap();
            let done = page.entries.iter().any(|e| {
                matches!(
                    &e.payload,
                    SessionPayload::Event(daemon_protocol::AgentEvent::TurnFinished { .. })
                )
            });
            if done {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }

        let first = &page.entries[0];
        assert_eq!(first.seq, 1);
        assert_eq!(first.direction, Direction::Inbound);
        assert!(matches!(
            first.payload,
            SessionPayload::Command(AgentCommand::StartTurn { .. })
        ));
        assert!(page
            .entries
            .iter()
            .any(|e| e.direction == Direction::Outbound));

        // Non-destructive: a second read from cursor 0 returns the same head, and a cursor past the
        // head returns nothing while reporting the same head.
        let again = api.log_after(session.clone(), 0, 0).await.unwrap();
        assert_eq!(again.head_seq, page.head_seq);
        assert_eq!(again.entries.len(), page.entries.len());

        let tail = api
            .log_after(session.clone(), page.head_seq, 0)
            .await
            .unwrap();
        assert!(tail.entries.is_empty());
        assert_eq!(tail.head_seq, page.head_seq);
    }

    #[tokio::test]
    async fn submit_from_attributes_inbound_to_origin() {
        // `submit_from` carries per-event provenance onto the merged log: the inbound entry is
        // attributed to the submitting surface's origin, not the host-local `ffi` default.
        let api = CoreSessionApi::new();
        let session = SessionId::new("attributed");
        let origin = Origin::new("telegram", OriginScope::Dm { user: "u1".into() });

        api.submit_from(
            session.clone(),
            origin.clone(),
            AgentCommand::StartTurn {
                input: UserMsg::new("hi"),
                request_id: ReqId(1),
            },
        )
        .await
        .expect("submit_from");

        let page = api.log_after(session.clone(), 0, 0).await.unwrap();
        let inbound = page
            .entries
            .iter()
            .find(|e| e.direction == Direction::Inbound)
            .expect("an inbound entry");
        assert_eq!(inbound.origin, origin);
        assert_ne!(inbound.origin, ffi_origin());
    }

    #[tokio::test]
    async fn record_meta_is_observable_but_not_in_history() {
        // A `Transport` meta event lands on the merged live log (observable via `log_after`) but is
        // never folded into durable history (`session_history`) — observability, not context.
        let api = CoreSessionApi::new();
        let session = SessionId::new("meta");

        // Open the session so it has a live log to record onto.
        api.submit(
            session.clone(),
            AgentCommand::StartTurn {
                input: UserMsg::new("hi"),
                request_id: ReqId(1),
            },
        )
        .await
        .expect("submit");

        let origin = Origin::new(
            "gui",
            OriginScope::Api {
                key: "owner".into(),
            },
        );
        api.record_meta(session.clone(), origin, "attach".into(), vec![1, 2, 3])
            .await
            .expect("record_meta");

        let page = api.log_after(session.clone(), 0, 0).await.unwrap();
        let meta = page
            .entries
            .iter()
            .find(|e| matches!(&e.payload, SessionPayload::Meta { .. }))
            .expect("a meta entry on the live log");
        assert_eq!(meta.disposition, Disposition::Transport);

        // The FFI exposes no durable journal, so history is empty: the meta event is observability
        // only and never graduates into durable history.
        let history = api.session_history(session.clone(), 0, 0).await;
        assert!(history.entries.is_empty());
    }
}
