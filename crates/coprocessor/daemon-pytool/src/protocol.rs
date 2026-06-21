//! The Python-tool worker wire protocol — [`Command`]/[`Event`] frames + a JSON codec.
//!
//! The daemon (`daemon-pytool-client`'s `PyToolHost`) and the Python worker exchange these frames
//! over a length-framed stdio cut ([`daemon_provision::CutChannel`], [`Framing::Length`]). Each
//! frame body is **JSON** (so the Python SDK is stdlib-only); the `u32`-length prefix is handled by
//! the channel, so this module only owns the body [`encode`]/[`decode`].
//!
//! Both enums are internally tagged (`"op"` / `"event"`, snake_case) so the Python side dispatches
//! on a single discriminator field. Every request-bearing command (`ListTools`, `CallTool`, `Ping`)
//! carries a `request_id`; its reply (`Tools`, `Result`, `Pong`) echoes it so the client can route
//! concurrent in-flight calls. `Initialize`, `Cancel`, and `Shutdown` are fire-and-forget
//! notifications (no reply).
//!
//! [`Framing::Length`]: daemon_provision::Framing::Length

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

/// The protocol version this build speaks. Bumped on any breaking wire change; the worker reports
/// the version it implements in [`Event::Ready`] and the client warns on a mismatch.
pub const PROTOCOL_VERSION: u32 = 1;

/// A tool's batch-concurrency class, mirroring `daemon_core::tools::ToolConcurrency`. A worker
/// declares it per tool in its [`ToolManifest`]; the proxy maps it onto the engine's class so a
/// read-only Python tool can opt into concurrent batch execution.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Concurrency {
    /// Side-effect-free / read-only: safe to run alongside other parallel calls in a batch.
    Parallel,
    /// Must run alone (the default): the tool mutates state or has ordered side effects.
    #[default]
    Exclusive,
}

/// One tool the worker exposes, discovered via [`Command::ListTools`] / [`Event::Tools`].
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolManifest {
    /// The tool's stable name as exposed to the model.
    pub name: String,
    /// A human/model-facing description (defaults to the schema's `description`).
    #[serde(default)]
    pub description: String,
    /// The argument JSON-schema, as a JSON string (the daemon offers it to the model verbatim).
    pub schema: String,
    /// The tool's batch-concurrency class.
    #[serde(default)]
    pub concurrency: Concurrency,
    /// Whether this tool's results are external/untrusted by default (web/scrape/MCP-style). The
    /// §12 pipeline fences untrusted content before budgeting.
    #[serde(default)]
    pub untrusted: bool,
}

/// A structured result detail (the §17 `ToolResultView::detail` envelope). The `body` is arbitrary
/// JSON the GUI renders per `kind`; the proxy serializes it to bytes for `daemon_protocol::ToolDetail`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResultDetail {
    /// The stable renderer discriminator (e.g. the tool name).
    pub kind: String,
    /// The opaque structured payload (JSON), decoded by the consumer per `kind`.
    pub body: serde_json::Value,
}

/// A classified worker failure (mirrors the metta/infer taxonomy so the client maps it onto the
/// daemon failure taxonomy for supervision/recovery).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorClass {
    /// A bad request (undecodable args, unknown tool) — the caller must fix it.
    BadRequest,
    /// The op / tool is not supported by this worker build.
    Unsupported,
    /// A transient error — a retry (possibly on a fresh worker) may succeed.
    Transient,
    /// Unrecoverable: an internal worker bug / corrupt state.
    Fatal,
    /// The call was cancelled / timed out cooperatively.
    Cancelled,
}

/// A `parent -> worker` command frame.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Command {
    /// Notify the worker of the negotiated protocol version (fire-and-forget; no reply). Sent once
    /// after the worker's [`Event::Ready`], before discovery.
    Initialize {
        /// The protocol version the client speaks.
        protocol_version: u32,
    },
    /// Discover the worker's tools (answered with [`Event::Tools`]).
    ListTools {
        /// Correlates the reply.
        request_id: u64,
    },
    /// Invoke a tool (answered with [`Event::Result`] or [`Event::Error`]).
    CallTool {
        /// Correlates the reply.
        request_id: u64,
        /// The engine's call id (echoed back so the proxy pairs it with the originating call).
        call_id: String,
        /// The tool name.
        name: String,
        /// The (already repaired) argument payload as a JSON string.
        args: String,
        /// The calling session id (so a tool can scope per-session state).
        session_id: String,
        /// A cooperative wall-clock deadline in ms (`0` = the worker default / none).
        #[serde(default)]
        deadline_ms: u64,
    },
    /// Ask the worker to cancel an in-flight call (fire-and-forget; the worker still sends the
    /// call's [`Event::Result`]/[`Event::Error`], typically [`ErrorClass::Cancelled`]).
    Cancel {
        /// The call to cancel.
        call_id: String,
    },
    /// Liveness probe (answered with [`Event::Pong`]).
    Ping {
        /// Correlates the reply.
        request_id: u64,
    },
    /// Ask the worker to exit cleanly (fire-and-forget).
    Shutdown,
}

impl Command {
    /// The request id this command correlates on (`None` for the notifications).
    pub fn request_id(&self) -> Option<u64> {
        match self {
            Command::ListTools { request_id }
            | Command::CallTool { request_id, .. }
            | Command::Ping { request_id } => Some(*request_id),
            Command::Initialize { .. } | Command::Cancel { .. } | Command::Shutdown => None,
        }
    }

    /// Overwrite the correlating request id (the supervised client assigns ids centrally). A no-op
    /// for the notifications, which carry none.
    pub fn set_request_id(&mut self, id: u64) {
        match self {
            Command::ListTools { request_id }
            | Command::CallTool { request_id, .. }
            | Command::Ping { request_id } => *request_id = id,
            Command::Initialize { .. } | Command::Cancel { .. } | Command::Shutdown => {}
        }
    }
}

/// A `worker -> parent` event frame.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum Event {
    /// The worker is up; reports its identity, SDK version, and implemented protocol version. The
    /// unsolicited first frame after spawn.
    Ready {
        /// The worker identifier (e.g. `"daemon_pytool"`).
        worker: String,
        /// The worker SDK version string.
        sdk_version: String,
        /// The protocol version the worker implements.
        protocol_version: u32,
    },
    /// The reply to [`Command::ListTools`]: the worker's discovered tools.
    Tools {
        /// The request this answers.
        request_id: u64,
        /// The discovered tools.
        tools: Vec<ToolManifest>,
    },
    /// The reply to [`Command::CallTool`]: a tool result.
    Result {
        /// The request this answers.
        request_id: u64,
        /// The originating engine call id.
        call_id: String,
        /// Whether the tool succeeded.
        ok: bool,
        /// The textual result content.
        content: String,
        /// An optional structured detail envelope for a rich GUI consumer.
        #[serde(default)]
        detail: Option<ResultDetail>,
        /// Whether the content is external/untrusted (overrides the manifest default per-call).
        #[serde(default)]
        untrusted: bool,
    },
    /// A classified failure for `request_id` (or worker-level when `None`).
    Error {
        /// The request this fails (`None` = a worker-level fault failing every in-flight call).
        #[serde(default)]
        request_id: Option<u64>,
        /// The failure class.
        class: ErrorClass,
        /// A human-readable message.
        message: String,
    },
    /// Liveness reply to [`Command::Ping`].
    Pong {
        /// The ping this answers.
        request_id: u64,
    },
}

/// A JSON codec error.
#[derive(Debug, thiserror::Error)]
pub enum CodecError {
    /// Encoding a frame to JSON failed.
    #[error("json encode: {0}")]
    Encode(String),
    /// Decoding a frame from JSON failed.
    #[error("json decode: {0}")]
    Decode(String),
}

/// Encode a frame body to JSON bytes (the [`CutChannel`](daemon_provision::CutChannel) adds the
/// length prefix).
pub fn encode<T: Serialize>(value: &T) -> Result<Vec<u8>, CodecError> {
    serde_json::to_vec(value).map_err(|e| CodecError::Encode(e.to_string()))
}

/// Decode a JSON frame body.
pub fn decode<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, CodecError> {
    serde_json::from_slice(bytes).map_err(|e| CodecError::Decode(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip_command(cmd: Command) {
        let bytes = encode(&cmd).expect("encode command");
        let back: Command = decode(&bytes).expect("decode command");
        assert_eq!(cmd, back);
    }

    fn round_trip_event(ev: Event) {
        let bytes = encode(&ev).expect("encode event");
        let back: Event = decode(&bytes).expect("decode event");
        assert_eq!(ev, back);
    }

    #[test]
    fn commands_round_trip() {
        round_trip_command(Command::Initialize {
            protocol_version: PROTOCOL_VERSION,
        });
        round_trip_command(Command::ListTools { request_id: 1 });
        round_trip_command(Command::CallTool {
            request_id: 2,
            call_id: "c-1".into(),
            name: "py_echo".into(),
            args: r#"{"text":"hi"}"#.into(),
            session_id: "s-1".into(),
            deadline_ms: 0,
        });
        round_trip_command(Command::Cancel {
            call_id: "c-1".into(),
        });
        round_trip_command(Command::Ping { request_id: 3 });
        round_trip_command(Command::Shutdown);
    }

    #[test]
    fn events_round_trip() {
        round_trip_event(Event::Ready {
            worker: "daemon_pytool".into(),
            sdk_version: "0.1.0".into(),
            protocol_version: PROTOCOL_VERSION,
        });
        round_trip_event(Event::Tools {
            request_id: 1,
            tools: vec![ToolManifest {
                name: "py_echo".into(),
                description: "echo".into(),
                schema: r#"{"type":"object"}"#.into(),
                concurrency: Concurrency::Parallel,
                untrusted: false,
            }],
        });
        round_trip_event(Event::Result {
            request_id: 2,
            call_id: "c-1".into(),
            ok: true,
            content: "hi".into(),
            detail: Some(ResultDetail {
                kind: "py_echo".into(),
                body: serde_json::json!({"echoed": "hi"}),
            }),
            untrusted: false,
        });
        round_trip_event(Event::Error {
            request_id: Some(2),
            class: ErrorClass::BadRequest,
            message: "bad args".into(),
        });
        round_trip_event(Event::Pong { request_id: 3 });
    }

    /// The Python side dispatches on the `"op"` / `"event"` discriminator; pin the wire shape so a
    /// rename can't silently break the stdlib decoder.
    #[test]
    fn wire_tag_shape_is_stable() {
        let bytes = encode(&Command::Ping { request_id: 7 }).unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["op"], "ping");
        assert_eq!(v["request_id"], 7);

        let bytes = encode(&Event::Pong { request_id: 7 }).unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["event"], "pong");
        assert_eq!(v["request_id"], 7);
    }
}
