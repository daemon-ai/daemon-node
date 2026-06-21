//! The worker wire protocol — engine-agnostic [`Command`]/[`Event`] frames + a CBOR codec.
//!
//! The daemon (`daemon-providers`' `LocalProvider`) and the `daemon-infer` worker exchange these
//! frames over a length-framed stdio cut ([`daemon_provision::CutChannel`], [`Framing::Length`]).
//! Each frame body is CBOR; the `u32`-length prefix is handled by the channel, so this module only
//! owns the body [`encode`]/[`decode`].
//!
//! These are standalone wire types (not `daemon-core`'s), so the protocol — and any crate that only
//! needs it (e.g. `daemon-providers` with `default-features = false`) — stays light: `serde` +
//! `ciborium` only, no engine and no `daemon-core` dependency.
//!
//! [`Framing::Length`]: daemon_provision::Framing::Length

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

/// Which local inference engine the worker hosts (selected at spawn via `--engine`, confirmed in
/// [`Command::Load`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Engine {
    /// llama.cpp via the `llama-cpp-4` bindings.
    Llama,
    /// mistral.rs via the `mistralrs` crate.
    MistralRs,
}

impl Engine {
    /// Parse the `--engine` flag value.
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "llama" | "llama-cpp" | "llamacpp" => Some(Engine::Llama),
            "mistralrs" | "mistral-rs" | "mistral.rs" => Some(Engine::MistralRs),
            _ => None,
        }
    }

    /// The canonical flag spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            Engine::Llama => "llama",
            Engine::MistralRs => "mistralrs",
        }
    }
}

/// How a model encodes tool calls (a mirror of `daemon_core::ToolCallFormat`, kept independent so
/// the protocol crate carries no `daemon-core` dependency).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToolCallFormat {
    /// Native structured tool calls.
    Native,
    /// Anthropic tool-use blocks.
    AnthropicToolUse,
    /// Hermes-style XML.
    HermesXml,
}

/// Declared backend/model capabilities, reported in [`Event::Ready`] after a successful load.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Capabilities {
    /// Whether the model emits native tool calls (vs. text the worker must parse).
    pub supports_native_tools: bool,
    /// Whether the backend streams tokens incrementally.
    pub supports_streaming: bool,
    /// The tool-call wire format the model expects.
    pub tool_call_format: ToolCallFormat,
    /// The model's maximum context window, if known.
    pub max_context: Option<u32>,
}

/// Model load parameters (the subset both engines understand; engine-specific knobs are optional).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelParams {
    /// llama.cpp: number of layers offloaded to the GPU (`0` = CPU only).
    pub n_gpu_layers: u32,
    /// The context window to allocate (`0` = the model's training default).
    pub n_ctx: u32,
    /// Threads used for generation/prompt processing (`None` = engine default).
    pub n_threads: Option<u32>,
    /// Enable Flash Attention where the backend supports it.
    pub flash_attn: bool,
    /// In-situ quantization spec for mistral.rs (e.g. `"Q8_0"`); `None` = load as-is.
    pub isq: Option<String>,
    /// Load the model in **embedding mode** (a pooled-embedding context) rather than for generation.
    /// A worker loaded this way answers [`Command::Embed`] and refuses [`Command::Generate`].
    pub embeddings: bool,
}

/// Token-sampling parameters for one [`Command::Generate`].
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Sampling {
    /// Softmax temperature.
    pub temperature: f32,
    /// Nucleus (top-p) cutoff.
    pub top_p: f32,
    /// Top-k cutoff (`0` = disabled).
    pub top_k: u32,
    /// RNG seed for reproducible sampling.
    pub seed: u64,
}

impl Default for Sampling {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_p: 0.95,
            top_k: 40,
            seed: 0,
        }
    }
}

/// A grammar constraint applied to one generation, bounding the model's output to a formal grammar.
///
/// Each engine accepts a different dialect, so the constraint carries both renderings and each
/// backend picks its own: mistral.rs consumes [`Constraint::lark`] via
/// `RequestBuilder::set_constraint`; llama.cpp consumes [`Constraint::gbnf`] (root rule `root`) via
/// `LlamaSampler::grammar`. A backend whose dialect is absent ignores the constraint (with a
/// warning) rather than failing the generation.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Constraint {
    /// A Lark grammar (mistral.rs / llguidance), if available.
    #[serde(default)]
    pub lark: Option<String>,
    /// A GBNF grammar (llama.cpp, root rule `root`), if available.
    #[serde(default)]
    pub gbnf: Option<String>,
}

/// One flattened conversation message (mirrors `daemon_core::RequestMsg`, preserving native
/// tool-call linkage so the worker can apply a faithful chat template).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Msg {
    /// The role: `system`, `user`, `assistant`, or `tool`.
    pub role: String,
    /// The message text (assistant/user text, or a tool result payload).
    pub content: String,
    /// For an `assistant` message: the tool calls it emitted.
    pub tool_calls: Vec<ToolCall>,
    /// For a `tool` message: the id of the call this result answers.
    pub tool_call_id: Option<String>,
}

/// A tool offered for the turn, with its JSON-Schema (used for chat-template tool advertisement and
/// grammar-constrained argument decoding).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolDef {
    /// The tool's stable name.
    pub name: String,
    /// The tool's JSON-Schema (as a string).
    pub schema: String,
}

/// A tool call produced by the model.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    /// Correlates the call with its result.
    pub call_id: String,
    /// The tool name.
    pub name: String,
    /// The (JSON) argument payload.
    pub args: String,
}

/// Token usage accrued by one generation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    /// Prompt (input) tokens.
    pub input_tokens: u64,
    /// Generated (output) tokens.
    pub output_tokens: u64,
}

/// A classified backend failure — maps onto the daemon's `Failure` taxonomy so the existing §8
/// recovery loop can decide retry/compact/abort without engine-specific knowledge.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorClass {
    /// The prompt exceeded the context window — the daemon compacts and retries (`ContextOverflow`).
    ContextOverflow,
    /// A GPU/host allocator OOM (VRAM exhausted) — the daemon retries then declares fatal
    /// (`ProviderOverloaded`).
    OutOfMemory,
    /// A transient/internal generation or decode error — retry (`TransientTransport`).
    Transient,
    /// Unrecoverable: no backend compiled, the model is unloadable, or an internal bug — abort
    /// (`Fatal`).
    Fatal,
    /// Generation was cancelled cooperatively (`Cancelled`).
    Cancelled,
}

/// A parent -> worker command frame.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Command {
    /// Load a model into the (already engine-selected) worker. The worker answers [`Event::Ready`]
    /// on success or [`Event::Error`] on failure.
    Load {
        /// The engine the worker was spawned for (a cross-check against `--engine`).
        engine: Engine,
        /// The model: a local GGUF path (llama) or a directory / Hugging Face id (mistral.rs).
        model: String,
        /// Load knobs.
        params: ModelParams,
    },
    /// Run one generation, streaming [`Event::TextDelta`]/[`Event::ToolCall`] then [`Event::Done`].
    Generate {
        /// Correlates events and a later [`Command::Cancel`] with this request.
        request_id: u64,
        /// The system prompt.
        system: String,
        /// The flattened conversation.
        messages: Vec<Msg>,
        /// The tools offered this turn.
        tools: Vec<ToolDef>,
        /// Sampling parameters.
        sampling: Sampling,
        /// The output-token cap (`0` = the worker's default).
        max_tokens: u32,
        /// An optional grammar constraint bounding the output (e.g. MeTTa). `None` = unconstrained.
        #[serde(default)]
        constraint: Option<Constraint>,
    },
    /// Embed a batch of texts against an embedding-mode model (loaded with
    /// [`ModelParams::embeddings`]). The worker answers [`Event::Embeddings`] or [`Event::Error`].
    Embed {
        /// Correlates the [`Event::Embeddings`]/[`Event::Error`] reply with this request.
        request_id: u64,
        /// The texts to embed (one vector is returned per text, in order).
        texts: Vec<String>,
    },
    /// Cancel an in-flight generation cooperatively.
    Cancel {
        /// The request to cancel.
        request_id: u64,
    },
    /// Ask the worker to exit cleanly.
    Shutdown,
    /// Liveness probe (answered with [`Event::Pong`]).
    Ping,
}

/// A worker -> parent event frame.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Event {
    /// The model loaded; reports its capabilities.
    Ready {
        /// The loaded model's capabilities.
        capabilities: Capabilities,
    },
    /// Incremental assistant text for `request_id`.
    TextDelta {
        /// The generation this delta belongs to.
        request_id: u64,
        /// The text chunk.
        text: String,
    },
    /// Incremental reasoning text for `request_id`.
    ReasoningDelta {
        /// The generation this delta belongs to.
        request_id: u64,
        /// The reasoning chunk.
        text: String,
    },
    /// A decoded tool call for `request_id`.
    ToolCall {
        /// The generation this call belongs to.
        request_id: u64,
        /// The tool call.
        call: ToolCall,
    },
    /// Generation finished; carries the final usage.
    Done {
        /// The generation that finished.
        request_id: u64,
        /// Token usage for the generation.
        usage: Usage,
    },
    /// The embeddings for a [`Command::Embed`] — one vector per input text, in order.
    Embeddings {
        /// The embed request these vectors answer.
        request_id: u64,
        /// One embedding vector per input text.
        vectors: Vec<Vec<f32>>,
        /// The embedding dimensionality (length of each vector).
        dims: u32,
    },
    /// A classified failure. `request_id` is `None` for load/worker-level errors.
    Error {
        /// The affected generation, if any.
        request_id: Option<u64>,
        /// The failure class (maps to the daemon's `Failure`).
        class: ErrorClass,
        /// A short human-readable detail.
        message: String,
    },
    /// Liveness reply to [`Command::Ping`].
    Pong,
    /// A health snapshot (emitted on demand / at startup for diagnostics).
    Health {
        /// The compiled backend identifier (e.g. `"llama"`, `"mistralrs"`, `"stub"`).
        backend: String,
        /// Whether a model is currently loaded.
        model_loaded: bool,
    },
}

/// A CBOR codec error.
#[derive(Debug, thiserror::Error)]
pub enum CodecError {
    /// Encoding a frame to CBOR failed.
    #[error("cbor encode: {0}")]
    Encode(String),
    /// Decoding a frame from CBOR failed.
    #[error("cbor decode: {0}")]
    Decode(String),
}

/// Encode a frame body to CBOR bytes (the [`CutChannel`] adds the length prefix).
///
/// [`CutChannel`]: daemon_provision::CutChannel
pub fn encode<T: Serialize>(value: &T) -> Result<Vec<u8>, CodecError> {
    let mut buf = Vec::new();
    ciborium::into_writer(value, &mut buf).map_err(|e| CodecError::Encode(e.to_string()))?;
    Ok(buf)
}

/// Decode a CBOR frame body.
pub fn decode<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, CodecError> {
    ciborium::from_reader(bytes).map_err(|e| CodecError::Decode(e.to_string()))
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
        round_trip_command(Command::Load {
            engine: Engine::Llama,
            model: "/models/llama-3.2-3b-q4.gguf".into(),
            params: ModelParams {
                n_gpu_layers: 99,
                n_ctx: 8192,
                n_threads: Some(8),
                flash_attn: true,
                isq: None,
                embeddings: false,
            },
        });
        round_trip_command(Command::Generate {
            request_id: 7,
            system: "you are helpful".into(),
            messages: vec![
                Msg {
                    role: "user".into(),
                    content: "hi".into(),
                    ..Default::default()
                },
                Msg {
                    role: "assistant".into(),
                    content: String::new(),
                    tool_calls: vec![ToolCall {
                        call_id: "call-0".into(),
                        name: "read_file".into(),
                        args: r#"{"path":"a"}"#.into(),
                    }],
                    tool_call_id: None,
                },
                Msg {
                    role: "tool".into(),
                    content: "contents".into(),
                    tool_calls: Vec::new(),
                    tool_call_id: Some("call-0".into()),
                },
            ],
            tools: vec![ToolDef {
                name: "read_file".into(),
                schema: r#"{"type":"object"}"#.into(),
            }],
            sampling: Sampling::default(),
            max_tokens: 512,
            constraint: Some(Constraint {
                gbnf: Some("root ::= \"a\"".into()),
                lark: None,
            }),
        });
        round_trip_command(Command::Embed {
            request_id: 9,
            texts: vec!["hello".into(), "world".into()],
        });
        round_trip_command(Command::Cancel { request_id: 7 });
        round_trip_command(Command::Shutdown);
        round_trip_command(Command::Ping);
    }

    #[test]
    fn events_round_trip() {
        round_trip_event(Event::Ready {
            capabilities: Capabilities {
                supports_native_tools: false,
                supports_streaming: true,
                tool_call_format: ToolCallFormat::HermesXml,
                max_context: Some(8192),
            },
        });
        round_trip_event(Event::TextDelta {
            request_id: 1,
            text: "hel".into(),
        });
        round_trip_event(Event::ReasoningDelta {
            request_id: 1,
            text: "thinking".into(),
        });
        round_trip_event(Event::ToolCall {
            request_id: 1,
            call: ToolCall {
                call_id: "call-1".into(),
                name: "shell".into(),
                args: r#"{"cmd":"ls"}"#.into(),
            },
        });
        round_trip_event(Event::Done {
            request_id: 1,
            usage: Usage {
                input_tokens: 10,
                output_tokens: 20,
            },
        });
        round_trip_event(Event::Embeddings {
            request_id: 1,
            vectors: vec![vec![0.5, -0.25, 0.0], vec![0.125, 0.75, -1.0]],
            dims: 3,
        });
        for class in [
            ErrorClass::ContextOverflow,
            ErrorClass::OutOfMemory,
            ErrorClass::Transient,
            ErrorClass::Fatal,
            ErrorClass::Cancelled,
        ] {
            round_trip_event(Event::Error {
                request_id: Some(1),
                class,
                message: "boom".into(),
            });
        }
        round_trip_event(Event::Pong);
        round_trip_event(Event::Health {
            backend: "stub".into(),
            model_loaded: false,
        });
    }

    #[test]
    fn engine_parse_roundtrips() {
        assert_eq!(Engine::parse("llama"), Some(Engine::Llama));
        assert_eq!(Engine::parse("MistralRS"), Some(Engine::MistralRs));
        assert_eq!(Engine::parse("nope"), None);
        assert_eq!(Engine::parse(Engine::Llama.as_str()), Some(Engine::Llama));
        assert_eq!(
            Engine::parse(Engine::MistralRs.as_str()),
            Some(Engine::MistralRs)
        );
    }
}
