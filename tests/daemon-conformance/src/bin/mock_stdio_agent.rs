//! A minimal **foreign** agent: a non-`daemon-core` brain that speaks §17 over a process cut.
//!
//! It exists to prove the §17 leaf is a *universal agent-runner contract*: this binary has no
//! dependency on `daemon-core`, yet a `daemon-host` `ProcessAgentUnit` drives it as an ordinary
//! `Engine`-leaf `ManagedUnit`. It reads `Inbound` frames from stdin and, on a `StartTurn`,
//! emits a canned turn up its stdout:
//! `TurnStarted` -> `Usage` -> `ToolStarted{detail}` -> `ContentDelta` -> `ToolFinished{detail}` ->
//! `TextDelta` -> `TurnFinished{Completed}`. The structured tool `detail` and the `ContentDelta`
//! carry **opaque** payloads (kinds the daemon has never seen) to prove they survive the foreign
//! process -> CBOR cut -> host -> node-surface round-trip untouched. `Shutdown` ends the loop.

use daemon_common::UsageDelta;
use daemon_protocol::{
    AgentCommand, AgentEvent, EndReason, Inbound, Outbound, ToolCallView, ToolDetail,
    ToolResultView, TurnSummary, TurnTrigger,
};
use daemon_provision::CutChannel;

// Opaque structured payloads the daemon must pass through byte-for-byte (the test re-checks these).
/// The `kind` discriminator on the tool-result `detail` envelope.
pub const TOOL_DETAIL_KIND: &str = "search.results";
/// The opaque encoded body on the tool-result `detail` (a tool/GUI-private shape, not CBOR-parsed
/// by the daemon).
pub const TOOL_DETAIL_BODY: &[u8] = b"[{\"title\":\"Rust\",\"url\":\"https://rust-lang.org\"}]";
/// The reserved `kind` for a terminal/PTY stream carried via `ContentDelta`.
pub const CONTENT_KIND: &str = "ansi-stream";
/// The opaque terminal bytes (raw ANSI) carried via `ContentDelta`.
pub const CONTENT_BODY: &[u8] = b"\x1b[32m$ cargo build\x1b[0m\n   Compiling ok\n";

fn encode_up(frame: &Outbound) -> Vec<u8> {
    let mut buf = Vec::new();
    ciborium::into_writer(frame, &mut buf).expect("encode Outbound");
    buf
}

fn decode_down(bytes: &[u8]) -> Option<Inbound> {
    ciborium::from_reader(bytes).ok()
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let (writer, mut reader) = CutChannel::from_stdio().split();

    let mut seq = 0u64;
    let mut next = || {
        let s = seq;
        seq += 1;
        s
    };

    while let Some(bytes) = reader.recv().await {
        match decode_down(&bytes) {
            Some(Inbound::Command(AgentCommand::StartTurn { .. })) => {
                let frames = [
                    Outbound::Event(AgentEvent::TurnStarted {
                        seq: next(),
                        trigger: TurnTrigger::User,
                    }),
                    Outbound::Event(AgentEvent::Usage {
                        seq: next(),
                        delta: UsageDelta {
                            input_tokens: 7,
                            output_tokens: 3,
                            api_calls: 1,
                            ..Default::default()
                        },
                    }),
                    Outbound::Event(AgentEvent::ToolStarted {
                        seq: next(),
                        call: ToolCallView {
                            call_id: "call-1".into(),
                            name: "web_search".into(),
                            args_summary: "query=rustlang".into(),
                            // Opaque structured arguments object for a rich consumer.
                            detail: Some(ToolDetail::new(
                                "search.args",
                                b"{\"query\":\"rustlang\"}".to_vec(),
                            )),
                        },
                    }),
                    // Stream content not tied to a tool: a terminal/PTY chunk under a reserved kind.
                    Outbound::Event(AgentEvent::ContentDelta {
                        seq: next(),
                        kind: CONTENT_KIND.into(),
                        body: CONTENT_BODY.to_vec(),
                    }),
                    Outbound::Event(AgentEvent::ToolFinished {
                        seq: next(),
                        result: ToolResultView {
                            call_id: "call-1".into(),
                            ok: true,
                            summary: "1 result".into(),
                            // Opaque structured results payload (a kind the daemon never parses).
                            detail: Some(ToolDetail::new(
                                TOOL_DETAIL_KIND,
                                TOOL_DETAIL_BODY.to_vec(),
                            )),
                        },
                    }),
                    Outbound::Event(AgentEvent::TextDelta {
                        seq: next(),
                        text: "foreign agent reporting in".into(),
                    }),
                    Outbound::Event(AgentEvent::TurnFinished {
                        seq: next(),
                        summary: TurnSummary::ended(EndReason::Completed),
                    }),
                ];
                for frame in &frames {
                    if writer.send(&encode_up(frame)).await.is_err() {
                        return;
                    }
                }
            }
            Some(Inbound::Command(AgentCommand::Shutdown)) | None => return,
            _ => {}
        }
    }
}
