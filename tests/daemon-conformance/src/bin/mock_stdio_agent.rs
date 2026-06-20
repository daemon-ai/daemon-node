//! A minimal **foreign** agent: a non-`daemon-core` brain that speaks §17 over a process cut.
//!
//! It exists to prove the §17 leaf is a *universal agent-runner contract*: this binary has no
//! dependency on `daemon-core`, yet a `daemon-host` `ProcessAgentUnit` drives it as an ordinary
//! `Engine`-leaf `ManagedUnit`. It reads `Section17Down` frames from stdin and, on a `StartTurn`,
//! emits a canned turn (`TurnStarted` -> `Usage` -> `TextDelta` -> `TurnFinished{Completed}`) up its
//! stdout. `Shutdown` ends the loop.

use daemon_common::UsageDelta;
use daemon_protocol::{
    AgentCommand, AgentEvent, EndReason, Section17Down, Section17Up, TurnSummary, TurnTrigger,
};
use daemon_provision::CutChannel;

fn encode_up(frame: &Section17Up) -> Vec<u8> {
    let mut buf = Vec::new();
    ciborium::into_writer(frame, &mut buf).expect("encode Section17Up");
    buf
}

fn decode_down(bytes: &[u8]) -> Option<Section17Down> {
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
            Some(Section17Down::Command(AgentCommand::StartTurn { .. })) => {
                let frames = [
                    Section17Up::Event(AgentEvent::TurnStarted {
                        seq: next(),
                        trigger: TurnTrigger::User,
                    }),
                    Section17Up::Event(AgentEvent::Usage {
                        seq: next(),
                        delta: UsageDelta {
                            input_tokens: 7,
                            output_tokens: 3,
                            api_calls: 1,
                        },
                    }),
                    Section17Up::Event(AgentEvent::TextDelta {
                        seq: next(),
                        text: "foreign agent reporting in".into(),
                    }),
                    Section17Up::Event(AgentEvent::TurnFinished {
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
            Some(Section17Down::Command(AgentCommand::Shutdown)) | None => return,
            _ => {}
        }
    }
}
