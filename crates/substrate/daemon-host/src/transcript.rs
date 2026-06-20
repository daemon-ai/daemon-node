//! Coalescing the fine-grained §17 stream into finished transcript blocks for the verifiable
//! journal.
//!
//! The live drain ([`daemon_api::Outbound`]) carries every streaming fragment — text deltas,
//! reasoning deltas, raw content chunks, tool start/finish, host requests, lifecycle. The durable
//! journal must record only **finished** units (host-spec: "we care about the final message, the
//! finished blocks"), so a [`BlockCoalescer`] folds the stream: text deltas accumulate into a
//! message; a tool boundary flushes the message and emits tool blocks; content chunks of one kind
//! concatenate into a content block; a host request becomes a request block; and a turn boundary
//! flushes everything then signals a **seal**. Reasoning deltas and per-fragment usage/rate-limit
//! deltas are dropped from the durable journal (they remain on the live drain).
//!
//! The coalescer is a pure state machine returning [`JournalAction`]s; the production wiring applies
//! them to a [`crate::journal::JournalSink`] (append management/blocks, seal at turn end).

use daemon_protocol::{AgentEvent, Outbound, TranscriptBlock, TranscriptRole};

/// One action the coalescer asks the journal to take, in stream order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum JournalAction {
    /// Append a coarse management record (lifecycle).
    Management {
        /// The kind label (`mgmt.*`).
        kind: String,
        /// Human/structured detail.
        detail: String,
    },
    /// Append a finished chat block.
    Block(TranscriptBlock),
    /// Seal the open segment (a turn boundary): the prior actions form one signed segment.
    Seal,
}

/// Folds the §17 stream into finished [`TranscriptBlock`]s + lifecycle management records, sealing
/// at turn boundaries. One per journaled stream; driven by the rich `Outbound` tap.
#[derive(Default)]
pub struct BlockCoalescer {
    /// Assistant text accumulated from `TextDelta`s, flushed at a tool/turn boundary.
    pending_text: String,
    /// Opaque structured content accumulated from `ContentDelta`s of one kind, flushed at a
    /// boundary or when the kind changes.
    pending_content: Option<(String, Vec<u8>)>,
}

impl BlockCoalescer {
    /// A fresh coalescer with no pending blocks.
    pub fn new() -> Self {
        Self::default()
    }

    /// Fold one upbound frame, returning the journal actions it produces (often none until a
    /// boundary flushes a finished block).
    pub fn push(&mut self, frame: &Outbound) -> Vec<JournalAction> {
        match frame {
            Outbound::Event(event) => self.push_event(event),
            Outbound::Request(req) => {
                // A blocking host request is a finished item the moment it is raised.
                let mut out = self.flush();
                out.push(JournalAction::Block(TranscriptBlock::Request {
                    request_id: req.request_id,
                    kind: req.kind.clone(),
                }));
                out
            }
            // `Outbound` is `#[non_exhaustive]`; a future frame is not journaled until handled.
            _ => Vec::new(),
        }
    }

    fn push_event(&mut self, event: &AgentEvent) -> Vec<JournalAction> {
        match event {
            AgentEvent::TurnStarted { trigger, .. } => {
                let mut out = self.flush();
                out.push(JournalAction::Management {
                    kind: "mgmt.turn_started".into(),
                    detail: format!("{trigger:?}"),
                });
                out
            }
            AgentEvent::TextDelta { text, .. } => {
                self.pending_text.push_str(text);
                Vec::new()
            }
            // Reasoning is never journaled (host-spec §17.2 scrubbing).
            AgentEvent::ReasoningDelta { .. } => Vec::new(),
            AgentEvent::ContentDelta { kind, body, .. } => {
                let mut out = Vec::new();
                match &mut self.pending_content {
                    Some((pending_kind, buf)) if pending_kind == kind => buf.extend_from_slice(body),
                    _ => {
                        // A new content kind: flush the previous content block first.
                        if let Some(block) = self.take_content() {
                            out.push(JournalAction::Block(block));
                        }
                        self.pending_content = Some((kind.clone(), body.clone()));
                    }
                }
                out
            }
            AgentEvent::ToolStarted { call, .. } => {
                let mut out = self.flush();
                out.push(JournalAction::Block(TranscriptBlock::ToolCall {
                    call_id: call.call_id.clone(),
                    name: call.name.clone(),
                    args_summary: call.args_summary.clone(),
                    detail: call.detail.clone(),
                }));
                out
            }
            AgentEvent::ToolFinished { result, .. } => {
                vec![JournalAction::Block(TranscriptBlock::ToolResult {
                    call_id: result.call_id.clone(),
                    ok: result.ok,
                    summary: result.summary.clone(),
                    detail: result.detail.clone(),
                })]
            }
            // Per-fragment usage / rate-limit deltas are aggregated into the turn summary; not
            // journaled individually (kept on the live drain).
            AgentEvent::Usage { .. } | AgentEvent::RateLimit { .. } => Vec::new(),
            AgentEvent::Steered {
                request_id,
                accepted,
                ..
            } => vec![JournalAction::Management {
                kind: "mgmt.steered".into(),
                detail: format!("request_id={request_id:?} accepted={accepted}"),
            }],
            // The snapshot reply is a read-only projection, not history.
            AgentEvent::Snapshot { .. } => Vec::new(),
            AgentEvent::TurnFinished { summary, .. } => {
                // If the turn carried only a final text (no deltas), surface it as the message.
                if self.pending_text.is_empty() {
                    if let Some(text) = &summary.final_text {
                        self.pending_text = text.clone();
                    }
                }
                let mut out = self.flush();
                out.push(JournalAction::Management {
                    kind: "mgmt.turn_finished".into(),
                    detail: format!(
                        "end_reason={:?} usage={:?}",
                        summary.end_reason, summary.usage
                    ),
                });
                out.push(JournalAction::Seal);
                out
            }
            AgentEvent::Error { failure, .. } => {
                let mut out = self.flush();
                out.push(JournalAction::Management {
                    kind: "mgmt.error".into(),
                    detail: failure.clone(),
                });
                out.push(JournalAction::Seal);
                out
            }
            // `AgentEvent` is `#[non_exhaustive]`; an unknown future event is dropped from the
            // durable journal (it still rides the live drain).
            _ => Vec::new(),
        }
    }

    /// Flush any pending message + content blocks (a boundary was reached).
    fn flush(&mut self) -> Vec<JournalAction> {
        let mut out = Vec::new();
        if !self.pending_text.is_empty() {
            let text = std::mem::take(&mut self.pending_text);
            out.push(JournalAction::Block(TranscriptBlock::Message {
                role: TranscriptRole::Assistant,
                text,
            }));
        }
        if let Some(block) = self.take_content() {
            out.push(JournalAction::Block(block));
        }
        out
    }

    fn take_content(&mut self) -> Option<TranscriptBlock> {
        self.pending_content
            .take()
            .map(|(kind, body)| TranscriptBlock::Content { kind, body })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use daemon_common::ReqId;
    use daemon_protocol::{
        EndReason, HostRequest, HostRequestKind, ToolCallView, ToolResultView, TurnSummary,
        TurnTrigger,
    };

    fn ev(e: AgentEvent) -> Outbound {
        Outbound::Event(e)
    }

    #[test]
    fn text_deltas_coalesce_into_one_message_at_turn_end() {
        let mut c = BlockCoalescer::new();
        assert!(c
            .push(&ev(AgentEvent::TurnStarted { seq: 0, trigger: TurnTrigger::User }))
            .iter()
            .any(|a| matches!(a, JournalAction::Management { .. })));
        assert!(c.push(&ev(AgentEvent::TextDelta { seq: 1, text: "Hello ".into() })).is_empty());
        assert!(c.push(&ev(AgentEvent::TextDelta { seq: 2, text: "world".into() })).is_empty());
        let out = c.push(&ev(AgentEvent::TurnFinished {
            seq: 3,
            summary: TurnSummary::ended(EndReason::Completed),
        }));
        // message block, then a turn_finished management record, then a seal.
        assert!(matches!(
            &out[0],
            JournalAction::Block(TranscriptBlock::Message { text, .. }) if text == "Hello world"
        ));
        assert!(matches!(out[1], JournalAction::Management { .. }));
        assert_eq!(out[2], JournalAction::Seal);
    }

    #[test]
    fn tool_boundary_flushes_message_then_emits_tool_blocks() {
        let mut c = BlockCoalescer::new();
        c.push(&ev(AgentEvent::TextDelta { seq: 0, text: "thinking".into() }));
        let started = c.push(&ev(AgentEvent::ToolStarted {
            seq: 1,
            call: ToolCallView {
                call_id: "c1".into(),
                name: "search".into(),
                args_summary: "q=rust".into(),
                detail: None,
            },
        }));
        assert!(matches!(&started[0], JournalAction::Block(TranscriptBlock::Message { .. })));
        assert!(matches!(&started[1], JournalAction::Block(TranscriptBlock::ToolCall { .. })));
        let finished = c.push(&ev(AgentEvent::ToolFinished {
            seq: 2,
            result: ToolResultView {
                call_id: "c1".into(),
                ok: true,
                summary: "3 hits".into(),
                detail: None,
            },
        }));
        assert!(matches!(&finished[0], JournalAction::Block(TranscriptBlock::ToolResult { ok: true, .. })));
    }

    #[test]
    fn content_deltas_concatenate_into_one_block() {
        let mut c = BlockCoalescer::new();
        c.push(&ev(AgentEvent::ContentDelta { seq: 0, kind: "pty".into(), body: b"foo".to_vec() }));
        c.push(&ev(AgentEvent::ContentDelta { seq: 1, kind: "pty".into(), body: b"bar".to_vec() }));
        let out = c.push(&ev(AgentEvent::TurnFinished {
            seq: 2,
            summary: TurnSummary::ended(EndReason::Completed),
        }));
        assert!(out.iter().any(|a| matches!(
            a,
            JournalAction::Block(TranscriptBlock::Content { kind, body }) if kind == "pty" && body == b"foobar"
        )));
    }

    #[test]
    fn reasoning_is_dropped() {
        let mut c = BlockCoalescer::new();
        assert!(c
            .push(&ev(AgentEvent::ReasoningDelta { seq: 0, text: "secret".into() }))
            .is_empty());
    }

    #[test]
    fn host_request_becomes_a_request_block() {
        let mut c = BlockCoalescer::new();
        let out = c.push(&Outbound::Request(HostRequest {
            request_id: ReqId(7),
            kind: HostRequestKind::Input { prompt: "name?".into() },
        }));
        assert!(matches!(
            &out[0],
            JournalAction::Block(TranscriptBlock::Request { request_id, .. }) if *request_id == ReqId(7)
        ));
    }
}
