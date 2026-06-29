// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The foreign-agent codec seam: one generic session driver over any line/length transport.
//!
//! A foreign brain (a CLI agent we did not write) speaks some protocol over its stdio that is *not*
//! the in-process typed channel `daemon-core` uses. Every such protocol reduces to the same shape:
//! bytes arrive framed on the agent's stdout and must be translated into §17 [`Outbound`] frames
//! (events up, blocking requests up); §17 [`Inbound`] frames (commands down, request replies down)
//! must be translated back into bytes on the agent's stdin. That translation is the **only** thing
//! that varies per protocol — so it is captured in one trait, [`Codec`], and driven by one reusable
//! [`CodecSession`].
//!
//! The framing (length-prefixed vs newline-delimited) is a runtime property of the
//! [`CutChannel`](daemon_provision::CutChannel) ([`daemon_provision::Framing`]), so the driver is
//! generic only over the [`Codec`]; the same driver runs over either transport. The native CBOR cut
//! our own placed `daemon-core` children speak is just the first codec ([`NativeCutCodec`]); the
//! Claude-Code stream-json codec ([`crate::streamjson::StreamJsonCodec`]) is the second.

use crate::agent_session::AgentSession;
use async_trait::async_trait;
use daemon_protocol::{AgentCommand, AgentEvent, HostRequestHandler, Inbound, Outbound};
use daemon_provision::{ChildGuard, CutChannel, CutWriter};
use std::sync::{Arc, Mutex};
use tokio::sync::broadcast;

/// Translates between a foreign agent's on-wire bytes and the §17 [`Inbound`]/[`Outbound`] frames.
///
/// One message in / out at a time, but each call may fan out to zero-or-more frames (a single
/// transport line can carry several content blocks; a single §17 command may need several lines).
/// `&mut self` lets a codec carry state across calls (monotonic event `seq`, in-flight tool calls,
/// a prompt-turn counter). Implementations live in `daemon-host` ([`NativeCutCodec`],
/// [`crate::streamjson::StreamJsonCodec`]); the driver never inspects the protocol itself.
pub trait Codec: Send + 'static {
    /// Translate one received transport message into zero or more §17 [`Outbound`] frames. A
    /// malformed / unrecognized message yields an empty `Vec` (forward-compatible: ignore it).
    fn decode(&mut self, msg: &[u8]) -> Vec<Outbound>;

    /// Translate one §17 [`Inbound`] frame into zero or more transport messages to write back to
    /// the agent (each is framed by the channel). Empty when the frame has no on-wire effect.
    fn encode(&mut self, inbound: Inbound) -> Vec<Vec<u8>>;
}

/// A [`AgentSession`] over a foreign agent process, generic over its wire [`Codec`].
///
/// Owns the single reader task (recv → `decode` → events to the broadcast / blocking requests
/// through the [`HostRequestHandler`], whose replies are `encode`d back down) and retains the writer
/// for [`AgentSession::submit`]. The codec is shared (a `std::sync::Mutex`) between the reader
/// task and `submit`; its critical sections are pure CPU (no `.await` held across the lock).
pub struct CodecSession<C: Codec> {
    writer: CutWriter,
    codec: Arc<Mutex<C>>,
    events: broadcast::Sender<AgentEvent>,
    /// Owns the child process (when placed over a real cut); killed on drop so a unit never leaks an
    /// OS process. `None` when driven over an in-memory channel (tests).
    _child: Option<ChildGuard>,
}

impl<C: Codec> CodecSession<C> {
    /// Start pumping a foreign agent over `channel` with `codec`: spawn the reader task and retain
    /// the writer for `submit`. `child`, when present, is the owned OS process (killed on drop).
    pub fn from_channel(
        channel: CutChannel,
        child: Option<ChildGuard>,
        host: Arc<dyn HostRequestHandler>,
        codec: C,
    ) -> Self {
        let (writer, mut reader) = channel.split();
        let (events, _) = broadcast::channel::<AgentEvent>(256);
        let codec = Arc::new(Mutex::new(codec));

        let events_relay = events.clone();
        let reply_writer = writer.clone();
        let codec_task = codec.clone();
        tokio::spawn(async move {
            while let Some(bytes) = reader.recv().await {
                // Decode under the lock (pure CPU), release before awaiting the host / the writer.
                let frames = codec_task.lock().unwrap().decode(&bytes);
                for frame in frames {
                    match frame {
                        Outbound::Event(ev) => {
                            let _ = events_relay.send(ev);
                        }
                        Outbound::Request(req) => {
                            let resp = host.request(req).await;
                            let replies =
                                codec_task.lock().unwrap().encode(Inbound::Response(resp));
                            for reply in replies {
                                let _ = reply_writer.send(&reply).await;
                            }
                        }
                        // `Outbound` is `#[non_exhaustive]`; a future variant has no relay here.
                        _ => {}
                    }
                }
            }
        });

        Self {
            writer,
            codec,
            events,
            _child: child,
        }
    }
}

#[async_trait]
impl<C: Codec> AgentSession for CodecSession<C> {
    async fn submit(&self, cmd: AgentCommand) {
        let frames = self.codec.lock().unwrap().encode(Inbound::Command(cmd));
        for frame in frames {
            let _ = self.writer.send(&frame).await;
        }
    }

    fn subscribe(&self) -> broadcast::Receiver<AgentEvent> {
        self.events.subscribe()
    }
}

/// The native `daemon` cut codec: CBOR-encoded §17 frames over the length-framed transport. This is
/// the dialect our own placed `daemon-core` children speak (a §17 brain driven across a process
/// cut); it carries the §17 frames verbatim, so `decode`/`encode` are a single CBOR (de)serialize.
#[derive(Default)]
pub struct NativeCutCodec;

impl Codec for NativeCutCodec {
    fn decode(&mut self, msg: &[u8]) -> Vec<Outbound> {
        decode_outbound(msg).into_iter().collect()
    }

    fn encode(&mut self, inbound: Inbound) -> Vec<Vec<u8>> {
        vec![encode_inbound(&inbound)]
    }
}

/// Encode an [`Inbound`] frame (CBOR). Frame types are always serializable; a failure is a bug.
pub fn encode_inbound(frame: &Inbound) -> Vec<u8> {
    let mut buf = Vec::new();
    ciborium::into_writer(frame, &mut buf).expect("encode Inbound");
    buf
}

/// Decode an [`Outbound`] frame (CBOR); `None` on a malformed frame.
pub fn decode_outbound(bytes: &[u8]) -> Option<Outbound> {
    ciborium::from_reader(bytes).ok()
}
