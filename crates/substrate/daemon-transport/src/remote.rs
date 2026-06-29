// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The `remote` transport: a length-framed TCP socket carrying [`Wire`] envelopes.

use std::sync::Arc;
use std::time::Duration;

use daemon_common::{
    Budget, Epoch, FenceToken, ReqId, SessionId, SnapshotBlob, TraceId, WireVersion,
};
use daemon_store::{Checkpoint, SessionStatus, SessionStore, StoreErrorWire};
use daemon_supervision::{Ack, ManageCommand, ManageEvent, ManagedUnit, WorkRef};
use daemon_telemetry::{current_trace, fields, restore_trace_span, with_trace, SpanKind};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::tcp::{OwnedReadHalf, OwnedWriteHalf};
use tokio::net::{TcpListener, TcpStream, ToSocketAddrs};

/// The on-wire envelope: a versioned frame carrying the sender's trace context and a body.
#[derive(Debug, Serialize, Deserialize)]
struct Wire<B> {
    wire_version: WireVersion,
    trace: TraceId,
    body: B,
}

/// A client→server request.
#[derive(Debug, Serialize, Deserialize)]
enum Req {
    /// Version handshake.
    Hello { version: WireVersion },
    /// Drive the server's hosted unit through one turn with the given inline work.
    Drive { work: String },
    /// Cross-node lease: acquire a fresh fencing token for `session`.
    AcquireFence { session: SessionId },
    /// Cross-node commit: mark `session` completed under `fence` (fenced by the authority).
    Commit {
        session: SessionId,
        epoch: Epoch,
        fence: FenceToken,
    },
    /// Read a session's durable status.
    Status { session: SessionId },
}

fn req_kind(req: &Req) -> &'static str {
    match req {
        Req::Hello { .. } => "Hello",
        Req::Drive { .. } => "Drive",
        Req::AcquireFence { .. } => "AcquireFence",
        Req::Commit { .. } => "Commit",
        Req::Status { .. } => "Status",
    }
}

/// A server→client reply.
#[derive(Debug, Serialize, Deserialize)]
enum Resp {
    /// Handshake result.
    Hello { version: WireVersion, ok: bool },
    /// The hosted unit was driven; carries the terminal reason and the trace the server *restored*
    /// from the request (the round-trip proof).
    Driven {
        ok: bool,
        end_reason: String,
        observed_trace: TraceId,
    },
    /// An `AcquireFence` result.
    Fence(Result<FenceToken, StoreErrorWire>),
    /// A `Commit` result (the cross-node fence verdict).
    Commit(Result<(), StoreErrorWire>),
    /// A `Status` result.
    Status(Option<SessionStatus>),
}

/// The outcome of driving a remote unit over the socket.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DriveOutcome {
    /// Whether the unit reached a successful terminal state.
    pub ok: bool,
    /// The terminal reason (debug-rendered `EndReason`, or an error label).
    pub end_reason: String,
    /// The trace the server restored from the request — equals the client's trace when propagation
    /// holds across the socket.
    pub observed_trace: TraceId,
}

/// Errors from the transport client.
#[derive(Debug, thiserror::Error)]
pub enum TransportError {
    /// Underlying socket I/O error.
    #[error("transport io: {0}")]
    Io(#[from] std::io::Error),
    /// The peer closed the connection before replying.
    #[error("transport closed by peer")]
    Closed,
    /// The peer sent an unexpected reply for the request.
    #[error("unexpected reply on transport")]
    Protocol,
    /// The authoritative store rejected the operation (notably a stale remote fence).
    #[error("remote store rejected: {0:?}")]
    Store(StoreErrorWire),
}

async fn write_frame<W, B>(w: &mut W, frame: &Wire<B>) -> std::io::Result<()>
where
    W: AsyncWriteExt + Unpin,
    B: Serialize,
{
    let mut buf = Vec::new();
    ciborium::into_writer(frame, &mut buf)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
    w.write_all(&(buf.len() as u32).to_be_bytes()).await?;
    w.write_all(&buf).await?;
    w.flush().await?;
    Ok(())
}

async fn read_frame<R, B>(r: &mut R) -> std::io::Result<Option<Wire<B>>>
where
    R: AsyncReadExt + Unpin,
    B: DeserializeOwned,
{
    let mut len = [0u8; 4];
    match r.read_exact(&mut len).await {
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e),
    }
    let n = u32::from_be_bytes(len) as usize;
    let mut buf = vec![0u8; n];
    r.read_exact(&mut buf).await?;
    let frame = ciborium::from_reader(&buf[..])
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
    Ok(Some(frame))
}

// ---------------------------------------------------------------------------
// Server
// ---------------------------------------------------------------------------

/// The transport server: holds the authoritative store and a hosted unit, and serves the wire
/// protocol (handshake, drive-unit, cross-node fence) over accepted connections.
pub struct RemoteHost {
    store: Arc<dyn SessionStore>,
    unit: Arc<dyn ManagedUnit>,
    drive_timeout: Duration,
}

impl RemoteHost {
    /// Build a server over an authoritative `store` and a hosted `unit`.
    pub fn new(store: Arc<dyn SessionStore>, unit: Arc<dyn ManagedUnit>) -> Self {
        Self {
            store,
            unit,
            drive_timeout: Duration::from_secs(10),
        }
    }

    /// Accept connections on `listener` until it errors or is dropped, serving each concurrently.
    pub async fn serve(self: Arc<Self>, listener: TcpListener) -> std::io::Result<()> {
        loop {
            let (stream, _peer) = listener.accept().await?;
            let me = self.clone();
            tokio::spawn(async move {
                let _ = me.handle(stream).await;
            });
        }
    }

    /// Serve a single connection (used directly in tests).
    pub async fn handle(&self, stream: TcpStream) -> std::io::Result<()> {
        let (mut r, mut w) = stream.into_split();
        // A trace scope so `set_trace` (restore-on-decode) governs the replies we stamp back.
        with_trace(TraceId::NONE, async move {
            while let Some(frame) = read_frame::<OwnedReadHalf, Req>(&mut r).await? {
                let operation = req_kind(&frame.body);
                let span = restore_trace_span(
                    frame.trace,
                    fields::span::TRANSPORT_REQUEST,
                    SpanKind::Transport,
                );
                let _guard = span.enter();
                tracing::debug!(
                    trace_id = %frame.trace,
                    wire = "tcp",
                    operation,
                    event = fields::event::TRANSPORT_REQUEST,
                    "transport request received"
                );
                let body = self.dispatch(frame.body).await;
                let reply = Wire {
                    wire_version: WireVersion::CURRENT,
                    trace: current_trace(),
                    body,
                };
                write_frame::<OwnedWriteHalf, Resp>(&mut w, &reply).await?;
            }
            Ok(())
        })
        .await
    }

    async fn dispatch(&self, req: Req) -> Resp {
        match req {
            Req::Hello { version } => Resp::Hello {
                version: WireVersion::CURRENT,
                ok: WireVersion::CURRENT.is_compatible(&version),
            },
            Req::Drive { work } => {
                // The trace was just restored from the request; the unit runs under it.
                let observed = current_trace();
                let (ok, end_reason) = self.drive_unit(&work).await;
                Resp::Driven {
                    ok,
                    end_reason,
                    observed_trace: observed,
                }
            }
            Req::AcquireFence { session } => Resp::Fence(
                self.store
                    .acquire_activation_lease(&session)
                    .await
                    .map_err(|e| StoreErrorWire::from(&e)),
            ),
            Req::Commit {
                session,
                epoch,
                fence,
            } => {
                let checkpoint = Checkpoint::new(session, epoch, SnapshotBlob::default());
                Resp::Commit(
                    self.store
                        .mark_completed(checkpoint, fence)
                        .await
                        .map_err(|e| StoreErrorWire::from(&e)),
                )
            }
            Req::Status { session } => Resp::Status(self.store.status(&session).await),
        }
    }

    /// Drive the hosted unit through one assigned turn, returning `(ok, terminal_reason)`.
    async fn drive_unit(&self, work: &str) -> (bool, String) {
        let mut events = self.unit.events();
        let ack = self
            .unit
            .command(ManageCommand::Assign {
                request_id: ReqId(1),
                work: WorkRef::inline("remote-w1", work),
                budget: Budget::unlimited(),
            })
            .await;
        if ack != Ack::Accepted {
            return (false, format!("rejected: {ack:?}"));
        }
        loop {
            match tokio::time::timeout(self.drive_timeout, events.recv()).await {
                Ok(Ok(ManageEvent::Finished { outcome, .. })) => {
                    return (true, format!("{:?}", outcome.end_reason));
                }
                Ok(Ok(ManageEvent::Error { failure, .. })) => {
                    return (false, format!("error: {failure:?}"));
                }
                Ok(Ok(_)) => continue,
                Ok(Err(_)) => return (false, "event stream closed".into()),
                Err(_) => return (false, "drive timed out".into()),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

/// A sequential request/reply client over the transport. Each call stamps the current trace onto
/// the request and restores the peer's trace from the reply (elfo network path).
pub struct RemoteClient {
    reader: OwnedReadHalf,
    writer: OwnedWriteHalf,
}

impl RemoteClient {
    /// Connect to a [`RemoteHost`] at `addr`.
    pub async fn connect(addr: impl ToSocketAddrs) -> Result<Self, TransportError> {
        let stream = TcpStream::connect(addr).await?;
        let (reader, writer) = stream.into_split();
        Ok(Self { reader, writer })
    }

    async fn call(&mut self, body: Req) -> Result<Resp, TransportError> {
        let frame = Wire {
            wire_version: WireVersion::CURRENT,
            trace: current_trace(),
            body,
        };
        write_frame::<OwnedWriteHalf, Req>(&mut self.writer, &frame).await?;
        let reply = read_frame::<OwnedReadHalf, Resp>(&mut self.reader)
            .await?
            .ok_or(TransportError::Closed)?;
        // Restore the peer's trace context (so the journal/logs here correlate with the server).
        let span = restore_trace_span(
            reply.trace,
            fields::span::TRANSPORT_REPLY,
            SpanKind::Transport,
        );
        let _guard = span.enter();
        tracing::debug!(
            trace_id = %reply.trace,
            wire = "tcp",
            event = fields::event::TRANSPORT_REPLY,
            "transport reply received"
        );
        Ok(reply.body)
    }

    /// Perform the version handshake; returns whether the peer is compatible.
    pub async fn hello(&mut self) -> Result<bool, TransportError> {
        match self
            .call(Req::Hello {
                version: WireVersion::CURRENT,
            })
            .await?
        {
            Resp::Hello { ok, .. } => Ok(ok),
            _ => Err(TransportError::Protocol),
        }
    }

    /// Drive the server's hosted unit through one turn over the socket.
    pub async fn drive(&mut self, work: &str) -> Result<DriveOutcome, TransportError> {
        match self.call(Req::Drive { work: work.into() }).await? {
            Resp::Driven {
                ok,
                end_reason,
                observed_trace,
            } => Ok(DriveOutcome {
                ok,
                end_reason,
                observed_trace,
            }),
            _ => Err(TransportError::Protocol),
        }
    }

    /// Acquire a fresh fencing token for `session` from the remote authority (cross-node lease).
    pub async fn acquire_fence(
        &mut self,
        session: &SessionId,
    ) -> Result<FenceToken, TransportError> {
        match self
            .call(Req::AcquireFence {
                session: session.clone(),
            })
            .await?
        {
            Resp::Fence(Ok(token)) => Ok(token),
            Resp::Fence(Err(e)) => Err(TransportError::Store(e)),
            _ => Err(TransportError::Protocol),
        }
    }

    /// Commit `session` completed under `fence` at the remote authority (cross-node fence check).
    pub async fn commit(
        &mut self,
        session: &SessionId,
        epoch: Epoch,
        fence: FenceToken,
    ) -> Result<(), TransportError> {
        match self
            .call(Req::Commit {
                session: session.clone(),
                epoch,
                fence,
            })
            .await?
        {
            Resp::Commit(Ok(())) => Ok(()),
            Resp::Commit(Err(e)) => Err(TransportError::Store(e)),
            _ => Err(TransportError::Protocol),
        }
    }

    /// Read a session's durable status from the remote authority.
    pub async fn status(
        &mut self,
        session: &SessionId,
    ) -> Result<Option<SessionStatus>, TransportError> {
        match self
            .call(Req::Status {
                session: session.clone(),
            })
            .await?
        {
            Resp::Status(s) => Ok(s),
            _ => Err(TransportError::Protocol),
        }
    }
}
