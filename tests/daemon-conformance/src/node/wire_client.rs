// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! A transport-generic multiplexed client for the node api wire protocol (the L0 mux envelope:
//! `Hello` handshake, correlated `Call`/`Reply`, the SASL `AuthStart`/`AuthStep`/`AuthResume`
//! exchange, and `Open`/`Item`/`End` streams). Unlike [`daemon_host::MuxApiClient`] (Unix-only) this
//! is generic over any [`AsyncRead`] + [`AsyncWrite`] byte stream, so the Auth 7 e2e/negative suites
//! drive the *same* client over both the plaintext Unix socket and a real `rustls` TLS/TCP
//! connection — proving the shared `serve_mux` auth path on every transport.
//!
//! Framing matches the server (`daemon_host::socket`): a 4-byte big-endian length prefix + a CBOR
//! body, with the wire types from [`daemon_api`]. It speaks the same `rsasl` SCRAM-SHA-256 client a
//! real GUI/TUI uses, so the handshake is exercised end to end (no internal shortcut).

#![allow(dead_code)]

use daemon_api::{
    from_cbor, to_cbor, ApiError, ApiRequest, ApiResponse, PrincipalView, WireC2S, WireS2C,
    WIRE_FEATURE_MUX, WIRE_FEATURE_STREAM, WIRE_VERSION,
};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

/// A multiplexed wire client over a byte stream `S` (a `UnixStream`, a `tokio_rustls` TLS stream,
/// etc.). Single-stream-at-a-time in shape (the reader is `&mut self`), which suits conformance.
pub struct MuxConn<S> {
    stream: S,
    next_id: u64,
    /// The mechanisms the server advertised on its `Hello` (empty under local trust).
    pub mechanisms: Vec<String>,
}

impl<S> MuxConn<S>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    /// Complete the `Hello` handshake over an already-connected stream, capturing the advertised
    /// `auth_mechanisms`.
    pub async fn handshake(mut stream: S) -> Result<Self, ApiError> {
        let hello = WireC2S::Hello {
            wire_version: WIRE_VERSION,
            features: vec![
                WIRE_FEATURE_MUX.to_string(),
                WIRE_FEATURE_STREAM.to_string(),
            ],
        };
        write_frame(&mut stream, &to_cbor(&hello))
            .await
            .map_err(|e| ApiError::Other(format!("send hello: {e}")))?;
        let bytes = read_frame(&mut stream)
            .await
            .map_err(|e| ApiError::Other(format!("recv hello: {e}")))?
            .ok_or_else(|| ApiError::Other("closed before hello ack".into()))?;
        match from_cbor::<WireS2C>(&bytes)? {
            WireS2C::Hello {
                auth_mechanisms, ..
            } => Ok(Self {
                stream,
                next_id: 1,
                mechanisms: auth_mechanisms,
            }),
            other => Err(ApiError::Other(format!(
                "expected Hello ack, got {other:?}"
            ))),
        }
    }

    fn take_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    async fn send(&mut self, frame: WireC2S) -> Result<(), ApiError> {
        write_frame(&mut self.stream, &to_cbor(&frame))
            .await
            .map_err(|e| ApiError::Other(format!("send: {e}")))
    }

    /// Read the next server frame.
    pub async fn next(&mut self) -> Result<WireS2C, ApiError> {
        let bytes = read_frame(&mut self.stream)
            .await
            .map_err(|e| ApiError::Other(format!("recv: {e}")))?
            .ok_or_else(|| ApiError::Other("connection closed".into()))?;
        from_cbor::<WireS2C>(&bytes)
    }

    /// Send a one-shot `Call` and await its correlated `Reply` (or an `End` carrying the error).
    pub async fn call(&mut self, req: ApiRequest) -> Result<ApiResponse, ApiError> {
        let id = self.take_id();
        self.send(WireC2S::Call { id, req }).await?;
        loop {
            match self.next().await? {
                WireS2C::Reply { id: rid, res } if rid == id => return Ok(res),
                WireS2C::End { id: rid, error } if rid == id => {
                    return Err(error.unwrap_or_else(|| {
                        ApiError::Other("stream ended without a reply".into())
                    }));
                }
                _ => continue,
            }
        }
    }

    /// Open a server-stream for a streaming request; read its frames with [`MuxConn::next`].
    pub async fn open(&mut self, req: ApiRequest) -> Result<u64, ApiError> {
        let id = self.take_id();
        self.send(WireC2S::Open { id, req }).await?;
        Ok(id)
    }

    /// Cancel a streaming exchange.
    pub async fn cancel(&mut self, id: u64) -> Result<(), ApiError> {
        self.send(WireC2S::Cancel { id }).await
    }

    /// Send a raw `AuthStart` and return the server's first reply frame (used by the negative suite
    /// to assert that a malformed/unknown mechanism yields `AuthError`).
    pub async fn auth_start(
        &mut self,
        mechanism: &str,
        initial: Vec<u8>,
    ) -> Result<WireS2C, ApiError> {
        self.send(WireC2S::AuthStart {
            mechanism: mechanism.to_string(),
            initial,
        })
        .await?;
        self.next().await
    }

    /// Drive a full `SCRAM-SHA-256` exchange with a real `rsasl` client, returning the authenticated
    /// [`PrincipalView`] and the session token from `AuthOk` (the token a client presents on
    /// reconnect via [`MuxConn::authenticate_resume`]).
    pub async fn authenticate_scram(
        &mut self,
        username: &str,
        password: &str,
    ) -> Result<(PrincipalView, String), ApiError> {
        use rsasl::prelude::{Mechname, SASLClient, SASLConfig, State as ClientState};

        let config = SASLConfig::with_credentials(None, username.into(), password.into())
            .map_err(|e| ApiError::Other(format!("sasl client config: {e}")))?;
        let mechname = Mechname::parse(daemon_host::MECH_SCRAM_SHA_256.as_bytes())
            .map_err(|e| ApiError::Other(format!("mechname: {e}")))?;
        let mut session = SASLClient::new(config)
            .start_suggested_iter([mechname])
            .map_err(|e| ApiError::Other(format!("sasl client start: {e}")))?;

        // SCRAM is client-first: produce the client-first message with no input.
        let mut out = Vec::new();
        session
            .step(None, &mut out)
            .map_err(|e| ApiError::Other(format!("sasl client step: {e}")))?;
        self.send(WireC2S::AuthStart {
            mechanism: daemon_host::MECH_SCRAM_SHA_256.to_string(),
            initial: out.clone(),
        })
        .await?;

        loop {
            match self.next().await? {
                WireS2C::AuthChallenge { data } => {
                    out.clear();
                    let state = session
                        .step(Some(&data), &mut out)
                        .map_err(|e| ApiError::Unauthenticated(format!("sasl: {e}")))?;
                    if !out.is_empty() {
                        self.send(WireC2S::AuthStep { data: out.clone() }).await?;
                    } else if state == ClientState::Running {
                        // Running with no output is unexpected for SCRAM; keep awaiting AuthOk.
                    }
                }
                WireS2C::AuthOk { principal, token } => return Ok((principal, token)),
                WireS2C::AuthError { reason } => return Err(ApiError::Unauthenticated(reason)),
                other => {
                    return Err(ApiError::Other(format!(
                        "unexpected frame during authentication: {other:?}"
                    )))
                }
            }
        }
    }

    /// The reconnect fast-path: present a prior session `token` via `AuthResume`, returning the
    /// re-bound [`PrincipalView`] (or the coarse `AuthError` for an invalid/revoked/expired token).
    pub async fn authenticate_resume(&mut self, token: &str) -> Result<PrincipalView, ApiError> {
        self.send(WireC2S::AuthResume {
            token: token.to_string(),
        })
        .await?;
        match self.next().await? {
            WireS2C::AuthOk { principal, .. } => Ok(principal),
            WireS2C::AuthError { reason } => Err(ApiError::Unauthenticated(reason)),
            other => Err(ApiError::Other(format!(
                "unexpected frame during resume: {other:?}"
            ))),
        }
    }
}

/// Read one length-framed message (4-byte big-endian length + body). `Ok(None)` on a clean EOF.
async fn read_frame<R: AsyncRead + Unpin>(stream: &mut R) -> std::io::Result<Option<Vec<u8>>> {
    let mut len_buf = [0u8; 4];
    match stream.read_exact(&mut len_buf).await {
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e),
    }
    let len = u32::from_be_bytes(len_buf) as usize;
    let mut buf = vec![0u8; len];
    stream.read_exact(&mut buf).await?;
    Ok(Some(buf))
}

/// Write one length-framed message.
async fn write_frame<W: AsyncWrite + Unpin>(stream: &mut W, bytes: &[u8]) -> std::io::Result<()> {
    let len = u32::try_from(bytes.len())
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidInput, "frame too large"))?;
    stream.write_all(&len.to_be_bytes()).await?;
    stream.write_all(bytes).await?;
    stream.flush().await?;
    Ok(())
}
