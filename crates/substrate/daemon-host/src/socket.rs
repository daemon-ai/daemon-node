//! The Unix-domain-socket transport adapter for the [`daemon_api`] surface.
//!
//! A thin shell over the shared [`daemon_api::dispatch`]: it moves length-framed CBOR
//! [`ApiRequest`]/[`ApiResponse`] bytes over a socket and calls `dispatch` — *exactly* the same
//! surface the in-process caller and the C FFI reach, only the byte transport differs. Framing is a
//! 4-byte big-endian length prefix followed by the CBOR payload.

use daemon_api::{dispatch, from_cbor, to_cbor, ApiError, ApiRequest, ApiResponse, NodeApi};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{UnixListener, UnixStream};

/// Serve the node surface over a bound [`UnixListener`] until it errors. Each connection is a
/// request/response stream: read a framed [`ApiRequest`], `dispatch`, write the framed
/// [`ApiResponse`], repeat. Runs forever; spawn it as a background task after `host.start()`.
pub async fn serve_api_unix(listener: UnixListener, api: Arc<dyn NodeApi>) {
    loop {
        match listener.accept().await {
            Ok((stream, _addr)) => {
                let api = api.clone();
                tokio::spawn(async move {
                    if let Err(e) = serve_conn(stream, api).await {
                        tracing::debug!("api socket connection ended: {e}");
                    }
                });
            }
            Err(e) => {
                tracing::warn!("api socket accept failed: {e}");
                return;
            }
        }
    }
}

async fn serve_conn(mut stream: UnixStream, api: Arc<dyn NodeApi>) -> std::io::Result<()> {
    while let Some(bytes) = read_frame(&mut stream).await? {
        let response = match from_cbor::<ApiRequest>(&bytes) {
            Ok(request) => dispatch(api.as_ref(), request).await,
            Err(e) => ApiResponse::Error(e),
        };
        write_frame(&mut stream, &to_cbor(&response)).await?;
    }
    Ok(())
}

/// A one-shot client over the Unix-socket adapter: connect, send one request, read one response.
/// Cheap to clone (it only holds the socket path); each [`ApiClient::call`] opens a fresh
/// connection — the model an operator CLI wants.
#[derive(Clone)]
pub struct ApiClient {
    path: PathBuf,
}

impl ApiClient {
    /// A client targeting the socket at `path`.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    /// Connect, send `request`, and await the single framed response.
    pub async fn call(&self, request: ApiRequest) -> Result<ApiResponse, ApiError> {
        let mut stream = UnixStream::connect(&self.path)
            .await
            .map_err(|e| ApiError::Other(format!("connect {}: {e}", self.path.display())))?;
        write_frame(&mut stream, &to_cbor(&request))
            .await
            .map_err(|e| ApiError::Other(format!("send: {e}")))?;
        let bytes = read_frame(&mut stream)
            .await
            .map_err(|e| ApiError::Other(format!("recv: {e}")))?
            .ok_or_else(|| ApiError::Other("connection closed before a response".into()))?;
        from_cbor::<ApiResponse>(&bytes)
    }
}

/// Read one length-framed message. Returns `Ok(None)` on a clean EOF at a frame boundary.
async fn read_frame(stream: &mut UnixStream) -> std::io::Result<Option<Vec<u8>>> {
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
async fn write_frame(stream: &mut UnixStream, bytes: &[u8]) -> std::io::Result<()> {
    let len = u32::try_from(bytes.len())
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidInput, "frame too large"))?;
    stream.write_all(&len.to_be_bytes()).await?;
    stream.write_all(bytes).await?;
    stream.flush().await?;
    Ok(())
}
