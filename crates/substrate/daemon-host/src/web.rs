// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The single-origin web front for the browser GUI: ONE plain-HTTP listener that serves the Qt
//! WASM app bundle as static files AND hosts the same mux-over-WebSocket carrier ([`crate::ws`])
//! on `GET /ws` — so a browser loads the GUI *from the daemon* and connects back to the very
//! origin it was loaded from, with zero CORS/origin configuration. The groundwork for running the
//! daemon as a self-contained appliance (e.g. a microvm serving its own GUI).
//!
//! * **Same-origin upgrades work with zero config** — the upgrade gate derives the listener's own
//!   origin from the request's `Host` header (`http://<host>`, since this listener is plain HTTP)
//!   and accepts an `Origin` matching it automatically; `[api].ws_allowed_origins` additionally
//!   applies for deliberate cross-origin allowance; every other *browser* origin is refused with
//!   403. An upgrade with **no** `Origin` header (a non-browser client) is accepted, same as on
//!   the standalone listener: the origin gate is a browser CSRF defense, and non-browser clients
//!   are gated by the mandatory authentication on `/ws` instead (see [`crate::ws`] for the full
//!   rationale). Behind a TLS-terminating reverse proxy the browser's `Origin` is `https://…` and
//!   no longer matches the derived `http://…` self-origin — add the public origin to
//!   `[api].ws_allowed_origins` there.
//! * **Static files are public, the api is not** — `/ws` runs the identical
//!   [`serve_mux_over_ws`] posture as the standalone `[api].ws_addr` listener: authentication
//!   ALWAYS required, plaintext transport, SCRAM only.
//! * **Traversal is impossible by construction** — the bundle directory is scanned ONCE at
//!   startup into an allow-map of regular files ([`WebRoot::scan`]); a request is a pure map
//!   lookup (after percent-decoding), never a filesystem path computation, so no request string
//!   can name a file outside the map. The flip side (the reload caveat): files added to the
//!   directory after startup are not served until the daemon restarts.
//! * **Content negotiation for the fat artifacts** — correct Content-Types (notably
//!   `application/wasm`, required for `WebAssembly.compileStreaming`), plus `Accept-Encoding`
//!   negotiation onto precompressed `.br`/`.gz` siblings scanned next to their identity files
//!   (`Content-Encoding` + the *underlying* Content-Type + `Vary: Accept-Encoding`).
//! * **GET/HEAD only**, sequential HTTP/1.1 keep-alive, unknown paths 404. `https://` (and thus
//!   `wss://`) terminate at a reverse proxy for now, exactly like the other TCP listeners.
//!
//! The HTTP/1.1 front is deliberately hand-rolled on tokio: the daemon's layering keeps axum
//! isolated in the `daemon-http` adapter crate, and the request surface here (two verbs, no
//! bodies, an allow-map) is small enough that a framework would only add dependencies.

use std::collections::{BTreeMap, HashMap};
use std::io;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use daemon_api::NodeApi;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, ReadBuf};
use tokio::net::{TcpListener, TcpStream};
use tokio_tungstenite::tungstenite::handshake::server::{Request, Response};
use tokio_tungstenite::tungstenite::http::header::HOST;
use tokio_tungstenite::WebSocketStream;

use crate::authn::Authenticator;
use crate::ws::{apply_upgrade_policy, serve_mux_over_ws};

/// The bundle's entry page, served for `/` (the Qt wasm installer's flat layout).
const INDEX_FILE: &str = "daemon-app.html";
/// The reserved WebSocket upgrade path on the single-origin listener.
const WS_PATH: &str = "/ws";
/// Upper bound on one request head (request line + headers) — far beyond any browser's.
const MAX_HEAD_BYTES: usize = 16 * 1024;
/// Directory-recursion bound for the startup scan (a symlink-loop guard; bundles are flat).
const MAX_SCAN_DEPTH: usize = 16;

// --- the startup-scanned allow-map ------------------------------------------------------------

/// One servable file: its identity path, its Content-Type, and the precompressed siblings found
/// next to it at scan time.
struct FileEntry {
    path: PathBuf,
    content_type: &'static str,
    br: Option<PathBuf>,
    gz: Option<PathBuf>,
}

impl FileEntry {
    /// Whether any precompressed variant exists (=> responses carry `Vary: Accept-Encoding`).
    fn has_variants(&self) -> bool {
        self.br.is_some() || self.gz.is_some()
    }

    /// Pick the representation for an `Accept-Encoding` header: the `.br` sibling when brotli is
    /// acceptable, else the `.gz` sibling when gzip is, else the identity file. Returns the file
    /// to stream and the `Content-Encoding` token to declare (`None` = identity).
    fn negotiate(&self, accept_encoding: Option<&str>) -> (&Path, Option<&'static str>) {
        if let Some(accept) = accept_encoding {
            if let Some(br) = &self.br {
                if accepts_coding(accept, "br") {
                    return (br, Some("br"));
                }
            }
            if let Some(gz) = &self.gz {
                if accepts_coding(accept, "gzip") {
                    return (gz, Some("gzip"));
                }
            }
        }
        (&self.path, None)
    }
}

/// The startup-scanned allow-map of servable files under the bundle root. Requests resolve by
/// exact map lookup only — the filesystem is never walked per-request, so path traversal cannot
/// reach outside the scanned set (see the module docs for the reload caveat).
pub struct WebRoot {
    /// `/`-separated path relative to the root (e.g. `daemon-app.wasm`) → the served file.
    files: HashMap<String, FileEntry>,
}

impl WebRoot {
    /// Scan `root` (which must be an existing directory) into the allow-map: every regular file,
    /// recursively, symlinks resolved (Nix-store bundle layouts link freely at deploy time —
    /// scan-time resolution is operator-controlled, not request-controlled). `<name>.br` /
    /// `<name>.gz` files are attached to their identity sibling as negotiable variants rather
    /// than listed as pages of their own. Fails fast so a misconfigured `[web].root` stops boot.
    pub fn scan(root: &Path) -> io::Result<Self> {
        let meta = std::fs::metadata(root)?;
        if !meta.is_dir() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "not a directory",
            ));
        }
        let mut found = BTreeMap::new();
        collect_files(root, "", &mut found, 0)?;
        let mut files = HashMap::new();
        for (key, path) in &found {
            if key.ends_with(".br") || key.ends_with(".gz") {
                continue;
            }
            files.insert(
                key.clone(),
                FileEntry {
                    path: path.clone(),
                    content_type: content_type_for(key),
                    br: found.get(&format!("{key}.br")).cloned(),
                    gz: found.get(&format!("{key}.gz")).cloned(),
                },
            );
        }
        if !files.contains_key(INDEX_FILE) {
            tracing::warn!(
                root = %root.display(),
                "web root has no {INDEX_FILE}: `/` will 404 (is this the app bundle directory?)"
            );
        }
        Ok(Self { files })
    }

    /// How many servable pages the scan found (identity files; variants attach to them).
    pub fn len(&self) -> usize {
        self.files.len()
    }

    /// Whether the scan found nothing servable.
    pub fn is_empty(&self) -> bool {
        self.files.is_empty()
    }

    /// Resolve a percent-decoded request path to a scanned entry. `/` aliases the index page. A
    /// `..` segment can never match (map keys come from our own walk), but is refused explicitly
    /// anyway so the choke point reads as fail-closed.
    fn lookup(&self, decoded_path: &str) -> Option<&FileEntry> {
        let rel = decoded_path.strip_prefix('/')?;
        if rel.split('/').any(|segment| segment == "..") {
            return None;
        }
        let rel = if rel.is_empty() { INDEX_FILE } else { rel };
        self.files.get(rel)
    }
}

/// Recursively collect regular files under `dir` into `out`, keyed by `/`-joined relative path.
/// Unreadable entries (e.g. a dangling symlink) are skipped with a log rather than failing boot.
fn collect_files(
    dir: &Path,
    prefix: &str,
    out: &mut BTreeMap<String, PathBuf>,
    depth: usize,
) -> io::Result<()> {
    if depth > MAX_SCAN_DEPTH {
        tracing::warn!(dir = %dir.display(), "web root deeper than {MAX_SCAN_DEPTH} levels; not descending");
        return Ok(());
    }
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name();
        // A non-UTF8 name can never match a request path; skip it.
        let Some(name) = name.to_str() else { continue };
        let path = entry.path();
        let key = if prefix.is_empty() {
            name.to_string()
        } else {
            format!("{prefix}/{name}")
        };
        let meta = match std::fs::metadata(&path) {
            Ok(meta) => meta,
            Err(e) => {
                tracing::warn!(path = %path.display(), "skipping unreadable web root entry: {e}");
                continue;
            }
        };
        if meta.is_dir() {
            collect_files(&path, &key, out, depth + 1)?;
        } else if meta.is_file() {
            out.insert(key, path);
        }
    }
    Ok(())
}

/// The Content-Type for a served path, by extension. `application/wasm` is load-bearing (the
/// browser refuses `WebAssembly.compileStreaming` without it); `.data` is Qt's packed asset blob.
fn content_type_for(path: &str) -> &'static str {
    let ext = path.rsplit_once('.').map(|(_, ext)| ext).unwrap_or("");
    match ext {
        "html" | "htm" => "text/html; charset=utf-8",
        "js" | "mjs" => "application/javascript",
        "wasm" => "application/wasm",
        "json" => "application/json",
        "css" => "text/css; charset=utf-8",
        "svg" => "image/svg+xml",
        "png" => "image/png",
        "ico" => "image/x-icon",
        "txt" => "text/plain; charset=utf-8",
        // `.data` and anything unknown: an opaque byte blob.
        _ => "application/octet-stream",
    }
}

/// Whether an `Accept-Encoding` header value accepts `coding`: the token must be listed (matched
/// case-insensitively) and not disabled with `q=0`. Wildcards are ignored — every real browser
/// lists `br`/`gzip` explicitly, and identity remains the always-correct fallback.
fn accepts_coding(accept_encoding: &str, coding: &str) -> bool {
    accept_encoding.split(',').any(|part| {
        let mut params = part.split(';');
        let token = params.next().unwrap_or("").trim();
        if !token.eq_ignore_ascii_case(coding) {
            return false;
        }
        for param in params {
            if let Some(q) = param.trim().strip_prefix("q=") {
                return q.trim().parse::<f64>().map(|v| v > 0.0).unwrap_or(false);
            }
        }
        true
    })
}

// --- the request head -------------------------------------------------------------------------

/// A parsed HTTP/1.x request head (the routing surface: method, target, headers). Bodies are
/// never read — GET/HEAD carry none, and a request declaring one downgrades to `Connection:
/// close` so the unread bytes can never desynchronize the next request.
struct RequestHead {
    method: String,
    target: String,
    /// `false` only for HTTP/1.0 (no default keep-alive).
    http_11: bool,
    /// Lower-cased names, trimmed values, in wire order.
    headers: Vec<(String, String)>,
}

impl RequestHead {
    /// The path portion of the request target (query stripped; never percent-decoded here).
    fn path(&self) -> &str {
        self.target.split(['?', '#']).next().unwrap_or("")
    }

    /// The first value of header `name` (`name` must be lower-case).
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.as_str())
    }

    /// Whether any value of header `name` contains `token` in its comma-separated list
    /// (case-insensitive) — the `Connection` / `Upgrade` token grammar.
    fn header_has_token(&self, name: &str, token: &str) -> bool {
        self.headers
            .iter()
            .filter(|(n, _)| n == name)
            .any(|(_, v)| {
                v.split(',')
                    .any(|item| item.trim().eq_ignore_ascii_case(token))
            })
    }

    /// Whether the request declares a body (which this server never reads).
    fn has_body(&self) -> bool {
        self.header("content-length")
            .is_some_and(|v| v.trim() != "0")
            || self.header("transfer-encoding").is_some()
    }

    /// Whether the connection stays open after responding to this request.
    fn keep_alive(&self) -> bool {
        self.http_11 && !self.header_has_token("connection", "close") && !self.has_body()
    }

    /// Whether this is the WebSocket upgrade for the mux carrier (`GET /ws` + `Upgrade:
    /// websocket`). A plain `GET /ws` without the upgrade header falls through to static
    /// serving (404 — the bundle has no `ws` file); tungstenite enforces the rest of RFC 6455.
    fn is_ws_upgrade(&self) -> bool {
        self.path() == WS_PATH && self.header_has_token("upgrade", "websocket")
    }
}

/// Parse one request head (everything up to and including the blank line). `None` = malformed.
fn parse_request_head(bytes: &[u8]) -> Option<RequestHead> {
    let text = std::str::from_utf8(bytes).ok()?;
    let mut lines = text.split("\r\n");
    let request_line = lines.next()?;
    let mut parts = request_line.split_ascii_whitespace();
    let method = parts.next()?.to_string();
    let target = parts.next()?.to_string();
    let version = parts.next()?;
    if parts.next().is_some() || !version.starts_with("HTTP/1.") {
        return None;
    }
    let mut headers = Vec::new();
    for line in lines {
        if line.is_empty() {
            continue; // the terminating blank line
        }
        let (name, value) = line.split_once(':')?;
        headers.push((name.trim().to_ascii_lowercase(), value.trim().to_string()));
    }
    Some(RequestHead {
        method,
        target,
        http_11: version != "HTTP/1.0",
        headers,
    })
}

/// Percent-decode a request path. `None` on a malformed escape (refused as 400). The decoded
/// string is only ever used as an allow-map key, so decoded `/`, `..`, NUL etc. can redirect the
/// lookup but never reach the filesystem.
fn percent_decode(path: &str) -> Option<String> {
    let bytes = path.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            let hi = (*bytes.get(i + 1)? as char).to_digit(16)?;
            let lo = (*bytes.get(i + 2)? as char).to_digit(16)?;
            out.push((hi * 16 + lo) as u8);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(out).ok()
}

/// Find the end of the request head (the index just past `\r\n\r\n`).
fn head_end(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n").map(|p| p + 4)
}

// --- the listener -----------------------------------------------------------------------------

/// Serve the single-origin web front until the listener errors: static bundle files from the
/// startup-scanned `site`, plus the authenticated mux-over-WebSocket carrier on `GET /ws` (the
/// same serving as [`serve_mux_ws`](crate::ws::serve_mux_ws) — same-origin upgrades pass with
/// zero config, `allowed_origins` grants extra cross-origin allowance). Spawn it as a background
/// task alongside the other listeners.
pub async fn serve_web(
    listener: TcpListener,
    site: WebRoot,
    api: Arc<dyn NodeApi>,
    auth: Arc<Authenticator>,
    allowed_origins: Vec<String>,
) {
    let site = Arc::new(site);
    let allowed = Arc::new(allowed_origins);
    loop {
        match listener.accept().await {
            Ok((stream, _addr)) => {
                let site = site.clone();
                let api = api.clone();
                let auth = auth.clone();
                let allowed = allowed.clone();
                tokio::spawn(async move {
                    // A failed conversation (malformed request, refused upgrade, an aborted
                    // download) is dropped cleanly — never panics the accept loop.
                    if let Err(e) = handle_conn(stream, &site, api, auth, &allowed).await {
                        tracing::debug!("web connection ended: {e}");
                    }
                });
            }
            Err(e) => {
                tracing::warn!("web accept failed: {e}");
                return;
            }
        }
    }
}

/// One connection: sequential HTTP/1.1 requests served off the same socket until the client
/// closes (or asks to), with `GET /ws` + `Upgrade: websocket` handing the connection over to the
/// mux carrier for the rest of its life.
async fn handle_conn(
    mut stream: TcpStream,
    site: &WebRoot,
    api: Arc<dyn NodeApi>,
    auth: Arc<Authenticator>,
    allowed_origins: &[String],
) -> io::Result<()> {
    let mut carry = Vec::new();
    loop {
        let Some((head_bytes, rest)) = read_head(&mut stream, carry).await? else {
            return Ok(()); // clean EOF between requests
        };
        let Some(head) = parse_request_head(&head_bytes) else {
            write_simple(&mut stream, 400, "Bad Request", None, false).await?;
            return Ok(());
        };

        if head.is_ws_upgrade() {
            // Hand tungstenite the exact bytes already consumed (the head, plus anything read
            // past it) so its server handshake re-reads the same request off the live stream.
            let mut replay = head_bytes;
            replay.extend_from_slice(&rest);
            let ws = accept_web_ws(Rewind::new(replay, stream), allowed_origins)
                .await
                .map_err(io::Error::other)?;
            return serve_mux_over_ws(ws, api, auth).await;
        }

        let keep = respond_static(&mut stream, site, &head).await?;
        if !keep {
            return Ok(());
        }
        carry = rest;
    }
}

/// Read one request head off the stream, `carry` being bytes already read past the previous
/// request. Returns the head (through the blank line) and any surplus bytes, or `None` on a
/// clean EOF before any byte of a new request.
async fn read_head(
    stream: &mut TcpStream,
    carry: Vec<u8>,
) -> io::Result<Option<(Vec<u8>, Vec<u8>)>> {
    let mut buf = carry;
    loop {
        if let Some(end) = head_end(&buf) {
            let rest = buf.split_off(end);
            return Ok(Some((buf, rest)));
        }
        if buf.len() > MAX_HEAD_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "request head too large",
            ));
        }
        let mut chunk = [0u8; 4096];
        let n = stream.read(&mut chunk).await?;
        if n == 0 {
            return if buf.is_empty() {
                Ok(None)
            } else {
                Err(io::ErrorKind::UnexpectedEof.into())
            };
        }
        buf.extend_from_slice(&chunk[..n]);
    }
}

/// Run the WebSocket server handshake with the mux upgrade gate ([`apply_upgrade_policy`])
/// applied over the *effective* allow-list: the configured origins plus the listener's own origin
/// derived from the request's `Host` header (`http://<host>` — this listener is plain HTTP; see
/// the module docs for the reverse-proxy caveat). That derived entry is what makes same-origin
/// browser pages work with zero configuration.
// tungstenite's `Callback` trait dictates the `Result<Response, ErrorResponse>` shape inside the
// closure; not ours to shrink.
#[allow(clippy::result_large_err)]
async fn accept_web_ws<S>(
    stream: S,
    allowed_origins: &[String],
) -> Result<WebSocketStream<S>, tokio_tungstenite::tungstenite::Error>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    tokio_tungstenite::accept_hdr_async(stream, |req: &Request, resp: Response| {
        let mut effective = allowed_origins.to_vec();
        if let Some(own) = self_origin(req) {
            effective.push(own);
        }
        apply_upgrade_policy(req, resp, &effective)
    })
    .await
}

/// The listener's own origin as seen by this request: `http://` + the `Host` header verbatim
/// (host, plus the port whenever the browser included one — i.e. always, except on default-port
/// deployments, where both sides omit it and still agree).
fn self_origin(req: &Request) -> Option<String> {
    let host = req.headers().get(HOST)?.to_str().ok()?.trim();
    if host.is_empty() {
        return None;
    }
    Some(format!("http://{host}"))
}

/// Serve one static request (everything that is not the `/ws` upgrade): GET/HEAD only, allow-map
/// lookup, `Accept-Encoding` negotiation, streamed body. Returns whether the connection stays
/// open for the next request.
async fn respond_static(
    stream: &mut TcpStream,
    site: &WebRoot,
    head: &RequestHead,
) -> io::Result<bool> {
    let keep = head.keep_alive();
    if head.method != "GET" && head.method != "HEAD" {
        write_simple(
            stream,
            405,
            "Method Not Allowed",
            Some("allow: GET, HEAD"),
            keep,
        )
        .await?;
        return Ok(keep);
    }
    let Some(decoded) = percent_decode(head.path()) else {
        write_simple(stream, 400, "Bad Request", None, false).await?;
        return Ok(false);
    };
    let Some(entry) = site.lookup(&decoded) else {
        write_simple(stream, 404, "Not Found", None, keep).await?;
        return Ok(keep);
    };

    let (file_path, encoding) = entry.negotiate(head.header("accept-encoding"));
    // The allow-map was scanned at startup; a since-vanished file (the reload caveat) is a 404,
    // not a connection error.
    let mut file = match tokio::fs::File::open(file_path).await {
        Ok(file) => file,
        Err(e) => {
            tracing::debug!(path = %file_path.display(), "scanned web file no longer readable: {e}");
            write_simple(stream, 404, "Not Found", None, keep).await?;
            return Ok(keep);
        }
    };
    let len = file.metadata().await?.len();

    let mut resp = format!(
        "HTTP/1.1 200 OK\r\ncontent-type: {}\r\ncontent-length: {len}\r\n",
        entry.content_type
    );
    if let Some(encoding) = encoding {
        resp.push_str(&format!("content-encoding: {encoding}\r\n"));
    }
    if entry.has_variants() {
        // Cache correctness: the representation depends on the request's Accept-Encoding.
        resp.push_str("vary: accept-encoding\r\n");
    }
    if !keep {
        resp.push_str("connection: close\r\n");
    }
    resp.push_str("\r\n");
    stream.write_all(resp.as_bytes()).await?;
    if head.method == "GET" {
        tokio::io::copy(&mut file, stream).await?;
    }
    stream.flush().await?;
    Ok(keep)
}

/// Write a minimal non-200 response with a plain-text body.
async fn write_simple(
    stream: &mut TcpStream,
    status: u16,
    reason: &str,
    extra_header: Option<&str>,
    keep: bool,
) -> io::Result<()> {
    let body = reason.to_ascii_lowercase();
    let mut resp = format!(
        "HTTP/1.1 {status} {reason}\r\ncontent-type: text/plain; charset=utf-8\r\ncontent-length: {}\r\n",
        body.len()
    );
    if let Some(extra) = extra_header {
        resp.push_str(extra);
        resp.push_str("\r\n");
    }
    if !keep {
        resp.push_str("connection: close\r\n");
    }
    resp.push_str("\r\n");
    resp.push_str(&body);
    stream.write_all(resp.as_bytes()).await?;
    stream.flush().await
}

// --- the rewind shim --------------------------------------------------------------------------

/// A stream that replays already-consumed bytes before reading on: the routing above reads the
/// request head off the TCP stream, and tungstenite's server handshake needs to read that same
/// request itself. Writes pass straight through.
struct Rewind<S> {
    pre: Vec<u8>,
    pos: usize,
    inner: S,
}

impl<S> Rewind<S> {
    fn new(pre: Vec<u8>, inner: S) -> Self {
        Self { pre, pos: 0, inner }
    }
}

impl<S: AsyncRead + Unpin> AsyncRead for Rewind<S> {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        out: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        let this = self.get_mut();
        if this.pos < this.pre.len() {
            let n = out.remaining().min(this.pre.len() - this.pos);
            out.put_slice(&this.pre[this.pos..this.pos + n]);
            this.pos += n;
            if this.pos == this.pre.len() {
                this.pre = Vec::new(); // replayed in full; drop the buffer
                this.pos = 0;
            }
            return Poll::Ready(Ok(()));
        }
        Pin::new(&mut this.inner).poll_read(cx, out)
    }
}

impl<S: AsyncWrite + Unpin> AsyncWrite for Rewind<S> {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        data: &[u8],
    ) -> Poll<io::Result<usize>> {
        Pin::new(&mut self.get_mut().inner).poll_write(cx, data)
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.get_mut().inner).poll_flush(cx)
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.get_mut().inner).poll_shutdown(cx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Write a fake bundle into a tempdir and scan it.
    fn scanned(files: &[(&str, &[u8])]) -> (tempfile::TempDir, WebRoot) {
        let dir = tempfile::tempdir().expect("tempdir");
        for (name, contents) in files {
            let path = dir.path().join(name);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).expect("mkdir");
            }
            std::fs::write(path, contents).expect("write");
        }
        let root = WebRoot::scan(dir.path()).expect("scan");
        (dir, root)
    }

    /// The allow-map: `/` aliases the index, scanned files resolve, anything else — including
    /// every traversal shape — misses. Nothing outside the map is reachable by construction.
    #[test]
    fn lookup_is_allow_map_only() {
        let (_dir, root) = scanned(&[
            ("daemon-app.html", b"<html/>"),
            ("daemon-app.wasm", b"\0asm"),
            ("assets/icon.png", b"png"),
        ]);
        assert_eq!(root.len(), 3);
        assert!(root.lookup("/").is_some(), "/ must alias {INDEX_FILE}");
        assert!(root.lookup("/daemon-app.wasm").is_some());
        assert!(root.lookup("/assets/icon.png").is_some());
        for miss in [
            "/nope.js",
            "/../Cargo.toml",
            "/..",
            "/assets/../../etc/passwd",
            "//etc/passwd",
            "/daemon-app.wasm/",
            "daemon-app.wasm", // no leading slash: not a valid request path
        ] {
            assert!(root.lookup(miss).is_none(), "{miss} must not resolve");
        }
    }

    /// Precompressed siblings attach to their identity file (and are not pages of their own),
    /// and negotiation prefers br, then gzip, then identity — q=0 disables a listed coding.
    #[test]
    fn variants_attach_and_negotiate() {
        let (_dir, root) = scanned(&[
            ("daemon-app.wasm", b"idn"),
            ("daemon-app.wasm.br", b"bro"),
            ("daemon-app.wasm.gz", b"gzp"),
            ("daemon-app.html", b"<html/>"),
        ]);
        assert_eq!(root.len(), 2, "variants must not be listed as pages");
        assert!(root.lookup("/daemon-app.wasm.br").is_none());

        let entry = root.lookup("/daemon-app.wasm").expect("entry");
        assert!(entry.has_variants());
        let ends = |p: &Path, suffix: &str| p.to_string_lossy().ends_with(suffix);

        let (p, enc) = entry.negotiate(Some("gzip, br"));
        assert!(ends(p, ".wasm.br") && enc == Some("br"), "br wins");
        let (p, enc) = entry.negotiate(Some("gzip;q=0.5"));
        assert!(ends(p, ".wasm.gz") && enc == Some("gzip"));
        let (p, enc) = entry.negotiate(Some("br;q=0, gzip"));
        assert!(
            ends(p, ".wasm.gz") && enc == Some("gzip"),
            "q=0 disables br"
        );
        let (p, enc) = entry.negotiate(Some("identity"));
        assert!(ends(p, ".wasm") && enc.is_none());
        let (p, enc) = entry.negotiate(None);
        assert!(ends(p, ".wasm") && enc.is_none());

        let plain = root.lookup("/daemon-app.html").expect("entry");
        assert!(!plain.has_variants());
        let (_, enc) = plain.negotiate(Some("br, gzip"));
        assert!(enc.is_none(), "no variant on disk => identity");
    }

    /// The Content-Type table: the four bundle-critical types plus the octet-stream default.
    #[test]
    fn content_types_for_the_bundle() {
        assert_eq!(
            content_type_for("daemon-app.html"),
            "text/html; charset=utf-8"
        );
        assert_eq!(content_type_for("qtloader.js"), "application/javascript");
        assert_eq!(content_type_for("daemon-app.wasm"), "application/wasm");
        assert_eq!(
            content_type_for("daemon-app.data"),
            "application/octet-stream"
        );
        assert_eq!(content_type_for("no-extension"), "application/octet-stream");
    }

    /// Percent-decoding feeds the allow-map only: encoded traversals decode faithfully (and then
    /// miss the map), malformed escapes are refused outright.
    #[test]
    fn percent_decoding_is_strict() {
        assert_eq!(percent_decode("/a%20b").as_deref(), Some("/a b"));
        assert_eq!(percent_decode("/%2e%2e/x").as_deref(), Some("/../x"));
        assert_eq!(percent_decode("/..%2fx").as_deref(), Some("/../x"));
        assert_eq!(percent_decode("/plain").as_deref(), Some("/plain"));
        assert!(percent_decode("/bad%zz").is_none(), "malformed escape");
        assert!(percent_decode("/bad%2").is_none(), "truncated escape");
        assert!(
            percent_decode("/nul%00").is_some(),
            "NUL decodes (and then misses the map)"
        );
    }

    /// The request-head parser: routing fields (method/path/version/headers) and the keep-alive +
    /// upgrade predicates.
    #[test]
    fn request_head_parses_and_routes() {
        let head = parse_request_head(
            b"GET /ws?x=1 HTTP/1.1\r\nHost: gui.local:8080\r\nConnection: keep-alive, Upgrade\r\nUpgrade: WebSocket\r\n\r\n",
        )
        .expect("parse");
        assert_eq!(head.method, "GET");
        assert_eq!(head.path(), "/ws");
        assert_eq!(head.header("host"), Some("gui.local:8080"));
        assert!(head.is_ws_upgrade(), "upgrade tokens are case-insensitive");
        assert!(head.keep_alive());

        let head = parse_request_head(b"GET /ws HTTP/1.1\r\nHost: h\r\n\r\n").expect("parse");
        assert!(
            !head.is_ws_upgrade(),
            "GET /ws without Upgrade is static (404)"
        );

        let head =
            parse_request_head(b"POST / HTTP/1.1\r\nContent-Length: 4\r\n\r\n").expect("parse");
        assert!(
            !head.keep_alive(),
            "an unread body forces connection: close"
        );

        let head = parse_request_head(b"GET / HTTP/1.0\r\n\r\n").expect("parse");
        assert!(!head.keep_alive(), "HTTP/1.0 has no default keep-alive");

        assert!(parse_request_head(b"NOT-HTTP\r\n\r\n").is_none());
        assert!(parse_request_head(b"GET / SPDY/3\r\n\r\n").is_none());
    }

    /// The Accept-Encoding token scan handles q-values and case, and never matches substrings.
    #[test]
    fn accept_encoding_tokens() {
        assert!(accepts_coding("gzip, br", "br"));
        assert!(accepts_coding("BR;q=0.9", "br"));
        assert!(!accepts_coding("abr, gzipx", "br"), "no substring matches");
        assert!(!accepts_coding("br;q=0", "br"));
        assert!(!accepts_coding("br;q=0.0", "br"));
        assert!(!accepts_coding("gzip", "br"));
        assert!(
            !accepts_coding("*", "br"),
            "wildcard is deliberately ignored"
        );
    }

    /// The rewind shim replays the consumed bytes, then continues on the live stream.
    #[tokio::test]
    async fn rewind_replays_then_reads_through() {
        let (client, server) = tokio::io::duplex(64);
        let mut client = client;
        client.write_all(b" live").await.expect("write");
        drop(client);
        let mut rewound = Rewind::new(b"replayed".to_vec(), server);
        let mut all = String::new();
        rewound.read_to_string(&mut all).await.expect("read");
        assert_eq!(all, "replayed live");
    }
}
