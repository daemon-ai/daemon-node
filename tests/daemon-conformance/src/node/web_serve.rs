// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! THE SINGLE-ORIGIN WEB GATE: one `[web].addr`-style listener serving the Qt WASM app bundle as
//! static files AND the same mux-over-WebSocket carrier on `GET /ws` — proven over a real TCP
//! listener with a raw HTTP/1.1 client (so nothing normalizes the traversal attempts away) and
//! the same pinned `WsMuxClient` the standalone-listener gate uses:
//!
//! - `/` serves `daemon-app.html`; the bundle's Content-Types are exact (`application/wasm` is
//!   required for streaming compilation); HEAD mirrors GET without a body; unknown paths 404;
//!   non-GET/HEAD methods 405;
//! - `Accept-Encoding` negotiation serves precompressed `.br`/`.gz` siblings with
//!   `Content-Encoding`, the *underlying* Content-Type, and `Vary: Accept-Encoding`;
//! - path traversal — plain, percent-encoded, nested — can never leave the bundle directory (the
//!   allow-map design: a sentinel file OUTSIDE the root stays unreachable);
//! - a same-origin WebSocket upgrade on `/ws` (Origin == the listener's own origin, derived from
//!   `Host`) completes subprotocol + Hello + SCRAM + Health with ZERO origin configuration;
//!   a cross-origin upgrade is refused with 403 unless allow-listed via the extra origins.

use super::harness::*;
use super::ws_transport::WsMuxClient;
use daemon_auth::{AuthStore, Role};
use daemon_host::{Authenticator, WebRoot};
use std::collections::HashMap;
use std::path::Path;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio_tungstenite::tungstenite::Error as WsError;

/// The fabricated app bundle (the Qt wasm installer's flat layout, tiny placeholder bytes).
const BUNDLE: &[(&str, &[u8])] = &[
    ("daemon-app.html", b"<html>daemon-app</html>"),
    ("daemon-app.js", b"export {};"),
    ("daemon-app.wasm", b"\0asm-fake-wasm-bytes"),
    ("daemon-app.data", b"qt-packed-assets"),
    ("qtloader.js", b"export {};"),
];

/// Write `files` (flat placeholder names -> bytes) into `dir`.
fn write_bundle(dir: &Path, files: &[(&str, &[u8])]) {
    for (name, contents) in files {
        std::fs::write(dir.join(name), contents).expect("write bundle file");
    }
}

/// A node + seeded authenticator serving the single-origin web front over `root` on an ephemeral
/// port with `allowed_origins` as the extra (cross-origin) allowance. Mirrors `serve_ws` in the
/// WebSocket-carrier gate.
async fn serve_root(
    root: &Path,
    allowed_origins: &[&str],
) -> (
    std::net::SocketAddr,
    tokio::task::JoinHandle<()>,
    daemon_host::SupervisorHandle,
) {
    let (node, handle) = assemble();
    let store = Arc::new(AuthStore::open_in_memory().expect("auth store"));
    store
        .create_user("operator", "op-pw", &[Role::Operator])
        .expect("create operator");
    let auth = Arc::new(Authenticator::new(store));
    let site = WebRoot::scan(root).expect("scan web root");
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("local addr");
    let server = tokio::spawn(daemon_host::serve_web(
        listener,
        site,
        node,
        auth,
        allowed_origins.iter().map(|s| s.to_string()).collect(),
    ));
    (addr, server, handle)
}

/// Send one raw HTTP/1.1 request (the caller writes the request head verbatim, so traversal
/// shapes reach the server unnormalized) and read the response to EOF.
async fn http_exchange(
    addr: std::net::SocketAddr,
    raw_request: &str,
) -> (u16, HashMap<String, String>, Vec<u8>) {
    let mut stream = TcpStream::connect(addr).await.expect("connect");
    stream
        .write_all(raw_request.as_bytes())
        .await
        .expect("send request");
    let mut response = Vec::new();
    stream
        .read_to_end(&mut response)
        .await
        .expect("read response");
    let head_end = response
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .expect("response head")
        + 4;
    let head = std::str::from_utf8(&response[..head_end]).expect("utf8 head");
    let mut lines = head.split("\r\n");
    let status: u16 = lines
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|code| code.parse().ok())
        .expect("status code");
    let mut headers = HashMap::new();
    for line in lines {
        if let Some((name, value)) = line.split_once(':') {
            headers.insert(name.trim().to_ascii_lowercase(), value.trim().to_string());
        }
    }
    (status, headers, response[head_end..].to_vec())
}

/// `GET target` with optional extra header lines (each `\r\n`-terminated), `Connection: close`.
async fn get(
    addr: std::net::SocketAddr,
    target: &str,
    extra_headers: &str,
) -> (u16, HashMap<String, String>, Vec<u8>) {
    let raw = format!(
        "GET {target} HTTP/1.1\r\nHost: {addr}\r\n{extra_headers}Connection: close\r\n\r\n"
    );
    http_exchange(addr, &raw).await
}

/// Read exactly one `Content-Length`-delimited response off a kept-alive connection.
async fn read_one_response(stream: &mut TcpStream) -> (u16, Vec<u8>) {
    let mut buf = Vec::new();
    let head_end = loop {
        if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break pos + 4;
        }
        let mut chunk = [0u8; 4096];
        let n = stream.read(&mut chunk).await.expect("read head");
        assert!(n > 0, "connection must stay open mid-response");
        buf.extend_from_slice(&chunk[..n]);
    };
    let head = std::str::from_utf8(&buf[..head_end]).expect("utf8 head");
    let status: u16 = head
        .split_whitespace()
        .nth(1)
        .and_then(|code| code.parse().ok())
        .expect("status code");
    let len: usize = head
        .lines()
        .find_map(|l| {
            l.to_ascii_lowercase()
                .strip_prefix("content-length:")
                .map(str::trim)
                .map(String::from)
        })
        .and_then(|v| v.parse().ok())
        .expect("content-length");
    while buf.len() < head_end + len {
        let mut chunk = [0u8; 4096];
        let n = stream.read(&mut chunk).await.expect("read body");
        assert!(n > 0, "connection must stay open mid-body");
        buf.extend_from_slice(&chunk[..n]);
    }
    (status, buf[head_end..head_end + len].to_vec())
}

/// Index + the bundle's Content-Types, HEAD parity, unknown-path 404, non-GET/HEAD 405.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn web_serves_the_bundle_with_exact_content_types() {
    let dir = tempfile::tempdir().expect("tempdir");
    write_bundle(dir.path(), BUNDLE);
    let (addr, server, handle) = serve_root(dir.path(), &[]).await;

    // `/` is the app page (text/html), byte-identical to daemon-app.html.
    let (status, headers, body) = get(addr, "/", "").await;
    assert_eq!(status, 200);
    assert_eq!(
        headers.get("content-type").map(String::as_str),
        Some("text/html; charset=utf-8")
    );
    assert_eq!(body, b"<html>daemon-app</html>");

    // The four bundle-critical Content-Types (wasm's is required for streaming compilation).
    for (target, want_type) in [
        ("/daemon-app.html", "text/html; charset=utf-8"),
        ("/daemon-app.js", "application/javascript"),
        ("/qtloader.js", "application/javascript"),
        ("/daemon-app.wasm", "application/wasm"),
        ("/daemon-app.data", "application/octet-stream"),
    ] {
        let (status, headers, _) = get(addr, target, "").await;
        assert_eq!(status, 200, "{target} must serve");
        assert_eq!(
            headers.get("content-type").map(String::as_str),
            Some(want_type),
            "{target} content-type"
        );
    }

    // HEAD: the GET headers (including the true length) without a body.
    let (status, headers, body) = http_exchange(
        addr,
        &format!("HEAD /daemon-app.wasm HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n"),
    )
    .await;
    assert_eq!(status, 200);
    assert_eq!(
        headers.get("content-length").map(String::as_str),
        Some("20"),
        "HEAD must carry the entity length"
    );
    assert!(body.is_empty(), "HEAD must not carry a body");

    // Unknown paths 404; methods other than GET/HEAD 405.
    let (status, _, _) = get(addr, "/not-in-the-bundle.js", "").await;
    assert_eq!(status, 404);
    let (status, headers, _) = http_exchange(
        addr,
        &format!("POST / HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n"),
    )
    .await;
    assert_eq!(status, 405, "the static surface is GET/HEAD only");
    assert_eq!(headers.get("allow").map(String::as_str), Some("GET, HEAD"));

    server.abort();
    handle.shutdown().await;
}

/// HTTP/1.1 keep-alive: a browser fetches the whole bundle over one connection — sequential
/// requests on the same socket each get their own correct response.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn web_serves_sequential_requests_on_one_connection() {
    let dir = tempfile::tempdir().expect("tempdir");
    write_bundle(dir.path(), BUNDLE);
    let (addr, server, handle) = serve_root(dir.path(), &[]).await;

    let mut stream = TcpStream::connect(addr).await.expect("connect");
    for (target, want_body) in [
        ("/", &b"<html>daemon-app</html>"[..]),
        ("/qtloader.js", b"export {};"),
        ("/daemon-app.wasm", b"\0asm-fake-wasm-bytes"),
    ] {
        stream
            .write_all(format!("GET {target} HTTP/1.1\r\nHost: {addr}\r\n\r\n").as_bytes())
            .await
            .expect("send request");
        let (status, body) = read_one_response(&mut stream).await;
        assert_eq!(status, 200, "{target} over the kept-alive connection");
        assert_eq!(body, want_body, "{target} body");
    }

    server.abort();
    handle.shutdown().await;
}

/// `Accept-Encoding` negotiation onto precompressed siblings: brotli preferred, gzip next,
/// identity always available — with `Content-Encoding`, the underlying Content-Type, and
/// `Vary: Accept-Encoding` on every representation of a file that has variants.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn web_negotiates_precompressed_siblings() {
    let dir = tempfile::tempdir().expect("tempdir");
    write_bundle(dir.path(), BUNDLE);
    // The size-optimization branch ships `.br`/`.gz` siblings of the big files.
    write_bundle(
        dir.path(),
        &[
            ("daemon-app.wasm.br", b"brotli-compressed-wasm"),
            ("daemon-app.wasm.gz", b"gzip-compressed-wasm"),
        ],
    );
    let (addr, server, handle) = serve_root(dir.path(), &[]).await;

    // A brotli-capable browser gets the .br sibling, typed as the UNDERLYING wasm.
    let (status, headers, body) =
        get(addr, "/daemon-app.wasm", "Accept-Encoding: gzip, br\r\n").await;
    assert_eq!(status, 200);
    assert_eq!(body, b"brotli-compressed-wasm");
    assert_eq!(
        headers.get("content-encoding").map(String::as_str),
        Some("br")
    );
    assert_eq!(
        headers.get("content-type").map(String::as_str),
        Some("application/wasm"),
        "the encoded representation must keep the underlying Content-Type"
    );
    assert_eq!(
        headers.get("vary").map(String::as_str),
        Some("accept-encoding"),
        "a negotiated resource must declare Vary"
    );

    // gzip-only negotiates the .gz sibling; a q=0 brotli is a refusal.
    let (_, headers, body) = get(addr, "/daemon-app.wasm", "Accept-Encoding: gzip\r\n").await;
    assert_eq!(body, b"gzip-compressed-wasm");
    assert_eq!(
        headers.get("content-encoding").map(String::as_str),
        Some("gzip")
    );
    let (_, _, body) = get(
        addr,
        "/daemon-app.wasm",
        "Accept-Encoding: br;q=0, gzip\r\n",
    )
    .await;
    assert_eq!(body, b"gzip-compressed-wasm", "q=0 must disable a coding");

    // No Accept-Encoding: the identity bytes, still marked Vary (the response depends on it).
    let (_, headers, body) = get(addr, "/daemon-app.wasm", "").await;
    assert_eq!(body, b"\0asm-fake-wasm-bytes");
    assert!(!headers.contains_key("content-encoding"));
    assert_eq!(
        headers.get("vary").map(String::as_str),
        Some("accept-encoding")
    );

    // A file without siblings negotiates nothing (and the variants are not pages of their own —
    // the loader requests `daemon-app.wasm`, never the `.br` name).
    let (_, headers, _) = get(addr, "/daemon-app.html", "Accept-Encoding: br, gzip\r\n").await;
    assert!(!headers.contains_key("content-encoding"));
    assert!(!headers.contains_key("vary"));
    let (status, _, _) = get(addr, "/daemon-app.wasm.br", "").await;
    assert_eq!(
        status, 404,
        "precompressed variants are not directly addressable"
    );

    server.abort();
    handle.shutdown().await;
}

/// Path traversal is impossible: a sentinel file OUTSIDE the served root stays unreachable
/// through every traversal shape — plain `..`, percent-encoded dots and slashes, nested inside a
/// valid prefix, doubled slashes. The raw client sends the attack targets verbatim (an HTTP URL
/// client would normalize them away before they ever hit the server).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn web_path_traversal_is_impossible() {
    let outer = tempfile::tempdir().expect("tempdir");
    std::fs::write(outer.path().join("secret.txt"), b"TOP-SECRET").expect("write sentinel");
    let root = outer.path().join("bundle");
    std::fs::create_dir(&root).expect("mkdir bundle");
    write_bundle(&root, BUNDLE);
    let (addr, server, handle) = serve_root(&root, &[]).await;

    for target in [
        "/../secret.txt",
        "/%2e%2e/secret.txt",
        "/..%2fsecret.txt",
        "/%2e%2e%2fsecret.txt",
        "/daemon-app.html/../../secret.txt",
        "//../secret.txt",
        "/./../secret.txt",
        "/..",
        "/../../../../etc/passwd",
        "/..%252fsecret.txt", // double-encoded: decodes to the literal `..%2fsecret.txt`, no file
    ] {
        let (status, _, body) = get(addr, target, "").await;
        assert!(
            status == 404 || status == 400,
            "{target} must be refused, got {status}"
        );
        assert!(
            !body.windows(10).any(|w| w == b"TOP-SECRET"),
            "{target} must never leak bytes from outside the root"
        );
    }

    // The map still serves the real bundle after all that.
    let (status, _, _) = get(addr, "/daemon-app.html", "").await;
    assert_eq!(status, 200);

    server.abort();
    handle.shutdown().await;
}

/// The single-origin promise: a browser page loaded from this listener upgrades `/ws` with its
/// own origin (derived server-side from `Host`) and NO origin configuration, then completes the
/// full pinned mux flow — subprotocol echo, Hello, SCRAM-SHA-256, an authenticated Health call.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn web_ws_same_origin_upgrade_serves_the_mux() {
    let dir = tempfile::tempdir().expect("tempdir");
    write_bundle(dir.path(), BUNDLE);
    let (addr, server, handle) = serve_root(dir.path(), &[]).await;

    // What a browser sends for a page served from http://{addr}: Origin == that very origin.
    let mut client =
        WsMuxClient::connect_url(&format!("ws://{addr}/ws"), Some(&format!("http://{addr}")))
            .await
            .expect("same-origin upgrade must pass with zero configuration");
    let view = client
        .authenticate_scram("operator", "op-pw")
        .await
        .expect("scram over the single-origin /ws");
    assert_eq!(view.username, "operator");
    let res = client.call(ApiRequest::Health).await.expect("health call");
    assert!(
        !matches!(res, ApiResponse::Error(_)),
        "an authenticated same-origin browser client must be served, got {res:?}"
    );

    server.abort();
    handle.shutdown().await;
}

/// The cross-origin gate on `/ws`: a foreign browser origin is refused with 403 at the upgrade
/// unless `[api].ws_allowed_origins` grants it; non-browser clients (no Origin) still connect;
/// and a plain GET of `/ws` (no upgrade) is just an unknown static path.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn web_ws_cross_origin_upgrades_are_gated() {
    let dir = tempfile::tempdir().expect("tempdir");
    write_bundle(dir.path(), BUNDLE);

    // Zero extra allowance: only the listener's own origin passes.
    let (addr, server, handle) = serve_root(dir.path(), &[]).await;
    match WsMuxClient::connect_url(&format!("ws://{addr}/ws"), Some("https://evil.example.com"))
        .await
    {
        Err(e) => match *e {
            WsError::Http(response) => assert_eq!(
                response.status(),
                403,
                "a foreign Origin must be refused with 403"
            ),
            other => panic!("expected an HTTP 403 refusal, got {other}"),
        },
        Ok(_) => panic!("a foreign Origin must not complete the upgrade"),
    }
    let mut headless = WsMuxClient::connect_url(&format!("ws://{addr}/ws"), None)
        .await
        .expect("a non-browser client (no Origin) connects");
    headless
        .authenticate_scram("operator", "op-pw")
        .await
        .expect("scram without an Origin header");
    let (status, _, _) = get(addr, "/ws", "").await;
    assert_eq!(status, 404, "a non-upgrade GET /ws is not a page");
    server.abort();
    handle.shutdown().await;

    // The extra allow-list grants a deliberate cross-origin GUI host.
    let (addr, server, handle) = serve_root(dir.path(), &["https://gui.example.com"]).await;
    let mut cross =
        WsMuxClient::connect_url(&format!("ws://{addr}/ws"), Some("https://gui.example.com"))
            .await
            .expect("an allow-listed cross origin connects");
    cross
        .authenticate_scram("operator", "op-pw")
        .await
        .expect("scram over the allow-listed cross-origin upgrade");
    let res = cross.call(ApiRequest::Health).await.expect("health call");
    assert!(!matches!(res, ApiResponse::Error(_)));
    server.abort();
    handle.shutdown().await;
}
