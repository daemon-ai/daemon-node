// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-egress` — the ONE SSRF-safe outbound HTTP client for node network tools.
//!
//! The OpenClaw CVE class this closes: a tool `check_url`s the *initial* URL once, then hands it to
//! an HTTP client that silently auto-follows redirects — so a public URL that `302`s to
//! `http://169.254.169.254/` or a loopback/RFC-1918 host bypasses the egress gate, and any
//! `Authorization`/`Cookie` header rides along across the origin change. This crate makes that form
//! unrepresentable: the inner [`reqwest::Client`] follows **no** redirects on its own
//! ([`reqwest::redirect::Policy::none`]); [`EgressClient`] follows them **manually**, re-validating
//! **every hop** with [`daemon_core::check_url`], dropping credential headers when the origin
//! changes, and capping the hop count.
//!
//! Redirect behaviour is a **surfaced, per-call** parameter ([`Redirects`]) — visible at each call
//! site rather than hidden in a client builder. Callers pick [`Redirects::FollowValidated`]
//! (browser-like, hop-capped, re-validated) or [`Redirects::None`] (never follow — for trusted,
//! non-redirecting peers whose host may legitimately be private/loopback and must therefore *not*
//! be run through the public-host gate).
//!
//! The *initial* URL is deliberately **not** re-checked here — that is the caller's pre-flight
//! decision (mirroring `daemon-tool-vision`): agent-facing callers gate it with `check_url`, while
//! an operator-configured peer on a private host is reached without tripping the public-host gate.
//! Only redirect *hops* are re-validated.

#![forbid(unsafe_code)]
// Phase 4 anchor: this crate IS the one sanctioned home for a raw `reqwest::Client`. The
// workspace-wide `disallowed_types` ban points every other crate here; the inner client is pinned
// to `redirect::Policy::none()` so redirects are followed manually and re-validated per hop.
#![allow(clippy::disallowed_types)]

use std::time::Duration;

use daemon_core::{check_url, UrlReject};
use reqwest::header::{
    HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE, COOKIE, LOCATION, PROXY_AUTHORIZATION,
};
use reqwest::{Method, StatusCode, Url};

/// Credential-bearing headers stripped when a validated redirect changes the origin — so a token
/// minted for origin A never leaks to origin B (the OpenClaw redirect credential-leak vector).
const CREDENTIAL_HEADERS: [reqwest::header::HeaderName; 3] =
    [AUTHORIZATION, COOKIE, PROXY_AUTHORIZATION];

/// The **surfaced** per-call redirect policy. This is the user-visible choice; the
/// [`reqwest::redirect::Policy::none`] on the inner client is just the plumbing that disables the
/// *silent library* auto-follow so this manual, re-validating loop can own the decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Redirects {
    /// Do not follow redirects. A `3xx` response is returned to the caller unchanged. Use for
    /// trusted, non-redirecting peers (e.g. an operator-configured sync server that may live on a
    /// private/LAN/loopback host). Refusing to follow kills both the redirect-SSRF and the
    /// credential-leak vectors without subjecting the peer's host to the public-host gate.
    None,
    /// Follow redirects browser-style, but **manually**: every hop's target is re-validated with
    /// [`daemon_core::check_url`] (reject on failure), credential headers are dropped when the
    /// origin changes, and at most `max_hops` redirects are followed.
    FollowValidated {
        /// The maximum number of redirects to follow before giving up.
        max_hops: usize,
    },
}

impl Redirects {
    /// The browser-like default: follow up to five validated hops (matches the vision tool's
    /// `MAX_REDIRECT_HOPS`).
    pub const DEFAULT: Redirects = Redirects::FollowValidated { max_hops: 5 };
}

/// Why an egress request failed.
#[derive(Debug, thiserror::Error)]
pub enum EgressError {
    /// A redirect hop's target was rejected by the egress policy (private/loopback/link-local/
    /// non-http(s)). Carries the underlying [`UrlReject`].
    #[error("egress blocked: {0}")]
    Blocked(#[from] UrlReject),
    /// A `3xx` response lacked a usable `Location` header.
    #[error("redirect without a usable Location header (status {0})")]
    BadRedirect(u16),
    /// The redirect chain exceeded the configured hop cap.
    #[error("too many redirects (limit {0})")]
    TooManyRedirects(usize),
    /// A URL (initial or a joined redirect target) could not be parsed.
    #[error("invalid url: {0}")]
    InvalidUrl(String),
    /// The request body could not be encoded.
    #[error("request body encode failed: {0}")]
    Encode(String),
    /// A transport/TLS failure issuing the request.
    #[error("request failed: {0}")]
    Transport(String),
}

/// Construction knobs for [`EgressClient`].
#[derive(Clone, Debug, Default)]
pub struct EgressConfig {
    /// The `User-Agent` sent on every request (a plain product identity). `None` uses reqwest's
    /// default.
    pub user_agent: Option<String>,
    /// A per-request deadline applied by the inner client. `None` leaves reqwest's default.
    pub timeout: Option<Duration>,
}

/// A request the [`EgressClient`] issues (and re-issues per hop). The body is retained so it can be
/// re-sent when a `307`/`308` preserves the method.
#[derive(Clone, Debug)]
pub struct EgressRequest {
    method: Method,
    url: String,
    headers: HeaderMap,
    body: Option<Vec<u8>>,
}

impl EgressRequest {
    /// A `GET` for `url`.
    pub fn get(url: impl Into<String>) -> Self {
        Self {
            method: Method::GET,
            url: url.into(),
            headers: HeaderMap::new(),
            body: None,
        }
    }

    /// A `POST` of `body` serialized as JSON (sets `Content-Type: application/json`).
    pub fn post_json<T: serde::Serialize>(
        url: impl Into<String>,
        body: &T,
    ) -> Result<Self, EgressError> {
        let bytes = serde_json::to_vec(body).map_err(|e| EgressError::Encode(e.to_string()))?;
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        Ok(Self {
            method: Method::POST,
            url: url.into(),
            headers,
            body: Some(bytes),
        })
    }

    /// Set a header (best-effort: an invalid name/value is ignored). Chainable.
    #[must_use]
    pub fn header(mut self, name: &str, value: &str) -> Self {
        if let (Ok(n), Ok(v)) = (
            reqwest::header::HeaderName::from_bytes(name.as_bytes()),
            HeaderValue::from_str(value),
        ) {
            self.headers.insert(n, v);
        }
        self
    }

    /// Set `Authorization: Bearer <token>` (dropped automatically on a cross-origin redirect).
    #[must_use]
    pub fn bearer_auth(mut self, token: &str) -> Self {
        if let Ok(v) = HeaderValue::from_str(&format!("Bearer {token}")) {
            self.headers.insert(AUTHORIZATION, v);
        }
        self
    }
}

/// The one SSRF-safe outbound HTTP client. Its inner [`reqwest::Client`] follows no redirects on its
/// own; [`EgressClient::execute`] performs the manual, per-hop-revalidated redirect loop. Raw
/// [`reqwest::Client`]s should be built nowhere else (a future clippy disallow-list enforces this).
#[derive(Clone)]
pub struct EgressClient {
    http: reqwest::Client,
}

impl EgressClient {
    /// Build a client. The inner reqwest client is pinned to
    /// [`reqwest::redirect::Policy::none`] so the library never silently auto-follows a redirect
    /// past the egress gate. Fails only when the TLS backend cannot initialize (a boot-environment
    /// defect) — surfaced rather than swapping in a default (redirect-following) client.
    pub fn new(cfg: EgressConfig) -> Result<Self, EgressError> {
        let mut builder = reqwest::Client::builder().redirect(reqwest::redirect::Policy::none());
        if let Some(ua) = cfg.user_agent {
            builder = builder.user_agent(ua);
        }
        if let Some(timeout) = cfg.timeout {
            builder = builder.timeout(timeout);
        }
        let http = builder
            .build()
            .map_err(|e| EgressError::Transport(e.to_string()))?;
        Ok(Self { http })
    }

    /// A convenience `GET`. The initial `url` is **not** re-checked here (the caller's pre-flight
    /// owns that); redirect hops are re-validated when `redirects` is
    /// [`Redirects::FollowValidated`].
    pub async fn get(
        &self,
        url: &str,
        redirects: Redirects,
    ) -> Result<reqwest::Response, EgressError> {
        self.execute(EgressRequest::get(url), redirects).await
    }

    /// Execute `req` under `redirects`, returning the final resolved response.
    ///
    /// Under [`Redirects::None`] a `3xx` is returned unchanged. Under
    /// [`Redirects::FollowValidated`] each hop's target is joined against the current URL,
    /// re-validated with [`daemon_core::check_url`] (a failure returns [`EgressError::Blocked`]),
    /// credential headers are dropped on an origin change, and the request method is rewritten per
    /// the redirect status (`301`/`302`/`303` on a `POST` → `GET` without body; `307`/`308`
    /// preserve method and body).
    pub async fn execute(
        &self,
        req: EgressRequest,
        redirects: Redirects,
    ) -> Result<reqwest::Response, EgressError> {
        let max_hops = match redirects {
            Redirects::None => 0,
            Redirects::FollowValidated { max_hops } => max_hops,
        };

        let mut url = Url::parse(&req.url).map_err(|e| EgressError::InvalidUrl(e.to_string()))?;
        let mut method = req.method;
        let mut headers = req.headers;
        let mut body = req.body;

        for hop in 0..=max_hops {
            let mut builder = self.http.request(method.clone(), url.clone());
            builder = builder.headers(headers.clone());
            if let Some(bytes) = &body {
                builder = builder.body(bytes.clone());
            }
            let resp = builder
                .send()
                .await
                .map_err(|e| EgressError::Transport(e.to_string()))?;
            let status = resp.status();

            if !status.is_redirection() {
                return Ok(resp);
            }
            // A 3xx. Under `None` we hand the redirect back to the caller unfollowed.
            if let Redirects::None = redirects {
                return Ok(resp);
            }
            if hop == max_hops {
                return Err(EgressError::TooManyRedirects(max_hops));
            }

            let location = resp
                .headers()
                .get(LOCATION)
                .and_then(|v| v.to_str().ok())
                .ok_or_else(|| EgressError::BadRedirect(status.as_u16()))?;
            let next = resolve_next_hop(&url, location)?;
            if !same_origin(&url, &next) {
                strip_credentials(&mut headers);
            }
            rewrite_method_for_redirect(status, &mut method, &mut body);
            url = next;
        }

        Err(EgressError::TooManyRedirects(max_hops))
    }
}

/// Join a redirect `location` (absolute or relative) against the current URL and re-validate the
/// target against the egress policy — the per-hop SSRF guard. Every followed hop passes through
/// here (mirrors the vision tool's `next_hop`).
fn resolve_next_hop(current: &Url, location: &str) -> Result<Url, EgressError> {
    let next = current
        .join(location)
        .map_err(|e| EgressError::InvalidUrl(format!("invalid redirect location: {e}")))?;
    check_url(next.as_str())?;
    Ok(next)
}

/// Whether two URLs share an origin (scheme + host + effective port). `http`/`https` default ports
/// (80/443) are normalized so `http://h/` and `http://h:80/` compare equal.
fn same_origin(a: &Url, b: &Url) -> bool {
    a.scheme() == b.scheme()
        && a.host_str() == b.host_str()
        && a.port_or_known_default() == b.port_or_known_default()
}

/// Remove every credential-bearing header (used when a redirect crosses an origin boundary).
fn strip_credentials(headers: &mut HeaderMap) {
    for name in CREDENTIAL_HEADERS {
        headers.remove(&name);
    }
}

/// Rewrite the request method after a redirect: `301`/`302`/`303` demote a `POST` to a bodyless
/// `GET` (browser behaviour); `307`/`308` preserve the method and body.
fn rewrite_method_for_redirect(
    status: StatusCode,
    method: &mut Method,
    body: &mut Option<Vec<u8>>,
) {
    match status {
        StatusCode::MOVED_PERMANENTLY | StatusCode::FOUND | StatusCode::SEE_OTHER
            if *method == Method::POST =>
        {
            *method = Method::GET;
            *body = None;
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[test]
    fn next_hop_joins_and_revalidates_redirect_targets() {
        // Absolute public target: allowed.
        let base = Url::parse("https://example.com/a").unwrap();
        assert_eq!(
            resolve_next_hop(&base, "https://cdn.example.net/img.png")
                .unwrap()
                .as_str(),
            "https://cdn.example.net/img.png"
        );
        // Relative target joins against the current URL.
        let base = Url::parse("https://example.com/dir/a").unwrap();
        assert_eq!(
            resolve_next_hop(&base, "b.png").unwrap().as_str(),
            "https://example.com/dir/b.png"
        );
        // Redirects into private / loopback / metadata space are rejected mid-chain.
        let base = Url::parse("https://example.com/a").unwrap();
        for target in [
            "http://169.254.169.254/latest/meta-data/",
            "http://localhost:8080/x",
            "http://10.0.0.5/x",
            "http://[::1]/x",
        ] {
            assert!(
                matches!(
                    resolve_next_hop(&base, target),
                    Err(EgressError::Blocked(_))
                ),
                "expected {target} to be blocked"
            );
        }
        // A scheme downgrade to a non-http scheme is rejected by the same policy.
        assert!(matches!(
            resolve_next_hop(&base, "file:///etc/passwd"),
            Err(EgressError::Blocked(_))
        ));
    }

    #[test]
    fn same_origin_normalizes_default_ports() {
        let a = Url::parse("http://host/x").unwrap();
        let b = Url::parse("http://host:80/y").unwrap();
        assert!(same_origin(&a, &b), "default http port is equivalent");

        let https = Url::parse("https://host/x").unwrap();
        assert!(!same_origin(&a, &https), "scheme change is a new origin");

        let other_host = Url::parse("http://evil/x").unwrap();
        assert!(!same_origin(&a, &other_host), "host change is a new origin");

        let other_port = Url::parse("http://host:8080/x").unwrap();
        assert!(!same_origin(&a, &other_port), "port change is a new origin");
    }

    #[test]
    fn credentials_dropped_only_on_cross_origin() {
        let make_headers = || {
            let mut h = HeaderMap::new();
            h.insert(AUTHORIZATION, HeaderValue::from_static("Bearer secret"));
            h.insert(COOKIE, HeaderValue::from_static("sid=abc"));
            h.insert(PROXY_AUTHORIZATION, HeaderValue::from_static("Basic zzz"));
            h
        };

        // Same origin: nothing stripped (the caller would not call strip in this case).
        let a = Url::parse("https://host/a").unwrap();
        let same = Url::parse("https://host/b").unwrap();
        assert!(same_origin(&a, &same));
        let kept = make_headers();
        assert!(kept.contains_key(AUTHORIZATION));

        // Cross origin: every credential header is removed.
        let cross = Url::parse("https://evil/b").unwrap();
        assert!(!same_origin(&a, &cross));
        let mut stripped = make_headers();
        strip_credentials(&mut stripped);
        assert!(!stripped.contains_key(AUTHORIZATION));
        assert!(!stripped.contains_key(COOKIE));
        assert!(!stripped.contains_key(PROXY_AUTHORIZATION));
    }

    #[test]
    fn method_rewrite_follows_browser_rules() {
        // 303 on a POST -> GET without body.
        let mut m = Method::POST;
        let mut b = Some(vec![1u8, 2, 3]);
        rewrite_method_for_redirect(StatusCode::SEE_OTHER, &mut m, &mut b);
        assert_eq!(m, Method::GET);
        assert!(b.is_none());

        // 302 on a POST -> GET without body.
        let mut m = Method::POST;
        let mut b = Some(vec![9u8]);
        rewrite_method_for_redirect(StatusCode::FOUND, &mut m, &mut b);
        assert_eq!(m, Method::GET);
        assert!(b.is_none());

        // 307 preserves method + body.
        let mut m = Method::POST;
        let mut b = Some(vec![7u8]);
        rewrite_method_for_redirect(StatusCode::TEMPORARY_REDIRECT, &mut m, &mut b);
        assert_eq!(m, Method::POST);
        assert_eq!(b, Some(vec![7u8]));
    }

    fn client() -> EgressClient {
        EgressClient::new(EgressConfig::default()).expect("build egress client")
    }

    /// The core repro: a public (here loopback-mock) URL that `302`s into link-local metadata space
    /// is rejected mid-chain instead of being followed. Pre-fix, the library auto-followed the hop.
    #[tokio::test]
    async fn redirect_to_blocked_host_is_rejected_midchain() {
        for target in [
            "http://169.254.169.254/latest/meta-data/",
            "http://127.0.0.1:9/secret",
            "http://10.0.0.5/x",
            "http://[::1]/x",
        ] {
            let server = MockServer::start().await;
            Mock::given(method("GET"))
                .and(path("/start"))
                .respond_with(ResponseTemplate::new(302).insert_header("location", target))
                .mount(&server)
                .await;

            let err = client()
                .get(&format!("{}/start", server.uri()), Redirects::DEFAULT)
                .await
                .expect_err("redirect into blocked space must be rejected");
            assert!(
                matches!(err, EgressError::Blocked(_)),
                "target {target}: expected Blocked, got {err:?}"
            );
        }
    }

    /// A plain `200` (no redirect) is returned unchanged.
    #[tokio::test]
    async fn non_redirect_response_is_returned() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/ok"))
            .respond_with(ResponseTemplate::new(200).set_body_string("hello"))
            .mount(&server)
            .await;

        let resp = client()
            .get(&format!("{}/ok", server.uri()), Redirects::DEFAULT)
            .await
            .expect("200 returned");
        assert_eq!(resp.status().as_u16(), 200);
        assert_eq!(resp.text().await.unwrap(), "hello");
    }

    /// Under `Redirects::None`, a `3xx` is handed back to the caller unfollowed and the redirect
    /// target is never requested.
    #[tokio::test]
    async fn redirects_none_returns_3xx_unfollowed() {
        let server = MockServer::start().await;
        // Only /start is mounted; if the client followed the redirect it would 404 (or, for a
        // public target, leave the mock) — either way we assert we got the 302 back verbatim.
        Mock::given(method("GET"))
            .and(path("/start"))
            .respond_with(
                ResponseTemplate::new(302).insert_header("location", "https://example.com/next"),
            )
            .mount(&server)
            .await;

        let resp = client()
            .get(&format!("{}/start", server.uri()), Redirects::None)
            .await
            .expect("None returns the 3xx");
        assert_eq!(resp.status().as_u16(), 302);
        assert_eq!(
            resp.headers().get("location").and_then(|v| v.to_str().ok()),
            Some("https://example.com/next")
        );
    }
}
