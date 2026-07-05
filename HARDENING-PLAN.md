# Cluster D — Shared SSRF egress client (Phase 1) — implementation plan

Branch: `hardening/egress-client` · worktree: `/home/j/experiments/daemon-worktrees/egress-client`

Status: **PHASE 1 — plan only.** No source changed, nothing committed. Awaiting coordinator review.

Guiding principle (from the master plan): *make the unsafe form unrepresentable.* Today three
network callers `check_url` once (or not at all) and then hand the URL to an HTTP client that
auto-follows redirects with no per-hop re-check — so a `302 → http://169.254.169.254/` (or any
loopback / RFC-1918 host) slips straight past the SSRF gate, and a bearer token rides across an
origin change. The vision tool already does it right (manual, re-validated per hop). This track
lifts that proven shape into ONE shared client and migrates the offenders onto it.

---

## 1. The bug, per caller (current behaviour, file:line)

### 1a. `web_extract` (agent-facing, untrusted model input)
- Tool pre-flight checks the **initial** URL once:
  `tools/daemon-tool-web/src/extract_tool.rs:75` — `check_url(&args.url)`.
- The local backend builds a reqwest client with the **default redirect policy** (reqwest
  auto-follows up to 10 hops):
  `tools/daemon-tool-web/src/local.rs:30-34` — `reqwest::Client::builder().user_agent(ua).build().unwrap_or_default()`.
- The fetch auto-follows redirects with **no re-check**:
  `tools/daemon-tool-web/src/local.rs:51-56` — `self.http.get(url).send().await`.
- ⇒ A public page that 302s to `http://169.254.169.254/latest/meta-data/` is fetched and its body
  returned to the model. **SSRF bypass.**
- (The `firecrawl` backend, `tools/daemon-tool-web/src/firecrawl.rs:32,72-79`, POSTs the URL to a
  fixed trusted endpoint `api.firecrawl.dev`; the page fetch happens server-side, so there is no
  client-side redirect SSRF there. Out of scope — see §7.)

### 1b. `browser` (agent-facing, untrusted model input)
- Tool pre-flight checks the **initial** URL once:
  `tools/daemon-tool-browser/src/tool.rs:203` — `check_url(url)`.
- Navigation redirects are followed **inside Chromium**, not by an HTTP library:
  `tools/daemon-tool-browser/src/supervisor.rs:99` — `page.goto(url)`, then
  `wait_for_navigation()` and `page.url()` (`supervisor.rs:102-109`).
- ⇒ A redirect to a blocked host is followed by the browser with no re-check. **SSRF bypass.**
- **Important correction to the master plan:** the browser does *not* use "the HTTP library"; it
  uses Chromium's own network stack over CDP. The reqwest `EgressClient` **cannot** intercept
  Chromium's redirects. The browser fix therefore reuses the same *primitive* (`check_url`) but
  through a different *mechanism* (CDP request interception), not the shared reqwest client. See §5.

### 1c. `daemon-mnemosyne` sync (operator-trusted config, not agent input)
- `SyncEngine::http_post` builds a reqwest client with the **default redirect policy** and attaches
  the sync bearer token:
  `crates/memory/daemon-mnemosyne/src/sync/mod.rs:921-931` —
  `reqwest::Client::builder().timeout(30s).build()`, `client.post(&url).json(body)`,
  `req.bearer_auth(key)`.
- The send auto-follows redirects, carrying the bearer across any origin change:
  `crates/memory/daemon-mnemosyne/src/sync/mod.rs:932` — `req.send().await`.
- ⇒ A redirecting / hostile sync server can (a) bounce the client into SSRF and (b) **exfiltrate
  the `sync_token`** to another origin. (`sync/mod.rs:914-945`, `http_post`; callers `sync_with`
  at `:987` and `:1028`.)
- **Trust nuance:** the remote is operator-configured (`MnemosyneConfig.sync_remote`) and may be a
  **private/LAN/loopback** host by design — so it must **not** be run through `check_url`'s
  public-host gate. The right hardening here is to **refuse to follow redirects at all**
  (`Redirects::None`): a trusted sync peer needs no client redirect, and refusing eliminates both
  the redirect-SSRF and the token-leak vectors while preserving private-network sync.

### 1d. The reference done right — `daemon-tool-vision` (reuse this shape)
- No-redirect inner client: `tools/daemon-tool-vision/src/lib.rs:114-132`
  (`reqwest::redirect::Policy::none()`).
- Manual, per-hop-revalidated fetch loop: `.../lib.rs:141-201` (`fetch_image`, `0..=MAX_REDIRECT_HOPS`).
- Per-hop join + `check_url`: `.../lib.rs:498-507` (`next_hop`).
- Initial URL checked by the caller, not the loop: `.../lib.rs:213-216` (`analyze`).
- Pure-function test to mirror: `.../lib.rs:539-570` (`next_hop_joins_and_revalidates_redirect_targets`).

---

## 2. Where the shared client lives + why

**New crate: `daemon-egress`** at `crates/engine/daemon-egress/` (sibling of `daemon-core`, which
owns `check_url`). Rationale:

- `check_url` lives in `daemon-core` and is re-exported at `daemon_core::{check_url, CheckedUrl,
  UrlReject}` (`crates/engine/daemon-core/src/lib.rs:94`, `.../safety/mod.rs:12`).
- `daemon-core` is deliberately **dependency-free / lean** — its `safety/url.rs` header says so
  ("dependency-free ... so the core engine crate stays lean"), and `daemon-core` does **not**
  currently depend on `reqwest`. Adding `reqwest` (rustls + aws-lc-rs) to `daemon-core` would bloat
  compile for the whole tree. So the HTTP client goes in its own crate that depends on
  `daemon-core` for the primitive.
- A single dedicated crate is exactly the "egress module" the master plan's **Phase 4** lint targets
  ("ban raw `reqwest::Client` outside the egress module"). One home = one lintable boundary.
- Dependency direction stays acyclic: `daemon-egress → daemon-core`; consumers
  (`daemon-tool-web`, `daemon-tool-vision`, `daemon-tool-browser`*, `daemon-mnemosyne`) →
  `daemon-egress`. (*browser reuses only `check_url`, not the client — see §5.)
- Workspace wiring: `members = ["crates/*/*", ...]` (root `Cargo.toml:3`) auto-includes the new
  crate; add one line to `[workspace.dependencies]` (`daemon-egress = { path = "crates/engine/daemon-egress" }`).
- No new third-party crate enters the tree: `reqwest` is already a workspace dep
  (`Cargo.toml:71`, features `json,stream,rustls-tls`), so `cargo deny` sees nothing new.

*(Location is coordinator-adjustable — `crates/substrate/daemon-egress` is an equally valid home
since it is shared network infrastructure. I default to `engine` for proximity to `check_url`.)*

---

## 3. API of `daemon-egress` (with the SURFACED `Redirects` policy)

```rust
// crates/engine/daemon-egress/src/lib.rs
#![forbid(unsafe_code)]

use std::time::Duration;
use daemon_core::{check_url, UrlReject};
use reqwest::header::{HeaderMap, HeaderName, AUTHORIZATION, COOKIE, PROXY_AUTHORIZATION};

/// SURFACED, per-CALL redirect policy — the choice is visible at every call site, never hidden
/// in a client builder. This is the user-visible policy; `Policy::none()` on the inner reqwest
/// client is just plumbing that disables the *silent library* auto-follow.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Redirects {
    /// Do not follow redirects. A 3xx is returned to the caller unchanged. Use for trusted,
    /// non-redirecting peers (e.g. an operator-configured Mnemosyne sync server on a private host).
    None,
    /// Follow redirects browser-style, but MANUALLY: every hop's target is re-validated with
    /// `check_url` (reject on failure), credential headers are dropped when the origin changes,
    /// and at most `max_hops` are followed.
    FollowValidated { max_hops: usize },
}

impl Redirects {
    /// Browser-like default: follow up to 5 validated hops (matches vision's MAX_REDIRECT_HOPS).
    pub const DEFAULT: Redirects = Redirects::FollowValidated { max_hops: 5 };
}

/// Credential headers stripped when a validated redirect changes the origin.
const CREDENTIAL_HEADERS: [HeaderName; 3] = [AUTHORIZATION, COOKIE, PROXY_AUTHORIZATION];

#[derive(Debug, thiserror::Error)]
pub enum EgressError {
    #[error("egress blocked: {0}")]
    Blocked(#[from] UrlReject),          // a redirect hop failed check_url
    #[error("redirect without a usable Location header (status {0})")]
    BadRedirect(u16),
    #[error("too many redirects (limit {0})")]
    TooManyRedirects(usize),
    #[error("invalid url: {0}")]
    InvalidUrl(String),
    #[error("request failed: {0}")]
    Transport(String),
}

/// Construction knobs (user agent + timeout).
#[derive(Clone, Debug, Default)]
pub struct EgressConfig {
    pub user_agent: Option<String>,
    pub timeout: Option<Duration>,
}

/// The ONE SSRF-safe outbound HTTP client. Inner reqwest client follows NO redirects on its own;
/// redirect handling is manual + re-validated here. Build raw reqwest clients nowhere else.
#[derive(Clone)]
pub struct EgressClient {
    http: reqwest::Client,
}

/// A pending request the client (re-)issues per hop. Body is retained so it can be re-sent.
pub struct EgressRequest {
    method: reqwest::Method,
    url: String,
    headers: HeaderMap,
    body: Option<Vec<u8>>,
}

impl EgressRequest {
    pub fn get(url: impl Into<String>) -> Self;
    pub fn post_json<T: serde::Serialize>(url: impl Into<String>, body: &T) -> Result<Self, EgressError>;
    pub fn header(self, name: &str, value: &str) -> Self;
    pub fn bearer_auth(self, token: &str) -> Self; // sets Authorization: Bearer <token>
}

impl EgressClient {
    /// Build the client. Inner reqwest client uses `redirect::Policy::none()` so the library never
    /// silently auto-follows. Fails only if the TLS backend cannot initialise (a boot defect) —
    /// we surface that rather than swapping in a default (redirect-following) client.
    pub fn new(cfg: EgressConfig) -> Result<Self, EgressError>;

    /// Convenience GET. The INITIAL url is NOT re-checked here (caller's pre-flight decision, per
    /// the vision pattern); redirect hops ARE re-checked when `redirects` is `FollowValidated`.
    pub async fn get(&self, url: &str, redirects: Redirects)
        -> Result<reqwest::Response, EgressError>;

    /// Execute an arbitrary request under `redirects`, returning the final resolved response.
    pub async fn execute(&self, req: EgressRequest, redirects: Redirects)
        -> Result<reqwest::Response, EgressError>;
}
```

**Design notes**
- **Returns `reqwest::Response`** so each caller keeps its own body handling: `web_extract` calls
  `.text()`, `vision` streams `.chunk()` with a byte cap, `mnemosyne` calls `.json()`. (Phase-4's
  lint bans raw `reqwest::Client`, not the `Response` type, so this leaks nothing that matters.)
- **Initial URL is the caller's responsibility** (matches vision). This is deliberate: it lets
  agent-facing callers gate the initial URL with `check_url` while letting mnemosyne target a
  private host without tripping the public-host gate — and it keeps the client testable against a
  loopback mock server.
- **Manual loop** (`execute`): parse `url`; record origin `(scheme, host, port)`; loop
  `0..=max_hops` (0 for `None`): issue request with current method/headers/body; if `None`, return
  the first response as-is; on a 3xx read `Location`, `url.join(loc)`, **`check_url(next)` → reject
  on failure**, and if the origin changed, `remove` every `CREDENTIAL_HEADERS` entry; rewrite
  method per status (301/302/303 → GET + drop body; 307/308 → keep method + body); continue. On a
  non-3xx, return it. Falling off the loop → `TooManyRedirects`.
- **Origin change = drop credentials**: `origin(a) != origin(b)` where origin is
  scheme+host+port. Covers host change, port change, and http→https downgrade/upgrade.

---

## 4. Migration — `web_extract` and `mnemosyne` (reqwest callers)

### 4a. `daemon-tool-web`
- `tools/daemon-tool-web/Cargo.toml`: add `daemon-egress = { workspace = true }`. Keep `reqwest`
  (still used by `firecrawl.rs`).
- `tools/daemon-tool-web/src/local.rs`:
  - Replace the `reqwest::Client` field with `EgressClient` (built via `EgressClient::new` with the
    UA + a sensible timeout; drop the `unwrap_or_default()` that could silently yield a
    redirect-following client — `new` returns `Result`, surface the error at construction).
  - `fetch`: `self.egress.get(url, Redirects::DEFAULT).await` then `.text()`. Map
    `EgressError::Blocked` → `WebError::Rejected(..)` (variant already exists,
    `backend.rs:40`), other `EgressError` → `WebError::Http(..)`.
  - The tool-level initial `check_url` stays (`extract_tool.rs:75`); redirect hops are now
    validated by the client. **Net effect: the redirect SSRF is closed.**
- No change to `extract_tool.rs` logic (only relies on the backend now being safe).

### 4b. `daemon-mnemosyne`
- `crates/memory/daemon-mnemosyne/Cargo.toml`: add `daemon-egress` as an **optional** dep and put
  it in the existing `sync` feature: `sync = ["dep:reqwest", "dep:daemon-egress", ...]`. (Only the
  `sync` path does HTTP.)
- `crates/memory/daemon-mnemosyne/src/sync/mod.rs` `http_post` (`:914-945`):
  - Build an `EgressClient` (keep per-call construction to stay surgical) with the 30s timeout.
  - `EgressRequest::post_json(&url, body)` + optional `.bearer_auth(key)`, then
    `egress.execute(req, Redirects::None)`. `Redirects::None` = never follow ⇒ no token leak, no
    redirect-SSRF, and **no `check_url` on the operator's (possibly private) remote**.
  - Preserve the existing "never raises" contract: map any `EgressError` into the same
    `{"status":"error","error":...}` JSON shape already returned.
  - `resp.json::<Value>()` handling unchanged.

---

## 5. Migration — `browser` (CDP, NOT the reqwest client)

The reqwest `EgressClient` cannot see Chromium's redirects, so the browser reuses the **primitive**
(`check_url`) via a **CDP mechanism**. Recommended (coordinator to confirm scope):

- **Primary (true per-hop guard):** in `supervisor.rs`, enable the CDP **Fetch** domain and, for
  every paused request/redirect (`EventRequestPaused`), run `check_url(request.url)`; `ContinueRequest`
  on pass, `FailRequest`(BlockedByClient) on reject. This is the browser-native equivalent of the
  manual loop and makes a redirect to a blocked host unreachable. It lives entirely in the
  `cdp`-gated `supervisor.rs`.
- **Belt-and-suspenders:** after `page.goto`/`wait_for_navigation`, re-validate the final
  `page.url()` (`supervisor.rs:105-109`) with `check_url`; on failure, tear the page down and return
  a rejection.
- Keep the existing pre-navigation `check_url` (`tool.rs:203`).

**Gate caveat (must flag):** the `browser`/`cdp` feature is **off by default**
(`daemon-tool-browser/Cargo.toml:8-12`), so the required workspace gate
(`cargo clippy --workspace --all-targets`, which builds *default* features) **does not compile the
browser code at all.** Browser changes must be separately validated:
`cargo clippy -p daemon-tool-browser --features cdp --all-targets -- -D warnings` and
`cargo test -p daemon-tool-browser --features cdp`. **Coordinator decision requested:** approve the
full CDP-interception scope, or accept the minimal final-URL re-check for this phase and defer the
interceptor. (If the CDP interceptor is deemed too large for Cluster D, I recommend at minimum the
final-URL re-check now and a tracked follow-up for interception.)

---

## 6. Optional (recommended) — refactor `daemon-tool-vision` onto the shared client

Vision is the *reference*, not a named migration target, but it currently owns a duplicate of the
loop. Because `EgressClient::execute` returns `reqwest::Response`, vision can delegate the
redirect loop to the shared client while keeping its Content-Length pre-check + streamed byte cap.
This removes the duplication and pre-empts Phase 4's "raw reqwest outside egress" lint. **Marked
optional / separable** — default plan includes it only if the coordinator wants a single
implementation; otherwise vision is left untouched to keep the diff focused.

---

## 7. Explicitly out of scope
- `firecrawl.rs` (`daemon-tool-web`) — POSTs to a fixed trusted endpoint; the page fetch is
  server-side, no client redirect SSRF. Left on raw reqwest (note for the Phase-4 lint: either
  allow it or route the fixed endpoint through `EgressClient` with `Redirects::None` later).
- `tavily.rs`, model/HF clients, telemetry OTLP, matrix — not agent-supplied-URL fetchers; not part
  of Cluster D.
- Hardening `check_url` itself (trailing-dot/IDNA/connect-time IP) is **Phase 2 (checkurl-identity)**,
  which builds on this crate.

---

## 8. Tests (Phase 2 — write the bug repros FIRST, watch them fail, then fix)

### In `daemon-egress` (`crates/engine/daemon-egress/`, dev-deps: `tokio`, `wiremock`, `tokio-util`)
1. **Unit — next-hop join + revalidate** (mirror vision `lib.rs:539-570`): absolute public target
   allowed; relative joins against current; targets `169.254.169.254`, `localhost`, `10.0.0.5`,
   `[::1]`, and a `file://` scheme are `Err(Blocked)`.
2. **Unit — credentials dropped on origin change**: helper over `HeaderMap` — same origin keeps
   `Authorization`; host change / port change / scheme change drops `Authorization`, `Cookie`,
   `Proxy-Authorization`. (Pure fn ⇒ hermetic; the cross-origin *follow* can't be integration-tested
   because the loopback mock trips `check_url` on the hop — same constraint vision has.)
3. **Integration (wiremock) — redirect to blocked host rejected mid-chain** *(core repro)*: mock on
   127.0.0.1 returns `302` + `Location: http://169.254.169.254/latest/meta-data/`;
   `client.get(uri, Redirects::DEFAULT)` ⇒ `Err(EgressError::Blocked(_))`. Repeat for `Location`
   into `http://127.0.0.1:<other>/`, `http://10.0.0.5/`, `http://[::1]/`. (Initial URL is the
   loopback mock and is intentionally *not* checked, so the request reaches the server.)
4. **Integration (wiremock) — happy path**: `200` body returned unchanged (no redirect).
5. **Integration (wiremock) — `Redirects::None` does not follow**: `302` returned to caller as a
   3xx response (assert status 302), redirect target never requested.
6. **Unit — method rewrite**: 303 on a POST → GET without body; 307 → POST with body preserved.

### In `daemon-tool-web` (`tools/daemon-tool-web/tests/web_tools.rs`, wiremock already a dev-dep)
7. **Bug repro at the caller** (mirrors `local_fetch_extracts_html_to_markdown` at
   `web_tools.rs:126-149`): mock returns `302 → Location: http://169.254.169.254/`;
   `LocalFetch::fetch(uri, ..)` ⇒ `Err(WebError::Rejected(_))`. Pre-fix this would follow and try to
   fetch metadata.

### In `daemon-mnemosyne` (add `wiremock` to dev-deps, gate the test on `feature = "sync"`)
8. **Token-leak repro**: server A (127.0.0.1:pa) returns `302 → http://127.0.0.1:pb/`; server B
   (pb) records requests. `SyncEngine::http_post(A, .., bearer=secret)` ⇒ server B receives **zero**
   requests (redirect not followed ⇒ token never crosses origin). Hermetic: both loopback, initial
   not checked, `None` doesn't follow so no hop check is needed.

### Browser (only if CDP scope approved) — `--features cdp`
9. If feasible without a real Chromium in CI, a focused test that the Fetch interceptor's URL
   predicate rejects a blocked hop (unit-test the `check_url`-based predicate directly, not a live
   browser). Full live-browser redirect tests likely need a Chromium and may be gated/skipped.

---

## 9. Verification gate (from `AGENTS.md`, run from the worktree root)
- `nix develop --command cargo fmt`
- `nix develop --command cargo clippy --workspace --all-targets -- -D warnings`
- `nix develop --command cargo deny check`
- `nix develop --command cargo test --workspace`
- **Browser extras (default gate skips `cdp`):**
  `nix develop --command cargo clippy -p daemon-tool-browser --features cdp --all-targets -- -D warnings`
  and `nix develop --command cargo test -p daemon-tool-browser --features cdp`.
- **No wire types change** (no `ApiRequest`/`ApiResponse` touch) ⇒ no CDDL update and no
  `cargo test -p daemon-api --features arbitrary` needed. Will re-confirm nothing wire-reachable was
  touched before claiming done.

---

## 10. Risks & ambiguities (coordinator input welcome)
1. **Browser ≠ reqwest.** The master plan assumes the browser uses "the HTTP library"; it uses
   Chromium/CDP. The shared reqwest client cannot cover it. Resolution: reuse `check_url` via CDP
   Fetch interception (+ final-URL re-check). **Decision needed:** full interceptor now vs. minimal
   final-URL re-check now + tracked follow-up (§5).
2. **`cdp` feature not in the default gate** — browser changes aren't compiled/tested by the
   required workspace gate; they need the explicit `--features cdp` runs above. Flagging so "gate
   green" isn't mistaken for "browser validated".
3. **Mnemosyne trust / private hosts.** Using `Redirects::None` (not `FollowValidated`) is a
   deliberate choice so `check_url`'s public-host gate never rejects an operator's legitimate
   private/LAN/loopback sync remote, while still killing the redirect-SSRF + token-leak. If the
   coordinator instead wants mnemosyne to *follow* redirects, we'd need a policy variant that
   re-validates hops but permits private hosts — more surface; I recommend `None`.
4. **Crate location** (`engine` vs `substrate`) — cosmetic; either satisfies the Phase-4 "one
   egress module" lint. Defaulting to `crates/engine/daemon-egress`.
5. **Vision refactor** (§6) — optional; included only on request to keep the diff surgical.
6. **POST-redirect semantics** — implementing standard method rewriting (301/302/303→GET,
   307/308→preserve). Rare for the trusted mnemosyne peer (which uses `None`), but correct for any
   future `FollowValidated` POST caller.
```
