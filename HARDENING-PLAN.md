# HARDENING-PLAN — Phase 4 / Cluster F: central ingress governor

Branch: `hardening/ingress-governor` (worktree `/home/j/experiments/daemon-worktrees/ingress-governor`, base
`hardening/integration` — carries all Phase 1–3 hardening incl. the Phase 1 ingress-bounds:
`MAX_FRAME_BYTES`, the pre-alloc frame guards in socket.rs/remote.rs, the WS `ws_config()` cap).

## Coordinator decisions (APPROVED — implementing)

1. **Secure-by-default, not opt-in.** The governor is enforced by default with *generous* limits proven not to
   reject any legitimate conformance/e2e traffic. Concrete defaults (`IngressLimits::default()` ==
   `IngressGovernor::secure_default()`):
   - `max_frame_bytes = MAX_FRAME_BYTES` (640 MiB) — unchanged pre-alloc ceiling.
   - `max_decoded_bytes = MAX_FRAME_BYTES` (640 MiB) — ≥ `MAX_BLOB_SIZE` (256 MiB), so no legit blob is newly
     rejected; the real bite is a future compressed carrier (via `ws_config` min) + the O(1) blob check.
   - `max_connections = Some(1024)` — high networked concurrency ceiling.
   - `peer_conn_rate = Some(RateSpec { burst: 256, refill_per_sec: 128 })` — a per-peer new-connection rate
     that comfortably exceeds any real client burst.
   - `max_tracked_peers = 4096` — the limiter's own memory bound (overflow → shared bucket, §1.5).
   All configurable via `[api]` (§6). Proof of generosity = the full suite stays green with defaults active.
2. **Governor home:** `daemon-common` behind a new `governor` feature — APPROVED (§1.1).
3. **Threading:** param-thread the governor/limits (no process-global) — APPROVED (§2, and the signature
   refinement below).
4. **remote.rs gets the FULL governor now** (frame + decoded + per-peer rate + connection concurrency), its
   `MAX_FRAME_BYTES` check unified in — APPROVED (§2).

**Signature refinement (minimize churn, keep behavior secure):** the **local-trust** carriers
(`serve_api_unix[_authenticated]`, `serve_api_windows_pipe[_authenticated]`) keep their current signatures and
build `IngressGovernor::secure_default()` internally — they are rate/concurrency-EXEMPT (§1.6), so only the
frame/decoded caps apply, at the safe 640 MiB default (no regression, and ~30 conformance call sites stay
untouched). The **networked** carriers (`serve_api_tls_tcp`, `serve_mux_ws`, `serve_web`, `RemoteHost`) take an
injected `Arc<IngressGovernor>` so `[api]` config drives the meaningful (conn/rate) limits; only **3**
conformance call sites (ws_transport, negative_auth-tls, web_serve) + the 3 bins/daemon spawns pass one.
`RemoteHost::new` keeps its signature (builds `secure_default()` internally — full governance per decision 4);
`RemoteHost::with_governor` injects a custom one for tests.

Guiding principle (per the roadmap): *make the unsafe form unrepresentable, not "remember to check."*
Today the ingress limits are **scattered and partial**: a frame cap duplicated across three read paths,
a decoded/expansion cap that exists only implicitly (blob store, WS post-inflate), and **no** per-peer
rate limit or connection-concurrency cap anywhere (every accept loop spawns an unbounded per-connection
task). This track collapses them into ONE `IngressGovernor` with a single explicit policy that every
carrier funnels through, and enforces NO-WILDCARD-ARM discipline on the governor's own decision code.

---

## 0. The ingress path today (the mechanism being unified)

Six carriers reach the node surface; all but daemon-http share the length-framed mux/legacy loops:

| Carrier | Entry (daemon-host unless noted) | Accept loop | Frame read | Trust |
|---|---|---|---|---|
| Unix socket | `serve_api_unix[_authenticated]` `socket.rs:88,96` | `accept_unix` `socket.rs:109-127` | `read_frame` `socket.rs:1266-1287` | local-trust (or SASL) |
| Windows pipe | `serve_api_windows_pipe[_authenticated]` `socket.rs:172,180` | `accept_windows_pipe` `socket.rs:197-232` | `read_frame` | local-trust (or SASL) |
| TLS/TCP | `serve_api_tls_tcp` `tls.rs:151-190` | inline `tls.rs:158-189` | `read_frame` (via `serve_mux`) | SASL always |
| Plain WS | `serve_mux_ws` `ws.rs:74-106` | inline `ws.rs:81-105` | `WsFrameReader::poll_read` `ws.rs:276-330` (+ `ws_config` `ws.rs:63-67`) | SASL always |
| Single-origin web | `serve_web` `web.rs:547-580` | inline `web.rs:558-579` | as WS via `serve_mux_over_ws` | SASL on `/ws` |
| Cross-node `remote` | `RemoteHost::serve` `daemon-transport/src/remote.rs:171-179` | inline | `read_frame` `remote.rs:121-146` | (cross-node) |
| in-proc HTTP (axum) | `daemon-http::serve_http` (out of primary scope, see §7) | axum | axum body limits | local-trust or deny-all |

**What each has / lacks today:**

- **Frame cap (pre-alloc):** present but *duplicated* — `if len > daemon_common::MAX_FRAME_BYTES { InvalidData }`
  literally repeated in `socket.rs:1278-1283`, `remote.rs:135-140`, and `ws.rs:300-311`
  (`WsFrameReader`), plus the tungstenite `ws_config()` cap (`ws.rs:63-67`). One value
  (`daemon_common::MAX_FRAME_BYTES` = 640 MiB, `daemon-common/src/limits.rs:24`), three-plus copies of the check.
- **Decoded / expansion cap:** *implicit only*. The 640 MiB frame cap is deliberately coarse (sized for a
  256 MiB `BlobPut` × ~2 as a CBOR array-of-ints); the only true "post-expansion" bound today is
  (a) the blob store's `len > MAX_BLOB_SIZE` (`blob_store.rs:82`, 256 MiB) applied *after* full decode+buffering,
  and (b) tungstenite's `max_message_size` which caps the **post-inflate** WS message (`ws_config`).
- **Per-peer rate limit:** **none.** Every accept loop `listener.accept()` → `tokio::spawn` with no throttle
  (`accept_unix`, `accept_windows_pipe`, `serve_api_tls_tcp`, `serve_mux_ws`, `serve_web`, `RemoteHost::serve`).
  The peer `SocketAddr` is captured then dropped (`_addr`/`_peer`).
- **Connection concurrency cap:** **none.** Unbounded spawned per-connection tasks — a connection flood
  exhausts memory/fds with no ceiling. This is the widest open gap.
- **Fail-closed decision code:** the frame checks fail closed already; there is no consolidated decision type,
  so no place to enforce the no-wildcard discipline the roadmap asks for on the governor layer.

**Pre-auth exposure that motivates a tighter cap:** `read_frame` runs *before* the SASL state check in
`serve_mux` (an unauthenticated TLS/WS peer can force a 640 MiB read buffer per frame), and the accept loops
admit unlimited connections before any auth. The governor introduces a **tighter pre-auth frame cap** and a
**connection budget + per-peer rate** that bite before that cost is paid.

---

## 1. The governor: type + policy design

### 1.1 Where it lives — `daemon-common` behind a `governor` feature

`daemon-common` is the **only** crate both `daemon-host` (socket/tls/ws/web) and `daemon-transport` (remote)
depend on, and it already hosts `MAX_FRAME_BYTES` (`limits.rs`). It also already establishes the exact pattern
we need: a pure-data policy always compiled, plus a runtime helper gated behind a feature that pulls in tokio
(`env_policy.rs` + the `process` feature, `Cargo.toml:18-20`).

Add a new module `crates/contracts/daemon-common/src/ingress.rs` and a new feature:

```toml
# daemon-common/Cargo.toml [features]
governor = ["dep:tokio"]   # IngressGovernor needs tokio::sync::Semaphore + time; pure IngressLimits stays runtime-free
```

- `IngressLimits` (pure data, **always compiled**): the cap values. `Copy`, cheap to pass by value/ref.
- `IngressGovernor` (**behind `governor`**): owns the connection semaphore + per-peer bucket map + the limits.
- `daemon-host/Cargo.toml` and `daemon-transport/Cargo.toml` enable `daemon-common/governor`.

`lib.rs` hunk stays minimal (trivial to merge alongside any concurrent daemon-common change):
`pub mod ingress;` + `pub use ingress::{IngressLimits, IngressGovernor, IngressReject, PeerKey};`.

### 1.2 The policy (pure data)

```rust
/// The single ingress policy every transport enforces (fail-closed). Pure data.
#[derive(Clone, Copy, Debug)]
pub struct IngressLimits {
    /// Max accepted length-framed wire frame, rejected BEFORE the receive buffer is allocated.
    /// The pre-decode allocation ceiling. Defaults to `MAX_FRAME_BYTES` (640 MiB) to preserve
    /// today's behavior; an operator may set a tighter networked cap.
    pub max_frame_bytes: usize,
    /// Max post-transport-decode payload size (post-decompression for a compressed carrier;
    /// the decoded byte-payload of a request for the blob-carrying variants). Catches an
    /// expansion step turning a small frame into a large in-memory payload. Defaults to
    /// `MAX_BLOB_SIZE`-derived (see §1.4).
    pub max_decoded_bytes: usize,
    /// Max concurrent live connections across the governed (networked) carriers. `None` = unbounded
    /// (the pre-governor behavior, explicit).
    pub max_connections: Option<usize>,
    /// Per-peer token-bucket rate for NEW connections: `burst` capacity, `refill_per_sec` tokens/s.
    /// `None` = no per-peer rate limit.
    pub peer_conn_rate: Option<RateSpec>,
    /// Upper bound on distinct peers tracked by the rate limiter (its own memory-DoS guard, §1.5).
    pub max_tracked_peers: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct RateSpec { pub burst: f64, pub refill_per_sec: f64 }
```

`IngressLimits::default()` = today's posture made explicit: `max_frame_bytes = MAX_FRAME_BYTES`,
`max_decoded_bytes = 2 * MAX_BLOB_SIZE` (matches the frame headroom rationale), `max_connections = None`,
`peer_conn_rate = None`. A conservative production preset (`IngressLimits::networked_default()`) sets a
tighter pre-auth frame cap, a finite `max_connections`, and a peer rate — surfaced as config (§6), never
silently on, so behavior is a visible operator choice.

`IngressLimits::unlimited()` (all caps off / `usize::MAX`) for the client helpers and existing tests that
must not be throttled.

### 1.3 The governor (stateful, feature-gated)

```rust
pub struct IngressGovernor {
    limits: IngressLimits,
    /// Connection budget. `None` when max_connections is None (admit always).
    conns: Option<Arc<Semaphore>>,          // tokio::sync::Semaphore
    /// Per-peer new-connection buckets. Leaf lock (see §5). Bounded by max_tracked_peers.
    peers: Mutex<HashMap<PeerKey, TokenBucket>>,
}
```

- **`limits()` / `check_frame_len(len) -> Result<(), IngressReject>`** — the pre-alloc frame cap. O(1), no
  state. Replaces the three duplicated inline checks. Also usable via `IngressLimits::check_frame_len` so
  `read_frame` can enforce it without the stateful governor when a caller only has the limits.
- **`check_decoded_len(actual: usize) -> Result<(), IngressReject>`** — the post-decode payload cap (§1.4).
- **`admit_connection() -> Option<ConnectionPermit>`** — the concurrency cap. `try_acquire_owned()` on the
  semaphore; `Err` (at cap) → `None` = **refuse** (fail closed; never queue — queuing unbounded accepts is
  itself a DoS). The returned `ConnectionPermit` wraps the `OwnedSemaphorePermit`; dropping it (on connection
  close, including revocation teardown) frees the slot. `None` limits → always returns a no-op permit.
- **`check_peer(&PeerKey) -> Result<(), IngressReject>`** — per-peer token bucket for a new connection
  (§1.5). Empty bucket → `Err(IngressReject::PeerRateExceeded)` = refuse.

`PeerKey` = the peer **IP** (`IpAddr`), NOT the ephemeral `SocketAddr` port, so a rapidly-reconnecting client
is still throttled. Local-trust carriers use a sentinel `PeerKey::Local` that the governor **exempts** from
concurrency + rate (see §1.6).

### 1.4 Measuring decoded size "without a second full allocation"

The concern (roadmap): a small frame that *expands* — decompression or CBOR structural blow-up — into a large
in-memory payload. Concretely, for this wire:

1. **Compressed carrier (future / WS permessage-deflate):** a small compressed frame inflates to a large
   message. tungstenite applies `max_message_size` to the **already-inflated** message, so `ws_config()` is
   *the* decoded-size enforcement point for WS. The governor's `max_decoded_bytes` is wired into `ws_config()`
   (it already caps at `MAX_FRAME_BYTES`; it becomes `min(max_frame_bytes, max_decoded_bytes)`), giving one
   place any future compressed transport enforces the decoded bound.
2. **Length-framed carriers (unix/pipe/tls/remote — uncompressed today):** the frame bytes ARE the pre-decode
   bytes, already bounded by `max_frame_bytes`. CBOR of the workspace types never *expands* a byte payload
   (`Vec<u8>` is an array-of-ints: ~1–2 wire bytes → 1 in-memory byte, i.e. decoded ≤ frame). So the frame
   cap already bounds structural decoded size; a counting/second-pass deserializer would add cost for no gain
   ("simplicity first").
3. **The one genuine typed-decode expansion of interest — carried blob bytes:** `BlobPut`/`FsWriteFromBlob`
   carry a decoded `Vec<u8>` whose semantic size is what `blob_store.rs:82` checks (`len > MAX_BLOB_SIZE`)
   *after* the full buffer already exists. The governor centralizes this as an **O(1)** check: on the already-
   decoded request, read `blob.len()` / `BlobRef.size` and call `check_decoded_len(...)`. No re-encode, no
   second allocation — it inspects the Vec length already in hand. This makes "reject oversize decoded payload"
   a single ingress policy rather than a per-handler afterthought.

So `max_decoded_bytes` is enforced at three honest points — the WS post-inflate cap, the frame cap on
length-framed carriers, and the O(1) blob-payload check — and is the designated hook the moment a compressed
carrier is added. This is documented in the module so the intent is unmissable.

### 1.5 Rate-limit algorithm — token bucket

Per-peer **token bucket** (chosen over fixed-window: smooth, burst-tolerant, the standard):
`TokenBucket { tokens: f64, last_refill: Instant }`, capacity `burst`, refill `refill_per_sec`. On a new
connection: lazily refill `min(burst, tokens + elapsed*rate)`, then if `tokens >= 1.0` consume one → allow,
else refuse. `check_peer` takes the `peers` Mutex only for this O(1) update (leaf lock, never across `.await`).

**The limiter's own memory bound (a required DoS guard):** many distinct source IPs would grow the map
unboundedly. Enforce `max_tracked_peers`: when the map is full and a new peer arrives, either evict a full-and-
idle bucket (a bucket back at `burst` and untouched > an idle window is safe to forget — a fresh peer starts
full anyway) or, if none is evictable, fall back to a shared "overflow" bucket so tracking can never itself be
the DoS. Lazy eviction on insert (no background task).

Granularity: rate-limit at **connection accept** (primary — stops connection floods cheaply). Per-frame
(intra-connection) rate is intentionally **not** added in v1: a single authenticated mux connection's request
rate is already bounded by `WRITER_QUEUE` backpressure (`socket.rs:44`, 256) and spawned-task concurrency;
throttling frames would also risk starving legitimate streaming (`Subscribe`). Flagged as residual (§8).

### 1.6 Local-trust exemption (surfaced policy)

The connection budget + per-peer rate apply to **networked** carriers (TLS/TCP, WS, web, cross-node remote).
The **unix socket / Windows pipe local-trust** carriers get only the frame/decoded caps — no permit, no peer
rate — because they are the deliberate trusted local admin/FFI/CLI path and share no meaningful peer identity,
and because a networked connection flood must **not** be able to starve the local operator CLI of the shared
connection budget. This mirrors the existing `local_trust` trust split and is stated in config (§6). (An
authenticated unix socket with `local_trust` disabled is still local and stays exempt from the *networked*
budget; its frame/decoded caps still apply.)

---

## 2. How each transport funnels through the governor

One `Arc<IngressGovernor>` is constructed at boot (§6) and shared into every `serve_*`. Two narrow consult
points per carrier: **(a) accept** (permit + peer rate) and **(b) frame read** (frame cap). Decoded cap rides
`ws_config()` + the blob-variant O(1) check.

- **`socket.rs`**
  - `read_frame` (`1266-1287`): replace the inline `if len > MAX_FRAME_BYTES` (`1275-1283`) with
    `limits.check_frame_len(len)`. `read_frame` gains a `limits: &IngressLimits` param; client/test callers
    pass `&IngressLimits::unlimited()` (via a defaulted wrapper to keep those call sites tiny).
  - `accept_unix` (`109-127`) / `accept_windows_pipe` (`197-232`): **local-trust → frame cap only**; thread
    the governor's `limits` into `serve_conn_split`/`serve_conn`/`serve_legacy`/`serve_mux`. No permit / no
    peer rate (local exempt, §1.6).
  - `serve_mux` (`466`), `serve_conn_split` (`145`), `serve_conn` (`130`), `serve_legacy` (`241`): gain a
    `limits: IngressLimits` (Copy) param, forwarded to their `read_frame` calls.
- **`tls.rs`** — `serve_api_tls_tcp` accept (`158-189`): `governor.check_peer(&PeerKey::ip(addr))?` then
  `governor.admit_connection()` (else drop + debug-log, fail closed) BEFORE the handshake spawn; move the
  permit into the task; pass `governor.limits()` into `serve_mux`.
- **`ws.rs`** — `serve_mux_ws` accept (`81-105`): peer-rate + permit as TLS; `ws_config()` (`63-67`) caps at
  `min(max_frame_bytes, max_decoded_bytes)`; `WsFrameReader::poll_read` (`300-311`) frame check routes through
  `limits.check_frame_len`; `serve_mux_over_ws` (`112`) gains `limits`.
- **`web.rs`** — `serve_web` accept (`558-579`): peer-rate + permit for the connection (covers both static and
  the `/ws` upgrade path); pass `governor` into `serve_mux_over_ws`. Static-file serving is unaffected beyond
  the shared connection budget/rate (which is correct — a static-file flood is also a DoS).
- **`daemon-transport/src/remote.rs`** — `read_frame` (`121-146`): replace inline check (`135-140`) with
  `limits.check_frame_len`. `RemoteHost` gains an `Arc<IngressGovernor>` (constructed in `RemoteHost::new` or a
  `with_governor` builder; default `unlimited()` to keep existing `new` callers working); `RemoteHost::serve`
  (`171-179`) does peer-rate + permit per accept. remote frames are all tiny control messages, so this carrier
  can take a much tighter `max_frame_bytes` than the blob-carrying api carriers (surfaced).

**Public-signature churn & test impact:** the public `serve_*` fns gain an `Arc<IngressGovernor>` param. The
conformance harness (`tests/daemon-conformance/src/node/harness.rs`) constructs them in one place — update it
to pass `IngressGovernor::unlimited()` except in the new governor tests (§4). `MuxApiClient`/`ApiClient`/unit
tests that call `read_frame` pass `&IngressLimits::unlimited()`. All flagged for the clippy-disallow sibling
in §9.

---

## 3. Fail-closed semantics + NO-WILDCARD discipline

`IngressReject` is the single decision-reject enum; every consult returns `Result<_, IngressReject>` and every
match on it is **exhaustive (no `_` arm)** — the roadmap's "new variant = build break" discipline applied to
the governor layer:

```rust
#[non_exhaustive-NOT-used]   // deliberately exhaustive: adding a variant must break every match site
pub enum IngressReject {
    FrameTooLarge { len: usize, max: usize },
    DecodedTooLarge { len: usize, max: usize },
    ConnectionCapReached { max: usize },
    PeerRateExceeded,
}
```

- Frame/decoded over cap → map to `io::Error(InvalidData, ...)` → connection dropped (today's shape, now via
  the governor).
- `admit_connection()` at cap → refuse the new connection (drop the accepted stream immediately, debug-log);
  never queue.
- `check_peer` empty bucket → refuse the new connection (drop).
- Every governor decision method and every call-site `match` enumerates all arms; the default of any ambiguity
  is **reject**. Documented as: the governor never has a catch-all that could silently permit.

`IngressReject` is internal (server-side); it is **not** a wire type (see §5). Where a reject must reach a mux
client mid-connection (only the decoded/blob path — the frame/accept rejects happen before/around a live
exchange), it maps to an existing `ApiError` on the `Reply`/`End`; no new wire variant.

---

## 4. Tests (added FIRST — must fail before the fix, pass after)

Unit tests in `daemon-common` (pure + `governor`-feature); integration in `tests/daemon-conformance`
(model on `auth_transport.rs` / `revocation_transport.rs`; real `NodeApiImpl`, `serve_api_tls_tcp` on a
loopback listener with a tiny-limit governor, `MuxApiClient`).

1. **Oversize decoded payload rejected (fail-closed).**
   - Unit: `IngressLimits{ max_decoded_bytes: N, .. }.check_decoded_len(N+1) == Err(DecodedTooLarge)`;
     `check_decoded_len(N) == Ok`.
   - Integration: authenticated client sends a `BlobPut` whose decoded blob exceeds `max_decoded_bytes` →
     `ApiError` (refused at the ingress boundary), and an at-limit blob succeeds. (Repro: today only the blob
     store's post-buffering `MAX_BLOB_SIZE` check catches it; the governor makes it an ingress decision.)
   - Keep + re-route the existing pre-alloc frame tests (`socket.rs:1365-1392`, `remote.rs:420-433`,
     `ws.rs:527-531`) so they exercise the governed check; add `IngressLimits::check_frame_len` unit tests.
2. **Per-peer rate exceeded → throttled/refused (fail-closed).**
   - Unit: token bucket `burst=N` → N `check_peer(ip)` Ok, the (N+1)th `Err(PeerRateExceeded)`; after
     `refill_per_sec` elapses (inject a clock/`Instant` seam or `tokio::time::pause`) a token returns.
   - Integration: governor with `peer_conn_rate = { burst: N }`; open N+K connections from one loopback peer
     rapidly → exactly the excess are refused (dropped), a different peer is unaffected (per-peer, not global).
3. **Concurrency cap enforced (fail-closed).**
   - Unit: `max_connections = Some(N)` → N `admit_connection()` return `Some`, the (N+1)th `None`; dropping one
     permit frees a slot (the next `admit` is `Some`).
   - Integration: configure `max_connections = N`, hold N connections open (park after `Hello`), the (N+1)th
     is dropped by the server; closing one lets a subsequent connect succeed.
4. **All decisions fail closed.** A zero-permit governor admits nothing; a zero-token bucket refuses; an
   over-cap frame/decoded errors — each asserted to REJECT, never permit. (The no-wildcard discipline is
   enforced structurally by exhaustive matches; documented, not runtime-testable.)
5. **Local-trust exemption + no-regression.** A unix-socket local-trust connection is served with a
   finite-`max_connections` governor even when the networked budget is exhausted (the local path is exempt),
   and the frame cap still applies to it. Existing conformance/e2e stay green with `unlimited()`.

---

## 5. Concurrency: lock/permit ordering vs. the Phase-2 secret-epoch teardown (no deadlock)

The governor introduces two runtime resources — the connection **semaphore permit** and the per-peer
**bucket Mutex**. Both are designed as *leaves* that never interact with the revocation path:

- **Bucket `Mutex` is a strict leaf** (same rule as `SessionRevocations.users`, `revocation.rs:26-32`): held
  only for the O(1) refill+consume in `check_peer`, **never across an `.await`**, and never nested under any
  other lock (store mutex, `AuthStore` connection mutex, `SessionRevocations.users`). `check_peer` is called at
  accept time, entirely outside `serve_mux`'s revocation `select!` (`socket.rs:519-531`).
- **`ConnectionPermit` is RAII, acquired non-blocking (`try_acquire_owned`), released by `Drop`.** It is
  acquired **once at accept, before `serve_mux`**, and dropped when `serve_mux` returns — including on the
  revocation teardown path (`socket.rs:522-528, 713-724, 762-774`), which `break`s the loop → drops `tx` →
  `writer.await` → returns → permit drops. So a revoked connection **releases its slot** exactly like a normal
  close; revocation never *waits on* the governor, and the governor never *waits on* revocation. Because the
  permit is never blocked-on (try-acquire) and never held while awaiting another lock, there is no acquire-
  order to invert → no deadlock possible.
- **No new lock is introduced into `serve_mux`'s hot loop.** The permit lives as a moved-in value in the
  per-connection task; the governor is consulted only at the accept boundary. The teardown `select!` arms are
  unchanged. The stream pumps' `revoked_or_never` arms (`socket.rs:341-346`) are untouched.

Net: the governor is additive to the Cluster-F teardown; permit lifetime is a superset of the revocation
guard's, and both release cleanly on the same connection-close path.

---

## 6. Configuration + construction (node-side, not wire)

New `[api]` knobs on `ApiConfig` (`bins/daemon/src/config.rs:757-804`), plain serde/figment (NOT wire types):
`max_connections: Option<usize>`, `peer_conn_burst`/`peer_conn_rate_per_sec` (→ `RateSpec`), an optional
tighter `max_frame_bytes`, and `max_decoded_bytes`; a `remote`-specific tighter `max_frame_bytes` for the
cross-node transport. Defaults preserve today's behavior (`IngressLimits::default()`): a first-class,
opt-in-to-tighten posture, consistent with the roadmap's "surfaced policy" theme. Documented in the generated
config table (`config.rs` help text, near `:1507`).

Construct one `Arc<IngressGovernor>` in `main.rs` right before the listener block (`~2519`) and clone it into
each `serve_*` spawn (`2528/2531`, `2547/2550`, `2590`, `2615`, `2687`) and into `RemoteHost` if wired. The
unix/pipe carriers receive the governor but it applies frame/decoded caps only (local exempt).

---

## 7. Wire / CDDL impact

**None.** No `ApiRequest`/`ApiResponse` variant, field, or wire-reachable type changes: the governor, its
limits, `IngressReject`, and the config knobs are all server-side. `daemon-api.cddl` and
`cargo test -p daemon-api --features arbitrary` are unaffected — the arbitrary gate is still run (per the
mandate) and is a no-op here. daemon-http (axum) is **out of primary scope** (it has its own
`DefaultBodyLimit`/router limits and is local-trust-or-deny-all); wiring it to the same governor is noted as
residual (§8), not required for this track.

---

## 8. Residual coverage (documented, not closed here)

- **daemon-http (axum) carrier** funnels through axum's own limits, not this governor — a follow-on could
  share `IngressGovernor` (concurrency/peer-rate) via a tower layer.
- **Per-frame (intra-connection) rate** is not added (bounded today by `WRITER_QUEUE` backpressure +
  spawned-task concurrency); surfaced as a future knob if a single-connection request flood is observed.
- **Rate-limiter map memory** is self-bounded (`max_tracked_peers` + idle eviction / overflow bucket, §1.5) —
  called out so it can never become a DoS itself.
- **Decoded-size for arbitrary structures** relies on the frame cap (uncompressed, CBOR non-expanding) + the
  O(1) blob-variant check + the WS post-inflate cap; a general counting deserializer is deliberately avoided
  ("simplicity first") and would be the escalation if a compressed non-WS carrier is ever added.
- **Local-trust exemption** from concurrency/peer-rate is a deliberate policy (protects the operator path
  under network attack), documented in config.

---

## 9. Edit surface (for the clippy-disallow sibling — keep its lints off my lines)

This track owns the ingress/governor layer's no-wildcard discipline. Files/functions I will touch:

- **new** `crates/contracts/daemon-common/src/ingress.rs`; `daemon-common/src/lib.rs` (`pub mod ingress;` +
  re-exports); `daemon-common/Cargo.toml` (`governor` feature).
- `crates/substrate/daemon-host/src/socket.rs`: `read_frame` (+ callers `MuxApiClient`/`ApiClient` for the new
  `limits` param), `accept_unix`, `accept_windows_pipe`, `serve_conn`, `serve_conn_split`, `serve_legacy`,
  `serve_mux`, and the public `serve_api_unix[_authenticated]` / `serve_api_windows_pipe[_authenticated]`
  signatures. **These are the socket.rs "stragglers" — please keep clippy-disallow edits off these fns.**
- `crates/substrate/daemon-host/src/tls.rs`: `serve_api_tls_tcp`.
- `crates/substrate/daemon-host/src/ws.rs`: `serve_mux_ws`, `serve_mux_over_ws`, `ws_config`,
  `WsFrameReader::poll_read`.
- `crates/substrate/daemon-host/src/web.rs`: `serve_web`.
- `crates/substrate/daemon-host/Cargo.toml`: enable `daemon-common/governor`.
- `crates/substrate/daemon-transport/src/remote.rs`: `read_frame`, `RemoteHost`(`new`/builder), `serve`;
  `daemon-transport/Cargo.toml`: enable `daemon-common/governor`.
- `bins/daemon/src/config.rs` (`ApiConfig` knobs + help text); `bins/daemon/src/main.rs` (construct + thread
  the governor).
- `tests/daemon-conformance/src/node/harness.rs` (pass `unlimited()`); new governor test module(s).

I merge **before** clippy-disallow (per the wave plan), so it rebases onto these lines.

---

## 10. Gate (Phase 2, from worktree root; paste tails)

- `nix develop --command cargo fmt --all -- --check`
- `nix develop --command cargo clippy --workspace --all-targets -- -D warnings`
- `nix develop --command cargo test --workspace --no-fail-fast`
- `nix develop --command cargo test -p daemon-api --features arbitrary`  (no wire change; run anyway)

Machine-load: if `bins/daemon/tests/host_launch.rs` fails under the full parallel run, re-run
`cargo test -p daemon --test host_launch -- --test-threads=2` isolated before treating it real. Known
pre-existing load flakes (do NOT chase): `detached_delegation::detached_notice_reaches_a_parked_durable_parent`,
`detached_delegation::detached_fanout_materializes_distinct_children`,
`process_notify::injected_input…store_seam`.

Commit on `hardening/ingress-governor`. Do NOT merge; do NOT remove the worktree.

---

## 11. Open questions for coordinator review

1. **Default posture:** keep `IngressLimits::default()` == today (all networked caps *off*, opt-in via config),
   or ship a finite `networked_default()` (tighter pre-auth frame cap + a `max_connections` + a peer rate) as
   the out-of-box default? Recommend **opt-in defaults preserve behavior**, with a documented recommended
   preset — but a "secure by default" finite `max_connections` is defensible for the networked carriers.
2. **Governor home:** `daemon-common` behind a `governor` feature (recommended — the shared DAG root, mirrors
   `env_policy`), vs. a small new `daemon-ingress` crate. daemon-common keeps the merge surface trivial and
   both consumers already depend on it.
3. **`read_frame` param threading** vs. a thread-local/`OnceLock` global limits: I propose the explicit
   `&IngressLimits` param (greppable, testable, no hidden global) even though it touches the client/test
   callers; confirm that's acceptable churn vs. a process-global default.
4. **remote.rs governor:** wire the cross-node transport to the same governor with a tighter control-frame
   `max_frame_bytes`, or leave `RemoteHost` on `unlimited()` + just the tighter frame const for now? Recommend
   wiring it (it is networked and today wholly unthrottled).
