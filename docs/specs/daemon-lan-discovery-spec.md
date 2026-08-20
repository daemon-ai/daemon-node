# LAN Node Discovery — mDNS/DNS-SD Advertise (node) + Browse (app)

Status: BINDING SPEC (lan-discovery track, phase 1), design accepted, implementation staged.
Where this document and code disagree, the code is wrong until a stage brings it into
compliance. Ported (with corrections) from the legacy `daemon-q1-2026` peer-discovery stack; §8
records exactly what carries over and what is deliberately dropped. Companion:
[`daemon-pairing-spec.md`](daemon-pairing-spec.md) (phase 2) replaces the TOFU + password
first contact with a SPAKE2 enrollment ceremony; this spec stands alone without it — discovery
finds nodes, pairing makes a found node trusted.

## 1. Problem and scope

A `daemon-app` that wants to attach to a `daemon-node` elsewhere on the local network today
requires the user to hand-type `host:port` into the connection picker. The node already exposes
a network API carrier — the always-TLS TCP listener (`[api].tls_addr` / `tls_cert` / `tls_key`,
bound in `bins/daemon/src/main.rs`) — but nothing announces it, and the app has no way to
enumerate candidates.

The **predominant consumer is mobile**: a phone or tablet on the same Wi-Fi finding the
household/office node is the flagship flow — a desktop user can type `host:port`; a phone user
realistically will not. Desktop GUI/TUI get the same feature, and land first only because they
are the cheapest place to prove the contract (§10, §12) — order of delivery, not order of
importance.

**Resolution:** a split design.

- `daemon-node` **advertises** its bound TLS API listener over mDNS/DNS-SD (RFC 6762/6763).
- `daemon-app` **browses** for advertisements and surfaces them in the first-run wizard and in
  Settings → Connection as selectable "Nearby nodes".
- Everything after selection is unchanged and remains authoritative: the mux `Hello` +
  `api/<N>` feature gate, TLS certificate validation (CA / pinned leaf), SASL login, and the
  existing liveness state machine.

**Authority note (binding).** "The node decides, the apps render" is not violated: browsing is a
*connection bootstrap* concern — the app must locate a node before it can call any node API. This
is the same class of app-local platform code as the scheme picker and the TLS dialer, not domain
state. No domain question is answered client-side; the moment a connection exists, the node is
authoritative as usual. Discovery results are untrusted hints in every respect (§9).

**Scope (binding):**

- v1 advertises the **TLS/TCP carrier only**. The Unix socket, Windows pipe, in-process HTTP,
  plaintext WebSocket (`[api].ws_addr`), web front, and gateway are never advertised (§9).
- v1 browse targets **desktop (Linux/macOS/Windows) AND Android/iOS**. Only WASM is excluded,
  and only because the exclusion is physical: a browser page has no multicast sockets and no
  mDNS. Mobile's current WebSocket-only posture is a *build posture*, not a platform constraint
  — §5.3 binds the transport work that makes a discovered node connectable from mobile, and the
  mobile stages of §12 are gated on the mobile build lane existing (those builds are not yet
  built or tested; this spec defines their requirements now so the §2 contract never has to
  change for them).
- No wire-contract change: no CDDL edit, no `just update-codec`, no `WireVersion` bump, no new
  `ApiRequest`/`ApiResponse`. A typed runtime advertise-toggle API is explicitly out of scope
  (§11).
- Discovery carries no trust. mDNS solves *finding* a node, not *trusting* it. First-contact
  trust is, in preference order: the pairing ceremony
  ([`daemon-pairing-spec.md`](daemon-pairing-spec.md)) when available, else the explicit
  user-confirmed TOFU affordance of §6.3, always followed by the existing SASL login.

## 2. DNS-SD contract (normative)

This section is the cross-repo, cross-platform contract. The node publishes it; every app
platform consumes it; neither may extend it without bumping `txtvers`.

### 2.1 Service registration

| Field | Value (binding) |
|---|---|
| Service type | `_daemon-node._tcp.local.` |
| Instance name | The display name (§3.3). User-visible per RFC 6763; spaces allowed. Uniqueness on the LAN is delegated to standard DNS-SD conflict resolution (auto-rename); instance names are NOT identity — TXT `node` is (§2.2). |
| SRV host | The machine hostname as `<hostname>.local.` |
| SRV port | The **actually bound** TLS port, taken from `TcpListener::local_addr()` after a successful bind — never the configured string (a `:0` bind would otherwise advertise a lie). |
| TTLs / goodbye | Library defaults (`mdns-sd`). A clean node shutdown MUST unregister so goodbye records (TTL 0) go out. |

Platform API spelling: the type is one service; the trailing labels differ by API convention.
qmdnsengine and `mdns-sd` take `_daemon-node._tcp.local.`; Android `NsdManager` and Apple's
`NWBrowser` take `_daemon-node._tcp` (no `.local.`). Backends encapsulate this; nothing else in
the app may hardcode the type string.

The service type is deliberately not `_daemon-api._tcp` or `_daemon-tls._tcp`: one type names the
product's node, and v1 binds that type to the TLS carrier. If a future revision advertises a
second carrier it registers a **distinct type** (e.g. `_daemon-node-ws._tcp`), because a DNS-SD
instance has exactly one SRV port; it does not overload this one.

### 2.2 TXT record, `txtvers=1`

| Key | Value | Notes |
|---|---|---|
| `txtvers` | `1` | Schema version. The app MUST ignore records with an unknown `txtvers` (forward-compat: unknown *keys* are ignored, unknown *versions* are skipped). |
| `node` | 32 lowercase hex | Persistent node identity (§3.2). The app's dedup key. |
| `name` | UTF-8 display string | From `[api.mdns].name`, default hostname. Display only. |
| `api` | decimal, currently `52` | The API contract version: `daemon_common::WireVersion::CURRENT` (`crates/contracts/daemon-common`), i.e. the `N` in the Hello feature `api/N`. Compatibility is exact equality, matching `WireVersion::is_compatible`. |
| `wire` | decimal, currently `2` | The mux envelope version `daemon_api::wire::WIRE_VERSION`. Distinct from `api` — do not conflate. |
| `ver` | node SemVer (`X.Y.Z`, no build suffix) | Display only. |
| `auth` | `scram` | Pre-warns the app that login follows. Literal in v1 (pairing adds no TXT key; its mechanism is advertised in `Hello`, not here). |
| `fp` | 64 lowercase hex | SHA-256 of the leaf certificate **DER** — byte-for-byte the format of the app's `conn/tls/pinnedSha256` (`QSslCertificate::digest(QCryptographicHash::Sha256).toHex()`) and of the node's `TlsState.peer_cert_fingerprint`, so an accepted fingerprint drops in with no transformation. Computed node-side from the first certificate in the configured `api.tls_cert` PEM. An untrusted hint (§9); never anything more. |

Budget: ~170–200 bytes with a typical name — one TXT record, one packet. Keys beyond these MUST
NOT be added under `txtvers=1`. In particular `inc` (incarnation) is deliberately absent:
incarnation is minted per process (`mint_incarnation`,
`crates/substrate/daemon-host/src/node_api/internals.rs`) and belongs to the Bootstrap/Hello
surface, not to a record that would churn on every restart. `part` (PartitionId) is likewise not
a LAN-facing concept. There is also **no** "ready to pair" hint — pairing state is never
broadcast (pairing spec §1).

## 3. Node side — advertiser

### 3.1 Crate

New adapter crate `crates/adapters/daemon-mdns` (workspace membership is automatic via the
`crates/adapters/*` glob; add the `[lints] workspace = true` stanza). Built on the pure-Rust
**`mdns-sd`** crate — no Avahi/Bonjour runtime dependency, which keeps the sealed bundle
self-contained. The new dependency needs a `cargo deny` pass (first mDNS dep in the workspace).

Surface stays minimal and synchronous-at-the-edges:

```rust
pub struct ServiceSpec {
    pub instance: String,              // display name
    pub hostname: String,              // "<host>.local."
    pub port: u16,                     // from local_addr()
    pub txt: BTreeMap<String, String>, // §2.2, fully composed by the caller
}

pub struct Advertiser { /* owns the mdns-sd daemon */ }
impl Advertiser {
    pub fn start(spec: ServiceSpec) -> anyhow::Result<Self>;
    pub fn shutdown(self); // unregister => goodbye records
}
```

The crate knows nothing about `NodeConfig`, TLS, or version constants — `bins/daemon` composes
the `ServiceSpec` and owns policy. Interface churn (address add/remove) is handled inside
`mdns-sd`'s daemon; the advertiser does not hand-roll republish logic.

### 3.2 Persistent node identity (new primitive)

There is no durable node identity today: incarnation is per-process, `PartitionId` is an
ownership domain, and the VHC base key is VHC-scoped. Discovery needs a stable dedup key, so:

- New file `<data_dir>/node-id`: 32 lowercase hex chars, minted once (16 random bytes via
  `getrandom`, hex-encoded — the `mint_incarnation` recipe) on first need, then reused forever.
  Mint-and-persist MUST be atomic-write (tmp + rename), and a corrupt/short file is re-minted.
- New **top-level** `NodeConfig` field `node_id: Option<String>` (next to `partition`): when set
  (or via env `DAEMON_NODE_ID`), it overrides the file. It is node identity generally, not a
  discovery detail — other consumers (pairing URIs carry it; sync, fleet) SHOULD reuse it rather
  than mint their own.

### 3.3 Configuration

Nested under `[api]`, because the advertiser announces the API listener and nothing else. A bare
`[discovery]` section is forbidden: "discovery" already means model-vendor discovery, MCP tool
discovery, and VHC `RunDiscovery` in this codebase.

```toml
[api]
tls_addr = "0.0.0.0:7443"
tls_cert = "/path/server.pem"
tls_key  = "/path/server.key"

[api.mdns]
enabled = true            # default: true — but inert unless the TLS listener binds non-loopback
name    = "Office Daemon" # default: machine hostname
```

Env overrides ride the standard figment scheme: `DAEMON_API__MDNS__ENABLED`,
`DAEMON_API__MDNS__NAME`.

**Default-on is safe by construction (binding):** advertisement happens only when ALL hold —

1. `[api.mdns].enabled` is true;
2. the TLS listener exists (`tls_addr`+`tls_cert`+`tls_key` configured) and **bound
   successfully**;
3. the bound address is not loopback.

A default install (Unix socket only) therefore advertises nothing. Binding the TLS carrier to a
non-loopback address is already the operator's deliberate act of exposing the API to the network;
advertising adds no meaningful exposure a port scan wouldn't find, and buys the zero-config UX.
When condition 2 or 3 fails with `enabled = true` explicitly set, log at `info` why advertisement
is skipped.

### 3.4 Lifecycle in `bins/daemon/src/main.rs`

Spawned from `run_as_host` immediately after the TLS listener bind succeeds (the only place the
real bound port is knowable), following the existing config-gated optional-subsystem pattern
(like the WS/web servers, not the gateway `ManagedResource` — there is no runtime toggle in v1):

1. Bind TLS listener (existing code).
2. If §3.3 conditions hold: load/mint `node_id`, read `api.tls_cert`, compute `fp`, compose
   TXT + `ServiceSpec` (port from `local_addr()`), `Advertiser::start`.
3. On shutdown (after `shutdown_signal()`), call `Advertiser::shutdown` **before** aborting the
   TLS server task, so goodbye records precede the listener disappearing.

Advertiser failure (e.g. multicast socket denied) is non-fatal: log `warn` and continue serving.
A node that cannot advertise is degraded, not down.

### 3.5 Hostname / certificate reality (binding disclosure)

The SRV hostname (`<host>.local.`) will usually NOT match a self-signed certificate's SAN. That
is expected: LAN trust is established by pairing or the pin/TOFU path (§6.3), not hostname
verification. The node MUST NOT refuse to advertise on SAN mismatch (the app handles it), but
SHOULD log an `info` note when the cert's SANs cover neither the hostname nor any advertised
address, so operators running real CAs can fix their cert.

## 4. App side — browser

### 4.1 Module and per-platform backends

New `src/core/discovery/` → CMake target `da_discovery`. Separate from `da_connection`, which is
deliberately Qt6::Core-only; this target links Qt6::Network plus the platform backend's
dependencies. Class names carry the `Node` prefix so "discovery" stays unambiguous next to
provider/agent discovery.

Common files:

| File | Contents |
|---|---|
| `discovered_node.h` | Value DTO (§4.2). |
| `inode_discovery.h` | The seam: `Q_PROPERTY(bool available)`, `Q_PROPERTY(QString unavailableReason)`, `Q_PROPERTY(bool scanning)`, `Q_INVOKABLE start()/stop()/refresh()`, signals `nodeUpdated(DiscoveredNode)`, `nodeLost(QString nodeId)`, `errorOccurred(QString)`. Browse-only — the app NEVER advertises. |
| `mock_node_discovery.{h,cpp}` | Deterministic fixture backend for tests and `DAEMON_APP_SERVICE_MODE=mock` (hermetic GUI/a11y runs): a scripted set of appear/update/disappear events, including one incompatible-`api` node and one carrying `fp`. Compiled on all platforms. |
| `discovered_nodes_model.{h,cpp}` | `QAbstractListModel` over the seam (§4.3). |

One backend per platform, selected by the CMake platform-source swap pattern of
`src/core/daemon/CMakeLists.txt` (a TXT-record parser shared by all of them lives in a common
TU and is unit-tested once):

| Platform | Backend source | Binding requirements |
|---|---|---|
| Desktop (Linux/macOS/Windows) | `mdns_node_discovery.{h,cpp}` — qmdnsengine | Browse + resolve on the main thread (non-blocking `QUdpSocket` event work; the legacy `QThread` hosted TLS links and SQLite sync, none of which comes along). Handle `serviceAdded`/`serviceUpdated`/`serviceRemoved`. |
| Android | `nsd_node_discovery.{h,cpp}` + a JNI Java helper — `NsdManager` | The platform-blessed API; raw multicast (qmdnsengine) is unreliable on Android. Port the legacy `NsdHelper` semantics **with its defects fixed**: `onServiceLost` wired through to `nodeLost` (the backend keeps an instance-name → `nodeId` map, since the lost callback carries only the name); TXT parsed via `NsdServiceInfo.getAttributes()` (legacy ignored TXT); the serialized resolve queue kept (one `resolveService` in flight, queue + de-dup set); the Wi-Fi multicast lock acquired only while scanning and always released on stop/background. Use `getHostAddresses()` (API 34+) for all addresses, else the single `getHost()`. |
| iOS | `bonjour_node_discovery.{h,mm}` — `NWBrowser` (Network.framework) | Bonjour-API browsing does NOT require the `com.apple.developer.networking.multicast` entitlement that raw multicast would — this is why qmdnsengine is not used here. Browse with the TXT-carrying descriptor (`bonjourWithTXTRecord`) so metadata arrives without a second resolve step; resolve addresses on demand for selected/updated results. Map the browser's denied/failed states to `unavailableReason = "permission-denied"` (§4.5). No legacy iOS backend existed; this is new code bound by the same seam tests. |
| WASM | `null_node_discovery.{h,cpp}` | `available == false`, `unavailableReason = "unsupported"`, all ops no-op. |

Desktop dependency: qmdnsengine, pinned at the legacy-proven rev (`nitroshare/qmdnsengine` @
`9de38dfbd1cb989b977ed80c512187f0775abbbd`, static, `BUILD_SHARED_LIBS=OFF`), added to
`daemon-app/flake.nix` as a flake input + derivation following the `posixsignalmanager-qt6`
pattern, on the devShell `CMAKE_PREFIX_PATH`, desktop-only (alongside `DAEMON_APP_DESKTOP_DEPS`).
On desktop builds it is **required** (`find_package(qmdnsengine CONFIG REQUIRED)`) — the legacy
"silently compile without discovery" flag is not ported; a desktop build either has discovery or
fails to configure. (qmdnsengine is effectively unmaintained upstream; it is small, static, and
proven — accepted risk, revisit if it ever blocks a Qt upgrade.)

### 4.2 `DiscoveredNode` DTO

| Field | Source | Notes |
|---|---|---|
| `nodeId` | TXT `node` | Dedup key. Records missing `node` or `txtvers=1` are dropped. |
| `displayName` | TXT `name` | Fallback: DNS-SD instance name. |
| `instanceName` | DNS-SD instance | |
| `hostname` | SRV host | Kept for display and future re-resolve (§11). |
| `port` | SRV port | |
| `addresses` | all resolved A/AAAA | Aggregated across resolver callbacks (legacy emitted one candidate per address — not ported). IPv6 link-local addresses retain their scope id. |
| `apiVersion` / `wireVersion` | TXT `api` / `wire` | |
| `nodeVersion` | TXT `ver` | Display. |
| `auth` | TXT `auth` | |
| `fingerprint` | TXT `fp` | Untrusted hint. |
| `lastSeen` / `stale` | bookkeeping | §4.4. |

Dedup is by `nodeId`: `serviceAdded`/`serviceUpdated`/late resolver results upsert one row.

### 4.3 Model and wiring

- `AppServiceGraph` gains `discovery::INodeDiscovery* discovery`, constructed in
  `createAppServiceGraph()`: the §4.1 platform backend in Daemon mode, `MockNodeDiscovery` in
  Mock mode.
- `Application::registerContext()` exposes it as context property **`NodeDiscovery`** alongside
  `Connection`/`ConnSchemes`.
- `DiscoveredNodesModel` follows the `SessionsListModel` precedent (`QML_ELEMENT` in a QML
  module), roles: `nodeId`, `displayName`, `hostname`, `target` (§6.1), `nodeVersion`,
  `compatible` (bool: TXT `api` == the app codec's API version, exact equality),
  `incompatibleReason` (localized, e.g. "node speaks api/51, this app needs api/52"),
  `hasFingerprint`, `stale`. The TUI consumes the same model through `DisplayRoleAdapter`.
- Compatibility, ordering (compatible first, then by `displayName`), and target composition live
  in C++ (model/backend) — never in QML or TUI widget code.

### 4.4 Browse lifecycle

- **Scan on demand, not always-on:** `start()` when a surface showing the list becomes visible
  (wizard connect phase, Settings Connection section), `stop()` when it hides. `refresh()`
  restarts the browser and re-queries.
- **Removal:** goodbye/lost events (`serviceRemoved` on desktop, `onServiceLost` on Android,
  browse-result removal on iOS) → `nodeLost` → row removed. The legacy desktop backend never
  connected this signal and Android never wired it; those defects are not ported.
- **Staleness fallback:** a row not confirmed for 60 s is marked `stale` (greyed, still
  selectable); a stale row unconfirmed for a further 120 s is dropped. This replaces the legacy
  15 s hard TTL, which fought DNS-SD's own liveness model.
- **Network churn:** on `QNetworkInformation` reachability/transport change (debounced 500 ms),
  restart the browser and mark all rows stale pending reconfirmation — the legacy
  `NetworkEpochManager` semantic without the 1 s fingerprint-poll machinery.
- **Mobile app lifecycle (binding):** on `Qt::ApplicationSuspended` the backend stops scanning
  and releases platform resources (the Android multicast lock especially); scanning resumes only
  when a discovery surface is visible again. Background scanning is forbidden — battery and
  platform policy both say so.
- **Self-connections are not filtered:** a managed local node that binds TLS legitimately appears;
  the app advertises nothing, so the legacy self-filter has no subject.

App-side toggle: `conn/discovery/enabled` (default true) in the settings store gates all
scanning; surfaced in GUI Settings and TUI Settings (§7). This controls *browsing* on this
device; the node-side advertise toggle is node config (§3.3), out of the app's reach in v1.

### 4.5 Availability and permission semantics (binding)

`available: bool` alone cannot express mobile reality; the seam pairs it with
`unavailableReason`, one of:

| Reason | Meaning | UI obligation |
|---|---|---|
| `""` (available) | Backend operational. | Show the Nearby section. |
| `unsupported` | Platform cannot browse (WASM null backend). | Hide the section entirely. |
| `permission-denied` | The OS denied local-network access — iOS Local Network privacy (the prompt fires on the app's first browse attempt; the user may decline, and may later flip it in system settings). | Show the section with guidance: "Allow local network access in system settings to find nearby nodes" — never a silent empty list, which reads as "no nodes exist". |

The iOS backend re-checks on activation (the user may have granted permission in Settings while
the app was backgrounded). Android NSD needs no runtime permission; `permission-denied` is not
expected there but the seam handles it uniformly if a vendor build produces it.

### 4.6 Mobile packaging requirements (binding)

Recorded here so the mobile build lane inherits them as requirements, not archaeology:

- **iOS `Info.plist`:** `NSBonjourServices` MUST list `_daemon-node._tcp` (without it, iOS 14+
  silently denies Bonjour browsing regardless of the privacy prompt), and
  `NSLocalNetworkUsageDescription` MUST carry a purpose string ("Find daemon nodes on your local
  network").
- **Android manifest:** `INTERNET`, `ACCESS_NETWORK_STATE`, and `CHANGE_WIFI_MULTICAST_STATE`
  (for the multicast lock). A proguard/R8 keep rule for the JNI-called NSD helper class — the
  legacy repo's `proguard-rules.pro` keep for its `NsdHelper` is the precedent.

## 5. What discovery hands to the connection seam

Selection composes existing seam inputs; `IConnectionService` and
`connectTo(mode, target, token)` are unchanged.

### 5.1 Target composition (binding)

A selected node becomes scheme `tls` → seam mode `remote`, target `host:port` where host is the
**best resolved address**: prefer a routable IPv4, else a routable IPv6 (bracketed), else a
link-local IPv6 with scope id. This rule is one shared C++ implementation used by every platform
and both surfaces. v1 dials by address, not by the `.local.` hostname — `.local` resolution via
the system resolver (nss-mdns/Bonjour) is not universally present, and the pin-based trust path
(§6.3) is hostname-indifferent. The known cost: a CA-signed node whose cert lacks IP SANs will
fail hostname verification when dialed by address; such deployments keep manual entry (type the
DNS name the cert covers). Re-resolution by `nodeId` after DHCP churn is a future extension
(§11) — v1 persists whatever target the user connected to, exactly as manual entry does today.

### 5.2 Per-target certificate pins (binding change)

Today `conn/tls/pinnedSha256` is a single global key — switching between two self-signed LAN
nodes would clobber the pin. Following the per-target token precedent
(`conn/tokens/<sha256(target)[:16]>`), `tlsConfigFromSettings()` gains a per-target lookup:
`conn/tls/pins/<sha256(target)[:16]>` consulted first, global `conn/tls/pinnedSha256` as
fallback (compatibility; existing setups keep working). TOFU acceptances (§6.3) and the pairing
ceremony (pairing spec §6.2) write the per-target key only. Pin semantics inside the transport
are untouched: consulted on `sslErrors`, leaf-DER SHA-256 hex, colon-stripped, case-insensitive
compare.

### 5.3 Mobile TLS transport prerequisite (binding)

Discovery on mobile is useless unless a discovered node is *connectable* from mobile — and today
it is not. The facts, and why they are posture rather than physics:

- `src/core/daemon/CMakeLists.txt` lumps `ANDROID OR IOS` with `DAEMON_APP_WASM`: both get the
  stubbed stream carriers (`daemon_transport_stream_wasm.cpp`) and the stub launcher, because
  the flake's mobile Qt builds were made without the `ssl` feature (no bundled OpenSSL). Only
  the WASM half of that lump is physical (Qt-for-wasm genuinely lacks `ssl`/`process`/raw
  sockets).
- `daemon_connection_service.cpp` enforces the lump at runtime: on WASM/Android/iOS it refuses
  every mode except `remote-ws` (the gates at the `connectTo` and `testConnection` entry
  points).
- The node advertises only the TLS carrier (§2.1), plaintext `ws://` is never advertised (§9),
  no WSS carrier exists node-side — and the app's `wss` transport has **no pin knobs** (platform
  default verification only), so even a hypothetical WSS carrier could not reach a self-signed
  LAN node. Every path leads to the same conclusion:

**Mobile becomes a first-class TLS client (binding).** `remote-ws` remains the browser
transport; Android/iOS get the same native TLS stream transport as desktop:

1. **Flake (Qt builds):** the Android Qt build gains the `ssl` feature with an OpenSSL provider
   bundled into the APK (Qt's `openssl` TLS backend; the KDAB `android_openssl` arrangement is
   the established precedent). The iOS Qt build enables Qt's **SecureTransport** TLS backend —
   no OpenSSL needed on iOS; the statically built backend plugin must be linked in. Both are
   verified in the mobile-lane stage (§12), since these builds have never been exercised.
2. **CMake:** the platform-source swap splits — transport and launcher gate independently. WASM
   keeps both stubs; Android/iOS compile the real `daemon_transport_stream.cpp` and keep the
   stub launcher TU (a phone cannot spawn or attach to a local node; renaming the shared stub
   from `*_wasm.cpp` to `*_stub.cpp` is housekeeping, not a requirement).
3. **Connection service:** the WS-only gates narrow to `Q_OS_WASM` only. Mobile accepts modes
   `remote` (tls) and `remote-ws`; `local`/managed remain refused there with the existing
   "needs setup" message. The refusal text updates accordingly per platform.
4. **Scheme catalog:** `tls` becomes selectable on Android/iOS in `ConnectionSchemes`;
   `managed`/`unix` stay desktop-only.
5. **Trust parity for free:** the pin/TOFU path lives in the shared transport's `sslErrors`
   handler and the per-target pin store (§5.2) — compiling that transport on mobile carries the
   entire §6.3 trust model over unchanged. (Pairing's mobile story is separately deferred by the
   pairing spec; until it lands, mobile first contact is TOFU + SCRAM, same as desktop without
   pairing.)

This subsection is a prerequisite stage for mobile browse (§12 stage 8), not for anything on
desktop; desktop stages neither wait for it nor depend on it.

## 6. Wizard and Settings UX (GUI)

### 6.1 Nearby nodes in `ConnectionPicker.qml`

One edit covers both surfaces on every GUI platform: `ConnectionPicker` is shared by
`FirstRunGate.qml` (connect phase) and Settings `ConnectionSection.qml`. Below the scheme+target
row, when `NodeDiscovery.available && conn/discovery/enabled`:

- A "Nearby nodes" `SectionLabel` + `Kit.ListRow` list (`DiscoveredNodesModel`), a Refresh
  action, and a scanning indicator bound to `NodeDiscovery.scanning`.
- States: scanning / empty ("No nodes found on this network") / rows / permission-denied
  (the §4.5 guidance text). Incompatible rows are disabled with `incompatibleReason` as
  subtitle. Stale rows render dimmed.
- The section never appears on WASM (`unavailableReason == "unsupported"`), collapses when the
  discovery toggle is off, and on mobile appears from §12 stage 9 onward.

**Selecting a row** (binding): switch the scheme to `tls`, fill the target (§5.1), mark the
target user-edited, and run the existing `testConnection` probe. It MUST NOT auto-connect —
Connect stays the single explicit action, and manual entry remains fully available. Never
auto-connect to a first/only result either.

### 6.2 After selection

Connect drives the unchanged flow: TLS handshake → trust establishment if the target is unknown
(pairing when available, else §6.3) → SASL credential prompt → `setLastConnection` persistence.
Discovery adds no auth shortcut of any kind.

### 6.3 Trust affordance (TOFU, explicit)

When a selected node's advertisement carried `fp`, no CA covers the connection, and the TLS
probe fails verification (the self-signed LAN case — the only case the pin path serves):

- Surface an explicit "Trust this node's certificate?" affordance showing the node's display
  name and the abbreviated fingerprint (first/last 8 hex chars, full value on demand), plus the
  fingerprint actually presented in the handshake.
- Only on explicit user acceptance is the pin written — to the per-target key (§5.2) — and the
  value written is the **handshake-observed** leaf fingerprint, with a warning shown if it
  differs from the advertised `fp` (advertisement is unauthenticated; the wire observation is
  the thing being trusted).
- Declining leaves everything untouched; the connection fails closed exactly as today.

This is the fallback trust path for never-paired targets; the preferred successor to the legacy
pairing ceremony is the SPAKE2 enrollment in [`daemon-pairing-spec.md`](daemon-pairing-spec.md),
which writes the per-target pin as part of its ceremony. TOFU here is leap-of-faith-equivalent;
the displayed fingerprint enables out-of-band verification for users who want it. The app MUST
NOT write a pin from TXT data without one of these ceremonies, ever.

## 7. TUI parity (mandatory)

GUI-only discovery is an incomplete feature by repo policy. In the same delivery:

- **First-run** (`dialogs/first_run_dialog.cpp`): the connect step gains a Nearby-nodes list
  bound to the same `DiscoveredNodesModel` via `DisplayRoleAdapter`, plus a Scan/Refresh action.
  Selection fills scheme+target through the same C++ path as the GUI (no TUI-local target
  composition); Connect remains explicit.
- **Settings** (`pages/hub_settings.cpp`): the Connection section gains the Nearby-nodes list,
  the `conn/discovery/enabled` toggle, and a connect action for a selected row — which closes
  the documented "reconnect via the GUI picker" gap at least for the discovery path. The TOFU
  affordance (§6.3) gets a TUI dialog rendering the same strings.

The TUI ships on desktop only, so mobile has a single (GUI) surface; no TUI counterpart exists
to keep in parity there.

## 8. Legacy port ledger

From `daemon-q1-2026/apps/daemon` (`src/core/services/peer/`).

**Ported (with corrections):**

| Legacy semantic | Fate here |
|---|---|
| mDNS browse via qmdnsengine, same pinned rev | Kept (desktop app, §4.1) — browse-only. |
| Stable id in TXT (`id=`) + versioned metadata (`protocol=`) | Kept as `node=` + `txtvers`/`api`/`wire` (§2.2). |
| Async SRV/A/AAAA resolution | Kept; addresses aggregated per node instead of one candidate per address. |
| Network-change epoch → browser restart (`NetworkEpochManager`, 200 ms debounce) | Kept as `QNetworkInformation` + 500 ms debounce + stale-marking (§4.4), without the 1 s interface poll. |
| Source provenance / last-seen bookkeeping | Kept in the DTO. |
| Android NSD backend: multicast lock + serialized resolve queue | **Ported in v1** (§4.1, stage 9) with its defects fixed: `onServiceLost` wired, TXT parsed via `getAttributes()`, lock held only while scanning. |

**Dropped (deliberately, do not resurrect):**

- `_llamachat._tcp` service type; UDP identity broadcast on 1716; the TCP 1716–1764 listen
  range; the mDNS→directed-UDP→TCP connect sequence.
- App-side advertisement (`Provider`/`Hostname`) and self-filtering.
- Peer replication, legacy pairing, trust store, lexicographic-initiator logic (the pairing
  *concern* returns, redesigned, in `daemon-pairing-spec.md` — its ledger is §9 there).
- The 15 s hard TTL prune; raw `QVariantMap` peer models; numeric-address-only identity
  *as a model concept* (v1 still dials numeric, §5.1, but the DTO keeps hostname + nodeId).

**Legacy defects fixed, not ported:** desktop ignoring `serviceRemoved`; Android `peerLost`
unwired; Android ignoring TXT; discovered candidates never reaching the dialer directly. (No
legacy iOS backend existed; the §4.1 NWBrowser backend is new work.)

## 9. Security invariants (binding, never relaxed)

1. Advertise only the TLS carrier. Never the Unix socket, pipe, HTTP, plaintext WS, web front,
   or gateway; never anything reachable under `local_trust`.
2. Never place credentials, tokens, or any secret in TXT. The TXT record is public broadcast.
3. Everything read from mDNS (instance, TXT, SRV, addresses) is untrusted attacker-controllable
   input until the TLS handshake and SASL login succeed. It selects *what to try*, never *what
   to trust*: no auto-connect, no auto-pin, no version-gate bypass, no auth shortcut.
4. The pin written by TOFU is the handshake-observed leaf fingerprint after explicit user
   confirmation (§6.3) — never the advertised `fp` directly. (Pairing binds the same rule
   cryptographically: pairing spec §10.)
5. Advertisement only after successful non-loopback bind; goodbye on clean shutdown (§3.3–3.4).
6. Fail-closed transport behavior is untouched on every platform, including mobile once §5.3
   lands: no protocol bytes before TLS verification completes (pin path included), SASL-only on
   the network carriers.
7. Parsing robustness: TXT parsing (node-side composition, app-side consumption on all
   backends) treats oversized values, non-UTF-8, duplicate keys, and absurd counts as
   record-drop conditions, not crashes. Fuzz-adjacent unit coverage required (§10).
8. No background scanning on mobile (§4.4); the multicast lock and browse sockets are released
   when discovery surfaces are not visible.

## 10. Verification

Node-first, because `avahi-browse -r _daemon-node._tcp` proves the entire §2 contract with zero
app code — any later app-side failure is then unambiguously app-side.

- **daemon-mdns unit tests:** ServiceSpec→registration mapping, TXT composition (golden record
  for §2.2), shutdown unregisters. `cargo test -p daemon-mdns` in the devShell.
- **bins/daemon:** config gating matrix (§3.3 conditions), node-id mint/persist/override
  (atomicity, corrupt-file re-mint), fp computation from PEM. Existing per-edit cadence
  (`cargo check -p`, name-filtered tests); `just deny` in the stage that adds `mdns-sd`.
- **App unit tests** (`tests/unit/tst_node_discovery.cpp`, mock + a fake record layer feeding
  the shared TXT parser): TXT v1 parsing incl. §9.7 hostile inputs, `txtvers` skip, dedup by
  `node`, upsert/removal/stale timing, address preference and IPv6 scope handling (§5.1), target
  composition, `compatible` gating, availability/permission state transitions (§4.5),
  per-target pin read/write precedence (§5.2). Extend `tst_connection_schemes.cpp` for the
  DTO→(mode, target) round-trip and per-platform scheme availability (§5.3 item 4), and
  `tst_tls_transport.cpp` for per-target pin fallback.
- **TUI tests:** extend `tests/tui/tst_first_run_dialog.cpp` and `tst_settings_page.cpp` with
  the mock backend.
- **GUI/a11y:** new rows, the permission-denied state, and the TOFU dialog carry accessible
  names; a kwin-mcp mock-mode journey (wizard → nearby row → target filled, no auto-connect)
  using `MockNodeDiscovery`.
- **i18n:** all new strings through `qsTr`/`tr` + the 12 catalogs (`scripts/i18n-drift.sh`
  gates).
- **Mobile (stages 8–9, gated on the mobile build lane):** the JNI-free parts of the Android
  backend (name→nodeId mapping, resolve-queue ordering, TXT handling) are unit-tested against
  the seam like any backend. End-to-end browse verification requires **real devices on a real
  LAN**: the Android emulator's multicast/mDNS support is notoriously broken (a pass proves
  little, a failure proves nothing), and the iOS simulator does not faithfully reproduce Local
  Network privacy behavior. The §5.3 transport is verified by connecting a device build to a
  LAN node via `tls` + pin. These runs are release-checklist items, not CI gates.
- **Cross-process LAN smoke** (real node advertising, real app browsing, isolated network):
  valuable but multicast-flaky in sandboxes — an opt-in scenario, NOT added to `just e2e` or any
  default gate.
- Standard repo gates per tier: diff-scoped `just lint`, `just verify-stage` per stage,
  `just verify-landing` + builds at the end. No codec/CDDL gates apply (nothing on the wire
  changed).

## 11. Out of scope / future extensions

- **Runtime advertise toggle over the API** (`MdnsStatus`/`MdnsSet` + invalidation event): only
  if a real need appears; it is a wire-contract change (CDDL + `update-codec`) and an
  admin-authorization question. Do not resurrect a generic `ConfigSet`.
- **Re-resolve by `nodeId` on reconnect** (DHCP churn resilience): browse briefly for the
  remembered `nodeId` before dialing a persisted discovered target; requires persisting
  `nodeId`+hostname alongside the target. Design-ready via the DTO fields; not in v1. Worth the
  most on mobile, where network moves are the norm.
- **WS/WSS carrier advertisement**: serves nobody for discovery — WASM cannot browse regardless
  of carrier, and mobile dials TLS natively per §5.3. Only relevant if a future non-browsing
  consumer wants it; plaintext `ws://` is never advertised regardless.
- **Pairing on mobile**: phase-2 pairing ([`daemon-pairing-spec.md`](daemon-pairing-spec.md)) is
  desktop-first by its own scope; once §5.3 gives mobile the native TLS transport with client
  certificates, extending pairing to mobile (QR camera scanning is the natural entry) is the
  follow-up recorded in pairing spec §12. Until then mobile first contact is TOFU (§6.3) +
  SCRAM.
- **Node-side browse** (node discovering nodes): nothing here precludes it; the §2 contract is
  producer-agnostic.

## 12. Staged delivery

Each stage lands green and independently (per-stage `just verify-stage`). Stages 1–7 are
desktop-complete and do not wait on mobile; stages 8–9 carry an explicit prerequisite: **the
mobile build lane exists, builds, and boots** (those builds are currently unbuilt/untested —
their requirements are bound here so the lane inherits them).

1. **Contract + node identity:** this spec; `node-id` persistence + `node_id` config; no
   behavior change elsewhere.
2. **`daemon-mdns` crate + advertiser wiring:** §3 complete; `just deny` for `mdns-sd`; manual
   `avahi-browse` verification of the §2 golden record.
3. **App discovery core:** `da_discovery` seam + desktop/null/mock backends + shared TXT parser
   + model + qmdnsengine Nix dep; unit tests; no UI yet.
4. **Per-target pins:** §5.2 in `tlsConfigFromSettings` + tests (independent, small, unblocks
   TOFU — and later the pairing ceremony writes through the same store).
5. **GUI surfaces:** `ConnectionPicker` nearby section + TOFU affordance + Settings toggle;
   kwin-mcp mock journey; i18n.
6. **TUI parity:** §7 both surfaces; TUI tests; i18n.
7. **Desktop landing:** `just verify-landing`, builds, opt-in LAN smoke run once on real
   hardware.
8. **Mobile TLS transport (§5.3)** *(prerequisite: mobile build lane)*: flake Qt `ssl` +
   OpenSSL (Android) / SecureTransport (iOS), CMake transport/launcher split, connection-service
   gate narrowing, scheme catalog; verified by a device build connecting to a LAN node via
   `tls` + pin.
9. **Mobile browse backends (§4.1)** *(prerequisite: stage 8)*: Android NSD + iOS NWBrowser
   backends, §4.5 permission UX, §4.6 packaging; real-device LAN smoke per §10.
