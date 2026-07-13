# Swarm P1 + Transport — lane ledger **B2** (iroh gossip control plane + self-hosted relay)

Lane **B2** of the "Swarm P1 + Transport" program, Wave 2. Owns
`crates/swarm/daemon-swarm-net/src/iroh_gossip.rs` + `lib.rs` re-exports + the crate `Cargo.toml`
(iroh-relay features / dev-deps) + the relay dev runner and its docs under the lane-owned
`crates/swarm/daemon-swarm-net/dev/`. Read `swarm-p1-ledger.md` (program ledger) and
`swarm-mvp-ledger.md` (frozen MVP surfaces) first — this ledger only records B2's deltas.

## Base + branch

- **Repo:** `daemon-node` (worktree `/home/j/experiments/daemon-worktree/swarm-runtime`).
- **Branch:** `swarm/b2`, based at `bd2cb5b` (`mirror(merge-1): freeze Wave-1 interfaces`) on
  `integrations/swarm-p1` (Merge 1 HEAD).
- **Frozen surfaces consumed (never modified):**
  - `ControlPlane { async publish(&[u8]), subscribe() -> ControlSubscription }` +
    `ControlSubscription` (`transport.rs`) — the seam `IrohGossip` implements.
  - `Deduper` (`dedupe.rs`) — content-hash (proto blake3) dedupe; the reusable NET-6 rule.
  - `LoopbackGossip` (`gossip.rs`) — the in-process conformance twin `IrohGossip` must match.
  - proto `SignedMessage` (canonical CBOR, ed25519) — signing stays proto-side; the plane carries
    opaque already-signed bytes. `SignedMessage::verify()` is the **consumer's** gate, not the
    plane's (§7.1: gossip is dissemination, never arbitration).
  - `Join.iroh_id: IrohId` (`messages.rs`) — the node-key <-> iroh-key binding the Join flow carries.
  - `FrozenEnvelope::hash()` (`envelope.rs`) — the blake3 envelope hash, our topic-derivation input.
- **FROZEN files (integration owner only, never touched here):** root `Cargo.toml` (iroh pins already
  in `[workspace.dependencies]`), `deny.toml`, `flake.nix` (ships `iroh-relay` 1.0.0 on the devShell).

## The pin (CRITICAL Wave-2 note — the plan says 0.97, the tree resolved 1.0)

Per the program ledger "Resolved dependency pins": iroh **0.97/0.98 are unresolvable** against the
frozen `sha2 0.11` tree (they pull `sha2 =0.11.0-rc.*`, disjoint from slack-morphism's stable
`sha2 0.11.0`). iroh **1.0 dropped sha2 entirely**, so the tree pins **iroh 1.0.2 / iroh-gossip
0.101.0 / iroh-relay 1.0.2**, all behind `daemon-swarm-net`'s off-default `iroh` feature. B2's
mandate: **port Psyche's verified 0.97 gossip patterns (reference pack) to the iroh 1.0 API and
record every delta.** Good news up front: the reference checkout
(`/home/j/experiments/decentralised-llm-training/psyche`, pinned 0.97) already uses the modernized
iroh surface (`Endpoint::builder(presets::N0)`, `EndpointId`/`EndpointAddr`, `address_lookup`,
`QuicTransportConfig`, `Gossip::builder().spawn`), so 0.97 -> 1.0 is small and mostly module-path
churn. The delta table below is the program's iroh-churn mitigation (Risk 4).

## 0.97 -> 1.0 API delta table (Psyche anchor -> iroh 1.0.2 / iroh-gossip 0.101.0)

| Concern | Psyche 0.97 anchor | iroh 1.0.2 / gossip 0.101.0 | Delta / note |
|---|---|---|---|
| Endpoint build | `lib.rs:343-378` `Endpoint::builder(presets::N0)` | `Endpoint::builder(presets::Minimal)` (or `N0`) | Same shape. We use `presets::Minimal` (crypto provider only) — **no** DNS/pkarr discovery (N0 preset adds the public n0 DNS lookup; we prefer explicit roster addrs, per brief). `presets::{Empty,Minimal,N0,N0DisableRelay}` all present. |
| Node identity type | `EndpointId` (psyche 0.97 already) | `EndpointId` (alias of `PublicKey`) | 0.97 had already renamed `NodeId` -> `EndpointId`; unchanged into 1.0. Our proto `IrohId` is 32 raw bytes -> `EndpointId::from_bytes`. |
| Node address type | `EndpointAddr` (`lib.rs:337`) | `EndpointAddr { id, addrs: BTreeSet<TransportAddr> }` | 1.0 generalizes to `TransportAddr::{Relay(RelayUrl),Ip(SocketAddr),Custom}`. Builders `EndpointAddr::new(id).with_relay_url(u).with_ip_addr(sa)`. (0.97 had explicit `relay_url` + `direct_addresses` fields.) |
| Static/explicit discovery | `local_discovery::LocalTestDiscovery` / `MemoryLookup` (`lib.rs:366-368`) | `iroh::address_lookup::memory::MemoryLookup` | Available on **default** iroh features (not `test-utils`-gated). `MemoryLookup::new()`, `.add_endpoint_info(EndpointAddr)`; add post-bind via `endpoint.address_lookup()?.add(lookup)`. This is our explicit-roster dialing path — **no public discovery service** for this program. |
| Relay mode | `RelayMode::{Disabled,Default,Custom(RelayMap)}` (`lib.rs:350-354`) | identical | `RelayMap: FromIterator<RelayUrl>` + `From<RelayUrl>`; `RelayUrl: FromStr`. We build `RelayMode::Custom(RelayMap::from_iter(urls))` from config, `RelayMode::Disabled` when no URLs. |
| QUIC transport tuning | `QuicTransportConfig::builder().max_idle_timeout(..).keep_alive_interval(..)` (`lib.rs:344-348`) | same (`iroh::endpoint::{QuicTransportConfig,QuicTransportConfigBuilder}`) | `max_idle_timeout(Option<IdleTimeout>)` where `IdleTimeout: TryFrom<Duration>`; `keep_alive_interval(Duration)`. We keep Psyche's **120s idle / 5s keepalive**. |
| Gossip init | `Gossip::builder().max_message_size(4096).membership_config(Hyparview).broadcast_config(Plumtree).spawn(ep)` (`lib.rs:459-474`) | **identical** in 0.101 | `HyparviewConfig{active_view_capacity, shuffle_interval, neighbor_request_timeout,..}`, `PlumtreeConfig{graft_timeout_2, message_cache_retention, message_id_retention,..}`. We keep 4096 + Psyche's tuning. |
| Subscribe + bootstrap | `gossip.subscribe(topic, bootstrap_endpoint_ids).await?.split()` (`lib.rs:500-503`) | `gossip.subscribe(topic, Vec<EndpointId>).await?` -> `GossipTopic`; `.split() -> (GossipSender, GossipReceiver)` | Identical. Bootstrap arg is `Vec<EndpointId>`. `subscribe_and_join` variant waits for first neighbor. |
| Broadcast | `gossip_tx.broadcast(Bytes).await` (`lib.rs:580`) | `GossipSender::broadcast(bytes::Bytes)` | Identical. `Bytes: From<Vec<u8>>`. |
| Join peers (roster update) | `gossip_tx.join_peers(Vec<EndpointId>).await` (`lib.rs:557`) | `GossipSender::join_peers(Vec<EndpointId>)` | Identical. |
| Receive events | `iroh_gossip::api::Event::{Received(Message{content,delivered_from,scope}),NeighborUp,NeighborDown,Lagged}` (`lib.rs:898-946`) | **identical** in 0.101 | `GossipReceiver: Stream<Item=Result<Event, ApiError>>`; tracks `neighbors()` internally. |
| Neighbors | `gossip_rx.neighbors()` (`lib.rs:867-869`) | `GossipReceiver::neighbors()` / `GossipTopic::neighbors()` | Identical. |
| Router / ALPN | `Router::builder(ep).accept(iroh_gossip::ALPN, gossip).spawn()` (`router.rs:32-46`) | identical (`iroh::protocol::Router`, `iroh_gossip::ALPN`) | We accept **only** the gossip ALPN (no blobs — P4; no model-sharing). |
| Topic derivation | `sha256("psyche gossip" ++ run_id)` -> `TopicId` (`util.rs:5-13`) | `TopicId::from_bytes([u8;32])` | **DELTA (ours):** `blake3(frozen envelope hash)` -> `TopicId`. blake3 not sha256 (spec §6.4 content-addressing); envelope hash not run_id (binds the topic to the exact frozen run, not just its name). Uses `daemon_swarm_proto::blake3_hash`. |
| Signed message | postcard `SignedMessage{from,data,sig}` (`signed_message.rs:17-38`); verify-on-receive (`lib.rs:898-922`) | n/a — we do not use iroh's signing | **DELTA (ours):** our `SignedMessage` is **canonical CBOR + ed25519** via `daemon_swarm_proto`; the plane carries **opaque already-signed bytes** and never signs/verifies (verify is the consumer's gate). Psyche signs *inside* the network layer with the iroh key; we sign *outside* with the **node** key and keep the iroh key transport-only. |
| Gossip message id | (implicit) | `MessageId = blake3(content)`, **validated** on receive; deduped for `message_id_retention` (2 min) — `proto/plumtree.rs:26-36,212-214` | The linchpin for the rebroadcast design below. |
| In-process test harness | `router.rs` tests (`MemoryLookup`, `RelayMode::Disabled`) | `iroh-gossip-0.101/src/net.rs::gossip_net_smoke`, `api.rs::test_rpc` | Our multi-node harness ports these: N endpoints, `RelayMode::Disabled`, `MemoryLookup` seeded with each peer's `EndpointAddr`, subscribe with bootstrap ids. |
| Relay server (self-host) | none (Psyche hardcodes hostnames) | `iroh-relay` 1.0 binary `--dev` (plain HTTP, port 3340) | **DELTA (ours):** self-hosted from the start; relay URLs pinned in the envelope, consumed by config. See "Relay dev-run story". |

## Rebroadcast / dedupe design decision (the round-critical delivery-assurance knob)

**Problem.** iroh-gossip guarantees ~99.9% delivery. Psyche periodically re-broadcasts live results
with a **nonce bump** (`client.rs:490-505`) so a peer that missed a message still gets it. But two
dedupe layers now interact:

1. **iroh-gossip's Plumtree layer** dedupes by `MessageId = blake3(broadcast content)` for
   `message_id_retention` (2 min). Re-broadcasting **identical bytes** is a no-op at this layer — it
   will not re-flood. So a naive rebroadcast defeats delivery-assurance.
2. **Our app-layer `Deduper`** dedupes by `blake3(inner signed message bytes)` and must drop
   duplicates so the consumer sees each message once (NET-6).

**Resolution — a thin rebroadcast frame OUTSIDE the signed payload.** Every gossip broadcast carries
`frame = [nonce: u64 little-endian][payload: already-signed CBOR bytes]`. The receiver strips the
8-byte nonce prefix and dedupes by `Deduper::id(payload)` (the **inner** bytes). Therefore:

- **First publish:** `nonce = 0`. Floods once; intermediate gossip hops re-transmit the *same* frame
  bytes -> same `MessageId` -> efficient single flood + built-in Plumtree dedupe across the mesh.
- **Rebroadcast (origin only, every `interval`, default on):** bump that payload's nonce ->
  **different frame bytes -> different `MessageId`** -> Plumtree treats it as new -> **re-floods**.
  Receivers strip the frame and the app-layer `Deduper` drops it (payload already delivered).
- **Net effect (exactly what we want):** gossip-layer id changes force re-flood for
  delivery-assurance; app-layer content-hash dedupe guarantees one delivery. The signature is over
  the inner payload only, so a nonce bump never invalidates it.

**Verified against 0.101:** `MessageId::from_content` is `blake3(content)` and is *validated* on
receive (`proto/plumtree.rs:26-36,212-214`), so changing the outer bytes is both necessary and
sufficient to force a re-flood. Frame overhead is 8 bytes; signed control messages are sub-4 KB
(§7.1), so `payload <= max_message_size - 8` (4088 B) — enforced/asserted in `publish`.

**Self-delivery (matching `LoopbackGossip`).** `IrohGossip::publish` also fans the message out to the
node's **own** local subscribers (after the same dedupe), because a single `IrohGossip` instance is
one node and iroh-gossip does not loop a broadcast back to its origin. This keeps the delivery
contract identical to `LoopbackGossip` ("publish -> every subscriber, once"), which is what the
parametric conformance suite asserts over both impls and what B3's `RoundEngine` builds on.

## Relay dev-run story (TLS / insecure findings — investigated honestly)

- The devShell ships **iroh-relay 1.0.0** on PATH (flake Wave-0 lane). Its binary CLI (verified from
  the `iroh-relay` 1.0.2 crate `main.rs`) has a first-class dev mode: **`iroh-relay --dev`** runs
  "in localhost development mode over **plain HTTP**", default bind **`[::]:3340`**, and "**ignores
  any config file fields pertaining to TLS**" (it sets `dangerous_http_only`). This is the
  insecure/dev relay for LAN/loopback — **no certs, no ACME/LetsEncrypt** needed.
- Production/TLS path (out of scope here, recorded): a TOML `--config-path` with a `[tls]` section
  (`cert_mode = "manual" | "letsencrypt"`, `manual_cert_path`/`manual_key_path` or ACME
  `hostname`/`contact`), plus optional `enable_quic_addr_discovery` (requires TLS). The envelope
  pins the relay **URLs** the run author operates; B2's config only consumes them.
- **Dev runner:** `crates/swarm/daemon-swarm-net/dev/run-relay.sh` wraps `iroh-relay --dev`
  (port overridable via `$IROH_RELAY_PORT`), prints the `http://localhost:<port>` relay URL to feed
  into `IrohGossipConfig.relay_urls` / the envelope. Ops notes (ports, URL shape, how to point a
  client at it) live alongside in `dev/README.md`.
- **Relay-path test:** an `#[ignore]`-free live test spawns the devShell `iroh-relay --dev` binary,
  points clients at it via `RelayMode::Custom` with **relay-url-only** roster addrs, and asserts
  gossip forms + delivers through the relay. It **skips cleanly** (returns early) when `iroh-relay`
  is not on PATH (standalone checkout without the devShell). The spawn uses a single, reasoned
  `#[allow(clippy::disallowed_methods)]` on the test-only helper (the workspace bans
  `std::process::Command::new` for production shell-exec; spawning a known dev tool from a test is
  the sanctioned exception, documented here).

## Exported seams (freeze at Merge 2)

1. **`IrohGossip: ControlPlane`** — construction surface `IrohGossip::connect(IrohGossipConfig).await`
   + `IrohGossipConfig { secret_key: [u8;32], relay_urls: Vec<String>, roster: Vec<IrohPeer>,
   topic_input: [u8;32] (frozen envelope hash), rebroadcast: RebroadcastConfig, bind_addr:
   Option<SocketAddr> }`; `IrohPeer { endpoint_id: [u8;32], direct_addrs: Vec<SocketAddr>, relay_url:
   Option<String> }`; `RebroadcastConfig { enabled: bool (default true), interval: Duration,
   ring_capacity: usize }`. Accessors: `node_id() -> [u8;32]` (the iroh `EndpointId`, for
   `Join.iroh_id`), `local_peer() -> IrohPeer` (dialable self, for roster wiring), `neighbor_count()`.
   Roster-update: `update_roster(&self, Vec<IrohPeer>)` (re-seeds discovery + `join_peers`, capped ~3
   neighbors — `ensure_gossip_connected` semantics). `shutdown(&self)`.
2. **The dev relay runner + config shape** — `dev/run-relay.sh` + `dev/README.md`; the `relay_urls`
   field shape in `IrohGossipConfig` (envelope carries them; config consumes them).
3. **The parametric `ControlPlane` conformance suite** — `tests/control_plane_conformance.rs`: a
   mesh-factory-parametric behavior suite run over `LoopbackGossip` (always) and `IrohGossip`
   (behind `iroh` + an in-process multi-node harness).

## Planned slices (TDD; each commit passes lane gates)

1. `mirror(B2): ledger` (this file) — first commit.
2. `feat(swarm-net): IrohGossip endpoint+gossip+topic scaffold behind iroh feature (green)`.
3. `feat(swarm-net): IrohGossip publish/subscribe over rebroadcast frame + dedupe (green)`.
4. `feat(swarm-net): roster bootstrap + update (ensure_gossip_connected) (green)`.
5. `feat(swarm-net): parametric ControlPlane conformance (loopback + iroh) (green)`.
6. `feat(swarm-net): NET-6 + iroh multi-node harness (fanout/roster/partition) (green)`.
7. `feat(swarm-net): self-hosted relay dev runner + relay-path test (green)`.

## Gates (B2)

`cargo fmt --check` && `cargo clippy --workspace --all-targets -- -D warnings` &&
`cargo clippy -p daemon-swarm-net --features iroh --all-targets -- -D warnings` &&
`cargo test --workspace` && `cargo test -p daemon-swarm-net` &&
`cargo test -p daemon-swarm-net --features iroh` && `typos docs/specs`. Known pre-existing flake
(never modified): the `daemon-conformance` detached-delegation trio — pass-in-isolation = green.

## Results / deviations

### Commits (oldest -> newest)

| Commit | Subject |
|---|---|
| `mirror(B2): ledger` | this ledger |
| `feat(swarm-net): IrohGossip control plane over iroh 1.0 gossip (green)` | the plane + config + rebroadcast frame |
| `feat(swarm-net): parametric ControlPlane conformance suite (loopback + iroh) (green)` | shared behavior suite + iroh harness |
| `feat(swarm-net): NET-6 + iroh multi-node harness + self-hosted relay dev runner (green)` | NET-6 + churn + relay-path + dev runner |
| `mirror(B2): ledger results` | this results section |

### Test counts (all green)

- `cargo test -p daemon-swarm-net` (default, no iroh): **69** = 67 lib + 2 loopback conformance.
- `cargo test -p daemon-swarm-net --features iroh`: **82** = 71 lib (67 + 4 `iroh_gossip` unit
  tests: frame/topic/relay-map) + 4 conformance (2 loopback + 2 iroh) + 7 `iroh_gossip` integration.
- B2 net-new: **15** tests (4 unit + 2 loopback-conformance + 2 iroh-conformance + 7 iroh-integration).
- The 7 iroh integration tests: `signed_gossip_bad_sig_rejected` (NET-6),
  `ws_gossip_duplicate_message_dedupes` (NET-6), `iroh_fanout_reaches_all_nodes`,
  `iroh_roster_update_reconnects_new_peer`, `iroh_partition_rejoin_smoke`,
  `rebroadcast_refloods_without_duplicate_delivery`, `relay_path_delivers_through_self_hosted_relay`.
  The **relay-path test ran green** (not skipped) — the devShell `iroh-relay` 1.0.0 is on PATH and
  the plain-HTTP `--dev` relay routed gossip end to end.

### Full gate results

`cargo fmt --check` OK · `cargo clippy --workspace --all-targets -- -D warnings` OK ·
`cargo clippy -p daemon-swarm-net --features iroh --all-targets -- -D warnings` OK ·
`cargo test --workspace` OK **except** the documented pre-existing `daemon-conformance`
detached-delegation/operator-steer trio flake (this run: `injected_input_reaches_a_parked_durable_
session_via_the_store_seam` failed under the full parallel run, **passes in isolation** — verified;
never modified, and impossible for B2 to have caused since all iroh code is behind the off-default
`iroh` feature and is not compiled in the default workspace build) · `cargo test -p daemon-swarm-net`
+ `--features iroh` OK · `typos docs/specs` OK.

### Findings / deviations (what Merge-2 + B3 must know)

1. **Direct loopback mesh forms with NO relay.** The in-process harness uses
   `RelayMode::Disabled` + `MemoryLookup` seeded with each node's `127.0.0.1:<port>` `EndpointAddr`;
   the mesh forms and delivers in ~1 s. The relay is only needed for NAT/WAN reachability — the
   relay-path test proves that path separately.
2. **`presets::Minimal`, not `presets::N0`.** `N0` adds the public n0 DNS/pkarr discovery; the
   program prefers explicit roster addressing (no public discovery service), so B2 uses `Minimal`
   (crypto provider only) + `MemoryLookup`. If a future deployment wants global discovery it is a
   one-line preset swap, but the envelope-roster path is the sanctioned one.
3. **Self-delivery is intentional.** `IrohGossip::publish` fans the message to the node's own local
   subscribers (after dedupe) as well as gossiping it, because a single iroh node does not receive
   its own broadcast back. This keeps the delivery contract byte-identical to `LoopbackGossip`
   ("publish -> every subscriber, once"), which the parametric conformance suite asserts over both.
   B3's `RoundEngine` therefore sees the *same* contract whether it runs on loopback or iroh.
4. **Rebroadcast/dedupe interplay is verified, not assumed.** `MessageId = blake3(content)` is
   *validated* by iroh-gossip 0.101 on receive, so the nonce frame is both necessary and sufficient
   to force a re-flood; `rebroadcast_refloods_without_duplicate_delivery` proves the app `Deduper`
   still yields exactly one delivery. Default rebroadcast is **on, 10 s, ring 32**; B3 can tune per
   round-criticality (Commitments/Attestations want it on; Heartbeats can leave it default).
5. **Relay dev mode is plain HTTP, port 3340, zero certs** (`iroh-relay --dev`). `dev/run-relay.sh`
   + `dev/README.md` document it; production TLS/ACME is recorded but out of P1 scope (P2 WAN gate).
6. **One scoped `#[allow(clippy::disallowed_methods)]`** on the test-only `spawn_dev_relay` helper
   (the workspace bans `std::process::Command::new`; spawning a known dev tool from a test is the
   sanctioned, documented exception). No production code spawns anything.
7. **B3 wiring notes:** construct `IrohGossip::connect(IrohGossipConfig{..})` with the node's iroh
   secret key, the envelope-pinned `relay_urls`, the admission roster (`IrohPeer` per admitted peer,
   `endpoint_id` = that peer's `Join.iroh_id`), and `topic_input` = `FrozenEnvelope::hash()`. Call
   `node_id()` to fill the local peer's `Join.iroh_id` at join. Call `update_roster(..)` on every
   roster change (admission/drop). The plane carries **already-signed** proto `SignedMessage` bytes
   — sign before `publish`, and `verify()` after `subscribe().recv()` (the plane never verifies).
8. **The `iroh-relay` crate stays declared-but-unreferenced** (machete-ignored): the relay is an
   external binary. If a future lane wants an in-process relay for tests, add `iroh/test-utils`
   (`iroh::test_utils::run_relay_server`) as a dev-feature rather than fighting the `Command` ban.
