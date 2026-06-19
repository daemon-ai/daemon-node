# daemon-core GUI Surfaces: Embeddable Core, One Protocol, Thin Shells

How to approach the GUI/front-end targets in the Rust rewrite. The thesis is short and the whole
document follows from it:

> **Build one embeddable core library and one typed protocol crate. Everything else — TUI,
> desktop, web, editor, gateway, API — becomes a thin wrapper around that boundary. The core can
> then be embedded anywhere, and no surface owns runtime logic.**

This is the front-end companion to the host-interface boundary in
[daemon-core-host-interface.md](daemon-core-host-interface.md) (the typed
`AgentCommand`/`AgentEvent`/`HostRequest`/`TurnSummary` contract), the runtime model in
[daemon-core-runtime-model.md](daemon-core-runtime-model.md) (the agent actor), and the redesign
scoping in [daemon-core-redesign.md](daemon-core-redesign.md). It is written against the current
Python/JS GUI stack mapped in [hermes-desktop-architecture.md](../../../../docs/research/hermes/hermes-desktop-architecture.md)
and [hermes-gui-surfaces.md](../../../../docs/research/hermes/hermes-gui-surfaces.md).

> **Status — downstream / deferred.** GUI/TUI/desktop is **out of scope** for the current
> in-process `daemon-core` + `daemon-host` vertical slice and is explicitly deferred (see
> [daemon-core-spec.md](daemon-core-spec.md) §1.2 non-goals). This doc is the *direction* for that
> later track, not a committed design. Crate names here (`daemon-core`/`daemon-protocol`/
> `daemon-store`/`daemon-server`) and the event model are aligned to the authoritative engine spec;
> nothing in it is built until the engine/host seams are proven.

---

## 1. Reframe: it is not "five GUIs to port"

The Python/JS stack today is, in effect:

- **Five front-ends:** Electron desktop, web SPA, Ink TUI, plus the Tauri installer and the ACP
  editor adapters.
- **Three duplicate gateway clients:** `apps/shared/src/json-rpc-gateway.ts` (desktop),
  `web/src/lib/gatewayClient.ts` (web), `ui-tui/src/gatewayClient.ts` (TUI) — three
  hand-maintained implementations of the same JSON-RPC dialect.
- **Two backends bolted together:** FastAPI REST (`hermes_cli/web_server.py`) for management +
  `tui_gateway` JSON-RPC for chat, glued inside one `hermes dashboard` process.
- **A large amount of plumbing whose only job is to make non-Python front-ends reach Python:**
  the desktop `bootstrap-runner` + `install.sh/ps1` stage machine, native-deps staging, ephemeral
  port handshake (`HERMES_DASHBOARD_READY port=…`), the `dashboard-token` adoption dance, the
  Node→Python TUI sandwich (`/api/pty` spawning `ui-tui` spawning `tui_gateway.entry`).

Almost all of that exists for one reason: **the Python core is not a library the other surfaces
can link.** Each surface had to re-implement the client, and the runtime had to be reached as a
spawned subprocess over a socket.

Rust removes that constraint. The core is a crate. So the real design is not "port five GUIs" — it
is:

```
one core library  +  one protocol crate  +  one server  →  N thin shells
```

---

## 2. The target shape

```mermaid
graph TD
  subgraph ws [one cargo workspace]
    CORE[daemon-core<br/>agent actor · typed Conversation · resilience]
    PROTO[daemon-protocol<br/>AgentCommand / AgentEvent / HostRequest / TurnSummary<br/>serde + non_exhaustive]
    STORE[daemon-store<br/>SessionStore trait · sqlite + fake]
    SERVER[daemon-server<br/>axum: WS/SSE + embedded static assets<br/>embeds core in-process]
  end

  subgraph shells [thin shells / adapters]
    TUI[ratatui TUI]
    DESK[Tauri desktop]
    WEB[web SPA]
    ACP[editor ACP]
    API[HTTP / SSE API]
    MCPB[MCP bridge]
  end

  CORE --- PROTO
  CORE --- STORE
  SERVER --> CORE
  TUI -->|in-proc channels| CORE
  DESK -->|in-proc channels| CORE
  WEB -->|WS/SSE| SERVER
  ACP -->|in-proc or stdio| CORE
  API --> SERVER
  MCPB --> CORE
  TUI -. remote .->|same protocol over WS| SERVER
  DESK -. remote .->|same protocol over WS| SERVER
```

Three load-bearing decisions; framework choices fall out of them.

### Decision A — one protocol crate, generated for TS

`daemon-protocol` holds the typed boundary as serde enums (the shape is already drafted in
[daemon-core-host-interface.md](daemon-core-host-interface.md), authoritatively pinned in
[daemon-core-spec.md](daemon-core-spec.md) §17):

- **In-process hosts** (TUI, Tauri-embedded, ACP) use the Rust enums directly over typed channels
  — **zero serialization, no JSON-RPC, no socket.** Event delivery follows the **lossless-primary +
  `seq`-resync** model (daemon-core-spec.md §17.1 item 1), not a lossy `broadcast`: a slow consumer
  applies backpressure or resyncs from `SessionStore`, never silently drops events.
- **Out-of-process hosts** (browser, remote desktop, API) get the **same** enums serialized over
  WebSocket/SSE.
- TS surfaces **generate** their types from the Rust enums (`ts-rs` / `typeshare`), so the wire
  contract cannot drift. This single move **deletes two of the three duplicate gateway clients**:
  there is one source of truth, and the TS client is generated from it.
- Versioning is explicit from day one: `#[non_exhaustive]` + a serde tagging policy so new event
  variants never break a deployed UI client (one of the open questions called out in the
  host-interface doc).

### Decision B — collapse the backend into the core (single binary)

In Python, "the backend" is two servers welded together and reached by spawning. In Rust it is one
crate embedded in one process:

- A single static binary contains `daemon-core` + `daemon-server` (axum) + embedded web assets +
  a PTY crate (`portable-pty`) where a terminal is needed.
- **Local desktop/TUI:** no subprocess, no ephemeral port, no token handshake — call the
  in-process actor directly.
- **Dashboard mode:** the *same* binary additionally runs axum to serve remote browsers/desktops
  over the protocol.
- The standalone **"gateway" stops being a separate component** — it is just the core exposing its
  session protocol over a transport.

### Decision C — one frontend, two shells (stop duplicating web vs desktop)

Today `web/` and `apps/desktop/` are two separate React apps that duplicate the gateway client,
themes, i18n, model picker, and most of the chat surface. The rewrite is the moment to merge them:
**one frontend codebase, rendered in two shells** — a plain browser (no native APIs) and a Tauri
window (native APIs via Tauri commands). The browser/desktop difference becomes a capability layer,
not a second application.

---

## 3. The embedding spectrum (what Rust unlocks)

Because the core is a library, each surface chooses *how tightly* to embed. Same code path, three
deployment shapes:

| Mode | Who | How it talks to the core | Transport |
| --- | --- | --- | --- |
| **In-process actor** | TUI, local desktop, ACP, tests | links `daemon-core`, sends `AgentCommand`, subscribes to `AgentEvent` | typed channels (lossless-primary), no serialization |
| **In-process + server** | dashboard / self-hosting | same binary also runs axum exposing the protocol | WS/SSE for remote clients |
| **Out-of-process** | remote desktop, browser, CI, SDK | connects to a remote `daemon-server` | serialized `daemon-protocol` |

Python had to *fake* this spectrum with subprocess spawning, port handshakes, and token adoption.
In Rust it is one trait boundary with three call sites. **This is the property that makes "embed
our project anywhere" true rather than aspirational** — embedding is just choosing a mode.

---

## 4. Per-surface plan

| Surface | Approach | What collapses |
| --- | --- | --- |
| **TUI** | **ratatui + crossterm**, linking `daemon-core` in-process | Deletes the Node + Ink + Python three-language sandwich and `/api/pty`; no IPC for local use |
| **Desktop** | **Tauri 2**, web frontend, core embedded in the binary | Drops Electron's weight *and* the Python spawn / bootstrap / token machinery; keeps the rich renderer |
| **Web** | Same SPA served by `daemon-server` | Replaces FastAPI `web_server.py`; becomes the no-native-APIs shell of the shared frontend |
| **Installer** | Shrinks to a thin self-installer or folds into desktop first-run | The static binary removes the bootstrap stage machine's reason to exist |
| **Gateway** | Not a separate component | It is the core exposing the protocol over a transport |
| **ACP / API / MCP** | Thin adapters over the same protocol | Each is an encoding of `AgentCommand`/`AgentEvent`, not a bespoke integration |

### 4.1 TUI — the unambiguous Rust win

`ratatui` is mature and idiomatic. The local TUI links `daemon-core` and drives the actor directly;
remote use connects over the same protocol. This removes the most awkward part of the current stack
(a Node Ink app spawning a Python gateway, embedded in xterm by the dashboard) and replaces it with
a single linked crate.

### 4.2 Desktop — Tauri shell, keep the rich renderer (for v1)

The desktop renderer is the crown jewel of the existing UX: the incremental assistant-ui streaming
runtime, virtualized transcript with a render budget, shiki/katex/ANSI rendering, the composer
(slash/`@`/queue/steer/voice), and perf instrumentation. Rewriting that in an immature Rust GUI
toolkit would discard a lot of hard-won work.

**Tauri 2** keeps that frontend while replacing Electron and, crucially, **embeds the agent core
inside the desktop binary** — so there is no Python to spawn, no venv to bootstrap, no port/token
handshake. The team already ships a Tauri app (the installer), so the toolchain is in hand.

### 4.3 Web — one frontend, served by the core

The browser is the shell of the shared frontend without native APIs, served as embedded static
assets by `daemon-server`. Keep the good ideas from today's dashboard: the plugin slot system, YAML
themes, and the schema-driven config editor.

### 4.4 Installer & gateway — mostly delete

The elaborate `bootstrap-runner` + `install.sh/ps1` stages, native-deps staging, and update
hand-off exist because a Python venv + Node + `node-pty` is hard to ship. A single static binary
makes installation "place one binary + register PATH/app/auto-update." Keep a tiny self-installer
for OS integration, or fold it into the desktop app's first run. The standalone gateway disappears
into the core.

---

## 5. The one genuinely open fork: frontend *language*

Rust settles core/server/TUI cleanly. It does **not** settle what renders the desktop/web chat
surface. This is the real decision to make consciously.

| | **Option A — pragmatic (recommended for v1)** | **Option B — Rust-maximal** |
| --- | --- | --- |
| Web + desktop frontend | React/TS, **one codebase, two shells** (browser + Tauri) | Dioxus / Leptos (one Rust UI across web-WASM + desktop) |
| TUI | ratatui | ratatui |
| Wire types | generated from Rust (`ts-rs`/`typeshare`) | native Rust types throughout |
| Risk | low — reuses the mature streaming chat renderer | high — Rust GUI is still immature for streaming-markdown-heavy chat (incremental render, virtualized transcript, code highlighting) |
| Payoff | deletes Electron + Python-spawn + 2 of 3 clients; keeps UX | also removes TS entirely; one language end to end |

**Recommendation:** ship **Option A** for v1 and treat **Option B** as a migration target once the
protocol and core are stable. The TUI is ratatui in both options, so it can be the first native-Rust
surface regardless.

---

## 6. What to keep from the current design

A Rust rewrite is a chance to drop accidental complexity, not proven ideas. Keep:

- **The event model.** The JSON-RPC streaming event vocabulary is good; the typed boundary is a
  refinement of it, not a replacement.
- **The desktop renderer's streaming architecture** (incremental runtime, render budget,
  stream-stable selectors) — if Option A, port it largely intact.
- **The web plugin slot system, per-profile theming, and schema-driven config UI.**
- **Read-only session ops behind `SessionStore`, not `AgentCommand`** (title/usage/history/search)
  — this is what keeps the command enum from degenerating into the callback bag the redesign
  rejects.

---

## 7. Phasing

1. **`daemon-protocol` crate first** — the typed boundary + serde + TS generation. Everything else
   depends on it, and it is testable on its own (script commands, assert the event transcript).
2. **`daemon-core` actor + `SessionStore`** exposing the protocol in-process.
3. **TUI (ratatui)** as the first thin shell — proves the in-process embedding end to end with no
   server.
4. **`daemon-server` (axum)** — the same protocol over WS/SSE + embedded assets; proves
   out-of-process embedding.
5. **Web SPA** against the server, then **Tauri desktop** wrapping the same frontend with the core
   embedded.
6. **Installer shrink / fold-in; ACP, HTTP API, MCP** as additional thin adapters over the now-proven
   protocol.

---

## 8. One-line summary

Stop building front-ends that each re-reach a spawned backend. Build **one embeddable core and one
typed protocol crate**; then the TUI, desktop, web, editor, API, and MCP surfaces are thin wrappers
that pick an embedding mode. That deletes Electron, the Python-spawn / bootstrap / token machinery,
the standalone gateway, most of the installer, and two of the three duplicate gateway clients — the
bulk of the accidental complexity in today's GUI stack — and makes "embed Hermes anywhere" a
property of the architecture rather than a one-off integration each time.
