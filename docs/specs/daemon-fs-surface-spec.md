# daemon-api filesystem / workspace surface

Status: implemented (this document tracks the shipped surface). Driven by the GUI design at
`../../../daemon-app/docs/file-browser-workspace-design.md`.

## Why

A GUI/TUI file browser and file viewer/editor must read and write workspace files **through
the node**, not off the operator's local disk: a unit's files live in its per-session
`ExecutionEnvironment` (host-spec 7) on whichever node/process runs that engine, and the node
may be reached in-process (`embedded`), over a Unix socket (`local`), or over HTTP/WS
(`remote`). The only correct source of truth is the node surface.

There is no such surface today: agents touch files via the `fs`/`shell` tools
(`tools/daemon-tool-fs`), which go through `ExecutionEnvironment`; `daemon-api` exposes only
checkpoint rewind for workspace state. This spec adds a first-class filesystem surface.

## Design principles (validated against the code)

- **Group `fs_*` on `ControlApi`, not a new sub-trait.** `NodeApi` is a marker supertrait
  (`SessionApi + ControlApi + ModelApi + ProfileApi + CredentialApi + AuthApi`) with a blanket
  impl; adding a 7th sub-trait would force every implementor to add it. The repo convention is
  to group long-tail ops on `ControlApi` with `Unsupported` defaults - we follow it.
- **The FS layer does not touch the running engine.** An engine's `ExecutionEnvironment` is
  moved into its actor and is unreachable by `SessionId`/`UnitId`; durable sessions have no
  resident engine between turns. So the FS layer resolves an `FsRootId` to a **directory path**
  and serves files itself, using the public free function `daemon_core::exec::contain()` +
  `tokio::fs`. This also lets the GUI browse a dormant durable session's workspace.
- **Engine and FS share one root.** A `Provisioner` roots each session's engine
  (`EngineProfile::with_exec`) and the FS resolver at the same directory, so operator and agent
  see one filesystem.
- **Containment is the boundary.** Every path op is `contain(root, requested)`-ed; escapes
  (`..`, absolute paths outside root) are rejected, exactly as the agent `fs` tool is bounded.

## Root kinds (`FsRootId`)

- `Host(String)` - browse the node's **own machine** for discovery, bounded by an operator
  policy. Shipped: the node registers the user's **home directory** as the sole browse root
  (`daemon-node` builds `browse = [("home", $HOME)]`). An operator allowlist of additional roots
  and a recents/bookmarks list are **future** (the policy structure is in place via
  `WorkspaceRoots::with_browse_roots`). Read/discovery only - `fs_write` is rejected for `Host`.
  This is what lets a user discover directories on a *remote* node before binding (the Hermes
  `RemoteFolderPicker` / VS Code Remote "Open Folder" pattern).
- `Workspace` - the node's configured workspace root.
- `Session(SessionId)` - a session/unit's workspace sandbox (its `ExecutionEnvironment` root).

`fs_roots` advertises the available roots so a client can present a picker.

## Workspace binding (the "work on my directory" case)

A session's workspace root is one of two bindings, selected at session open via a new
`SessionOverlay.workspace: Option<WorkspaceBinding>` field (the overlay is `#[serde(default)]`
and already round-trips on the wire, so this is additive; no new "bind" op is needed):

- `Isolated` (default) - `<workspace_root>/<session_id>`, for ephemeral/parallel/untrusted
  agents.
- `Bound(PathBuf)` - the operator-specified directory directly, edited **in place** (mirrors
  Hermes cwd / Cursor workspace). Containment keeps the agent inside it.

The binding is realized by the `WorkspaceRoots` resolver + `EngineProfile::with_exec` closures in
`daemon-node` (`root_profile` for the base profiles, and `resolve_effective` for the per-session
overlay path), which root each engine at `WorkspaceRoots::session_root(id)` (the bound dir when set,
else `<workspace_root>/<session_id>`) and record the resolved root so the FS surface serves the same
directory - replacing the `$TMP/daemon-ws-{session}` default. (`ProcessProvisioner::workspace`
remains unused; the resolver, not the provisioner, owns rooting.)

## API surface

Methods on `ControlApi` (all `#[async_trait]`, `Unsupported` defaults):

- `fs_roots() -> Vec<FsRoot>` - the node's `Host` browse roots + `Workspace` + opened
  `Session` roots.
- `fs_list(root, dir, show_ignored) -> Result<Vec<FsEntry>, ApiError>` - one directory's
  children; entries matching the built-in artifact/VCS `IGNORED_NAMES` set (`.git`, `node_modules`,
  `target`, `__pycache__`, ...) are marked `ignored` (and dropped when `show_ignored` is false).
  Full `.gitignore` evaluation is **future** (the `ignore` crate was intentionally not added, to
  keep the dependency/nix-vendoring surface small).
- `fs_stat(root, path) -> Result<FsEntry, ApiError>`.
- `fs_read(root, path, max_bytes) -> Result<FsContent, ApiError>` - bytes + an `FsRevision`
  etag + a `truncated` flag.
- `fs_write(root, path, bytes, base_revision, force) -> Result<FsRevision, ApiError>` - optimistic
  concurrency; `Workspace`/`Session` roots only; `force` overrides the sensitive-path / `Deny` gate.
- `fs_search(root, query) -> Result<FsSearchPage, ApiError>` - server-side content/regex
  search, paginated.
- `fs_watch_after(root, dir, after_seq, max) -> Result<FsWatchPageView, ApiError>` - the
  implemented change cursor: each call re-scans the directory, folds created/modified/removed into
  a bounded per-dir ring keyed by a monotonic `seq` (on-demand diff; no OS watcher), and returns
  events after `after_seq`. `fs_watch(root, dir) -> Result<FsWatchStream, ApiError>` is the
  push-stream seam but ships as the default empty stream; HTTP `GET /fs/watch` provides live
  delivery by polling `fs_watch_after`.

### DTOs

- `FsRootId { Host(String), Workspace, Session(SessionId) }`.
- `FsRoot { id: FsRootId, label: String, kind: FsRootKind, session: Option<SessionId> }`.
- `FsEntry { name, path, kind: FsEntryKind (File|Dir|Symlink), size, mtime_ms, ignored }`.
- `FsContent { bytes, revision: FsRevision, truncated }`.
- `FsRevision { mtime_ms, size }` - a cheap opaque etag (not `daemon_common::Revision`, which
  is profile/skill versioning). Avoids re-reading the file to validate a write base.
- `FsSearchQuery { query, regex, case_sensitive, max_results, page }`,
  `FsSearchPage { hits: Vec<FsSearchHit { path, line, col, preview }>, has_more }`.
- `FsChange { path, kind: FsChangeKind }`, `FsWatchPageView { events: Vec<FsChange>, next_seq }`.

## Write gating (operator writes reuse agent policy, minus the host-ask)

The operator *is* the human, so operator writes do **not** route through
`HostRequestKind::Approval`. They reuse the rest of the agent gating:

- `daemon_core::approval::is_sensitive_path` (`.git`/`.ssh`/dotenv/keys) + the session's
  `ApprovalMode`: `Deny` and sensitive paths block unless an explicit `force` is set.
- Capture a checkpoint via `CheckpointStore::capture` before mutating (a transient
  `LocalEnvironment::new(root)` is built to satisfy the `ExecutionEnvironment` it expects), so
  operator edits are rewindable just like agent edits.
- Reject with `ApiError::Conflict` when `base_revision` is stale, or when the target `Session`
  is live and mid-turn (avoid racing the engine).

## Transports

`fs_*` ride the shared `dispatch`, so the Unix socket and FFI need no changes. The HTTP/WS
adapter gets them via `POST /api` for free; `fs_watch` adds an SSE route (`GET /fs/watch`)
mirroring `subscribe_sse`, with `fs_watch_after` as the socket/long-poll cursor.

The new enum variants + the defaulted `SessionOverlay.workspace` field are additive, so
`API_WIRE_VERSION` / `WireVersion::CURRENT` is **kept at 14** (the cddl labels the FS ops as wire
v14); no bump is required.

## Out of scope (future)

- **Placed/remote *child unit* browsing** (distinct from connecting to a remote node, which
  `Host` browse covers): a unit deep in the tree running in another process/host. `UnitNode`
  carries no location field today; this needs `(UnitId -> host)` on the wire plus an
  `FsCall`/`FsReply` frame over the placement cut (`daemon-host/src/cut.rs`) so the parent
  proxies `fs_*` to the owning child - the same pattern `RemoteStoreClient` already uses.
- Collaborative / OT-CRDT editing; LSP / completion.
