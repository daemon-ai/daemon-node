// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

// Phase 4: fs here is the daemon-internal engine inbox/outbox IPC dirs under the node data root
// (not attacker-influenced); raw fs allowed file-wide. No process spawns in this file.
#![allow(clippy::disallowed_methods)]

//! The engine ⇄ activation-seam adapter (host-spec §3.1).
//!
//! `daemon-core` is deliberately free of the durable substrate (it depends only on
//! `daemon-protocol`). [`CoreIncarnation`] is the host-owned bridge that lets the activation layer
//! drive a real engine through the protocol-agnostic [`Incarnation`] seam: it decodes the durable
//! [`SnapshotBlob`] into the engine's typed [`Snapshot`], applies unapplied completions, runs one
//! turn, and maps the engine's terminal/suspension outcome back onto the seam's [`Step`].
//!
//! Background delegation on this path is resolved by [`DelegateResolver`], a built-in
//! [`HostRequestHandler`] that mints the deterministic durable `JobId` the activation outbox keys on
//! — the live management-protocol escalation path is the actor-backed `EngineUnit` (see
//! [`crate::unit`]).

use crate::background::BackgroundSpawner;
use crate::blob_store::BlobStore;
use crate::journal::{JournalFeeder, JournalSink};
use crate::node_api::{decode_overlay, DurableProfileResolver};
use crate::workspace_fs::WorkspaceRoots;
use async_trait::async_trait;
use daemon_activation::{EngineError, EngineFactory, Incarnation, SnapshotBlob, Step, TurnCtx};
use daemon_common::{Epoch, JobId, JournalStreamId, ProfileRef, ReqId, SessionId};
use daemon_core::{
    Completion, Conversation, Effect, Engine, EngineProfile, EventSink, Failure, MockProvider,
    Provider, Snapshot, SystemPrompt, Tool, ToolCall, ToolOutcome, ToolRegistry, ToolResult,
    TurnControl, TurnCx, TurnOutcome,
};
use daemon_protocol::{
    HostRequest, HostRequestHandler, HostRequestKind, HostResponse, HostResponseBody, Outbound,
};
use daemon_store::{JobCommand, JobCompletion, ParkedApproval, SessionStore};
use daemon_telemetry::TraceSigner;
use std::sync::{Arc, Mutex};

/// The store + signer a durable incarnation journals into. Injected by the composition root; when
/// absent the durable path runs without journaling (e.g. the substrate conformance suite).
#[derive(Clone)]
pub struct JournalConfig {
    /// The authoritative store the journal is appended to + sealed in.
    pub store: Arc<dyn SessionStore>,
    /// The node's segment-root signer.
    pub signer: Arc<TraceSigner>,
}

// The provider/credential builder type aliases now live with the [`EngineProfile`] in `daemon-core`
// (the one composition seam); re-exported here for callers that still reference them by this path.
pub use daemon_core::{CredentialBuilder, ProviderBuilder};

/// Builds core-backed [`Incarnation`]s from a shared [`EngineProfile`] — the durable activation
/// path's view of the one engine-construction seam.
#[derive(Clone)]
pub struct CoreEngineFactory {
    profile: EngineProfile,
    journal: Option<JournalConfig>,
    /// The §4.3 background-spawn materializer, when configured. Threaded into every incarnation so
    /// (a) `Effect::Spawn` host requests materialize attached non-joining children, and (b) a
    /// background child session hydrates under its constrained review profile instead of `profile`.
    background: Option<Arc<BackgroundSpawner>>,
    /// The per-session profile resolver (bound profile ref + persisted overlay -> `EngineProfile`),
    /// injected by the node. When set, a durable session with a recorded bound profile rehydrates
    /// from *its own* profile + overlay (the unified resolution path) instead of pinning this
    /// factory's fixed `profile`; `None` (or no recorded binding) falls back to `profile`.
    resolver: Option<DurableProfileResolver>,
    /// The content store + workspace roots for node-mediated artifact transfer (content-transfer
    /// Phase 2a): at a child's terminal completion the incarnation captures its `outbox/` into the
    /// store; on a parent's hydrate it materializes a child's returned artifacts into the parent's
    /// `inbox/`. `None` disables artifact capture/materialization (the legacy `child:{id}` marker).
    content: Option<ContentTransfer>,
    /// The constrained profile a **cron-fired** session (`session_meta.scheduled_job.is_some()`)
    /// runs its turn under (I15/G3): an orchestrator-free, `cron`-tool-free toolset so a scheduled
    /// run cannot self-schedule or self-delegate (runaway prevention). When set it overrides the
    /// resolver/fallback for any scheduled session; `None` leaves cron sessions on the default path.
    cron_profile: Option<EngineProfile>,
}

/// The node-side content-transfer handles threaded into a durable incarnation (blob store +
/// workspace roots), used to capture/materialize delegated artifacts.
#[derive(Clone)]
struct ContentTransfer {
    blobs: Arc<dyn BlobStore>,
    roots: Arc<WorkspaceRoots>,
}

/// A minimal conformance/test delegation tool standing in for the node's `orchestrate` tool on the
/// pure-`daemon-core` paths (the substrate conformance harness + the §17 ⇄ management translation
/// gate), which cannot link the real `daemon-tool-orchestrate` (a higher-level crate). Its `spawn`
/// raises the blocking `HostRequest::Delegate` and yields the durable [`Effect::Delegate`] the
/// engine suspends on, carrying the same `DelegationInput::task(label)` payload the real tool
/// encodes — so the "delegate → suspend → resume → complete" cycle is exercised without the retired
/// `daemon-core` `delegate` tool.
pub struct OrchestrateShim {
    label: String,
}

impl OrchestrateShim {
    /// A shim that labels its delegated work with `label`.
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
        }
    }
}

#[async_trait]
impl Tool for OrchestrateShim {
    fn name(&self) -> &str {
        "orchestrate"
    }

    fn schema(&self) -> &str {
        r#"{"type":"object","properties":{}}"#
    }

    async fn run(&self, call: &ToolCall, cx: &TurnCx<'_>) -> ToolOutcome {
        let req = HostRequest {
            request_id: ReqId(0),
            kind: HostRequestKind::Delegate {
                label: self.label.clone(),
                budget: cx.budget,
            },
        };
        let resp = cx.host.request(req).await;
        let job_id = match resp.body {
            HostResponseBody::Delegated(job) => job,
            _ => JobId::new(format!("{}:unresolved", cx.session_id)),
        };
        // The same structured job payload the real orchestrate tool encodes (task only, no
        // attachments), so the node-side worker seeds the child identically.
        let payload = daemon_protocol::DelegationInput::task(self.label.clone()).encode();
        ToolOutcome {
            result: ToolResult {
                call_id: call.call_id.clone(),
                ok: true,
                content: format!("delegated:{job_id}"),
            },
            effects: vec![Effect::Delegate {
                job: job_id,
                payload,
            }],
            detail: None,
            untrusted: false,
        }
    }
}

impl CoreEngineFactory {
    /// A factory whose engines delegate one unit of background work and then complete — the durable
    /// "delegate → suspend → resume → complete" cycle the substrate conformance suite drives.
    pub fn delegating() -> Self {
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(OrchestrateShim::new("background-work")));
        let profile = EngineProfile::new(
            Arc::new(|| {
                Arc::new(MockProvider::delegating("orchestrate", "work complete"))
                    as Arc<dyn Provider>
            }),
            Arc::new(registry),
            SystemPrompt::new("daemon-core conformance engine"),
        );
        Self {
            profile,
            journal: None,
            background: None,
            resolver: None,
            content: None,
            cron_profile: None,
        }
    }

    /// A factory over a custom provider builder, tool registry, and system prompt.
    pub fn with_provider(
        provider: ProviderBuilder,
        registry: Arc<ToolRegistry>,
        system: SystemPrompt,
    ) -> Self {
        Self {
            profile: EngineProfile::new(provider, registry, system),
            journal: None,
            background: None,
            resolver: None,
            content: None,
            cron_profile: None,
        }
    }

    /// A factory over an already-assembled [`EngineProfile`] (the binary's composition root).
    pub fn from_profile(profile: EngineProfile) -> Self {
        Self {
            profile,
            journal: None,
            background: None,
            resolver: None,
            content: None,
            cron_profile: None,
        }
    }

    /// Inject the verifiable-journal store + signer so every durable incarnation this factory builds
    /// seals its turn into the unified journal (the durable production journaling path).
    pub fn with_journal(mut self, store: Arc<dyn SessionStore>, signer: Arc<TraceSigner>) -> Self {
        self.journal = Some(JournalConfig { store, signer });
        self
    }

    /// Inject the §4.3 background-spawn materializer so this factory's incarnations can spawn
    /// attached, non-joining review children and hydrate them under their constrained profile.
    pub fn with_background(mut self, background: Arc<BackgroundSpawner>) -> Self {
        self.background = Some(background);
        self
    }

    /// Inject the per-session profile resolver so durable sessions rehydrate from their own bound
    /// profile + persisted overlay (unified live/durable resolution) instead of this factory's fixed
    /// profile. Requires the journal store (the source of the session metadata) to be set too.
    pub fn with_session_resolver(mut self, resolver: DurableProfileResolver) -> Self {
        self.resolver = Some(resolver);
        self
    }

    /// Inject the constrained cron-run profile (I15/G3): a cron-fired session
    /// (`session_meta.scheduled_job.is_some()`) hydrates under this orchestrator-free, `cron`-free
    /// toolset instead of the resolver/fallback profile, so a scheduled run cannot self-schedule or
    /// self-delegate. Leave unset to run cron sessions on the default profile path.
    pub fn with_cron_profile(mut self, profile: EngineProfile) -> Self {
        self.cron_profile = Some(profile);
        self
    }

    /// Inject the content store + workspace roots so this factory's incarnations capture a child's
    /// `outbox/` artifacts at completion and materialize a child's returned artifacts into a parent's
    /// `inbox/` on hydrate (daemon-content-transfer-spec.md Phase 2a, node-mediated).
    pub fn with_content(mut self, blobs: Arc<dyn BlobStore>, roots: Arc<WorkspaceRoots>) -> Self {
        self.content = Some(ContentTransfer { blobs, roots });
        self
    }

    /// Inject an authority-backed (or brokered) credential provider + profile into every engine
    /// this factory builds — the host bridge for the §7 port (host-spec §6).
    pub fn with_credentials(mut self, credentials: CredentialBuilder, profile: ProfileRef) -> Self {
        self.profile = self.profile.with_credentials(credentials, profile);
        self
    }
}

impl EngineFactory for CoreEngineFactory {
    fn create(&self) -> Box<dyn Incarnation> {
        Box::new(CoreIncarnation {
            profile: self.profile.clone(),
            engine: None,
            journal: self.journal.clone(),
            background: self.background.clone(),
            resolver: self.resolver.clone(),
            content: self.content.clone(),
            cron_profile: self.cron_profile.clone(),
            completion_payload: None,
            ctx: None,
            turn_seal: None,
        })
    }
}

/// One core-backed engine incarnation driven through the activation seam.
pub struct CoreIncarnation {
    profile: EngineProfile,
    engine: Option<Engine>,
    journal: Option<JournalConfig>,
    /// The §4.3 background-spawn materializer (when configured): drives `Effect::Spawn` requests and
    /// selects the constrained review profile when *this* incarnation is itself a background child.
    background: Option<Arc<BackgroundSpawner>>,
    /// The per-session profile resolver (when configured): re-resolves this session's `EngineProfile`
    /// from its bound profile + persisted overlay at hydrate, so a durable session honors its own
    /// profile + restored session override instead of the factory's fixed profile.
    resolver: Option<DurableProfileResolver>,
    /// Node-side content transfer (blob store + workspace roots) for capturing/materializing
    /// delegated artifacts; `None` disables it.
    content: Option<ContentTransfer>,
    /// The constrained cron-run profile (I15/G3): used in place of the resolver/fallback when this
    /// incarnation hydrates a cron-fired session, so the run carries no `cron`/`orchestrate` tools.
    cron_profile: Option<EngineProfile>,
    /// The structured completion payload captured at `Step::Completed` (a CBOR `DelegationResult`
    /// over the child's `outbox/`), surfaced via [`Incarnation::completion_payload`]. `None` => no
    /// artifacts captured (the store falls back to the legacy `child:{id}` marker).
    completion_payload: Option<Vec<u8>>,
    /// The per-activation turn context (session-unification §5), stashed at hydrate: the persisted
    /// execution policy (terminal-vs-idle at the turn boundary), the in-flight turn's journal
    /// segment, and the activation fence every durable append rides.
    ctx: Option<TurnCtx>,
    /// The committed turn's deferred journal seal (interactive-root path), captured after the
    /// post-turn journaling and handed to the manager via [`Incarnation::take_turn_seal`] for the
    /// `commit_turn` transaction.
    turn_seal: Option<daemon_store::TurnSeal>,
}

fn map_failure(failure: Failure) -> EngineError {
    EngineError::Other(failure.to_string())
}

/// The store [`ChildLifetime`](daemon_store::ChildLifetime) a delegation suspension's opaque
/// payload declares: decode the CBOR [`DelegationInput`](daemon_protocol::DelegationInput) and map
/// its protocol-level lifetime onto the store's. Legacy / undecodable payloads (including the
/// §12 approval-park marker) default to `Persistent`, the historical managed-child behaviour.
fn lifetime_from_payload(payload: &[u8]) -> daemon_store::ChildLifetime {
    match daemon_protocol::DelegationInput::decode(payload).lifetime {
        daemon_protocol::DelegationLifetime::Ephemeral => daemon_store::ChildLifetime::Ephemeral,
        daemon_protocol::DelegationLifetime::Persistent => daemon_store::ChildLifetime::Persistent,
    }
}

impl CoreIncarnation {
    /// Re-resolve `session`'s effective [`EngineProfile`] from its host-level metadata (the bound
    /// profile plus the persisted overlay) via the injected resolver. Returns `None` when no
    /// resolver/journal store is wired or no profile binding is recorded, so the caller then falls
    /// back to the factory's default profile. This is the durable half of the one resolution path
    /// shared with the live surface, so a durable session honors its own profile and restored
    /// session override.
    async fn resolve_session_profile(&self, session: &SessionId) -> Option<EngineProfile> {
        let resolver = self.resolver.as_ref()?;
        let store = self.journal.as_ref().map(|cfg| &cfg.store)?;
        let meta = store.session_meta(session).await.unwrap_or_default();
        let overlay = decode_overlay(&meta.overlay);
        resolver(session, meta.bound_profile, &meta.inline_profile, &overlay)
    }

    /// Capture this (child) session's `outbox/` into the content store as a structured
    /// [`DelegationResult`](daemon_protocol::DelegationResult): each regular file is `blob_put` and
    /// referenced by name. Returns `None` (legacy marker) when content transfer is unwired or the
    /// `outbox/` is absent/empty. Best-effort: an unreadable file or store error is skipped.
    async fn capture_outbox(&self, session: &SessionId) -> Option<Vec<u8>> {
        let content = self.content.as_ref()?;
        let outbox = content.roots.session_root(session.as_str()).join("outbox");
        let mut artifacts = Vec::new();
        for entry in std::fs::read_dir(&outbox).ok()?.flatten() {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let Ok(bytes) = std::fs::read(&path) else {
                continue;
            };
            let Ok(mut blob_ref) = content.blobs.put(&bytes).await else {
                continue;
            };
            blob_ref.name = path.file_name().map(|n| n.to_string_lossy().into_owned());
            artifacts.push(blob_ref);
        }
        if artifacts.is_empty() {
            return None;
        }
        let summary = format!("completed with {} artifact(s)", artifacts.len());
        Some(daemon_protocol::DelegationResult { summary, artifacts }.encode())
    }

    /// Materialize the artifacts a delegated child returned (decoded from its completion payload)
    /// into this (parent) session's `inbox/`, fetching each from the content store. Best-effort and a
    /// no-op when content transfer is unwired, the payload is legacy/structureless, or there are no
    /// artifacts. The basename guards against a name escaping `inbox/`.
    async fn materialize_artifacts(&self, session: &SessionId, payload: &[u8]) {
        let Some(content) = &self.content else {
            return;
        };
        let result = daemon_protocol::DelegationResult::decode(payload);
        if result.artifacts.is_empty() {
            return;
        }
        let inbox = content.roots.session_root(session.as_str()).join("inbox");
        if std::fs::create_dir_all(&inbox).is_err() {
            return;
        }
        for art in &result.artifacts {
            let Ok(bytes) = content.blobs.get(&art.hash, None).await else {
                continue;
            };
            let name = art
                .name
                .clone()
                .unwrap_or_else(|| format!("{}.bin", art.hash.to_hex()));
            let base = std::path::Path::new(&name)
                .file_name()
                .map(|n| n.to_owned())
                .unwrap_or_else(|| std::ffi::OsStr::new("artifact").to_owned());
            let _ = std::fs::write(inbox.join(base), bytes);
        }
    }
}

#[async_trait]
impl Incarnation for CoreIncarnation {
    async fn hydrate(
        &mut self,
        snapshot: SnapshotBlob,
        unapplied: Vec<JobCompletion>,
        splices: Vec<daemon_store::InboxSplice>,
        ctx: TurnCtx,
    ) -> Result<(), EngineError> {
        if snapshot.is_empty() {
            return Err(EngineError::Other(
                "core incarnation hydrated from an empty snapshot".into(),
            ));
        }
        self.ctx = Some(ctx);
        let snap = Snapshot::decode(&snapshot)?;
        let session_id = snap.session_id.clone();
        // I15: read the cron origin once — it drives both the profile selection here (a cron-fired
        // session runs under the constrained cron profile) and the `TurnTrigger::Scheduled` arming
        // below, so we avoid a second `session_meta` round-trip.
        let scheduled_job = if let Some(store) = self.journal.as_ref().map(|cfg| &cfg.store) {
            store
                .session_meta(&session_id)
                .await
                .and_then(|m| m.scheduled_job)
        } else {
            None
        };
        // A background child (§4.3) hydrates under its constrained review profile (skills-only /
        // memory-only tools + bounded budget + nudges off), not the parent's full profile. A
        // cron-fired session (I15/G3) hydrates under the constrained cron profile (no `cron`/
        // `orchestrate` tools) so it cannot self-schedule. Otherwise, when a per-session resolver +
        // journal store are wired, re-resolve this session's profile from its persisted bound profile
        // + overlay (unified resolution: a durable session honors its own model/tools/approval
        // override). Falls back to the factory's fixed profile when no binding is recorded (e.g.
        // delegated orchestrator children).
        let profile = if let Some(bg_profile) = self
            .background
            .as_ref()
            .and_then(|bg| bg.profile_for(&snap.session_id))
        {
            bg_profile
        } else if scheduled_job.is_some() {
            // I15/G3 + Phase 2 shaping: a cron-fired session resolves its bound profile overlaid with
            // the run's persisted `SessionOverlay` (model/provider/tool-allowlist/workdir) through the
            // SAME unified resolver the live/durable paths use. That resolver is G3-safe **by
            // construction** — it builds the session tool registry from fs+shell+node-extras+skills and
            // never wires the `cron`/`orchestrate` tools — so honoring the overlay cannot let a
            // scheduled run self-schedule or self-delegate. Falls back to the explicitly-constrained
            // `cron_profile` (then the factory default) when no resolver/binding is wired.
            if let Some(resolved) = self.resolve_session_profile(&snap.session_id).await {
                resolved
            } else if let Some(cron_profile) = &self.cron_profile {
                cron_profile.clone()
            } else {
                self.profile.clone()
            }
        } else if let Some(resolved) = self.resolve_session_profile(&snap.session_id).await {
            resolved
        } else {
            self.profile.clone()
        };
        let mut engine = profile.from_snapshot(snap);
        // Node-side: materialize any artifacts the completed children returned into this (parent)
        // session's `inbox/` before the engine folds the completions (the engine sees only the
        // summary text; the files land on disk). Best-effort; no-op without content transfer.
        for completion in &unapplied {
            self.materialize_artifacts(&session_id, &completion.payload)
                .await;
        }
        let completions = unapplied
            .into_iter()
            .map(|c| Completion {
                job_id: c.job_id,
                payload: c.payload,
            })
            .collect();
        engine.apply_completions(completions);
        // Durable inbox seam (session-unification §4): fold the claimed splices — a background
        // process-exit notification, a message to a delegated child (the durable `send` path), a
        // seeded first input, an F4 durable-resume turn — into the conversation before the turn
        // runs. The splices were CAS-claimed under this activation's fence inside the load
        // transaction (replacing the old destructive `take_session_inputs` drain); rows at or
        // below the snapshot's consumed cursor were captured by an earlier commit and are
        // skipped. Each payload decodes as a CBOR `UserMsg` (bare text falls back); an `Observe`
        // splice folds context-only (`push_observe` — it never opens a turn by itself). The
        // engine's cursor advances with each fold, so the commit that persists this snapshot
        // flips exactly the folded prefix to `Consumed` in the same transaction.
        let cursor = engine.consumed_splice_seq();
        for splice in splices {
            if splice.splice_seq <= cursor {
                continue;
            }
            let msg = daemon_protocol::UserMsg::decode(&splice.payload);
            match splice.kind {
                daemon_store::SpliceKind::Observe => engine.push_observe(msg),
                daemon_store::SpliceKind::StartTurn | daemon_store::SpliceKind::Steer => {
                    engine.push_user(msg)
                }
            }
            engine.note_consumed_splice(splice.splice_seq);
        }
        // I15: a cron-fired session carries `SessionMeta::scheduled_job` (read above). Arm the next
        // turn's trigger as `TurnTrigger::Scheduled { job }` so the fired turn reports its scheduled
        // origin instead of the durable wake path's default `User`. One-shot (consumed by `run_turn`).
        if let Some(job) = scheduled_job {
            engine.set_next_trigger(daemon_protocol::TurnTrigger::Scheduled { job });
        }
        self.engine = Some(engine);
        Ok(())
    }

    async fn run(&mut self) -> Result<Step, EngineError> {
        let engine = self
            .engine
            .as_mut()
            .ok_or_else(|| EngineError::Other("run before hydrate".into()))?;
        let session_id = engine.snapshot().session_id.clone();
        // Session-unification §5: the durable journal segment is the in-flight TURN, not the
        // incarnation epoch — a resumed suspension continues the same open segment. The persisted
        // execution policy decides the turn boundary: an interactive root commits back to
        // Idle/Ready (`Step::TurnCommitted`, seal deferred into the `commit_turn` transaction);
        // every other policy (and legacy `None`) stays terminal.
        let ctx = self
            .ctx
            .ok_or_else(|| EngineError::Other("run before hydrate".into()))?;
        let interactive = matches!(
            ctx.policy,
            Some(daemon_store::ExecutionPolicy::InteractiveRoot)
        );
        // When background spawn is enabled, capture a clone of the parent's live conversation so a
        // mid-turn `Effect::Spawn` can seed the review child `FromConversation` without a store read.
        let seed_conversation = self
            .background
            .as_ref()
            .map(|_| engine.snapshot().conversation.clone());
        let host = DelegateResolver {
            session_id: session_id.clone(),
            epoch: engine.epoch(),
            background: self.background.clone(),
            seed_conversation,
            approval_seq: Mutex::new(0),
        };
        // When journaling, capture the engine's events so they can be coalesced into finished blocks
        // and sealed after the turn, and so the turn's token usage can be folded into the durable
        // per-session usage surface (the tree projection's usage source); otherwise discard.
        let captured: Arc<Mutex<Vec<daemon_protocol::AgentEvent>>> =
            Arc::new(Mutex::new(Vec::new()));
        let sink = if self.journal.is_some() {
            let cap = captured.clone();
            EventSink::new(move |ev| cap.lock().unwrap().push(ev))
        } else {
            EventSink::discarding()
        };
        let control = TurnControl::new();
        let outcome = engine
            .run_turn(&host, &sink, &control)
            .await
            .map_err(map_failure)?;

        // Fold this turn's token usage into the durable per-session usage surface so the management
        // tree projects real, recovery-survivable usage at every node (replacing the in-memory fleet
        // fan-in for durable sessions).
        if let Some(cfg) = &self.journal {
            let mut delta = daemon_common::UsageDelta::default();
            for ev in captured.lock().unwrap().iter() {
                if let daemon_protocol::AgentEvent::Usage { delta: d, .. } = ev {
                    delta.add(d);
                }
            }
            if delta != daemon_common::UsageDelta::default() {
                cfg.store.record_usage(&session_id, delta).await;
            }
        }

        // Journal this turn into the unified verifiable journal, keyed by the TURN's segment and
        // fenced by the activation lease on every append (session-unification §5: a stale
        // incarnation can neither append into nor seal the winning segment). On the interactive
        // path the seal is DEFERRED: the coalescer's turn-boundary `Seal` computes + signs the
        // root, and the resulting `TurnSeal` rides the `commit_turn` transaction so the root
        // lands atomically with the snapshot it covers. Terminal/suspension paths seal directly
        // (a suspension emits no turn-boundary event, leaving its segment open for the resume).
        if let Some(cfg) = &self.journal {
            let stream = JournalStreamId::session(&session_id);
            let jsink = Arc::new(
                JournalSink::for_turn(
                    cfg.store.clone(),
                    cfg.signer.clone(),
                    stream,
                    ctx.fence,
                    ctx.turn_seq,
                    interactive,
                )
                .await,
            );
            let feeder = JournalFeeder::new(jsink.clone());
            let events = std::mem::take(&mut *captured.lock().unwrap());
            for ev in events {
                feeder.feed(&Outbound::Event(ev)).await;
            }
            if interactive {
                self.turn_seal = jsink.take_pending_seal();
            }
        }

        // Re-index this session's searchable text at the turn boundary (the durable half of the
        // `session_search` FTS surface; the live pump indexes interactive sessions): the coalesced
        // full conversation (user + assistant text + tool names) replaces the prior row, so search
        // reflects the whole conversation, not just the opening turn. Best-effort by construction
        // (`index_session_text` swallows store errors).
        if let Some(cfg) = &self.journal {
            let title = cfg
                .store
                .session_meta(&session_id)
                .await
                .and_then(|m| m.title);
            let turns =
                crate::session_index::turns_from_conversation(&engine.snapshot().conversation);
            let body = crate::session_index::coalesce_body(&turns);
            if !body.trim().is_empty() {
                cfg.store
                    .index_session_text(&session_id, title, &body)
                    .await;
            }
        }

        match outcome {
            TurnOutcome::Completed(_) if interactive => {
                // The interactive-root turn boundary (session-unification §3/§5): the session is
                // NOT terminal — the turn commits back to `Idle`/`Ready` through the fenced
                // `commit_turn`, and a failed turn stays retryable (retry = a new user action; the
                // failure is already journaled by the coalescer's error record). No LCM finalize,
                // no outbox capture: both are terminal-only.
                Ok(Step::TurnCommitted)
            }
            TurnOutcome::Completed(_) => {
                // Terminal deactivation (§10/§11): `Step::Completed` marks the session `Completed`
                // in the store (never re-activated), so flush the context engine + memory providers
                // (LCM final ingest + lifecycle finalize) before the final checkpoint is taken.
                engine.end_session().await;
                // Terminal: capture this child's `outbox/` artifacts into the content store as the
                // structured completion payload (the parent materializes them on its wake). `None`
                // when content transfer is unwired or no artifacts were produced (legacy marker).
                self.completion_payload = self.capture_outbox(&session_id).await;
                Ok(Step::Completed)
            }
            // §12 HITL: an approval park records its parked rows for the operator surface and enqueues
            // no runnable job (the activation layer routes it to `park_approval`). The snapshot keeps
            // the typed `PendingApproval`s (with the deferred `ToolCall`); these are the store rows.
            TurnOutcome::Suspended(suspension)
                if suspension.payload == daemon_core::APPROVAL_SUSPEND_PAYLOAD =>
            {
                let approvals = engine
                    .snapshot()
                    .pending_approvals
                    .iter()
                    .map(|p| ParkedApproval {
                        session_id: session_id.clone(),
                        job_id: p.job_id.clone(),
                        epoch: suspension.epoch,
                        prompt: p.prompt.clone(),
                        path: p.path.clone(),
                        // wire v28: carry the resolved command fingerprint onto the durable row so
                        // the operator surface can display it structurally (hex digest; the
                        // approve-then-swap enforcement stays on the engine's typed `PendingApproval`).
                        fingerprint: p.fingerprint.as_ref().map(|f| f.as_str().to_string()),
                        decision: None,
                    })
                    .collect();
                Ok(Step::ParkApproval { approvals })
            }
            TurnOutcome::Suspended(suspension) => {
                // The delegating parent's declared child lifetime rides inside the opaque payload
                // (a CBOR `DelegationInput`); surface it onto the durable job so the fleet worker
                // derives the child's roster/tree role (managed vs ephemeral subagent).
                let lifetime = lifetime_from_payload(&suspension.payload);
                Ok(Step::Suspended {
                    job: JobCommand {
                        job_id: suspension.job_id,
                        session_id,
                        epoch: suspension.epoch,
                        payload: suspension.payload,
                        lifetime,
                        child: None,
                    },
                })
            }
        }
    }

    fn checkpoint(&self) -> Result<SnapshotBlob, EngineError> {
        let engine = self
            .engine
            .as_ref()
            .ok_or_else(|| EngineError::Other("checkpoint before hydrate".into()))?;
        Ok(engine.snapshot().encode()?)
    }

    fn epoch(&self) -> Epoch {
        self.engine.as_ref().map(|e| e.epoch()).unwrap_or_default()
    }

    fn completion_payload(&self) -> Option<Vec<u8>> {
        self.completion_payload.clone()
    }

    fn consumed_splices(&self) -> Option<u64> {
        // The engine's snapshot cursor (advanced at hydrate as claimed splices folded): the store
        // flips exactly that prefix `Consumed` in the same transaction that persists the snapshot.
        self.engine
            .as_ref()
            .map(|e| e.consumed_splice_seq())
            .filter(|seq| *seq > 0)
    }

    fn take_turn_seal(&mut self) -> Option<daemon_store::TurnSeal> {
        self.turn_seal.take()
    }
}

/// The substrate-path host handler: resolves a delegation to the deterministic durable `JobId` the
/// activation outbox dedupes on, materializes an attached non-joining background child for a
/// `Spawn` (§4.3, fire-and-forget — never suspends the parent), and trivially answers the other §17
/// request kinds.
struct DelegateResolver {
    session_id: SessionId,
    epoch: Epoch,
    /// The §4.3 background-spawn materializer, when configured.
    background: Option<Arc<BackgroundSpawner>>,
    /// The parent's live conversation snapshot, captured before the turn so a `Spawn` seeds the
    /// review child `FromConversation` without a store round-trip (only `Some` when spawn is on).
    seed_conversation: Option<Conversation>,
    /// A per-run counter minting a deterministic `JobId` for each §12 edit-approval ask in turn
    /// order, so a gated tool on the durable path defers to a parked operator decision. Deterministic
    /// per `(session, post-bump epoch, ordinal)` so a recovery re-park reuses the same id (dedupe).
    approval_seq: Mutex<u32>,
}

#[async_trait]
impl HostRequestHandler for DelegateResolver {
    async fn request(&self, req: HostRequest) -> HostResponse {
        let body = match req.kind {
            HostRequestKind::Delegate { .. } => {
                // Deterministic per (session, post-bump epoch) so a recovery re-enqueue dedupes.
                let job_id = JobId::new(format!("{}:{}:job", self.session_id, self.epoch.next().0));
                HostResponseBody::Delegated(job_id)
            }
            HostRequestKind::Spawn { spec } => {
                // Fire-and-forget: materialize the attached non-joining child now and return its id;
                // the parent neither suspends nor waits. Unknown kind / no spawner -> no-op.
                let child = match &self.background {
                    Some(bg) => bg
                        .spawn(
                            &self.session_id,
                            self.epoch,
                            &spec,
                            self.seed_conversation.clone(),
                        )
                        .await
                        .unwrap_or_else(|| self.session_id.clone()),
                    None => self.session_id.clone(),
                };
                HostResponseBody::Spawned(child)
            }
            HostRequestKind::Approval { .. } => {
                // §12 durable HITL: a gated tool on the durable path asks only when its policy is
                // `Ask` (the engine already auto-allowed/denied otherwise). There is no synchronous
                // operator on this headless path, so defer: mint the deterministic parked `JobId` the
                // engine records + suspends on, to be answered later by `ApprovalDecide`.
                let mut seq = self.approval_seq.lock().unwrap();
                let job_id = JobId::new(format!(
                    "{}:{}:approval:{}",
                    self.session_id,
                    self.epoch.next().0,
                    *seq
                ));
                *seq += 1;
                HostResponseBody::Deferred(job_id)
            }
            HostRequestKind::Input { .. } => HostResponseBody::Input(String::new()),
            HostRequestKind::Choice { .. } => HostResponseBody::Chosen(0),
            _ => HostResponseBody::Approved {
                approved: true,
                allow_permanent: false,
                reason: None,
            },
        };
        HostResponse {
            request_id: req.request_id,
            body,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blob_store::FileBlobStore;
    use daemon_core::{MockProvider, Provider, SystemPrompt, ToolRegistry};

    fn unique_dir(tag: &str) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("daemon-inc-{tag}-{}-{n}", std::process::id()))
    }

    fn incarnation_with_content(
        blobs: Arc<dyn BlobStore>,
        roots: Arc<WorkspaceRoots>,
    ) -> CoreIncarnation {
        let profile = EngineProfile::new(
            Arc::new(|| Arc::new(MockProvider::completing("done")) as Arc<dyn Provider>),
            Arc::new(ToolRegistry::new()),
            SystemPrompt::new("test"),
        );
        CoreIncarnation {
            profile,
            engine: None,
            journal: None,
            background: None,
            resolver: None,
            content: Some(ContentTransfer { blobs, roots }),
            cron_profile: None,
            completion_payload: None,
            ctx: None,
            turn_seal: None,
        }
    }

    /// A child's `outbox/` is captured into the content store as a structured `DelegationResult`,
    /// and that result materializes back into a (parent) session's `inbox/` — the node-mediated
    /// artifact round-trip (daemon-content-transfer-spec.md Phase 2a, completion-up).
    #[tokio::test]
    async fn outbox_capture_round_trips_into_parent_inbox() {
        let ws = unique_dir("ws");
        let cas = unique_dir("cas");
        let roots = Arc::new(WorkspaceRoots::new(ws.clone()));
        let blobs: Arc<dyn BlobStore> =
            Arc::new(FileBlobStore::open(cas.clone()).expect("open blob store"));
        let inc = incarnation_with_content(blobs.clone(), roots.clone());

        let child = SessionId::new("parent/c1");
        let parent = SessionId::new("parent");

        // The child writes an artifact into its outbox/.
        let outbox = roots.session_root(child.as_str()).join("outbox");
        std::fs::create_dir_all(&outbox).unwrap();
        std::fs::write(outbox.join("report.txt"), b"final report").unwrap();

        // Capture: the outbox is folded into a DelegationResult referencing the stored blob.
        let payload = inc
            .capture_outbox(&child)
            .await
            .expect("a non-empty outbox yields a structured payload");
        let result = daemon_protocol::DelegationResult::decode(&payload);
        assert_eq!(result.artifacts.len(), 1);
        assert_eq!(result.artifacts[0].name.as_deref(), Some("report.txt"));

        // Materialize: the parent's inbox/ receives the artifact bytes fetched from the store.
        inc.materialize_artifacts(&parent, &payload).await;
        let landed = roots.session_root(parent.as_str()).join("inbox/report.txt");
        assert_eq!(std::fs::read(&landed).unwrap(), b"final report");

        // An empty outbox captures nothing (the store falls back to the legacy marker).
        let empty_child = SessionId::new("parent/c2");
        assert!(inc.capture_outbox(&empty_child).await.is_none());

        let _ = std::fs::remove_dir_all(&ws);
        let _ = std::fs::remove_dir_all(&cas);
    }

    /// The durable inbox seam (session-unification §4): claimed splices are folded at hydrate —
    /// decoded as `UserMsg`s and appended to the conversation in sequence order, before the turn
    /// runs — the checkpointed snapshot records the consumed cursor, and a re-hydrate FROM that
    /// checkpoint skips the already-folded prefix (nothing folds twice). A non-CBOR payload folds
    /// as bare text rather than failing the activation (`UserMsg::decode` fallback), and an
    /// `Observe` splice folds context-only.
    #[tokio::test]
    async fn hydrate_folds_claimed_splices_into_conversation() {
        use daemon_store::{InMemoryStore, NewSplice, SessionStore, SpliceKind};

        let store: Arc<dyn SessionStore> = Arc::new(InMemoryStore::new());
        let session = SessionId::new("notify-target");
        let snapshot = daemon_core::Snapshot::fresh(session.clone());
        store
            .create_session(session.clone(), daemon_common::PartitionId::DEFAULT, {
                snapshot.encode().unwrap()
            })
            .await
            .unwrap();

        // Two well-formed inputs (FIFO) plus one bare-text payload (folds via the decode fallback).
        let encode = |text: &str| {
            let mut buf = Vec::new();
            ciborium::into_writer(&daemon_protocol::UserMsg::new(text), &mut buf).unwrap();
            buf
        };
        let splice = |kind: SpliceKind, payload: Vec<u8>, op: &str| NewSplice {
            session_id: session.clone(),
            kind,
            payload,
            origin_op: op.into(),
            origin: "test".into(),
        };
        store
            .append_splice(splice(
                SpliceKind::StartTurn,
                encode("[proc done] first"),
                "n-1",
            ))
            .await
            .unwrap();
        store
            .append_splice(splice(SpliceKind::Steer, b"not-cbor".to_vec(), "n-2"))
            .await
            .unwrap();
        store
            .append_splice(splice(
                SpliceKind::Observe,
                encode("[proc done] second"),
                "n-3",
            ))
            .await
            .unwrap();

        let profile = EngineProfile::new(
            Arc::new(|| Arc::new(MockProvider::completing("ok")) as Arc<dyn Provider>),
            Arc::new(ToolRegistry::new()),
            SystemPrompt::new("test"),
        );
        let factory = CoreEngineFactory::from_profile(profile)
            .with_journal(store.clone(), Arc::new(TraceSigner::generate()));
        let mut inc = factory.create();
        let fence = store.acquire_activation_lease(&session).await.unwrap();
        let activation = store.load_for_activation(&session, fence).await.unwrap();
        assert_eq!(activation.splices.len(), 3, "the load claims the inbox");
        inc.hydrate(
            activation.snapshot,
            activation.unapplied,
            activation.splices,
            TurnCtx {
                policy: activation.policy,
                turn_seq: activation.turn_seq,
                fence,
            },
        )
        .await
        .unwrap();

        // The checkpointed conversation carries all three inputs, in sequence order (the bare-text
        // payload folds through the decode fallback rather than being dropped; the Observe folds
        // context-only into the same conversation), and the consumed cursor is stamped.
        let checkpoint_blob = inc.checkpoint().unwrap();
        let snap = Snapshot::decode(&checkpoint_blob).unwrap();
        let users: Vec<String> = snap
            .conversation
            .turns
            .iter()
            .filter_map(|t| match t {
                daemon_core::Turn::User(msg) => Some(msg.text.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(
            users,
            vec![
                "[proc done] first".to_string(),
                "not-cbor".to_string(),
                "[proc done] second".to_string()
            ]
        );
        assert_eq!(
            snap.consumed_splice_seq, 3,
            "the snapshot records the folded high-water mark"
        );
        assert_eq!(
            inc.consumed_splices(),
            Some(3),
            "the incarnation stamps the cursor onto its commit"
        );

        // A re-hydrate FROM the checkpoint with the same claimed set (a crash between fold and
        // commit re-claims under a newer fence) skips the already-captured prefix.
        let fence2 = store.acquire_activation_lease(&session).await.unwrap();
        let again = store.load_for_activation(&session, fence2).await.unwrap();
        let mut inc2 = factory.create();
        inc2.hydrate(
            checkpoint_blob,
            again.unapplied,
            again.splices,
            TurnCtx {
                policy: again.policy,
                turn_seq: again.turn_seq,
                fence: fence2,
            },
        )
        .await
        .unwrap();
        let snap2 = Snapshot::decode(&inc2.checkpoint().unwrap()).unwrap();
        assert_eq!(
            snap2.conversation.turns.len(),
            snap.conversation.turns.len(),
            "splices at/below the snapshot cursor never re-fold"
        );
    }

    /// The suspension payload's declared lifetime is surfaced onto the durable `JobCommand`:
    /// an ephemeral `DelegationInput` maps to `ChildLifetime::Ephemeral`; a persistent/legacy
    /// payload keeps the historical managed-child default.
    #[test]
    fn lifetime_threads_from_delegation_payload() {
        let ephemeral = daemon_protocol::DelegationInput {
            task: "quick check".into(),
            attachments: Vec::new(),
            lifetime: daemon_protocol::DelegationLifetime::Ephemeral,
            source: daemon_protocol::ChildSource::Default,
            detached: false,
        }
        .encode();
        assert_eq!(
            lifetime_from_payload(&ephemeral),
            daemon_store::ChildLifetime::Ephemeral
        );

        let persistent = daemon_protocol::DelegationInput::task("long-lived work").encode();
        assert_eq!(
            lifetime_from_payload(&persistent),
            daemon_store::ChildLifetime::Persistent
        );
        // Legacy plain-text payloads (pre-upgrade jobs) stay managed children.
        assert_eq!(
            lifetime_from_payload(b"delegated-work"),
            daemon_store::ChildLifetime::Persistent
        );
    }

    /// Splices queued while the session was dehydrated (the durable `send` seam) fold exactly
    /// once: the commit that persists the folded snapshot consumes them in the same transaction,
    /// so the NEXT activation's load claims nothing and the conversation gains no duplicates.
    #[tokio::test]
    async fn commit_consumes_folded_splices_exactly_once() {
        use daemon_store::{Checkpoint, NewSplice, SpliceKind};

        let store: Arc<dyn SessionStore> = Arc::new(daemon_store::InMemoryStore::new());
        let session = SessionId::new("dormant");
        store
            .create_session(
                session.clone(),
                daemon_common::PartitionId::DEFAULT,
                daemon_core::Snapshot::fresh(session.clone())
                    .encode()
                    .unwrap(),
            )
            .await
            .unwrap();
        for (text, op) in [("first ping", "s-1"), ("bare follow-up", "s-2")] {
            store
                .append_splice(NewSplice {
                    session_id: session.clone(),
                    kind: SpliceKind::Steer,
                    payload: daemon_protocol::UserMsg::new(text).encode(),
                    origin_op: op.into(),
                    origin: "test".into(),
                })
                .await
                .unwrap();
        }

        let profile = EngineProfile::new(
            Arc::new(|| Arc::new(MockProvider::completing("done")) as Arc<dyn Provider>),
            Arc::new(ToolRegistry::new()),
            SystemPrompt::new("test"),
        );
        let factory = CoreEngineFactory::from_profile(profile)
            .with_journal(store.clone(), Arc::new(TraceSigner::generate()));
        let mut inc = factory.create();
        let fence = store.acquire_activation_lease(&session).await.unwrap();
        let activation = store.load_for_activation(&session, fence).await.unwrap();
        inc.hydrate(
            activation.snapshot,
            activation.unapplied,
            activation.splices,
            TurnCtx {
                policy: activation.policy,
                turn_seq: activation.turn_seq,
                fence,
            },
        )
        .await
        .unwrap();
        let folded = Snapshot::decode(&inc.checkpoint().unwrap()).unwrap();
        assert_eq!(folded.consumed_splice_seq, 2);

        // Commit the folded snapshot with the cursor: consumption rides the same transaction.
        store
            .mark_completed(
                Checkpoint::new(session.clone(), inc.epoch(), inc.checkpoint().unwrap())
                    .with_consumed_splices(inc.consumed_splices()),
                fence,
            )
            .await
            .unwrap();

        // The next activation claims nothing — the folded splices are consumed, not re-deliverable.
        let fence2 = store.acquire_activation_lease(&session).await.unwrap();
        let again = store.load_for_activation(&session, fence2).await.unwrap();
        assert!(
            again.splices.is_empty(),
            "consumed splices never re-claim: the fold happened exactly once"
        );
    }
}
