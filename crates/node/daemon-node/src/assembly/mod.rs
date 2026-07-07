// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The phase-wired composition root: [`assemble`] builds and starts the default host node — durable
//! substrate + resident services, the orchestration fleet as the real job worker, the credential
//! seam, and the live session surface — all from one shared [`EngineProfile`] per role.
//!
//! `assemble` is a short, ordered spine over the private `build_*` / `bind_*` phase helpers in this
//! module. The composition contract is the **order** (cron stack before the orchestrator profile;
//! `host.start()` before `NodeApiImpl`; the late `cron_worker.set_delivery(node)` last) and the
//! **shared single-instance `Arc`s** ([`Shared`] threads `fleet_events`, `node_events`, `signer`,
//! `background`, `workspace_roots`, `blob_store` so every phase captures the SAME instance). The
//! helpers move phase bodies verbatim; they never re-spec behavior.

use std::path::PathBuf;
use std::sync::Arc;

use daemon_activation::EngineFactory;
use daemon_api::{from_cbor, EngineSelector, ProfileSpec, SessionOverlay, TreeEvent};
use daemon_common::{ProfileRef, SessionId};
use daemon_core::{ApprovalPolicy, Config, EngineProfile, StablePromptSource, SystemPrompt, Tool};
use daemon_host::{
    BackgroundSpawner, BlobStore, BlueprintSource, CoreEngineFactory, CronFiring, CronOps,
    CronScheduler, DurableProfileResolver, FileBlobStore, FleetControl, Host, JournalConfig,
    ModelProviderFactory, NodeApiImpl, NodeApiParts, NodeEventFeed, ProfileOps, ProfileStore,
    RoutingBuilder, RoutingRegistry, SessionBackend, SessionEngineBuilder, WorkspaceFs,
    WorkspaceRoots,
};
use daemon_orchestration::{ChildSpawner, DefaultAnswerPolicy, FleetRuntime};
use daemon_supervision::ManageRequestHandler;
use daemon_telemetry::TraceSigner;

use crate::cron::worker::{CronSkillLoader, CronWorker};
use crate::fleet::foreign_incarnation::{DispatchingEngineFactory, ForeignConfig};
use crate::fleet::job_worker::FleetJobWorker;
use crate::fleet::spawner::ProfileChildSpawner;
use crate::fleet::view::FleetViewImpl;
use crate::profiles::dress::{
    core_tool_registry_with_skills, dress, provider_for, root_profile, ProcessToolkit,
    CHILD_PROFILE, ORCHESTRATOR_PROFILE,
};
use crate::profiles::registry::background_registry;
use crate::profiles::resolve::SessionFactoryCtx;
use crate::types::{AssembledNode, NodeAssembly};

/// The node-wide single-instance handles every phase shares. Threading one [`Shared`] keeps the
/// capture lists identical across the phase helpers: the SAME `fleet_events` broadcast sender flows
/// into [`FleetRuntime`], [`FleetJobWorker`], and the tree-push surface; the SAME `node_events` feed
/// backs both the fleet-change bridge and the model download-progress hook; etc.
struct Shared {
    /// The node's workspace-root resolver, shared by every engine exec builder + the `fs_*` surface.
    workspace_roots: Option<Arc<WorkspaceRoots>>,
    /// The content store (blob CAS), shared by the durable worker + the `blob_*`/`fs_*` surface.
    blob_store: Option<Arc<dyn BlobStore>>,
    /// The host-owned fleet event bus (I4/I8): the single broadcast sender the orchestration
    /// producers ping and `tree_subscribe` subscribes to.
    fleet_events: tokio::sync::broadcast::Sender<TreeEvent>,
    /// The node-wide event feed (L3 `EventsSince`).
    node_events: Arc<NodeEventFeed>,
    /// The node's one verifiable-journal signer (durable + live + fleet-child seal with this key).
    signer: Arc<TraceSigner>,
    /// The §4.3 background-review spawner, shared by the durable factory + the live surface.
    background: Arc<BackgroundSpawner>,
}

/// The resident cron scheduler (I15) + its shared ops surface and agent veneer, built BEFORE the
/// agent profiles so the agent `cron` tool wraps the same `CronOps` the operator `cron_*` ops use.
struct CronStack {
    /// The constrained cron-run profile (the `child_profile` shape — no `orchestrate`/`cron`).
    cron_run_profile: EngineProfile,
    /// The resident scheduler worker (5th supervised service + manual-fire seam + catch-up tick).
    cron_worker: Arc<CronWorker>,
    /// The shared cron operations surface backing both the operator ops and the agent tool.
    cron_ops: Arc<CronOps>,
    /// The agent veneer over [`CronStack::cron_ops`], registered into the agent-facing profiles.
    cron_tool: Arc<dyn Tool>,
}

/// The shared profile-authoring surface (Phase 2) + its agent veneer, built BEFORE the orchestrator
/// profile so the agent `profile_manage` tool wraps the same [`ProfileOps`] the operator
/// `profile_create` op uses (one validation + persistence + revision path, not two). Present only
/// when the node hosts profile management (`NodeAssembly.profiles`).
struct ProfileStack {
    /// The shared profile ops surface backing both the operator ops and the agent tool. Its
    /// validator is late-bound to the assembled node (`ProfileValidator = NodeApiImpl`).
    profile_ops: Arc<ProfileOps>,
    /// The agent veneer over [`ProfileStack::profile_ops`], registered onto the orchestrator profile.
    profile_tool: Arc<dyn Tool>,
}

/// Assemble and start the default host node: durable substrate + resident services, the
/// orchestration fleet as the real job worker, the credential seam, and the live session surface,
/// all built from one shared [`EngineProfile`] per role so the durable, live, and fleet-child paths
/// share provider/credential/tunable policy.
pub fn assemble(a: NodeAssembly) -> AssembledNode {
    // The node's one verifiable-journal signer: every engine path (durable, live, fleet child) seals
    // its per-stream chain with this key, and the control surface publishes the verifying half.
    let signer = Arc::new(
        a.journal_seed
            .map(|seed| TraceSigner::from_seed(&seed))
            .unwrap_or_else(TraceSigner::generate),
    );
    let journal = JournalConfig {
        store: a.store.clone(),
        signer: signer.clone(),
    };

    // Resolve the launch agent's own skills once (store + tools + index), keyed on the node profile.
    // The role engines (fleet child, orchestrator, fixed session) and the background skill_review
    // child all run as the launch agent, so they share this resolution; per-session interactive /
    // durable engines re-resolve per their own `spec.id` through the `SessionFactoryCtx`.
    let launch_skills = a.skills_resolver.as_ref().map(|r| r(&a.profile));
    let launch_index = launch_skills.as_ref().map(|s| &s.index);
    let launch_skill_tools: Vec<Arc<dyn Tool>> = launch_skills
        .as_ref()
        .map(|s| s.tools.clone())
        .unwrap_or_default();

    // Autonomous durable engines (the orchestrator, every delegated child, the fleet job worker)
    // run headless with no operator to answer an edit-approval ask, so they must never gate on a
    // human (an `Ask` would suspend the turn forever). Force `AutoAllow` for these roles; the
    // *interactive* session path keeps the operator-selectable base policy (default `Ask`).
    let autonomous_config = Config {
        approval_policy: ApprovalPolicy::AutoAllow,
        ..a.engine_config
    };
    let workspace_roots = build_workspace_roots(&a);
    let blob_store = build_blob_store(&a);

    // The resident background-process registry: host-owned (never engine-owned — a background
    // process outlives the turn that spawned it), constructed before the role profiles so every
    // tool registry captures the SAME instance. Its notifier is late-bound onto the NodeApi below,
    // mirroring the cron delivery handle.
    let procs = ProcessToolkit {
        registry: Arc::new(daemon_processes::ProcessRegistry::new(
            a.processes.registry,
            Arc::new(daemon_processes::RealClock::new()),
        )),
        shell: a.processes.shell,
    };

    // The fleet child: one shared profile, driven as the real job worker so every child gets the same
    // provider + brokered credentials. Each child journals into the shared store keyed by its UnitId.
    let child_profile = build_child_profile(
        &a,
        &launch_skill_tools,
        launch_index,
        autonomous_config,
        &workspace_roots,
        &procs,
    );
    // The legacy synchronous placement seam (in-process live engine children + foreign agents). The
    // durable Core delegation path no longer uses this — it materializes children as durable
    // sessions through the shared activation manager (see `FleetJobWorker`) — so this spawner is
    // retained only for the foreign/ephemeral coarse lifecycle and the live management escalation.
    let spawner: Arc<dyn ChildSpawner> = Arc::new(
        ProfileChildSpawner::core(child_profile.clone())
            .with_journal(journal.clone())
            .with_rewind(a.store.clone(), a.checkpoints.clone()),
    );
    let (fleet_events, _) = tokio::sync::broadcast::channel::<TreeEvent>(256);
    let node_events = NodeEventFeed::new(1024);
    let fleet = FleetRuntime::new(
        a.store.clone(),
        a.partition,
        spawner,
        Arc::new(DefaultAnswerPolicy),
        None::<Arc<dyn ManageRequestHandler>>,
    )
    .with_event_sink(fleet_events.clone());

    let cron = build_cron_stack(&a, &child_profile, &workspace_roots);
    // The shared profile-authoring surface + `profile_manage` tool (Phase 2), built before the
    // orchestrator profile so the tool is registered onto it. `None` on a node without profile mgmt.
    let profile_stack = build_profile_stack(&a, &node_events);

    // The one orchestrator-capable engine shape, used at *every* durable level: the top session and
    // every delegated child are built from this profile, so a child is itself an orchestrator that
    // can delegate (the recursive durable graph).
    let orchestrator_profile = build_orchestrator_profile(
        &a,
        &fleet,
        &launch_skill_tools,
        launch_index,
        &cron.cron_tool,
        profile_stack.as_ref().map(|s| &s.profile_tool),
        autonomous_config,
        &workspace_roots,
        &procs,
    );
    // The §4.3 background-review spawner: shared by the durable factory (so a review child raised
    // mid-turn resolves its constrained profile during hydrate) and the live surface (so a `Spawn`
    // host request from an interactive session is materialized fire-and-forget). Inert when the
    // registry is empty (no skills/memory tools) — `Effect::Spawn` then no-ops.
    let background = Arc::new(BackgroundSpawner::new(
        a.store.clone(),
        a.partition,
        background_registry(&a, &launch_skill_tools),
    ));

    // From here on every phase shares one set of node-wide singletons (the capture-list contract).
    let shared = Shared {
        workspace_roots,
        blob_store,
        fleet_events,
        node_events,
        signer,
        background,
    };

    // The ephemeral-subagent reaper (host background sweep): archives `EphemeralSubagent` sessions
    // a grace period after their terminal state, so transient-child churn ages out of the default
    // roster/tree scopes on its own. Detached like the fleet-change bridge; inert when disabled.
    if a.reaper.enabled {
        crate::fleet::EphemeralReaper::new(a.store.clone(), a.reaper.grace)
            .with_events(shared.node_events.clone())
            .spawn(a.reaper.interval);
    }

    // The one per-session resolution context, shared by the live session builder and the durable
    // rehydration resolver so both paths resolve a session's engine identically (bound profile +
    // overlay). Present only when the node carries a profile store + provider resolver; otherwise
    // sessions fall back to the single fixed `session_profile` (legacy single-profile behavior).
    let session_ctx = build_session_ctx(&a, &shared, &cron.cron_tool, &procs);

    // The durable path journals too, and (when per-session resolution is available) re-resolves a
    // durable session's engine from its recorded bound profile + overlay on rehydration.
    let factory = build_factory(
        &a,
        &orchestrator_profile,
        &cron.cron_run_profile,
        &session_ctx,
        &shared,
    );
    // One durable job worker for the whole node: every delegation (top or nested) materializes a
    // parent-bound durable child session seeded from the same orchestrator profile (moved in here).
    let job_worker = build_job_worker(&a, orchestrator_profile, &session_ctx, &shared);
    // The resident cron scheduler (`cron.cron_worker`, built above) drives the 5th supervised service.
    let host = Host::new(a.store.clone(), factory, a.host_config)
        .with_job_worker(Arc::new(job_worker))
        .with_cron_scheduler(cron.cron_worker.clone() as Arc<dyn CronScheduler>);
    let handle = host.start();

    // The interactive (session sub-surface) engines + the profile-aware session builder.
    let session_profile = build_session_profile(
        &a,
        &launch_skill_tools,
        launch_index,
        &cron.cron_tool,
        &shared.workspace_roots,
        &procs,
    );
    let session_builder = build_session_builder(
        &session_ctx,
        session_profile,
        a.store.clone(),
        a.foreign_gateway.clone(),
    );

    let mut node_api = NodeApiImpl::new(NodeApiParts {
        supervisor: handle.observer(),
        store: a.store.clone(),
        manager: host.manager().clone(),
        partition: a.partition,
        engine_builder: session_builder,
        fleet: Some({
            let mut view = FleetViewImpl::new(a.store.clone(), fleet.clone());
            // Wire v29 tree enrichment: a bound unit's node denormalizes its profile's engine
            // selector, so the projection needs the profile store when the node manages profiles.
            if let Some(profiles) = a.profiles.clone() {
                view = view.with_profiles(profiles);
            }
            Arc::new(view) as Arc<dyn FleetControl>
        }),
    })
    // Live interactive sessions journal per turn; also records the signer so history reads verify.
    .with_journal(a.store.clone(), shared.signer.clone())
    // Surface the resident telemetry aggregator through the `telemetry` control op.
    .with_metrics(host.metrics().clone())
    // Subscribe the tree-push surface to the host fleet bus (I4/I8): `tree_subscribe` now forwards
    // live spawn/terminal/progress deltas instead of re-projecting on a fixed poll interval.
    .with_fleet_events(shared.fleet_events.clone())
    // The node-wide event feed (L3): `events_since` serves from this ring and the §5 emit hooks
    // push onto it.
    .with_node_events(shared.node_events.clone())
    // The read-only guardrail caps (`Caps`, wire v29): the EFFECTIVE orchestrate ceilings — the
    // same composition the tool registration below enforces.
    .with_caps(daemon_api::CapsReport {
        orchestrate_max_depth: a.orchestrate.max_depth.min(a.nesting_depth + 1) as u32,
        orchestrate_max_fanout: a.orchestrate.max_fanout as u32,
        max_composed_profiles: a.orchestrate.max_composed_profiles as u32,
        max_ephemeral_per_session: a.orchestrate.max_ephemeral_per_session as u32,
    });
    // Background session-title generation (hermes title_generator parity), when the binary resolved
    // an auxiliary provider for it.
    if let Some(aux) = a.title_aux.clone() {
        node_api = node_api.with_title_aux(aux);
    }
    // Bind every optional sub-surface (workspace/blob/cron/models/profiles/credentials/auth/skills/
    // routing/cloud/acp/model-factory/background/checkpoints) + the fleet-change bridge. Order
    // preserved from the original inline block.
    node_api = bind_node_api_surfaces(node_api, &a, &cron.cron_ops, &shared);
    // Bind the shared profile-authoring surface (Phase 2): the SAME `ProfileOps` the agent
    // `profile_manage` tool wraps, so the operator create/update ops and the tool author through one
    // validation + persistence + revision path.
    if let Some(stack) = &profile_stack {
        node_api = node_api.with_profile_ops(stack.profile_ops.clone());
    }

    let node = Arc::new(node_api);
    // Late-bind the profile-ops validator now that the node exists: the node IS the engine/inference
    // validator (`validate_engine` + `validate_inference`), so the operator ops and the agent tool
    // share the exact same validation. (No profile_create is reachable before serving starts.)
    if let Some(stack) = &profile_stack {
        stack
            .profile_ops
            .set_validator(node.clone() as Arc<dyn daemon_host::ProfileValidator>);
    }
    // Late-bind the cron post-settle delivery handle now that the `NodeApiImpl` exists: it implements
    // `CronDelivery` over its `DeliverySink` registry, so a finished cron run's `deliver` pushes
    // through the same outbound path live replies use.
    cron.cron_worker
        .set_delivery(node.clone() as Arc<dyn daemon_host::CronDelivery>);
    // Late-bind the process exit/watch notifier the same way: notifications inject into the owning
    // session through `inject_session_input` — a live `StartTurn` for actor sessions, the durable
    // pending-input + wake for activation-lifecycle sessions (drained at hydrate).
    procs
        .registry
        .set_notifier(Arc::new(NodeProcessNotifier { node: node.clone() }));

    // The detached-delegation notice worker (W9): drains the durable completion-notice outbox for
    // `spawn wait:false` children and injects each terminal outcome into its parent through the same
    // one lifecycle-aware `inject_session_input` seam the process notifier uses. Detached like the
    // reaper/fleet-change bridge; ticks at the wake/job dispatch cadence (no new config field — W10
    // executes in parallel and must not conflict on `NodeAssembly`).
    crate::fleet::NoticeWorker::new(a.store.clone(), node.clone())
        .spawn(a.host_config.dispatch_interval);

    AssembledNode {
        node,
        handle,
        fleet,
        signer: shared.signer,
        processes: procs.registry,
    }
}

/// The process-notification adapter: routes a formatted `[IMPORTANT: ...]` message into the owning
/// session through the assembled node's one lifecycle-aware inject seam.
struct NodeProcessNotifier {
    node: Arc<NodeApiImpl>,
}

#[async_trait::async_trait]
impl daemon_processes::ProcessNotifier for NodeProcessNotifier {
    async fn notify(&self, owner: &SessionId, text: String) {
        if let Err(e) = self.node.inject_session_input(owner, text).await {
            tracing::warn!(
                owner = %owner,
                error = %e,
                "background-process notification could not be injected"
            );
        }
    }
}

/// The node's workspace-root resolver: shared by every engine's exec-env builder (so agents root
/// under it) and the filesystem surface (so operator + agent see one filesystem). `None` keeps the
/// per-session temp-sandbox default (tests / nodes without a workspace).
fn build_workspace_roots(a: &NodeAssembly) -> Option<Arc<WorkspaceRoots>> {
    a.workspace_root.clone().map(|base| {
        // Host browse roots for discovery before binding (daemon-fs-surface-spec.md): default to the
        // node user's home directory. (An operator allowlist / recents can extend this later.)
        let mut browse = Vec::new();
        if let Some(home) = std::env::var_os("HOME") {
            browse.push(("home".to_string(), PathBuf::from(home)));
        }
        Arc::new(WorkspaceRoots::new(base).with_browse_roots(browse))
    })
}

/// The content store (blob CAS), shared by the durable job worker (materializing delegated
/// attachments) and the NodeApi `blob_*`/`fs_write_from_blob` surface. A failed open leaves it
/// unbound (those ops resolve to Unsupported; attachment transfer is a no-op).
fn build_blob_store(a: &NodeAssembly) -> Option<Arc<dyn BlobStore>> {
    a.blob_root.as_ref().and_then(|root| {
        FileBlobStore::open(root.clone())
            .ok()
            .map(|s| Arc::new(s) as Arc<dyn BlobStore>)
    })
}

/// Build the constrained fleet-child profile: the core fs + shell + skills toolset (no orchestrate /
/// cron), dressed with the node subsystems + credentials and forced to the autonomous policy.
fn build_child_profile(
    a: &NodeAssembly,
    launch_skill_tools: &[Arc<dyn Tool>],
    launch_index: Option<&Arc<dyn StablePromptSource>>,
    autonomous_config: Config,
    workspace_roots: &Option<Arc<WorkspaceRoots>>,
    procs: &ProcessToolkit,
) -> EngineProfile {
    root_profile(
        dress(
            EngineProfile::new(
                provider_for(&a.providers, CHILD_PROFILE),
                Arc::new(core_tool_registry_with_skills(
                    &a.extra_tools,
                    launch_skill_tools,
                    &a.fs,
                    procs,
                )),
                SystemPrompt::new("fleet child"),
            ),
            a,
            launch_index,
        )
        .with_config(autonomous_config),
        workspace_roots,
    )
}

/// Build the orchestrator-capable engine shape: the core toolset *plus* the orchestrate tool (depth
/// guard `nesting_depth + 1`) *plus* the `cron` scheduling tool, dressed + forced autonomous.
#[allow(
    clippy::too_many_arguments,
    reason = "the assemble() phase helpers thread the shared single-instance handles positionally; \
              keeping the signature flat keeps the wiring diff minimal across workstreams"
)]
fn build_orchestrator_profile(
    a: &NodeAssembly,
    fleet: &FleetRuntime,
    launch_skill_tools: &[Arc<dyn Tool>],
    launch_index: Option<&Arc<dyn StablePromptSource>>,
    cron_tool: &Arc<dyn Tool>,
    profile_tool: Option<&Arc<dyn Tool>>,
    autonomous_config: Config,
    workspace_roots: &Option<Arc<WorkspaceRoots>>,
    procs: &ProcessToolkit,
) -> EngineProfile {
    let mut registry =
        core_tool_registry_with_skills(&a.extra_tools, launch_skill_tools, &a.fs, procs);
    registry.register(Arc::new(
        daemon_tool_orchestrate::OrchestrateTool::new(fleet.clone())
            // The effective depth guard composes the `[orchestrate].max_depth` policy ceiling with
            // the assembly recursion budget (`nesting_depth + 1`): policy may narrow the
            // structural budget but never widen it (the pre-v29 `nesting_depth + 1` behavior is
            // the default, since the default cap of 8 exceeds it).
            .with_max_depth(a.orchestrate.max_depth.min(a.nesting_depth + 1))
            .with_max_fanout(a.orchestrate.max_fanout)
            // The ephemeral (transient-subagent) fan cap — the agent-created-agents guardrail, sibling
            // to fanout but scoped to ephemeral-role children (joining or detached).
            .with_max_ephemeral(a.orchestrate.max_ephemeral_per_session)
            // The durable session graph backs the tool's `send` (pending-input + wake) and
            // per-child `status` verbs.
            .with_store(a.store.clone()),
    ));
    registry.register(cron_tool.clone());
    // Phase 2: the `profile_manage` tool goes on the orchestrator-capable profile (alongside
    // `orchestrate`/`cron`), so an authored profile id can feed a later `spawn { source: Profile }`.
    // Absent on a node without profile management. Constrained child/cron-run profiles never get it.
    if let Some(profile_tool) = profile_tool {
        registry.register(profile_tool.clone());
    }
    root_profile(
        dress(
            EngineProfile::new(
                provider_for(&a.providers, ORCHESTRATOR_PROFILE),
                Arc::new(registry),
                SystemPrompt::new("daemon host node"),
            ),
            a,
            launch_index,
        )
        .with_config(autonomous_config),
        workspace_roots,
    )
}

/// The resident cron scheduler (I15) + its shared ops surface, built BEFORE the agent profiles so
/// the agent-facing `cron` tool can wrap the same `CronOps` the operator `cron_*` control ops use
/// (one job engine, not two). The worker seeds its isolated cron sessions from the *constrained*
/// `child_profile` shape (no `orchestrate`/`cron`) — and the durable factory re-hydrates every
/// cron-fired session under that same constrained `cron_profile` (G3), so a scheduled run can never
/// self-schedule or self-delegate.
fn build_cron_stack(
    a: &NodeAssembly,
    child_profile: &EngineProfile,
    workspace_roots: &Option<Arc<WorkspaceRoots>>,
) -> CronStack {
    let cron_run_profile = child_profile.clone();
    let mut cron_worker = CronWorker::new(a.store.clone(), a.partition, cron_run_profile.clone());
    if let Some(roots) = workspace_roots {
        cron_worker = cron_worker.with_scripts_dir(roots.workspace_root().join("scripts"));
    }
    // Preload `CronSpec::skills` from the launch profile's skill library (the same library the
    // constrained cron-run profile exposes via `skill_*`), so a scheduled run carries the skill
    // bodies a chat would have `skill_view`'d. No skills subsystem -> on-demand `skill_*` only.
    if let Some(skills) = &a.skills {
        let provider = skills.clone();
        let profile_id = a.profile.as_str().to_string();
        let loader: CronSkillLoader =
            Arc::new(move |name: &str| provider.for_profile(&profile_id).view(name, None).ok());
        cron_worker = cron_worker.with_skill_loader(loader);
    }
    let cron_worker = Arc::new(cron_worker);
    let mut cron_ops_builder =
        CronOps::new(a.store.clone()).with_firing(cron_worker.clone() as Arc<dyn CronFiring>);
    // The `metadata.daemon.blueprint` skill bridge: scan the launch profile's skills (cheaply, on
    // each suggestion seed) and offer any runnable blueprint as a consent-first cron suggestion.
    if let Some(skills) = &a.skills {
        let provider = skills.clone();
        let profile_id = a.profile.as_str().to_string();
        let source: BlueprintSource = Arc::new(move || {
            provider
                .for_profile(&profile_id)
                .discover()
                .into_iter()
                .filter_map(|entry| {
                    let bp = entry.frontmatter.blueprint()?;
                    daemon_host::blueprint_suggestion(&entry.name, bp)
                })
                .collect()
        });
        cron_ops_builder = cron_ops_builder.with_blueprints(source);
    }
    let cron_ops = Arc::new(cron_ops_builder);
    // The agent veneer over the cron ops; registered into the agent-facing profiles (and into the
    // interactive `SessionFactoryCtx`), but deliberately NOT into `child_profile` / `cron_run_profile`,
    // so it is absent from cron-fired runs (defense in depth alongside the tool's in-cron refusal guard).
    let cron_tool = Arc::new(daemon_tool_cron::CronTool::new(cron_ops.clone())) as Arc<dyn Tool>;
    CronStack {
        cron_run_profile,
        cron_worker,
        cron_ops,
        cron_tool,
    }
}

/// The shared profile-authoring surface (Phase 2): one [`ProfileOps`] over the node's profile store
/// (+ revision log) backing BOTH the operator `profile_create`/`profile_update` ops and the agent
/// `profile_manage` tool, so both author through one validation + persistence + revision path. The
/// facade's engine/inference validator is late-bound to the assembled node (`ProfileValidator =
/// NodeApiImpl`). The tool holds the durable session store for the subtree-authorization check.
/// `None` on a node without profile management.
fn build_profile_stack(a: &NodeAssembly, node_events: &Arc<NodeEventFeed>) -> Option<ProfileStack> {
    let profiles = a.profiles.clone()?;
    let mut ops = ProfileOps::new(profiles);
    if let Some(revisions) = a.revisions.clone() {
        ops = ops.with_revisions(revisions);
    }
    // The node-wide `ProfilesChanged` emit sink (Phase 3): a create/update/delete through this shared
    // facade (operator ops OR the agent `profile_manage` tool) pings the feed so a thin client
    // refetches the profile list. The feed IS the sink (`impl ProfileEvents for NodeEventFeed`).
    ops = ops.with_events(node_events.clone() as Arc<dyn daemon_host::ProfileEvents>);
    let profile_ops = Arc::new(ops);
    let profile_tool = Arc::new(
        daemon_tool_profile::ProfileManageTool::new(profile_ops.clone(), a.store.clone())
            // The composed-profiles cap — the agent-created-agents guardrail; the same
            // `[orchestrate].max_composed_profiles` policy surfaced read-only via `Caps`.
            .with_max_composed(a.orchestrate.max_composed_profiles),
    ) as Arc<dyn Tool>;
    Some(ProfileStack {
        profile_ops,
        profile_tool,
    })
}

/// The one per-session resolution context, shared by the live session builder and the durable
/// rehydration resolver. Present only when the node carries a profile store + provider resolver.
fn build_session_ctx(
    a: &NodeAssembly,
    shared: &Shared,
    cron_tool: &Arc<dyn Tool>,
    procs: &ProcessToolkit,
) -> Option<(Arc<dyn ProfileStore>, Arc<SessionFactoryCtx>)> {
    match (a.profiles.clone(), a.provider_resolver.clone()) {
        (Some(store), Some(resolver)) => {
            // Interactive sessions get the `cron` tool too (so a chatting agent can schedule),
            // alongside the node's `extra_tools`. Cron-fired runs never resolve through this ctx
            // (they hydrate under the constrained `cron_run_profile`), so this stays agent-only.
            let mut session_extra = a.extra_tools.clone();
            session_extra.push(cron_tool.clone());
            let ctx = Arc::new(SessionFactoryCtx {
                resolver,
                extra_tools: session_extra,
                engine_config: a.engine_config,
                credentials: a.credentials.clone(),
                context: a.context.clone(),
                context_builder: a.context_builder.clone(),
                memory: a.memory.clone(),
                memory_builder: a.memory_builder.clone(),
                prompt_sources: a.prompt_sources.clone(),
                skills_resolver: a.skills_resolver.clone(),
                workspace_roots: shared.workspace_roots.clone(),
                fs_config: a.fs.clone(),
                procs: procs.clone(),
            });
            Some((store, ctx))
        }
        _ => None,
    }
}

/// The durable engine factory: journals per turn into the shared store, wires the background-review
/// spawner + the constrained cron profile, optionally re-resolves a durable session from its bound
/// profile + overlay on rehydration, and threads the content store + workspace roots.
///
/// When the node hosts profiles, the Core factory is wrapped in a [`DispatchingEngineFactory`] so a
/// delegated child whose bound profile is `Foreign{agent}` runs as that ACP / stream-json agent (via
/// [`ForeignIncarnation`](crate::fleet::foreign_incarnation)) instead of silently falling back to
/// Core. Without a profile store there are no foreign bindings, so the plain Core factory is used.
fn build_factory(
    a: &NodeAssembly,
    orchestrator_profile: &EngineProfile,
    cron_run_profile: &EngineProfile,
    session_ctx: &Option<(Arc<dyn ProfileStore>, Arc<SessionFactoryCtx>)>,
    shared: &Shared,
) -> Arc<dyn EngineFactory> {
    let mut factory = CoreEngineFactory::from_profile(orchestrator_profile.clone())
        .with_journal(a.store.clone(), shared.signer.clone())
        .with_background(shared.background.clone())
        // I15/G3: a cron-fired session (`session_meta.scheduled_job`) hydrates under the constrained,
        // `cron`/`orchestrate`-free profile so a scheduled run cannot self-schedule or self-delegate.
        .with_cron_profile(cron_run_profile.clone());
    if let Some((store, ctx)) = session_ctx {
        let store = store.clone();
        let ctx = ctx.clone();
        // Re-resolve a durable session's engine at hydrate. Precedence:
        //   1. an INLINE sub-agent spec (Phase 1): the opaque `ProfileSpec` persisted in
        //      `SessionMeta.inline_profile` (bound_profile is `None` for an inline child). A Core
        //      inline spec resolves here; a Foreign inline spec is routed to the dispatching
        //      factory's foreign incarnation, so this returns `None` (the foreign path).
        //   2. a bound profile name -> the profile store.
        //   3. neither (e.g. a delegated orchestrator child) -> `None`, so the factory keeps its
        //      orchestrator profile.
        // `resolve_effective` stays Core-only: a `Foreign{agent}` binding/inline is routed to the
        // foreign incarnation, so the resolver only ever builds Core specs.
        let resolver: DurableProfileResolver = Arc::new(
            move |bound: Option<ProfileRef>, inline: &[u8], overlay: &SessionOverlay| {
                if !inline.is_empty() {
                    let spec = from_cbor::<ProfileSpec>(inline).ok()?;
                    return match spec.engine {
                        EngineSelector::Core => Some(ctx.resolve_effective(&spec, overlay)),
                        EngineSelector::Foreign { .. } => None,
                    };
                }
                let bound = bound?;
                let spec = store.get(bound.as_str()).ok().flatten()?;
                Some(ctx.resolve_effective(&spec, overlay))
            },
        );
        factory = factory.with_session_resolver(resolver);
    }
    // Give durable incarnations the content store + workspace roots so a completed child captures
    // its outbox/ into blobs and a waking parent materializes the returned artifacts into its inbox/.
    if let (Some(roots), Some(blobs)) = (&shared.workspace_roots, &shared.blob_store) {
        factory = factory.with_content(blobs.clone(), roots.clone());
    }
    match session_ctx {
        // Profiles present: dispatch Core vs `Foreign{agent}` per delegated child at hydrate.
        Some((profiles, _)) => Arc::new(DispatchingEngineFactory::new(
            factory,
            ForeignConfig {
                profiles: profiles.clone(),
                store: a.store.clone(),
                signer: shared.signer.clone(),
                gateway: a.foreign_gateway.clone(),
            },
        )),
        // No profile store: no foreign bindings are possible, so the plain Core factory suffices.
        None => Arc::new(factory),
    }
}

/// One durable job worker for the whole node: every delegation (top or nested) materializes a
/// parent-bound durable child session seeded from the (moved-in) `orchestrator_profile`.
fn build_job_worker(
    a: &NodeAssembly,
    orchestrator_profile: EngineProfile,
    session_ctx: &Option<(Arc<dyn ProfileStore>, Arc<SessionFactoryCtx>)>,
    shared: &Shared,
) -> FleetJobWorker {
    let mut job_worker = FleetJobWorker::new(a.store.clone(), a.partition, orchestrator_profile)
        .with_event_sink(shared.fleet_events.clone());
    // Give the worker the workspace roots + content store so it can materialize delegated
    // attachments from the parent's workspace into the child's inbox/ (node-mediated).
    if let (Some(roots), Some(blobs)) = (&shared.workspace_roots, &shared.blob_store) {
        job_worker = job_worker.with_workspace(roots.clone(), blobs.clone());
    }
    // Give the worker the profile store so a delegated child bound to a `Foreign{agent}` profile is
    // seeded for the durable foreign path (empty snapshot + task on the input seam).
    if let Some(profiles) = a.profiles.clone() {
        job_worker = job_worker.with_profiles(profiles);
    }
    // Give the worker the per-session resolution ctx so a Core INLINE sub-agent (Phase 1) seeds its
    // first snapshot from `resolve_effective(inline_spec)` (persona/toolset reflected from the start).
    if let Some((_, ctx)) = session_ctx {
        job_worker = job_worker.with_session_ctx(ctx.clone());
    }
    job_worker
}

/// Build the constrained fixed-session profile (carries the `cron` tool so a single-profile node's
/// chatting agent can also schedule work), dressed with the node subsystems + credentials.
fn build_session_profile(
    a: &NodeAssembly,
    launch_skill_tools: &[Arc<dyn Tool>],
    launch_index: Option<&Arc<dyn StablePromptSource>>,
    cron_tool: &Arc<dyn Tool>,
    workspace_roots: &Option<Arc<WorkspaceRoots>>,
    procs: &ProcessToolkit,
) -> EngineProfile {
    let mut session_registry =
        core_tool_registry_with_skills(&a.extra_tools, launch_skill_tools, &a.fs, procs);
    session_registry.register(cron_tool.clone());
    root_profile(
        dress(
            EngineProfile::new(
                provider_for(&a.providers, a.profile.as_str()),
                Arc::new(session_registry),
                SystemPrompt::new("interactive session"),
            ),
            a,
            launch_index,
        ),
        workspace_roots,
    )
}

/// The profile-aware interactive session builder: when the node carries a profile store + provider
/// resolver, each session resolves its bound profile bundle at open, applies the persisted session
/// overlay, and materializes its backend from the result (the same `resolve_effective` the durable
/// path uses). Otherwise sessions are built from the single fixed `session_profile` (moved in here).
///
/// The profile's `engine` selector picks the backend: `Core` runs the native in-process engine
/// (provider/model resolution as before); `Foreign { agent }` returns a deferred foreign factory
/// that resolves the agent's catalog recipe + protocol node-side at spawn (`fleet::foreign_live`)
/// — the genai provider/model path is bypassed entirely for foreign engines. `session_store`
/// supplies the durable agent registrations that resolution reads.
fn build_session_builder(
    session_ctx: &Option<(Arc<dyn ProfileStore>, Arc<SessionFactoryCtx>)>,
    session_profile: EngineProfile,
    session_store: Arc<dyn daemon_store::SessionStore>,
    foreign_gateway: Option<crate::GatewayCoords>,
) -> SessionEngineBuilder {
    match session_ctx {
        Some((store, ctx)) => {
            let store = store.clone();
            let ctx = ctx.clone();
            let fallback = session_profile;
            Arc::new(
                move |id: SessionId, requested: Option<ProfileRef>, overlay: &SessionOverlay| {
                    // Routing's agent-selection seam: build from the explicitly-requested profile when
                    // one is supplied, else the node's active default (the legacy single-profile path).
                    let spec = match requested {
                        Some(profile) => store.get(profile.as_str()).ok().flatten(),
                        None => store
                            .active()
                            .ok()
                            .flatten()
                            .and_then(|active| store.get(&active).ok().flatten()),
                    };
                    match spec {
                        Some(spec) => match &spec.engine {
                            daemon_api::EngineSelector::Core => SessionBackend::Core(
                                ctx.resolve_effective(&spec, overlay).fresh(id),
                            ),
                            daemon_api::EngineSelector::Foreign { agent } => {
                                // A persisted per-session model override still steers a foreign
                                // session at open (Phase 3 makes `SetSessionModel` fully
                                // foreign-aware); it takes precedence over the profile's foreign
                                // backend model. The shared `spawn_foreign_session` helper (reused by
                                // the durable foreign incarnation) resolves the recipe node-side and
                                // applies the AgentNative/NodeProvider backend policy + gateway token.
                                let overlay_model =
                                    overlay.model.clone().filter(|m| !m.trim().is_empty());
                                let agent = agent.clone();
                                let backend = spec.foreign_backend.clone();
                                let store = session_store.clone();
                                let gateway = foreign_gateway.clone();
                                SessionBackend::Foreign(Box::new(move |host| {
                                    Box::pin(async move {
                                        crate::fleet::foreign_live::spawn_foreign_session(
                                            agent,
                                            backend,
                                            overlay_model,
                                            id,
                                            store,
                                            gateway,
                                            host,
                                        )
                                        .await
                                    })
                                }))
                            }
                        },
                        None => SessionBackend::Core(fallback.fresh(id)),
                    }
                },
            )
        }
        None => {
            let profile = session_profile;
            Arc::new(
                move |id: SessionId, _requested: Option<ProfileRef>, _overlay: &SessionOverlay| {
                    SessionBackend::Core(profile.fresh(id))
                },
            )
        }
    }
}

/// Bind every optional `NodeApiImpl` sub-surface + the L3 fleet-change bridge, in the original order.
/// Each surface is installed only when the node carries the backing policy (`if let Some(..)`), so
/// these binders — not `assemble` — own that branching. The order is the contract; the work is split
/// across cohesive sub-binders. The shared `node_events`/`fleet_events`/`background` come from [`Shared`].
fn bind_node_api_surfaces(
    node_api: NodeApiImpl,
    a: &NodeAssembly,
    cron_ops: &Arc<CronOps>,
    shared: &Shared,
) -> NodeApiImpl {
    spawn_fleet_change_bridge(shared);
    let node_api = bind_storage_surfaces(node_api, cron_ops, shared);
    let node_api = bind_model_surface(node_api, a, shared);
    let node_api = bind_identity_surfaces(node_api, a);
    let node_api = install_routing(node_api, a);
    bind_discovery_surfaces(node_api, a, shared)
}

/// L3 fleet liveness: bridge the fleet topology bus (`fleet_events`, consumed by `tree_subscribe`)
/// onto the node-wide feed as a coalesced `FleetChanged`, so `events_since` clients learn the
/// subagent tree changed (spawn / state / finish) and re-fetch `Tree` live - without threading the
/// feed through the orchestration crate (only `NodeApiImpl`/`LiveSessions` can reach it directly).
/// `FleetChanged` coalesces in the feed ring, so a spawn burst is one client refetch; a `Lagged`
/// (the bridge fell behind the bus) is itself just "the tree changed".
fn spawn_fleet_change_bridge(shared: &Shared) {
    let feed = shared.node_events.clone();
    let mut rx = shared.fleet_events.subscribe();
    tokio::spawn(async move {
        use tokio::sync::broadcast::error::RecvError;
        // Loop until the bus closes; both a value and a `Lagged` mean "the tree changed".
        while let Ok(_) | Err(RecvError::Lagged(_)) = rx.recv().await {
            let rev = feed.note_fleet_change();
            feed.emit(daemon_api::NodeEvent::FleetChanged { rev });
        }
    });
}

/// Bind the workspace (`fs_*`), content-store (`blob_*`), and cron (I15) surfaces.
fn bind_storage_surfaces(
    mut node_api: NodeApiImpl,
    cron_ops: &Arc<CronOps>,
    shared: &Shared,
) -> NodeApiImpl {
    // Bind the filesystem / workspace surface (`fs_*`) over the SAME `WorkspaceRoots` the engine
    // exec builders root at, so operator and agent see one filesystem.
    if let Some(roots) = &shared.workspace_roots {
        node_api = node_api.with_workspace(Arc::new(WorkspaceFs::new(roots.clone())));
    }
    // Bind the content store (blob CAS) surface, reusing the shared store built above.
    if let Some(blobs) = &shared.blob_store {
        node_api = node_api.with_blobs(blobs.clone());
    }
    // Bind the cron operations surface (I15): the SAME shared `CronOps` (with the resident
    // `CronWorker` as its manual-fire handle) that backs the agent `cron` tool, so the operator
    // control ops and the agent tool create/trigger through one path.
    node_api.with_cron(cron_ops.clone())
}

/// Bind the model-management sub-surface (when this node hosts local-inference model management),
/// fanning download progress + catalog changes onto the node-wide feed so the client renders both
/// without polling.
fn bind_model_surface(mut node_api: NodeApiImpl, a: &NodeAssembly, shared: &Shared) -> NodeApiImpl {
    if let Some(models) = a.models.clone() {
        // L3: pct is derived from the byte counters; state mirrors the wire string; the raw byte
        // counters ride the event so the client renders real progress (wire v26).
        let feed = shared.node_events.clone();
        models.set_download_progress(Arc::new(move |status: daemon_common::DownloadStatus| {
            let pct = status
                .downloaded_bytes
                .saturating_mul(100)
                .checked_div(status.total_bytes)
                .unwrap_or(0)
                .min(100) as u32;
            let state = match status.state {
                daemon_common::DownloadState::Queued => "Queued",
                daemon_common::DownloadState::Downloading => "Downloading",
                daemon_common::DownloadState::Completed => "Completed",
                daemon_common::DownloadState::Paused => "Paused",
                daemon_common::DownloadState::Cancelled => "Cancelled",
                daemon_common::DownloadState::Failed => "Failed",
            };
            feed.emit(daemon_api::NodeEvent::DownloadProgress {
                id: status.id,
                pct,
                state: state.to_string(),
                downloaded_bytes: status.downloaded_bytes,
                total_bytes: status.total_bytes,
            });
        }));
        // L3 (wire v26): the installed-model registry changed (a completed download was cataloged
        // / a model was deleted) — clients refetch ModelCatalog instead of polling.
        let feed = shared.node_events.clone();
        models.set_catalog_changed(Arc::new(move || {
            feed.emit(daemon_api::NodeEvent::CatalogChanged);
        }));
        node_api = node_api.with_models(models, a.profile.as_str().to_string());
    }
    node_api
}

/// Bind the identity sub-surfaces: profiles, credentials, interactive-auth families, the revision
/// log, and the skills provider — each installed only when this node carries the backing policy.
fn bind_identity_surfaces(mut node_api: NodeApiImpl, a: &NodeAssembly) -> NodeApiImpl {
    // Bind the profile/config sub-surface when this node hosts profile management.
    if let Some(profiles) = a.profiles.clone() {
        node_api = node_api.with_profiles(profiles);
    }
    // Bind the credential sub-surface when this node hosts credential management.
    if let Some(credentials) = a.credential_store.clone() {
        node_api = node_api.with_credential_store(credentials);
    }
    // Register the interactive-auth families (Matrix SSO, future OAuth2/OIDC) when any are supplied,
    // so a decoupled client can drive a browser-redirect login over the wire `AuthApi`.
    if !a.auth_factories.is_empty() {
        node_api = node_api.with_auth_factories(a.auth_factories.clone());
    }
    // Bind the profile/skill versioning surface when this node hosts a revision log.
    if let Some(revisions) = a.revisions.clone() {
        node_api = node_api.with_revisions(revisions);
    }
    if let Some(skills) = a.skills.clone() {
        node_api = node_api.with_skills(skills);
    }
    node_api
}

/// Install the host routing registry (§5.9) so routed submits select the session's profile +
/// delivery from the inbound origin. The account→profile baseline (precedence step 2) is derived
/// from every profile's `bound_accounts` (§5.9.4): profile-declared instance bindings fill the
/// registry's `instance_profiles`, while any explicit config `[[routing.instance_profile]]` already
/// present wins (operator override).
///
/// When the node carries a profile store, the derivation is installed as the routing **rebuild
/// hook** ([`RoutingBuilder`], the §5.9 hot-reload seam) rather than a boot-time snapshot: every
/// `rebuild_routing()` (fired by `profile_update` / `auth_complete` / the `routing_*` ops)
/// recomputes the baseline from the LIVE profile store, so binding or un-binding an account takes
/// effect without a restart (EIO-3/EIO-7/EIO-11 — the `routing_builder` hole). Without a profile
/// store there is nothing live to re-derive from, so the static config table (if any) is installed
/// as before.
fn install_routing(node_api: NodeApiImpl, a: &NodeAssembly) -> NodeApiImpl {
    let config_base = a.routing.clone();
    match a.profiles.clone() {
        Some(profiles) => {
            let builder: RoutingBuilder = Arc::new(move || {
                let specs = profiles.list().unwrap_or_default();
                match config_base.clone() {
                    Some(reg) => reg.bind_instances_from_profiles(&specs),
                    None => RoutingRegistry::new().bind_instances_from_profiles(&specs),
                }
            });
            // The consuming builder runs the hook once to seed the live table (boot-equivalent to
            // the previous static derivation).
            node_api.with_routing_builder(builder)
        }
        None => match config_base {
            Some(reg) => node_api.with_routing(reg),
            None => node_api,
        },
    }
}

/// Bind the discovery + live-rebind surfaces: the cloud-model catalog, the (always-on) ACP discovery
/// hook, the live model-switch factory, the background-review spawner, and the checkpoint store.
//
// Transport-adapter registry seam (daemon-transport-adapter-spec.md §3.4): the declarative companion
// to routing. No adapter implements `TransportAdapter` yet (the `serve` spawns still live in
// `bins/daemon`), so the registry stays empty/inert here; populating it + driving lifecycle from the
// registry is deferred (spec §7 P1).
fn bind_discovery_surfaces(
    mut node_api: NodeApiImpl,
    a: &NodeAssembly,
    shared: &Shared,
) -> NodeApiImpl {
    // Bind the live cloud-model discovery hook when the binary provided one.
    if let Some(cloud_catalog) = a.cloud_catalog.clone() {
        node_api = node_api.with_cloud_catalog(cloud_catalog);
    }
    // Wire the server-side foreign-agent discovery hook (I7): the host's `agent_discover` op probes
    // the curated direct-binary recipe table (ACP entries via the `initialize` handshake). The host
    // cannot link the ACP runtime directly (`daemon-acp` depends on `daemon-host`), so the
    // discoverer is injected here.
    node_api = node_api.with_agent_discovery(Arc::new(daemon_acp::AcpDiscoverer::new()));
    // Bind the live model-switch factory when this node resolves per-session profiles: a
    // `SetSessionModel` rebuilds a running session's provider for the new model id from the
    // (model-overridden) profile bundle via the same provider resolver.
    if let Some(resolver) = a.provider_resolver.clone() {
        let factory: ModelProviderFactory = Arc::new(move |spec| (resolver)(spec)());
        node_api = node_api.with_model_factory(factory);
    }
    // Bind the background-review spawner so live sessions materialize `Spawn` requests fire-and-forget.
    node_api = node_api.with_background(shared.background.clone());
    // Bind the §12 tool-checkpoint store so the `Checkpoint{List,Rewind}` ops see the same rewind
    // points the engines record.
    if let Some(checkpoints) = a.checkpoints.clone() {
        node_api = node_api.with_checkpoints(checkpoints);
    }
    node_api
}
