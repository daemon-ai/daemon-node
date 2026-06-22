//! `daemon-node` — the single host-composition root.
//!
//! Phases 1-11 grew the node's wiring (durable store + resident services, the orchestration fleet as
//! the real job worker, the credential broker, and the live session surface) inline in `bins/daemon`,
//! with a near-identical copy in the conformance harness. [`assemble`] collapses that into one place:
//! both the binary and the gate build their node through it, so there is exactly one composition to
//! keep correct. It lives above `daemon-host` because the fleet + orchestrate-tool glue is
//! composition-layer policy — `daemon-host` deliberately does not depend on `daemon-orchestration`.
//!
//! Callers supply only *policy*: the store, the [`ProviderRegistry`] (provider selection seam),
//! optional brokered credentials, the session/credential [`ProfileRef`], and the engine
//! [`Config`](daemon_core::Config). [`assemble`] does the standard plumbing (three role
//! `EngineProfile`s, the fleet, the durable factory, the host, and the [`NodeApiImpl`]).

#![forbid(unsafe_code)]

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use daemon_api::{EngineTunables, FleetReport, ProfileSpec};
use daemon_common::{Budget, JournalStreamId, PartitionId, ProfileRef, SessionId, UnitId};
use daemon_core::{
    ApprovalPolicy, Config, ContextEngine, ContextEngineBuilder, CredentialBuilder, EngineProfile,
    MemoryBuilder, MemoryProvider, ProviderBuilder, ProviderRegistry, StablePromptSource,
    SystemPrompt, Tool, ToolRegistry,
};
use daemon_host::{
    AgentSession, AgentUnit, BackgroundProfile, BackgroundProfileRegistry, BackgroundSpawner,
    CodecSession, CoreEngineFactory, CredentialStore, EngineUnit, FleetControl, Host, HostConfig,
    JobWorker, JournalConfig, JournalFeeder, JournalSink, ModelProviderFactory, NodeApiImpl,
    ProcessAgentUnit, ProfileStore, ServiceError, SessionEngineBuilder, StreamJsonCodec,
    SupervisorHandle,
};
use daemon_orchestration::{ChildSpawner, DefaultAnswerPolicy, FleetRuntime};
use daemon_protocol::HostRequestHandler;
use daemon_provision::{PlacementSpec, ProcessProvisioner, Provisioner};
use daemon_supervision::{DelegationSpec, ManageRequestHandler, ManagedUnit};
use daemon_telemetry::TraceSigner;

/// The provider-registry profile name the orchestrator (parent) engine resolves to.
const ORCHESTRATOR_PROFILE: &str = "orchestrator";
/// The provider-registry profile name the (legacy synchronous) fleet-child engine resolves to.
const CHILD_PROFILE: &str = "child";

/// The skills toolset names a `skill_review` background child is constrained to (hermes' skills-only
/// review whitelist). Kept in sync with `daemon_tool_skill::SKILL_TOOL_NAMES`.
const SKILL_TOOL_NAMES: [&str; 3] = ["skills_list", "skill_view", "skill_manage"];
/// The name prefix of Mnemosyne memory tools a `memory_review` background child is constrained to.
const MEMORY_TOOL_PREFIX: &str = "mnemosyne_";
/// The bounded iteration cap for a background-review child (hermes `max_iterations=16`).
const BACKGROUND_MAX_ITERATIONS: u32 = 16;

/// The `skill_review` background child's seeding instruction (a condensed port of hermes'
/// `_SKILL_REVIEW_PROMPT`): curate skills from what just happened, preferring to patch existing
/// umbrella skills, never editing bundled/hub skills, and writing only to the local skills dir.
const SKILL_REVIEW_PROMPT: &str = "\
You are a background skill curator reviewing the conversation that just completed. Identify any \
durable, reusable procedure, preference, or pitfall worth capturing as a skill. Prefer `patch`ing \
an existing, loaded skill over creating a new one; create a new skill only for a genuinely new, \
class-level capability. Do not edit bundled or hub-installed skills. Keep skills concise and \
general. If nothing is worth saving, do nothing and finish. Use only the skills tools.";

/// The `memory_review` background child's seeding instruction: persist durable facts/preferences from
/// the conversation into long-term memory.
const MEMORY_REVIEW_PROMPT: &str = "\
You are a background memory curator reviewing the conversation that just completed. Persist any \
durable facts, user preferences, or decisions worth remembering into long-term memory using the \
memory tools. Be precise and avoid duplicating what is already stored. If nothing is worth saving, \
do nothing and finish.";

/// Resolves a [`ProviderBuilder`] for a profile bundle — the seam letting the binary map a
/// [`ProfileSpec`]'s `provider`/`model`/`base_url` onto a concrete provider client without
/// `daemon-node` depending on `daemon-providers`. When a node supplies a resolver and a
/// [`ProfileStore`], interactive sessions resolve their provider/persona/tools/budget per session
/// from the active profile (so a GUI can switch model/provider live); otherwise the node falls back
/// to a single fixed session profile.
pub type ProviderResolver = Arc<dyn Fn(&ProfileSpec) -> ProviderBuilder + Send + Sync>;

/// The policy inputs for [`assemble`]: everything that varies between a production node and a test
/// node. The standard plumbing (role profiles, fleet, factory, host, session surface) is derived.
pub struct NodeAssembly {
    /// The durable store backend (shared by the host, fleet, and control surface).
    pub store: Arc<dyn daemon_store::SessionStore>,
    /// The partition this node owns.
    pub partition: PartitionId,
    /// Resident-service cadence + supervision policy.
    pub host_config: HostConfig,
    /// The provider *selection* seam: the orchestrator/child engines resolve `"orchestrator"`/
    /// `"child"`, the session engine resolves `profile` (falling back to the registry default).
    pub providers: ProviderRegistry,
    /// The brokered credential builder applied uniformly to every engine (durable, live, child);
    /// `None` leaves engines on their embedded L1 pool (tests).
    pub credentials: Option<CredentialBuilder>,
    /// The session + credential profile name.
    pub profile: ProfileRef,
    /// The engine tunables (§20) every engine this node builds runs under.
    pub engine_config: Config,
    /// The 32-byte seed for the node's verifiable-journal signer, so its verifying key is stable
    /// across restarts (auditors keep verifying old segments). `None` generates an ephemeral key
    /// (fine for tests; a fresh key each boot otherwise).
    pub journal_seed: Option<[u8; 32]>,
    /// How many orchestrator levels the top fleet materializes before its leaves. `0` (default) is a
    /// flat fleet of engine leaves; `1` makes every top child an orchestrator owning a sub-fleet of
    /// leaves (fleets-of-fleets), `n` nests `n` deep — the tree the GUI projects and addresses.
    pub nesting_depth: usize,
    /// A shared (session-independent) §10 context engine injected into every engine this node builds.
    /// `None` keeps the in-core [`BudgetedContextEngine`](daemon_core::BudgetedContextEngine) (tests
    /// / CI). For stateful engines prefer [`Self::context_builder`].
    pub context: Option<Arc<dyn ContextEngine>>,
    /// A per-session §10 context-engine builder (e.g. LCM, which keeps per-session compaction state).
    /// Takes precedence over [`Self::context`] so each session gets its own instance.
    pub context_builder: Option<ContextEngineBuilder>,
    /// The default §11 memory providers (e.g. a frozen `FileMemory`) injected into every engine this
    /// node builds. Empty keeps memory off (tests / CI). For session-scoped backends prefer
    /// [`Self::memory_builder`].
    pub memory: Vec<Arc<dyn MemoryProvider>>,
    /// A per-session §11 memory builder (e.g. Mnemosyne, scoped by `session_id` over a shared bank).
    /// Takes precedence over [`Self::memory`] so each session gets its own provider set.
    pub memory_builder: Option<MemoryBuilder>,
    /// Extra tools (e.g. `mnemosyne_*` / `lcm_*`) registered into every role's tool registry on top
    /// of the core fs + shell toolset, so the model can drive memory/context backends.
    pub extra_tools: Vec<Arc<dyn Tool>>,
    /// The model-management facade backing the node's `ModelApi` sub-surface (search/download/
    /// catalog/activate). `None` builds a node without local-inference model management (tests, a
    /// remote-only node); the `ModelApi` calls then resolve to `ApiError::Unsupported`.
    pub models: Option<Arc<daemon_models::ModelManager>>,
    /// The durable profile store backing the node's `ProfileApi` sub-surface and per-session engine
    /// resolution. `None` builds a node without profile management (the `profile` field then fixes
    /// every interactive session's engine shape, the legacy behavior).
    pub profiles: Option<Arc<dyn ProfileStore>>,
    /// Maps an active [`ProfileSpec`] onto a concrete provider client. Required (with `profiles`) for
    /// per-session profile resolution; `None` keeps the fixed session profile.
    pub provider_resolver: Option<ProviderResolver>,
    /// The persisted credential store backing the node's `CredentialApi` sub-surface (the same store
    /// the credential authority provisions from). `None` builds a node without credential management.
    pub credential_store: Option<Arc<dyn CredentialStore>>,
    /// The live networked-model discovery hook backing `ModelApi::models()` (the binary's
    /// `genai`-backed catalog; the host never links `genai`). `None` lists only the static cloud
    /// catalog + local models.
    pub cloud_catalog: Option<Arc<dyn daemon_host::CloudCatalog>>,
    /// Generic stable-tier prompt sources (§10) folded into every engine's system prompt — e.g. the
    /// skills *index* ([`daemon_skills::SkillsPromptSource`](https://docs.rs)). Empty keeps the
    /// system prompt unchanged. The §4.3 background-review spawner is derived automatically from the
    /// skills/memory tools in [`Self::extra_tools`] and is inert unless the engine's review nudge
    /// intervals (`engine_config.skill_review_interval` / `memory_review_interval`) are non-zero.
    pub prompt_sources: Vec<Arc<dyn StablePromptSource>>,
}

/// The assembled node: the bound surface, its started resident-service handle, and the fleet handle.
pub struct AssembledNode {
    /// The one [`daemon_api`] surface (control + session + fleet sub-surfaces).
    pub node: Arc<NodeApiImpl>,
    /// The started resident-service tree; drive shutdown via [`SupervisorHandle::shutdown`].
    pub handle: SupervisorHandle,
    /// The orchestration fleet handle (e.g. for inspection in tests).
    pub fleet: FleetRuntime,
    /// The node's verifiable-journal signer — its verifying key is published so auditors can verify
    /// sealed history (`ControlApi::verifying_key`).
    pub signer: Arc<TraceSigner>,
}

/// Apply the engine tunables, the default context engine + memory providers (§10/§11), and the
/// optional brokered credentials uniformly to a role profile (credentials bound to the node profile).
fn dress(profile: EngineProfile, a: &NodeAssembly) -> EngineProfile {
    dress_with_credential(profile, a, a.profile.clone())
}

/// Like [`dress`] but binds credentials to an explicit profile ref (the per-session credential ref).
fn dress_with_credential(
    profile: EngineProfile,
    a: &NodeAssembly,
    cred_profile: ProfileRef,
) -> EngineProfile {
    let mut profile = profile.with_config(a.engine_config);
    // Per-session builders (stateful/session-scoped backends) take precedence over shared instances.
    if let Some(builder) = &a.context_builder {
        profile = profile.with_context_engine_builder(builder.clone());
    } else if let Some(context) = &a.context {
        profile = profile.with_context_engine(context.clone());
    }
    if let Some(builder) = &a.memory_builder {
        profile = profile.with_memory_builder(builder.clone());
    } else if !a.memory.is_empty() {
        profile = profile.with_memory(a.memory.clone());
    }
    for source in &a.prompt_sources {
        profile = profile.with_prompt_block(source.clone());
    }
    match &a.credentials {
        Some(credentials) => profile.with_credentials(credentials.clone(), cred_profile),
        None => profile,
    }
}

/// Whether `tool` belongs in a background child's constrained toolset: its name matches `names`
/// exactly or (when set) starts with `prefix`.
fn tool_matches(tool: &Arc<dyn Tool>, names: &[&str], prefix: Option<&str>) -> bool {
    let name = tool.name();
    names.contains(&name) || prefix.is_some_and(|p| name.starts_with(p))
}

/// Build a [`ToolRegistry`] holding only the tools in `extra` matching `names`/`prefix` — the
/// constrained toolset of a background-review child.
fn constrained_registry(
    extra: &[Arc<dyn Tool>],
    names: &[&str],
    prefix: Option<&str>,
) -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    for tool in extra {
        if tool_matches(tool, names, prefix) {
            registry.register(tool.clone());
        }
    }
    registry
}

/// Build the §4.3 background-review profile registry from the node's tools: a `skill_review` child
/// constrained to the skills tools, and a `memory_review` child constrained to the Mnemosyne memory
/// tools. Each runs under a bounded iteration cap with review nudges disabled (no recursion) and
/// inherits the node's provider + credentials, but starts from a clean base (no memory/context/index
/// — the reviewer drives its tools directly). A kind is registered only when its tools are present;
/// the returned registry may be empty (spawn is then a no-op).
fn background_registry(a: &NodeAssembly) -> BackgroundProfileRegistry {
    let mut registry = BackgroundProfileRegistry::new();
    let bg_config = Config {
        max_iterations: BACKGROUND_MAX_ITERATIONS,
        skill_review_interval: 0,
        memory_review_interval: 0,
        // A background-review child runs autonomously (no operator attached): never gate its tool
        // actions on a human, or the headless turn would suspend forever.
        approval_policy: ApprovalPolicy::AutoAllow,
        ..a.engine_config
    };
    // A clean base carrying only the node's provider (orchestrator selection) + brokered credentials.
    let base = |names: &[&str], prefix: Option<&str>, persona: &str| -> EngineProfile {
        let profile = EngineProfile::new(
            provider_for(&a.providers, ORCHESTRATOR_PROFILE),
            Arc::new(constrained_registry(&a.extra_tools, names, prefix)),
            SystemPrompt::new(persona),
        )
        .with_config(bg_config);
        match &a.credentials {
            Some(c) => profile.with_credentials(c.clone(), a.profile.clone()),
            None => profile,
        }
    };

    if a
        .extra_tools
        .iter()
        .any(|t| tool_matches(t, &SKILL_TOOL_NAMES, None))
    {
        registry = registry.with(
            "skill_review",
            BackgroundProfile::new(base(&SKILL_TOOL_NAMES, None, "skill curator"), SKILL_REVIEW_PROMPT),
        );
    }
    if a
        .extra_tools
        .iter()
        .any(|t| tool_matches(t, &[], Some(MEMORY_TOOL_PREFIX)))
    {
        registry = registry.with(
            "memory_review",
            BackgroundProfile::new(
                base(&[], Some(MEMORY_TOOL_PREFIX), "memory curator"),
                MEMORY_REVIEW_PROMPT,
            ),
        );
    }
    registry
}

/// Overlay a [`ProfileSpec`]'s engine-tunable overrides onto the node's base [`Config`].
fn merged_config(base: Config, t: &EngineTunables) -> Config {
    let mut c = base;
    if let Some(v) = t.model_retry_attempts {
        c.model_retry_attempts = v;
    }
    if let Some(v) = t.context_budget_tokens {
        c.context_budget_tokens = Some(v);
    }
    if let Some(v) = t.max_iterations {
        c.max_iterations = v;
    }
    if let Some(v) = t.tool_result_budget {
        c.tool_result_budget = v;
    }
    c
}

/// Build the interactive tool registry for a session: the core fs + shell toolset plus node-level
/// `extra` tools, optionally narrowed to an allowlist of tool names.
fn session_tool_registry(extra: &[Arc<dyn Tool>], allowlist: Option<&[String]>) -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    let mut candidates: Vec<Arc<dyn Tool>> = vec![
        Arc::new(daemon_tool_fs::FsTool::new()) as Arc<dyn Tool>,
        Arc::new(daemon_tool_shell::ShellTool::new()) as Arc<dyn Tool>,
    ];
    candidates.extend(extra.iter().cloned());
    for tool in candidates {
        let allowed = match allowlist {
            Some(list) => list.iter().any(|n| n == tool.name()),
            None => true,
        };
        if allowed {
            registry.register(tool);
        }
    }
    registry
}

/// The captured node context a profile-aware session builder needs to materialize a per-session
/// [`EngineProfile`] from the active [`ProfileSpec`] (provider + persona + tools + budget +
/// context/memory + credentials), without re-borrowing the consumed [`NodeAssembly`].
struct SessionFactoryCtx {
    resolver: ProviderResolver,
    extra_tools: Vec<Arc<dyn Tool>>,
    engine_config: Config,
    credentials: Option<CredentialBuilder>,
    context: Option<Arc<dyn ContextEngine>>,
    context_builder: Option<ContextEngineBuilder>,
    memory: Vec<Arc<dyn MemoryProvider>>,
    memory_builder: Option<MemoryBuilder>,
    prompt_sources: Vec<Arc<dyn StablePromptSource>>,
}

impl SessionFactoryCtx {
    /// Materialize a per-session engine profile from a profile bundle.
    fn profile_from_spec(&self, spec: &ProfileSpec) -> EngineProfile {
        let provider = (self.resolver)(spec);
        let registry = session_tool_registry(&self.extra_tools, spec.tool_allowlist.as_deref());
        let persona = if spec.system_prompt.trim().is_empty() {
            "interactive session".to_string()
        } else {
            spec.system_prompt.clone()
        };
        let mut profile = EngineProfile::new(provider, Arc::new(registry), SystemPrompt::new(persona))
            .with_config(merged_config(self.engine_config, &spec.tunables));
        if spec.budget.tokens.is_some() || spec.budget.wall_ms.is_some() {
            profile = profile.with_budget(Budget {
                tokens: spec.budget.tokens,
                wall_ms: spec.budget.wall_ms,
            });
        }
        if let Some(builder) = &self.context_builder {
            profile = profile.with_context_engine_builder(builder.clone());
        } else if let Some(context) = &self.context {
            profile = profile.with_context_engine(context.clone());
        }
        if let Some(builder) = &self.memory_builder {
            profile = profile.with_memory_builder(builder.clone());
        } else if !self.memory.is_empty() {
            profile = profile.with_memory(self.memory.clone());
        }
        for source in &self.prompt_sources {
            profile = profile.with_prompt_block(source.clone());
        }
        if let Some(credentials) = &self.credentials {
            profile =
                profile.with_credentials(credentials.clone(), ProfileRef::new(spec.credential_profile()));
            // A configured fallback credential profile composes a failover chain on top of the
            // per-profile multi-key pool: the engine re-keys to it when the primary is exhausted.
            if let Some(fallback) = spec.fallback_credential_profile() {
                profile = profile.with_fallback_profile(ProfileRef::new(fallback));
            }
        }
        profile
    }
}

/// Resolve a provider builder for `name`, falling back to the registry default.
fn provider_for(providers: &ProviderRegistry, name: &str) -> daemon_core::ProviderBuilder {
    providers
        .builder_for(&ProfileRef::new(name))
        .unwrap_or_else(|| panic!("no provider registered for {name:?} and no default set"))
}

/// A registry seeded with the core local toolset (fs + shell) every daemon-core engine carries, so a
/// leaf or session can do real work in its contained workspace (§12/§13), plus any node-level
/// `extra` tools (e.g. `mnemosyne_*` / `lcm_*`). Callers add role tools (e.g. orchestrate) on top.
fn core_tool_registry(extra: &[Arc<dyn Tool>]) -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(daemon_tool_fs::FsTool::new()));
    registry.register(Arc::new(daemon_tool_shell::ShellTool::new()));
    for tool in extra {
        registry.register(tool.clone());
    }
    registry
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

    // The fleet child: one shared profile, driven as the real job worker so every child gets the same
    // provider + brokered credentials. Each child journals into the shared store keyed by its UnitId.
    // Autonomous durable engines (the orchestrator, every delegated child, the fleet job worker)
    // run headless with no operator to answer an edit-approval ask, so they must never gate on a
    // human (an `Ask` would suspend the turn forever). Force `AutoAllow` for these roles; the
    // *interactive* session path keeps the operator-selectable base policy (default `Ask`).
    let autonomous_config = Config {
        approval_policy: ApprovalPolicy::AutoAllow,
        ..a.engine_config
    };
    let child_profile = dress(
        EngineProfile::new(
            provider_for(&a.providers, CHILD_PROFILE),
            Arc::new(core_tool_registry(&a.extra_tools)),
            SystemPrompt::new("fleet child"),
        ),
        &a,
    )
    .with_config(autonomous_config);
    // The legacy synchronous placement seam (in-process live engine children + foreign agents). The
    // durable Core delegation path no longer uses this — it materializes children as durable
    // sessions through the shared activation manager (see `FleetJobWorker`) — so this spawner is
    // retained only for the foreign/ephemeral coarse lifecycle and the live management escalation.
    let spawner: Arc<dyn ChildSpawner> =
        Arc::new(ProfileChildSpawner::core(child_profile).with_journal(journal.clone()));
    let fleet = FleetRuntime::new(
        a.store.clone(),
        a.partition,
        spawner,
        Arc::new(DefaultAnswerPolicy),
        None::<Arc<dyn ManageRequestHandler>>,
    );

    // The one orchestrator-capable engine shape, used at *every* durable level: the top session and
    // every delegated child are built from this profile, so a child is itself an orchestrator that
    // can delegate (the recursive durable graph). The orchestrate tool's depth guard (cap =
    // `nesting_depth + 1`) terminates the chain: `nesting_depth = 0` is a single delegation level
    // (top -> leaf child), `n` allows `n + 1` levels of nested delegation.
    // The orchestrator-capable engine carries the core local toolset (fs + shell) *plus* orchestrate,
    // so a node can both do real local work and delegate.
    let mut registry = core_tool_registry(&a.extra_tools);
    registry.register(Arc::new(
        daemon_tool_orchestrate::OrchestrateTool::new(fleet.clone())
            .with_max_depth(a.nesting_depth + 1),
    ));
    let orchestrator_profile = dress(
        EngineProfile::new(
            provider_for(&a.providers, ORCHESTRATOR_PROFILE),
            Arc::new(registry),
            SystemPrompt::new("daemon host node"),
        ),
        &a,
    )
    .with_config(autonomous_config);
    // The §4.3 background-review spawner: shared by the durable factory (so a review child raised
    // mid-turn resolves its constrained profile during hydrate) and the live surface (so a `Spawn`
    // host request from an interactive session is materialized fire-and-forget). Inert when the
    // registry is empty (no skills/memory tools) — `Effect::Spawn` then no-ops.
    let background = Arc::new(BackgroundSpawner::new(
        a.store.clone(),
        a.partition,
        background_registry(&a),
    ));

    // The durable path journals too: replace the discarding sink with one sealing per turn into the
    // shared store, keyed by the durable `SessionId`.
    let factory = CoreEngineFactory::from_profile(orchestrator_profile.clone())
        .with_journal(a.store.clone(), signer.clone())
        .with_background(background.clone());

    // One durable job worker for the whole node: every delegation (top or nested) materializes a
    // parent-bound durable child session seeded from the same orchestrator profile.
    let host =
        Host::new(a.store.clone(), Arc::new(factory), a.host_config).with_job_worker(Arc::new(
            FleetJobWorker::new(a.store.clone(), a.partition, orchestrator_profile),
        ));
    let handle = host.start();

    // The interactive (session sub-surface) engines: built from the same seam (resolved provider +
    // brokered credentials), so the live path is not credential-asymmetric with the durable one.
    let session_profile = dress(
        EngineProfile::new(
            provider_for(&a.providers, a.profile.as_str()),
            Arc::new(core_tool_registry(&a.extra_tools)),
            SystemPrompt::new("interactive session"),
        ),
        &a,
    );
    // Profile-aware interactive session builder: when the node carries a profile store + provider
    // resolver, each session resolves the *active* profile bundle at open and materializes its
    // engine (provider/persona/tools/budget/credentials) from it, so a GUI can switch model/provider
    // live. Otherwise sessions are built from the single fixed `session_profile` (legacy behavior).
    let session_builder: SessionEngineBuilder =
        match (a.profiles.clone(), a.provider_resolver.clone()) {
            (Some(store), Some(resolver)) => {
                let ctx = SessionFactoryCtx {
                    resolver,
                    extra_tools: a.extra_tools.clone(),
                    engine_config: a.engine_config,
                    credentials: a.credentials.clone(),
                    context: a.context.clone(),
                    context_builder: a.context_builder.clone(),
                    memory: a.memory.clone(),
                    memory_builder: a.memory_builder.clone(),
                    prompt_sources: a.prompt_sources.clone(),
                };
                let fallback = session_profile;
                Arc::new(move |id: SessionId| {
                    let spec = store
                        .active()
                        .ok()
                        .flatten()
                        .and_then(|active| store.get(&active).ok().flatten());
                    match spec {
                        Some(spec) => ctx.profile_from_spec(&spec).fresh(id),
                        None => fallback.fresh(id),
                    }
                })
            }
            _ => {
                let profile = session_profile;
                Arc::new(move |id: SessionId| profile.fresh(id))
            }
        };

    let mut node_api = NodeApiImpl::new(
        handle.observer(),
        a.store.clone(),
        host.manager().clone(),
        a.partition,
        session_builder,
        Some(Arc::new(FleetViewImpl::new(a.store.clone(), fleet.clone())) as Arc<dyn FleetControl>),
    )
    // Live interactive sessions journal per turn; also records the signer so history reads verify.
    .with_journal(a.store.clone(), signer.clone())
    // Surface the resident telemetry aggregator through the `telemetry` control op.
    .with_metrics(host.metrics().clone());
    // Bind the model-management sub-surface when this node hosts local-inference model management.
    if let Some(models) = a.models.clone() {
        node_api = node_api.with_models(models, a.profile.as_str().to_string());
    }
    // Bind the profile/config sub-surface when this node hosts profile management.
    if let Some(profiles) = a.profiles.clone() {
        node_api = node_api.with_profiles(profiles);
    }
    // Bind the credential sub-surface when this node hosts credential management.
    if let Some(credentials) = a.credential_store.clone() {
        node_api = node_api.with_credential_store(credentials);
    }
    // Bind the live cloud-model discovery hook when the binary provided one.
    if let Some(cloud_catalog) = a.cloud_catalog.clone() {
        node_api = node_api.with_cloud_catalog(cloud_catalog);
    }
    // Bind the live model-switch factory when this node resolves per-session profiles: a
    // `SetSessionModel` rebuilds a running session's provider for the new model id from the
    // (model-overridden) profile bundle via the same provider resolver.
    if let Some(resolver) = a.provider_resolver.clone() {
        let factory: ModelProviderFactory = Arc::new(move |spec| (resolver)(spec)());
        node_api = node_api.with_model_factory(factory);
    }
    // Bind the background-review spawner so live sessions materialize `Spawn` requests fire-and-forget.
    node_api = node_api.with_background(background.clone());
    let node = Arc::new(node_api);

    AssembledNode {
        node,
        handle,
        fleet,
        signer,
    }
}

// ---------------------------------------------------------------------------
// Composition-layer glue (moved here from `bins/daemon` so the binary and the
// conformance harness share one implementation).
// ---------------------------------------------------------------------------

/// A foreign agent launch profile: how to start a non-`daemon-core` brain that speaks §17 over a
/// process cut (mirrors [`daemon_provision::PlacementSpec`]). The reference brain needs none of this;
/// it is the home for "manage the foreign process's environment" the way `EngineProfile` is for ours.
pub struct LaunchProfile {
    /// The program to exec.
    pub program: PathBuf,
    /// Its CLI arguments.
    pub args: Vec<String>,
    /// Environment overrides applied to the child.
    pub env: Vec<(String, String)>,
    /// Which foreign wire protocol the agent speaks (selects the transport + codec / adapter).
    pub protocol: ForeignProtocol,
}

/// The wire protocol a foreign agent speaks — the selector that decides which transport + codec (or
/// out-of-tree adapter) materializes the child. All three present up the tree as a
/// `UnitKind::Engine` `ManagedUnit` and journal identically; only the bytes on the cut differ.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ForeignProtocol {
    /// The native `daemon` cut: CBOR §17 frames over the length-framed transport (our own placed
    /// `daemon-core` children, or any brain that speaks the native dialect).
    #[default]
    NativeCut,
    /// Claude-Code `stream-json`: NDJSON event envelope over the line transport (also Amp, Cursor).
    StreamJson,
    /// Agent Client Protocol: symmetric JSON-RPC 2.0 over stdio, via the `daemon-acp` adapter.
    Acp,
}

/// How to construct a child brain. `Core` is the in-process reference engine; `Foreign` launches an
/// external agent process. Both are presented up the tree as a `UnitKind::Engine` `ManagedUnit`, so
/// the fleet/orchestrator (and the GUI above it) cannot tell them apart.
pub enum AgentBackend {
    /// The in-process reference engine, built from a shared [`EngineProfile`].
    Core(EngineProfile),
    /// An external agent process launched from a [`LaunchProfile`].
    Foreign(LaunchProfile),
}

/// The profile-driven placement seam: materialize each child as the configured [`AgentBackend`],
/// uniformly presented as a `ManagedUnit`.
pub struct ProfileChildSpawner {
    backend: AgentBackend,
    provisioner: Arc<dyn Provisioner>,
    /// The verifiable-journal store + signer; when set, each spawned child journals its transcript
    /// (finished blocks + lifecycle) sealed per turn into the shared store, keyed by its `UnitId`.
    journal: Option<JournalConfig>,
}

impl ProfileChildSpawner {
    /// A spawner that materializes children from the in-process reference engine profile.
    pub fn core(profile: EngineProfile) -> Self {
        Self {
            backend: AgentBackend::Core(profile),
            provisioner: Arc::new(ProcessProvisioner::new()),
            journal: None,
        }
    }

    /// A spawner that materializes children by launching a foreign agent process.
    pub fn foreign(launch: LaunchProfile) -> Self {
        Self {
            backend: AgentBackend::Foreign(launch),
            provisioner: Arc::new(ProcessProvisioner::new()),
            journal: None,
        }
    }

    /// Journal every spawned child into the unified verifiable journal (keyed by `UnitId`).
    pub fn with_journal(mut self, journal: JournalConfig) -> Self {
        self.journal = Some(journal);
        self
    }

    /// Build a per-child journal feeder keyed by `id`, when journaling is configured.
    fn feeder(&self, id: &UnitId) -> Option<Arc<JournalFeeder>> {
        self.journal.as_ref().map(|cfg| {
            let sink = JournalSink::new(
                cfg.store.clone(),
                cfg.signer.clone(),
                JournalStreamId::unit(id),
            );
            Arc::new(JournalFeeder::new(Arc::new(sink)))
        })
    }
}

#[async_trait]
impl ChildSpawner for ProfileChildSpawner {
    async fn spawn(&self, id: UnitId, _spec: &DelegationSpec) -> Arc<dyn ManagedUnit> {
        let feeder = self.feeder(&id);
        match &self.backend {
            AgentBackend::Core(profile) => {
                let engine = profile.fresh(SessionId::new(id.as_str()));
                Arc::new(EngineUnit::spawn_journaled(id, engine, feeder))
            }
            AgentBackend::Foreign(launch) => {
                let session = SessionId::new(id.as_str());
                let spec = PlacementSpec {
                    program: launch.program.clone(),
                    args: launch.args.clone(),
                    env: launch.env.clone(),
                };
                match launch.protocol {
                    ForeignProtocol::NativeCut => {
                        let placement = self
                            .provisioner
                            .place(&session, spec)
                            .await
                            .expect("place native-cut foreign agent");
                        Arc::new(ProcessAgentUnit::start_journaled(id, placement, feeder))
                    }
                    ForeignProtocol::StreamJson => {
                        // NDJSON over the line transport, driven by the generic codec session driver.
                        let placement = self
                            .provisioner
                            .place_lines(&session, spec)
                            .await
                            .expect("place stream-json foreign agent");
                        let daemon_provision::Placement { channel, child } = placement;
                        Arc::new(AgentUnit::start_journaled(
                            id,
                            feeder,
                            move |host: Arc<dyn HostRequestHandler>| {
                                Arc::new(CodecSession::from_channel(
                                    channel,
                                    Some(child),
                                    host,
                                    StreamJsonCodec::new(),
                                )) as Arc<dyn AgentSession>
                            },
                        ))
                    }
                    ForeignProtocol::Acp => {
                        // The ACP adapter owns its own subprocess + stdio (it does not use the cut).
                        let acp = daemon_acp::AcpLaunch::new(launch.program.clone())
                            .args(launch.args.clone())
                            .env(launch.env.clone());
                        Arc::new(daemon_acp::acp_unit(id, acp, feeder))
                    }
                }
            }
        }
    }
}

/// Drives the durable job outbox by materializing each delegation as a *durable child session*:
/// seed a fresh orchestrator-capable engine snapshot with the delegated work, create the child row,
/// bind it to the parent's job (so its terminal completion wakes the parent — store-parent-link),
/// and enqueue a wake. The one shared [`daemon_activation::ActivationManager`] then drives the child
/// through the same `CoreIncarnation` path as the top session; if the child itself delegates it
/// suspends and enqueues its own job (parent = child), so nesting is recursive and crash-recoverable
/// at every depth. The legacy synchronous `FleetRuntime::spawn_and_run` is retained only for the
/// foreign/ephemeral coarse lifecycle, not this path.
pub struct FleetJobWorker {
    store: Arc<dyn daemon_store::SessionStore>,
    partition: PartitionId,
    /// The orchestrator-capable profile every durable session (top and child) is built from — one
    /// engine shape at every level. Used here to seed a fresh child's first turn.
    profile: EngineProfile,
}

impl FleetJobWorker {
    /// A durable job worker that seeds children from `profile` into `store` under `partition`.
    pub fn new(
        store: Arc<dyn daemon_store::SessionStore>,
        partition: PartitionId,
        profile: EngineProfile,
    ) -> Self {
        Self {
            store,
            partition,
            profile,
        }
    }

    /// The deterministic id of the child session a delegation job materializes: the parent's id plus
    /// a `/c{epoch}` path segment. Deterministic so a re-enqueued/recovered job dedupes onto the same
    /// child, and the `/`-delimited path encodes the tree depth the orchestrate-tool guard reads.
    fn child_id(job: &daemon_store::JobCommand) -> SessionId {
        SessionId::new(format!("{}/c{}", job.session_id, job.epoch.0))
    }
}

#[async_trait]
impl JobWorker for FleetJobWorker {
    async fn process_jobs_once(&self) -> Result<(), ServiceError> {
        while let Some(job) = self.store.dequeue_job().await {
            let child = Self::child_id(&job);
            // Create-if-absent: a fresh durable child session seeded with the delegated work as its
            // first turn (recovery-idempotent — a re-processed job finds the child already present).
            if self.store.status(&child).await.is_none() {
                let work = String::from_utf8_lossy(&job.payload).into_owned();
                let mut engine = self.profile.fresh(child.clone());
                engine.push_user(daemon_protocol::UserMsg::new(work));
                let blob = engine.snapshot().encode().map_err(ServiceError::new)?;
                self.store
                    .create_session(child.clone(), self.partition, blob)
                    .await
                    .map_err(ServiceError::new)?;
            }
            // Durable tree edge: the child's terminal completion fulfills this job and wakes the
            // parent (in the store's mark_completed transaction). Idempotent.
            self.store
                .bind_delegation(child.clone(), job.clone())
                .await
                .map_err(ServiceError::new)?;
            // Kick the child into its first turn via the shared wake dispatcher.
            self.store.enqueue_wake(child).await;
        }
        Ok(())
    }
}

/// Projects the management tree for the node control surface directly from the **durable session
/// graph** (the GUI/TUI's real surface). Structure (parent->children), state, per-node work label,
/// and folded usage are all re-sourced from the store — so the tree is recovery-survivable and
/// shows every durable session (top, child, grandchild, ...) at its true depth, addressable by id.
/// The legacy in-memory `FleetRuntime` projection is retained only for the synchronous foreign path;
/// `cancel` still routes through it.
pub struct FleetViewImpl {
    store: Arc<dyn daemon_store::SessionStore>,
    fleet: FleetRuntime,
}

impl FleetViewImpl {
    /// A control-surface projection over the durable `store`, with `fleet` for cancel routing.
    pub fn new(store: Arc<dyn daemon_store::SessionStore>, fleet: FleetRuntime) -> Self {
        Self { store, fleet }
    }

    /// Build the tree node for one durable session from its status + durable child edge.
    async fn node_for(
        &self,
        session: &SessionId,
        status: &daemon_store::SessionStatus,
        children: &[SessionId],
    ) -> daemon_api::UnitNode {
        use daemon_store::SessionStatus;
        // A node is an orchestrator iff it actually delegated (has durable children), else a leaf.
        let kind = if children.is_empty() {
            daemon_api::UnitKind::Engine
        } else {
            daemon_api::UnitKind::Orchestrator
        };
        let state = match status {
            SessionStatus::Completed => daemon_api::UnitState::Finished {
                end_reason: "Completed".to_string(),
            },
            _ => daemon_api::UnitState::Running,
        };
        daemon_api::UnitNode {
            id: UnitId::new(session.as_str()),
            kind,
            state,
            work: self.store.delegation_work(session).await,
            usage: self.store.usage_of(session).await,
            children: children.iter().map(|c| UnitId::new(c.as_str())).collect(),
        }
    }
}

#[async_trait]
impl FleetControl for FleetViewImpl {
    async fn report(&self) -> FleetReport {
        let mut usage = daemon_common::UsageDelta::default();
        let mut children = Vec::new();
        for (session, _) in self.store.list_sessions().await {
            usage.add(&self.store.usage_of(&session).await);
            children.push(UnitId::new(session.as_str()));
        }
        FleetReport { children, usage }
    }

    async fn cancel(&self, child: &UnitId) -> bool {
        self.fleet.cancel_child(child).await
    }

    async fn tree(&self) -> daemon_api::TreeReport {
        let sessions = self.store.list_sessions().await;
        let mut nodes = Vec::with_capacity(sessions.len());
        let mut is_child = std::collections::HashSet::new();
        for (session, status) in &sessions {
            let children = self.store.children_of(session).await;
            for c in &children {
                is_child.insert(c.clone());
            }
            nodes.push(self.node_for(session, status, &children).await);
        }
        // The root is the single top (parentless) session, if there is exactly one; otherwise the
        // node holds a forest and `root` is left unset (the nodes still carry the full structure).
        let roots: Vec<&SessionId> = sessions
            .iter()
            .map(|(s, _)| s)
            .filter(|s| !is_child.contains(*s))
            .collect();
        let root = match roots.as_slice() {
            [only] => Some(UnitId::new(only.as_str())),
            _ => None,
        };
        daemon_api::TreeReport { root, nodes }
    }

    async fn unit(&self, id: &UnitId) -> Option<daemon_api::UnitNode> {
        let session = SessionId::new(id.as_str());
        let status = self.store.status(&session).await?;
        let children = self.store.children_of(&session).await;
        Some(self.node_for(&session, &status, &children).await)
    }

    async fn unit_events(&self, id: &UnitId, max: u32) -> Vec<daemon_api::ManageEventView> {
        use daemon_store::SessionStatus;
        let session = SessionId::new(id.as_str());
        // Coarse lifecycle views synthesized from the durable status (the rich, byte-faithful
        // transcript is the verifiable journal, read via `unit_history`). A durable session has at
        // least Started; a terminal one also has Finished.
        let Some(status) = self.store.status(&session).await else {
            return Vec::new();
        };
        let mut views = vec![daemon_api::ManageEventView::Started { seq: 0 }];
        if matches!(status, SessionStatus::Completed) {
            views.push(daemon_api::ManageEventView::Finished {
                seq: 1,
                end_reason: "Completed".to_string(),
                summary: None,
            });
        }
        if max != 0 && (max as usize) < views.len() {
            let skip = views.len() - max as usize;
            views.drain(0..skip);
        }
        views
    }

    async fn unit_outbound(&self, _id: &UnitId, _max: u32) -> Vec<daemon_api::Outbound> {
        // Durable sessions retain no live §17 stream; their transcript is the durable journal.
        Vec::new()
    }

    async fn pause(&self, _id: &UnitId) -> bool {
        // Vestigial on the durable path: a durable session has no live scheduling to pause.
        false
    }

    async fn resume(&self, _id: &UnitId) -> bool {
        false
    }

    async fn scale(&self, _id: &UnitId, _n: u32) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    //! Composition smoke test for the wired-in defaults: an [`EngineProfile`] dressed with the LCM
    //! context engine + the Mnemosyne memory provider (the same way [`dress`] wires them from
    //! [`NodeAssembly`]) runs one full turn end-to-end, exercising the §10/§11 seams against the real
    //! port implementations and the once-per-incarnation lifecycle hooks.

    use super::*;
    use daemon_common::SessionId;
    use daemon_context_lcm::{LcmConfig, LcmContextEngine};
    use daemon_core::{
        EventSink, MockProvider, Provider, ToolCall, ToolOutcome, TurnControl, TurnOutcome,
    };
    use daemon_mnemosyne::{MnemosyneConfig, MnemosyneProvider};
    use daemon_protocol::{
        HostRequest, HostRequestHandler, HostResponse, HostResponseBody, UserMsg,
    };
    use std::collections::HashMap;
    use std::sync::Mutex;

    struct NoopHost;

    #[async_trait]
    impl HostRequestHandler for NoopHost {
        async fn request(&self, req: HostRequest) -> HostResponse {
            HostResponse {
                request_id: req.request_id,
                body: HostResponseBody::Approved(true),
            }
        }
    }

    /// A shared per-session Mnemosyne bank cache (mirrors the binary's `MnemosyneBanks`): one
    /// provider per session over the same on-disk bank, shared by the memory builder and the tools so
    /// the §11 hook and the `mnemosyne_*` tools always hit the same per-session instance.
    struct TestBanks {
        dir: std::path::PathBuf,
        sessions: Mutex<HashMap<SessionId, Arc<MnemosyneProvider>>>,
    }

    impl TestBanks {
        fn new(dir: std::path::PathBuf) -> Self {
            Self {
                dir,
                sessions: Mutex::new(HashMap::new()),
            }
        }

        fn get_or_open(&self, session: &SessionId) -> Arc<MnemosyneProvider> {
            let mut sessions = self.sessions.lock().unwrap();
            if let Some(p) = sessions.get(session) {
                return p.clone();
            }
            let cfg = MnemosyneConfig {
                data_dir: self.dir.clone(),
                session_id: session.as_str().to_string(),
                ..MnemosyneConfig::default()
            };
            let p = Arc::new(MnemosyneProvider::open(cfg).expect("open mnemosyne bank"));
            sessions.insert(session.clone(), p.clone());
            p
        }
    }

    /// A §12 tool adapter resolving the calling session's provider from the shared cache by
    /// `cx.session_id` (mirrors the binary's `MemoryProviderTool`).
    struct MemTool {
        banks: Arc<TestBanks>,
        def: daemon_core::ToolDef,
    }

    #[async_trait]
    impl Tool for MemTool {
        fn name(&self) -> &str {
            &self.def.name
        }
        fn schema(&self) -> &str {
            &self.def.schema
        }
        async fn run(&self, call: &ToolCall, cx: &daemon_core::TurnCx<'_>) -> ToolOutcome {
            let args = serde_json::from_str(&call.args).unwrap_or(serde_json::Value::Null);
            let provider = self.banks.get_or_open(&cx.session_id);
            let out = provider.call_tool(&self.def.name, args).await;
            ToolOutcome::text(call.call_id.clone(), true, out)
        }
    }

    #[tokio::test]
    async fn lcm_and_mnemosyne_defaults_run_a_turn() {
        // A unique on-disk Mnemosyne bank so parallel tests do not collide.
        let dir = std::env::temp_dir().join(format!("daemon-node-smoke-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let banks = Arc::new(TestBanks::new(dir.clone()));

        // Register the mnemosyne_* tools (session-resolved through the shared cache), exactly as the
        // composition layer does. Tool defs are session-independent; read them from a probe instance.
        let mut registry = ToolRegistry::new();
        for def in banks.get_or_open(&SessionId::new("__probe__")).tools() {
            registry.register(Arc::new(MemTool {
                banks: banks.clone(),
                def,
            }) as Arc<dyn Tool>);
        }
        assert!(
            registry.get("mnemosyne_recall").is_some(),
            "memory tools registered"
        );

        // Per-session builders, exactly as [`dress`] wires them from [`NodeAssembly`]: LCM gets a
        // fresh instance per session; Mnemosyne resolves the session's bank from the shared cache.
        let context_builder: ContextEngineBuilder = Arc::new(|id: &SessionId| {
            let aux: Arc<dyn Provider> = Arc::new(MockProvider::completing("summary"));
            Arc::new(LcmContextEngine::open_for_session(LcmConfig::in_memory(), id, aux).expect("lcm"))
                as Arc<dyn ContextEngine>
        });
        let memory_builder: MemoryBuilder = {
            let banks = banks.clone();
            Arc::new(move |id: &SessionId| vec![banks.get_or_open(id) as Arc<dyn MemoryProvider>])
        };

        let profile = EngineProfile::new(
            Arc::new(|| Arc::new(MockProvider::completing("done")) as Arc<dyn Provider>),
            Arc::new(registry),
            SystemPrompt::new("smoke"),
        )
        .with_context_engine_builder(context_builder)
        .with_memory_builder(memory_builder);

        let mut engine = profile.fresh(SessionId::new("smoke"));
        engine.push_user(UserMsg::new("remember that the sky is blue today"));
        let outcome = engine
            .run_turn(&NoopHost, &EventSink::discarding(), &TurnControl::new())
            .await
            .expect("turn runs through the wired LCM + Mnemosyne defaults");
        assert!(matches!(outcome, TurnOutcome::Completed(_)));
        engine.end_session().await;

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn lcm_tools_dispatch_through_the_shared_engine() {
        // Mirror the binary's `LcmBanks`: the context builder and the `lcm_*` tools resolve the same
        // per-session engine, so a tool call observes that session's live state + durable store.
        let session = SessionId::new("lcm-tools");
        let aux: Arc<dyn Provider> = Arc::new(MockProvider::completing("summary"));
        let lcm = Arc::new(
            LcmContextEngine::open_for_session(LcmConfig::in_memory(), &session, aux).expect("lcm"),
        );

        // The advisory names and the §12 tool defs both cover the seven tools.
        assert_eq!(ContextEngine::tools(lcm.as_ref()).len(), 7);
        assert_eq!(lcm.tool_defs().len(), 7);

        // A §12 adapter (mirrors the binary's `LcmTool`) dispatches by name to the shared engine.
        struct LcmStatusTool {
            lcm: Arc<LcmContextEngine>,
        }
        #[async_trait]
        impl Tool for LcmStatusTool {
            fn name(&self) -> &str {
                "lcm_status"
            }
            fn schema(&self) -> &str {
                "{}"
            }
            async fn run(&self, call: &ToolCall, _cx: &daemon_core::TurnCx<'_>) -> ToolOutcome {
                let out = self.lcm.call_tool("lcm_status", serde_json::Value::Null).await;
                ToolOutcome::text(call.call_id.clone(), true, out)
            }
        }
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(LcmStatusTool { lcm: lcm.clone() }) as Arc<dyn Tool>);
        assert!(registry.get("lcm_status").is_some());

        // Calling through the engine returns well-formed status JSON for this session.
        let status: serde_json::Value =
            serde_json::from_str(&lcm.call_tool("lcm_status", serde_json::json!({})).await)
                .expect("lcm_status returns JSON");
        assert_eq!(status["session_id"], "lcm-tools");
        assert!(status["store"]["session_messages"].is_number());
    }
}
