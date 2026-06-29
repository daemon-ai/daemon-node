// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

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
use daemon_api::{
    ApprovalMode, CatchUpPolicy, ContextEngineSel, CronSpec, EngineTunables, FleetReport,
    ManageEventView, MemoryProviderSel, OverlapPolicy, ProfileSpec, SessionOverlay, SubagentPhase,
    TreeEvent, WorkspaceBinding,
};
use daemon_common::{Budget, JournalStreamId, PartitionId, ProfileRef, SessionId, UnitId};
use daemon_core::{
    ApprovalPolicy, Config, ContextEngine, ContextEngineBuilder, CredentialBuilder, EngineProfile,
    ExecutionEnvironment, LocalEnvironment, MemoryBuilder, MemoryProvider, ProviderBuilder,
    ProviderRegistry, StablePromptSource, SystemPrompt, Tool, ToolRegistry,
};
use daemon_host::{
    AgentSession, AgentUnit, BackgroundProfile, BackgroundProfileRegistry, BackgroundSpawner,
    BlobStore, CodecSession, CoreEngineFactory, CredentialStore, CronFiring, CronScheduler,
    DurableProfileResolver, EngineUnit, FileBlobStore, FleetControl, Host, HostConfig, JobWorker,
    JournalConfig, JournalFeeder, JournalSink, ModelProviderFactory, NodeApiImpl, NodeEventFeed,
    ProcessAgentUnit, ProfileStore, RoutingRegistry, ServiceError, SessionEngineBuilder,
    StreamJsonCodec, SupervisorHandle, WorkspaceFs, WorkspaceRoots,
};
use daemon_orchestration::{ChildSpawner, DefaultAnswerPolicy, FleetRuntime};
use daemon_protocol::HostRequestHandler;
use daemon_provision::{PlacementSpec, ProcessProvisioner, Provisioner};
use daemon_schedule::Schedule;
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

/// The per-agent skills trio resolved for one profile — the analogue of [`MemoryBuilder`] /
/// [`ContextEngineBuilder`] for the skills subsystem. Skills were the one engine subsystem still
/// built once at node startup over the launch agent's home and shared across every session; this
/// makes them *resolved per agent* like memory and context. Carries the model-facing `skill_*`
/// tools bound to that profile's [`SkillStore`](daemon_skills::SkillStore) plus the
/// progressive-disclosure index ([`SkillsPromptSource`](daemon_skills::SkillsPromptSource)).
pub struct ResolvedSkills {
    /// The `skill_*` tools bound to the resolved profile's store (registered subject to the
    /// session's `tool_allowlist`, like any other tool).
    pub tools: Vec<Arc<dyn Tool>>,
    /// The profile's progressive-disclosure index, folded into the stable system-prompt tier when
    /// the profile actually carries skills tools (hermes' `valid_tool_names` gate).
    pub index: Arc<dyn StablePromptSource>,
}

/// Resolve the per-profile skills trio for a routed/identity [`ProfileRef`]. The binary closes over
/// the node's [`SkillsProvider`](daemon_skills::SkillsProvider) and `daemon_tool_skill::skill_tools`
/// so `daemon-node` stays free of the concrete tool crate (mirrors [`ProviderResolver`]).
pub type SkillsResolver = Arc<dyn Fn(&ProfileRef) -> ResolvedSkills + Send + Sync>;

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
    /// The append-only revision log backing profile + skill versioning. `None` builds a node without
    /// versioning (the history/revert ops resolve to `ApiError::Unsupported`). When set, it is the
    /// same log the [`Self::skills`] store records through, so operator + agent edits share a history.
    pub revisions: Option<Arc<dyn daemon_common::RevisionLog>>,
    /// The per-profile skills provider backing the node's skill versioning, distribution, and
    /// curation surface. Resolves an `Arc<SkillStore>` per profile id (rooted at that agent's home),
    /// so skill ops act on the right agent's library. `None` builds a node without a skills subsystem.
    pub skills: Option<Arc<daemon_skills::SkillsProvider>>,
    /// The per-profile skills resolver wired into the engine path: each session's engine gets its
    /// own profile's `skill_*` tools + index (resolved against `spec.id`), rather than a node-global
    /// set baked at startup. `None` keeps skills out of the engine entirely. Pairs with [`Self::skills`].
    pub skills_resolver: Option<SkillsResolver>,
    /// The host routing registry (daemon-event-io-spec §5.9): maps an inbound `Origin` to the
    /// session + profile + delivery a routed submit (`SessionApi::submit_routed`) opens. `None` (the
    /// default) installs an empty registry — routed submits then derive the session with `PerThread`
    /// and run the node's active default profile (the legacy single-profile behavior).
    pub routing: Option<RoutingRegistry>,
    /// The §12 tool-checkpoint store: wired into every engine (records a workspace checkpoint before
    /// a mutating tool runs) and into the control surface (the `Checkpoint{List,Rewind}` ops).
    /// `None` builds a node without checkpointing (tests / read-only nodes).
    pub checkpoints: Option<Arc<dyn daemon_core::CheckpointStore>>,
    /// The interactive-auth factories backing the node's `AuthApi` sub-surface (the client-driven
    /// SSO/OAuth2 login seam, `daemon-interactive-auth-spec`). Each factory serves one transport/
    /// provider family (e.g. the Matrix SSO factory). Empty (the default) builds a node whose
    /// `auth_begin`/`auth_complete` resolve to `ApiError::Unsupported` and whose `auth_providers` is
    /// empty. Completion writes through the same credential + profile stores wired above.
    pub auth_factories: Vec<Arc<dyn daemon_host::AuthFlowFactory>>,
    /// The node's workspace root: the parent directory of per-session sandboxes and the
    /// `FsRootId::Workspace` browse root. When `Some`, every engine this node builds is rooted under
    /// it (`<workspace_root>/<session_id>` for an isolated session, or the operator-bound directory
    /// for a `Bound` session) instead of the per-session `$TMP/daemon-ws-*` sandbox, and the
    /// filesystem surface (`fs_*`) serves files from there. `None` keeps the temp-sandbox default
    /// and leaves the filesystem surface unbound (tests / nodes without a workspace).
    pub workspace_root: Option<PathBuf>,
    /// The node content store (blob CAS, daemon-content-transfer-spec.md) root. When `Some`, the
    /// `blob_*` ops + `fs_write_from_blob` are served from a `FileBlobStore` rooted here, and
    /// `fs_read` attaches a `BlobRef` to untruncated reads. `None` leaves the content surface
    /// unbound (the ops resolve to `ApiError::Unsupported`).
    pub blob_root: Option<PathBuf>,
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
fn dress(
    profile: EngineProfile,
    a: &NodeAssembly,
    skills_index: Option<&Arc<dyn StablePromptSource>>,
) -> EngineProfile {
    dress_with_credential(profile, a, a.profile.clone(), skills_index)
}

/// Like [`dress`] but binds credentials to an explicit profile ref (the per-session credential ref).
/// `skills_index` is the launch agent's progressive-disclosure index (the role engines run as the
/// launch agent), folded into the system prompt alongside the node's other stable prompt sources.
fn dress_with_credential(
    profile: EngineProfile,
    a: &NodeAssembly,
    cred_profile: ProfileRef,
    skills_index: Option<&Arc<dyn StablePromptSource>>,
) -> EngineProfile {
    let mut profile = profile
        .with_config(a.engine_config)
        // Scope §10/§11 subsystem stores to the node's launch profile (the legacy single-profile
        // home), so the durable/orchestrator/fixed-session engines share one bank as before.
        .with_profile_ref(a.profile.clone());
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
    if let Some(index) = skills_index {
        profile = profile.with_prompt_block(index.clone());
    }
    if let Some(checkpoints) = &a.checkpoints {
        profile = profile.with_checkpoints(checkpoints.clone());
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
fn background_registry(
    a: &NodeAssembly,
    skill_tools: &[Arc<dyn Tool>],
) -> BackgroundProfileRegistry {
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
    // The skills review child curates the launch agent's own skills, so it draws its constrained
    // toolset from the resolved per-profile skill tools (no longer node-global `extra_tools`); the
    // memory review child still draws the `mnemosyne_*` tools from `extra_tools`.
    let skill_pool: Vec<Arc<dyn Tool>> = skill_tools.to_vec();
    // A clean base carrying only the node's provider (orchestrator selection) + brokered credentials.
    let base = |pool: &[Arc<dyn Tool>],
                names: &[&str],
                prefix: Option<&str>,
                persona: &str|
     -> EngineProfile {
        let profile = EngineProfile::new(
            provider_for(&a.providers, ORCHESTRATOR_PROFILE),
            Arc::new(constrained_registry(pool, names, prefix)),
            SystemPrompt::new(persona),
        )
        .with_config(bg_config);
        match &a.credentials {
            Some(c) => profile.with_credentials(c.clone(), a.profile.clone()),
            None => profile,
        }
    };

    if skill_pool
        .iter()
        .any(|t| tool_matches(t, &SKILL_TOOL_NAMES, None))
    {
        registry = registry.with(
            "skill_review",
            BackgroundProfile::new(
                base(&skill_pool, &SKILL_TOOL_NAMES, None, "skill curator"),
                SKILL_REVIEW_PROMPT,
            ),
        );
    }
    if a.extra_tools
        .iter()
        .any(|t| tool_matches(t, &[], Some(MEMORY_TOOL_PREFIX)))
    {
        registry = registry.with(
            "memory_review",
            BackgroundProfile::new(
                base(
                    &a.extra_tools,
                    &[],
                    Some(MEMORY_TOOL_PREFIX),
                    "memory curator",
                ),
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
    skills_resolver: Option<SkillsResolver>,
    /// The node's workspace-root resolver (shared with the engine exec builder + the filesystem
    /// surface). `None` keeps engines on the temp-sandbox default.
    workspace_roots: Option<Arc<WorkspaceRoots>>,
}

impl SessionFactoryCtx {
    /// Materialize a per-session engine profile by resolving the **bound profile** overlaid with the
    /// session's **overlay** — the one resolution path shared by the live surface and the durable
    /// rehydration path. The overlay's model/provider/tool-allowlist are applied to the spec; its
    /// approval-mode override is baked into the engine config. Unset overlay fields fall through to
    /// the profile, so an empty overlay resolves straight from the profile bundle.
    fn resolve_effective(&self, base: &ProfileSpec, overlay: &SessionOverlay) -> EngineProfile {
        let mut spec = base.clone();
        overlay.apply_to(&mut spec);
        let spec = &spec;
        let provider = (self.resolver)(spec);
        let mut registry = session_tool_registry(&self.extra_tools, spec.tool_allowlist.as_deref());
        // Resolve this agent's own skills (store + tools + index) keyed on its id — the per-profile
        // analogue of the memory/context builders. The `skill_*` tools are registered subject to the
        // same `tool_allowlist` as any other tool, and the index is attached only when the profile
        // actually carries skills tools (hermes' `valid_tool_names` gate), so a profile that excludes
        // them gets no skills block.
        let skills_index = self.skills_resolver.as_ref().and_then(|resolve| {
            let resolved = resolve(&ProfileRef::new(&spec.id));
            let mut any = false;
            for tool in resolved.tools {
                let allowed = match spec.tool_allowlist.as_deref() {
                    Some(list) => list.iter().any(|n| n == tool.name()),
                    None => true,
                };
                if allowed {
                    registry.register(tool);
                    any = true;
                }
            }
            any.then_some(resolved.index)
        });
        let persona = if spec.system_prompt.trim().is_empty() {
            "interactive session".to_string()
        } else {
            spec.system_prompt.clone()
        };
        // The §20 tunables config, with the overlay's edit-approval override (if any) baked in so a
        // per-session mode switch is honored by both the live actor and a rehydrated durable engine.
        let mut config = merged_config(self.engine_config, &spec.tunables);
        if let Some(mode) = overlay.approval_mode {
            config.approval_policy = approval_mode_to_policy(mode);
        }
        let mut profile =
            EngineProfile::new(provider, Arc::new(registry), SystemPrompt::new(persona))
                .with_config(config)
                // Scope the §10/§11 subsystem stores to the profile's own id (its on-disk key), so two
                // rooms routed to two profiles get isolated context/memory banks under their own homes.
                .with_profile_ref(ProfileRef::new(&spec.id));
        if spec.budget.tokens.is_some() || spec.budget.wall_ms.is_some() {
            profile = profile.with_budget(Budget {
                tokens: spec.budget.tokens,
                wall_ms: spec.budget.wall_ms,
            });
        }
        // Honor the profile's §10 context-engine selector. `Budgeted` is the in-core
        // `BudgetedContextEngine` default (attach nothing); `Lcm` wires the node's LCM builder when
        // the node configured one.
        match spec.context_engine {
            ContextEngineSel::Lcm => {
                if let Some(builder) = &self.context_builder {
                    profile = profile.with_context_engine_builder(builder.clone());
                } else if let Some(context) = &self.context {
                    profile = profile.with_context_engine(context.clone());
                }
            }
            ContextEngineSel::Budgeted => {}
        }
        // Honor the profile's §11 memory-provider selector. `None` attaches no memory; `Mnemosyne`
        // wires the node's session-scoped builder (the default); `File` uses the node's shared
        // frozen `FileMemory` providers. When the node didn't wire the requested backend, fall back
        // to whatever memory it does carry (we currently ship one default each — LCM + Mnemosyne).
        match spec.memory_provider {
            MemoryProviderSel::Mnemosyne => {
                if let Some(builder) = &self.memory_builder {
                    profile = profile.with_memory_builder(builder.clone());
                } else if !self.memory.is_empty() {
                    profile = profile.with_memory(self.memory.clone());
                }
            }
            MemoryProviderSel::File => {
                if !self.memory.is_empty() {
                    profile = profile.with_memory(self.memory.clone());
                }
            }
            MemoryProviderSel::None => {}
        }
        for source in &self.prompt_sources {
            profile = profile.with_prompt_block(source.clone());
        }
        if let Some(index) = skills_index {
            profile = profile.with_prompt_block(index);
        }
        if let Some(credentials) = &self.credentials {
            profile = profile.with_credentials(
                credentials.clone(),
                ProfileRef::new(spec.credential_profile()),
            );
            // A configured fallback credential profile composes a failover chain on top of the
            // per-profile multi-key pool: the engine re-keys to it when the primary is exhausted.
            if let Some(fallback) = spec.fallback_credential_profile() {
                profile = profile.with_fallback_profile(ProfileRef::new(fallback));
            }
        }
        // Root the engine's execution environment (§13) at the session's workspace: the operator-bound
        // directory when the overlay carries `Bound(path)`, else the isolated `<workspace_root>/<id>`
        // sandbox. Record the resolved root so the filesystem surface (`fs_*`) serves the *same*
        // directory the agent's fs/shell tools operate in. No-op when no workspace root is configured.
        if let Some(roots) = &self.workspace_roots {
            let roots = roots.clone();
            let binding = overlay.workspace.clone();
            profile = profile.with_exec(Arc::new(move |id: &SessionId| {
                let root = match &binding {
                    Some(WorkspaceBinding::Bound(p)) => p.clone(),
                    _ => roots.isolated_root(id.as_str()),
                };
                roots.record(id.as_str(), root.clone());
                Arc::new(LocalEnvironment::new(root)) as Arc<dyn ExecutionEnvironment>
            }));
        }
        profile
    }
}

/// Map a wire-level [`ApprovalMode`] onto the engine's [`ApprovalPolicy`] (the §12 session mode),
/// for baking a session overlay's approval override into the resolved engine config.
fn approval_mode_to_policy(mode: ApprovalMode) -> ApprovalPolicy {
    match mode {
        ApprovalMode::Ask => ApprovalPolicy::Ask,
        ApprovalMode::AcceptEdits => ApprovalPolicy::AcceptEdits,
        ApprovalMode::AutoAllow => ApprovalPolicy::AutoAllow,
        ApprovalMode::Deny => ApprovalPolicy::Deny,
    }
}

/// Root a base profile's engines under the node `workspace_root` (isolated per-session sandbox),
/// recording the resolved root so the filesystem surface serves the same directory. No-op when no
/// workspace root is configured (engines then fall back to the per-session temp sandbox).
fn root_profile(profile: EngineProfile, roots: &Option<Arc<WorkspaceRoots>>) -> EngineProfile {
    match roots {
        Some(roots) => {
            let roots = roots.clone();
            profile.with_exec(Arc::new(move |id: &SessionId| {
                let root = roots.session_root(id.as_str());
                roots.record(id.as_str(), root.clone());
                Arc::new(LocalEnvironment::new(root)) as Arc<dyn ExecutionEnvironment>
            }))
        }
        None => profile,
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

/// Like [`core_tool_registry`] but additionally registers the launch agent's resolved `skill_*`
/// tools. The role engines (fleet child, orchestrator, fixed session) run as the launch agent, so
/// they carry that agent's per-profile skills rather than a node-global set.
fn core_tool_registry_with_skills(
    extra: &[Arc<dyn Tool>],
    skills: &[Arc<dyn Tool>],
) -> ToolRegistry {
    let mut registry = core_tool_registry(extra);
    for tool in skills {
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
    // The node's workspace-root resolver: shared by every engine's exec-env builder (so agents root
    // under it) and the filesystem surface (so operator + agent see one filesystem). `None` keeps
    // the per-session temp-sandbox default (tests / nodes without a workspace).
    let workspace_roots: Option<Arc<WorkspaceRoots>> = a.workspace_root.clone().map(|base| {
        // Host browse roots for discovery before binding (daemon-fs-surface-spec.md): default to the
        // node user's home directory. (An operator allowlist / recents can extend this later.)
        let mut browse = Vec::new();
        if let Some(home) = std::env::var_os("HOME") {
            browse.push(("home".to_string(), PathBuf::from(home)));
        }
        Arc::new(WorkspaceRoots::new(base).with_browse_roots(browse))
    });
    // The content store (blob CAS), shared by the durable job worker (materializing delegated
    // attachments) and the NodeApi `blob_*`/`fs_write_from_blob` surface. A failed open leaves it
    // unbound (those ops resolve to Unsupported; attachment transfer is a no-op).
    let blob_store: Option<Arc<dyn BlobStore>> = a.blob_root.as_ref().and_then(|root| {
        FileBlobStore::open(root.clone())
            .ok()
            .map(|s| Arc::new(s) as Arc<dyn BlobStore>)
    });
    let child_profile = root_profile(
        dress(
            EngineProfile::new(
                provider_for(&a.providers, CHILD_PROFILE),
                Arc::new(core_tool_registry_with_skills(
                    &a.extra_tools,
                    &launch_skill_tools,
                )),
                SystemPrompt::new("fleet child"),
            ),
            &a,
            launch_index,
        )
        .with_config(autonomous_config),
        &workspace_roots,
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
    // The host-owned fleet event bus (I4/I8): the single broadcast sender the orchestration
    // producers (this in-memory `FleetRuntime` + the durable `FleetJobWorker`) ping on a real
    // topology change, and `NodeApiImpl::tree_subscribe` subscribes to for live push. Capacity is a
    // burst cushion; a slow subscriber that lags re-syncs with a fresh snapshot.
    let (fleet_events, _) = tokio::sync::broadcast::channel::<TreeEvent>(256);
    // The node-wide event feed (L3 `EventsSince`): a retained, cursored ring of payload-free
    // notifications the client subscribes to so it learns out-of-focus changes without polling. The
    // ring depth bounds how far behind a reconnecting client can be before it gets `ResyncNeeded`.
    let node_events = NodeEventFeed::new(1024);
    let fleet = FleetRuntime::new(
        a.store.clone(),
        a.partition,
        spawner,
        Arc::new(DefaultAnswerPolicy),
        None::<Arc<dyn ManageRequestHandler>>,
    )
    .with_event_sink(fleet_events.clone());

    // The resident cron scheduler (I15) + its shared ops surface, built BEFORE the agent profiles so
    // the agent-facing `cron` tool can wrap the same `CronOps` the operator `cron_*` control ops use
    // (one job engine, not two). The worker seeds its isolated cron sessions from the *constrained*
    // `child_profile` shape (no `orchestrate`/`cron`) — and the durable factory re-hydrates every
    // cron-fired session under that same constrained `cron_profile` (G3), so a scheduled run can
    // never self-schedule or self-delegate. The same `Arc` backs the 5th supervised service, the
    // manual-fire seam (`cron_trigger` / the tool's "run now"), and the catch-up tick.
    let cron_run_profile = child_profile.clone();
    let mut cron_worker = CronWorker::new(a.store.clone(), a.partition, cron_run_profile.clone());
    if let Some(roots) = &workspace_roots {
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
    let mut cron_ops_builder = daemon_host::CronOps::new(a.store.clone())
        .with_firing(cron_worker.clone() as Arc<dyn CronFiring>);
    // The `metadata.daemon.blueprint` skill bridge: scan the launch profile's skills (cheaply, on
    // each suggestion seed) and offer any runnable blueprint as a consent-first cron suggestion.
    if let Some(skills) = &a.skills {
        let provider = skills.clone();
        let profile_id = a.profile.as_str().to_string();
        let source: daemon_host::BlueprintSource = Arc::new(move || {
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
    // The agent veneer over the cron ops; registered into the agent-facing profiles below (and into
    // the interactive `SessionFactoryCtx`), but deliberately NOT into `child_profile` /
    // `cron_run_profile`, so it is absent from cron-fired runs (defense in depth alongside the
    // tool's own in-cron-session refusal guard).
    let cron_tool = Arc::new(daemon_tool_cron::CronTool::new(cron_ops.clone())) as Arc<dyn Tool>;

    // The one orchestrator-capable engine shape, used at *every* durable level: the top session and
    // every delegated child are built from this profile, so a child is itself an orchestrator that
    // can delegate (the recursive durable graph). The orchestrate tool's depth guard (cap =
    // `nesting_depth + 1`) terminates the chain: `nesting_depth = 0` is a single delegation level
    // (top -> leaf child), `n` allows `n + 1` levels of nested delegation.
    // The orchestrator-capable engine carries the core local toolset (fs + shell) *plus* orchestrate
    // *plus* the `cron` scheduling tool, so a node can do real local work, delegate, and schedule.
    let mut registry = core_tool_registry_with_skills(&a.extra_tools, &launch_skill_tools);
    registry.register(Arc::new(
        daemon_tool_orchestrate::OrchestrateTool::new(fleet.clone())
            .with_max_depth(a.nesting_depth + 1),
    ));
    registry.register(cron_tool.clone());
    let orchestrator_profile = root_profile(
        dress(
            EngineProfile::new(
                provider_for(&a.providers, ORCHESTRATOR_PROFILE),
                Arc::new(registry),
                SystemPrompt::new("daemon host node"),
            ),
            &a,
            launch_index,
        )
        .with_config(autonomous_config),
        &workspace_roots,
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

    // The one per-session resolution context, shared by the live session builder and the durable
    // rehydration resolver so both paths resolve a session's engine identically (bound profile +
    // overlay). Present only when the node carries a profile store + provider resolver; otherwise
    // sessions fall back to the single fixed `session_profile` (legacy single-profile behavior).
    let session_ctx: Option<(Arc<dyn ProfileStore>, Arc<SessionFactoryCtx>)> =
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
                    workspace_roots: workspace_roots.clone(),
                });
                Some((store, ctx))
            }
            _ => None,
        };

    // The durable path journals too: replace the discarding sink with one sealing per turn into the
    // shared store, keyed by the durable `SessionId`. When per-session resolution is available, the
    // factory also re-resolves a durable session's engine from its recorded bound profile + overlay
    // on rehydration (the unified resolution path), instead of always using the orchestrator profile.
    let mut factory = CoreEngineFactory::from_profile(orchestrator_profile.clone())
        .with_journal(a.store.clone(), signer.clone())
        .with_background(background.clone())
        // I15/G3: a cron-fired session (`session_meta.scheduled_job`) hydrates under the constrained,
        // `cron`/`orchestrate`-free profile so a scheduled run cannot self-schedule or self-delegate.
        .with_cron_profile(cron_run_profile.clone());
    if let Some((store, ctx)) = &session_ctx {
        let store = store.clone();
        let ctx = ctx.clone();
        // Re-resolve a durable session from its bound profile + overlay; no recorded binding (e.g. a
        // delegated orchestrator child) yields `None`, so the factory keeps its orchestrator profile.
        let resolver: DurableProfileResolver =
            Arc::new(move |bound: Option<ProfileRef>, overlay: &SessionOverlay| {
                let bound = bound?;
                let spec = store.get(bound.as_str()).ok().flatten()?;
                Some(ctx.resolve_effective(&spec, overlay))
            });
        factory = factory.with_session_resolver(resolver);
    }
    // Give durable incarnations the content store + workspace roots so a completed child captures
    // its outbox/ into blobs and a waking parent materializes the returned artifacts into its inbox/.
    if let (Some(roots), Some(blobs)) = (&workspace_roots, &blob_store) {
        factory = factory.with_content(blobs.clone(), roots.clone());
    }

    // One durable job worker for the whole node: every delegation (top or nested) materializes a
    // parent-bound durable child session seeded from the same orchestrator profile.
    let mut job_worker = FleetJobWorker::new(a.store.clone(), a.partition, orchestrator_profile)
        .with_event_sink(fleet_events.clone());
    // Give the worker the workspace roots + content store so it can materialize delegated
    // attachments from the parent's workspace into the child's inbox/ (node-mediated).
    if let (Some(roots), Some(blobs)) = (&workspace_roots, &blob_store) {
        job_worker = job_worker.with_workspace(roots.clone(), blobs.clone());
    }
    // The resident cron scheduler (`cron_worker`, built above) drives the 5th supervised service.
    let host = Host::new(a.store.clone(), Arc::new(factory), a.host_config)
        .with_job_worker(Arc::new(job_worker))
        .with_cron_scheduler(cron_worker.clone() as Arc<dyn CronScheduler>);
    let handle = host.start();

    // The interactive (session sub-surface) engines: built from the same seam (resolved provider +
    // brokered credentials), so the live path is not credential-asymmetric with the durable one.
    // Carries the `cron` tool so a single-profile node's chatting agent can also schedule work.
    let mut session_registry = core_tool_registry_with_skills(&a.extra_tools, &launch_skill_tools);
    session_registry.register(cron_tool.clone());
    let session_profile = root_profile(
        dress(
            EngineProfile::new(
                provider_for(&a.providers, a.profile.as_str()),
                Arc::new(session_registry),
                SystemPrompt::new("interactive session"),
            ),
            &a,
            launch_index,
        ),
        &workspace_roots,
    );
    // Profile-aware interactive session builder: when the node carries a profile store + provider
    // resolver, each session resolves its bound profile bundle at open, applies the persisted
    // session overlay, and materializes its engine from the result (the same `resolve_effective` the
    // durable path uses). Otherwise sessions are built from the single fixed `session_profile`.
    let session_builder: SessionEngineBuilder = match &session_ctx {
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
                        Some(spec) => ctx.resolve_effective(&spec, overlay).fresh(id),
                        None => fallback.fresh(id),
                    }
                },
            )
        }
        None => {
            let profile = session_profile;
            Arc::new(
                move |id: SessionId, _requested: Option<ProfileRef>, _overlay: &SessionOverlay| {
                    profile.fresh(id)
                },
            )
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
    .with_metrics(host.metrics().clone())
    // Subscribe the tree-push surface to the host fleet bus (I4/I8): `tree_subscribe` now forwards
    // live spawn/terminal/progress deltas instead of re-projecting on a fixed poll interval.
    .with_fleet_events(fleet_events.clone())
    // The node-wide event feed (L3): `events_since` serves from this ring and the §5 emit hooks
    // push onto it.
    .with_node_events(node_events.clone());
    // L3 fleet liveness: bridge the fleet topology bus (`fleet_events`, consumed by `tree_subscribe`)
    // onto the node-wide feed as a coalesced `FleetChanged`, so `events_since` clients learn the
    // subagent tree changed (spawn / state / finish) and re-fetch `Tree` live - without threading the
    // feed through the orchestration crate (only `NodeApiImpl`/`LiveSessions` can reach it directly).
    // `FleetChanged` coalesces in the feed ring, so a spawn burst is one client refetch; a `Lagged`
    // (the bridge fell behind the bus) is itself just "the tree changed".
    {
        let feed = node_events.clone();
        let mut rx = fleet_events.subscribe();
        tokio::spawn(async move {
            use tokio::sync::broadcast::error::RecvError;
            // Loop until the bus closes; both a value and a `Lagged` mean "the tree changed".
            while let Ok(_) | Err(RecvError::Lagged(_)) = rx.recv().await {
                let rev = feed.note_fleet_change();
                feed.emit(daemon_api::NodeEvent::FleetChanged { rev });
            }
        });
    }
    // Bind the filesystem / workspace surface (`fs_*`) over the SAME `WorkspaceRoots` the engine
    // exec builders root at, so operator and agent see one filesystem.
    if let Some(roots) = &workspace_roots {
        node_api = node_api.with_workspace(Arc::new(WorkspaceFs::new(roots.clone())));
    }
    // Bind the content store (blob CAS) surface, reusing the shared store built above.
    if let Some(blobs) = &blob_store {
        node_api = node_api.with_blobs(blobs.clone());
    }
    // Bind the cron operations surface (I15): the SAME shared `CronOps` (with the resident
    // `CronWorker` as its manual-fire handle) that backs the agent `cron` tool, so the operator
    // control ops and the agent tool create/trigger through one path.
    node_api = node_api.with_cron(cron_ops.clone());
    // Bind the model-management sub-surface when this node hosts local-inference model management.
    if let Some(models) = a.models.clone() {
        // L3: fan download progress onto the node-wide feed so the client renders it without the
        // 600ms poll. pct is derived from the byte counters; state mirrors the wire string.
        let feed = node_events.clone();
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
            });
        }));
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
    // Install the host routing registry (§5.9) so routed submits select the session's profile +
    // delivery from the inbound origin. The account→profile baseline (precedence step 2) is derived
    // here from every profile's `bound_accounts` (§5.9.4): profile-declared instance bindings fill
    // the registry's `instance_profiles`, while any explicit config `[[routing.instance_profile]]`
    // already present wins (operator override). A registry is installed when configured *or* when a
    // profile contributes a binding, even if the config route table was empty.
    {
        let profile_specs = a
            .profiles
            .as_ref()
            .and_then(|p| p.list().ok())
            .unwrap_or_default();
        let routing = match a.routing.clone() {
            Some(reg) => Some(reg.bind_instances_from_profiles(&profile_specs)),
            None => {
                let reg = RoutingRegistry::new().bind_instances_from_profiles(&profile_specs);
                (!reg.is_empty()).then_some(reg)
            }
        };
        if let Some(routing) = routing {
            node_api = node_api.with_routing(routing);
        }
    }
    // Transport-adapter registry seam (daemon-transport-adapter-spec.md §3.4): the declarative
    // companion to routing. `NodeApiImpl::with_adapters(AdapterRegistry::new().with_adapter(..))`
    // installs the node's self-describing events-IO adapters so `transport_adapters` enumerates them
    // for the GUI "Add channel" picker. No adapter implements `TransportAdapter` yet (the `serve`
    // spawns still live in `bins/daemon`), so the registry stays empty/inert here; populating it +
    // driving lifecycle from the registry is deferred (spec §7 P1).
    // Bind the live cloud-model discovery hook when the binary provided one.
    if let Some(cloud_catalog) = a.cloud_catalog.clone() {
        node_api = node_api.with_cloud_catalog(cloud_catalog);
    }
    // Wire the server-side ACP discovery hook (I7): the host's `acp_discover` op probes the curated
    // direct-binary recipe table via the ACP `initialize` handshake. The host cannot link the ACP
    // runtime directly (`daemon-acp` depends on `daemon-host`), so the discoverer is injected here.
    node_api = node_api.with_acp_discovery(Arc::new(daemon_acp::AcpDiscoverer::new()));
    // Bind the live model-switch factory when this node resolves per-session profiles: a
    // `SetSessionModel` rebuilds a running session's provider for the new model id from the
    // (model-overridden) profile bundle via the same provider resolver.
    if let Some(resolver) = a.provider_resolver.clone() {
        let factory: ModelProviderFactory = Arc::new(move |spec| (resolver)(spec)());
        node_api = node_api.with_model_factory(factory);
    }
    // Bind the background-review spawner so live sessions materialize `Spawn` requests fire-and-forget.
    node_api = node_api.with_background(background.clone());
    // Bind the §12 tool-checkpoint store so the `Checkpoint{List,Rewind}` ops see the same rewind
    // points the engines record.
    if let Some(checkpoints) = a.checkpoints.clone() {
        node_api = node_api.with_checkpoints(checkpoints);
    }
    let node = Arc::new(node_api);
    // Late-bind the cron post-settle delivery handle now that the `NodeApiImpl` exists: it implements
    // `CronDelivery` over its `DeliverySink` registry, so a finished cron run's `deliver` pushes
    // through the same outbound path live replies use.
    cron_worker.set_delivery(node.clone() as Arc<dyn daemon_host::CronDelivery>);

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
// Built once per child at placement time, not stored in bulk - the variant size delta is irrelevant
// here, and boxing would leak into this pub enum's construction/match sites for no real benefit.
#[allow(clippy::large_enum_variant)]
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
    /// The durable session store + §12 checkpoint store, threaded into a `Core` child's managed
    /// engine so a conversation rewind on it applies the same journal seal + workspace rollback the
    /// live-session path applies (conversation-rewind spec §6). `None` keeps the engine-only truncate.
    rewind_store: Option<Arc<dyn daemon_store::SessionStore>>,
    rewind_checkpoints: Option<Arc<dyn daemon_core::CheckpointStore>>,
}

impl ProfileChildSpawner {
    /// A spawner that materializes children from the in-process reference engine profile.
    pub fn core(profile: EngineProfile) -> Self {
        Self {
            backend: AgentBackend::Core(profile),
            provisioner: Arc::new(ProcessProvisioner::new()),
            journal: None,
            rewind_store: None,
            rewind_checkpoints: None,
        }
    }

    /// A spawner that materializes children by launching a foreign agent process.
    pub fn foreign(launch: LaunchProfile) -> Self {
        Self {
            backend: AgentBackend::Foreign(launch),
            provisioner: Arc::new(ProcessProvisioner::new()),
            journal: None,
            rewind_store: None,
            rewind_checkpoints: None,
        }
    }

    /// Thread the durable seal + workspace-rollback handles into the spawned `Core` children so a
    /// conversation rewind on a managed engine matches the live path (conversation-rewind spec §6).
    pub fn with_rewind(
        mut self,
        store: Arc<dyn daemon_store::SessionStore>,
        checkpoints: Option<Arc<dyn daemon_core::CheckpointStore>>,
    ) -> Self {
        self.rewind_store = Some(store);
        self.rewind_checkpoints = checkpoints;
        self
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
                let session = SessionId::new(id.as_str());
                let engine = profile.fresh(session.clone());
                // Thread the durable seal/rollback handles into the managed engine so a rewind on it
                // matches the live path (the §17⇄management seam fix); `None` store => engine-only.
                let rewind = self
                    .rewind_store
                    .clone()
                    .map(|store| daemon_host::RewindHooks {
                        store,
                        checkpoints: self.rewind_checkpoints.clone(),
                        journaled: feeder.is_some(),
                        session,
                    });
                Arc::new(EngineUnit::spawn_rewindable(id, engine, feeder, rewind))
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
    /// The host fleet event bus (I4/I8). On a real durable child create the worker pushes a
    /// [`TreeEvent::Subagent`] spawn marker so `tree_subscribe` shows the new subagent row promptly
    /// (before any poll interval). `None` => no live push from the durable delegation seam.
    events: Option<tokio::sync::broadcast::Sender<TreeEvent>>,
    /// Monotonic sequence for the spawn markers the worker emits onto the bus.
    bus_seq: std::sync::atomic::AtomicU64,
    /// Workspace roots for materializing delegated attachments (parent -> child inbox/). `None`
    /// disables attachment transfer.
    workspace_roots: Option<Arc<WorkspaceRoots>>,
    /// The content store used to put/fetch delegated attachment bytes. `None` disables transfer.
    blobs: Option<Arc<dyn BlobStore>>,
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
            events: None,
            bus_seq: std::sync::atomic::AtomicU64::new(0),
            workspace_roots: None,
            blobs: None,
        }
    }

    /// Give the worker the workspace roots + content store so it materializes a delegation's
    /// attachment paths (read from the parent's workspace, round-tripped through the content store)
    /// into the child's `inbox/` before the child's first turn. No-op transfer when unset.
    pub fn with_workspace(mut self, roots: Arc<WorkspaceRoots>, blobs: Arc<dyn BlobStore>) -> Self {
        self.workspace_roots = Some(roots);
        self.blobs = Some(blobs);
        self
    }

    /// Materialize a delegation's attachment paths from the parent workspace into the child's
    /// `inbox/`, round-tripping each through the content store (dedup + integrity; federation-ready).
    /// Best-effort: a missing/contained-rejected path or store error is skipped, never failing the
    /// job. No-op when no workspace/blob store is wired or there are no attachments.
    async fn materialize_attachments(
        &self,
        parent: &SessionId,
        child: &SessionId,
        paths: &[String],
    ) {
        let (Some(roots), Some(blobs)) = (&self.workspace_roots, &self.blobs) else {
            return;
        };
        if paths.is_empty() {
            return;
        }
        let parent_root = roots.session_root(parent.as_str());
        let inbox = roots.session_root(child.as_str()).join("inbox");
        if std::fs::create_dir_all(&inbox).is_err() {
            return;
        }
        for path in paths {
            let Ok(src) = daemon_core::exec::contain(&parent_root, std::path::Path::new(path))
            else {
                continue;
            };
            let Ok(bytes) = std::fs::read(&src) else {
                continue;
            };
            let Ok(blob_ref) = blobs.put(&bytes).await else {
                continue;
            };
            let Ok(out) = blobs.get(&blob_ref.hash, None).await else {
                continue;
            };
            let name = std::path::Path::new(path)
                .file_name()
                .unwrap_or_else(|| std::ffi::OsStr::new("attachment"));
            let _ = std::fs::write(inbox.join(name), out);
        }
    }

    /// Inject the host fleet event bus so a durable child create pushes a live spawn delta. Call
    /// during assembly with the same sender wired into `NodeApiImpl`/`FleetRuntime`.
    pub fn with_event_sink(mut self, events: tokio::sync::broadcast::Sender<TreeEvent>) -> Self {
        self.events = Some(events);
        self
    }

    /// Push the spawn marker for a freshly-created durable child onto the fleet bus (role from the
    /// job's `ChildLifetime`, active count = the parent's current durable child total). A no-op when
    /// no bus is wired.
    async fn emit_spawn(
        &self,
        parent: &SessionId,
        child: &SessionId,
        role: daemon_api::SessionRole,
    ) {
        let Some(events) = &self.events else {
            return;
        };
        let active_children = self.store.children_of(parent).await.len() as u32;
        let seq = self
            .bus_seq
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let _ = events.send(TreeEvent::Subagent(ManageEventView::Subagent {
            seq,
            child: UnitId::new(child.as_str()),
            role,
            phase: SubagentPhase::Spawned,
            active_children,
        }));
    }

    /// The deterministic id of the child session a delegation job materializes: the parent's id plus
    /// a `/c{epoch}` path segment. Deterministic so a re-enqueued/recovered job dedupes onto the same
    /// child, and the `/`-delimited path encodes the tree depth the orchestrate-tool guard reads.
    fn child_id(job: &daemon_store::JobCommand) -> SessionId {
        SessionId::new(format!("{}/c{}", job.session_id, job.epoch.0))
    }
}

/// Map a durable-store session role to its wire-surface equivalent (for the fleet bus markers).
fn map_store_role(role: daemon_store::SessionRole) -> daemon_api::SessionRole {
    match role {
        daemon_store::SessionRole::Primary => daemon_api::SessionRole::Primary,
        daemon_store::SessionRole::ManagedChild => daemon_api::SessionRole::ManagedChild,
        daemon_store::SessionRole::EphemeralSubagent => daemon_api::SessionRole::EphemeralSubagent,
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
                // Decode the structured delegation (task + attachment paths), falling back to a
                // legacy plain-text task for pre-upgrade jobs. Seed the child with the real task and
                // materialize any attachments into its inbox/ before the first turn.
                let input = daemon_protocol::DelegationInput::decode(&job.payload);
                self.materialize_attachments(&job.session_id, &child, &input.attachments)
                    .await;
                let mut engine = self.profile.fresh(child.clone());
                engine.push_user(daemon_protocol::UserMsg::new(input.task));
                let blob = engine.snapshot().encode().map_err(ServiceError::new)?;
                self.store
                    .create_session(child.clone(), self.partition, blob)
                    .await
                    .map_err(ServiceError::new)?;
                // Stamp the hierarchy edge so the child is excluded from the `TopLevel` roster and
                // reached only by walking the tree: it is a non-`Primary` child of the delegating
                // session. Read-modify-write preserves any bound profile/overlay; the role is
                // derived from the job's declared `ChildLifetime` (managed vs ephemeral subagent).
                let mut meta = self.store.session_meta(&child).await.unwrap_or_default();
                meta.parent = Some(job.session_id.clone());
                let child_role = job.lifetime.role();
                meta.role = Some(child_role);
                self.store
                    .set_session_meta(&child, meta)
                    .await
                    .map_err(ServiceError::new)?;
                // Real topology change: push the spawn delta so a live `tree_subscribe` shows the new
                // subagent row promptly (the conformance "push before poll" guarantee).
                self.emit_spawn(&job.session_id, &child, map_store_role(child_role))
                    .await;
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

/// The maximum lateness (seconds) a [`CatchUpPolicy::Skip`] job tolerates before a missed fire is
/// fast-forwarded instead of run. Small so Skip never catches up after real downtime, but generous
/// enough to absorb a slow tick / brief pause (`Grace` tolerates `>= MIN_GRACE_SECS`, `Always` is
/// unbounded — the three policies stay monotonically ordered).
const CRON_SKIP_TOLERANCE_SECS: u64 = 60;

/// Cap (bytes) on a single `context_from` chained-output injection, so a chatty upstream job cannot
/// blow up a downstream job's seed prompt.
const CRON_CONTEXT_CHARS: usize = 8192;

/// Cap (bytes) on a single preloaded skill body injected into a cron seed prompt (v16 `skills`), so
/// a large skill cannot blow up the prompt. Mirrors [`CRON_CONTEXT_CHARS`].
const CRON_SKILL_CHARS: usize = 8192;

/// The sentinel a cron agent run emits (as its entire final message) to suppress delivery — "nothing
/// worth reporting this tick" (ported from Hermes). A run whose captured output is exactly this is
/// recorded `ok` but is not delivered to any transport.
const CRON_SILENT_SENTINEL: &str = "[SILENT]";

/// Whether a captured cron run output is the `[SILENT]` delivery-suppression sentinel (trimmed).
fn is_cron_silent(text: &str) -> bool {
    text.trim() == CRON_SILENT_SENTINEL
}

/// Resolves a skill's full `SKILL.md` body by name for cron `skills` preloading. Injected into
/// [`CronWorker`] from the launch profile's [`SkillStore`](daemon_skills::SkillStore) in `assemble`;
/// `None` (no skills subsystem) makes `skills` preloading a no-op.
pub type CronSkillLoader = Arc<dyn Fn(&str) -> Option<String> + Send + Sync>;

/// Truncate `s` to at most `max` bytes without splitting a UTF-8 char (a plain `String::truncate`
/// panics on a non-boundary index). Used to cap cron seed-prompt injections.
fn cap_on_boundary(mut s: String, max: usize) -> String {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    s.truncate(end);
    s
}

/// The resident cron scheduler (I15): the `CronScheduler`/`CronFiring` worker. Mirrors
/// [`FleetJobWorker`] — it seeds isolated `cron_{id}_{ts}` sessions from a profile into the store and
/// enqueues their wake, leaving the existing wake-outbox dispatcher to run the turn. The scheduler
/// only computes next-fire (via `daemon-schedule`/croner) and enqueues; it never runs a turn itself.
///
/// Correctness: each due job's `next_fire` is advanced **first** (at-most-once across a crash), with
/// stale-miss fast-forward vs grace catch-up; overlap is deduped per `OverlapPolicy` (which also
/// closes the manual-trigger-vs-tick double-fire race); a `repeat`-exhausted job auto-deletes. Each
/// tick also reconciles in-flight runs (a settled cron session stamps its run `finished`).
pub struct CronWorker {
    store: Arc<dyn daemon_store::SessionStore>,
    partition: PartitionId,
    /// The seed engine shape for a cron session (the orchestrator-capable profile, minus the cron
    /// tool — see the cron-session safety gate in `assemble`). The durable factory re-resolves the
    /// bound profile (`spec.target`) from `session_meta` on wake, mirroring `FleetJobWorker`.
    profile: EngineProfile,
    /// Root for `no_agent` scripts; a job's `script` is contained under this dir. `None` disables
    /// the script path (a `no_agent` job then records an error run).
    scripts_dir: Option<PathBuf>,
    /// Resolves a `CronSpec::skills` name to its body for seed-prompt preloading (v16). `None`
    /// (no skills subsystem) skips preloading — the run still sees the launch agent's `skill_*`
    /// tools + index in its profile and can `skill_view` on demand.
    skill_loader: Option<CronSkillLoader>,
    /// The post-settle delivery handle (Phase 2 `deliver`): pushes a finished run's captured result
    /// to its `CronSpec::deliver` transport(s) through the host's existing `DeliverySink` registry.
    /// Late-bound (the handle is `NodeApiImpl`, built after the worker) via [`set_delivery`](Self::set_delivery);
    /// unset => store-only runs (no transport delivery).
    delivery: std::sync::OnceLock<Arc<dyn daemon_host::CronDelivery>>,
}

impl CronWorker {
    /// A cron worker that seeds sessions from `profile` into `store` under `partition`.
    pub fn new(
        store: Arc<dyn daemon_store::SessionStore>,
        partition: PartitionId,
        profile: EngineProfile,
    ) -> Self {
        Self {
            store,
            partition,
            profile,
            scripts_dir: None,
            skill_loader: None,
            delivery: std::sync::OnceLock::new(),
        }
    }

    /// Set the root directory `no_agent` job scripts are resolved (and contained) under.
    pub fn with_scripts_dir(mut self, dir: PathBuf) -> Self {
        self.scripts_dir = Some(dir);
        self
    }

    /// Set the loader used to preload `CronSpec::skills` bodies into an agent run's seed prompt.
    pub fn with_skill_loader(mut self, loader: CronSkillLoader) -> Self {
        self.skill_loader = Some(loader);
        self
    }

    /// Late-bind the post-settle delivery handle (the `NodeApiImpl`, built after the worker is
    /// `Arc`-wrapped). Idempotent: the first set wins; subsequent calls are ignored.
    pub fn set_delivery(&self, delivery: Arc<dyn daemon_host::CronDelivery>) {
        let _ = self.delivery.set(delivery);
    }

    /// Wall-clock now in unix seconds.
    fn now_unix() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }

    /// Decode the wire `CronSpec` from a stored job's opaque blob (falling back to a name-only spec
    /// from the column on a decode failure, so a corrupt row never wedges the whole tick).
    fn decode_spec(job: &daemon_store::StoredCronJob) -> CronSpec {
        daemon_api::from_cbor::<CronSpec>(&job.spec).unwrap_or_else(|_| CronSpec {
            schedule: job.schedule.clone(),
            ..CronSpec::default()
        })
    }

    /// Parse a [`CronSpec::provider`] string to the wire [`ProviderSelector`](daemon_api::ProviderSelector),
    /// accepting the canonical snake_case names plus the legacy adapter aliases that all collapse to
    /// `GenAi`. An unrecognized value yields `None` (inherit the profile's provider).
    fn parse_provider(s: &str) -> Option<daemon_api::ProviderSelector> {
        use daemon_api::ProviderSelector::*;
        match s.trim().to_lowercase().as_str() {
            "mock" => Some(Mock),
            "genai" | "openai" | "anthropic" | "gemini" | "groq" | "deep_seek" | "deepseek"
            | "xai" | "open_router" | "openrouter" | "cohere" => Some(GenAi),
            "llama_cpp" | "llamacpp" => Some(LlamaCpp),
            "mistral_rs" | "mistralrs" => Some(MistralRs),
            _ => None,
        }
    }

    /// Project the run-shaping fields of a [`CronSpec`] into a [`SessionOverlay`] (Phase 2): a
    /// per-job `model`/`provider`/`workdir`/`enabled_toolsets` override layered onto the cron base
    /// profile at hydrate. Unset fields inherit.
    fn overlay_from_spec(spec: &CronSpec) -> daemon_api::SessionOverlay {
        daemon_api::SessionOverlay {
            model: spec.model.clone(),
            provider: spec.provider.as_deref().and_then(Self::parse_provider),
            tool_allowlist: match &spec.enabled_toolsets {
                Some(list) => daemon_api::ToolsOverride::Allowlist(list.clone()),
                None => daemon_api::ToolsOverride::Inherit,
            },
            approval_mode: None,
            workspace: spec
                .workdir
                .as_deref()
                .map(|w| daemon_common::WorkspaceBinding::Bound(PathBuf::from(w))),
        }
    }

    /// Build the `daemon-schedule` `Schedule` from a spec (schedule string + tz + repeat + jitter).
    fn schedule_of(spec: &CronSpec) -> Result<Schedule, daemon_schedule::ScheduleError> {
        Schedule::parse(&spec.schedule)?
            .with_timezone(spec.timezone.as_deref())
            .map(|s| s.with_repeat(spec.repeat).with_jitter(spec.jitter_secs))
    }

    /// Whether a job's most recent run is still in flight: it has no `finished_unix` and its cron
    /// session is not yet settled (still `Active`/`Suspended`). Used for `OverlapPolicy` dedup.
    async fn in_flight(&self, job_id: &str) -> bool {
        let Some(run) = self
            .store
            .cron_runs_list(job_id, 1)
            .await
            .into_iter()
            .next()
        else {
            return false;
        };
        if run.finished_unix.is_some() {
            return false;
        }
        match &run.session {
            Some(session) => matches!(
                self.store.status(session).await,
                Some(daemon_store::SessionStatus::Active)
                    | Some(daemon_store::SessionStatus::Suspended { .. })
            ),
            None => false,
        }
    }

    /// Reconcile a job's latest in-flight run: if the cron session has settled (`Ready`/`Completed`/
    /// gone), stamp the run `finished` and fold the outcome into the job's `last_*` bookkeeping.
    async fn reconcile(&self, job: &daemon_store::StoredCronJob) {
        let Some(mut run) = self
            .store
            .cron_runs_list(&job.id, 1)
            .await
            .into_iter()
            .next()
        else {
            return;
        };
        if run.finished_unix.is_some() {
            return;
        }
        let Some(session) = run.session.clone() else {
            return;
        };
        let settled = !matches!(
            self.store.status(&session).await,
            Some(daemon_store::SessionStatus::Active)
                | Some(daemon_store::SessionStatus::Suspended { .. })
        );
        if !settled {
            return;
        }
        let now = Self::now_unix();
        run.finished_unix = Some(now);
        // Capture the run's real outcome: the cron session's final assistant message (read-only from
        // the durable journal). A run that produced output is `ok`; one that journaled no assistant
        // message (errored before producing output) is recorded failed. This is also what
        // `context_from` chains downstream and what the delivery step below sends.
        let captured = self.captured_output(&session).await;
        // A run whose entire output is the `[SILENT]` sentinel succeeded but reports nothing —
        // recorded `ok` but never delivered (ported from Hermes).
        let silent = captured.as_deref().is_some_and(is_cron_silent);
        let ok = captured.is_some();
        let detail = captured
            .clone()
            .unwrap_or_else(|| "no output captured".into());
        run.ok = ok;
        run.detail = Some(detail.clone());
        // Re-append the finished run (the store keys runs by job id; the latest row is updated by a
        // fresh append + bounded retention drops the stale unfinished copy on the next trim — here we
        // instead update the job's last_* directly, which is what the GUI list reads).
        let mut updated = job.clone();
        updated.last_run_unix = Some(run.started_unix);
        updated.last_ok = Some(ok);
        updated.last_detail = Some(detail);
        let _ = self.store.cron_run_append(run).await;
        let _ = self.store.cron_set(updated).await;
        // Post-settle delivery (Phase 2): push the captured result to the job's `deliver` transport(s)
        // through the host's existing `DeliverySink` registry. Suppressed for a `[SILENT]` run, a
        // failed/empty run, a store-only (`deliver = None`) job, or a node with no delivery handle.
        if ok && !silent {
            if let (Some(delivery), Some(text)) = (self.delivery.get(), captured) {
                let spec = Self::decode_spec(job);
                if let Some(deliver) = spec
                    .deliver
                    .as_deref()
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                {
                    delivery.deliver(deliver, spec.origin.as_ref(), &text).await;
                }
            }
        }
    }

    /// Read a settled cron session's final assistant message text from the durable verifiable
    /// journal (read-only — the same coalesced [`TranscriptBlock`](daemon_protocol::TranscriptBlock)s
    /// `session_history` serves; no lease, so it never disturbs recovery). Returns the last
    /// assistant message, size-capped on a char boundary; `None` if the session journaled none.
    async fn captured_output(&self, session: &SessionId) -> Option<String> {
        let stream = daemon_common::JournalStreamId::session(session);
        let mut after = 0u64;
        let mut last: Option<String> = None;
        // Page the whole stream (a cron run's transcript is short); bounded against a runaway loop.
        for _ in 0..64 {
            let page = self.store.load_journal(&stream, after, 256).await;
            if page.entries.is_empty() {
                break;
            }
            for je in &page.entries {
                let Ok(view) = daemon_telemetry::decode_entry(&je.entry.bytes) else {
                    continue;
                };
                if let daemon_telemetry::JournalPayload::Block { body } = view.payload {
                    if let Ok(daemon_protocol::TranscriptBlock::Message {
                        role: daemon_protocol::TranscriptRole::Assistant,
                        text,
                    }) = daemon_api::from_cbor::<daemon_protocol::TranscriptBlock>(&body)
                    {
                        last = Some(text);
                    }
                }
            }
            if page.next_cursor <= after || after >= page.head_cursor {
                break;
            }
            after = page.next_cursor;
        }
        last.map(|t| cap_on_boundary(t, CRON_CONTEXT_CHARS))
    }

    /// Build the seed prompt: the job payload (utf-8) preceded by any preloaded `skills` bodies
    /// (v16) and any `context_from` upstream outputs. Order is skills (instructions) → context
    /// (data) → body (task). Each injection is size-capped on a char boundary.
    async fn seed_prompt(&self, spec: &CronSpec) -> String {
        let body = String::from_utf8_lossy(&spec.payload).into_owned();
        let mut prefix = String::new();
        // Preloaded skills first — the agent's instructional context, mirroring a chat that had
        // `skill_view`'d them. Missing/loader-less skills are skipped (the run keeps the on-demand
        // `skill_*` tools), so preloading is best-effort.
        if !spec.skills.is_empty() {
            if let Some(loader) = &self.skill_loader {
                for name in &spec.skills {
                    if let Some(body) = loader(name) {
                        let snippet = cap_on_boundary(body, CRON_SKILL_CHARS);
                        prefix.push_str(&format!("# Skill `{name}`\n{snippet}\n\n"));
                    }
                }
            }
        }
        // Then chained upstream outputs (the latest run detail of each referenced job).
        for upstream in &spec.context_from {
            if let Some(run) = self
                .store
                .cron_runs_list(upstream, 1)
                .await
                .into_iter()
                .next()
            {
                if let Some(detail) = run.detail {
                    let snippet = cap_on_boundary(detail, CRON_CONTEXT_CHARS);
                    prefix.push_str(&format!("# Context from job `{upstream}`\n{snippet}\n\n"));
                }
            }
        }
        if prefix.is_empty() {
            body
        } else {
            format!("{prefix}{body}")
        }
    }

    /// Run a `no_agent` job's script under the contained scripts dir, returning `(ok, detail)`.
    /// Best-effort: a missing scripts dir / contained-rejected path / spawn error is a failed run.
    async fn run_script(&self, rel: &str) -> (bool, String) {
        let Some(dir) = &self.scripts_dir else {
            return (false, "no scripts directory configured".into());
        };
        let Ok(path) = daemon_core::exec::contain(dir, std::path::Path::new(rel)) else {
            return (false, format!("script path escapes scripts dir: {rel}"));
        };
        let out =
            tokio::task::spawn_blocking(move || std::process::Command::new(&path).output()).await;
        match out {
            Ok(Ok(output)) => {
                let mut detail = String::from_utf8_lossy(&output.stdout).into_owned();
                detail.truncate(CRON_CONTEXT_CHARS);
                (output.status.success(), detail)
            }
            Ok(Err(e)) => (false, format!("script spawn failed: {e}")),
            Err(e) => (false, format!("script join failed: {e}")),
        }
    }

    /// Materialize and fire one occurrence of `job`: a `no_agent` script run (recorded inline) or an
    /// isolated `cron_{id}_{ts}` agent session (seeded + wake-enqueued, run recorded as in-flight).
    /// `manual` marks an out-of-band `cron_trigger`. Does not touch the schedule (the caller advances).
    async fn fire(&self, job: &daemon_store::StoredCronJob, spec: &CronSpec, manual: bool) {
        let now = Self::now_unix();
        // Script-only path: run inline, record a completed run, no agent turn.
        if spec.no_agent {
            let (ok, detail) = match &spec.script {
                Some(script) => self.run_script(script).await,
                None => (false, "no_agent job has no script".into()),
            };
            let _ = self
                .store
                .cron_run_append(daemon_store::StoredCronRun {
                    job_id: job.id.clone(),
                    started_unix: now,
                    finished_unix: Some(Self::now_unix()),
                    ok,
                    detail: Some(detail.clone()),
                    session: None,
                    manual,
                })
                .await;
            let mut job = job.clone();
            job.last_run_unix = Some(now);
            job.last_ok = Some(ok);
            job.last_detail = Some(detail);
            job.fire_count = job.fire_count.saturating_add(1);
            let _ = self.store.cron_set(job).await;
            return;
        }

        // Agent path: an isolated cron session seeded with the (context-chained) payload.
        let session = SessionId::new(format!("cron_{}_{}", job.id, now));
        if self.store.status(&session).await.is_none() {
            let prompt = self.seed_prompt(spec).await;
            let mut engine = self.profile.fresh(session.clone());
            engine.push_user(daemon_protocol::UserMsg::new(prompt));
            let Ok(blob) = engine.snapshot().encode() else {
                return;
            };
            if self
                .store
                .create_session(session.clone(), self.partition, blob)
                .await
                .is_err()
            {
                return;
            }
            // Stamp the cron origin + bound profile + isolation role. `scheduled_job` tells the
            // incarnation to set `TurnTrigger::Scheduled`; the `EphemeralSubagent` role keeps the
            // cron run out of the top-level roster (it is a transient, isolated session).
            let mut meta = self.store.session_meta(&session).await.unwrap_or_default();
            meta.scheduled_job = Some(daemon_common::JobId::from(job.id.as_str()));
            meta.role = Some(daemon_store::SessionRole::EphemeralSubagent);
            if let Some(target) = &spec.target {
                meta.bound_profile = Some(ProfileRef::new(target));
            }
            // Phase 2 shaping: persist the run's model/provider/toolset/workdir as a `SessionOverlay`
            // so the durable factory applies it when hydrating this cron session (see
            // `engine_incarnation::hydrate`). The constrained cron profile is the base; the overlay
            // narrows/overrides it (the resolver path is G3-safe — it never wires `cron`/`orchestrate`).
            let overlay = Self::overlay_from_spec(spec);
            if !overlay.is_empty() {
                meta.overlay = daemon_host::encode_overlay(&overlay);
            }
            let _ = self.store.set_session_meta(&session, meta).await;
        }
        let _ = self
            .store
            .cron_run_append(daemon_store::StoredCronRun {
                job_id: job.id.clone(),
                started_unix: now,
                finished_unix: None,
                ok: true,
                detail: None,
                session: Some(session.clone()),
                manual,
            })
            .await;
        let mut updated = job.clone();
        updated.last_run_unix = Some(now);
        updated.fire_count = updated.fire_count.saturating_add(1);
        let _ = self.store.cron_set(updated).await;
        // Kick the cron session into its turn via the shared wake dispatcher.
        self.store.enqueue_wake(session).await;
    }

    /// Whether a fire that was scheduled for `scheduled_fire` and observed at `now` should run, given
    /// the catch-up policy and the schedule's grace window (the rest fast-forwards).
    fn should_fire(spec: &CronSpec, schedule: &Schedule, scheduled_fire: u64, now: u64) -> bool {
        let lateness = now.saturating_sub(scheduled_fire);
        let tolerance = match spec.catch_up {
            CatchUpPolicy::Always => u64::MAX,
            CatchUpPolicy::Grace => schedule.grace_secs(now),
            CatchUpPolicy::Skip => CRON_SKIP_TOLERANCE_SECS,
        };
        lateness <= tolerance
    }

    /// Advance a job's `next_fire` past `now` (fast-forwarding stale misses so a long downtime fires
    /// at most once), applying jitter. Returns the updated job; `next_fire` is `None` when the
    /// schedule is exhausted (a past one-shot).
    fn advanced(
        job: &daemon_store::StoredCronJob,
        schedule: &Schedule,
        now: u64,
    ) -> daemon_store::StoredCronJob {
        let mut next = schedule.next_after(now);
        // Fast-forward: keep advancing while the computed fire is not strictly in the future, so a
        // multi-period downtime collapses to a single next occurrence (no thundering herd).
        let mut guard = 0;
        while let Some(t) = next {
            if t > now || guard > 4096 {
                break;
            }
            next = schedule.next_after(t);
            guard += 1;
        }
        let mut job = job.clone();
        job.next_fire_unix = next.map(|t| t.saturating_add(schedule.jitter_offset(t)));
        job
    }
}

#[async_trait]
impl CronScheduler for CronWorker {
    async fn tick_once(&self) -> Result<(), ServiceError> {
        let now = Self::now_unix();
        for job in self.store.cron_due(now).await {
            self.reconcile(&job).await;
            let spec = Self::decode_spec(&job);
            let schedule = match Self::schedule_of(&spec) {
                Ok(s) => s,
                Err(_) => {
                    // Unparsable schedule: clear next_fire so it stops being due (operator must fix).
                    let mut job = job.clone();
                    job.next_fire_unix = None;
                    let _ = self.store.cron_set(job).await;
                    continue;
                }
            };
            let scheduled_fire = job.next_fire_unix.unwrap_or(now);
            let in_flight = self.in_flight(&job.id).await;

            // OverlapPolicy::Queue defers (no advance) while a previous run is in flight, so the
            // occurrence runs once the prior finishes. Skip/Allow advance now (at-most-once).
            if matches!(spec.overlap, OverlapPolicy::Queue) && in_flight {
                continue;
            }

            let advanced = Self::advanced(&job, &schedule, now);
            let exhausted_oneshot = advanced.next_fire_unix.is_none();

            // Should this occurrence actually fire? (overlap dedup + catch-up grace)
            let blocked_by_overlap = in_flight && matches!(spec.overlap, OverlapPolicy::Skip);
            let fire =
                !blocked_by_overlap && Self::should_fire(&spec, &schedule, scheduled_fire, now);

            // Persist the advance first (at-most-once) unless we are firing — in which case `fire`
            // writes the updated job (fire_count/last_run) and we layer the new next_fire on top.
            if fire {
                self.fire(&advanced, &spec, false).await;
                // Re-read to fold the fire's bookkeeping, then persist the advanced next_fire.
                if let Some(mut latest) = self.store.cron_get(&job.id).await {
                    latest.next_fire_unix = advanced.next_fire_unix;
                    // repeat / auto-delete: a job that has reached its fire cap is removed.
                    if spec.repeat.is_some_and(|max| latest.fire_count >= max) || exhausted_oneshot
                    {
                        let _ = self.store.cron_remove(&job.id).await;
                    } else {
                        let _ = self.store.cron_set(latest).await;
                    }
                }
            } else {
                // Not firing (fast-forward or overlap-skip): just persist the advance, or delete an
                // exhausted one-shot that will never fire again.
                if exhausted_oneshot {
                    let _ = self.store.cron_remove(&job.id).await;
                } else {
                    let _ = self.store.cron_set(advanced).await;
                }
            }
        }
        Ok(())
    }
}

#[async_trait]
impl CronFiring for CronWorker {
    async fn fire_now(&self, id: &str) -> Result<(), ServiceError> {
        let Some(job) = self.store.cron_get(id).await else {
            return Err(ServiceError::new(format!("cron job not found: {id}")));
        };
        let spec = Self::decode_spec(&job);
        // Honor overlap dedup for the manual path too (closes the trigger-vs-tick double-fire race).
        if matches!(spec.overlap, OverlapPolicy::Skip) && self.in_flight(id).await {
            return Ok(());
        }
        self.fire(&job, &spec, true).await;
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
        // Enrich the node with the session's durable identity (profile/title/role) so a GUI tree
        // drill-down carries the same identity as the roster line, sourced from the same host meta.
        let meta = self.store.session_meta(session).await.unwrap_or_default();
        let role = match meta.role {
            Some(daemon_store::SessionRole::Primary) | None => daemon_api::SessionRole::Primary,
            Some(daemon_store::SessionRole::ManagedChild) => daemon_api::SessionRole::ManagedChild,
            Some(daemon_store::SessionRole::EphemeralSubagent) => {
                daemon_api::SessionRole::EphemeralSubagent
            }
        };
        daemon_api::UnitNode {
            id: UnitId::new(session.as_str()),
            kind,
            state,
            work: self.store.delegation_work(session).await,
            usage: self.store.usage_of(session).await,
            children: children.iter().map(|c| UnitId::new(c.as_str())).collect(),
            profile: meta.bound_profile,
            session: Some(session.clone()),
            title: meta.title,
            role: Some(role),
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

    /// The session overlay is applied to the bound profile *before* the provider/registry are
    /// resolved: the resolver observes the overridden model, and the tool registry is narrowed to the
    /// overlay's allowlist. This is the one resolution path shared by the live and durable surfaces,
    /// so proving it here proves both.
    #[test]
    fn resolve_effective_applies_overlay_before_resolution() {
        use daemon_api::{ProviderSelector, ToolsOverride};
        use std::sync::Mutex as StdMutex;

        // Capture the spec the provider resolver is handed, to assert the overlay was applied first.
        type Captured = Arc<StdMutex<Option<(String, Option<Vec<String>>)>>>;
        let seen: Captured = Arc::new(StdMutex::new(None));
        let seen2 = seen.clone();
        let resolver: ProviderResolver = Arc::new(move |spec: &ProfileSpec| {
            *seen2.lock().unwrap() = Some((spec.model.clone(), spec.tool_allowlist.clone()));
            Arc::new(|| Arc::new(MockProvider::completing("ok")) as Arc<dyn Provider>)
                as ProviderBuilder
        });
        let ctx = SessionFactoryCtx {
            resolver,
            extra_tools: Vec::new(),
            engine_config: Config::default(),
            credentials: None,
            context: None,
            context_builder: None,
            memory: Vec::new(),
            memory_builder: None,
            prompt_sources: Vec::new(),
            skills_resolver: None,
            workspace_roots: None,
        };

        let base = ProfileSpec::new("p", ProviderSelector::GenAi, "base-model");
        let overlay = SessionOverlay {
            model: Some("override-model".to_string()),
            provider: None,
            tool_allowlist: ToolsOverride::Allowlist(vec!["fs".to_string()]),
            approval_mode: Some(ApprovalMode::AutoAllow),
            workspace: None,
        };
        let _profile = ctx.resolve_effective(&base, &overlay);

        let (model, allow) = seen.lock().unwrap().clone().expect("resolver ran");
        assert_eq!(model, "override-model", "overlay model override is applied");
        assert_eq!(
            allow,
            Some(vec!["fs".to_string()]),
            "overlay tool allowlist is applied before the registry is built"
        );

        // An empty overlay is a pure inherit: the resolver sees the profile's own model untouched.
        let _ = ctx.resolve_effective(&base, &SessionOverlay::default());
        assert_eq!(seen.lock().unwrap().clone().unwrap().0, "base-model");
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
        let context_builder: ContextEngineBuilder =
            Arc::new(|_profile: Option<&ProfileRef>, id: &SessionId| {
                let aux: Arc<dyn Provider> = Arc::new(MockProvider::completing("summary"));
                Arc::new(
                    LcmContextEngine::open_for_session(LcmConfig::in_memory(), id, aux)
                        .expect("lcm"),
                ) as Arc<dyn ContextEngine>
            });
        let memory_builder: MemoryBuilder = {
            let banks = banks.clone();
            Arc::new(move |_profile: Option<&ProfileRef>, id: &SessionId| {
                vec![banks.get_or_open(id) as Arc<dyn MemoryProvider>]
            })
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
                let out = self
                    .lcm
                    .call_tool("lcm_status", serde_json::Value::Null)
                    .await;
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

    /// The fleet job worker materializes a delegation's attachment paths from the parent's workspace
    /// into the child's `inbox/`, round-tripping through the content store (content-transfer Phase 2a,
    /// delegation-down).
    #[tokio::test]
    async fn worker_materializes_attachment_into_child_inbox() {
        let ws = std::env::temp_dir().join(format!("daemon-worker-ws-{}", std::process::id()));
        let cas = std::env::temp_dir().join(format!("daemon-worker-cas-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&ws);
        let _ = std::fs::remove_dir_all(&cas);
        let roots = Arc::new(WorkspaceRoots::new(ws.clone()));
        let blobs: Arc<dyn BlobStore> = Arc::new(FileBlobStore::open(cas.clone()).unwrap());
        let store: Arc<dyn daemon_store::SessionStore> =
            Arc::new(daemon_store::InMemoryStore::new());
        let profile = EngineProfile::new(
            Arc::new(|| Arc::new(MockProvider::completing("x")) as Arc<dyn Provider>),
            Arc::new(ToolRegistry::new()),
            SystemPrompt::new("t"),
        );
        let worker = FleetJobWorker::new(store, PartitionId::DEFAULT, profile)
            .with_workspace(roots.clone(), blobs);

        let parent = SessionId::new("parent");
        let child = SessionId::new("parent/c1");
        let pdir = roots.session_root(parent.as_str());
        std::fs::create_dir_all(&pdir).unwrap();
        std::fs::write(pdir.join("input.txt"), b"hand me down").unwrap();

        worker
            .materialize_attachments(&parent, &child, &["input.txt".to_string()])
            .await;

        let landed = roots.session_root(child.as_str()).join("inbox/input.txt");
        assert_eq!(std::fs::read(&landed).unwrap(), b"hand me down");

        let _ = std::fs::remove_dir_all(&ws);
        let _ = std::fs::remove_dir_all(&cas);
    }

    #[test]
    fn cap_on_boundary_never_splits_utf8() {
        // A multi-byte char straddling the cap is dropped whole (no panic / no broken char).
        let s = "aé".to_string(); // 'é' is 2 bytes -> total len 3
        assert_eq!(cap_on_boundary(s.clone(), 2), "a");
        assert_eq!(cap_on_boundary(s, 10), "aé");
    }

    fn mock_profile() -> EngineProfile {
        EngineProfile::new(
            Arc::new(|| Arc::new(MockProvider::completing("x")) as Arc<dyn Provider>),
            Arc::new(ToolRegistry::new()),
            SystemPrompt::new("t"),
        )
    }

    #[tokio::test]
    async fn seed_prompt_preloads_skill_bodies_ahead_of_payload() {
        let store: Arc<dyn daemon_store::SessionStore> =
            Arc::new(daemon_store::InMemoryStore::new());
        let loader: CronSkillLoader = Arc::new(|name: &str| match name {
            "briefing" => Some("BRIEFING BODY".to_string()),
            _ => None,
        });
        let worker =
            CronWorker::new(store, PartitionId::DEFAULT, mock_profile()).with_skill_loader(loader);
        let spec = CronSpec {
            name: "j".into(),
            schedule: "0 9 * * *".into(),
            payload: b"do the task".to_vec(),
            skills: vec!["briefing".into(), "missing".into()],
            ..CronSpec::default()
        };
        let prompt = worker.seed_prompt(&spec).await;
        assert!(prompt.contains("# Skill `briefing`"));
        assert!(prompt.contains("BRIEFING BODY"));
        // A skill the loader can't resolve is skipped, not errored.
        assert!(!prompt.contains("# Skill `missing`"));
        // The skill block precedes the task body.
        let skill_at = prompt.find("BRIEFING BODY").unwrap();
        let body_at = prompt.find("do the task").unwrap();
        assert!(skill_at < body_at, "skills must precede the payload");
    }

    #[tokio::test]
    async fn seed_prompt_is_just_payload_without_skills_or_context() {
        let store: Arc<dyn daemon_store::SessionStore> =
            Arc::new(daemon_store::InMemoryStore::new());
        let worker = CronWorker::new(store, PartitionId::DEFAULT, mock_profile());
        let spec = CronSpec {
            name: "j".into(),
            schedule: "0 9 * * *".into(),
            payload: b"only this".to_vec(),
            ..CronSpec::default()
        };
        assert_eq!(worker.seed_prompt(&spec).await, "only this");
    }

    #[test]
    fn silent_sentinel_is_recognized_trimmed() {
        assert!(is_cron_silent("[SILENT]"));
        assert!(is_cron_silent("  [SILENT]\n"));
        assert!(!is_cron_silent("[SILENT] but also this"));
        assert!(!is_cron_silent("all good, here is the digest"));
    }

    #[test]
    fn overlay_from_spec_projects_shaping_fields() {
        use daemon_api::ToolsOverride;
        let spec = CronSpec {
            name: "j".into(),
            schedule: "0 9 * * *".into(),
            model: Some("gpt-5".into()),
            provider: Some("openai".into()),
            enabled_toolsets: Some(vec!["fs".into(), "shell".into()]),
            workdir: Some("/srv/proj".into()),
            ..CronSpec::default()
        };
        let overlay = CronWorker::overlay_from_spec(&spec);
        assert_eq!(overlay.model.as_deref(), Some("gpt-5"));
        // The legacy adapter alias collapses to the GenAi selector.
        assert_eq!(overlay.provider, Some(daemon_api::ProviderSelector::GenAi));
        assert_eq!(
            overlay.tool_allowlist,
            ToolsOverride::Allowlist(vec!["fs".into(), "shell".into()])
        );
        assert_eq!(
            overlay.workspace,
            Some(daemon_common::WorkspaceBinding::Bound(PathBuf::from(
                "/srv/proj"
            )))
        );
    }

    #[test]
    fn overlay_from_spec_is_empty_when_unshaped() {
        let spec = CronSpec {
            name: "j".into(),
            schedule: "0 9 * * *".into(),
            ..CronSpec::default()
        };
        assert!(CronWorker::overlay_from_spec(&spec).is_empty());
    }

    #[test]
    fn parse_provider_accepts_canonical_and_aliases() {
        use daemon_api::ProviderSelector::*;
        assert_eq!(CronWorker::parse_provider("genai"), Some(GenAi));
        assert_eq!(CronWorker::parse_provider("Anthropic"), Some(GenAi));
        assert_eq!(CronWorker::parse_provider("mock"), Some(Mock));
        assert_eq!(CronWorker::parse_provider("llama_cpp"), Some(LlamaCpp));
        assert_eq!(CronWorker::parse_provider("mistral_rs"), Some(MistralRs));
        assert_eq!(CronWorker::parse_provider("nonsense"), None);
    }
}
