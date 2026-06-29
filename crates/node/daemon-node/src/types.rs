// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The policy inputs ([`NodeAssembly`]) and the assembled output ([`AssembledNode`]) of
//! [`assemble`](crate::assemble), plus the provider/skills resolution seams the binary supplies.

use std::path::PathBuf;
use std::sync::Arc;

use daemon_api::ProfileSpec;
use daemon_common::{PartitionId, ProfileRef};
use daemon_core::{
    Config, ContextEngine, ContextEngineBuilder, CredentialBuilder, MemoryBuilder, MemoryProvider,
    ProviderBuilder, ProviderRegistry, StablePromptSource, Tool,
};
use daemon_host::{HostConfig, NodeApiImpl, ProfileStore, RoutingRegistry, SupervisorHandle};
use daemon_orchestration::FleetRuntime;
use daemon_telemetry::TraceSigner;

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

/// The policy inputs for [`assemble`](crate::assemble): everything that varies between a production
/// node and a test node. The standard plumbing (role profiles, fleet, factory, host, session
/// surface) is derived.
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
    pub credential_store: Option<Arc<dyn daemon_host::CredentialStore>>,
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
