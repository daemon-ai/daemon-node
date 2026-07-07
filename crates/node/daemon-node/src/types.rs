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
    /// The `[fs]` tool configuration (read caps, search caps, deny paths, post-edit lint) applied
    /// to every `fs` tool this node registers — the role registries and each per-session registry
    /// alike. `Default::default()` keeps the tool's built-in caps (tests).
    pub fs: daemon_tool_fs::FsConfig,
    /// The background-process service policy (`[processes]` registry limits + `[shell]` tool
    /// limits). The default carries the hermes-parity limits (200 KB ring, 64 tracked, 30 min TTL,
    /// 180 s/600 s foreground timeouts, watch rate limits), so tests just pass
    /// `Default::default()`.
    pub processes: daemon_processes::ProcessesConfig,
    /// The auxiliary provider for background session-title generation (resolved by the binary the
    /// same way as the LCM/Mnemosyne aux providers): after a live session's first exchange, one
    /// best-effort `task = "title_generation"` call replaces the truncation-seeded roster title.
    /// `None` keeps seeded titles only (tests / nodes without an aux provider).
    pub title_aux: Option<Arc<dyn daemon_core::Provider>>,
    /// The ephemeral-subagent reaper policy ([`EphemeralReaper`](crate::fleet::EphemeralReaper)):
    /// archive `EphemeralSubagent` sessions `grace` after their terminal state, swept every
    /// `interval`. The default is enabled (300s grace / 60s interval); the first sweep runs one
    /// interval after start, so short-lived test nodes never observe one.
    pub reaper: crate::fleet::ReaperConfig,
    /// The delegation guardrail caps the `orchestrate` tool enforces (wire v29): the TOML
    /// `[orchestrate].max_depth` / `.max_fanout` policy ceilings, surfaced read-only via the
    /// `Caps` op. The effective depth guard composes with [`Self::nesting_depth`] (the assembly
    /// recursion budget): the tool declines past `min(max_depth, nesting_depth + 1)`, so the
    /// policy cap can narrow the structural budget but never widen it.
    pub orchestrate: OrchestrateCaps,
    /// The node gateway's loopback coordinates + per-session token minter, for routing
    /// `NodeProvider`-backed OpenAI-wire foreign agents through the gateway. `None` (the default)
    /// leaves foreign agents on their own backend; the binary sets it when the gateway has a bind
    /// address.
    pub foreign_gateway: Option<GatewayCoords>,
}

/// The loopback coordinates of the node's OpenAI-compatible gateway (`daemon-gateway`), threaded
/// into the interactive session builder so a `NodeProvider`-backed OpenAI-wire foreign agent
/// (codex/opencode) is spawned pointed at the gateway (`OPENAI_BASE_URL`/`OPENAI_API_KEY`) instead
/// of holding a real provider key. `None` on [`NodeAssembly::foreign_gateway`] leaves foreign agents
/// on their own backend; the binary sets it whenever the gateway has a bind address.
///
/// The bearer is no longer a single global token: routing is now per-profile (Phase 2). When a
/// `NodeProvider` foreign session opens, the builder mints a PER-SESSION token via [`minter`] bound
/// to that session's `{provider, model, credential_ref}` and injects it as `OPENAI_API_KEY`; the
/// gateway resolves the provider+model+credential from the presented token (node-side), so no
/// provider key ever reaches the agent. Injection is env-only — the launch recipe still comes from
/// the catalog by name, preserving the foreign-engine security invariant.
///
/// [`minter`]: GatewayCoords::minter
#[derive(Clone)]
pub struct GatewayCoords {
    /// The gateway base URL an agent's `OPENAI_BASE_URL` is set to (e.g. `http://127.0.0.1:8081/v1`).
    pub base_url: String,
    /// The per-session token registry: mint a loopback bearer bound to a foreign session's routing,
    /// revoked when the session ends. Implemented by the binary over the gateway backend's registry.
    pub minter: Arc<dyn GatewayTokenMinter>,
}

/// The node-side routing a per-session gateway token is bound to (Phase 2): a foreign
/// `NodeProvider` session's turns run on this `{provider, model, credential_ref}`. Mirrors
/// [`daemon_api::ForeignBackend::NodeProvider`] but stays a plain host-side struct (never wire), so
/// the token→binding table can live node-side and the agent only ever holds an opaque bearer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GatewayBinding {
    /// The node provider the gateway routes the token's turns to.
    pub provider: daemon_api::ProviderSelector,
    /// The model id the gateway routes to.
    pub model: String,
    /// The stored credential the gateway acquires the provider bearer from (`None` = the routed
    /// profile's own credential). A name, never the secret.
    pub credential_ref: Option<String>,
}

/// The per-session gateway-token registry seam the binary implements over the gateway backend's
/// token→binding table. `daemon-node` never links `daemon-gateway`/the binary, so this trait is the
/// injection point (mirroring [`ProviderResolver`]): the session builder mints a token bound to a
/// foreign session's routing at open and revokes it at close, keeping the real provider key
/// node-side.
pub trait GatewayTokenMinter: Send + Sync {
    /// Mint a fresh loopback bearer bound to `binding`, returning the opaque token to inject as the
    /// agent's `OPENAI_API_KEY`.
    fn mint(&self, binding: GatewayBinding) -> String;

    /// Revoke a previously-minted token (idempotent), dropping its binding from the registry.
    fn revoke(&self, token: &str);
}

/// A drop guard tying a minted per-session gateway token to the lifetime of the foreign session it
/// was minted for: when the session's [`AgentSession`](daemon_host::AgentSession) is dropped (the
/// session closed / the node shut down), the guard revokes the token so its binding never outlives
/// the session and the registry cannot grow unbounded.
pub struct GatewayLease {
    minter: Arc<dyn GatewayTokenMinter>,
    token: String,
}

impl GatewayLease {
    /// Build a lease over a freshly-minted `token`; dropping it revokes via `minter`.
    pub fn new(minter: Arc<dyn GatewayTokenMinter>, token: String) -> Self {
        Self { minter, token }
    }
}

impl Drop for GatewayLease {
    fn drop(&mut self) {
        self.minter.revoke(&self.token);
    }
}

/// The delegation guardrail caps (`[orchestrate].max_depth` / `.max_fanout`) threaded into the
/// `orchestrate` tool and the read-only `Caps` surface. Defaults mirror the tool's own built-in
/// ceilings (8 / 8).
#[derive(Clone, Copy, Debug)]
pub struct OrchestrateCaps {
    /// The delegation-tree depth ceiling a `spawn` is declined past.
    pub max_depth: usize,
    /// The concurrent detached-children ceiling per parent a `spawn wait:false` is declined past.
    pub max_fanout: usize,
}

impl Default for OrchestrateCaps {
    fn default() -> Self {
        Self {
            max_depth: 8,
            max_fanout: 8,
        }
    }
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
    /// The resident background-process registry. The shutdown path calls
    /// [`ProcessRegistry::shutdown`](daemon_processes::ProcessRegistry::shutdown) so no spawned
    /// process group outlives the daemon.
    pub processes: Arc<daemon_processes::ProcessRegistry>,
}
