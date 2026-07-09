// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! [`EngineProfile`] — the one place that says *how to construct an engine*.
//!
//! Phases 1-8 grew three separate engine-construction sites (the durable activation factory, the
//! live interactive-session builder, and the fleet child spawner), each hardcoding its own
//! provider, tools, and — unevenly — credentials. `EngineProfile` collapses that into a single,
//! cloneable bundle of the engine's *environment* (provider builder, tool registry, system prompt,
//! credentials, budget) with one `fresh`/`from_snapshot` constructor every site shares. The host
//! and binary fill it once; the durable and live lifecycles then build engines the same way, so the
//! credential seam (and any future tunable) is uniform across all three paths.
//!
//! The two engine *lifecycles* remain intentionally distinct — the durable [`crate::engine::Engine`]
//! is driven one-turn-at-a-time through the activation seam, while the live actor
//! ([`crate::actor`]) serves §17 commands over a mailbox — but both are now *constructed* from the
//! same `EngineProfile`.

use crate::command::CommandProviderHandle;
use crate::config::Config;
use crate::context::{ContextEngine, StablePromptSource};
use crate::conversation::SystemPrompt;
use crate::credentials::CredentialProvider;
use crate::engine::Engine;
use crate::exec::ExecutionEnvironment;
use crate::memory::MemoryProvider;
use crate::provider::Provider;
use crate::snapshot::Snapshot;
use crate::tools::ToolRegistry;
use daemon_common::{Budget, ProfileRef, SessionId};
use std::sync::Arc;

/// Builds the model [`Provider`] for one engine (a fresh provider per constructed engine, so a
/// brokered/stateful client is re-bound per activation).
pub type ProviderBuilder = Arc<dyn Fn() -> Arc<dyn Provider> + Send + Sync>;

/// Builds the [`CredentialProvider`] for one engine. A fresh handle per engine lets a brokered
/// client re-bind to the live cut (host-spec §6).
pub type CredentialBuilder = Arc<dyn Fn() -> Arc<dyn CredentialProvider> + Send + Sync>;

/// Builds the [`ExecutionEnvironment`] (§13) for one engine, keyed by its [`SessionId`] — the seam
/// the host uses to root each session's tools in its provisioned workspace (or, later, to route
/// fs/exec to a host-owned/remote env). The default builds a per-session [`LocalEnvironment`] sandbox.
pub type ExecEnvBuilder = Arc<dyn Fn(&SessionId) -> Arc<dyn ExecutionEnvironment> + Send + Sync>;

/// Builds the §10 [`ContextEngine`] for one engine, keyed by its owning [`ProfileRef`] (the §5.9
/// routed/identity profile, `None` => the node default) and its [`SessionId`]. A stateful engine
/// (e.g. LCM, which tracks per-session compaction state) needs a *fresh instance per session* so
/// concurrent sessions never share mutable state; the profile key additionally roots each session's
/// store under that profile's home, so two rooms routed to two profiles never share a context bank.
/// Prefer over [`EngineProfile::with_context_engine`] for any engine that is not stateless.
pub type ContextEngineBuilder =
    Arc<dyn Fn(Option<&ProfileRef>, &SessionId) -> Arc<dyn ContextEngine> + Send + Sync>;

/// Builds the §11 [`MemoryProvider`] set for one engine, keyed by its owning [`ProfileRef`] (the
/// §5.9 routed/identity profile, `None` => the node default) and its [`SessionId`] — so a backend
/// scoped by session (e.g. Mnemosyne's `session_id` row column) is constructed per-session, and the
/// bank itself is rooted under the resolved profile's home (per-profile memory isolation under
/// per-room binding). Prefer over [`EngineProfile::with_memory`] for any session-scoped backend.
pub type MemoryBuilder =
    Arc<dyn Fn(Option<&ProfileRef>, &SessionId) -> Vec<Arc<dyn MemoryProvider>> + Send + Sync>;

/// The engine's construction environment, shared by every construction site (durable factory, live
/// session builder, fleet child spawner).
#[derive(Clone)]
pub struct EngineProfile {
    provider: ProviderBuilder,
    registry: Arc<ToolRegistry>,
    system: SystemPrompt,
    credentials: Option<(CredentialBuilder, ProfileRef)>,
    /// The credential profile to fall back to when the primary credential profile is exhausted
    /// (the `Recovery::Fallback` hop). `None` => no fallback (a single-profile engine).
    fallback_profile: Option<ProfileRef>,
    /// The owning *identity* profile (the §5.9 routed profile), used to scope this engine's §10/§11
    /// subsystem stores (LCM/Mnemosyne banks under `<data_dir>/<profile>/`) and surfaced to tools via
    /// [`TurnCx::profile`](crate::turn::TurnCx). Distinct from the credential profile (which mutates on
    /// a fallback hop); `None` => the node default profile (legacy single-profile behavior).
    profile: Option<ProfileRef>,
    budget: Budget,
    config: Config,
    exec: Option<ExecEnvBuilder>,
    context: Option<Arc<dyn ContextEngine>>,
    context_builder: Option<ContextEngineBuilder>,
    memory: Vec<Arc<dyn MemoryProvider>>,
    memory_builder: Option<MemoryBuilder>,
    /// Generic stable prompt sources (§10), e.g. the skills index — independent of memory;
    /// composed into the system prompt once per session.
    prompt_sources: Vec<Arc<dyn StablePromptSource>>,
    /// The §12 tool-checkpoint store every engine this profile builds records pre-mutation
    /// checkpoints into (shared across sessions; rewound via the control surface). `None` => off.
    checkpoints: Option<Arc<dyn crate::checkpoint::CheckpointStore>>,
    /// Node-scoped command providers contributed explicitly (the `register_command` analog), folded
    /// into the node command registry alongside the providers derived from the context engine /
    /// memory set. Separate from the per-turn engine wiring: these back the out-of-band command
    /// surface, not the turn loop.
    command_providers: Vec<CommandProviderHandle>,
}

impl EngineProfile {
    /// A profile over a provider builder, tool registry, and system prompt. Credentials default to
    /// the engine's embedded L1 pool until [`EngineProfile::with_credentials`] injects an
    /// authority-backed one; the budget defaults to unlimited and tunables to [`Config::default`].
    pub fn new(
        provider: ProviderBuilder,
        registry: Arc<ToolRegistry>,
        system: SystemPrompt,
    ) -> Self {
        Self {
            provider,
            registry,
            system,
            credentials: None,
            fallback_profile: None,
            profile: None,
            budget: Budget::unlimited(),
            config: Config::default(),
            exec: None,
            context: None,
            context_builder: None,
            memory: Vec::new(),
            memory_builder: None,
            prompt_sources: Vec::new(),
            checkpoints: None,
            command_providers: Vec::new(),
        }
    }

    /// Inject the credential provider builder + profile every engine this profile builds acquires
    /// capabilities from (the host bridge for the §7 port; applied uniformly to the durable, live,
    /// and fleet-child paths).
    pub fn with_credentials(mut self, credentials: CredentialBuilder, profile: ProfileRef) -> Self {
        self.credentials = Some((credentials, profile));
        self
    }

    /// Set the credential profile every engine this profile builds falls back to when its primary
    /// credential profile is exhausted (the engine's `Recovery::Fallback` hop). The fallback uses
    /// the same provider client/broker, only re-keying the acquired capability — so it composes a
    /// cross-credential failover chain on top of the per-profile multi-key pool.
    pub fn with_fallback_profile(mut self, profile: ProfileRef) -> Self {
        self.fallback_profile = Some(profile);
        self
    }

    /// Bind the owning *identity* profile (the §5.9 routed profile) that scopes every engine this
    /// profile builds: its §10/§11 subsystem stores are rooted under that profile's home and the
    /// builders/tools resolve the profile-keyed bank. Distinct from the credential profile.
    pub fn with_profile_ref(mut self, profile: ProfileRef) -> Self {
        self.profile = Some(profile);
        self
    }

    /// Govern every engine this profile builds with `budget`.
    pub fn with_budget(mut self, budget: Budget) -> Self {
        self.budget = budget;
        self
    }

    /// Inject the engine tunables (§20) every engine this profile builds runs under.
    pub fn with_config(mut self, config: Config) -> Self {
        self.config = config;
        self
    }

    /// Inject the execution-environment builder (§13) every engine this profile builds runs its tools
    /// in — the host roots each session in its provisioned workspace (or a host-routed env). Without
    /// it, engines fall back to the per-session [`LocalEnvironment`] sandbox.
    pub fn with_exec(mut self, exec: ExecEnvBuilder) -> Self {
        self.exec = Some(exec);
        self
    }

    /// Inject the §12 tool-checkpoint store every engine this profile builds records pre-mutation
    /// checkpoints into. Without it, checkpointing is off (read-only / no-rewind engines).
    pub fn with_checkpoints(
        mut self,
        checkpoints: Arc<dyn crate::checkpoint::CheckpointStore>,
    ) -> Self {
        self.checkpoints = Some(checkpoints);
        self
    }

    /// Inject the context engine (§10) every engine this profile builds assembles/compacts context
    /// with. Without it, engines use the default [`BudgetedContextEngine`](crate::context::BudgetedContextEngine).
    pub fn with_context_engine(mut self, context: Arc<dyn ContextEngine>) -> Self {
        self.context = Some(context);
        self
    }

    /// Inject a per-session context-engine builder (§10), keyed by [`SessionId`]. Takes precedence
    /// over [`EngineProfile::with_context_engine`]: each constructed engine gets its *own* instance
    /// (the seam for stateful engines like LCM that must not share mutable state across sessions).
    pub fn with_context_engine_builder(mut self, context: ContextEngineBuilder) -> Self {
        self.context_builder = Some(context);
        self
    }

    /// Register the memory providers (§11) every engine this profile builds consults around each
    /// turn (default empty — memory is opt-in).
    pub fn with_memory(mut self, memory: Vec<Arc<dyn MemoryProvider>>) -> Self {
        self.memory = memory;
        self
    }

    /// Inject a per-session memory builder (§11), keyed by [`SessionId`]. Takes precedence over
    /// [`EngineProfile::with_memory`]: each constructed engine gets its own provider set (the seam
    /// for session-scoped backends like Mnemosyne).
    pub fn with_memory_builder(mut self, memory: MemoryBuilder) -> Self {
        self.memory_builder = Some(memory);
        self
    }

    /// Register a generic stable prompt source (§10) composed into the system prompt of every
    /// engine this profile builds — the seam the skills *index* uses (cache-stable; full bodies load
    /// on demand via `skill_view`). Independent of memory.
    pub fn with_prompt_block(mut self, source: Arc<dyn StablePromptSource>) -> Self {
        self.prompt_sources.push(source);
        self
    }

    /// Replace the tool registry every engine this profile builds is constructed with (e.g. to
    /// constrain a background-review child to a skills-only / memory-only toolset).
    pub fn with_registry(mut self, registry: Arc<ToolRegistry>) -> Self {
        self.registry = registry;
        self
    }

    /// Replace the system prompt every engine this profile builds opens with (e.g. a review persona).
    pub fn with_system(mut self, system: SystemPrompt) -> Self {
        self.system = system;
        self
    }

    /// Register a node-scoped [`CommandProvider`](crate::command::CommandProvider) (the
    /// `register_command` analog) whose commands the node folds into its command registry. Use for
    /// providers that are not the engine's own context/memory instances (e.g. a plugin, or a
    /// node-level maintenance handle). Context-engine / memory-provider commands are picked up
    /// automatically by [`command_providers`](Self::command_providers).
    pub fn with_command_provider(mut self, provider: CommandProviderHandle) -> Self {
        self.command_providers.push(provider);
        self
    }

    /// Collect every command provider this profile contributes: the explicitly-registered ones plus
    /// the [`CommandProvider`](crate::command::CommandProvider) views of the configured context
    /// engine and memory providers (when they opt in via `command_provider()`). The node command
    /// registry calls this once to build its catalog. Per-session builders are not invoked here —
    /// only the shared instances a node-level catalog can enumerate without a session.
    pub fn command_providers(&self) -> Vec<CommandProviderHandle> {
        let mut providers = self.command_providers.clone();
        if let Some(context) = &self.context {
            if let Some(p) = context.clone().command_provider() {
                providers.push(p);
            }
        }
        for memory in &self.memory {
            if let Some(p) = memory.clone().command_provider() {
                providers.push(p);
            }
        }
        providers
    }

    /// The tool registry shared by engines this profile builds.
    pub fn registry(&self) -> Arc<ToolRegistry> {
        self.registry.clone()
    }

    /// The system prompt engines this profile builds open with.
    pub fn system(&self) -> &SystemPrompt {
        &self.system
    }

    /// Apply the profile's credentials, budget, tunables, and execution environment to a freshly
    /// constructed engine.
    fn dress(&self, mut engine: Engine) -> Engine {
        if let Some((build, profile)) = &self.credentials {
            engine = engine.with_credentials(build(), profile.clone());
        }
        if let Some(fallback) = &self.fallback_profile {
            engine = engine.with_fallback_profile(fallback.clone());
        }
        // The owning identity profile (for §10/§11 store scoping + tool resolution via `TurnCx`).
        engine = engine.with_subsystem_profile(self.profile.clone());
        if let Some(build) = &self.exec {
            let exec = build(&engine.snapshot().session_id);
            engine = engine.with_exec(exec);
        }
        if let Some(checkpoints) = &self.checkpoints {
            engine = engine.with_checkpoints(checkpoints.clone());
        }
        // Context/memory: a per-session builder (for stateful/session-scoped backends) takes
        // precedence over a shared instance; both are keyed on the engine's owning profile + session
        // id, so two sessions routed to two profiles never share a context/memory bank.
        if let Some(build) = &self.context_builder {
            let context = build(self.profile.as_ref(), &engine.snapshot().session_id);
            engine = engine.with_context_engine(context);
        } else if let Some(context) = &self.context {
            engine = engine.with_context_engine(context.clone());
        }
        if let Some(build) = &self.memory_builder {
            let memory = build(self.profile.as_ref(), &engine.snapshot().session_id);
            if !memory.is_empty() {
                engine = engine.with_memory(memory);
            }
        } else if !self.memory.is_empty() {
            engine = engine.with_memory(self.memory.clone());
        }
        if !self.prompt_sources.is_empty() {
            engine = engine.with_prompt_sources(self.prompt_sources.clone());
        }
        engine.with_budget(self.budget).with_config(self.config)
    }

    /// Build a fresh engine for a new session id.
    pub fn fresh(&self, id: SessionId) -> Engine {
        let engine = Engine::fresh(
            id,
            self.system.clone(),
            (self.provider)(),
            self.registry.clone(),
        );
        self.dress(engine)
    }

    /// Build an engine over an existing (rehydrated) snapshot.
    pub fn from_snapshot(&self, snapshot: Snapshot) -> Engine {
        let engine = Engine::from_snapshot(snapshot, (self.provider)(), self.registry.clone());
        self.dress(engine)
    }
}
