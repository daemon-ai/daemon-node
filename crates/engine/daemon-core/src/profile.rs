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

use crate::config::Config;
use crate::context::ContextEngine;
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

/// Builds the §10 [`ContextEngine`] for one engine, keyed by its [`SessionId`]. A stateful engine
/// (e.g. LCM, which tracks per-session compaction state) needs a *fresh instance per session* so
/// concurrent sessions never share mutable state; this builder is the seam that gives each one its
/// own, while all instances target the same agent-wide store. Prefer over
/// [`EngineProfile::with_context_engine`] for any engine that is not stateless.
pub type ContextEngineBuilder = Arc<dyn Fn(&SessionId) -> Arc<dyn ContextEngine> + Send + Sync>;

/// Builds the §11 [`MemoryProvider`] set for one engine, keyed by its [`SessionId`] — so a backend
/// scoped by session (e.g. Mnemosyne's `session_id` row column over a shared bank) is constructed
/// per-session. Prefer over [`EngineProfile::with_memory`] for any session-scoped backend.
pub type MemoryBuilder = Arc<dyn Fn(&SessionId) -> Vec<Arc<dyn MemoryProvider>> + Send + Sync>;

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
    budget: Budget,
    config: Config,
    exec: Option<ExecEnvBuilder>,
    context: Option<Arc<dyn ContextEngine>>,
    context_builder: Option<ContextEngineBuilder>,
    memory: Vec<Arc<dyn MemoryProvider>>,
    memory_builder: Option<MemoryBuilder>,
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
            budget: Budget::unlimited(),
            config: Config::default(),
            exec: None,
            context: None,
            context_builder: None,
            memory: Vec::new(),
            memory_builder: None,
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
        if let Some(build) = &self.exec {
            let exec = build(&engine.snapshot().session_id);
            engine = engine.with_exec(exec);
        }
        // Context/memory: a per-session builder (for stateful/session-scoped backends) takes
        // precedence over a shared instance; both are keyed on the engine's own session id.
        if let Some(build) = &self.context_builder {
            let context = build(&engine.snapshot().session_id);
            engine = engine.with_context_engine(context);
        } else if let Some(context) = &self.context {
            engine = engine.with_context_engine(context.clone());
        }
        if let Some(build) = &self.memory_builder {
            let memory = build(&engine.snapshot().session_id);
            if !memory.is_empty() {
                engine = engine.with_memory(memory);
            }
        } else if !self.memory.is_empty() {
            engine = engine.with_memory(self.memory.clone());
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
