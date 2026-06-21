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

/// The engine's construction environment, shared by every construction site (durable factory, live
/// session builder, fleet child spawner).
#[derive(Clone)]
pub struct EngineProfile {
    provider: ProviderBuilder,
    registry: Arc<ToolRegistry>,
    system: SystemPrompt,
    credentials: Option<(CredentialBuilder, ProfileRef)>,
    budget: Budget,
    config: Config,
    exec: Option<ExecEnvBuilder>,
    context: Option<Arc<dyn ContextEngine>>,
    memory: Vec<Arc<dyn MemoryProvider>>,
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
            budget: Budget::unlimited(),
            config: Config::default(),
            exec: None,
            context: None,
            memory: Vec::new(),
        }
    }

    /// Inject the credential provider builder + profile every engine this profile builds acquires
    /// capabilities from (the host bridge for the §7 port; applied uniformly to the durable, live,
    /// and fleet-child paths).
    pub fn with_credentials(mut self, credentials: CredentialBuilder, profile: ProfileRef) -> Self {
        self.credentials = Some((credentials, profile));
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

    /// Register the memory providers (§11) every engine this profile builds consults around each
    /// turn (default empty — memory is opt-in).
    pub fn with_memory(mut self, memory: Vec<Arc<dyn MemoryProvider>>) -> Self {
        self.memory = memory;
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
        if let Some(build) = &self.exec {
            let exec = build(&engine.snapshot().session_id);
            engine = engine.with_exec(exec);
        }
        if let Some(context) = &self.context {
            engine = engine.with_context_engine(context.clone());
        }
        if !self.memory.is_empty() {
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
