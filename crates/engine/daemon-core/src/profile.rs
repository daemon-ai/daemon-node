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
use crate::conversation::SystemPrompt;
use crate::credentials::CredentialProvider;
use crate::engine::Engine;
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
}

impl EngineProfile {
    /// A profile over a provider builder, tool registry, and system prompt. Credentials default to
    /// the engine's embedded L1 pool until [`EngineProfile::with_credentials`] injects an
    /// authority-backed one; the budget defaults to unlimited and tunables to [`Config::default`].
    pub fn new(provider: ProviderBuilder, registry: Arc<ToolRegistry>, system: SystemPrompt) -> Self {
        Self {
            provider,
            registry,
            system,
            credentials: None,
            budget: Budget::unlimited(),
            config: Config::default(),
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

    /// The tool registry shared by engines this profile builds.
    pub fn registry(&self) -> Arc<ToolRegistry> {
        self.registry.clone()
    }

    /// The system prompt engines this profile builds open with.
    pub fn system(&self) -> &SystemPrompt {
        &self.system
    }

    /// Apply the profile's credentials, budget, and tunables to a freshly constructed engine.
    fn dress(&self, mut engine: Engine) -> Engine {
        if let Some((build, profile)) = &self.credentials {
            engine = engine.with_credentials(build(), profile.clone());
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
