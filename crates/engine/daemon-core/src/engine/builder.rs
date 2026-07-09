// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `Engine` construction: the snapshot/fresh constructors and the chained `with_*` builder setters
//! (§4.1 wiring of provider, credentials, exec, context, memory, prompt sources, profiles). Split
//! out of `engine.rs` so the module body is the turn machinery; these are pure field wiring with no
//! turn-loop behavior. Behavior-preserving verbatim move.

use super::*;

impl Engine {
    /// Construct an engine over an existing snapshot.
    pub fn from_snapshot(
        snapshot: Snapshot,
        provider: Arc<dyn Provider>,
        registry: Arc<ToolRegistry>,
    ) -> Self {
        // Default sandbox keyed by session id; a host injects a workspace-rooted env via the profile.
        let exec: Arc<dyn ExecutionEnvironment> =
            Arc::new(LocalEnvironment::sandbox(snapshot.session_id.as_str()));
        Self {
            snapshot,
            provider,
            registry,
            pending: Vec::new(),
            budget: Budget::unlimited(),
            // L1 default: the in-tree embedded pool. Under a host an authority-backed provider is
            // injected via `with_credentials` (host-spec §6).
            credentials: Arc::new(EmbeddedCredentialPool::single_key()),
            profile: ProfileRef::new("default"),
            fallback_profile: None,
            subsystem_profile: None,
            config: Config::default(),
            exec,
            checkpoints: None,
            context: Arc::new(BudgetedContextEngine::default()),
            composed: None,
            composed_dirty: false,
            turn_injection: TurnInjection::default(),
            memory: Vec::new(),
            prompt_sources: Vec::new(),
            async_sources: Vec::new(),
            model_sources: Vec::new(),
            model_id: None,
            nudge_sources: Vec::new(),
            tool_observers: Vec::new(),
            next_trigger: None,
            lifecycle_started: false,
        }
    }

    /// Construct a fresh engine for a new session with the given system prompt.
    pub fn fresh(
        session_id: SessionId,
        system: SystemPrompt,
        provider: Arc<dyn Provider>,
        registry: Arc<ToolRegistry>,
    ) -> Self {
        let mut snapshot = Snapshot::fresh(session_id);
        snapshot.conversation = Conversation::new(system);
        Self::from_snapshot(snapshot, provider, registry)
    }

    /// Set the budget governing this engine's turns.
    pub fn with_budget(mut self, budget: Budget) -> Self {
        self.budget = budget;
        self
    }

    /// Inject the engine tunables (§20) the host loaded from its config.
    pub fn with_config(mut self, config: Config) -> Self {
        self.config = config;
        self
    }

    /// Inject the execution environment (§13) this engine's tools run in (a per-session
    /// workspace-rooted [`LocalEnvironment`], or a host-routed env).
    pub fn with_exec(mut self, exec: Arc<dyn ExecutionEnvironment>) -> Self {
        self.exec = exec;
        self
    }

    /// Inject the checkpoint store (§12 safety): the pipeline will record a workspace checkpoint
    /// before each [`mutates`](crate::tools::Tool::mutates) tool runs. Without it, checkpointing is off.
    pub fn with_checkpoints(
        mut self,
        checkpoints: Arc<dyn crate::checkpoint::CheckpointStore>,
    ) -> Self {
        self.checkpoints = Some(checkpoints);
        self
    }

    /// Inject the context engine (§10) this engine assembles/compacts context with (the default is
    /// the cheap [`BudgetedContextEngine`]).
    pub fn with_context_engine(mut self, context: Arc<dyn ContextEngine>) -> Self {
        self.context = context;
        self
    }

    /// Register the memory providers (§11) this engine consults around each turn (default empty).
    pub fn with_memory(mut self, memory: Vec<Arc<dyn MemoryProvider>>) -> Self {
        self.memory = memory;
        self
    }

    /// Register generic stable prompt sources (§10) composed into the system prompt at session
    /// start (e.g. the skills index). Independent of memory; expected to be cache-stable.
    pub fn with_prompt_sources(mut self, sources: Vec<Arc<dyn StablePromptSource>>) -> Self {
        self.prompt_sources = sources;
        self
    }

    /// Register async prompt sources (§10) gathered over the session's execution environment at
    /// the same composition boundaries (e.g. the workspace context files, the environment hints).
    pub fn with_async_sources(mut self, sources: Vec<Arc<dyn AsyncPromptSource>>) -> Self {
        self.async_sources = sources;
        self
    }

    /// Register model-keyed prompt sources (§10), re-resolved at every composition against the
    /// engine's live model identity (e.g. tool-use enforcement + model-family guidance).
    pub fn with_model_sources(mut self, sources: Vec<Arc<dyn ModelPromptSource>>) -> Self {
        self.model_sources = sources;
        self
    }

    /// Set the engine's model identity — the resolved spec's model id, used to key
    /// [`ModelPromptSource`]s and the composed-prompt stale-identity check. Without it the engine
    /// falls back to its credential-profile label.
    pub fn with_model_id(mut self, model: impl Into<String>) -> Self {
        self.model_id = Some(model.into());
        self
    }

    /// Register per-turn nudge sources (§10/§11): consulted when a user-triggered turn opens;
    /// contributions ride that turn's [`TurnInjection`] (e.g. the USER.md save nudge).
    pub fn with_nudge_sources(mut self, sources: Vec<Arc<dyn NudgeSource>>) -> Self {
        self.nudge_sources = sources;
        self
    }

    /// Register per-tool-call observers (§10/§12): each executed call's result may gain appended
    /// hint text (e.g. the subdirectory context-file hints).
    pub fn with_tool_observers(mut self, observers: Vec<Arc<dyn ToolCallObserver>>) -> Self {
        self.tool_observers = observers;
        self
    }

    /// The contained execution environment (§13) this engine's tools run in — exposed so the
    /// profile layer can construct per-session helpers (e.g. tool observers) over the same root
    /// the engine operates in.
    pub fn exec(&self) -> &Arc<dyn ExecutionEnvironment> {
        &self.exec
    }

    /// Inject the credential provider + profile this engine acquires capabilities from (the host
    /// injects an authority-backed or brokered impl; the default is the embedded L1 pool).
    pub fn with_credentials(
        mut self,
        credentials: Arc<dyn CredentialProvider>,
        profile: ProfileRef,
    ) -> Self {
        self.credentials = credentials;
        self.profile = profile;
        self
    }

    /// Set the single fallback profile the §8 recovery loop hops to when the active profile cannot
    /// recover a model failure (persistent auth/billing/content-policy).
    pub fn with_fallback_profile(mut self, profile: ProfileRef) -> Self {
        self.fallback_profile = Some(profile);
        self
    }

    /// Bind the owning *identity* profile (§5.9) this engine's §10/§11 stores are scoped under and
    /// that is surfaced to tools via [`TurnCx::profile`](crate::turn::TurnCx). `None` => node default.
    pub fn with_subsystem_profile(mut self, profile: Option<ProfileRef>) -> Self {
        self.subsystem_profile = profile;
        self
    }
}
