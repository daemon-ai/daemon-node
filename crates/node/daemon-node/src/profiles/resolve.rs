// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

// Phase 4: the fs here creates the daemon-internal profile/data dir (operator-configured node root),
// not attacker-influenced; raw fs allowed file-wide. No process spawns in this file.
#![allow(clippy::disallowed_methods)]

//! Per-session engine resolution: the one [`resolve_effective`](SessionFactoryCtx::resolve_effective)
//! path shared by the live session surface and the durable rehydration resolver — a session's
//! engine is materialized from its bound [`ProfileSpec`] overlaid with its [`SessionOverlay`].

use std::sync::Arc;

use daemon_api::{
    ApprovalMode, ContextEngineSel, MemoryProviderSel, ProfileSpec, SessionOverlay,
    WorkspaceBinding,
};
use daemon_common::{Budget, ProfileRef, SessionId};
use daemon_core::{
    ApprovalPolicy, Config, ContextEngine, ContextEngineBuilder, CredentialBuilder, EngineProfile,
    ExecutionEnvironment, LocalEnvironment, MemoryBuilder, MemoryProvider, StablePromptSource,
    SystemPrompt, Tool, ToolRegistry,
};
use daemon_host::WorkspaceRoots;

use crate::profiles::registry::{merged_config, session_tool_registry};
use crate::types::{ProviderResolver, SkillsResolver};

/// The captured node context a profile-aware session builder needs to materialize a per-session
/// [`EngineProfile`] from the active [`ProfileSpec`] (provider + persona + tools + budget +
/// context/memory + credentials), without re-borrowing the consumed [`NodeAssembly`](crate::NodeAssembly).
pub(crate) struct SessionFactoryCtx {
    pub(crate) resolver: ProviderResolver,
    pub(crate) extra_tools: Vec<Arc<dyn Tool>>,
    pub(crate) engine_config: Config,
    pub(crate) credentials: Option<CredentialBuilder>,
    pub(crate) context: Option<Arc<dyn ContextEngine>>,
    pub(crate) context_builder: Option<ContextEngineBuilder>,
    pub(crate) memory: Vec<Arc<dyn MemoryProvider>>,
    pub(crate) memory_builder: Option<MemoryBuilder>,
    pub(crate) prompt_sources: Vec<Arc<dyn StablePromptSource>>,
    pub(crate) skills_resolver: Option<SkillsResolver>,
    /// The node's workspace-root resolver (shared with the engine exec builder + the filesystem
    /// surface). `None` keeps engines on the temp-sandbox default.
    pub(crate) workspace_roots: Option<Arc<WorkspaceRoots>>,
    /// The node's `[fs]` tool configuration, applied to each session's `fs` tool.
    pub(crate) fs_config: daemon_tool_fs::FsConfig,
    /// The resident process-service handles (background shell + process tool), shared node-wide.
    pub(crate) procs: crate::profiles::dress::ProcessToolkit,
}

impl SessionFactoryCtx {
    /// Materialize a per-session engine profile by resolving the **bound profile** overlaid with the
    /// session's **overlay** — the one resolution path shared by the live surface and the durable
    /// rehydration path. The overlay's model/provider/tool-allowlist are applied to the spec; its
    /// approval-mode override is baked into the engine config. Unset overlay fields fall through to
    /// the profile, so an empty overlay resolves straight from the profile bundle.
    pub(crate) fn resolve_effective(
        &self,
        base: &ProfileSpec,
        overlay: &SessionOverlay,
    ) -> EngineProfile {
        let mut spec = base.clone();
        overlay.apply_to(&mut spec);
        let spec = &spec;
        let provider = (self.resolver)(spec);
        let mut registry = session_tool_registry(
            &self.extra_tools,
            spec.tool_allowlist.as_deref(),
            &self.fs_config,
            &self.procs,
        );
        let skills_index = self.resolve_skills_into_registry(spec, &mut registry);
        // TODO(prompt-arch Lane E): persona resolution moves node-side (PersonaSource ->
        // PersonaStore SOUL.md); `ProfileSpec.system_prompt` left the wire at v36. This
        // placeholder keeps the pre-existing empty-persona fallback until Lane E lands.
        let persona = "interactive session".to_string();
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
        profile = self.apply_context_selector(profile, spec);
        profile = self.apply_memory_selector(profile, spec);
        for source in &self.prompt_sources {
            profile = profile.with_prompt_block(source.clone());
        }
        if let Some(index) = skills_index {
            profile = profile.with_prompt_block(index);
        }
        profile = self.apply_credentials(profile, spec);
        self.apply_workspace_exec(profile, overlay)
    }

    /// Resolve this agent's own skills (store + tools + index) keyed on its id — the per-profile
    /// analogue of the memory/context builders — registering each `skill_*` tool into `registry`
    /// subject to the same `tool_allowlist` as any other tool. Returns the progressive-disclosure
    /// index only when the profile actually carries skills tools (hermes' `valid_tool_names` gate),
    /// so a profile that excludes them gets no skills block.
    fn resolve_skills_into_registry(
        &self,
        spec: &ProfileSpec,
        registry: &mut ToolRegistry,
    ) -> Option<Arc<dyn StablePromptSource>> {
        self.skills_resolver.as_ref().and_then(|resolve| {
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
        })
    }

    /// Honor the profile's §10 context-engine selector. `Budgeted` is the in-core
    /// `BudgetedContextEngine` default (attach nothing); `Lcm` wires the node's LCM builder when the
    /// node configured one (else its shared context engine, if any).
    fn apply_context_selector(
        &self,
        mut profile: EngineProfile,
        spec: &ProfileSpec,
    ) -> EngineProfile {
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
        profile
    }

    /// Honor the profile's §11 memory-provider selector. `None` attaches no memory; `Mnemosyne`
    /// wires the node's session-scoped builder (the default); `File` uses the node's shared frozen
    /// `FileMemory` providers. When the node didn't wire the requested backend, fall back to whatever
    /// memory it does carry (we currently ship one default each — LCM + Mnemosyne).
    fn apply_memory_selector(
        &self,
        mut profile: EngineProfile,
        spec: &ProfileSpec,
    ) -> EngineProfile {
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
        profile
    }

    /// Bind the brokered credentials to the spec's credential profile, composing a failover chain
    /// onto the per-profile multi-key pool when a fallback credential profile is configured (the
    /// engine re-keys to it when the primary is exhausted). No-op when the node carries no credentials.
    fn apply_credentials(&self, mut profile: EngineProfile, spec: &ProfileSpec) -> EngineProfile {
        if let Some(credentials) = &self.credentials {
            profile = profile.with_credentials(
                credentials.clone(),
                ProfileRef::new(spec.credential_profile()),
            );
            if let Some(fallback) = spec.fallback_credential_profile() {
                profile = profile.with_fallback_profile(ProfileRef::new(fallback));
            }
        }
        profile
    }

    /// Root the engine's execution environment (§13) at the session's workspace: the operator-bound
    /// directory when the overlay carries `Bound(path)`, else the isolated `<workspace_root>/<id>`
    /// sandbox. Record the resolved root so the filesystem surface (`fs_*`) serves the *same*
    /// directory the agent's fs/shell tools operate in. No-op when no workspace root is configured.
    fn apply_workspace_exec(
        &self,
        mut profile: EngineProfile,
        overlay: &SessionOverlay,
    ) -> EngineProfile {
        if let Some(roots) = &self.workspace_roots {
            let roots = roots.clone();
            let binding = overlay.workspace.clone();
            profile = profile.with_exec(Arc::new(move |id: &SessionId| {
                // A `Bound` root is an operator-specified external directory whose contents may be
                // attacker-influenced — mark it UNTRUSTED so workspace-discovered artifacts (a
                // planted `.venv` interpreter) are not auto-trusted (Cluster E). The isolated
                // per-session sandbox is node-managed and trusted.
                let (root, trusted) = match &binding {
                    Some(WorkspaceBinding::Bound(p)) => (canonicalize_bound(p), false),
                    _ => (roots.isolated_root(id.as_str()), true),
                };
                roots.record(id.as_str(), root.clone());
                Arc::new(LocalEnvironment::with_trust(root, trusted))
                    as Arc<dyn ExecutionEnvironment>
            }));
        }
        profile
    }
}

/// Canonicalize an operator-`Bound` workspace root at bind time (Cluster C): create it if missing,
/// then resolve symlinks/`.`/`..` in the root's own prefix to a stable absolute real path. The
/// resulting path is what the engine roots at and records to the FS surface, so the `ContainedRoot`'s
/// root fd opens a stable target and `RESOLVE_BENEATH` is well-defined regardless of symlinks in the
/// bound path's prefix. We deliberately do NOT require `Bound` to live under the node root — `Bound`
/// is by design the external "work on my repo" directory, and the operator-tier capability that gates
/// setting it (Phase 2) is the "explicitly allowed" condition. Falls back to the raw path when the
/// directory cannot be created/canonicalized (e.g. a not-yet-existent mount), preserving prior behavior.
fn canonicalize_bound(p: &std::path::Path) -> std::path::PathBuf {
    let _ = std::fs::create_dir_all(p);
    std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf())
}

/// Map a wire-level [`ApprovalMode`] onto the engine's [`ApprovalPolicy`] (the §12 session mode),
/// for baking a session overlay's approval override into the resolved engine config.
pub(crate) fn approval_mode_to_policy(mode: ApprovalMode) -> ApprovalPolicy {
    match mode {
        ApprovalMode::Ask => ApprovalPolicy::Ask,
        ApprovalMode::AcceptEdits => ApprovalPolicy::AcceptEdits,
        ApprovalMode::AutoAllow => ApprovalPolicy::AutoAllow,
        ApprovalMode::Deny => ApprovalPolicy::Deny,
    }
}

#[cfg(test)]
mod tests {
    //! Composition smoke test for the wired-in defaults: an [`EngineProfile`] dressed with the LCM
    //! context engine + the Mnemosyne memory provider (the same way `dress` wires them from
    //! [`NodeAssembly`](crate::NodeAssembly)) runs one full turn end-to-end, exercising the §10/§11
    //! seams against the real port implementations and the once-per-incarnation lifecycle hooks.

    use super::*;
    use daemon_api::ProfileSpec;
    use daemon_common::SessionId;
    use daemon_context_lcm::{LcmConfig, LcmContextEngine};
    use daemon_core::{
        EventSink, MockProvider, Provider, ProviderBuilder, ToolCall, ToolOutcome, ToolRegistry,
        TurnControl, TurnOutcome,
    };
    use daemon_mnemosyne::{MnemosyneConfig, MnemosyneProvider};
    use daemon_protocol::{
        HostRequest, HostRequestHandler, HostResponse, HostResponseBody, UserMsg,
    };
    use std::collections::HashMap;
    use std::sync::Mutex;

    use async_trait::async_trait;

    struct NoopHost;

    #[async_trait]
    impl HostRequestHandler for NoopHost {
        async fn request(&self, req: HostRequest) -> HostResponse {
            HostResponse {
                request_id: req.request_id,
                body: HostResponseBody::Approved {
                    approved: true,
                    allow_permanent: false,
                    reason: None,
                },
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
            fs_config: daemon_tool_fs::FsConfig::default(),
            procs: crate::profiles::dress::ProcessToolkit {
                registry: Arc::new(daemon_processes::ProcessRegistry::new(
                    daemon_processes::RegistryConfig::default(),
                    Arc::new(daemon_processes::RealClock::new()),
                )),
                shell: daemon_processes::ShellConfig::default(),
            },
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

        // Per-session builders, exactly as `dress` wires them from [`NodeAssembly`]: LCM gets a
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
        assert!(status["store"]["messages"].is_number());
    }
}
