// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Role-profile dressing: apply the node's shared §10/§11 subsystem stores, brokered credentials,
//! workspace root, and core toolset onto each role [`EngineProfile`] the node builds.

use std::sync::Arc;

use daemon_common::{ProfileRef, SessionId};
use daemon_core::{
    EngineProfile, ExecutionEnvironment, LocalEnvironment, ProviderBuilder, ProviderRegistry,
    StablePromptSource, Tool, ToolRegistry,
};
use daemon_host::WorkspaceRoots;

use crate::types::NodeAssembly;

/// The provider-registry profile name the orchestrator (parent) engine resolves to.
pub(crate) const ORCHESTRATOR_PROFILE: &str = "orchestrator";
/// The provider-registry profile name the (legacy synchronous) fleet-child engine resolves to.
pub(crate) const CHILD_PROFILE: &str = "child";

/// Apply the engine tunables, the default context engine + memory providers (§10/§11), and the
/// optional brokered credentials uniformly to a role profile (credentials bound to the node profile).
pub(crate) fn dress(
    profile: EngineProfile,
    a: &NodeAssembly,
    skills_index: Option<&Arc<dyn StablePromptSource>>,
) -> EngineProfile {
    dress_with_credential(profile, a, a.profile.clone(), skills_index)
}

/// Like [`dress`] but binds credentials to an explicit profile ref (the per-session credential ref).
/// `skills_index` is the launch agent's progressive-disclosure index (the role engines run as the
/// launch agent), folded into the system prompt alongside the node's other stable prompt sources.
pub(crate) fn dress_with_credential(
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

/// Root a base profile's engines under the node `workspace_root` (isolated per-session sandbox),
/// recording the resolved root so the filesystem surface serves the same directory. No-op when no
/// workspace root is configured (engines then fall back to the per-session temp sandbox).
pub(crate) fn root_profile(
    profile: EngineProfile,
    roots: &Option<Arc<WorkspaceRoots>>,
) -> EngineProfile {
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
pub(crate) fn provider_for(providers: &ProviderRegistry, name: &str) -> ProviderBuilder {
    providers
        .builder_for(&ProfileRef::new(name))
        .unwrap_or_else(|| panic!("no provider registered for {name:?} and no default set"))
}

/// A registry seeded with the core local toolset (fs + shell) every daemon-core engine carries, so a
/// leaf or session can do real work in its contained workspace (§12/§13), plus any node-level
/// `extra` tools (e.g. `mnemosyne_*` / `lcm_*`). Callers add role tools (e.g. orchestrate) on top.
pub(crate) fn core_tool_registry(extra: &[Arc<dyn Tool>]) -> ToolRegistry {
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
pub(crate) fn core_tool_registry_with_skills(
    extra: &[Arc<dyn Tool>],
    skills: &[Arc<dyn Tool>],
) -> ToolRegistry {
    let mut registry = core_tool_registry(extra);
    for tool in skills {
        registry.register(tool.clone());
    }
    registry
}
