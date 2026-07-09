// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Tool-registry construction for the node's role engines: the interactive/session toolset, the
//! §20 tunables overlay, and the §4.3 background-review profile registry.

use std::sync::Arc;

use daemon_api::EngineTunables;
use daemon_core::{ApprovalPolicy, Config, EngineProfile, SystemPrompt, Tool, ToolRegistry};
use daemon_host::{BackgroundProfile, BackgroundProfileRegistry};
use daemon_prompt::{role_persona, RolePersona};

use crate::profiles::dress::{provider_for, ORCHESTRATOR_PROFILE};
use crate::types::NodeAssembly;

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
pub(crate) fn background_registry(
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
                role: RolePersona|
     -> EngineProfile {
        let profile = EngineProfile::new(
            provider_for(&a.providers, ORCHESTRATOR_PROFILE),
            Arc::new(constrained_registry(pool, names, prefix)),
            SystemPrompt::new(role_persona(role)),
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
                base(
                    &skill_pool,
                    &SKILL_TOOL_NAMES,
                    None,
                    RolePersona::SkillCurator,
                ),
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
                    RolePersona::MemoryCurator,
                ),
                MEMORY_REVIEW_PROMPT,
            ),
        );
    }
    registry
}

/// Overlay a [`ProfileSpec`](daemon_api::ProfileSpec)'s engine-tunable overrides onto the node's
/// base [`Config`].
pub(crate) fn merged_config(base: Config, t: &EngineTunables) -> Config {
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

/// Build the interactive tool registry for a session: the core fs + shell + process toolset plus
/// node-level `extra` tools, optionally narrowed to an allowlist of tool names. `fs` is the node's
/// `[fs]` tool configuration (caps / deny paths / lint).
pub(crate) fn session_tool_registry(
    extra: &[Arc<dyn Tool>],
    allowlist: Option<&[String]>,
    fs: &daemon_tool_fs::FsConfig,
    procs: &crate::profiles::dress::ProcessToolkit,
) -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    let mut candidates: Vec<Arc<dyn Tool>> = vec![
        Arc::new(daemon_tool_fs::FsTool::with_config(fs.clone())) as Arc<dyn Tool>,
        Arc::new(daemon_tool_shell::ShellTool::with_processes(
            procs.registry.clone(),
            procs.shell,
        )) as Arc<dyn Tool>,
        Arc::new(daemon_tool_process::ProcessTool::new(
            procs.registry.clone(),
            procs.shell,
        )) as Arc<dyn Tool>,
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
