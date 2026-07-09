// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Gated guidance-block builders for the composed system prompt — ports of the hermes-agent
//! `agent/prompt_builder.py` guidance constants and dynamic builders, with tool names adapted to
//! the daemon toolset (`shell` / `fs` / `execute_code` instead of `terminal` / `read_file` /
//! `search_files`).
//!
//! Every builder returns `Option<String>`: `None` means the block contributes nothing. Builders
//! are pure functions over explicit inputs — no clocks, no env vars, no ambient config — so a
//! caller that snapshots them once per session gets a byte-stable prefix. The node fills
//! [`EnvironmentInput`] from its `ExecutionEnvironment` and configuration; this crate never
//! probes the world itself.
//!
//! Deliberately NOT here (owned by existing seams): the Mnemosyne memory block
//! (`daemon-mnemosyne` `prompt_block`), the skills index header (`daemon-skills`), and tool
//! guardrails (per-tool schemas).

/// Universal "finish the job" guidance — applied to ALL models, not gated by family. Addresses
/// two cross-model failure modes: stopping after a stub, and fabricating output when a real path
/// is blocked. Short on purpose: it ships in every cached system prompt.
pub const TASK_COMPLETION_GUIDANCE: &str = "# Finishing the job\n\
When the user asks you to build, run, or verify something, the deliverable is a working artifact \
backed by real tool output — not a description of one. Do not stop after writing a stub, a plan, \
or a single command. Keep working until you have actually exercised the code or produced the \
requested result, then report what real execution returned.\n\
If a tool, install, or network call fails and blocks the real path, say so directly and try an \
alternative (different package manager, different approach, ask the user). NEVER substitute \
plausible-looking fabricated output (made-up data, invented file contents, synthesised API \
responses) for results you couldn't actually produce. Reporting a blocker honestly is always \
better than inventing a result.";

/// Tool-use enforcement guidance: models that narrate intended actions instead of calling tools.
pub const TOOL_USE_ENFORCEMENT_GUIDANCE: &str = "# Tool-use enforcement\n\
You MUST use your tools to take action — do not describe what you would do or plan to do without \
actually doing it. When you say you will perform an action (e.g. 'I will run the tests', 'Let me \
check the file', 'I will create the project'), you MUST immediately make the corresponding tool \
call in the same response. Never end your turn with a promise of future action — execute it \
now.\n\
Keep working until the task is actually complete. Do not stop with a summary of what you plan to \
do next time. If you have tools available that can accomplish the task, use them instead of \
telling the user what you would do.\n\
Every response should either (a) contain tool calls that make progress, or (b) deliver a final \
result to the user. Responses that only describe intentions without acting are not acceptable.";

/// Model-id substrings that trigger tool-use enforcement guidance in [`ToolUseMode::Auto`]. Add
/// new entries when a model family needs explicit steering.
pub const TOOL_USE_ENFORCEMENT_MODELS: &[&str] = &[
    "gpt", "codex", "gemini", "gemma", "grok", "glm", "qwen", "deepseek",
];

/// OpenAI GPT/Codex-specific execution guidance; also applied to xAI Grok (same failure modes in
/// practice). The body is family-agnostic; the `OPENAI_` prefix reflects origin, not exclusivity.
pub const OPENAI_MODEL_EXECUTION_GUIDANCE: &str = "# Execution discipline\n\
<tool_persistence>\n\
- Use tools whenever they improve correctness, completeness, or grounding.\n\
- Do not stop early when another tool call would materially improve the result.\n\
- If a tool returns empty or partial results, retry with a different query or strategy before \
giving up.\n\
- Keep calling tools until: (1) the task is complete, AND (2) you have verified the result.\n\
</tool_persistence>\n\
\n\
<mandatory_tool_use>\n\
NEVER answer these from memory or mental computation — ALWAYS use a tool:\n\
- Arithmetic, math, calculations → use shell or execute_code\n\
- Hashes, encodings, checksums → use shell (e.g. sha256sum, base64)\n\
- Current time, date, timezone → use shell (e.g. date)\n\
- System state: OS, CPU, memory, disk, ports, processes → use shell\n\
- File contents, sizes, line counts → use the fs tool or shell\n\
- Git history, branches, diffs → use shell\n\
- Current facts (weather, news, versions) → use web_search\n\
Your user profile describes the USER, not the system you are running on. The execution \
environment may differ from what the user profile says about their personal setup.\n\
</mandatory_tool_use>\n\
\n\
<act_dont_ask>\n\
When a question has an obvious default interpretation, act on it immediately instead of asking \
for clarification. Examples:\n\
- 'Is port 443 open?' → check THIS machine (don't ask 'open where?')\n\
- 'What OS am I running?' → check the live system (don't use the user profile)\n\
- 'What time is it?' → run `date` (don't guess)\n\
Only ask for clarification when the ambiguity genuinely changes what tool you would call.\n\
</act_dont_ask>\n\
\n\
<prerequisite_checks>\n\
- Before taking an action, check whether prerequisite discovery, lookup, or context-gathering \
steps are needed.\n\
- Do not skip prerequisite steps just because the final action seems obvious.\n\
- If a task depends on output from a prior step, resolve that dependency first.\n\
</prerequisite_checks>\n\
\n\
<verification>\n\
Before finalizing your response:\n\
- Correctness: does the output satisfy every stated requirement?\n\
- Grounding: are factual claims backed by tool outputs or provided context?\n\
- Formatting: does the output match the requested format or schema?\n\
- Safety: if the next step has side effects (file writes, commands, API calls), confirm scope \
before executing.\n\
</verification>\n\
\n\
<missing_context>\n\
- If required context is missing, do NOT guess or hallucinate an answer.\n\
- Use the appropriate lookup tool when missing information is retrievable (fs, web_search, \
etc.).\n\
- Ask a clarifying question only when the information cannot be retrieved by tools.\n\
- If you must proceed with incomplete information, label assumptions explicitly.\n\
</missing_context>";

/// Gemini/Gemma-specific operational guidance, adapted from OpenCode's gemini.txt.
pub const GOOGLE_MODEL_OPERATIONAL_GUIDANCE: &str = "# Google model operational directives\n\
Follow these operational rules strictly:\n\
- **Absolute paths:** Always construct and use absolute file paths for all file system \
operations. Combine the project root with relative paths.\n\
- **Verify first:** Use the fs tool to check file contents and project structure before making \
changes. Never guess at file contents.\n\
- **Dependency checks:** Never assume a library is available. Check package.json, \
requirements.txt, Cargo.toml, etc. before importing.\n\
- **Conciseness:** Keep explanatory text brief — a few sentences, not paragraphs. Focus on \
actions and results over narration.\n\
- **Parallel tool calls:** When you need to perform multiple independent operations (e.g. \
reading several files), make all the tool calls in a single response rather than sequentially.\n\
- **Non-interactive commands:** Use flags like -y, --yes, --non-interactive to prevent CLI tools \
from hanging on prompts.\n\
- **Keep going:** Work autonomously until the task is fully resolved. Don't stop with a plan — \
execute it.\n";

/// The always-on core agentic / task-completion block ([`TASK_COMPLETION_GUIDANCE`]).
pub fn core_agentic_guidance() -> Option<String> {
    Some(TASK_COMPLETION_GUIDANCE.to_string())
}

/// How the tool-use enforcement block is gated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolUseMode {
    /// Emit when the registry is non-empty AND the model family is on the enforcement list.
    Auto,
    /// Always emit (when the registry is non-empty — the text is meaningless with no tools).
    On,
    /// Never emit.
    Off,
}

/// The tool-use enforcement block, gated on the tool registry contents + `mode` + model family.
pub fn tool_use_guidance(tool_names: &[&str], mode: ToolUseMode, model_id: &str) -> Option<String> {
    let has_tools = !tool_names.is_empty();
    let emit = match mode {
        ToolUseMode::Off => false,
        ToolUseMode::On => has_tools,
        ToolUseMode::Auto => {
            let id = model_id.to_lowercase();
            has_tools && TOOL_USE_ENFORCEMENT_MODELS.iter().any(|m| id.contains(m))
        }
    };
    emit.then(|| TOOL_USE_ENFORCEMENT_GUIDANCE.to_string())
}

/// Model-family operational guidance, keyed on the model id: GPT/Codex/Grok get the execution
/// discipline block; Gemini/Gemma get the Google operational directives; others get nothing
/// (Claude needs no steering; glm/qwen/deepseek get only the enforcement block via
/// [`tool_use_guidance`]).
pub fn model_family_guidance(model_id: &str) -> Option<String> {
    let id = model_id.to_lowercase();
    if ["gpt", "codex", "grok"].iter().any(|m| id.contains(m)) {
        Some(OPENAI_MODEL_EXECUTION_GUIDANCE.to_string())
    } else if ["gemini", "gemma"].iter().any(|m| id.contains(m)) {
        Some(GOOGLE_MODEL_OPERATIONAL_GUIDANCE.to_string())
    } else {
        None
    }
}

/// Facts about the execution environment, filled by the node from its `ExecutionEnvironment` and
/// configuration. This crate never probes the world: modeling the inputs as a plain struct keeps
/// the builder pure and the composed prompt byte-stable.
#[derive(Clone, Debug, Default)]
pub struct EnvironmentInput {
    /// Whether the session's backend is a remote/sandboxed environment (tools do not touch the
    /// daemon host). When `true`, ALL host details below are suppressed — the agent's tools
    /// can't reach the host, so host facts would only mislead it.
    pub sandboxed: bool,
    /// The backend label for a sandboxed environment (e.g. `docker`, `modal`, `ssh`).
    pub backend_label: Option<String>,
    /// Pre-formatted backend state lines (OS / user / home / cwd inside the backend), when the
    /// node probed them. Rendered verbatim under the backend block.
    pub backend_details: Option<String>,
    /// The host OS description, e.g. `Linux (6.8.0-generic)` (local backends only).
    pub host_os: Option<String>,
    /// The host user's home directory (local backends only).
    pub user_home: Option<String>,
    /// The session's WORKSPACE working directory — the root the agent's tools operate in, which
    /// is preferred over the daemon process's own cwd (the hermes `TERMINAL_CWD`-over-launch-dir
    /// fix, #24882 class).
    pub workspace_cwd: Option<String>,
    /// An operator/embedder-supplied environment description, appended after the factual block.
    pub extra_hint: Option<String>,
}

/// Environment hints for the system prompt: a factual block describing where the agent's tools
/// run. For local backends this names the host OS, home, and workspace cwd; for sandboxed
/// backends host details are suppressed and the backend is described instead.
pub fn environment_hints(input: &EnvironmentInput) -> Option<String> {
    let mut blocks: Vec<String> = Vec::new();

    if input.sandboxed {
        let label = input.backend_label.as_deref().unwrap_or("sandboxed");
        let block = match &input.backend_details {
            Some(details) => format!(
                "Execution backend: {label}. Your `shell`, `fs`, and `execute_code` tools all \
                 operate inside this {label} environment — NOT on the machine where the daemon \
                 itself is running. The host OS, home, and cwd of the daemon process are \
                 irrelevant; only the following backend state matters:\n{details}"
            ),
            None => format!(
                "Execution backend: {label}. Your `shell`, `fs`, and `execute_code` tools all \
                 operate inside this {label} environment — NOT on the machine where the daemon \
                 itself runs. The backend's current user, $HOME, and working directory are \
                 unknown from here. If you need them, probe directly with a shell call like \
                 `uname -a && whoami && pwd`."
            ),
        };
        blocks.push(block);
    } else {
        let mut host_lines: Vec<String> = Vec::new();
        if let Some(os) = &input.host_os {
            host_lines.push(format!("Host: {os}"));
        }
        if let Some(home) = &input.user_home {
            host_lines.push(format!("User home directory: {home}"));
        }
        if let Some(cwd) = &input.workspace_cwd {
            host_lines.push(format!("Current working directory: {cwd}"));
        }
        if !host_lines.is_empty() {
            blocks.push(host_lines.join("\n"));
        }
    }

    if let Some(extra) = input
        .extra_hint
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        blocks.push(extra.to_string());
    }

    if blocks.is_empty() {
        return None;
    }
    Some(blocks.join("\n\n"))
}

/// The per-surface formatting hint for a transport `family` — the daemon analogue of hermes'
/// platform hints. Keyed on the transport FAMILY string (the node splits an instance-qualified
/// id like `matrix/@bot:hs` down to its family before calling), since a surface renders replies
/// differently and the agent gets formatting guidance for it. Only families this crate knows
/// rules for return a hint; an unknown family (the common case for socket clients — GUI/TUI are
/// indistinguishable at wire v36 — and any transport without a documented rendering rule yet)
/// returns `None`. Adding a new family (Telegram, Discord, …) is one match arm.
pub fn transport_hints(family: &str) -> Option<&'static str> {
    match family {
        "matrix" => Some(
            "Session surface: a Matrix room. Replies are delivered as Matrix messages with \
             markdown formatting. Keep them concise and chat-shaped: short paragraphs, bullet \
             lists, fenced code blocks for anything long. Other room members may see your \
             replies, so avoid dumping large raw output into the room.",
        ),
        _ => None,
    }
}

/// The date-only stamp: `Conversation started: <today>`. `today` is the caller-formatted date
/// (hermes: `%A, %B %d, %Y`, e.g. `Thursday, July 09, 2026`) — this crate takes it as an input
/// so the builder stays pure. Date-ONLY on purpose: a time component would change every
/// composition and bust the provider prefix cache.
pub fn date_stamp(today: &str) -> Option<String> {
    let today = today.trim();
    if today.is_empty() {
        return None;
    }
    Some(format!("Conversation started: {today}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Tool-use enforcement constants ────────────────────────────────

    #[test]
    fn guidance_mentions_tool_calls() {
        assert!(TOOL_USE_ENFORCEMENT_GUIDANCE.contains("tool call"));
    }

    #[test]
    fn guidance_forbids_description_only() {
        assert!(TOOL_USE_ENFORCEMENT_GUIDANCE.contains("do not describe what you would do"));
    }

    #[test]
    fn guidance_requires_action() {
        assert!(TOOL_USE_ENFORCEMENT_GUIDANCE.contains("execute it now"));
    }

    #[test]
    fn enforcement_models_cover_known_families() {
        for family in [
            "gpt", "codex", "grok", "qwen", "deepseek", "gemini", "gemma", "glm",
        ] {
            assert!(TOOL_USE_ENFORCEMENT_MODELS.contains(&family), "{family}");
        }
    }

    // ── OpenAI execution guidance ─────────────────────────────────────

    #[test]
    fn openai_guidance_covers_tool_persistence() {
        assert!(OPENAI_MODEL_EXECUTION_GUIDANCE.contains("<tool_persistence>"));
        assert!(OPENAI_MODEL_EXECUTION_GUIDANCE.contains("Do not stop early"));
    }

    #[test]
    fn openai_guidance_covers_prerequisite_checks() {
        assert!(OPENAI_MODEL_EXECUTION_GUIDANCE.contains("<prerequisite_checks>"));
    }

    #[test]
    fn openai_guidance_covers_verification() {
        assert!(OPENAI_MODEL_EXECUTION_GUIDANCE.contains("<verification>"));
        assert!(OPENAI_MODEL_EXECUTION_GUIDANCE.contains("Grounding"));
    }

    #[test]
    fn openai_guidance_covers_missing_context() {
        assert!(OPENAI_MODEL_EXECUTION_GUIDANCE.contains("<missing_context>"));
        assert!(OPENAI_MODEL_EXECUTION_GUIDANCE.contains("do NOT guess"));
    }

    #[test]
    fn openai_guidance_uses_daemon_tool_names() {
        // The hermes original said `terminal` / `read_file` / `search_files`; the daemon port
        // must reference tools that actually exist here.
        assert!(OPENAI_MODEL_EXECUTION_GUIDANCE.contains("use shell"));
        assert!(OPENAI_MODEL_EXECUTION_GUIDANCE.contains("fs tool"));
        assert!(OPENAI_MODEL_EXECUTION_GUIDANCE.contains("web_search"));
        assert!(!OPENAI_MODEL_EXECUTION_GUIDANCE.contains("terminal"));
        assert!(!OPENAI_MODEL_EXECUTION_GUIDANCE.contains("read_file"));
        assert!(!OPENAI_MODEL_EXECUTION_GUIDANCE.contains("search_files"));
    }

    // ── Gating: tool_use_guidance ─────────────────────────────────────

    const TOOLS: &[&str] = &["fs", "shell"];

    #[test]
    fn off_never_emits() {
        assert!(tool_use_guidance(TOOLS, ToolUseMode::Off, "gpt-5.5").is_none());
    }

    #[test]
    fn on_emits_with_tools_only() {
        assert!(tool_use_guidance(TOOLS, ToolUseMode::On, "claude-4.6-opus").is_some());
        assert!(tool_use_guidance(&[], ToolUseMode::On, "gpt-5.5").is_none());
    }

    #[test]
    fn auto_gates_on_registry_and_model_family() {
        assert!(tool_use_guidance(TOOLS, ToolUseMode::Auto, "gpt-5.5").is_some());
        assert!(tool_use_guidance(TOOLS, ToolUseMode::Auto, "openai/GPT-5.5-codex").is_some());
        assert!(tool_use_guidance(TOOLS, ToolUseMode::Auto, "claude-4.6-opus").is_none());
        assert!(tool_use_guidance(&[], ToolUseMode::Auto, "gpt-5.5").is_none());
    }

    // ── Model-family guidance ─────────────────────────────────────────

    #[test]
    fn gpt_codex_grok_get_execution_discipline() {
        for id in ["gpt-5.5", "codex-mini", "grok-4", "openai/gpt-5.3-codex"] {
            assert_eq!(
                model_family_guidance(id).as_deref(),
                Some(OPENAI_MODEL_EXECUTION_GUIDANCE),
                "{id}"
            );
        }
    }

    #[test]
    fn gemini_gemma_get_google_directives() {
        for id in ["gemini-3-pro", "gemma-3-27b"] {
            assert_eq!(
                model_family_guidance(id).as_deref(),
                Some(GOOGLE_MODEL_OPERATIONAL_GUIDANCE),
                "{id}"
            );
        }
    }

    #[test]
    fn other_families_get_none() {
        assert!(model_family_guidance("claude-4.6-opus").is_none());
        // qwen/deepseek/glm are enforcement-only families: no operational block.
        assert!(model_family_guidance("qwen-3max").is_none());
        assert!(model_family_guidance("deepseek-v4").is_none());
    }

    // ── Environment hints ─────────────────────────────────────────────

    fn local_input() -> EnvironmentInput {
        EnvironmentInput {
            sandboxed: false,
            backend_label: None,
            backend_details: None,
            host_os: Some("Linux (6.8.0-generic)".into()),
            user_home: Some("/home/user".into()),
            workspace_cwd: Some("/ws/session".into()),
            extra_hint: None,
        }
    }

    #[test]
    fn local_backend_emits_full_host_block() {
        let result = environment_hints(&local_input()).unwrap();
        assert!(result.contains("Host: Linux (6.8.0-generic)"));
        assert!(result.contains("User home directory: /home/user"));
        assert!(result.contains("Current working directory: /ws/session"));
    }

    #[test]
    fn workspace_cwd_is_the_emitted_cwd() {
        // The struct carries only the WORKSPACE cwd — the node fills it from the session's
        // ExecutionEnvironment root, never the daemon process cwd (hermes #24882 class).
        let result = environment_hints(&local_input()).unwrap();
        assert!(result.contains("Current working directory: /ws/session"));
    }

    #[test]
    fn sandboxed_backend_suppresses_host_details() {
        let input = EnvironmentInput {
            sandboxed: true,
            backend_label: Some("docker".into()),
            ..local_input()
        };
        let result = environment_hints(&input).unwrap();
        assert!(!result.contains("Host:"));
        assert!(!result.contains("User home directory:"));
        assert!(!result.contains("/home/user"));
        assert!(result.contains("Execution backend: docker"));
        assert!(result.to_lowercase().contains("inside"));
        // Probe-less fallback tells the model how to discover backend state itself.
        assert!(result.contains("uname -a && whoami && pwd"));
    }

    #[test]
    fn sandboxed_backend_renders_probe_details_when_available() {
        let input = EnvironmentInput {
            sandboxed: true,
            backend_label: Some("modal".into()),
            backend_details: Some(
                "  OS: Linux 6.8.0\n  User: root\n  Home: /root\n  Working directory: /workspace"
                    .into(),
            ),
            ..EnvironmentInput::default()
        };
        let result = environment_hints(&input).unwrap();
        assert!(result.contains("Execution backend: modal"));
        assert!(result.contains("Linux 6.8.0"));
        assert!(result.contains("/workspace"));
    }

    #[test]
    fn extra_hint_is_appended_after_the_host_block() {
        let input = EnvironmentInput {
            extra_hint: Some("Running inside an OpenShell sandbox.".into()),
            ..local_input()
        };
        let result = environment_hints(&input).unwrap();
        assert!(result.contains("Running inside an OpenShell sandbox."));
        let host_at = result.find("Host:").unwrap();
        let extra_at = result.find("OpenShell").unwrap();
        assert!(host_at < extra_at, "factual host block must come first");
    }

    #[test]
    fn blank_extra_hint_is_ignored() {
        let input = EnvironmentInput {
            extra_hint: Some("   ".into()),
            ..local_input()
        };
        let result = environment_hints(&input).unwrap();
        assert!(!result.ends_with("\n\n"));
        assert!(result.contains("Host:"));
    }

    #[test]
    fn empty_input_emits_nothing() {
        assert!(environment_hints(&EnvironmentInput::default()).is_none());
    }

    // ── Transport hints ───────────────────────────────────────────────

    #[test]
    fn matrix_family_has_a_nonempty_surface_hint() {
        let hint = transport_hints("matrix").expect("the matrix family maps to a hint");
        assert!(hint.len() > 50);
        assert!(hint.starts_with("Session surface:"));
        assert!(hint.contains("Matrix room"));
    }

    #[test]
    fn unknown_families_compose_no_hint() {
        // Families this crate has no documented rendering rule for (socket clients, transports not
        // yet mapped, empty) return None — the node treats that as "no hint".
        for family in ["gui", "tui", "acp", "telegram", "discord", "api", ""] {
            assert!(transport_hints(family).is_none(), "{family}");
        }
    }

    // ── Date stamp ────────────────────────────────────────────────────

    #[test]
    fn date_stamp_wraps_the_caller_formatted_date() {
        assert_eq!(
            date_stamp("Thursday, July 09, 2026").as_deref(),
            Some("Conversation started: Thursday, July 09, 2026")
        );
    }

    #[test]
    fn date_stamp_empty_is_none() {
        assert!(date_stamp("").is_none());
        assert!(date_stamp("   ").is_none());
    }

    // ── Core agentic guidance ─────────────────────────────────────────

    #[test]
    fn core_guidance_is_the_task_completion_block() {
        let block = core_agentic_guidance().unwrap();
        assert!(block.starts_with("# Finishing the job"));
        assert!(block.contains("NEVER substitute"));
        assert!(block.contains("Reporting a blocker honestly"));
    }
}
