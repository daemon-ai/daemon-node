// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Slot adapters binding `daemon-prompt`'s content builders into the engine's composed prompt
//! (prompt-arch Phase 3): each adapter implements one of daemon-core's §10 source seams and is
//! attached by [`attach_prompt_sources`] to every full-capability engine profile the node builds
//! (per-session resolution + the orchestrator / fleet-child / fixed-session roles). The
//! background curators (skill/memory review) deliberately get none of these — internal roles run
//! persona-only prompts (the hermes `skip_context_files` analogue) and must stay byte-stable and
//! small.
//!
//! Composition-order note (intra-Guidance): the engine folds sync sources, then model-keyed
//! sources, then async sources — so the Guidance slot reads core guidance → tool-use enforcement
//! → model-family guidance → environment hints (with the LCM note, when that engine is active,
//! composed first via `ContextEngine::guidance_block`).

use std::path::PathBuf;
use std::sync::Arc;

use daemon_core::{
    AsyncPromptSource, EngineProfile, ExecutionEnvironment, ModelPromptSource, NudgeCx,
    NudgeSource, SlotKind, StablePromptSource, ToolCallObserver,
};
use daemon_prompt::{
    core_agentic_guidance, date_stamp, environment_hints, model_family_guidance, tool_use_guidance,
    ContextFilesLoader, EnvironmentInput, SubdirHintTracker, ToolUseMode, UserProfileStore,
    USER_PROFILE_NUDGE,
};

use crate::types::{PromptAssembly, PromptPolicy};

/// A fixed Guidance-slot block (the core agentic guidance, a transport hint).
pub(crate) struct StaticGuidance(pub(crate) String);

impl StablePromptSource for StaticGuidance {
    fn block(&self) -> Option<String> {
        (!self.0.is_empty()).then(|| self.0.clone())
    }
}

/// The date-only stamp, formatted at composition time (session start / model switch) — per
/// session, never per turn, so the composed prefix stays byte-stable for the session while a
/// fresh session picks up the current date. Date-ONLY on purpose (a time component would bust the
/// prefix cache every composition).
struct DateStampSource;

impl StablePromptSource for DateStampSource {
    fn block(&self) -> Option<String> {
        // hermes format: "%A, %B %d, %Y" (e.g. "Thursday, July 09, 2026").
        date_stamp(&chrono::Local::now().format("%A, %B %d, %Y").to_string())
    }

    fn slot_kind(&self) -> SlotKind {
        SlotKind::Stamp
    }
}

/// The per-profile USER.md snapshot (deduplicated, sanitized, usage-headed). Read at composition
/// boundaries only — the engine snapshots it once per session (the frozen-snapshot contract), so
/// mid-session `user_profile` writes are durable on disk but reach the prompt next session.
struct UserProfileSlotSource {
    store: Arc<UserProfileStore>,
    profile: String,
}

impl StablePromptSource for UserProfileSlotSource {
    fn block(&self) -> Option<String> {
        self.store.snapshot(&self.profile)
    }

    fn slot_kind(&self) -> SlotKind {
        SlotKind::UserProfile
    }
}

/// Tool-use enforcement guidance, keyed on the live model identity + the registry contents (the
/// hermes gating: only models on the enforcement list get it, and only when tools exist).
struct ToolUseGuidanceSource {
    tool_names: Vec<String>,
    mode: ToolUseMode,
}

impl ModelPromptSource for ToolUseGuidanceSource {
    fn block(&self, model_id: &str) -> Option<String> {
        let names: Vec<&str> = self.tool_names.iter().map(String::as_str).collect();
        tool_use_guidance(&names, self.mode, model_id)
    }
}

/// Model-family operational guidance (GPT/Codex/Grok execution discipline, Gemini/Gemma
/// directives), keyed on the live model identity so a live model switch re-keys it.
struct ModelFamilySource;

impl ModelPromptSource for ModelFamilySource {
    fn block(&self, model_id: &str) -> Option<String> {
        model_family_guidance(model_id)
    }
}

/// Environment hints: host facts captured once at node assembly (OS/arch, home) plus the
/// session's WORKSPACE cwd read from its execution environment at composition time — the agent's
/// tools operate there, never in the daemon process cwd (hermes #24882 class).
struct EnvHintsSource {
    host_os: Option<String>,
    user_home: Option<String>,
}

#[async_trait::async_trait]
impl AsyncPromptSource for EnvHintsSource {
    async fn block(&self, exec: &dyn ExecutionEnvironment) -> Option<String> {
        environment_hints(&EnvironmentInput {
            // Every current backend is a local contained workspace; a remote/sandboxed backend
            // sets `sandboxed` + backend details when one lands.
            sandboxed: false,
            backend_label: None,
            backend_details: None,
            host_os: self.host_os.clone(),
            user_home: self.user_home.clone(),
            workspace_cwd: Some(exec.cwd().display().to_string()),
            extra_hint: None,
        })
    }
}

/// The workspace context files (`DAEMON.md` > `AGENTS.md` > `CLAUDE.md` > `.cursorrules`),
/// loaded through the session's execution environment at composition time and snapshotted for
/// the session (cache-stable).
struct ContextFilesSource {
    loader: ContextFilesLoader,
}

#[async_trait::async_trait]
impl AsyncPromptSource for ContextFilesSource {
    async fn block(&self, exec: &dyn ExecutionEnvironment) -> Option<String> {
        // The session cwd IS the workspace root (relative "" against the environment).
        self.loader.build(exec, std::path::Path::new("")).await
    }

    fn slot_kind(&self) -> SlotKind {
        SlotKind::ContextFiles
    }
}

/// The USER.md save nudge: fires every `interval` user turns (the [`daemon_prompt::NudgeCounter`]
/// cadence, derived statelessly from the conversation's user-turn count so it self-hydrates on
/// restore), and only while the store actually has a live profile to save into.
struct UserProfileNudge {
    interval: u32,
}

impl NudgeSource for UserProfileNudge {
    fn nudge(&self, cx: &NudgeCx) -> Option<String> {
        if self.interval == 0 || !cx.user_turns.is_multiple_of(u64::from(self.interval)) {
            return None;
        }
        Some(USER_PROFILE_NUDGE.to_string())
    }
}

/// The per-surface transport hint as an origin-aware [`NudgeSource`] (prompt-arch, `[prompt]`-gated):
/// on a turn opened by a routed submit it injects the formatting guidance daemon-prompt knows for
/// that submit's origin transport — Matrix today. It is per-TURN by construction (keyed on
/// [`NudgeCx::origin`], the one-shot origin of *this* submit), so it is correct across the ways a
/// session is multi-transport (GUI `Submit` to any session, chat pins, `Handover`, rooms fan-out):
/// a no-origin turn (durable rehydrate, injected store inputs, background completions, cron, steer,
/// observe) carries no origin and therefore composes nothing. Socket clients compose none (GUI/TUI
/// are indistinguishable at wire v36), and any transport without a documented rendering rule maps to
/// nothing — the family match lives in `daemon_prompt::transport_hints`.
struct TransportHintSource;

impl NudgeSource for TransportHintSource {
    fn nudge(&self, cx: &NudgeCx) -> Option<String> {
        // Transport ids are instance-qualified (`matrix/@bot:hs`); the FAMILY segment keys the
        // hint (node-side policy — splitting is not daemon-prompt's concern).
        let family = cx.origin?.as_str().split('/').next().unwrap_or_default();
        daemon_prompt::transport_hints(family).map(str::to_string)
    }
}

/// The mid-session subdirectory hint tracker (hermes `subdirectory_hints.py`) behind the engine's
/// [`ToolCallObserver`] seam: watches executed tool calls for newly-visited workspace
/// subdirectories and returns each directory's context-file hint exactly once; the engine appends
/// it to the triggering call's result. Per-session state (the load-once set) lives behind a
/// `tokio::sync::Mutex` since the seam is `&self`.
struct SubdirHints {
    tracker: tokio::sync::Mutex<SubdirHintTracker>,
}

impl SubdirHints {
    fn new(root: PathBuf) -> Self {
        Self {
            tracker: tokio::sync::Mutex::new(SubdirHintTracker::new(root)),
        }
    }
}

#[async_trait::async_trait]
impl ToolCallObserver for SubdirHints {
    async fn on_tool_call(
        &self,
        exec: &dyn ExecutionEnvironment,
        name: &str,
        args_json: &str,
    ) -> Option<String> {
        let args: serde_json::Value = serde_json::from_str(args_json).ok()?;
        self.tracker
            .lock()
            .await
            .on_tool_call(exec, name, &args)
            .await
    }
}

/// Attach the full prompt-architecture source set to a full-capability engine profile: guidance
/// blocks (core / tool-use / model-family / environment), the workspace context files + the
/// subdirectory hint observer, the USER.md snapshot + save nudge (scoped to `profile_id`), and
/// the date stamp — each gated by the `[prompt]` policy. Internal roles (curators/reviewers)
/// deliberately never call this.
pub(crate) fn attach_prompt_sources(
    mut profile: EngineProfile,
    prompt: &PromptAssembly,
    profile_id: &str,
) -> EngineProfile {
    let policy: &PromptPolicy = &prompt.policy;
    if policy.core_guidance {
        if let Some(text) = core_agentic_guidance() {
            profile = profile.with_prompt_block(Arc::new(StaticGuidance(text)));
        }
    }
    let tool_names = profile.registry().names();
    profile = profile.with_model_prompt_block(Arc::new(ToolUseGuidanceSource {
        tool_names,
        mode: policy.tool_use_guidance,
    }));
    if policy.model_guidance {
        profile = profile.with_model_prompt_block(Arc::new(ModelFamilySource));
    }
    if policy.environment_hints {
        profile = profile.with_async_prompt_block(Arc::new(EnvHintsSource {
            host_os: Some(format!(
                "{} ({})",
                std::env::consts::OS,
                std::env::consts::ARCH
            )),
            user_home: std::env::var("HOME").ok().filter(|h| !h.is_empty()),
        }));
    }
    if policy.context_files {
        profile = profile.with_async_prompt_block(Arc::new(ContextFilesSource {
            loader: ContextFilesLoader::with_max_chars(policy.context_file_max_chars),
        }));
        // Mid-session subdirectory hints share the context-files gate: a fresh tracker per
        // session, rooted at the engine's resolved workspace (the exec env's cwd).
        profile = profile.with_tool_observers(Arc::new(
            |_session: &daemon_common::SessionId, exec: &Arc<dyn ExecutionEnvironment>| {
                vec![Arc::new(SubdirHints::new(exec.cwd().to_path_buf()))
                    as Arc<dyn ToolCallObserver>]
            },
        ));
    }
    if let Some(store) = &prompt.user_profiles {
        profile = profile.with_prompt_block(Arc::new(UserProfileSlotSource {
            store: store.clone(),
            profile: profile_id.to_string(),
        }));
        if policy.nudge_interval > 0 {
            profile = profile.with_nudge_source(Arc::new(UserProfileNudge {
                interval: policy.nudge_interval,
            }));
        }
    }
    if policy.transport_hints {
        // Origin-aware, per-turn: injects the surface hint only on a turn opened by a routed
        // submit whose origin family daemon-prompt knows rules for. No-origin turns compose none.
        profile = profile.with_nudge_source(Arc::new(TransportHintSource));
    }
    if policy.date_stamp {
        profile = profile.with_prompt_block(Arc::new(DateStampSource));
    }
    profile
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn static_guidance_composes_nonempty_only() {
        assert_eq!(
            StaticGuidance("hello".into()).block().as_deref(),
            Some("hello")
        );
        assert!(StaticGuidance(String::new()).block().is_none());
        assert_eq!(StaticGuidance("x".into()).slot_kind(), SlotKind::Guidance);
    }

    #[test]
    fn date_stamp_is_date_only_and_owns_the_stamp_slot() {
        let source = DateStampSource;
        let block = source.block().expect("a date always renders");
        assert!(block.starts_with("Conversation started: "));
        // Date-only: no clock component may leak in (it would bust the prefix cache).
        assert!(!block.contains(':') || block.matches(':').count() == 1);
        assert!(!block.contains("AM") && !block.contains("PM"));
        assert_eq!(source.slot_kind(), SlotKind::Stamp);
    }

    #[test]
    fn user_profile_slot_reads_the_profile_snapshot() {
        let dir = tempfile::tempdir().unwrap();
        let store =
            Arc::new(UserProfileStore::open(dir.path(), daemon_prompt::DEFAULT_USER_CAP).unwrap());
        store.add("opus", "prefers rust").unwrap();
        let source = UserProfileSlotSource {
            store: store.clone(),
            profile: "opus".into(),
        };
        let block = source.block().unwrap();
        assert!(block.contains("USER PROFILE"));
        assert!(block.contains("prefers rust"));
        assert_eq!(source.slot_kind(), SlotKind::UserProfile);
        // Another profile's slot is isolated.
        let other = UserProfileSlotSource {
            store,
            profile: "other".into(),
        };
        assert!(other.block().is_none());
    }

    #[test]
    fn tool_use_guidance_gates_on_registry_and_model() {
        let with_tools = ToolUseGuidanceSource {
            tool_names: vec!["fs".into(), "shell".into()],
            mode: ToolUseMode::Auto,
        };
        assert!(with_tools.block("gpt-5.5").is_some());
        assert!(with_tools.block("claude-4.6-opus").is_none());
        let no_tools = ToolUseGuidanceSource {
            tool_names: Vec::new(),
            mode: ToolUseMode::Auto,
        };
        assert!(no_tools.block("gpt-5.5").is_none(), "no tools, no block");
    }

    /// A `NudgeCx` at cadence position `user_turns` with no origin (the cadence-only case).
    fn cadence(user_turns: u64) -> NudgeCx<'static> {
        NudgeCx {
            user_turns,
            origin: None,
        }
    }

    #[test]
    fn nudge_fires_on_the_interval_only() {
        let nudge = UserProfileNudge { interval: 3 };
        assert!(nudge.nudge(&cadence(1)).is_none());
        assert!(nudge.nudge(&cadence(2)).is_none());
        assert_eq!(
            nudge.nudge(&cadence(3)).as_deref(),
            Some(USER_PROFILE_NUDGE)
        );
        assert!(nudge.nudge(&cadence(4)).is_none());
        assert!(nudge.nudge(&cadence(6)).is_some());
        let disabled = UserProfileNudge { interval: 0 };
        assert!((0..10).all(|n| disabled.nudge(&cadence(n)).is_none()));
    }

    #[test]
    fn transport_hint_maps_by_family_and_ignores_origin_less_turns() {
        let source = TransportHintSource;
        // Instance-qualified matrix ids (`matrix/<mxid>`, the adapter's stamp) map by family.
        for id in ["matrix", "matrix/@bot:hs.org"] {
            let origin = daemon_protocol::TransportId::new(id);
            let hint = source
                .nudge(&NudgeCx {
                    user_turns: 1,
                    origin: Some(&origin),
                })
                .expect("the matrix family injects a hint");
            assert!(hint.contains("Matrix room"), "{id}");
        }
        // A socket/unmapped family composes nothing.
        let api = daemon_protocol::TransportId::new("api");
        assert!(source
            .nudge(&NudgeCx {
                user_turns: 1,
                origin: Some(&api),
            })
            .is_none());
        let telegram = daemon_protocol::TransportId::new("telegram/bot-1");
        assert!(source
            .nudge(&NudgeCx {
                user_turns: 1,
                origin: Some(&telegram),
            })
            .is_none());
        // No origin (durable rehydrate, background completion, steer, observe): no hint, ever.
        assert!(source.nudge(&cadence(1)).is_none());
    }

    /// End-to-end over a real exec env: the async sources read the WORKSPACE (context files from
    /// the root, the cwd hint naming it), and the subdir observer hints exactly once per
    /// directory.
    #[tokio::test]
    async fn async_sources_and_subdir_hints_read_the_workspace() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("AGENTS.md"), "Root rules: be tidy.").unwrap();
        std::fs::create_dir_all(dir.path().join("sub")).unwrap();
        std::fs::write(
            dir.path().join("sub/AGENTS.md"),
            "Subdir rules: extra care.",
        )
        .unwrap();
        let exec = daemon_core::LocalEnvironment::new(dir.path().to_path_buf());

        let ctx = ContextFilesSource {
            loader: ContextFilesLoader::with_max_chars(20_000),
        };
        let block = ctx.block(&exec).await.expect("AGENTS.md loads");
        assert!(block.starts_with("# Project Context"));
        assert!(block.contains("Root rules: be tidy."));

        let env = EnvHintsSource {
            host_os: Some("linux (x86_64)".into()),
            user_home: Some("/home/u".into()),
        };
        let hints = env.block(&exec).await.unwrap();
        assert!(hints.contains("Current working directory:"));
        assert!(hints.contains(&dir.path().display().to_string()));

        let observer = SubdirHints::new(exec.cwd().to_path_buf());
        let args = serde_json::json!({"op": "read", "path": "sub/main.rs"}).to_string();
        let hint = observer
            .on_tool_call(&exec, "fs", &args)
            .await
            .expect("first touch of sub/ yields its AGENTS.md hint");
        assert!(hint.contains("Subdir rules: extra care."));
        assert!(
            observer.on_tool_call(&exec, "fs", &args).await.is_none(),
            "load-once per directory"
        );
    }
}
