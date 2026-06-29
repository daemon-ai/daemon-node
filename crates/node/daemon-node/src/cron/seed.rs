// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The cron worker's seed-prompt assembly + run capture: build a run's seed prompt (preloaded
//! skills + chained context + payload), capture a settled run's final assistant message, run a
//! `no_agent` script, and project a [`CronSpec`] into a [`SessionOverlay`](daemon_api::SessionOverlay).

use std::path::PathBuf;

use daemon_api::CronSpec;
use daemon_common::SessionId;

use super::worker::{cap_on_boundary, CronWorker, CRON_CONTEXT_CHARS, CRON_SKILL_CHARS};

impl CronWorker {
    /// Read a settled cron session's final assistant message text from the durable verifiable
    /// journal (read-only — the same coalesced [`TranscriptBlock`](daemon_protocol::TranscriptBlock)s
    /// `session_history` serves; no lease, so it never disturbs recovery). Returns the last
    /// assistant message, size-capped on a char boundary; `None` if the session journaled none.
    pub(crate) async fn captured_output(&self, session: &SessionId) -> Option<String> {
        let stream = daemon_common::JournalStreamId::session(session);
        let mut after = 0u64;
        let mut last: Option<String> = None;
        // Page the whole stream (a cron run's transcript is short); bounded against a runaway loop.
        for _ in 0..64 {
            let page = self.store.load_journal(&stream, after, 256).await;
            if page.entries.is_empty() {
                break;
            }
            for je in &page.entries {
                let Ok(view) = daemon_telemetry::decode_entry(&je.entry.bytes) else {
                    continue;
                };
                if let daemon_telemetry::JournalPayload::Block { body } = view.payload {
                    if let Ok(daemon_protocol::TranscriptBlock::Message {
                        role: daemon_protocol::TranscriptRole::Assistant,
                        text,
                    }) = daemon_api::from_cbor::<daemon_protocol::TranscriptBlock>(&body)
                    {
                        last = Some(text);
                    }
                }
            }
            if page.next_cursor <= after || after >= page.head_cursor {
                break;
            }
            after = page.next_cursor;
        }
        last.map(|t| cap_on_boundary(t, CRON_CONTEXT_CHARS))
    }

    /// Build the seed prompt: the job payload (utf-8) preceded by any preloaded `skills` bodies
    /// (v16) and any `context_from` upstream outputs. Order is skills (instructions) → context
    /// (data) → body (task). Each injection is size-capped on a char boundary.
    pub(crate) async fn seed_prompt(&self, spec: &CronSpec) -> String {
        let body = String::from_utf8_lossy(&spec.payload).into_owned();
        let mut prefix = String::new();
        // Preloaded skills first — the agent's instructional context, mirroring a chat that had
        // `skill_view`'d them. Missing/loader-less skills are skipped (the run keeps the on-demand
        // `skill_*` tools), so preloading is best-effort.
        if !spec.skills.is_empty() {
            if let Some(loader) = &self.skill_loader {
                for name in &spec.skills {
                    if let Some(body) = loader(name) {
                        let snippet = cap_on_boundary(body, CRON_SKILL_CHARS);
                        prefix.push_str(&format!("# Skill `{name}`\n{snippet}\n\n"));
                    }
                }
            }
        }
        // Then chained upstream outputs (the latest run detail of each referenced job).
        for upstream in &spec.context_from {
            if let Some(run) = self
                .store
                .cron_runs_list(upstream, 1)
                .await
                .into_iter()
                .next()
            {
                if let Some(detail) = run.detail {
                    let snippet = cap_on_boundary(detail, CRON_CONTEXT_CHARS);
                    prefix.push_str(&format!("# Context from job `{upstream}`\n{snippet}\n\n"));
                }
            }
        }
        if prefix.is_empty() {
            body
        } else {
            format!("{prefix}{body}")
        }
    }

    /// Run a `no_agent` job's script under the contained scripts dir, returning `(ok, detail)`.
    /// Best-effort: a missing scripts dir / contained-rejected path / spawn error is a failed run.
    pub(crate) async fn run_script(&self, rel: &str) -> (bool, String) {
        let Some(dir) = &self.scripts_dir else {
            return (false, "no scripts directory configured".into());
        };
        let Ok(path) = daemon_core::exec::contain(dir, std::path::Path::new(rel)) else {
            return (false, format!("script path escapes scripts dir: {rel}"));
        };
        let out =
            tokio::task::spawn_blocking(move || std::process::Command::new(&path).output()).await;
        match out {
            Ok(Ok(output)) => {
                let mut detail = String::from_utf8_lossy(&output.stdout).into_owned();
                detail.truncate(CRON_CONTEXT_CHARS);
                (output.status.success(), detail)
            }
            Ok(Err(e)) => (false, format!("script spawn failed: {e}")),
            Err(e) => (false, format!("script join failed: {e}")),
        }
    }

    /// Project the run-shaping fields of a [`CronSpec`] into a [`SessionOverlay`](daemon_api::SessionOverlay)
    /// (Phase 2): a per-job `model`/`provider`/`workdir`/`enabled_toolsets` override layered onto the
    /// cron base profile at hydrate. Unset fields inherit.
    pub(crate) fn overlay_from_spec(spec: &CronSpec) -> daemon_api::SessionOverlay {
        daemon_api::SessionOverlay {
            model: spec.model.clone(),
            provider: spec.provider.as_deref().and_then(Self::parse_provider),
            tool_allowlist: match &spec.enabled_toolsets {
                Some(list) => daemon_api::ToolsOverride::Allowlist(list.clone()),
                None => daemon_api::ToolsOverride::Inherit,
            },
            approval_mode: None,
            workspace: spec
                .workdir
                .as_deref()
                .map(|w| daemon_common::WorkspaceBinding::Bound(PathBuf::from(w))),
        }
    }
}
