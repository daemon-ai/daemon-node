// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The resident cron scheduler (I15): the [`CronWorker`] that seeds isolated cron sessions from a
//! profile into the store and enqueues their wake, reconciling settled runs and delivering results.

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use daemon_api::{CronSpec, OverlapPolicy};
use daemon_common::{PartitionId, ProfileRef, SessionId};
use daemon_core::EngineProfile;
use daemon_host::{CronDelivery, CronFiring, CronScheduler, ServiceError};

/// The maximum lateness (seconds) a [`CatchUpPolicy::Skip`](daemon_api::CatchUpPolicy) job tolerates
/// before a missed fire is fast-forwarded instead of run. Small so Skip never catches up after real
/// downtime, but generous enough to absorb a slow tick / brief pause (`Grace` tolerates
/// `>= MIN_GRACE_SECS`, `Always` is unbounded — the three policies stay monotonically ordered).
pub(crate) const CRON_SKIP_TOLERANCE_SECS: u64 = 60;

/// Cap (bytes) on a single `context_from` chained-output injection, so a chatty upstream job cannot
/// blow up a downstream job's seed prompt.
pub(crate) const CRON_CONTEXT_CHARS: usize = 8192;

/// Cap (bytes) on a single preloaded skill body injected into a cron seed prompt (v16 `skills`), so
/// a large skill cannot blow up the prompt. Mirrors [`CRON_CONTEXT_CHARS`].
pub(crate) const CRON_SKILL_CHARS: usize = 8192;

/// The sentinel a cron agent run emits (as its entire final message) to suppress delivery — "nothing
/// worth reporting this tick" (ported from Hermes). A run whose captured output is exactly this is
/// recorded `ok` but is not delivered to any transport.
const CRON_SILENT_SENTINEL: &str = "[SILENT]";

/// Whether a captured cron run output is the `[SILENT]` delivery-suppression sentinel (trimmed).
fn is_cron_silent(text: &str) -> bool {
    text.trim() == CRON_SILENT_SENTINEL
}

/// Resolves a skill's full `SKILL.md` body by name for cron `skills` preloading. Injected into
/// [`CronWorker`] from the launch profile's [`SkillStore`](daemon_skills::SkillStore) in
/// [`assemble`](crate::assemble); `None` (no skills subsystem) makes `skills` preloading a no-op.
pub type CronSkillLoader = Arc<dyn Fn(&str) -> Option<String> + Send + Sync>;

/// Truncate `s` to at most `max` bytes without splitting a UTF-8 char (a plain `String::truncate`
/// panics on a non-boundary index). Used to cap cron seed-prompt injections.
pub(crate) fn cap_on_boundary(mut s: String, max: usize) -> String {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    s.truncate(end);
    s
}

/// The resident cron scheduler (I15): the `CronScheduler`/`CronFiring` worker. Mirrors
/// [`FleetJobWorker`](crate::FleetJobWorker) — it seeds isolated `cron_{id}_{ts}` sessions from a
/// profile into the store and enqueues their wake, leaving the existing wake-outbox dispatcher to run
/// the turn. The scheduler only computes next-fire (via `daemon-schedule`/croner) and enqueues; it
/// never runs a turn itself.
///
/// Correctness: each due job's `next_fire` is advanced **first** (at-most-once across a crash), with
/// stale-miss fast-forward vs grace catch-up; overlap is deduped per `OverlapPolicy` (which also
/// closes the manual-trigger-vs-tick double-fire race); a `repeat`-exhausted job auto-deletes. Each
/// tick also reconciles in-flight runs (a settled cron session stamps its run `finished`).
pub struct CronWorker {
    pub(crate) store: Arc<dyn daemon_store::SessionStore>,
    pub(crate) partition: PartitionId,
    /// The seed engine shape for a cron session (the orchestrator-capable profile, minus the cron
    /// tool — see the cron-session safety gate in `assemble`). The durable factory re-resolves the
    /// bound profile (`spec.target`) from `session_meta` on wake, mirroring `FleetJobWorker`.
    pub(crate) profile: EngineProfile,
    /// Root for `no_agent` scripts; a job's `script` is contained under this dir. `None` disables
    /// the script path (a `no_agent` job then records an error run).
    pub(crate) scripts_dir: Option<PathBuf>,
    /// Resolves a `CronSpec::skills` name to its body for seed-prompt preloading (v16). `None`
    /// (no skills subsystem) skips preloading — the run still sees the launch agent's `skill_*`
    /// tools + index in its profile and can `skill_view` on demand.
    pub(crate) skill_loader: Option<CronSkillLoader>,
    /// The post-settle delivery handle (Phase 2 `deliver`): pushes a finished run's captured result
    /// to its `CronSpec::deliver` transport(s) through the host's existing `DeliverySink` registry.
    /// Late-bound (the handle is `NodeApiImpl`, built after the worker) via [`set_delivery`](Self::set_delivery);
    /// unset => store-only runs (no transport delivery).
    pub(crate) delivery: std::sync::OnceLock<Arc<dyn CronDelivery>>,
}

impl CronWorker {
    /// A cron worker that seeds sessions from `profile` into `store` under `partition`.
    pub fn new(
        store: Arc<dyn daemon_store::SessionStore>,
        partition: PartitionId,
        profile: EngineProfile,
    ) -> Self {
        Self {
            store,
            partition,
            profile,
            scripts_dir: None,
            skill_loader: None,
            delivery: std::sync::OnceLock::new(),
        }
    }

    /// Set the root directory `no_agent` job scripts are resolved (and contained) under.
    pub fn with_scripts_dir(mut self, dir: PathBuf) -> Self {
        self.scripts_dir = Some(dir);
        self
    }

    /// Set the loader used to preload `CronSpec::skills` bodies into an agent run's seed prompt.
    pub fn with_skill_loader(mut self, loader: CronSkillLoader) -> Self {
        self.skill_loader = Some(loader);
        self
    }

    /// Late-bind the post-settle delivery handle (the `NodeApiImpl`, built after the worker is
    /// `Arc`-wrapped). Idempotent: the first set wins; subsequent calls are ignored.
    pub fn set_delivery(&self, delivery: Arc<dyn CronDelivery>) {
        let _ = self.delivery.set(delivery);
    }

    /// Whether a job's most recent run is still in flight: it has no `finished_unix` and its cron
    /// session is not yet settled (still `Active`/`Suspended`). Used for `OverlapPolicy` dedup.
    async fn in_flight(&self, job_id: &str) -> bool {
        let Some(run) = self
            .store
            .cron_runs_list(job_id, 1)
            .await
            .into_iter()
            .next()
        else {
            return false;
        };
        if run.finished_unix.is_some() {
            return false;
        }
        match &run.session {
            Some(session) => matches!(
                self.store.status(session).await,
                Some(daemon_store::SessionStatus::Active)
                    | Some(daemon_store::SessionStatus::Suspended { .. })
            ),
            None => false,
        }
    }

    /// Reconcile a job's latest in-flight run: if the cron session has settled (`Ready`/`Completed`/
    /// gone), stamp the run `finished` and fold the outcome into the job's `last_*` bookkeeping.
    async fn reconcile(&self, job: &daemon_store::StoredCronJob) {
        let Some(mut run) = self
            .store
            .cron_runs_list(&job.id, 1)
            .await
            .into_iter()
            .next()
        else {
            return;
        };
        if run.finished_unix.is_some() {
            return;
        }
        let Some(session) = run.session.clone() else {
            return;
        };
        let settled = !matches!(
            self.store.status(&session).await,
            Some(daemon_store::SessionStatus::Active)
                | Some(daemon_store::SessionStatus::Suspended { .. })
        );
        if !settled {
            return;
        }
        let now = Self::now_unix();
        run.finished_unix = Some(now);
        // Capture the run's real outcome: the cron session's final assistant message (read-only from
        // the durable journal). A run that produced output is `ok`; one that journaled no assistant
        // message (errored before producing output) is recorded failed. This is also what
        // `context_from` chains downstream and what the delivery step below sends.
        let captured = self.captured_output(&session).await;
        // A run whose entire output is the `[SILENT]` sentinel succeeded but reports nothing —
        // recorded `ok` but never delivered (ported from Hermes).
        let silent = captured.as_deref().is_some_and(is_cron_silent);
        let ok = captured.is_some();
        let detail = captured
            .clone()
            .unwrap_or_else(|| "no output captured".into());
        run.ok = ok;
        run.detail = Some(detail.clone());
        // Re-append the finished run (the store keys runs by job id; the latest row is updated by a
        // fresh append + bounded retention drops the stale unfinished copy on the next trim — here we
        // instead update the job's last_* directly, which is what the GUI list reads).
        let mut updated = job.clone();
        updated.last_run_unix = Some(run.started_unix);
        updated.last_ok = Some(ok);
        updated.last_detail = Some(detail);
        let _ = self.store.cron_run_append(run).await;
        let _ = self.store.cron_set(updated).await;
        // Post-settle delivery (Phase 2): push the captured result to the job's `deliver` transport(s)
        // through the host's existing `DeliverySink` registry. Suppressed for a `[SILENT]` run, a
        // failed/empty run, a store-only (`deliver = None`) job, or a node with no delivery handle.
        if ok && !silent {
            if let (Some(delivery), Some(text)) = (self.delivery.get(), captured) {
                let spec = Self::decode_spec(job);
                if let Some(deliver) = spec
                    .deliver
                    .as_deref()
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                {
                    delivery.deliver(deliver, spec.origin.as_ref(), &text).await;
                }
            }
        }
    }

    /// Materialize and fire one occurrence of `job`: a `no_agent` script run (recorded inline) or an
    /// isolated `cron_{id}_{ts}` agent session (seeded + wake-enqueued, run recorded as in-flight).
    /// `manual` marks an out-of-band `cron_trigger`. Does not touch the schedule (the caller advances).
    async fn fire(&self, job: &daemon_store::StoredCronJob, spec: &CronSpec, manual: bool) {
        let now = Self::now_unix();
        // Script-only path: run inline, record a completed run, no agent turn.
        if spec.no_agent {
            let (ok, detail) = match &spec.script {
                Some(script) => self.run_script(script).await,
                None => (false, "no_agent job has no script".into()),
            };
            let _ = self
                .store
                .cron_run_append(daemon_store::StoredCronRun {
                    job_id: job.id.clone(),
                    started_unix: now,
                    finished_unix: Some(Self::now_unix()),
                    ok,
                    detail: Some(detail.clone()),
                    session: None,
                    manual,
                })
                .await;
            let mut job = job.clone();
            job.last_run_unix = Some(now);
            job.last_ok = Some(ok);
            job.last_detail = Some(detail);
            job.fire_count = job.fire_count.saturating_add(1);
            let _ = self.store.cron_set(job).await;
            return;
        }

        // Agent path: an isolated cron session seeded with the (context-chained) payload.
        let session = SessionId::new(format!("cron_{}_{}", job.id, now));
        if self.store.status(&session).await.is_none() {
            let prompt = self.seed_prompt(spec).await;
            let mut engine = self.profile.fresh(session.clone());
            engine.push_user(daemon_protocol::UserMsg::new(prompt));
            let Ok(blob) = engine.snapshot().encode() else {
                return;
            };
            if self
                .store
                .create_session(session.clone(), self.partition, blob)
                .await
                .is_err()
            {
                return;
            }
            // Stamp the cron origin + bound profile + isolation role. `scheduled_job` tells the
            // incarnation to set `TurnTrigger::Scheduled`; the `EphemeralSubagent` role keeps the
            // cron run out of the top-level roster (it is a transient, isolated session).
            let mut meta = self.store.session_meta(&session).await.unwrap_or_default();
            meta.scheduled_job = Some(daemon_common::JobId::from(job.id.as_str()));
            meta.role = Some(daemon_store::SessionRole::EphemeralSubagent);
            // Auth 4: the spawned cron session is owned by the job's creator (captured at
            // `cron_create` from the request principal), so a scheduled run is visible to (and
            // controllable by) the user who scheduled it. `None` on legacy/system jobs.
            meta.owner = job.owner.clone();
            if let Some(target) = &spec.target {
                meta.bound_profile = Some(ProfileRef::new(target));
            }
            // Phase 2 shaping: persist the run's model/provider/toolset/workdir as a `SessionOverlay`
            // so the durable factory applies it when hydrating this cron session (see
            // `engine_incarnation::hydrate`). The constrained cron profile is the base; the overlay
            // narrows/overrides it (the resolver path is G3-safe — it never wires `cron`/`orchestrate`).
            let overlay = Self::overlay_from_spec(spec);
            if !overlay.is_empty() {
                meta.overlay = daemon_host::encode_overlay(&overlay);
            }
            let _ = self.store.set_session_meta(&session, meta).await;
        }
        let _ = self
            .store
            .cron_run_append(daemon_store::StoredCronRun {
                job_id: job.id.clone(),
                started_unix: now,
                finished_unix: None,
                ok: true,
                detail: None,
                session: Some(session.clone()),
                manual,
            })
            .await;
        let mut updated = job.clone();
        updated.last_run_unix = Some(now);
        updated.fire_count = updated.fire_count.saturating_add(1);
        let _ = self.store.cron_set(updated).await;
        // Kick the cron session into its turn via the shared wake dispatcher.
        self.store.enqueue_wake(session).await;
    }
}

#[async_trait]
impl CronScheduler for CronWorker {
    async fn tick_once(&self) -> Result<(), ServiceError> {
        let now = Self::now_unix();
        for job in self.store.cron_due(now).await {
            self.reconcile(&job).await;
            let spec = Self::decode_spec(&job);
            let schedule = match Self::schedule_of(&spec) {
                Ok(s) => s,
                Err(_) => {
                    // Unparsable schedule: clear next_fire so it stops being due (operator must fix).
                    let mut job = job.clone();
                    job.next_fire_unix = None;
                    let _ = self.store.cron_set(job).await;
                    continue;
                }
            };
            let scheduled_fire = job.next_fire_unix.unwrap_or(now);
            let in_flight = self.in_flight(&job.id).await;

            // OverlapPolicy::Queue defers (no advance) while a previous run is in flight, so the
            // occurrence runs once the prior finishes. Skip/Allow advance now (at-most-once).
            if matches!(spec.overlap, OverlapPolicy::Queue) && in_flight {
                continue;
            }

            let advanced = Self::advanced(&job, &schedule, now);
            let exhausted_oneshot = advanced.next_fire_unix.is_none();

            // Should this occurrence actually fire? (overlap dedup + catch-up grace)
            let blocked_by_overlap = in_flight && matches!(spec.overlap, OverlapPolicy::Skip);
            let fire =
                !blocked_by_overlap && Self::should_fire(&spec, &schedule, scheduled_fire, now);

            // Persist the advance first (at-most-once) unless we are firing — in which case `fire`
            // writes the updated job (fire_count/last_run) and we layer the new next_fire on top.
            if fire {
                self.fire(&advanced, &spec, false).await;
                // Re-read to fold the fire's bookkeeping, then persist the advanced next_fire.
                if let Some(mut latest) = self.store.cron_get(&job.id).await {
                    latest.next_fire_unix = advanced.next_fire_unix;
                    // repeat / auto-delete: a job that has reached its fire cap is removed.
                    if spec.repeat.is_some_and(|max| latest.fire_count >= max) || exhausted_oneshot
                    {
                        let _ = self.store.cron_remove(&job.id).await;
                    } else {
                        let _ = self.store.cron_set(latest).await;
                    }
                }
            } else {
                // Not firing (fast-forward or overlap-skip): just persist the advance, or delete an
                // exhausted one-shot that will never fire again.
                if exhausted_oneshot {
                    let _ = self.store.cron_remove(&job.id).await;
                } else {
                    let _ = self.store.cron_set(advanced).await;
                }
            }
        }
        Ok(())
    }
}

#[async_trait]
impl CronFiring for CronWorker {
    async fn fire_now(&self, id: &str) -> Result<(), ServiceError> {
        let Some(job) = self.store.cron_get(id).await else {
            return Err(ServiceError::new(format!("cron job not found: {id}")));
        };
        let spec = Self::decode_spec(&job);
        // Honor overlap dedup for the manual path too (closes the trigger-vs-tick double-fire race).
        if matches!(spec.overlap, OverlapPolicy::Skip) && self.in_flight(id).await {
            return Ok(());
        }
        self.fire(&job, &spec, true).await;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use daemon_core::{MockProvider, Provider, SystemPrompt, ToolRegistry};

    fn mock_profile() -> EngineProfile {
        EngineProfile::new(
            Arc::new(|| Arc::new(MockProvider::completing("x")) as Arc<dyn Provider>),
            Arc::new(ToolRegistry::new()),
            SystemPrompt::new("t"),
        )
    }

    #[test]
    fn cap_on_boundary_never_splits_utf8() {
        // A multi-byte char straddling the cap is dropped whole (no panic / no broken char).
        let s = "aé".to_string(); // 'é' is 2 bytes -> total len 3
        assert_eq!(cap_on_boundary(s.clone(), 2), "a");
        assert_eq!(cap_on_boundary(s, 10), "aé");
    }

    #[tokio::test]
    async fn seed_prompt_preloads_skill_bodies_ahead_of_payload() {
        let store: Arc<dyn daemon_store::SessionStore> =
            Arc::new(daemon_store::InMemoryStore::new());
        let loader: CronSkillLoader = Arc::new(|name: &str| match name {
            "briefing" => Some("BRIEFING BODY".to_string()),
            _ => None,
        });
        let worker =
            CronWorker::new(store, PartitionId::DEFAULT, mock_profile()).with_skill_loader(loader);
        let spec = CronSpec {
            name: "j".into(),
            schedule: "0 9 * * *".into(),
            payload: b"do the task".to_vec(),
            skills: vec!["briefing".into(), "missing".into()],
            ..CronSpec::default()
        };
        let prompt = worker.seed_prompt(&spec).await;
        assert!(prompt.contains("# Skill `briefing`"));
        assert!(prompt.contains("BRIEFING BODY"));
        // A skill the loader can't resolve is skipped, not errored.
        assert!(!prompt.contains("# Skill `missing`"));
        // The skill block precedes the task body.
        let skill_at = prompt.find("BRIEFING BODY").unwrap();
        let body_at = prompt.find("do the task").unwrap();
        assert!(skill_at < body_at, "skills must precede the payload");
    }

    #[tokio::test]
    async fn seed_prompt_is_just_payload_without_skills_or_context() {
        let store: Arc<dyn daemon_store::SessionStore> =
            Arc::new(daemon_store::InMemoryStore::new());
        let worker = CronWorker::new(store, PartitionId::DEFAULT, mock_profile());
        let spec = CronSpec {
            name: "j".into(),
            schedule: "0 9 * * *".into(),
            payload: b"only this".to_vec(),
            ..CronSpec::default()
        };
        assert_eq!(worker.seed_prompt(&spec).await, "only this");
    }

    #[test]
    fn silent_sentinel_is_recognized_trimmed() {
        assert!(is_cron_silent("[SILENT]"));
        assert!(is_cron_silent("  [SILENT]\n"));
        assert!(!is_cron_silent("[SILENT] but also this"));
        assert!(!is_cron_silent("all good, here is the digest"));
    }

    #[test]
    fn overlay_from_spec_projects_shaping_fields() {
        use daemon_api::ToolsOverride;
        let spec = CronSpec {
            name: "j".into(),
            schedule: "0 9 * * *".into(),
            model: Some("gpt-5".into()),
            provider: Some("openai".into()),
            enabled_toolsets: Some(vec!["fs".into(), "shell".into()]),
            workdir: Some("/srv/proj".into()),
            ..CronSpec::default()
        };
        let overlay = CronWorker::overlay_from_spec(&spec);
        assert_eq!(overlay.model.as_deref(), Some("gpt-5"));
        // The legacy adapter alias collapses to the GenAi selector.
        assert_eq!(overlay.provider, Some(daemon_api::ProviderSelector::GenAi));
        assert_eq!(
            overlay.tool_allowlist,
            ToolsOverride::Allowlist(vec!["fs".into(), "shell".into()])
        );
        assert_eq!(
            overlay.workspace,
            Some(daemon_common::WorkspaceBinding::Bound(PathBuf::from(
                "/srv/proj"
            )))
        );
    }

    #[test]
    fn overlay_from_spec_is_empty_when_unshaped() {
        let spec = CronSpec {
            name: "j".into(),
            schedule: "0 9 * * *".into(),
            ..CronSpec::default()
        };
        assert!(CronWorker::overlay_from_spec(&spec).is_empty());
    }

    #[test]
    fn parse_provider_accepts_canonical_and_aliases() {
        use daemon_api::ProviderSelector::*;
        assert_eq!(CronWorker::parse_provider("genai"), Some(GenAi));
        assert_eq!(CronWorker::parse_provider("Anthropic"), Some(GenAi));
        assert_eq!(CronWorker::parse_provider("mock"), Some(Mock));
        assert_eq!(CronWorker::parse_provider("llama_cpp"), Some(LlamaCpp));
        assert_eq!(CronWorker::parse_provider("mistral_rs"), Some(MistralRs));
        assert_eq!(CronWorker::parse_provider("nonsense"), None);
    }
}
