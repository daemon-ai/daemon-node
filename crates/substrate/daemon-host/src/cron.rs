// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `CronOps` — the shared cron-store surface (I15) behind both `NodeApiImpl`'s `ControlApi` cron
//! ops and the agent-facing `cron` tool.
//!
//! It owns the durable store CRUD, the schedule validation + next-fire computation (via
//! `daemon-schedule`), and the wire `CronSpec`/`CronJob`/`CronRun`/`CronSuggestion` (de)serialization
//! (the store keeps the spec as an opaque CBOR blob, protocol-free). The actual *firing* of a job —
//! materializing an isolated cron session — is delegated to an injected [`CronFiring`](crate::CronFiring)
//! handle (the node's `CronWorker`); `CronOps` never builds an engine itself, so it stays free of the
//! engine/composition layer. A single surface means the operator (`cron_*` control ops), the agent
//! tool, and the catalog all create through the exact same validation + id discipline.

use crate::CronFiring;
use daemon_api::{
    from_cbor, to_cbor, ApiError, ChatRoute, CronJob, CronRun, CronSpec, CronSuggestion,
    RunTrigger, SuggestionStatus,
};
use daemon_schedule::Schedule;
use daemon_store::{
    SessionStore, StoredCronJob, StoredCronRun, StoredCronSuggestion, CRON_RUN_RETENTION,
};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

/// Process-unique counter mixed into generated job/suggestion ids so two creates within the same
/// nanosecond still get distinct ids.
static ID_SALT: AtomicU64 = AtomicU64::new(0);

/// A late-bound source of blueprint-derived [`CronSuggestion`]s (the `metadata.daemon.blueprint`
/// skill bridge). Injected from the node so `CronOps` stays free of the skills subsystem; called
/// (cheaply) during suggestion seeding so a freshly-installed blueprint skill surfaces as a pending
/// suggestion without a restart.
pub type BlueprintSource = Arc<dyn Fn() -> Vec<CronSuggestion> + Send + Sync>;

/// The shared cron operations surface over a durable [`SessionStore`].
#[derive(Clone)]
pub struct CronOps {
    store: Arc<dyn SessionStore>,
    /// The manual-fire seam (the node's `CronWorker`). `None` => `trigger` is unsupported (a node
    /// with the store surface but no resident scheduler).
    firing: Option<Arc<dyn CronFiring>>,
    /// Source of blueprint-skill suggestions, scanned during seeding. `None` => no skill bridge.
    blueprints: Option<BlueprintSource>,
}

impl CronOps {
    /// A cron surface over `store` with no firing handle (trigger unsupported).
    pub fn new(store: Arc<dyn SessionStore>) -> Self {
        Self {
            store,
            firing: None,
            blueprints: None,
        }
    }

    /// Attach the manual-fire seam so `trigger` materializes a run.
    pub fn with_firing(mut self, firing: Arc<dyn CronFiring>) -> Self {
        self.firing = Some(firing);
        self
    }

    /// Attach the blueprint-skill suggestion source (the `metadata.daemon.blueprint` bridge).
    pub fn with_blueprints(mut self, source: BlueprintSource) -> Self {
        self.blueprints = Some(source);
        self
    }

    /// Resolve the originating [`Origin`](daemon_protocol::Origin) of a session by reverse-scanning
    /// the durable chat→session routing pins (§5.9, I5): the pin whose `session_id` matches carries
    /// the creating chat's `Origin` in its CBOR descriptor. The agent `cron` tool calls this to stamp
    /// `CronSpec::origin` (wire v17) at create time, so a run's `deliver = "origin"` can route its
    /// result back to the chat that asked for the job. `None` when the session has no routing pin
    /// (a deterministic/unpinned session or a CLI/operator create) — `"origin"` then falls back to
    /// store-only.
    pub async fn origin_for_session(
        &self,
        session: &daemon_common::SessionId,
    ) -> Option<daemon_protocol::Origin> {
        self.store
            .routing_list()
            .await
            .iter()
            .find(|r| &r.session_id == session)
            .and_then(|r| from_cbor::<ChatRoute>(&r.descriptor).ok())
            .map(|route| route.origin)
    }

    fn now_unix() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }

    /// Mint a fresh opaque id with the given prefix (`cron`/`sug`), unique per process+nanosecond.
    fn gen_id(prefix: &str) -> String {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let salt = ID_SALT.fetch_add(1, Ordering::Relaxed);
        format!("{prefix}-{nanos:x}{salt:x}")
    }

    /// Validate a spec and return its parsed [`Schedule`]. The single gate every create/update runs:
    /// the schedule must parse, a `no_agent` job must carry a `script`, and the timezone (if any)
    /// must be a valid IANA name.
    pub fn validate(spec: &CronSpec) -> Result<Schedule, ApiError> {
        if spec.no_agent && spec.script.as_deref().unwrap_or("").trim().is_empty() {
            return Err(ApiError::Other(
                "no_agent cron job requires a script".into(),
            ));
        }
        Schedule::parse(&spec.schedule)
            .and_then(|s| s.with_timezone(spec.timezone.as_deref()))
            .map(|s| s.with_repeat(spec.repeat).with_jitter(spec.jitter_secs))
            .map_err(|e| ApiError::Other(format!("invalid cron schedule: {e}")))
    }

    /// The first fire of a freshly-(re)scheduled job: next occurrence strictly after `now`, plus the
    /// schedule's jitter. `None` for an already-past one-shot.
    fn first_fire(schedule: &Schedule, now: u64) -> Option<u64> {
        schedule
            .next_after(now)
            .map(|t| t.saturating_add(schedule.jitter_offset(t)))
    }

    /// Project a stored job to its wire [`CronJob`].
    fn to_wire_job(stored: &StoredCronJob) -> CronJob {
        let spec = from_cbor::<CronSpec>(&stored.spec).unwrap_or_default();
        CronJob {
            id: stored.id.clone(),
            spec,
            next_fire_unix: stored.next_fire_unix,
            paused: stored.paused,
            last_run_unix: stored.last_run_unix,
            last_ok: stored.last_ok,
            last_detail: stored.last_detail.clone(),
            fire_count: stored.fire_count,
        }
    }

    /// Project a stored run to its wire [`CronRun`].
    fn to_wire_run(stored: StoredCronRun) -> CronRun {
        CronRun {
            started_unix: stored.started_unix,
            ok: stored.ok,
            detail: stored.detail,
            finished_unix: stored.finished_unix,
            session: stored.session,
            trigger: if stored.manual {
                RunTrigger::Manual
            } else {
                RunTrigger::Scheduled
            },
        }
    }

    /// Project a stored suggestion to its wire [`CronSuggestion`].
    fn to_wire_suggestion(stored: &StoredCronSuggestion) -> CronSuggestion {
        CronSuggestion {
            id: stored.id.clone(),
            title: stored.title.clone(),
            description: stored.description.clone(),
            source: stored.source.clone(),
            spec: from_cbor::<CronSpec>(&stored.spec).unwrap_or_default(),
            dedup_key: stored.dedup_key.clone(),
            status: parse_status(&stored.status),
        }
    }

    /// List every scheduled job.
    pub async fn list(&self) -> Vec<CronJob> {
        self.store
            .cron_list()
            .await
            .iter()
            .map(Self::to_wire_job)
            .collect()
    }

    /// Create a job from `spec`, returning the new opaque id. Validates the schedule, computes the
    /// first fire, and persists the job (paused iff `!spec.enabled`).
    pub async fn create(&self, spec: CronSpec) -> Result<String, ApiError> {
        require_operator_for_security_fields(&spec)?;
        let schedule = Self::validate(&spec)?;
        let now = Self::now_unix();
        let id = Self::gen_id("cron");
        let next_fire_unix = if spec.enabled {
            Self::first_fire(&schedule, now)
        } else {
            None
        };
        let stored = StoredCronJob {
            id: id.clone(),
            schedule: spec.schedule.clone(),
            spec: to_cbor(&spec),
            next_fire_unix,
            paused: !spec.enabled,
            last_run_unix: None,
            last_ok: None,
            last_detail: None,
            fire_count: 0,
            created_unix: now,
            // Auth 4: record the creating principal so the worker can stamp each spawned cron
            // session's owner. `None` for a principal-less (system/local) create.
            owner: crate::request_context::current_principal().map(|p| p.user_id),
        };
        self.store
            .cron_set(stored)
            .await
            .map_err(|e| ApiError::Other(format!("cron store: {e}")))?;
        Ok(id)
    }

    /// Replace an existing job's spec (preserving run bookkeeping), recomputing its next fire.
    pub async fn update(&self, id: String, spec: CronSpec) -> Result<(), ApiError> {
        require_operator_for_security_fields(&spec)?;
        let schedule = Self::validate(&spec)?;
        let Some(mut stored) = self.store.cron_get(&id).await else {
            return Err(ApiError::Other(format!("unknown cron job: {id}")));
        };
        let now = Self::now_unix();
        stored.schedule = spec.schedule.clone();
        stored.paused = !spec.enabled;
        stored.next_fire_unix = if spec.enabled {
            Self::first_fire(&schedule, now)
        } else {
            None
        };
        stored.spec = to_cbor(&spec);
        self.store
            .cron_set(stored)
            .await
            .map_err(|e| ApiError::Other(format!("cron store: {e}")))
    }

    /// Delete a job (idempotent).
    pub async fn delete(&self, id: String) -> Result<(), ApiError> {
        self.store
            .cron_remove(&id)
            .await
            .map_err(|e| ApiError::Other(format!("cron store: {e}")))
    }

    /// Pause or resume a job. Resuming recomputes the next fire from now.
    pub async fn pause(&self, id: String, paused: bool) -> Result<(), ApiError> {
        let Some(mut stored) = self.store.cron_get(&id).await else {
            return Err(ApiError::Other(format!("unknown cron job: {id}")));
        };
        stored.paused = paused;
        if paused {
            stored.next_fire_unix = None;
        } else {
            // Resume: re-parse the spec and recompute the next fire from now.
            let spec = from_cbor::<CronSpec>(&stored.spec).unwrap_or_default();
            if let Ok(schedule) = Self::validate(&spec) {
                stored.next_fire_unix = Self::first_fire(&schedule, Self::now_unix());
            }
        }
        self.store
            .cron_set(stored)
            .await
            .map_err(|e| ApiError::Other(format!("cron store: {e}")))
    }

    /// Fire a job now (manual). Requires an injected firing handle.
    pub async fn trigger(&self, id: String) -> Result<(), ApiError> {
        match &self.firing {
            Some(firing) => firing
                .fire_now(&id)
                .await
                .map_err(|e| ApiError::Other(format!("cron fire: {e}"))),
            None => Err(ApiError::Unsupported("cron_trigger".into())),
        }
    }

    /// A job's recent runs (newest first, bounded by the store's retention).
    pub async fn runs(&self, id: String) -> Vec<CronRun> {
        self.store
            .cron_runs_list(&id, CRON_RUN_RETENTION)
            .await
            .into_iter()
            .map(Self::to_wire_run)
            .collect()
    }

    /// The pending (un-acted) suggestions. Seeds the built-in starter catalog first (idempotent +
    /// latched by `dedup_key`), so a fresh node offers the starters and an accepted/dismissed one is
    /// never re-offered (H1/H2 consent-first UX).
    pub async fn suggestions(&self) -> Vec<CronSuggestion> {
        self.seed_catalog().await;
        self.store
            .cron_suggestions_list()
            .await
            .iter()
            .filter(|s| s.status == "pending")
            .map(Self::to_wire_suggestion)
            .collect()
    }

    /// Seed the built-in starter catalog ([`crate::cron_catalog::starter_suggestions`]) plus any
    /// blueprint-skill suggestions (the `metadata.daemon.blueprint` bridge) as `Pending`. Idempotent:
    /// a suggestion whose `dedup_key` already exists (in any state) is skipped, so this never
    /// resurrects an accepted/dismissed one and is cheap to call repeatedly.
    pub async fn seed_catalog(&self) {
        let existing: std::collections::HashSet<String> = self
            .store
            .cron_suggestions_list()
            .await
            .into_iter()
            .map(|s| s.dedup_key)
            .collect();
        let blueprint_suggestions = match &self.blueprints {
            Some(source) => source(),
            None => Vec::new(),
        };
        for suggestion in crate::cron_catalog::starter_suggestions()
            .into_iter()
            .chain(blueprint_suggestions)
        {
            if existing.contains(&suggestion.dedup_key) {
                continue;
            }
            let _ = self.upsert_suggestion(suggestion).await;
        }
    }

    /// Upsert a suggestion (used by the catalog/blueprint seeding). Latched per `dedup_key`: a
    /// suggestion whose key was already accepted/dismissed is not re-offered (re-inserting it as
    /// pending is a no-op).
    pub async fn upsert_suggestion(&self, suggestion: CronSuggestion) -> Result<(), ApiError> {
        // Latch: if any existing suggestion shares this dedup_key and is not pending, skip.
        let latched = self
            .store
            .cron_suggestions_list()
            .await
            .into_iter()
            .any(|s| s.dedup_key == suggestion.dedup_key && s.status != "pending");
        if latched {
            return Ok(());
        }
        let stored = StoredCronSuggestion {
            id: suggestion.id.clone(),
            title: suggestion.title,
            description: suggestion.description,
            source: suggestion.source,
            spec: to_cbor(&suggestion.spec),
            dedup_key: suggestion.dedup_key,
            status: status_str(suggestion.status).into(),
            created_unix: Self::now_unix(),
        };
        self.store
            .cron_suggestion_set(stored)
            .await
            .map_err(|e| ApiError::Other(format!("cron store: {e}")))
    }

    /// Accept a suggestion: create the backing job (returning its id) and latch the suggestion
    /// `accepted` so it is never re-offered.
    pub async fn accept_suggestion(&self, id: String) -> Result<String, ApiError> {
        let Some(mut stored) = self.store.cron_suggestion_get(&id).await else {
            return Err(ApiError::Other(format!("unknown cron suggestion: {id}")));
        };
        let spec = from_cbor::<CronSpec>(&stored.spec).unwrap_or_default();
        let job_id = self.create(spec).await?;
        stored.status = "accepted".into();
        self.store
            .cron_suggestion_set(stored)
            .await
            .map_err(|e| ApiError::Other(format!("cron store: {e}")))?;
        Ok(job_id)
    }

    /// Dismiss a suggestion: latch it `dismissed` (kept in the store so its `dedup_key` is not
    /// re-offered).
    pub async fn dismiss_suggestion(&self, id: String) -> Result<(), ApiError> {
        let Some(mut stored) = self.store.cron_suggestion_get(&id).await else {
            return Err(ApiError::Other(format!("unknown cron suggestion: {id}")));
        };
        stored.status = "dismissed".into();
        self.store
            .cron_suggestion_set(stored)
            .await
            .map_err(|e| ApiError::Other(format!("cron store: {e}")))
    }
}

/// Cluster E (policy partition): the security-relevant `CronSpec` fields — `workdir` (projected to a
/// `WorkspaceBinding::Bound` at hydrate, a per-session sandbox escape) and `enabled_toolsets` (pins an
/// unattended run's tool surface) — require an operator-tier capability
/// ([`SessionControlAny`](daemon_auth::Capability::SessionControlAny)). This is the single choke point
/// for both control-plane cron ops and `accept_suggestion` (which calls `create`); the agent `cron`
/// tool additionally refuses these fields outright as defense in depth. Fail-closed on no principal.
fn require_operator_for_security_fields(spec: &CronSpec) -> Result<(), ApiError> {
    let sets_workdir = spec
        .workdir
        .as_deref()
        .is_some_and(|w| !w.trim().is_empty());
    let sets_toolset = spec
        .enabled_toolsets
        .as_ref()
        .is_some_and(|t| !t.is_empty());
    if !(sets_workdir || sets_toolset) {
        return Ok(());
    }
    match crate::request_context::current_principal() {
        Some(p) if p.has(daemon_auth::Capability::SessionControlAny) => Ok(()),
        Some(_) => Err(ApiError::Forbidden(
            "cron workdir/enabled_toolsets require an operator-tier capability".into(),
        )),
        None => Err(ApiError::Unauthenticated(
            "no authenticated principal bound to this request".into(),
        )),
    }
}

/// Map a wire [`SuggestionStatus`] to its stored string.
fn status_str(status: SuggestionStatus) -> &'static str {
    match status {
        SuggestionStatus::Pending => "pending",
        SuggestionStatus::Accepted => "accepted",
        SuggestionStatus::Dismissed => "dismissed",
    }
}

/// Parse a stored status string back to the wire [`SuggestionStatus`].
fn parse_status(s: &str) -> SuggestionStatus {
    match s {
        "accepted" => SuggestionStatus::Accepted,
        "dismissed" => SuggestionStatus::Dismissed,
        _ => SuggestionStatus::Pending,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use daemon_store::InMemoryStore;

    /// A `CronOps` over a fresh in-memory store, plus the store handle for direct assertions.
    fn ops() -> (CronOps, Arc<dyn SessionStore>) {
        let store: Arc<dyn SessionStore> = Arc::new(InMemoryStore::new());
        (CronOps::new(store.clone()), store)
    }

    fn spec(schedule: &str, enabled: bool) -> CronSpec {
        CronSpec {
            name: "job".into(),
            schedule: schedule.into(),
            enabled,
            ..CronSpec::default()
        }
    }

    /// `test_scheduled_task.c` `/schedule/normal` (the "scheduled" state): an enabled create parses
    /// the schedule and arms a first fire.
    #[tokio::test]
    async fn create_enabled_sets_next_fire() {
        let (ops, store) = ops();
        let id = ops.create(spec("@every 1h", true)).await.expect("create");
        let jobs = ops.list().await;
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].id, id);
        assert!(!jobs[0].paused);
        assert!(
            jobs[0].next_fire_unix.is_some(),
            "an enabled job is armed with a first fire"
        );
        // It is due once we advance past the fire (the store's `cron_due` selects it).
        let fire = jobs[0].next_fire_unix.unwrap();
        assert_eq!(store.cron_due(fire).await.len(), 1);
    }

    /// `/scheduled-task/new` + `/properties` (default state UNSCHEDULED, no execute-at): a disabled
    /// create is stored paused with no armed fire — the daemon "unscheduled" analogue.
    #[tokio::test]
    async fn create_disabled_has_no_next_fire() {
        let (ops, store) = ops();
        ops.create(spec("@every 1h", false)).await.expect("create");
        let jobs = ops.list().await;
        assert_eq!(jobs.len(), 1);
        assert!(jobs[0].paused);
        assert!(
            jobs[0].next_fire_unix.is_none(),
            "unscheduled: no execute-at"
        );
        // Never due while unscheduled.
        assert!(store.cron_due(u64::MAX).await.is_empty());
    }

    /// `/scheduled-task/schedule/cancelled` (cancel → does not fire): pausing clears the armed fire
    /// so the job is excluded from `cron_due`; resuming recomputes it.
    #[tokio::test]
    async fn pause_clears_next_fire_resume_recomputes() {
        let (ops, store) = ops();
        let id = ops.create(spec("@every 1h", true)).await.expect("create");
        assert!(!store.cron_due(u64::MAX).await.is_empty(), "armed => due");

        // Cancel.
        ops.pause(id.clone(), true).await.expect("pause");
        let job = store.cron_get(&id).await.expect("still present");
        assert!(job.paused);
        assert!(job.next_fire_unix.is_none());
        assert!(
            store.cron_due(u64::MAX).await.is_empty(),
            "a cancelled job never fires"
        );

        // Resume re-arms.
        ops.pause(id.clone(), false).await.expect("resume");
        let job = store.cron_get(&id).await.expect("present");
        assert!(!job.paused);
        assert!(job.next_fire_unix.is_some(), "resume recomputes the fire");
    }

    /// `/scheduled-task/schedule/reschedule` (reschedule replaces the previous execute-at): update
    /// replaces the spec + recomputes the fire in place — still exactly one job.
    #[tokio::test]
    async fn update_replaces_schedule_single_job() {
        let (ops, store) = ops();
        let id = ops.create(spec("@every 1h", true)).await.expect("create");
        let before = store.cron_get(&id).await.unwrap().next_fire_unix.unwrap();

        ops.update(id.clone(), spec("@every 30m", true))
            .await
            .expect("update");
        let jobs = ops.list().await;
        assert_eq!(jobs.len(), 1, "reschedule replaces in place, no duplicate");
        assert_eq!(jobs[0].id, id);
        assert_eq!(jobs[0].spec.schedule, "@every 30m");
        let after = store.cron_get(&id).await.unwrap().next_fire_unix.unwrap();
        assert!(
            after <= before,
            "a 30m reschedule fires no later than the previous 1h fire"
        );
    }

    /// `update`/`pause` on an unknown job id error cleanly (daemon-native edge); `delete` is
    /// idempotent (a no-op success), mirroring the store's tolerant remove.
    #[tokio::test]
    async fn mutate_unknown_job_errors() {
        let (ops, _store) = ops();
        assert!(ops
            .update("nope".into(), spec("@every 1h", true))
            .await
            .is_err());
        assert!(ops.pause("nope".into(), true).await.is_err());
        // Delete of an unknown id is a tolerant no-op.
        assert!(ops.delete("nope".into()).await.is_ok());
    }
}
