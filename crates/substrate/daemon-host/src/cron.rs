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
        };
        self.store
            .cron_set(stored)
            .await
            .map_err(|e| ApiError::Other(format!("cron store: {e}")))?;
        Ok(id)
    }

    /// Replace an existing job's spec (preserving run bookkeeping), recomputing its next fire.
    pub async fn update(&self, id: String, spec: CronSpec) -> Result<(), ApiError> {
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
