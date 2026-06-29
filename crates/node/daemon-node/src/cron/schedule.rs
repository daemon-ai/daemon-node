// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The cron worker's schedule arithmetic: spec decoding, provider parsing, `daemon-schedule`
//! construction, catch-up gating, and next-fire advancement (all pure, no `&self`).

use daemon_api::{CatchUpPolicy, CronSpec};
use daemon_schedule::Schedule;

use super::worker::{CronWorker, CRON_SKIP_TOLERANCE_SECS};

impl CronWorker {
    /// Wall-clock now in unix seconds.
    pub(crate) fn now_unix() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }

    /// Decode the wire `CronSpec` from a stored job's opaque blob (falling back to a name-only spec
    /// from the column on a decode failure, so a corrupt row never wedges the whole tick).
    pub(crate) fn decode_spec(job: &daemon_store::StoredCronJob) -> CronSpec {
        daemon_api::from_cbor::<CronSpec>(&job.spec).unwrap_or_else(|_| CronSpec {
            schedule: job.schedule.clone(),
            ..CronSpec::default()
        })
    }

    /// Parse a [`CronSpec::provider`] string to the wire [`ProviderSelector`](daemon_api::ProviderSelector),
    /// accepting the canonical snake_case names plus the legacy adapter aliases that all collapse to
    /// `GenAi`. An unrecognized value yields `None` (inherit the profile's provider).
    pub(crate) fn parse_provider(s: &str) -> Option<daemon_api::ProviderSelector> {
        use daemon_api::ProviderSelector::*;
        match s.trim().to_lowercase().as_str() {
            "mock" => Some(Mock),
            "genai" | "openai" | "anthropic" | "gemini" | "groq" | "deep_seek" | "deepseek"
            | "xai" | "open_router" | "openrouter" | "cohere" => Some(GenAi),
            "llama_cpp" | "llamacpp" => Some(LlamaCpp),
            "mistral_rs" | "mistralrs" => Some(MistralRs),
            _ => None,
        }
    }

    /// Build the `daemon-schedule` `Schedule` from a spec (schedule string + tz + repeat + jitter).
    pub(crate) fn schedule_of(spec: &CronSpec) -> Result<Schedule, daemon_schedule::ScheduleError> {
        Schedule::parse(&spec.schedule)?
            .with_timezone(spec.timezone.as_deref())
            .map(|s| s.with_repeat(spec.repeat).with_jitter(spec.jitter_secs))
    }

    /// Whether a fire that was scheduled for `scheduled_fire` and observed at `now` should run, given
    /// the catch-up policy and the schedule's grace window (the rest fast-forwards).
    pub(crate) fn should_fire(
        spec: &CronSpec,
        schedule: &Schedule,
        scheduled_fire: u64,
        now: u64,
    ) -> bool {
        let lateness = now.saturating_sub(scheduled_fire);
        let tolerance = match spec.catch_up {
            CatchUpPolicy::Always => u64::MAX,
            CatchUpPolicy::Grace => schedule.grace_secs(now),
            CatchUpPolicy::Skip => CRON_SKIP_TOLERANCE_SECS,
        };
        lateness <= tolerance
    }

    /// Advance a job's `next_fire` past `now` (fast-forwarding stale misses so a long downtime fires
    /// at most once), applying jitter. Returns the updated job; `next_fire` is `None` when the
    /// schedule is exhausted (a past one-shot).
    pub(crate) fn advanced(
        job: &daemon_store::StoredCronJob,
        schedule: &Schedule,
        now: u64,
    ) -> daemon_store::StoredCronJob {
        let mut next = schedule.next_after(now);
        // Fast-forward: keep advancing while the computed fire is not strictly in the future, so a
        // multi-period downtime collapses to a single next occurrence (no thundering herd).
        let mut guard = 0;
        while let Some(t) = next {
            if t > now || guard > 4096 {
                break;
            }
            next = schedule.next_after(t);
            guard += 1;
        }
        let mut job = job.clone();
        job.next_fire_unix = next.map(|t| t.saturating_add(schedule.jitter_offset(t)));
        job
    }
}
