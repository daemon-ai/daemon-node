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

#[cfg(test)]
mod tests {
    use super::*;
    use daemon_api::CatchUpPolicy;
    use daemon_store::StoredCronJob;

    fn spec(schedule: &str, catch_up: CatchUpPolicy) -> CronSpec {
        CronSpec {
            name: "job".into(),
            schedule: schedule.into(),
            catch_up,
            ..CronSpec::default()
        }
    }

    fn job(schedule: &str, next_fire_unix: Option<u64>) -> StoredCronJob {
        StoredCronJob {
            id: "cron-1".into(),
            schedule: schedule.into(),
            spec: Vec::new(),
            next_fire_unix,
            paused: false,
            last_run_unix: None,
            last_ok: None,
            last_detail: None,
            fire_count: 0,
            created_unix: 0,
            owner: None,
        }
    }

    /// `test_scheduled_task.c` `/schedule/normal` catch-up analogue: a fire scheduled just in the
    /// recent past still runs (within the grace window), i.e. it "fires promptly".
    #[test]
    fn should_fire_recent_past_due_within_grace() {
        let now = 1_000_000;
        let sched = CronWorker::schedule_of(&spec("@every 1h", CatchUpPolicy::Grace)).unwrap();
        // 10s late — well inside the 120s grace floor.
        assert!(CronWorker::should_fire(
            &spec("@every 1h", CatchUpPolicy::Grace),
            &sched,
            now - 10,
            now
        ));
    }

    /// A stale miss beyond the Skip tolerance is not run (the schedule fast-forwards instead).
    #[test]
    fn should_fire_skips_stale_beyond_tolerance() {
        let now = 1_000_000;
        let sched = CronWorker::schedule_of(&spec("@every 1h", CatchUpPolicy::Skip)).unwrap();
        // 1000s late — far past CRON_SKIP_TOLERANCE_SECS (60s).
        assert!(!CronWorker::should_fire(
            &spec("@every 1h", CatchUpPolicy::Skip),
            &sched,
            now - 1000,
            now
        ));
        // Always ignores lateness entirely.
        assert!(CronWorker::should_fire(
            &spec("@every 1h", CatchUpPolicy::Always),
            &sched,
            now - 1_000_000,
            now
        ));
    }

    /// `/scheduled-task/schedule/reuse` (re-schedule after execution fires again): advancing a
    /// recurring job past `now` re-arms it with a single future fire.
    #[test]
    fn advanced_recurring_rearms_future() {
        let now = 1_000_000;
        let sched = CronWorker::schedule_of(&spec("@every 1h", CatchUpPolicy::Grace)).unwrap();
        let advanced = CronWorker::advanced(&job("@every 1h", Some(now - 10)), &sched, now);
        let next = advanced.next_fire_unix.expect("recurring job re-arms");
        assert!(next > now, "the re-armed fire is strictly in the future");
        assert!(next <= now + 3600, "and within one period");
    }

    /// Multi-period downtime collapses to a single next occurrence (no thundering-herd backlog):
    /// a very stale `next_fire` still advances to just the next future fire.
    #[test]
    fn advanced_fast_forwards_stale_downtime() {
        let now = 1_600_000_000;
        // Hourly cron; the job's armed fire is ~28h stale.
        let sched = CronWorker::schedule_of(&spec("0 * * * *", CatchUpPolicy::Grace)).unwrap();
        let advanced = CronWorker::advanced(&job("0 * * * *", Some(now - 100_000)), &sched, now);
        let next = advanced.next_fire_unix.expect("recurring advances");
        assert!(next > now, "collapses stale backlog to one future fire");
        assert!(
            next <= now + 3600,
            "a single next hourly occurrence, not a backlog replay"
        );
    }

    /// `/scheduled-task/schedule/past` — **divergence**: libpurple refuses a past execute-at with an
    /// error; the daemon instead yields `next_fire None` for a past one-shot, so the tick auto-deletes
    /// it (it can never fire again). No error path.
    #[test]
    fn advanced_one_shot_past_exhausts() {
        let now = 1_600_000_000; // well after the year-2000 timestamp below
        let sched =
            CronWorker::schedule_of(&spec("2000-01-01T00:00:00Z", CatchUpPolicy::Grace)).unwrap();
        assert!(sched.is_one_shot());
        let advanced =
            CronWorker::advanced(&job("2000-01-01T00:00:00Z", Some(946_684_800)), &sched, now);
        assert!(
            advanced.next_fire_unix.is_none(),
            "a past one-shot is exhausted (no fire), not an error"
        );
    }
}
