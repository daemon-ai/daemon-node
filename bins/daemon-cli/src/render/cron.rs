// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Cron-surface responses: scheduled jobs, created job ids, run history, and suggestions.

use daemon_api::ApiResponse;

pub(super) fn try_render(resp: ApiResponse) -> Option<ApiResponse> {
    match resp {
        ApiResponse::CronJobs(jobs) => {
            for j in jobs {
                let next = j
                    .next_fire_unix
                    .map(|t| t.to_string())
                    .unwrap_or_else(|| "-".to_string());
                println!(
                    "  - {} {} ({}){} next={} fires={}",
                    j.id,
                    j.spec.name,
                    j.spec.schedule,
                    if j.paused { " [paused]" } else { "" },
                    next,
                    j.fire_count,
                );
            }
        }
        ApiResponse::CronId(id) => println!("cron: {id}"),
        ApiResponse::CronRuns(runs) => {
            for r in runs {
                let session = r.session.as_ref().map(|s| s.as_str()).unwrap_or("-");
                println!(
                    "  - started={} ok={} trigger={:?} session={}",
                    r.started_unix, r.ok, r.trigger, session
                );
            }
        }
        ApiResponse::CronSuggestions(suggestions) => {
            println!("cron suggestions: {}", suggestions.len());
            for s in suggestions {
                println!(
                    "  - {} \"{}\" [{}] ({})",
                    s.id, s.title, s.spec.schedule, s.source
                );
            }
        }
        other => return Some(other),
    }
    None
}
