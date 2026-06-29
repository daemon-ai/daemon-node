// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Core node responses: ack/route, health, stats, telemetry, and errors.

use daemon_api::ApiResponse;

pub(super) fn try_render(resp: ApiResponse) -> Option<ApiResponse> {
    match resp {
        ApiResponse::Ok => println!("ok"),
        ApiResponse::Routed { session } => println!("routed: session={session}"),
        ApiResponse::Health(h) => {
            println!("health: all_ok={}", h.all_ok);
            for s in h.services {
                let detail = s.detail.map(|d| format!(" ({d})")).unwrap_or_default();
                println!(
                    "  - {} ok={} restarts={}{}",
                    s.name, s.ok, s.restarts, detail
                );
            }
        }
        ApiResponse::Stats(s) => println!(
            "stats: jobs={} wakes={} sessions={} active={} usage(in={} out={} cache_r={} cache_w={} reason={} cost=${:.4})",
            s.pending_jobs,
            s.pending_wakes,
            s.sessions,
            s.active,
            s.usage.input_tokens,
            s.usage.output_tokens,
            s.usage.cache_read_tokens,
            s.usage.cache_write_tokens,
            s.usage.reasoning_tokens,
            s.usage.cost_micros as f64 / 1_000_000.0,
        ),
        ApiResponse::Telemetry(d) => println!(
            "telemetry: healthy={} events={} jobs={} wakes={} sessions={} active={} usage(in={} out={} cache_r={} cache_w={} reason={} api_calls={} cost=${:.4})",
            d.healthy,
            d.events,
            d.pending_jobs,
            d.pending_wakes,
            d.sessions,
            d.active,
            d.usage.input_tokens,
            d.usage.output_tokens,
            d.usage.cache_read_tokens,
            d.usage.cache_write_tokens,
            d.usage.reasoning_tokens,
            d.usage.api_calls,
            d.usage.cost_micros as f64 / 1_000_000.0,
        ),
        ApiResponse::Error(e) => println!("error: {e}"),
        other => return Some(other),
    }
    None
}
