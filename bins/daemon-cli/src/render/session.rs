// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Session-surface responses: rosters, pages, detail, grouping, search, approvals,
//! checkpoints, and drained inbox items.

use daemon_api::ApiResponse;

pub(super) fn try_render(resp: ApiResponse) -> Option<ApiResponse> {
    match resp {
        ApiResponse::Sessions(list) => {
            println!("sessions: {}", list.len());
            for info in list {
                println!("  - {} {:?}", info.session, info.state);
            }
        }
        ApiResponse::Approvals(list) => {
            println!("pending approvals: {}", list.len());
            for info in list {
                let path = info.path.map(|p| format!(" path={p}")).unwrap_or_default();
                println!(
                    "  - {} req={}{} :: {}",
                    info.session, info.request_id, path, info.prompt
                );
            }
        }
        ApiResponse::Checkpoints(list) => {
            println!("checkpoints: {}", list.len());
            for info in list {
                println!(
                    "  - {} session={} tool={} created={}",
                    info.id, info.session, info.tool, info.created_unix
                );
            }
        }
        ApiResponse::Drained(items) => {
            println!("drained: {} item(s)", items.len());
            for item in items {
                println!("  - {item:?}");
            }
        }
        ApiResponse::SessionPage(page) => {
            println!("sessions: {}", page.sessions.len());
            for s in page.sessions {
                let profile = s
                    .bound_profile
                    .map(|p| p.as_str().to_string())
                    .unwrap_or_else(|| "-".to_string());
                let title = s.title.unwrap_or_else(|| "(untitled)".to_string());
                println!(
                    "  - {} [{:?}/{:?}] profile={} {}",
                    s.session, s.lifecycle, s.role, profile, title
                );
            }
            if let Some(c) = page.next_cursor {
                println!("  next_cursor={c}");
            }
        }
        ApiResponse::SessionDetail(detail) => match detail {
            Some(d) => println!(
                "session {}: state={:?} role={:?} model={:?} children={} checkpoints={}",
                d.info.session,
                d.info.state,
                d.info.role,
                d.model,
                d.children.len(),
                d.checkpoints
            ),
            None => println!("session: not found"),
        },
        ApiResponse::SessionsByProfile(groups) => {
            for (profile, sessions) in groups {
                println!(
                    "profile {}: {} session(s)",
                    profile.as_str(),
                    sessions.len()
                );
                for s in sessions {
                    println!("  - {}", s.session);
                }
            }
        }
        ApiResponse::SessionSearch(hits) => {
            println!("hits: {}", hits.len());
            for h in hits {
                println!("  - {} {}: {}", h.session, h.title, h.snippet);
            }
        }
        other => return Some(other),
    }
    None
}
