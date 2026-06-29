// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Curator-surface responses: the skill ledger and the result of a curator run.

use daemon_api::ApiResponse;

pub(super) fn try_render(resp: ApiResponse) -> Option<ApiResponse> {
    match resp {
        ApiResponse::CuratorSkills(entries) => {
            println!("skills: {}", entries.len());
            for e in entries {
                let u = &e.usage;
                let cat = e.category.as_deref().unwrap_or("general");
                let origin = match u.created_by {
                    daemon_api::SkillCreator::Agent => "agent",
                    daemon_api::SkillCreator::User => "user",
                    daemon_api::SkillCreator::Bundled => "bundled",
                };
                let state = match u.state {
                    daemon_api::SkillState::Active => "active",
                    daemon_api::SkillState::Stale => "stale",
                    daemon_api::SkillState::Archived => "archived",
                };
                let mut flags = Vec::new();
                if u.pinned {
                    flags.push("pinned");
                }
                if e.is_bundled {
                    flags.push("bundled");
                }
                let flags = if flags.is_empty() {
                    String::new()
                } else {
                    format!(" [{}]", flags.join(","))
                };
                println!(
                    "  - {} [{}] {}/{} views={} uses={} patches={}{}",
                    e.name, cat, state, origin, u.view_count, u.use_count, u.patch_count, flags
                );
            }
        }
        ApiResponse::CuratorRun(changes) => {
            if changes.is_empty() {
                println!("curator: no changes");
            } else {
                println!("curator: {} change(s)", changes.len());
                for c in changes {
                    println!("  - {}: {:?} -> {:?}", c.name, c.from, c.to);
                }
            }
        }
        other => return Some(other),
    }
    None
}
