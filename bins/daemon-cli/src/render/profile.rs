// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Profile/credential-surface responses: profile roster/detail, credentials, auth flows,
//! distribution bundles, imports, revisions, and skill bundles.

use daemon_api::ApiResponse;

pub(super) fn try_render(resp: ApiResponse) -> Option<ApiResponse> {
    match resp {
        ApiResponse::Profiles(profiles) => {
            println!("profiles: {}", profiles.len());
            for p in profiles {
                let active = if p.is_active { " *" } else { "" };
                println!("  - {} [{:?}] {}{}", p.id, p.provider, p.model, active);
            }
        }
        ApiResponse::Profile(spec) => match spec {
            Some(s) => {
                println!("profile: {}", s.id);
                println!("  provider: {:?}", s.provider);
                println!("  model: {}", s.model);
                if let Some(base) = &s.base_url {
                    println!("  base_url: {base}");
                }
                println!("  credential_ref: {}", s.credential_profile());
            }
            None => println!("profile: none"),
        },
        ApiResponse::Credentials(creds) => {
            println!("credentials: {}", creds.len());
            for c in creds {
                let state = if c.present {
                    c.hint.clone()
                } else {
                    "(none)".to_string()
                };
                println!("  - {} {}", c.profile, state);
            }
        }
        ApiResponse::AuthBegun(b) => {
            println!("auth begun: flow_id={} ({:?})", b.flow_id, b.flow_kind);
            println!("  open this URL in a browser:\n    {}", b.authorization_url);
            println!(
                "  redirect_uri={} expires_at={}",
                b.redirect_uri, b.expires_at
            );
        }
        ApiResponse::AuthCompleted(c) => {
            let bound = c
                .bound_profile
                .map(|p| format!(" bound_profile={p}"))
                .unwrap_or_default();
            println!(
                "auth completed: account={} credential_ref={} instance={}{}",
                c.account_label,
                c.credential_ref,
                c.transport_instance.as_str(),
                bound
            );
        }
        ApiResponse::AuthProviders(list) => {
            println!("auth providers: {}", list.len());
            for p in list {
                let fields: Vec<String> = p
                    .params_schema
                    .iter()
                    .map(|f| {
                        if f.required {
                            format!("{}*", f.key)
                        } else {
                            f.key.clone()
                        }
                    })
                    .collect();
                println!(
                    "  - {} [{:?}] \"{}\" params=[{}]",
                    p.family,
                    p.flow_kind,
                    p.display_name,
                    fields.join(", ")
                );
            }
        }
        ApiResponse::Distribution(d) => {
            println!(
                "distribution: {} (wire v{})",
                d.profile.id, d.wire_version.0
            );
            println!("  provider: {:?}", d.profile.provider);
            println!("  model: {}", d.profile.model);
            println!("  credential_ref: {}", d.profile.credential_profile());
            println!("  skills: {}", d.skills.len());
            for s in &d.skills {
                println!("    - {}", s.name);
            }
            if let Some(seq) = d.head_seq {
                println!("  head revision: {seq}");
            }
        }
        ApiResponse::ProfileId(id) => println!("imported profile: {id}"),
        ApiResponse::Revisions(page) => {
            println!("revisions: {}", page.items.len());
            for r in page.items {
                let author = match &r.author {
                    daemon_api::Author::Operator => "operator".to_string(),
                    daemon_api::Author::Agent(label) => format!("agent:{label}"),
                };
                println!(
                    "  - #{} [{}] {} (parent {:?})",
                    r.seq, author, r.reason, r.parent
                );
            }
            if let Some(next) = page.next {
                println!("  next={next}");
            }
        }
        ApiResponse::SkillBundle(b) => {
            let cat = b.category.as_deref().unwrap_or("general");
            println!("skill: {} [{}] ({} file(s))", b.name, cat, b.files.len());
            for path in b.files.keys() {
                println!("  - {path}");
            }
        }
        other => return Some(other),
    }
    None
}
