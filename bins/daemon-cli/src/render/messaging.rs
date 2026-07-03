// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Messaging-surface responses: chat routes/pins, rooms, transport instances, conversations,
//! contacts, and the action menu.

use daemon_api::ApiResponse;

pub(super) fn try_render(resp: ApiResponse) -> Option<ApiResponse> {
    match resp {
        ApiResponse::ChatRoutes(page) => {
            println!("chat routes: {}", page.items.len());
            for r in page.items {
                let profile = r
                    .profile
                    .map(|p| p.as_str().to_string())
                    .unwrap_or_else(|| "-".to_string());
                println!(
                    "  - {}/{:?} -> {} profile={}",
                    r.origin.transport.as_str(),
                    r.origin.scope,
                    r.session,
                    profile
                );
            }
            if let Some(next) = page.next {
                println!("  next={next}");
            }
        }
        ApiResponse::ChatRoute(route) => match route {
            Some(r) => println!(
                "pin: {}/{:?} -> {}",
                r.origin.transport.as_str(),
                r.origin.scope,
                r.session
            ),
            None => println!("pin: (none)"),
        },
        ApiResponse::Rooms(page) => {
            println!("rooms: {}", page.items.len());
            for r in page.items {
                let session = r
                    .session
                    .map(|s| s.as_str().to_string())
                    .unwrap_or_else(|| "-".to_string());
                println!(
                    "  - {} {} session={}",
                    r.transport.as_str(),
                    r.room,
                    session
                );
            }
            if let Some(next) = page.next {
                println!("  next={next}");
            }
        }
        ApiResponse::TransportInstances(instances) => {
            println!("transport instances: {}", instances.len());
            for i in instances {
                println!(
                    "  - {} [{}] {:?}/{:?}",
                    i.transport.as_str(),
                    i.family,
                    i.connection,
                    i.presence
                );
            }
        }
        ApiResponse::Conversations(page) => {
            println!("conversations: {}", page.items.len());
            for c in page.items {
                let title = c.title.unwrap_or_else(|| "(untitled)".to_string());
                println!(
                    "  - {}/{} [{:?}] \"{}\" members={}",
                    c.transport.as_str(),
                    c.id,
                    c.kind,
                    title,
                    c.members.len()
                );
            }
            if let Some(next) = page.next {
                println!("  next={next}");
            }
        }
        ApiResponse::Conversation(conv) => match conv {
            Some(c) => {
                println!(
                    "conversation: {}/{} [{:?}]",
                    c.transport.as_str(),
                    c.id,
                    c.kind
                );
                if let Some(t) = &c.title {
                    println!("  title: {t}");
                }
                if let Some(t) = &c.topic {
                    println!("  topic: {t}");
                }
                println!("  members: {}", c.members.len());
                for m in &c.members {
                    let session = m
                        .session
                        .as_ref()
                        .map(|s| s.as_str().to_string())
                        .unwrap_or_else(|| "-".to_string());
                    println!("    - {} [{:?}] session={}", m.contact.id, m.role, session);
                }
            }
            None => println!("conversation: not found"),
        },
        ApiResponse::ContactProfile(profile) => println!("profile:\n{profile}"),
        ApiResponse::Contacts(contacts) => {
            println!("contacts: {}", contacts.len());
            for c in contacts {
                let name = c.display_name.unwrap_or_else(|| "(no name)".to_string());
                println!("  - {} \"{}\"", c.id, name);
            }
        }
        ApiResponse::ActionMenu(menu) => match menu {
            Some(m) => {
                println!("action menu: {} item(s)", m.items.len());
                for item in m.items {
                    println!("  - {item}");
                }
            }
            None => println!("action menu: none"),
        },
        other => return Some(other),
    }
    None
}
