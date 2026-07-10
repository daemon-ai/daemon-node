// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

// Phase 4: xtask is dev/build tooling (codegen, CI helpers) run by maintainers, not a runtime
// security surface. Its fs (build artifacts) and spawns (cbindgen/cc/bash build steps) are
// developer-controlled; the hardening bans target the shipped node, so xtask is allowed crate-wide.
#![allow(clippy::disallowed_methods)]

//! `xtask` — repo automation (codegen, CI helpers).
//!
//! Subcommands:
//! - `gen-headers` — run `cbindgen` over both binding crates to (re)generate the committed C
//!   headers `bindings/daemon-core-ffi/include/daemon_core.h` (the L1 brain seam) and
//!   `bindings/daemon-ffi/include/daemon.h` (the L2 durable-host seam). The generated headers plus
//!   the published `daemon-api.cddl` are the complete non-Rust contract (daemon-ffi-spec §3.6).
//! - `cddl` — check the `daemon-api` mirror CDDL artifact covers the Rust wire enum variants.
//! - `api-fixtures` — write canonical CBOR request/response fixtures for non-Rust clients.
//! - `gen-zcbor` — generate the client zcbor C codec from a CDDL (the artifact `daemon-app` vendors).
//! - `verify-codec` — decode every CBOR fixture with the generated C codec, proving the CDDL/zcbor
//!   path stays byte-compatible with the serde/ciborium runtime wire format.

#![forbid(unsafe_code)]

use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};
use std::process::Command;

/// `xtask` — repo automation (codegen, CI helpers).
#[derive(Parser)]
#[command(name = "xtask", about)]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// (Re)generate the committed C headers for both binding crates via `cbindgen`.
    GenHeaders,
    /// Check the `daemon-api` mirror CDDL covers the Rust wire enum variants.
    Cddl,
    /// Write canonical CBOR request/response fixtures for non-Rust clients.
    ApiFixtures,
    /// Generate the client zcbor C codec from a CDDL.
    GenZcbor {
        /// The CDDL contract (defaults to the pinned `daemon-api.cddl`).
        #[arg(long)]
        cddl: Option<PathBuf>,
        /// The output directory (defaults to `target/zcbor-codec`).
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Decode every CBOR fixture with the generated C codec (wire-compat gate).
    VerifyCodec,
}

fn main() -> anyhow::Result<()> {
    match Cli::parse().command {
        Cmd::GenHeaders => gen_headers(),
        Cmd::Cddl => check_cddl(),
        Cmd::ApiFixtures => gen_api_fixtures(),
        Cmd::GenZcbor { cddl, out } => gen_zcbor(cddl, out),
        Cmd::VerifyCodec => verify_codec(),
    }
}

/// The workspace root (xtask's manifest dir is `<root>/xtask`).
fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask lives under the workspace root")
        .to_path_buf()
}

/// Generate the committed C headers for both binding crates via `cbindgen`.
fn gen_headers() -> anyhow::Result<()> {
    let root = workspace_root();
    // (crate name, crate dir relative to root, output header relative to the crate dir).
    let crates = [
        (
            "daemon-core-ffi",
            "bindings/daemon-core-ffi",
            "include/daemon_core.h",
        ),
        ("daemon-ffi", "bindings/daemon-ffi", "include/daemon.h"),
    ];
    for (name, dir, header) in crates {
        gen_one_header(&root, name, dir, header)?;
    }
    Ok(())
}

/// Run `cbindgen` over one binding crate, writing its committed header.
fn gen_one_header(root: &Path, name: &str, dir: &str, header: &str) -> anyhow::Result<()> {
    let crate_dir = root.join(dir);
    let config = crate_dir.join("cbindgen.toml");
    let out = crate_dir.join(header);
    std::fs::create_dir_all(out.parent().unwrap())?;

    let status = Command::new("cbindgen")
        .arg("--config")
        .arg(&config)
        .arg("--crate")
        .arg(name)
        .arg("--output")
        .arg(&out)
        .arg(&crate_dir)
        .status()
        .map_err(|e| anyhow::anyhow!("failed to run cbindgen (is it on PATH?): {e}"))?;
    anyhow::ensure!(status.success(), "cbindgen exited with {status} for {name}");

    println!("generated {}", out.display());
    Ok(())
}

/// Check that the `daemon-api` CDDL mirror artifact exists and names every Rust request/response
/// variant. This is intentionally a syntactic parity gate: schema validation/codegen is handled by
/// downstream CDDL tooling, but adding a Rust wire variant without updating the published contract
/// must fail CI.
fn check_cddl() -> anyhow::Result<()> {
    let root = workspace_root();
    let path = root.join("crates/contracts/daemon-api/daemon-api.cddl");
    let text = read_to_string(&path)?;
    anyhow::ensure!(!text.trim().is_empty(), "{} is empty", path.display());
    for rule in [
        "api-request",
        "api-response",
        "wire_version",
        // wire v2: the merged live session event log shapes.
        "session-log-entry",
        "session-payload",
        "log-page-view",
        "direction",
        "disposition",
        "origin",
        // wire v2: outbound delivery targets + handover (§5.4).
        "delivery-target",
        "sink-kind",
        "route-addr",
    ] {
        anyhow::ensure!(
            text.contains(rule),
            "{} is missing the `{rule}` rule",
            path.display()
        );
    }
    // ApiRequest / ApiResponse live in the `wire` submodule of daemon-api.
    let rust = read_to_string(&root.join("crates/contracts/daemon-api/src/wire.rs"))?;
    assert_cddl_covers_enum(&text, &rust, "ApiRequest", "api-request")?;
    assert_cddl_covers_enum(&text, &rust, "ApiResponse", "api-response")?;
    println!("ok: {} defines the api mirror", path.display());
    Ok(())
}

fn read_to_string(path: &Path) -> anyhow::Result<String> {
    std::fs::read_to_string(path).map_err(|e| anyhow::anyhow!("reading {}: {e}", path.display()))
}

fn assert_cddl_covers_enum(
    cddl: &str,
    rust: &str,
    enum_name: &str,
    rule_name: &str,
) -> anyhow::Result<()> {
    let variants = rust_enum_variants(rust, enum_name)?;
    let missing: Vec<_> = variants
        .iter()
        .filter(|variant| !cddl_rule_mentions_variant(cddl, rule_name, variant))
        .cloned()
        .collect();
    anyhow::ensure!(
        missing.is_empty(),
        "{rule_name} is missing Rust {enum_name} variants: {}",
        missing.join(", ")
    );
    Ok(())
}

fn rust_enum_variants(rust: &str, enum_name: &str) -> anyhow::Result<Vec<String>> {
    let marker = format!("pub enum {enum_name}");
    let start = rust
        .find(&marker)
        .ok_or_else(|| anyhow::anyhow!("could not find `{marker}`"))?;
    let after_marker = &rust[start + marker.len()..];
    let open = after_marker
        .find('{')
        .ok_or_else(|| anyhow::anyhow!("could not find body for `{enum_name}`"))?;
    let body_start = start + marker.len() + open + 1;
    let mut depth = 1i32;
    let mut end = None;
    for (offset, ch) in rust[body_start..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    end = Some(body_start + offset);
                    break;
                }
            }
            _ => {}
        }
    }
    let body_end = end.ok_or_else(|| anyhow::anyhow!("unterminated `{enum_name}` body"))?;
    let mut variants = Vec::new();
    let mut depth = 1i32;
    for line in rust[body_start..body_end].lines() {
        let trimmed = line.trim();
        if depth == 1
            && !trimmed.is_empty()
            && !trimmed.starts_with("///")
            && !trimmed.starts_with("#[")
            && !trimmed.starts_with("//")
            && !trimmed.starts_with('}')
        {
            let ident: String = trimmed
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect();
            if !ident.is_empty() {
                variants.push(ident);
            }
        }
        for ch in line.chars() {
            match ch {
                '{' | '(' => depth += 1,
                '}' | ')' => depth -= 1,
                _ => {}
            }
        }
    }
    variants.sort();
    variants.dedup();
    Ok(variants)
}

/// A Rust enum variant is "covered" when the CDDL carries its externally-tagged wire key as a
/// quoted string `"Variant"`. In the unified CDDL each `api-request`/`api-response` arm is its own
/// named rule (e.g. `request-submit = { "Submit": ... }`, `request-health = "Health"`), so the key
/// lives in the arm rule rather than inline in the union block; searching the whole file is the
/// format-stable parity check. `rule_name` is kept for call-site clarity.
fn cddl_rule_mentions_variant(cddl: &str, rule_name: &str, variant: &str) -> bool {
    let _ = rule_name;
    cddl.contains(&format!("\"{variant}\""))
}

fn gen_api_fixtures() -> anyhow::Result<()> {
    use daemon_api::{
        AccountSettingsSchema, AdapterCapabilities, AdapterInfo, ApiRequest, ApiResponse,
        ApprovalInfo, ChatMessage, CommandInvocation, CommandOutput, ConnectionState, ContactInfo,
        ContactsOps, ConvChange, ConversationOps, CredentialInfo, DisconnectReason, EventsPage,
        HealthReport, JournalPageView, JournalRecord, JournalRecordPayload, LogPageView,
        MembershipChange, MembershipOps, MessageAttachment, ModelDescriptor, NodeEvent,
        Participant, PolicyEntry, PresenceState, ProfileInfo, ProfileSpec, ProviderDescriptor,
        ProviderKindWire, ProviderSelector, ProviderSignIn, RosterOps, ServiceHealth, SessionPage,
        TransportInstanceInfo,
    };
    use daemon_common::{Author, ProfileRef, ReqId, SessionId};
    use daemon_protocol::{AgentCommand, ToolDetail, TransportId, UserMsg};

    let root = workspace_root();
    let out = root.join("crates/contracts/daemon-api/fixtures/cbor");
    std::fs::create_dir_all(&out)?;

    write_cbor(&out, "request-health.cbor", &ApiRequest::Health)?;
    write_cbor(
        &out,
        "request-sessions-query.cbor",
        &ApiRequest::SessionsQuery {
            query: daemon_api::SessionQuery {
                scope: daemon_api::SessionScope::TopLevel,
                after: None,
                limit: 25,
                since_rev: None,
            },
        },
    )?;
    write_cbor(
        &out,
        "request-subscribe.cbor",
        &ApiRequest::Subscribe {
            session: SessionId::new("fixture-session"),
            after_seq: 0,
            max: 64,
        },
    )?;
    write_cbor(
        &out,
        "request-events-since.cbor",
        &ApiRequest::EventsSince {
            cursor: 0,
            wait_ms: Some(1000),
        },
    )?;
    write_cbor(
        &out,
        "request-submit.cbor",
        &ApiRequest::Submit {
            session: SessionId::new("fixture-session"),
            command: AgentCommand::StartTurn {
                input: UserMsg::new("hello from daemon-app"),
                request_id: ReqId(1),
            },
            origin: None,
            profile: Some(ProfileRef::new("default")),
        },
    )?;
    write_cbor(
        &out,
        "request-session-create.cbor",
        &ApiRequest::SessionCreate {
            session: Some(SessionId::new("fixture-session")),
            profile: Some(ProfileRef::new("default")),
        },
    )?;
    // Cluster B / allow_permanent: a committed fixture exercising the additive optional field at v28,
    // so the CDDL↔Rust agreement on `request-approval-decide` is proven on a real ciborium payload.
    write_cbor(
        &out,
        "request-approval-decide.cbor",
        &ApiRequest::ApprovalDecide {
            session: SessionId::new("fixture-session"),
            request_id: "fixture-request".into(),
            allow: true,
            allow_permanent: true,
            reason: Some("fixture reason".into()),
        },
    )?;
    // The read-only guardrail caps (wire v29).
    write_cbor(&out, "request-caps.cbor", &ApiRequest::Caps)?;
    write_cbor(
        &out,
        "response-caps.cbor",
        &ApiResponse::Caps(daemon_api::CapsReport {
            orchestrate_max_depth: 1,
            orchestrate_max_fanout: 8,
            // wire v31: the agent-created-agents guardrail caps.
            max_composed_profiles: 32,
            max_ephemeral_per_session: 8,
        }),
    )?;
    // Fingerprint management (wire v29): the allow-list list/revoke ops + the list response, so
    // `verify-codec` proves the generated zcbor C decoder accepts the new shapes.
    write_cbor(
        &out,
        "request-fingerprint-list.cbor",
        &ApiRequest::FingerprintList {
            session: SessionId::new("fixture-session"),
        },
    )?;
    write_cbor(
        &out,
        "request-fingerprint-revoke.cbor",
        &ApiRequest::FingerprintRevoke {
            session: SessionId::new("fixture-session"),
            fingerprint: "ab12cd34".into(),
        },
    )?;
    write_cbor(
        &out,
        "response-fingerprints.cbor",
        &ApiResponse::Fingerprints(vec![daemon_api::RememberedFingerprint {
            fingerprint: "ab12cd34".into(),
            // Provenance (wire v30): a populated label + capture timestamp.
            label: Some("git status".into()),
            remembered_at_ms: 1_700_000_000_000,
        }]),
    )?;
    write_cbor(
        &out,
        "response-session-created.cbor",
        &ApiResponse::SessionCreated {
            session: SessionId::new("fixture-session"),
        },
    )?;
    // ----- wire v30 batch -----
    // Item 1: transport lifecycle ops.
    write_cbor(
        &out,
        "request-transport-disconnect.cbor",
        &ApiRequest::TransportDisconnect {
            transport: TransportId::new("matrix/@bot:hs.org"),
        },
    )?;
    write_cbor(
        &out,
        "request-transport-remove.cbor",
        &ApiRequest::TransportRemove {
            transport: TransportId::new("matrix/@bot:hs.org"),
        },
    )?;
    // Item 2: an instance carrying a fatal auth failure (reason/message/fatal + Error state).
    write_cbor(
        &out,
        "response-transport-instances.cbor",
        &ApiResponse::TransportInstances(vec![TransportInstanceInfo {
            transport: TransportId::new("matrix/@bot:hs.org"),
            family: "matrix".into(),
            display_name: "@bot:hs.org".into(),
            connection: ConnectionState::Error,
            presence: PresenceState::Offline,
            bound_profile: Some(ProfileRef::new("default")),
            reason: Some(DisconnectReason::AuthenticationFailed),
            message: Some("M_FORBIDDEN: invalid access token".into()),
            fatal: true,
            // Wire v35: this instance carries a custom label + is enabled (the desired-state
            // overlay), so the one fixture exercises the populated decode of both new fields.
            enabled: true,
            label: Some("Work bot".into()),
        }]),
    )?;
    // Item 4: adapter policies — matrix reports auto_accept_invites; a second adapter reports none.
    // Wire v33: the matrix row also carries the per-verb ops descriptors (Some(..) with mixed
    // flags + directory=true) while the room row leaves every new field at its default (None ops +
    // directory=false, encoded as absent) — so the one fixture proves both the populated and the
    // back-compat "absent" decode of the v33 additive fields.
    write_cbor(
        &out,
        "response-adapters.cbor",
        &ApiResponse::Adapters(vec![
            AdapterInfo {
                family: "matrix".into(),
                display_name: "Matrix".into(),
                capabilities: AdapterCapabilities {
                    rooms: true,
                    direct_messages: true,
                    presence: true,
                    room_enumeration: true,
                    file_transfer: false,
                    interactive_auth: true,
                },
                account_schema: AccountSettingsSchema::default(),
                policies: vec![PolicyEntry {
                    key: "auto_accept_invites".into(),
                    label: "Automatically accept room invites".into(),
                    value: "true".into(),
                }],
                conversation_ops: Some(ConversationOps {
                    create: true,
                    join_channel: true,
                    leave: true,
                    delete: false,
                    send: true,
                    set_topic: true,
                    set_title: true,
                    set_description: true,
                }),
                membership_ops: Some(MembershipOps {
                    invite: true,
                    remove: true,
                    ban: true,
                    set_role: true,
                }),
                contacts_ops: Some(ContactsOps {
                    get_profile: true,
                    action_menu: false,
                    set_alias: false,
                }),
                roster_ops: Some(RosterOps {
                    list: true,
                    add: false,
                    update: false,
                    remove: false,
                }),
                directory: true,
            },
            AdapterInfo {
                family: "room".into(),
                display_name: "Rooms (internal)".into(),
                capabilities: AdapterCapabilities::default(),
                account_schema: AccountSettingsSchema::default(),
                policies: Vec::new(),
                conversation_ops: None,
                membership_ops: None,
                contacts_ops: None,
                roster_ops: None,
                directory: false,
            },
        ]),
    )?;
    // Item 6: the tool-override op.
    write_cbor(
        &out,
        "request-tool-set-enabled.cbor",
        &ApiRequest::ToolSetEnabled {
            tool: "browser".into(),
            enabled: false,
        },
    )?;
    // Item 7: an fs/edit approval carrying a node-computed diff detail.
    write_cbor(
        &out,
        "response-approvals.cbor",
        &ApiResponse::Approvals(daemon_api::WirePage {
            items: vec![ApprovalInfo {
                session: SessionId::new("fixture-session"),
                request_id: "fixture-approval".into(),
                prompt: "Apply edit to src/lib.rs".into(),
                path: Some("src/lib.rs".into()),
                fingerprint: None,
                detail: Some(ToolDetail::new(
                    "fs.diff",
                    br#"{"path":"src/lib.rs","diff":"@@ -1 +1 @@\n-old\n+new\n"}"#.to_vec(),
                )),
            }],
            next: None,
        }),
    )?;
    write_cbor(&out, "request-profile-list.cbor", &ApiRequest::ProfileList)?;
    write_cbor(
        &out,
        "request-model-current.cbor",
        &ApiRequest::ModelCurrent {
            profile: Some("default".into()),
        },
    )?;
    write_cbor(&out, "request-fs-roots.cbor", &ApiRequest::FsRoots)?;
    // Paged fs_list (wire v24/v25, the uniform WirePage shape): a resume request (after = the
    // previous page's `next`) and a page response carrying items + a set `next` cursor, so
    // `verify-codec` proves the generated zcbor C decoder accepts the fs-list-page shape.
    write_cbor(
        &out,
        "request-fs-list.cbor",
        &ApiRequest::FsList {
            root: daemon_api::FsRootId::Workspace,
            dir: "src".into(),
            show_ignored: false,
            after: Some("src/main.rs".into()),
        },
    )?;
    write_cbor(
        &out,
        "response-fs-list.cbor",
        &ApiResponse::FsList(daemon_api::FsListPage {
            items: vec![
                daemon_api::FsEntry {
                    name: "vendor".into(),
                    path: "src/vendor".into(),
                    kind: daemon_api::FsEntryKind::Dir,
                    size: 0,
                    mtime_ms: 1_700_000_000_000,
                    ignored: false,
                },
                daemon_api::FsEntry {
                    name: "lib.rs".into(),
                    path: "src/lib.rs".into(),
                    kind: daemon_api::FsEntryKind::File,
                    size: 4096,
                    mtime_ms: 1_700_000_000_001,
                    ignored: false,
                },
            ],
            next: Some("src/lib.rs".into()),
        }),
    )?;
    // Paged conv_list (wire v25): a resume request + a page with a set `next` cursor, proving the
    // generated zcbor C decoder accepts the conv-page shape.
    write_cbor(
        &out,
        "request-conv-list.cbor",
        &ApiRequest::ConvList {
            transport: daemon_protocol::TransportId::new("rooms"),
            after: Some("conv-063".into()),
        },
    )?;
    write_cbor(
        &out,
        "response-conv-list.cbor",
        &ApiResponse::Conversations(daemon_api::WirePage {
            items: vec![daemon_api::ConversationInfo {
                transport: daemon_protocol::TransportId::new("rooms"),
                id: "conv-064".into(),
                kind: daemon_api::ConversationType::Channel,
                title: Some("General".into()),
                topic: None,
                description: None,
                members: Vec::new(),
                parent: None,
            }],
            next: Some("conv-064".into()),
        }),
    )?;
    // Conversation hierarchy (wire v38): a structural `Space` container (a root — no `parent`) and
    // a child `Channel` naming that space via `parent`, so verify-codec proves the generated zcbor C
    // decoder accepts the new `ConversationType::Space` variant + the additive `parent` member.
    write_cbor(
        &out,
        "response-conv-hierarchy.cbor",
        &ApiResponse::Conversations(daemon_api::WirePage {
            items: vec![
                daemon_api::ConversationInfo {
                    transport: daemon_protocol::TransportId::new("matrix/@me:hs.org"),
                    id: "!space:hs.org".into(),
                    kind: daemon_api::ConversationType::Space,
                    title: Some("Engineering".into()),
                    topic: None,
                    description: None,
                    members: Vec::new(),
                    parent: None,
                },
                daemon_api::ConversationInfo {
                    transport: daemon_protocol::TransportId::new("matrix/@me:hs.org"),
                    id: "!room:hs.org".into(),
                    kind: daemon_api::ConversationType::Channel,
                    title: Some("general".into()),
                    topic: Some("chit-chat".into()),
                    description: None,
                    members: Vec::new(),
                    parent: Some("!space:hs.org".into()),
                },
            ],
            next: None,
        }),
    )?;
    // Server-side roster (wire v34): a paged list request + resume, the mutation requests carrying a
    // ContactInfo, and a ContactPage response with a set `next` cursor — so verify-codec proves the
    // generated zcbor C decoder accepts the contact-page shape + the four new request variants.
    write_cbor(
        &out,
        "request-roster-list.cbor",
        &ApiRequest::RosterList {
            transport: TransportId::new("matrix/@me:hs.org"),
            after: Some("@aaa:matrix.org".into()),
        },
    )?;
    write_cbor(
        &out,
        "request-roster-add.cbor",
        &ApiRequest::RosterAdd {
            transport: TransportId::new("matrix/@me:hs.org"),
            contact: daemon_api::ContactInfo {
                id: "@bob:matrix.org".into(),
                display_name: Some("Bob".into()),
                presence: daemon_api::Presence::default(),
                permission: daemon_api::ContactPermission::Allow,
            },
        },
    )?;
    write_cbor(
        &out,
        "request-roster-remove.cbor",
        &ApiRequest::RosterRemove {
            transport: TransportId::new("matrix/@me:hs.org"),
            contact: daemon_api::ContactInfo {
                id: "@bob:matrix.org".into(),
                display_name: None,
                presence: daemon_api::Presence::default(),
                permission: daemon_api::ContactPermission::Unset,
            },
        },
    )?;
    write_cbor(
        &out,
        "response-contact-page.cbor",
        &ApiResponse::ContactPage(daemon_api::WirePage {
            items: vec![daemon_api::ContactInfo {
                id: "@bob:matrix.org".into(),
                display_name: Some("Bob".into()),
                presence: daemon_api::Presence::default(),
                permission: daemon_api::ContactPermission::Allow,
            }],
            next: Some("@bob:matrix.org".into()),
        }),
    )?;
    // Notifications (wire v37; port-notify): the read-only list op + a response carrying an
    // authorization-request and a connection-error notification, so verify-codec proves the
    // generated zcbor C decoder accepts the notification-info shape + its typed kinds.
    write_cbor(
        &out,
        "request-notification-list.cbor",
        &ApiRequest::NotificationList,
    )?;
    // Deterministic `created_ms`: the `NotificationInfo::new_*` constructors stamp wall-clock
    // `now_ms()`, which would churn this fixture on every run. Pin it to a fixed epoch AFTER
    // construction (runtime behavior is untouched — this is a fixture-only override).
    const FIXTURE_NOTIF_CREATED_MS: u64 = 1_700_000_000_000;
    let mut notif_authz = daemon_api::NotificationInfo::new_authorization(
        Some("notif-authz".into()),
        daemon_api::AuthorizationRequest::new(daemon_api::ContactInfo {
            id: "@bob:matrix.org".into(),
            display_name: Some("Bob".into()),
            presence: daemon_api::Presence::default(),
            permission: daemon_api::ContactPermission::Unset,
        }),
    );
    notif_authz.created_ms = FIXTURE_NOTIF_CREATED_MS;
    let mut notif_conn = daemon_api::NotificationInfo::new_connection_error(
        Some("notif-conn".into()),
        TransportId::new("matrix/@me:hs.org"),
    );
    notif_conn.created_ms = FIXTURE_NOTIF_CREATED_MS;
    write_cbor(
        &out,
        "response-notifications.cbor",
        &ApiResponse::Notifications(vec![notif_authz, notif_conn]),
    )?;
    // Persons / metacontacts (wire v37; port-person): the read-only list op + a response carrying
    // an aliased, avatared, multi-endpoint person, so verify-codec proves the generated zcbor C
    // decoder accepts the person shape (incl. the first wire-reachable `image` rule).
    write_cbor(&out, "request-person-list.cbor", &ApiRequest::PersonList)?;
    write_cbor(
        &out,
        "response-persons.cbor",
        &ApiResponse::Persons(vec![daemon_api::Person {
            id: "person-ada".into(),
            alias: Some("Ada".into()),
            avatar: Some(daemon_api::Image {
                blob: daemon_common::BlobRef::new(daemon_common::ContentHash::new([7u8; 32]), 3),
            }),
            endpoints: vec![
                daemon_api::PersonEndpoint::new(
                    TransportId::new("matrix/@me:hs.org"),
                    daemon_api::ContactInfo {
                        id: "@ada:hs.org".into(),
                        display_name: Some("Ada L.".into()),
                        presence: daemon_api::Presence::default(),
                        permission: daemon_api::ContactPermission::Allow,
                    },
                ),
                daemon_api::PersonEndpoint::new(
                    TransportId::new("discord/bot"),
                    daemon_api::ContactInfo {
                        id: "ada#1234".into(),
                        display_name: None,
                        presence: daemon_api::Presence::default(),
                        permission: daemon_api::ContactPermission::Unset,
                    },
                ),
            ],
        }]),
    )?;
    // Account management (wire v35): the reversible-connect + persisted enabled/label + credential
    // rename requests, so verify-codec proves the generated zcbor C decoder accepts all four new
    // request variants (including the `? label` optional in both set/clear shapes).
    write_cbor(
        &out,
        "request-transport-connect.cbor",
        &ApiRequest::TransportConnect {
            transport: TransportId::new("matrix/@bot:hs.org"),
        },
    )?;
    write_cbor(
        &out,
        "request-transport-set-enabled.cbor",
        &ApiRequest::TransportSetEnabled {
            transport: TransportId::new("matrix/@bot:hs.org"),
            enabled: false,
        },
    )?;
    write_cbor(
        &out,
        "request-transport-set-label.cbor",
        &ApiRequest::TransportSetLabel {
            transport: TransportId::new("matrix/@bot:hs.org"),
            label: Some("Work bot".into()),
        },
    )?;
    write_cbor(
        &out,
        "request-credential-set-label.cbor",
        &ApiRequest::CredentialSetLabel {
            profile: "default".into(),
            label: Some("Personal key".into()),
        },
    )?;
    write_cbor(&out, "request-command-list.cbor", &ApiRequest::CommandList)?;
    write_cbor(
        &out,
        "request-command-invoke.cbor",
        &ApiRequest::CommandInvoke {
            invocation: CommandInvocation {
                name: "help".into(),
                ..Default::default()
            },
        },
    )?;
    // Onboarding (CON-4 / CON-6): credentials + model discovery/selection.
    write_cbor(
        &out,
        "request-credential-set.cbor",
        &ApiRequest::CredentialSet {
            profile: "default".into(),
            secret: "sk-fixture-secret".into(),
        },
    )?;
    write_cbor(
        &out,
        "request-credential-list.cbor",
        &ApiRequest::CredentialList,
    )?;
    write_cbor(
        &out,
        "request-credential-remove.cbor",
        &ApiRequest::CredentialRemove {
            profile: "default".into(),
        },
    )?;
    // Multi-step interactive auth (wire v31): the AuthStep op across every AuthStepInput arm, the
    // reshaped AuthBegun (initial challenge), and AuthStepped across every AuthChallenge +
    // AuthStepResult arm — so verify-codec proves the generated zcbor decoder accepts the new shapes.
    {
        use daemon_api::{
            AuthBeginResponse, AuthChallenge, AuthCompleteResponse, AuthFieldKind, AuthFlowKind,
            AuthParamField, AuthProviderInfo, AuthStepInput, AuthStepRequest, AuthStepResult,
        };
        write_cbor(
            &out,
            "request-auth-step-fields.cbor",
            &ApiRequest::AuthStep(AuthStepRequest {
                flow_id: "flow-1".into(),
                input: AuthStepInput::Fields(std::collections::BTreeMap::from([(
                    "otp".to_string(),
                    "123456".to_string(),
                )])),
            }),
        )?;
        write_cbor(
            &out,
            "request-auth-step-callback.cbor",
            &ApiRequest::AuthStep(AuthStepRequest {
                flow_id: "flow-1".into(),
                input: AuthStepInput::Callback("https://cb.example/?code=xyz&state=s".into()),
            }),
        )?;
        write_cbor(
            &out,
            "request-auth-step-poll.cbor",
            &ApiRequest::AuthStep(AuthStepRequest {
                flow_id: "flow-1".into(),
                input: AuthStepInput::Poll,
            }),
        )?;
        write_cbor(
            &out,
            "response-auth-begun.cbor",
            &ApiResponse::AuthBegun(AuthBeginResponse {
                flow_id: "flow-1".into(),
                challenge: AuthChallenge::Redirect {
                    authorization_url: "https://idp.example/authorize?state=s".into(),
                },
                expires_at: 1_700_000_600,
            }),
        )?;
        write_cbor(
            &out,
            "response-auth-stepped-form.cbor",
            &ApiResponse::AuthStepped(AuthStepResult::Challenge(AuthChallenge::Form {
                title: "Enter the code we texted you".into(),
                fields: vec![AuthParamField {
                    key: "otp".into(),
                    label: "One-time code".into(),
                    required: true,
                    // wire v38: exercise the enriched metadata (a numeric OTP with a hint) so
                    // verify-codec proves the generated C decoder accepts the new optional members.
                    kind: daemon_api::AuthFieldKind::Number,
                    placeholder: Some("123456".into()),
                    ..Default::default()
                }],
            })),
        )?;
        write_cbor(
            &out,
            "response-auth-stepped-qr.cbor",
            &ApiResponse::AuthStepped(AuthStepResult::Challenge(AuthChallenge::Qr {
                payload: "wa://link?token=abc".into(),
                image: Some(vec![0x89, 0x50, 0x4e, 0x47]),
                poll_interval_ms: 2000,
            })),
        )?;
        write_cbor(
            &out,
            "response-auth-stepped-message.cbor",
            &ApiResponse::AuthStepped(AuthStepResult::Challenge(AuthChallenge::Message {
                text: "Approve the login on your other device".into(),
            })),
        )?;
        write_cbor(
            &out,
            "response-auth-stepped-completed.cbor",
            &ApiResponse::AuthStepped(AuthStepResult::Completed(AuthCompleteResponse {
                credential_ref: "matrix/@bot:hs.org".into(),
                account_label: "@bot:hs.org".into(),
                transport_instance: daemon_protocol::TransportId::new("matrix/@bot:hs.org"),
                bound_profile: Some(ProfileRef::new("default")),
            })),
        )?;
        // wire v38: an AuthProviders discovery response advertising the new UserPassword flow with
        // an enriched params schema across every AuthFieldKind (a plain-text username, a MASKED
        // password, and a defaulted Choice) — so verify-codec proves the generated zcbor C decoder
        // accepts the enriched auth-param-field + the new auth-flow-kind arm.
        write_cbor(
            &out,
            "response-auth-providers.cbor",
            &ApiResponse::AuthProviders(vec![AuthProviderInfo {
                family: "userpass".into(),
                flow_kind: AuthFlowKind::UserPassword,
                display_name: "Username & password".into(),
                params_schema: vec![
                    AuthParamField {
                        key: "username".into(),
                        label: "Username".into(),
                        required: true,
                        kind: AuthFieldKind::Text,
                        placeholder: Some("you@example.org".into()),
                        ..Default::default()
                    },
                    AuthParamField {
                        key: "password".into(),
                        label: "Password".into(),
                        required: true,
                        kind: AuthFieldKind::Password,
                        ..Default::default()
                    },
                    AuthParamField {
                        key: "region".into(),
                        label: "Region".into(),
                        required: false,
                        kind: AuthFieldKind::Choice,
                        default: Some("us".into()),
                        choices: vec!["us".into(), "eu".into()],
                        ..Default::default()
                    },
                ],
            }]),
        )?;
    }
    write_cbor(
        &out,
        "request-models.cbor",
        &ApiRequest::Models { after: None },
    )?;
    write_cbor(
        &out,
        "request-set-session-model.cbor",
        &ApiRequest::SetSessionModel {
            session: SessionId::new("fixture-session"),
            model: "claude-opus-4-8".into(),
            provider: Some(ProviderSelector::GenAi),
        },
    )?;
    // Profiles CRUD (PRO-2/3/4): exercise the now-concrete profile-spec (the optional arms -
    // tool_allowlist Some - and the nested budget/tunables maps).
    let mut fixture_spec = ProfileSpec::new("work", ProviderSelector::GenAi, "claude-opus-4-8");
    fixture_spec.tool_allowlist = Some(vec!["read".into(), "search".into()]);
    write_cbor(
        &out,
        "request-profile-create.cbor",
        &ApiRequest::ProfileCreate {
            spec: fixture_spec.clone(),
        },
    )?;
    write_cbor(
        &out,
        "request-profile-update.cbor",
        &ApiRequest::ProfileUpdate {
            spec: fixture_spec.clone(),
        },
    )?;
    write_cbor(
        &out,
        "request-profile-get.cbor",
        &ApiRequest::ProfileGet { id: "work".into() },
    )?;
    write_cbor(
        &out,
        "request-profile-clone.cbor",
        &ApiRequest::ProfileClone {
            source: "default".into(),
            new_id: "work".into(),
        },
    )?;
    write_cbor(
        &out,
        "response-profile.cbor",
        &ApiResponse::Profile(Some(fixture_spec)),
    )?;
    // Persona ops (wire v36): the SoulGet/SoulSet requests + the SoulText response, so
    // `verify-codec` proves the generated zcbor C decoder accepts the new persona shapes (the
    // composed system prompt itself never travels — this is the SOUL.md source text only).
    write_cbor(
        &out,
        "request-soul-get.cbor",
        &ApiRequest::SoulGet { id: "work".into() },
    )?;
    write_cbor(
        &out,
        "request-soul-set.cbor",
        &ApiRequest::SoulSet {
            id: "work".into(),
            text: "You are a focused work assistant.".into(),
        },
    )?;
    write_cbor(
        &out,
        "response-soul-text.cbor",
        &ApiResponse::SoulText("You are a focused work assistant.".into()),
    )?;
    // The profile listing (PRO-1) exercising the wire v31 provenance on `profile-info`: one
    // operator-authored (created_by "operator", no owner) and one agent-authored
    // (created_by {agent}, owner = the authoring session) row, so `verify-codec` proves the
    // generated zcbor C decoder accepts the new optional `created_by`/`owner` fields on both arms.
    let mut op_info = ProfileInfo::from_spec(
        &ProfileSpec::new("work", ProviderSelector::GenAi, "claude-opus-4-8"),
        true,
    );
    op_info.created_by = Some(Author::Operator);
    let mut agent_info = ProfileInfo::from_spec(
        &ProfileSpec::new("agent/s1/helper", ProviderSelector::Mock, "m"),
        false,
    );
    agent_info.created_by = Some(Author::Agent("profile_manage".into()));
    agent_info.owner = Some("s1".into());
    write_cbor(
        &out,
        "response-profiles.cbor",
        &ApiResponse::Profiles(vec![op_info, agent_info]),
    )?;
    // The daemon-api gateway selector (wire `"daemon_api"`): a full profile-spec exercising the new
    // additive `provider-selector` value so `verify-codec` proves the generated zcbor C decoder
    // accepts it (OpenRouter-style `author/slug` model id + the pinned OpenAI-compatible base URL).
    let daemon_api_spec = ProfileSpec {
        base_url: Some("https://api.daemon.ai/api/v1/".into()),
        ..ProfileSpec::new(
            "daemon",
            ProviderSelector::DaemonApi,
            "anthropic/claude-sonnet-4-5",
        )
    };
    write_cbor(
        &out,
        "response-profile-daemon-api.cbor",
        &ApiResponse::Profile(Some(daemon_api_spec)),
    )?;
    // The foreign-engine selector (wire v23; generalized v29): a profile-spec whose `engine` is
    // the foreign arm (`{"Foreign": {"agent": tstr}}` — catalog name only, never a recipe), so
    // `verify-codec` proves the generated zcbor C decoder accepts the `engine-selector` union. The
    // other profile fixtures above exercise the default "Core" arm (always present on new
    // encodings).
    let foreign_engine_spec = ProfileSpec {
        engine: daemon_api::EngineSelector::Foreign {
            agent: "gemini".into(),
        },
        ..ProfileSpec::new("foreign", ProviderSelector::Mock, "")
    };
    write_cbor(
        &out,
        "response-profile-foreign-engine.cbor",
        &ApiResponse::Profile(Some(foreign_engine_spec)),
    )?;
    // The `NodeProvider` foreign backend (wire v30): a Foreign profile routed through the node
    // gateway to a provider+model, so `verify-codec` proves the generated zcbor C decoder accepts
    // the `foreign-backend` union's `NodeProvider` arm (the `foreign-engine` fixture above exercises
    // the default `AgentNative` arm, present on every profile encoding).
    let foreign_node_provider_spec = ProfileSpec {
        engine: daemon_api::EngineSelector::Foreign {
            agent: "codex".into(),
        },
        foreign_backend: daemon_api::ForeignBackend::NodeProvider {
            provider: ProviderSelector::GenAi,
            model: "gpt-4o".into(),
            credential_ref: Some("openai".into()),
        },
        ..ProfileSpec::new("routed", ProviderSelector::Mock, "")
    };
    write_cbor(
        &out,
        "response-profile-foreign-node-provider.cbor",
        &ApiResponse::Profile(Some(foreign_node_provider_spec)),
    )?;
    // The foreign-agent catalog (wire v29): one ACP entry + one stream-json entry, so
    // `verify-codec` proves the generated zcbor C decoder accepts the renamed `agent-entry` shape
    // and both `agent-protocol` values.
    write_cbor(
        &out,
        "response-agent-catalog.cbor",
        &ApiResponse::AgentCatalog(vec![
            daemon_api::AgentEntry {
                name: "gemini".into(),
                recipe: daemon_api::AgentRecipe {
                    program: Some("gemini".into()),
                    args: vec!["--experimental-acp".into()],
                    env: Vec::new(),
                    endpoint: None,
                },
                source: daemon_api::AgentSource::Builtin,
                protocol: daemon_api::AgentProtocol::Acp,
                installed: true,
                version: Some("1".into()),
                capabilities: vec![("fs".into(), "true".into())],
                verification: daemon_api::AgentVerification::Verified,
            },
            daemon_api::AgentEntry {
                name: "claude".into(),
                recipe: daemon_api::AgentRecipe {
                    program: Some("claude".into()),
                    args: vec!["--output-format".into(), "stream-json".into()],
                    env: Vec::new(),
                    endpoint: None,
                },
                source: daemon_api::AgentSource::Manual,
                protocol: daemon_api::AgentProtocol::StreamJson,
                installed: true,
                version: None,
                capabilities: Vec::new(),
                verification: daemon_api::AgentVerification::Unverified,
            },
        ]),
    )?;

    let fixture_descriptor = ModelDescriptor {
        id: "claude-opus-4-8".into(),
        provider: ProviderSelector::GenAi,
        display_name: None,
        context_length: Some(200_000),
        input_price_micros_per_mtok: Some(15_000_000),
        output_price_micros_per_mtok: Some(75_000_000),
        local: false,
    };
    write_cbor(
        &out,
        "response-credentials.cbor",
        &ApiResponse::Credentials(vec![CredentialInfo {
            profile: "default".into(),
            present: true,
            hint: "\u{2026}cret".into(),
            // Wire v35: the node-overlaid human label (`None` on an un-labeled credential).
            label: Some("Personal key".into()),
        }]),
    )?;
    write_cbor(
        &out,
        "response-models.cbor",
        &ApiResponse::Models(daemon_api::WirePage {
            items: vec![fixture_descriptor.clone()],
            next: Some(fixture_descriptor.id.clone()),
        }),
    )?;
    write_cbor(
        &out,
        "response-model-current.cbor",
        &ApiResponse::ModelCurrent(Some(fixture_descriptor)),
    )?;
    // Provider + model discovery (v22): the enumeration op, a credential-aware per-provider listing
    // (with a transient key), and their responses. The response descriptors exercise the additive
    // `provider-descriptor` shape and a `model-descriptor` carrying the optional `display_name`.
    write_cbor(
        &out,
        "request-provider-catalog.cbor",
        &ApiRequest::ProviderCatalog,
    )?;
    write_cbor(
        &out,
        "request-provider-models.cbor",
        &ApiRequest::ProviderModels {
            provider: "anthropic".into(),
            credential_ref: None,
            transient_key: Some("sk-fixture-transient".into()),
            after: None,
        },
    )?;
    write_cbor(
        &out,
        "response-provider-catalog.cbor",
        &ApiResponse::ProviderCatalog(vec![
            ProviderDescriptor {
                id: "daemon_cloud".into(),
                display_name: "Daemon Cloud".into(),
                kind: ProviderKindWire::DaemonCloud,
                wire_selector: ProviderSelector::DaemonApi,
                // Daemon Cloud needs a key to run turns (lists keyless — host-spec semantics).
                requires_key: true,
                supports_model_discovery: true,
                default_base_url: Some("https://api.daemon.ai/api/v1/".into()),
                sign_in: None,
            },
            // The OpenRouter genai row advertises interactive sign-in (wire v30, CON-15): the node
            // states the auth family + label; the client calls `auth_begin { family, params: {} }`.
            ProviderDescriptor {
                id: "open_router".into(),
                display_name: "OpenRouter".into(),
                kind: ProviderKindWire::Cloud,
                wire_selector: ProviderSelector::GenAi,
                requires_key: true,
                supports_model_discovery: true,
                default_base_url: None,
                sign_in: Some(ProviderSignIn {
                    family: "provider/openrouter".into(),
                    label: "Sign in with OpenRouter".into(),
                }),
            },
        ]),
    )?;
    write_cbor(
        &out,
        "response-provider-models.cbor",
        &ApiResponse::ProviderModels(daemon_api::WirePage {
            items: vec![ModelDescriptor {
                id: "anthropic/claude-sonnet-4-5".into(),
                provider: ProviderSelector::DaemonApi,
                display_name: Some("Claude Sonnet 4.5".into()),
                context_length: Some(200_000),
                input_price_micros_per_mtok: Some(3_000_000),
                output_price_micros_per_mtok: Some(15_000_000),
                local: false,
            }],
            next: None,
        }),
    )?;
    // Custom providers (generalized Daemon Cloud): the write-model CRUD ops + the list response, so
    // a non-Rust client and verify-codec exercise the `custom-provider` shape end-to-end.
    write_cbor(
        &out,
        "request-custom-provider-list.cbor",
        &ApiRequest::CustomProviderList,
    )?;
    write_cbor(
        &out,
        "request-custom-provider-set.cbor",
        &ApiRequest::CustomProviderSet {
            provider: daemon_api::CustomProvider {
                id: "custom/my-gateway".into(),
                display_name: "My Gateway".into(),
                base_url: "https://my-gateway.example/v1/".into(),
                wire_selector: ProviderSelector::DaemonApi,
                requires_key: true,
                credential_ref: Some("custom/my-gateway".into()),
                source: daemon_api::CustomProviderSource::User,
            },
        },
    )?;
    write_cbor(
        &out,
        "request-custom-provider-remove.cbor",
        &ApiRequest::CustomProviderRemove {
            id: "custom/my-gateway".into(),
        },
    )?;
    write_cbor(
        &out,
        "response-custom-providers.cbor",
        &ApiResponse::CustomProviders(vec![daemon_api::CustomProvider {
            id: "custom/my-gateway".into(),
            display_name: "My Gateway".into(),
            base_url: "https://my-gateway.example/v1/".into(),
            wire_selector: ProviderSelector::DaemonApi,
            requires_key: true,
            credential_ref: None,
            source: daemon_api::CustomProviderSource::Config,
        }]),
    )?;
    write_cbor(&out, "response-ok.cbor", &ApiResponse::Ok)?;
    write_cbor(
        &out,
        "response-session-page.cbor",
        &ApiResponse::SessionPage(SessionPage {
            sessions: Vec::new(),
            next_cursor: None,
            rev: 0,
            removed: Vec::new(),
        }),
    )?;
    // W6: the pure-local session recap op (request + a populated response), so verify-codec proves
    // the generated C decoder takes the new shapes end-to-end.
    write_cbor(
        &out,
        "request-session-recap.cbor",
        &ApiRequest::SessionRecap {
            session: SessionId::new("fixture-session"),
        },
    )?;
    write_cbor(
        &out,
        "response-session-recap.cbor",
        &ApiResponse::SessionRecap(Some(daemon_api::SessionRecap {
            title: Some("Docker Networking Help".into()),
            user_turns: 3,
            assistant_turns: 4,
            tool_results: 2,
            top_tools: vec![("fs".into(), 2), ("web_search".into(), 1)],
            files_touched: vec!["src/lib.rs".into()],
            last_ask: Some("why does the bridge drop packets".into()),
            last_reply: Some("the MTU mismatch was the culprit".into()),
        })),
    )?;
    write_cbor(
        &out,
        "response-log-page.cbor",
        &ApiResponse::LogPage(LogPageView {
            entries: Vec::new(),
            next_seq: 0,
            head_seq: 0,
            epoch: 0,
        }),
    )?;
    // wire v37 (W2-E): the richer ChatMessage on the conversation-history surface, so conformance
    // + verify-codec prove the CDDL↔Rust agreement on the new `JournalRecordPayload::Chat` arm and
    // the `chat-message` / `message-attachment` shapes on a real ciborium payload.
    write_cbor(
        &out,
        "response-journal.cbor",
        &ApiResponse::Journal(JournalPageView {
            entries: vec![JournalRecord {
                cursor: 3,
                segment: 1,
                seq: 3,
                epoch: 0,
                trace: 0,
                kind: "block.message".into(),
                timestamp_ms: 911_347_200_000,
                verified: true,
                payload: JournalRecordPayload::Chat {
                    message: Box::new(ChatMessage {
                        id: Some("$evt:hs.org".into()),
                        author: Some(Participant::Contact(ContactInfo {
                            id: "@alice:hs.org".into(),
                            display_name: Some("Alice Smith".into()),
                            ..Default::default()
                        })),
                        replying_to: Some("$prev:hs.org".into()),
                        text: "Now that is a big door".into(),
                        attachments: vec![MessageAttachment {
                            id: "att-1".into(),
                            content_type: Some("image/png".into()),
                            is_inline: true,
                            local_uri: None,
                            remote_uri: Some("mxc://hs.org/abc".into()),
                            size: 4096,
                        }],
                        timestamp: Some(911_347_200),
                        delivered_at: Some(911_347_201),
                        edited_at: None,
                        error: None,
                        title: Some("Titled".into()),
                        highlight_color: Some("#FF00FF".into()),
                        action: false,
                        event: false,
                        notice: false,
                        system: false,
                        highlighted: true,
                    }),
                },
            }],
            next_cursor: 3,
            head_cursor: 3,
            sealed_after: None,
        }),
    )?;
    write_cbor(
        &out,
        "response-events-page.cbor",
        &ApiResponse::EventsPage(EventsPage {
            events: vec![
                NodeEvent::RosterChanged { rev: 7 },
                // v31: the profile-list-changed pointer, so verify-codec proves the generated
                // decoder accepts the new node-event arm.
                NodeEvent::ProfilesChanged { rev: 3 },
                NodeEvent::ApprovalPending {
                    session: SessionId::new("fixture-session"),
                    request_id: "req-1".into(),
                },
                // v26: byte counters on the throttled download-progress event + the payload-free
                // catalog-changed pointer, so verify-codec proves the generated decoder takes both.
                NodeEvent::DownloadProgress {
                    id: daemon_common::DownloadId(1),
                    pct: 50,
                    state: "Downloading".into(),
                    downloaded_bytes: 46_000_000,
                    total_bytes: 92_000_000,
                },
                NodeEvent::CatalogChanged,
                // v29: the presence-push event, so verify-codec proves the generated decoder
                // accepts the new node-event arm + the connection/presence enums it carries.
                NodeEvent::TransportChanged {
                    transport: TransportId::new("matrix/@bot:hs.org"),
                    connection: ConnectionState::Connected,
                    presence: PresenceState::Unknown,
                    reason: None,
                    message: None,
                    fatal: false,
                },
                // v30: a disconnect transition carrying a reason/message + the transient
                // Disconnecting state (reconnect/backoff is node-owned; fatal:false = will retry).
                NodeEvent::TransportChanged {
                    transport: TransportId::new("matrix/@bot:hs.org"),
                    connection: ConnectionState::Disconnecting,
                    presence: PresenceState::Offline,
                    reason: Some(DisconnectReason::NetworkError),
                    message: Some("connection reset by peer".into()),
                    fatal: false,
                },
                // v30: the two membership-push tiers.
                NodeEvent::ConversationsChanged {
                    transport: TransportId::new("matrix/@bot:hs.org"),
                    conv: "!room:hs.org".into(),
                    change: ConvChange::Added,
                },
                NodeEvent::MembershipChanged {
                    transport: TransportId::new("matrix/@bot:hs.org"),
                    conv: "!room:hs.org".into(),
                    member: "@bot:hs.org".into(),
                    change: MembershipChange::Kicked,
                    actor: Some("@admin:hs.org".into()),
                    reason: Some("cleanup".into()),
                    is_self: true,
                },
                // v34: the roster-changed pointer, so verify-codec proves the generated decoder
                // accepts the new node-event arm.
                NodeEvent::ContactsChanged {
                    transport: TransportId::new("matrix/@bot:hs.org"),
                },
                // wire v37: the payload-free notifications-changed pointer (port-notify).
                NodeEvent::NotificationsChanged,
                // wire v37: the payload-free persons-changed pointer (port-person).
                NodeEvent::PersonsChanged,
                // wire v38: the per-message conversation-history pointer (chat journal), so
                // verify-codec proves the generated decoder accepts the new node-event arm.
                NodeEvent::MessagesChanged {
                    transport: TransportId::new("matrix/@bot:hs.org"),
                    conv: "!room:hs.org".into(),
                },
            ],
            next_cursor: 13,
            head_cursor: 13,
        }),
    )?;
    write_cbor(
        &out,
        "response-fs-roots.cbor",
        &ApiResponse::FsRoots(Vec::new()),
    )?;
    write_cbor(
        &out,
        "response-commands.cbor",
        &ApiResponse::Commands(Vec::new()),
    )?;
    write_cbor(
        &out,
        "response-command-output.cbor",
        &ApiResponse::CommandOutput(CommandOutput::default()),
    )?;
    write_cbor(
        &out,
        "response-health.cbor",
        &ApiResponse::Health(HealthReport {
            all_ok: true,
            services: vec![ServiceHealth {
                name: "fixture".into(),
                ok: true,
                restarts: 0,
                detail: None,
            }],
        }),
    )?;
    // Local model track (Phase 2): exercise the model arrays + ModelRef/ModelSource through the
    // regenerated (cap-bumped to 64) C codec. The quant is the chosen GGUF file carried as
    // ModelRef::Hf{ file: Some(...) }; ModelId is content-derived (no quant in it).
    {
        use daemon_common::{
            DownloadState, DownloadStatus, InstalledModel, ModelEngine, ModelFile, ModelId,
            ModelRef, ModelSource, QuantCandidate, QuantRecommendation, SearchHit, SearchPage,
            SearchQuery, SearchSort,
        };
        let repo = "bartowski/SmolLM2-135M-Instruct-GGUF";
        let gguf = "SmolLM2-135M-Instruct-Q4_K_M.gguf";
        let hf_ref = || {
            ModelRef::new(
                ModelEngine::Llama,
                ModelSource::Hf {
                    repo: repo.into(),
                    file: Some(gguf.into()),
                    revision: "main".into(),
                },
            )
        };
        write_cbor(
            &out,
            "request-model-search.cbor",
            &ApiRequest::ModelSearch {
                query: SearchQuery {
                    text: "SmolLM2".into(),
                    engine: ModelEngine::Llama,
                    sort: SearchSort::Trending,
                    page: 0,
                    limit: 25,
                },
            },
        )?;
        write_cbor(
            &out,
            "request-model-files.cbor",
            &ApiRequest::ModelFiles {
                repo: repo.into(),
                revision: None,
                engine: ModelEngine::Llama,
                after: None,
            },
        )?;
        write_cbor(
            &out,
            "request-model-download.cbor",
            &ApiRequest::ModelDownload { model: hf_ref() },
        )?;
        write_cbor(
            &out,
            "request-model-downloads.cbor",
            &ApiRequest::ModelDownloads,
        )?;
        write_cbor(
            &out,
            "request-model-catalog.cbor",
            &ApiRequest::ModelCatalog,
        )?;
        write_cbor(
            &out,
            "request-model-recommend.cbor",
            &ApiRequest::ModelRecommend(daemon_api::ModelRecommendArgs {
                repo: repo.into(),
                revision: None,
                engine: ModelEngine::Llama,
                budget_bytes: Some(6 * 1024 * 1024 * 1024),
            }),
        )?;
        // Responses exercise the bumped array caps: search-page.results, [model-file],
        // [download-status], [installed-model], plus the nested quant candidate list.
        write_cbor(
            &out,
            "response-model-search.cbor",
            &ApiResponse::ModelSearch(SearchPage {
                page: 0,
                results: vec![SearchHit {
                    repo: repo.into(),
                    author: Some("bartowski".into()),
                    downloads: 12_345,
                    likes: 42,
                    num_parameters: Some(135_000_000),
                    pipeline_tag: Some("text-generation".into()),
                    last_modified: Some("2025-01-01T00:00:00Z".into()),
                    gated: false,
                    private: false,
                }],
                has_more: false,
            }),
        )?;
        write_cbor(
            &out,
            "response-model-files.cbor",
            &ApiResponse::ModelFiles(daemon_api::WirePage {
                items: vec![
                    ModelFile {
                        path: gguf.into(),
                        size_bytes: 92_000_000,
                        quant: Some("Q4_K_M".into()),
                        is_split: false,
                        is_first_shard: false,
                        is_mmproj: false,
                    },
                    ModelFile {
                        path: "SmolLM2-135M-Instruct-Q8_0.gguf".into(),
                        size_bytes: 145_000_000,
                        quant: Some("Q8_0".into()),
                        is_split: false,
                        is_first_shard: false,
                        is_mmproj: false,
                    },
                    // A vision-projector companion row (wire v27): listed + downloadable, badged
                    // by the client, never a chat model.
                    ModelFile {
                        path: "mmproj-SmolLM2-135M-Instruct-Q8_0.gguf".into(),
                        size_bytes: 6_000_000,
                        quant: Some("Q8_0".into()),
                        is_split: false,
                        is_first_shard: false,
                        is_mmproj: true,
                    },
                ],
                next: None,
            }),
        )?;
        write_cbor(
            &out,
            "response-model-downloads.cbor",
            &ApiResponse::ModelDownloads(vec![DownloadStatus {
                id: daemon_common::DownloadId(1),
                model: hf_ref(),
                state: DownloadState::Downloading,
                downloaded_bytes: 46_000_000,
                total_bytes: 92_000_000,
                files_done: 0,
                files_total: 1,
                error: None,
            }]),
        )?;
        write_cbor(
            &out,
            "response-model-catalog.cbor",
            &ApiResponse::ModelCatalog(vec![InstalledModel {
                id: ModelId::new("smollm2-135m-q4km"),
                model: hf_ref(),
                display_name: "SmolLM2-135M-Instruct".into(),
                local_path: "/cache/models/SmolLM2-135M-Instruct-Q4_K_M.gguf".into(),
                size_bytes: 92_000_000,
                quant: Some("Q4_K_M".into()),
                installed_at_ms: 1_700_000_000_000,
                arch: Some("llama".into()),
                context_length: Some(8192),
                file_type: Some("Q4_K_M".into()),
                // The paired vision-projector companion (wire v27); null for text-only models.
                mmproj_path: Some("/cache/models/mmproj-SmolLM2-135M-Instruct-Q8_0.gguf".into()),
                // The node-local pinned artifact hash surfaced for display (wire v28).
                sha256: Some(
                    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855".into(),
                ),
            }]),
        )?;
        write_cbor(
            &out,
            "response-model-recommend.cbor",
            &ApiResponse::ModelRecommend(QuantRecommendation {
                engine: ModelEngine::Llama,
                repo: repo.into(),
                file: Some(gguf.into()),
                quant: "Q4_K_M".into(),
                size_bytes: Some(92_000_000),
                budget_bytes: 6 * 1024 * 1024 * 1024,
                fits: true,
                reason: "best quality that fits the detected ~6 GiB budget".into(),
                candidates: vec![
                    QuantCandidate {
                        quant: "Q8_0".into(),
                        file: Some("SmolLM2-135M-Instruct-Q8_0.gguf".into()),
                        size_bytes: Some(145_000_000),
                        fits: true,
                    },
                    QuantCandidate {
                        quant: "Q4_K_M".into(),
                        file: Some(gguf.into()),
                        size_bytes: Some(92_000_000),
                        fits: true,
                    },
                ],
            }),
        )?;
        write_cbor(
            &out,
            "response-model-download-started.cbor",
            &ApiResponse::ModelDownloadStarted(daemon_common::DownloadId(1)),
        )?;
    }
    // Multiplexed/server-streaming envelope (wire L0): prove the Rust serde shapes match the
    // wire-c2s / wire-s2c CDDL rules. The client hand-codes this envelope, so these fixtures are the
    // schema gate that keeps both sides in agreement.
    {
        use daemon_api::{WireC2S, WireS2C, WIRE_FEATURE_MUX, WIRE_FEATURE_STREAM, WIRE_VERSION};
        let features = vec![
            WIRE_FEATURE_MUX.to_string(),
            WIRE_FEATURE_STREAM.to_string(),
        ];
        write_cbor(
            &out,
            "wire-c2s-hello.cbor",
            &WireC2S::Hello {
                wire_version: WIRE_VERSION,
                features: features.clone(),
            },
        )?;
        write_cbor(
            &out,
            "wire-c2s-call.cbor",
            &WireC2S::Call {
                id: 1,
                req: ApiRequest::Subscribe {
                    session: SessionId::new("fixture-session"),
                    after_seq: 0,
                    max: 64,
                },
            },
        )?;
        write_cbor(
            &out,
            "wire-c2s-open.cbor",
            &WireC2S::Open {
                id: 2,
                req: ApiRequest::Subscribe {
                    session: SessionId::new("fixture-session"),
                    after_seq: 0,
                    max: 64,
                },
            },
        )?;
        write_cbor(&out, "wire-c2s-cancel.cbor", &WireC2S::Cancel { id: 1 })?;
        write_cbor(
            &out,
            "wire-s2c-hello.cbor",
            &WireS2C::Hello {
                wire_version: WIRE_VERSION,
                features,
                auth_mechanisms: Vec::new(),
            },
        )?;
        write_cbor(
            &out,
            "wire-s2c-reply.cbor",
            &WireS2C::Reply {
                id: 1,
                res: ApiResponse::Ok,
            },
        )?;
        write_cbor(
            &out,
            "wire-s2c-item.cbor",
            &WireS2C::Item {
                id: 1,
                res: ApiResponse::LogPage(LogPageView {
                    entries: Vec::new(),
                    next_seq: 0,
                    head_seq: 0,
                    epoch: 0,
                }),
            },
        )?;
        write_cbor(
            &out,
            "wire-s2c-end.cbor",
            &WireS2C::End { id: 1, error: None },
        )?;
        write_cbor(
            &out,
            "wire-s2c-reset.cbor",
            &WireS2C::Reset {
                id: 1,
                epoch: 0,
                head_seq: 0,
            },
        )?;
    }

    // ----- access control (Auth 5) -----
    write_cbor(
        &out,
        "request-user-create.cbor",
        &ApiRequest::UserCreate {
            username: "alice".into(),
            password: "correct horse".into(),
            roles: vec!["user".into()],
        },
    )?;
    write_cbor(&out, "request-user-list.cbor", &ApiRequest::UserList)?;
    write_cbor(&out, "request-who-am-i.cbor", &ApiRequest::WhoAmI)?;
    write_cbor(
        &out,
        "request-session-revoke.cbor",
        &ApiRequest::SessionRevoke {
            user_id: "u1".into(),
        },
    )?;
    write_cbor(
        &out,
        "request-resource-grant-create.cbor",
        &ApiRequest::ResourceGrantCreate {
            user_id: "u1".into(),
            resource_kind: "session".into(),
            resource_id: "s1".into(),
            capability: "session_read".into(),
        },
    )?;
    write_cbor(
        &out,
        "response-access-user.cbor",
        &ApiResponse::AccessUser(daemon_api::AccessUser {
            user_id: "u1".into(),
            username: "alice".into(),
            disabled: false,
            created_at: 0,
            roles: vec!["user".into()],
        }),
    )?;
    write_cbor(
        &out,
        "response-access-users.cbor",
        &ApiResponse::AccessUsers(Vec::new()),
    )?;
    write_cbor(
        &out,
        "response-access-roles.cbor",
        &ApiResponse::AccessRoles(vec![daemon_api::RoleInfo {
            role: "admin".into(),
            capabilities: vec!["access_admin".into()],
        }]),
    )?;
    write_cbor(
        &out,
        "response-who-am-i.cbor",
        &ApiResponse::WhoAmI(daemon_api::PrincipalView {
            user_id: "u1".into(),
            username: "alice".into(),
            roles: vec!["admin".into()],
            capabilities: vec!["access_admin".into()],
        }),
    )?;

    // -- user feedback over OpenTelemetry (N1; wire v31) -----------------------------------------
    write_cbor(
        &out,
        "request-feedback-submit.cbor",
        &ApiRequest::FeedbackSubmit {
            kind: daemon_api::FeedbackKind::Response,
            target: Some(daemon_api::FeedbackTarget {
                session: "s-fixture".into(),
                cursor: 42,
                trace: Some(daemon_common::TraceId(0x1234)),
            }),
            rating: Some(daemon_api::FeedbackRating::Up),
            comment: Some("nailed it".into()),
            include_content: true,
            diagnostics: Some(daemon_api::FeedbackDiagnostics {
                app_version: Some("1.2.3".into()),
                os: Some("linux".into()),
            }),
            surface: "transcript".into(),
        },
    )?;
    write_cbor(
        &out,
        "request-telemetry-consent-get.cbor",
        &ApiRequest::TelemetryConsentGet,
    )?;
    write_cbor(
        &out,
        "request-telemetry-consent-set.cbor",
        &ApiRequest::TelemetryConsentSet { enabled: true },
    )?;
    write_cbor(
        &out,
        "response-feedback-ack.cbor",
        &ApiResponse::FeedbackAck(daemon_api::FeedbackAck {
            accepted: true,
            queued: true,
        }),
    )?;
    write_cbor(
        &out,
        "response-telemetry-consent.cbor",
        &ApiResponse::TelemetryConsent { enabled: true },
    )?;
    // Saved presences (W2-F; wire v37): the list/save/delete/set-active ops + the listing reply.
    {
        use daemon_api::{PresencePrimitive, SavedPresence};
        write_cbor(
            &out,
            "request-presence-list.cbor",
            &ApiRequest::PresenceList,
        )?;
        let fixture_presence = SavedPresence {
            id: "ffffffff-ffff-ffff-ffff-ffffffffffff".into(),
            name: Some("Streaming".into()),
            primitive: PresencePrimitive::Streaming,
            message: Some("live on twitch".into()),
            emoji: Some("💀".into()),
            last_used: Some(1_700_000_000),
            use_count: 7,
        };
        write_cbor(
            &out,
            "request-presence-save.cbor",
            &ApiRequest::PresenceSave {
                presence: fixture_presence.clone(),
            },
        )?;
        write_cbor(
            &out,
            "request-presence-delete.cbor",
            &ApiRequest::PresenceDelete {
                id: "ffffffff-ffff-ffff-ffff-ffffffffffff".into(),
            },
        )?;
        write_cbor(
            &out,
            "request-presence-set-active.cbor",
            &ApiRequest::PresenceSetActive {
                id: "ffffffff-ffff-ffff-ffff-ffffffffffff".into(),
            },
        )?;
        write_cbor(
            &out,
            "response-saved-presences.cbor",
            &ApiResponse::SavedPresences(vec![fixture_presence]),
        )?;
    }

    // -- file transfer (W2-H; wire v37) --------------------------------------------------------
    {
        use daemon_api::{FileTransfer, FileTransferDirection, FileTransferState};
        use daemon_common::{BlobRef, ContentHash};

        let blob = BlobRef {
            hash: ContentHash::new([7u8; 32]),
            size: 1337,
            name: Some("cat.png".into()),
            mime: Some("image/png".into()),
        };
        write_cbor(
            &out,
            "request-ft-send.cbor",
            &ApiRequest::FtSend {
                transport: TransportId::new("matrix/@bot:localhost"),
                transfer: FileTransfer {
                    name: "cat.png".into(),
                    blob: blob.clone(),
                    direction: FileTransferDirection::Send,
                    state: FileTransferState::Negotiating,
                    file_size: 1337,
                    content_type: Some("image/png".into()),
                    message: Some("here you go".into()),
                    ..Default::default()
                },
            },
        )?;
        write_cbor(
            &out,
            "request-ft-receive.cbor",
            &ApiRequest::FtReceive {
                transport: TransportId::new("matrix/@bot:localhost"),
                transfer: FileTransfer {
                    name: "cat.png".into(),
                    blob,
                    direction: FileTransferDirection::Receive,
                    file_size: 1337,
                    source: Some("mxc://localhost/abc123".into()),
                    ..Default::default()
                },
            },
        )?;
    }

    // -- transport account settings (N2; wire v38) --------------------------------------------
    // The settings read + merge-edit of a transport instance's persisted NON-SECRET values, so
    // verify-codec proves the generated zcbor C decoder accepts the map-carrying shapes.
    {
        use daemon_api::AccountSettingsValues;

        let mut values = std::collections::BTreeMap::new();
        values.insert("server".to_string(), "hs.example.org".to_string());
        values.insert("nick".to_string(), "daemon-bot".to_string());
        write_cbor(
            &out,
            "request-transport-settings.cbor",
            &ApiRequest::TransportSettings {
                transport: TransportId::new("matrix/@bot:hs.org"),
            },
        )?;
        write_cbor(
            &out,
            "request-transport-configure.cbor",
            &ApiRequest::TransportConfigure {
                transport: TransportId::new("matrix/@bot:hs.org"),
                settings: AccountSettingsValues {
                    values: values.clone(),
                },
            },
        )?;
        write_cbor(
            &out,
            "response-transport-settings.cbor",
            &ApiResponse::TransportSettings(AccountSettingsValues { values }),
        )?;
    }

    println!("generated CBOR fixtures in {}", out.display());
    Ok(())
}

fn write_cbor<T: serde::Serialize>(dir: &Path, name: &str, value: &T) -> anyhow::Result<()> {
    let bytes = daemon_api::to_cbor(value);
    std::fs::write(dir.join(name), bytes)?;
    Ok(())
}

/// Base name passed to the codegen script; the generated entry types are `api_request`/`api_response`.
const ZCBOR_BASENAME: &str = "daemon_api_client";

fn codegen_script(root: &Path) -> PathBuf {
    root.join("crates/contracts/daemon-api/zcbor-codegen.sh")
}

/// The single authoritative CDDL. It is authored in zcbor dialect (quoted map keys, named union
/// arms, labeled tuples, `any` for opaque fields, plus a few `-t` rule-name disambiguators) so the
/// one file both documents the full wire contract and generates the client C codec. `verify-codec`
/// proves the generated decoder accepts real ciborium fixtures; the `daemon-api` cddl-cat
/// conformance tests prove the schema matches the serde wire format.
fn default_cddl(root: &Path) -> PathBuf {
    root.join("crates/contracts/daemon-api/daemon-api.cddl")
}

/// Run the canonical codegen script. `extra` forwards flags such as `--copy-sources`.
fn run_codegen(root: &Path, cddl: &Path, out: &Path, extra: &[&str]) -> anyhow::Result<()> {
    let status = Command::new("bash")
        .arg(codegen_script(root))
        .arg(cddl)
        .arg(out)
        .args(extra)
        .status()
        .map_err(|e| {
            anyhow::anyhow!(
                "running zcbor-codegen.sh (is zcbor on PATH / in the flake shell?): {e}"
            )
        })?;
    anyhow::ensure!(status.success(), "zcbor codegen failed with {status}");
    Ok(())
}

/// `gen-zcbor [--cddl <path>] [--out <dir>]` — (re)generate the client CBOR codec.
///
/// A thin dev wrapper over `zcbor-codegen.sh`. daemon-node owns generation because the CDDL is
/// authoritative here and zcbor lives in this flake; the output is the committed artifact
/// `daemon-app` vendors (no Python/zcbor in the Qt build). The superproject's pure
/// `packages.daemon-zcbor-codec` derivation invokes the same script.
fn gen_zcbor(cddl: Option<PathBuf>, out: Option<PathBuf>) -> anyhow::Result<()> {
    let root = workspace_root();
    let cddl = cddl.unwrap_or_else(|| default_cddl(&root));
    let out = out.unwrap_or_else(|| root.join("target/zcbor-codec"));
    run_codegen(&root, &cddl, &out, &[])?;
    println!(
        "generated zcbor codec from {} in {}",
        cddl.display(),
        out.display()
    );
    Ok(())
}

/// The verify-codec harness: decode every ciborium-produced fixture with the zcbor-generated decoder.
/// A `response-*` filename is decoded as `api_response`, anything else as `api_request`; success
/// means the generated decoder accepted the bytes (ZCBOR_SUCCESS) and consumed all of them.
const VERIFY_CODEC_C: &str = r#"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "daemon_api_client_decode.h"

static unsigned char buf[1u << 20];

int main(int argc, char **argv) {
    int failures = 0;
    for (int i = 1; i < argc; i++) {
        const char *path = argv[i];
        FILE *f = fopen(path, "rb");
        if (!f) {
            fprintf(stderr, "FAIL %s: cannot open\n", path);
            failures++;
            continue;
        }
        size_t n = fread(buf, 1, sizeof buf, f);
        fclose(f);

        const char *base = strrchr(path, '/');
        base = base ? base + 1 : path;

        size_t consumed = 0;
        int ret;
        if (strncmp(base, "response", 8) == 0) {
            struct api_response_r *r = calloc(1, sizeof *r);
            ret = cbor_decode_api_response(buf, n, r, &consumed);
            free(r);
        } else {
            struct api_request_r *r = calloc(1, sizeof *r);
            ret = cbor_decode_api_request(buf, n, r, &consumed);
            free(r);
        }

        if (ret != 0) {
            fprintf(stderr, "FAIL %s: zcbor decode error %d\n", base, ret);
            failures++;
        } else if (consumed != n) {
            fprintf(stderr, "FAIL %s: decoded %zu of %zu bytes\n", base, consumed, n);
            failures++;
        } else {
            fprintf(stderr, "ok   %s (%zu bytes)\n", base, n);
        }
    }

    if (failures) {
        fprintf(stderr, "%d fixture(s) failed to decode\n", failures);
        return 1;
    }
    fprintf(stderr, "all fixtures decoded with the generated zcbor codec\n");
    return 0;
}
"#;

/// `verify-codec` — prove the generated C codec accepts real ciborium wire bytes.
///
/// Closes the loop the syntactic `cddl` gate cannot: generate the codec from the CDDL, compile its
/// decoder with the zcbor runtime, then decode every `fixtures/cbor/*.cbor` (each emitted by
/// `api-fixtures` through ciborium — the runtime truth) and assert success + full consumption. Any
/// drift between the serde wire format and the CDDL/zcbor path fails here.
fn verify_codec() -> anyhow::Result<()> {
    let root = workspace_root();

    let fixtures_dir = root.join("crates/contracts/daemon-api/fixtures/cbor");
    if !fixtures_dir.exists() {
        gen_api_fixtures()?;
    }
    let mut fixtures: Vec<PathBuf> = std::fs::read_dir(&fixtures_dir)
        .map_err(|e| anyhow::anyhow!("reading {}: {e}", fixtures_dir.display()))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().map(|ext| ext == "cbor").unwrap_or(false))
        // The multiplexed-envelope fixtures (`wire-c2s-*` / `wire-s2c-*`) are NOT `api-request` /
        // `api-response`, and the vendored C codec is deliberately scoped to those two entry types
        // (the client hand-codes the tiny envelope). Their schema is covered by the cddl-cat
        // conformance test against `wire-c2s` / `wire-s2c`, not this generated-decoder harness.
        .filter(|path| {
            !path
                .file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with("wire-"))
                .unwrap_or(false)
        })
        .collect();
    fixtures.sort();
    anyhow::ensure!(
        !fixtures.is_empty(),
        "no CBOR fixtures in {}",
        fixtures_dir.display()
    );

    // Decode every committed fixture with the generated codec (an independent C cross-check of the
    // serde wire bytes). Per-variant coverage is no longer asserted here: the unified CDDL now spans
    // the full surface (~150 variants), and the comprehensive "Rust output always matches the CDDL"
    // gate is the cddl-cat round-trip + proptest conformance in the `daemon-api` crate. This harness
    // proves the zcbor-generated decoder agrees with ciborium on the fixtures that exist.
    let work = std::env::temp_dir().join(format!("daemon-verify-codec-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&work);
    let codec = work.join("codec");
    std::fs::create_dir_all(&codec)?;
    // `--copy-sources` drops the zcbor C runtime flat alongside the generated codec.
    run_codegen(&root, &default_cddl(&root), &codec, &["--copy-sources"])?;

    let harness_c = work.join("verify_codec.c");
    std::fs::write(&harness_c, VERIFY_CODEC_C)?;
    let bin = work.join("verify-codec");

    let status = Command::new("cc")
        .arg(&harness_c)
        .arg(codec.join(format!("{ZCBOR_BASENAME}_decode.c")))
        .arg(codec.join("zcbor_decode.c"))
        .arg(codec.join("zcbor_common.c"))
        .arg(format!("-I{}", codec.display()))
        .arg("-o")
        .arg(&bin)
        .status()
        .map_err(|e| anyhow::anyhow!("failed to run cc (is it in the flake shell?): {e}"))?;
    anyhow::ensure!(
        status.success(),
        "compiling the verify harness failed with {status}"
    );

    let status = Command::new(&bin).args(&fixtures).status()?;
    anyhow::ensure!(
        status.success(),
        "codec verification failed: a fixture did not decode with the generated codec"
    );
    let _ = std::fs::remove_dir_all(&work);
    println!(
        "verified {} fixtures decode with the generated zcbor codec",
        fixtures.len()
    );
    Ok(())
}
