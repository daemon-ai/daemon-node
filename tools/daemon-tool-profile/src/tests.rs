// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Unit tests for the `profile_manage` tool: id namespacing, name validation, and the
//! subtree-scoped create/edit/delete dispatch (over a validator-less [`ProfileOps`] — the shared
//! validation engine is proved end-to-end in the conformance suite).

use super::*;
use daemon_host::{MemProfileStore, ProfileStore};
use daemon_store::{InMemoryStore, SessionMeta};
use std::sync::Arc;

/// A tool over fresh in-memory stores (no validator wired => store-only, so create/edit persist
/// without a node); returns the profile store so a test can inspect the persisted spec.
fn tool_with_stores() -> (ProfileManageTool, Arc<dyn ProfileStore>, Arc<InMemoryStore>) {
    let profiles: Arc<dyn ProfileStore> = Arc::new(MemProfileStore::new());
    let ops = Arc::new(ProfileOps::new(profiles.clone()));
    let sessions = Arc::new(InMemoryStore::new());
    let tool = ProfileManageTool::new(ops, sessions.clone());
    (tool, profiles, sessions)
}

fn create_args(name: &str) -> ManageArgs {
    ManageArgs {
        action: "create".into(),
        name: Some(name.into()),
        ..Default::default()
    }
}

#[test]
fn id_namespacing_round_trips() {
    let id = agent_profile_id(&SessionId::new("s1"), "helper");
    assert_eq!(id, "agent/s1/helper");
    assert_eq!(
        parse_authoring_session(&id).unwrap().as_str(),
        "s1",
        "the authoring session is recovered from the id"
    );
    // A session id that itself embeds lineage is preserved (the NAME is the final segment).
    let nested = agent_profile_id(&SessionId::new("s1/c2"), "helper");
    assert_eq!(nested, "agent/s1/c2/helper");
    assert_eq!(parse_authoring_session(&nested).unwrap().as_str(), "s1/c2");
    // An operator profile (no `agent/` prefix) and a malformed id are not agent-owned.
    assert!(parse_authoring_session("opus").is_none());
    assert!(parse_authoring_session("agent/onlyone").is_none());
    assert!(parse_authoring_session("agent//p").is_none());
    assert!(parse_authoring_session("agent/s1/").is_none());
}

#[test]
fn name_validation_rejects_separators_control_and_oversize() {
    assert!(check_name("helper").is_ok());
    assert!(check_name("").is_err());
    assert!(check_name("  ").is_err());
    assert!(check_name("a/b").is_err());
    assert!(check_name("a\\b").is_err());
    assert!(check_name("a\0b").is_err());
    assert!(check_name(&"x".repeat(MAX_NAME_LEN + 1)).is_err());
}

#[tokio::test]
async fn create_persists_under_the_agent_namespace() {
    let (tool, profiles, _sessions) = tool_with_stores();
    let caller = SessionId::new("s1");
    let mut args = create_args("researcher");
    args.system_prompt = Some("a focused researcher".into());
    args.tool_allowlist = Some(vec!["fs".into(), "web_search".into()]);
    args.engine = Some(EngineSelector::Foreign {
        agent: "gemini".into(),
    });

    let msg = tool.dispatch(&caller, args).await.expect("create succeeds");
    assert!(msg.contains("agent/s1/researcher"));

    let spec = profiles
        .get("agent/s1/researcher")
        .unwrap()
        .expect("the created profile is persisted under the agent namespace");
    assert_eq!(spec.system_prompt, "a focused researcher");
    assert_eq!(
        spec.tool_allowlist,
        Some(vec!["fs".to_string(), "web_search".to_string()])
    );
    assert_eq!(
        spec.engine,
        EngineSelector::Foreign {
            agent: "gemini".into()
        },
        "the authored engine building block is set as an ordinary field"
    );
}

#[tokio::test]
async fn create_requires_a_valid_name() {
    let (tool, _profiles, _sessions) = tool_with_stores();
    let caller = SessionId::new("s1");
    // Missing name.
    let mut args = ManageArgs {
        action: "create".into(),
        ..Default::default()
    };
    assert!(tool.dispatch(&caller, args).await.is_err());
    // A separator in the name would escape the namespace.
    args = create_args("a/b");
    assert!(tool.dispatch(&caller, args).await.is_err());
}

#[tokio::test]
async fn edit_and_delete_manage_own_profile() {
    let (tool, profiles, _sessions) = tool_with_stores();
    let caller = SessionId::new("s1");
    tool.dispatch(&caller, create_args("helper"))
        .await
        .expect("create");

    // Edit the caller's OWN profile (reflexive ownership).
    let edit = ManageArgs {
        action: "edit".into(),
        id: Some("agent/s1/helper".into()),
        model: Some("claude-opus-4-8".into()),
        ..Default::default()
    };
    tool.dispatch(&caller, edit)
        .await
        .expect("edit own profile");
    assert_eq!(
        profiles.get("agent/s1/helper").unwrap().unwrap().model,
        "claude-opus-4-8"
    );

    // Delete the caller's OWN profile.
    let del = ManageArgs {
        action: "delete".into(),
        id: Some("agent/s1/helper".into()),
        ..Default::default()
    };
    tool.dispatch(&caller, del)
        .await
        .expect("delete own profile");
    assert!(profiles.get("agent/s1/helper").unwrap().is_none());
}

#[tokio::test]
async fn edit_allowed_for_a_descendant_via_id_prefix() {
    let (tool, profiles, _sessions) = tool_with_stores();
    // A descendant session (`s1/c2`) authored a profile; the ancestor `s1` owns the subtree.
    profiles
        .create(ProfileSpec::new(
            "agent/s1/c2/child-helper",
            ProviderSelector::Mock,
            "m",
        ))
        .unwrap();
    let ancestor = SessionId::new("s1");
    let edit = ManageArgs {
        action: "edit".into(),
        id: Some("agent/s1/c2/child-helper".into()),
        system_prompt: Some("edited by ancestor".into()),
        ..Default::default()
    };
    tool.dispatch(&ancestor, edit)
        .await
        .expect("an ancestor may manage a descendant's profile");
    assert_eq!(
        profiles
            .get("agent/s1/c2/child-helper")
            .unwrap()
            .unwrap()
            .system_prompt,
        "edited by ancestor"
    );
}

#[tokio::test]
async fn manage_denied_outside_subtree_and_for_operator_profiles() {
    let (tool, profiles, _sessions) = tool_with_stores();
    // A sibling-authored agent profile and an operator profile both exist.
    profiles
        .create(ProfileSpec::new(
            "agent/s1/c3/sibling",
            ProviderSelector::Mock,
            "m",
        ))
        .unwrap();
    profiles
        .create(ProfileSpec::new(
            "opus",
            ProviderSelector::GenAi,
            "claude-opus-4-8",
        ))
        .unwrap();

    let caller = SessionId::new("s1/c2");
    // A sibling's profile is outside the caller's subtree.
    let edit_sibling = ManageArgs {
        action: "edit".into(),
        id: Some("agent/s1/c3/sibling".into()),
        system_prompt: Some("nope".into()),
        ..Default::default()
    };
    assert!(
        tool.dispatch(&caller, edit_sibling).await.is_err(),
        "a sibling's profile must not be editable"
    );

    // An ancestor's profile is outside the caller's subtree (not reflexive, not a descendant).
    let del_ancestor = ManageArgs {
        action: "delete".into(),
        id: Some("agent/s1/helper".into()),
        ..Default::default()
    };
    // (No such profile is even needed — the ownership check fails first.)
    assert!(tool.dispatch(&caller, del_ancestor).await.is_err());

    // An operator profile is never agent-managed.
    let del_operator = ManageArgs {
        action: "delete".into(),
        id: Some("opus".into()),
        ..Default::default()
    };
    assert!(
        tool.dispatch(&caller, del_operator).await.is_err(),
        "an operator profile must not be deletable by the agent tool"
    );
    // ...and it is untouched.
    assert!(profiles.get("opus").unwrap().is_some());
}

#[tokio::test]
async fn subtree_walk_fallback_authorizes_a_reparented_authoring_session() {
    let (tool, profiles, sessions) = tool_with_stores();
    // A profile whose authoring session id does NOT embed the caller (a re-parented session), so the
    // id-prefix fast path misses and the bounded `SessionMeta.parent` walk must authorize it.
    profiles
        .create(ProfileSpec::new(
            "agent/orphan/p",
            ProviderSelector::Mock,
            "m",
        ))
        .unwrap();
    sessions
        .set_session_meta(
            &SessionId::new("orphan"),
            SessionMeta {
                parent: Some(SessionId::new("root")),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let root = SessionId::new("root");
    let edit = ManageArgs {
        action: "edit".into(),
        id: Some("agent/orphan/p".into()),
        model: Some("m2".into()),
        ..Default::default()
    };
    tool.dispatch(&root, edit)
        .await
        .expect("the parent walk authorizes a re-parented authoring session");
    assert_eq!(profiles.get("agent/orphan/p").unwrap().unwrap().model, "m2");
}
