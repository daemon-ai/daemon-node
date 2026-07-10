// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `MessagingProtocol` conformance suite — the daemon port of libpurple's protocol test fixtures
//! (`libpurple/tests/test_protocol*.c`). Each module mirrors one libpurple test file.
//!
//! libpurple's async callback + `implements_*` idioms translate to daemon-api as: `supported()`
//! ops flags (⟵ `implements_*`), `Ok`/`Err` returns (⟵ the async `finish` result), and the
//! per-fixture `should_error` boolean (⟵ [`FakeProtocol::failing`]). The Empty side maps to
//! [`EmptyProtocol`], the non-empty side to [`FakeProtocol`].

use std::sync::Arc;

use daemon_api::{
    AccountSettingsSchema, AccountSettingsValues, ApiError, ContactInfo, ConvSendArgs,
    CreateConversationDetails, MessagingProtocol, TransportAdapter,
};
use daemon_api_testkit::{
    assert_ops_match_behavior, sentinels, EmptyProtocol, FailSwitches, FakeProtocol,
};
use daemon_protocol::{TransportId, UserMsg};

/// A stable transport id for the ported cases.
fn t() -> TransportId {
    TransportId::new("conformance")
}

/// Assert `res` is exactly the capability-sentinel `Unsupported` (the libpurple
/// "not implemented → error" outcome).
#[track_caller]
fn assert_unsupported<T: std::fmt::Debug>(res: Result<T, ApiError>, sentinel: &str) {
    match res {
        Err(ApiError::Unsupported(s)) => assert_eq!(s, sentinel, "wrong sentinel"),
        other => panic!("expected Err(Unsupported({sentinel:?})), got {other:?}"),
    }
}

/// Assert `res` is a non-`Unsupported` error (the libpurple `should_error` fixture path). The Fake
/// returns [`ApiError::Other`] so an error path never collides with the capability sentinel.
#[track_caller]
fn assert_error<T: std::fmt::Debug>(res: Result<T, ApiError>) {
    match res {
        Err(ApiError::Other(_)) => {}
        other => panic!("expected Err(Other(_)), got {other:?}"),
    }
}

// ===========================================================================
// test_protocol_conversation.c
// ===========================================================================
mod conversation {
    use super::*;

    fn empty() -> Arc<dyn daemon_api::SupportsConversations> {
        EmptyProtocol::new().conversations().unwrap()
    }
    fn fake() -> Arc<dyn daemon_api::SupportsConversations> {
        FakeProtocol::new().conversations().unwrap()
    }
    fn fake_failing() -> Arc<dyn daemon_api::SupportsConversations> {
        FakeProtocol::failing().conversations().unwrap()
    }

    // ---- Empty ----
    #[tokio::test]
    async fn conv_empty_implements_create() {
        assert!(!empty().supported().create);
    }
    #[tokio::test]
    async fn conv_empty_create_details_default() {
        // Divergence: libpurple's empty returns NULL + warning; daemon models this as an infallible
        // getter that returns the default value.
        assert_eq!(
            empty().create_details(t()).await,
            CreateConversationDetails::default()
        );
    }
    #[tokio::test]
    async fn conv_empty_create_unsupported() {
        assert_unsupported(
            empty()
                .create(t(), CreateConversationDetails::default())
                .await,
            sentinels::CONV_CREATE,
        );
    }
    #[tokio::test]
    async fn conv_empty_implements_leave() {
        assert!(!empty().supported().leave);
    }
    #[tokio::test]
    async fn conv_empty_leave_unsupported() {
        assert_unsupported(empty().leave(t(), "c".into()).await, sentinels::CONV_LEAVE);
    }
    #[tokio::test]
    async fn conv_empty_implements_send() {
        assert!(!empty().supported().send);
    }
    #[tokio::test]
    async fn conv_empty_send_unsupported() {
        let args = ConvSendArgs {
            transport: t(),
            conv: "c".into(),
            from: None,
            message: UserMsg::new("hi"),
        };
        assert_unsupported(empty().send(args).await, sentinels::CONV_SEND);
    }
    #[tokio::test]
    async fn conv_empty_implements_set_topic() {
        assert!(!empty().supported().set_topic);
    }
    #[tokio::test]
    async fn conv_empty_set_topic_unsupported() {
        assert_unsupported(
            empty().set_topic(t(), "c".into(), Some("x".into())).await,
            sentinels::CONV_SET_TOPIC,
        );
    }
    #[tokio::test]
    async fn conv_empty_channel_join_details_default() {
        // Divergence: libpurple's empty returns NULL + warning; daemon returns the default value.
        assert_eq!(empty().channel_join_details(t()).await, Default::default());
    }
    #[tokio::test]
    async fn conv_empty_join_channel_unsupported() {
        assert_unsupported(
            empty().join_channel(t(), Default::default()).await,
            sentinels::CONV_JOIN,
        );
    }
    #[tokio::test]
    async fn conv_empty_implements_set_title() {
        assert!(!empty().supported().set_title);
    }
    #[tokio::test]
    async fn conv_empty_set_title_unsupported() {
        assert_unsupported(
            empty().set_title(t(), "c".into(), Some("x".into())).await,
            sentinels::CONV_SET_TITLE,
        );
    }
    #[tokio::test]
    async fn conv_empty_implements_set_description() {
        assert!(!empty().supported().set_description);
    }
    #[tokio::test]
    async fn conv_empty_set_description_unsupported() {
        assert_unsupported(
            empty()
                .set_description(t(), "c".into(), Some("x".into()))
                .await,
            sentinels::CONV_SET_DESCRIPTION,
        );
    }

    // ---- Normal (Fake) ----
    #[tokio::test]
    async fn conv_fake_implements_create() {
        assert!(fake().supported().create);
    }
    #[tokio::test]
    async fn conv_fake_create_details_value() {
        // Mirrors libpurple's fixture `purple_create_conversation_details_new(10)`.
        assert_eq!(fake().create_details(t()).await.max_participants, 10);
    }
    #[tokio::test]
    async fn conv_fake_create_ok() {
        assert!(fake()
            .create(t(), CreateConversationDetails::default())
            .await
            .is_ok());
    }
    #[tokio::test]
    async fn conv_fake_create_error() {
        assert_error(
            fake_failing()
                .create(t(), CreateConversationDetails::default())
                .await,
        );
    }
    #[tokio::test]
    async fn conv_fake_implements_leave() {
        assert!(fake().supported().leave);
    }
    #[tokio::test]
    async fn conv_fake_leave_ok() {
        assert!(fake().leave(t(), "c".into()).await.is_ok());
    }
    #[tokio::test]
    async fn conv_fake_leave_error() {
        assert_error(fake_failing().leave(t(), "c".into()).await);
    }
    #[tokio::test]
    async fn conv_fake_implements_send() {
        assert!(fake().supported().send);
    }
    fn send_args() -> ConvSendArgs {
        ConvSendArgs {
            transport: t(),
            conv: "c".into(),
            from: None,
            message: UserMsg::new("hi"),
        }
    }
    #[tokio::test]
    async fn conv_fake_send_ok() {
        assert!(fake().send(send_args()).await.is_ok());
    }
    #[tokio::test]
    async fn conv_fake_send_error() {
        assert_error(fake_failing().send(send_args()).await);
    }
    #[tokio::test]
    async fn conv_fake_channel_join_details_value() {
        // Mirrors `purple_channel_join_details_new(16, TRUE, 16, TRUE, 0)`.
        let d = fake().channel_join_details(t()).await;
        assert_eq!(d.name_max_length, 16);
        assert!(d.nickname_supported);
        assert_eq!(d.nickname_max_length, 16);
        assert!(d.password_supported);
        assert_eq!(d.password_max_length, 0);
    }
    #[tokio::test]
    async fn conv_fake_join_channel_ok() {
        assert!(fake().join_channel(t(), Default::default()).await.is_ok());
    }
    #[tokio::test]
    async fn conv_fake_join_channel_error() {
        assert_error(fake_failing().join_channel(t(), Default::default()).await);
    }
    #[tokio::test]
    async fn conv_fake_set_topic_ok() {
        assert!(fake()
            .set_topic(t(), "c".into(), Some("x".into()))
            .await
            .is_ok());
    }
    #[tokio::test]
    async fn conv_fake_set_topic_error() {
        assert_error(
            fake_failing()
                .set_topic(t(), "c".into(), Some("x".into()))
                .await,
        );
    }
    #[tokio::test]
    async fn conv_fake_set_title_ok() {
        assert!(fake()
            .set_title(t(), "c".into(), Some("x".into()))
            .await
            .is_ok());
    }
    #[tokio::test]
    async fn conv_fake_set_title_error() {
        assert_error(
            fake_failing()
                .set_title(t(), "c".into(), Some("x".into()))
                .await,
        );
    }
    #[tokio::test]
    async fn conv_fake_set_description_ok() {
        assert!(fake()
            .set_description(t(), "c".into(), Some("x".into()))
            .await
            .is_ok());
    }
    #[tokio::test]
    async fn conv_fake_set_description_error() {
        assert_error(
            fake_failing()
                .set_description(t(), "c".into(), Some("x".into()))
                .await,
        );
    }
}

// ===========================================================================
// test_protocol_contacts.c
// ===========================================================================
mod contacts {
    use super::*;

    fn empty() -> Arc<dyn daemon_api::SupportsContacts> {
        EmptyProtocol::new().contacts().unwrap()
    }
    fn fake() -> Arc<dyn daemon_api::SupportsContacts> {
        FakeProtocol::new().contacts().unwrap()
    }
    fn fake_failing() -> Arc<dyn daemon_api::SupportsContacts> {
        FakeProtocol::failing().contacts().unwrap()
    }

    #[tokio::test]
    async fn contacts_empty_implements_get_profile() {
        assert!(!empty().supported().get_profile);
    }
    #[tokio::test]
    async fn contacts_empty_get_profile_unsupported() {
        assert_unsupported(
            empty().get_profile(t(), ContactInfo::default()).await,
            sentinels::CONTACT_GET_PROFILE,
        );
    }
    #[tokio::test]
    async fn contacts_empty_implements_action_menu() {
        assert!(!empty().supported().action_menu);
    }
    #[tokio::test]
    async fn contacts_empty_action_menu_none() {
        assert!(empty().action_menu(t(), ContactInfo::default()).is_none());
    }
    #[tokio::test]
    async fn contacts_empty_implements_set_alias() {
        assert!(!empty().supported().set_alias);
    }
    #[tokio::test]
    async fn contacts_empty_set_alias_unsupported() {
        assert_unsupported(
            empty()
                .set_alias(t(), ContactInfo::default(), Some("a".into()))
                .await,
            sentinels::CONTACT_SET_ALIAS,
        );
    }

    #[tokio::test]
    async fn contacts_fake_implements_get_profile() {
        assert!(fake().supported().get_profile);
    }
    #[tokio::test]
    async fn contacts_fake_get_profile_ok() {
        // Mirrors the libpurple fixture returning `"profile data"`.
        assert_eq!(
            fake()
                .get_profile(t(), ContactInfo::default())
                .await
                .unwrap(),
            "profile data"
        );
    }
    #[tokio::test]
    async fn contacts_fake_get_profile_error() {
        assert_error(
            fake_failing()
                .get_profile(t(), ContactInfo::default())
                .await,
        );
    }
    #[tokio::test]
    async fn contacts_fake_implements_action_menu() {
        assert!(fake().supported().action_menu);
    }
    #[tokio::test]
    async fn contacts_fake_action_menu_some() {
        assert!(fake().action_menu(t(), ContactInfo::default()).is_some());
    }
    #[tokio::test]
    async fn contacts_fake_implements_set_alias() {
        assert!(fake().supported().set_alias);
    }
    #[tokio::test]
    async fn contacts_fake_set_alias_ok() {
        assert!(fake()
            .set_alias(t(), ContactInfo::default(), Some("new-alias".into()))
            .await
            .is_ok());
    }
    #[tokio::test]
    async fn contacts_fake_set_alias_error() {
        assert_error(
            fake_failing()
                .set_alias(t(), ContactInfo::default(), Some("bad".into()))
                .await,
        );
    }
}

// ===========================================================================
// test_protocol_roster.c
// ===========================================================================
mod roster {
    use super::*;

    fn empty() -> Arc<dyn daemon_api::SupportsRoster> {
        EmptyProtocol::new().roster().unwrap()
    }
    fn fake() -> Arc<dyn daemon_api::SupportsRoster> {
        FakeProtocol::new().roster().unwrap()
    }
    fn fake_failing() -> Arc<dyn daemon_api::SupportsRoster> {
        FakeProtocol::failing().roster().unwrap()
    }

    #[tokio::test]
    async fn roster_empty_add_unsupported() {
        assert_unsupported(
            empty().add(t(), ContactInfo::default()).await,
            sentinels::ROSTER_ADD,
        );
    }
    #[tokio::test]
    async fn roster_empty_update_unsupported() {
        assert_unsupported(
            empty().update(t(), ContactInfo::default()).await,
            sentinels::ROSTER_UPDATE,
        );
    }
    #[tokio::test]
    async fn roster_empty_remove_unsupported() {
        assert_unsupported(
            empty().remove(t(), ContactInfo::default()).await,
            sentinels::ROSTER_REMOVE,
        );
    }

    #[tokio::test]
    async fn roster_fake_add_ok() {
        assert!(fake().add(t(), ContactInfo::default()).await.is_ok());
    }
    #[tokio::test]
    async fn roster_fake_add_error() {
        assert_error(fake_failing().add(t(), ContactInfo::default()).await);
    }
    #[tokio::test]
    async fn roster_fake_update_ok() {
        assert!(fake().update(t(), ContactInfo::default()).await.is_ok());
    }
    #[tokio::test]
    async fn roster_fake_update_error() {
        assert_error(fake_failing().update(t(), ContactInfo::default()).await);
    }
    #[tokio::test]
    async fn roster_fake_remove_ok() {
        assert!(fake().remove(t(), ContactInfo::default()).await.is_ok());
    }
    #[tokio::test]
    async fn roster_fake_remove_error() {
        assert_error(fake_failing().remove(t(), ContactInfo::default()).await);
    }
}

// ===========================================================================
// test_protocol_directory.c
// ===========================================================================
mod directory {
    use super::*;

    fn fake() -> Arc<dyn daemon_api::SupportsDirectory> {
        FakeProtocol::new().directory().unwrap()
    }
    fn fake_failing() -> Arc<dyn daemon_api::SupportsDirectory> {
        FakeProtocol::failing().directory().unwrap()
    }

    #[tokio::test]
    async fn directory_fake_search_ok() {
        assert!(fake()
            .search_contacts(t(), Some("bob".into()))
            .await
            .is_ok());
    }
    #[tokio::test]
    async fn directory_fake_search_error() {
        assert_error(
            fake_failing()
                .search_contacts(t(), Some("bob".into()))
                .await,
        );
    }
}

// ===========================================================================
// test_protocol_file_transfer.c (EMPTY side only; normal/error skipped → W2-H)
// ===========================================================================
mod file_transfer {
    use super::*;

    fn empty() -> Arc<dyn daemon_api::SupportsFileTransfer> {
        EmptyProtocol::new().file_transfer().unwrap()
    }
    fn fake() -> Arc<dyn daemon_api::SupportsFileTransfer> {
        FakeProtocol::new().file_transfer().unwrap()
    }
    fn fake_failing() -> Arc<dyn daemon_api::SupportsFileTransfer> {
        FakeProtocol::failing().file_transfer().unwrap()
    }

    fn transfer() -> daemon_api::FileTransfer {
        daemon_api::FileTransfer {
            name: "file.png".into(),
            blob: daemon_common::BlobRef::new(daemon_common::ContentHash::new([0u8; 32]), 0),
            ..Default::default()
        }
    }

    // ---- Empty (Wave 1) ----
    #[tokio::test]
    async fn ft_empty_implements_and_send_unsupported() {
        let ft = empty();
        assert!(!ft.supported().send);
        assert_unsupported(
            ft.send(t(), transfer()).await,
            sentinels::FILE_TRANSFER_SEND,
        );
    }
    #[tokio::test]
    async fn ft_empty_implements_and_receive_unsupported() {
        let ft = empty();
        assert!(!ft.supported().receive);
        assert_unsupported(
            ft.receive(t(), transfer()).await,
            sentinels::FILE_TRANSFER_RECEIVE,
        );
    }

    // ---- Normal / error (← /protocol-file-transfer/normal/*; W2-H) ----
    #[tokio::test]
    async fn ft_fake_implements_and_send_ok() {
        // ⟵ /protocol-file-transfer/normal/send-normal: implements_send + a successful finish.
        let ft = fake();
        assert!(ft.supported().send);
        assert!(ft.send(t(), transfer()).await.is_ok());
    }
    #[tokio::test]
    async fn ft_fake_send_error() {
        // ⟵ /protocol-file-transfer/normal/send-error: should_error → a non-sentinel Err.
        assert_error(fake_failing().send(t(), transfer()).await);
    }
    #[tokio::test]
    async fn ft_fake_implements_and_receive_ok() {
        // ⟵ /protocol-file-transfer/normal/receive-normal.
        let ft = fake();
        assert!(ft.supported().receive);
        assert!(ft.receive(t(), transfer()).await.is_ok());
    }
    #[tokio::test]
    async fn ft_fake_receive_error() {
        // ⟵ /protocol-file-transfer/normal/receive-error.
        assert_error(fake_failing().receive(t(), transfer()).await);
    }
}

// ===========================================================================
// test_protocol.c (validate/account scope; account-manager lifecycle skipped → node-owned)
// ===========================================================================
mod protocol {
    use super::*;

    #[tokio::test]
    async fn protocol_fake_descriptor_identity() {
        // ⟵ /protocol/properties: adapter identity (family + display name).
        let p = FakeProtocol::new();
        assert_eq!(p.family(), "fake");
        assert_eq!(p.info().display_name, "Fake");
    }
    #[tokio::test]
    async fn protocol_fake_default_account_settings() {
        // ⟵ /protocol/get-default-account-settings: the adapter's account-setup form.
        assert_eq!(
            FakeProtocol::new().info().account_schema,
            AccountSettingsSchema::default()
        );
    }
    #[tokio::test]
    async fn protocol_fake_validate_account_ok() {
        // ⟵ /protocol/validate-account.
        assert!(FakeProtocol::new()
            .validate_account(&AccountSettingsValues::default())
            .await
            .is_ok());
    }
    #[tokio::test]
    async fn protocol_empty_validate_account_ok() {
        // The trait-default `validate_account` accepts (daemon-specific: no libpurple empty row).
        assert!(EmptyProtocol::new()
            .validate_account(&AccountSettingsValues::default())
            .await
            .is_ok());
    }
    #[tokio::test]
    async fn protocol_fake_validate_account_error() {
        // daemon-specific: the Fake's validate switch surfaces the `Result` error path.
        assert_error(
            FakeProtocol::failing()
                .validate_account(&AccountSettingsValues::default())
                .await,
        );
    }
    #[tokio::test]
    async fn protocol_fake_validate_account_rejects_marker_value() {
        // daemon-specific (N2, wire v38): an otherwise-operable Fake rejects a settings map
        // carrying the marker VALUE with a non-`Unsupported` error — the adapter-side validation
        // failure the host's `transport_configure` op must surface.
        let mut values = std::collections::BTreeMap::new();
        values.insert(
            "nick".to_string(),
            FakeProtocol::VALIDATE_REJECT_VALUE.to_string(),
        );
        assert_error(
            FakeProtocol::new()
                .validate_account(&AccountSettingsValues { values })
                .await,
        );
        // Marker-free maps still validate (the rejection is value-keyed, not a global switch).
        let mut ok = std::collections::BTreeMap::new();
        ok.insert("nick".to_string(), "daemon-bot".to_string());
        assert!(FakeProtocol::new()
            .validate_account(&AccountSettingsValues { values: ok })
            .await
            .is_ok());
    }
}

// ===========================================================================
// Harness self-tests — the ops<->behavior invariant against the reference impls.
// ===========================================================================
mod invariant {
    use super::*;

    #[tokio::test]
    async fn invariant_empty_matches_behavior() {
        let p: Arc<dyn MessagingProtocol> = EmptyProtocol::new();
        assert_ops_match_behavior(p).await;
    }
    #[tokio::test]
    async fn invariant_fake_matches_behavior() {
        let p: Arc<dyn MessagingProtocol> = FakeProtocol::new();
        assert_ops_match_behavior(p).await;
    }
    #[tokio::test]
    async fn invariant_fake_all_failing_matches_behavior() {
        // Even with every verb switched to fail, advertised verbs return `Other`, never the
        // capability sentinel — so the biconditional still holds.
        let p: Arc<dyn MessagingProtocol> = FakeProtocol::failing();
        assert_ops_match_behavior(p).await;
    }
    #[tokio::test]
    async fn invariant_fake_targeted_failures() {
        // A custom switch set is still sentinel-clean on the failing verbs.
        let p: Arc<dyn MessagingProtocol> =
            FakeProtocol::with_failures(FailSwitches::only(&[sentinels::CONV_SEND]));
        assert_ops_match_behavior(p).await;
    }
}
