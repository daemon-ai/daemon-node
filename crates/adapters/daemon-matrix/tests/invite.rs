// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

// Test code may use raw fs/reqwest/etc.; the --lib pass still guards production.
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

//! Vertical tests for the invite-acceptance path (EIO-11) over a wiremock-backed homeserver: a
//! stripped `m.room.member` invite for the bot arriving on sync drives `Room::join` (the adapter's
//! `on_stripped_member` handler), the joined room then enumerates through the adapter's
//! `SupportsConversations::list` (the wire `ConvList` surfacing), and the `auto_accept_invites:
//! false` policy leaves the invite pending (join endpoint never hit), and the `invite_allowlist`
//! narrows acceptance to listed senders (an unlisted sender stays pending).

use std::sync::Arc;
use std::time::{Duration, Instant};

use matrix_sdk::ruma::events::room::member::MembershipState;
use matrix_sdk::ruma::{room_id, user_id};
use matrix_sdk::test_utils::mocks::MatrixMockServer;
use matrix_sdk::{Client, RoomState};
use matrix_sdk_test::event_factory::EventFactory;
use matrix_sdk_test::InvitedRoomBuilder;

use daemon_api::SupportsConversations;
use daemon_matrix::{InviteCtx, MatrixAdapter, MatrixConfig};
use daemon_protocol::TransportId;

/// Register the invite handler on `client` for `transport`, with the given accept policy and
/// (possibly empty) sender allowlist.
fn register_invite_handler(
    client: &Client,
    transport: &TransportId,
    auto_accept: bool,
    invite_allowlist: Vec<String>,
) {
    let me = client
        .user_id()
        .expect("mock client is logged in")
        .to_owned();
    client.add_event_handler_context(InviteCtx {
        me,
        transport: transport.clone(),
        auto_accept,
        invite_allowlist,
    });
    client.add_event_handler(daemon_matrix::on_stripped_member);
}

/// A minimal `AccountProvisioning` stand-in: the adapter under test resolves its per-account
/// client from the live-clients registry, so provisioning is never consulted.
struct NoProvisioning;
impl daemon_host::AccountProvisioning for NoProvisioning {
    fn bound_accounts(&self, _family: &str) -> Vec<daemon_host::ProvisionedAccount> {
        Vec::new()
    }
    fn account_credential(&self, _credential_ref: &str) -> Option<String> {
        None
    }
    fn store_account_credential(
        &self,
        _credential_ref: &str,
        _blob: &str,
    ) -> Result<(), daemon_api::ApiError> {
        Ok(())
    }
}

/// An adapter whose live-clients registry maps `transport` to `client` (the shape `serve`
/// publishes at bring-up), so `SupportsConversations::list` resolves the mock client.
async fn adapter_over(transport: &TransportId, client: Client) -> Arc<MatrixAdapter> {
    let adapter = MatrixAdapter::new(Arc::new(NoProvisioning), MatrixConfig::default());
    adapter
        .register_live_client(transport.clone(), client)
        .await;
    adapter
}

/// Sync a stripped `m.room.member` invite for the logged-in bot into `client`'s state.
async fn sync_invite(server: &MatrixMockServer, client: &Client, room: &matrix_sdk::ruma::RoomId) {
    let me = client.user_id().expect("logged in").to_owned();
    let factory = EventFactory::new();
    server
        .sync_room(
            client,
            InvitedRoomBuilder::new(room).add_state_event(
                factory
                    .member(&me)
                    .membership(MembershipState::Invite)
                    .sender(user_id!("@alice:localhost")),
            ),
        )
        .await;
}

/// An invite addressed to the bot is auto-accepted: the join endpoint is hit, the room flips to
/// `Joined` locally, and — after the post-join sync — the room enumerates through the adapter's
/// `SupportsConversations::list` (what the wire `ConvList` serves).
#[tokio::test]
async fn invite_is_auto_accepted_and_room_reaches_conv_list() {
    let server = MatrixMockServer::new().await;
    let client = server.client_builder().build().await;
    let room = room_id!("!invited:localhost");
    server
        .mock_room_join(room)
        .ok()
        .expect(1)
        .named("join")
        .mount()
        .await;

    let transport = TransportId::new("matrix/@bot:localhost");
    register_invite_handler(&client, &transport, true, Vec::new());

    sync_invite(&server, &client, room).await;

    // The handler spawns the join off the sync task; wait for the local state flip.
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if client.get_room(room).map(|r| r.state()) == Some(RoomState::Joined) {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "the invited room never became Joined (state: {:?})",
            client.get_room(room).map(|r| r.state())
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    // The joined room surfaces through the adapter's conversations listing (the `ConvList` body).
    let adapter = adapter_over(&transport, client).await;
    let convs = SupportsConversations::list(&*adapter, transport).await;
    assert!(
        convs.iter().any(|c| c.id == room.as_str()),
        "the accepted room must enumerate via ConvList, got {convs:?}"
    );
}

/// With `auto_accept_invites: false` the invite stays pending: the join endpoint is never hit and
/// the room remains `Invited`.
#[tokio::test]
async fn invite_is_left_pending_when_auto_accept_is_off() {
    let server = MatrixMockServer::new().await;
    let client = server.client_builder().build().await;
    let room = room_id!("!pending:localhost");
    // Mounted so an unexpected join would be observable (and the expectation verifies zero calls).
    server
        .mock_room_join(room)
        .ok()
        .expect(0)
        .named("join-off")
        .mount()
        .await;

    let transport = TransportId::new("matrix/@bot:localhost");
    register_invite_handler(&client, &transport, false, Vec::new());

    sync_invite(&server, &client, room).await;

    // Give a (wrong) spawned join ample time to fire before verifying nothing did.
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(
        client.get_room(room).map(|r| r.state()),
        Some(RoomState::Invited),
        "the invite must stay pending when auto-accept is off"
    );
    server.server().verify().await;
}

/// A non-empty `invite_allowlist` that INCLUDES the inviter (`@alice:localhost`, per `sync_invite`)
/// still auto-accepts: the join endpoint is hit and the room flips to `Joined`.
#[tokio::test]
async fn invite_from_allowlisted_sender_is_accepted() {
    let server = MatrixMockServer::new().await;
    let client = server.client_builder().build().await;
    let room = room_id!("!allowed:localhost");
    server
        .mock_room_join(room)
        .ok()
        .expect(1)
        .named("join-allowed")
        .mount()
        .await;

    let transport = TransportId::new("matrix/@bot:localhost");
    register_invite_handler(
        &client,
        &transport,
        true,
        vec!["@alice:localhost".to_string()],
    );

    sync_invite(&server, &client, room).await;

    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if client.get_room(room).map(|r| r.state()) == Some(RoomState::Joined) {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "an allowlisted invite never became Joined (state: {:?})",
            client.get_room(room).map(|r| r.state())
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    server.server().verify().await;
}

/// A non-empty `invite_allowlist` that EXCLUDES the inviter leaves the invite pending even with
/// `auto_accept_invites: true`: the join endpoint is never hit and the room stays `Invited`.
#[tokio::test]
async fn invite_from_non_allowlisted_sender_is_left_pending() {
    let server = MatrixMockServer::new().await;
    let client = server.client_builder().build().await;
    let room = room_id!("!blocked:localhost");
    server
        .mock_room_join(room)
        .ok()
        .expect(0)
        .named("join-blocked")
        .mount()
        .await;

    let transport = TransportId::new("matrix/@bot:localhost");
    register_invite_handler(
        &client,
        &transport,
        true,
        vec!["@trusted:localhost".to_string()],
    );

    sync_invite(&server, &client, room).await;

    // Give a (wrong) spawned join ample time to fire before verifying nothing did.
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(
        client.get_room(room).map(|r| r.state()),
        Some(RoomState::Invited),
        "an invite from a non-allowlisted sender must stay pending"
    );
    server.server().verify().await;
}
