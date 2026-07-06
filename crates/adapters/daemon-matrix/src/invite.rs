// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Invite acceptance: auto-join rooms the bot account is invited to (EIO-11).
//!
//! Room invites arrive on the sync stream as **stripped** `m.room.member` state events (the
//! invited client only sees the room's stripped state until it joins). Without a handler the bot
//! stays invited forever — a user inviting the agent from their own Matrix client never gets it
//! into the room. This module registers the canonical matrix-sdk auto-join shape: on a stripped
//! member event *about this account* in a room this client is *invited* to, join the room (with a
//! short bounded retry, because a join issued immediately after the invite can race the
//! homeserver's state propagation).
//!
//! Once joined, the room shows up in `Client::rooms()` and therefore in the adapter's
//! `SupportsConversations::list` — i.e. the wire `ConvList` — with no further wiring; subsequent
//! `m.room.message`s flow through the normal inbound gate ([`crate::on_room_message`]).
//!
//! Policy: gated by [`MatrixConfig::auto_accept_invites`](crate::MatrixConfig) (default **on**;
//! see the field docs for the security tradeoff). A finer sender allowlist / owner-only policy is
//! a recorded follow-up.

use std::time::Duration;

use matrix_sdk::event_handler::Ctx;
use matrix_sdk::ruma::events::room::member::{MembershipState, StrippedRoomMemberEvent};
use matrix_sdk::ruma::OwnedUserId;
use matrix_sdk::{Room, RoomState};

use daemon_protocol::TransportId;

/// The shared, cloneable context threaded into the per-account stripped-member handler.
#[derive(Clone)]
pub struct InviteCtx {
    /// This account's own user id — only invites addressed to it are acted on.
    pub me: OwnedUserId,
    /// This account's instance-qualified transport id (log attribution).
    pub transport: TransportId,
    /// Whether to accept invites at all ([`crate::MatrixConfig::auto_accept_invites`]).
    pub auto_accept: bool,
}

/// The registered stripped `m.room.member` handler: accept an invite addressed to this account by
/// joining the room. Joining is spawned off the sync task (the retry sleeps must not stall event
/// dispatch for the rest of the sync batch).
pub async fn on_stripped_member(ev: StrippedRoomMemberEvent, room: Room, ctx: Ctx<InviteCtx>) {
    let ctx = ctx.0;
    // Only membership events about US — an invite for another user in a shared room is not ours.
    if ev.state_key != ctx.me {
        return;
    }
    // Only actual invites to a room we are currently invited to (stripped state also carries e.g.
    // the inviter's own join membership).
    if ev.content.membership != MembershipState::Invite || room.state() != RoomState::Invited {
        return;
    }
    if !ctx.auto_accept {
        tracing::info!(
            instance = %ctx.transport.as_str(),
            room = %room.room_id(),
            sender = %ev.sender,
            "matrix: invite received but auto_accept_invites is off; leaving it pending"
        );
        return;
    }

    tokio::spawn(async move {
        // A join issued immediately on the invite can race the homeserver (transient 404/403);
        // the canonical auto-join shape retries briefly, then gives up loudly.
        let mut delay = Duration::from_millis(200);
        for attempt in 1..=4u32 {
            match room.join().await {
                Ok(()) => {
                    tracing::info!(
                        instance = %ctx.transport.as_str(),
                        room = %room.room_id(),
                        sender = %ev.sender,
                        "matrix: accepted room invite"
                    );
                    return;
                }
                Err(e) if attempt < 4 => {
                    tracing::debug!(
                        error = %e,
                        room = %room.room_id(),
                        attempt,
                        "matrix: invite join failed; retrying"
                    );
                    tokio::time::sleep(delay).await;
                    delay *= 2;
                }
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        instance = %ctx.transport.as_str(),
                        room = %room.room_id(),
                        "matrix: giving up accepting room invite"
                    );
                }
            }
        }
    });
}
