// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
// [waveA:node-v30]

//! Membership push (wire v30, item 3): translate Matrix `m.room.member` state transitions into the
//! node's two-tier membership events via the [`LifecycleSink`]. The adapter only classifies the
//! transition (Join/Leave/Invite/Ban, and leave-by-other = kick); the node owns the consequences
//! (routing reconciliation on a self removal, event emission). A self join/leave also emits the
//! coarse `ConversationsChanged` so a client stops re-polling `ConvList`.

use matrix_sdk::event_handler::Ctx;
use matrix_sdk::ruma::events::room::member::{MembershipState, OriginalSyncRoomMemberEvent};
use matrix_sdk::ruma::OwnedUserId;
use matrix_sdk::Room;
use std::sync::Arc;

use daemon_api::{ConvChange, LifecycleSink, MembershipChange};
use daemon_protocol::TransportId;

/// The shared, cloneable context threaded into the per-account `m.room.member` handler.
#[derive(Clone)]
pub struct MembershipCtx {
    /// The node-owned lifecycle sink the adapter reports transitions through.
    pub sink: Arc<dyn LifecycleSink>,
    /// This account's instance-qualified transport id.
    pub transport: TransportId,
    /// This account's own user id (drives the `is_self` flag).
    pub me: OwnedUserId,
}

/// The registered `m.room.member` handler: classify the transition and report it to the node.
pub async fn on_room_member(ev: OriginalSyncRoomMemberEvent, room: Room, ctx: Ctx<MembershipCtx>) {
    let ctx = ctx.0;
    let member = ev.state_key.to_string();
    let actor = ev.sender.to_string();
    let is_self = ev.state_key == ctx.me;
    let reason = ev.content.reason.clone();
    let conv = room.room_id().as_str().to_string();

    let change = match ev.content.membership {
        MembershipState::Join => MembershipChange::Joined,
        MembershipState::Invite => MembershipChange::Invited,
        MembershipState::Ban => MembershipChange::Banned,
        // A leave whose sender is the member is a voluntary leave; set by another, it is a kick.
        MembershipState::Leave => {
            if ev.sender == ev.state_key {
                MembershipChange::Left
            } else {
                MembershipChange::Kicked
            }
        }
        // Knock (and any future state) is not modeled by the two-tier push.
        _ => return,
    };

    // Coarse tier: this account's own conversation set changed on a self join/departure.
    if is_self {
        let conv_change = match change {
            MembershipChange::Joined => Some(ConvChange::Added),
            MembershipChange::Left | MembershipChange::Kicked | MembershipChange::Banned => {
                Some(ConvChange::Removed)
            }
            MembershipChange::Invited => None,
        };
        if let Some(cc) = conv_change {
            ctx.sink
                .conversations_changed(ctx.transport.clone(), conv.clone(), cc)
                .await;
        }
    }

    // Granular tier: the node reconciles routing on a self removal BEFORE emitting the event.
    ctx.sink
        .membership_changed(
            ctx.transport.clone(),
            conv,
            member,
            change,
            Some(actor),
            reason,
            is_self,
        )
        .await;
}
