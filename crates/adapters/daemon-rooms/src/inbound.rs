// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Inbound: a Room post -> a per-member `submit_from` fan-out.
//!
//! A Room is the loopback analogue of a Matrix room. Where `daemon-matrix` receives one event and
//! gates it for the single bot account, the RoomRouter fans **one** post out to **every** member: the
//! room id maps to `OriginScope::Group { chat: room_id }` under the room's loopback `TransportId`, and
//! each member is its own pre-resolved session (the membership table binds `(room, member) ->
//! SessionId`). The floor-control policy upstream decides who is *addressed* (opens a `StartTurn`) vs.
//! who merely *observes* (`Observe`); this module owns the room-specific fan-out that submits the
//! chosen command to each member session via `submit_from` (so per-event attribution is recorded).
//!
//! The per-member *busy* gating (queue-while-busy / ambient-fold) that `daemon-ingest` owns for a
//! single account is driven on the outbound side here — the loopback `Projector` notes each member
//! session's `TurnStarted`/`TurnFinished` against the shared [`Ingestor`](daemon_ingest::Ingestor) —
//! so a follow-up can route the inbound through the gate per member without a second subscription.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use daemon_api::NodeApi;
use daemon_common::{ReqId, SessionId};
use daemon_protocol::{AgentCommand, Origin, OriginScope, RoomId, RoomMember, UserMsg};

/// The in-memory membership table for the rooms this router owns: `room_id -> members`. Loaded from
/// the store (`room_members`) at bring-up and mutated by the `room_add_member`/`room_remove_member`
/// ops. The RoomRouter reads it to know whom to fan a post out to.
#[derive(Default)]
pub struct Membership {
    by_room: HashMap<RoomId, Vec<RoomMember>>,
}

impl Membership {
    /// An empty membership table.
    pub fn new() -> Self {
        Self::default()
    }

    /// The members of `room` (empty if the room is unknown).
    pub fn members(&self, room: &RoomId) -> &[RoomMember] {
        self.by_room.get(room).map(Vec::as_slice).unwrap_or(&[])
    }

    /// Add (or replace) a member of `room`, keyed by its handle.
    pub fn upsert(&mut self, room: RoomId, member: RoomMember) {
        let members = self.by_room.entry(room).or_default();
        if let Some(slot) = members.iter_mut().find(|m| m.member == member.member) {
            *slot = member;
        } else {
            members.push(member);
        }
    }

    /// Remove a member from `room` by handle (idempotent).
    pub fn remove(&mut self, room: &RoomId, member: &str) {
        if let Some(members) = self.by_room.get_mut(room) {
            members.retain(|m| m.member != member);
        }
    }

    /// Drop a room's entire membership (on room delete; idempotent).
    pub fn remove_room(&mut self, room: &RoomId) {
        self.by_room.remove(room);
    }

    /// Reverse lookup: the `(room, member handle)` bound to `session`, if any. The outbound loop uses
    /// this to resolve which Room a finished member turn belongs to before re-injecting it.
    pub fn find_by_session(&self, session: &SessionId) -> Option<(RoomId, String)> {
        for (room, members) in &self.by_room {
            if let Some(m) = members.iter().find(|m| &m.session == session) {
                return Some((room.clone(), m.member.clone()));
            }
        }
        None
    }

    /// Every member session across every room (the set the outbound loop subscribes for re-injection).
    pub fn all_member_sessions(&self) -> Vec<SessionId> {
        self.by_room
            .values()
            .flat_map(|ms| ms.iter().map(|m| m.session.clone()))
            .collect()
    }
}

/// The room-specific inbound fan-out half. Submits one §17 command per member to that member's
/// session via `submit_from`, attributed to the room's loopback [`Origin`].
pub struct RoomInbound {
    api: Arc<dyn NodeApi>,
    next_req: AtomicU64,
}

impl RoomInbound {
    /// Construct the fan-out over the node `api`.
    pub fn new(api: Arc<dyn NodeApi>) -> Self {
        Self {
            api,
            next_req: AtomicU64::new(1),
        }
    }

    fn req(&self) -> ReqId {
        ReqId(self.next_req.fetch_add(1, Ordering::Relaxed))
    }

    /// The loopback [`Origin`] of a post into `room` (attribution rides in the text body).
    fn origin(room: &RoomId) -> Origin {
        Origin::new(
            room.transport(),
            OriginScope::Group {
                chat: room.as_str().to_string(),
                thread: None,
            },
        )
    }

    /// Fan a post (`sender`: `text`) out to `members`. `addressed(member)` is the per-member floor
    /// decision the [`FloorControl`](crate::policy::FloorControl) made (true opens a `StartTurn`;
    /// false is ambient `Observe`); the sender never receives its own post.
    pub async fn fan_out<F>(
        &self,
        room: &RoomId,
        sender: &str,
        text: &str,
        members: &[RoomMember],
        addressed: F,
    ) where
        F: Fn(&RoomMember) -> bool,
    {
        let origin = Self::origin(room);
        let attributed = format!("{sender}: {text}");
        for member in members {
            if member.member == sender {
                continue;
            }
            let input = UserMsg::new(attributed.clone());
            let command = if addressed(member) {
                AgentCommand::StartTurn {
                    input,
                    request_id: self.req(),
                }
            } else {
                AgentCommand::Observe {
                    input,
                    request_id: self.req(),
                }
            };
            if let Err(e) = self
                .api
                .submit_from(member.session.clone(), origin.clone(), command)
                .await
            {
                tracing::warn!(error = %e, member = %member.member, "rooms: submit_from failed");
            }
        }
    }
}
