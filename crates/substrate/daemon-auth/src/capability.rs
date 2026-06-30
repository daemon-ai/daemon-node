// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The RBAC vocabulary: [`Capability`] (what an authenticated caller may do), [`Role`] (the
//! admin-assignable bundle of capabilities), and [`Principal`] (the resolved identity + its
//! effective capability set, derived once at authentication time and carried per request).
//!
//! Capabilities are coarse, aligned to the node's API operation categories (the `serve_*` groups in
//! `daemon-api`'s dispatch). Authorization is a two-step check: the per-request *capability* gate
//! here, plus a per-resource *ownership* check enforced by the session layer
//! ([`Capability::SessionSeeAll`] / [`Capability::SessionControlAny`] are the operator overrides).
//! Finer per-resource grants (sharing one session/profile with one user) are a deliberate future
//! extension — see the reserved `resource_grants` table in [`crate::store`].

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// A single coarse permission over one API operation category. `Read` variants gate the listing /
/// inspection ops; `Write` variants gate the mutating ops. Two ownership overrides
/// (`Session{SeeAll,ControlAny}`) let an operator transcend per-user session ownership.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Capability {
    /// Read one's own sessions / transcripts / roster.
    SessionRead,
    /// Submit to / respond to / cancel one's own sessions.
    SessionWrite,
    /// See sessions owned by *any* user (operator roster). Override of per-user ownership.
    SessionSeeAll,
    /// Control (respond/cancel/handover) sessions owned by *any* user. Override of ownership.
    SessionControlAny,
    /// Read node health / stats / telemetry / approvals / checkpoints.
    ControlRead,
    /// Mutate control surface (assign, cancel, approvals, rewind).
    ControlWrite,
    /// Read the fleet / orchestration tree.
    FleetRead,
    /// Drive the fleet (pause/resume/scale/unit control).
    FleetWrite,
    /// List models / catalog / current.
    ModelsRead,
    /// Download / select / configure models.
    ModelsWrite,
    /// Read profiles / skills.
    ProfileRead,
    /// Create / edit / delete profiles + skills.
    ProfileWrite,
    /// List credential refs (never the secret material).
    CredentialRead,
    /// Set / remove provider credentials and run interactive (OAuth) auth flows.
    CredentialWrite,
    /// Read cron schedules + runs.
    CronRead,
    /// Create / edit cron jobs.
    CronWrite,
    /// Read the routing registry.
    RoutingRead,
    /// Edit routing (set rules, bind/unbind chats).
    RoutingWrite,
    /// Read messaging adapters / conversations / contacts / directory.
    MessagingRead,
    /// Send / manage messaging conversations + members.
    MessagingWrite,
    /// Read the ACP / provider / tool / command / config registry.
    RegistryRead,
    /// Mutate the registry / node config.
    RegistryWrite,
    /// Read files / blobs exposed by the node.
    FsRead,
    /// Write files / blobs.
    FsWrite,
    /// Administer the access-control system itself: create users, assign roles, manage sessions.
    AccessAdmin,
}

/// An admin-assignable bundle of capabilities. Stored per user as text in `user_roles`; the
/// effective capability set of a [`Principal`] is the union over its roles ([`Role::capabilities`]).
/// The four built-ins form a strict ladder (Viewer < User < Operator < Admin) for the common case;
/// finer control is possible by composing roles (and, later, per-resource grants).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// Read-only access to one's own surfaces.
    Viewer,
    /// Full control of one's *own* sessions/profiles; no visibility into other users' work.
    User,
    /// Node operator: sees and controls all sessions/fleet, but cannot administer users.
    Operator,
    /// Full control including user/role administration.
    Admin,
}

impl Role {
    /// All built-in roles, lowest to highest privilege.
    pub const ALL: [Role; 4] = [Role::Viewer, Role::User, Role::Operator, Role::Admin];

    /// The stable text form persisted in the store and carried on the wire.
    pub fn as_str(self) -> &'static str {
        match self {
            Role::Viewer => "viewer",
            Role::User => "user",
            Role::Operator => "operator",
            Role::Admin => "admin",
        }
    }

    /// Parse the persisted/wire text form. Unknown strings are rejected (caller treats as no role).
    pub fn from_wire(s: &str) -> Option<Role> {
        match s {
            "viewer" => Some(Role::Viewer),
            "user" => Some(Role::User),
            "operator" => Some(Role::Operator),
            "admin" => Some(Role::Admin),
            _ => None,
        }
    }

    /// The capabilities this role grants. Higher roles are supersets of lower ones, so a multi-role
    /// principal simply unions these.
    pub fn capabilities(self) -> Vec<Capability> {
        use Capability::*;
        // Read-only over one's own surfaces.
        let viewer = [
            SessionRead,
            ControlRead,
            FleetRead,
            ModelsRead,
            ProfileRead,
            CredentialRead,
            CronRead,
            RoutingRead,
            MessagingRead,
            RegistryRead,
            FsRead,
        ];
        // Adds write over one's own resources.
        let user_extra = [
            SessionWrite,
            ProfileWrite,
            CredentialWrite,
            CronWrite,
            MessagingWrite,
            FsWrite,
        ];
        // Adds node-wide visibility/control (the ownership overrides + control/fleet/routing write).
        let operator_extra = [
            SessionSeeAll,
            SessionControlAny,
            ControlWrite,
            FleetWrite,
            ModelsWrite,
            RoutingWrite,
            RegistryWrite,
        ];
        match self {
            Role::Viewer => viewer.to_vec(),
            Role::User => [viewer.as_slice(), user_extra.as_slice()].concat(),
            Role::Operator => [
                viewer.as_slice(),
                user_extra.as_slice(),
                operator_extra.as_slice(),
            ]
            .concat(),
            Role::Admin => {
                let mut all = [
                    viewer.as_slice(),
                    user_extra.as_slice(),
                    operator_extra.as_slice(),
                ]
                .concat();
                all.push(AccessAdmin);
                all
            }
        }
    }
}

/// A resolved, authenticated identity plus its effective capability set. Built once at
/// authentication time ([`Principal::from_roles`]) and carried with every request (see
/// `RequestContext` in the host) so the capability gate and ownership checks never re-query the
/// store on the hot path.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Principal {
    /// Stable opaque user id.
    pub user_id: String,
    /// Human-facing username (for display + audit).
    pub username: String,
    /// The roles assigned to this user.
    pub roles: Vec<Role>,
    /// The precomputed union of capabilities across `roles`.
    pub capabilities: BTreeSet<Capability>,
}

impl Principal {
    /// Build a principal, computing the effective capability set as the union over `roles`.
    pub fn from_roles(
        user_id: impl Into<String>,
        username: impl Into<String>,
        roles: Vec<Role>,
    ) -> Principal {
        let capabilities = roles.iter().flat_map(|r| r.capabilities()).collect();
        Principal {
            user_id: user_id.into(),
            username: username.into(),
            roles,
            capabilities,
        }
    }

    /// Whether this principal holds `cap`.
    pub fn has(&self, cap: Capability) -> bool {
        self.capabilities.contains(&cap)
    }

    /// Whether this principal may see/operate sessions it does not own (operator override).
    pub fn can_see_all_sessions(&self) -> bool {
        self.has(Capability::SessionSeeAll)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn role_ladder_is_monotonic() {
        let viewer = Role::Viewer.capabilities().len();
        let user = Role::User.capabilities().len();
        let operator = Role::Operator.capabilities().len();
        let admin = Role::Admin.capabilities().len();
        assert!(viewer < user && user < operator && operator < admin);
        assert!(Role::Admin
            .capabilities()
            .contains(&Capability::AccessAdmin));
        assert!(!Role::Operator
            .capabilities()
            .contains(&Capability::AccessAdmin));
    }

    #[test]
    fn role_text_round_trips() {
        for r in Role::ALL {
            assert_eq!(Role::from_wire(r.as_str()), Some(r));
        }
        assert_eq!(Role::from_wire("nope"), None);
    }

    #[test]
    fn principal_unions_capabilities_and_overrides() {
        let p = Principal::from_roles("u1", "alice", vec![Role::User]);
        assert!(p.has(Capability::SessionWrite));
        assert!(!p.can_see_all_sessions());
        let op = Principal::from_roles("u2", "bob", vec![Role::Operator]);
        assert!(op.can_see_all_sessions());
        assert!(!op.has(Capability::AccessAdmin));
    }
}
