// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The control-surface projection of the management tree, sourced directly from the durable session
//! graph (recovery-survivable), with the in-memory [`FleetRuntime`] retained only for cancel routing.

use std::sync::Arc;

use async_trait::async_trait;
use daemon_api::{FleetReport, ManageEventView};
use daemon_common::{SessionId, UnitId};
use daemon_host::FleetControl;
use daemon_orchestration::FleetRuntime;

/// Projects the management tree for the node control surface directly from the **durable session
/// graph** (the GUI/TUI's real surface). Structure (parent->children), state, per-node work label,
/// and folded usage are all re-sourced from the store — so the tree is recovery-survivable and
/// shows every durable session (top, child, grandchild, ...) at its true depth, addressable by id.
/// The legacy in-memory `FleetRuntime` projection is retained only for the synchronous foreign path;
/// `cancel` still routes through it.
pub struct FleetViewImpl {
    store: Arc<dyn daemon_store::SessionStore>,
    fleet: FleetRuntime,
    /// The profile store backing per-node engine enrichment (wire v29 `UnitNode.engine`): a bound
    /// profile's `EngineSelector` is denormalized onto its tree node. `None` (a node without
    /// profile management) leaves `engine` unset.
    profiles: Option<Arc<dyn daemon_host::ProfileStore>>,
}

impl FleetViewImpl {
    /// A control-surface projection over the durable `store`, with `fleet` for cancel routing.
    pub fn new(store: Arc<dyn daemon_store::SessionStore>, fleet: FleetRuntime) -> Self {
        Self {
            store,
            fleet,
            profiles: None,
        }
    }

    /// Attach the profile store so each bound unit's tree node carries its profile's
    /// [`EngineSelector`](daemon_api::EngineSelector) (wire v29 enrichment).
    pub fn with_profiles(mut self, profiles: Arc<dyn daemon_host::ProfileStore>) -> Self {
        self.profiles = Some(profiles);
        self
    }

    /// Build the tree node for one durable session from its status + durable child edge.
    async fn node_for(
        &self,
        session: &SessionId,
        status: &daemon_store::SessionStatus,
        children: &[SessionId],
    ) -> daemon_api::UnitNode {
        use daemon_store::SessionStatus;
        // A node is an orchestrator iff it actually delegated (has durable children), else a leaf.
        let kind = if children.is_empty() {
            daemon_api::UnitKind::Engine
        } else {
            daemon_api::UnitKind::Orchestrator
        };
        let state = match status {
            SessionStatus::Completed => daemon_api::UnitState::Finished {
                end_reason: "Completed".to_string(),
            },
            _ => daemon_api::UnitState::Running,
        };
        // Enrich the node with the session's durable identity (profile/title/role) so a GUI tree
        // drill-down carries the same identity as the roster line, sourced from the same host meta.
        let meta = self.store.session_meta(session).await.unwrap_or_default();
        let role = match meta.role {
            Some(daemon_store::SessionRole::Primary) | None => daemon_api::SessionRole::Primary,
            Some(daemon_store::SessionRole::ManagedChild) => daemon_api::SessionRole::ManagedChild,
            Some(daemon_store::SessionRole::EphemeralSubagent) => {
                daemon_api::SessionRole::EphemeralSubagent
            }
        };
        // The declared delegation lifetime (wire v29), derived from the durable role — the SERVER
        // owns the role->lifetime rule (`ChildLifetime::role` is the forward direction stamped at
        // materialize time), so clients never re-implement the inversion. Primary units carry none.
        let lifetime = match role {
            daemon_api::SessionRole::ManagedChild => {
                Some(daemon_protocol::DelegationLifetime::Persistent)
            }
            daemon_api::SessionRole::EphemeralSubagent => {
                Some(daemon_protocol::DelegationLifetime::Ephemeral)
            }
            daemon_api::SessionRole::Primary => None,
        };
        // Denormalize the bound profile's engine selector (wire v29) so a tree render needs no
        // per-node ProfileGet. Absent profile store / unbound unit / vanished profile => None.
        let engine = match (&self.profiles, &meta.bound_profile) {
            (Some(profiles), Some(profile)) => profiles
                .get(profile.as_str())
                .ok()
                .flatten()
                .map(|spec| spec.engine),
            _ => None,
        };
        daemon_api::UnitNode {
            id: UnitId::new(session.as_str()),
            kind,
            state,
            work: self.store.delegation_work(session).await,
            usage: self.store.usage_of(session).await,
            children: children.iter().map(|c| UnitId::new(c.as_str())).collect(),
            profile: meta.bound_profile,
            session: Some(session.clone()),
            title: meta.title,
            role: Some(role),
            lifetime,
            engine,
        }
    }
}

#[async_trait]
impl FleetControl for FleetViewImpl {
    async fn report(&self) -> FleetReport {
        let mut usage = daemon_common::UsageDelta::default();
        let mut children = Vec::new();
        for (session, _) in self.store.list_sessions().await {
            usage.add(&self.store.usage_of(&session).await);
            children.push(UnitId::new(session.as_str()));
        }
        FleetReport { children, usage }
    }

    async fn cancel(&self, child: &UnitId) -> bool {
        self.fleet.cancel_child(child).await
    }

    async fn tree(&self) -> daemon_api::TreeReport {
        let sessions = self.store.list_sessions().await;
        let mut nodes = Vec::with_capacity(sessions.len());
        let mut is_child = std::collections::HashSet::new();
        for (session, status) in &sessions {
            let children = self.store.children_of(session).await;
            for c in &children {
                is_child.insert(c.clone());
            }
            nodes.push(self.node_for(session, status, &children).await);
        }
        // The root is the single top (parentless) session, if there is exactly one; otherwise the
        // node holds a forest and `root` is left unset (the nodes still carry the full structure).
        let roots: Vec<&SessionId> = sessions
            .iter()
            .map(|(s, _)| s)
            .filter(|s| !is_child.contains(*s))
            .collect();
        let root = match roots.as_slice() {
            [only] => Some(UnitId::new(only.as_str())),
            _ => None,
        };
        daemon_api::TreeReport {
            root,
            nodes,
            // The full (unpaged) projection: the wire `ControlApi::tree` handler slices it into
            // cursor pages; in-process consumers (the fleet bus) take it whole.
            next: None,
            // rung 1: the wire handler stamps the authoritative fleet-rev echo; this lower-level
            // projection leaves it 0.
            rev: 0,
        }
    }

    async fn unit(&self, id: &UnitId) -> Option<daemon_api::UnitNode> {
        let session = SessionId::new(id.as_str());
        let status = self.store.status(&session).await?;
        let children = self.store.children_of(&session).await;
        Some(self.node_for(&session, &status, &children).await)
    }

    async fn unit_events(&self, id: &UnitId, max: u32) -> Vec<daemon_api::ManageEventView> {
        use daemon_store::SessionStatus;
        let session = SessionId::new(id.as_str());
        // Coarse lifecycle views synthesized from the durable status (the rich, byte-faithful
        // transcript is the verifiable journal, read via `unit_history`). A durable session has at
        // least Started; a terminal one also has Finished.
        let Some(status) = self.store.status(&session).await else {
            return Vec::new();
        };
        let mut views = vec![ManageEventView::Started { seq: 0 }];
        if matches!(status, SessionStatus::Completed) {
            views.push(ManageEventView::Finished {
                seq: 1,
                end_reason: "Completed".to_string(),
                summary: None,
            });
        }
        if max != 0 && (max as usize) < views.len() {
            let skip = views.len() - max as usize;
            views.drain(0..skip);
        }
        views
    }

    async fn unit_outbound(&self, _id: &UnitId, _max: u32) -> Vec<daemon_api::Outbound> {
        // Durable sessions retain no live §17 stream; their transcript is the durable journal.
        Vec::new()
    }

    async fn pause(&self, _id: &UnitId) -> bool {
        // Vestigial on the durable path: a durable session has no live scheduling to pause.
        false
    }

    async fn resume(&self, _id: &UnitId) -> bool {
        false
    }

    async fn scale(&self, _id: &UnitId, _n: u32) -> bool {
        false
    }
}
