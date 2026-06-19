//! The child registry + fleet-state fold (layout §4: `UnitId -> status/work/lease`, Usage fan-in).
//!
//! The runtime materializes the live fleet view from the children it has spawned: each
//! [`ChildRecord`] tracks the unit handle, its assigned work, and its lifecycle status, while the
//! incremental [`daemon_common::UsageDelta`] events fold into one fleet total (supervision
//! invariant #4 — usage aggregates up the tree by construction).

use daemon_supervision::{ManagedUnit, Outcome, WorkRef};
use std::sync::Arc;

/// Where a child sits in its lifecycle, as folded from its `ManageEvent` stream.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ChildStatus {
    /// Spawned and registered, no `Started` seen yet.
    Spawned,
    /// `Started` observed; work in flight.
    Running,
    /// `Finished` observed; carries the terminal [`Outcome`].
    Finished,
    /// `Error` observed or the event stream closed before `Finished`.
    Failed,
}

/// One child's entry in the fleet registry.
#[derive(Clone)]
pub struct ChildRecord {
    /// The upward unit handle (routing is by id, but the runtime drives it directly in-process).
    pub unit: Arc<dyn ManagedUnit>,
    /// The work the child was assigned.
    pub work: WorkRef,
    /// The child's current lifecycle status.
    pub status: ChildStatus,
    /// The terminal outcome, once `Finished`/`Failed`.
    pub outcome: Option<Outcome>,
}

impl ChildRecord {
    /// A freshly spawned, not-yet-started child.
    pub fn new(unit: Arc<dyn ManagedUnit>, work: WorkRef) -> Self {
        Self {
            unit,
            work,
            status: ChildStatus::Spawned,
            outcome: None,
        }
    }
}
