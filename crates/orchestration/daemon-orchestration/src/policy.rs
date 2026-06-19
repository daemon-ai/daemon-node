//! The child-request answer/escalation policy (layout §4; synthesis §3.1 #3).
//!
//! A parent is the **answer-authority** for its children: a child's blocking `ManageRequest`
//! (`Approval`/`Input`/`Choice`/`Resource`/`Escalate`) flows up through the runtime's request
//! handler, which either answers it by policy here, or escalates it to *its* supervisor. (Growing
//! the tree — `Delegate` — is handled by the runtime itself, since answering it spawns children.)
//! The policy is the deterministic half; surfacing to a live agent as a §17 `HostRequest` is the
//! agent-driven half and rides the same handler in `bins/daemon`.

use daemon_supervision::{ManageRequest, ManageRequestKind, ManageResponseBody};

/// What the runtime should do with a child's request.
pub enum Decision {
    /// Answer it locally with this body.
    Answer(ManageResponseBody),
    /// Cannot answer locally — re-raise to the runtime's own supervisor.
    Escalate,
}

/// The policy a [`crate::FleetRuntime`] consults for each child request it cannot grow the tree for.
pub trait AnswerPolicy: Send + Sync {
    /// Decide how to handle `req`.
    fn decide(&self, req: &ManageRequest) -> Decision;
}

/// The default deterministic policy: approve approvals, answer input/choice with a safe default,
/// and escalate everything a parent cannot resolve on its own (`Escalate`/`Resource`).
pub struct DefaultAnswerPolicy;

impl AnswerPolicy for DefaultAnswerPolicy {
    fn decide(&self, req: &ManageRequest) -> Decision {
        match &req.kind {
            ManageRequestKind::Approval(_) => Decision::Answer(ManageResponseBody::Approved(true)),
            ManageRequestKind::Input(_) => Decision::Answer(ManageResponseBody::Input(String::new())),
            ManageRequestKind::Choice(_) => Decision::Answer(ManageResponseBody::Chosen(0)),
            // Delegate is grown by the runtime before the policy is consulted; if it reaches here
            // (e.g. over the child budget) it escalates like any unresolvable request.
            ManageRequestKind::Delegate(_)
            | ManageRequestKind::Escalate(_)
            | ManageRequestKind::Resource(_) => Decision::Escalate,
            _ => Decision::Escalate,
        }
    }
}
