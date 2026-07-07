// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! THE PHASE-3 GATE: §17 ⇄ management protocol round-trips (`daemon-workspace-layout.md` §7
//! phase-3 gate). The host presents a real `daemon-core` engine as a `UnitKind::Engine`
//! [`ManagedUnit`]; driving it with `ManageCommand`s and observing `ManageEvent`s / a
//! `ManageRequest` exercises the supervision §4 mapping table end to end (host-spec §9,
//! supervision invariant #7).

use async_trait::async_trait;
use daemon_common::{Budget, ReqId, SessionId, UnitId};
use daemon_core::{Engine, MockProvider, Provider, SystemPrompt, Tool, ToolRegistry};
use daemon_host::{EngineUnit, OrchestrateShim};
use daemon_supervision::{
    Ack, Concurrency, EndReason, ManageCommand, ManageEvent, ManageRequest, ManageRequestHandler,
    ManageRequestKind, ManageResponse, ManageResponseBody, ManagedUnit, ProgressDelta,
    StartTrigger, UnitKind, WorkRef,
};
use std::sync::Arc;
use std::time::Duration;

/// Build a managed unit over a real engine driven by `provider`.
fn engine_unit(provider: Arc<dyn Provider>) -> daemon_host::AgentUnit {
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(OrchestrateShim::new("background-work")));
    let engine = Engine::fresh(
        SessionId::new("u1"),
        SystemPrompt::new("translation gate engine"),
        provider,
        Arc::new(registry),
    );
    EngineUnit::spawn(UnitId::new("u1"), engine)
}

/// The legacy `daemon-core` `delegate` tool is retired: the conformance delegation surface is the
/// `orchestrate` shim (named for the node tool it stands in for), so the mock provider that drives
/// the delegation cycle below must call `orchestrate`, never `delegate`.
#[test]
fn delegate_tool_is_retired_in_favor_of_orchestrate() {
    assert_eq!(
        OrchestrateShim::new("background-work").name(),
        "orchestrate"
    );
}

/// `Assign` drives a turn whose §17 events surface as `Started → Progress → Finished` upward.
#[tokio::test]
async fn assign_round_trips_to_finished() {
    let unit = engine_unit(Arc::new(MockProvider::completing("all done")));
    assert_eq!(unit.kind(), UnitKind::Engine);
    assert_eq!(unit.id(), UnitId::new("u1"));

    let mut events = unit.events();
    let ack = unit
        .command(ManageCommand::Assign {
            request_id: ReqId(1),
            work: WorkRef::inline("w1", "do the thing"),
            budget: Budget::unlimited(),
        })
        .await;
    assert_eq!(ack, Ack::Accepted);

    let mut saw_started = false;
    let mut saw_progress = false;
    let outcome = loop {
        match tokio::time::timeout(Duration::from_secs(2), events.recv()).await {
            Ok(Ok(ManageEvent::Started {
                trigger: StartTrigger::Assigned(_),
                ..
            })) => saw_started = true,
            Ok(Ok(ManageEvent::Progress {
                delta: ProgressDelta::Text(_),
                ..
            })) => saw_progress = true,
            Ok(Ok(ManageEvent::Finished { outcome, .. })) => break outcome,
            Ok(Ok(_)) => {}
            Ok(Err(_)) => panic!("event stream closed before Finished"),
            Err(_) => panic!("timed out waiting for ManageEvent::Finished"),
        }
    };

    assert!(
        saw_started,
        "no Started{{Assigned}} mapped from TurnStarted"
    );
    assert!(saw_progress, "no Progress{{Text}} mapped from TextDelta");
    assert_eq!(outcome.end_reason, EndReason::Completed);
}

/// `Pause`/`Resume`/`Scale` are no-ops at an engine leaf — the partial-downward `Ack::Unsupported`.
#[tokio::test]
async fn pause_resume_scale_are_unsupported() {
    let unit = engine_unit(Arc::new(MockProvider::completing("done")));
    assert_eq!(unit.command(ManageCommand::Pause).await, Ack::Unsupported);
    assert_eq!(unit.command(ManageCommand::Resume).await, Ack::Unsupported);
    assert_eq!(
        unit.command(ManageCommand::Scale {
            target: Concurrency(4)
        })
        .await,
        Ack::Unsupported
    );
}

/// A blocking §17 `HostRequest` raised inside a turn surfaces upward as a correlated
/// `ManageRequest` through the installed handler (supervision §2.3 / §4).
#[tokio::test]
async fn host_request_maps_to_manage_request() {
    struct Recorder {
        tx: tokio::sync::mpsc::UnboundedSender<ManageRequest>,
    }

    #[async_trait]
    impl ManageRequestHandler for Recorder {
        async fn request(&self, req: ManageRequest) -> ManageResponse {
            let request_id = req.request_id;
            let _ = self.tx.send(req);
            ManageResponse {
                request_id,
                body: ManageResponseBody::Delegated(vec![UnitId::new("child-1")]),
            }
        }
    }

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let unit = engine_unit(Arc::new(MockProvider::delegating("orchestrate", "done")));
    unit.install_request_handler(Arc::new(Recorder { tx }));

    let ack = unit
        .command(ManageCommand::Assign {
            request_id: ReqId(7),
            work: WorkRef::inline("w1", "needs background work"),
            budget: Budget::unlimited(),
        })
        .await;
    assert_eq!(ack, Ack::Accepted);

    let got = tokio::time::timeout(Duration::from_secs(2), rx.recv())
        .await
        .expect("timed out waiting for the escalated ManageRequest")
        .expect("a ManageRequest");
    assert!(
        matches!(got.kind, ManageRequestKind::Delegate(_)),
        "the §17 HostRequest::Delegate did not map to ManageRequestKind::Delegate"
    );
}
