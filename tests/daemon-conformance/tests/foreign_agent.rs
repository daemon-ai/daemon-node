//! Phase-10 foreign-agent proof: a non-`daemon-core` process is a first-class fleet member.
//!
//! A `daemon-host` `ProcessAgentUnit` drives the `mock_stdio_agent` binary (which has no
//! `daemon-core` dependency) over a §17 process cut, and a real `FleetRuntime` delegates to it
//! exactly as it would to an in-process engine. This proves the §17 leaf is engine-agnostic: the
//! foreign brain presents as `UnitKind::Engine`, is driven to a terminal `Completed` outcome, and
//! its usage folds into the fleet total — all over a real OS process boundary, with no LLM.

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use daemon_api::Outbound;
use daemon_common::{Budget, PartitionId, ReqId, SessionId, UnitId};
use daemon_host::ProcessAgentUnit;
use daemon_orchestration::{ChildSpawner, DefaultAnswerPolicy, FleetRuntime};
use daemon_protocol::AgentEvent;
use daemon_provision::{PlacementSpec, ProcessProvisioner, Provisioner};
use daemon_store::{InMemoryStore, SessionStore};
use daemon_supervision::{
    DelegationSpec, EndReason, ManageRequest, ManageRequestKind, ManageResponseBody, ManagedUnit,
    UnitKind, WorkRef,
};

// The opaque payloads the `mock_stdio_agent` emits (mirrored here; the bin's consts live in a
// separate compilation unit). The daemon must carry these through byte-for-byte without parsing.
const EXPECT_TOOL_DETAIL_KIND: &str = "search.results";
const EXPECT_TOOL_DETAIL_BODY: &[u8] = b"[{\"title\":\"Rust\",\"url\":\"https://rust-lang.org\"}]";
const EXPECT_CONTENT_KIND: &str = "ansi-stream";
const EXPECT_CONTENT_BODY: &[u8] = b"\x1b[32m$ cargo build\x1b[0m\n   Compiling ok\n";

/// The mock foreign-agent binary path (Cargo sets this for integration tests).
fn mock_agent() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_mock_stdio_agent"))
}

/// A `ChildSpawner` that materializes each child as a foreign agent process behind a §17 cut.
struct ForeignSpawner {
    program: PathBuf,
    provisioner: Arc<dyn Provisioner>,
}

#[async_trait]
impl ChildSpawner for ForeignSpawner {
    async fn spawn(&self, id: UnitId, _spec: &DelegationSpec) -> Arc<dyn ManagedUnit> {
        let placement = self
            .provisioner
            .place(
                &SessionId::new(id.as_str()),
                PlacementSpec {
                    program: self.program.clone(),
                    args: Vec::new(),
                    env: Vec::new(),
                },
            )
            .await
            .expect("place foreign agent");
        Arc::new(ProcessAgentUnit::start(id, placement))
    }
}

/// A placed foreign agent presents as an `Engine` leaf — indistinguishable from a `daemon-core` one.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn foreign_agent_presents_as_engine_leaf() {
    let provisioner = ProcessProvisioner::new();
    let placement = provisioner
        .place(
            &SessionId::new("foreign-direct"),
            PlacementSpec {
                program: mock_agent(),
                args: Vec::new(),
                env: Vec::new(),
            },
        )
        .await
        .expect("place foreign agent");
    let unit = ProcessAgentUnit::start(UnitId::new("foreign-direct"), placement);
    assert_eq!(unit.kind(), UnitKind::Engine);
}

/// The fleet delegates to a foreign process and drives it to `Completed`, folding its usage.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fleet_drives_a_foreign_agent_to_completion() {
    let store: Arc<dyn SessionStore> = Arc::new(InMemoryStore::new());
    let spawner = Arc::new(ForeignSpawner {
        program: mock_agent(),
        provisioner: Arc::new(ProcessProvisioner::new()),
    });
    let fleet = FleetRuntime::new(
        store,
        PartitionId::DEFAULT,
        spawner,
        Arc::new(DefaultAnswerPolicy),
        None,
    );

    // Delegate one unit of work through the fleet's answer-authority (the same path an engine's
    // `Delegate` host-request takes): spawn + drive the foreign child to a terminal outcome.
    let response = fleet
        .request_handler()
        .request(ManageRequest {
            request_id: ReqId(1),
            kind: ManageRequestKind::Delegate(vec![DelegationSpec {
                work: WorkRef::inline("w-1", "do the foreign thing"),
                budget: Budget::unlimited(),
                toolset: Vec::new(),
            }]),
        })
        .await;

    let ids = match response.body {
        ManageResponseBody::Delegated(ids) => ids,
        other => panic!("expected Delegated, got {other:?}"),
    };
    assert_eq!(ids.len(), 1, "exactly one foreign child should attach");
    let child = &ids[0];

    let outcome = fleet
        .child_outcome(child)
        .expect("the foreign child should have a recorded outcome");
    assert_eq!(
        outcome.end_reason,
        EndReason::Completed,
        "the foreign agent should drive to Completed"
    );
    assert!(
        fleet.fleet_usage().api_calls >= 1,
        "the foreign agent's usage should fold into the fleet total, got {:?}",
        fleet.fleet_usage()
    );

    // The foreign process appears in the GUI tree projection as an indistinguishable Engine leaf.
    let tree = fleet.tree();
    let node = tree
        .nodes
        .iter()
        .find(|n| &n.id == child)
        .expect("the foreign child should appear in the tree");
    assert_eq!(
        node.kind,
        daemon_api::UnitKind::Engine,
        "a foreign brain is an Engine leaf in the tree"
    );
    assert!(
        fleet
            .unit_events(child, 0)
            .iter()
            .any(|e| matches!(e, daemon_api::ManageEventView::Finished { .. })),
        "the foreign child's drill-down events should include Finished"
    );

    // The rich, transcript-fidelity drill-down: the full §17 Outbound stream is addressable by
    // UnitId, and the opaque structured payloads survive the foreign process -> CBOR cut -> host ->
    // surface round-trip byte-for-byte (the daemon never parsed `kind`/`body`).
    let outbound = fleet.unit_outbound(child, 0);
    assert!(
        outbound.len() >= 6,
        "the rich drain should carry the full §17 turn, got {} item(s)",
        outbound.len()
    );

    let tool_detail = outbound.iter().find_map(|o| match o {
        Outbound::Event(AgentEvent::ToolFinished { result, .. }) => result.detail.clone(),
        _ => None,
    });
    let tool_detail = tool_detail.expect("ToolFinished should carry a structured detail envelope");
    assert_eq!(tool_detail.kind, EXPECT_TOOL_DETAIL_KIND);
    assert_eq!(
        tool_detail.body, EXPECT_TOOL_DETAIL_BODY,
        "the opaque tool payload must pass through untouched"
    );

    let content = outbound.iter().find_map(|o| match o {
        Outbound::Event(AgentEvent::ContentDelta { kind, body, .. }) => {
            Some((kind.clone(), body.clone()))
        }
        _ => None,
    });
    let (content_kind, content_body) =
        content.expect("a tool-independent ContentDelta should reach the surface");
    assert_eq!(content_kind, EXPECT_CONTENT_KIND);
    assert_eq!(
        content_body, EXPECT_CONTENT_BODY,
        "the opaque terminal/PTY stream must pass through untouched"
    );

    // A second drain is empty: the rich view is a destructive poll, like the per-session drain.
    assert!(
        fleet.unit_outbound(child, 0).is_empty(),
        "unit_outbound is a destructive drain"
    );
}
