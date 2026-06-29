// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Foreign native-protocol adapters: a Claude-Code `stream-json` agent and an ACP agent each attach
//! as a first-class fleet member through the `ProfileChildSpawner` protocol selector.
//!
//! For each protocol this proves the three properties that make a foreign brain indistinguishable
//! from `daemon-core` at the management surface: (a) `Assign` drives it to a terminal
//! `Finished{Completed}` exactly like an engine, (b) a blocking permission request the agent raises
//! round-trips through the installed answer-authority, and (c) the unit's transcript seals into a
//! verifiable journal segment that checks out under the node's signing key — all over a real OS
//! process boundary, with no LLM.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use daemon_api::Outbound;
use daemon_common::{Budget, ContentHash, JournalStreamId, ReqId, UnitId};
use daemon_host::JournalConfig;
use daemon_node::{ForeignProtocol, LaunchProfile, ProfileChildSpawner};
use daemon_orchestration::ChildSpawner;
use daemon_store::{InMemoryStore, SessionStore, TraceSegment};
use daemon_supervision::{
    Ack, DelegationSpec, EndReason, EventStream, ManageCommand, ManageEvent, ManageRequest,
    ManageRequestHandler, ManageResponse, ManageResponseBody, UnitKind, WorkRef,
};
use daemon_telemetry::{verify_segment, SegmentInput, TraceSigner, GENESIS_ROOT};

/// The node journal signer seed for these tests; the sealed chain verifies under its verifying key.
const SEED: [u8; 32] = [0x22; 32];

/// A supervisor handler that approves everything (the answer-authority for the foreign unit).
struct Approver;

#[async_trait]
impl ManageRequestHandler for Approver {
    async fn request(&self, req: ManageRequest) -> ManageResponse {
        ManageResponse {
            request_id: req.request_id,
            body: ManageResponseBody::Approved(true),
        }
    }
}

fn mock(bin_env: &str) -> PathBuf {
    PathBuf::from(bin_env)
}

fn delegation() -> DelegationSpec {
    DelegationSpec {
        work: WorkRef::inline("w1", "do the work"),
        budget: Budget::unlimited(),
        toolset: Vec::new(),
    }
}

/// The loaded segment's entries shaped for `verify_segment`.
fn loaded_entries(seg: &TraceSegment) -> Vec<(u64, Vec<u8>, ContentHash)> {
    seg.entries
        .iter()
        .map(|e| (e.seq, e.bytes.clone(), e.content_hash))
        .collect()
}

async fn await_terminal(events: &mut EventStream<ManageEvent>) -> ManageEvent {
    loop {
        match tokio::time::timeout(Duration::from_secs(15), events.recv()).await {
            Ok(Ok(ev @ (ManageEvent::Finished { .. } | ManageEvent::Error { .. }))) => return ev,
            Ok(Ok(_)) => continue,
            Ok(Err(_)) => panic!("event stream closed before a terminal event"),
            Err(_) => panic!("timed out waiting for a terminal event from the foreign agent"),
        }
    }
}

/// Drive one foreign protocol end-to-end: spawn the mock through the protocol selector, drive a
/// turn, and assert it (a) completes like an engine, (b) round-tripped a permission request, and
/// (c) sealed a verifiable journal segment.
async fn drive_protocol(program: PathBuf, protocol: ForeignProtocol, label: &str) {
    let store: Arc<dyn SessionStore> = Arc::new(InMemoryStore::new());
    let signer = Arc::new(TraceSigner::from_seed(&SEED));
    let spawner = ProfileChildSpawner::foreign(LaunchProfile {
        program,
        args: Vec::new(),
        env: Vec::new(),
        protocol,
    })
    .with_journal(JournalConfig {
        store: store.clone(),
        signer: signer.clone(),
    });

    let id = UnitId::new(label);
    let unit = spawner.spawn(id.clone(), &delegation()).await;
    assert_eq!(
        unit.kind(),
        UnitKind::Engine,
        "a foreign agent presents as an Engine leaf"
    );
    unit.install_request_handler(Arc::new(Approver));
    let mut events = unit.events();

    let ack = unit
        .command(ManageCommand::Assign {
            request_id: ReqId(1),
            work: WorkRef::inline("w1", "do the work"),
            budget: Budget::unlimited(),
        })
        .await;
    assert_eq!(
        ack,
        Ack::Accepted,
        "the foreign unit should accept the work"
    );

    let terminal = await_terminal(&mut events).await;
    assert!(
        matches!(
            terminal,
            ManageEvent::Finished { ref outcome, .. } if outcome.end_reason == EndReason::Completed
        ),
        "the foreign agent should map up to Finished{{Completed}}, got {terminal:?}"
    );

    // (b) A blocking permission request the agent raised round-tripped up the rich §17 drain.
    let raised_permission = unit
        .drain_outbound(0)
        .iter()
        .any(|o| matches!(o, Outbound::Request(_)));
    assert!(
        raised_permission,
        "the agent's permission request should round-trip up the §17 drain"
    );

    // (c) The unit's transcript sealed a verifiable segment under the node's signing key.
    let stream = JournalStreamId::unit(&id);
    let seg = store
        .load_trace_segment(&stream, 0)
        .await
        .expect("the foreign unit should have journaled a sealed segment keyed by its UnitId");
    let committed = seg
        .committed
        .clone()
        .expect("the segment should be sealed (committed root present) after the turn");
    let verifying = TraceSigner::from_seed(&SEED).verifying_key();
    let entries = loaded_entries(&seg);
    verify_segment(
        &SegmentInput {
            stream: &stream,
            segment: 0,
            prior: GENESIS_ROOT,
            entries: &entries,
        },
        &committed.root,
        &committed.signature,
        &verifying,
    )
    .expect("the foreign unit's sealed segment must verify under the node verifying key");

    unit.command(ManageCommand::Shutdown { drain: false }).await;
}

/// A Claude-Code `stream-json` agent attaches via the line transport + `StreamJsonCodec`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stream_json_agent_maps_up_and_seals_a_journal() {
    drive_protocol(
        mock(env!("CARGO_BIN_EXE_mock_stream_json_agent")),
        ForeignProtocol::StreamJson,
        "sj-1",
    )
    .await;
}

/// An ACP agent attaches via the `daemon-acp` adapter.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn acp_agent_maps_up_and_seals_a_journal() {
    drive_protocol(
        mock(env!("CARGO_BIN_EXE_mock_acp_agent")),
        ForeignProtocol::Acp,
        "acp-1",
    )
    .await;
}
