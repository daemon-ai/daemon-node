//! Integration test: drive the real `daemon-metta` worker process over the length-framed cut.
//!
//! Spawns the compiled worker binary (via `CARGO_BIN_EXE_daemon-metta`) through the same
//! `ProcessProvisioner` the supervised client uses, then exercises the SKILL round-trip
//! `assert -> match -> eval -> explain`, durable persistence across a respawn, and a clean
//! shutdown — the worker-side mirror of the inference worker's protocol tests.

use daemon_metta::protocol::{self, Command, Event, OpResponse, Provenance, Space};
use daemon_provision::{
    CutReader, CutWriter, Placement, PlacementSpec, ProcessProvisioner, Provisioner,
};

struct Conn {
    writer: CutWriter,
    reader: CutReader,
    child: daemon_provision::ChildGuard,
    next_id: u64,
}

async fn spawn(state_dir: Option<&std::path::Path>) -> Conn {
    let mut args = Vec::new();
    if let Some(dir) = state_dir {
        args.push("--state-dir".to_string());
        args.push(dir.to_string_lossy().into_owned());
    }
    let spec = PlacementSpec {
        program: env!("CARGO_BIN_EXE_daemon-metta").into(),
        args,
        env: Vec::new(),
    };
    let session = daemon_common::SessionId::new("test-metta");
    let Placement { channel, child } = ProcessProvisioner::new()
        .place(&session, spec)
        .await
        .expect("spawn worker");
    let (writer, mut reader) = channel.split();
    // First frame must be `Ready`.
    let bytes = reader.recv().await.expect("ready frame");
    match protocol::decode::<Event>(&bytes).expect("decode ready") {
        Event::Ready { .. } => {}
        other => panic!("expected Ready, got {other:?}"),
    }
    Conn {
        writer,
        reader,
        child,
        next_id: 1,
    }
}

impl Conn {
    async fn call(&mut self, mut cmd: Command) -> OpResponse {
        let id = self.next_id;
        self.next_id += 1;
        cmd.set_request_id(id);
        let bytes = protocol::encode(&cmd).unwrap();
        self.writer.send(&bytes).await.unwrap();
        loop {
            let frame = self.reader.recv().await.expect("reply frame");
            match protocol::decode::<Event>(&frame).expect("decode event") {
                Event::Reply(resp) if resp.request_id == id => return resp,
                Event::Error {
                    request_id,
                    class,
                    message,
                } if request_id == Some(id) => panic!("worker error {class:?}: {message}"),
                _ => {}
            }
        }
    }

    async fn shutdown(mut self) {
        let bytes = protocol::encode(&Command::Shutdown).unwrap();
        let _ = self.writer.send(&bytes).await;
        self.child.shutdown().await;
    }
}

#[tokio::test]
async fn assert_match_eval_explain_roundtrip() {
    let mut conn = spawn(None).await;

    let asserted = conn
        .call(Command::Assert {
            request_id: 0,
            atoms: vec!["(owns alice artifact-42)".into()],
            space: Space::Semantic,
            provenance: Provenance {
                source: Some("user-message msg-1".into()),
                ..Default::default()
            },
            idempotency_key: None,
            expected_snapshot: None,
        })
        .await;
    assert!(asserted.ok);
    assert_eq!(asserted.committed_ids.len(), 1);
    let rec_id = asserted.committed_ids[0].clone();

    let matched = conn
        .call(Command::Match {
            request_id: 0,
            pattern: "(owns $p artifact-42)".into(),
            space: Space::Semantic,
            limit: 0,
            cursor: None,
        })
        .await;
    assert!(matched.ok);
    assert_eq!(matched.results.len(), 1);
    assert!(matched.results[0].contains("$p = alice"));

    let evaled = conn
        .call(Command::Eval {
            request_id: 0,
            expression: "(+ 2 3)".into(),
            space: Space::Working,
            bounds: protocol::Bounds::default(),
            allow_grounded: false,
        })
        .await;
    assert_eq!(evaled.results, vec!["5".to_string()]);

    let explained = conn
        .call(Command::Explain {
            request_id: 0,
            target: rec_id.clone(),
            max_depth: 3,
        })
        .await;
    assert!(explained.ok);
    assert!(explained.results.iter().any(|l| l.contains("artifact-42")));
    assert!(explained.results.iter().any(|l| l.contains("msg-1")));

    conn.shutdown().await;
}

#[tokio::test]
async fn durable_state_survives_respawn() {
    let dir = tempfile::tempdir().unwrap();

    // First worker: assert and promote a procedure, then exit.
    {
        let mut conn = spawn(Some(dir.path())).await;
        conn.call(Command::Assert {
            request_id: 0,
            atoms: vec!["(fact persisted)".into()],
            space: Space::Semantic,
            provenance: Provenance::default(),
            idempotency_key: None,
            expected_snapshot: None,
        })
        .await;
        let defined = conn
            .call(Command::Define {
                request_id: 0,
                program: "(= (f) 1)".into(),
                metadata: Some("{\"id\":\"proc-x\"}".into()),
                tests: vec![],
                status: protocol::Status::Candidate,
            })
            .await;
        assert!(defined.ok);
        conn.call(Command::Promote {
            request_id: 0,
            candidate_id: "proc-x".into(),
            evidence: vec!["traj-1".into()],
            expected_version: Some(1),
        })
        .await;
        conn.shutdown().await;
    }

    // Respawn over the same state dir: the journal replay must reproduce the data + active version.
    {
        let mut conn = spawn(Some(dir.path())).await;
        let matched = conn
            .call(Command::Match {
                request_id: 0,
                pattern: "(fact $x)".into(),
                space: Space::Semantic,
                limit: 0,
                cursor: None,
            })
            .await;
        assert_eq!(matched.results.len(), 1, "asserted fact survived respawn");
        let explained = conn
            .call(Command::Explain {
                request_id: 0,
                target: "proc-x".into(),
                max_depth: 1,
            })
            .await;
        assert!(
            explained.results.iter().any(|l| l.contains("active=Some(1)")),
            "promoted procedure version survived respawn: {:?}",
            explained.results
        );
        conn.shutdown().await;
    }
}
