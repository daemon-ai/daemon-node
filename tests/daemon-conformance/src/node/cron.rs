// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

use super::harness::*;

/// Assemble a node retaining a handle to its shared durable store, so a cron test can observe the
/// isolated `cron_*` session (an `EphemeralSubagent`, excluded from the top-level roster) directly.
fn assemble_with_store() -> (
    Arc<NodeApiImpl>,
    daemon_host::SupervisorHandle,
    Arc<dyn SessionStore>,
) {
    let store: Arc<dyn SessionStore> = Arc::new(InMemoryStore::new());
    let AssembledNode { node, handle, .. } =
        assemble_over(store.clone(), 0, [0x5c; 32], fast_host_config());
    (node, handle, store)
}

/// I15(a): `cron_create` -> `cron_list` surfaces a computed `next_fire_unix`, and the in-process
/// trait call and the Unix-socket round-trip agree (transport parity).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cron_create_lists_with_next_fire_over_socket() {
    let (node, handle) = assemble();
    let path = temp_socket();
    let _ = std::fs::remove_file(&path);
    let listener = UnixListener::bind(&path).expect("bind api socket");
    let server = tokio::spawn(serve_api_unix(listener, node.clone()));
    let client = ApiClient::new(path.clone());

    let spec = daemon_api::CronSpec {
        name: "daily".into(),
        schedule: "0 9 * * *".into(),
        payload: b"do the thing".to_vec(),
        enabled: true,
        ..daemon_api::CronSpec::default()
    };
    let id = match client.call(ApiRequest::CronCreate { spec }).await.unwrap() {
        ApiResponse::CronId(id) => id,
        other => panic!("expected CronId, got {other:?}"),
    };
    assert!(!id.is_empty(), "create must mint a job id");

    let jobs = match client.call(ApiRequest::CronList).await.unwrap() {
        ApiResponse::CronJobs(jobs) => jobs,
        other => panic!("expected CronJobs, got {other:?}"),
    };
    let job = jobs
        .iter()
        .find(|j| j.id == id)
        .expect("created job must be listed");
    assert_eq!(job.spec.name, "daily");
    assert!(
        job.next_fire_unix.is_some(),
        "an enabled cron job must have a computed next fire"
    );
    assert!(!job.paused);

    // Transport parity: the in-process surface agrees with the socket round-trip.
    let inproc = node.cron_list().await;
    assert_eq!(
        inproc.iter().map(|j| j.id.clone()).collect::<Vec<_>>(),
        jobs.iter().map(|j| j.id.clone()).collect::<Vec<_>>(),
        "cron_list must agree across transports"
    );

    server.abort();
    handle.shutdown().await;
    let _ = std::fs::remove_file(&path);
}

/// I15(b): `cron_trigger` materializes an isolated `cron_{id}_{ts}` session that the resident
/// activation path drives to `Completed`, and records a `CronRun` (`trigger = Manual`, carrying
/// the fired session) discoverable via `cron_runs`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cron_trigger_fires_isolated_session_and_records_run() {
    let (node, handle, store) = assemble_with_store();
    let path = temp_socket();
    let _ = std::fs::remove_file(&path);
    let listener = UnixListener::bind(&path).expect("bind api socket");
    let server = tokio::spawn(serve_api_unix(listener, node.clone()));
    let client = ApiClient::new(path.clone());

    let spec = daemon_api::CronSpec {
        name: "manual".into(),
        schedule: "0 9 * * *".into(),
        payload: b"run now please".to_vec(),
        enabled: true,
        ..daemon_api::CronSpec::default()
    };
    let id = match client.call(ApiRequest::CronCreate { spec }).await.unwrap() {
        ApiResponse::CronId(id) => id,
        other => panic!("expected CronId, got {other:?}"),
    };

    assert!(matches!(
        client
            .call(ApiRequest::CronTrigger { id: id.clone() })
            .await
            .unwrap(),
        ApiResponse::Ok
    ));

    // A run is recorded carrying the isolated cron session.
    let deadline = Instant::now() + Duration::from_secs(10);
    let session = loop {
        let runs = match client
            .call(ApiRequest::CronRuns { id: id.clone() })
            .await
            .unwrap()
        {
            ApiResponse::CronRuns(runs) => runs,
            other => panic!("expected CronRuns, got {other:?}"),
        };
        if let Some(run) = runs.first() {
            assert_eq!(
                run.trigger,
                daemon_api::RunTrigger::Manual,
                "a cron_trigger run is Manual"
            );
            if let Some(session) = run.session.clone() {
                assert!(
                    session.as_str().starts_with("cron_"),
                    "the fired session is an isolated cron_* session, got {session}"
                );
                break session;
            }
        }
        assert!(
            Instant::now() < deadline,
            "cron_trigger never recorded a run"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    };

    // The activation path drives the isolated session to completion.
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if matches!(
            store.status(&session).await,
            Some(daemon_store::SessionStatus::Completed)
        ) {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "the fired cron session never reached Completed"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    server.abort();
    handle.shutdown().await;
    let _ = std::fs::remove_file(&path);
}

/// Phase 2 (outcome + deliver): a scheduled cron run captures its session's **real** final
/// assistant text into `CronRun.detail` (replacing the hardcoded `"completed"`), and a
/// `deliver = "<transport>:<chat>"` directive pushes that text through the host's existing
/// [`DeliverySink`](daemon_api::DeliverySink) registry — the same outbound path a live reply uses.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cron_run_captures_output_and_delivers_to_sink() {
    use daemon_host::DeliveryHost;

    let (node, handle, _store) = assemble_with_store();

    // A test sink capturing the assistant text of every entry delivered to transport "test".
    struct CapturingSink {
        seen: Arc<std::sync::Mutex<Vec<String>>>,
    }
    #[async_trait::async_trait]
    impl daemon_api::DeliverySink for CapturingSink {
        async fn deliver(
            &self,
            _target: daemon_protocol::DeliveryTarget,
            entry: daemon_protocol::SessionLogEntry,
        ) {
            if let daemon_protocol::SessionPayload::Event(
                daemon_protocol::AgentEvent::TextDelta { text, .. },
            ) = entry.payload
            {
                self.seen.lock().unwrap().push(text);
            }
        }
    }
    let seen = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    node.register_delivery_sink(
        daemon_protocol::TransportId::new("test"),
        Arc::new(CapturingSink { seen: seen.clone() }),
    );

    // A job that fires every second and delivers its result to `test:room1`. The constrained cron
    // base runs the `child` mock provider, whose final assistant message is `"child done"`.
    let spec = daemon_api::CronSpec {
        name: "deliverer".into(),
        schedule: "@every 1s".into(),
        payload: b"summarize".to_vec(),
        deliver: Some("test:room1".into()),
        enabled: true,
        ..daemon_api::CronSpec::default()
    };
    let id = node.cron_create(spec).await.expect("create cron job");

    // The resident scheduler fires the job (1s cadence), the activation path settles it, and a
    // subsequent tick reconciles it: the real output is captured and delivered to the sink.
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        if seen.lock().unwrap().iter().any(|t| t == "child done") {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "cron run output was never delivered to the sink; saw {:?}",
            seen.lock().unwrap()
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    // The captured outcome is also folded into the durable run log (not a hardcoded "completed").
    let finished = node
        .cron_runs(id)
        .await
        .into_iter()
        .find(|r| r.finished_unix.is_some())
        .expect("a finished run is recorded");
    assert!(finished.ok, "a run that produced output is recorded ok");
    assert_eq!(
        finished.detail.as_deref(),
        Some("child done"),
        "the run detail carries the session's real final assistant text"
    );

    handle.shutdown().await;
}

/// I15(H): the consent-first suggestion catalog is seeded on first read, accepting a suggestion
/// creates its backing job (and drops it from the pending list), and the surface is
/// transport-agnostic.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cron_suggestions_seed_accept_and_dismiss() {
    let (node, handle) = assemble();
    let path = temp_socket();
    let _ = std::fs::remove_file(&path);
    let listener = UnixListener::bind(&path).expect("bind api socket");
    let server = tokio::spawn(serve_api_unix(listener, node.clone()));
    let client = ApiClient::new(path.clone());

    let pending = match client.call(ApiRequest::CronSuggestions).await.unwrap() {
        ApiResponse::CronSuggestions(s) => s,
        other => panic!("expected CronSuggestions, got {other:?}"),
    };
    assert!(
        pending.len() >= 4,
        "the starter catalog seeds at least four suggestions, got {}",
        pending.len()
    );
    let accept_id = pending[0].id.clone();
    let dismiss_id = pending[1].id.clone();

    // Accept -> a backing job is created.
    let job_id = match client
        .call(ApiRequest::CronAcceptSuggestion {
            id: accept_id.clone(),
        })
        .await
        .unwrap()
    {
        ApiResponse::CronId(id) => id,
        other => panic!("expected CronId, got {other:?}"),
    };
    let jobs = match client.call(ApiRequest::CronList).await.unwrap() {
        ApiResponse::CronJobs(jobs) => jobs,
        other => panic!("expected CronJobs, got {other:?}"),
    };
    assert!(
        jobs.iter().any(|j| j.id == job_id),
        "accepting a suggestion creates a job"
    );

    // Dismiss -> latched.
    assert!(matches!(
        client
            .call(ApiRequest::CronDismissSuggestion {
                id: dismiss_id.clone()
            })
            .await
            .unwrap(),
        ApiResponse::Ok
    ));

    // Neither the accepted nor the dismissed suggestion is re-offered (latched by dedup_key).
    let remaining = node.cron_suggestions().await;
    assert!(
        !remaining
            .iter()
            .any(|s| s.id == accept_id || s.id == dismiss_id),
        "accepted/dismissed suggestions are latched out of the pending list"
    );

    server.abort();
    handle.shutdown().await;
    let _ = std::fs::remove_file(&path);
}
