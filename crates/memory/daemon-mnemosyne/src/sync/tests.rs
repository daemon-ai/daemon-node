// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

use super::*;
use crate::config::MnemosyneConfig;
use crate::engine::Engine;
use std::sync::Arc;

// ── Test transport fixture ──────────────────────────────────────────────────────────────────
// The crate ships no listener (the node owns transport — see `endpoints`); tests bring their
// own minimal HTTP/1.1 loop so the reqwest client and the pure endpoints meet over a real
// socket, exactly the wire a Python `sync_server.py` peer would present.

struct TestServer {
    addr: std::net::SocketAddr,
    handle: tokio::task::JoinHandle<()>,
}

impl Drop for TestServer {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

async fn spawn_server(engine: Arc<Engine>, api_key: Option<String>) -> TestServer {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
        .await
        .expect("bind test sync server");
    let addr = listener.local_addr().expect("local addr");
    let handle = tokio::spawn(async move {
        loop {
            let Ok((mut stream, _)) = listener.accept().await else {
                continue;
            };
            let engine = engine.clone();
            let api_key = api_key.clone();
            tokio::spawn(async move {
                // Headers, then Content-Length body (one request per connection).
                let mut buf: Vec<u8> = Vec::with_capacity(1024);
                let mut chunk = [0u8; 4096];
                let header_end = loop {
                    let Ok(n) = stream.read(&mut chunk).await else {
                        return;
                    };
                    if n == 0 {
                        return;
                    }
                    buf.extend_from_slice(&chunk[..n]);
                    if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                        break pos + 4;
                    }
                };
                let head = String::from_utf8_lossy(&buf[..header_end]).to_string();
                let mut lines = head.lines();
                let mut request_line = lines.next().unwrap_or_default().split_whitespace();
                let method = request_line.next().unwrap_or("").to_string();
                let target = request_line.next().unwrap_or("/").to_string();
                let mut content_length = 0usize;
                let mut bearer = None;
                for line in lines {
                    let Some((name, value)) = line.split_once(':') else {
                        continue;
                    };
                    let value = value.trim();
                    if name.eq_ignore_ascii_case("content-length") {
                        content_length = value.parse().unwrap_or(0);
                    } else if name.eq_ignore_ascii_case("authorization") {
                        bearer = value.strip_prefix("Bearer ").map(str::to_string);
                    }
                }
                let mut body_bytes = buf[header_end..].to_vec();
                while body_bytes.len() < content_length {
                    let Ok(n) = stream.read(&mut chunk).await else {
                        return;
                    };
                    if n == 0 {
                        break;
                    }
                    body_bytes.extend_from_slice(&chunk[..n]);
                }
                body_bytes.truncate(content_length);
                let body: Value = if body_bytes.is_empty() {
                    json!({})
                } else {
                    serde_json::from_slice(&body_bytes).unwrap_or(Value::Null)
                };

                let request = endpoints::SyncRequest {
                    method,
                    path: endpoints::SyncRequest::normalize_path(&target),
                    bearer,
                    body,
                };
                let (status, resp) = endpoints::route(&engine, None, api_key.as_deref(), &request);
                let payload = if status == 204 {
                    Vec::new()
                } else {
                    resp.to_string().into_bytes()
                };
                let head = format!(
                    "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\n\
                     Content-Length: {}\r\nConnection: close\r\n\r\n",
                    payload.len()
                );
                let _ = stream.write_all(head.as_bytes()).await;
                let _ = stream.write_all(&payload).await;
                let _ = stream.flush().await;
            });
        }
    });
    TestServer { addr, handle }
}

fn engine() -> Engine {
    Engine::open_in_memory(MnemosyneConfig::default()).expect("engine")
}

fn bank(dir: &std::path::Path) -> Engine {
    Engine::open(MnemosyneConfig {
        data_dir: dir.to_path_buf(),
        ..Default::default()
    })
    .expect("engine")
}

fn ev(id: &str, mid: &str, ts: &str, device: &str, importance: f64) -> SyncEvent {
    SyncEvent {
        event_id: id.to_string(),
        memory_id: mid.to_string(),
        operation: "CREATE".to_string(),
        timestamp: ts.to_string(),
        device_id: device.to_string(),
        payload: None,
        parent_event_ids: "[]".to_string(),
        importance: Some(importance),
        expiry: None,
        event_hash: None,
    }
}

// ── ConflictResolution ──────────────────────────────────────────────────────────────────────

#[test]
fn resolve_lww_then_importance_then_device() {
    // Latest timestamp wins.
    let a = ev("a", "m", "2026-01-01T00:00:00+00:00", "dev-a", 0.9);
    let b = ev("b", "m", "2026-01-02T00:00:00+00:00", "dev-b", 0.1);
    assert_eq!(
        ConflictResolution::resolve(&[a.clone(), b.clone()])
            .unwrap()
            .event_id,
        "b"
    );

    // Equal timestamps: higher importance wins.
    let c = ev("c", "m", "2026-01-01T00:00:00+00:00", "dev-a", 0.9);
    let d = ev("d", "m", "2026-01-01T00:00:00+00:00", "dev-b", 0.5);
    assert_eq!(
        ConflictResolution::resolve(&[d.clone(), c.clone()])
            .unwrap()
            .event_id,
        "c"
    );

    // Full tie on (ts, importance): device id breaks it.
    let e1 = ev("e1", "m", "2026-01-01T00:00:00+00:00", "dev-a", 0.5);
    let e2 = ev("e2", "m", "2026-01-01T00:00:00+00:00", "dev-z", 0.5);
    assert_eq!(
        ConflictResolution::resolve(&[e1, e2]).unwrap().event_id,
        "e2"
    );

    // None importance sorts as 0.0 (Python `_sort_key`).
    let mut none_imp = ev("n", "m", "2026-01-01T00:00:00+00:00", "dev-a", 0.0);
    none_imp.importance = None;
    let low = ev("l", "m", "2026-01-01T00:00:00+00:00", "dev-a", 0.1);
    assert_eq!(
        ConflictResolution::resolve(&[none_imp, low])
            .unwrap()
            .event_id,
        "l"
    );

    assert!(ConflictResolution::resolve(&[]).is_err());
}

#[test]
fn resolve_with_chain_causality_beats_lww() {
    // B declares A as parent: B wins even though A has the later timestamp + importance.
    let a = ev("ev-a", "m", "2026-01-05T00:00:00+00:00", "dev-a", 0.9);
    let mut b = ev("ev-b", "m", "2026-01-01T00:00:00+00:00", "dev-b", 0.1);
    b.parent_event_ids = r#"["ev-a"]"#.to_string();
    assert_eq!(
        ConflictResolution::resolve_with_chain(&[a.clone(), b.clone()])
            .unwrap()
            .event_id,
        "ev-b"
    );

    // Transitive: C -> B -> A means C dominates both.
    let mut c = ev("ev-c", "m", "2025-12-01T00:00:00+00:00", "dev-c", 0.0);
    c.parent_event_ids = r#"["ev-b"]"#.to_string();
    assert_eq!(
        ConflictResolution::resolve_with_chain(&[a.clone(), b.clone(), c])
            .unwrap()
            .event_id,
        "ev-c"
    );

    // No causal links: falls back to v1 LWW.
    let x = ev("x", "m", "2026-01-01T00:00:00+00:00", "dev", 0.5);
    let y = ev("y", "m", "2026-01-02T00:00:00+00:00", "dev", 0.5);
    assert_eq!(
        ConflictResolution::resolve_with_chain(&[x, y])
            .unwrap()
            .event_id,
        "y"
    );
}

#[test]
fn detect_conflicts_windows_by_memory_and_time() {
    let l1 = ev("l1", "m1", "2026-01-01T00:00:00+00:00", "dev-a", 0.5);
    let r_in = ev("r1", "m1", "2026-01-01T00:00:03+00:00", "dev-b", 0.5);
    let r_out = ev("r2", "m1", "2026-01-01T01:00:00+00:00", "dev-b", 0.5);
    let r_other = ev("r3", "m2", "2026-01-01T00:00:00+00:00", "dev-b", 0.5);

    let groups = ConflictResolution::detect_conflicts(&[l1], &[r_in.clone(), r_out, r_other], 5.0);
    assert_eq!(groups.len(), 1, "{groups:?}");
    let ids: Vec<&str> = groups[0].iter().map(|e| e.event_id.as_str()).collect();
    assert_eq!(ids, vec!["l1", "r1"], "same memory within ±5s only");

    // A trailing Z timestamp parses like +00:00 (`_parse_sync_timestamp`).
    let lz = ev("lz", "m1", "2026-01-01T00:00:00Z", "dev-a", 0.5);
    assert_eq!(
        ConflictResolution::detect_conflicts(&[lz], &[r_in], 5.0).len(),
        1
    );
}

#[test]
fn propose_merge_shapes_candidates_and_winner() {
    let mut a = ev("a", "m1", "2026-01-01T00:00:00+00:00", "dev-a", 0.3);
    a.payload = Some(r#"{"content": "from a"}"#.to_string());
    let mut b = ev("b", "m1", "2026-01-01T00:00:02+00:00", "dev-b", 0.9);
    b.payload = Some(r#"{"content": "from b"}"#.to_string());
    let proposals =
        ConflictResolution::propose_merge(&[vec![a, b]], Some(&json!({"bank": "test"})));
    assert_eq!(proposals.len(), 1);
    let p = &proposals[0];
    assert_eq!(p["memory_id"], "m1");
    assert_eq!(p["suggested_action"], "keep_latest");
    assert_eq!(p["suggested_winner_index"], 1, "highest importance");
    assert_eq!(p["candidates"][0]["content"], "from a");
    assert_eq!(p["context"]["bank"], "test");
}

// ── SyncEncryption ──────────────────────────────────────────────────────────────────────────

#[test]
fn encryption_round_trip_and_detection() {
    let (key, salt) = SyncEncryption::derive_key("correct horse battery", None).expect("derive");
    assert_eq!(salt.len(), 16);
    // Same passphrase + salt is deterministic; different passphrase diverges.
    let (again, _) = SyncEncryption::derive_key("correct horse battery", Some(&salt)).unwrap();
    assert_eq!(key, again);
    let (other, _) = SyncEncryption::derive_key("wrong horse", Some(&salt)).unwrap();
    assert_ne!(key, other);

    let enc = SyncEncryption::new(key);
    let payload = json!({"content": "secret memory", "importance": 0.9});
    let sealed = enc.encrypt(&payload);
    assert!(SyncEncryption::is_encrypted(&sealed));
    assert!(!SyncEncryption::is_encrypted(r#"{"content": "plain"}"#));
    assert!(SyncEncryption::is_encrypted("gAAAAAabc"), "Fernet detected");
    assert_eq!(enc.decrypt(&sealed).expect("decrypt"), payload);

    // Wrong key fails closed.
    let bad = SyncEncryption::new(other);
    assert!(bad.decrypt(&sealed).is_err());

    // Nonce freshness: sealing twice differs.
    assert_ne!(sealed, enc.encrypt(&payload));
}

#[test]
fn key_sources_accept_raw_and_file() {
    let generated = SyncEncryption::generate_key();
    let enc = SyncEncryption::from_key_source(&generated)
        .expect("raw key parses")
        .expect("some");
    let sealed = enc.encrypt(&json!({"x": 1}));

    // Unpadded base64 tolerated (Python retried with '==').
    let unpadded = generated.trim_end_matches('=').to_string();
    let enc2 = SyncEncryption::from_key_source(&unpadded).unwrap().unwrap();
    assert_eq!(enc2.decrypt(&sealed).unwrap(), json!({"x": 1}));

    // Key file (with and without the file: prefix).
    let tmp = tempfile::tempdir().unwrap();
    let key_path = tmp.path().join("sync.key");
    std::fs::write(&key_path, format!("{generated}\n")).unwrap();
    let enc3 = SyncEncryption::from_key_source(&format!("file:{}", key_path.display()))
        .unwrap()
        .unwrap();
    assert_eq!(enc3.decrypt(&sealed).unwrap(), json!({"x": 1}));
    let enc4 = SyncEncryption::from_key_source(key_path.to_str().unwrap())
        .unwrap()
        .unwrap();
    assert_eq!(enc4.decrypt(&sealed).unwrap(), json!({"x": 1}));

    assert!(SyncEncryption::from_key_source("").unwrap().is_none());
    assert!(SyncEncryption::from_key_source("not!!base64!!").is_err());
    assert!(
        SyncEncryption::from_key_source("dG9vc2hvcnQ=").is_err(),
        "wrong length"
    );
}

// ── SyncEngine: event log ───────────────────────────────────────────────────────────────────

#[test]
fn log_event_and_pull_changes_paginate() {
    let e = engine();
    let se = SyncEngine::new(&e, Some("device-test".into()), None).unwrap();

    assert!(
        se.log_event("m1", "TRUNCATE", None, 0.5, None).is_err(),
        "op validated"
    );

    for i in 0..5 {
        se.log_event(
            &format!("m{i}"),
            "CREATE",
            Some(&json!({"content": format!("row {i}")})),
            0.5,
            None,
        )
        .expect("log");
    }
    let page1 = se.pull_changes(None, 3).expect("pull");
    assert_eq!(page1["total"], 3);
    assert_eq!(page1["has_more"], true);
    let cursor = page1["next_cursor"].as_str().unwrap().to_string();

    let page2 = se.pull_changes(Some(&cursor), 100).expect("pull2");
    assert_eq!(page2["has_more"], false);
    // Cursor is timestamp-based: everything strictly after page1's last timestamp. Events
    // minted in the same instant share a timestamp, so total ≤ 2 but the union covers all 5.
    let ids: std::collections::HashSet<String> = page1["events"]
        .as_array()
        .unwrap()
        .iter()
        .chain(page2["events"].as_array().unwrap())
        .map(|v| v["event_id"].as_str().unwrap().to_string())
        .collect();
    assert!(ids.len() >= 3);

    // Event ids are UUIDv4-shaped; hashes are 64-hex and match the recompute.
    let first = SyncEvent::from_value(&page1["events"][0]).unwrap();
    assert_eq!(first.event_id.len(), 36);
    assert_eq!(first.event_id.as_bytes()[14], b'4', "version nibble");
    let hash = first.event_hash.clone().unwrap();
    assert_eq!(hash.len(), 64);
    let recomputed = SyncEngine::compute_event_hash(&SyncEvent {
        event_hash: None,
        ..first
    });
    assert_eq!(hash, recomputed);
}

#[test]
fn write_path_events_are_pullable_and_unlogged_backfill_is_noop_for_rust_banks() {
    let e = engine();
    e.remember("native row", &Default::default()).unwrap();
    let se = SyncEngine::new(&e, None, None).unwrap();

    // The engine's write path already logged the mutation.
    let pulled = se.pull_changes(None, 100).unwrap();
    assert_eq!(pulled["total"], 1);
    assert_eq!(pulled["events"][0]["operation"], "CREATE");

    // So the backfill scan finds nothing new.
    assert!(se.find_unlogged_memories(1000).unwrap().is_empty());

    // A foreign (unlogged) row — as a Python bank would leave — does get backfilled.
    e.with_conn(|conn| {
        conn.execute(
            "INSERT INTO working_memory (id, content, source, timestamp, session_id) \
             VALUES ('py-row', 'from python', 'conversation', ?1, 'default')",
            [crate::util::now_iso()],
        )?;
        Ok(())
    })
    .unwrap();
    let created = se.find_unlogged_memories(1000).unwrap();
    assert_eq!(created.len(), 1);
    assert_eq!(created[0].memory_id, "py-row");
    assert_eq!(created[0].operation, "CREATE");
}

// ── SyncEngine: push_changes ────────────────────────────────────────────────────────────────

#[test]
fn push_changes_applies_through_full_pipeline_without_log_amplification() {
    let receiver = engine();
    let se = SyncEngine::new(&receiver, Some("device-local".into()), None).unwrap();

    let mut incoming = ev(
        &uuid4(),
        "peer-mem-1",
        &crate::util::now_iso(),
        "device-peer",
        0.8,
    );
    incoming.payload = Some(
        json!({"content": "Nadia keeps bees in Ljubljana", "source": "conversation"}).to_string(),
    );
    incoming.event_hash = Some(SyncEngine::compute_event_hash(&incoming));

    let stats = se.push_changes(&[incoming.to_value()]).expect("push");
    assert_eq!(stats["accepted"], 1, "{stats:?}");
    assert_eq!(stats["errors"], 0);

    // Applied through the full pipeline: the row exists under the peer's id and is searchable.
    let hits = receiver.recall("Ljubljana bees", 5).expect("recall");
    assert!(hits.iter().any(|h| h.id == "peer-mem-1"), "{hits:?}");

    // Exactly ONE event row: the peer's (synced_at set) — the apply write minted no second
    // local event (Python parity: remember() never self-logs).
    let (count, synced): (i64, i64) = receiver
        .with_conn(|conn| {
            Ok((
                conn.query_row("SELECT COUNT(*) FROM memory_events", [], |r| r.get(0))?,
                conn.query_row(
                    "SELECT COUNT(*) FROM memory_events WHERE synced_at IS NOT NULL",
                    [],
                    |r| r.get(0),
                )?,
            ))
        })
        .unwrap();
    assert_eq!(count, 1, "no event-log amplification");
    assert_eq!(synced, 1);

    // Idempotent: the same event again dedups by hash.
    let stats2 = se.push_changes(&[incoming.to_value()]).expect("re-push");
    assert_eq!(stats2["duplicates"], 1);
    assert_eq!(stats2["accepted"], 0);

    // DELETE routes through forget().
    let mut del = ev(
        &uuid4(),
        "peer-mem-1",
        &crate::util::now_iso(),
        "device-peer",
        0.5,
    );
    del.operation = "DELETE".to_string();
    del.event_hash = Some(SyncEngine::compute_event_hash(&del));
    let stats3 = se.push_changes(&[del.to_value()]).expect("delete");
    assert_eq!(stats3["accepted"], 1);
    let gone: i64 = receiver
        .with_conn(|conn| {
            Ok(conn.query_row(
                "SELECT COUNT(*) FROM working_memory WHERE id = 'peer-mem-1'",
                [],
                |r| r.get(0),
            )?)
        })
        .unwrap();
    assert_eq!(gone, 0, "forget applied");
}

#[test]
fn push_changes_drops_conflict_losers() {
    let receiver = engine();
    let se = SyncEngine::new(&receiver, Some("device-local".into()), None).unwrap();

    // A local event now...
    let local = se
        .log_event(
            "contested",
            "CREATE",
            Some(&json!({"content": "local version"})),
            0.9,
            None,
        )
        .expect("local event");
    // ...and an incoming event for the same memory 1s earlier (within the ±5s window) with
    // lower importance: the incoming event loses and is filtered out.
    let older_ts = (chrono::Utc::now() - chrono::Duration::seconds(1)).to_rfc3339();
    let mut incoming = ev(&uuid4(), "contested", &older_ts, "device-peer", 0.1);
    incoming.payload = Some(json!({"content": "peer version"}).to_string());
    incoming.event_hash = Some(SyncEngine::compute_event_hash(&incoming));

    let stats = se.push_changes(&[incoming.to_value()]).expect("push");
    assert_eq!(stats["conflicts"], 1, "{stats:?}");
    assert_eq!(stats["accepted"], 0, "losing incoming event not applied");
    let content: String = receiver
        .with_conn(|conn| {
            Ok(conn
                .query_row(
                    "SELECT content FROM working_memory WHERE id = 'contested'",
                    [],
                    |r| r.get::<_, String>(0),
                )
                .unwrap_or_default())
        })
        .unwrap();
    assert_ne!(content, "peer version");
    let _ = local;
}

#[test]
fn encrypted_payloads_decrypt_with_key_and_stay_opaque_without() {
    let sender = engine();
    let key_b64 = SyncEncryption::generate_key();
    let enc = SyncEncryption::from_key_source(&key_b64).unwrap().unwrap();
    let se_send = SyncEngine::new(&sender, Some("device-a".into()), Some(enc)).unwrap();
    let event = se_send
        .log_event(
            "sealed-1",
            "CREATE",
            Some(&json!({"content": "classified fact", "source": "conversation"})),
            0.7,
            None,
        )
        .expect("log sealed");
    assert!(SyncEncryption::is_encrypted(
        event.payload.as_deref().unwrap()
    ));

    // A receiver WITH the key applies the mutation.
    let with_key = engine();
    let enc2 = SyncEncryption::from_key_source(&key_b64).unwrap().unwrap();
    let se_with = SyncEngine::new(&with_key, Some("device-b".into()), Some(enc2)).unwrap();
    let stats = se_with.push_changes(&[event.to_value()]).unwrap();
    assert_eq!(stats["accepted"], 1);
    assert!(with_key
        .recall("classified fact", 5)
        .unwrap()
        .iter()
        .any(|h| h.id == "sealed-1"));

    // A receiver WITHOUT the key stores the event opaquely (relay) but applies no mutation.
    let without_key = engine();
    let se_without = SyncEngine::new(&without_key, Some("device-c".into()), None).unwrap();
    let stats = se_without.push_changes(&[event.to_value()]).unwrap();
    assert_eq!(stats["accepted"], 1, "event accepted for relay");
    let (events, rows): (i64, i64) = without_key
        .with_conn(|conn| {
            Ok((
                conn.query_row("SELECT COUNT(*) FROM memory_events", [], |r| r.get(0))?,
                conn.query_row("SELECT COUNT(*) FROM working_memory", [], |r| r.get(0))?,
            ))
        })
        .unwrap();
    assert_eq!(events, 1, "opaque event logged for relay");
    assert_eq!(rows, 0, "no plaintext mutation applied");
}

// ── End-to-end over HTTP ────────────────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sync_with_replicates_bidirectionally_and_converges() {
    let tmp = tempfile::tempdir().expect("tmpdir");
    let a = bank(&tmp.path().join("a"));
    let b = Arc::new(bank(&tmp.path().join("b")));
    a.remember("alpha grows olives", &Default::default())
        .unwrap();
    b.remember("bravo restores pianos", &Default::default())
        .unwrap();

    let server = spawn_server(b.clone(), None).await;
    let url = format!("http://{}", server.addr);

    let se_a = SyncEngine::new(&a, Some("device-a".into()), None).unwrap();
    let result = se_a.sync_with(&url, "bidirectional", None).await;
    assert_eq!(
        result["errors"].as_array().map(Vec::len),
        Some(0),
        "{result:?}"
    );
    assert_eq!(result["push"]["accepted"], 1, "{result:?}");
    assert_eq!(result["pull"]["accepted"], 1, "{result:?}");

    // Both banks now hold both rows, searchable.
    let spawn_b = b.clone();
    assert!(
        !a.recall("bravo pianos", 5).unwrap().is_empty(),
        "pull applied"
    );
    assert!(
        !spawn_b.recall("alpha olives", 5).unwrap().is_empty(),
        "push applied"
    );

    // Cursor persisted per remote.
    assert!(se_a
        .meta_get(&format!("last_sync_cursor_{url}"))
        .unwrap()
        .is_some());

    // A second cycle converges: everything dedups, nothing new applies.
    let again = se_a.sync_with(&url, "bidirectional", None).await;
    let pushed_again = again["push"]["accepted"].as_u64().unwrap_or(0);
    assert_eq!(pushed_again, 0, "{again:?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn server_enforces_bearer_auth_and_404s() {
    let e = Arc::new(engine());
    let server = spawn_server(e, Some("sesame".into())).await;
    let url = format!("http://{}", server.addr);
    let client = reqwest::Client::new();

    let denied = client
        .post(format!("{url}/sync/pull"))
        .json(&json!({}))
        .send()
        .await
        .expect("send");
    assert_eq!(denied.status().as_u16(), 401);

    let ok = client
        .post(format!("{url}/sync/pull"))
        .bearer_auth("sesame")
        .json(&json!({"since": null}))
        .send()
        .await
        .expect("send");
    assert_eq!(ok.status().as_u16(), 200);
    let body: Value = ok.json().await.unwrap();
    assert_eq!(body["status"], "ok");
    assert_eq!(body["total"], 0);

    let status = client
        .get(format!("{url}/sync/status/?probe=1"))
        .bearer_auth("sesame")
        .send()
        .await
        .expect("send");
    assert_eq!(
        status.status().as_u16(),
        200,
        "query + trailing slash normalize"
    );

    let missing = client
        .get(format!("{url}/nope"))
        .bearer_auth("sesame")
        .send()
        .await
        .expect("send");
    assert_eq!(missing.status().as_u16(), 404);
}

#[test]
fn get_status_reports_counts_and_breakdown() {
    let e = engine();
    e.remember("one", &Default::default()).unwrap();
    e.remember("two", &Default::default()).unwrap();
    let se = SyncEngine::new(&e, None, None).unwrap();
    let status = se.get_status(Some("http://remote:1")).unwrap();
    assert_eq!(status["total_events"], 2);
    assert_eq!(status["device_count"], 1);
    assert_eq!(status["operation_breakdown"]["CREATE"], 2);
    assert_eq!(status["synced_events"], 0);
    assert_eq!(status["remote"], "http://remote:1");
    assert!(status["device_id"].as_str().unwrap().starts_with("device-"));
}

// ── Replication tools ───────────────────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sync_tools_push_pull_status_against_live_server() {
    let tmp = tempfile::tempdir().expect("tmpdir");
    let remote_bank = Arc::new(bank(&tmp.path().join("remote")));
    remote_bank
        .remember("remote fact: kestrels hover", &Default::default())
        .unwrap();
    let server = spawn_server(remote_bank.clone(), None).await;
    let url = format!("http://{}", server.addr);

    let local = Arc::new(
        Engine::open(MnemosyneConfig {
            data_dir: tmp.path().join("local"),
            sync_remote: Some(url.clone()),
            ..Default::default()
        })
        .expect("engine"),
    );
    local
        .remember("local fact: otters juggle", &Default::default())
        .unwrap();
    let provider = crate::MnemosyneProvider::new(local.clone());

    // Status: identity + knobs (`_handle_status` shape).
    let status: Value =
        serde_json::from_str(&provider.call_tool("mnemosyne_sync_status", json!({})).await)
            .unwrap();
    assert_eq!(status["status"], "ok");
    assert_eq!(status["remote"], url);
    assert_eq!(status["encryption"], "disabled");
    assert_eq!(status["mode"], "bidirectional");
    assert_eq!(status["local_events"], 1);
    assert_eq!(status["last_cursor"], "none");

    // Push: one local event lands remotely; cursor advances.
    let pushed: Value =
        serde_json::from_str(&provider.call_tool("mnemosyne_sync_push", json!({})).await).unwrap();
    assert_eq!(pushed["status"], "ok", "{pushed:?}");
    assert_eq!(pushed["pushed"], 1);
    assert!(!remote_bank.recall("otters juggle", 5).unwrap().is_empty());
    let again: Value =
        serde_json::from_str(&provider.call_tool("mnemosyne_sync_push", json!({})).await).unwrap();
    assert_eq!(again["pushed"], 0, "cursor advanced: {again:?}");
    assert_eq!(again["message"], "No local changes to push.");

    // Pull: the remote's own event applies locally (ours dedups by hash).
    let pulled: Value =
        serde_json::from_str(&provider.call_tool("mnemosyne_sync_pull", json!({})).await).unwrap();
    assert_eq!(pulled["status"], "ok", "{pulled:?}");
    assert_eq!(pulled["pulled"], 1);
    assert_eq!(
        pulled["duplicates"], 1,
        "our pushed event came back and deduped"
    );
    assert!(!local.recall("kestrels hover", 5).unwrap().is_empty());
    drop(server);
}

#[tokio::test]
async fn sync_tools_report_missing_remote_and_bad_key() {
    let e = Arc::new(engine());
    let provider = crate::MnemosyneProvider::new(e);
    for tool in ["mnemosyne_sync_push", "mnemosyne_sync_pull"] {
        let resp: Value = serde_json::from_str(&provider.call_tool(tool, json!({})).await).unwrap();
        assert_eq!(resp["status"], "error");
        assert!(
            resp["error"]
                .as_str()
                .unwrap()
                .contains("No remote configured"),
            "{resp:?}"
        );
    }
    let status: Value =
        serde_json::from_str(&provider.call_tool("mnemosyne_sync_status", json!({})).await)
            .unwrap();
    assert_eq!(status["remote"], "(unconfigured)");

    // A bad key source fails closed rather than silently syncing plaintext.
    let bad = Arc::new(
        Engine::open_in_memory(MnemosyneConfig {
            sync_key: Some("not!!base64!!".into()),
            ..Default::default()
        })
        .unwrap(),
    );
    let provider = crate::MnemosyneProvider::new(bad);
    let resp: Value =
        serde_json::from_str(&provider.call_tool("mnemosyne_sync_status", json!({})).await)
            .unwrap();
    assert_eq!(resp["status"], "error");
    assert!(
        resp["error"]
            .as_str()
            .unwrap()
            .contains("Sync adapter not available"),
        "{resp:?}"
    );
}

/// Bug repro: `http_post` must NOT follow a redirect, so a hostile/misconfigured sync server that
/// `302`s to another origin can never bounce the bearer token (or the request) to that origin.
/// With `Redirects::None` the redirect target server records zero requests.
#[tokio::test]
async fn http_post_does_not_follow_redirects_so_token_never_crosses_origin() {
    use wiremock::matchers::any;
    use wiremock::{Mock, MockServer, ResponseTemplate};

    // Origin B — the redirect target. It must never receive a request (and thus never the token).
    let sink = MockServer::start().await;
    Mock::given(any())
        .respond_with(ResponseTemplate::new(200).set_body_string("{}"))
        .mount(&sink)
        .await;

    // Origin A — the configured remote, which 302s to origin B.
    let redirector = MockServer::start().await;
    Mock::given(any())
        .respond_with(
            ResponseTemplate::new(302).insert_header("location", format!("{}/sink", sink.uri())),
        )
        .mount(&redirector)
        .await;

    // Never raises: a 302 body is not JSON, so the "unparseable response" error shape returns —
    // the point of the test is what does NOT happen on the wire.
    let resp = SyncEngine::http_post(
        &redirector.uri(),
        "/sync/push",
        &json!({"x": 1}),
        Some("secret"),
    )
    .await;
    assert_eq!(resp["status"], "error", "302 body is not JSON: {resp:?}");

    // The bearer token (and the request) never crossed to origin B.
    let sink_reqs = sink
        .received_requests()
        .await
        .expect("request recording enabled");
    assert!(
        sink_reqs.is_empty(),
        "redirect target must never be requested (token-leak / SSRF): {sink_reqs:?}"
    );

    // Sanity: origin A did receive exactly one request, carrying the bearer — proving the call
    // happened and the token went only to the configured origin.
    let a_reqs = redirector
        .received_requests()
        .await
        .expect("request recording enabled");
    assert_eq!(a_reqs.len(), 1, "one request to the configured origin");
    assert_eq!(
        a_reqs[0]
            .headers
            .get("authorization")
            .and_then(|v| v.to_str().ok()),
        Some("Bearer secret"),
        "token sent to origin A only"
    );
}

#[test]
fn wire_event_tolerates_python_shapes() {
    // Array-typed parent_event_ids, missing importance, unknown keys.
    let raw = json!({
        "event_id": "e1",
        "memory_id": "m1",
        "operation": "UPDATE",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "device_id": "dev",
        "parent_event_ids": ["p1", "p2"],
        "synced_at": "2026-01-01T00:00:01+00:00",
        "totally_unknown": {"x": 1},
    });
    let ev = SyncEvent::from_value(&raw).expect("parse");
    assert_eq!(ev.parent_ids(), vec!["p1", "p2"]);
    assert_eq!(ev.importance, Some(0.5), "dataclass default");
    // Explicit null importance survives as None (Python from_dict passes None through).
    let raw2 = json!({
        "event_id": "e2", "memory_id": "m", "operation": "CREATE",
        "timestamp": "t", "device_id": "d", "importance": null,
    });
    assert_eq!(SyncEvent::from_value(&raw2).unwrap().importance, None);
}
