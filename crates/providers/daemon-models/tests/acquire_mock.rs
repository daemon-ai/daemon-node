// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Acquisition-path tests against an in-process `wiremock` Hub (no live network): companion
//! (mmproj) plan expansion + cataloging, the size-integrity gate, the local pairing scan, and the
//! projector activate/resolve guards.

use std::path::PathBuf;
use std::time::Duration;

use daemon_common::{DownloadState, InstalledModel, ModelEngine, ModelRef, ModelSource};
use daemon_models::{ManagerConfig, ModelManager, Registry};
use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

/// Serve one repo file the way the Hub's `resolve` endpoint does for `hf-hub`: the `bytes=0-0`
/// metadata probe answers with `etag` + `x-repo-commit` + a `content-range` carrying the total
/// size; ranged chunk requests answer with the byte slice (clamped to the file length).
struct ResolveFile {
    bytes: Vec<u8>,
    commit: String,
}

impl Respond for ResolveFile {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let len = self.bytes.len();
        let range = request
            .headers
            .get("range")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("bytes="))
            .and_then(|v| {
                let (a, b) = v.split_once('-')?;
                Some((a.parse::<usize>().ok()?, b.parse::<usize>().ok()?))
            });
        match range {
            Some((0, 0)) => ResponseTemplate::new(206)
                .insert_header("etag", "\"blob-etag-1\"")
                .insert_header("x-repo-commit", self.commit.as_str())
                .insert_header("content-range", format!("bytes 0-0/{len}").as_str())
                .set_body_bytes(vec![self.bytes.first().copied().unwrap_or(0)]),
            Some((start, stop)) => {
                let stop = stop.min(len.saturating_sub(1));
                ResponseTemplate::new(206)
                    .insert_header(
                        "content-range",
                        format!("bytes {start}-{stop}/{len}").as_str(),
                    )
                    .set_body_bytes(self.bytes[start..=stop].to_vec())
            }
            None => ResponseTemplate::new(200).set_body_bytes(self.bytes.clone()),
        }
    }
}

/// A fake GGUF payload: the magic + deterministic filler up to `len` bytes.
fn gguf_bytes(len: usize) -> Vec<u8> {
    let mut bytes = b"GGUF".to_vec();
    bytes.resize(len, 0xAB);
    bytes
}

/// Mount a repo tree listing + per-file resolve responders on `server`.
async fn mount_repo(server: &MockServer, repo: &str, files: &[(&str, &[u8], u64)]) {
    let tree: Vec<_> = files
        .iter()
        .map(|(name, _, declared)| json!({"type": "file", "path": name, "size": declared}))
        .collect();
    Mock::given(method("GET"))
        .and(path(format!("/api/models/{repo}/tree/main")))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!(tree)))
        .mount(server)
        .await;
    for (name, bytes, _) in files {
        Mock::given(method("GET"))
            .and(path(format!("/{repo}/resolve/main/{name}")))
            .respond_with(ResolveFile {
                bytes: bytes.to_vec(),
                commit: "0123456789abcdef0123456789abcdef01234567".into(),
            })
            .mount(server)
            .await;
    }
}

/// Mount a single-file repo whose tree advertises a git-LFS `oid` (sha256), so the acquisition
/// path can verify the downloaded bytes against the Hub-declared hash (Phase 3 / Cluster E, L1).
async fn mount_repo_lfs(
    server: &MockServer,
    repo: &str,
    name: &str,
    bytes: &[u8],
    declared: u64,
    oid: &str,
) {
    let tree = json!([{
        "type": "file", "path": name, "size": declared,
        "lfs": { "oid": oid, "size": declared },
    }]);
    Mock::given(method("GET"))
        .and(path(format!("/api/models/{repo}/tree/main")))
        .respond_with(ResponseTemplate::new(200).set_body_json(tree))
        .mount(server)
        .await;
    Mock::given(method("GET"))
        .and(path(format!("/{repo}/resolve/main/{name}")))
        .respond_with(ResolveFile {
            bytes: bytes.to_vec(),
            commit: "0123456789abcdef0123456789abcdef01234567".into(),
        })
        .mount(server)
        .await;
}

/// Lowercase-hex sha256 of `bytes` (the pin/oid format).
fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(bytes);
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

/// The node-local pin sidecar path for an artifact (`<artifact>.sha256`).
fn sidecar_of(artifact: &std::path::Path) -> PathBuf {
    let mut p = artifact.as_os_str().to_os_string();
    p.push(".sha256");
    PathBuf::from(p)
}

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("daemon-models-acq-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

async fn manager_over(dir: &std::path::Path, endpoint: String) -> ModelManager {
    ModelManager::new(ManagerConfig {
        cache_dir: Some(dir.join("hub")),
        fallback_cache_dir: None,
        registry_path: Some(dir.join("catalog.json")),
        endpoint: Some(endpoint),
        quantize_worker_bin: None,
    })
    .await
    .expect("manager")
}

/// Await a terminal download state, panicking on timeout.
async fn await_terminal(manager: &ModelManager, id: daemon_common::DownloadId) -> DownloadState {
    for _ in 0..200 {
        let status = manager
            .downloads()
            .await
            .into_iter()
            .find(|s| s.id == id)
            .expect("job status");
        match status.state {
            DownloadState::Queued | DownloadState::Downloading => {
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
            terminal => return terminal,
        }
    }
    panic!("download did not settle");
}

/// Await the catalog record for `model` (the watcher catalogs asynchronously).
async fn await_record(manager: &ModelManager, model: &ModelRef) -> InstalledModel {
    for _ in 0..200 {
        if let Some(r) = manager
            .catalog()
            .await
            .into_iter()
            .find(|r| &r.model == model)
        {
            return r;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("record was not cataloged");
}

/// A text-GGUF download auto-pairs the repo's projector in the SAME job (files_total = 2), the
/// text record carries `mmproj_path`, and the companion gets no separate catalog record.
#[tokio::test]
async fn download_pairs_repo_mmproj_companion() {
    let server = MockServer::start().await;
    let text = gguf_bytes(300);
    let proj = gguf_bytes(120);
    mount_repo(
        &server,
        "org/smolvlm",
        &[
            ("SmolVLM-256M-Instruct-Q8_0.gguf", &text, 300),
            ("mmproj-SmolVLM-256M-Instruct-Q8_0.gguf", &proj, 120),
        ],
    )
    .await;

    let dir = temp_dir("pair");
    let manager = manager_over(&dir, server.uri()).await;
    let model = ModelRef::new(
        ModelEngine::Llama,
        ModelSource::hf_file("org/smolvlm", "SmolVLM-256M-Instruct-Q8_0.gguf"),
    );
    let id = manager.download(model.clone()).await.expect("download");
    assert_eq!(await_terminal(&manager, id).await, DownloadState::Completed);

    let status = manager
        .downloads()
        .await
        .into_iter()
        .find(|s| s.id == id)
        .unwrap();
    assert_eq!(status.files_total, 2, "companion rides the same job");
    assert_eq!(status.files_done, 2);

    let record = await_record(&manager, &model).await;
    let mmproj = record
        .mmproj_path
        .expect("companion cataloged on the text record");
    assert!(
        mmproj.contains("mmproj-SmolVLM-256M-Instruct-Q8_0.gguf"),
        "unexpected companion path: {mmproj}"
    );
    assert!(std::path::Path::new(&mmproj).exists(), "companion on disk");
    assert_eq!(
        manager.catalog().await.len(),
        1,
        "the companion is not a standalone record"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// An mmproj request downloads plainly: no companion lookup, and the record is a projector —
/// excluded from activation with the actionable error.
#[tokio::test]
async fn standalone_mmproj_download_stays_a_projector_record() {
    let server = MockServer::start().await;
    let proj = gguf_bytes(120);
    mount_repo(
        &server,
        "org/smolvlm",
        &[("mmproj-SmolVLM-256M-Instruct-Q8_0.gguf", &proj, 120)],
    )
    .await;

    let dir = temp_dir("standalone");
    let manager = manager_over(&dir, server.uri()).await;
    let model = ModelRef::new(
        ModelEngine::Llama,
        ModelSource::hf_file("org/smolvlm", "mmproj-SmolVLM-256M-Instruct-Q8_0.gguf"),
    );
    let id = manager.download(model.clone()).await.expect("download");
    assert_eq!(await_terminal(&manager, id).await, DownloadState::Completed);
    let status = manager
        .downloads()
        .await
        .into_iter()
        .find(|s| s.id == id)
        .unwrap();
    assert_eq!(status.files_total, 1, "no companion lookup for an mmproj");

    let record = await_record(&manager, &model).await;
    assert!(record.mmproj_path.is_none());

    // Activation is rejected with the actionable projector error.
    let err = manager
        .activate(&record.id, "default")
        .await
        .expect_err("projector must not activate");
    let msg = err.to_string();
    assert!(
        msg.contains("vision projector") && msg.contains("text weights"),
        "unexpected error: {msg}"
    );

    // Resolve of the projector reference is rejected too (the stale-profile path).
    let err = manager.resolve(&model).await.expect_err("resolve guard");
    assert!(err.to_string().contains("vision projector"));

    let _ = std::fs::remove_dir_all(&dir);
}

/// A truncated transfer (on-disk bytes != Hub-declared size) fails the job with a clear error
/// even though the GGUF magic is intact.
#[tokio::test]
async fn size_mismatch_fails_the_job() {
    let server = MockServer::start().await;
    let text = gguf_bytes(200); // served: 200 bytes
    mount_repo(
        &server,
        "org/truncated",
        &[("Model-Q8_0.gguf", &text, 260)], // declared: 260 bytes
    )
    .await;

    let dir = temp_dir("size");
    let manager = manager_over(&dir, server.uri()).await;
    let model = ModelRef::new(
        ModelEngine::Llama,
        ModelSource::hf_file("org/truncated", "Model-Q8_0.gguf"),
    );
    let id = manager.download(model.clone()).await.expect("download");
    assert_eq!(await_terminal(&manager, id).await, DownloadState::Failed);
    let status = manager
        .downloads()
        .await
        .into_iter()
        .find(|s| s.id == id)
        .unwrap();
    let err = status.error.expect("failure reason");
    assert!(err.contains("size mismatch"), "unexpected error: {err}");
    assert!(manager.catalog().await.is_empty(), "nothing cataloged");

    let _ = std::fs::remove_dir_all(&dir);
}

/// The post-download pairing scan links a PRE-EXISTING cataloged projector to a freshly
/// downloaded text model (quant-compatible, recall above the local threshold) — the repo listing
/// itself carries no projector here.
#[tokio::test]
async fn pairing_scan_links_preexisting_projector_record() {
    let server = MockServer::start().await;
    let text = gguf_bytes(300);
    mount_repo(
        &server,
        "org/smolvlm",
        &[("SmolVLM-256M-Instruct-Q8_0.gguf", &text, 300)],
    )
    .await;

    let dir = temp_dir("scan");
    // Seed a projector record whose artifact exists on disk (as the wizard's earlier standalone
    // mmproj download would have left).
    let proj_path = dir.join("mmproj-SmolVLM-256M-Instruct-Q8_0.gguf");
    std::fs::write(&proj_path, gguf_bytes(64)).unwrap();
    let proj_ref = ModelRef::new(
        ModelEngine::Llama,
        ModelSource::Local {
            path: proj_path.clone(),
        },
    );
    {
        let registry = Registry::open(dir.join("catalog.json")).await.unwrap();
        registry
            .upsert(InstalledModel {
                id: daemon_models::model_id(&proj_ref),
                model: proj_ref.clone(),
                display_name: "mmproj-SmolVLM-256M-Instruct-Q8_0.gguf".into(),
                local_path: proj_path.clone(),
                size_bytes: 64,
                quant: Some("Q8_0".into()),
                installed_at_ms: 1,
                arch: Some("clip".into()),
                context_length: None,
                file_type: None,
                mmproj_path: None,
            })
            .await
            .unwrap();
    }

    let manager = manager_over(&dir, server.uri()).await;
    let model = ModelRef::new(
        ModelEngine::Llama,
        ModelSource::hf_file("org/smolvlm", "SmolVLM-256M-Instruct-Q8_0.gguf"),
    );
    let id = manager.download(model.clone()).await.expect("download");
    assert_eq!(await_terminal(&manager, id).await, DownloadState::Completed);

    let record = await_record(&manager, &model).await;
    assert_eq!(
        record.mmproj_path.as_deref(),
        Some(proj_path.to_string_lossy().as_ref()),
        "the scan links the pre-existing projector"
    );
    // The projector record itself never gains a pairing.
    let proj_record = manager
        .catalog()
        .await
        .into_iter()
        .find(|r| r.model == proj_ref)
        .expect("projector record kept");
    assert!(proj_record.mmproj_path.is_none());

    let _ = std::fs::remove_dir_all(&dir);
}

/// L1 provenance: a download whose bytes do NOT hash to the Hub-declared git-LFS `oid` fails the
/// job (even though size + `GGUF` magic are intact) and is never cataloged.
#[tokio::test]
async fn download_oid_mismatch_fails_and_is_not_cataloged() {
    let server = MockServer::start().await;
    let bytes = gguf_bytes(200);
    let wrong_oid = "0".repeat(64); // does not match the served bytes
    mount_repo_lfs(
        &server,
        "org/oidbad",
        "Model-Q8_0.gguf",
        &bytes,
        200,
        &wrong_oid,
    )
    .await;

    let dir = temp_dir("oidbad");
    let manager = manager_over(&dir, server.uri()).await;
    let model = ModelRef::new(
        ModelEngine::Llama,
        ModelSource::hf_file("org/oidbad", "Model-Q8_0.gguf"),
    );
    let id = manager.download(model.clone()).await.expect("download");
    assert_eq!(await_terminal(&manager, id).await, DownloadState::Failed);
    let status = manager
        .downloads()
        .await
        .into_iter()
        .find(|s| s.id == id)
        .unwrap();
    let err = status.error.expect("failure reason");
    assert!(err.contains("sha256 mismatch"), "unexpected error: {err}");
    assert!(manager.catalog().await.is_empty(), "nothing cataloged");

    let _ = std::fs::remove_dir_all(&dir);
}

/// L1 provenance: a download whose bytes hash to the Hub-declared `oid` completes and records the
/// pin in a node-local `<artifact>.sha256` sidecar equal to the oid.
#[tokio::test]
async fn download_oid_match_records_pin() {
    let server = MockServer::start().await;
    let bytes = gguf_bytes(200);
    let oid = sha256_hex(&bytes);
    mount_repo_lfs(&server, "org/oidok", "Model-Q8_0.gguf", &bytes, 200, &oid).await;

    let dir = temp_dir("oidok");
    let manager = manager_over(&dir, server.uri()).await;
    let model = ModelRef::new(
        ModelEngine::Llama,
        ModelSource::hf_file("org/oidok", "Model-Q8_0.gguf"),
    );
    let id = manager.download(model.clone()).await.expect("download");
    assert_eq!(await_terminal(&manager, id).await, DownloadState::Completed);

    let record = await_record(&manager, &model).await;
    let sidecar = sidecar_of(&record.local_path);
    let pinned =
        std::fs::read_to_string(&sidecar).expect("pin sidecar written beside the artifact");
    assert_eq!(pinned.trim(), oid, "the pin equals the Hub-declared oid");

    let _ = std::fs::remove_dir_all(&dir);
}
