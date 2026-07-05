// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

use super::harness::*;

/// The filesystem / workspace surface (daemon-fs-surface-spec.md) end to end through a fully
/// assembled node: a configured `workspace_root` binds the `fs_*` ops to a real directory, the
/// node advertises its roots, write/read round-trips in the workspace root, the sensitive-path
/// gate blocks a dotenv write unless forced, and a containment escape is rejected.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn filesystem_surface_round_trips_and_gates() {
    use daemon_api::{ControlApi, FsRootId};

    let ws = std::env::temp_dir().join(format!("daemon-fs-it-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&ws);
    std::fs::create_dir_all(&ws).unwrap();

    let AssembledNode { node, handle, .. } = assemble_node(NodeAssembly {
        store: Arc::new(InMemoryStore::new()),
        partition: PARTITION,
        host_config: fast_host_config(),
        providers: gate_providers(),
        credentials: None,
        profile: ProfileRef::new("openai"),
        engine_config: daemon_core::Config::default(),
        journal_seed: Some([0x45; 32]),
        nesting_depth: 0,
        context: None,
        context_builder: None,
        memory: Vec::new(),
        memory_builder: None,
        extra_tools: Vec::new(),
        models: None,
        profiles: None,
        provider_resolver: None,
        credential_store: None,
        cloud_catalog: None,
        prompt_sources: vec![],
        revisions: None,
        skills: None,
        skills_resolver: None,
        routing: None,
        checkpoints: None,
        auth_factories: vec![],
        workspace_root: Some(ws.clone()),
        blob_root: None,
        fs: Default::default(),
    });

    // The node advertises at least the writable workspace root.
    let roots = node.fs_roots().await;
    assert!(
        roots.iter().any(|r| matches!(r.id, FsRootId::Workspace)),
        "fs_roots should advertise the workspace root, got {roots:?}"
    );

    // Write + read round-trips in the workspace root.
    let rev = node
        .fs_write(daemon_api::FsWriteArgs {
            root: FsRootId::Workspace,
            path: "notes/hello.txt".into(),
            bytes: b"hi".to_vec(),
            base_revision: None,
            force: false,
        })
        .await
        .expect("write");
    assert_eq!(rev.size, 2);
    let content = node
        .fs_read(FsRootId::Workspace, "notes/hello.txt".into(), 0)
        .await
        .expect("read");
    assert_eq!(content.bytes, b"hi");
    // The bytes are on disk under the configured workspace root (the same dir an agent's tools
    // would operate in).
    assert_eq!(std::fs::read(ws.join("notes/hello.txt")).unwrap(), b"hi");

    let listing = node
        .fs_list(FsRootId::Workspace, "notes".into(), false, None)
        .await
        .expect("list");
    assert!(listing.items.iter().any(|e| e.name == "hello.txt"));
    assert!(listing.next.is_none(), "one small dir fits one page");

    // The sensitive-path gate blocks a dotenv write unless forced.
    let blocked = node
        .fs_write(daemon_api::FsWriteArgs {
            root: FsRootId::Workspace,
            path: ".env".into(),
            bytes: b"SECRET=1".to_vec(),
            base_revision: None,
            force: false,
        })
        .await;
    assert!(blocked.is_err(), "a .env write should be gated");
    let forced = node
        .fs_write(daemon_api::FsWriteArgs {
            root: FsRootId::Workspace,
            path: ".env".into(),
            bytes: b"SECRET=1".to_vec(),
            base_revision: None,
            force: true,
        })
        .await;
    assert!(forced.is_ok(), "force overrides the sensitive-path gate");

    // Containment: a path escaping the root is rejected.
    assert!(node
        .fs_read(FsRootId::Workspace, "../escape".into(), 0)
        .await
        .is_err());

    handle.shutdown().await;
    let _ = std::fs::remove_dir_all(&ws);
}

/// Wire pagination (v24) end to end over the mux socket: `fs_list` on a >64-entry directory pages
/// at the WIRE_PAGE_MAX bound, the `next` cursor chains the pages through real dispatch/CBOR, and
/// the reassembled listing is complete, ordered, and duplicate-free.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fs_list_pages_over_the_wire() {
    use daemon_api::{FsRootId, WIRE_PAGE_MAX};
    use daemon_host::{serve_api_unix, MuxApiClient};
    use tokio::net::UnixListener;

    let ws = std::env::temp_dir().join(format!("daemon-fs-page-it-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&ws);
    std::fs::create_dir_all(ws.join("big")).unwrap();
    for n in 0..70 {
        std::fs::write(ws.join("big").join(format!("f{n:03}.txt")), b"x").unwrap();
    }

    let AssembledNode { node, handle, .. } = assemble_node(NodeAssembly {
        store: Arc::new(InMemoryStore::new()),
        partition: PARTITION,
        host_config: fast_host_config(),
        providers: gate_providers(),
        credentials: None,
        profile: ProfileRef::new("openai"),
        engine_config: daemon_core::Config::default(),
        journal_seed: Some([0x48; 32]),
        nesting_depth: 0,
        context: None,
        context_builder: None,
        memory: Vec::new(),
        memory_builder: None,
        extra_tools: Vec::new(),
        models: None,
        profiles: None,
        provider_resolver: None,
        credential_store: None,
        cloud_catalog: None,
        prompt_sources: vec![],
        revisions: None,
        skills: None,
        skills_resolver: None,
        routing: None,
        checkpoints: None,
        auth_factories: vec![],
        workspace_root: Some(ws.clone()),
        blob_root: None,
        fs: Default::default(),
    });
    let path = temp_socket();
    let _ = std::fs::remove_file(&path);
    let listener = UnixListener::bind(&path).expect("bind api socket");
    let server = tokio::spawn(serve_api_unix(listener, node.clone()));

    let mut mux = MuxApiClient::connect(path.clone())
        .await
        .expect("mux connect + hello");
    let mut all: Vec<String> = Vec::new();
    let mut sizes = Vec::new();
    let mut after: Option<String> = None;
    loop {
        let page = match mux
            .call(ApiRequest::FsList {
                root: FsRootId::Workspace,
                dir: "big".into(),
                show_ignored: false,
                after: after.clone(),
            })
            .await
            .expect("fs_list call")
        {
            ApiResponse::FsList(page) => page,
            other => panic!("expected FsList, got {other:?}"),
        };
        assert!(
            page.items.len() <= WIRE_PAGE_MAX,
            "a wire page must never exceed WIRE_PAGE_MAX, got {}",
            page.items.len()
        );
        sizes.push(page.items.len());
        all.extend(page.items.iter().map(|e| e.name.clone()));
        match page.next {
            Some(next) => after = Some(next),
            None => break,
        }
        assert!(sizes.len() <= 4, "runaway pagination");
    }

    assert_eq!(sizes, vec![64, 6]);
    let expected: Vec<String> = (0..70).map(|n| format!("f{n:03}.txt")).collect();
    assert_eq!(all, expected, "complete, ordered, duplicate-free listing");

    handle.shutdown().await;
    server.abort();
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir_all(&ws);
}

/// The content store (blob CAS, daemon-content-transfer-spec.md Phase 1) end to end through a
/// fully assembled node: blob_put -> blob_get round-trips, identical content dedupes to one
/// BlobRef, fs_read attaches a matching blob_ref, fs_write_from_blob materializes the blob into
/// the workspace, and a tampered store file fails the integrity check.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn content_store_round_trips_and_materializes() {
    use daemon_api::{ControlApi, FsRootId};

    let ws = std::env::temp_dir().join(format!("daemon-blob-it-ws-{}", std::process::id()));
    let blobs = std::env::temp_dir().join(format!("daemon-blob-it-cas-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&ws);
    let _ = std::fs::remove_dir_all(&blobs);
    std::fs::create_dir_all(&ws).unwrap();

    let AssembledNode { node, handle, .. } = assemble_node(NodeAssembly {
        store: Arc::new(InMemoryStore::new()),
        partition: PARTITION,
        host_config: fast_host_config(),
        providers: gate_providers(),
        credentials: None,
        profile: ProfileRef::new("openai"),
        engine_config: daemon_core::Config::default(),
        journal_seed: Some([0x46; 32]),
        nesting_depth: 0,
        context: None,
        context_builder: None,
        memory: Vec::new(),
        memory_builder: None,
        extra_tools: Vec::new(),
        models: None,
        profiles: None,
        provider_resolver: None,
        credential_store: None,
        cloud_catalog: None,
        prompt_sources: vec![],
        revisions: None,
        skills: None,
        skills_resolver: None,
        routing: None,
        checkpoints: None,
        auth_factories: vec![],
        workspace_root: Some(ws.clone()),
        blob_root: Some(blobs.clone()),
        fs: Default::default(),
    });

    // put -> get round-trip.
    let r = node
        .blob_put(b"content-addressed".to_vec())
        .await
        .expect("put");
    assert_eq!(r.size, 17);
    assert_eq!(
        node.blob_get(r.hash, None).await.expect("get"),
        b"content-addressed"
    );
    assert!(node.blob_stat(r.hash).await.present);

    // Dedup: identical bytes -> identical ref.
    let r2 = node.blob_put(b"content-addressed".to_vec()).await.unwrap();
    assert_eq!(r.hash, r2.hash);

    // fs_read attaches a matching blob_ref for an untruncated read.
    node.fs_write(daemon_api::FsWriteArgs {
        root: FsRootId::Workspace,
        path: "doc.txt".into(),
        bytes: b"hi there".to_vec(),
        base_revision: None,
        force: false,
    })
    .await
    .unwrap();
    let read = node
        .fs_read(FsRootId::Workspace, "doc.txt".into(), 0)
        .await
        .unwrap();
    let read_ref = read.blob_ref.expect("blob_ref attached");
    assert_eq!(read_ref.size, 8);
    // The attached ref resolves to the same bytes via the content store.
    assert_eq!(
        node.blob_get(read_ref.hash, None).await.unwrap(),
        b"hi there"
    );

    // fs_write_from_blob materializes a blob into the workspace in place.
    node.fs_write_from_blob(daemon_api::FsWriteFromBlobArgs {
        root: FsRootId::Workspace,
        path: "from_blob.txt".into(),
        hash: r.hash,
        base_revision: None,
        force: false,
    })
    .await
    .expect("materialize");
    assert_eq!(
        std::fs::read(ws.join("from_blob.txt")).unwrap(),
        b"content-addressed"
    );

    // Integrity: tampering with the on-disk blob fails a full get.
    let path = blobs.join(format!("{}.bin", r.hash.to_hex()));
    std::fs::write(&path, b"tampered").unwrap();
    assert!(node.blob_get(r.hash, None).await.is_err());

    handle.shutdown().await;
    let _ = std::fs::remove_dir_all(&ws);
    let _ = std::fs::remove_dir_all(&blobs);
}

/// Inbound message attachments (daemon-content-transfer-spec.md Phase 2b) end to end through a
/// fully assembled node: a client `blob_put`s file bytes, then submits a `StartTurn` carrying the
/// `BlobRef`; the node materializes it into the session's `inbox/` before the turn, where the
/// agent's filesystem surface (and tools) can read it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn inbound_attachment_materializes_into_session_inbox() {
    use daemon_api::{ControlApi, FsRootId, SessionApi};
    use daemon_common::{BlobRef, ReqId};
    use daemon_protocol::{AgentCommand, UserMsg};

    let ws = std::env::temp_dir().join(format!("daemon-attach-it-ws-{}", std::process::id()));
    let blobs = std::env::temp_dir().join(format!("daemon-attach-it-cas-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&ws);
    let _ = std::fs::remove_dir_all(&blobs);
    std::fs::create_dir_all(&ws).unwrap();

    let AssembledNode { node, handle, .. } = assemble_node(NodeAssembly {
        store: Arc::new(InMemoryStore::new()),
        partition: PARTITION,
        host_config: fast_host_config(),
        providers: gate_providers(),
        credentials: None,
        profile: ProfileRef::new("openai"),
        engine_config: daemon_core::Config::default(),
        journal_seed: Some([0x47; 32]),
        nesting_depth: 0,
        context: None,
        context_builder: None,
        memory: Vec::new(),
        memory_builder: None,
        extra_tools: Vec::new(),
        models: None,
        profiles: None,
        provider_resolver: None,
        credential_store: None,
        cloud_catalog: None,
        prompt_sources: vec![],
        revisions: None,
        skills: None,
        skills_resolver: None,
        routing: None,
        checkpoints: None,
        auth_factories: vec![],
        workspace_root: Some(ws.clone()),
        blob_root: Some(blobs.clone()),
        fs: Default::default(),
    });

    // The client stages the attachment in the content store, then names it on the turn.
    let r = node
        .blob_put(b"attached payload".to_vec())
        .await
        .expect("put");
    let att = BlobRef::new(r.hash, r.size).with_name("hello.txt");
    let session = SessionId::new("attach-session");
    node.submit(
        session.clone(),
        AgentCommand::StartTurn {
            input: UserMsg::new("see attached").with_attachments(vec![att]),
            request_id: ReqId(1),
        },
    )
    .await
    .expect("submit");

    // The node materialized the blob into the session's inbox/ (visible via the fs surface, and
    // on disk where the agent's tools operate).
    let read = node
        .fs_read(
            FsRootId::Session(session.clone()),
            "inbox/hello.txt".into(),
            0,
        )
        .await
        .expect("read materialized attachment");
    assert_eq!(read.bytes, b"attached payload");

    handle.shutdown().await;
    let _ = std::fs::remove_dir_all(&ws);
    let _ = std::fs::remove_dir_all(&blobs);
}
