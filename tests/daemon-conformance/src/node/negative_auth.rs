// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! THE AUTH 7 NEGATIVE (FAIL-CLOSED) SUITE: the cross-cutting invariant that the absence or
//! ambiguity of identity DENIES — encoded as explicit, over-the-wire tests on the integrated server.
//! Complements (does not duplicate) the in-process `access_control`/`ownership` suites and the
//! `authn`/`authz`/`tls` unit tests by proving the guarantees *through the real transports*:
//!
//! - unauthenticated `Call`/`Open` before `AuthOk` -> `Unauthenticated`, connection stays unelevated;
//! - a malformed/unknown-mechanism `AuthStart` -> `AuthError`, still unauthenticated;
//! - wrong password and unknown user are indistinguishable (no account oracle); a bogus token resume
//!   is refused;
//! - the TCP/TLS transport *requires* auth (pre-auth -> `Unauthenticated`; SCRAM unlocks);
//! - the local Unix socket binds the **explicit** `local_trust` system principal (a named principal,
//!   never admin-by-absence), and offers NO SASL mechanisms;
//! - the admin API denies a non-admin caller over the wire;
//! - login (ok/fail) and admin CRUD are audited with NO secret material, and the journal verifies.

use super::harness::*;
use super::wire_client::MuxConn;

use daemon_api::{ApiError, ApiRequest, ApiResponse, WireS2C};
use daemon_auth::{AuthStore, Role};
use daemon_common::{ContentHash, JournalStreamId, UnitId};
use daemon_host::{
    auth_audit::AUTH_JOURNAL_UNIT, build_server_config, serve_api_tls_tcp,
    serve_api_unix_authenticated, ApiTlsConfig, AuthAudit, Authenticator, SYSTEM_USERNAME,
};
use daemon_telemetry::{
    decode_entry, verify_segment, JournalPayload, SegmentInput, TraceSigner, GENESIS_ROOT,
};
use tokio::net::{TcpListener, TcpStream, UnixStream};

/// A node + authenticator over one shared identity store (admin `root` + user `alice`), an in-memory
/// audit store the test can read back, and the started resident services.
struct Fixture {
    node: Arc<NodeApiImpl>,
    auth: Arc<Authenticator>,
    audit_store: Arc<dyn SessionStore>,
    signer: Arc<TraceSigner>,
    handle: daemon_host::SupervisorHandle,
}

fn fixture() -> Fixture {
    let (node, handle) = assemble();
    let store = Arc::new(AuthStore::open_in_memory().expect("auth store"));
    store
        .create_user("root", "rootpw", &[Role::Admin])
        .expect("create admin");
    store
        .create_user("alice", "alicepw", &[Role::User])
        .expect("create alice");
    let audit_store: Arc<dyn SessionStore> = Arc::new(InMemoryStore::new());
    let signer = Arc::new(TraceSigner::generate());
    let audit = AuthAudit::shared(audit_store.clone(), signer.clone());
    let node = Arc::new(
        (*node)
            .clone()
            .with_auth_store(store.clone())
            .with_auth_audit(audit.clone()),
    );
    let auth = Arc::new(Authenticator::new(store.clone()).with_audit(audit));
    Fixture {
        node,
        auth,
        audit_store,
        signer,
        handle,
    }
}

async fn connect(path: &std::path::Path) -> MuxConn<UnixStream> {
    let stream = UnixStream::connect(path).await.expect("connect socket");
    MuxConn::handshake(stream).await.expect("hello handshake")
}

/// Pre-auth `Call`/`Open` are refused; a malformed `AuthStart`, wrong password, unknown user, and a
/// bogus token are all rejected without ever elevating the connection — then a correct SCRAM unlocks.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fail_closed_until_a_valid_credential() {
    let f = fixture();
    let path = temp_socket();
    let listener = UnixListener::bind(&path).expect("bind socket");
    let server = tokio::spawn(serve_api_unix_authenticated(
        listener,
        f.node.clone(),
        f.auth.clone(),
    ));

    // Pre-auth Call -> Unauthenticated.
    let mut c = connect(&path).await;
    assert!(
        matches!(
            c.call(ApiRequest::Health).await.expect("pre-auth call"),
            ApiResponse::Error(ApiError::Unauthenticated(_))
        ),
        "a pre-auth Call must be Unauthenticated"
    );
    // Pre-auth Open (a streaming Subscribe) -> End { Unauthenticated }.
    let id = c
        .open(ApiRequest::Subscribe {
            session: SessionId::new("nope"),
            after_seq: 0,
            max: 1,
        })
        .await
        .expect("open pre-auth");
    match c.next().await.expect("open reply") {
        WireS2C::End { id: rid, error } => {
            assert_eq!(rid, id);
            assert!(matches!(error, Some(ApiError::Unauthenticated(_))));
        }
        other => panic!("expected End(Unauthenticated), got {other:?}"),
    }

    // An unknown mechanism -> AuthError; the connection stays unauthenticated.
    match c
        .auth_start("GSSAPI", Vec::new())
        .await
        .expect("auth start")
    {
        WireS2C::AuthError { .. } => {}
        other => panic!("unknown mechanism must be AuthError, got {other:?}"),
    }
    assert!(
        matches!(
            c.call(ApiRequest::Health)
                .await
                .expect("call after bad mech"),
            ApiResponse::Error(ApiError::Unauthenticated(_))
        ),
        "the connection must stay unelevated after a rejected mechanism"
    );

    // Wrong password and unknown user are indistinguishable (same coarse AuthError, no oracle).
    let wrong = connect(&path)
        .await
        .authenticate_scram("alice", "wrongpw")
        .await;
    let ghost = connect(&path)
        .await
        .authenticate_scram("ghost", "wrongpw")
        .await;
    let (wrong_reason, ghost_reason) = match (wrong, ghost) {
        (Err(ApiError::Unauthenticated(a)), Err(ApiError::Unauthenticated(b))) => (a, b),
        other => panic!("wrong-pw and unknown-user must both be Unauthenticated, got {other:?}"),
    };
    assert_eq!(
        wrong_reason, ghost_reason,
        "wrong password and unknown user must report the SAME reason (no account oracle)"
    );

    // A bogus token resume is refused, and that connection stays unauthenticated.
    let mut r = connect(&path).await;
    assert!(matches!(
        r.authenticate_resume("deadbeef-not-a-real-token").await,
        Err(ApiError::Unauthenticated(_))
    ));
    assert!(
        matches!(
            r.call(ApiRequest::Health)
                .await
                .expect("call after bad token"),
            ApiResponse::Error(ApiError::Unauthenticated(_))
        ),
        "a failed resume must leave the connection unelevated"
    );

    // Sanity: a correct credential DOES unlock (the socket is not just always-denying).
    let mut ok = connect(&path).await;
    ok.authenticate_scram("root", "rootpw")
        .await
        .expect("valid scram unlocks");
    assert!(
        !matches!(
            ok.call(ApiRequest::Health).await.expect("post-auth call"),
            ApiResponse::Error(_)
        ),
        "a Call after AuthOk must succeed"
    );

    server.abort();
    f.handle.shutdown().await;
}

/// The admin AccessControl surface denies a non-admin caller over the real transport: an
/// authenticated `User` is `Forbidden` on every admin op (the capability gate runs on the wire, not
/// just in-process).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn admin_api_denies_a_non_admin_over_the_wire() {
    let f = fixture();
    let path = temp_socket();
    let listener = UnixListener::bind(&path).expect("bind socket");
    let server = tokio::spawn(serve_api_unix_authenticated(
        listener,
        f.node.clone(),
        f.auth.clone(),
    ));

    let mut alice = connect(&path).await;
    alice
        .authenticate_scram("alice", "alicepw")
        .await
        .expect("alice scram");

    for (label, req) in [
        ("UserList", ApiRequest::UserList),
        (
            "UserCreate",
            ApiRequest::UserCreate {
                username: "mallory".into(),
                password: "x".into(),
                roles: vec!["admin".into()],
            },
        ),
        ("RoleList", ApiRequest::RoleList),
        (
            "SessionRevoke",
            ApiRequest::SessionRevoke {
                user_id: "whoever".into(),
            },
        ),
    ] {
        assert!(
            matches!(
                alice.call(req).await.expect("admin op call"),
                ApiResponse::Error(ApiError::Forbidden(_))
            ),
            "a User must be Forbidden on admin op {label}"
        );
    }

    // WhoAmI, by contrast, is allowed for any authenticated principal.
    match alice.call(ApiRequest::WhoAmI).await.expect("whoami") {
        ApiResponse::WhoAmI(view) => assert_eq!(view.username, "alice"),
        other => panic!("expected WhoAmI, got {other:?}"),
    }

    server.abort();
    f.handle.shutdown().await;
}

/// The local Unix socket (`local_trust`) binds the EXPLICIT system principal — a deliberate, named
/// principal, never admin-by-absence — and advertises NO SASL mechanisms.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn local_trust_binds_the_explicit_system_principal() {
    let (node, handle) = assemble();
    let path = temp_socket();
    let listener = UnixListener::bind(&path).expect("bind socket");
    // `serve_api_unix` is the local-trust entry point (binds `RequestContext::system`).
    let server = tokio::spawn(serve_api_unix(listener, node.clone()));

    let mut c = connect(&path).await;
    // Local trust offers no mechanisms: the deliberate "no SASL on this transport" signal.
    assert!(
        c.mechanisms.is_empty(),
        "local trust must advertise no auth mechanisms, got {:?}",
        c.mechanisms
    );
    match c.call(ApiRequest::WhoAmI).await.expect("whoami") {
        ApiResponse::WhoAmI(view) => {
            assert_eq!(
                view.username, SYSTEM_USERNAME,
                "the local-trust principal must be the explicit system principal"
            );
            assert!(
                view.capabilities.contains(&"access_admin".to_string()),
                "the system principal is a deliberate full-trust admin (by configuration, not by \
                 absence of identity)"
            );
        }
        other => panic!("expected WhoAmI, got {other:?}"),
    }

    server.abort();
    handle.shutdown().await;
}

/// Recompute + signature-verify one sealed `node-auth` segment, chaining onto the prior root (the
/// same check the production history reader performs).
async fn audit_segment_verifies(
    store: &dyn SessionStore,
    signer: &TraceSigner,
    segment: u64,
) -> bool {
    let s = JournalStreamId::unit(&UnitId::new(AUTH_JOURNAL_UNIT));
    let Some(seg) = store.load_trace_segment(&s, segment).await else {
        return false;
    };
    let Some(committed) = seg.committed else {
        return false;
    };
    let prior = if segment == 0 {
        GENESIS_ROOT
    } else {
        match store
            .load_trace_segment(&s, segment - 1)
            .await
            .and_then(|p| p.committed)
        {
            Some(c) => c.root,
            None => return false,
        }
    };
    let entries: Vec<(u64, Vec<u8>, ContentHash)> = seg
        .entries
        .into_iter()
        .map(|e| (e.seq, e.bytes, e.content_hash))
        .collect();
    let input = SegmentInput {
        stream: &s,
        segment,
        prior,
        entries: &entries,
    };
    verify_segment(
        &input,
        &committed.root,
        &committed.signature,
        &signer.verifying_key(),
    )
    .is_ok()
}

/// Login (success + failure) and admin CRUD driven over the real socket are recorded onto the
/// verifiable `node-auth` journal, with NO credential material in any payload, and the chain verifies.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn wire_auth_events_are_audited_without_secrets_and_journal_verifies() {
    let f = fixture();
    let path = temp_socket();
    let listener = UnixListener::bind(&path).expect("bind socket");
    let server = tokio::spawn(serve_api_unix_authenticated(
        listener,
        f.node.clone(),
        f.auth.clone(),
    ));

    // A successful admin login (-> auth.login_ok) + an admin UserCreate (-> auth.user_created).
    let mut admin = connect(&path).await;
    admin
        .authenticate_scram("root", "rootpw")
        .await
        .expect("admin scram");
    match admin
        .call(ApiRequest::UserCreate {
            username: "bob".into(),
            password: "super-secret-bobpw".into(),
            roles: vec!["user".into()],
        })
        .await
        .expect("user create")
    {
        ApiResponse::AccessUser(u) => assert_eq!(u.username, "bob"),
        other => panic!("expected AccessUser, got {other:?}"),
    }
    // A failed login on a fresh connection (-> auth.login_fail). The password must never be recorded.
    let _ = connect(&path)
        .await
        .authenticate_scram("alice", "totally-wrong-pw")
        .await;

    // The audit hooks are awaited best-effort off the dispatch path (login_ok rides the writer queue);
    // poll the chain until the three expected kinds have landed.
    let stream = JournalStreamId::unit(&UnitId::new(AUTH_JOURNAL_UNIT));
    let wanted = ["auth.login_ok", "auth.login_fail", "auth.user_created"];
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let page = f.audit_store.load_journal(&stream, 0, 100).await;
        let kinds: Vec<String> = page
            .entries
            .iter()
            .filter_map(|je| decode_entry(&je.entry.bytes).ok())
            .map(|v| v.kind)
            .collect();
        if wanted.iter().all(|w| kinds.iter().any(|k| k == w)) {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "expected audit kinds did not all land; saw {kinds:?}"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    // No secret material anywhere in the payloads, and every sealed segment verifies.
    let page = f.audit_store.load_journal(&stream, 0, 100).await;
    let secrets = ["rootpw", "super-secret-bobpw", "totally-wrong-pw"];
    let markers = ["$argon2", "$scram", "stored_key", "server_key"];
    for je in &page.entries {
        let view = decode_entry(&je.entry.bytes).expect("decode audit entry");
        if let JournalPayload::Management { detail } = view.payload {
            for s in secrets.iter().chain(markers.iter()) {
                assert!(
                    !detail.contains(s),
                    "audit payload must carry no credential material, found {s:?} in: {detail}"
                );
            }
        }
    }
    let n = page.entries.len() as u64;
    assert!(n >= 3, "expected at least 3 audited events, got {n}");
    for seg in 0..n {
        assert!(
            audit_segment_verifies(f.audit_store.as_ref(), &f.signer, seg).await,
            "node-auth segment {seg} must verify (tamper-evident chain)"
        );
    }

    server.abort();
    f.handle.shutdown().await;
}

/// A throwaway server PKI: a CA and a CA-signed server certificate (SAN `localhost`).
fn server_pki() -> (
    tokio_rustls::rustls::pki_types::CertificateDer<'static>,
    String,
    String,
) {
    use rcgen::{BasicConstraints, CertificateParams, IsCa, KeyPair};
    let ca_key = KeyPair::generate().unwrap();
    let mut ca_params = CertificateParams::new(Vec::<String>::new()).unwrap();
    ca_params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
    let ca_cert = ca_params.self_signed(&ca_key).unwrap();

    let server_key = KeyPair::generate().unwrap();
    let server_cert = CertificateParams::new(vec!["localhost".to_string()])
        .unwrap()
        .signed_by(&server_key, &ca_cert, &ca_key)
        .unwrap();
    (
        ca_cert.der().clone(),
        server_cert.pem(),
        server_key.serialize_pem(),
    )
}

/// The TCP/TLS transport ALWAYS requires authentication: a real `rustls` client completes the
/// handshake, sees SCRAM/PLAIN/EXTERNAL advertised, is `Unauthenticated` pre-auth, and unlocks
/// dispatch only after a full SCRAM exchange — proving auth is enforced on the networked transport.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn tls_tcp_requires_auth_and_scram_unlocks() {
    use daemon_host::MECH_PLAIN;
    use tokio_rustls::rustls::pki_types::ServerName;
    use tokio_rustls::rustls::{ClientConfig, RootCertStore};
    use tokio_rustls::TlsConnector;

    let f = fixture();

    // Build the real server TLS config from on-disk PEM (the production `build_server_config` path).
    let (ca_der, server_pem, key_pem) = server_pki();
    let dir = tempfile::tempdir().unwrap();
    let cert_path = dir.path().join("server.pem");
    let key_path = dir.path().join("server.key");
    std::fs::write(&cert_path, server_pem).unwrap();
    std::fs::write(&key_path, key_pem).unwrap();
    let tls = build_server_config(&ApiTlsConfig {
        cert_path,
        key_path,
        require_client_cert: false,
        client_ca_path: None,
    })
    .expect("server tls config");

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(serve_api_tls_tcp(
        listener,
        tls,
        f.node.clone(),
        f.auth.clone(),
        daemon_common::IngressGovernor::secure_default(),
    ));

    // A client that trusts the CA.
    let mut roots = RootCertStore::empty();
    roots.add(ca_der).unwrap();
    let provider = Arc::new(tokio_rustls::rustls::crypto::aws_lc_rs::default_provider());
    let client_cfg = ClientConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()
        .unwrap()
        .with_root_certificates(roots)
        .with_no_client_auth();
    let connector = TlsConnector::from(Arc::new(client_cfg));
    let tcp = TcpStream::connect(addr).await.unwrap();
    let name = ServerName::try_from("localhost").unwrap();
    let tls_stream = connector.connect(name, tcp).await.expect("tls handshake");

    let mut c = MuxConn::handshake(tls_stream)
        .await
        .expect("hello over tls");
    // Over TLS the server advertises SCRAM + the TLS-only mechanisms.
    assert!(c
        .mechanisms
        .iter()
        .any(|m| m == daemon_host::MECH_SCRAM_SHA_256));
    assert!(
        c.mechanisms.iter().any(|m| m == MECH_PLAIN),
        "PLAIN must be advertised over TLS, got {:?}",
        c.mechanisms
    );

    // Pre-auth over TLS -> Unauthenticated.
    assert!(
        matches!(
            c.call(ApiRequest::Health).await.expect("pre-auth tls call"),
            ApiResponse::Error(ApiError::Unauthenticated(_))
        ),
        "the TCP/TLS transport must require authentication"
    );

    // SCRAM over TLS unlocks dispatch.
    let (view, _token) = c
        .authenticate_scram("root", "rootpw")
        .await
        .expect("scram over tls");
    assert_eq!(view.username, "root");
    assert!(
        !matches!(
            c.call(ApiRequest::Health)
                .await
                .expect("post-auth tls call"),
            ApiResponse::Error(_)
        ),
        "a Call after AuthOk over TLS must succeed"
    );

    server.abort();
    f.handle.shutdown().await;
}
