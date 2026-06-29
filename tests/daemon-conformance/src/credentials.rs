// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! THE PHASE-7 GATE: the credential authority (`daemon-workspace-layout.md` §7 phase-7 gate).
//! A placed child several cuts down runs under only a brokered, attenuated, short-lived
//! capability lease — no raw secret crosses the cut. The authority (owner) mints signed,
//! scoped, TTL-bounded `CapabilityLease`s; intermediate hosts re-broker upward, intersecting
//! scope at each hop (least privilege); the three modes (`Native`/`Bearer`/`Proxied`) trade
//! isolation against cost; a stale incarnation cannot acquire; an expired/edited capability is
//! refused; every lifecycle step is journaled into the phase-6 verifiable trace and verifies;
//! and a cost ceiling feeds back into `Budget`.

use daemon_common::{
    ContentHash, CredError, CredMode, CredScope, FenceToken, JournalStreamId, PartitionId,
    ProfileRef, SessionId, SnapshotBlob, UnitId,
};
use daemon_credentials::{
    CapabilitySigner, CredAuditKind, CredentialAuthority, StubCredentialSource,
};
use daemon_host::{
    serve_credentials, CredentialBroker, FenceGuard, JournalSink, OwnerBroker, RelayBroker,
    RemoteCredentialClient,
};
use daemon_provision::CutChannel;
use daemon_store::{InMemoryStore, SessionStore, TraceSegment};
use daemon_telemetry::{verify_segment, SegmentInput, TraceSigner, GENESIS_ROOT};
use std::sync::{Arc, Mutex};

const PARTITION: PartitionId = PartitionId::DEFAULT;

/// A connected pair of in-process cut channels (parent end, child end) over a duplex pipe — a
/// cut without spawning a process, so the broker chain is exercised over the real frame codec.
fn cut_pair() -> (CutChannel, CutChannel) {
    let (a, b) = tokio::io::duplex(1 << 16);
    let (ar, aw) = tokio::io::split(a);
    let (br, bw) = tokio::io::split(b);
    (
        CutChannel::from_parts(Box::new(ar), Box::new(aw)),
        CutChannel::from_parts(Box::new(br), Box::new(bw)),
    )
}

/// Build the 2-hop chain A -> B -> C over two real cuts. A is the owner (mints); B is a relay
/// granting at most `grant_b` (optionally fenced); C gets the descendant-side client. Returns
/// `(client_at_C, authority_A)`.
fn build_chain(
    mode: CredMode,
    grant_a: CredScope,
    grant_b: CredScope,
    fence_b: Option<FenceGuard>,
) -> (Arc<RemoteCredentialClient>, Arc<CredentialAuthority>) {
    let signer = Arc::new(CapabilitySigner::generate());
    let source = Arc::new(StubCredentialSource::minting("openai", "sk-configured"));
    let authority = Arc::new(CredentialAuthority::new(
        grant_a, mode, 60_000, signer, source,
    ));

    // Cut A<->B: A serves as the owner.
    let (a_parent, a_child) = cut_pair();
    let owner = Arc::new(OwnerBroker::new(authority.clone())) as Arc<dyn CredentialBroker>;
    tokio::spawn(serve_credentials(a_parent, owner));
    let client_to_a = RemoteCredentialClient::connect(a_child); // lives at B

    // Cut B<->C: B re-brokers upward as a relay (narrowing by grant_b).
    let mut relay = RelayBroker::new(client_to_a as Arc<dyn CredentialBroker>, grant_b);
    if let Some(f) = fence_b {
        relay = relay.with_fence(f);
    }
    let relay = Arc::new(relay) as Arc<dyn CredentialBroker>;
    let (b_parent, b_child) = cut_pair();
    tokio::spawn(serve_credentials(b_parent, relay));
    let client_to_b = RemoteCredentialClient::connect(b_child); // lives at C

    (client_to_b, authority)
}

fn unit_c() -> Option<UnitId> {
    Some(UnitId::new("unit-C"))
}

fn loaded_entries(seg: &TraceSegment) -> Vec<(u64, Vec<u8>, ContentHash)> {
    seg.entries
        .iter()
        .map(|e| (e.seq, e.bytes.clone(), e.content_hash))
        .collect()
}

/// (1) + (3): re-brokering composes across two cuts, and the effective scope is the
/// intersection along the whole path — a descendant can never exceed `grant_A ∩ grant_B`.
#[tokio::test]
async fn two_hop_chain_composes_and_attenuates() {
    let grant_a = CredScope::new(["openai"], ["chat", "embed"], Some(1_000));
    let grant_b = CredScope::new(["openai"], ["chat"], Some(500));
    let (c, _auth) = build_chain(CredMode::Native, grant_a, grant_b, None);

    // C asks for *more* than either hop grants: extra profile/actions, a bigger ceiling.
    let broad = CredScope::new(
        ["openai", "anthropic"],
        ["chat", "embed", "admin"],
        Some(10_000),
    );
    let lease = c
        .acquire(unit_c(), &ProfileRef::new("openai"), &broad)
        .await
        .expect("the chain mints a capability for C");

    // Effective scope = grant_A ∩ grant_B ∩ request.
    assert!(lease.scope.profiles.contains("openai"));
    assert!(!lease.scope.profiles.contains("anthropic"));
    assert!(lease.scope.actions.contains("chat"));
    assert!(
        !lease.scope.actions.contains("embed"),
        "embed is not in grant_B"
    );
    assert!(
        !lease.scope.actions.contains("admin"),
        "admin is in no grant"
    );
    assert_eq!(
        lease.scope.max_tokens,
        Some(500),
        "ceiling clamps to the tightest hop"
    );
    assert!(lease.secret.is_some(), "Native carries a short-lived token");

    // A request with no overlap is denied at the narrowing hop (never forwarded to the owner).
    let off_grant = CredScope::new(["ghost"], ["chat"], None);
    let err = c
        .acquire(unit_c(), &ProfileRef::new("openai"), &off_grant)
        .await
        .unwrap_err();
    assert_eq!(err, CredError::ScopeDenied);
}

/// (2 Proxied) + (1 Proxied): the lease C holds is a handle (no secret); resolution must
/// round-trip to the owner A, which returns only a *result* — the raw key never crosses to B/C.
#[tokio::test]
async fn proxied_use_round_trips_to_owner_without_leaking_key() {
    let grant = CredScope::new(["openai"], ["chat"], None);
    let (c, auth) = build_chain(CredMode::Proxied, grant.clone(), grant.clone(), None);

    let lease = c
        .acquire(unit_c(), &ProfileRef::new("openai"), &grant)
        .await
        .unwrap();
    assert!(lease.secret.is_none(), "Proxied hands C only a handle");

    let result = c.use_capability(unit_c(), &lease).await.unwrap();
    assert_ne!(
        result.expose(),
        "sk-configured",
        "the raw key must never reach C/B"
    );
    assert!(
        result.expose().starts_with("proxied-result:"),
        "owner returns a result"
    );

    // The owner recorded the use (the round-trip reached A).
    assert!(
        auth.audit_log()
            .iter()
            .any(|e| e.kind == CredAuditKind::Use),
        "the proxied use must be audited at the owner"
    );
}

/// (2 Bearer): a long-lived-key profile hands over a usable key; the compensating control is
/// the mandatory audit record. With a minting source the key is fresh per-grant.
#[tokio::test]
async fn bearer_hands_over_key_and_is_audited() {
    let grant = CredScope::new(["openai"], ["chat"], Some(1_000));
    let (c, auth) = build_chain(CredMode::Bearer, grant.clone(), grant.clone(), None);

    let lease = c
        .acquire(unit_c(), &ProfileRef::new("openai"), &grant)
        .await
        .unwrap();
    let key = lease
        .secret
        .as_ref()
        .expect("Bearer carries a usable key")
        .expose();
    assert!(
        key.starts_with("sk-fresh-"),
        "a minting source issues a fresh per-grant key"
    );

    let granted = auth
        .audit_log()
        .into_iter()
        .find(|e| e.kind == CredAuditKind::Grant)
        .expect("the issuance is audited");
    assert_eq!(
        granted.requester,
        unit_c(),
        "the audit answers *who* was issued the key"
    );
}

/// (4): a stale incarnation cannot acquire — the superseded hop (here the relay B) rejects with
/// `Fenced`, exactly as the dual-ownership store fence does across a cut.
#[tokio::test]
async fn stale_fence_acquire_is_rejected() {
    let grant = CredScope::new(["openai"], ["chat"], None);
    let live = Arc::new(Mutex::new(FenceToken(1)));
    let guard = FenceGuard::new(FenceToken(1), live.clone());
    let (c, _auth) = build_chain(CredMode::Native, grant.clone(), grant.clone(), Some(guard));

    // While B's incarnation is current, acquire succeeds.
    c.acquire(unit_c(), &ProfileRef::new("openai"), &grant)
        .await
        .expect("current incarnation acquires");

    // A newer activation supersedes B.
    *live.lock().unwrap() = FenceToken(2);
    let err = c
        .acquire(unit_c(), &ProfileRef::new("openai"), &grant)
        .await
        .unwrap_err();
    assert_eq!(
        err,
        CredError::Fenced,
        "the superseded hop must reject the acquire"
    );
}

/// (5): a capability whose signed fields were edited fails verification (signature), and a
/// zero-TTL capability fails verification (expiry).
#[tokio::test]
async fn edited_and_expired_capabilities_are_refused() {
    let grant = CredScope::new(["openai"], ["chat"], Some(1_000));
    let (c, auth) = build_chain(CredMode::Native, grant.clone(), grant.clone(), None);
    let mut lease = c
        .acquire(unit_c(), &ProfileRef::new("openai"), &grant)
        .await
        .unwrap();
    auth.verify(&lease)
        .expect("a freshly minted capability verifies");

    // Tamper with a signed field: verification fails.
    lease.scope.max_tokens = Some(999_999);
    assert_eq!(auth.verify(&lease).unwrap_err(), CredError::BadSignature);

    // A zero-TTL authority mints an already-expired capability.
    let signer = Arc::new(CapabilitySigner::generate());
    let source = Arc::new(StubCredentialSource::new("openai", "sk"));
    let auth0 = CredentialAuthority::new(grant.clone(), CredMode::Native, 0, signer, source);
    let ctx = daemon_credentials::AcquireCtx::default();
    let lease0 = auth0
        .acquire(&ctx, &ProfileRef::new("openai"), &grant)
        .unwrap();
    assert_eq!(auth0.verify(&lease0).unwrap_err(), CredError::Expired);
}

/// (6): the credential audit trail is journaled into the phase-6 verifiable trace and verifies
/// end-to-end — the sealed, signed segment is the tamper-evident answer to "who requested which
/// credential, when."
#[tokio::test]
async fn audit_trail_is_journaled_and_verifies() {
    let grant = CredScope::new(["openai"], ["chat"], Some(1_000));
    let (c, auth) = build_chain(CredMode::Native, grant.clone(), grant.clone(), None);
    c.acquire(unit_c(), &ProfileRef::new("openai"), &grant)
        .await
        .unwrap();

    // Journal A's credential audit log into a sealed, signed trace segment.
    let store = Arc::new(InMemoryStore::new());
    let id = SessionId::new("cred-audit");
    store
        .create_session(id.clone(), PARTITION, SnapshotBlob::default())
        .await
        .unwrap();
    let fence = store.acquire_activation_lease(&id).await.unwrap();
    let tsigner = Arc::new(TraceSigner::generate());
    let stream = JournalStreamId::session(&id);
    let sink = JournalSink::for_incarnation(
        store.clone() as Arc<dyn SessionStore>,
        tsigner.clone(),
        stream.clone(),
        fence,
        0,
    )
    .await;

    let events = auth.audit_log();
    assert!(events.iter().any(|e| e.kind == CredAuditKind::Request));
    assert!(events.iter().any(|e| e.kind == CredAuditKind::Grant));
    for ev in &events {
        sink.record_credential(ev).await.unwrap();
    }
    let root = sink.seal().await.unwrap();

    let seg = store.load_trace_segment(&stream, 0).await.unwrap();
    let committed = seg.committed.clone().expect("segment sealed");
    assert_eq!(committed.root, root);
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
        &tsigner.verifying_key(),
    )
    .expect("the sealed credential-audit segment verifies end to end");

    // The journaled detail carries the requester (the "who").
    let grant_ev = events
        .iter()
        .find(|e| e.kind == CredAuditKind::Grant)
        .unwrap();
    assert!(grant_ev.summary().contains("unit-C"));
}

/// (7): a fleet cost ceiling feeds back into `Budget` — under the ceiling there is headroom,
/// once reached the budget is throttled to zero (which a supervisor enforces as a cap).
#[tokio::test]
async fn cost_ceiling_feeds_budget() {
    let signer = Arc::new(CapabilitySigner::generate());
    let source = Arc::new(StubCredentialSource::new("openai", "sk"));
    let auth = CredentialAuthority::new(
        CredScope::new(["openai"], ["chat"], Some(1_000)),
        CredMode::Native,
        60_000,
        signer,
        source,
    )
    .with_cost_ceiling(100);

    assert_eq!(
        auth.charge(60).tokens,
        Some(40),
        "headroom remains under the ceiling"
    );
    assert_eq!(
        auth.charge(60).tokens,
        Some(0),
        "throttled once the ceiling is reached"
    );
    assert_eq!(auth.spent_tokens(), 120);
}
