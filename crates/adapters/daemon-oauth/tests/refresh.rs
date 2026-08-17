// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Hermetic refresh-mechanics fixture (credential plan Phase 4): a wiremock token endpoint plays
//! the vendor, so expiry → refresh → rotation → failure classification is proven without any live
//! account. No currently supported provider exercises refresh live (OpenRouter mints a static
//! key; Hugging Face needs an operator-registered client id), so THIS fixture is the gate.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use wiremock::matchers::{body_string_contains, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use daemon_common::{CredentialEnvelope, OAuthTokenSet};
use daemon_host::{CredentialRefresher, CredentialStore, MemCredentialStore};
use daemon_oauth::{RefreshEndpoint, TokenSetRefresher};

const REF: &str = "provider/huggingface";
const METHOD: &str = "provider/huggingface";

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

/// A token set expiring `in_secs` from now (relative so the fixture never goes stale).
fn token_set(in_secs: i64, refresh_token: Option<&str>) -> OAuthTokenSet {
    OAuthTokenSet {
        provider_id: "huggingface".into(),
        method_id: METHOD.into(),
        access_token: "at-old".into(),
        refresh_token: refresh_token.map(str::to_string),
        expires_at: Some(unix_now().saturating_add_signed(in_secs)),
        token_endpoint: None,
        client_id: None,
    }
}

fn seed(store: &Arc<dyn CredentialStore>, ts: OAuthTokenSet) {
    store
        .set(REF, &CredentialEnvelope::OAuthTokenSet(ts).encode())
        .unwrap();
}

fn stored_token_set(store: &Arc<dyn CredentialStore>) -> OAuthTokenSet {
    match CredentialEnvelope::parse(&store.get(REF).unwrap()).unwrap() {
        CredentialEnvelope::OAuthTokenSet(ts) => ts,
        other => panic!("expected a token set, got {other:?}"),
    }
}

/// A refresher whose curated table points METHOD at the mock server (the Hugging Face wiring
/// shape: endpoint + client id from the live registration, not the envelope).
fn curated_refresher(store: Arc<dyn CredentialStore>, server_uri: &str) -> TokenSetRefresher {
    let mut curated = BTreeMap::new();
    curated.insert(
        METHOD.to_string(),
        RefreshEndpoint {
            token_endpoint: format!("{server_uri}/token"),
            client_id: "hf-client".into(),
            client_secret: None,
        },
    );
    TokenSetRefresher::new(store, curated).unwrap()
}

/// The happy path: a near-expiry set is refreshed through one RFC 6749 §6 grant, the store is
/// rewritten with the new access token AND the ROTATED refresh token, and the new expiry rides
/// the response's `expires_in`.
#[tokio::test]
async fn near_expiry_refreshes_and_rotates_the_refresh_token() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/token"))
        .and(body_string_contains("grant_type=refresh_token"))
        .and(body_string_contains("refresh_token=rt-old"))
        .and(body_string_contains("client_id=hf-client"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "access_token": "at-new",
            "token_type": "Bearer",
            "refresh_token": "rt-rotated",
            "expires_in": 3600,
        })))
        .expect(1)
        .mount(&server)
        .await;

    let store: Arc<dyn CredentialStore> = Arc::new(MemCredentialStore::new());
    seed(&store, token_set(30, Some("rt-old")));
    let refresher = curated_refresher(store.clone(), &server.uri());

    assert!(refresher.refresh_if_stale(REF).await.unwrap(), "rewrote");
    let renewed = stored_token_set(&store);
    assert_eq!(renewed.access_token, "at-new");
    assert_eq!(
        renewed.refresh_token.as_deref(),
        Some("rt-rotated"),
        "the rotated refresh token replaced the old one in the same rewrite"
    );
    let expires_at = renewed.expires_at.unwrap();
    assert!(
        expires_at > unix_now() + 3000,
        "expiry rides the response's expires_in ({expires_at})"
    );
    assert_eq!(renewed.provider_id, "huggingface", "identity carried over");
}

/// An endpoint that does NOT rotate keeps the old refresh token (it stays valid per §6).
#[tokio::test]
async fn unrotated_refresh_token_is_kept() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/token"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "access_token": "at-new",
            "expires_in": 3600,
        })))
        .expect(1)
        .mount(&server)
        .await;

    let store: Arc<dyn CredentialStore> = Arc::new(MemCredentialStore::new());
    seed(&store, token_set(30, Some("rt-keep")));
    let refresher = curated_refresher(store.clone(), &server.uri());

    assert!(refresher.refresh_if_stale(REF).await.unwrap());
    assert_eq!(
        stored_token_set(&store).refresh_token.as_deref(),
        Some("rt-keep")
    );
}

/// Nothing stale means nothing happens: a fresh set (outside the skew window), a set without an
/// expiry, a bare key, and a missing row all pass through with ZERO vendor exchanges.
#[tokio::test]
async fn fresh_sets_bare_keys_and_missing_rows_are_untouched() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/token"))
        .respond_with(ResponseTemplate::new(200))
        .expect(0)
        .mount(&server)
        .await;

    let store: Arc<dyn CredentialStore> = Arc::new(MemCredentialStore::new());
    let refresher = curated_refresher(store.clone(), &server.uri());

    seed(&store, token_set(100_000, Some("rt")));
    assert!(!refresher.refresh_if_stale(REF).await.unwrap(), "fresh");

    seed(
        &store,
        OAuthTokenSet {
            expires_at: None,
            ..token_set(0, Some("rt"))
        },
    );
    assert!(!refresher.refresh_if_stale(REF).await.unwrap(), "no expiry");

    store.set(REF, "hf_pasted_bare_key").unwrap();
    assert!(!refresher.refresh_if_stale(REF).await.unwrap(), "bare key");

    store.remove(REF).unwrap();
    assert!(!refresher.refresh_if_stale(REF).await.unwrap(), "no row");
}

/// Single-flight: concurrent acquires near expiry trigger exactly ONE vendor exchange (wiremock
/// enforces `expect(1)`); the losers re-check under the lock and see the winner's fresh rewrite.
#[tokio::test]
async fn concurrent_refreshes_single_flight_one_exchange() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/token"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_delay(Duration::from_millis(150))
                .set_body_json(serde_json::json!({
                    "access_token": "at-new",
                    "refresh_token": "rt-rotated",
                    "expires_in": 3600,
                })),
        )
        .expect(1)
        .mount(&server)
        .await;

    let store: Arc<dyn CredentialStore> = Arc::new(MemCredentialStore::new());
    seed(&store, token_set(30, Some("rt-old")));
    let refresher = Arc::new(curated_refresher(store.clone(), &server.uri()));

    let tasks: Vec<_> = (0..8)
        .map(|_| {
            let refresher = refresher.clone();
            tokio::spawn(async move { refresher.refresh_if_stale(REF).await })
        })
        .collect();
    let mut rewrites = 0;
    for task in tasks {
        if task.await.unwrap().unwrap() {
            rewrites += 1;
        }
    }
    assert_eq!(rewrites, 1, "one winner rewrote; the rest saw it fresh");
    assert_eq!(stored_token_set(&store).access_token, "at-new");
}

/// A rejected grant (4xx `invalid_grant`: revoked / already-rotated-away token) is TERMINAL —
/// classified `reauth_required`, and the stored set is left as-is for the manager to show.
#[tokio::test]
async fn rejected_grant_classifies_reauth_required() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/token"))
        .respond_with(ResponseTemplate::new(400).set_body_json(serde_json::json!({
            "error": "invalid_grant",
        })))
        .expect(1)
        .mount(&server)
        .await;

    let store: Arc<dyn CredentialStore> = Arc::new(MemCredentialStore::new());
    seed(&store, token_set(30, Some("rt-revoked")));
    let refresher = curated_refresher(store.clone(), &server.uri());

    let err = refresher.refresh_if_stale(REF).await.unwrap_err();
    assert!(
        err.to_string().starts_with("reauth_required:"),
        "terminal classification: {err}"
    );
    assert_eq!(stored_token_set(&store).access_token, "at-old");
}

/// A vendor 5xx is TRANSIENT — classified `refresh_failed`, retrying the turn may succeed.
#[tokio::test]
async fn server_error_classifies_transient_refresh_failed() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/token"))
        .respond_with(ResponseTemplate::new(503))
        .expect(1)
        .mount(&server)
        .await;

    let store: Arc<dyn CredentialStore> = Arc::new(MemCredentialStore::new());
    seed(&store, token_set(30, Some("rt")));
    let refresher = curated_refresher(store, &server.uri());

    let err = refresher.refresh_if_stale(REF).await.unwrap_err();
    assert!(
        err.to_string().starts_with("refresh_failed (transient):"),
        "transient classification: {err}"
    );
}

/// A stale set with NO refresh path is `reauth_required`, split by cause: the grant issued no
/// refresh token, or (a dynamic set) no refresh context is registered nor persisted.
#[tokio::test]
async fn stale_without_a_refresh_path_is_reauth_required() {
    let store: Arc<dyn CredentialStore> = Arc::new(MemCredentialStore::new());
    // Curated context present, but the grant issued no refresh token.
    let refresher = curated_refresher(store.clone(), "http://127.0.0.1:1");
    seed(&store, token_set(30, None));
    let err = refresher.refresh_if_stale(REF).await.unwrap_err();
    assert!(err.to_string().starts_with("reauth_required:"), "{err}");
    assert!(err.to_string().contains("no refresh token"), "{err}");

    // Refresh token present, but NO context: unregistered method, nothing persisted — the
    // explicitly non-refreshable-across-restart dynamic shape.
    let bare = TokenSetRefresher::new(store.clone(), BTreeMap::new()).unwrap();
    seed(&store, token_set(30, Some("rt")));
    let err = bare.refresh_if_stale(REF).await.unwrap_err();
    assert!(err.to_string().starts_with("reauth_required:"), "{err}");
    assert!(err.to_string().contains("no refresh context"), "{err}");
}

/// A DYNAMIC set (no curated registration) refreshes through the context its completion
/// persisted in the envelope — the generic-flow shape.
#[tokio::test]
async fn dynamic_sets_refresh_through_their_persisted_context() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/token"))
        .and(body_string_contains("client_id=dyn-client"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "access_token": "at-new",
            "expires_in": 600,
        })))
        .expect(1)
        .mount(&server)
        .await;

    let store: Arc<dyn CredentialStore> = Arc::new(MemCredentialStore::new());
    seed(
        &store,
        OAuthTokenSet {
            method_id: "oauth2".into(),
            token_endpoint: Some(format!("{}/token", server.uri())),
            client_id: Some("dyn-client".into()),
            ..token_set(30, Some("rt"))
        },
    );
    // NO curated table at all — the envelope's persisted context is the only path.
    let refresher = TokenSetRefresher::new(store.clone(), BTreeMap::new()).unwrap();

    assert!(refresher.refresh_if_stale(REF).await.unwrap());
    let renewed = stored_token_set(&store);
    assert_eq!(renewed.access_token, "at-new");
    assert_eq!(
        renewed.token_endpoint.as_deref(),
        Some(format!("{}/token", server.uri()).as_str()),
        "the persisted dynamic context carries forward through the rewrite"
    );
}

/// A magic-marked blob this build cannot decode fails CLOSED at the refresh seam with the
/// actionable classification (the projector could never serve it anyway).
#[tokio::test]
async fn undecodable_envelope_fails_closed_as_reauth_required() {
    let store: Arc<dyn CredentialStore> = Arc::new(MemCredentialStore::new());
    store
        .set(REF, r#"{"daemon_credential":99,"kind":"oauth_token_set"}"#)
        .unwrap();
    let refresher = TokenSetRefresher::new(store, BTreeMap::new()).unwrap();
    let err = refresher.refresh_if_stale(REF).await.unwrap_err();
    assert!(err.to_string().starts_with("reauth_required:"), "{err}");
}

/// The skew window refreshes BEFORE expiry, and the window is the knob: one set valid for
/// another 200s is fresh under a 60s skew and stale under a 300s one.
#[tokio::test]
async fn skew_window_bounds_the_refresh_decision() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/token"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "access_token": "at-new",
        })))
        .expect(1)
        .mount(&server)
        .await;

    let store: Arc<dyn CredentialStore> = Arc::new(MemCredentialStore::new());
    seed(&store, token_set(200, Some("rt")));

    // 60s skew: 200s of validity is comfortably fresh.
    let narrow = curated_refresher(store.clone(), &server.uri()).with_skew(Duration::from_secs(60));
    assert!(!narrow.refresh_if_stale(REF).await.unwrap());

    // 300s skew: the same set is inside the window and refreshes.
    let wide = curated_refresher(store.clone(), &server.uri()).with_skew(Duration::from_secs(300));
    assert!(wide.refresh_if_stale(REF).await.unwrap());
}
