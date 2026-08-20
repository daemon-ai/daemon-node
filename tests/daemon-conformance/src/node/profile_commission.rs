// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `ProfileCommission` (wire v53) conformance: the composite first-run/editor commit — profile
//! upsert + credential + default selection + provider probe as ONE strictly-ordered call.
//!
//! This op exists to retire a bug class: the client-orchestrated `ProfileCreate → CredentialSet →
//! ProfileSelect → ProviderModels` fire-and-forget sequence raced its own probe past its own
//! credential write (the socket dispatches each `Call` concurrently), failing first-run setup with
//! `auth_required` on a freshly pasted, valid key. Here we prove the node-side ordering: the probe
//! ALWAYS sees the credential stored earlier in the same call — including through the
//! provider-global redirect — and a probe failure persists everything with a structured verdict.

use super::harness::{
    as_system, assemble_node, fast_host_config, gate_providers, AssembledNode, NodeApiImpl,
    NodeAssembly, PARTITION,
};
use daemon_api::{
    dispatch_with_effects, ApiRequest, ApiResponse, ModelDescriptor, ProfileApi,
    ProfileCommissionArgs, ProfileCommissionOutcome, ProfileSpec, ProjectionId, ProviderDescriptor,
    ProviderListError, ProviderListErrorKind, ProviderSelector,
};
use daemon_host::{
    CloudCatalog, CredentialStore, MemCredentialStore, MemProfileStore, ProfileStore,
};
use daemon_store::{InMemoryStore, SessionStore, SqliteStore};
use std::sync::{Arc, Mutex};

/// A fake discovery hook recording the LIST key each `provider_models` call received, so a test
/// can assert the commission's probe authenticated with the credential stored EARLIER IN THE SAME
/// CALL. Returns one synthesized model per call.
struct RecordingCatalog {
    last_key: Arc<Mutex<Option<Option<String>>>>,
}

#[async_trait::async_trait]
impl CloudCatalog for RecordingCatalog {
    async fn list(&self) -> Vec<ModelDescriptor> {
        Vec::new()
    }
    async fn providers(&self) -> Vec<ProviderDescriptor> {
        Vec::new()
    }
    async fn provider_models(
        &self,
        provider_id: &str,
        key: Option<String>,
    ) -> Result<Vec<ModelDescriptor>, ProviderListError> {
        *self.last_key.lock().unwrap() = Some(key.clone());
        Ok(vec![ModelDescriptor {
            id: format!("{provider_id}/model-1"),
            provider: ProviderSelector::GenAi,
            display_name: None,
            context_length: None,
            input_price_micros_per_mtok: None,
            output_price_micros_per_mtok: None,
            local: false,
        }])
    }
}

/// A discovery hook whose listing always fails `auth_required`, for the persist-with-verdict leg.
struct FailingCatalog;

#[async_trait::async_trait]
impl CloudCatalog for FailingCatalog {
    async fn list(&self) -> Vec<ModelDescriptor> {
        Vec::new()
    }
    async fn providers(&self) -> Vec<ProviderDescriptor> {
        Vec::new()
    }
    async fn provider_models(
        &self,
        _: &str,
        _: Option<String>,
    ) -> Result<Vec<ModelDescriptor>, ProviderListError> {
        Err(ProviderListError {
            kind: ProviderListErrorKind::AuthRequired,
            message: "an API key is required".into(),
        })
    }
}

/// Assemble a node with a discovery hook + credential store + the caller's profile store, over
/// the caller's session-store backend (the both-backend axis).
fn assemble_commission(
    store: Arc<dyn SessionStore>,
    catalog: Arc<dyn CloudCatalog>,
    creds: Arc<dyn CredentialStore>,
    profiles: Arc<dyn ProfileStore>,
) -> Arc<NodeApiImpl> {
    let AssembledNode { node, .. } = assemble_node(NodeAssembly {
        store,
        partition: PARTITION,
        host_config: fast_host_config(),
        providers: gate_providers(),
        credentials: None,
        profile: daemon_common::ProfileRef::new("openai"),
        engine_config: daemon_core::Config::default(),
        journal_seed: Some([0x67; 32]),
        nesting_depth: 0,
        context: None,
        context_builder: None,
        memory: Vec::new(),
        memory_builder: None,
        extra_tools: Vec::new(),
        models: None,
        profiles: Some(profiles),
        provider_resolver: None,
        credential_store: Some(creds),
        cloud_catalog: Some(catalog),
        prompt_sources: vec![],
        revisions: None,
        skills: None,
        skills_resolver: None,
        routing: None,
        checkpoints: None,
        auth_factories: vec![],
        workspace_root: None,
        blob_root: None,
        fs: Default::default(),
        processes: Default::default(),
        title_aux: None,
        reaper: Default::default(),
        orchestrate: Default::default(),
        foreign_gateway: None,
        prompt: Default::default(),
    });
    node
}

/// A commission intent for `spec` with the pasted `credential`, probing `provider`.
fn commission_req(spec: ProfileSpec, credential: &str, provider: &str) -> ApiRequest {
    ApiRequest::ProfileCommission(ProfileCommissionArgs {
        spec,
        credential: Some(credential.into()),
        soul: None,
        activate_model: None,
        set_default: true,
        probe: Some(provider.into()),
        expected_rev: None,
    })
}

/// Unwrap a `ProfileCommissioned` reply.
fn commissioned(res: ApiResponse) -> ProfileCommissionOutcome {
    match res {
        ApiResponse::ProfileCommissioned(outcome) => outcome,
        other => panic!("expected ProfileCommissioned, got {other:?}"),
    }
}

/// The regression this op exists for, end to end: ONE commission stores the pasted key, sets the
/// default, and its probe authenticates with that key — where the old client-orchestrated
/// sequence raced and failed `NoKey`. The one receipt carries BOTH mutated domains
/// (Profiles + Credentials). Asserted against both store backends.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commission_probes_through_the_just_stored_credential() {
    for store in [
        Arc::new(InMemoryStore::new()) as Arc<dyn SessionStore>,
        Arc::new(SqliteStore::open_in_memory().expect("open sqlite store"))
            as Arc<dyn SessionStore>,
    ] {
        as_system(commission_probes_through_the_just_stored_credential_impl(
            store,
        ))
        .await;
    }
}
async fn commission_probes_through_the_just_stored_credential_impl(store: Arc<dyn SessionStore>) {
    let last_key = Arc::new(Mutex::new(None));
    let catalog = Arc::new(RecordingCatalog {
        last_key: last_key.clone(),
    });
    let creds: Arc<dyn CredentialStore> = Arc::new(MemCredentialStore::new());
    let node = assemble_commission(store, catalog, creds, Arc::new(MemProfileStore::new()));

    // An UN-namespaced model id: the credential stays profile-keyed (no provider-global mint) —
    // the exact shape the first-run wizard commits: a slug id plus the operator's free-text label
    // (wire v54 display_name), which the commission must carry through whole.
    let mut spec = ProfileSpec::new("daemon-anthropic", ProviderSelector::GenAi, "claude-x");
    spec.display_name = Some("Daemon (Anthropic)".into());
    let (res, effects) = dispatch_with_effects(
        node.as_ref(),
        commission_req(spec, "sk-pasted-123", "anthropic"),
    )
    .await;
    let outcome = commissioned(res);
    assert_eq!(outcome.profile_id, "daemon-anthropic");
    let verdict = outcome.probe.expect("a probe was requested");
    assert!(
        verdict.error.is_none(),
        "the probe must resolve the just-stored key, not race it: {verdict:?}"
    );
    assert_eq!(verdict.models.len(), 1);
    assert_eq!(
        *last_key.lock().unwrap(),
        Some(Some("sk-pasted-123".into())),
        "the probe authenticated with the credential stored earlier in the SAME call"
    );

    // One receipt, every mutated domain.
    let domains: Vec<ProjectionId> = effects.iter().map(|d| d.projection).collect();
    assert!(
        domains.contains(&ProjectionId::Profiles) && domains.contains(&ProjectionId::Credentials),
        "the composite receipt carries Profiles AND Credentials: {effects:?}"
    );

    // The committed state is all visible in the authoritative reads.
    let list = node.profile_list().await;
    let row = list
        .iter()
        .find(|p| p.id == "daemon-anthropic")
        .expect("committed profile listed");
    assert!(row.is_active, "set_default selected the committed profile");
    assert_eq!(
        row.display_name.as_deref(),
        Some("Daemon (Anthropic)"),
        "the list row carries the commissioned display_name (wire v54)"
    );
    let spec = node
        .profile_get("daemon-anthropic".into())
        .await
        .unwrap()
        .expect("committed spec readable");
    assert_eq!(spec.display_name.as_deref(), Some("Daemon (Anthropic)"));
}

/// The provider-global redirect leg: a NAMESPACED model mints `credential_ref =
/// provider/<vendor>`, `credential_set` redirects the pasted secret THERE — and the commission's
/// probe must read the same ref (the committed spec's `credential_profile()`), not the profile id
/// (where nothing was stored). The old client passed the profile id and would miss.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commission_probe_follows_the_provider_global_redirect() {
    as_system(commission_probe_follows_the_provider_global_redirect_impl()).await;
}
async fn commission_probe_follows_the_provider_global_redirect_impl() {
    let last_key = Arc::new(Mutex::new(None));
    let catalog = Arc::new(RecordingCatalog {
        last_key: last_key.clone(),
    });
    let creds: Arc<dyn CredentialStore> = Arc::new(MemCredentialStore::new());
    let node = assemble_commission(
        Arc::new(InMemoryStore::new()),
        catalog,
        creds.clone(),
        Arc::new(MemProfileStore::new()),
    );

    let spec = ProfileSpec::new(
        "daemon-anthropic",
        ProviderSelector::GenAi,
        "anthropic::claude-x",
    );
    let (res, _) = dispatch_with_effects(
        node.as_ref(),
        commission_req(spec, "sk-redirected-456", "anthropic"),
    )
    .await;
    let outcome = commissioned(res);
    let verdict = outcome.probe.expect("a probe was requested");
    assert!(
        verdict.error.is_none(),
        "redirected key resolved: {verdict:?}"
    );
    assert_eq!(
        *last_key.lock().unwrap(),
        Some(Some("sk-redirected-456".into())),
        "the probe read the provider-global ref the redirect stored under"
    );
    assert_eq!(
        creds.get("provider/anthropic").as_deref(),
        Some("sk-redirected-456"),
        "the secret landed under the node-minted provider-global ref"
    );
}

/// Persist-with-verdict: a probe failure does NOT unwind the commit — the profile, credential and
/// default selection all stay durable, and the reply carries the structured `auth_required`
/// verdict (never a hard `ApiError`). A transient vendor outage must not destroy a valid key.
/// Asserted against both store backends.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commission_persists_everything_on_probe_failure() {
    for store in [
        Arc::new(InMemoryStore::new()) as Arc<dyn SessionStore>,
        Arc::new(SqliteStore::open_in_memory().expect("open sqlite store"))
            as Arc<dyn SessionStore>,
    ] {
        as_system(commission_persists_everything_on_probe_failure_impl(store)).await;
    }
}
async fn commission_persists_everything_on_probe_failure_impl(store: Arc<dyn SessionStore>) {
    let creds: Arc<dyn CredentialStore> = Arc::new(MemCredentialStore::new());
    let node = assemble_commission(
        store,
        Arc::new(FailingCatalog),
        creds.clone(),
        Arc::new(MemProfileStore::new()),
    );

    let spec = ProfileSpec::new("daemon-anthropic", ProviderSelector::GenAi, "claude-x");
    let (res, effects) = dispatch_with_effects(
        node.as_ref(),
        commission_req(spec, "sk-still-kept-789", "anthropic"),
    )
    .await;
    let outcome = commissioned(res);
    let verdict = outcome.probe.expect("a probe was requested");
    let error = verdict.error.expect("the probe failed, structured");
    assert_eq!(error.kind, ProviderListErrorKind::AuthRequired);

    // Everything the mutations committed is still there.
    assert!(
        node.profile_get("daemon-anthropic".into())
            .await
            .expect("get")
            .is_some(),
        "the profile persisted through the probe failure"
    );
    assert_eq!(
        creds.get("daemon-anthropic").as_deref(),
        Some("sk-still-kept-789"),
        "the credential persisted through the probe failure"
    );
    let list = node.profile_list().await;
    assert!(
        list.iter()
            .any(|p| p.id == "daemon-anthropic" && p.is_active),
        "the default selection persisted through the probe failure"
    );
    let domains: Vec<ProjectionId> = effects.iter().map(|d| d.projection).collect();
    assert!(
        domains.contains(&ProjectionId::Profiles) && domains.contains(&ProjectionId::Credentials),
        "the mutations' receipt still carries every domain: {effects:?}"
    );
}

/// First-run replacement (wire v47) through the composite: a commission that CREATES retires the
/// still-seeded, session-free placeholder and the committed profile takes the default — the same
/// node-authoritative behavior as a plain operator `ProfileCreate`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commission_retires_the_seeded_placeholder() {
    as_system(commission_retires_the_seeded_placeholder_impl()).await;
}
async fn commission_retires_the_seeded_placeholder_impl() {
    let profiles = Arc::new(MemProfileStore::new());
    profiles
        .seed(ProfileSpec::new(
            "default",
            ProviderSelector::GenAi,
            "seed-model",
        ))
        .expect("seed placeholder");
    let creds: Arc<dyn CredentialStore> = Arc::new(MemCredentialStore::new());
    let node = assemble_commission(
        Arc::new(InMemoryStore::new()),
        Arc::new(RecordingCatalog {
            last_key: Arc::new(Mutex::new(None)),
        }),
        creds,
        profiles,
    );

    let spec = ProfileSpec::new("daemon-anthropic", ProviderSelector::GenAi, "claude-x");
    let (res, _) =
        dispatch_with_effects(node.as_ref(), commission_req(spec, "sk-abc", "anthropic")).await;
    let outcome = commissioned(res);
    assert_eq!(outcome.profile_id, "daemon-anthropic");

    let list = node.profile_list().await;
    assert!(
        !list.iter().any(|p| p.id == "default"),
        "the seeded placeholder was retired: {list:?}"
    );
    assert!(
        list.iter()
            .any(|p| p.id == "daemon-anthropic" && p.is_active),
        "the committed profile is the active default: {list:?}"
    );
}

/// The upsert leg: a commission naming an EXISTING id replaces the spec in place (update
/// semantics — no duplicate, `seeded` stays cleared), and a credential-less commission leaves the
/// stored key untouched.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commission_updates_an_existing_profile_in_place() {
    as_system(commission_updates_an_existing_profile_in_place_impl()).await;
}
async fn commission_updates_an_existing_profile_in_place_impl() {
    let creds: Arc<dyn CredentialStore> = Arc::new(MemCredentialStore::new());
    creds.set("daemon-anthropic", "sk-kept").expect("seed key");
    let node = assemble_commission(
        Arc::new(InMemoryStore::new()),
        Arc::new(RecordingCatalog {
            last_key: Arc::new(Mutex::new(None)),
        }),
        creds.clone(),
        Arc::new(MemProfileStore::new()),
    );
    node.profile_create(ProfileSpec::new(
        "daemon-anthropic",
        ProviderSelector::GenAi,
        "claude-old",
    ))
    .await
    .expect("create");

    let (res, _) = dispatch_with_effects(
        node.as_ref(),
        ApiRequest::ProfileCommission(ProfileCommissionArgs {
            spec: ProfileSpec::new("daemon-anthropic", ProviderSelector::GenAi, "claude-new"),
            credential: None,
            soul: None,
            activate_model: None,
            set_default: false,
            probe: None,
            expected_rev: None,
        }),
    )
    .await;
    let outcome = commissioned(res);
    assert!(outcome.probe.is_none(), "no probe was requested");

    let list = node.profile_list().await;
    let rows: Vec<&str> = list
        .iter()
        .filter(|p| p.id == "daemon-anthropic")
        .map(|p| p.model.as_str())
        .collect();
    assert_eq!(rows, vec!["claude-new"], "replaced in place, no duplicate");
    assert_eq!(
        creds.get("daemon-anthropic").as_deref(),
        Some("sk-kept"),
        "a credential-less commission leaves the stored key untouched"
    );
}

/// Stage 8 OCC through the composite: a stale `expected_rev` rejects the WHOLE commission with
/// `Conflict` before any step runs — no profile, no credential, no receipt.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commission_stale_expected_rev_conflicts_untouched() {
    as_system(commission_stale_expected_rev_conflicts_untouched_impl()).await;
}
async fn commission_stale_expected_rev_conflicts_untouched_impl() {
    let creds: Arc<dyn CredentialStore> = Arc::new(MemCredentialStore::new());
    let node = assemble_commission(
        Arc::new(InMemoryStore::new()),
        Arc::new(RecordingCatalog {
            last_key: Arc::new(Mutex::new(None)),
        }),
        creds.clone(),
        Arc::new(MemProfileStore::new()),
    );
    // Advance the Profiles domain past rev 0.
    node.profile_create(ProfileSpec::new("first", ProviderSelector::GenAi, "m"))
        .await
        .expect("create first");

    let (res, effects) = dispatch_with_effects(
        node.as_ref(),
        ApiRequest::ProfileCommission(ProfileCommissionArgs {
            spec: ProfileSpec::new("stale-loser", ProviderSelector::GenAi, "m"),
            credential: Some("sk-never-stored".into()),
            soul: None,
            activate_model: None,
            set_default: true,
            probe: Some("anthropic".into()),
            expected_rev: Some(0),
        }),
    )
    .await;
    assert!(
        matches!(res, ApiResponse::Error(daemon_api::ApiError::Conflict(_))),
        "a stale observation must Conflict: {res:?}"
    );
    assert!(
        effects.is_empty(),
        "a rejected commission mutates nothing: {effects:?}"
    );
    assert!(
        node.profile_get("stale-loser".into())
            .await
            .expect("get")
            .is_none(),
        "no profile landed"
    );
    assert_eq!(creds.get("stale-loser"), None, "no credential landed");
}
