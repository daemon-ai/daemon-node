// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

use super::harness::*;

/// A valid `SKILL.md` body (frontmatter `name` + `description` + a body) for versioning tests.
fn sample_skill_md(desc: &str) -> String {
    format!("---\nname: mine\ndescription: {desc}\n---\nDo the thing.\n")
}

/// Assemble a node wired for **profile + skill versioning + distribution**: a file-backed profile
/// store, the append-only `FileRevisionLog`, and a `SkillStore` recording through that same log —
/// all under `dir`. Returns the surface, its handle, and the shared skills store.
fn assemble_versioning(
    dir: &std::path::Path,
) -> (
    Arc<NodeApiImpl>,
    daemon_host::SupervisorHandle,
    Arc<daemon_skills::SkillsProvider>,
) {
    use daemon_host::{FileProfileStore, FileRevisionLog, ProfileStore};
    let profiles: Arc<dyn ProfileStore> =
        Arc::new(FileProfileStore::open(dir.join("profiles")).unwrap());
    let revisions: Arc<dyn daemon_common::RevisionLog> =
        Arc::new(FileRevisionLog::open(dir.join("revisions")).unwrap());
    // Per-profile skills: each profile id roots at `<dir>/<id>/skills`, recording through the
    // shared revision log, with a per-profile `.usage.json` sidecar (the curator's record).
    let skills = Arc::new(
        daemon_skills::SkillsProvider::per_profile(dir.to_path_buf())
            .with_revisions(revisions.clone())
            .with_usage(Arc::new(|root: &std::path::Path| {
                Arc::new(daemon_skills::FileSkillUsageLog::open(root))
                    as Arc<dyn daemon_common::SkillUsageLog>
            })),
    );
    let AssembledNode { node, handle, .. } = assemble_node(NodeAssembly {
        store: Arc::new(InMemoryStore::new()),
        partition: PARTITION,
        host_config: fast_host_config(),
        providers: gate_providers(),
        credentials: None,
        profile: ProfileRef::new("openai"),
        engine_config: daemon_core::Config::default(),
        journal_seed: Some([0x55; 32]),
        nesting_depth: 0,
        context: None,
        context_builder: None,
        memory: Vec::new(),
        memory_builder: None,
        extra_tools: Vec::new(),
        models: None,
        profiles: Some(profiles),
        provider_resolver: None,
        credential_store: None,
        cloud_catalog: None,
        prompt_sources: vec![],
        revisions: Some(revisions),
        skills: Some(skills.clone()),
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
    });
    (node, handle, skills)
}

/// Wire page bound (v25): a revision history past `WIRE_PAGE_MAX` entries is served in cursor
/// pages (the stringified-`seq` cursor, resumed numerically) — 71 revisions page as 64 + 7,
/// oldest-first, chaining to completion with no dup or gap.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn profile_history_pages_beyond_the_wire_bound() {
    use daemon_api::{ProfileApi, ProfileSpec, ProviderSelector, WIRE_PAGE_MAX};

    static SEQ: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "daemon-history-page-{}-{}",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    let (node, handle, _skills) = assemble_versioning(&dir);

    // 1 create + 70 updates = 71 revisions.
    let mut spec = ProfileSpec::new("pg", ProviderSelector::GenAi, "model-0");
    node.profile_create(spec.clone()).await.expect("create pg");
    for i in 1..=70 {
        spec.model = format!("model-{i}");
        node.profile_update(spec.clone()).await.expect("update pg");
    }

    let mut sizes = Vec::new();
    let mut seqs = Vec::new();
    let mut after: Option<String> = None;
    let mut pages = 0;
    loop {
        let page = node
            .profile_history("pg".into(), after.take())
            .await
            .expect("history");
        assert!(
            page.items.len() <= WIRE_PAGE_MAX,
            "a wire page must never exceed WIRE_PAGE_MAX, got {}",
            page.items.len()
        );
        sizes.push(page.items.len());
        seqs.extend(page.items.iter().map(|r| r.seq));
        pages += 1;
        assert!(pages <= 4, "pagination must terminate");
        match page.next {
            Some(next) => after = Some(next),
            None => break,
        }
    }
    assert_eq!(sizes, vec![WIRE_PAGE_MAX, 7], "71 revisions page as 64 + 7");
    let expected: Vec<u64> = (1..=71).collect();
    assert_eq!(
        seqs, expected,
        "pages chain oldest-first with no dup or gap"
    );

    handle.shutdown().await;
    let _ = std::fs::remove_dir_all(&dir);
}

/// Fail-fast inference validation (wire v26 cycle): a LOCAL-provider profile (llama.cpp /
/// mistral.rs) with an empty model is rejected at create AND update with a clear, actionable
/// error — it could never load an artifact and would otherwise only fail at first turn. Cloud
/// selectors keep the empty-model latitude (the unconfigured boot placeholder relies on it).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn local_provider_profile_requires_a_model() {
    use daemon_api::{ProfileApi, ProfileSpec, ProviderSelector};

    static SEQ: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "daemon-local-model-validate-{}-{}",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    let (node, handle, _skills) = assemble_versioning(&dir);

    // Create: a llama.cpp profile with no model fails fast with an actionable message.
    let err = node
        .profile_create(ProfileSpec::new("local", ProviderSelector::LlamaCpp, ""))
        .await
        .expect_err("empty-model llama profile must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("names no model"),
        "the rejection must say what is missing: {msg}"
    );

    // A cloud selector keeps the empty-model latitude (the seeded placeholder shape).
    node.profile_create(ProfileSpec::new("cloudy", ProviderSelector::DaemonApi, ""))
        .await
        .expect("cloud profiles may stay unconfigured");

    // A configured local profile passes; blanking its model via update is rejected.
    let mut spec = ProfileSpec::new("local", ProviderSelector::LlamaCpp, "abc123");
    node.profile_create(spec.clone())
        .await
        .expect("local profile with a model");
    spec.model = String::new();
    node.profile_update(spec)
        .await
        .expect_err("blanking a local profile's model must be rejected");
    assert_eq!(
        node.profile_get("local".into())
            .await
            .unwrap()
            .unwrap()
            .model,
        "abc123",
        "the rejected update must not have been applied"
    );

    handle.shutdown().await;
    let _ = std::fs::remove_dir_all(&dir);
}

/// THE VERSIONING + DISTRIBUTION GATE: a profile's edits are versioned in a native append-only
/// history with non-destructive revert (and roll-forward), skills (incl. agent-authored ones)
/// share the same mechanism, binary-bundled skills are read-only, and a profile exports/imports
/// as a self-contained distribution (spec + local skills, `credential_ref` kept) that survives a
/// restart.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn profile_and_skill_versioning_and_distribution() {
    use daemon_api::{ProfileApi, ProfileSpec, ProviderSelector};

    static SEQ: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "daemon-versioning-{}-{}",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    let (node, handle, skills) = assemble_versioning(&dir);

    // A node assembled with a bound revision log advertises the `versioning` capability (the
    // Hello feature the client gates its History/Revert UI on).
    assert!(
        node.supports_versioning(),
        "a versioned node reports supports_versioning"
    );

    // --- profile history + non-destructive revert + roll-forward ---
    let mut spec = ProfileSpec::new("p1", ProviderSelector::GenAi, "claude-opus-4-8");
    spec.credential_ref = Some("team-key".into());
    node.profile_create(spec).await.expect("create p1");
    assert_eq!(
        node.profile_history("p1".into(), None)
            .await
            .unwrap()
            .items
            .len(),
        1
    );

    // Edit the profile in full via `profile_update` (the only durable editor; Config removed).
    let mut edited = node.profile_get("p1".into()).await.unwrap().unwrap();
    edited.model = "claude-3-5-sonnet-latest".into();
    node.profile_update(edited).await.expect("update model");
    let hist = node.profile_history("p1".into(), None).await.unwrap();
    assert_eq!(hist.items.len(), 2, "create + update = 2 revisions");
    assert_eq!(hist.items[0].author, daemon_common::Author::Operator);
    assert_eq!(
        node.profile_get("p1".into()).await.unwrap().unwrap().model,
        "claude-3-5-sonnet-latest"
    );

    // Revert to seq 1 (the original opus model): non-destructive — appends a new head.
    node.profile_revert("p1".into(), 1).await.expect("revert");
    assert_eq!(
        node.profile_get("p1".into()).await.unwrap().unwrap().model,
        "claude-opus-4-8"
    );
    assert_eq!(
        node.profile_history("p1".into(), None)
            .await
            .unwrap()
            .items
            .len(),
        3
    );
    // Roll-forward = revert to the later seq 2 (the sonnet model).
    node.profile_revert("p1".into(), 2)
        .await
        .expect("roll forward");
    assert_eq!(
        node.profile_get("p1".into()).await.unwrap().unwrap().model,
        "claude-3-5-sonnet-latest"
    );
    assert_eq!(
        node.profile_history("p1".into(), None)
            .await
            .unwrap()
            .items
            .len(),
        4
    );
    // `profile_at` returns the recorded spec without mutating the live profile.
    assert_eq!(
        node.profile_at("p1".into(), 1).await.unwrap().model,
        "claude-opus-4-8"
    );

    // --- clone (fresh history, credential_ref carried) ---
    node.profile_clone("p1".into(), "p2".into())
        .await
        .expect("clone");
    assert_eq!(
        node.profile_history("p2".into(), None)
            .await
            .unwrap()
            .items
            .len(),
        1
    );
    let p2 = node.profile_get("p2".into()).await.unwrap().unwrap();
    assert_eq!(p2.model, "claude-3-5-sonnet-latest");
    assert_eq!(p2.credential_ref.as_deref(), Some("team-key"));

    // --- skill versioning (the agent's own write path records revisions) ---
    // Skills are per-profile: target p1's own library, and make p1 the active default so the
    // name-keyed skill revision ops (`skill_revert`) write back into p1's store.
    node.profile_select("p1".into()).await.expect("select p1");
    let p1_skills = skills.for_profile("p1");
    p1_skills
        .create("mine", &sample_skill_md("v1"), None)
        .expect("create skill");
    p1_skills
        .edit("mine", &sample_skill_md("v2"))
        .expect("edit skill");
    let sk_hist = node.skill_history("mine".into(), None).await.unwrap();
    assert_eq!(sk_hist.items.len(), 2, "create + edit = 2 skill revisions");
    assert_eq!(
        sk_hist.items[0].author,
        daemon_common::Author::Agent("skill_manage".into()),
        "tool writes are attributed to the agent"
    );
    // Revert the skill to its first revision (description v1).
    node.skill_revert("mine".into(), 1)
        .await
        .expect("skill revert");
    assert!(p1_skills
        .view("mine", None)
        .unwrap()
        .contains("description: v1"));

    // Binary-bundled skills are read-only: revert is rejected.
    let bundled = daemon_skills::bundled_names();
    let bundled_name = bundled.iter().next().expect("at least one bundled skill");
    let err = node
        .skill_revert(bundled_name.clone(), 1)
        .await
        .unwrap_err();
    assert!(
        matches!(err, daemon_api::ApiError::Conflict(_)),
        "bundled skill revert should be rejected, got {err:?}"
    );

    // --- export -> import roundtrip (spec + local skills, credential_ref kept) ---
    let dist = match node.profile_export("p1".into()).await {
        Ok(d) => d,
        Err(e) => panic!("export failed: {e}"),
    };
    assert_eq!(dist.profile.credential_ref.as_deref(), Some("team-key"));
    assert!(
        dist.skills.iter().any(|s| s.name == "mine"),
        "the distribution carries the local skill"
    );
    assert!(
        !dist.skills.iter().any(|s| &s.name == bundled_name),
        "the distribution never ships binary-bundled skills"
    );

    handle.shutdown().await;

    // Import into a *fresh* node over a *new* data root (a clean machine).
    let dir2 = std::env::temp_dir().join(format!(
        "daemon-versioning-import-{}-{}",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&dir2);
    let (node2, handle2, skills2) = assemble_versioning(&dir2);
    let new_id = node2
        .profile_import(dist, Some("imported".into()))
        .await
        .expect("import");
    assert_eq!(new_id, "imported");
    let imported = node2.profile_get("imported".into()).await.unwrap().unwrap();
    assert_eq!(imported.credential_ref.as_deref(), Some("team-key"));
    assert_eq!(imported.model, "claude-3-5-sonnet-latest");
    assert!(
        skills2.for_profile("imported").find("mine").is_ok(),
        "the imported distribution reconstituted the local skill into the imported profile's dir"
    );
    assert_eq!(
        node2
            .profile_history("imported".into(), None)
            .await
            .unwrap()
            .items
            .len(),
        1,
        "an imported profile seeds a fresh history"
    );
    handle2.shutdown().await;

    // --- restart survival: reopen the original data root; history is intact ---
    let (node3, handle3, _skills3) = assemble_versioning(&dir);
    assert_eq!(
        node3
            .profile_history("p1".into(), None)
            .await
            .unwrap()
            .items
            .len(),
        4,
        "profile history survives a node restart (durable revision log)"
    );
    assert_eq!(
        node3
            .skill_history("mine".into(), None)
            .await
            .unwrap()
            .items
            .len(),
        3,
        "skill history (create + edit + revert) survives a restart"
    );
    handle3.shutdown().await;

    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&dir2);
}

/// THE PER-PROFILE CURATOR GATE: skills are agent-owned libraries, and the curator surface acts on
/// the right agent's library. Proves, over the node api: (1) two profiles keep isolated skill
/// libraries + usage (a skill created for `p1` is invisible to `p2`), (2) `curator_list` surfaces
/// usage counts + lifecycle state, (3) pin protects an agent-created skill from `curator_run`'s
/// idle-archive while an unpinned idle one is archived (agent-created provenance is the eligibility
/// signal), and (4) archive/restore move a skill out of and back into discovery.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn curator_per_profile_lifecycle_over_node() {
    use daemon_api::{ProfileApi, ProfileSpec, ProviderSelector};
    use std::sync::atomic::{AtomicU64, Ordering};

    static SEQ: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "daemon-curator-{}-{}",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    let (node, handle, skills) = assemble_versioning(&dir);

    // Two profiles, each its own agent.
    for id in ["p1", "p2"] {
        let spec = ProfileSpec::new(id, ProviderSelector::GenAi, "claude-opus-4-8");
        node.profile_create(spec).await.expect("create profile");
    }
    node.profile_select("p1".into()).await.expect("select p1");

    // Pre-seed p1's usage sidecar with two *ancient* (idle-since-epoch) agent-created entries
    // BEFORE the store is first resolved, so the curator's staleness is deterministic (staleness
    // is wall-clock relative; a freshly-created skill is never idle). `beta` will be archived;
    // `delta` is the same but will be pinned (and thus protected).
    let p1_root = skills.root_for("p1");
    std::fs::create_dir_all(&p1_root).unwrap();
    let mut seed: std::collections::BTreeMap<String, daemon_common::SkillUsage> =
        std::collections::BTreeMap::new();
    for name in ["beta", "delta"] {
        seed.insert(
            name.to_string(),
            daemon_common::SkillUsage {
                created_by: daemon_api::SkillCreator::Agent,
                state: daemon_api::SkillState::Active,
                created_at_ms: 0,
                last_used_ms: Some(0),
                ..Default::default()
            },
        );
    }
    std::fs::write(
        p1_root.join(".usage.json"),
        serde_json::to_vec_pretty(&seed).unwrap(),
    )
    .unwrap();

    // p1 grows three skills (alpha fresh, beta/delta backed by the ancient usage entries); p2 one.
    // The agent's own write path defaults to Agent authorship, so all are curation-eligible.
    let p1 = skills.for_profile("p1");
    for name in ["alpha", "beta", "delta"] {
        p1.create(name, &sample_skill_md(name), None)
            .unwrap_or_else(|e| panic!("create {name}: {e}"));
    }
    let p2 = skills.for_profile("p2");
    p2.create("gamma", &sample_skill_md("gamma skill"), None)
        .expect("p2 gamma");

    // (1) Isolation: p1's listing has its own skills, never p2's gamma; and vice versa.
    let p1_list = node.curator_list(Some("p1".into())).await.expect("list p1");
    let p1_names: Vec<_> = p1_list.iter().map(|e| e.name.as_str()).collect();
    assert!(p1_names.contains(&"alpha") && p1_names.contains(&"beta"));
    assert!(
        !p1_names.contains(&"gamma"),
        "p2's skill must not leak into p1's library"
    );
    let p2_list = node.curator_list(Some("p2".into())).await.expect("list p2");
    let p2_names: Vec<_> = p2_list.iter().map(|e| e.name.as_str()).collect();
    assert!(p2_names.contains(&"gamma") && !p2_names.contains(&"alpha"));

    // (2) Usage view: a viewed (fresh) skill shows a non-zero view/use count + agent provenance.
    p1.view("alpha", None).expect("view alpha");
    let alpha = node
        .curator_list(Some("p1".into()))
        .await
        .unwrap()
        .into_iter()
        .find(|e| e.name == "alpha")
        .unwrap();
    assert!(alpha.usage.view_count >= 1 && alpha.usage.use_count >= 1);
    assert_eq!(alpha.usage.created_by, daemon_api::SkillCreator::Agent);

    // (3) Pin protects from auto-archive: pin delta, run the curator. The idle unpinned `beta`
    // archives; the idle but pinned `delta` survives; fresh `alpha` is untouched.
    node.curator_pin(Some("p1".into()), "delta".into())
        .await
        .expect("pin delta");
    let changes = node.curator_run(Some("p1".into())).await.expect("run");
    let archived: Vec<_> = changes
        .iter()
        .filter(|c| c.to == daemon_api::SkillState::Archived)
        .map(|c| c.name.as_str())
        .collect();
    assert!(
        archived.contains(&"beta"),
        "an idle, unpinned, agent-created skill is archived; got {changes:?}"
    );
    assert!(
        !archived.contains(&"delta"),
        "a pinned skill is protected from auto-archive; got {changes:?}"
    );
    assert!(
        !archived.contains(&"alpha"),
        "a fresh skill is not archived; got {changes:?}"
    );

    // beta left discovery; delta + alpha are still live.
    assert!(p1.find("beta").is_err(), "beta archived out of discovery");
    assert!(p1.find("delta").is_ok());
    assert!(p1.find("alpha").is_ok());

    // (4) Restore beta back into the live library.
    node.curator_restore(Some("p1".into()), "beta".into())
        .await
        .expect("restore beta");
    assert!(
        p1.find("beta").is_ok(),
        "restored beta is discoverable again"
    );

    handle.shutdown().await;
    let _ = std::fs::remove_dir_all(&dir);
}
