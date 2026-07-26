// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The assess / probe side of the worker (§6.5, §10.2).
//!
//! Owns the `Probe` hardware report, the `AssessRun` envelope→`(config, module)` resolution, and the
//! v2 claim()-funnel eligibility pass ([`assess`]). (The v1 `WasmBackend` construction / autotune
//! assess this file also carried retired with the v1 driver at the Phase-E sunset.)

use daemon_vhc_host::probe::DeviceLimits;
use daemon_vhc_host::{EngineConfig, Worker};
use daemon_vhc_net::{ArtifactRef, ArtifactResolver};
use daemon_vhc_proto::{from_canonical_slice, SignedEnvelope};
use daemon_vhc_session::protocol::{BackendCapability, Eligibility, Hardware, WorkerCapabilities};

/// A large sentinel (in MiB) used when a resource dimension is unknown, so the admission budget math
/// does not spuriously reject on an unprobed number (`u64::MAX / MiB`).
const UNKNOWN_BUDGET_MB: u64 = u64::MAX / (1 << 20);

/// Author the complete ABI §2.6 grants document for the run's worker role (§9.4 steps 8/11 hash
/// pinning — assess and join derive byte-identical copies). Admission authors the truth: the
/// document enumerates the complete capability surface the module actually links + the genesis
/// role grant list — the worlds the module imports (incl. `compute@2`/`data@2`), the role's
/// channel table, custom ops, artifacts, buffer limits, and rate/quota bounds. This replaces the
/// former hand-rolled `vhc@2`/`net@2`/`sys@2` subset (audit finding: grants inconsistent with
/// what the trainer uses).
///
/// Both assess and join call this with the SAME `(worker, module, genesis role)` inputs, so the
/// canonical bytes — and the grants hash — match by construction.
pub(crate) fn derive_grants(
    worker: &Worker,
    module: &[u8],
    role_grants: &daemon_vhc_proto::genesis::RoleGrants,
) -> Result<Vec<u8>, String> {
    let linked = daemon_vhc_host::linked_worlds(worker, module)
        .map_err(|e| format!("linked worlds for grants authoring: {e}"))?;
    Ok(daemon_vhc_proto::GrantsDoc::author(&linked, role_grants).to_canonical_bytes())
}

/// The experiment inputs a run resolves to: the worker role's opaque config CBOR + the module
/// `.wasm`, plus the envelope's per-role blake3 pin when one exists (the ABI §1.3 step-1
/// verify-before-compile input; `None` under the explicit `DAEMON_TRAIN_MODULE` module-source
/// override, which deliberately bypasses the artifact map and so carries no pin).
pub(crate) struct ResolvedRun {
    pub(crate) config: Vec<u8>,
    pub(crate) module: Vec<u8>,
    pub(crate) module_blake3: Option<[u8; 32]>,
    /// The worker role's typed `device_min` (ABI §9.3 stage-3 pre-screen input).
    pub(crate) device_min: Option<daemon_vhc_proto::DeviceMinimums>,
    /// The resolved genesis run (envelope v2). `Some` on every resolvable run — the schema-1
    /// envelope form refuses typed at [`ENVELOPE_SCHEMA_RETIRED`] before resolution.
    pub(crate) genesis: Option<GenesisRun>,
}

/// A resolved envelope-v2 (genesis) run: the decoded envelope, the frozen wire form (the
/// coordinator configuration source — `configure_coordinator` reads role config bytes
/// + the cryptographic `RunId` from it), and the joining worker role.
pub(crate) struct GenesisRun {
    pub(crate) env: daemon_vhc_proto::GenesisEnvelope,
    pub(crate) frozen: daemon_vhc_proto::FrozenGenesis,
    /// The worker role this node assessed/joins (the first non-coordinator role — the
    /// single-worker interim of decisions D6; role *selection* is node policy from Phase E).
    pub(crate) worker_role: String,
}

impl ResolvedRun {
    /// The envelope-v2 grants input for the admission funnel (D1 deliverable 4): the worker
    /// role's grant list + the run's artifact-map hashes, derived from the genesis envelope.
    /// `None` on the v1 path — the funnel's pre-D0 defaults stand there.
    pub(crate) fn envelope_grants(&self) -> Option<daemon_vhc_host::run::EnvelopeRoleGrants> {
        let g = self.genesis.as_ref()?;
        daemon_vhc_host::run::EnvelopeRoleGrants::from_genesis(&g.env, &g.worker_role)
    }
}

/// The typed refusal slug for the D0-retired unsigned legacy envelope path (refactor §8/D0:
/// "the worker's unsigned legacy envelope path is retired here with a typed refusal"). Stable —
/// tests and the node key on it, exactly like the ABI §1.5 refusal slugs.
pub(crate) const UNSIGNED_ENVELOPE_RETIRED: &str = "UnsignedEnvelopeRetired";

/// The typed refusal slug for the retired schema-major-1 (v1) envelope form: a genesis (schema-2)
/// envelope is the only run description this worker resolves — the v1 form cannot configure a
/// coordinator (no coordinator role entry, no `Authority`/identities section), so it refuses
/// HERE, at assess, before any module is fetched or inspected. Stable, like the slug above.
pub(crate) const ENVELOPE_SCHEMA_RETIRED: &str = "EnvelopeSchemaRetired";

/// Resolve the `AssessRun` envelope bytes into `(config, module)` (the §6.1/§6.5 seam).
///
/// The bytes MUST be the canonical [`SignedEnvelope`] wire form carrying a **genesis (schema-2)**
/// envelope: verify it, take the worker role's opaque config, and resolve the module by its
/// pinned artifact hash. `DAEMON_TRAIN_MODULE` remains the explicit dev/node-controlled
/// **module-source override inside the signed path** (it substitutes the artifact fetch, never
/// the envelope).
///
/// **D0: the unsigned legacy path is RETIRED.** Bytes that are not a signed-envelope wrapper
/// (the pre-A0 raw `[experiment.config]` CBOR direct-drive) are refused with the typed
/// [`UNSIGNED_ENVELOPE_RETIRED`] slug — never accepted, never guessed at.
///
/// **The schema-major-1 (v1) envelope form is RETIRED**: it refuses with the typed
/// [`ENVELOPE_SCHEMA_RETIRED`] slug at assess (a v1 envelope cannot configure a wasm
/// coordinator — no coordinator role, no `Authority`/identities section). Authors present a
/// genesis envelope v2 instead.
pub(crate) async fn resolve_run(
    envelope_bytes: &[u8],
    role: Option<&str>,
) -> Result<ResolvedRun, String> {
    let wire = from_canonical_slice::<SignedEnvelope>(envelope_bytes).map_err(|e| {
        format!(
            "{UNSIGNED_ENVELOPE_RETIRED}: AssessRun bytes are not a SignedEnvelope wire form \
             (the unsigned legacy raw-config path was retired at D0; author a signed envelope — \
             DAEMON_TRAIN_MODULE still overrides the module source inside it): {e}"
        )
    })?;
    // Route on the schema sniff: schema 2 (genesis) resolves; anything else refuses typed.
    match daemon_vhc_proto::peek_schema(&wire.bytes) {
        Some(daemon_vhc_proto::GENESIS_SCHEMA_MAJOR) => resolve_genesis_run(wire, role).await,
        Some(major) => Err(format!(
            "{ENVELOPE_SCHEMA_RETIRED}: envelope schema major {major} is retired — it cannot \
             configure a coordinator (no coordinator role entry to pin a module hash, no \
             Authority/identities section to name the signer); author a genesis envelope v2"
        )),
        None => Err(format!(
            "{ENVELOPE_SCHEMA_RETIRED}: the signed bytes carry no recognizable `[run].schema` \
             major; author a genesis envelope v2"
        )),
    }
}

/// Resolve an envelope-v2 **genesis** run (D1 deliverable 4; mixed-fleet retired-native-coordinator): verify the
/// signed genesis (hash re-derived over the received bytes, author signature checked), select the
/// worker + coordinator roles from the role set, decode the worker role's opaque config, and
/// resolve the worker role's module by its pinned artifact hash.
///
/// Role selection is NODE-DIRECTED: `directed_role` names the envelope role label to run (the
/// seat-claim path directs the coordinator role); a label absent from the genesis role set is a
/// typed refusal. Undirected (`None`), the default is the first role whose LANE is not
/// `coordinator` (the single-trainer interim — selection by declared lane, never by label
/// heuristics).
async fn resolve_genesis_run(
    wire: SignedEnvelope,
    directed_role: Option<&str>,
) -> Result<ResolvedRun, String> {
    let frozen = daemon_vhc_proto::FrozenGenesis::open(wire.bytes, wire.signature, wire.signer)
        .map_err(|e| format!("verify genesis envelope: {e}"))?;
    let env = frozen
        .decode()
        .map_err(|e| format!("decode genesis: {e}"))?;
    let worker_role = match directed_role {
        Some(label) => {
            if !env.roles.contains_key(label) {
                return Err(format!(
                    "directed role `{label}` is absent from the genesis role set (roles: {:?})",
                    env.roles.keys().collect::<Vec<_>>()
                ));
            }
            label.to_string()
        }
        None => env
            .roles
            .iter()
            .find(|(_, r)| r.lane != "coordinator")
            .map(|(name, _)| name.clone())
            .ok_or("genesis envelope has no non-coordinator role (validate should have refused)")?,
    };
    let role = &env.roles[&worker_role];
    let config = frozen
        .role_config_bytes(&worker_role)
        .map_err(|e| format!("role config: {e}"))?
        .ok_or("worker role vanished between decode and config extraction")?;
    let device_min = Some(role.device_min.clone());

    // Resolve the worker role's module by its pinned artifact hash — same override + fetch
    // discipline as before (DAEMON_TRAIN_MODULE is the explicit dev/node-controlled
    // module-source override inside the signed path).
    let (module, module_blake3) = if let Some(bytes) = module_from_env() {
        (bytes?, None)
    } else {
        let artifact = env.artifacts.get(&role.module).ok_or_else(|| {
            format!(
                "worker role module `{}` absent from the genesis artifact map",
                role.module
            )
        })?;
        let pin = Some(artifact.blake3.0);
        let module_name = role.module.clone();
        // The presign RunId is the genesis run id (blake3 of the frozen bytes) — the registry
        // namespaces `runs/<run>/modules/…` under it. Deriving it here means a correct r2 fetch
        // even though the worker's spawn env carries no per-run id (defect B: the node hands the
        // worker the presign BASE + AUTH via env; the run id comes from the genesis it decodes).
        let run_id_hex = frozen.run_id().to_hex();
        (
            fetch_genesis_artifact(artifact, &module_name, Some(&run_id_hex)).await?,
            pin,
        )
    };
    Ok(ResolvedRun {
        config,
        module,
        module_blake3,
        device_min,
        genesis: Some(GenesisRun {
            env,
            frozen,
            worker_role,
        }),
    })
}

/// Fetch one genesis artifact-map entry (blake3-verified): `file://` through the file resolver;
/// network URLs through the payload-store path under the `vhc-net` feature.
///
/// `run_id` is the presign RunId the store-fetch path namespaces `r2://` objects under
/// (`POST /runs/<run_id>/presign`). Pass the genesis run id (`frozen.run_id().to_hex()`) so the
/// module/corpus fetch targets the correct `runs/<run>/…` prefix regardless of the worker's spawn
/// env; `None` falls back to the env `DAEMON_VHC_RUN_ID` (the standalone prefetch path).
async fn fetch_genesis_artifact(
    artifact: &daemon_vhc_proto::SnapshotArtifact,
    name: &str,
    run_id: Option<&str>,
) -> Result<Vec<u8>, String> {
    let art = ArtifactRef::new(artifact.url.clone(), artifact.blake3);
    #[cfg(feature = "vhc-net")]
    if !artifact.url.starts_with("file://") {
        return fetch_artifact_from_store(&art, run_id)
            .await
            .map_err(|e| format!("fetch `{name}` ({}) from store: {e}", artifact.url));
    }
    #[cfg(not(feature = "vhc-net"))]
    let _ = run_id;
    ArtifactResolver::new()
        .fetch(&art)
        .await
        .map_err(|e| format!("resolve `{name}` ({}): {e}", artifact.url))
}

/// Resolve the genesis **coordinator role's** module by its pinned artifact hash (the in-process
/// self-driven join runs the run's real coordinator; consensus never runs outside the
/// sandboxed, content-addressed module). The `DAEMON_TRAIN_MODULE` override deliberately does
/// NOT apply here — it substitutes the WORKER module source only. Only the harness-featured
/// self-driven join resolves a coordinator module.
#[cfg(feature = "harness")]
pub(crate) async fn resolve_coordinator_module(
    env: &daemon_vhc_proto::GenesisEnvelope,
) -> Result<Vec<u8>, String> {
    let (_, role) = env
        .roles
        .iter()
        .find(|(_, r)| r.lane == "coordinator")
        .ok_or("genesis envelope has no role with lane `coordinator`")?;
    let artifact = env.artifacts.get(&role.module).ok_or_else(|| {
        format!(
            "coordinator role module `{}` absent from the genesis artifact map",
            role.module
        )
    })?;
    // Harness-only coordinator-module fetch (in-process self-driven join): the env
    // `DAEMON_VHC_RUN_ID` is authoritative here (no frozen handle at this call site).
    fetch_genesis_artifact(artifact, &role.module, None).await
}

/// The presign context the node sets when spawning the worker for a live run (small env strings, NOT
/// a pre-staged artifact): the coordinator base, run id, auth, and the on-disk cache dir/budget. This
/// is the fetch-by-hash analogue of `DAEMON_TRAIN_MODULE` — a node-controlled input at assess time.
#[cfg(feature = "vhc-net")]
pub(crate) struct StoreFetchContext {
    pub(crate) presign_base: Option<String>,
    pub(crate) run_id: String,
    pub(crate) ws_auth: daemon_vhc_session::protocol::WsAuthSpec,
    pub(crate) cache_dir: std::path::PathBuf,
    pub(crate) cache_gb: u32,
}

#[cfg(feature = "vhc-net")]
pub(crate) fn store_fetch_context() -> StoreFetchContext {
    use daemon_vhc_session::protocol::WsAuthSpec;
    let ws_auth = if let Ok(bearer) = std::env::var("DAEMON_VHC_BEARER") {
        WsAuthSpec::Bearer(bearer)
    } else if let (Ok(org_id), Ok(actor)) = (
        std::env::var("DAEMON_VHC_ORG"),
        std::env::var("DAEMON_VHC_ACTOR"),
    ) {
        WsAuthSpec::Internal { org_id, actor }
    } else {
        WsAuthSpec::None
    };
    let cache_dir = std::env::var_os("DAEMON_VHC_CACHE_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("daemon-vhc-cache"));
    let cache_gb = std::env::var("DAEMON_VHC_CACHE_GB")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    StoreFetchContext {
        presign_base: std::env::var("DAEMON_VHC_PRESIGN_BASE").ok(),
        run_id: std::env::var("DAEMON_VHC_RUN_ID").unwrap_or_else(|_| "run-unknown".to_string()),
        ws_auth,
        cache_dir,
        cache_gb,
    }
}

/// Build a content-addressed [`daemon_vhc_net::ContentCache`] from the node-set context.
#[cfg(feature = "vhc-net")]
pub(crate) fn open_content_cache(
    ctx: &StoreFetchContext,
) -> Result<daemon_vhc_net::ContentCache, String> {
    daemon_vhc_net::ContentCache::open_gb(&ctx.cache_dir, ctx.cache_gb)
        .map_err(|e| format!("open content cache {}: {e}", ctx.cache_dir.display()))
}

/// Build an [`ArtifactResolver`] wired for the network schemes from the node-set presign context
/// (egress for `https`/`hf`; egress + presign for `r2://`).
#[cfg(feature = "vhc-net")]
pub(crate) fn store_resolver(ctx: &StoreFetchContext) -> Result<ArtifactResolver, String> {
    use daemon_vhc_net::{HttpPresignClient, PresignClient, RunId};
    let egress = daemon_egress::EgressClient::new(daemon_egress::EgressConfig::default())
        .map_err(|e| format!("egress client: {e}"))?;
    let mut resolver = ArtifactResolver::with_egress(egress);
    if let Some(base) = &ctx.presign_base {
        use daemon_vhc_session::protocol::WsAuthSpec;
        let presign_egress =
            daemon_egress::EgressClient::new(daemon_egress::EgressConfig::default())
                .map_err(|e| format!("presign egress client: {e}"))?;
        let presign = match &ctx.ws_auth {
            WsAuthSpec::None => HttpPresignClient::new(presign_egress, base.clone()),
            WsAuthSpec::Bearer(t) => {
                HttpPresignClient::new(presign_egress, base.clone()).with_bearer(t.clone())
            }
            WsAuthSpec::Internal { org_id, actor } => {
                HttpPresignClient::new(presign_egress, base.clone())
                    .with_internal(org_id.clone(), actor.clone())
            }
        };
        let presign: std::sync::Arc<dyn PresignClient> = std::sync::Arc::new(presign);
        resolver = resolver.with_presign(presign, RunId::new(&ctx.run_id));
    }
    Ok(resolver)
}

/// Fetch `art` from the payload store (presigned GET), checking the on-disk content cache first and
/// caching the verified bytes on a miss (the fleet artifact-distribution path).
#[cfg(feature = "vhc-net")]
pub(crate) async fn fetch_artifact_from_store(
    art: &ArtifactRef,
    run_id: Option<&str>,
) -> Result<Vec<u8>, String> {
    let mut ctx = store_fetch_context();
    // The genesis-derived run id (when the caller has one) overrides the env default so the
    // presign request targets the right `runs/<run>/…` prefix.
    if let Some(run_id) = run_id {
        ctx.run_id = run_id.to_string();
    }
    let cache = open_content_cache(&ctx)?;
    if let Some(bytes) = cache.get(&art.blake3).await.map_err(|e| e.to_string())? {
        return Ok(bytes);
    }
    let resolver = store_resolver(&ctx)?;
    let bytes = resolver.fetch(art).await.map_err(|e| e.to_string())?;
    // Best-effort cache write (a cache failure must not fail a run whose bytes are already verified).
    if let Err(e) = cache.insert(&art.blake3, &bytes).await {
        eprintln!("[daemon-vhc-worker] content cache insert failed (continuing): {e}");
    }
    Ok(bytes)
}

/// Bind the run's **genesis-pinned artifact plane** over the session's committed-payload plane, so
/// the module-driven `data.fetch` seat resolves a pinned content id at the url the ENVELOPE commits
/// (`modules/<blake3>.wasm`, `corpus/<blake3>.cbor`, `corpus/<fold>.bin`, …) instead of looking for
/// it under the payload plane's own `payload/<hex>` key, where nothing publishes it.
///
/// This is the runtime twin of [`fetch_genesis_artifact`] — the same resolver, the same on-disk
/// content cache, the same presign context — for the artifacts the GUEST fetches rather than the
/// ones the worker fetches on its behalf. Hashes the envelope does not pin (committed payloads,
/// checkpoint documents, det-state family chunks) fall through to `content` unchanged.
#[cfg(feature = "vhc-net")]
pub(crate) fn pinned_artifact_plane(
    genesis: &GenesisRun,
    run_label: &str,
    content: std::sync::Arc<dyn daemon_vhc_net::ContentStore>,
) -> Result<std::sync::Arc<dyn daemon_vhc_net::ContentStore>, String> {
    let pinned: std::collections::BTreeMap<daemon_vhc_net::ContentHash, String> = genesis
        .env
        .artifacts
        .values()
        .map(|a| (a.blake3, a.url.clone()))
        .collect();
    if pinned.is_empty() {
        return Ok(content);
    }
    let mut ctx = store_fetch_context();
    // The presign RunId is the run label the SESSION's content plane is namespaced under
    // (`R2Store::new(.., RunId::new(run_label))`), because a pinned artifact and a committed
    // payload live in two namespaces of the SAME per-run bucket prefix. (On the fleet the run label
    // is the genesis hash hex, so this is also the module fetch's prefix.)
    ctx.run_id = run_label.to_string();
    // No presign base means no artifact plane to resolve against (the referenceless in-process
    // seat, or a filesystem content plane that already holds everything by content address) —
    // leave the content plane bound alone rather than wrapping it with a resolver that can only
    // refuse.
    if ctx.presign_base.is_none() {
        return Ok(content);
    }
    let cache = open_content_cache(&ctx).ok();
    let resolver = store_resolver(&ctx)?;
    Ok(std::sync::Arc::new(
        daemon_vhc_net::PinnedArtifactStore::new(pinned, resolver, cache, content),
    ))
}

/// Decode a 64-char lowercase-hex blake3 into a content hash.
#[cfg(feature = "vhc-net")]
pub(crate) fn hash_from_hex(s: &str) -> Option<daemon_vhc_net::ContentHash> {
    use daemon_vhc_net::ContentHash;
    let s = s.trim();
    if s.len() != ContentHash::LEN * 2 {
        return None;
    }
    let mut out = [0u8; ContentHash::LEN];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).ok()?;
    }
    Some(ContentHash::new(out))
}

/// The shard indices covering the sequence window `[start, start + count)` (wrapping;
/// `count == 0` or `>= total` = every shard) — mechanical staging arithmetic over the manifest
/// geometry (WHICH window to warm remains the operator's/module's choice).
#[cfg(feature = "vhc-net")]
fn shards_covering_window(
    manifest: &daemon_vhc_proto::CorpusManifest,
    start: u64,
    count: u64,
) -> Vec<usize> {
    let total = manifest.total_sequences();
    if total == 0 {
        return Vec::new();
    }
    if count == 0 || count >= total {
        return (0..manifest.shards.len()).collect();
    }
    let seq_len = u64::from(manifest.seq_len);
    let start = start % total;
    let ranges: [(u64, u64); 2] = if start + count <= total {
        [(start, start + count), (0, 0)]
    } else {
        [(start, total), (0, (start + count) - total)]
    };
    let mut out = Vec::new();
    let mut cum = 0u64;
    for (i, shard) in manifest.shards.iter().enumerate() {
        let seqs = shard.token_count / seq_len;
        let (bs, be) = (cum, cum + seqs);
        cum = be;
        for (rs, re) in ranges {
            if rs < re && bs < re && rs < be {
                out.push(i);
                break;
            }
        }
    }
    out
}

/// The `DAEMON_TRAIN_PREFETCH` cache-warming mode (the fleet staging entry point).
///
/// Runs on a bare fleet box (Windows cmd.exe, macOS, a RunPod container) with no CBOR framing:
/// fetch the run's module and/or corpus (manifest + windowed shards) **by content hash** from the
/// payload store into the on-disk [`daemon_vhc_net::ContentCache`], verify blake3, print per-object
/// `key / bytes / blake3 / source(cache|store) / ms`, then exit. A subsequent live run on the box
/// finds every artifact cache-warm. Idempotent (re-running is all cache hits).
///
/// Env (mirrors the assess-time fetch context; all plain strings so `set X=… && worker.exe` works):
/// `DAEMON_VHC_PRESIGN_BASE`, `DAEMON_VHC_RUN_ID`, `DAEMON_VHC_ORG`/`DAEMON_VHC_ACTOR` (or
/// `DAEMON_VHC_BEARER`), `DAEMON_VHC_CACHE_DIR`/`DAEMON_VHC_CACHE_GB`, plus what to warm:
/// `DAEMON_TRAIN_PREFETCH_MODULE=<blake3-hex>`, `DAEMON_TRAIN_PREFETCH_MANIFEST=<blake3-hex>`,
/// `DAEMON_TRAIN_PREFETCH_WINDOW=<start>:<count>` (optional; absent/0 = every shard).
#[cfg(feature = "vhc-net")]
pub(crate) async fn prefetch_main() -> Result<(), String> {
    let ctx = store_fetch_context();
    if ctx.presign_base.is_none() {
        return Err("DAEMON_TRAIN_PREFETCH needs DAEMON_VHC_PRESIGN_BASE".to_string());
    }
    let cache = open_content_cache(&ctx)?;
    let resolver = store_resolver(&ctx)?;
    println!(
        "prefetch: run={} presign_base={} cache_dir={} cache_gb={}",
        ctx.run_id,
        ctx.presign_base.as_deref().unwrap_or("-"),
        ctx.cache_dir.display(),
        ctx.cache_gb
    );

    /// Fetch one object, cache-first, printing the staging-evidence line.
    async fn warm(
        cache: &daemon_vhc_net::ContentCache,
        resolver: &ArtifactResolver,
        label: &str,
        art: &ArtifactRef,
    ) -> Result<Vec<u8>, String> {
        let t0 = std::time::Instant::now();
        let (bytes, source) = match cache.get(&art.blake3).await.map_err(|e| e.to_string())? {
            Some(b) => (b, "cache"),
            None => {
                let b = resolver
                    .fetch(art)
                    .await
                    .map_err(|e| format!("{label} ({}): {e}", art.url))?;
                if let Err(e) = cache.insert(&art.blake3, &b).await {
                    eprintln!("prefetch: cache insert failed for {label} (continuing): {e}");
                }
                (b, "store")
            }
        };
        println!(
            "prefetch: {label} bytes={} blake3={} source={source} ms={}",
            bytes.len(),
            art.blake3.to_hex(),
            t0.elapsed().as_millis()
        );
        Ok(bytes)
    }

    let mut warmed = 0u32;
    if let Ok(hex) = std::env::var("DAEMON_TRAIN_PREFETCH_MODULE") {
        let hash =
            hash_from_hex(&hex).ok_or_else(|| format!("bad DAEMON_TRAIN_PREFETCH_MODULE {hex}"))?;
        let art = ArtifactRef::new(format!("r2://modules/{}.wasm", hash.to_hex()), hash);
        warm(&cache, &resolver, "module", &art).await?;
        warmed += 1;
    }
    if let Ok(hex) = std::env::var("DAEMON_TRAIN_PREFETCH_MANIFEST") {
        let hash = hash_from_hex(&hex)
            .ok_or_else(|| format!("bad DAEMON_TRAIN_PREFETCH_MANIFEST {hex}"))?;
        let art = ArtifactRef::new(format!("r2://corpus/{}.cbor", hash.to_hex()), hash);
        let manifest_bytes = warm(&cache, &resolver, "manifest", &art).await?;
        let manifest = daemon_vhc_proto::CorpusManifest::from_canonical_bytes(&manifest_bytes)
            .map_err(|e| format!("parse corpus manifest: {e}"))?;

        // The tokenizer artifact is part of the corpus contract — warm it too.
        let tok = &manifest.tokenizer.hash;
        let tok_art = ArtifactRef::new(format!("r2://corpus/{}.json", tok.to_hex()), *tok);
        warm(&cache, &resolver, "tokenizer", &tok_art).await?;
        warmed += 1;

        let (start, count) = std::env::var("DAEMON_TRAIN_PREFETCH_WINDOW")
            .ok()
            .and_then(|w| {
                let (s, c) = w.split_once(':')?;
                Some((s.parse().ok()?, c.parse().ok()?))
            })
            .unwrap_or((0u64, 0u64));
        let indices = shards_covering_window(&manifest, start, count);
        println!(
            "prefetch: corpus window start={start} count={count} -> {}/{} shards",
            indices.len(),
            manifest.shards.len()
        );
        // Shards are chunk-addressed (identity = the chunk fold, not a whole-object hash):
        // fetch each selected shard's bytes as ONE open range, verify every chunk against the
        // manifest's chunk hashes, and warm the cache CHUNK-keyed (chunks are plain blake3
        // content — the cache's native unit; the live fetch path re-verifies regardless).
        for idx in indices {
            let entry = &manifest.shards[idx];
            let url = format!("r2://corpus/{}.bin", entry.shard_hash.to_hex());
            let t0 = std::time::Instant::now();
            let bytes = resolver
                .fetch_range(&url, 0, 0)
                .await
                .map_err(|e| format!("shard[{idx}] ({url}): {e}"))?;
            if bytes.len() as u64 != entry.byte_len {
                return Err(format!(
                    "shard[{idx}] is {} bytes, the manifest pins {}",
                    bytes.len(),
                    entry.byte_len
                ));
            }
            let mut cached_chunks = 0usize;
            for (i, chunk) in bytes
                .chunks(usize::try_from(manifest.chunk_size).unwrap_or(usize::MAX))
                .enumerate()
            {
                let expected = entry
                    .chunk_hashes
                    .get(i)
                    .ok_or_else(|| format!("shard[{idx}] chunk {i} past the manifest list"))?;
                let actual = daemon_vhc_proto::blake3_hash(chunk);
                if actual != *expected {
                    return Err(format!(
                        "shard[{idx}] chunk {i} does not hash to the manifest chunk hash"
                    ));
                }
                if let Err(e) = cache.insert(expected, chunk).await {
                    eprintln!("prefetch: cache insert failed for shard[{idx}] chunk {i}: {e}");
                }
                cached_chunks += 1;
            }
            println!(
                "prefetch: shard[{idx}] bytes={} fold={} chunks={cached_chunks} ms={}",
                bytes.len(),
                entry.shard_hash.to_hex(),
                t0.elapsed().as_millis()
            );
            warmed += 1;
        }
        warmed += 1;
    }
    if warmed == 0 {
        return Err(
            "DAEMON_TRAIN_PREFETCH set but neither DAEMON_TRAIN_PREFETCH_MODULE nor \
             DAEMON_TRAIN_PREFETCH_MANIFEST given — nothing to warm"
                .to_string(),
        );
    }
    println!("prefetch: OK ({warmed} objects verified + cache-warm)");
    Ok(())
}

/// The `.wasm` module bytes from `DAEMON_TRAIN_MODULE` (the dev / node-controlled override), if set.
/// `Some(Err(..))` means the var is set but the read failed.
fn module_from_env() -> Option<Result<Vec<u8>, String>> {
    let path = std::env::var("DAEMON_TRAIN_MODULE").ok()?;
    Some(std::fs::read(&path).map_err(|e| format!("reading module {path}: {e}")))
}

// ==== measured backend selection (the execution half of fleet heterogeneity) ====================
//
// The worker advertises what it can RUN (the structured `BackendCapability` records on the
// probe report) and selects what a join RUNS ON by the measured ladder over exactly those
// records — advertisement and selection cannot diverge. The ladder order is fixed
// cuda → wgpu → cpu; each device rung requires its feature compiled AND a passing runtime
// probe. **There is no silent fallback anywhere**: an explicit `DAEMON_TRAIN_BACKEND` naming a
// lane this build/host cannot serve is a typed refusal (the former quiet stderr-note downgrade
// to the CPU lane is deleted); `DAEMON_TRAIN_BACKEND=cpu` remains the operator's EXPLICIT
// escape hatch (an explicit CPU selection, not a fallback). The selection is recorded in the
// admitted tuple (`backend` + `gpu_index`) at assess, and the join's tuple rederivation reruns
// the identical ladder — a device that disappeared between assess and join rederives a
// different rung and refuses typed (the claim-revalidation flow; the node reassesses).

/// The measured selection one assessment/join runs under: the `BackendKind` wire slug, the
/// device backend CLASS it serves (`device_min.backend_class` vocabulary), and the device
/// placement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct BackendSelection {
    pub(crate) slug: String,
    pub(crate) class: String,
    pub(crate) gpu_index: u32,
}

/// The structured per-backend capability records this worker build advertises (one per
/// compiled lane whose runtime probe found a device, plus the always-present CPU record) —
/// the probe report's `Hardware::backends` and the measured ladder's input.
pub(crate) fn backend_inventory() -> Vec<BackendCapability> {
    // The CPU record first (unconditional — the final ladder rung exists on every build);
    // device records append per compiled lane + probe. Record order is not selection order
    // (the ladder searches by slug). Without a compiled device lane no push follows, hence
    // the gated allow.
    #[cfg_attr(not(any(feature = "wgpu", feature = "cuda")), allow(unused_mut))]
    let mut out = vec![BackendCapability {
        backend: "cpu".to_string(),
        class: "cpu".to_string(),
        adapter: "host".to_string(),
        device_index: 0,
        vram_mb: 0,
        max_alloc_mb: 0,
        shared_mb: 0,
        unified: false,
        ready: true,
    }];
    #[cfg(feature = "cuda")]
    if let Some(p) = daemon_vhc_host::probe::probe_cuda() {
        out.push(BackendCapability {
            backend: "cuda".to_string(),
            class: "cuda".to_string(),
            adapter: p.adapter.clone(),
            device_index: 0,
            vram_mb: p.vram_mb,
            max_alloc_mb: p.max_alloc_mb,
            shared_mb: 0,
            unified: false,
            // The two-leg NVRTC readiness gate: a CUDA device without its staged, driver-
            // matched runtime advertises the hardware but stays unselectable.
            ready: daemon_vhc_host::probe::cuda_nvrtc_ready(),
        });
    }
    #[cfg(feature = "wgpu")]
    if let Some(p) = daemon_vhc_host::probe::probe_wgpu() {
        let class = p.backend.to_lowercase();
        out.push(BackendCapability {
            backend: "wgpu".to_string(),
            // The graphics API the adapter came up on IS the device backend class the run
            // pre-screen matches ("vulkan" / "metal" / "dx12").
            class: class.clone(),
            adapter: p.adapter.clone(),
            device_index: 0,
            // The node's derived usable supply, or nothing.
            //
            // This record used to fall back to the per-buffer ceiling when sysfs had no dedicated
            // figure — which is how a two-gigabyte device supply was once advertised for a card with a
            // thirty-gigabyte budget, in the same report whose top-level figure was right. A ceiling
            // on one allocation is not a quantity of memory, and the two are not interchangeable in
            // either direction. Absence is `0` here (this record's documented "unknown"), which
            // advertises nothing rather than advertising something false.
            vram_mb: derived_supply_mb_for_class(&class),
            max_alloc_mb: p.max_alloc_mb,
            shared_mb: amdgpu_sysfs_mem_mb("mem_info_gtt_total"),
            unified: p.unified,
            ready: true,
        });
    }
    out
}

/// The measured selection ladder (pure — unit-tested over synthetic inventories; the callers
/// feed it [`backend_inventory`]). Fixed rung order cuda → wgpu → cpu; a rung serves only when
/// its record is `ready` and its class passes the run's `backend_class` constraint. `directed`
/// is the operator's explicit lane choice (`DAEMON_TRAIN_BACKEND`): it must name a servable,
/// class-allowed lane or the selection refuses typed — never a fallback. `gpu_index` is the
/// node-directed device placement: naming a device the inventory does not hold refuses typed.
///
/// # Errors
///
/// A human-readable reason carrying the `BackendUnavailable` vocabulary — the caller surfaces
/// it as the typed assess refusal / join error.
pub(crate) fn select_backend(
    inventory: &[BackendCapability],
    directed: Option<&str>,
    allowed_classes: &[String],
    gpu_index: Option<u32>,
) -> Result<BackendSelection, String> {
    let class_allowed =
        |class: &str| allowed_classes.is_empty() || allowed_classes.iter().any(|c| c == class);
    let place = |entry: &BackendCapability| -> Result<BackendSelection, String> {
        let index = match gpu_index {
            None => entry.device_index,
            Some(i) if i == entry.device_index => i,
            Some(i) => {
                return Err(format!(
                    "BackendUnavailable: device placement names device {i}, but backend `{}` \
                     probed device {} only (the placement must name a probed device)",
                    entry.backend, entry.device_index
                ))
            }
        };
        Ok(BackendSelection {
            slug: entry.backend.clone(),
            class: entry.class.clone(),
            gpu_index: index,
        })
    };
    let cpu_selection = |slug: &str| -> Result<BackendSelection, String> {
        if class_allowed("cpu") {
            Ok(BackendSelection {
                slug: slug.to_string(),
                class: "cpu".to_string(),
                gpu_index: 0,
            })
        } else {
            Err(format!(
                "BackendUnavailable: the run constrains backend classes to {allowed_classes:?}, \
                 which excludes the CPU lane"
            ))
        }
    };

    if let Some(slug) = directed {
        // The operator's explicit selection: exactly that lane or a typed refusal.
        return match slug {
            "" | "cpu" => cpu_selection("cpu"),
            // The explicit burn-ndarray lane is CPU-class (one real implementation).
            "burn-ndarray" => cpu_selection("burn-ndarray"),
            other => {
                let entry = inventory
                    .iter()
                    .find(|e| e.backend == other)
                    .ok_or_else(|| {
                        format!(
                            "BackendUnavailable: DAEMON_TRAIN_BACKEND={other} is not servable on \
                         this host (feature not compiled, or no device probed) — there is no \
                         fallback; select `cpu` explicitly for the CPU lane"
                        )
                    })?;
                if !entry.ready {
                    return Err(format!(
                        "BackendUnavailable: backend `{other}` probed a device but is not \
                         ready to serve (runtime not staged)"
                    ));
                }
                if !class_allowed(&entry.class) {
                    return Err(format!(
                        "BackendUnavailable: backend `{other}` serves class `{}`, outside the \
                         run's allowed backend classes {allowed_classes:?}",
                        entry.class
                    ));
                }
                place(entry)
            }
        };
    }

    // The measured ladder: cuda → wgpu → cpu (architecture: fixed order; each rung requires
    // its feature compiled AND a passing runtime probe — encoded here as inventory presence).
    for slug in ["cuda", "wgpu"] {
        if let Some(entry) = inventory.iter().find(|e| e.backend == slug) {
            if entry.ready && class_allowed(&entry.class) {
                return place(entry);
            }
        }
    }
    cpu_selection("cpu").map_err(|_| {
        format!(
            "BackendUnavailable: no compiled backend rung satisfies the run's backend classes \
             {allowed_classes:?} on this host"
        )
    })
}

/// Resolve a selected slug to the engine's `BackendKind` — the feature-compiled half of the
/// rung (the inventory only lists compiled device lanes, so a `None` here is a caller error on
/// the device slugs; `burn-ndarray` additionally requires its feature).
fn kind_for_slug(slug: &str) -> Option<daemon_vhc_host::BackendKind> {
    match slug {
        "cpu" => Some(daemon_vhc_host::BackendKind::Cpu),
        #[cfg(feature = "burn-ndarray")]
        "burn-ndarray" => Some(daemon_vhc_host::BackendKind::BurnNdarray),
        #[cfg(feature = "wgpu")]
        "wgpu" => Some(daemon_vhc_host::BackendKind::Wgpu),
        #[cfg(feature = "cuda")]
        "cuda" => Some(daemon_vhc_host::BackendKind::Cuda),
        _ => None,
    }
}

/// The operator's explicit lane choice (`DAEMON_TRAIN_BACKEND`), if any.
fn directed_backend() -> Option<String> {
    std::env::var("DAEMON_TRAIN_BACKEND").ok()
}

/// The node-directed device placement (`DAEMON_VHC_GPU_INDEX`), if any. A malformed value is a
/// typed error — placement is an owner input, never guessed.
fn directed_gpu_index() -> Result<Option<u32>, String> {
    match std::env::var("DAEMON_VHC_GPU_INDEX") {
        Err(_) => Ok(None),
        Ok(s) => {
            s.trim().parse::<u32>().map(Some).map_err(|e| {
                format!("BackendUnavailable: malformed DAEMON_VHC_GPU_INDEX `{s}`: {e}")
            })
        }
    }
}

/// Run the measured selection for this host + this run's constraints (the assess/join shared
/// path — both stages call this with the same inputs, so the tuple's `backend`/`gpu_index`
/// rederive exactly).
///
/// # Errors
///
/// The typed `BackendUnavailable` reason (no fallback).
pub(crate) fn measured_backend(
    device_min: Option<&daemon_vhc_proto::DeviceMinimums>,
) -> Result<BackendSelection, String> {
    let allowed = device_min
        .map(|m| m.backend_class.clone())
        .unwrap_or_default();
    select_backend(
        &backend_inventory(),
        directed_backend().as_deref(),
        &allowed,
        directed_gpu_index()?,
    )
}

/// The engine config a JOIN runs under: the measured backend selection materialized
/// (`EngineConfig.backend` + `gpu_index`), with the REAL-MODEL sandbox budgets
/// ([`EngineConfig::real_model`]) on the device lanes and under an explicit operator selection
/// (the defaults are tuned for the tiny reference model; a real geometry's fp32 steps trip the 5 s
/// epoch watchdog and its fresh-join seed-init expansion trips the default fuel budget) — and the
/// guest linear-memory cap taken from the **admitted claim** (`claim_host_bytes`, the claim's
/// hard-accountable host tier), so no host constant governs an admitted experiment's memory.
///
/// # Errors
///
/// The typed `BackendUnavailable` reason — the join refuses; there is no fallback.
pub(crate) fn engine_for_join(
    device_min: Option<&daemon_vhc_proto::DeviceMinimums>,
    claim_host_bytes: u64,
) -> Result<EngineConfig, String> {
    let sel = measured_backend(device_min)?;
    let backend = kind_for_slug(&sel.slug).ok_or_else(|| {
        format!(
            "BackendUnavailable: selected backend `{}` has no compiled engine lane in this \
             build",
            sel.slug
        )
    })?;
    let gpu_index = backend.is_device().then_some(sel.gpu_index);
    let base = if backend.is_device() || directed_backend().is_some() {
        EngineConfig::real_model(backend, gpu_index)
    } else {
        EngineConfig {
            backend,
            gpu_index,
            ..EngineConfig::default()
        }
    };
    Ok(base.with_claimed_memory(claim_host_bytes))
}

/// The engine config for the ASSESSMENT instance (the CPU-cheap admission path: manifest/claim
/// evaluation never runs compute, so assessment stays on the CPU engine regardless of the
/// measured selection) — roomy budgets under an explicit operator selection, exactly like the
/// join engine.
pub(crate) fn assess_engine_config() -> EngineConfig {
    if std::env::var_os("DAEMON_TRAIN_BACKEND").is_some() {
        EngineConfig::real_model(daemon_vhc_host::BackendKind::Cpu, None)
    } else {
        EngineConfig::default()
    }
}

/// The host capability vocabulary the probe advertises: the major-2 worlds with the implemented
/// minor (`<world>@<major>:<minor>` — the compatibility surface a run author matches modules
/// against), plus the versioned custom ops the host's registry advertises (`flash_attn@1`, …).
/// The retired `tabi@1` vocabulary is gone — the bridge is a typed `BridgeRetired` refusal.
fn host_ops() -> Vec<String> {
    let minor = daemon_vhc_abi::host_minor_for(daemon_vhc_abi::DA_ABI_MAJOR_V2).unwrap_or(0);
    let mut ops: Vec<String> = [
        daemon_vhc_abi::NS_VHC_V2,
        daemon_vhc_abi::NS_NET_V2,
        daemon_vhc_abi::NS_SYS_V2,
        daemon_vhc_abi::NS_DATA_V2,
        daemon_vhc_abi::NS_COMPUTE_V2,
    ]
    .iter()
    .map(|ns| format!("{ns}:{minor}"))
    .collect();
    ops.extend(
        daemon_vhc_abi::HOST_CUSTOM_OPS
            .iter()
            .map(|op| (*op).to_string()),
    );
    ops
}

pub(crate) fn host_capabilities() -> WorkerCapabilities {
    WorkerCapabilities {
        // The implemented ABI major (ABI §1.6): 2 — the only implemented major. The advertised
        // `ops` are the major-2 worlds at their implemented minor plus the custom-op registry.
        abi_version: daemon_vhc_abi::DA_ABI_MAJOR_V2 as u16,
        ops: host_ops(),
        payload_stores: Vec::new(),
    }
}

/// Total host RAM in MiB — the PORTABLE probe (Linux `/proc`, macOS `sysctl hw.memsize`, Windows
/// `GlobalMemoryStatusEx`). The pre-fix local reader was `/proc`-only, so a macOS/Windows trainer
/// reported `0` and the admission funnel refused the lane RAM floor spuriously.
fn host_ram_mb() -> u64 {
    daemon_vhc_host::probe::host_ram_mb()
}

/// Free disk space in MiB on the filesystem the run actually spills onto — the content cache
/// (corpus shards) or the durable run-state home, falling back to the process cwd — the portable
/// probe (unix `statvfs`, Windows `GetDiskFreeSpaceExW`). The pre-fix worker hardcoded
/// `disk_free_mb: 0` in every `hardware()` arm, so the trainer lane's disk floor was never met
/// off a device that happened to fill it, and macOS/Windows refused `below lane floor: ram/disk`
/// despite hundreds of GiB free.
fn host_disk_free_mb() -> u64 {
    daemon_vhc_host::probe::host_disk_free_mb(&disk_probe_path())
}

/// The path whose filesystem the disk-free probe measures: the first EXISTING ancestor of the
/// content-cache dir, else of the run-state home, else the process cwd (always existing).
/// `statvfs`/`GetDiskFreeSpaceExW` need a live path, and the cache/state dirs may not exist yet at
/// probe time (assess precedes the first fetch).
fn disk_probe_path() -> std::path::PathBuf {
    fn existing_ancestor(p: std::path::PathBuf) -> Option<std::path::PathBuf> {
        let mut cur: Option<&std::path::Path> = Some(p.as_path());
        while let Some(c) = cur {
            if c.exists() {
                return Some(c.to_path_buf());
            }
            cur = c.parent();
        }
        None
    }
    std::env::var_os("DAEMON_VHC_CACHE_DIR")
        .map(std::path::PathBuf::from)
        .and_then(existing_ancestor)
        .or_else(|| {
            std::env::var_os(daemon_vhc_session::journal_home::RUN_DIR_ENV)
                .map(std::path::PathBuf::from)
                .and_then(existing_ancestor)
        })
        .or_else(|| std::env::current_dir().ok())
        .unwrap_or_else(|| std::path::PathBuf::from("."))
}

/// Read an amdgpu sysfs memory-total file for the first DRM card that exposes it, in MiB.
///
/// `file` is `mem_info_vram_total` (dedicated VRAM — the true device lower bound) or
/// `mem_info_gtt_total` (the GTT / unified spillover pool). These are plain byte-count files under
/// `/sys/class/drm/card*/device/` — a legal direct file read in the worker binary (not the node).
/// Returns `0` when no card exposes the file (non-amdgpu / non-Linux), so callers fall back.
///
/// Parsing is delegated to [`daemon_vhc_host::probe::parse_amdgpu_mem_mb`] (unit-tested with
/// fixture strings); this wrapper only does the sysfs directory walk + read.
#[cfg(feature = "wgpu")]
fn amdgpu_sysfs_mem_mb(file: &str) -> u64 {
    let Ok(cards) = std::fs::read_dir("/sys/class/drm") else {
        return 0;
    };
    for entry in cards.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        // Only `cardN` device roots carry `device/mem_info_*` (skip `cardN-<connector>` outputs).
        if !(name.starts_with("card") && name[4..].bytes().all(|b| b.is_ascii_digit())) {
            continue;
        }
        let path = entry.path().join("device").join(file);
        if let Ok(contents) = std::fs::read_to_string(&path) {
            if let Some(mb) = daemon_vhc_host::probe::parse_amdgpu_mem_mb(&contents) {
                if mb > 0 {
                    return mb;
                }
            }
        }
    }
    0
}

/// The host hardware + capability report (§10.2). GPU count / VRAM come from a real wgpu adapter
/// probe + sysfs when the `wgpu` feature is on; a CPU-only build reports `gpus: 0` and the CPU lane.
///
/// **VRAM source (Merge-2 UMA fix).** wgpu has no total-VRAM query and clamps `max_buffer_size` to
/// i32::MAX (2047 MiB) on Linux/Mesa — a per-buffer limit, NOT the memory budget. `vram_mb` now
/// carries the sysfs *dedicated* VRAM (`mem_info_vram_total`, a true lower bound: 4096 MiB on this
/// box) when available, falling back to the `max_buffer_size` proxy only when sysfs is absent. The
/// additive `shared_mb` carries the GTT / unified spillover pool (`mem_info_gtt_total`), which is
/// where an integrated GPU actually pages large tensors.
pub(crate) fn hardware() -> Hardware {
    let ram_mb = host_ram_mb();
    // Windows: the DXGI/D3D12 probe (swarm-windows-vram-design.md §2) is authoritative and needs no
    // wgpu adapter — it queries D3D12 `ARCHITECTURE1.UMA` + DXGI budgets directly.
    #[cfg(windows)]
    {
        if let Some(dl) = daemon_vhc_host::probe::probe_windows_device_limits() {
            return Hardware {
                gpus: 1,
                vram_mb: dl.vram_mb,
                shared_mb: dl.shared_mb,
                ram_mb: if dl.ram_mb > 0 { dl.ram_mb } else { ram_mb },
                backend_lanes: vec!["dx12".to_string(), "vulkan".to_string(), "cpu".to_string()],
                backends: backend_inventory(),
                capabilities: host_capabilities(),
                up_kbps: 0,
                down_kbps: 0,
                disk_free_mb: host_disk_free_mb(),
                throughput_class: "c1".to_string(),
            };
        }
    }
    // macOS: the Metal probe (swarm-macos-uma-findings.md §4) sources the working-set budget +
    // maxBufferLength directly; unified => shared_mb == ram_mb (one DRAM pool).
    #[cfg(target_os = "macos")]
    {
        if let Some(dl) = daemon_vhc_host::probe::probe_macos_device_limits() {
            return Hardware {
                gpus: 1,
                vram_mb: dl.vram_mb,
                shared_mb: dl.shared_mb,
                ram_mb: if dl.ram_mb > 0 { dl.ram_mb } else { ram_mb },
                backend_lanes: vec!["metal".to_string(), "cpu".to_string()],
                backends: backend_inventory(),
                capabilities: host_capabilities(),
                up_kbps: 0,
                down_kbps: 0,
                disk_free_mb: host_disk_free_mb(),
                throughput_class: "c1".to_string(),
            };
        }
    }
    // CUDA lane: the driver exposes total dedicated VRAM directly (`cuDeviceTotalMem`), so a
    // discrete NVIDIA box reports real VRAM + `["cuda","cpu"]` lanes, no UMA.
    // Checked before wgpu so a box built `--features cuda` (RunPod 4090) reports the CUDA lane.
    #[cfg(feature = "cuda")]
    {
        if let Some(p) = daemon_vhc_host::probe::probe_cuda() {
            return Hardware {
                gpus: p.gpus,
                vram_mb: p.vram_mb,
                shared_mb: 0,
                ram_mb,
                backend_lanes: vec!["cuda".to_string(), "cpu".to_string()],
                backends: backend_inventory(),
                capabilities: host_capabilities(),
                up_kbps: 0,
                down_kbps: 0,
                disk_free_mb: host_disk_free_mb(),
                throughput_class: "c1".to_string(),
            };
        }
    }
    #[cfg(feature = "wgpu")]
    {
        if let Some(p) = daemon_vhc_host::probe::probe_wgpu() {
            // The derived usable supply, never the per-buffer ceiling: a ceiling on one allocation
            // says nothing about how much memory the device has, and substituting it here advertised
            // two gigabytes for a card with tens.
            let vram_mb = derived_supply_mb_for_class(&p.backend.to_lowercase());
            let shared_mb = amdgpu_sysfs_mem_mb("mem_info_gtt_total");
            return Hardware {
                gpus: p.gpus,
                vram_mb,
                shared_mb,
                ram_mb,
                backend_lanes: vec!["vulkan".to_string(), "cpu".to_string()],
                backends: backend_inventory(),
                capabilities: host_capabilities(),
                up_kbps: 0,
                down_kbps: 0,
                disk_free_mb: host_disk_free_mb(),
                throughput_class: "c1".to_string(),
            };
        }
    }
    Hardware {
        gpus: 0,
        vram_mb: 0,
        shared_mb: 0,
        ram_mb,
        backend_lanes: vec!["cpu".to_string()],
        backends: backend_inventory(),
        capabilities: host_capabilities(),
        up_kbps: 0,
        down_kbps: 0,
        disk_free_mb: host_disk_free_mb(),
        throughput_class: "c1".to_string(),
    }
}

/// The device budget the admission math is computed against (Merge-2 UMA fix) — post-sunset it
/// feeds the v2 claim funnel's [`daemon_vhc_host::run::DeviceProfile`].
///
/// With the `wgpu` feature + a usable adapter: `vram_mb` = sysfs dedicated VRAM (true lower bound),
/// `shared_mb` = sysfs GTT (the unified spillover pool), `max_alloc_mb` = the wgpu `max_buffer_size`
/// per-buffer ceiling, and `unified` = the adapter's device-type (IntegratedGpu/Cpu). On a unified
/// device the budget math then treats VRAM+GTT+RAM as one physical DRAM pool instead of rejecting
/// against the 2047 MiB per-buffer clamp. Without a GPU, the CPU lane runs in host RAM (no separate
/// VRAM constraint). Unknown dimensions use a large sentinel so an unprobed number never rejects.
pub(crate) fn device_limits() -> DeviceLimits {
    let ram_mb = {
        let r = host_ram_mb();
        if r == 0 {
            UNKNOWN_BUDGET_MB
        } else {
            r
        }
    };
    // Windows / macOS: the platform FFI probes are authoritative and need no wgpu feature.
    #[cfg(windows)]
    {
        if let Some(dl) = daemon_vhc_host::probe::probe_windows_device_limits() {
            return dl;
        }
    }
    #[cfg(target_os = "macos")]
    {
        if let Some(dl) = daemon_vhc_host::probe::probe_macos_device_limits() {
            return dl;
        }
    }
    // CUDA lane: discrete-device budget from the driver's total-VRAM query (24 GB on the
    // 4090), `shared_mb = 0`, `unified = false` — the discrete verdict path.
    #[cfg(feature = "cuda")]
    {
        if let Some(p) = daemon_vhc_host::probe::probe_cuda() {
            return daemon_vhc_host::probe::cuda_device_limits(p.vram_mb, p.max_alloc_mb, ram_mb);
        }
    }
    #[cfg(feature = "wgpu")]
    {
        if let Some(p) = daemon_vhc_host::probe::probe_wgpu() {
            let vram_sysfs = amdgpu_sysfs_mem_mb("mem_info_vram_total");
            // On a unified device without sysfs VRAM, dedicated VRAM is not a meaningful cap; the
            // pool is host RAM, so budget VRAM as RAM. On a discrete device fall back to the
            // per-buffer proxy (the honest lower bound wgpu can give).
            let vram_mb = if vram_sysfs > 0 {
                vram_sysfs
            } else if p.unified {
                ram_mb
            } else {
                p.max_alloc_mb
            };
            return DeviceLimits {
                vram_mb,
                ram_mb,
                max_alloc_mb: p.max_alloc_mb,
                shared_mb: amdgpu_sysfs_mem_mb("mem_info_gtt_total"),
                unified: p.unified,
            };
        }
    }
    DeviceLimits {
        vram_mb: ram_mb,
        ram_mb,
        max_alloc_mb: 0,
        shared_mb: 0,
        unified: false,
    }
}

/// The peer-side re-validation (ABI Draft 3 §1.3; decisions D2/D5):
///
/// 1. **Driver selection first** — [`daemon_vhc_host::select_driver`] runs the normative order
///    (hash-verify before compile → static-import inspection → candidate linker → instantiate →
///    `da_abi` cross-check). Any failure is a **typed admission refusal** returned as an
///    ineligible [`Eligibility`] carrying the split `refusal_code` slug — an `Assessed` outcome,
///    never an `Event::Error` (ABI §1.5). **Since the Phase-E v1 sunset a v1 module lands here
///    with `AbiUnsupportedMajor`** — the retained-v1 assess (import scan → meta pass → autotune
///    verdict) retired with the driver in the same step (decisions D5).
/// 2. A module selecting the **event-loop driver** runs the A2 claim()-admission funnel ([`assess`]):
///    lane floor, restricted-instance `da_manifest`/`da_claim`, lane claim bounds, owner
///    authorization — the ABI §9.3 owner-bracketed order.
///
/// Returns the eligibility verdict plus whether the module selected the **major-2** driver
/// (post-sunset: `true` on every eligible verdict; the flag stays so the `JoinRun` dispatch shape
/// is unchanged on the wire side).
pub(crate) fn assess(
    module: &[u8],
    config: &[u8],
    module_blake3: Option<&[u8; 32]>,
    device_min: Option<&daemon_vhc_proto::DeviceMinimums>,
    envelope_grants: Option<&daemon_vhc_host::run::EnvelopeRoleGrants>,
    tuple_identity: Option<TupleIdentity<'_>>,
) -> Result<(Eligibility, bool), String> {
    // `DAEMON_TRAIN_BACKEND` set ⇒ the roomy 160M-scale budgets (real-scale param layouts exceed
    // the tiny-model defaults). Assessment itself stays on the CPU-cheap path.
    let worker = Worker::new(assess_engine_config()).map_err(|e| format!("engine: {e}"))?;

    // ABI §1.3 steps 1–6: hash-verify → compile+inspect → candidate → instantiate → cross-check.
    match daemon_vhc_host::select_driver(&worker, module, module_blake3) {
        Ok(_sel) => {
            // The major-2 path: the claim()-based admission funnel (ABI §9.3, A2). Selection
            // re-runs inside admit (§9.4 step 1–3 order is normative); the arms below map the
            // funnel outcome onto the typed `Assessed` surface.
            Ok((
                assess_module(
                    &worker,
                    module,
                    config,
                    module_blake3,
                    device_min,
                    envelope_grants,
                    tuple_identity,
                ),
                true,
            ))
        }
        Err(refusal) => Ok((
            Eligibility {
                eligible: false,
                reasons: vec![refusal.to_string()],
                headroom: Vec::new(),
                refusal_code: Some(refusal.code.slug().to_string()),
                admitted_tuple: None,
            },
            false,
        )),
    }
}

/// The non-artifact identity fields the admitted tuple needs beyond what the module/config/grants
/// bytes provide: the run's genesis hash, the admitted role, and the role-instance incarnation.
#[derive(Clone, Copy)]
pub(crate) struct TupleIdentity<'a> {
    /// The run identity — the genesis-envelope hash.
    pub(crate) genesis_hash: [u8; 32],
    /// The envelope-level role admitted.
    pub(crate) role: &'a str,
    /// The role-instance incarnation.
    pub(crate) incarnation: u64,
}

/// Everything a production role-session join binds beyond the providers: the driver-facing run
/// binding (identity, certified per-run signing seed, admitted config + grants + quotas) and the
/// inbound trust configuration (own certificate, genesis-named trusted bases).
pub(crate) struct RoleBinding {
    pub(crate) run: daemon_vhc_host::run::RunConfig,
    pub(crate) own_cert: daemon_vhc_proto::RunKeyCertificate,
    pub(crate) trusted_bases: Vec<daemon_vhc_proto::PeerId>,
    /// The quotas the admission funnel derived — the grant-containment baseline a later live
    /// module switch compares its re-admitted quotas against (fail closed, ABI §10.3 step 3).
    pub(crate) quotas: Option<daemon_vhc_proto::AdmittedQuotas>,
    /// The admitted claim's hard-accountable HOST tier: the guest linear-memory ceiling the
    /// experiment declared for this exact config (`decl_for_config` -> `da_claim`), already checked
    /// against the lane's claim bounds and the owner's caps by the funnel. The join engine enforces
    /// it as the sandbox cap, so no host constant governs an admitted run's memory.
    pub(crate) claim_host_bytes: u64,
}

/// Author the role-session binding for a resolved genesis run: re-run the admission funnel over
/// the exact artifacts (the §9.4 join-time re-check — same lane, same grants derivation as
/// assess), resolve the certified per-run identity READ-ONLY against the node-provisioned
/// keystore, and assemble the run config with the admitted quotas applied.
///
/// Identity custody: the NODE mints the per-run key and issues its certificate at join
/// authorship (the base identity never leaves the node process); this worker resolves both by
/// reference and refuses typed when either is absent or the certificate does not bind the exact
/// execution identity about to run. The worker NEVER mints and NEVER touches `base.key`.
pub(crate) fn role_binding(
    resolved: &ResolvedRun,
    genesis: &GenesisRun,
    run_id: &str,
    incarnation: u64,
) -> Result<RoleBinding, String> {
    use daemon_vhc_session::keystore::VhcKeystore;

    // Admission re-check on the CPU-cheap path (assessment parity: same engine shape).
    let worker =
        Worker::new(assess_engine_config()).map_err(|e| format!("admission engine: {e}"))?;
    let module_hash = resolved
        .module_blake3
        .unwrap_or_else(|| *blake3::hash(&resolved.module).as_bytes());
    let envelope_grants = resolved.envelope_grants();
    let default_role = daemon_vhc_proto::genesis::RoleGrants::default();
    let role_grants = envelope_grants
        .as_ref()
        .map_or(&default_role, |eg| &eg.grants);
    let grants = derive_grants(&worker, &resolved.module, role_grants)?;
    let hw = hardware();
    let dl = device_limits();
    let device = admission_device_profile(&hw, &dl);
    let owner = daemon_vhc_host::run::OwnerPolicy {
        participation_enabled: true,
        vram_cap_bytes: 0,
        host_cap_bytes: 0,
    };
    let admission = daemon_vhc_host::run::admit(
        &worker,
        &resolved.module,
        Some(&module_hash),
        &resolved.config,
        &grants,
        &selected_lane(),
        &device,
        &owner,
        resolved.device_min.as_ref(),
        envelope_grants.as_ref(),
    )
    .map_err(|refusal| format!("join re-admission: {refusal}"))?;

    // Certified per-run identity, READ-ONLY from the node-provisioned keystore (a path
    // reference — key material never rides the command wire; the node minted the key and
    // issued the certificate at join authorship). Absence is a typed refusal: a worker never
    // mints identity, and base.key custody stays with the node.
    let keystore = VhcKeystore::from_env().map_err(|e| format!("identity store: {e}"))?;
    let genesis_hash = *genesis.frozen.run_id();
    let run_key = keystore
        .existing_run_signing_key(run_id, &genesis.worker_role, incarnation)
        .map_err(|e| format!("run key: {e}"))?
        .ok_or_else(|| {
            format!(
                "no per-run identity was provisioned for `{run_id}` role `{}` incarnation \
                 {incarnation} (the node mints keys and issues certificates at join authorship; \
                 the worker never mints)",
                genesis.worker_role
            )
        })?;
    let cert = keystore
        .run_certificate(run_id, &genesis.worker_role, incarnation)
        .map_err(|e| format!("run certificate: {e}"))?
        .ok_or_else(|| {
            format!(
                "no certificate was provisioned for `{run_id}` role `{}` incarnation {incarnation}",
                genesis.worker_role
            )
        })?;
    // The provisioned certificate must bind EXACTLY the execution identity about to run.
    let expected_scope = daemon_vhc_proto::CertScope {
        run_id: genesis_hash,
        epoch: 0,
        role: genesis.worker_role.clone(),
        instance: incarnation,
        module_hash: daemon_vhc_proto::Hash(module_hash),
    };
    if cert.body.scope != expected_scope {
        return Err(format!(
            "the provisioned certificate binds a different execution identity than this join \
             (certificate scope {:?}; joining {:?}) — the node reassesses/reprovisions",
            cert.body.scope, expected_scope
        ));
    }
    if cert.body.run_key != daemon_vhc_proto::peer_id(&run_key) {
        return Err(
            "the provisioned certificate does not certify the provisioned per-run key".into(),
        );
    }
    cert.verify_chain()
        .map_err(|e| format!("provisioned certificate chain: {e}"))?;

    let identity = daemon_vhc_host::run::RunIdentity {
        run_id: genesis_hash.0,
        epoch: 0,
        role: genesis.worker_role.clone(),
        instance: incarnation,
        module: module_hash,
    };
    let mut run = daemon_vhc_host::run::RunConfig::new(
        identity,
        run_key.to_bytes(),
        resolved.config.clone(),
        grants,
    );
    run.claim_bytes = admission.claim_bytes.clone();
    run.manifest_bytes = admission.manifest_bytes.clone();
    // The exactly-metered host-side staging ceiling, from the claim's peak tier (ABI §9.1): the
    // bytes a module stages host-side (`stage_state`, `create_from`, `buffer_append` — an outgoing
    // committed update is built through the last of these) above its linear-memory floor, which the
    // engine caps separately from the claim's hard tier. Previously left uncapped on this path;
    // wiring it is the other half of "the claim is what governs, not a host constant".
    run.hard_accountable_host_bytes = admission.claim.declared_peak.host;
    admission.apply_quotas(&mut run);
    // Provision the state plane from the genesis state contract (§6.3): the run-pinned
    // `state_chunk_size` the streamed det-lane fold runs under (`None` = no host-side state).
    if let Some(sc) = &genesis.env.state_contract {
        run.state_chunk_size = sc.chunk_size;
        // Disk-back the state store under the durable run-state home (design §8.1): the retained
        // det-lane roots (≈14.65 GiB at the ceremony tier) live on disk, not the memory-floor
        // peer's unified RAM. Absent a run-state home (ephemeral/dev), the store keeps its
        // resident RAM backing.
        run.state_dir = daemon_vhc_session::journal_home::run_dir_from_env().map(|root| {
            daemon_vhc_session::journal_home::state_dir(
                &root,
                &genesis.env.run.run_label,
                &genesis.worker_role,
                incarnation,
            )
        });
    }

    // Inbound trust: the base identities the genesis names — never ambient config.
    let trusted_bases = daemon_vhc_session::identity::TrustedBases::from_genesis(&genesis.env)
        .bases()
        .to_vec();
    Ok(RoleBinding {
        run,
        own_cert: cert,
        trusted_bases,
        quotas: admission.quotas.clone(),
        claim_host_bytes: admission.claim.hard_accountable.host,
    })
}

/// The dev / node-controlled module-source override for a live module switch (the upgrade-time
/// peer of `DAEMON_TRAIN_MODULE`): a filesystem path whose bytes MUST hash to the committed
/// target module (verified at pre-flight — the override substitutes the artifact fetch, never
/// the hash pin). Absent ⇒ the session resolves the target by content address from its bound
/// stores (or the store-fetch path resolves it here under the networked build).
pub(crate) const SWITCH_MODULE_ENV: &str = "DAEMON_VHC_SWITCH_MODULE";

/// Resolve a live-upgrade TARGET module's bytes by its committed content hash (the pre-switch
/// assessment's artifact source, hash-verified whichever source serves):
///
/// 1. the explicit [`SWITCH_MODULE_ENV`] override (dev / node-controlled),
/// 2. the run's filesystem content plane (the node-delivered shared payload root / run-state
///    dir — the single-host topology, where peers seed each other's content-addressed objects),
/// 3. the networked store-fetch path (content cache + presigned store) under `vhc-net`.
///
/// No resolvable source is a typed refusal — a switch target that cannot be fetched is never
/// guessed at.
async fn resolve_switch_module(run_label: &str, new_module: [u8; 32]) -> Result<Vec<u8>, String> {
    let verify = |bytes: Vec<u8>, source: &str| {
        if *blake3::hash(&bytes).as_bytes() == new_module {
            Ok(bytes)
        } else {
            Err(format!(
                "the switch target resolved from {source} does not hash to the committed module"
            ))
        }
    };
    if let Ok(path) = std::env::var(SWITCH_MODULE_ENV) {
        let bytes =
            std::fs::read(&path).map_err(|e| format!("reading switch module {path}: {e}"))?;
        return verify(bytes, "the explicit override");
    }
    // The filesystem content plane, exactly as the live attach roots it (shared payload root
    // override first, else the run's own state dir).
    let fs_dir = daemon_vhc_session::journal_home::payload_dir_from_env()
        .map(|shared| daemon_vhc_session::journal_home::payload_dir(&shared, run_label))
        .or_else(|| {
            daemon_vhc_session::journal_home::run_dir_from_env()
                .map(|root| daemon_vhc_session::journal_home::payload_dir(&root, run_label))
        });
    if let Some(dir) = fs_dir {
        use daemon_vhc_net::ContentStore as _;
        let store = daemon_vhc_net::FsContentStore::open(&dir)
            .map_err(|e| format!("fs content store {}: {e}", dir.display()))?;
        if let Ok(bytes) = store.get_content(&daemon_vhc_proto::Hash(new_module)).await {
            return verify(bytes, "the filesystem content plane");
        }
    }
    #[cfg(feature = "vhc-net")]
    {
        use daemon_vhc_net::ContentStore as _;
        let ctx = store_fetch_context();
        let cache = open_content_cache(&ctx)?;
        if let Some(bytes) = cache
            .get(&daemon_vhc_net::ContentHash::new(new_module))
            .await
            .map_err(|e| e.to_string())?
        {
            return verify(bytes, "the content cache");
        }
        if let Some(base) = &ctx.presign_base {
            let egress = daemon_egress::EgressClient::new(daemon_egress::EgressConfig::default())
                .map_err(|e| format!("egress client: {e}"))?;
            let presign_egress =
                daemon_egress::EgressClient::new(daemon_egress::EgressConfig::default())
                    .map_err(|e| format!("presign egress client: {e}"))?;
            let presign = daemon_vhc_net::HttpPresignClient::new(presign_egress, base.clone());
            let store = daemon_vhc_net::R2Store::new(
                presign,
                egress,
                daemon_vhc_net::RunId::new(run_label),
            );
            if let Ok(bytes) = store.get_content(&daemon_vhc_proto::Hash(new_module)).await {
                return verify(bytes, "the presigned content store");
            }
        }
    }
    Err(format!(
        "switch target {} is not resolvable from any worker-side source (no override, not on \
         the filesystem content plane, not in the content cache/store)",
        daemon_vhc_proto::Hash(new_module).to_hex()
    ))
}

/// Assess a live-upgrade TARGET (ABI §10.3 pre-switch assessment): resolve the hash-pinned
/// target bytes, re-derive the grants document against the committed record's grants anchor,
/// run the SAME claim admission funnel the join uses (empty config — upgrade records pin the
/// module and grants; config carriage arrives when they carry one), and produce the post-switch
/// admitted tuple, its claim hash computed here over the target's re-evaluated claim (the node
/// never touches module bytes). The session's pre-fence checks re-verify all of it fail-closed.
pub(crate) async fn assess_switch(
    resolved: &ResolvedRun,
    genesis: &GenesisRun,
    target: &daemon_vhc_session::protocol::SwitchTarget,
) -> Result<Eligibility, String> {
    let ineligible = |reason: String, code: Option<&str>| Eligibility {
        eligible: false,
        reasons: vec![reason],
        headroom: Vec::new(),
        refusal_code: code.map(str::to_string),
        admitted_tuple: None,
    };
    let run_label = genesis.env.run.run_label.clone();
    let bytes = match resolve_switch_module(&run_label, target.new_module).await {
        Ok(bytes) => bytes,
        Err(reason) => return Ok(ineligible(reason, Some("SwitchTargetUnresolvable"))),
    };
    let worker = Worker::new(assess_engine_config()).map_err(|e| format!("engine: {e}"))?;
    // The admitted role config carries UNCHANGED across the switch (upgrade records pin module
    // + grants; config carriage arrives when records carry one) — the target is assessed, and
    // later re-admitted at the fence, with the exact config the running instance was admitted
    // under, so the migrated instance initializes like its predecessor.
    let config = resolved.config.clone();
    // The grants anchor (§10.3 step 3): the re-derived document must hash to the committed
    // record's grants_hash — refused typed here, and re-checked by the session at the fence.
    let role_grants = &genesis.env.roles[&genesis.worker_role].grants;
    let grants = match derive_grants(&worker, &bytes, role_grants) {
        Ok(g) => g,
        Err(reason) => return Ok(ineligible(reason, None)),
    };
    if *blake3::hash(&grants).as_bytes() != target.grants_hash {
        return Ok(ineligible(
            "re-derived grants do not match the committed record's grants anchor".into(),
            Some("SwitchGrantsAnchorMismatch"),
        ));
    }
    let selection = match measured_backend(resolved.device_min.as_ref()) {
        Ok(sel) => sel,
        Err(reason) => return Ok(ineligible(reason, Some("BackendUnavailable"))),
    };
    let hw = hardware();
    let dl = device_limits();
    let device = admission_device_profile(&hw, &dl);
    let owner = daemon_vhc_host::run::OwnerPolicy {
        participation_enabled: true,
        vram_cap_bytes: 0,
        host_cap_bytes: 0,
    };
    let envelope_grants =
        daemon_vhc_host::run::EnvelopeRoleGrants::from_genesis(&genesis.env, &genesis.worker_role);
    match daemon_vhc_host::run::admit(
        &worker,
        &bytes,
        Some(&target.new_module),
        // The session's re-admission at the fence evaluates the same carried config, so the
        // claim bytes — and the tuple's claim hash — match by construction.
        &config,
        &grants,
        &selected_lane(),
        &device,
        &owner,
        resolved.device_min.as_ref(),
        envelope_grants.as_ref(),
    ) {
        Ok(admission) => Ok(Eligibility {
            eligible: true,
            reasons: vec![format!(
                "switch target admitted for epoch {}: device {} B / host {} B",
                target.epoch,
                admission.claim.device_total(),
                admission.claim.host_total(),
            )],
            headroom: Vec::new(),
            refusal_code: None,
            admitted_tuple: Some(daemon_vhc_session::protocol::AdmittedTuple {
                module_hash: target.new_module,
                config_hash: *blake3::hash(&config).as_bytes(),
                grants_hash: target.grants_hash,
                claim_hash: *blake3::hash(&admission.claim_bytes).as_bytes(),
                genesis_hash: genesis.frozen.run_id().0,
                role: genesis.worker_role.clone(),
                // 0 = unassigned; the node mints the post-switch incarnation and stamps it into
                // the tuple it delivers with SwitchModule (never here).
                incarnation: 0,
                device_profile_rev: 0,
                owner_policy_rev: 0,
                backend: selection.slug.clone(),
                gpu_index: selection.gpu_index,
            }),
        }),
        Err(refusal) => Ok(ineligible(
            format!("switch target re-admission refused: {refusal}"),
            None,
        )),
    }
}

/// Author the live-switch binding (the worker's pre-flight half of ABI §10.3): resolve the
/// node-provisioned POST-SWITCH identity read-only from the keystore (the node minted the new
/// incarnation's key and re-issued its certificate before sending the command — the worker never
/// mints), resolve the target module bytes where a worker-side source exists, and assemble the
/// admission inputs the session re-runs owner law with ahead of the fence. Every failure is a
/// typed refusal with the running instance untouched.
#[allow(clippy::too_many_arguments)]
pub(crate) fn switch_binding(
    genesis: &GenesisRun,
    run_id: &str,
    epoch: u64,
    new_module: [u8; 32],
    grants_hash: [u8; 32],
    deadline_ms: u64,
    tuple: daemon_vhc_session::protocol::AdmittedTuple,
    old_incarnation: u64,
    config: Vec<u8>,
) -> Result<daemon_vhc_session::role_session::SwitchBinding, String> {
    use daemon_vhc_session::keystore::VhcKeystore;

    let genesis_hash = *genesis.frozen.run_id();
    if tuple.genesis_hash != genesis_hash.0 {
        return Err("the post-switch tuple's genesis hash is not this run's".into());
    }
    if tuple.role != genesis.worker_role {
        return Err(format!(
            "the post-switch tuple's role `{}` is not the held role `{}`",
            tuple.role, genesis.worker_role
        ));
    }
    if tuple.incarnation <= old_incarnation {
        return Err(format!(
            "the post-switch incarnation {} does not supersede the running incarnation \
             {old_incarnation} (incarnations are never reused)",
            tuple.incarnation
        ));
    }

    // The re-issued identity, READ-ONLY from the node-provisioned keystore ([LT-8]: the node
    // mints keys and issues certificates; absence is a typed refusal).
    let keystore = VhcKeystore::from_env().map_err(|e| format!("identity store: {e}"))?;
    let run_key = keystore
        .existing_run_signing_key(run_id, &genesis.worker_role, tuple.incarnation)
        .map_err(|e| format!("post-switch run key: {e}"))?
        .ok_or_else(|| {
            format!(
                "no per-run identity was provisioned for `{run_id}` role `{}` incarnation {} \
                 (the node mints the post-switch key and re-issues its certificate before the \
                 switch; the worker never mints)",
                genesis.worker_role, tuple.incarnation
            )
        })?;
    let cert = keystore
        .run_certificate(run_id, &genesis.worker_role, tuple.incarnation)
        .map_err(|e| format!("post-switch certificate: {e}"))?
        .ok_or_else(|| {
            format!(
                "no certificate was re-issued for `{run_id}` role `{}` incarnation {}",
                genesis.worker_role, tuple.incarnation
            )
        })?;
    let expected_scope = daemon_vhc_proto::CertScope {
        run_id: genesis_hash,
        epoch,
        role: genesis.worker_role.clone(),
        instance: tuple.incarnation,
        module_hash: daemon_vhc_proto::Hash(new_module),
    };
    if cert.body.scope != expected_scope {
        return Err(format!(
            "the re-issued certificate binds a different execution identity than this switch \
             (certificate scope {:?}; switching to {:?}) — the node re-provisions",
            cert.body.scope, expected_scope
        ));
    }
    if cert.body.run_key != daemon_vhc_proto::peer_id(&run_key) {
        return Err(
            "the re-issued certificate does not certify the provisioned per-run key".into(),
        );
    }
    cert.verify_chain()
        .map_err(|e| format!("re-issued certificate chain: {e}"))?;

    // Target module bytes, when a worker-side source exists. The explicit override substitutes
    // the artifact fetch, never the hash pin (verified right here); with no worker-side source
    // the session resolves the target by content address from its bound stores.
    let module_bytes = match std::env::var(SWITCH_MODULE_ENV) {
        Ok(path) => {
            let bytes =
                std::fs::read(&path).map_err(|e| format!("reading switch module {path}: {e}"))?;
            if *blake3::hash(&bytes).as_bytes() != new_module {
                return Err(format!(
                    "the switch module override at {path} does not hash to the committed target"
                ));
            }
            Some(bytes)
        }
        Err(_) => None,
    };

    // Admission inputs: the same lane/device/owner surface the join evaluated, plus the genesis
    // role grants the new grants document derives from.
    let hw = hardware();
    let dl = device_limits();
    let device = admission_device_profile(&hw, &dl);
    let owner = daemon_vhc_host::run::OwnerPolicy {
        participation_enabled: true,
        vram_cap_bytes: 0,
        host_cap_bytes: 0,
    };

    // The seam journal (§8.1): with a durable run-state home, the switch CONTINUES the retiring
    // incarnation's file series (segment roll under the new identity); the referenceless
    // in-process seat records in memory.
    let journal: daemon_vhc_session::role_session::SeamJournal =
        match daemon_vhc_session::journal_home::run_dir_from_env() {
            Some(root) => {
                let dir = daemon_vhc_session::journal_home::journal_dir(
                    &root,
                    run_id,
                    &genesis.worker_role,
                    old_incarnation,
                );
                let key = *keystore
                    .journal_sidecar_key()
                    .map_err(|e| format!("seam journal: sidecar key: {e}"))?
                    .bytes();
                Box::new(move |identity: &daemon_vhc_host::run::RunIdentity| {
                    daemon_vhc_session::journal_home::DurableSink::open_continuation(
                        &dir, identity, key,
                    )
                    .map(|s| Box::new(s) as Box<dyn daemon_vhc_host::run::JournalSink>)
                    .map_err(|e| format!("seam journal open {}: {e}", dir.display()))
                })
            }
            None => Box::new(|_: &daemon_vhc_host::run::RunIdentity| {
                Ok(Box::new(daemon_vhc_host::run::MemorySink::new())
                    as Box<dyn daemon_vhc_host::run::JournalSink>)
            }),
        };

    Ok(daemon_vhc_session::role_session::SwitchBinding {
        epoch,
        new_module,
        grants_hash,
        tuple,
        module_bytes,
        // The admitted role config carries UNCHANGED across the switch (upgrade records pin
        // module + grants; config carriage arrives when records carry one): the migrated
        // instance initializes exactly like its predecessor.
        config,
        signing_seed: run_key.to_bytes(),
        own_cert: cert,
        role_grants: genesis.env.roles[&genesis.worker_role].grants.clone(),
        envelope_grants: daemon_vhc_host::run::EnvelopeRoleGrants::from_genesis(
            &genesis.env,
            &genesis.worker_role,
        ),
        lane: selected_lane(),
        device,
        owner,
        journal,
        deadline_ms,
        migrate_fuel: None,
    })
}

/// The owner's node-side lane configuration seam (§9.6: "numbers are deployment config").
/// Until the node client carries lane config, `DAEMON_VHC_LANE_GPU_OPTIONAL=1` selects a
/// CPU-admitting dev/t2 lane (GPU optional, no device floors, the same claim bounds) — the
/// owner's explicit choice, exactly like `DAEMON_TRAIN_BACKEND=cpu` on the v1 path. Shared by
/// assess and the v2 join so both stages evaluate the identical lane (§9.4 step-10 re-check).
pub(crate) fn selected_lane() -> daemon_vhc_host::run::ParticipationLane {
    if std::env::var_os("DAEMON_VHC_LANE_GPU_OPTIONAL").is_some_and(|v| v == "1") {
        daemon_vhc_host::run::ParticipationLane {
            gpu: 1,
            vram_bytes: 0,
            ram_bytes: 0,
            disk_bytes: 0,
            ..daemon_vhc_host::run::ParticipationLane::trainer_launch_defaults()
        }
    } else {
        daemon_vhc_host::run::ParticipationLane::trainer_launch_defaults()
    }
}

/// The major-2 assess arm: the owner-bracketed claim()-admission funnel (ABI §9.3;
/// `daemon_vhc_host::run::admission::admit`), mapped onto the typed `Assessed` surface.
///
/// Funnel inputs at the worker seam:
/// - **Stage 1 (owner participation)** is `true` here by construction: the node client — the
///   owner's agent — gates participation BEFORE issuing `AssessRun` (architecture §3.5 stage 1
///   lives node-side; the funnel function keeps the stage so host-level tests exercise it).
/// - **Stage 2 lane**: [`selected_lane`] — the ratified launch **Trainer** profile with its
///   deployment-config defaults (GPU required, the 16 GiB-class floor). On a CPU-only box a v2
///   module is refused "below lane floor" — the ratified behavior (Trainer is the only enabled
///   lane; numbers are node configuration, overridable when node-side lane config lands).
/// - **Stage 4.0 envelope grants** (D1): on the genesis path the worker role's grant list is
///   intersected tighten-only against the lane; `None` on the v1-envelope path.
/// - **Stage 5 owner caps** are uncapped at assess time: the standing resource policy
///   (`JoinPolicy{vram_cap_mb, …}`) arrives with `JoinRun`, where the join-time re-check judges
///   it (§9.4 step 10 re-runs stage 5 against the recorded claim).
fn assess_module(
    worker: &Worker,
    module: &[u8],
    config: &[u8],
    module_blake3: Option<&[u8; 32]>,
    device_min: Option<&daemon_vhc_proto::DeviceMinimums>,
    envelope_grants: Option<&daemon_vhc_host::run::EnvelopeRoleGrants>,
    tuple_identity: Option<TupleIdentity<'_>>,
) -> Eligibility {
    let lane = selected_lane();
    let hw = hardware();
    let dl = device_limits();
    // The measured backend selection (the execution half of eligibility): the ladder over the
    // advertised inventory, constrained by the run's `backend_class` and the operator's
    // explicit lane choice. No servable rung is a typed refusal — never a silent CPU
    // admission. The selection is stamped into the admitted tuple; the join reruns this exact
    // call and compares (device-claim revalidation).
    let selection = match measured_backend(device_min) {
        Ok(sel) => sel,
        Err(detail) => {
            return Eligibility {
                eligible: false,
                reasons: vec![detail],
                headroom: Vec::new(),
                refusal_code: Some("BackendUnavailable".to_string()),
                admitted_tuple: None,
            }
        }
    };
    let device = admission_device_profile(&hw, &dl);
    let owner = daemon_vhc_host::run::OwnerPolicy {
        participation_enabled: true,
        vram_cap_bytes: 0,
        host_cap_bytes: 0,
    };
    // The complete §2.6 grants document: the SAME deterministic derivation the v2 join uses
    // (worlds the module links ∪ the genesis role grant list), so assess and join evaluate
    // byte-identical (config, grants) pairs (§9.4 pinning). The no-envelope path authors from an
    // empty role grant (worlds still covered from the module's imports).
    let default_role = daemon_vhc_proto::genesis::RoleGrants::default();
    let role_grants = envelope_grants.map_or(&default_role, |eg| &eg.grants);
    let grants = match derive_grants(worker, module, role_grants) {
        Ok(g) => g,
        Err(detail) => {
            return Eligibility {
                eligible: false,
                reasons: vec![detail],
                headroom: Vec::new(),
                refusal_code: None,
                admitted_tuple: None,
            }
        }
    };
    match daemon_vhc_host::run::admit(
        worker,
        module,
        module_blake3,
        config,
        &grants,
        &lane,
        &device,
        &owner,
        device_min,
        // The envelope-v2 role grants (D1 deliverable 4): present on the genesis path — the
        // worker role's grant list from the genesis role set, intersected tighten-only against
        // the lane at stage 4.0 (mixed-fleet retired-native-coordinator). `None` on the v1-envelope path, where the
        // funnel's pre-D0 defaults stand.
        envelope_grants,
    ) {
        Ok(admission) => {
            // The immutable admitted tuple this assessment produced (architecture §6.3): the exact
            // assessed identity join rederives and compares. The artifact-addressed fields are hashes
            // of the very bytes admitted; the measured backend placement is the selection above
            // (rederiving differently at join = the device inventory changed = typed refusal);
            // the node stamps its device-profile / owner-policy revisions.
            let admitted_tuple =
                tuple_identity.map(|id| daemon_vhc_session::protocol::AdmittedTuple {
                    module_hash: module_blake3
                        .copied()
                        .unwrap_or_else(|| *blake3::hash(module).as_bytes()),
                    config_hash: *blake3::hash(config).as_bytes(),
                    grants_hash: *blake3::hash(&grants).as_bytes(),
                    claim_hash: *blake3::hash(&admission.claim_bytes).as_bytes(),
                    genesis_hash: id.genesis_hash,
                    role: id.role.to_string(),
                    incarnation: id.incarnation,
                    device_profile_rev: 0,
                    owner_policy_rev: 0,
                    backend: selection.slug.clone(),
                    gpu_index: selection.gpu_index,
                });
            Eligibility {
                eligible: true,
                reasons: vec![format!(
                    "major-2 claim admitted: device {} B / host {} B (disjoint tier sums), \
                     pressure order {:?}",
                    admission.claim.device_total(),
                    admission.claim.host_total(),
                    admission.claim.under_pressure,
                )],
                // The legacy declared tiers. A certification-minor admission additionally reports
                // the composed reservation under the `reservation_*` keys, which the node's ledger
                // projection prefers — the owner's memory charge and the governor's occupancy
                // reservation being the same reservation seen twice, read through one spelling
                // rather than two.
                headroom: vec![
                    (
                        "claim_device_bytes".to_string(),
                        admission.claim.device_total() as i64,
                    ),
                    (
                        "claim_host_bytes".to_string(),
                        admission.claim.host_total() as i64,
                    ),
                ],
                refusal_code: None,
                admitted_tuple,
            }
        }
        Err(refusal) => Eligibility {
            eligible: false,
            reasons: vec![refusal.to_string()],
            headroom: Vec::new(),
            refusal_code: refusal.code.map(|c| c.slug().to_string()),
            admitted_tuple: None,
        },
    }
}
/// The **backend implementation revision record** this binary makes about itself
/// (architecture §9.7 `[PC-10]`(1); `[RC-4]`'s revision binding).
///
/// It is assembled from values the probe path has already computed and, until now, discarded to a
/// `Debug` print. That print contained real information no code path could act on: nothing parsed
/// it, nothing carried it into the admitted tuple, and nothing could compare it to the revision
/// range a Backend Execution Profile names — so the revision binding was unenforceable and
/// "implementation identity is reported correctly" was untestable, because there was no report.
///
/// `produced_by` is [`ProducedBy::WorkerProbePath`] because this **is** the running binary's own
/// statement about itself, which is the only provenance admissible as admission evidence.
///
/// Fields this build cannot yet observe arrive as typed unavailability with an accurate reason
/// rather than as a value: the adapter UUID and the several driver numberings live on the
/// framework's adapter info, which the probe does not currently retain. Closing those is a probe
/// change, and until it lands `revision_signal()` falls back to the OS build — which is the
/// documented behaviour for a backend whose framework supplies no driver revision at all, and is
/// honest here rather than convenient.
#[must_use]
pub(crate) fn revision_record(
    capability: &BackendCapability,
) -> daemon_vhc_resource::BackendImplementationRevision {
    use daemon_vhc_resource::{
        AdapterDeviceType, AdapterIdentity, ApiSelectionSource, BackendClass, ComputeFramework,
        ComputeStackIdentity, DriverRevision, Maybe, OsFamily, PlatformApi, ProbeObservation,
        Unavailable,
    };

    let (class, api) = match capability.class.as_str() {
        "vulkan" => (BackendClass::Vulkan, PlatformApi::Vulkan),
        "metal" => (BackendClass::Metal, PlatformApi::Metal),
        "dx12" => (BackendClass::Dx12, PlatformApi::D3d12),
        "cuda" => (BackendClass::Cuda, PlatformApi::Cuda),
        _ => (BackendClass::Cpu, PlatformApi::None),
    };

    // What the framework's adapter info supplied — **only where that adapter serves THIS lane**.
    //
    // The graphics probe brings up one adapter under one selected API, so its findings describe one
    // lane and no other. Filling every lane's record from it put this box's Vulkan adapter identity
    // and its RADV driver strings into the CPU-lane record, which correctly reported a CPU class and
    // a software rasterizer beside them. A Vulkan profile naming a `Mesa 25.2.6` driver range would
    // then have matched the CPU-lane record: a device profile authenticating against a CPU lane, the
    // silent-device-fallback failure arriving through the authentication path rather than around it.
    //
    // Absent-for-this-lane is a *typed* absence, not a gap: a CPU lane has no device adapter and no
    // vendor driver, so there is nothing to report and no policy should tolerate a substitute.
    let probed: Option<ProbedAdapter> = probed_adapter().filter(|p| p.serves(class));
    let family = if cfg!(target_os = "linux") {
        OsFamily::Linux
    } else if cfg!(target_os = "macos") {
        OsFamily::Macos
    } else {
        OsFamily::Windows
    };

    let observation = ProbeObservation {
        backend_class: class,
        adapter_name: capability.adapter.clone(),
        device_type: if class == BackendClass::Cpu {
            AdapterDeviceType::Cpu
        } else if capability.unified {
            AdapterDeviceType::IntegratedGpu
        } else {
            AdapterDeviceType::DiscreteGpu
        },
        // Determined, not assumed. A CPU class is one definitionally; otherwise the framework probe's
        // own determination stands. `None` survives only for a platform that genuinely said nothing,
        // where the assembly is conservative so a device lane refuses rather than quietly running on
        // a rasterizer.
        is_software_rasterizer: if class == BackendClass::Cpu {
            Some(true)
        } else {
            probed.as_ref().map(|p| p.is_software)
        },
        identity: AdapterIdentity {
            vendor_id: probed
                .as_ref()
                .and_then(|p| p.vendor_id)
                .map_or_else(|| unobserved_for(class), Maybe::Available),
            device_id: probed
                .as_ref()
                .and_then(|p| p.device_id)
                .map_or_else(|| unobserved_for(class), Maybe::Available),
            // The bus address and the UUID are not on the framework's adapter info at all; the
            // platform-specific paths that carry them are a separate probe. On a CPU lane there is
            // no adapter to have them.
            pci_bus_id: unobserved_for(class),
            uuid: unobserved_for(class),
        },
        api,
        api_version: Maybe::Unavailable(Unavailable::NotExposedByFramework),
        driver: DriverRevision {
            // An empty string is how an absent driver name presents, and it is indistinguishable
            // from a driver whose name IS empty — so it becomes a typed unavailability here rather
            // than travelling as a value a range could be compared against.
            name: probed
                .as_ref()
                .map_or_else(|| unobserved_for(class), |p| text_or_unavailable(&p.driver)),
            version_text: probed.as_ref().map_or_else(
                || unobserved_for(class),
                |p| text_or_unavailable(&p.driver_info),
            ),
            ..Default::default()
        },
        kernel_driver: Maybe::Unavailable(Unavailable::NotExposedByPlatform),
        os: operating_system(family),
        // Whether this build can report allocator statistics **for this lane**, evaluated per record
        // rather than declared per binary: a lane compiled in whose sampler returns a reading can, and
        // a lane that is absent or silent cannot, in the same binary.
        //
        // Reported truthfully because a refusal depends on it — a profile whose terms were calibrated
        // from allocator statistics must not be accepted by a binary that cannot reproduce them.
        allocator_statistics_available: samples_this_lane(class),
        // WHICH boundaries, not merely whether. A profile calibrated from slice-end readings is not
        // reproducible on a binary that samples only at phase boundaries, even though both would
        // truthfully report statistics as available — so the comparison is between sets.
        sampled_points: if samples_this_lane(class) {
            SAMPLED_POINTS.iter().map(|p| (*p).to_string()).collect()
        } else {
            std::collections::BTreeSet::new()
        },
        // The pool configuration those readings were taken under. `bytes_reserved` above bytes-in-use
        // IS the pool's retention, so the same binary reports a different figure for the same workload
        // under a different pool configuration; a reading is only reproducible against a match.
        pool_configuration: if samples_this_lane(class) {
            Maybe::Available(POOL_CONFIGURATION.to_string())
        } else {
            unobserved_for(class)
        },
        graphics_api_selected: Maybe::Available(capability.class.clone()),
        graphics_api_selection_source: ApiSelectionSource::PlatformDefault,
    };

    let stack = ComputeStackIdentity {
        framework: ComputeFramework {
            name: "burn".to_string(),
            revision: BURN_REVISION.to_string(),
            runtime_name: "cubecl-runtime".to_string(),
            runtime_revision: CUBECL_REVISION.to_string(),
        },
        implementation_name: format!("cubecl-{}", capability.backend),
        implementation_revision: CUBECL_REVISION.to_string(),
        allocator_name: "cubecl-runtime/memory_management".to_string(),
        allocator_revision: CUBECL_REVISION.to_string(),
        allocation_mode: Maybe::Unavailable(Unavailable::NotExposedByFramework),
    };

    daemon_vhc_resource::BackendImplementationRevision::from_probe(
        observation,
        stack,
        sealed_binary_identity(),
    )
}

/// The adapter facts the compute framework's own info supplied, feature-independent.
struct ProbedAdapter {
    /// The graphics backend the probe brought the adapter up under, as the framework named it
    /// (`"Vulkan"`, `"Metal"`, `"Dx12"`). This is what makes the findings attributable to one lane.
    backend: String,
    is_software: bool,
    vendor_id: Option<u32>,
    device_id: Option<u32>,
    driver: String,
    driver_info: String,
}

impl ProbedAdapter {
    /// Whether this adapter is the one that serves `class`.
    ///
    /// The probe brings up a single adapter under a single selected graphics API, so its findings
    /// describe exactly one lane. A CPU lane is never served by it — the CPU backend allocates
    /// through the host and calls no vendor driver — and neither is CUDA, which is a different stack
    /// reached through a different probe. Reporting the graphics adapter under either would attribute
    /// one lane's hardware to another.
    fn serves(&self, class: daemon_vhc_resource::BackendClass) -> bool {
        use daemon_vhc_resource::BackendClass;
        match class {
            BackendClass::Vulkan => self.backend.eq_ignore_ascii_case("vulkan"),
            BackendClass::Metal => self.backend.eq_ignore_ascii_case("metal"),
            BackendClass::Dx12 => self.backend.eq_ignore_ascii_case("dx12"),
            BackendClass::Cpu | BackendClass::Cuda => false,
        }
    }
}

/// What the framework observed, or `None` where this build has no device lane to observe with.
fn probed_adapter() -> Option<ProbedAdapter> {
    #[cfg(feature = "wgpu")]
    {
        daemon_vhc_host::probe::probe_wgpu().map(|p| ProbedAdapter {
            backend: p.backend,
            is_software: p.is_software,
            vendor_id: p.vendor_id,
            device_id: p.device_id,
            driver: p.driver,
            driver_info: p.driver_info,
        })
    }
    #[cfg(not(feature = "wgpu"))]
    {
        None
    }
}

/// The phase boundaries this build's sampler is wired at, as the record spells them.
///
/// Kept beside the sampler's own `SamplePoint` slugs deliberately: these are the strings a profile's
/// reproducibility check compares, so they must be the boundaries actually sampled and not an
/// aspiration. `after-slice` is included because the slice seam is wired; per-dispatch sampling is
/// **not** here, and is not a wiring omission — attributing a workspace divergence to one operation
/// family needs a delta around each dispatch, which is a pricing-granularity decision rather than a
/// wiring one.
const SAMPLED_POINTS: &[&str] = &[
    "after-bring-up",
    "after-init",
    "after-migrate",
    "after-slice",
    "at-teardown",
];

/// How the allocator's pool is configured in this build.
///
/// The framework's own default, because nothing in this build calls the pool-configuration surface
/// yet. Named rather than left absent so a profile calibrated under this configuration can be told
/// apart from one calibrated under a configured pool — the figure that moves is `bytes_reserved`,
/// which is the one every pooling term is priced on.
const POOL_CONFIGURATION: &str = "framework-default";

/// Whether this build samples the allocator for `class`.
///
/// The conjunction the measurement wave recommended, evaluated per record: the lane must be compiled
/// in, and its sampler must actually return a reading. The CPU lane deliberately does not — its
/// allocations go through the host allocator, whose occupancy answers a different question than a
/// device profile's pooling terms ask.
fn samples_this_lane(class: daemon_vhc_resource::BackendClass) -> bool {
    use daemon_vhc_resource::BackendClass;
    match class {
        BackendClass::Vulkan | BackendClass::Metal | BackendClass::Dx12 => {
            cfg!(feature = "wgpu") && daemon_vhc_host::probe::wgpu_unavailability().is_none()
        }
        BackendClass::Cuda => cfg!(feature = "cuda"),
        BackendClass::Cpu => false,
    }
}

/// The operating system's own revision, read rather than restated.
///
/// `version` used to be `std::env::consts::OS` — a compile-time constant spelling the family, which
/// the family member already carries. It said nothing about the machine and, worse, said it in a
/// field a profile's permitted range would be evaluated against.
///
/// `build` mattered more: it is the implementation-revision signal for a backend whose framework
/// supplies no driver revision, which on Metal is *every* case. Hard-coding it unavailable meant
/// Metal had no comparable revision signal at all and no Metal profile could authenticate — a
/// platform blocked by an unread value rather than by a platform limitation.
fn operating_system(family: daemon_vhc_resource::OsFamily) -> daemon_vhc_resource::OperatingSystem {
    use daemon_vhc_resource::{Maybe, OperatingSystem, Unavailable};

    let kernel = read_first_line(&["uname", "-r"]).map_or(
        Maybe::Unavailable(Unavailable::ProbeFailed),
        Maybe::Available,
    );

    #[cfg(target_os = "macos")]
    let (version, build) = (
        read_first_line(&["sw_vers", "-productVersion"]),
        read_first_line(&["sw_vers", "-buildVersion"]),
    );

    // On Linux the distribution's `VERSION_ID` is the OS version, and the kernel release is the
    // closest thing to a build — it is what actually moves when the graphics stack moves.
    #[cfg(target_os = "linux")]
    let (version, build) = (
        os_release_field("VERSION_ID"),
        read_first_line(&["uname", "-r"]),
    );

    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    let (version, build) = (None, None);

    OperatingSystem {
        family,
        version: version.map_or(
            // Typed, not a restatement of the family: a range evaluated against a value that is
            // really a family name would be comparing the wrong thing while looking like it worked.
            Maybe::Unavailable(Unavailable::NotExposedByPlatform),
            Maybe::Available,
        ),
        build: build.map_or(
            Maybe::Unavailable(Unavailable::NotExposedByPlatform),
            Maybe::Available,
        ),
        kernel,
    }
}

/// The first line of a command's output, trimmed, or `None` when it cannot be read or is empty.
fn read_first_line(argv: &[&str]) -> Option<String> {
    let (program, args) = argv.split_first()?;
    let output = std::process::Command::new(program)
        .args(args)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let line = text.lines().next()?.trim();
    (!line.is_empty()).then(|| line.to_string())
}

/// One field of `/etc/os-release`, unquoted.
#[cfg(target_os = "linux")]
fn os_release_field(key: &str) -> Option<String> {
    let text = std::fs::read_to_string("/etc/os-release").ok()?;
    for line in text.lines() {
        if let Some(value) = line.strip_prefix(key).and_then(|r| r.strip_prefix('=')) {
            let unquoted = value.trim().trim_matches('"').trim();
            if !unquoted.is_empty() {
                return Some(unquoted.to_string());
            }
        }
    }
    None
}

/// Why a lane reports no adapter value.
///
/// A device lane whose probe supplied nothing has a framework gap — a defect or a platform
/// limitation, either way something that could in principle be closed. A CPU lane has no device
/// adapter and no vendor driver at all, so the absence is structural, and a policy meaning to
/// constrain a device driver must never treat it as a tolerable gap.
fn unobserved_for<T>(class: daemon_vhc_resource::BackendClass) -> daemon_vhc_resource::Maybe<T> {
    if class == daemon_vhc_resource::BackendClass::Cpu {
        daemon_vhc_resource::Maybe::Unavailable(
            daemon_vhc_resource::Unavailable::NotApplicableToLane,
        )
    } else {
        no_framework_value()
    }
}

/// A value the framework does not expose.
fn no_framework_value<T>() -> daemon_vhc_resource::Maybe<T> {
    daemon_vhc_resource::Maybe::Unavailable(daemon_vhc_resource::Unavailable::NotExposedByFramework)
}

/// A framework string, or a typed unavailability when it is empty.
///
/// An empty string is how an absent value presents on at least one backend, and it is
/// indistinguishable from a value that genuinely is empty — which is precisely the ambiguity a
/// profile's revision range cannot be evaluated against.
fn text_or_unavailable(value: &str) -> daemon_vhc_resource::Maybe<String> {
    if value.trim().is_empty() {
        no_framework_value()
    } else {
        daemon_vhc_resource::Maybe::Available(value.to_string())
    }
}

/// The compute-framework revision this binary links. A compile-time property of the binary, not of
/// the machine, and the one a profile's compatibility check keys on.
const BURN_REVISION: &str = "0.21.0";
/// The runtime revision beneath it. Its allocator's pooling and retention behaviour is what a
/// profile's pooling terms are calibrated against, and against nothing else.
const CUBECL_REVISION: &str = "0.10.0";

/// This binary's own identity, for the sealed-binary side of a profile's binding.
///
/// Read from the running executable. A record that could not identify its own binary would let a
/// profile bound to one binary be composed with by another, which is the comparison `[PC-12]`(3)
/// exists to make possible.
fn sealed_binary_identity() -> daemon_vhc_resource::SealedBinaryIdentity {
    let (blake3, size_bytes) = std::env::current_exe()
        .ok()
        .and_then(|path| std::fs::read(path).ok())
        .map_or(([0u8; 32], 0), |bytes| {
            (*blake3::hash(&bytes).as_bytes(), bytes.len() as u64)
        });
    daemon_vhc_resource::SealedBinaryIdentity { blake3, size_bytes }
}
/// The device figures the admission funnel compares a role's claim against.
///
/// The device-memory figure is the node's **derived usable supply**, not the dedicated carve-out the
/// legacy probe reports. On a unified part those differ by most of the machine: the carve-out is a true
/// fact about a BIOS reservation and a false statement about what the device may use, so a lane floor
/// or an envelope minimum above it refused a device that could have served the role.
///
/// No derivation means **zero**, and zero on a box that reports a GPU refuses at the lane floor. That is
/// the intended outcome: a device whose usable supply cannot be derived is not admissible, and the one
/// thing that must not happen is a substitute figure — a per-buffer ceiling or a physical-RAM total —
/// standing in for the measurement, because then the refusal arrives at an allocation instead.
fn admission_device_profile(
    hw: &Hardware,
    dl: &DeviceLimits,
) -> daemon_vhc_host::run::DeviceProfile {
    daemon_vhc_host::run::DeviceProfile {
        gpu: hw.gpus > 0,
        vram_bytes: derived_device_supply_bytes().unwrap_or(0),
        ram_bytes: dl.ram_mb << 20,
        disk_bytes: hw.disk_free_mb << 20,
    }
}

/// The device this node's platform ladder found, with the facts a supply derivation reads.
///
/// One ladder, used by the report and by the admission funnel alike, in the order
/// [`device_limits`] already dispatches: a second ordering here would let the two disagree about the
/// same machine, which is the class of defect that put a per-buffer ceiling on one of them.
struct ProbedDevice {
    facts: daemon_vhc_resource::HostDeviceFacts,
    class: daemon_vhc_resource::BackendClass,
    adapter_name: String,
}

/// Gather the platform facts a device-supply derivation reads, or `None` on a box with no device lane.
///
/// `None` is not a failure: a CPU-only participant has no device supply to state, and a report about a
/// device it does not have would be a fiction. What matters is that no *device* box lands here — the
/// derivation's own fail-closed path handles a device whose facts do not add up to a trustworthy figure.
// Every arm below is platform- or lane-gated, so a build with no device lane on this OS uses none of
// the three names — the same shape as the gated `allow` in `backend_inventory`.
#[cfg_attr(
    not(any(feature = "wgpu", feature = "cuda", windows, target_os = "macos")),
    allow(unused_imports)
)]
fn probed_device() -> Option<ProbedDevice> {
    use daemon_vhc_resource::{BackendClass, HostDeviceFacts, SupplyPlatform};

    let mib = |mb: u64| mb << 20;
    let ram_bytes = mib(host_ram_mb());

    #[cfg(windows)]
    {
        if let Some(dl) = daemon_vhc_host::probe::probe_windows_device_limits() {
            let raw = daemon_vhc_host::probe::probe_windows_adapter_memory();
            return Some(ProbedDevice {
                facts: HostDeviceFacts {
                    platform: SupplyPlatform::Windows,
                    unified: dl.unified,
                    dedicated_bytes: mib(dl.vram_mb),
                    // The STATIC borrowable ceiling, not the live budget the mapper folds into this
                    // figure on a unified part. Same reason Linux states its heap rather than its
                    // heap budget: a supply figure that moves with co-tenant pressure gives two probes
                    // of one idle machine two report digests, and the report is cited by digest.
                    shared_pool_bytes: match (&raw, dl.unified) {
                        (Some(raw), true) => raw.shared_system,
                        // A discrete adapter borrows nothing into its device budget by default.
                        (Some(_), false) => 0,
                        // Without the raw scalars the mapper's own figure is the only one in hand.
                        (None, _) => mib(dl.shared_mb),
                    },
                    host_ram_bytes: if dl.ram_mb > 0 {
                        mib(dl.ram_mb)
                    } else {
                        ram_bytes
                    },
                    // The live DXGI budget is dynamic by documentation, so it is a pressure reading for
                    // the governor and is printed beside the report rather than entering it. The static
                    // derivation above, clamped by physical RAM, is what the report states.
                    platform_budget_bytes: None,
                    advertised_device_heap_bytes: None,
                },
                class: BackendClass::Dx12,
                adapter_name: probed_adapter()
                    .map_or_else(|| "dx12 adapter".to_string(), |p| p.adapter),
            });
        }
    }
    #[cfg(target_os = "macos")]
    {
        if let Some(dl) = daemon_vhc_host::probe::probe_macos_device_limits() {
            return Some(ProbedDevice {
                facts: HostDeviceFacts {
                    platform: SupplyPlatform::Macos,
                    unified: dl.unified,
                    dedicated_bytes: mib(dl.vram_mb),
                    shared_pool_bytes: mib(dl.shared_mb),
                    host_ram_bytes: if dl.ram_mb > 0 {
                        mib(dl.ram_mb)
                    } else {
                        ram_bytes
                    },
                    // `macos_device_limits` maps `vram_mb` from `recommendedMaxWorkingSetSize`, which
                    // IS Metal's budget query — the platform saying what this process may use, not a
                    // total we inferred. So it enters as the budget it is.
                    platform_budget_bytes: Some(mib(dl.vram_mb)).filter(|b| *b > 0),
                    advertised_device_heap_bytes: None,
                },
                class: BackendClass::Metal,
                adapter_name: probed_adapter()
                    .map_or_else(|| "metal device".to_string(), |p| p.adapter),
            });
        }
    }
    #[cfg(feature = "cuda")]
    {
        if let Some(p) = daemon_vhc_host::probe::probe_cuda() {
            return Some(ProbedDevice {
                facts: HostDeviceFacts {
                    platform: SupplyPlatform::Cuda,
                    unified: false,
                    // A discrete card's dedicated memory IS its budget, and the CUDA driver reports it
                    // directly — the one platform here that needs no derivation.
                    dedicated_bytes: mib(p.vram_mb),
                    shared_pool_bytes: 0,
                    host_ram_bytes: ram_bytes,
                    platform_budget_bytes: None,
                    advertised_device_heap_bytes: None,
                },
                class: BackendClass::Cuda,
                adapter_name: p.adapter,
            });
        }
    }
    #[cfg(feature = "wgpu")]
    {
        if let Some(p) = daemon_vhc_host::probe::probe_wgpu() {
            let vulkan = daemon_vhc_host::probe::probe_vulkan_heap_budget();
            return Some(ProbedDevice {
                facts: HostDeviceFacts {
                    platform: SupplyPlatform::Linux,
                    unified: p.unified,
                    // The sysfs carve-out, as a FACT. It is a true lower bound on a unified part and
                    // was never a supply statement; reading it as one is the defect the derivation
                    // above exists to correct. Absent sysfs it contributes nothing rather than
                    // borrowing the per-buffer ceiling — a ceiling is not a capacity.
                    dedicated_bytes: mib(amdgpu_sysfs_mem_mb("mem_info_vram_total")),
                    // Read, never assumed: the GTT size is a kernel-cmdline setting on this box and
                    // the historical default is a different fraction of RAM entirely.
                    shared_pool_bytes: mib(amdgpu_sysfs_mem_mb("mem_info_gtt_total")),
                    host_ram_bytes: ram_bytes,
                    // The driver's live heap BUDGET is deliberately not the supply figure, though it
                    // is the more precise number: it moves with what else on the box holds memory, and
                    // two probes of one idle machine minutes apart produce two different budgets and
                    // therefore two different report digests. The report is cited by digest and
                    // compared across incarnations, and volatile occupancy pressure is the governor's
                    // business — so what enters here is the STABLE statement, and the live budget stays
                    // a pressure reading.
                    platform_budget_bytes: None,
                    // The heap the driver presents for device allocations, which on this class of part
                    // is a fixed fraction of one DRAM pool. It bounds the static derivation to what the
                    // backend will actually serve, and being a property of the device and driver rather
                    // than of the moment, it is reproducible.
                    advertised_device_heap_bytes: vulkan.map(|v| v.heap_size_bytes),
                },
                class: match p.backend.to_lowercase().as_str() {
                    "metal" => BackendClass::Metal,
                    "dx12" => BackendClass::Dx12,
                    _ => BackendClass::Vulkan,
                },
                adapter_name: p.adapter,
            });
        }
    }
    let _ = ram_bytes;
    None
}

/// The **Device Capability Report** this node states about its device, or `None` on a box with no
/// device lane.
///
/// This is the producer the report was missing. The measurement behind it was never lost — the
/// platform probes survived and are called on every assess — but nothing turned their facts into the
/// artifact admission compares against, so the capability report existed as a type with fixtures and no
/// production constructor.
///
/// Two things it deliberately does not do. It does not populate the measured per-allocation ceiling:
/// every ceiling reachable here is a *stated* one (a framework constant, a driver's advertised buffer
/// limit), and putting a stated ceiling in a field whose contract says "measured" re-creates the
/// substitution that once reported a two-gigabyte supply on a thirty-gigabyte card. And it does not
/// carry an owner's cap: that is node policy, applied afterwards and independently, so a hardware
/// statement never carries a preference.
pub(crate) fn device_capability_report() -> Option<daemon_vhc_resource::DeviceCapabilityReport> {
    use daemon_vhc_resource::{
        DeviceCapabilityReport, LinkCapacity, Maybe, MemoryPoolTopology, Unavailable,
        DEVICE_CAPABILITY_REPORT_SCHEMA,
    };

    let device = probed_device()?;
    let measured_or_failed = |mb: u64| {
        if mb > 0 {
            Maybe::Available(mb << 20)
        } else {
            // The probe ran and produced nothing usable. `ProbeFailed` rather than a platform gap:
            // every platform this runs on can report its memory and its free disk, so an absence here
            // is a defect on this node — and a zero would refuse the machine while looking measured.
            Maybe::Unavailable(Unavailable::ProbeFailed)
        }
    };

    let capability = BackendCapability {
        backend: device.class.slug().to_string(),
        class: device.class.slug().to_string(),
        adapter: device.adapter_name.clone(),
        device_index: 0,
        vram_mb: 0,
        max_alloc_mb: 0,
        shared_mb: 0,
        unified: device.facts.unified,
        ready: true,
    };
    let revision_digest =
        daemon_vhc_resource::revision_record_digest(&revision_record(&capability)).ok()?;

    Some(DeviceCapabilityReport {
        schema: DEVICE_CAPABILITY_REPORT_SCHEMA,
        backend_class: device.class,
        adapter_name: device.adapter_name,
        device_supply: daemon_vhc_resource::derive_device_supply(&device.facts),
        memory_pool: if device.facts.unified {
            MemoryPoolTopology::Unified
        } else {
            MemoryPoolTopology::Separate
        },
        measured_max_allocation_bytes: Maybe::Unavailable(Unavailable::NotExposedByFramework),
        host_memory_bytes: measured_or_failed(host_ram_mb()),
        disk_bytes: measured_or_failed(host_disk_free_mb()),
        // Not enumerated per device by anything in this build. Left empty rather than filled from the
        // module's own vocabulary: a family gate reading this set must refuse for want of evidence,
        // and inventing entries here would be the report asserting support nobody probed.
        supported_operation_families: std::collections::BTreeSet::new(),
        supported_dtypes: std::collections::BTreeSet::new(),
        // No link measurement exists on this path; the advertised throughput figures are zeros the
        // probe never measured, so they arrive as an absence instead of as a measured zero.
        link: LinkCapacity {
            uplink_bps: Maybe::Unavailable(Unavailable::ProbeFailed),
            downlink_bps: Maybe::Unavailable(Unavailable::ProbeFailed),
        },
        implementation_revision_digest: revision_digest,
        // No profile has been resolved at probe time: resolution is a selection against the run's and
        // the owner's policies, which do not exist here.
        applicable_profile_digest: Maybe::default(),
    })
}

/// The derived usable supply in MiB for one advertised backend class, or `0` when there is none.
///
/// Class-matched on purpose: a build carrying two device lanes probes one of them, and attributing that
/// lane's supply to the other lane's record is the same error as filling one lane's revision record from
/// another lane's adapter.
#[cfg(feature = "wgpu")]
fn derived_supply_mb_for_class(class: &str) -> u64 {
    probed_device()
        .filter(|device| device.class.slug() == class)
        .and_then(|device| {
            daemon_vhc_resource::derive_device_supply(&device.facts)
                .value()
                .map(|supply| supply.usable_bytes >> 20)
        })
        .unwrap_or(0)
}

/// Usable device supply in bytes as this node derives it, for the admission funnel's device figure.
///
/// The funnel used to be handed the amdgpu carve-out on a unified box — 4 GiB on a machine with tens of
/// gigabytes usable — so a lane floor above the carve-out refused a device that could have served the
/// role. Feeding it from the same derivation the report states keeps one answer per machine.
///
/// `None` when there is no device or no trustworthy derivation. The caller decides what that means; it
/// must not become a zero, and it must not become a ceiling borrowed from somewhere else.
pub(crate) fn derived_device_supply_bytes() -> Option<u64> {
    let device = probed_device()?;
    daemon_vhc_resource::derive_device_supply(&device.facts)
        .value()
        .map(|supply| supply.usable_bytes)
}

/// One allocator reading at the device bring-up boundary, taken on the probe path.
///
/// The same sample the run path takes at `AfterBringUp`, reachable without a seeded run, published
/// modules or a genesis envelope — which is what made the allocator terms unobtainable on a bare box.
///
/// `None` is an absence and not a zero: a backend that cannot report occupancy records nothing, and a
/// profile calibrated against a manufactured zero would be calibrated against nothing.
#[must_use]
pub(crate) fn probe_allocator_sample() -> Option<daemon_vhc_host::compute::AllocatorSample> {
    // The DEVICE lane, explicitly. The default is the CPU lane, whose sampler declines by design —
    // its allocations go through the host allocator, which answers a different question than a device
    // profile's pooling terms ask — so probing the default would report an absence on a box that has
    // an accelerator, which is exactly the kind of misreading this readout exists to prevent.
    let backend = {
        #[cfg(feature = "cuda")]
        {
            daemon_vhc_host::runtime::BackendKind::Cuda
        }
        #[cfg(all(feature = "wgpu", not(feature = "cuda")))]
        {
            daemon_vhc_host::runtime::BackendKind::Wgpu
        }
        // No device lane compiled in: there is nothing to sample, and the CPU lane declining is the
        // honest answer rather than a defect.
        #[cfg(not(any(feature = "wgpu", feature = "cuda")))]
        {
            daemon_vhc_host::runtime::BackendKind::Cpu
        }
    };
    let cfg = daemon_vhc_host::EngineConfig {
        backend,
        ..daemon_vhc_host::EngineConfig::default()
    };
    // Built on this thread, sampled on this thread: the runner is thread-pinned, so a sample taken
    // anywhere else would be a programming error rather than a measurement.
    let compute = daemon_vhc_host::compute::HostCompute::build(&cfg).ok()?;
    compute.sample_allocator()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cap(backend: &str, class: &str, ready: bool) -> BackendCapability {
        BackendCapability {
            backend: backend.to_string(),
            class: class.to_string(),
            adapter: format!("test {backend}"),
            device_index: 0,
            vram_mb: 24_000,
            max_alloc_mb: 24_000,
            shared_mb: 0,
            unified: false,
            ready,
        }
    }

    fn cpu_cap() -> BackendCapability {
        BackendCapability {
            vram_mb: 0,
            max_alloc_mb: 0,
            ..cap("cpu", "cpu", true)
        }
    }

    /// The measured ladder: cuda outranks wgpu outranks cpu; a missing/unready upper rung
    /// falls through to the next MEASURED rung (falling through the ladder is selection, not
    /// fallback — the selected rung is recorded and revalidated).
    #[test]
    fn ladder_prefers_cuda_then_wgpu_then_cpu() {
        let full = [
            cap("cuda", "cuda", true),
            cap("wgpu", "vulkan", true),
            cpu_cap(),
        ];
        assert_eq!(
            select_backend(&full, None, &[], None)
                .expect("selects")
                .slug,
            "cuda"
        );
        let no_cuda = [cap("wgpu", "vulkan", true), cpu_cap()];
        assert_eq!(
            select_backend(&no_cuda, None, &[], None)
                .expect("selects")
                .slug,
            "wgpu"
        );
        let cpu_only = [cpu_cap()];
        assert_eq!(
            select_backend(&cpu_only, None, &[], None)
                .expect("selects")
                .slug,
            "cpu"
        );
        // A CUDA device without its staged runtime advertises but never selects.
        let unready = [cap("cuda", "cuda", false), cpu_cap()];
        assert_eq!(
            select_backend(&unready, None, &[], None)
                .expect("selects")
                .slug,
            "cpu"
        );
    }

    /// The run's `backend_class` constraint filters rungs; no matching rung is the typed
    /// refusal (a cuda-required run on a vulkan-only box must refuse, never train on CPU).
    #[test]
    fn backend_class_constraint_filters_rungs_and_refuses_typed() {
        let vulkan_box = [cap("wgpu", "vulkan", true), cpu_cap()];
        let cuda_only = vec!["cuda".to_string()];
        let err = select_backend(&vulkan_box, None, &cuda_only, None)
            .expect_err("no rung serves class cuda");
        assert!(err.contains("BackendUnavailable"), "typed refusal: {err}");
        // The same box serves a vulkan-constrained run on the wgpu rung.
        let vulkan_only = vec!["vulkan".to_string()];
        let sel = select_backend(&vulkan_box, None, &vulkan_only, None).expect("selects");
        assert_eq!((sel.slug.as_str(), sel.class.as_str()), ("wgpu", "vulkan"));
        // A class constraint excluding cpu refuses the cpu-only box typed.
        let err =
            select_backend(&[cpu_cap()], None, &vulkan_only, None).expect_err("cpu lane excluded");
        assert!(err.contains("BackendUnavailable"), "typed refusal: {err}");
    }

    /// The operator's explicit selection: exactly that lane or a typed refusal — the former
    /// quiet downgrade to the CPU lane is gone. `cpu` stays the explicit escape hatch.
    #[test]
    fn explicit_selection_never_falls_back() {
        let cpu_only = [cpu_cap()];
        let err = select_backend(&cpu_only, Some("cuda"), &[], None)
            .expect_err("cuda unavailable must refuse, not fall back");
        assert!(err.contains("BackendUnavailable"), "typed refusal: {err}");
        assert!(err.contains("no fallback"), "names the rule: {err}");
        let err = select_backend(
            &[cap("cuda", "cuda", false), cpu_cap()],
            Some("cuda"),
            &[],
            None,
        )
        .expect_err("unready cuda must refuse");
        assert!(err.contains("not ready to serve"), "{err}");
        // The explicit escape hatch selects the CPU lane on any box.
        let sel = select_backend(&cpu_only, Some("cpu"), &[], None).expect("explicit cpu");
        assert_eq!(sel.slug, "cpu");
        // An explicit selection still honors the run's class constraint (an owner refusing to
        // supply the demanded class is a refusal, not a silent retrain on another class).
        let err = select_backend(
            &[cap("cuda", "cuda", true), cpu_cap()],
            Some("cpu"),
            &["cuda".to_string()],
            None,
        )
        .expect_err("explicit cpu against a cuda-constrained run refuses");
        assert!(err.contains("BackendUnavailable"), "typed refusal: {err}");
    }

    /// Device placement: `gpu_index` must name a probed device; anything else refuses typed.
    #[test]
    fn device_placement_must_name_a_probed_device() {
        let inv = [cap("cuda", "cuda", true), cpu_cap()];
        let sel = select_backend(&inv, None, &[], Some(0)).expect("placement on device 0");
        assert_eq!((sel.slug.as_str(), sel.gpu_index), ("cuda", 0));
        let err = select_backend(&inv, None, &[], Some(1)).expect_err("device 1 was never probed");
        assert!(err.contains("BackendUnavailable"), "typed refusal: {err}");
    }

    /// The advertised inventory always carries the CPU record (the final rung exists on every
    /// build), and — on this CPU-only default build — nothing else.
    #[test]
    fn inventory_always_advertises_the_cpu_lane() {
        let inv = backend_inventory();
        let cpu = inv
            .iter()
            .find(|e| e.backend == "cpu")
            .expect("cpu record always advertised");
        assert!(cpu.ready);
        assert_eq!(cpu.class, "cpu");
    }
}
