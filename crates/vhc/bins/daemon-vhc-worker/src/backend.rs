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
use daemon_vhc_session::protocol::{Eligibility, Hardware, WorkerCapabilities};

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
        (fetch_genesis_artifact(artifact, &module_name).await?, pin)
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
async fn fetch_genesis_artifact(
    artifact: &daemon_vhc_proto::SnapshotArtifact,
    name: &str,
) -> Result<Vec<u8>, String> {
    let art = ArtifactRef::new(artifact.url.clone(), artifact.blake3);
    #[cfg(feature = "vhc-net")]
    if !artifact.url.starts_with("file://") {
        return fetch_artifact_from_store(&art)
            .await
            .map_err(|e| format!("fetch `{name}` ({}) from store: {e}", artifact.url));
    }
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
    fetch_genesis_artifact(artifact, &role.module).await
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
/// caching the verified bytes on a miss (P3 lane S — the fleet distribution path).
#[cfg(feature = "vhc-net")]
pub(crate) async fn fetch_artifact_from_store(art: &ArtifactRef) -> Result<Vec<u8>, String> {
    let ctx = store_fetch_context();
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

/// The `DAEMON_TRAIN_PREFETCH` cache-warming mode (P3 lane S — the fleet staging entry point).
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

/// The engine config for this worker's wasm host, honoring `DAEMON_TRAIN_BACKEND` (P3 lane S — the
/// 160M staging rehearsal's Vulkan lane): `cpu` (default), `burn-ndarray`, or `wgpu` (feature-gated;
/// an unavailable selection falls back to the CPU det lane with a loud stderr note, never silently
/// changing semantics — the det digests are byte-identical across backends by the B3 contract, so
/// backend choice affects wall-clock only). When the var is set the roomy 160M-scale sandbox budgets
/// are applied (the defaults are tuned for the tiny reference model; a 768-wide model's real fp32
/// steps trip the 5 s epoch watchdog — mirrors `preset_160m.rs::roomy_engine`).
pub(crate) fn engine_config_from_env() -> daemon_vhc_host::EngineConfig {
    use std::time::Duration;
    let Ok(kind) = std::env::var("DAEMON_TRAIN_BACKEND") else {
        return daemon_vhc_host::EngineConfig::default();
    };
    let backend = match kind.as_str() {
        "cpu" | "" => daemon_vhc_host::BackendKind::Cpu,
        #[cfg(feature = "burn-ndarray")]
        "burn-ndarray" => daemon_vhc_host::BackendKind::BurnNdarray,
        #[cfg(feature = "wgpu")]
        "wgpu" => daemon_vhc_host::BackendKind::Wgpu,
        // P3 Merge-2: the CUDA lane (RunPod 4090) sets `DAEMON_TRAIN_BACKEND=cuda` for the roomy
        // 160M budgets. (The v1 live-attach probe ladder that composed over this —
        // `select_backend()` — retired with `live.rs` at the Phase-E sunset.)
        #[cfg(feature = "cuda")]
        "cuda" => daemon_vhc_host::BackendKind::Cuda,
        other => {
            eprintln!(
                "[daemon-vhc-worker] DAEMON_TRAIN_BACKEND={other} not available in this build \
                 (feature not compiled?) — falling back to the CPU det lane"
            );
            daemon_vhc_host::BackendKind::Cpu
        }
    };
    daemon_vhc_host::EngineConfig {
        fuel_per_call: 1 << 34,
        epoch_deadline: Duration::from_secs(600),
        op_budget: 1 << 30,
        max_step_handles: 1 << 24,
        backend,
        ..daemon_vhc_host::EngineConfig::default()
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

/// Host RAM in MiB from `/proc/meminfo` `MemTotal` (Linux, best effort). `0` if unavailable.
fn host_ram_mb() -> u64 {
    let Ok(text) = std::fs::read_to_string("/proc/meminfo") else {
        return 0;
    };
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            // `MemTotal:   16384000 kB`
            if let Some(kb) = rest
                .split_whitespace()
                .next()
                .and_then(|n| n.parse::<u64>().ok())
            {
                return kb / 1024;
            }
        }
    }
    0
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
                capabilities: host_capabilities(),
                up_kbps: 0,
                down_kbps: 0,
                disk_free_mb: 0,
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
                capabilities: host_capabilities(),
                up_kbps: 0,
                down_kbps: 0,
                disk_free_mb: 0,
                throughput_class: "c1".to_string(),
            };
        }
    }
    // CUDA (P3 Lane G): the driver exposes total dedicated VRAM directly (`cuDeviceTotalMem`), so a
    // discrete NVIDIA box reports real VRAM + `["cuda","cpu"]` lanes, no UMA (swarm-ledger-p3-g).
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
                capabilities: host_capabilities(),
                up_kbps: 0,
                down_kbps: 0,
                disk_free_mb: 0,
                throughput_class: "c1".to_string(),
            };
        }
    }
    #[cfg(feature = "wgpu")]
    {
        if let Some(p) = daemon_vhc_host::probe::probe_wgpu() {
            // Dedicated VRAM from sysfs (true lower bound); fall back to the max-alloc proxy.
            let vram_sysfs = amdgpu_sysfs_mem_mb("mem_info_vram_total");
            let vram_mb = if vram_sysfs > 0 {
                vram_sysfs
            } else {
                p.max_alloc_mb
            };
            let shared_mb = amdgpu_sysfs_mem_mb("mem_info_gtt_total");
            return Hardware {
                gpus: p.gpus,
                vram_mb,
                shared_mb,
                ram_mb,
                backend_lanes: vec!["vulkan".to_string(), "cpu".to_string()],
                capabilities: host_capabilities(),
                up_kbps: 0,
                down_kbps: 0,
                disk_free_mb: 0,
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
        capabilities: host_capabilities(),
        up_kbps: 0,
        down_kbps: 0,
        disk_free_mb: 0,
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
    // CUDA (P3 Lane G): discrete-device budget from the driver's total-VRAM query (24 GB on the
    // 4090), `shared_mb = 0`, `unified = false` — the discrete verdict path (swarm-ledger-p3-g D3).
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
    let engine = if std::env::var_os("DAEMON_TRAIN_BACKEND").is_some() {
        EngineConfig {
            backend: daemon_vhc_host::BackendKind::Cpu,
            ..engine_config_from_env()
        }
    } else {
        EngineConfig::default()
    };
    let worker = Worker::new(engine).map_err(|e| format!("engine: {e}"))?;

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
    let engine = EngineConfig {
        backend: daemon_vhc_host::BackendKind::Cpu,
        ..engine_config_from_env()
    };
    let worker = Worker::new(engine).map_err(|e| format!("admission engine: {e}"))?;
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
    let device = daemon_vhc_host::run::DeviceProfile {
        gpu: hw.gpus > 0,
        vram_bytes: dl.vram_mb << 20,
        ram_bytes: dl.ram_mb << 20,
        disk_bytes: hw.disk_free_mb << 20,
    };
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
    admission.apply_quotas(&mut run);

    // Inbound trust: the base identities the genesis names — never ambient config.
    let trusted_bases = daemon_vhc_session::identity::TrustedBases::from_genesis(&genesis.env)
        .bases()
        .to_vec();
    Ok(RoleBinding {
        run,
        own_cert: cert,
        trusted_bases,
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
    let device = daemon_vhc_host::run::DeviceProfile {
        gpu: hw.gpus > 0,
        vram_bytes: dl.vram_mb << 20,
        ram_bytes: dl.ram_mb << 20,
        disk_bytes: hw.disk_free_mb << 20,
    };
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
            // of the very bytes admitted; the node stamps its device-profile / owner-policy revisions.
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
