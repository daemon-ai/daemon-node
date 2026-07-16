// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The `WasmBackend` construction / assess / probe side of the worker (§6.5, §10.2).
//!
//! Owns the `Probe` hardware report, the `AssessRun` envelope→`(config, module)` resolution, and the
//! meta-mode eligibility pass. **G2** (Wave 2) evolves this file: real GPU `Hardware` numbers, VRAM
//! autotune / OOM probe, and the burn-wgpu backend behind `WasmBackend::assess`.

use std::collections::BTreeSet;

use daemon_vhc_host::autotune::{Autotune, DeviceLimits, DEFAULT_MAX_MICROBATCH};
use daemon_vhc_host::phase::PHASE_TABLE;
use daemon_vhc_host::{EngineConfig, Worker};
use daemon_vhc_net::{ArtifactRef, ArtifactResolver};
use daemon_vhc_proto::{from_canonical_slice, SignedEnvelope};
use daemon_vhc_session::protocol::{Eligibility, Hardware, WorkerCapabilities};

use crate::SEQ;

/// A large sentinel (in MiB) used when a resource dimension is unknown, so the autotune verdict does
/// not spuriously reject on an unprobed number (`u64::MAX / MiB`).
const UNKNOWN_BUDGET_MB: u64 = u64::MAX / (1 << 20);

/// The experiment inputs a run resolves to: the `[experiment.config]` CBOR + the module `.wasm`,
/// plus the envelope's per-role blake3 pin when one exists (the ABI §1.3 step-1 verify-before-
/// compile input; `None` under the explicit `DAEMON_TRAIN_MODULE` module-source override, which
/// deliberately bypasses the artifact map and so carries no pin).
pub(crate) struct ResolvedRun {
    pub(crate) config: Vec<u8>,
    pub(crate) module: Vec<u8>,
    pub(crate) module_blake3: Option<[u8; 32]>,
    /// The envelope's additive `device_min` section (ABI §9.3 stage-3 pre-screen; D3 cell 5
    /// interim-supported), parsed from the RAW frozen bytes — `None` when the envelope carries
    /// no such section. On the genesis path this is the worker role's typed `device_min`.
    pub(crate) device_min: Option<daemon_vhc_proto::DeviceMinimums>,
    /// The decoded typed v1 envelope (the coordinator-config source for the v2 self-driven join,
    /// mixed-fleet cell 5). `None` on the genesis path.
    pub(crate) envelope: Option<daemon_vhc_proto::Envelope>,
    /// The resolved genesis run (envelope v2 — mixed-fleet cell 6, D1 deliverable 4). `None` on
    /// the v1 path.
    pub(crate) genesis: Option<GenesisRun>,
}

/// A resolved envelope-v2 (genesis) run: the decoded envelope, its hash (the cryptographic
/// `RunId`, ABI §8.1), and the joining worker role. A genesis run's COORDINATION is its wasm
/// coordinator module (mixed-fleet cell 8) — the transitional cell-6 native-coordinator adapter
/// was retired at D2, so no coordinator-role config is read host-side anymore.
pub(crate) struct GenesisRun {
    pub(crate) env: daemon_vhc_proto::GenesisEnvelope,
    /// The genesis hash — the cryptographic `RunId` (architecture §5.1).
    pub(crate) run_id: [u8; 32],
    /// The worker role this node assessed/joins (the first non-coordinator role — the
    /// single-worker interim of decisions D6; role *selection* is node policy from Phase E).
    pub(crate) worker_role: String,
}

impl ResolvedRun {
    /// The envelope-v2 grants input for the admission funnel (D1 deliverable 4): the worker
    /// role's grant list + the run's artifact-map hashes, derived from the genesis envelope.
    /// `None` on the v1 path — the funnel's pre-D0 defaults stand there.
    pub(crate) fn envelope_grants(&self) -> Option<daemon_vhc_host::v2::EnvelopeRoleGrants> {
        let g = self.genesis.as_ref()?;
        daemon_vhc_host::v2::EnvelopeRoleGrants::from_genesis(&g.env, &g.worker_role)
    }
}

/// The typed refusal slug for the D0-retired unsigned legacy envelope path (refactor §8/D0:
/// "the worker's unsigned legacy envelope path is retired here with a typed refusal"). Stable —
/// tests and the node key on it, exactly like the ABI §1.5 refusal slugs.
pub(crate) const UNSIGNED_ENVELOPE_RETIRED: &str = "UnsignedEnvelopeRetired";

/// Resolve the `AssessRun` envelope bytes into `(config, module)` (the §6.1/§6.5 seam).
///
/// The bytes MUST be the canonical [`SignedEnvelope`] wire form: verify it, take
/// `config_bytes()`, and resolve the module from the envelope's artifact map via
/// [`ArtifactResolver`] (`file://`, blake3-verified). `DAEMON_TRAIN_MODULE` remains the explicit
/// dev/node-controlled **module-source override inside the signed path** (it substitutes the
/// artifact fetch, never the envelope).
///
/// **D0: the unsigned legacy path is RETIRED.** Bytes that are not a signed-envelope wrapper
/// (the pre-A0 raw `[experiment.config]` CBOR direct-drive) are refused with the typed
/// [`UNSIGNED_ENVELOPE_RETIRED`] slug — never accepted, never guessed at. Dev/test drives that
/// used raw config author a signed envelope instead (the worker-protocol suite's
/// `signed_envelope_wire()` shape), optionally keeping `DAEMON_TRAIN_MODULE` as the module
/// source.
pub(crate) async fn resolve_run(envelope_bytes: &[u8]) -> Result<ResolvedRun, String> {
    let wire = from_canonical_slice::<SignedEnvelope>(envelope_bytes).map_err(|e| {
        format!(
            "{UNSIGNED_ENVELOPE_RETIRED}: AssessRun bytes are not a SignedEnvelope wire form \
             (the unsigned legacy raw-config path was retired at D0; author a signed envelope — \
             DAEMON_TRAIN_MODULE still overrides the module source inside it): {e}"
        )
    })?;
    // Route on the schema sniff (decisions D3: the dual-driver worker enforces the mixed-fleet
    // matrix from the raw bytes — a v2 genesis run resolves through the genesis path, a v1 run
    // through the frozen-envelope path, byte-for-byte as before).
    if daemon_vhc_proto::peek_schema(&wire.bytes) == Some(daemon_vhc_proto::GENESIS_SCHEMA_MAJOR) {
        return resolve_genesis_run(wire).await;
    }
    // Verify (re-derives hash + config over the received bytes, checks the signature).
    let frozen = wire.open().map_err(|e| format!("verify envelope: {e}"))?;
    let config = frozen.config_bytes().to_vec();
    let device_min = frozen.device_min();
    let envelope = frozen.decode().ok();
    let (module, module_blake3) = resolve_module(&frozen).await?;
    Ok(ResolvedRun {
        config,
        module,
        module_blake3,
        device_min,
        envelope,
        genesis: None,
    })
}

/// Resolve an envelope-v2 **genesis** run (D1 deliverable 4; mixed-fleet cell 6): verify the
/// signed genesis (hash re-derived over the received bytes, author signature checked), select the
/// worker + coordinator roles from the role set, decode the worker role's opaque config, and
/// resolve the worker role's module by its pinned artifact hash.
///
/// Role selection is the single-worker interim (decisions D6): the first non-coordinator role is
/// the joining worker role, exactly the well-formedness split `GenesisEnvelope::validate`
/// guarantees exists; per-node role *selection* policy arrives with Phase-E multi-instance.
async fn resolve_genesis_run(wire: SignedEnvelope) -> Result<ResolvedRun, String> {
    let frozen = daemon_vhc_proto::FrozenGenesis::open(wire.bytes, wire.signature, wire.signer)
        .map_err(|e| format!("verify genesis envelope: {e}"))?;
    let env = frozen
        .decode()
        .map_err(|e| format!("decode genesis: {e}"))?;
    let worker_role = env
        .roles
        .keys()
        .find(|r| !r.contains("coordinator"))
        .cloned()
        .ok_or("genesis envelope has no worker role (validate should have refused)")?;
    let role = &env.roles[&worker_role];
    let config = frozen
        .role_config_bytes(&worker_role)
        .map_err(|e| format!("role config: {e}"))?
        .ok_or("worker role vanished between decode and config extraction")?;
    let device_min = Some(role.device_min.clone());

    // Resolve the worker role's module by its pinned artifact hash — same override + fetch
    // discipline as the v1 path (DAEMON_TRAIN_MODULE is the explicit dev/node-controlled
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
        let art = ArtifactRef::new(artifact.url.clone(), artifact.blake3);
        #[cfg(feature = "swarm-net")]
        if !artifact.url.starts_with("file://") {
            let bytes = fetch_artifact_from_store(&art).await.map_err(|e| {
                format!(
                    "fetch module `{}` ({}) from store: {e}",
                    role.module, artifact.url
                )
            })?;
            let run_id = *frozen.run_id().as_bytes();
            return finish_genesis_run(config, bytes, pin, device_min, env, run_id, worker_role);
        }
        let bytes = ArtifactResolver::new()
            .fetch(&art)
            .await
            .map_err(|e| format!("resolve module `{}` ({}): {e}", role.module, artifact.url))?;
        (bytes, pin)
    };
    let run_id = *frozen.run_id().as_bytes();
    finish_genesis_run(
        config,
        module,
        module_blake3,
        device_min,
        env,
        run_id,
        worker_role,
    )
}

/// Assemble the genesis [`ResolvedRun`] (split out so the feature-gated store-fetch arm can share
/// the tail without duplicating it).
#[allow(clippy::too_many_arguments)]
fn finish_genesis_run(
    config: Vec<u8>,
    module: Vec<u8>,
    module_blake3: Option<[u8; 32]>,
    device_min: Option<daemon_vhc_proto::DeviceMinimums>,
    env: daemon_vhc_proto::GenesisEnvelope,
    run_id: [u8; 32],
    worker_role: String,
) -> Result<ResolvedRun, String> {
    Ok(ResolvedRun {
        config,
        module,
        module_blake3,
        device_min,
        envelope: None,
        genesis: Some(GenesisRun {
            env,
            run_id,
            worker_role,
        }),
    })
}

/// Resolve the experiment module bytes for a verified envelope (P3 lane S — fetch-by-hash):
///
/// 1. `DAEMON_TRAIN_MODULE` set → read the local file (the **explicit** dev/test override — the only
///    remaining local-path path; the P2 pre-staging is gone).
/// 2. else, resolve the envelope's `experiment.module` artifact by its content hash. Under the
///    `swarm-net` feature a network URL (`r2://` / `https://` / `hf://`) is fetched from the payload
///    store via a presigned GET (context from the node-set env, [`store_fetch_context`]) and cached
///    content-addressed on disk ([`ContentCache`]); a `file://` URL uses the file-only resolver.
/// 3. default build: `file://` only (network schemes are `SchemeUnsupported`, as before).
///
/// Every path blake3-verifies the bytes against the artifact-map hash **before** `assess`/
/// instantiation ([`ArtifactResolver::fetch`] / [`ContentCache`]), so a tampered module is rejected
/// before the wasm engine loads it (§6.5, §12).
async fn resolve_module(
    frozen: &daemon_vhc_proto::FrozenEnvelope,
) -> Result<(Vec<u8>, Option<[u8; 32]>), String> {
    if let Some(bytes) = module_from_env() {
        // The explicit dev/node-controlled override deliberately bypasses the artifact map, so it
        // carries no envelope pin (the operator chose the bytes; there is nothing to verify them
        // against — matching the pre-A0 behavior of this path).
        return bytes.map(|b| (b, None));
    }
    let envelope = frozen
        .decode()
        .map_err(|e| format!("decode envelope: {e}"))?;
    let name = &envelope.experiment.module;
    let artifact = envelope
        .artifacts
        .get(name)
        .ok_or_else(|| format!("experiment module `{name}` absent from [artifacts]"))?;
    let pin = Some(artifact.blake3.0);
    let art = ArtifactRef::new(artifact.url.clone(), artifact.blake3);

    // Content-addressed fetch from the payload store (fleet distribution) — feature-gated because
    // egress/presign live behind `swarm-net`.
    #[cfg(feature = "swarm-net")]
    if !artifact.url.starts_with("file://") {
        return fetch_artifact_from_store(&art)
            .await
            .map(|b| (b, pin))
            .map_err(|e| format!("fetch module `{name}` ({}) from store: {e}", artifact.url));
    }

    ArtifactResolver::new()
        .fetch(&art)
        .await
        .map(|b| (b, pin))
        .map_err(|e| format!("resolve module `{name}` ({}): {e}", artifact.url))
}

/// The presign context the node sets when spawning the worker for a live run (small env strings, NOT
/// a pre-staged artifact): the coordinator base, run id, auth, and the on-disk cache dir/budget. This
/// is the fetch-by-hash analogue of `DAEMON_TRAIN_MODULE` — a node-controlled input at assess time.
#[cfg(feature = "swarm-net")]
pub(crate) struct StoreFetchContext {
    pub(crate) presign_base: Option<String>,
    pub(crate) run_id: String,
    pub(crate) ws_auth: daemon_vhc_session::protocol::WsAuthSpec,
    pub(crate) cache_dir: std::path::PathBuf,
    pub(crate) cache_gb: u32,
}

#[cfg(feature = "swarm-net")]
pub(crate) fn store_fetch_context() -> StoreFetchContext {
    use daemon_vhc_session::protocol::WsAuthSpec;
    let ws_auth = if let Ok(bearer) = std::env::var("DAEMON_SWARM_BEARER") {
        WsAuthSpec::Bearer(bearer)
    } else if let (Ok(org_id), Ok(actor)) = (
        std::env::var("DAEMON_SWARM_ORG"),
        std::env::var("DAEMON_SWARM_ACTOR"),
    ) {
        WsAuthSpec::Internal { org_id, actor }
    } else {
        WsAuthSpec::None
    };
    let cache_dir = std::env::var_os("DAEMON_SWARM_CACHE_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("daemon-swarm-cache"));
    let cache_gb = std::env::var("DAEMON_SWARM_CACHE_GB")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    StoreFetchContext {
        presign_base: std::env::var("DAEMON_SWARM_PRESIGN_BASE").ok(),
        run_id: std::env::var("DAEMON_SWARM_RUN_ID").unwrap_or_else(|_| "run-unknown".to_string()),
        ws_auth,
        cache_dir,
        cache_gb,
    }
}

/// Build a content-addressed [`daemon_vhc_net::ContentCache`] from the node-set context.
#[cfg(feature = "swarm-net")]
pub(crate) fn open_content_cache(
    ctx: &StoreFetchContext,
) -> Result<daemon_vhc_net::ContentCache, String> {
    daemon_vhc_net::ContentCache::open_gb(&ctx.cache_dir, ctx.cache_gb)
        .map_err(|e| format!("open content cache {}: {e}", ctx.cache_dir.display()))
}

/// Build an [`ArtifactResolver`] wired for the network schemes from the node-set presign context
/// (egress for `https`/`hf`; egress + presign for `r2://`).
#[cfg(feature = "swarm-net")]
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
#[cfg(feature = "swarm-net")]
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
#[cfg(feature = "swarm-net")]
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

/// The `DAEMON_TRAIN_PREFETCH` cache-warming mode (P3 lane S — the fleet staging entry point).
///
/// Runs on a bare fleet box (Windows cmd.exe, macOS, a RunPod container) with no CBOR framing:
/// fetch the run's module and/or corpus (manifest + windowed shards) **by content hash** from the
/// payload store into the on-disk [`daemon_vhc_net::ContentCache`], verify blake3, print per-object
/// `key / bytes / blake3 / source(cache|store) / ms`, then exit. A subsequent live run on the box
/// finds every artifact cache-warm. Idempotent (re-running is all cache hits).
///
/// Env (mirrors the assess-time fetch context; all plain strings so `set X=… && worker.exe` works):
/// `DAEMON_SWARM_PRESIGN_BASE`, `DAEMON_SWARM_RUN_ID`, `DAEMON_SWARM_ORG`/`DAEMON_SWARM_ACTOR` (or
/// `DAEMON_SWARM_BEARER`), `DAEMON_SWARM_CACHE_DIR`/`DAEMON_SWARM_CACHE_GB`, plus what to warm:
/// `DAEMON_TRAIN_PREFETCH_MODULE=<blake3-hex>`, `DAEMON_TRAIN_PREFETCH_MANIFEST=<blake3-hex>`,
/// `DAEMON_TRAIN_PREFETCH_WINDOW=<start>:<count>` (optional; absent/0 = every shard).
#[cfg(feature = "swarm-net")]
pub(crate) async fn prefetch_main() -> Result<(), String> {
    let ctx = store_fetch_context();
    if ctx.presign_base.is_none() {
        return Err("DAEMON_TRAIN_PREFETCH needs DAEMON_SWARM_PRESIGN_BASE".to_string());
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
        let art = ArtifactRef::new(format!("r2://corpus/{}.json", hash.to_hex()), hash);
        let manifest_bytes = warm(&cache, &resolver, "manifest", &art).await?;
        let manifest = daemon_vhc_session::data::Manifest::from_json(
            std::str::from_utf8(&manifest_bytes).map_err(|e| format!("manifest utf8: {e}"))?,
        )
        .map_err(|e| format!("parse corpus manifest: {e}"))?;

        let (start, count) = std::env::var("DAEMON_TRAIN_PREFETCH_WINDOW")
            .ok()
            .and_then(|w| {
                let (s, c) = w.split_once(':')?;
                Some((s.parse().ok()?, c.parse().ok()?))
            })
            .unwrap_or((0u64, 0u64));
        let indices = manifest.shards_covering(start, count);
        println!(
            "prefetch: corpus window start={start} count={count} -> {}/{} shards",
            indices.len(),
            manifest.shards.len()
        );
        for idx in indices {
            let desc = &manifest.shards[idx];
            let hash = hash_from_hex(&desc.blake3)
                .ok_or_else(|| format!("shard {idx} malformed blake3"))?;
            let art = ArtifactRef::new(format!("r2://corpus/{}.bin", desc.blake3), hash);
            warm(
                &cache,
                &resolver,
                &format!("shard[{idx}] {}", desc.name),
                &art,
            )
            .await?;
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
        // 160M budgets; the actual backend/gpu is still chosen by `select_backend()`'s probe ladder
        // (NVRTC-readiness-gated), which composes over this via `worker_engine_config()`.
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

/// The host `tabi@1` vocabulary (name-for-name with the phase table / SDK `TABI_IMPORTS`, all 66).
fn host_ops() -> Vec<String> {
    PHASE_TABLE.iter().map(|(n, _)| (*n).to_string()).collect()
}

pub(crate) fn host_capabilities() -> WorkerCapabilities {
    WorkerCapabilities {
        abi_version: daemon_vhc_host::TENSOR_ABI_MAJOR as u16,
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
/// Parsing is delegated to [`daemon_vhc_host::autotune::parse_amdgpu_mem_mb`] (unit-tested with
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
            if let Some(mb) = daemon_vhc_host::autotune::parse_amdgpu_mem_mb(&contents) {
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
        if let Some(dl) = daemon_vhc_host::autotune::probe_windows_device_limits() {
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
        if let Some(dl) = daemon_vhc_host::autotune::probe_macos_device_limits() {
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
        if let Some(p) = daemon_vhc_host::autotune::probe_cuda() {
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
        if let Some(p) = daemon_vhc_host::autotune::probe_wgpu() {
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

/// The device budget the autotune verdict is computed against (Merge-2 UMA fix).
///
/// With the `wgpu` feature + a usable adapter: `vram_mb` = sysfs dedicated VRAM (true lower bound),
/// `shared_mb` = sysfs GTT (the unified spillover pool), `max_alloc_mb` = the wgpu `max_buffer_size`
/// per-buffer ceiling, and `unified` = the adapter's device-type (IntegratedGpu/Cpu). On a unified
/// device the verdict then treats VRAM+GTT+RAM as one physical DRAM pool instead of rejecting
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
        if let Some(dl) = daemon_vhc_host::autotune::probe_windows_device_limits() {
            return dl;
        }
    }
    #[cfg(target_os = "macos")]
    {
        if let Some(dl) = daemon_vhc_host::autotune::probe_macos_device_limits() {
            return dl;
        }
    }
    // CUDA (P3 Lane G): discrete-device budget from the driver's total-VRAM query (24 GB on the
    // 4090), `shared_mb = 0`, `unified = false` — the discrete verdict path (swarm-ledger-p3-g D3).
    #[cfg(feature = "cuda")]
    {
        if let Some(p) = daemon_vhc_host::autotune::probe_cuda() {
            return daemon_vhc_host::autotune::cuda_device_limits(
                p.vram_mb,
                p.max_alloc_mb,
                ram_mb,
            );
        }
    }
    #[cfg(feature = "wgpu")]
    {
        if let Some(p) = daemon_vhc_host::autotune::probe_wgpu() {
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

/// The peer-side re-validation, A0 dual-dispatch shape (ABI Draft 3 §1.3; decisions D2):
///
/// 1. **Driver selection first** — [`daemon_vhc_host::select_driver`] runs the normative order
///    (hash-verify before compile → static-import inspection → candidate linker → instantiate →
///    `da_abi` cross-check). Any failure is a **typed admission refusal** returned as an
///    ineligible [`Eligibility`] carrying the split `refusal_code` slug — an `Assessed` outcome,
///    never an `Event::Error` (ABI §1.5).
/// 2. A module selecting the **v1 driver** proceeds down the byte-for-byte unchanged v1 assess:
///    the static import scan vs the host `tabi@1` vocabulary, then the host meta-mode pass +
///    autotune verdict.
/// 3. A module selecting the **v2 driver** runs the A2 claim()-admission funnel ([`assess_v2`]):
///    lane floor, restricted-instance `da_manifest`/`da_claim`, lane claim bounds, owner
///    authorization — the ABI §9.3 owner-bracketed order.
///
/// Returns the eligibility verdict plus whether the module selected the **major-2** driver (the
/// `JoinRun` dispatch needs it: the v2 run path — session pump attach — is not wired in this
/// worker yet, so a v2 join is refused loud instead of falling into the v1 self-drive).
pub(crate) fn assess(
    module: &[u8],
    config: &[u8],
    module_blake3: Option<&[u8; 32]>,
    device_min: Option<&daemon_vhc_proto::DeviceMinimums>,
    envelope_grants: Option<&daemon_vhc_host::v2::EnvelopeRoleGrants>,
) -> Result<(Eligibility, bool), String> {
    // `DAEMON_TRAIN_BACKEND` set ⇒ the roomy 160M-scale budgets (the meta pass over a real-scale
    // param layout exceeds the tiny-model defaults). The meta pass itself stays on the CPU-cheap
    // path (footprint estimation, backend-independent).
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
        Ok(sel) if sel.driver == daemon_vhc_abi::CandidateDriver::V1 => {
            // Fall through to the retained v1 assess below — unchanged behavior.
        }
        Ok(_sel) => {
            // The major-2 path: the claim()-based admission funnel (ABI §9.3, A2). Selection
            // re-runs inside admit_v2 (§9.4 step 1–3 order is normative); the arms below map the
            // funnel outcome onto the typed `Assessed` surface.
            return Ok((
                assess_v2(
                    &worker,
                    module,
                    config,
                    module_blake3,
                    device_min,
                    envelope_grants,
                ),
                true,
            ));
        }
        Err(refusal) => {
            return Ok((
                Eligibility {
                    eligible: false,
                    reasons: vec![refusal.to_string()],
                    headroom: Vec::new(),
                    refusal_code: Some(refusal.code.slug().to_string()),
                },
                false,
            ));
        }
    }

    let vocabulary: BTreeSet<String> = host_ops().into_iter().collect();
    let imports = worker
        .module_imports(module)
        .map_err(|e| format!("module import scan: {e}"))?;
    let missing: Vec<String> = imports
        .iter()
        .filter(|name| !vocabulary.contains(name.as_str()))
        .cloned()
        .collect();

    if !missing.is_empty() {
        return Ok((
            Eligibility {
                eligible: false,
                reasons: vec![format!(
                    "module imports ops outside host tabi@1: {}",
                    missing.join(", ")
                )],
                headroom: Vec::new(),
                refusal_code: None,
            },
            false,
        ));
    }

    let loaded = worker
        .load_module(module)
        .map_err(|e| format!("load module: {e}"))?;
    let mut inst = worker
        .instantiate(&loaded)
        .map_err(|e| format!("instantiate: {e}"))?;
    let report = inst
        .meta(config, 1, SEQ)
        .map_err(|e| format!("meta: {e}"))?;

    // G2 VRAM autotune (§5.1 planning, ABI §8): the meta-report footprint vs the probed device
    // budget → eligibility + chosen micro-batch. The MetaReport byte footprints are
    // backend-independent (shapes/dtypes), so the CPU meta pass is authoritative for the estimates;
    // the verdict compares them against the real device numbers from `device_limits`.
    let autotune = Autotune::from_meta(&report);
    let verdict = autotune.verdict(&device_limits(), DEFAULT_MAX_MICROBATCH);

    let mib = 1i64 << 20;
    let mut reasons = vec![format!(
        "tabi@1 satisfied ({} imports); meta pass ok",
        imports.len()
    )];
    reasons.extend(verdict.reasons.iter().cloned());

    Ok((
        Eligibility {
            eligible: verdict.eligible,
            reasons,
            refusal_code: None,
            headroom: vec![
                ("micro_batch".to_string(), i64::from(verdict.micro_batch)),
                ("vram_mb".to_string(), verdict.vram_mb_estimate as i64),
                ("ram_mb".to_string(), verdict.ram_mb_estimate as i64),
                (
                    "payload_bytes".to_string(),
                    verdict.payload_bytes_estimate as i64,
                ),
                (
                    "host_ram_mb".to_string(),
                    (report.host_ram_bytes_est as i64) / mib,
                ),
                ("param_bytes".to_string(), report.param_bytes as i64),
            ],
        },
        false,
    ))
}

/// The owner's node-side lane configuration seam (§9.6: "numbers are deployment config").
/// Until the node client carries lane config, `DAEMON_VHC_LANE_GPU_OPTIONAL=1` selects a
/// CPU-admitting dev/t2 lane (GPU optional, no device floors, the same claim bounds) — the
/// owner's explicit choice, exactly like `DAEMON_TRAIN_BACKEND=cpu` on the v1 path. Shared by
/// assess and the v2 join so both stages evaluate the identical lane (§9.4 step-10 re-check).
pub(crate) fn selected_lane() -> daemon_vhc_host::v2::ParticipationLane {
    if std::env::var_os("DAEMON_VHC_LANE_GPU_OPTIONAL").is_some_and(|v| v == "1") {
        daemon_vhc_host::v2::ParticipationLane {
            gpu: 1,
            vram_bytes: 0,
            ram_bytes: 0,
            disk_bytes: 0,
            ..daemon_vhc_host::v2::ParticipationLane::trainer_launch_defaults()
        }
    } else {
        daemon_vhc_host::v2::ParticipationLane::trainer_launch_defaults()
    }
}

/// The major-2 assess arm: the owner-bracketed claim()-admission funnel (ABI §9.3;
/// `daemon_vhc_host::v2::admission::admit_v2`), mapped onto the typed `Assessed` surface.
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
fn assess_v2(
    worker: &Worker,
    module: &[u8],
    config: &[u8],
    module_blake3: Option<&[u8; 32]>,
    device_min: Option<&daemon_vhc_proto::DeviceMinimums>,
    envelope_grants: Option<&daemon_vhc_host::v2::EnvelopeRoleGrants>,
) -> Eligibility {
    let lane = selected_lane();
    let hw = hardware();
    let dl = device_limits();
    let device = daemon_vhc_host::v2::DeviceProfile {
        gpu: hw.gpus > 0,
        vram_bytes: dl.vram_mb << 20,
        ram_bytes: dl.ram_mb << 20,
        disk_bytes: hw.disk_free_mb << 20,
    };
    let owner = daemon_vhc_host::v2::OwnerPolicy {
        participation_enabled: true,
        vram_cap_bytes: 0,
        host_cap_bytes: 0,
    };
    // The derived grants document (§2.6 stand-in): the SAME deterministic derivation the v2 join
    // uses, so assess and join evaluate byte-identical (config, grants) pairs (§9.4 pinning).
    let grants = crate::v2_session::derive_grants();
    match daemon_vhc_host::v2::admit_v2(
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
        // the lane at stage 4.0 (mixed-fleet cell 6). `None` on the v1-envelope path, where the
        // funnel's pre-D0 defaults stand.
        envelope_grants,
    ) {
        Ok(admission) => Eligibility {
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
        },
        Err(refusal) => Eligibility {
            eligible: false,
            reasons: vec![refusal.to_string()],
            headroom: Vec::new(),
            refusal_code: refusal.code.map(|c| c.slug().to_string()),
        },
    }
}

/// The engine backend + GPU index the live worker drives its **native** lane on (§10.5 verdict path;
/// swarm-ledger-p3-g D4 + the P3 fat-worker packaging decision).
///
/// Probe-ordered graceful degradation for the one-fat-binary packaging (ndarray + wgpu + cuda
/// unioned; runtime probe selects the arm): **CUDA → wgpu → CPU**. Each rung is taken only when its
/// feature is built AND its probe reports a usable device, so a cuda-featured worker on a non-NVIDIA
/// machine falls through to wgpu (if built + adapter present) or the CPU det lane — no panic, no
/// link-time dependency (cudarc is in dlopen mode; the memoized `probe_cuda` catches the
/// missing-libcuda unwind and reports `None`). The det lane stays host fp32 on every backend, so a
/// GPU peer's post-ingest digests are byte-identical to CPU peers ingesting the same committed set
/// (the consensus invariant is unchanged — the GPU arms only accelerate the tolerance-class native
/// lane).
///
/// `DAEMON_TRAIN_BACKEND=cpu` forces the CPU lane even when a GPU is present (an operator escape
/// hatch for a box whose driver-matched NVRTC is not staged — the backend would otherwise fail on
/// the first device op). Returns `(BackendKind, gpu_index)` for the [`daemon_vhc_host::EngineConfig`].
///
/// Only the live-attach path (`swarm-net`) consumes this; a probe-only or default worker never
/// selects a backend, so it is gated with its sole caller (`live.rs`).
#[cfg(feature = "swarm-net")]
pub(crate) fn select_backend() -> (daemon_vhc_host::BackendKind, Option<u32>) {
    let requested = std::env::var("DAEMON_TRAIN_BACKEND").ok();
    if requested.as_deref() == Some("cpu") {
        return (daemon_vhc_host::BackendKind::Cpu, None);
    }
    // Explicit operator override (P3 Merge-2): `DAEMON_TRAIN_BACKEND=wgpu` pins the wgpu rung even
    // when a CUDA device is present — the honest escape hatch for a box whose CUDA lane is unusable
    // (the Merge-2 fleet run hit a cubecl-cuda VRAM-exhaustion panic in the fleet engine loop that
    // the OOM ladder cannot catch: cubecl panics on its own thread instead of returning a
    // BudgetMemory trap; single-host 160M CUDA is green — recorded in swarm-p3-ledger). Probe-gated:
    // if the requested rung has no usable adapter, fall through the normal ladder below.
    #[cfg(feature = "wgpu")]
    if requested.as_deref() == Some("wgpu") {
        if let Some(p) = daemon_vhc_host::autotune::probe_wgpu() {
            eprintln!(
                "daemon-vhc-worker: DAEMON_TRAIN_BACKEND=wgpu override — selecting wgpu native \
                 lane (adapter: {}, backend {}); det lane stays host fp32 (consensus-invariant)",
                p.adapter, p.backend
            );
            return (daemon_vhc_host::BackendKind::Wgpu, None);
        }
        eprintln!(
            "daemon-vhc-worker: DAEMON_TRAIN_BACKEND=wgpu requested but no usable adapter — \
             falling through the probe ladder"
        );
    }
    #[cfg(feature = "cuda")]
    {
        if let Some(p) = daemon_vhc_host::autotune::probe_cuda() {
            // NVRTC readiness gate (fetch-on-demand contract, swarm-ledger-p3-g D6): the device
            // alone is not enough — burn-cuda JITs through libnvrtc, which must be staged
            // driver-matched (DAEMON_CUDA_RUNTIME_DIR). Until it is, downgrade to wgpu/CPU
            // instead of failing on the first tensor op.
            if daemon_vhc_host::autotune::cuda_nvrtc_ready() {
                eprintln!(
                    "daemon-vhc-worker: selecting CUDA native lane (device 0: {}, {} MiB VRAM); \
                     det lane stays host fp32 (consensus-invariant)",
                    p.adapter, p.vram_mb
                );
                return (daemon_vhc_host::BackendKind::Cuda, Some(0));
            }
            eprintln!(
                "daemon-vhc-worker: CUDA device present ({}) but the JIT runtime is not staged \
                 (libnvrtc loadable + CUDA_PATH/include/cuda_runtime.h required) — stage the \
                 driver-matched NVRTC runtime (DAEMON_CUDA_RUNTIME_DIR, export \
                 CUDA_PATH=$DAEMON_CUDA_RUNTIME_DIR) to enable the CUDA lane; downgrading \
                 (probe order: cuda -> wgpu -> cpu)",
                p.adapter
            );
        } else {
            eprintln!(
                "daemon-vhc-worker: cuda feature built but no CUDA device present — degrading \
                 (fat-worker probe order: cuda -> wgpu -> cpu)"
            );
        }
    }
    #[cfg(feature = "wgpu")]
    {
        if let Some(p) = daemon_vhc_host::autotune::probe_wgpu() {
            eprintln!(
                "daemon-vhc-worker: selecting wgpu native lane (adapter: {}, backend {}); \
                 det lane stays host fp32 (consensus-invariant)",
                p.adapter, p.backend
            );
            return (daemon_vhc_host::BackendKind::Wgpu, None);
        }
        eprintln!("daemon-vhc-worker: wgpu feature built but no usable adapter — CPU det lane");
    }
    (daemon_vhc_host::BackendKind::Cpu, None)
}
