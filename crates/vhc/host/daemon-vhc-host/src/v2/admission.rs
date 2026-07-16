// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `claim()` and the five-stage admission funnel (architecture §3.5; ABI §9) — Phase A2.
//!
//! Admission is a funnel **ordered by cost and bracketed by the owner at both ends**: cheap local
//! policy gates first, the claim-dependent resource authorization last. No later stage runs before
//! an earlier one passes; nothing before stage 4 fetches or executes guest code (refactor
//! invariant 8). [`admit_v2`] implements the stages for the major-2 path — the only admission
//! path since the Phase-E sunset retired the v1 driver's autotune-based admission.
//!
//! ## The Phase-A pre-screen branch (recorded determination — decisions D3 cell 5 / D7; ABI §9.3)
//!
//! **Branch B: interim v2 runs gate on the lane floor alone.** A2's optional additive
//! `device_min` envelope section is deliberately **not relied upon** here: the ratified decision
//! makes cell 5 conditional on a standing three-proviso whole-run fixture (old-reader open;
//! bytes/hash preserved end-to-end; no decode→re-freeze path). Static inspection of the frozen
//! decoder is favorable (no `deny_unknown_fields`; the hash is over received bytes; no
//! decode→re-freeze chain in-tree), but that is the decision's "preliminary evidence, not proof" —
//! the fixture needs the full store/forward path a live v2 run exercises, and until it passes,
//! stage 3 evaluates `max(lane floor, ∅)` — i.e. the lane floor of stage 2. D0 makes the
//! device-minimums section mandatory and re-points stage 3 at it.
//!
//! ## Owner arbitration scope (decisions D6)
//!
//! Stage 5 here is the conservative A2 shape: a single active worker role-instance per host (the
//! existing single-run supervision guard), judged against the owner's standing scalar caps. The
//! typed per-device + host-wide `OwnerBudget` ledgers, atomic check-and-reserve, preemption
//! ordering, and crash reconciliation are Phase E per the ratified decision — the claim tiers are
//! already summed disjointly here exactly as those ledgers will reserve them (§9.6 mapping).

use wasmtime::{ExternType, Linker, Module, Store};

use daemon_vhc_abi::{
    AbiRefusal, AbiRefusalCode, CandidateDriver, CLAIM_KEY_DECLARED_PEAK,
    CLAIM_KEY_HARD_ACCOUNTABLE, CLAIM_KEY_UNDER_PRESSURE, CLAIM_KEY_WORKSPACE,
    CLAIM_TIER_KEY_DEVICE, CLAIM_TIER_KEY_HOST, PHASE_A_DEFAULT_CHANNEL_TABLE,
};

use crate::runtime::{HostState, Worker};
use crate::select::{select_driver, Selection};

/// One tier of the §9.1 memory claim: `{device, host}` in raw bytes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TierBytes {
    /// Device-tier bytes (charged to the bound accelerator's ledger, decisions D6).
    pub device: u64,
    /// Host-tier bytes.
    pub host: u64,
}

/// The decoded §9.1 tiered memory claim.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryClaim {
    /// Resources the host meters exactly — the enforceable cap.
    pub hard_accountable: TierBytes,
    /// The expected high-water mark (admission input, not hard-enforced).
    pub declared_peak: TierBytes,
    /// Host-side costs the module is never blamed for.
    pub workspace: TierBytes,
    /// Ordered degradation steps (`CLAIM_PRESSURE_*`).
    pub under_pressure: Vec<u64>,
}

impl MemoryClaim {
    /// The disjoint tier sum this instance occupies on its bound device (§9.6 ledger mapping —
    /// charging only the hard tier would overcommit, decisions D6).
    #[must_use]
    pub fn device_total(&self) -> u64 {
        self.hard_accountable
            .device
            .saturating_add(self.declared_peak.device)
            .saturating_add(self.workspace.device)
    }

    /// The disjoint tier sum on the host ledger.
    #[must_use]
    pub fn host_total(&self) -> u64 {
        self.hard_accountable
            .host
            .saturating_add(self.declared_peak.host)
            .saturating_add(self.workspace.host)
    }
}

/// The §9.6 `ParticipationLane` fields the Phase-A funnel consumes. Lanes are **versioned
/// node-side configuration** — exact numbers are deployment config, never ABI constants; envelopes
/// may tighten a lane, never weaken it (`max(lane floor, envelope minimums)`).
#[derive(Debug, Clone)]
pub struct ParticipationLane {
    /// The lane name (`"trainer"` at launch; Verifier/Coordinator reserved).
    pub lane: String,
    /// The lane profile version.
    pub version: u64,
    /// The node-side owner switch (reserved lanes ship `false`).
    pub enabled: bool,
    /// Device minima: GPU requirement (0 = forbidden, 1 = optional, 2 = required).
    pub gpu: u64,
    /// Device minima: VRAM floor in raw bytes.
    pub vram_bytes: u64,
    /// Device minima: host RAM floor in raw bytes.
    pub ram_bytes: u64,
    /// Device minima: free-disk floor in raw bytes.
    pub disk_bytes: u64,
    /// Claim sanity bounds: `[min, max]` bytes the claim's device total must fall within.
    pub claim_bounds_device: [u64; 2],
    /// Claim sanity bounds: `[min, max]` bytes the claim's host total must fall within.
    pub claim_bounds_host: [u64; 2],
    /// The lane's grant/channel ceilings (ABI §9.6 `channel_ceilings` + event/buffer/op bounds +
    /// offered worlds/custom-ops) — the "lane profile ceilings" contributor to the §2.6 grants
    /// derivation. D0: an envelope role grant list is intersected against these, tighten-only
    /// (`daemon_vhc_proto::derive_admitted_quotas`); exceeding them is `GrantsExceedLane`.
    pub ceilings: daemon_vhc_proto::LaneCeilings,
}

impl ParticipationLane {
    /// The launch **Trainer** profile shape (ratified: GPU required, the 16 GiB-class VRAM floor,
    /// all four worlds, bridge allowed — ABI §9.6). Numbers here are the deployment-config
    /// defaults a node overrides; they are NOT ABI constants.
    #[must_use]
    pub fn trainer_launch_defaults() -> Self {
        Self {
            lane: "trainer".to_string(),
            version: 1,
            enabled: true,
            gpu: 2,               // required
            vram_bytes: 16 << 30, // the 16/24 GiB-class floor (deployment config)
            ram_bytes: 8 << 30,
            disk_bytes: 10 << 30,
            claim_bounds_device: [0, 64 << 30],
            claim_bounds_host: [0, 64 << 30],
            ceilings: daemon_vhc_proto::LaneCeilings {
                max_frame_bytes: 16 << 20,
                spool_frames: 1024,
                per_sender_quota: 256,
                replay_window: 4096,
                rate_per_min: 6000,
                advisory_depth: 1024,
                payload_depth: 1024,
                gossip_depth: 1024,
                max_live_handles: 1024,
                max_live_bytes: 1 << 30,
                max_readback_bytes: 64 << 20,
                max_outstanding_ops: 256,
                // The compute@2 queue-depth ceiling (C1; mirrors the `V2RunConfig` default).
                compute_queue_depth: 1024,
                // The worlds a Trainer-lane role may be granted (§9.6: all four capability
                // worlds; `vhc` loop mechanics always; `tabi` while the bridge is advertised).
                worlds: ["vhc", "net", "sys", "data", "compute", "tabi"]
                    .iter()
                    .map(|s| (*s).to_string())
                    .collect(),
                // The lane offers exactly what the host custom-op registry advertises (C2's
                // `HOST_CUSTOM_OPS` — `flash_attn@1` today); D0's tighten-only `custom_ops ⊆
                // lane` check composes with C2's stage-4.2½ registry gate.
                custom_ops: daemon_vhc_abi::HOST_CUSTOM_OPS
                    .iter()
                    .map(|s| (*s).to_string())
                    .collect(),
            },
        }
    }
}

/// The host-probed device profile the lane floor is judged against (the permanent
/// `daemon-vhc-probe` mechanism; module-independent).
#[derive(Debug, Clone, Copy, Default)]
pub struct DeviceProfile {
    /// A usable GPU is present.
    pub gpu: bool,
    /// Dedicated device memory in raw bytes.
    pub vram_bytes: u64,
    /// Host RAM in raw bytes.
    pub ram_bytes: u64,
    /// Free disk in raw bytes.
    pub disk_bytes: u64,
}

impl DeviceProfile {
    /// The guest-readable canonical-CBOR wire form: the bytes the run header journals (§8.3
    /// tag 0 `device`), `sys@2::device_profile` delivers (journaled per delivery as tag 15), and
    /// a module's autotune reads (architecture §3.5 — "the module reads the same device profile
    /// the probe measures"). A string-keyed map with raw-byte units, additive by minor.
    #[must_use]
    pub fn to_wire(&self) -> Vec<u8> {
        let uint = |v: u64| ciborium::value::Value::Integer(v.into());
        let text = |s: &str| ciborium::value::Value::Text(s.into());
        let v = ciborium::value::Value::Map(vec![
            (text("gpu"), ciborium::value::Value::Bool(self.gpu)),
            (text("vram_bytes"), uint(self.vram_bytes)),
            (text("ram_bytes"), uint(self.ram_bytes)),
            (text("disk_bytes"), uint(self.disk_bytes)),
        ]);
        daemon_vhc_proto::to_canonical_vec(&v).expect("device profile cbor")
    }
}

/// The owner's standing policy inputs to the funnel's two bracketing stages (architecture §3.5).
/// The A2 shape is scalar caps under the single-instance guard; the typed per-device ledgers are
/// Phase E (decisions D6).
#[derive(Debug, Clone, Copy)]
pub struct OwnerPolicy {
    /// Stage 1: the feature/participation gate. `false` refuses before any network/guest contact.
    pub participation_enabled: bool,
    /// Stage 5: the standing device-memory cap in raw bytes (`0` = uncapped).
    pub vram_cap_bytes: u64,
    /// Stage 5: the standing host-memory cap in raw bytes (`0` = uncapped).
    pub host_cap_bytes: u64,
}

/// The envelope-v2 grants input to the funnel (D0): one role's grant list from the genesis
/// envelope plus the run's committed artifact-map hashes. `None` at the funnel means "no envelope
/// grants" — the pre-D0 (v1-envelope / Phase-A default) path, where the driver defaults stand.
#[derive(Debug, Clone)]
pub struct EnvelopeRoleGrants {
    /// The role's grant list (`GenesisEnvelope::roles[role].grants`).
    pub grants: daemon_vhc_proto::RoleGrants,
    /// The blake3 hashes of the run's artifact map (granted artifacts must be a subset).
    pub run_artifacts: std::collections::BTreeSet<daemon_vhc_proto::Hash>,
}

impl EnvelopeRoleGrants {
    /// Extract one role's grants input from a decoded genesis envelope. `None` if the role is not
    /// in the envelope's role set (the caller refuses upstream — an unknown role never admits).
    #[must_use]
    pub fn from_genesis(env: &daemon_vhc_proto::GenesisEnvelope, role: &str) -> Option<Self> {
        env.roles.get(role).map(|entry| Self {
            grants: entry.grants.clone(),
            run_artifacts: env.artifacts.values().map(|a| a.blake3).collect(),
        })
    }
}

/// A successful admission: the selection, the decoded claim, the verbatim canonical bytes the
/// run header journals (§8.3 tag 0 `claim` / `manifest`), and — when the run carried envelope-v2
/// grants — the tighten-derived quotas the run config consumes.
#[derive(Debug, Clone)]
pub struct AdmissionV2 {
    /// The ABI §1.3 selection (driver, major, minor).
    pub selection: Selection,
    /// The decoded, bounds-checked claim.
    pub claim: MemoryClaim,
    /// The claim's verbatim CBOR bytes (byte-identity is part of the contract, §9.2).
    pub claim_bytes: Vec<u8>,
    /// The manifest's verbatim CBOR bytes.
    pub manifest_bytes: Vec<u8>,
    /// The admitted numeric quotas derived from `lane ∩ envelope role grants` (D0, ABI §2.6 core;
    /// tighten-only). `None` on the pre-D0 path (no envelope grants) — the `V2RunConfig` defaults
    /// stand there.
    pub quotas: Option<daemon_vhc_proto::AdmittedQuotas>,
}

impl AdmissionV2 {
    /// Copy the admitted quotas into a [`crate::v2::V2RunConfig`] — the D0 envelope→admission→
    /// run-config derivation seam (deliverable 2). A no-op when the admission carried no envelope
    /// grants (the config's Phase-A defaults stand). `granted_artifacts` REPLACES the config's
    /// set (the envelope is the authority; empty grants = no artifacts, fail closed).
    pub fn apply_quotas(&self, cfg: &mut crate::v2::V2RunConfig) {
        if let Some(q) = &self.quotas {
            apply_admitted_quotas(q, cfg);
        }
    }
}

/// Copy one derived [`daemon_vhc_proto::AdmittedQuotas`] into a [`crate::v2::V2RunConfig`] — the
/// single quota→config mapping shared by [`AdmissionV2::apply_quotas`] and the worker's
/// envelope-v2 join path (D1 deliverable 4: the join derives the role's quotas from the genesis
/// grants and applies them here, so assess and join agree on the mapping by construction).
///
/// `granted_artifacts` REPLACES the config's set (the envelope is the authority; empty grants =
/// no artifacts, fail closed). `0` admitted = "unbounded by this grant" (ABI §2.3): the driver
/// treats 0 the same way, so the values copy through verbatim.
pub fn apply_admitted_quotas(
    q: &daemon_vhc_proto::AdmittedQuotas,
    cfg: &mut crate::v2::V2RunConfig,
) {
    cfg.max_frame_bytes = u32::try_from(q.max_frame_bytes).unwrap_or(u32::MAX);
    cfg.advisory_depth = usize::try_from(q.advisory_depth).unwrap_or(usize::MAX);
    cfg.payload_depth = usize::try_from(q.payload_depth).unwrap_or(usize::MAX);
    cfg.gossip_depth = usize::try_from(q.gossip_depth).unwrap_or(usize::MAX);
    cfg.spool_frames = usize::try_from(q.spool_frames).unwrap_or(usize::MAX);
    cfg.per_sender_quota = usize::try_from(q.per_sender_quota).unwrap_or(usize::MAX);
    cfg.max_readback_bytes_per_slice = q.max_readback_bytes;
    cfg.max_live_buffer_handles = q.max_live_handles;
    cfg.max_live_buffer_bytes = q.max_live_bytes;
    cfg.max_outstanding_ops = q.max_outstanding_ops;
    cfg.compute_queue_depth = q.compute_queue_depth;
    cfg.granted_artifacts = q.granted_artifacts.iter().map(|h| h.0).collect();
}

/// A funnel refusal: which stage refused, the typed code where one is ratified (§9.5: stages 1–3
/// are local eligibility outcomes with no ABI code; stages 4–5 carry the split codes).
#[derive(Debug, Clone)]
pub struct FunnelRefusal {
    /// The refusing stage (1–5, architecture §3.5 numbering).
    pub stage: u8,
    /// The ratified refusal code, for stage-4/5 refusals.
    pub code: Option<AbiRefusalCode>,
    /// The human-readable reason (names observed vs required, §1.5 discipline).
    pub reason: String,
}

impl FunnelRefusal {
    fn local(stage: u8, reason: impl Into<String>) -> Self {
        Self {
            stage,
            code: None,
            reason: reason.into(),
        }
    }

    fn typed(stage: u8, refusal: AbiRefusal) -> Self {
        Self {
            stage,
            code: Some(refusal.code),
            reason: refusal.detail,
        }
    }
}

impl std::fmt::Display for FunnelRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.code {
            Some(code) => write!(f, "stage {}: {}: {}", self.stage, code.slug(), self.reason),
            None => write!(f, "stage {}: {}", self.stage, self.reason),
        }
    }
}

/// Run the owner-bracketed five-stage admission funnel for a major-2 module (architecture §3.5;
/// ABI §9.3/§9.4). Stage order is load-bearing and never inverted (refactor invariant 8).
///
/// # Errors
/// A [`FunnelRefusal`] naming the refusing stage; stages 4–5 carry the ratified split codes.
// The argument list mirrors the funnel's staged inputs one-to-one (module identity, admitted
// bytes, and the three policy sources bracketing it) — grouping them further would hide the
// stage structure the spec fixes.
#[allow(clippy::too_many_arguments)]
pub fn admit_v2(
    worker: &Worker,
    wasm: &[u8],
    expected_blake3: Option<&[u8; 32]>,
    config: &[u8],
    grants: &[u8],
    lane: &ParticipationLane,
    device: &DeviceProfile,
    owner: &OwnerPolicy,
    envelope_min: Option<&daemon_vhc_proto::DeviceMinimums>,
    envelope_grants: Option<&EnvelopeRoleGrants>,
) -> Result<AdmissionV2, FunnelRefusal> {
    // -- stage 1: owner participation policy — free, local, before ANY other work ----------------
    if !owner.participation_enabled {
        return Err(FunnelRefusal::local(
            1,
            "owner participation policy: feature not enabled",
        ));
    }

    // -- stage 2: lane feature floor — device profile vs the enabled lane, no module fetch --------
    if !lane.enabled {
        return Err(FunnelRefusal::local(
            2,
            format!("lane `{}` is not enabled on this node", lane.lane),
        ));
    }
    if lane.gpu == 2 && !device.gpu {
        return Err(FunnelRefusal::local(
            2,
            format!("below lane floor: lane `{}` requires a GPU", lane.lane),
        ));
    }
    if device.gpu && device.vram_bytes < lane.vram_bytes {
        return Err(FunnelRefusal::local(
            2,
            format!(
                "below lane floor: vram {} < lane floor {}",
                device.vram_bytes, lane.vram_bytes
            ),
        ));
    }
    if device.ram_bytes < lane.ram_bytes || device.disk_bytes < lane.disk_bytes {
        return Err(FunnelRefusal::local(2, "below lane floor: ram/disk"));
    }

    // -- stage 3: run pre-screen — `max(lane floor, envelope minimums)` ---------------------------
    // Branch shipped (D3 cell 5, recorded per the ratified conditional): the three-proviso
    // envelope fixture is GREEN (`daemon-vhc-proto/tests/cell5_envelope.rs` — old-reader open,
    // bytes/hash end-to-end, decode→re-freeze strip canary), so cell 5 is **interim-supported**
    // and the additive `device_min` section is consumed here. The section is parsed from the RAW
    // frozen bytes at the CBOR-value level (`FrozenEnvelope::device_min`), never via the typed
    // `Envelope` (which silently drops unknown keys — the proviso-3 trap). An envelope may only
    // TIGHTEN the lane floor (§9.5): each present field refuses below its minimum; absent fields
    // constrain nothing. Mandatory per-role minimums arrive with the D0 genesis schema.
    if let Some(min) = envelope_min {
        if min.gpu == Some(2) && !device.gpu {
            return Err(FunnelRefusal::local(
                3,
                "run pre-screen: envelope requires a GPU (device_min.gpu = 2)",
            ));
        }
        let below = |have: u64, want: Option<u64>| want.is_some_and(|w| have < w);
        if below(
            if device.gpu { device.vram_bytes } else { 0 },
            min.vram_bytes,
        ) || below(device.ram_bytes, min.ram_bytes)
            || below(device.disk_bytes, min.disk_bytes)
        {
            return Err(FunnelRefusal::local(
                3,
                "run pre-screen: device < max(lane floor, envelope minimums)",
            ));
        }
    }

    // -- stage 4: module fetch + assessment — selection, manifest, claim (§9.4 steps 1–7) ---------
    //
    // Stage-4 gates run in cost order, and the funnel stays COMPOSABLE (ratified admission-funnel
    // design):
    //   4.0  envelope grants vs lane (tighten-only derivation — pure, needs no module byte);
    //   4.1  hash-verify → compile → driver selection (§1.3);
    //   4.2  manifest decode + ABI echo + channel-table membership (vs the ADMITTED table);
    //   4.2½ C2's custom-op registry gate (after manifest decode — the manifest names the ops);
    //   4.3  claim evaluation vs lane claim bounds.
    //
    // 4.0 — the D0 grants derivation (ABI §2.6 lane ∩ envelope core): an envelope role grant list
    // exceeding the lane's ceilings is refused `GrantsExceedLane` BEFORE any module byte is
    // compiled (architecture §3.5 — "grants exceeding the lane's bounds are refused").
    let quotas = match envelope_grants {
        Some(eg) => Some(
            daemon_vhc_proto::derive_admitted_quotas(&eg.grants, &lane.ceilings, &eg.run_artifacts)
                .map_err(|e| {
                    FunnelRefusal::typed(
                        4,
                        AbiRefusal::new(AbiRefusalCode::GrantsExceedLane, e.to_string()),
                    )
                })?,
        ),
        None => None,
    };

    let selection = select_driver(worker, wasm, expected_blake3)
        .map_err(|refusal| FunnelRefusal::typed(4, refusal))?;
    if selection.driver != CandidateDriver::V2 {
        return Err(FunnelRefusal::local(
            4,
            "admit_v2 is the major-2 funnel; no other driver is admissible (the v1 driver \
             retired at the Phase-E sunset)",
        ));
    }
    let module = Module::new(worker.engine(), wasm)
        .map_err(|e| FunnelRefusal::local(4, format!("recompile: {e}")))?;
    let assessed = assess_instance(worker, &module, config, grants)
        .map_err(|refusal| FunnelRefusal::typed(4, refusal))?;

    // Manifest checks (§9.4 step 6, the Phase-A subset): the ABI echo must match the selection
    // (AbiDeclarationMismatch) and every channel the manifest names must be in the admitted
    // table — the Phase-A default table until D0 (GrantsExceedLane).
    let declared = (u64::from(selection.major) << 16) | u64::from(selection.minor);
    if assessed.manifest_abi != declared {
        return Err(FunnelRefusal::typed(
            4,
            AbiRefusal::new(
                AbiRefusalCode::AbiDeclarationMismatch,
                format!(
                    "manifest abi echo {:#x} contradicts the selected declaration {declared:#x}",
                    assessed.manifest_abi
                ),
            ),
        ));
    }
    // The ADMITTED channel table (§6.2): from D0 the per-role table comes from the genesis
    // envelope's grant list when present; the Phase-A default table serves the pre-D0 path.
    let channel_admitted = |ch: u64| -> bool {
        match envelope_grants {
            Some(eg) => eg.grants.channels.iter().any(|d| u64::from(d.id) == ch),
            None => PHASE_A_DEFAULT_CHANNEL_TABLE
                .iter()
                .any(|d| u64::from(d.id) == ch),
        }
    };
    for ch in &assessed.manifest_channels {
        if !channel_admitted(*ch) {
            return Err(FunnelRefusal::typed(
                4,
                AbiRefusal::new(
                    AbiRefusalCode::GrantsExceedLane,
                    format!(
                        "manifest names channel {ch}, absent from the admitted channel table \
                         (§6.2/§9.4 step 6)"
                    ),
                ),
            ));
        }
    }

    // Custom-op admission (§9.4 step 6; architecture §3.2): every custom op the manifest requires
    // MUST be advertised by the host custom-op registry, else a clean typed refusal — never a
    // trap. `flash_attn@1` is the first registered fusion; the RESERVED compute@2 OperationIr::Custom
    // wire (owned/refused by C1) resolves NAMED ops through this same registry (C2:custom-op).
    crate::v2::custom_op::CustomOpRegistry::default()
        .admit(&assessed.manifest_custom_ops)
        .map_err(|refusal| FunnelRefusal::typed(4, refusal))?;

    // Claim-vs-lane sanity bounds (§9.3 stage 4 tail: "claim within the lane's claim bounds").
    let claim = &assessed.claim;
    let dt = claim.device_total();
    let ht = claim.host_total();
    let [dmin, dmax] = lane.claim_bounds_device;
    let [hmin, hmax] = lane.claim_bounds_host;
    if dt < dmin || dt > dmax || ht < hmin || ht > hmax {
        return Err(FunnelRefusal::typed(
            4,
            AbiRefusal::new(
                AbiRefusalCode::ClaimExceedsPolicy,
                format!(
                    "claim outside lane `{}` bounds: device {dt} ∉ [{dmin}, {dmax}] or \
                     host {ht} ∉ [{hmin}, {hmax}]",
                    lane.lane
                ),
            ),
        ));
    }

    // -- stage 5: claim vs owner resource authorization — last (needs the claim), supreme ---------
    // Conservative A2 arbitration (decisions D6): the single-instance guard + scalar caps; the
    // disjoint tier sum is exactly what the Phase-E ledgers will reserve.
    if owner.vram_cap_bytes != 0 && dt > owner.vram_cap_bytes {
        return Err(FunnelRefusal::typed(
            5,
            AbiRefusal::new(
                AbiRefusalCode::ClaimExceedsPolicy,
                format!(
                    "claim device total {dt} exceeds the owner's standing cap {} (owner policy \
                     is supreme)",
                    owner.vram_cap_bytes
                ),
            ),
        ));
    }
    if owner.host_cap_bytes != 0 && ht > owner.host_cap_bytes {
        return Err(FunnelRefusal::typed(
            5,
            AbiRefusal::new(
                AbiRefusalCode::ClaimExceedsPolicy,
                format!(
                    "claim host total {ht} exceeds the owner's standing cap {}",
                    owner.host_cap_bytes
                ),
            ),
        ));
    }

    Ok(AdmissionV2 {
        selection,
        claim: assessed.claim,
        claim_bytes: assessed.claim_bytes,
        manifest_bytes: assessed.manifest_bytes,
        quotas,
    })
}

/// What the restricted assessment instance yielded (§9.2/§9.4 steps 4–7).
struct Assessed {
    claim: MemoryClaim,
    claim_bytes: Vec<u8>,
    manifest_bytes: Vec<u8>,
    manifest_abi: u64,
    manifest_channels: Vec<u64>,
    manifest_custom_ops: Vec<String>,
}

/// The fuel budget for the whole assessment pass (`da_manifest` + `da_claim` ×2): minimal — a
/// claim is a cheap compute-free function (§9.1); exhausting this is nonconforming.
const ASSESS_FUEL: u64 = 1 << 22;

/// §9.2's assessment instance: every capability import bound to a deterministic deny-on-call stub
/// (calling any of them traps `ClaimCapabilityDenied` and refuses admission), minimal fuel, a
/// tight epoch deadline — used for `da_manifest` + `da_claim`, then **discarded, never promoted**.
fn assess_instance(
    worker: &Worker,
    module: &Module,
    config: &[u8],
    grants: &[u8],
) -> Result<Assessed, AbiRefusal> {
    let mut store: Store<HostState> = Store::new(worker.engine(), HostState::new(worker.config()));
    store
        .set_fuel(ASSESS_FUEL)
        .map_err(|e| AbiRefusal::new(AbiRefusalCode::BadModule, format!("fuel seeding: {e}")))?;
    store.set_epoch_deadline(worker.epoch_ticks_pub());

    let mut linker: Linker<HostState> = Linker::new(worker.engine());
    for import in module.imports() {
        let ExternType::Func(func_ty) = import.ty() else {
            return Err(AbiRefusal::new(
                AbiRefusalCode::BadModule,
                format!(
                    "non-function import `{}::{}`",
                    import.module(),
                    import.name()
                ),
            ));
        };
        let denied = format!(
            "ClaimCapabilityDenied: capability import `{}::{}` called during assessment \
             (ABI §9.2 deny-on-call stub)",
            import.module(),
            import.name()
        );
        linker
            .func_new(import.module(), import.name(), func_ty, move |_, _, _| {
                Err(wasmtime::Error::msg(denied.clone()))
            })
            .map_err(|e| AbiRefusal::new(AbiRefusalCode::BadModule, format!("stub link: {e}")))?;
    }
    let instance = linker.instantiate(&mut store, module).map_err(|e| {
        AbiRefusal::new(
            AbiRefusalCode::BadModule,
            format!("assessment instantiation: {e}"),
        )
    })?;

    // Write the config + grants spans via da_alloc (outside import context, §2.4 / §9.4 step 4).
    let memory = instance
        .get_memory(&mut store, "memory")
        .ok_or_else(|| AbiRefusal::new(AbiRefusalCode::BadModule, "no exported memory"))?;
    let alloc = instance
        .get_typed_func::<(u32, u32), u32>(&mut store, "da_alloc")
        .map_err(|_| AbiRefusal::new(AbiRefusalCode::BadModule, "missing da_alloc"))?;
    let write_span = |store: &mut Store<HostState>, bytes: &[u8]| -> Result<u32, AbiRefusal> {
        if bytes.is_empty() {
            return Ok(0);
        }
        let ptr = alloc
            .call(&mut *store, (bytes.len() as u32, 1))
            .map_err(|e| deny_or_bad(e, store))?;
        if ptr == 0 {
            return Err(AbiRefusal::new(
                AbiRefusalCode::BadModule,
                "da_alloc returned 0 during assessment",
            ));
        }
        memory
            .write(&mut *store, ptr as usize, bytes)
            .map_err(|e| AbiRefusal::new(AbiRefusalCode::BadModule, format!("span write: {e}")))?;
        Ok(ptr)
    };
    let cfg_ptr = write_span(&mut store, config)?;
    let grants_ptr = write_span(&mut store, grants)?;

    // da_manifest — decode the Phase-A fields (abi echo + channels).
    let manifest_bytes = call_cbor_export(
        &instance,
        &mut store,
        "da_manifest",
        &[cfg_ptr, config.len() as u32],
    )?;
    let (manifest_abi, manifest_channels, manifest_custom_ops) = decode_manifest(&manifest_bytes)?;

    // da_claim — twice, byte-identical, deterministic (§9.2): `ClaimInconsistent` on divergence.
    let args = [
        cfg_ptr,
        config.len() as u32,
        grants_ptr,
        grants.len() as u32,
    ];
    let claim_bytes = call_cbor_export(&instance, &mut store, "da_claim", &args)?;
    let claim_again = call_cbor_export(&instance, &mut store, "da_claim", &args)?;
    if claim_bytes != claim_again {
        return Err(AbiRefusal::new(
            AbiRefusalCode::ClaimInconsistent,
            format!(
                "repeated da_claim invocations returned different bytes ({} vs {})",
                claim_bytes.len(),
                claim_again.len()
            ),
        ));
    }
    let claim = decode_claim(&claim_bytes)?;

    Ok(Assessed {
        claim,
        claim_bytes,
        manifest_bytes,
        manifest_abi,
        manifest_channels,
        manifest_custom_ops,
    })
    // `store` drops here — the assessment instance is discarded, never promoted (§9.2).
}

/// Map an assessment-time wasm error into a typed refusal. A deny-stub trap
/// (`ClaimCapabilityDenied`, §9.2) means the module's `da_manifest`/`da_claim` needed a
/// capability — nonconforming by §2.3/§9.1, refused `BadModule` with the offending import named
/// verbatim in the detail (the §1.5 exposed-code table routes assessment-instance faults under
/// the step-5/7 refusals; `ClaimCapabilityDenied` itself is the *trap* vocabulary, §7.6).
/// Fuel/epoch exhaustion under the minimal assessment budget refuses the same way (§9.2).
fn deny_or_bad(e: wasmtime::Error, _store: &mut Store<HostState>) -> AbiRefusal {
    let msg = e
        .chain()
        .map(std::string::ToString::to_string)
        .collect::<Vec<_>>()
        .join(": ");
    AbiRefusal::new(AbiRefusalCode::BadModule, msg)
}

/// Call a `(args…) -> (ptr<<32)|len` CBOR-returning export and copy the span out (§2.1).
fn call_cbor_export(
    instance: &wasmtime::Instance,
    store: &mut Store<HostState>,
    name: &str,
    args: &[u32],
) -> Result<Vec<u8>, AbiRefusal> {
    let packed = match args.len() {
        2 => instance
            .get_typed_func::<(u32, u32), u64>(&mut *store, name)
            .map_err(|_| {
                AbiRefusal::new(
                    AbiRefusalCode::BadModule,
                    format!("missing/mis-typed `{name}`"),
                )
            })?
            .call(&mut *store, (args[0], args[1])),
        4 => instance
            .get_typed_func::<(u32, u32, u32, u32), u64>(&mut *store, name)
            .map_err(|_| {
                AbiRefusal::new(
                    AbiRefusalCode::BadModule,
                    format!("missing/mis-typed `{name}`"),
                )
            })?
            .call(&mut *store, (args[0], args[1], args[2], args[3])),
        _ => unreachable!("assessment exports take 2 or 4 args"),
    }
    .map_err(|e| deny_or_bad(e, store))?;
    let (ptr, len) = ((packed >> 32) as u32, (packed & 0xffff_ffff) as usize);
    if len == 0 {
        return Ok(Vec::new());
    }
    let memory = instance
        .get_memory(&mut *store, "memory")
        .ok_or_else(|| AbiRefusal::new(AbiRefusalCode::BadModule, "no exported memory"))?;
    let start = ptr as usize;
    memory
        .data(&*store)
        .get(start..start + len)
        .map(<[u8]>::to_vec)
        .ok_or_else(|| {
            AbiRefusal::new(
                AbiRefusalCode::BadModule,
                format!("`{name}` returned an out-of-bounds span"),
            )
        })
}

/// Decode the §2.3 manifest fields the Phase-A funnel checks: the `abi` echo, `channels`, and the
/// versioned `custom_ops` the host custom-op registry admits (architecture §3.2).
fn decode_manifest(bytes: &[u8]) -> Result<(u64, Vec<u64>, Vec<String>), AbiRefusal> {
    let v: ciborium::value::Value = ciborium::de::from_reader(bytes)
        .map_err(|e| AbiRefusal::new(AbiRefusalCode::BadModule, format!("manifest CBOR: {e}")))?;
    let ciborium::value::Value::Map(entries) = v else {
        return Err(AbiRefusal::new(
            AbiRefusalCode::BadModule,
            "manifest is not a CBOR map",
        ));
    };
    let get = |key: &str| {
        entries.iter().find_map(|(k, val)| match k {
            ciborium::value::Value::Text(t) if t == key => Some(val),
            _ => None,
        })
    };
    let abi = get("abi")
        .and_then(ciborium::value::Value::as_integer)
        .map(|i| u64::try_from(i128::from(i)).unwrap_or(0))
        .ok_or_else(|| AbiRefusal::new(AbiRefusalCode::BadModule, "manifest missing `abi` echo"))?;
    let channels = match get("channels") {
        Some(ciborium::value::Value::Array(items)) => items
            .iter()
            .filter_map(|i| i.as_integer())
            .map(|i| u64::try_from(i128::from(i)).unwrap_or(u64::MAX))
            .collect(),
        _ => Vec::new(),
    };
    let custom_ops = match get("custom_ops") {
        Some(ciborium::value::Value::Array(items)) => items
            .iter()
            .filter_map(|v| match v {
                ciborium::value::Value::Text(t) => Some(t.clone()),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    };
    Ok((abi, channels, custom_ops))
}

/// Decode the §9.1 `memory-claim` map. Malformed → `BadModule` (the module violated its own
/// export contract); bounds/policy judgments happen in the funnel, not here.
fn decode_claim(bytes: &[u8]) -> Result<MemoryClaim, AbiRefusal> {
    let v: ciborium::value::Value = ciborium::de::from_reader(bytes)
        .map_err(|e| AbiRefusal::new(AbiRefusalCode::BadModule, format!("claim CBOR: {e}")))?;
    let ciborium::value::Value::Map(entries) = v else {
        return Err(AbiRefusal::new(
            AbiRefusalCode::BadModule,
            "claim is not a CBOR map",
        ));
    };
    let get = |key: &str| {
        entries.iter().find_map(|(k, val)| match k {
            ciborium::value::Value::Text(t) if t == key => Some(val),
            _ => None,
        })
    };
    let tier = |key: &str| -> Result<TierBytes, AbiRefusal> {
        let Some(ciborium::value::Value::Map(m)) = get(key) else {
            return Err(AbiRefusal::new(
                AbiRefusalCode::BadModule,
                format!("claim missing tier `{key}`"),
            ));
        };
        let field = |name: &str| {
            m.iter()
                .find_map(|(k, val)| match k {
                    ciborium::value::Value::Text(t) if t == name => val.as_integer(),
                    _ => None,
                })
                .map(|i| u64::try_from(i128::from(i)).unwrap_or(0))
                .unwrap_or(0)
        };
        Ok(TierBytes {
            device: field(CLAIM_TIER_KEY_DEVICE),
            host: field(CLAIM_TIER_KEY_HOST),
        })
    };
    let under_pressure = match get(CLAIM_KEY_UNDER_PRESSURE) {
        Some(ciborium::value::Value::Array(items)) => items
            .iter()
            .filter_map(|i| i.as_integer())
            .map(|i| u64::try_from(i128::from(i)).unwrap_or(u64::MAX))
            .collect(),
        _ => Vec::new(),
    };
    Ok(MemoryClaim {
        hard_accountable: tier(CLAIM_KEY_HARD_ACCOUNTABLE)?,
        declared_peak: tier(CLAIM_KEY_DECLARED_PEAK)?,
        workspace: tier(CLAIM_KEY_WORKSPACE)?,
        under_pressure,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lane() -> ParticipationLane {
        ParticipationLane {
            lane: "trainer".into(),
            version: 1,
            enabled: true,
            gpu: 1, // optional (test lane; numbers are deployment config)
            vram_bytes: 0,
            ram_bytes: 0,
            disk_bytes: 0,
            claim_bounds_device: [0, 4 << 30],
            claim_bounds_host: [0, 4 << 30],
            ceilings: ParticipationLane::trainer_launch_defaults().ceilings,
        }
    }

    #[test]
    fn stage1_owner_participation_gates_before_everything() {
        let worker = Worker::new(crate::runtime::EngineConfig::default()).unwrap();
        let owner = OwnerPolicy {
            participation_enabled: false,
            vram_cap_bytes: 0,
            host_cap_bytes: 0,
        };
        // Deliberately garbage wasm: stage 1 must refuse BEFORE any byte is inspected.
        let err = admit_v2(
            &worker,
            b"not wasm",
            None,
            &[],
            &[],
            &lane(),
            &DeviceProfile::default(),
            &owner,
            None,
            None,
        )
        .unwrap_err();
        assert_eq!(err.stage, 1);
        assert!(err.code.is_none(), "stage 1 is a local outcome (§9.5)");
    }

    #[test]
    fn stage2_lane_floor_gates_before_module_fetch() {
        let worker = Worker::new(crate::runtime::EngineConfig::default()).unwrap();
        let owner = OwnerPolicy {
            participation_enabled: true,
            vram_cap_bytes: 0,
            host_cap_bytes: 0,
        };
        let mut l = lane();
        l.gpu = 2; // required — the CPU-only device profile is below the floor
        let err = admit_v2(
            &worker,
            b"not wasm",
            None,
            &[],
            &[],
            &l,
            &DeviceProfile::default(),
            &owner,
            None,
            None,
        )
        .unwrap_err();
        assert_eq!(err.stage, 2);
        assert!(err.reason.contains("below lane floor"));
    }

    /// D0 deliverable 2: an envelope role grant list exceeding the lane's ceilings is a typed
    /// stage-4 `GrantsExceedLane` refusal — raised at gate 4.0, BEFORE any module byte is
    /// compiled (garbage wasm proves the ordering: the grants gate fires, not `BadModule`).
    #[test]
    fn stage4_envelope_grants_exceeding_lane_refused_before_compile() {
        let worker = Worker::new(crate::runtime::EngineConfig::default()).unwrap();
        let owner = OwnerPolicy {
            participation_enabled: true,
            vram_cap_bytes: 0,
            host_cap_bytes: 0,
        };
        let mut grants = daemon_vhc_proto::RoleGrants::default();
        grants.channels.push(daemon_vhc_proto::ChannelDecl {
            id: 0,
            name: "control".into(),
            class: 0,
            direction: 2,
            // Wider than the trainer lane's 16 MiB frame ceiling — tighten-only must refuse.
            max_frame_bytes: 1 << 30,
            rate_per_min: 60,
            spool_frames: Some(8),
            replay_window: Some(64),
            per_sender_quota: Some(4),
        });
        let eg = EnvelopeRoleGrants {
            grants,
            run_artifacts: std::collections::BTreeSet::new(),
        };
        let err = admit_v2(
            &worker,
            b"not wasm",
            None,
            &[],
            &[],
            &lane(),
            &DeviceProfile::default(),
            &owner,
            None,
            Some(&eg),
        )
        .unwrap_err();
        assert_eq!(err.stage, 4);
        assert_eq!(err.code, Some(AbiRefusalCode::GrantsExceedLane));
        assert!(err.reason.contains("max_frame_bytes"));
    }

    /// D1 deliverable 4 (grants threading): `EnvelopeRoleGrants::from_genesis` derives a role's
    /// grants straight from a genesis envelope and feeds the SAME admission seam the hand-built
    /// grants do — the production path the worker join threads in place of the pre-D1 `None`. Here
    /// the worker role's channel grant exceeds the trainer lane's frame ceiling, so the derived
    /// grants produce the identical stage-4 `GrantsExceedLane` refusal (before any compile).
    #[test]
    fn from_genesis_role_grants_feed_the_admission_seam() {
        use std::collections::BTreeMap;
        let worker = Worker::new(crate::runtime::EngineConfig::default()).unwrap();
        let owner = OwnerPolicy {
            participation_enabled: true,
            vram_cap_bytes: 0,
            host_cap_bytes: 0,
        };
        let mut grants = daemon_vhc_proto::RoleGrants::default();
        grants.channels.push(daemon_vhc_proto::ChannelDecl {
            id: 0,
            name: "control".into(),
            class: 0,
            direction: 2,
            max_frame_bytes: 1 << 30, // wider than the trainer lane's 16 MiB ceiling
            rate_per_min: 60,
            spool_frames: Some(8),
            replay_window: Some(64),
            per_sender_quota: Some(4),
        });
        let mut roles = BTreeMap::new();
        roles.insert(
            "worker".to_string(),
            daemon_vhc_proto::RoleEntry {
                lane: "trainer".into(),
                module: "worker-mod".into(),
                abi: "vhc@2".into(),
                config: ciborium::value::Value::Map(vec![]),
                grants,
                device_min: daemon_vhc_proto::DeviceMinimums::default(),
            },
        );
        roles.insert(
            "coordinator".to_string(),
            daemon_vhc_proto::RoleEntry {
                lane: "coordinator".into(),
                module: "coord-mod".into(),
                abi: "vhc@2".into(),
                config: ciborium::value::Value::Map(vec![]),
                grants: daemon_vhc_proto::RoleGrants::default(),
                device_min: daemon_vhc_proto::DeviceMinimums::default(),
            },
        );
        let mut artifacts = BTreeMap::new();
        artifacts.insert(
            "worker-mod".to_string(),
            daemon_vhc_proto::SnapshotArtifact {
                url: "r2://mods/worker.wasm".into(),
                blake3: daemon_vhc_proto::Hash([1u8; 32]),
                size: Some(4096),
            },
        );
        artifacts.insert(
            "coord-mod".to_string(),
            daemon_vhc_proto::SnapshotArtifact {
                url: "r2://mods/coord.wasm".into(),
                blake3: daemon_vhc_proto::Hash([2u8; 32]),
                size: Some(2048),
            },
        );
        let env = daemon_vhc_proto::GenesisEnvelope {
            run: daemon_vhc_proto::RunSectionV2 {
                schema: daemon_vhc_proto::GENESIS_SCHEMA_MAJOR,
                run_label: "grants-run".into(),
                min_peers: 1,
                max_peers: 8,
                access: daemon_vhc_proto::envelope::Access::Org,
            },
            roles,
            artifacts,
            authority: ciborium::value::Value::Map(vec![]),
            transport: daemon_vhc_proto::TransportSelection::default(),
            identities: daemon_vhc_proto::Identities::default(),
        };
        // The derivation the worker join threads (replacing the pre-D1 `None`).
        let eg = EnvelopeRoleGrants::from_genesis(&env, "worker").expect("worker role present");
        assert!(EnvelopeRoleGrants::from_genesis(&env, "no-such-role").is_none());

        let err = admit_v2(
            &worker,
            b"not wasm",
            None,
            &[],
            &[],
            &lane(),
            &DeviceProfile::default(),
            &owner,
            None,
            Some(&eg),
        )
        .unwrap_err();
        assert_eq!(err.stage, 4);
        assert_eq!(err.code, Some(AbiRefusalCode::GrantsExceedLane));
    }

    /// D0 deliverable 2: the admitted quotas copy into `V2RunConfig` (the Phase-B seam) —
    /// tightened values replace the defaults; the granted-artifact set replaces the config's.
    #[test]
    fn apply_quotas_copies_admitted_values_into_run_config() {
        let quotas = daemon_vhc_proto::AdmittedQuotas {
            max_frame_bytes: 4096,
            spool_frames: 32,
            per_sender_quota: 8,
            advisory_depth: 16,
            payload_depth: 24,
            gossip_depth: 12,
            max_live_handles: 40,
            max_live_bytes: 1 << 16,
            max_readback_bytes: 1 << 14,
            max_outstanding_ops: 6,
            compute_queue_depth: 48,
            granted_artifacts: [daemon_vhc_proto::Hash([7u8; 32])].into_iter().collect(),
        };
        let admission = AdmissionV2 {
            selection: Selection {
                driver: CandidateDriver::V2,
                major: 2,
                minor: 0,
            },
            claim: MemoryClaim {
                hard_accountable: TierBytes::default(),
                declared_peak: TierBytes::default(),
                workspace: TierBytes::default(),
                under_pressure: Vec::new(),
            },
            claim_bytes: Vec::new(),
            manifest_bytes: Vec::new(),
            quotas: Some(quotas),
        };
        let identity = crate::v2::RunIdentity {
            run_id: [0u8; 32],
            epoch: 0,
            role: "worker".into(),
            instance: 1,
            module: [0u8; 32],
        };
        let mut cfg = crate::v2::V2RunConfig::new(identity, [0u8; 32], vec![], vec![]);
        admission.apply_quotas(&mut cfg);
        assert_eq!(cfg.max_frame_bytes, 4096);
        assert_eq!(cfg.spool_frames, 32);
        assert_eq!(cfg.per_sender_quota, 8);
        assert_eq!(cfg.advisory_depth, 16);
        assert_eq!(cfg.payload_depth, 24);
        assert_eq!(cfg.gossip_depth, 12);
        assert_eq!(cfg.max_live_buffer_handles, 40);
        assert_eq!(cfg.max_live_buffer_bytes, 1 << 16);
        assert_eq!(cfg.max_readback_bytes_per_slice, 1 << 14);
        assert_eq!(cfg.max_outstanding_ops, 6);
        assert_eq!(cfg.compute_queue_depth, 48);
        assert_eq!(
            cfg.granted_artifacts,
            [[7u8; 32]]
                .into_iter()
                .collect::<std::collections::BTreeSet<_>>()
        );
    }

    #[test]
    fn decode_manifest_extracts_custom_ops_and_registry_gates_them() {
        // A manifest declaring one advertised + one unknown custom op.
        let manifest = ciborium::value::Value::Map(vec![
            (
                ciborium::value::Value::Text("abi".into()),
                ciborium::value::Value::Integer(0x2_0001.into()),
            ),
            (
                ciborium::value::Value::Text("custom_ops".into()),
                ciborium::value::Value::Array(vec![
                    ciborium::value::Value::Text("flash_attn@1".into()),
                    ciborium::value::Value::Text("fused_moe@1".into()),
                ]),
            ),
        ]);
        let mut bytes = Vec::new();
        ciborium::into_writer(&manifest, &mut bytes).unwrap();
        let (_abi, channels, custom_ops) = decode_manifest(&bytes).unwrap();
        assert!(channels.is_empty());
        assert_eq!(custom_ops, vec!["flash_attn@1", "fused_moe@1"]);
        // The registry that stage 4 runs refuses the unadvertised op, typed (not a trap).
        let err = crate::v2::custom_op::CustomOpRegistry::default()
            .admit(&custom_ops)
            .unwrap_err();
        assert_eq!(err.code, AbiRefusalCode::CustomOpUnsupported);
    }

    #[test]
    fn claim_decode_rejects_missing_tier_and_sums_disjointly() {
        let claim = ciborium::value::Value::Map(vec![(
            ciborium::value::Value::Text("hard_accountable".into()),
            ciborium::value::Value::Map(vec![]),
        )]);
        let mut bytes = Vec::new();
        ciborium::into_writer(&claim, &mut bytes).unwrap();
        assert_eq!(
            decode_claim(&bytes).unwrap_err().code,
            AbiRefusalCode::BadModule
        );

        let full = MemoryClaim {
            hard_accountable: TierBytes { device: 1, host: 2 },
            declared_peak: TierBytes {
                device: 10,
                host: 20,
            },
            workspace: TierBytes {
                device: 100,
                host: 200,
            },
            under_pressure: vec![0, 1],
        };
        assert_eq!(full.device_total(), 111);
        assert_eq!(full.host_total(), 222);
    }
}
