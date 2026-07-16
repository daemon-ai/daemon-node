// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `claim()` and the five-stage admission funnel (architecture §3.5; ABI §9) — Phase A2.
//!
//! Admission is a funnel **ordered by cost and bracketed by the owner at both ends**: cheap local
//! policy gates first, the claim-dependent resource authorization last. No later stage runs before
//! an earlier one passes; nothing before stage 4 fetches or executes guest code (refactor
//! invariant 8). [`admit_v2`] implements the stages for the major-2 path; the v1 path's
//! autotune-based admission is byte-for-byte untouched (it retires only at the Phase-E sunset).
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

/// A successful admission: the selection, the decoded claim, and the verbatim canonical bytes the
/// run header journals (§8.3 tag 0 `claim` / `manifest`).
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
    let selection = select_driver(worker, wasm, expected_blake3)
        .map_err(|refusal| FunnelRefusal::typed(4, refusal))?;
    if selection.driver != CandidateDriver::V2 {
        return Err(FunnelRefusal::local(
            4,
            "admit_v2 is the major-2 funnel; the v1 path keeps autotune admission (refactor §5 A2)",
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
    for ch in &assessed.manifest_channels {
        if !PHASE_A_DEFAULT_CHANNEL_TABLE
            .iter()
            .any(|d| u64::from(d.id) == *ch)
        {
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
        )
        .unwrap_err();
        assert_eq!(err.stage, 2);
        assert!(err.reason.contains("below lane floor"));
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
