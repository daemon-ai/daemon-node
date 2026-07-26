// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The **permanent device probe** (architecture §3.5; decisions D5 "the device probe stays
//! forever") — per-platform device-memory/limit readouts feeding the admission funnel's lane
//! feature floor and run pre-screen.
//!
//! Until the Phase-E v1 sunset this module also carried the v1 driver's **autotune admission**
//! (the `MetaReport`-derived resource model + micro-batch verdict + OOM-probe halving ladder);
//! that path retired with the v1 five-phase driver (decisions D5 — "only *module-measuring*
//! autotune retires"; major-2 admission is the `claim()` funnel). What remains is the
//! module-independent probe machinery: [`DeviceLimits`] and the per-platform sources
//! (amdgpu sysfs, wgpu adapter limits, CUDA, DXGI, Metal).
//!
//! ## What wgpu exposes for VRAM probing (honest inventory)
//!
//! wgpu's `Adapter` exposes `get_info()` (name/backend/device type) and `limits()`
//! (`max_buffer_size` — the largest single allocation, a hard per-tensor ceiling). It does **not**
//! expose total or free VRAM: the Vulkan / WebGPU surface has no device-memory-size query. So
//! [`DeviceLimits::max_alloc_mb`] is the one truly device-honest number; total VRAM
//! ([`DeviceLimits::vram_mb`]) is sourced by the caller from the GPU-governor policy cap (§10.5) or
//! the node's effective-resource computation, not from wgpu. See [`WgpuProbe`] (feature `wgpu`).

const MIB: u64 = 1 << 20;

/// Probed (or policy-capped) device limits the admission funnel compares against.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeviceLimits {
    /// Dedicated VRAM in MiB — the device's own memory (sysfs `mem_info_vram_total` on
    /// Linux/amdgpu; a true lower bound). NOT wgpu-queryable; the worker sources it (see module
    /// docs). On a unified/integrated device this is the small "carve-out", not the usable budget
    /// (which spills into `shared_mb`).
    pub vram_mb: u64,
    /// Effective host RAM in MiB.
    pub ram_mb: u64,
    /// Largest single GPU allocation in MiB (wgpu `max_buffer_size`); `0` = unknown / unbounded
    /// (the CPU / ndarray lanes, or when no device was probed).
    pub max_alloc_mb: u64,
    /// Shared / spillover memory budget in MiB (GTT on Linux/amdgpu — `mem_info_gtt_total`; the
    /// unified-memory pool the device can page tensors into beyond its dedicated VRAM carve-out).
    /// `0` = none (a classic discrete GPU with no host spill). Additive; `Default` = 0 preserves
    /// the pre-UMA behavior.
    pub shared_mb: u64,
    /// Whether the device shares host DRAM (an integrated/unified GPU, or the CPU lane — from
    /// `AdapterInfo.device_type == IntegratedGpu | Cpu`). When set, the budget math treats device +
    /// host footprints as competing for ONE physical DRAM pool (a joint budget), instead of two
    /// independent VRAM / RAM budgets. Additive; `Default` = `false` preserves the discrete path.
    pub unified: bool,
}

// (A2 inversion): `oom_error_class` — the one host→session symbol beyond the seam — moved to the
// worker binary (`daemon-vhc-worker::transport`), which links both sides. The runtime still maps
// a wasmtime "memory" trap to [`crate::TrapCode::BudgetMemory`]; the wire ErrorClass mapping is
// the worker's business.

/// Parse an amdgpu sysfs memory-total file (`mem_info_vram_total` / `mem_info_gtt_total`) into MiB.
///
/// The kernel exposes these as a single decimal **byte** count (e.g. `"4294967296\n"` = 4096 MiB;
/// `"125829120000\n"` = 120000 MiB). Returns `None` on an empty / non-numeric file so the caller
/// can fall back to another source. Pure (no I/O) so the worker's real file read stays a thin
/// wrapper and the parse is unit-tested with fixture strings.
#[must_use]
pub fn parse_amdgpu_mem_mb(contents: &str) -> Option<u64> {
    contents.trim().parse::<u64>().ok().map(|bytes| bytes / MIB)
}

// =====================================================================================
// Portable host RAM + free-disk probes (platform-agnostic; the admission funnel's lane
// device-floor inputs). The worker's `Hardware` report sources `ram_mb` / `disk_free_mb`
// from these on EVERY platform — the pre-fix worker hardcoded `disk_free_mb: 0` and read RAM
// only from Linux `/proc`, so a macOS/Windows trainer spuriously refused the lane floor
// (`below lane floor: ram/disk`) despite hundreds of GiB free. All decision logic stays in the
// admission funnel; these only gather raw OS scalars (fixture-free — they read the live host,
// so the unit test only asserts "nonzero on the host OS").
// =====================================================================================

/// Total physical host RAM in MiB, portably: Linux `/proc/meminfo` `MemTotal`, macOS
/// `sysctl hw.memsize`, Windows `GlobalMemoryStatusEx().ullTotalPhys`. `0` only when the
/// platform source is unavailable (the caller then uses the large "unknown" budget sentinel so an
/// unprobed number never spuriously rejects).
#[must_use]
pub fn host_ram_mb() -> u64 {
    #[cfg(target_os = "linux")]
    {
        linux_mem_total_mb()
    }
    #[cfg(target_os = "macos")]
    {
        mac_ffi::ram_mb()
    }
    #[cfg(windows)]
    {
        win_ffi::ram_mb()
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
    {
        0
    }
}

/// Free disk space in MiB on the filesystem containing `path`, portably: unix `statvfs`
/// (`f_bavail × f_frsize` — space available to a non-privileged process), Windows
/// `GetDiskFreeSpaceExW` (free bytes available to the caller). `0` when the query fails (a
/// non-existent path, or an unsupported platform) — callers pass an existing path (the run's
/// cache / state home, or the process cwd).
#[must_use]
pub fn host_disk_free_mb(path: &std::path::Path) -> u64 {
    #[cfg(unix)]
    {
        unix_ffi::disk_free_mb(path)
    }
    #[cfg(windows)]
    {
        win_ffi::disk_free_mb(path)
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = path;
        0
    }
}

// =====================================================================================
// wgpu graphics-API selection (the fleet's third GPU API: DX12 on Windows).
//
// cubecl-wgpu 0.10's `AutoGraphicsApi::backend()` hardcodes `wgpu::Backend::Vulkan` on every
// non-macOS / non-wasm target (its `AUTO_GRAPHICS_BACKEND` override is `#[cfg(test)]`-gated), so a
// released Windows worker ALWAYS came up Vulkan and `WGPU_BACKEND=dx12` had no effect — the fleet's
// G-1 three-distinct-GPU-API requirement could never be met (Windows would run Vulkan, same as
// Strix). We select the graphics API EXPLICITLY at bring-up instead: Windows defaults to Dx12,
// with an operator override that MIRRORS how `DAEMON_TRAIN_BACKEND` selects the lane. Linux/macOS
// behavior is unchanged (AutoGraphicsApi = Vulkan on Linux, Metal on macOS).
// =====================================================================================

/// The env knob that overrides the wgpu graphics API selection, mirroring `DAEMON_TRAIN_BACKEND`'s
/// lane selection: `dx12` | `vulkan` | `metal` | `auto`. Unset ⇒ the platform default (Dx12 on
/// Windows, Auto elsewhere).
pub const GRAPHICS_API_ENV: &str = "DAEMON_TRAIN_GRAPHICS_API";

/// Which wgpu graphics API a bring-up selects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphicsApiChoice {
    /// cubecl's `AutoGraphicsApi` — Vulkan on Linux, Metal on macOS (the pre-fix behavior, kept
    /// unchanged for those platforms).
    Auto,
    /// Force Vulkan.
    Vulkan,
    /// Force Metal.
    Metal,
    /// Force DirectX 12 (the Windows fleet lane).
    Dx12,
}

/// Resolve the wgpu graphics API from the target OS + an optional operator override
/// (`DAEMON_TRAIN_GRAPHICS_API`). Pure + unit-tested cross-platform (the feature-gated bring-up
/// maps the choice onto `init_setup::<G>`):
///
/// - an explicit, recognized override wins on every OS (`dx12`/`vulkan`/`metal`/`auto`, case- and
///   whitespace-insensitive; an unrecognized value falls back to the platform default so a typo
///   never silently forces the wrong API);
/// - otherwise **Windows defaults to Dx12** (AutoGraphicsApi would pick Vulkan there), and every
///   other OS keeps **Auto** (Vulkan on Linux, Metal on macOS — unchanged).
#[must_use]
pub fn resolve_graphics_api(target_os: &str, override_env: Option<&str>) -> GraphicsApiChoice {
    let platform_default = if target_os == "windows" {
        GraphicsApiChoice::Dx12
    } else {
        GraphicsApiChoice::Auto
    };
    match override_env.map(|s| s.trim().to_ascii_lowercase()) {
        Some(v) if v.is_empty() => platform_default,
        Some(v) => match v.as_str() {
            "dx12" | "directx12" | "d3d12" => GraphicsApiChoice::Dx12,
            "vulkan" | "vk" => GraphicsApiChoice::Vulkan,
            "metal" => GraphicsApiChoice::Metal,
            "auto" => GraphicsApiChoice::Auto,
            _ => platform_default,
        },
        None => platform_default,
    }
}

/// The wgpu graphics API this host selects right now (the live-env resolution of
/// [`resolve_graphics_api`]).
#[must_use]
pub fn selected_graphics_api() -> GraphicsApiChoice {
    resolve_graphics_api(
        std::env::consts::OS,
        std::env::var(GRAPHICS_API_ENV).ok().as_deref(),
    )
}

/// Bring up (register) the default-or-`device` wgpu client under the SELECTED graphics API. This is
/// the single choke point that replaces cubecl's `AutoGraphicsApi` bring-up: it branches on
/// [`selected_graphics_api`] and calls `init_setup::<G>` with the concrete graphics type, so a
/// Windows worker registers a **Dx12** client. Returns the adapter's `WgpuProbe`; `None` never —
/// callers wrap it in `catch_unwind` (cubecl panics when no adapter matches the requested API).
#[cfg(feature = "wgpu")]
fn init_setup_selected(
    device: &burn::backend::wgpu::WgpuDevice,
    options: burn::backend::wgpu::RuntimeOptions,
) -> burn::backend::wgpu::WgpuSetup {
    use burn::backend::wgpu::{
        graphics::{AutoGraphicsApi, Dx12, Metal, Vulkan},
        init_setup,
    };
    match selected_graphics_api() {
        GraphicsApiChoice::Dx12 => init_setup::<Dx12>(device, options),
        GraphicsApiChoice::Vulkan => init_setup::<Vulkan>(device, options),
        GraphicsApiChoice::Metal => init_setup::<Metal>(device, options),
        GraphicsApiChoice::Auto => init_setup::<AutoGraphicsApi>(device, options),
    }
}

/// Ensure the wgpu compute client for `device` is registered under the selected graphics API,
/// idempotently. The probe brings up `DefaultDevice` (so the ceremony's default-GPU path is already
/// correct via [`probe_wgpu`]); this is the bring-up for an explicitly-placed device, which the
/// probe did not register — without it the router's lazy bring-up would fall back to cubecl's
/// `AutoGraphicsApi` (Vulkan on Windows).
///
/// An already-registered device (an "already registered" panic) is benign and reported as success.
/// **Every other bring-up failure is returned**, with the panic's own message and location: this
/// used to swallow them, so a placement naming an adapter the host does not have registered nothing
/// and returned "fine" — and the run then met the poisoned router lock that the swallowed panic had
/// left behind, one call later and with no mention of the device.
///
/// # Errors
/// The recorded panic text when the device cannot be brought up.
#[cfg(feature = "wgpu")]
pub fn ensure_wgpu_registered(device: &burn::backend::wgpu::WgpuDevice) -> Result<(), String> {
    let outcome = crate::device_panic::catch(true, || {
        let _ = init_setup_selected(device, burn::backend::wgpu::RuntimeOptions::default());
    });
    match outcome {
        Ok(()) => Ok(()),
        Err(text) if text.contains("already registered") => Ok(()),
        Err(text) => Err(text),
    }
}

/// The `WgpuDevice` one measured device PLACEMENT names, honouring the adapter class the probe
/// actually found.
///
/// The placement is an ordinal over the devices the worker's own probe inventoried — it is NOT a
/// statement that the adapter is discrete. Mapping it unconditionally onto
/// `WgpuDevice::DiscreteGpu(i)` (which selects the i-th DISCRETE adapter) is correct only on a box
/// whose accelerator happens to be discrete; on a unified/integrated one — every Apple Silicon
/// seat — it names no adapter at all, the bring-up fails, and the failure only becomes visible as
/// the backend router's poisoned lock at the first tensor op.
///
/// Placement `0` is the device [`probe_wgpu`] measured and registered under the selected graphics
/// API, so it resolves to `DefaultDevice` — the same adapter whose limits the admitted tuple
/// records. A higher ordinal is an explicit operator placement (`DAEMON_VHC_GPU_INDEX`) and is
/// resolved within the probed adapter's class, then registered eagerly so an unbringable placement
/// refuses HERE rather than deep inside the first device op.
///
/// # Errors
/// A `BackendUnavailable`-vocabulary reason when no adapter is present, when the probed adapter
/// class cannot carry an ordinal placement, or when the placed device fails to register.
#[cfg(feature = "wgpu")]
pub fn wgpu_device_for_placement(
    placement: Option<u32>,
) -> Result<burn::backend::wgpu::WgpuDevice, String> {
    use burn::backend::wgpu::WgpuDevice;
    let Some(index) = placement else {
        return Ok(WgpuDevice::DefaultDevice);
    };
    let probe = probe_wgpu().ok_or_else(|| {
        "BackendUnavailable: the wgpu lane was selected but no adapter can be brought up on this \
         host"
            .to_string()
    })?;
    if index == 0 {
        // The measured device itself — already up under the selected graphics API.
        return Ok(WgpuDevice::DefaultDevice);
    }
    let device = match probe.device_type.as_str() {
        "DiscreteGpu" => WgpuDevice::DiscreteGpu(index as usize),
        "IntegratedGpu" => WgpuDevice::IntegratedGpu(index as usize),
        "VirtualGpu" => WgpuDevice::VirtualGpu(index as usize),
        other => {
            return Err(format!(
                "BackendUnavailable: device placement {index} cannot be resolved on an adapter the \
                 probe classifies as `{other}` (only the discrete/integrated/virtual classes carry \
                 an ordinal; placement 0 is the probed device)"
            ))
        }
    };
    ensure_wgpu_registered(&device).map_err(|e| {
        format!(
            "BackendUnavailable: device placement {index} ({device:?}) failed to register on the \
             `{}` adapter class: {e}",
            probe.device_type
        )
    })?;
    Ok(device)
}

/// Linux `MemTotal` (kB) → MiB. `0` when `/proc/meminfo` is unreadable / unparseable.
// Declared raw-fs site (Phase 4 hardening / clippy.toml): a fixed `/proc` pseudo-file, never an
// attacker- or guest-influenced path, and this crate links no `ContainedRoot` (the choke point is
// for contained trees, not for procfs stat reads).
#[allow(clippy::disallowed_methods)]
#[cfg(target_os = "linux")]
fn linux_mem_total_mb() -> u64 {
    let Ok(text) = std::fs::read_to_string("/proc/meminfo") else {
        return 0;
    };
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
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

// -------------------------------------------------------------------------------------
// Unix free-disk FFI (`statvfs`) — Linux + macOS. `libc` is a workspace dep already in the
// lock (lock-neutral edge); the raw call is the same scoped-`unsafe` pattern the CUDA /
// Windows / macOS probes use under the crate's `#![deny(unsafe_code)]`.
// -------------------------------------------------------------------------------------
#[cfg(unix)]
#[allow(unsafe_code)]
mod unix_ffi {
    use super::MIB;
    use std::os::unix::ffi::OsStrExt;
    use std::path::Path;

    /// Free space (available to a non-privileged process) in MiB on `path`'s filesystem, via
    /// `statvfs`. `0` on any error (e.g. a non-existent path).
    pub(super) fn disk_free_mb(path: &Path) -> u64 {
        let mut cpath: Vec<u8> = path.as_os_str().as_bytes().to_vec();
        cpath.push(0);
        // SAFETY: `cpath` is a NUL-terminated C string; `stat` is a valid, zero-initialized
        // `statvfs` out-pointer of the exact type the libc binding expects.
        let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
        let rc = unsafe { libc::statvfs(cpath.as_ptr().cast::<libc::c_char>(), &mut stat) };
        if rc != 0 {
            return 0;
        }
        let frsize = stat.f_frsize as u64;
        let avail = stat.f_bavail as u64;
        frsize.saturating_mul(avail) / MIB
    }
}

/// A honest snapshot of what a wgpu adapter exposes for resource planning (feature `wgpu`). Total
/// VRAM is **not** wgpu-queryable (see the module docs), so [`Self::vram_mb`] reports the adapter's
/// `max_buffer_size` as a documented lower-bound proxy; [`Self::max_alloc_mb`] is the same limit
/// used as the hard per-allocation ceiling. The GPU-governor policy is the authoritative VRAM
/// budget for eligibility.
#[cfg(feature = "wgpu")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WgpuProbe {
    /// Usable adapters found (1 when a default-device adapter initializes; wgpu adapter enumeration
    /// across all devices needs a direct `wgpu` dep = a frozen-root change, so this is "≥1 usable").
    pub gpus: u32,
    /// `max_buffer_size` in MiB — the largest single allocation (the one device-honest number),
    /// also used as the per-buffer ceiling.
    pub max_alloc_mb: u64,
    /// The adapter name (`get_info().name`, e.g. "AMD Radeon … (RADV …)").
    pub adapter: String,
    /// The graphics backend (`get_info().backend`, e.g. "Vulkan").
    pub backend: String,
    /// The adapter device type (`get_info().device_type` — e.g. "IntegratedGpu", "DiscreteGpu",
    /// "Cpu"). Debug-formatted so no direct `wgpu`-type dependency is needed.
    pub device_type: String,
    /// Whether this is a unified-memory device (`device_type` is `IntegratedGpu` or `Cpu`): the GPU
    /// shares host DRAM, so the budget math uses a joint memory pool (see [`DeviceLimits`]).
    pub unified: bool,
    /// Whether this adapter is a **software rasterizer**.
    ///
    /// The framework exposes no explicit flag the way DXGI does, so this is a determination rather
    /// than a read: a `Cpu` device type is one definitionally, and the known software
    /// implementations are matched by name. It matters because a device-lane role admitted against a
    /// rasterizer that reports a device backend class is the silent CPU fallback the platform rules
    /// forbid — and the loader on at least one fleet box enumerates one beside the real adapter, so
    /// it is reachable without any operator intending it.
    pub is_software: bool,
    /// The adapter's PCI vendor id (`get_info().vendor`), `None` where the platform supplies none.
    pub vendor_id: Option<u32>,
    /// The adapter's PCI device id (`get_info().device`).
    pub device_id: Option<u32>,
    /// The driver name (`get_info().driver`, e.g. `"radv"`). Empty where the platform supplies none —
    /// which is why the caller converts an empty string to a typed unavailability rather than
    /// recording it as a driver whose name is the empty string.
    pub driver: String,
    /// The driver's own version text (`get_info().driver_info`, e.g. `"Mesa 25.2.6"`). This is the
    /// numbering a profile's permitted revision range constrains.
    pub driver_info: String,
}

/// Names the known software rasterizers carry. Matched case-insensitively as substrings, because a
/// platform decorates them differently (`"llvmpipe (LLVM 18.1.0, 256 bits)"`,
/// `"Microsoft Basic Render Driver"`).
const SOFTWARE_RASTERIZER_NAMES: &[&str] = &[
    "llvmpipe",
    "lavapipe",
    "swiftshader",
    "warp",
    "basic render driver",
    "software rasterizer",
];

/// Whether an adapter of this name and device type is a software rasterizer.
///
/// Pure, so the determination is testable without a device: the fleet's reachable case is a loader
/// that enumerates `llvmpipe` beside real hardware and reports it under a device backend class.
#[must_use]
pub fn is_software_adapter(adapter_name: &str, device_type: &str) -> bool {
    if device_type == "Cpu" {
        return true;
    }
    let lowered = adapter_name.to_ascii_lowercase();
    SOFTWARE_RASTERIZER_NAMES
        .iter()
        .any(|name| lowered.contains(name))
}

/// Why no device could be probed. A **typed** answer, replacing the inference a caller had to make
/// from a zero reading.
///
/// A zero resource reading is an admission refusal wearing a measurement's clothes: it refuses the
/// machine rather than reporting the defect. Worse, in the case that produced this type, it did
/// something actively misleading — a GPU-capable box whose graphics loader was absent from the
/// process environment reported `gpus: 0` and was silently reclassified as CPU-only, which a run
/// would then admit as a CPU participant. The environment was the fault, and nothing said so.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DeviceProbeUnavailability {
    /// This binary carries no device lane at all — the backend feature was compiled out. A `cfg`
    /// fact, not an observation about the machine, and the first thing to rule out when a box that
    /// has an accelerator reports none.
    FeatureCompiledOut,
    /// The lane is compiled in and the adapter could not be brought up. The captured reason is
    /// carried **verbatim**, never parsed for policy: it distinguishes an absent loader from absent
    /// hardware for a human reading it, and the two are not yet distinguished mechanically — that
    /// needs a loader probe, and guessing from message text would be a policy decision resting on a
    /// vendor's phrasing.
    BringUpFailed(String),
}

impl DeviceProbeUnavailability {
    /// The stable slug for evidence and logs.
    #[must_use]
    pub fn slug(&self) -> &'static str {
        match self {
            Self::FeatureCompiledOut => "DeviceLaneCompiledOut",
            Self::BringUpFailed(_) => "DeviceBringUpFailed",
        }
    }
}

impl std::fmt::Display for DeviceProbeUnavailability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FeatureCompiledOut => write!(
                f,
                "{}: this binary was built without a device backend lane, so it reports no \
                 accelerator regardless of what the machine has",
                self.slug()
            ),
            Self::BringUpFailed(reason) => write!(
                f,
                "{}: the device lane is compiled in but no adapter could be brought up — check the \
                 graphics loader is present in this process's environment before concluding the \
                 machine has no accelerator ({reason})",
                self.slug()
            ),
        }
    }
}

/// The recorded reason the last wgpu bring-up failed, for [`wgpu_unavailability`].
#[cfg(feature = "wgpu")]
fn wgpu_bring_up_failure() -> Option<String> {
    WGPU_BRING_UP_FAILURE
        .lock()
        .ok()
        .and_then(|guard| guard.clone())
}

/// Record it. Set once by the memoized probe; a later reader gets the reason the probe actually saw.
#[cfg(feature = "wgpu")]
fn record_wgpu_bring_up_failure(reason: &str) {
    if let Ok(mut guard) = WGPU_BRING_UP_FAILURE.lock() {
        *guard = Some(reason.to_string());
    }
}

#[cfg(feature = "wgpu")]
static WGPU_BRING_UP_FAILURE: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);

/// The typed reason no wgpu device is available, or `None` when one is.
///
/// Callers use this instead of inferring absence from a zero device count.
#[must_use]
pub fn wgpu_unavailability() -> Option<DeviceProbeUnavailability> {
    #[cfg(feature = "wgpu")]
    {
        if probe_wgpu().is_some() {
            return None;
        }
        Some(DeviceProbeUnavailability::BringUpFailed(
            wgpu_bring_up_failure().unwrap_or_else(|| "no reason was recorded".to_string()),
        ))
    }
    #[cfg(not(feature = "wgpu"))]
    {
        Some(DeviceProbeUnavailability::FeatureCompiledOut)
    }
}

/// Probe the default wgpu device for its adapter info + limits (feature `wgpu`). Returns `None`
/// (never panics) when no adapter can be brought up. Reads only wgpu-queryable fields (`get_info`
/// incl. `device_type`, `limits`).
///
/// **Memoized process-wide.** cubecl's `ComputeClient::init` panics with "already registered" if
/// the default device's client is already up, so `probe_wgpu` is the canonical bring-up — call it
/// (or [`crate::wgpu_adapter_available`], which delegates here) before any wgpu tensor work.
///
/// **Register-or-reuse (fragility fix).** The bring-up (`init_setup`) both requests the adapter and
/// registers the client. If a burn tensor op won the race and registered the default device first,
/// `init_setup` panics; rather than caching `None` (which would make a *present* GPU look absent
/// for the rest of the process), the probe recognizes the "already registered" panic as proof an
/// adapter exists and returns a reuse marker (`Some`) so availability stays correct. The worker's
/// assess path always probes first (its meta pass runs on the CPU engine — no prior wgpu op), so it
/// always takes the full tier-1 path and gets the real `device_type` / `max_alloc`.
#[cfg(feature = "wgpu")]
#[must_use]
pub fn probe_wgpu() -> Option<WgpuProbe> {
    use std::sync::OnceLock;
    static PROBE: OnceLock<Option<WgpuProbe>> = OnceLock::new();
    PROBE.get_or_init(probe_wgpu_uncached).clone()
}

#[cfg(feature = "wgpu")]
fn probe_wgpu_uncached() -> Option<WgpuProbe> {
    use burn::backend::wgpu::{RuntimeOptions, WgpuDevice};

    // Tier 1 — canonical bring-up: `init_setup_selected` requests the adapter AND registers the
    // default client UNDER THE SELECTED GRAPHICS API (Dx12 on Windows, else Auto — the DX12 fix),
    // instead of cubecl's `AutoGraphicsApi` which hardcodes Vulkan off macOS. Full adapter info
    // incl. `device_type` (→ `unified`).
    //
    // Captured `quiet` (an absent adapter is an expected probe outcome, not an incident) through
    // the RECORDING capture rather than a blanked global hook: the old idiom silenced every
    // thread's panic report for the duration of the probe, which is how a device runner thread's
    // primary failure went unreported on a fleet box.
    let attempt = crate::device_panic::catch(true, || {
        let setup = init_setup_selected(&WgpuDevice::DefaultDevice, RuntimeOptions::default());
        let info = setup.adapter.get_info();
        let limits = setup.adapter.limits();
        let max_alloc_mb = (limits.max_buffer_size / MIB).max(1);
        let device_type = format!("{:?}", info.device_type);
        let unified = matches!(device_type.as_str(), "IntegratedGpu" | "Cpu");
        WgpuProbe {
            gpus: 1,
            max_alloc_mb,
            is_software: is_software_adapter(&info.name, &device_type),
            vendor_id: Some(info.vendor),
            device_id: Some(info.device),
            driver: info.driver.clone(),
            driver_info: info.driver_info.clone(),
            adapter: info.name.clone(),
            backend: format!("{:?}", info.backend),
            device_type,
            unified,
        }
    });

    match attempt {
        Ok(probe) => Some(probe),
        Err(msg) => {
            // The reason is recorded so `wgpu_unavailability` can report it verbatim rather than
            // leaving a caller to infer absence from a zero.
            record_wgpu_bring_up_failure(&msg);
            // Tier 2 — register-or-reuse: an "already registered" panic means an adapter is up
            // (a burn op registered the default client before this probe). Report availability
            // rather than caching `None`; the per-buffer limit / device_type are unknown via reuse
            // (the worker path never hits this — it probes before any wgpu op).
            if msg.contains("already registered") {
                Some(WgpuProbe {
                    gpus: 1,
                    max_alloc_mb: 0,
                    adapter: "reused (default client already registered)".to_string(),
                    backend: "wgpu".to_string(),
                    device_type: "Unknown".to_string(),
                    unified: false,
                    // Nothing was observed through a reuse, so nothing is claimed. An adapter whose
                    // class is unknown is conservatively a rasterizer here: the alternative is
                    // asserting real hardware on the strength of a race we lost.
                    is_software: true,
                    vendor_id: None,
                    device_id: None,
                    driver: String::new(),
                    driver_info: String::new(),
                })
            } else {
                None
            }
        }
    }
}

// =====================================================================================
// CUDA device-memory probe (the CUDA backend lane). Unlike wgpu, the CUDA driver exposes
// total device memory (`cuDeviceTotalMem`), so `vram_mb` is the real dedicated VRAM (24564 MiB on the
// RunPod 4090) — a discrete-device honest number (no UMA on this card). The pure mapper is
// unconditional (fixture-tested); the driver query is `#[cfg(feature = "cuda")]` (dlopen'd libcuda).
// =====================================================================================

/// Map probed CUDA device numbers to [`DeviceLimits`] — a **discrete** NVIDIA GPU: dedicated
/// `vram_mb` (from `cuDeviceTotalMem`), **no** shared spill pool, **no** UMA. Unconditional + pure so
/// it is fixture-tested on every platform; the `catch_unwind`-wrapped driver query
/// ([`probe_cuda`]) is feature-gated.
///
/// `shared_mb = 0` / `unified = false` put the budget math on the discrete path (dedicated
/// VRAM is the whole GPU budget; the joint-pool UMA math never applies), matching the archived
/// hardware-validation probe matrix for the 4090 (24 GB discrete).
#[must_use]
pub fn cuda_device_limits(vram_mb: u64, max_alloc_mb: u64, ram_mb: u64) -> DeviceLimits {
    DeviceLimits {
        vram_mb,
        ram_mb,
        max_alloc_mb,
        shared_mb: 0,
        unified: false,
    }
}

/// A honest snapshot of what a CUDA device exposes for resource planning (feature `cuda`).
///
/// The CUDA driver DOES expose total VRAM (unlike wgpu), so [`Self::vram_mb`] is the real dedicated
/// memory. CUDA has no wgpu-style per-buffer ceiling (a single `cudaMalloc` may span most of VRAM), so
/// [`Self::max_alloc_mb`] reports total VRAM as an honest upper bound (the per-buffer gate
/// then only rejects a single tensor larger than the whole card). Discrete NVIDIA ⇒ `unified = false`.
#[cfg(feature = "cuda")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CudaProbe {
    /// Usable devices found (`1` when device 0 initializes; multi-device enumeration is not needed
    /// for the single-host lane).
    pub gpus: u32,
    /// Total dedicated VRAM in MiB (`cuDeviceTotalMem`) — the true device budget (24564 on the 4090).
    pub vram_mb: u64,
    /// Largest single allocation in MiB — reported as total VRAM (no CUDA per-buffer ceiling).
    pub max_alloc_mb: u64,
    /// The device name (`cuDeviceGetName`, e.g. "NVIDIA GeForce RTX 4090").
    pub adapter: String,
    /// Discrete NVIDIA GPUs do not share host DRAM: always `false` (the discrete budget path).
    pub unified: bool,
}

/// Probe CUDA device 0 for its total VRAM + name (feature `cuda`). Returns `None` (never panics) when
/// no device / driver is present — the GPU-skip convention (mirrors [`probe_wgpu`]).
///
/// **Memoized process-wide.** The query is `cuInit` + `cuDeviceGet` + `cuDeviceTotalMem` +
/// `cuDeviceGetName` via `cudarc` (which dlopens libcuda under nix glibc — proven on the 4090
/// container during the archived hardware validation). Wrapped in `catch_unwind` so a
/// missing-libcuda dlopen panic (e.g.
/// on a CUDA-less host running the feature build) reports "no device" instead of aborting the process.
#[cfg(feature = "cuda")]
#[must_use]
pub fn probe_cuda() -> Option<CudaProbe> {
    use std::sync::OnceLock;
    static PROBE: OnceLock<Option<CudaProbe>> = OnceLock::new();
    PROBE.get_or_init(probe_cuda_uncached).clone()
}

#[cfg(feature = "cuda")]
fn probe_cuda_uncached() -> Option<CudaProbe> {
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let attempt = std::panic::catch_unwind(cuda_ffi::query_device0);
    std::panic::set_hook(prev);
    attempt.ok().flatten()
}

/// Whether the NVRTC runtime library is loadable (feature `cuda`) — the **fetch-on-demand readiness
/// gate** (an archived CUDA-lane decision). A CUDA *device* being present ([`probe_cuda`]) is necessary but not
/// sufficient for the CUDA engine arm: burn-cuda JIT-compiles kernels through NVRTC, which most
/// containers do not ship and which must be **driver-matched** (an nvrtc newer than the driver's CUDA
/// level emits PTX the driver rejects — `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`, an archived
/// container-validation finding). The
/// operator (or the future fetch-on-demand stager) provides it via `DAEMON_CUDA_RUNTIME_DIR` on
/// `LD_LIBRARY_PATH`; cudarc dlopens `libnvrtc.so.12` by soname on first use.
///
/// This check has **two legs**, both required (a live-attach smoke finding from the CUDA-lane
/// validation):
///
/// 1. **`libnvrtc` loadability** — compile-and-free a trivial NVRTC program inside `catch_unwind`:
///    in cudarc's dlopen mode a missing `libnvrtc` surfaces as a panic on first symbol resolution,
///    which is caught and reported as **not ready**.
/// 2. **cudart JIT headers** — cubecl-cuda resolves `#include <cuda_runtime.h>` for every kernel it
///    JITs via its `cuda_path()` rule (`$CUDA_PATH`, else `/usr/local/cuda`, `/opt/cuda`, or `/usr`
///    when `/usr/bin/nvcc` exists) and **panics at kernel-compile time** when none resolves. The lib
///    being loadable is NOT sufficient — a worker that passed leg 1 alone would select CUDA and then
///    panic-spam on the first tensor op (observed live on the 4090 when spawned without `CUDA_PATH`).
///    So readiness also requires `<cuda_path>/include/cuda_runtime.h` to exist (the same search rule,
///    checked here without panicking).
///
/// The worker's fat-binary probe order downgrades to wgpu/CPU when either leg fails. Memoized
/// process-wide (dlopen'd libraries + the env are process-start state).
#[cfg(feature = "cuda")]
#[must_use]
pub fn cuda_nvrtc_ready() -> bool {
    use std::sync::OnceLock;
    static READY: OnceLock<bool> = OnceLock::new();
    *READY.get_or_init(|| {
        if !cuda_jit_headers_present() {
            return false;
        }
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let attempt = std::panic::catch_unwind(cuda_ffi::nvrtc_loads);
        std::panic::set_hook(prev);
        matches!(attempt, Ok(true))
    })
}

/// Whether cubecl-cuda's JIT include path resolves to a real `cuda_runtime.h` — leg 2 of
/// [`cuda_nvrtc_ready`]. Mirrors cubecl's `cuda_path()` search order exactly (`$CUDA_PATH`, then
/// `/usr/local/cuda`, `/opt/cuda`, `/usr` iff `/usr/bin/nvcc`) but returns `false` instead of
/// panicking, and additionally requires the header itself (an empty/incomplete dir is not ready).
/// The fetch-on-demand stager satisfies this by shipping the cudart `include/` inside the runtime
/// dir and the launcher exporting `CUDA_PATH=$DAEMON_CUDA_RUNTIME_DIR` (the staged
/// `/root/cuda-rt-124` already carries `include/cuda_runtime.h`).
#[cfg(feature = "cuda")]
fn cuda_jit_headers_present() -> bool {
    let base = if let Ok(p) = std::env::var("CUDA_PATH") {
        Some(std::path::PathBuf::from(p))
    } else if std::path::Path::new("/usr/local/cuda").exists() {
        Some(std::path::PathBuf::from("/usr/local/cuda"))
    } else if std::path::Path::new("/opt/cuda").exists() {
        Some(std::path::PathBuf::from("/opt/cuda"))
    } else if std::path::Path::new("/usr/bin/nvcc").exists() {
        Some(std::path::PathBuf::from("/usr"))
    } else {
        None
    };
    base.is_some_and(|b| b.join("include").join("cuda_runtime.h").exists())
}

// The one cudarc-touching module. `cuDeviceTotalMem` / `destroy_program` are `cudarc` `unsafe fn`s,
// so this module carries the scoped `#[allow(unsafe_code)]` under the crate's `#![deny(unsafe_code)]`
// — the identical pattern the Windows/macOS FFI probes use. cudarc is a
// cuda-gated, lock-neutral dep (already resolved via cubecl-cuda).
#[cfg(feature = "cuda")]
#[allow(unsafe_code)]
mod cuda_ffi {
    use super::{CudaProbe, MIB};

    /// Query device 0's total VRAM + name via the CUDA driver API. `None` on any driver error.
    pub(super) fn query_device0() -> Option<CudaProbe> {
        use cudarc::driver::result::{device, init};
        init().ok()?;
        let dev = device::get(0).ok()?;
        // SAFETY: `dev` is a `CUdevice` handle just returned by `device::get(0)`, the exact contract
        // `total_mem`'s safety comment requires.
        let total_bytes = unsafe { device::total_mem(dev) }.ok()?;
        let vram_mb = (total_bytes as u64) / MIB;
        if vram_mb == 0 {
            return None;
        }
        let adapter = device::get_name(dev).unwrap_or_else(|_| "NVIDIA CUDA device".to_string());
        Some(CudaProbe {
            gpus: 1,
            vram_mb,
            max_alloc_mb: vram_mb,
            adapter,
            unified: false,
        })
    }

    /// Create (and free) a trivial NVRTC program — proves `libnvrtc` dlopens and its symbols
    /// resolve. Panics (caught by the caller) when the library is absent; `false` on a soft error.
    pub(super) fn nvrtc_loads() -> bool {
        use cudarc::nvrtc::result::{create_program, destroy_program};
        match create_program(
            c"extern \"C\" __global__ void daemon_nvrtc_probe() {}",
            None,
        ) {
            Ok(prog) => {
                // SAFETY: `prog` was just created by `create_program` and not yet destroyed.
                let _ = unsafe { destroy_program(prog) };
                true
            }
            Err(_) => false,
        }
    }
}

// =====================================================================================
// Windows DXGI/D3D12 device-memory probe (swarm-windows-vram-design.md §2 mapping).
// The pure mapper + raw struct are unconditional (fixture-tested on every platform); the
// actual DXGI/D3D12 FFI is `#[cfg(windows)]` + target-gated `windows` dep.
// =====================================================================================

/// Static + live memory numbers for one DXGI adapter, gathered by the Windows FFI (or a fixture).
/// All byte counts as reported by the OS; the pure [`windows_device_limits`] mapper turns these
/// into [`DeviceLimits`] per the design's §2 field mapping and its trap rules.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DxgiAdapterMemory {
    /// `GetDesc3().DedicatedVideoMemory` — physical VRAM on discrete (correct > 4 GiB); on an APU's
    /// "Variable Graphics Memory" this is the *configured allocation*, not physical RAM.
    pub dedicated_video: u64,
    /// `GetDesc3().DedicatedSystemMemory` — BIOS carve-out some iGPUs reserve (usually 0). Carried
    /// as a telemetry note; **never folded into `vram_mb`**.
    pub dedicated_system: u64,
    /// `GetDesc3().SharedSystemMemory` — the static **ceiling** on borrowable system RAM (~½ RAM),
    /// a limit, NOT usage.
    pub shared_system: u64,
    /// `DXGI_ADAPTER_FLAG3_SOFTWARE` — the WARP software rasterizer; skip during enumeration.
    pub is_software: bool,
    /// `D3D12_FEATURE_DATA_ARCHITECTURE1.UMA` — authoritative unified flag (queried, not inferred).
    pub uma: bool,
    /// `D3D12_FEATURE_DATA_ARCHITECTURE1.CacheCoherentUMA` — coherent cache hierarchy (telemetry).
    pub cache_coherent_uma: bool,
    /// `QueryVideoMemoryInfo(node 0, LOCAL).Budget` — the live OS-granted budget for the LOCAL
    /// segment group (on UMA this is the shared-pool grant; on discrete ≈ 0.9 × VRAM, the number
    /// Task Manager's GPU tab shows). Same WDDM source as Task Manager → trivial cross-check.
    pub budget_local: u64,
    /// `QueryVideoMemoryInfo(node 0, NON_LOCAL).Budget` — the live NON_LOCAL budget (≈ ½ RAM on
    /// discrete; ≈ 0 on UMA). Recorded for telemetry; contributes **0** to the discrete GPU budget.
    pub budget_non_local: u64,
}

/// Map one non-WARP DXGI adapter's memory numbers to [`DeviceLimits`] (design §2).
///
/// - `unified` ← `ARCHITECTURE1.UMA` (authoritative; replaces-and-validates the wgpu heuristic).
/// - `vram_mb` ← `DedicatedVideoMemory` (physical VRAM / configured VGM allocation).
/// - `shared_mb`: **UMA** → `min(SharedSystemMemory, LOCAL.Budget)` (on UMA everything is LOCAL, so
///   the live LOCAL budget is the shared-pool grant, statically capped by `SharedSystemMemory`);
///   **discrete** → **0** (NON_LOCAL spill is PCIe-speed and contributes 0 to the effective GPU
///   budget by default, per the program's discrete-spill rule — the NON_LOCAL budget is recorded on
///   [`DxgiAdapterMemory`] for telemetry, not fed to the budget math).
/// - `max_alloc_mb` ← wgpu `max_buffer_size` (passed in; the DX12 `i32::MAX` constant when wgpu is
///   absent) — the per-tensor gate, unchanged.
/// - `ram_mb` ← `GlobalMemoryStatusEx().ullTotalPhys` (passed in).
///
/// Returns `None` for a WARP / software adapter (the caller skips it during enumeration).
///
/// **VGM safety:** on Variable-Graphics-Memory APUs `dedicated_video` can present tens of GB of
/// unified RAM; because the unified-path budget math clamps the joint pool to
/// `min(vram + 90%·shared, ram)`, the physical-RAM ceiling caps the inflated VRAM figure — the
/// design's "never conflate configured allocation with physical RAM" rule holds, and `ram_mb` is
/// the true physical bound.
#[must_use]
pub fn windows_device_limits(
    adapter: &DxgiAdapterMemory,
    ram_mb: u64,
    max_alloc_mb: u64,
) -> Option<DeviceLimits> {
    if adapter.is_software {
        return None; // WARP / software rasterizer — skip (trap rule).
    }
    let shared_mb = if adapter.uma {
        // UMA: the LOCAL budget is the live shared-pool grant; SharedSystemMemory caps it statically.
        adapter.shared_system.min(adapter.budget_local) / MIB
    } else {
        // Discrete: NON_LOCAL spill contributes 0 to the effective GPU budget by default.
        0
    };
    Some(DeviceLimits {
        vram_mb: adapter.dedicated_video / MIB,
        ram_mb,
        max_alloc_mb,
        shared_mb,
        unified: adapter.uma,
    })
}

/// The DX12 per-buffer ceiling wgpu reports (`max_buffer_size`) when the probe has no live wgpu
/// adapter: `i32::MAX` bytes ("Dx12 does not expose a maximum buffer size in the API",
/// `wgpu-hal dx12/adapter.rs:891-894`). In MiB (2047) — a wgpu-enforced per-tensor gate, never a
/// capacity number.
pub const DX12_MAX_BUFFER_MB: u64 = (i32::MAX as u64) / MIB;

// =====================================================================================
// macOS Metal device-budget probe (swarm-macos-uma-findings.md §4 mapping).
// =====================================================================================

/// Metal device scalars gathered by the macOS FFI (or a fixture). All byte counts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MetalAdapterMemory {
    /// `MTLDevice.recommendedMaxWorkingSetSize` — the "allocate up to this" GPU budget (≈ ⅔ RAM on
    /// Apple Silicon); the working-set analogue of the three-platform budget symmetry.
    pub recommended_working_set: u64,
    /// `MTLDevice.maxBufferLength` — the per-allocation ceiling (≈ ½ RAM); honest on Metal (wgpu's
    /// `max_buffer_size` agrees exactly, so this doubles as `max_alloc_mb`).
    pub max_buffer_length: u64,
    /// `sysctl hw.memsize` (== `ProcessInfo.physicalMemory`) — full physical RAM.
    pub phys_ram: u64,
    /// `MTLDevice.hasUnifiedMemory` — Apple Silicon is always true.
    pub has_unified: bool,
}

/// Map Metal device scalars to [`DeviceLimits`] (findings §4): `vram_mb` = the working-set budget
/// (NOT 0, NOT `max_buffer_size`), `shared_mb` = `ram_mb` (the unified physical pool that drives the
/// joint check), `max_alloc_mb` = `maxBufferLength`, `unified` = `hasUnifiedMemory`.
#[must_use]
pub fn macos_device_limits(metal: &MetalAdapterMemory) -> DeviceLimits {
    let ram_mb = metal.phys_ram / MIB;
    DeviceLimits {
        vram_mb: metal.recommended_working_set / MIB,
        ram_mb,
        max_alloc_mb: metal.max_buffer_length / MIB,
        // The unified physical pool CPU+GPU jointly draw from; drives the joint-pool check so
        // `fixed_vram + host_ram` is validated against one pool. On Apple Silicon = physical RAM.
        shared_mb: if metal.has_unified { ram_mb } else { 0 },
        unified: metal.has_unified,
    }
}

// -------------------------------------------------------------------------------------
// Windows FFI (DXGI/D3D12). Compiled only for the Windows target; the `windows` crate is a
// target-gated dep. All decision logic lives in `windows_device_limits` above (fixture-tested
// everywhere); this module only gathers raw scalars.
// -------------------------------------------------------------------------------------
#[cfg(windows)]
#[allow(unsafe_code)]
mod win_ffi {
    use super::{DeviceLimits, DxgiAdapterMemory, DX12_MAX_BUFFER_MB, MIB};
    use windows::Win32::Foundation::HANDLE;
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
    use windows::Win32::Graphics::Direct3D12::{
        D3D12CreateDevice, ID3D12Device, D3D12_FEATURE_ARCHITECTURE1,
        D3D12_FEATURE_DATA_ARCHITECTURE1,
    };
    use windows::Win32::Graphics::Dxgi::{
        CreateDXGIFactory2, IDXGIAdapter4, IDXGIFactory6, DXGI_ADAPTER_FLAG3_SOFTWARE,
        DXGI_CREATE_FACTORY_FLAGS, DXGI_GPU_PREFERENCE_UNSPECIFIED,
        DXGI_MEMORY_SEGMENT_GROUP_LOCAL, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL,
        DXGI_QUERY_VIDEO_MEMORY_INFO,
    };
    use windows::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};

    /// Physical RAM in MiB from `GlobalMemoryStatusEx().ullTotalPhys`; `0` on failure.
    pub(super) fn ram_mb() -> u64 {
        let mut status = MEMORYSTATUSEX {
            dwLength: size_of::<MEMORYSTATUSEX>() as u32,
            ..Default::default()
        };
        // SAFETY: `status` is a valid, `dwLength`-initialized MEMORYSTATUSEX out-pointer.
        match unsafe { GlobalMemoryStatusEx(&mut status) } {
            Ok(()) => status.ullTotalPhys / MIB,
            Err(_) => 0,
        }
    }

    /// Free disk space (available to the caller) in MiB on `path`'s volume, from
    /// `GetDiskFreeSpaceExW`. `0` on any error.
    pub(super) fn disk_free_mb(path: &std::path::Path) -> u64 {
        use windows::core::HSTRING;
        use windows::Win32::Storage::FileSystem::GetDiskFreeSpaceExW;
        let dir = HSTRING::from(path.as_os_str());
        let mut free_to_caller: u64 = 0;
        // SAFETY: `dir` is a valid wide C string; `free_to_caller` is a valid u64 out-pointer;
        // the two unused out-params are `None`.
        match unsafe { GetDiskFreeSpaceExW(&dir, Some(&mut free_to_caller), None, None) } {
            Ok(()) => free_to_caller / MIB,
            Err(_) => 0,
        }
    }

    /// Probe the first non-WARP DXGI adapter → [`DeviceLimits`], plus the raw numbers for logging.
    /// Returns `None` when no usable adapter is found (or DXGI is unavailable).
    pub(super) fn probe() -> Option<(DeviceLimits, DxgiAdapterMemory)> {
        let ram = ram_mb();
        // SAFETY: DXGI factory/adapter/device calls with correctly-typed out-pointers; every result
        // is checked. COM objects are dropped at scope end (windows-crate RAII).
        unsafe {
            let factory: IDXGIFactory6 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
            let mut i = 0u32;
            loop {
                let adapter: IDXGIAdapter4 =
                    match factory.EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_UNSPECIFIED) {
                        Ok(a) => a,
                        Err(_) => return None, // exhausted enumeration with no usable adapter
                    };
                i += 1;

                let desc = match adapter.GetDesc3() {
                    Ok(d) => d,
                    Err(_) => continue,
                };
                let is_software = (desc.Flags.0 & DXGI_ADAPTER_FLAG3_SOFTWARE.0) != 0;
                if is_software {
                    continue; // skip WARP (trap rule)
                }

                // UMA is queried via a D3D12 device (authoritative), not inferred.
                let mut device: Option<ID3D12Device> = None;
                if D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut device).is_err() {
                    continue;
                }
                let Some(device) = device else { continue };
                let mut arch = D3D12_FEATURE_DATA_ARCHITECTURE1::default();
                let _ = device.CheckFeatureSupport(
                    D3D12_FEATURE_ARCHITECTURE1,
                    (&mut arch as *mut D3D12_FEATURE_DATA_ARCHITECTURE1).cast(),
                    size_of::<D3D12_FEATURE_DATA_ARCHITECTURE1>() as u32,
                );

                let mut local = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
                let mut non_local = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
                let _ =
                    adapter.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut local);
                let _ = adapter.QueryVideoMemoryInfo(
                    0,
                    DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL,
                    &mut non_local,
                );

                let raw = DxgiAdapterMemory {
                    dedicated_video: desc.DedicatedVideoMemory as u64,
                    dedicated_system: desc.DedicatedSystemMemory as u64,
                    shared_system: desc.SharedSystemMemory as u64,
                    is_software,
                    uma: arch.UMA.as_bool(),
                    cache_coherent_uma: arch.CacheCoherentUMA.as_bool(),
                    budget_local: local.Budget,
                    budget_non_local: non_local.Budget,
                };
                // wgpu's DX12 `max_buffer_size` is the fixed i32::MAX constant; use it directly (no
                // live wgpu adapter is needed for the probe-only cross build).
                let limits = super::windows_device_limits(&raw, ram, DX12_MAX_BUFFER_MB)?;
                let _ = HANDLE::default(); // (budget-change event handle wiring is §3, not probe-time)
                return Some((limits, raw));
            }
        }
    }
}

/// Probe Windows GPU memory via DXGI/D3D12 → [`DeviceLimits`] (design §2). `None` off Windows or
/// when no usable (non-WARP) adapter is found. Safe wrapper over the `#[cfg(windows)]` FFI.
#[must_use]
pub fn probe_windows_device_limits() -> Option<DeviceLimits> {
    #[cfg(windows)]
    {
        win_ffi::probe().map(|(limits, raw)| {
            eprintln!(
                "daemon-vhc-host probe (windows/DXGI): {raw:?} dedicated_system_mb={} \
                 budget_local_mb={} budget_non_local_mb={} -> {limits:?}",
                raw.dedicated_system / MIB,
                raw.budget_local / MIB,
                raw.budget_non_local / MIB,
            );
            limits
        })
    }
    #[cfg(not(windows))]
    {
        None
    }
}

// -------------------------------------------------------------------------------------
// macOS FFI (Metal + libSystem sysctl). Compiled only for macOS. No new dependency — raw `extern`
// FFI to the Objective-C runtime + Metal/Foundation frameworks + libSystem `sysctlbyname`. All
// mapping lives in `macos_device_limits` above (fixture-tested everywhere).
// -------------------------------------------------------------------------------------
#[cfg(target_os = "macos")]
#[allow(unsafe_code)]
mod mac_ffi {
    use super::{macos_device_limits, DeviceLimits, MetalAdapterMemory};
    use core::ffi::{c_char, c_void};

    #[link(name = "Metal", kind = "framework")]
    unsafe extern "C" {
        fn MTLCreateSystemDefaultDevice() -> *mut c_void;
    }
    unsafe extern "C" {
        fn sel_registerName(name: *const c_char) -> *mut c_void;
        fn objc_msgSend();
        fn sysctlbyname(
            name: *const c_char,
            oldp: *mut c_void,
            oldlenp: *mut usize,
            newp: *mut c_void,
            newlen: usize,
        ) -> i32;
    }

    /// `[obj selector]` returning an unsigned integer (NSUInteger / u64).
    unsafe fn msg_u64(obj: *mut c_void, sel: *mut c_void) -> u64 {
        let f: unsafe extern "C" fn(*mut c_void, *mut c_void) -> u64 =
            unsafe { core::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
        unsafe { f(obj, sel) }
    }

    /// `[obj selector]` returning an Objective-C `BOOL` (a signed char on arm64 macOS).
    unsafe fn msg_bool(obj: *mut c_void, sel: *mut c_void) -> bool {
        let f: unsafe extern "C" fn(*mut c_void, *mut c_void) -> bool =
            unsafe { core::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
        unsafe { f(obj, sel) }
    }

    unsafe fn sel(name: &core::ffi::CStr) -> *mut c_void {
        unsafe { sel_registerName(name.as_ptr()) }
    }

    /// Physical RAM in MiB from `sysctl hw.memsize`; `0` on failure.
    pub(super) fn ram_mb() -> u64 {
        sysctl_u64(c"hw.memsize") / super::MIB
    }

    fn sysctl_u64(name: &core::ffi::CStr) -> u64 {
        let mut val: u64 = 0;
        let mut len = size_of::<u64>();
        // SAFETY: `val`/`len` are valid out-pointers sized for a u64 sysctl scalar.
        let rc = unsafe {
            sysctlbyname(
                name.as_ptr(),
                (&mut val as *mut u64).cast(),
                &mut len,
                core::ptr::null_mut(),
                0,
            )
        };
        if rc == 0 {
            val
        } else {
            0
        }
    }

    pub(super) fn probe() -> Option<DeviceLimits> {
        // SAFETY: MTLCreateSystemDefaultDevice returns a valid MTLDevice (or null → bail); the
        // selectors are no-argument accessors returning scalar NSUInteger/BOOL, called via a
        // correctly-typed objc_msgSend. The device is intentionally leaked (probe runs once).
        unsafe {
            let device = MTLCreateSystemDefaultDevice();
            if device.is_null() {
                return None;
            }
            let working_set = msg_u64(device, sel(c"recommendedMaxWorkingSetSize"));
            let max_buffer = msg_u64(device, sel(c"maxBufferLength"));
            let has_unified = msg_bool(device, sel(c"hasUnifiedMemory"));
            let metal = MetalAdapterMemory {
                recommended_working_set: working_set,
                max_buffer_length: max_buffer,
                phys_ram: sysctl_u64(c"hw.memsize"),
                has_unified,
            };
            let limits = macos_device_limits(&metal);
            eprintln!("daemon-vhc-host probe (macos/Metal): {metal:?} -> {limits:?}");
            Some(limits)
        }
    }
}

/// Probe macOS GPU budget via Metal (`recommendedMaxWorkingSetSize`/`maxBufferLength`/
/// `hasUnifiedMemory`) + `sysctl hw.memsize` → [`DeviceLimits`] (findings §4). `None` off macOS or
/// when no Metal device is available.
#[must_use]
pub fn probe_macos_device_limits() -> Option<DeviceLimits> {
    #[cfg(target_os = "macos")]
    {
        mac_ffi::probe()
    }
    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

/// Whether a usable wgpu adapter can be brought up (the **GPU-skip test convention**, TDD §8.1
/// tier-2): device-needing tests skip loudly when this is `false`, so the default CI gate stays
/// green on GPU-less runners while the `.#vulkan` devShell runs the full suite. Delegates to the
/// memoized [`probe_wgpu`] (a `catch_unwind` around cubecl's default-device setup — cubecl panics
/// when no adapter matches). The probe registers the default device's compute client exactly
/// once; subsequent tensor work reuses it.
#[cfg(feature = "wgpu")]
#[must_use]
pub fn wgpu_adapter_available() -> bool {
    probe_wgpu().is_some()
}

/// Whether a usable CUDA device can be brought up — the CUDA analogue of
/// [`wgpu_adapter_available`]. Returns `false` — never panics — when no device / driver is
/// present, so CUDA-needing tests skip cleanly on GPU-less runners. Delegates to the memoized
/// [`probe_cuda`], which wraps the `cuInit`/device query in `catch_unwind`.
#[cfg(feature = "cuda")]
#[must_use]
pub fn cuda_adapter_available() -> bool {
    probe_cuda().is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The portable host probes return real, nonzero numbers on the host OS (the cross-platform
    /// leg of the disk/RAM fix — the pre-fix worker hardcoded `disk_free_mb: 0` and read RAM only
    /// from Linux `/proc`, spuriously failing the trainer lane floor off-Linux).
    #[test]
    fn host_ram_and_disk_free_are_nonzero_on_this_host() {
        assert!(
            host_ram_mb() > 0,
            "host RAM probe must report a nonzero total"
        );
        let cwd = std::env::current_dir().expect("cwd");
        assert!(
            host_disk_free_mb(&cwd) > 0,
            "free-disk probe must report nonzero on the cwd filesystem"
        );
        // A non-existent path yields 0 (never a spurious large number).
        assert_eq!(
            host_disk_free_mb(std::path::Path::new(
                "/definitely/not/a/real/path/for/vhc/probe/test"
            )),
            0
        );
    }

    /// The DX12 selection fix: Windows defaults to Dx12 (the fleet's third GPU API), Linux/macOS
    /// keep Auto (Vulkan/Metal — unchanged), and the operator override wins on every OS.
    #[test]
    fn graphics_api_selection_defaults_and_override() {
        // Platform defaults (no override): Windows → Dx12; Linux/macOS → Auto.
        assert_eq!(
            resolve_graphics_api("windows", None),
            GraphicsApiChoice::Dx12,
            "Windows must default to Dx12 (AutoGraphicsApi would pick Vulkan there)"
        );
        assert_eq!(resolve_graphics_api("linux", None), GraphicsApiChoice::Auto);
        assert_eq!(resolve_graphics_api("macos", None), GraphicsApiChoice::Auto);
        // The override wins everywhere, case/whitespace-insensitive.
        assert_eq!(
            resolve_graphics_api("linux", Some(" DX12 ")),
            GraphicsApiChoice::Dx12
        );
        assert_eq!(
            resolve_graphics_api("windows", Some("vulkan")),
            GraphicsApiChoice::Vulkan
        );
        assert_eq!(
            resolve_graphics_api("windows", Some("auto")),
            GraphicsApiChoice::Auto
        );
        assert_eq!(
            resolve_graphics_api("macos", Some("metal")),
            GraphicsApiChoice::Metal
        );
        // An empty or unrecognized override falls back to the platform default (no silent wrong API).
        assert_eq!(
            resolve_graphics_api("windows", Some("")),
            GraphicsApiChoice::Dx12
        );
        assert_eq!(
            resolve_graphics_api("windows", Some("nonsense")),
            GraphicsApiChoice::Dx12
        );
        assert_eq!(
            resolve_graphics_api("linux", Some("nonsense")),
            GraphicsApiChoice::Auto
        );
    }

    #[test]
    fn parse_amdgpu_mem_mb_fixtures() {
        // This machine's real values: 4 GiB VRAM, 120000 MiB GTT.
        assert_eq!(parse_amdgpu_mem_mb("4294967296\n"), Some(4096));
        assert_eq!(parse_amdgpu_mem_mb("125829120000\n"), Some(120_000));
        assert_eq!(parse_amdgpu_mem_mb("  4294967296  "), Some(4096));
        // Non-numeric / empty → None (caller falls back to another source).
        assert_eq!(parse_amdgpu_mem_mb(""), None);
        assert_eq!(parse_amdgpu_mem_mb("N/A"), None);
    }

    // ---- Windows DXGI/D3D12 probe mapping (swarm-windows-vram-design.md §2) ----
    //
    // The mapper tests survive the Phase-E sunset (the probe is permanent, decisions D5); the
    // Autotune-verdict legs that used to ride them retired with the autotune admission.

    const GIB: u64 = 1024 * MIB;

    /// A discrete card (RTX-5090-shaped: 32 GiB dedicated, NON_LOCAL budget ≈ ½ RAM) maps to
    /// `vram_mb = DedicatedVideoMemory`, `shared_mb = 0` (NON_LOCAL spill contributes 0 by
    /// default), `unified = false`.
    #[test]
    fn windows_discrete_maps_dedicated_vram_shared_zero() {
        let adapter = DxgiAdapterMemory {
            dedicated_video: 32 * GIB,
            dedicated_system: 0,
            shared_system: 32 * GIB, // ≈ ½ of 64 GiB RAM
            is_software: false,
            uma: false,
            cache_coherent_uma: false,
            budget_local: 30 * GIB, // ≈ 0.9 × VRAM (what Task Manager shows)
            budget_non_local: 30 * GIB, // ≈ ½ RAM (telemetry only)
        };
        let limits =
            windows_device_limits(&adapter, 64 * 1024, DX12_MAX_BUFFER_MB).expect("not WARP");
        assert_eq!(limits.vram_mb, 32 * 1024);
        assert_eq!(limits.shared_mb, 0, "discrete NON_LOCAL contributes 0");
        assert!(!limits.unified);
        assert_eq!(limits.max_alloc_mb, DX12_MAX_BUFFER_MB); // 2047 — the DX12 constant
    }

    /// An integrated/UMA adapter (small dedicated carve-out, large LOCAL budget) maps to
    /// `unified = true`, `shared_mb = min(SharedSystemMemory, LOCAL.Budget)`.
    #[test]
    fn windows_uma_uses_local_budget() {
        let adapter = DxgiAdapterMemory {
            dedicated_video: 512 * MIB, // iGPU carve-out
            dedicated_system: 0,
            shared_system: 16 * GIB, // static ceiling (~½ of 32 GiB)
            is_software: false,
            uma: true,
            cache_coherent_uma: true,
            budget_local: 12 * GIB, // live shared-pool grant (< static ceiling)
            budget_non_local: 0,    // UMA => ~0
        };
        let limits =
            windows_device_limits(&adapter, 32 * 1024, DX12_MAX_BUFFER_MB).expect("not WARP");
        assert!(limits.unified);
        assert_eq!(limits.vram_mb, 512);
        // min(16 GiB ceiling, 12 GiB live budget) = 12 GiB.
        assert_eq!(limits.shared_mb, 12 * 1024);
    }

    /// The AMD "Variable Graphics Memory" trap — a Strix-Halo-on-Windows APU can present tens of
    /// GB as `DedicatedVideoMemory`. The mapper keeps the configured allocation in `vram_mb`
    /// (never conflated with RAM); it is never summed with `shared_mb` into a fake capacity.
    #[test]
    fn windows_variable_graphics_memory_clamped_to_ram() {
        let adapter = DxgiAdapterMemory {
            dedicated_video: 48 * GIB, // configured VGM allocation (huge)
            dedicated_system: 0,
            shared_system: 60 * GIB,
            is_software: false,
            uma: true,
            cache_coherent_uma: true,
            budget_local: 56 * GIB,
            budget_non_local: 0,
        };
        let ram_mb = 128 * 1024; // 128 GiB physical
        let limits = windows_device_limits(&adapter, ram_mb, DX12_MAX_BUFFER_MB).expect("not WARP");
        assert_eq!(limits.vram_mb, 48 * 1024, "configured allocation, not RAM");
        assert!(limits.unified);
        assert_eq!(limits.ram_mb, ram_mb);
    }

    /// The RunPod-4090 numbers map to a discrete budget — `vram_mb` = dedicated VRAM
    /// (`cuDeviceTotalMem`), `shared_mb = 0`, `unified = false` (the CUDA backend lane).
    #[test]
    fn cuda_discrete_maps_dedicated_vram_no_uma() {
        // 4090: 24564 MiB dedicated VRAM, 124 GiB host RAM (RunPod container).
        let limits = cuda_device_limits(24_564, 24_564, 124_000);
        assert_eq!(limits.vram_mb, 24_564);
        assert_eq!(limits.shared_mb, 0, "discrete: no shared spill pool");
        assert!(!limits.unified, "no UMA on the 4090");
    }

    /// A software (WARP) adapter maps to `None` so enumeration skips it.
    #[test]
    fn windows_warp_skipped() {
        let warp = DxgiAdapterMemory {
            dedicated_video: 0,
            is_software: true,
            ..Default::default()
        };
        assert!(windows_device_limits(&warp, 16 * 1024, DX12_MAX_BUFFER_MB).is_none());
    }

    // ---- macOS Metal probe mapping (swarm-macos-uma-findings.md §4) ----

    /// The measured M1-mini numbers (8 GiB) map to `vram_mb = recommendedMaxWorkingSetSize`
    /// (⅔ RAM), `max_alloc_mb = maxBufferLength` (½ RAM), `shared_mb = ram_mb`, `unified = true`.
    #[test]
    fn macos_m1_working_set_and_joint_pool() {
        let metal = MetalAdapterMemory {
            recommended_working_set: 5_726_633_984, // 5461 MiB (⅔ of 8 GiB, measured)
            max_buffer_length: 4 * GIB,             // 4096 MiB (½ of 8 GiB, measured)
            phys_ram: 8 * GIB,
            has_unified: true,
        };
        let limits = macos_device_limits(&metal);
        assert_eq!(limits.vram_mb, 5461);
        assert_eq!(limits.max_alloc_mb, 4096);
        assert_eq!(limits.ram_mb, 8192);
        assert_eq!(limits.shared_mb, 8192);
        assert!(limits.unified);
    }
}

#[cfg(test)]
mod software_adapter_tests {
    use super::{is_software_adapter, DeviceProbeUnavailability};

    /// The determination is pure, so it is testable without a device — which matters because the
    /// reachable case on a fleet box is a loader enumerating a rasterizer beside real hardware and
    /// reporting it under a device backend class.
    #[test]
    fn known_software_rasterizers_are_recognized_whatever_the_platform_calls_them() {
        for (name, device_type) in [
            ("llvmpipe (LLVM 18.1.0, 256 bits)", "Cpu"),
            ("lavapipe (LLVM 18.1.0, 256 bits)", "Other"),
            ("SwiftShader Device (Subzero)", "Other"),
            ("Microsoft Basic Render Driver", "Other"),
            ("Microsoft Direct3D12 (WARP)", "Other"),
        ] {
            assert!(
                is_software_adapter(name, device_type),
                "`{name}` is a software rasterizer"
            );
        }

        // A CPU device type is one definitionally, whatever it is called.
        assert!(is_software_adapter("Some Vendor Compute Device", "Cpu"));
    }

    /// Real hardware is not swept up by the name match — a floor that refused real adapters would be
    /// worse than the fallback it replaces.
    #[test]
    fn real_adapters_are_not_mistaken_for_rasterizers() {
        for (name, device_type) in [
            ("Radeon 8060S Graphics (RADV GFX1151)", "IntegratedGpu"),
            ("NVIDIA GeForce RTX 5090", "DiscreteGpu"),
            ("Apple M4 Pro", "IntegratedGpu"),
            ("Intel(R) Arc(tm) A770 Graphics", "DiscreteGpu"),
        ] {
            assert!(
                !is_software_adapter(name, device_type),
                "`{name}` is real hardware"
            );
        }
    }

    /// The absence of a device is reported as a typed reason, and the two reasons are distinguishable
    /// — the first thing to rule out when a box that has an accelerator reports none is that the
    /// binary was built without the lane at all.
    #[test]
    fn device_absence_carries_a_typed_reason_that_names_the_environment() {
        let compiled_out = DeviceProbeUnavailability::FeatureCompiledOut;
        assert_eq!(compiled_out.slug(), "DeviceLaneCompiledOut");
        assert!(compiled_out
            .to_string()
            .contains("without a device backend lane"));

        let bring_up = DeviceProbeUnavailability::BringUpFailed("no adapter matched".into());
        assert_eq!(bring_up.slug(), "DeviceBringUpFailed");
        let rendered = bring_up.to_string();
        assert!(
            rendered.contains("graphics loader is present in this process's environment"),
            "the message points at the environment before the hardware: {rendered}"
        );
        // The captured reason travels verbatim; it is never parsed for policy.
        assert!(rendered.contains("no adapter matched"));
        assert_ne!(compiled_out, bring_up);
    }
}
