// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The wasmtime host runtime (ABI §2): the engine profile the sandbox runs under.
//!
//! The [`Worker`] owns the wasmtime [`Engine`] (fuel on, epoch on, NaN canonicalization, pooling
//! allocator, no WASI). The major-2 event-loop driver ([`crate::v2`]) builds its stores and
//! linkers over this engine; the admission-time selection layer ([`crate::select`]) compiles +
//! inspects modules through it.
//!
//! The `tabi@1` dispatch layer that lived here is RETIRED with the transitional compute bridge:
//! a module importing `tabi@1` meets a typed `BridgeRetired` admission refusal at the §1.3 front
//! door — compute crosses the boundary exclusively through the `compute@2` world
//! ([`crate::compute`]).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use wasmtime::{Config, Engine, InstanceAllocationStrategy, Module, PoolingAllocationConfig};

use crate::TrainError;

/// The engine's device-backend selection seam.
///
/// The default is [`BackendKind::Cpu`]. The GPU arms name the device lanes the worker binary's
/// feature flags compile (`wgpu` / `cuda`); the driver constructs the matching per-instance
/// [`crate::compute::HostCompute`] runner from this seam ([`EngineConfig::backend`] +
/// [`EngineConfig::gpu_index`]). **A selected backend that is unavailable at run start is a
/// typed [`crate::run::RunError::BackendUnavailable`] refusal — never a silent CPU run** (the
/// det lane stays host fp32 on every rung, so backend choice affects only the native
/// tolerance-class lane; the refusal protects capacity, not determinism). Selection is data
/// only; nothing burn-specific leaks across the worker protocol.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum BackendKind {
    /// The CPU lane (the det lane is bit-exact everywhere by construction).
    #[default]
    Cpu,
    /// The burn-ndarray lane.
    #[cfg(feature = "burn-ndarray")]
    BurnNdarray,
    /// The burn-wgpu lane (Vulkan/RADV). Device chosen by [`EngineConfig::gpu_index`].
    #[cfg(feature = "wgpu")]
    Wgpu,
    /// The burn-cuda lane (NVIDIA CUDA / NVRTC JIT). Device chosen by [`EngineConfig::gpu_index`].
    #[cfg(feature = "cuda")]
    Cuda,
}

impl BackendKind {
    /// The stable wire slug for this lane (`"cpu"` / `"burn-ndarray"` / `"wgpu"` / `"cuda"`) —
    /// what the admitted tuple records and the worker's capability advertisement names. Slugs
    /// are stable identifiers; renaming one is a wire change.
    #[must_use]
    pub fn slug(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            #[cfg(feature = "burn-ndarray")]
            Self::BurnNdarray => "burn-ndarray",
            #[cfg(feature = "wgpu")]
            Self::Wgpu => "wgpu",
            #[cfg(feature = "cuda")]
            Self::Cuda => "cuda",
        }
    }

    /// Whether this lane executes on a GPU device (the lanes bound by the per-process compute
    /// slot and the runtime availability probe; the CPU/ndarray lanes are always available).
    #[must_use]
    pub fn is_device(self) -> bool {
        #[cfg(feature = "wgpu")]
        if matches!(self, Self::Wgpu) {
            return true;
        }
        #[cfg(feature = "cuda")]
        if matches!(self, Self::Cuda) {
            return true;
        }
        false
    }
}

/// Fixed host-side settings that affect observable semantics (ABI §2.2/§8).
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Fuel budget per entry point (ABI §8). The deterministic budget.
    pub fuel_per_call: u64,
    /// Wall-clock epoch deadline per call (ABI §8) — the pure-guest-compute watchdog.
    pub epoch_deadline: Duration,
    /// How often the background thread ticks the engine epoch.
    pub epoch_tick: Duration,
    /// Linear-memory cap (ABI §8, T1).
    pub max_memory_bytes: usize,
    /// Live step-handle cap (ABI §8).
    pub max_step_handles: usize,
    /// Host-op-call cap per entry point (ABI §8).
    pub op_budget: u64,
    /// The device-backend lane ([`BackendKind`]).
    pub backend: BackendKind,
    /// GPU device selection for the GPU lanes. `None` = the best available adapter (honoring
    /// `CUBECL_WGPU_DEFAULT_DEVICE` on wgpu); `Some(i)` = discrete device `i`. Ignored by the
    /// CPU / ndarray lanes.
    pub gpu_index: Option<u32>,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            fuel_per_call: 1 << 26,
            epoch_deadline: Duration::from_secs(5),
            epoch_tick: Duration::from_millis(100),
            max_memory_bytes: 64 * 1024 * 1024,
            max_step_handles: 1 << 20,
            op_budget: 1 << 22,
            backend: BackendKind::Cpu,
            gpu_index: None,
        }
    }
}

impl EngineConfig {
    /// The REAL-MODEL sandbox profile: the budgets a production-geometry run needs, as opposed to
    /// [`EngineConfig::default`], whose values are sized for the tiny reference model the
    /// acceptance tier trains.
    ///
    /// The two budgets a real geometry blows past are both per-slice (fuel and the epoch deadline
    /// both reset on every delivered event, ABI §4.1/§5.5) and both are dominated by the same
    /// slice: the fresh-join seed-init, which expands every parameter element from
    /// `daemon-vhc-det`'s counter-based distribution, folds it into the master family, and uploads
    /// it to the device. At the fleet-ceremony geometry (786_507_264 parameters) that ONE slice
    /// measures **323.3 G fuel / ~100 s** on the CPU lane (`daemon-vhc-testkit`'s
    /// `ceremony_geometry` gate), so `fuel_per_call` is `1 << 39` (549.8 G — the measured cost with
    /// ~1.7× headroom) and the epoch deadline is the 600 s device-lane wall. The wall-clock guard
    /// against a wedged guest is that epoch watchdog; fuel is the deterministic budget, and a
    /// budget under the init cost turns a healthy fresh join into a `BudgetFuel` trap.
    ///
    /// The ROUND path was measured against this same budget rather than assumed under it
    /// (`ceremony_round`, the round-path gate: θ export → `make_update` → ingest → quiesce at the
    /// frozen geometry, instrumenting the per-slice fuel at the `next_event` seam). Its worst slice
    /// measures **1.015 G fuel** — the ingest walk's sealing slice — with the next at 0.77 G and
    /// the streamed window slices at ~0.38 G. That is 316× under the init slice and ~540× under
    /// this budget, because every round-path walk is bounded per window by construction while init
    /// is one unbroken pass. So the budget above stands on the init measurement alone; the rounds
    /// do not move it.
    ///
    /// `max_memory_bytes` is deliberately NOT raised: a conforming guest streams state families
    /// window-by-window (design §3.2 — the fold walks, the seed expansion, the checkpoint
    /// rehydration), so its linear-memory high-water is O(window), independent of the geometry.
    /// Raising the cap here would hide exactly the class of guest regression the ceremony-geometry
    /// init gate exists to catch.
    ///
    /// Both the worker's join engine and the ceremony-geometry init gate build from this one
    /// definition, so a budget change is reviewed against a test rather than transcribed.
    #[must_use]
    pub fn real_model(backend: BackendKind, gpu_index: Option<u32>) -> Self {
        Self {
            fuel_per_call: 1 << 39,
            epoch_deadline: Duration::from_secs(600),
            op_budget: 1 << 30,
            max_step_handles: 1 << 24,
            backend,
            gpu_index,
            ..Self::default()
        }
    }
}

// -- worker + engine ------------------------------------------------------------------------------

struct EpochThread {
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl Drop for EpochThread {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// The worker: the wasmtime engine profile every sandbox store/linker is built over.
pub struct Worker {
    engine: Engine,
    config: EngineConfig,
    _epoch: EpochThread,
}

impl Worker {
    /// Build the engine with the ABI §2.2 host profile.
    ///
    /// # Errors
    ///
    /// [`TrainError::Sandbox`] if the engine cannot be configured.
    pub fn new(config: EngineConfig) -> Result<Self, TrainError> {
        let mut c = Config::new();
        c.consume_fuel(true);
        c.epoch_interruption(true);
        c.cranelift_nan_canonicalization(true);
        // Threads/atomics and relaxed-simd are off by default (default-features = false); the ABI
        // forbids them (§2.1) and NaN canonicalization + no threads gives deterministic guest
        // execution (§2.2). No WASI is linked.
        let mut pool = PoolingAllocationConfig::default();
        pool.max_memory_size(config.max_memory_bytes);
        pool.total_memories(64);
        pool.total_core_instances(64);
        c.allocation_strategy(InstanceAllocationStrategy::Pooling(pool));
        let engine = Engine::new(&c).map_err(|e| TrainError::Sandbox(e.to_string()))?;

        let stop = Arc::new(AtomicBool::new(false));
        let eng = engine.clone();
        let tick = config.epoch_tick;
        let stop_thread = stop.clone();
        let handle = std::thread::spawn(move || {
            while !stop_thread.load(Ordering::Relaxed) {
                std::thread::sleep(tick);
                eng.increment_epoch();
            }
        });

        Ok(Self {
            engine,
            config,
            _epoch: EpochThread {
                stop,
                handle: Some(handle),
            },
        })
    }

    /// The import names a module requests (the peer-side re-validation input, spec §6.5): compile
    /// the module and list its declared imports. A worker rejects a run whose module imports a
    /// symbol outside the host vocabulary before ever instantiating it.
    ///
    /// # Errors
    ///
    /// [`TrainError::Sandbox`] if the module fails to validate / compile.
    pub fn module_imports(&self, wasm: &[u8]) -> Result<Vec<String>, TrainError> {
        let module =
            Module::new(&self.engine, wasm).map_err(|e| TrainError::Sandbox(e.to_string()))?;
        Ok(module.imports().map(|i| i.name().to_string()).collect())
    }

    fn epoch_ticks(&self) -> u64 {
        let d = self.config.epoch_deadline.as_millis();
        let t = self.config.epoch_tick.as_millis().max(1);
        (d / t).max(1) as u64
    }

    /// The wasmtime [`Engine`] (fuel/epoch/NaN-canonicalized, pooling) — used by the admission-time
    /// driver-selection layer ([`crate::select`], ABI §1.3) to compile + inspect + assessment-read a
    /// module.
    pub(crate) fn engine(&self) -> &Engine {
        &self.engine
    }

    /// The engine config (fuel/epoch budgets the assessment read reuses, ABI §9.2).
    pub(crate) fn config(&self) -> &EngineConfig {
        &self.config
    }

    /// The epoch-deadline tick count for one entry point (shared with the assessment read).
    pub(crate) fn epoch_ticks_pub(&self) -> u64 {
        self.epoch_ticks()
    }
}
