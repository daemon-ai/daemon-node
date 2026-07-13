// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The wasmtime host runtime (ABI §2): engine profile, host state, the `tabi@1` dispatch layer, and
//! the lifecycle driver.
//!
//! The [`Worker`] owns the wasmtime [`Engine`] (fuel on, epoch on, NaN canonicalization, pooling
//! allocator, no WASI) and the linked import table; [`LoadedModule`] holds an `InstancePre` for
//! µs-scale re-instantiation (T3). [`Instance`] drives one module through
//! `da_abi → da_manifest → da_build → step/inner_update/make_update/ingest`, enforcing the phase
//! table ([`crate::phase`]), lane/handle rules ([`crate::handle`]), and the budgets (fuel/epoch/
//! memory/op-count/handle-count), dispatching math to the [`OpBackend`] ([`CpuBackend`] this wave).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use wasmtime::{
    Caller, Config, Engine, InstanceAllocationStrategy, InstancePre, Linker, Memory, Module,
    PoolingAllocationConfig, Store, StoreLimits, StoreLimitsBuilder,
};

use crate::backend::{AdamwHp, CpuBackend, OpBackend, TensorId};
use crate::handle::{self, HandleClass, Lane, StepArena};
use crate::phase::{self, Phase};
use crate::trap::{Trap, TrapCode};
use crate::TrainError;

/// Which [`OpBackend`] the host instantiates behind the ABI dispatch layer (G1 seam).
///
/// The default is [`BackendKind::Cpu`] — the frozen fixed-order fp32 tape (`CpuBackend`), the MVP
/// cross-peer bit-identity engine. [`BackendKind::BurnNdarray`] selects the burn-ndarray autodiff
/// engine ([`crate::burn_backend::BurnBackend`]); G2 adds a `Wgpu` arm behind the `wgpu` feature.
/// Nothing burn leaks across the `TrainerBackend`/`WasmBackend` seam — selection is data only.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum BackendKind {
    /// The `CpuBackend` fixed-order fp32 tape (MVP behavior; det lane is bit-exact everywhere).
    #[default]
    Cpu,
    /// The burn-ndarray autodiff engine (native lane = tolerance class; det lane = det-core fp32).
    #[cfg(feature = "burn-ndarray")]
    BurnNdarray,
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
    /// Which [`OpBackend`] to instantiate (G1 seam). Defaults to [`BackendKind::Cpu`].
    pub backend: BackendKind,
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
        }
    }
}

/// A registered param's public description (ABI §6.3/§6.4).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamInfo {
    /// Registration name.
    pub name: String,
    /// Shape.
    pub shape: Vec<u32>,
    /// Dtype code (ABI §3.2).
    pub dtype: u32,
    /// The stable handle assigned (deterministic across re-instantiation, T3).
    pub handle: u64,
}

/// The module cadence + round-mode block (`da_manifest`, ABI §6.2).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Manifest {
    /// Module name.
    pub name: String,
    /// Module version.
    pub version: String,
    /// SDK version.
    pub sdk: String,
    /// H — inner-step cadence.
    pub steps_per_round: u32,
    /// Round modes the module tolerates.
    pub round_modes: Vec<String>,
    /// Minimum viable round interval (ms).
    pub min_round_interval_ms: u32,
}

// -- host state ---------------------------------------------------------------------------------

struct ParamReg {
    name: String,
    shape: Vec<u32>,
    dtype: u32,
    storage: TensorId,
    master: TensorId,
    grad: TensorId,
    round_base: TensorId,
}

struct StateReg {
    #[allow(dead_code)]
    name: String,
    shape: Vec<u32>,
    #[allow(dead_code)]
    dtype: u32,
    #[allow(dead_code)]
    class: u32,
    tensor: TensorId,
}

enum Section {
    Bytes(Vec<u8>),
    Tensor { data: Vec<f32>, shape: Vec<u32> },
}

struct Container {
    sections: Vec<Section>,
}

/// The canonical wire form of one update-container section (the opaque payload the swarm moves +
/// hashes but never parses, ABI §4.3/§7.3). A `da_make_update` container serializes to a
/// `Vec<SectionWire>`; a received payload deserializes back into host [`Section`]s for ingest.
#[derive(serde::Serialize, serde::Deserialize)]
enum SectionWire {
    /// An experiment-defined opaque byte section (`upd_push_bytes`).
    Bytes(Vec<u8>),
    /// A tensor section (`upd_push_tensor`), staged det-lane at ingest.
    Tensor {
        /// Row-major fp32 data.
        data: Vec<f32>,
        /// The tensor shape.
        shape: Vec<u32>,
    },
}

/// The canonical wire form of the full worker state dict (Wave-3 checkpoint body, §9). Stores every
/// param master + round base and **all** persistents (both classes) so `load → continue` is
/// bit-exact; the replicated (`class = 1`) subset a cross-peer resync needs is included.
#[derive(serde::Serialize, serde::Deserialize)]
struct CheckpointWire {
    /// Param fp32 masters, registration order.
    params: Vec<Vec<f32>>,
    /// Param round bases (the outer-step anchor), registration order.
    round_base: Vec<Vec<f32>>,
    /// All native persistents (both classes), registration order.
    persistents: Vec<Vec<f32>>,
    /// All det persistents (both classes), registration order.
    det_persistents: Vec<Vec<f32>>,
}

struct BatchData {
    tokens: Vec<u32>,
    batch: u32,
    seq: u32,
}

/// The per-instance host state (the wasmtime `Store` data).
struct HostState {
    phase: Option<Phase>,
    backend: Box<dyn OpBackend>,
    params: Vec<ParamReg>,
    persistents: Vec<StateReg>,
    det_persistents: Vec<StateReg>,
    step_native: StepArena,
    step_det: StepArena,
    containers: Vec<Container>,
    staged: Vec<usize>,
    batches: Vec<BatchData>,
    metrics: Vec<(String, f32)>,
    names: std::collections::HashSet<String>,
    limits: StoreLimits,
    op_calls: u64,
    op_budget: u64,
    handle_budget: usize,
    trap: Option<Trap>,
    /// Every import charged over this instance's life (meta-mode `ops_used`, ABI §6.4). Unlike
    /// `op_calls` (reset per entry point), this accumulates across the whole probe.
    ops_used: std::collections::BTreeSet<&'static str>,
}

impl HostState {
    fn new(cfg: &EngineConfig) -> Self {
        let backend: Box<dyn OpBackend> = match cfg.backend {
            BackendKind::Cpu => Box::new(CpuBackend::new()),
            #[cfg(feature = "burn-ndarray")]
            BackendKind::BurnNdarray => Box::new(crate::burn_backend::BurnNdarrayBackend::new()),
        };
        Self {
            phase: None,
            backend,
            params: Vec::new(),
            persistents: Vec::new(),
            det_persistents: Vec::new(),
            step_native: StepArena::new(Lane::Native),
            step_det: StepArena::new(Lane::Det),
            containers: Vec::new(),
            staged: Vec::new(),
            batches: Vec::new(),
            metrics: Vec::new(),
            names: std::collections::HashSet::new(),
            limits: StoreLimitsBuilder::new()
                .memory_size(cfg.max_memory_bytes)
                .build(),
            op_calls: 0,
            op_budget: cfg.op_budget,
            handle_budget: cfg.max_step_handles,
            trap: None,
            ops_used: std::collections::BTreeSet::new(),
        }
    }

    fn charge_op(&mut self, import: &'static str, phase: Phase) -> Result<(), Trap> {
        self.op_calls += 1;
        self.ops_used.insert(import);
        if self.op_calls > self.op_budget {
            return Err(Trap::new(
                TrapCode::BudgetOps,
                import,
                Some(phase),
                "host-op-call budget exhausted",
            ));
        }
        Ok(())
    }

    fn live_step_handles(&self) -> usize {
        self.step_native.live() + self.step_det.live()
    }

    fn alloc_native(&mut self, tensor: TensorId, shape: Vec<u32>) -> Result<u64, Trap> {
        if self.live_step_handles() >= self.handle_budget {
            return Err(Trap::bare(
                TrapCode::BudgetHandles,
                "step-handle budget exhausted",
            ));
        }
        Ok(self.step_native.alloc(tensor, shape))
    }

    fn alloc_det(&mut self, tensor: TensorId, shape: Vec<u32>) -> Result<u64, Trap> {
        if self.live_step_handles() >= self.handle_budget {
            return Err(Trap::bare(
                TrapCode::BudgetHandles,
                "step-handle budget exhausted",
            ));
        }
        Ok(self.step_det.alloc(tensor, shape))
    }

    /// Resolve a native-lane handle → (backend tensor, shape). Cross-lane use traps.
    fn native(&self, import: &'static str, h: u64) -> Result<(TensorId, Vec<u32>), Trap> {
        match handle::classify(h) {
            Some(HandleClass::Param) => {
                let p = self
                    .params
                    .get((handle::stable_index(h) - 1) as usize)
                    .ok_or_else(|| {
                        Trap::new(TrapCode::InvalidHandle, import, self.phase, "param")
                    })?;
                Ok((p.storage, p.shape.clone()))
            }
            Some(HandleClass::Persistent) => {
                let s = self
                    .persistents
                    .get((handle::stable_index(h) - 1) as usize)
                    .ok_or_else(|| {
                        Trap::new(TrapCode::InvalidHandle, import, self.phase, "persist")
                    })?;
                Ok((s.tensor, s.shape.clone()))
            }
            Some(HandleClass::Step(Lane::Native)) => self
                .step_native
                .resolve(h)
                .map(|(t, s)| (t, s.to_vec()))
                .map_err(|c| Trap::new(c, import, self.phase, "native step")),
            Some(_) => Err(Trap::new(
                TrapCode::LaneMismatch,
                import,
                self.phase,
                "expected native lane",
            )),
            None => Err(Trap::new(
                TrapCode::InvalidHandle,
                import,
                self.phase,
                "unknown handle",
            )),
        }
    }

    /// Resolve a det-lane handle → (backend tensor, shape). Cross-lane use traps.
    fn det(&self, import: &'static str, h: u64) -> Result<(TensorId, Vec<u32>), Trap> {
        match handle::classify(h) {
            Some(HandleClass::DetPersistent) => {
                let s = self
                    .det_persistents
                    .get((handle::stable_index(h) - 1) as usize)
                    .ok_or_else(|| {
                        Trap::new(TrapCode::InvalidHandle, import, self.phase, "detpersist")
                    })?;
                Ok((s.tensor, s.shape.clone()))
            }
            Some(HandleClass::Step(Lane::Det)) => self
                .step_det
                .resolve(h)
                .map(|(t, s)| (t, s.to_vec()))
                .map_err(|c| Trap::new(c, import, self.phase, "det step")),
            Some(_) => Err(Trap::new(
                TrapCode::LaneMismatch,
                import,
                self.phase,
                "expected det lane",
            )),
            None => Err(Trap::new(
                TrapCode::InvalidHandle,
                import,
                self.phase,
                "unknown handle",
            )),
        }
    }

    fn numel(shape: &[u32]) -> usize {
        shape.iter().map(|&d| d as usize).product()
    }
}

// -- worker + module + instance -----------------------------------------------------------------

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

/// The worker: the wasmtime engine profile + the linked `tabi@1` import table.
pub struct Worker {
    engine: Engine,
    linker: Linker<HostState>,
    config: EngineConfig,
    _epoch: EpochThread,
}

impl Worker {
    /// Build the engine with the ABI §2.2 host profile and link the `tabi@1` imports.
    ///
    /// # Errors
    ///
    /// [`TrainError::Sandbox`] if the engine cannot be configured or the imports cannot be linked.
    pub fn new(config: EngineConfig) -> Result<Self, TrainError> {
        let mut c = Config::new();
        c.consume_fuel(true);
        c.epoch_interruption(true);
        c.cranelift_nan_canonicalization(true);
        // Threads/atomics and relaxed-simd are off by default (default-features = false); the ABI
        // forbids them (§2.1) and NaN canonicalization + no threads gives deterministic guest
        // execution (§2.2). No WASI is linked — the only imports are `tabi@1`.
        let mut pool = PoolingAllocationConfig::default();
        pool.max_memory_size(config.max_memory_bytes);
        pool.total_memories(64);
        pool.total_core_instances(64);
        c.allocation_strategy(InstanceAllocationStrategy::Pooling(pool));
        let engine = Engine::new(&c).map_err(|e| TrainError::Sandbox(e.to_string()))?;

        let mut linker = Linker::new(&engine);
        link_tabi(&mut linker).map_err(|e| TrainError::Sandbox(e.to_string()))?;

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
            linker,
            config,
            _epoch: EpochThread {
                stop,
                handle: Some(handle),
            },
        })
    }

    /// Compile a module and pre-link it for cheap re-instantiation (ABI §2.2, `InstancePre`).
    ///
    /// # Errors
    ///
    /// [`TrainError::Sandbox`] on a validation/compile/link failure.
    pub fn load_module(&self, wasm: &[u8]) -> Result<LoadedModule, TrainError> {
        let module =
            Module::new(&self.engine, wasm).map_err(|e| TrainError::Sandbox(e.to_string()))?;
        let pre = self
            .linker
            .instantiate_pre(&module)
            .map_err(|e| TrainError::Sandbox(e.to_string()))?;
        Ok(LoadedModule { pre })
    }

    /// The import names a module requests (the peer-side re-validation input, spec §6.5): compile
    /// the module and list its declared imports (e.g. `matmul@1`). A worker rejects a run whose
    /// module imports an op outside the host `tabi@1` vocabulary before ever instantiating it.
    ///
    /// # Errors
    ///
    /// [`TrainError::Sandbox`] if the module fails to validate / compile.
    pub fn module_imports(&self, wasm: &[u8]) -> Result<Vec<String>, TrainError> {
        let module =
            Module::new(&self.engine, wasm).map_err(|e| TrainError::Sandbox(e.to_string()))?;
        Ok(module.imports().map(|i| i.name().to_string()).collect())
    }

    /// Instantiate a loaded module with a fresh host state (T3: always safe at any boundary).
    ///
    /// # Errors
    ///
    /// [`TrainError::Sandbox`] on instantiation failure; [`TrainError::Trap`] if the module fails
    /// the `da_abi` major/minor gate (ABI §4).
    pub fn instantiate(&self, module: &LoadedModule) -> Result<Instance, TrainError> {
        let mut store = Store::new(&self.engine, HostState::new(&self.config));
        store.limiter(|s| &mut s.limits);
        let epoch_ticks = self.epoch_ticks();
        // Fuel starts at 0 under `consume_fuel`; seed it before instantiation runs any start glue.
        store
            .set_fuel(self.config.fuel_per_call)
            .map_err(|e| TrainError::Sandbox(e.to_string()))?;
        store.set_epoch_deadline(epoch_ticks);
        let instance = module
            .pre
            .instantiate(&mut store)
            .map_err(|e| TrainError::Sandbox(e.to_string()))?;
        let mut inst = Instance {
            store,
            instance,
            fuel: self.config.fuel_per_call,
            epoch_ticks,
        };
        inst.abi_gate()?;
        Ok(inst)
    }

    fn epoch_ticks(&self) -> u64 {
        let d = self.config.epoch_deadline.as_millis();
        let t = self.config.epoch_tick.as_millis().max(1);
        (d / t).max(1) as u64
    }
}

/// A compiled, pre-linked module (ABI §2.2).
pub struct LoadedModule {
    pre: InstancePre<HostState>,
}

/// One live module instance driven through the lifecycle (ABI §2.3).
pub struct Instance {
    store: Store<HostState>,
    instance: wasmtime::Instance,
    fuel: u64,
    epoch_ticks: u64,
}

impl Instance {
    fn abi_gate(&mut self) -> Result<(), TrainError> {
        let da_abi = self
            .instance
            .get_typed_func::<(), u32>(&mut self.store, "da_abi")
            .map_err(|_| Trap::bare(TrapCode::BadModule, "missing da_abi export"))?;
        let v = da_abi
            .call(&mut self.store, ())
            .map_err(|e| TrainError::Sandbox(e.to_string()))?;
        let (major, minor) = (v >> 16, v & 0xffff);
        if major != crate::TENSOR_ABI_MAJOR || minor > crate::TENSOR_ABI_MINOR {
            return Err(Trap::bare(
                TrapCode::AbiMismatch,
                format!(
                    "module abi {major}.{minor} vs host 1.{}",
                    crate::TENSOR_ABI_MINOR
                ),
            )
            .into());
        }
        Ok(())
    }

    fn memory(&mut self) -> Result<Memory, TrainError> {
        self.instance
            .get_memory(&mut self.store, "memory")
            .ok_or_else(|| Trap::bare(TrapCode::BadModule, "no exported memory").into())
    }

    fn alloc_write(&mut self, bytes: &[u8]) -> Result<u32, TrainError> {
        if bytes.is_empty() {
            return Ok(0);
        }
        let alloc = self
            .instance
            .get_typed_func::<(u32, u32), u32>(&mut self.store, "da_alloc")
            .map_err(|_| Trap::bare(TrapCode::BadModule, "missing da_alloc"))?;
        let ptr = alloc
            .call(&mut self.store, (bytes.len() as u32, 1))
            .map_err(|e| TrainError::Sandbox(e.to_string()))?;
        if ptr == 0 {
            return Err(Trap::bare(TrapCode::AllocFail, "da_alloc returned 0").into());
        }
        let mem = self.memory()?;
        mem.write(&mut self.store, ptr as usize, bytes)
            .map_err(|_| Trap::bare(TrapCode::MemOob, "config write out of bounds"))?;
        Ok(ptr)
    }

    fn free(&mut self, ptr: u32, len: u32) {
        if ptr == 0 {
            return;
        }
        if let Ok(f) = self
            .instance
            .get_typed_func::<(u32, u32, u32), ()>(&mut self.store, "da_free")
        {
            let _ = f.call(&mut self.store, (ptr, len, 1));
        }
    }

    fn prep_entry(&mut self, phase: Phase) -> Result<(), TrainError> {
        self.store
            .set_fuel(self.fuel)
            .map_err(|e| TrainError::Sandbox(e.to_string()))?;
        self.store.set_epoch_deadline(self.epoch_ticks);
        let d = self.store.data_mut();
        d.op_calls = 0;
        d.trap = None;
        d.phase = Some(phase);
        // A `da_step` is a differentiable pass: record the autodiff tape and retain step tensors so
        // `backward@1` can read its inputs (HOST-9). Other entry points don't build a tape.
        if phase == Phase::Step {
            d.backend.begin_pass();
        }
        Ok(())
    }

    fn finish_entry(&mut self, result: Result<(), wasmtime::Error>) -> Result<(), TrainError> {
        // End any differentiable pass first (stop recording so the frees below actually recycle,
        // apply deferred frees, and clear the tape), then free step handles wholesale at return
        // (ABI §3.3), regardless of outcome.
        self.store.data_mut().backend.end_pass();
        let (native_freed, det_freed) = {
            let d = self.store.data_mut();
            (d.step_native.clear(), d.step_det.clear())
        };
        {
            let d = self.store.data_mut();
            for t in native_freed.into_iter().chain(det_freed) {
                d.backend.free(t);
            }
            d.phase = None;
        }
        match result {
            Ok(()) => Ok(()),
            Err(e) => Err(self.map_error(e)),
        }
    }

    fn map_error(&mut self, e: wasmtime::Error) -> TrainError {
        if let Some(trap) = self.store.data_mut().trap.take() {
            return TrainError::Trap(trap);
        }
        // The trap reason (fuel/epoch/oob/unreachable) is often a source in the chain, not the
        // top-level message ("error while executing at wasm backtrace: …").
        let msg = e
            .chain()
            .map(|c| c.to_string())
            .collect::<Vec<_>>()
            .join(": ");
        let low = msg.to_lowercase();
        let code = if low.contains("fuel") {
            TrapCode::BudgetFuel
        } else if low.contains("epoch") {
            TrapCode::BudgetEpoch
        } else if low.contains("unreachable") {
            TrapCode::GuestPanic
        } else if low.contains("out of bounds") {
            TrapCode::MemOob
        } else if low.contains("memory") {
            TrapCode::BudgetMemory
        } else {
            return TrainError::Sandbox(msg);
        };
        TrainError::Trap(Trap::bare(code, msg))
    }

    /// `da_manifest` — pure function of the config bytes (ABI §6.2). Returns the parsed cadence.
    ///
    /// # Errors
    ///
    /// [`TrainError::Sandbox`]/[`TrainError::Trap`] on a call/decode failure.
    pub fn manifest(&mut self, config: &[u8]) -> Result<Manifest, TrainError> {
        self.store
            .set_fuel(self.fuel)
            .map_err(|e| TrainError::Sandbox(e.to_string()))?;
        self.store.set_epoch_deadline(self.epoch_ticks);
        let ptr = self.alloc_write(config)?;
        let func = self
            .instance
            .get_typed_func::<(u32, u32), u64>(&mut self.store, "da_manifest")
            .map_err(|_| Trap::bare(TrapCode::BadModule, "missing da_manifest"))?;
        let packed = func
            .call(&mut self.store, (ptr, config.len() as u32))
            .map_err(|e| TrainError::Sandbox(e.to_string()))?;
        self.free(ptr, config.len() as u32);
        let (out_ptr, out_len) = ((packed >> 32) as u32, (packed & 0xffff_ffff) as u32);
        let mem = self.memory()?;
        let start = out_ptr as usize;
        let bytes = mem
            .data(&self.store)
            .get(start..start + out_len as usize)
            .ok_or_else(|| Trap::bare(TrapCode::MemOob, "manifest span out of bounds"))?
            .to_vec();
        self.free(out_ptr, out_len);
        ciborium::from_reader(bytes.as_slice())
            .map_err(|e| TrainError::Sandbox(format!("manifest CBOR: {e}")))
    }

    /// `da_build` — register params/persistents from the config (ABI §6.3).
    ///
    /// # Errors
    ///
    /// [`TrainError::Trap`] on a registration trap (e.g. `NameCollision`).
    pub fn build(&mut self, config: &[u8]) -> Result<(), TrainError> {
        self.prep_entry(Phase::Build)?;
        let ptr = self.alloc_write(config)?;
        let func = self
            .instance
            .get_typed_func::<(u32, u32), ()>(&mut self.store, "da_build")
            .map_err(|_| Trap::bare(TrapCode::BadModule, "missing da_build"))?;
        let r = func.call(&mut self.store, (ptr, config.len() as u32));
        let out = self.finish_entry(r);
        self.free(ptr, config.len() as u32);
        // The build snapshot is the first round base (ABI §5.1).
        let d = self.store.data_mut();
        for p in 0..d.params.len() {
            let base = d.backend.view(d.params[p].master).to_vec();
            d.backend.write(d.params[p].round_base, &base);
        }
        out
    }

    /// The registered param list (canonical state dict order, ABI §6.3) — deterministic across
    /// re-instantiation (T3).
    #[must_use]
    pub fn params(&self) -> Vec<ParamInfo> {
        self.store
            .data()
            .params
            .iter()
            .enumerate()
            .map(|(i, p)| ParamInfo {
                name: p.name.clone(),
                shape: p.shape.clone(),
                dtype: p.dtype,
                handle: handle::param_handle((i + 1) as u32),
            })
            .collect()
    }

    /// Register a host micro-batch and return its handle.
    pub fn register_batch(&mut self, tokens: Vec<u32>, batch: u32, seq: u32) -> u64 {
        let d = self.store.data_mut();
        d.batches.push(BatchData { tokens, batch, seq });
        handle::batch_handle(d.batches.len() as u32)
    }

    /// `da_step` — one micro-batch forward + backward (ABI §2.3).
    ///
    /// # Errors
    ///
    /// [`TrainError::Trap`] on any trap (phase/lane/handle/budget) raised during the call.
    pub fn step(
        &mut self,
        batch: u64,
        inner_step: u32,
        mb_index: u32,
        mb_count: u32,
        step_seqs: u32,
    ) -> Result<(), TrainError> {
        self.prep_entry(Phase::Step)?;
        let func = self
            .instance
            .get_typed_func::<(u64, u32, u32, u32, u32), ()>(&mut self.store, "da_step")
            .map_err(|_| Trap::bare(TrapCode::BadModule, "missing da_step"))?;
        let r = func.call(
            &mut self.store,
            (batch, inner_step, mb_index, mb_count, step_seqs),
        );
        self.finish_entry(r)
    }

    /// `da_inner_update` — apply the inner optimizer (ABI §2.3).
    ///
    /// # Errors
    ///
    /// [`TrainError::Trap`] on any trap raised during the call.
    pub fn inner_update(&mut self, inner_step: u32) -> Result<(), TrainError> {
        self.prep_entry(Phase::InnerUpdate)?;
        let func = self
            .instance
            .get_typed_func::<u32, ()>(&mut self.store, "da_inner_update")
            .map_err(|_| Trap::bare(TrapCode::BadModule, "missing da_inner_update"))?;
        let r = func.call(&mut self.store, inner_step);
        self.finish_entry(r)
    }

    /// `da_make_update` — compress local progress; returns the container handle (ABI §2.3).
    ///
    /// # Errors
    ///
    /// [`TrainError::Trap`] on any trap raised during the call.
    pub fn make_update(&mut self, round: u64) -> Result<u64, TrainError> {
        self.prep_entry(Phase::MakeUpdate)?;
        let func = self
            .instance
            .get_typed_func::<u64, u64>(&mut self.store, "da_make_update")
            .map_err(|_| Trap::bare(TrapCode::BadModule, "missing da_make_update"))?;
        let r = func.call(&mut self.store, round);
        match r {
            Ok(handle) => {
                self.finish_entry(Ok(()))?;
                Ok(handle)
            }
            Err(e) => {
                self.finish_entry(Err(e))?;
                unreachable!("finish_entry returns Err on a wasmtime error")
            }
        }
    }

    /// Stage a built update container for the next ingest (self-inclusive; the host would stage the
    /// committed set in record order, ABI §5.11).
    pub fn stage(&mut self, container: u64) {
        if let Some(HandleClass::Update) = handle::classify(container) {
            let idx = (handle::stable_index(container) - 1) as usize;
            self.store.data_mut().staged.push(idx);
        }
    }

    /// `da_ingest_updates` — decode + aggregate + outer step (det lane, ABI §2.3).
    ///
    /// # Errors
    ///
    /// [`TrainError::Trap`] on any trap raised during the call.
    pub fn ingest(&mut self, round: u64, count: u32) -> Result<(), TrainError> {
        self.prep_entry(Phase::Ingest)?;
        let func = self
            .instance
            .get_typed_func::<(u64, u32), ()>(&mut self.store, "da_ingest_updates")
            .map_err(|_| Trap::bare(TrapCode::BadModule, "missing da_ingest_updates"))?;
        let r = func.call(&mut self.store, (round, count));
        let out = self.finish_entry(r);
        // The post-ingest master is the next round's base (barrier snapshot, ABI §2.3/§5.9).
        let d = self.store.data_mut();
        for p in 0..d.params.len() {
            let base = d.backend.view(d.params[p].master).to_vec();
            d.backend.write(d.params[p].round_base, &base);
        }
        out
    }

    /// The current fp32 master of param `name` (inspection / determinism checks).
    #[must_use]
    pub fn param_master(&self, name: &str) -> Option<Vec<f32>> {
        let d = self.store.data();
        d.params
            .iter()
            .find(|p| p.name == name)
            .map(|p| d.backend.view(p.master).to_vec())
    }

    /// The metrics reported via `metric@1`.
    #[must_use]
    pub fn metrics(&self) -> Vec<(String, f32)> {
        self.store.data().metrics.clone()
    }

    /// Run the full lifecycle once in `meta` mode and produce the `MetaReport` (ABI §6.4, HOST-8):
    /// `da_build` → one `da_step` at the representative `(batch, seq)` shape → `da_inner_update` →
    /// `da_make_update` → `da_ingest_updates` **twice** (1 and 2 staged) to fit the linear per-peer
    /// ingest cost. This build measures a real execute pass on the CPU backend (see `meta.rs`).
    ///
    /// # Errors
    ///
    /// [`TrainError::Trap`]/[`TrainError::Sandbox`] on any lifecycle failure.
    pub fn meta(
        &mut self,
        config: &[u8],
        batch: u32,
        seq: u32,
    ) -> Result<crate::MetaReport, TrainError> {
        let abi = {
            let f = self
                .instance
                .get_typed_func::<(), u32>(&mut self.store, "da_abi")
                .map_err(|_| Trap::bare(TrapCode::BadModule, "missing da_abi"))?;
            f.call(&mut self.store, ())
                .map_err(|e| TrainError::Sandbox(e.to_string()))?
        };

        self.build(config)?;
        let op_build = self.store.data().op_calls;

        let b = self.register_batch(vec![0u32; (batch * seq) as usize], batch, seq);
        self.step(b, 0, 0, 1, batch)?;
        let op_step = self.store.data().op_calls;

        self.inner_update(0)?;
        let op_inner = self.store.data().op_calls;

        let container = self.make_update(0)?;
        let op_make = self.store.data().op_calls;
        let payload_bytes_est = self.container_bytes(container);

        self.stage(container);
        self.ingest(0, 1)?;
        let op_ingest1 = self.store.data().op_calls;
        self.stage(container);
        self.ingest(0, 2)?;
        let op_ingest2 = self.store.data().op_calls;
        let ingest_op_calls_per_peer = op_ingest2.saturating_sub(op_ingest1);

        let d = self.store.data();
        let params: Vec<(String, Vec<u32>, u32)> = d
            .params
            .iter()
            .map(|p| (p.name.clone(), p.shape.clone(), p.dtype))
            .collect();
        let persistent: Vec<(String, Vec<u32>, u32, u32)> = d
            .persistents
            .iter()
            .map(|s| (s.name.clone(), s.shape.clone(), s.dtype, s.class))
            .collect();
        let det_persistent: Vec<(String, Vec<u32>, u32)> = d
            .det_persistents
            .iter()
            .map(|s| (s.name.clone(), s.shape.clone(), s.class))
            .collect();

        let param_bytes: u64 = d
            .params
            .iter()
            .map(|p| (HostState::numel(&p.shape) * dtype_size(p.dtype)) as u64)
            .sum();
        let master_bytes: u64 = d
            .params
            .iter()
            .map(|p| (HostState::numel(&p.shape) * 4) as u64)
            .sum();
        let grad_bytes = master_bytes;
        let max_master: u64 = d
            .params
            .iter()
            .map(|p| (HostState::numel(&p.shape) * 4) as u64)
            .max()
            .unwrap_or(0);

        let mut op_calls = std::collections::BTreeMap::new();
        op_calls.insert("da_build".to_string(), op_build);
        op_calls.insert("da_step".to_string(), op_step);
        op_calls.insert("da_inner_update".to_string(), op_inner);
        op_calls.insert("da_make_update".to_string(), op_make);
        op_calls.insert("da_ingest_updates".to_string(), op_ingest1);

        let ops_used: Vec<String> = d.ops_used.iter().map(|s| (*s).to_string()).collect();

        Ok(crate::MetaReport {
            abi,
            params,
            persistent,
            det_persistent,
            param_bytes,
            master_bytes,
            grad_bytes,
            // Coarse activation proxy (shape-only propagation is a later refinement).
            act_bytes_est: master_bytes,
            payload_bytes_est,
            // Streaming ingest peak ≈ ~2 dense tensors + staged payloads (§5.9).
            ingest_bytes_est: payload_bytes_est.saturating_mul(2) + 2 * max_master,
            host_ram_bytes_est: master_bytes * 2 + payload_bytes_est.saturating_mul(2),
            op_calls,
            ingest_op_calls_per_peer,
            ops_used,
            value_dependent: false,
        })
    }

    fn container_bytes(&self, container: u64) -> u64 {
        if let Some(HandleClass::Update) = handle::classify(container) {
            let idx = (handle::stable_index(container) - 1) as usize;
            if let Some(c) = self.store.data().containers.get(idx) {
                return c
                    .sections
                    .iter()
                    .map(|s| match s {
                        Section::Bytes(b) => b.len() as u64,
                        Section::Tensor { data, .. } => (data.len() * 4) as u64,
                    })
                    .sum();
            }
        }
        0
    }

    // -- Wave-3 E↔R wiring (additive) -----------------------------------------------------------

    /// Seal a `da_make_update` container handle to canonical CBOR — the opaque payload the swarm
    /// moves + hashes (never parses, ABI §4.3/§7.3). Round-trips through [`Self::ingest_payloads`].
    ///
    /// # Errors
    ///
    /// [`TrainError::Trap`] ([`TrapCode::InvalidHandle`]) if `container` is not an update container;
    /// [`TrainError::Engine`] on a (should-never-happen) CBOR encode failure.
    pub fn update_bytes(&self, container: u64) -> Result<Vec<u8>, TrainError> {
        let idx = match handle::classify(container) {
            Some(HandleClass::Update) => (handle::stable_index(container) - 1) as usize,
            _ => return Err(Trap::bare(TrapCode::InvalidHandle, "not an update container").into()),
        };
        let c =
            self.store.data().containers.get(idx).ok_or_else(|| {
                Trap::bare(TrapCode::InvalidHandle, "update container out of range")
            })?;
        let wire: Vec<SectionWire> = c
            .sections
            .iter()
            .map(|s| match s {
                Section::Bytes(b) => SectionWire::Bytes(b.clone()),
                Section::Tensor { data, shape } => SectionWire::Tensor {
                    data: data.clone(),
                    shape: shape.clone(),
                },
            })
            .collect();
        let mut buf = Vec::new();
        ciborium::into_writer(&wire, &mut buf)
            .map_err(|e| TrainError::Engine(format!("update encode: {e}")))?;
        Ok(buf)
    }

    /// Stage the record-ordered committed set (each payload from [`Self::update_bytes`]) through the
    /// `upd_*` ABI — one container per staged payload, in caller order — then run
    /// `da_ingest_updates` (the outer step). The post-ingest master becomes the next round base.
    ///
    /// The caller MUST stage in `RoundRecord` order (sorted by node pubkey bytes, §6.4 I3); ordering
    /// is a consensus input.
    ///
    /// # Errors
    ///
    /// [`TrainError::Engine`] on a payload decode failure; [`TrainError::Trap`] on any ingest trap.
    pub fn ingest_payloads(&mut self, round: u64, payloads: &[Vec<u8>]) -> Result<(), TrainError> {
        {
            let d = self.store.data_mut();
            // Fresh container arena for this ingest: make_update's container (already sealed to
            // bytes by the caller) and any prior round's staging are no longer needed.
            d.containers.clear();
            d.staged.clear();
        }
        for payload in payloads {
            let wire: Vec<SectionWire> = ciborium::from_reader(payload.as_slice())
                .map_err(|e| TrainError::Engine(format!("payload decode: {e}")))?;
            let sections = wire
                .into_iter()
                .map(|s| match s {
                    SectionWire::Bytes(b) => Section::Bytes(b),
                    SectionWire::Tensor { data, shape } => Section::Tensor { data, shape },
                })
                .collect();
            let d = self.store.data_mut();
            d.containers.push(Container { sections });
            let idx = d.containers.len() - 1;
            d.staged.push(idx);
        }
        self.ingest(round, payloads.len() as u32)
    }

    /// The canonical state bytes the round digest is taken over (spec §5.6): every param fp32 master,
    /// then the `class = 1` **replicated** native persistents, then the replicated det persistents —
    /// all in registration order, little-endian. Local (`class = 0`) persistents are excluded (peers
    /// rebuild them, ABI §5.1). Feed this to `daemon_swarm_proto::digest::digest_state`.
    #[must_use]
    pub fn canonical_state_bytes(&self) -> Vec<u8> {
        let d = self.store.data();
        let mut buf = Vec::new();
        for p in &d.params {
            for v in d.backend.view(p.master) {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
        for s in d.persistents.iter().filter(|s| s.class == 1) {
            for v in d.backend.view(s.tensor) {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
        for s in d.det_persistents.iter().filter(|s| s.class == 1) {
            for v in d.backend.view(s.tensor) {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
        buf
    }

    /// Serialize the full worker state dict with blake3 integrity (§9): `blake3(body) ++ body`,
    /// `body` = canonical CBOR of masters + round bases + all persistents + all det persistents.
    /// Restored bit-exactly by [`Self::restore_checkpoint`].
    #[must_use]
    pub fn checkpoint_bytes(&self) -> Vec<u8> {
        let d = self.store.data();
        let wire = CheckpointWire {
            params: d
                .params
                .iter()
                .map(|p| d.backend.view(p.master).to_vec())
                .collect(),
            round_base: d
                .params
                .iter()
                .map(|p| d.backend.view(p.round_base).to_vec())
                .collect(),
            persistents: d
                .persistents
                .iter()
                .map(|s| d.backend.view(s.tensor).to_vec())
                .collect(),
            det_persistents: d
                .det_persistents
                .iter()
                .map(|s| d.backend.view(s.tensor).to_vec())
                .collect(),
        };
        let mut body = Vec::new();
        ciborium::into_writer(&wire, &mut body)
            .expect("CheckpointWire is always CBOR-serializable");
        let sum = blake3::hash(&body);
        let mut out = Vec::with_capacity(32 + body.len());
        out.extend_from_slice(sum.as_bytes());
        out.extend_from_slice(&body);
        out
    }

    /// Restore state from [`Self::checkpoint_bytes`], verifying the blake3 integrity prefix and the
    /// per-slot lengths. Sets each param's master + storage + round base and every persistent, so a
    /// re-instantiated instance (T3) continues bit-exactly.
    ///
    /// # Errors
    ///
    /// [`TrainError::Engine`] on a short/corrupt buffer, a blake3 mismatch, or a layout mismatch
    /// (wrong param/persistent count or element count vs the current registration).
    pub fn restore_checkpoint(&mut self, bytes: &[u8]) -> Result<(), TrainError> {
        let (sum, body) = bytes
            .split_at_checked(32)
            .ok_or_else(|| TrainError::Engine("checkpoint too short".into()))?;
        if blake3::hash(body).as_bytes() != sum {
            return Err(TrainError::Engine("checkpoint blake3 mismatch".into()));
        }
        let wire: CheckpointWire = ciborium::from_reader(body)
            .map_err(|e| TrainError::Engine(format!("checkpoint decode: {e}")))?;
        let d = self.store.data_mut();
        let np = d.params.len();
        if wire.params.len() != np
            || wire.round_base.len() != np
            || wire.persistents.len() != d.persistents.len()
            || wire.det_persistents.len() != d.det_persistents.len()
        {
            return Err(TrainError::Engine(
                "checkpoint layout does not match the built registration".into(),
            ));
        }
        let expect = |got: usize, want: usize, what: &str| {
            if got == want {
                Ok(())
            } else {
                Err(TrainError::Engine(format!(
                    "checkpoint {what} length {got} != registered {want}"
                )))
            }
        };
        for i in 0..np {
            let (master, storage, rb, want) = (
                d.params[i].master,
                d.params[i].storage,
                d.params[i].round_base,
                HostState::numel(&d.params[i].shape),
            );
            expect(wire.params[i].len(), want, "param")?;
            expect(wire.round_base[i].len(), want, "round_base")?;
            d.backend.write(master, &wire.params[i]);
            d.backend.write(storage, &wire.params[i]);
            d.backend.write(rb, &wire.round_base[i]);
        }
        for (i, s) in d.persistents.iter().enumerate() {
            expect(
                wire.persistents[i].len(),
                HostState::numel(&s.shape),
                "persistent",
            )?;
        }
        for (i, s) in d.det_persistents.iter().enumerate() {
            expect(
                wire.det_persistents[i].len(),
                HostState::numel(&s.shape),
                "det_persistent",
            )?;
        }
        let persist_ids: Vec<TensorId> = d.persistents.iter().map(|s| s.tensor).collect();
        let det_ids: Vec<TensorId> = d.det_persistents.iter().map(|s| s.tensor).collect();
        for (t, data) in persist_ids.into_iter().zip(&wire.persistents) {
            d.backend.write(t, data);
        }
        for (t, data) in det_ids.into_iter().zip(&wire.det_persistents) {
            d.backend.write(t, data);
        }
        Ok(())
    }

    /// The number of host imports charged over this instance's life (the HOST-15 manifest-purity
    /// probe: `da_manifest` must charge zero — it enters no phase, so any import would trap).
    #[must_use]
    pub fn imports_charged(&self) -> usize {
        self.store.data().ops_used.len()
    }
}

/// Byte size of a stored element of `dtype` (ABI §3.2).
fn dtype_size(dtype: u32) -> usize {
    match dtype {
        3 => 8,     // I64
        1 | 2 => 2, // BF16 / F16
        6 | 7 => 1, // U8 / Bool
        _ => 4,     // F32 / I32 / U32
    }
}

// -- memory helpers (host-function side) --------------------------------------------------------

fn mem_of(caller: &mut Caller<'_, HostState>) -> Result<Memory, Trap> {
    caller
        .get_export("memory")
        .and_then(wasmtime::Extern::into_memory)
        .ok_or_else(|| Trap::bare(TrapCode::BadModule, "module has no exported memory"))
}

fn read_bytes(caller: &mut Caller<'_, HostState>, ptr: u32, len: u32) -> Result<Vec<u8>, Trap> {
    let mem = mem_of(caller)?;
    let data = mem.data(&caller);
    let (start, end) = (ptr as usize, ptr as usize + len as usize);
    data.get(start..end)
        .map(<[u8]>::to_vec)
        .ok_or_else(|| Trap::bare(TrapCode::MemOob, "span out of bounds"))
}

fn read_str(caller: &mut Caller<'_, HostState>, ptr: u32, len: u32) -> Result<String, Trap> {
    let bytes = read_bytes(caller, ptr, len)?;
    String::from_utf8(bytes).map_err(|_| Trap::bare(TrapCode::MemOob, "name is not utf-8"))
}

fn read_dims(caller: &mut Caller<'_, HostState>, ptr: u32, rank: u32) -> Result<Vec<u32>, Trap> {
    if rank > 8 {
        return Err(Trap::bare(TrapCode::RankOverflow, "rank > 8"));
    }
    let bytes = read_bytes(caller, ptr, rank * 4)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn read_handles(
    caller: &mut Caller<'_, HostState>,
    ptr: u32,
    count: u32,
) -> Result<Vec<u64>, Trap> {
    let bytes = read_bytes(caller, ptr, count * 8)?;
    Ok(bytes
        .chunks_exact(8)
        .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

fn stash<T>(caller: &mut Caller<'_, HostState>, r: Result<T, Trap>) -> Result<T, wasmtime::Error> {
    r.map_err(|t| {
        let msg = t.to_string();
        caller.data_mut().trap = Some(t);
        wasmtime::Error::msg(msg)
    })
}

fn enter(caller: &mut Caller<'_, HostState>, import: &'static str) -> Result<Phase, Trap> {
    let phase = caller.data().phase.ok_or_else(|| {
        Trap::new(
            TrapCode::PhaseViolation,
            import,
            None,
            "outside an entry point",
        )
    })?;
    phase::guard(import, phase)?;
    caller.data_mut().charge_op(import, phase)?;
    Ok(phase)
}

fn fake_init(name: &str, n: usize, init: u32, p0: f64, p1: f64) -> Vec<f32> {
    match init {
        0 => vec![0.0; n],
        1 => vec![1.0; n],
        _ => {
            // Deterministic-by-name pseudo values. The real host seeds from (run_id, name); T3 only
            // needs determinism across re-instantiation, which name-hashing provides.
            let mut s = 0xcbf2_9ce4_8422_2325_u64;
            for b in name.bytes() {
                s ^= u64::from(b);
                s = s.wrapping_mul(0x0000_0100_0000_01b3);
            }
            let (p0, span) = (p0 as f32, (p1 - p0) as f32);
            (0..n)
                .map(|i| {
                    s ^= (i as u64).wrapping_add(1);
                    s = s.wrapping_mul(0x2545_F491_4F6C_DD1D);
                    let u = ((s >> 40) as f32) / ((1u64 << 24) as f32);
                    p0 + span * u
                })
                .collect()
        }
    }
}

impl HostState {
    // registration ------------------------------------------------------------------------------

    fn register_param(
        &mut self,
        name: &str,
        dims: &[u32],
        dtype: u32,
        init: u32,
        p0: f64,
        p1: f64,
    ) -> Result<u64, Trap> {
        if dtype > 7 {
            return Err(Trap::new(TrapCode::BadEnum, "param@1", self.phase, "dtype"));
        }
        if init > 4 {
            return Err(Trap::new(TrapCode::BadEnum, "param@1", self.phase, "init"));
        }
        if !self.names.insert(format!("param:{name}")) {
            return Err(Trap::new(
                TrapCode::NameCollision,
                "param@1",
                self.phase,
                name.to_string(),
            ));
        }
        let n = Self::numel(dims);
        let init_v = fake_init(name, n, init, p0, p1);
        let master = self.backend.create(init_v.clone());
        let storage = self.backend.create(init_v.clone());
        let grad = self.backend.zeros(n);
        let round_base = self.backend.create(init_v);
        self.params.push(ParamReg {
            name: name.to_string(),
            shape: dims.to_vec(),
            dtype,
            storage,
            master,
            grad,
            round_base,
        });
        Ok(handle::param_handle(self.params.len() as u32))
    }

    fn register_persistent(
        &mut self,
        name: &str,
        dims: &[u32],
        dtype: u32,
        class: u32,
    ) -> Result<u64, Trap> {
        if dtype > 7 || class > 1 {
            return Err(Trap::new(
                TrapCode::BadEnum,
                "persistent@1",
                self.phase,
                "dtype/class",
            ));
        }
        if !self.names.insert(format!("persist:{name}")) {
            return Err(Trap::new(
                TrapCode::NameCollision,
                "persistent@1",
                self.phase,
                name.to_string(),
            ));
        }
        let tensor = self.backend.zeros(Self::numel(dims));
        self.persistents.push(StateReg {
            name: name.to_string(),
            shape: dims.to_vec(),
            dtype,
            class,
            tensor,
        });
        Ok(handle::persistent_handle(self.persistents.len() as u32))
    }

    fn register_det_persistent(
        &mut self,
        name: &str,
        dims: &[u32],
        class: u32,
    ) -> Result<u64, Trap> {
        if class > 1 {
            return Err(Trap::new(
                TrapCode::BadEnum,
                "det_persistent@1",
                self.phase,
                "class",
            ));
        }
        if !self.names.insert(format!("detpersist:{name}")) {
            return Err(Trap::new(
                TrapCode::NameCollision,
                "det_persistent@1",
                self.phase,
                name.to_string(),
            ));
        }
        let tensor = self.backend.zeros(Self::numel(dims));
        self.det_persistents.push(StateReg {
            name: name.to_string(),
            shape: dims.to_vec(),
            dtype: 0,
            class,
            tensor,
        });
        Ok(handle::det_persistent_handle(
            self.det_persistents.len() as u32
        ))
    }

    // creation ----------------------------------------------------------------------------------

    fn op_create(&mut self, dims: &[u32], value: f32) -> Result<u64, Trap> {
        let n = Self::numel(dims);
        let t = self.backend.create(vec![value; n]);
        self.alloc_native(t, dims.to_vec())
    }

    // native math -------------------------------------------------------------------------------

    fn op_matmul(&mut self, a: u64, b: u64) -> Result<u64, Trap> {
        let (at, ash) = self.native("matmul@1", a)?;
        let (bt, bsh) = self.native("matmul@1", b)?;
        if ash.len() != 2 || bsh.len() != 2 || ash[1] != bsh[0] {
            return Err(Trap::new(
                TrapCode::ShapeMismatch,
                "matmul@1",
                self.phase,
                "2-D contract",
            ));
        }
        let (m, k, n) = (ash[0] as usize, ash[1] as usize, bsh[1] as usize);
        let out = self.backend.matmul(at, m, k, bt, n);
        self.alloc_native(out, vec![ash[0], bsh[1]])
    }

    fn op_add(&mut self, a: u64, b: u64) -> Result<u64, Trap> {
        let (at, ash) = self.native("add@1", a)?;
        let (bt, bsh) = self.native("add@1", b)?;
        if ash == bsh {
            let out = self.backend.add(at, bt);
            self.alloc_native(out, ash)
        } else if bsh.len() == 1 && Some(&bsh[0]) == ash.last() {
            let cols = bsh[0] as usize;
            let rows = Self::numel(&ash) / cols;
            let out = self.backend.add_bias(at, bt, rows, cols);
            self.alloc_native(out, ash)
        } else {
            Err(Trap::new(
                TrapCode::ShapeMismatch,
                "add@1",
                self.phase,
                "broadcast",
            ))
        }
    }

    fn op_binary_same(&mut self, import: &'static str, a: u64, b: u64) -> Result<u64, Trap> {
        let (at, ash) = self.native(import, a)?;
        let (bt, bsh) = self.native(import, b)?;
        if ash != bsh {
            return Err(Trap::new(
                TrapCode::ShapeMismatch,
                import,
                self.phase,
                "elementwise",
            ));
        }
        let out = match import {
            "sub@1" => self.backend.sub(at, bt),
            _ => self.backend.mul(at, bt),
        };
        self.alloc_native(out, ash)
    }

    fn op_mul_s(&mut self, x: u64, v: f64) -> Result<u64, Trap> {
        let (xt, xsh) = self.native("mul_s@1", x)?;
        let out = self.backend.mul_s(xt, v);
        self.alloc_native(out, xsh)
    }

    fn op_relu(&mut self, x: u64) -> Result<u64, Trap> {
        let (xt, xsh) = self.native("relu@1", x)?;
        let out = self.backend.relu(xt);
        self.alloc_native(out, xsh)
    }

    fn op_cross_entropy(&mut self, logits: u64, targets: u64, ignore: i64) -> Result<u64, Trap> {
        let (lt, lsh) = self.native("cross_entropy@1", logits)?;
        if lsh.len() != 2 {
            return Err(Trap::new(
                TrapCode::ShapeMismatch,
                "cross_entropy@1",
                self.phase,
                "rank",
            ));
        }
        let (tt, _) = self.native("cross_entropy@1", targets)?;
        let tgt: Vec<i64> = self.backend.view(tt).iter().map(|&f| f as i64).collect();
        let out = self
            .backend
            .cross_entropy(lt, lsh[0] as usize, lsh[1] as usize, &tgt, ignore);
        self.alloc_native(out, Vec::new())
    }

    // autodiff ----------------------------------------------------------------------------------

    fn op_backward(&mut self, loss: u64) -> Result<(), Trap> {
        let (lt, _) = self.native("backward@1", loss)?;
        if self.backend.view(lt).len() != 1 {
            return Err(Trap::new(
                TrapCode::NotScalar,
                "backward@1",
                self.phase,
                "loss numel != 1",
            ));
        }
        // Reverse-mode autodiff over the recorded tape (HOST-9), then fold each param's
        // dL/d(storage) into its `grad` tensor — accumulating across micro-batch `da_step` passes
        // (the guest clears it via `zero_grads@1`). `grad@1` / `adamw_step@1` read that tensor.
        self.backend.backward(lt);
        let params: Vec<(TensorId, TensorId)> =
            self.params.iter().map(|p| (p.storage, p.grad)).collect();
        for (storage, grad_t) in params {
            if let Some(g) = self.backend.grad_of(storage) {
                let mut cur = self.backend.view(grad_t).to_vec();
                for (c, v) in cur.iter_mut().zip(g.iter()) {
                    *c += v;
                }
                self.backend.write(grad_t, &cur);
            }
        }
        Ok(())
    }

    fn op_grad(&mut self, p: u64) -> Result<u64, Trap> {
        let index = self.param_index("grad@1", p)?;
        let (grad, shape) = (self.params[index].grad, self.params[index].shape.clone());
        let view = self.backend.clone_tensor(grad);
        self.alloc_native(view, shape)
    }

    fn op_zero_grads(&mut self) {
        let ids: Vec<(TensorId, usize)> = self
            .params
            .iter()
            .map(|p| (p.grad, Self::numel(&p.shape)))
            .collect();
        for (grad, n) in ids {
            self.backend.write(grad, &vec![0.0; n]);
        }
    }

    fn op_assign(&mut self, dst: u64, src: u64) -> Result<(), Trap> {
        let (st, _) = self.native("assign@1", src)?;
        let data = self.backend.view(st).to_vec();
        match handle::classify(dst) {
            Some(HandleClass::Param) => {
                let idx = (handle::stable_index(dst) - 1) as usize;
                let p = self.params.get(idx).ok_or_else(|| {
                    Trap::new(TrapCode::InvalidHandle, "assign@1", self.phase, "param")
                })?;
                let (storage, master) = (p.storage, p.master);
                self.backend.write(storage, &data);
                self.backend.write(master, &data);
                Ok(())
            }
            Some(HandleClass::Persistent) => {
                let idx = (handle::stable_index(dst) - 1) as usize;
                let t = self
                    .persistents
                    .get(idx)
                    .ok_or_else(|| {
                        Trap::new(TrapCode::InvalidHandle, "assign@1", self.phase, "persist")
                    })?
                    .tensor;
                self.backend.write(t, &data);
                Ok(())
            }
            _ => Err(Trap::new(
                TrapCode::InvalidHandle,
                "assign@1",
                self.phase,
                "assign target",
            )),
        }
    }

    fn param_index(&self, import: &'static str, p: u64) -> Result<usize, Trap> {
        match handle::classify(p) {
            Some(HandleClass::Param) => {
                let idx = (handle::stable_index(p) - 1) as usize;
                if idx < self.params.len() {
                    Ok(idx)
                } else {
                    Err(Trap::new(
                        TrapCode::InvalidHandle,
                        import,
                        self.phase,
                        "param range",
                    ))
                }
            }
            _ => Err(Trap::new(
                TrapCode::InvalidHandle,
                import,
                self.phase,
                "not a param",
            )),
        }
    }

    fn op_param_round_base(&mut self, p: u64) -> Result<u64, Trap> {
        let index = self.param_index("param_round_base@1", p)?;
        let (rb, shape) = (
            self.params[index].round_base,
            self.params[index].shape.clone(),
        );
        let view = self.backend.clone_tensor(rb);
        self.alloc_native(view, shape)
    }

    #[allow(clippy::too_many_arguments)]
    fn op_adamw_step(
        &mut self,
        p: u64,
        g: u64,
        m: u64,
        v: u64,
        step: u32,
        lr: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
        wd: f64,
    ) -> Result<(), Trap> {
        let pi = self.param_index("adamw_step@1", p)?;
        let (gt, _) = self.native("adamw_step@1", g)?;
        let mt = self.persist_tensor("adamw_step@1", m)?;
        let vt = self.persist_tensor("adamw_step@1", v)?;
        let (master, storage) = (self.params[pi].master, self.params[pi].storage);
        self.backend.adamw_step(
            master,
            gt,
            mt,
            vt,
            AdamwHp {
                step,
                lr,
                beta1,
                beta2,
                eps,
                wd,
            },
        );
        let m_v = self.backend.view(master).to_vec();
        self.backend.write(storage, &m_v);
        Ok(())
    }

    fn persist_tensor(&self, import: &'static str, h: u64) -> Result<TensorId, Trap> {
        match handle::classify(h) {
            Some(HandleClass::Persistent) => self
                .persistents
                .get((handle::stable_index(h) - 1) as usize)
                .map(|s| s.tensor)
                .ok_or_else(|| Trap::new(TrapCode::InvalidHandle, import, self.phase, "persist")),
            _ => Err(Trap::new(
                TrapCode::InvalidHandle,
                import,
                self.phase,
                "not a persistent",
            )),
        }
    }

    // readouts ----------------------------------------------------------------------------------

    fn scalar_tensor(&self, import: &'static str, x: u64) -> Result<TensorId, Trap> {
        match handle::classify(x) {
            Some(
                HandleClass::Step(Lane::Native) | HandleClass::Param | HandleClass::Persistent,
            ) => Ok(self.native(import, x)?.0),
            Some(HandleClass::Step(Lane::Det) | HandleClass::DetPersistent) => {
                Ok(self.det(import, x)?.0)
            }
            _ => Err(Trap::new(
                TrapCode::InvalidHandle,
                import,
                self.phase,
                "scalar handle",
            )),
        }
    }

    fn op_scalar(&self, x: u64) -> Result<f64, Trap> {
        let t = self.scalar_tensor("scalar@1", x)?;
        let data = self.backend.view(t);
        if data.len() != 1 {
            return Err(Trap::new(
                TrapCode::NotScalar,
                "scalar@1",
                self.phase,
                "numel != 1",
            ));
        }
        Ok(f64::from(data[0]))
    }

    fn op_metric(&mut self, name: &str, x: u64) -> Result<(), Trap> {
        let t = self.scalar_tensor("metric@1", x)?;
        let val = self.backend.view(t).first().copied().unwrap_or(0.0);
        self.metrics.push((name.to_string(), val));
        Ok(())
    }

    // batch -------------------------------------------------------------------------------------

    fn batch_ref(&self, import: &'static str, b: u64) -> Result<&BatchData, Trap> {
        match handle::classify(b) {
            Some(HandleClass::Batch) => self
                .batches
                .get((handle::stable_index(b) - 1) as usize)
                .ok_or_else(|| Trap::new(TrapCode::InvalidHandle, import, self.phase, "batch")),
            _ => Err(Trap::new(
                TrapCode::InvalidHandle,
                import,
                self.phase,
                "not a batch",
            )),
        }
    }

    fn op_batch_tokens(&mut self, b: u64) -> Result<u64, Trap> {
        let (data, batch, seq) = {
            let bd = self.batch_ref("batch_tokens@1", b)?;
            (
                bd.tokens.iter().map(|&t| t as f32).collect::<Vec<f32>>(),
                bd.batch,
                bd.seq,
            )
        };
        let t = self.backend.create(data);
        self.alloc_native(t, vec![batch, seq])
    }

    // update container --------------------------------------------------------------------------

    fn op_upd_new(&mut self) -> u64 {
        self.containers.push(Container {
            sections: Vec::new(),
        });
        handle::update_handle(self.containers.len() as u32)
    }

    fn container_index(&self, import: &'static str, u: u64) -> Result<usize, Trap> {
        match handle::classify(u) {
            Some(HandleClass::Update) => {
                let idx = (handle::stable_index(u) - 1) as usize;
                if idx < self.containers.len() {
                    Ok(idx)
                } else {
                    Err(Trap::new(
                        TrapCode::InvalidHandle,
                        import,
                        self.phase,
                        "container",
                    ))
                }
            }
            _ => Err(Trap::new(
                TrapCode::InvalidHandle,
                import,
                self.phase,
                "not a container",
            )),
        }
    }

    fn op_upd_push_bytes(&mut self, u: u64, data: Vec<u8>) -> Result<(), Trap> {
        let idx = self.container_index("upd_push_bytes@1", u)?;
        self.containers[idx].sections.push(Section::Bytes(data));
        Ok(())
    }

    fn op_upd_push_tensor(&mut self, u: u64, x: u64) -> Result<(), Trap> {
        let (xt, xsh) = self.native("upd_push_tensor@1", x)?;
        let data = self.backend.view(xt).to_vec();
        let idx = self.container_index("upd_push_tensor@1", u)?;
        self.containers[idx]
            .sections
            .push(Section::Tensor { data, shape: xsh });
        Ok(())
    }

    fn staged_section(&self, import: &'static str, i: u32, s: u32) -> Result<&Section, Trap> {
        let ci = *self.staged.get(i as usize).ok_or_else(|| {
            Trap::new(TrapCode::InvalidHandle, import, self.phase, "staged index")
        })?;
        self.containers[ci]
            .sections
            .get(s as usize)
            .ok_or_else(|| Trap::new(TrapCode::InvalidHandle, import, self.phase, "section index"))
    }

    fn op_upd_sections(&self, i: u32) -> Result<u32, Trap> {
        let ci = *self.staged.get(i as usize).ok_or_else(|| {
            Trap::new(
                TrapCode::InvalidHandle,
                "upd_sections@1",
                self.phase,
                "staged",
            )
        })?;
        Ok(self.containers[ci].sections.len() as u32)
    }

    fn op_upd_kind(&self, i: u32, s: u32) -> Result<u32, Trap> {
        Ok(match self.staged_section("upd_kind@1", i, s)? {
            Section::Bytes(_) => 0,
            Section::Tensor { .. } => 1,
        })
    }

    fn op_upd_bytes_len(&self, i: u32, s: u32) -> Result<u32, Trap> {
        Ok(match self.staged_section("upd_bytes_len@1", i, s)? {
            Section::Bytes(b) => b.len() as u32,
            Section::Tensor { .. } => 0,
        })
    }

    fn op_upd_bytes(&self, i: u32, s: u32) -> Result<Vec<u8>, Trap> {
        Ok(match self.staged_section("upd_read_bytes@1", i, s)? {
            Section::Bytes(b) => b.clone(),
            Section::Tensor { .. } => Vec::new(),
        })
    }

    fn op_upd_tensor(&mut self, i: u32, s: u32) -> Result<u64, Trap> {
        let (data, shape) = match self.staged_section("upd_tensor@1", i, s)? {
            Section::Tensor { data, shape } => (data.clone(), shape.clone()),
            Section::Bytes(_) => {
                return Err(Trap::new(
                    TrapCode::DtypeMismatch,
                    "upd_tensor@1",
                    self.phase,
                    "bytes section",
                ))
            }
        };
        let t = self.backend.create(data);
        self.alloc_det(t, shape)
    }

    // det lane ----------------------------------------------------------------------------------

    fn op_det_zeros(&mut self, dims: &[u32]) -> Result<u64, Trap> {
        let t = self.backend.zeros(Self::numel(dims));
        self.alloc_det(t, dims.to_vec())
    }

    fn op_det_sum(&mut self, handles: &[u64]) -> Result<u64, Trap> {
        let mut ids = Vec::with_capacity(handles.len());
        for &h in handles {
            ids.push(self.det("det_sum@1", h)?.0);
        }
        let shape = handles
            .first()
            .map(|&h| self.det("det_sum@1", h).map(|(_, s)| s))
            .transpose()?
            .unwrap_or_default();
        let out = self
            .backend
            .det_sum(&ids)
            .map_err(|c| Trap::new(c, "det_sum@1", self.phase, "shapes"))?;
        self.alloc_det(out, shape)
    }

    fn op_det_scale(&mut self, x: u64, alpha: f64) -> Result<u64, Trap> {
        let (xt, xsh) = self.det("det_scale@1", x)?;
        let out = self.backend.det_scale(xt, alpha);
        self.alloc_det(out, xsh)
    }

    fn op_det_l2norm(&self, x: u64) -> Result<f64, Trap> {
        let (xt, _) = self.det("det_l2norm@1", x)?;
        Ok(f64::from(self.backend.det_l2norm(xt)))
    }

    fn op_det_sign(&mut self, x: u64) -> Result<u64, Trap> {
        let (xt, xsh) = self.det("det_sign@1", x)?;
        let out = self.backend.det_sign(xt);
        self.alloc_det(out, xsh)
    }

    fn op_det_binary(&mut self, import: &'static str, a: u64, b: u64) -> Result<u64, Trap> {
        let (at, ash) = self.det(import, a)?;
        let (bt, _) = self.det(import, b)?;
        let out = match import {
            "det_add@1" => self.backend.det_add(at, bt),
            "det_sub@1" => self.backend.det_sub(at, bt),
            _ => self.backend.det_mul(at, bt),
        }
        .map_err(|c| Trap::new(c, import, self.phase, "shapes"))?;
        self.alloc_det(out, ash)
    }

    fn op_det_absmax_unpack(&mut self, packed: u64, chunk: u32, bits: u32) -> Result<u64, Trap> {
        let (pt, _) = self.det("det_absmax_unpack@1", packed)?;
        let out = self
            .backend
            .det_absmax_unpack(pt, chunk as usize, bits)
            .map_err(|c| Trap::new(c, "det_absmax_unpack@1", self.phase, "layout"))?;
        let n = self.backend.view(out).len() as u32;
        self.alloc_det(out, vec![n])
    }

    fn op_det_chunk_scatter_add(
        &mut self,
        acc: u64,
        vals: u64,
        idx: u64,
        chunk: u32,
    ) -> Result<(), Trap> {
        let (acct, _) = self.det("det_chunk_scatter_add@1", acc)?;
        let (valst, _) = self.det("det_chunk_scatter_add@1", vals)?;
        let (idxt, _) = self.det("det_chunk_scatter_add@1", idx)?;
        self.backend
            .det_chunk_scatter_add(acct, valst, idxt, chunk as usize)
            .map_err(|c| Trap::new(c, "det_chunk_scatter_add@1", self.phase, "layout"))
    }

    fn op_det_chunk_scatter(
        &mut self,
        vals: u64,
        idx: u64,
        chunk: u32,
        dims: &[u32],
    ) -> Result<u64, Trap> {
        let (valst, _) = self.det("det_chunk_scatter@1", vals)?;
        let (idxt, _) = self.det("det_chunk_scatter@1", idx)?;
        let out = self
            .backend
            .det_chunk_scatter(valst, idxt, chunk as usize, Self::numel(dims))
            .map_err(|c| Trap::new(c, "det_chunk_scatter@1", self.phase, "layout"))?;
        self.alloc_det(out, dims.to_vec())
    }

    fn op_det_assign(&mut self, dst: u64, src: u64) -> Result<(), Trap> {
        let (st, _) = self.det("det_assign@1", src)?;
        let data = self.backend.view(st).to_vec();
        match handle::classify(dst) {
            Some(HandleClass::DetPersistent) => {
                let t = self
                    .det_persistents
                    .get((handle::stable_index(dst) - 1) as usize)
                    .ok_or_else(|| {
                        Trap::new(
                            TrapCode::InvalidHandle,
                            "det_assign@1",
                            self.phase,
                            "detpersist",
                        )
                    })?
                    .tensor;
                self.backend.write(t, &data);
                Ok(())
            }
            Some(HandleClass::Step(Lane::Det)) => {
                let (t, _) = self.det("det_assign@1", dst)?;
                self.backend.write(t, &data);
                Ok(())
            }
            _ => Err(Trap::new(
                TrapCode::InvalidHandle,
                "det_assign@1",
                self.phase,
                "det target",
            )),
        }
    }

    fn op_det_param(&mut self, p: u64) -> Result<u64, Trap> {
        let index = self.param_index("det_param@1", p)?;
        let (rb, shape) = (
            self.params[index].round_base,
            self.params[index].shape.clone(),
        );
        let view = self.backend.clone_tensor(rb);
        self.alloc_det(view, shape)
    }

    fn op_det_reset_param_to_base(&mut self, p: u64) -> Result<(), Trap> {
        let index = self.param_index("det_reset_param_to_base@1", p)?;
        let (master, storage, rb) = (
            self.params[index].master,
            self.params[index].storage,
            self.params[index].round_base,
        );
        let base = self.backend.view(rb).to_vec();
        self.backend.write(master, &base);
        self.backend.write(storage, &base);
        Ok(())
    }

    fn op_det_axpy_param(&mut self, p: u64, x: u64, alpha: f64) -> Result<(), Trap> {
        let index = self.param_index("det_axpy_param@1", p)?;
        let (xt, _) = self.det("det_axpy_param@1", x)?;
        let (master, storage) = (self.params[index].master, self.params[index].storage);
        self.backend
            .det_axpy(master, alpha, xt)
            .map_err(|c| Trap::new(c, "det_axpy_param@1", self.phase, "shapes"))?;
        let m_v = self.backend.view(master).to_vec();
        self.backend.write(storage, &m_v);
        Ok(())
    }

    // -- Wave-2 NN / shape (native lane) --------------------------------------------------------

    fn op_embedding(&mut self, w: u64, ids: u64) -> Result<u64, Trap> {
        let (wt, wsh) = self.native("embedding@1", w)?;
        if wsh.len() != 2 {
            return Err(Trap::new(
                TrapCode::ShapeMismatch,
                "embedding@1",
                self.phase,
                "weight rank",
            ));
        }
        let d = wsh[1] as usize;
        let (idst, idsh) = self.native("embedding@1", ids)?;
        let id_usize: Vec<usize> = self
            .backend
            .view(idst)
            .iter()
            .map(|&f| f as usize)
            .collect();
        let out = self.backend.embedding(wt, &id_usize, d);
        let mut shape = idsh;
        shape.push(wsh[1]);
        self.alloc_native(out, shape)
    }

    fn op_rmsnorm(&mut self, x: u64, w: u64, eps: f64) -> Result<u64, Trap> {
        let (xt, xsh) = self.native("rmsnorm@1", x)?;
        let (wt, _) = self.native("rmsnorm@1", w)?;
        let d = *xsh
            .last()
            .ok_or_else(|| Trap::new(TrapCode::ShapeMismatch, "rmsnorm@1", self.phase, "rank"))?
            as usize;
        let rows = Self::numel(&xsh) / d;
        let out = self.backend.rmsnorm(xt, wt, rows, d, eps);
        self.alloc_native(out, xsh)
    }

    fn op_silu(&mut self, x: u64) -> Result<u64, Trap> {
        let (xt, xsh) = self.native("silu@1", x)?;
        let out = self.backend.silu(xt);
        self.alloc_native(out, xsh)
    }

    fn op_softmax(&mut self, x: u64, dim: u32) -> Result<u64, Trap> {
        let (xt, xsh) = self.native("softmax@1", x)?;
        let dim = dim as usize;
        if dim >= xsh.len() {
            return Err(Trap::new(
                TrapCode::ShapeMismatch,
                "softmax@1",
                self.phase,
                "dim",
            ));
        }
        let dimlen = xsh[dim] as usize;
        let inner: usize = xsh[dim + 1..].iter().map(|&d| d as usize).product();
        let outer: usize = xsh[..dim].iter().map(|&d| d as usize).product();
        let out = self.backend.softmax(xt, outer, dimlen, inner);
        self.alloc_native(out, xsh)
    }

    fn op_rope(
        &mut self,
        x: u64,
        pos_start: u32,
        theta: f64,
        interleaved: u32,
    ) -> Result<u64, Trap> {
        let (xt, xsh) = self.native("rope@1", x)?;
        if xsh.len() < 2 {
            return Err(Trap::new(
                TrapCode::ShapeMismatch,
                "rope@1",
                self.phase,
                "rank",
            ));
        }
        let hd = xsh[xsh.len() - 1] as usize;
        let seq = xsh[xsh.len() - 2] as usize;
        let rows = Self::numel(&xsh) / hd;
        let out = self.backend.rope(
            xt,
            rows,
            seq,
            hd,
            pos_start as usize,
            theta,
            interleaved != 0,
        );
        self.alloc_native(out, xsh)
    }

    fn op_flash_attn(
        &mut self,
        q: u64,
        k: u64,
        v: u64,
        causal: u32,
        scale: f64,
    ) -> Result<u64, Trap> {
        let (qt, qsh) = self.native("flash_attn@1", q)?;
        let (kt, _) = self.native("flash_attn@1", k)?;
        let (vt, _) = self.native("flash_attn@1", v)?;
        if qsh.len() != 4 {
            return Err(Trap::new(
                TrapCode::ShapeMismatch,
                "flash_attn@1",
                self.phase,
                "expects [b,h,s,d]",
            ));
        }
        let (b, h, s, d) = (
            qsh[0] as usize,
            qsh[1] as usize,
            qsh[2] as usize,
            qsh[3] as usize,
        );
        let out = self
            .backend
            .flash_attn(qt, kt, vt, b * h, s, d, causal != 0, scale);
        self.alloc_native(out, qsh)
    }

    fn op_reshape(&mut self, x: u64, dims: &[u32]) -> Result<u64, Trap> {
        let (xt, xsh) = self.native("reshape@1", x)?;
        if Self::numel(&xsh) != Self::numel(dims) {
            return Err(Trap::new(
                TrapCode::ShapeMismatch,
                "reshape@1",
                self.phase,
                "numel",
            ));
        }
        // Identity data with a new shape, recorded on the tape so gradients pass through (HOST-9).
        let out = self.backend.reshape(xt);
        self.alloc_native(out, dims.to_vec())
    }

    fn op_transpose(&mut self, x: u64, d0: u32, d1: u32) -> Result<u64, Trap> {
        let (xt, xsh) = self.native("transpose@1", x)?;
        let (d0, d1) = (d0 as usize, d1 as usize);
        if d0 >= xsh.len() || d1 >= xsh.len() {
            return Err(Trap::new(
                TrapCode::ShapeMismatch,
                "transpose@1",
                self.phase,
                "axis",
            ));
        }
        let shape_in: Vec<usize> = xsh.iter().map(|&d| d as usize).collect();
        let out = self.backend.transpose(xt, &shape_in, d0, d1);
        let mut shape = xsh;
        shape.swap(d0, d1);
        self.alloc_native(out, shape)
    }

    fn op_slice(&mut self, x: u64, dim: u32, start: u32, end: u32) -> Result<u64, Trap> {
        let (xt, xsh) = self.native("slice@1", x)?;
        let (dim, start, end) = (dim as usize, start as usize, end as usize);
        if dim >= xsh.len() || start > end || end > xsh[dim] as usize {
            return Err(Trap::new(
                TrapCode::ShapeMismatch,
                "slice@1",
                self.phase,
                "range",
            ));
        }
        let shape_in: Vec<usize> = xsh.iter().map(|&d| d as usize).collect();
        let out = self.backend.slice(xt, &shape_in, dim, start, end);
        let mut shape = xsh;
        shape[dim] = (end - start) as u32;
        self.alloc_native(out, shape)
    }

    // -- Wave-2 compression natives -------------------------------------------------------------

    /// Returns `(values_handle, indices_handle)`; the caller writes the indices to the guest.
    fn op_topk_chunk(&mut self, x: u64, chunk: u32, k: u32) -> Result<(u64, u64), Trap> {
        let (xt, xsh) = self.native("topk_chunk@1", x)?;
        let numel = Self::numel(&xsh);
        let n_chunks = if chunk == 0 {
            0
        } else {
            numel / chunk as usize
        };
        let (vt, it) = self
            .backend
            .topk_chunk(xt, chunk as usize, k as usize)
            .map_err(|c| Trap::new(c, "topk_chunk@1", self.phase, "layout"))?;
        let shape = vec![n_chunks as u32, k];
        let vh = self.alloc_native(vt, shape.clone())?;
        let ih = self.alloc_native(it, shape)?;
        Ok((vh, ih))
    }

    fn op_chunk_scatter(
        &mut self,
        vals: u64,
        idx: u64,
        chunk: u32,
        dims: &[u32],
    ) -> Result<u64, Trap> {
        let (valst, _) = self.native("chunk_scatter@1", vals)?;
        let (idxt, _) = self.native("chunk_scatter@1", idx)?;
        let out = self
            .backend
            .det_chunk_scatter(valst, idxt, chunk as usize, Self::numel(dims))
            .map_err(|c| Trap::new(c, "chunk_scatter@1", self.phase, "layout"))?;
        self.alloc_native(out, dims.to_vec())
    }

    fn op_absmax_pack(&mut self, x: u64, chunk: u32, bits: u32) -> Result<u64, Trap> {
        let (xt, _) = self.native("absmax_pack@1", x)?;
        let out = self
            .backend
            .absmax_pack(xt, chunk as usize, bits)
            .map_err(|c| Trap::new(c, "absmax_pack@1", self.phase, "layout"))?;
        let n = self.backend.view(out).len() as u32;
        self.alloc_native(out, vec![n])
    }

    fn op_absmax_unpack(&mut self, packed: u64, chunk: u32, bits: u32) -> Result<u64, Trap> {
        let (pt, _) = self.native("absmax_unpack@1", packed)?;
        let out = self
            .backend
            .det_absmax_unpack(pt, chunk as usize, bits)
            .map_err(|c| Trap::new(c, "absmax_unpack@1", self.phase, "layout"))?;
        let n = self.backend.view(out).len() as u32;
        self.alloc_native(out, vec![n])
    }

    fn op_dct2(&mut self, x: u64, tile: u32, import: &'static str) -> Result<u64, Trap> {
        let (xt, xsh) = self.native(import, x)?;
        let out = self
            .backend
            .dct2(xt, tile as usize)
            .map_err(|c| Trap::new(c, import, self.phase, "tile"))?;
        self.alloc_native(out, xsh)
    }

    fn op_idct2_native(&mut self, x: u64, tile: u32) -> Result<u64, Trap> {
        let (xt, xsh) = self.native("idct2@1", x)?;
        let out = self
            .backend
            .idct2(xt, tile as usize)
            .map_err(|c| Trap::new(c, "idct2@1", self.phase, "tile"))?;
        self.alloc_native(out, xsh)
    }

    fn op_det_idct2(&mut self, x: u64, tile: u32) -> Result<u64, Trap> {
        let (xt, xsh) = self.det("det_idct2@1", x)?;
        let out = self
            .backend
            .idct2(xt, tile as usize)
            .map_err(|c| Trap::new(c, "det_idct2@1", self.phase, "tile"))?;
        self.alloc_det(out, xsh)
    }

    fn op_drop(&mut self, h: u64) -> Result<(), Trap> {
        match handle::classify(h) {
            Some(HandleClass::Step(Lane::Native)) => {
                let t = self
                    .step_native
                    .free(h)
                    .map_err(|c| Trap::new(c, "drop@1", self.phase, "native step"))?;
                self.backend.free(t);
                Ok(())
            }
            Some(HandleClass::Step(Lane::Det)) => {
                let t = self
                    .step_det
                    .free(h)
                    .map_err(|c| Trap::new(c, "drop@1", self.phase, "det step"))?;
                self.backend.free(t);
                Ok(())
            }
            Some(HandleClass::Update) => Ok(()), // container drop: freed wholesale at return
            Some(_) => Err(Trap::new(
                TrapCode::InvalidHandle,
                "drop@1",
                self.phase,
                "cannot drop a stable handle",
            )),
            None => Err(Trap::new(
                TrapCode::InvalidHandle,
                "drop@1",
                self.phase,
                "unknown handle",
            )),
        }
    }
}

/// Wire every `tabi@1` import this host implements (the Merge-1 vocabulary subset) into `linker`.
#[allow(clippy::too_many_lines)]
fn link_tabi(linker: &mut Linker<HostState>) -> Result<(), wasmtime::Error> {
    macro_rules! import {
        ($name:literal, |$c:ident $(, $a:ident : $t:ty)*| -> $ret:ty $body:block) => {
            linker.func_wrap(
                "tabi@1",
                $name,
                |mut $c: Caller<'_, HostState> $(, $a: $t)*| -> Result<$ret, wasmtime::Error> {
                    let r: Result<$ret, Trap> = (|| { $body })();
                    stash(&mut $c, r)
                },
            )?;
        };
    }

    import!("param@1", |c,
                        np: u32,
                        nl: u32,
                        dp: u32,
                        dr: u32,
                        dt: u32,
                        init: u32,
                        p0: f64,
                        p1: f64|
     -> u64 {
        enter(&mut c, "param@1")?;
        let name = read_str(&mut c, np, nl)?;
        let dims = read_dims(&mut c, dp, dr)?;
        c.data_mut().register_param(&name, &dims, dt, init, p0, p1)
    });
    import!("persistent@1", |c,
                             np: u32,
                             nl: u32,
                             dp: u32,
                             dr: u32,
                             dt: u32,
                             class: u32|
     -> u64 {
        enter(&mut c, "persistent@1")?;
        let name = read_str(&mut c, np, nl)?;
        let dims = read_dims(&mut c, dp, dr)?;
        c.data_mut().register_persistent(&name, &dims, dt, class)
    });
    import!("det_persistent@1", |c,
                                 np: u32,
                                 nl: u32,
                                 dp: u32,
                                 dr: u32,
                                 class: u32|
     -> u64 {
        enter(&mut c, "det_persistent@1")?;
        let name = read_str(&mut c, np, nl)?;
        let dims = read_dims(&mut c, dp, dr)?;
        c.data_mut().register_det_persistent(&name, &dims, class)
    });
    import!("drop@1", |c, h: u64| -> () {
        enter(&mut c, "drop@1")?;
        c.data_mut().op_drop(h)
    });
    import!("param_round_base@1", |c, p: u64| -> u64 {
        enter(&mut c, "param_round_base@1")?;
        c.data_mut().op_param_round_base(p)
    });
    import!("backward@1", |c, loss: u64| -> () {
        enter(&mut c, "backward@1")?;
        c.data_mut().op_backward(loss)
    });
    import!("grad@1", |c, p: u64| -> u64 {
        enter(&mut c, "grad@1")?;
        c.data_mut().op_grad(p)
    });
    import!("zero_grads@1", |c| -> () {
        enter(&mut c, "zero_grads@1")?;
        c.data_mut().op_zero_grads();
        Ok(())
    });
    import!("assign@1", |c, dst: u64, src: u64| -> () {
        enter(&mut c, "assign@1")?;
        c.data_mut().op_assign(dst, src)
    });
    import!("zeros@1", |c, dp: u32, dr: u32, _dt: u32| -> u64 {
        enter(&mut c, "zeros@1")?;
        let dims = read_dims(&mut c, dp, dr)?;
        c.data_mut().op_create(&dims, 0.0)
    });
    import!("ones@1", |c, dp: u32, dr: u32, _dt: u32| -> u64 {
        enter(&mut c, "ones@1")?;
        let dims = read_dims(&mut c, dp, dr)?;
        c.data_mut().op_create(&dims, 1.0)
    });
    import!("full@1", |c,
                       dp: u32,
                       dr: u32,
                       _dt: u32,
                       value: f64|
     -> u64 {
        enter(&mut c, "full@1")?;
        let dims = read_dims(&mut c, dp, dr)?;
        c.data_mut().op_create(&dims, value as f32)
    });
    import!("add@1", |c, a: u64, b: u64| -> u64 {
        enter(&mut c, "add@1")?;
        c.data_mut().op_add(a, b)
    });
    import!("sub@1", |c, a: u64, b: u64| -> u64 {
        enter(&mut c, "sub@1")?;
        c.data_mut().op_binary_same("sub@1", a, b)
    });
    import!("mul@1", |c, a: u64, b: u64| -> u64 {
        enter(&mut c, "mul@1")?;
        c.data_mut().op_binary_same("mul@1", a, b)
    });
    import!("mul_s@1", |c, x: u64, v: f64| -> u64 {
        enter(&mut c, "mul_s@1")?;
        c.data_mut().op_mul_s(x, v)
    });
    import!("matmul@1", |c, a: u64, b: u64| -> u64 {
        enter(&mut c, "matmul@1")?;
        c.data_mut().op_matmul(a, b)
    });
    import!("relu@1", |c, x: u64| -> u64 {
        enter(&mut c, "relu@1")?;
        c.data_mut().op_relu(x)
    });
    import!("cross_entropy@1", |c,
                                logits: u64,
                                targets: u64,
                                ignore: i64|
     -> u64 {
        enter(&mut c, "cross_entropy@1")?;
        c.data_mut().op_cross_entropy(logits, targets, ignore)
    });
    import!("adamw_step@1", |c,
                             p: u64,
                             g: u64,
                             m: u64,
                             v: u64,
                             step: u32,
                             lr: f64,
                             b1: f64,
                             b2: f64,
                             eps: f64,
                             wd: f64|
     -> () {
        enter(&mut c, "adamw_step@1")?;
        c.data_mut()
            .op_adamw_step(p, g, m, v, step, lr, b1, b2, eps, wd)
    });
    import!("batch_tokens@1", |c, b: u64| -> u64 {
        enter(&mut c, "batch_tokens@1")?;
        c.data_mut().op_batch_tokens(b)
    });
    import!("batch_size@1", |c, b: u64| -> u32 {
        enter(&mut c, "batch_size@1")?;
        Ok(c.data().batch_ref("batch_size@1", b)?.batch)
    });
    import!("batch_seq_len@1", |c, b: u64| -> u32 {
        enter(&mut c, "batch_seq_len@1")?;
        Ok(c.data().batch_ref("batch_seq_len@1", b)?.seq)
    });
    import!("scalar@1", |c, x: u64| -> f64 {
        enter(&mut c, "scalar@1")?;
        c.data().op_scalar(x)
    });
    import!("metric@1", |c, np: u32, nl: u32, x: u64| -> () {
        enter(&mut c, "metric@1")?;
        let name = read_str(&mut c, np, nl)?;
        c.data_mut().op_metric(&name, x)
    });
    import!("log@1", |c, _level: u32, _mp: u32, _ml: u32| -> () {
        enter(&mut c, "log@1")?;
        Ok(())
    });
    import!("abi_minor@1", |c| -> u32 {
        enter(&mut c, "abi_minor@1")?;
        Ok(crate::TENSOR_ABI_MINOR)
    });
    import!("upd_new@1", |c| -> u64 {
        enter(&mut c, "upd_new@1")?;
        Ok(c.data_mut().op_upd_new())
    });
    import!("upd_push_bytes@1", |c, u: u64, dp: u32, dl: u32| -> () {
        enter(&mut c, "upd_push_bytes@1")?;
        let data = read_bytes(&mut c, dp, dl)?;
        c.data_mut().op_upd_push_bytes(u, data)
    });
    import!("upd_push_tensor@1", |c, u: u64, x: u64| -> () {
        enter(&mut c, "upd_push_tensor@1")?;
        c.data_mut().op_upd_push_tensor(u, x)
    });
    import!("upd_sections@1", |c, i: u32| -> u32 {
        enter(&mut c, "upd_sections@1")?;
        c.data().op_upd_sections(i)
    });
    import!("upd_kind@1", |c, i: u32, s: u32| -> u32 {
        enter(&mut c, "upd_kind@1")?;
        c.data().op_upd_kind(i, s)
    });
    import!("upd_bytes_len@1", |c, i: u32, s: u32| -> u32 {
        enter(&mut c, "upd_bytes_len@1")?;
        c.data().op_upd_bytes_len(i, s)
    });
    import!("upd_read_bytes@1", |c,
                                 i: u32,
                                 s: u32,
                                 dp: u32,
                                 dl: u32|
     -> u32 {
        enter(&mut c, "upd_read_bytes@1")?;
        let bytes = c.data().op_upd_bytes(i, s)?;
        let n = (bytes.len()).min(dl as usize);
        let mem = mem_of(&mut c)?;
        mem.write(&mut c, dp as usize, &bytes[..n])
            .map_err(|_| Trap::bare(TrapCode::MemOob, "upd_read_bytes dst"))?;
        Ok(n as u32)
    });
    import!("upd_tensor@1", |c, i: u32, s: u32| -> u64 {
        enter(&mut c, "upd_tensor@1")?;
        c.data_mut().op_upd_tensor(i, s)
    });
    import!("det_zeros@1", |c, dp: u32, dr: u32| -> u64 {
        enter(&mut c, "det_zeros@1")?;
        let dims = read_dims(&mut c, dp, dr)?;
        c.data_mut().op_det_zeros(&dims)
    });
    import!("det_sum@1", |c, hp: u32, hc: u32| -> u64 {
        enter(&mut c, "det_sum@1")?;
        let handles = read_handles(&mut c, hp, hc)?;
        c.data_mut().op_det_sum(&handles)
    });
    import!("det_scale@1", |c, x: u64, alpha: f64| -> u64 {
        enter(&mut c, "det_scale@1")?;
        c.data_mut().op_det_scale(x, alpha)
    });
    import!("det_l2norm@1", |c, x: u64| -> f64 {
        enter(&mut c, "det_l2norm@1")?;
        c.data().op_det_l2norm(x)
    });
    import!("det_sign@1", |c, x: u64| -> u64 {
        enter(&mut c, "det_sign@1")?;
        c.data_mut().op_det_sign(x)
    });
    import!("det_add@1", |c, a: u64, b: u64| -> u64 {
        enter(&mut c, "det_add@1")?;
        c.data_mut().op_det_binary("det_add@1", a, b)
    });
    import!("det_sub@1", |c, a: u64, b: u64| -> u64 {
        enter(&mut c, "det_sub@1")?;
        c.data_mut().op_det_binary("det_sub@1", a, b)
    });
    import!("det_mul@1", |c, a: u64, b: u64| -> u64 {
        enter(&mut c, "det_mul@1")?;
        c.data_mut().op_det_binary("det_mul@1", a, b)
    });
    import!("det_absmax_unpack@1", |c,
                                    packed: u64,
                                    chunk: u32,
                                    bits: u32|
     -> u64 {
        enter(&mut c, "det_absmax_unpack@1")?;
        c.data_mut().op_det_absmax_unpack(packed, chunk, bits)
    });
    import!("det_chunk_scatter_add@1", |c,
                                        acc: u64,
                                        vals: u64,
                                        idx: u64,
                                        chunk: u32|
     -> () {
        enter(&mut c, "det_chunk_scatter_add@1")?;
        c.data_mut().op_det_chunk_scatter_add(acc, vals, idx, chunk)
    });
    import!("det_chunk_scatter@1", |c,
                                    vals: u64,
                                    idx: u64,
                                    chunk: u32,
                                    dp: u32,
                                    dr: u32|
     -> u64 {
        enter(&mut c, "det_chunk_scatter@1")?;
        let dims = read_dims(&mut c, dp, dr)?;
        c.data_mut().op_det_chunk_scatter(vals, idx, chunk, &dims)
    });
    import!("det_assign@1", |c, dst: u64, src: u64| -> () {
        enter(&mut c, "det_assign@1")?;
        c.data_mut().op_det_assign(dst, src)
    });
    import!("det_param@1", |c, p: u64| -> u64 {
        enter(&mut c, "det_param@1")?;
        c.data_mut().op_det_param(p)
    });
    import!("det_reset_param_to_base@1", |c, p: u64| -> () {
        enter(&mut c, "det_reset_param_to_base@1")?;
        c.data_mut().op_det_reset_param_to_base(p)
    });
    import!("det_axpy_param@1", |c, p: u64, x: u64, alpha: f64| -> () {
        enter(&mut c, "det_axpy_param@1")?;
        c.data_mut().op_det_axpy_param(p, x, alpha)
    });
    // -- Wave-2 additions -----------------------------------------------------------------------
    import!("embedding@1", |c, w: u64, ids: u64| -> u64 {
        enter(&mut c, "embedding@1")?;
        c.data_mut().op_embedding(w, ids)
    });
    import!("rmsnorm@1", |c, x: u64, w: u64, eps: f64| -> u64 {
        enter(&mut c, "rmsnorm@1")?;
        c.data_mut().op_rmsnorm(x, w, eps)
    });
    import!("softmax@1", |c, x: u64, dim: u32| -> u64 {
        enter(&mut c, "softmax@1")?;
        c.data_mut().op_softmax(x, dim)
    });
    import!("silu@1", |c, x: u64| -> u64 {
        enter(&mut c, "silu@1")?;
        c.data_mut().op_silu(x)
    });
    import!("rope@1", |c,
                       x: u64,
                       pos: u32,
                       theta: f64,
                       il: u32|
     -> u64 {
        enter(&mut c, "rope@1")?;
        c.data_mut().op_rope(x, pos, theta, il)
    });
    import!("flash_attn@1", |c,
                             q: u64,
                             k: u64,
                             v: u64,
                             causal: u32,
                             scale: f64|
     -> u64 {
        enter(&mut c, "flash_attn@1")?;
        c.data_mut().op_flash_attn(q, k, v, causal, scale)
    });
    import!("reshape@1", |c, x: u64, dp: u32, dr: u32| -> u64 {
        enter(&mut c, "reshape@1")?;
        let dims = read_dims(&mut c, dp, dr)?;
        c.data_mut().op_reshape(x, &dims)
    });
    import!("transpose@1", |c, x: u64, d0: u32, d1: u32| -> u64 {
        enter(&mut c, "transpose@1")?;
        c.data_mut().op_transpose(x, d0, d1)
    });
    import!("slice@1", |c,
                        x: u64,
                        dim: u32,
                        start: u32,
                        end: u32|
     -> u64 {
        enter(&mut c, "slice@1")?;
        c.data_mut().op_slice(x, dim, start, end)
    });
    import!("topk_chunk@1", |c,
                             x: u64,
                             chunk: u32,
                             k: u32,
                             out_idx: u32|
     -> u64 {
        enter(&mut c, "topk_chunk@1")?;
        let (vh, ih) = c.data_mut().op_topk_chunk(x, chunk, k)?;
        let mem = mem_of(&mut c)?;
        mem.write(&mut c, out_idx as usize, &ih.to_le_bytes())
            .map_err(|_| Trap::bare(TrapCode::MemOob, "topk_chunk out_idx"))?;
        Ok(vh)
    });
    import!("chunk_scatter@1", |c,
                                vals: u64,
                                idx: u64,
                                chunk: u32,
                                dp: u32,
                                dr: u32|
     -> u64 {
        enter(&mut c, "chunk_scatter@1")?;
        let dims = read_dims(&mut c, dp, dr)?;
        c.data_mut().op_chunk_scatter(vals, idx, chunk, &dims)
    });
    import!("absmax_pack@1", |c, x: u64, chunk: u32, bits: u32| -> u64 {
        enter(&mut c, "absmax_pack@1")?;
        c.data_mut().op_absmax_pack(x, chunk, bits)
    });
    import!("absmax_unpack@1", |c,
                                packed: u64,
                                chunk: u32,
                                bits: u32,
                                _dt: u32|
     -> u64 {
        enter(&mut c, "absmax_unpack@1")?;
        c.data_mut().op_absmax_unpack(packed, chunk, bits)
    });
    import!("dct2@1", |c, x: u64, tile: u32| -> u64 {
        enter(&mut c, "dct2@1")?;
        c.data_mut().op_dct2(x, tile, "dct2@1")
    });
    import!("idct2@1", |c, x: u64, tile: u32| -> u64 {
        enter(&mut c, "idct2@1")?;
        c.data_mut().op_idct2_native(x, tile)
    });
    import!("det_idct2@1", |c, x: u64, tile: u32| -> u64 {
        enter(&mut c, "det_idct2@1")?;
        c.data_mut().op_det_idct2(x, tile)
    });
    Ok(())
}
