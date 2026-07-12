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
}

impl HostState {
    fn new(cfg: &EngineConfig) -> Self {
        Self {
            phase: None,
            backend: Box::new(CpuBackend::new()),
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
        }
    }

    fn charge_op(&mut self, import: &'static str, phase: Phase) -> Result<(), Trap> {
        self.op_calls += 1;
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
        Ok(())
    }

    fn finish_entry(&mut self, result: Result<(), wasmtime::Error>) -> Result<(), TrainError> {
        // Free step handles wholesale at return (ABI §3.3), regardless of outcome.
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
        self.backend.backward(lt);
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
    Ok(())
}
