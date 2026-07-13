// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! [`WasmBackend`] — a [`daemon_swarm_run::backend::TrainerBackend`] over the wasm host runtime.
//!
//! This is the E↔R wiring (spec §5.1/§10.2): the participant [`RoundEngine`] (lane R) drives the
//! round structure over the engine-agnostic `TrainerBackend` seam; `WasmBackend` fills in the math
//! by driving a real wasm experiment module through the [`crate::Worker`] host runtime. Nothing
//! wasm/burn leaks across the trait — payloads and checkpoints cross it as opaque bytes.
//!
//! Lifecycle mapping (ABI §2.3): `build` → `da_abi` gate + `da_manifest` + `da_build`; `train_step`
//! → `register_batch` + `da_step`; `inner_update` → `da_inner_update`; `make_update` →
//! `da_make_update` sealed to canonical CBOR ([`crate::Instance::update_bytes`]); `ingest` → the
//! record-ordered payloads staged through the `upd_*` ABI + `da_ingest_updates`
//! ([`crate::Instance::ingest_payloads`]); `checkpoint_save`/`load` → the blake3-tagged full state
//! dict ([`crate::Instance::checkpoint_bytes`] / [`crate::Instance::restore_checkpoint`]).
//!
//! **Digest**: the post-ingest state digest is [`crate::Instance::canonical_state_bytes`] (params +
//! replicated persistents) fed to `daemon_swarm_proto::digest_state` (seed-keyed xxh3-128, seeded
//! by the round, full sampling) — the frozen proto digest, not a re-derived one. Two `WasmBackend`s
//! over the same module + config + batches + staged set are **bit-identical** every round (ABI §7 /
//! the MVP's core claim; see `tests/wasm_backend_determinism.rs`).
//!
//! **Preemption-as-churn** (§10.5, T3): [`WasmBackend::pause`] checkpoints then drops the instance
//! (releasing wasm/GPU memory, keeping the CPU checkpoint); [`WasmBackend::resume`] re-instantiates
//! from the `InstancePre`, re-runs `da_build`, and restores — bit-identical to the pre-pause state.
//!
//! [`RoundEngine`]: daemon_swarm_run::engine::RoundEngine

use daemon_swarm_proto::{digest_state, Seed};
use daemon_swarm_run::backend::{
    AssessMeta, Assessment, BatchRef, StagedPayload, StateDigest, StepCtx, StepStats,
    TrainerBackend,
};
use daemon_swarm_run::seam::RoundId;

use crate::autotune::{Autotune, DeviceLimits, DEFAULT_MAX_MICROBATCH};
use crate::runtime::{EngineConfig, Instance, LoadedModule, Manifest, Worker};
use crate::TrainError;

/// Construction inputs for a [`WasmBackend`]: the experiment module bytes + the host engine profile.
///
/// The runner hands the `.wasm` bytes (from the envelope's artifact) and the per-call budgets; the
/// `[experiment.config]` CBOR arrives later via [`TrainerBackend::build`].
#[derive(Clone, Debug)]
pub struct WasmBackendConfig {
    /// The experiment module bytes (a `wasm32-unknown-unknown` `cdylib`).
    pub wasm: Vec<u8>,
    /// The host engine profile (fuel / epoch / memory / op budgets, ABI §8).
    pub engine: EngineConfig,
}

/// Errors surfaced by [`WasmBackend`].
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum WasmBackendError {
    /// A host-runtime error (trap / sandbox / engine).
    #[error("{0}")]
    Train(#[from] TrainError),
    /// A training / checkpoint entry point was called before `build` (no live instance + no config).
    #[error("wasm backend used before build()")]
    NotBuilt,
}

/// A [`TrainerBackend`] that drives a real wasm experiment module through the [`Worker`] host.
pub struct WasmBackend {
    worker: Worker,
    module: LoadedModule,
    instance: Option<Instance>,
    /// The `[experiment.config]` bytes (held for churn re-build after a `pause`).
    config: Option<Vec<u8>>,
    /// The VRAM/RAM resource model captured at `build` (G2 autotune, from the param layout). The
    /// worker's meta path carries the higher-fidelity `Autotune::from_meta` (real `act_bytes_est`).
    autotune: Option<Autotune>,
    /// The checkpoint captured by [`WasmBackend::pause`] (restored on `resume`).
    paused_state: Option<Vec<u8>>,
}

impl WasmBackend {
    /// Load the module + build the host engine (no instance yet — [`TrainerBackend::build`]
    /// instantiates once the config arrives).
    ///
    /// # Errors
    ///
    /// [`WasmBackendError::Train`] if the engine cannot be configured or the module fails to
    /// validate / compile / link.
    pub fn new(cfg: WasmBackendConfig) -> Result<Self, WasmBackendError> {
        let worker = Worker::new(cfg.engine)?;
        let module = worker.load_module(&cfg.wasm)?;
        Ok(Self {
            worker,
            module,
            instance: None,
            config: None,
            autotune: None,
            paused_state: None,
        })
    }

    fn inst_mut(&mut self) -> Result<&mut Instance, WasmBackendError> {
        self.instance.as_mut().ok_or(WasmBackendError::NotBuilt)
    }

    /// Instantiate + `da_build` from the stored config (used by `build`/`resume`/`checkpoint_load`).
    fn fresh_instance(&self) -> Result<Instance, WasmBackendError> {
        let config = self.config.as_deref().ok_or(WasmBackendError::NotBuilt)?;
        let mut inst = self.worker.instantiate(&self.module)?;
        inst.build(config)?;
        Ok(inst)
    }

    /// The module manifest (cadence + round modes) for the built config.
    ///
    /// # Errors
    ///
    /// [`WasmBackendError::NotBuilt`] before `build`; [`WasmBackendError::Train`] on a call failure.
    pub fn manifest(&mut self) -> Result<Manifest, WasmBackendError> {
        let config = self.config.clone().ok_or(WasmBackendError::NotBuilt)?;
        Ok(self.inst_mut()?.manifest(&config)?)
    }

    /// The inner-step cadence `H` (`da_manifest.steps_per_round`) the round loop paces.
    ///
    /// # Errors
    ///
    /// As [`WasmBackend::manifest`].
    pub fn steps_per_round(&mut self) -> Result<u32, WasmBackendError> {
        Ok(self.manifest()?.steps_per_round)
    }

    /// Preemption-as-churn (§10.5): checkpoint the live state, then drop the wasm instance (release
    /// its linear memory / GPU allocations), keeping only the CPU-side checkpoint. Idempotent.
    ///
    /// # Errors
    ///
    /// Never fails once built; returns [`WasmBackendError::NotBuilt`] if called before `build`.
    pub fn pause(&mut self) -> Result<(), WasmBackendError> {
        if let Some(inst) = &self.instance {
            self.paused_state = Some(inst.checkpoint_bytes());
            self.instance = None;
        } else if self.config.is_none() {
            return Err(WasmBackendError::NotBuilt);
        }
        Ok(())
    }

    /// Resume after a [`WasmBackend::pause`]: re-instantiate from the `InstancePre`, re-run
    /// `da_build` (deterministic under T3), and restore the paused checkpoint bit-exactly. A no-op
    /// if an instance is already live.
    ///
    /// # Errors
    ///
    /// [`WasmBackendError::NotBuilt`] if never built; [`WasmBackendError::Train`] on a
    /// re-instantiation / restore failure.
    pub fn resume(&mut self) -> Result<(), WasmBackendError> {
        if self.instance.is_some() {
            return Ok(());
        }
        let mut inst = self.fresh_instance()?;
        if let Some(saved) = &self.paused_state {
            inst.restore_checkpoint(saved)?;
        }
        self.instance = Some(inst);
        Ok(())
    }

    /// The post-ingest round state digest: the canonical state (params + replicated persistents)
    /// through the frozen proto digest schedule, seed-keyed by the round with full sampling so it is
    /// a bit-exact function of the whole canonical state (equal across peers, ABI §7).
    fn digest_of(inst: &Instance, round: RoundId) -> StateDigest {
        let mut seed = [0u8; 32];
        seed[..8].copy_from_slice(&round.to_le_bytes());
        let state = inst.canonical_state_bytes();
        // block_size 64, sample_count = all blocks (u32::MAX min num_blocks) → a full digest.
        let d = digest_state(&Seed(seed), 64, u32::MAX, &state);
        StateDigest(*d.as_bytes())
    }
}

impl TrainerBackend for WasmBackend {
    type Error = WasmBackendError;

    fn build(&mut self, config: &[u8]) -> Result<(), Self::Error> {
        self.config = Some(config.to_vec());
        let inst = self.fresh_instance()?;
        // Capture the resource model from the registered param layout (G2 autotune). The activation
        // term is a coarse proxy here (no meta pass at build); the worker's assess path runs a real
        // meta pass and uses `Autotune::from_meta` with the measured `act_bytes_est`.
        let params: Vec<(Vec<u32>, u32)> = inst
            .params()
            .iter()
            .map(|p| (p.shape.clone(), p.dtype))
            .collect();
        self.autotune = Some(Autotune::from_params(&params));
        self.instance = Some(inst);
        self.paused_state = None;
        Ok(())
    }

    fn assess(&self, meta: &AssessMeta) -> Result<Assessment, Self::Error> {
        let autotune = self.autotune.clone().unwrap_or_default();
        // The node's effective resources are the VRAM/RAM budget (governor policy caps applied,
        // §10.5). `max_alloc_mb = 0` = unbounded here (the worker's meta+probe path supplies the
        // wgpu-queryable per-allocation ceiling); this trait path budgets on total resources.
        // This trait path budgets on the node's total effective resources (governor caps already
        // applied); it is not device-type aware, so it keeps the discrete (independent VRAM/RAM)
        // interpretation — `shared_mb = 0`, `unified = false`. The worker's meta+probe path
        // (`daemon-train-worker/backend.rs`) supplies the real unified-memory limits.
        let limits = DeviceLimits {
            vram_mb: meta.effective_vram_mb,
            ram_mb: meta.effective_ram_mb,
            max_alloc_mb: 0,
            ..Default::default()
        };
        let v = autotune.verdict(&limits, DEFAULT_MAX_MICROBATCH);
        Ok(Assessment {
            eligible: v.eligible,
            reasons: v.reasons,
            vram_mb_estimate: v.vram_mb_estimate,
            ram_mb_estimate: v.ram_mb_estimate,
            payload_bytes_estimate: v.payload_bytes_estimate,
        })
    }

    fn train_step(&mut self, batch: &BatchRef, ctx: StepCtx) -> Result<StepStats, Self::Error> {
        let seq = batch.seq_len.max(1);
        let sequences = (batch.tokens.len() as u32 / seq).max(1);
        let inst = self.inst_mut()?;
        let handle = inst.register_batch(batch.tokens.clone(), sequences, batch.seq_len);
        inst.step(
            handle,
            ctx.inner_step,
            ctx.mb_index,
            ctx.mb_count,
            ctx.step_seqs,
        )?;
        let loss = inst
            .metrics()
            .into_iter()
            .rev()
            .find(|(name, _)| name == "loss")
            .map_or(f32::NAN, |(_, v)| v);
        Ok(StepStats { loss })
    }

    fn inner_update(&mut self, inner_step: u32) -> Result<(), Self::Error> {
        self.inst_mut()?.inner_update(inner_step)?;
        Ok(())
    }

    fn make_update(&mut self, round: RoundId) -> Result<Vec<u8>, Self::Error> {
        let inst = self.inst_mut()?;
        let container = inst.make_update(round)?;
        Ok(inst.update_bytes(container)?)
    }

    fn ingest(
        &mut self,
        round: RoundId,
        staged: &[StagedPayload],
    ) -> Result<StateDigest, Self::Error> {
        let payloads: Vec<Vec<u8>> = staged.iter().map(|p| p.bytes.clone()).collect();
        let inst = self.inst_mut()?;
        inst.ingest_payloads(round, &payloads)?;
        Ok(Self::digest_of(inst, round))
    }

    fn checkpoint_save(&self) -> Result<Vec<u8>, Self::Error> {
        Ok(self
            .instance
            .as_ref()
            .ok_or(WasmBackendError::NotBuilt)?
            .checkpoint_bytes())
    }

    fn checkpoint_load(&mut self, bytes: &[u8]) -> Result<(), Self::Error> {
        // The checkpoint restores onto the built registration, so `build(config)` must have
        // established it first (a rejoiner builds from the envelope config, then loads). If the
        // instance was dropped (paused) but the config is known, re-instantiate + build first.
        if self.instance.is_none() {
            self.instance = Some(self.fresh_instance()?);
        }
        self.inst_mut()?.restore_checkpoint(bytes)?;
        Ok(())
    }
}
