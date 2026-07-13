// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The `JoinRun` / coordinator-attach side of the worker (B3).
//!
//! Drives one representative round in-process (train × H → make_update → ingest) and streams
//! `Metric`/`RoundOutcome`, now with the **Wave-3 worker lifecycle glue** the brief calls for:
//!
//! - **micro-batch from the autotune verdict** — the node's `Eligibility.headroom["micro_batch"]`
//!   (G2's `Autotune`) is threaded in and consumed in-process (reported as a `Metric`), and seeds the
//!   OOM-probe ladder start.
//! - **live OOM trial** — the round runs inside a `daemon_train::autotune::probe_microbatch`-style
//!   halving ladder (§10.5): on a `BudgetMemory` trap the worker churns the instance (releasing its
//!   memory) and retries at half the micro-batch until it fits or hits the floor. The `wasmtime`
//!   memory trap → `TrapCode::BudgetMemory` mapping (`runtime.rs`) makes this a **real** catch, not a
//!   simulation; `oom_error_class()` names the class.
//!
//! The self-driven representative round runs at the **budget-sized** shape (`SEQS`) the meta pass
//! sized the instance for, so it stays within the op/mem budget (a full-scale driven micro-batch
//! needs the instance budgets re-sized — a `backend.rs`/G2 follow-on, recorded in the B3 ledger).
//!
//! The full **live multi-round attach** — construct a `RoundEngine` over `IrohGossip` + `R2Store`
//! inside the worker subprocess and run the coordinator loop — is proven end to end by
//! `daemon-swarm-run::live_harness` (which builds exactly that `RoundEngine`-over-live-transport
//! wiring) and the `daemon-swarm-e2e` live gate; wiring it into the worker *subprocess* (adding the
//! iroh/QUIC tree to the `daemon-train` binary + the join credentials plumbing) is the recorded B3
//! remainder (see the ledger "Deviations").

use daemon_provision::CutWriter;
use daemon_swarm_proto::{blake3_hash, PeerId};
use daemon_swarm_run::backend::{BatchRef, StagedPayload, StateDigest, StepCtx, TrainerBackend};
use daemon_swarm_run::protocol::Event;
use daemon_train::autotune::oom_error_class;
use daemon_train::{
    EngineConfig, TrainError, TrapCode, WasmBackend, WasmBackendConfig, WasmBackendError,
};

use crate::{send, SEQ, SEQS};

/// A reserved self-peer id for the MVP self-driven round's committed set (not a real node identity).
const SELF_PEER: PeerId = PeerId([0xA1; 32]);

/// Construct the backend, emit `RunPhase{train}` (the supervisor's `join` resolves here), consume the
/// autotune `micro_batch`, then self-drive one round through the §10.5 OOM-halving ladder and stream
/// `Metric`/`RoundOutcome`. Returns the live backend (kept for `Throttle`).
pub(crate) async fn join_and_run_round(
    module: &[u8],
    config: &[u8],
    run_id: &str,
    assessed_micro_batch: u32,
    writer: &CutWriter,
) -> Result<WasmBackend, String> {
    let mut backend = build_backend(module, config)?;
    let steps = backend.steps_per_round().map_err(err_detail)?;

    send(
        writer,
        &Event::RunPhase {
            run_id: run_id.to_string(),
            phase: "train".to_string(),
            epoch: 0,
            round: 0,
        },
    )
    .await;

    // Consume the node's autotune verdict in-process (G2's Eligibility.headroom["micro_batch"]): it
    // seeds the OOM-probe ladder start + the driven shape (clamped below). Logged (not a new protocol
    // event — the frozen worker `Event` stream, §10.2, is pinned by `worker_protocol`).
    eprintln!("daemon-train-worker: autotune micro_batch={assessed_micro_batch} (§10.5 verdict)");

    // §10.5 live OOM ladder (mirrors `daemon_train::autotune::probe_microbatch`): run the round at
    // `mb`; on a real `BudgetMemory` trap, churn the instance (fresh instance releases memory) and
    // retry at half until it fits or hits the floor 1. The driven shape is budget-sized (SEQS) so the
    // ladder never fires for tiny-llama, but the recovery seam G2 left mechanical is now wired.
    let mut mb = assessed_micro_batch.clamp(1, SEQS);
    let mut halvings: u32 = 0;
    let (last_loss, digest) = loop {
        match run_round(&mut backend, steps, mb) {
            Ok(out) => break out,
            Err(e) if is_oom(&e) && mb > 1 => {
                let next = mb / 2;
                eprintln!(
                    "daemon-train-worker: {:?} at micro_batch={mb}; halving to {next} and re-probing (§10.5)",
                    oom_error_class()
                );
                halvings += 1;
                // Churn: a fresh instance releases the OOMing instance's memory (WasmBackend::pause
                // semantics), then re-`build` deterministically before the retry.
                backend = build_backend(module, config)?;
                mb = next;
            }
            Err(e) => return Err(err_detail(e)),
        }
    };
    if halvings > 0 {
        eprintln!(
            "daemon-train-worker: recovered after {halvings} OOM halving(s); micro_batch={mb}"
        );
    }

    send(
        writer,
        &Event::Metric {
            name: "loss".to_string(),
            value: f64::from(last_loss),
        },
    )
    .await;
    send(
        writer,
        &Event::RoundOutcome {
            round: 0,
            committed: 1,
            ingested: 1,
            stalled: false,
            digest: *digest.as_bytes(),
        },
    )
    .await;

    Ok(backend)
}

/// Build + `da_build` a fresh [`WasmBackend`] from the module + config (also the OOM-churn rebuild).
fn build_backend(module: &[u8], config: &[u8]) -> Result<WasmBackend, String> {
    let mut backend = WasmBackend::new(WasmBackendConfig {
        wasm: module.to_vec(),
        engine: EngineConfig::default(),
    })
    .map_err(err_detail)?;
    backend.build(config).map_err(err_detail)?;
    Ok(backend)
}

/// Run one representative round at `mb` sequences/step: `steps` inner steps → `make_update` →
/// self-`ingest`, returning `(last_loss, post-ingest digest)`. Any [`WasmBackendError`] (incl. a
/// `BudgetMemory` trap the OOM ladder catches) propagates.
fn run_round(
    backend: &mut WasmBackend,
    steps: u32,
    mb: u32,
) -> Result<(f32, StateDigest), WasmBackendError> {
    let mb = mb.max(1);
    let mut last_loss = f32::NAN;
    for step in 0..steps {
        let stats = backend.train_step(
            &BatchRef {
                tokens: vec![0u32; (mb * SEQ) as usize],
                seq_len: SEQ,
            },
            StepCtx {
                inner_step: step,
                mb_index: 0,
                mb_count: 1,
                step_seqs: mb,
            },
        )?;
        last_loss = stats.loss;
        backend.inner_update(step)?;
    }
    let payload = backend.make_update(0)?;
    let digest = backend.ingest(
        0,
        &[StagedPayload {
            peer: SELF_PEER,
            hash: blake3_hash(&payload),
            bytes: payload,
        }],
    )?;
    Ok((last_loss, digest))
}

/// Whether a backend error is an out-of-memory trap (`TrapCode::BudgetMemory`) — the §10.5 recovery
/// trigger the OOM ladder halves on. Other errors (module traps, budgets) are hard failures.
fn is_oom(e: &WasmBackendError) -> bool {
    matches!(
        e,
        WasmBackendError::Train(TrainError::Trap(t)) if t.code == TrapCode::BudgetMemory
    )
}

/// Render a backend error for an `Event::Error` detail.
fn err_detail(e: WasmBackendError) -> String {
    e.to_string()
}
