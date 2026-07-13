// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The `JoinRun` / coordinator-attach side of the worker.
//!
//! Today this self-drives one round in-process (train × H → make_update → ingest) and streams
//! `Metric`/`RoundOutcome` — the MVP path. **B3** (Wave 3) replaces the self-driven loop with a live
//! connection to `JoinRun.coordinator`: construct a `RoundEngine` over `IrohGossip` + `R2Store` +
//! `WasmBackend` and run the real multi-round loop (see the E3/B3 ledgers).

use daemon_provision::CutWriter;
use daemon_swarm_proto::{blake3_hash, PeerId};
use daemon_swarm_run::backend::{BatchRef, StagedPayload, StepCtx, TrainerBackend};
use daemon_swarm_run::protocol::Event;
use daemon_train::{EngineConfig, WasmBackend, WasmBackendConfig, WasmBackendError};

use crate::{send, SEQ, SEQS};

/// A reserved self-peer id for the MVP self-driven round's committed set (not a real node identity).
const SELF_PEER: PeerId = PeerId([0xA1; 32]);

/// Construct the backend, emit `RunPhase{train}` (the supervisor's `join` resolves here), then
/// self-drive one round and stream `Metric`/`RoundOutcome`. Returns the live backend (kept for
/// `Throttle`). The round loop is self-driven for the MVP — connecting to `JoinRun.coordinator` is a
/// B3 Wave-3 decision (see the E3 ledger).
pub(crate) async fn join_and_run_round(
    module: &[u8],
    config: &[u8],
    run_id: &str,
    writer: &CutWriter,
) -> Result<WasmBackend, String> {
    let mut backend = WasmBackend::new(WasmBackendConfig {
        wasm: module.to_vec(),
        engine: EngineConfig::default(),
    })
    .map_err(err_detail)?;
    backend.build(config).map_err(err_detail)?;
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

    let mut last_loss = f32::NAN;
    for step in 0..steps {
        let stats = backend
            .train_step(
                &BatchRef {
                    tokens: vec![0u32; (SEQS * SEQ) as usize],
                    seq_len: SEQ,
                },
                StepCtx {
                    inner_step: step,
                    mb_index: 0,
                    mb_count: 1,
                    step_seqs: SEQS,
                },
            )
            .map_err(err_detail)?;
        last_loss = stats.loss;
        backend.inner_update(step).map_err(err_detail)?;
    }

    let payload = backend.make_update(0).map_err(err_detail)?;
    let digest = backend
        .ingest(
            0,
            &[StagedPayload {
                peer: SELF_PEER,
                hash: blake3_hash(&payload),
                bytes: payload,
            }],
        )
        .map_err(err_detail)?;

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

/// Render a backend error for an `Event::Error` detail.
fn err_detail(e: WasmBackendError) -> String {
    e.to_string()
}
