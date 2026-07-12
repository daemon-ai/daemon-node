// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `test-abi-basic` — an ABI-surface exerciser guest for the host runtime tests (ABI §11).
//!
//! `da_build` registers a param; `da_step`'s behavior is selected by the config `mode`:
//! - `0` normal: a native op (`add@1`) + a `metric@1` readout — the happy path.
//! - `1` phase violation: call a det op (`det_zeros@1`) in `da_step`, illegal outside ingest
//!   (ABI §3.5) ⇒ the host traps `PhaseViolation`.
//! - `2` fuel spin: a large pure-guest loop ⇒ the host traps `BudgetFuel` under a low fuel budget
//!   (ABI §8).

use daemon_train_sdk::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Default)]
struct Cfg {
    mode: u32,
}

struct TestAbi {
    w: Param,
    mode: u32,
}

impl Experiment for TestAbi {
    fn manifest(_cfg: &Config) -> Manifest {
        Manifest::new("test-abi-basic", env!("CARGO_PKG_VERSION"), 1)
    }

    fn build(cfg: &Config) -> Self {
        let cfg: Cfg = cfg.parse();
        let w = Param::new("w", &[2, 2], Dtype::F32, Init::Ones, 0.0, 0.0);
        Self { w, mode: cfg.mode }
    }

    fn step(&mut self, _batch: &Batch, _ctx: &StepCtx) {
        match self.mode {
            // Happy path: an op call + a telemetry readback.
            0 => {
                let a = Tensor::ones(&[2, 2], Dtype::F32);
                let sum = a.add(self.w.tensor());
                sum.metric("probe");
            }
            // Phase violation: a det-lane op is illegal in da_step (ABI §3.5).
            1 => {
                let _illegal = det_zeros(&[4]);
            }
            // Fuel spin: a pure-guest loop that exceeds a low fuel budget (ABI §8).
            _ => {
                let mut acc = 0_u64;
                for i in 0..5_000_000_u64 {
                    acc = acc.wrapping_add(i).wrapping_mul(3);
                }
                core::hint::black_box(acc);
            }
        }
    }

    fn inner_update(&mut self, _inner_step: u32) {}

    fn make_update(&mut self, _round: u64) -> UpdateBuilder {
        UpdateBuilder::new()
    }

    fn ingest(&mut self, _round: u64, _updates: &UpdatesView) {}
}

daemon_train_sdk::experiment!(TestAbi);
