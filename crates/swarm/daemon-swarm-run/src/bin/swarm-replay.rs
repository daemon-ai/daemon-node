// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `swarm-replay` — the recorded-run verification entry point (spec §6.4 I1, §14; TDD PROTO-20; B2).
//!
//! Reads the two `daemon-swarm-observe` artifacts a `swarm-local --observe <dir>` (or a live gate
//! run) wrote: `<run>.dsmlog` (the node-visible signed
//! [`MessageLog`](daemon_swarm_observe::MessageLog)) and `<run>.dsmcap` (the coordinator's
//! reproducible `tick` [`RunCapture`](daemon_swarm_observe::RunCapture)). It re-runs the **pure**
//! `daemon-swarm-coordinator` `tick` from the captured genesis and asserts every wire-recorded
//! `RoundRecord` re-derives byte-identically (the run's per-round consensus / digest is
//! reproducible), prints the per-round health projection, and exits non-zero on any divergence —
//! the gate-ceremony "anyone can re-derive the coordinator" check.
//!
//! Usage: `swarm-replay <dir>` (verifies the single recorded run in `dir`). Requires `harness`.

use std::process::ExitCode;

use daemon_swarm_run::harness::verify_observe_dir;

fn main() -> ExitCode {
    let Some(dir) = std::env::args().nth(1) else {
        eprintln!("usage: swarm-replay <observe-dir>");
        return ExitCode::from(2);
    };
    if dir == "-h" || dir == "--help" {
        eprintln!(
            "swarm-replay <observe-dir>  # verify a recorded run's per-round consensus re-derives"
        );
        return ExitCode::SUCCESS;
    }

    let report = match verify_observe_dir(std::path::Path::new(&dir)) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("swarm-replay: {e}");
            return ExitCode::FAILURE;
        }
    };

    println!(
        "swarm-replay: run={} rounds_verified={}/{}",
        report.run_id, report.rounds_verified, report.logged_records
    );
    for rh in &report.health.rounds {
        println!(
            "  round {:>3}: committed={} attested={} finalized={} digest_agreed={}",
            rh.round, rh.committed, rh.attested_coverage, rh.finalized, rh.digest_agreed
        );
    }

    if report.all_verified() {
        println!(
            "swarm-replay: OK — all {} recorded round records re-derived byte-identically",
            report.logged_records
        );
        ExitCode::SUCCESS
    } else {
        eprintln!(
            "swarm-replay: FAILED — {}/{} records re-derived",
            report.rounds_verified, report.logged_records
        );
        ExitCode::FAILURE
    }
}
