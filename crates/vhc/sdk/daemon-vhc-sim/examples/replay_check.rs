// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `replay_check` — the recorded-run re-derivation developer runnable, on the v2 (vhc-sim)
//! substrate.
//!
//! Runs the identical whole-run setup twice and confirms the two decision transcripts are
//! byte-identical — the SDK-side analogue of the host's §8.7 input replay and the successor to the
//! retired `replay` dev runner ("anyone can re-derive the run"). The host-side re-derivation
//! of a *recorded production run* through the sandboxed coordinator module stays a library
//! capability (`daemon_vhc_session::harness::verify_observe_dir`, exercised by the observe gate);
//! this example is the fast native equivalent for iterating on policy code.
//!
//! Exits non-zero on any divergence. Usage:
//! `cargo run -p daemon-vhc-sim --example replay_check -- [PEERS] [TICKS]`.

use std::process::ExitCode;

use daemon_vhc_sim::toys::SparseAverager;
use daemon_vhc_sim::{RunLimits, RunTranscript, Simulator, Trace, VirtualNet};

fn run(peers: usize, ticks: u32) -> RunTranscript {
    let modules: Vec<SparseAverager> = (0..peers)
        .map(|i| SparseAverager::new(vec![(i as f32) + 1.0], 60, ticks))
        .collect();
    // A seeded lossy/churn trace: determinism must hold even off the reliable happy path.
    let trace = Trace::seeded(0xABCD, 8, 20, 100);
    let (_, transcript) =
        Simulator::new(VirtualNet::mesh(peers), trace).run(modules, RunLimits::default());
    transcript
}

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let peers: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(4).max(1);
    let ticks: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(8);

    let a = run(peers, ticks);
    let b = run(peers, ticks);

    println!("replay check — {peers} peers, {ticks} ticks (v2 substrate)");
    println!(
        "  run A: {} publishes, {} events; run B: {} publishes, {} events",
        a.publishes.len(),
        a.events_delivered,
        b.publishes.len(),
        b.events_delivered
    );
    if a.publishes == b.publishes && a.events_delivered == b.events_delivered {
        println!("  OK — the run re-derives byte-identically (deterministic transcript)");
        ExitCode::SUCCESS
    } else {
        eprintln!("  DIVERGED — the two runs produced different transcripts");
        ExitCode::FAILURE
    }
}
