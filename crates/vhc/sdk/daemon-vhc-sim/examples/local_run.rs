// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `local_run` — the local-mode vhc developer runnable, on the v2 (vhc-sim) substrate.
//!
//! Runs N native SPARTA-shaped averager peers (`daemon_vhc_sim::toys::SparseAverager`) over the
//! virtual worlds — a whole run of timer-driven gossip that converges to the exact global mean —
//! and prints the per-peer transcript. This is the successor to the retired local-mode dev runner (`--backend
//! stub` dev runner: no native/stub consensus path survives as a standalone binary; the local-run
//! developer experience lives here as a vhc-sim example on the v2 substrate (the SPARTA whole run
//! whose wasm twin runs under the host testkit).
//!
//! Usage: `cargo run -p daemon-vhc-sim --example local_run -- [PEERS] [TICKS]`.

use daemon_vhc_sim::toys::SparseAverager;
use daemon_vhc_sim::{RunLimits, Simulator, Trace, VirtualNet};

fn main() {
    let mut args = std::env::args().skip(1);
    let peers: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(3);
    let ticks: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(6);
    let peers = peers.max(1);

    // Each peer opens at a distinct value; the reliable mesh converges every peer to the global
    // mean (the deterministic whole-run oracle, architecture §6/§9).
    let initials: Vec<f32> = (0..peers).map(|i| (i as f32) + 1.0).collect();
    let global_mean = initials.iter().sum::<f32>() / peers as f32;
    let modules: Vec<SparseAverager> = initials
        .iter()
        .map(|&x| SparseAverager::new(vec![x], 50, ticks))
        .collect();

    let sim = Simulator::new(VirtualNet::mesh(peers), Trace::reliable(10));
    let (final_peers, transcript) = sim.run(modules, RunLimits::default());

    println!("local vhc — {peers} peers, {ticks} mixing ticks (SPARTA averager, v2 substrate)");
    println!("  opening values : {initials:?}");
    for (i, p) in final_peers.iter().enumerate() {
        println!("  peer {i:>2} converged : {:.6}", p.value()[0]);
    }
    println!("  global mean    : {global_mean:.6}");
    println!(
        "  gossip publishes: {}  (events delivered: {})",
        transcript.publishes.len(),
        transcript.events_delivered
    );
    let agreed = final_peers
        .iter()
        .all(|p| (p.value()[0] - final_peers[0].value()[0]).abs() < 1e-6);
    println!("  all peers agree : {agreed}");
}
