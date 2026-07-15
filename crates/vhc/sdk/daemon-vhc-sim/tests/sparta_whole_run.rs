// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The SDK-side Phase-B acceptance: the SPARTA-shaped continuous-averaging toy run natively over
//! vhc-sim's virtual worlds (architecture §6, §9). No rounds, no coordinator — timers + gossip
//! publish only. Asserts: (1) the mesh converges to the exact global mean over a reliable trace;
//! (2) the run is bit-for-bit deterministic (two runs → identical transcript, the SDK-side analogue
//! of the host §8.7 input replay); (3) it still runs, bounded, under a lossy/churn trace.

use daemon_vhc_sim::toys::SparseAverager;
use daemon_vhc_sim::{RunLimits, Simulator, Trace, VirtualNet};

fn averagers(initials: &[f32], tick_ms: u64, ticks: u32) -> Vec<SparseAverager> {
    initials
        .iter()
        .map(|&x| SparseAverager::new(vec![x], tick_ms, ticks))
        .collect()
}

#[test]
fn reliable_mesh_converges_to_the_global_mean() {
    let initials = [1.0_f32, 4.0, 7.0];
    let global_mean = initials.iter().sum::<f32>() / initials.len() as f32; // 4.0
    let sim = Simulator::new(VirtualNet::mesh(3), Trace::reliable(10));
    let (peers, transcript) = sim.run(averagers(&initials, 50, 5), RunLimits::default());

    for (i, p) in peers.iter().enumerate() {
        let v = p.value()[0];
        assert!(
            (v - global_mean).abs() < 1e-4,
            "peer {i} converged to {v}, expected the global mean {global_mean}"
        );
    }
    // Every peer holds the same converged value (agreement, not just proximity).
    let v0 = peers[0].value()[0];
    assert!(peers.iter().all(|p| (p.value()[0] - v0).abs() < 1e-6));
    // The run actually gossiped: init publish (3) + per-tick publishes.
    assert!(
        transcript.publishes.len() >= 3,
        "expected gossip traffic, got {}",
        transcript.publishes.len()
    );
}

#[test]
fn the_whole_run_is_deterministic() {
    let initials = [2.0_f32, 5.0, 9.0, 13.0];
    let run = || {
        let sim = Simulator::new(VirtualNet::mesh(4), Trace::seeded(0xABCD, 8, 20, 0));
        sim.run(averagers(&initials, 60, 6), RunLimits::default())
    };
    let (_, a) = run();
    let (_, b) = run();
    assert_eq!(
        a.publishes, b.publishes,
        "two runs of the identical setup must produce byte-identical decision transcripts"
    );
    assert_eq!(a.events_delivered, b.events_delivered);
}

#[test]
fn runs_bounded_under_a_lossy_churn_trace() {
    let initials = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
    // 30% per-message loss + one peer offline for a mid-run window (session lapse / churn).
    let trace = Trace::seeded(0x51EED, 5, 15, 300).offline(2, 100, 300);
    let sim = Simulator::new(VirtualNet::mesh(5), trace);
    let limits = RunLimits {
        max_events: 10_000,
        horizon_ms: 5_000,
        deliver_stop: true,
    };
    let (peers, transcript) = sim.run(averagers(&initials, 40, 20), limits);
    // The gate here is liveness under adversity: the run terminates within the bound and every peer
    // holds a finite estimate (exact convergence under loss/churn is a statistical claim, not a
    // det-equality one — architecture §3.6).
    assert!(transcript.events_delivered <= 10_000);
    for p in &peers {
        assert!(p.value()[0].is_finite());
    }
}
