// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `swarm-local` — the local-mode swarm runner (spec §10.1; B3).
//!
//! Drives an in-process N-peer swarm through the full round protocol against the real
//! `daemon-swarm-coordinator` `tick` loop, over a **selectable transport** — the in-process
//! `LoopbackGossip` (`--transport loopback`, the default) or a **real per-node `IrohGossip` mesh**
//! (`--transport iroh`) — with a filesystem payload store (`--store fs`). It prints the agreed
//! per-round digest transcript so a human can eyeball a run or wire it into `just swarm-dev`
//! (W1's proposed recipe). The deterministic `StubBackend` is used so a run is reproducible without
//! a GPU / guest build; the worker-backed backend rides the `daemon-train-worker` binary path.
//!
//! Usage:
//! ```text
//! swarm-local [--transport loopback|iroh] [--store fs] [--peers N] [--rounds N] [--relay <url>]
//! ```
//!
//! This bin requires the `iroh` feature (so it can offer both transports from one binary).

use std::process::ExitCode;

use daemon_swarm_run::harness::{run_swarm, SwarmConfig};
use daemon_swarm_run::live_harness::{run_live_swarm, LiveSwarmConfig};

/// Parsed CLI options.
struct Opts {
    transport: Transport,
    peers: usize,
    rounds: u64,
    relay: Option<String>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Transport {
    Loopback,
    Iroh,
}

fn parse_args() -> Result<Opts, String> {
    let mut transport = Transport::Loopback;
    let mut peers = 3usize;
    let mut rounds = 10u64;
    let mut relay: Option<String> = None;
    let mut store = "fs".to_string();

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--transport" => {
                transport = match args.next().as_deref() {
                    Some("loopback") => Transport::Loopback,
                    Some("iroh") => Transport::Iroh,
                    other => {
                        return Err(format!("--transport expects loopback|iroh, got {other:?}"))
                    }
                }
            }
            "--store" => {
                store = args
                    .next()
                    .ok_or_else(|| "--store expects a value (fs)".to_string())?;
            }
            "--peers" => {
                peers = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or_else(|| "--peers expects an integer".to_string())?;
            }
            "--rounds" => {
                rounds = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or_else(|| "--rounds expects an integer".to_string())?;
            }
            "--relay" => {
                relay = Some(
                    args.next()
                        .ok_or_else(|| "--relay expects a url".to_string())?,
                );
            }
            "-h" | "--help" => return Err("help".to_string()),
            other => return Err(format!("unknown arg: {other}")),
        }
    }
    if store != "fs" {
        // `--store r2` (presigned R2 via B1's `R2Store` over BC's `apps/swarm` presign endpoint) is a
        // recorded follow-on — it needs BC's live endpoint (or wrangler-dev). The exit gate uses the
        // filesystem-backed `PayloadStore` (a shared object store).
        return Err(format!(
            "--store {store} not supported yet (only `fs`; r2 is a BC follow-on)"
        ));
    }
    Ok(Opts {
        transport,
        peers,
        rounds,
        relay,
    })
}

#[tokio::main]
async fn main() -> ExitCode {
    let opts = match parse_args() {
        Ok(o) => o,
        Err(msg) => {
            if msg == "help" {
                eprintln!(
                    "swarm-local [--transport loopback|iroh] [--store fs] [--peers N] [--rounds N] [--relay <url>]"
                );
                return ExitCode::SUCCESS;
            }
            eprintln!("swarm-local: {msg}");
            return ExitCode::from(2);
        }
    };

    let base = SwarmConfig {
        num_peers: opts.peers,
        num_rounds: opts.rounds,
        ..SwarmConfig::small(opts.rounds)
    };

    let transport = match opts.transport {
        Transport::Loopback => "loopback",
        Transport::Iroh => "iroh",
    };
    println!(
        "swarm-local: transport={transport} store=fs peers={} rounds={}",
        opts.peers, opts.rounds
    );

    let run = match opts.transport {
        Transport::Loopback => run_swarm(base).await,
        Transport::Iroh => {
            let mut cfg = LiveSwarmConfig::new(base);
            if let Some(url) = opts.relay {
                cfg = cfg.with_relay(url);
            }
            run_live_swarm(cfg).await
        }
    };

    let run = match run {
        Ok(r) => r,
        Err(e) => {
            eprintln!("swarm-local: run failed: {e}");
            return ExitCode::FAILURE;
        }
    };

    let by_round = run.digests_by_round();
    for (round, digest) in run.agreed_transcript() {
        let peers = by_round
            .get(&round)
            .map_or(0, std::collections::BTreeMap::len);
        println!(
            "round {round:>3}: {} peers  digest={}",
            peers,
            digest.to_hex()
        );
    }
    if run.all_agree() && run.left_peers().is_empty() {
        println!(
            "swarm-local: OK — {} rounds, all peers agree every round",
            by_round.len()
        );
        ExitCode::SUCCESS
    } else {
        eprintln!(
            "swarm-local: DIVERGENCE — all_agree={}, left_peers={:?}",
            run.all_agree(),
            run.left_peers()
        );
        ExitCode::FAILURE
    }
}
