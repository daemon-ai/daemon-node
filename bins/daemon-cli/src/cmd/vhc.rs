// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The `vhc` subcommand family — the ceremony operator's drive surface over the node API.
//!
//! Every verb marshals one existing `daemon_api::ApiRequest` variant (no wire change) and renders
//! the reply, with `--json` for machine consumption. The sole exception is `identity`, which reads
//! the LOCAL vhc keystore via the `daemon-vhc-session` keystore API — no wire call, so it works
//! with the node stopped (the preflight identity-collection step). The digest fields the `detail
//! --watch` loop renders ride the additive wire-v44 `VhcRunDetail.last_round_digest` /
//! `VhcEvent::RoundOutcome.digest` surface.

use std::path::PathBuf;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use daemon_api::{ApiRequest, ApiResponse, VhcEvent, VhcPolicy, VhcPolicyMode, VhcRunDetail};
use daemon_host::ApiClient;

use crate::cli::VhcCmd;

/// Mint a per-invocation idempotency key (ADR-006). A direct CLI invocation is one operation; a
/// re-run is a new operation, so a fresh unique key per call is the correct idempotency contract.
fn mint_op_id(verb: &str, run_id: &str) -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("cli-vhc-{verb}-{run_id}-{nanos}")
}

/// Parse a `--policy` selector into a [`VhcPolicy`] (ceremony boxes participate unconditionally
/// under `always`; the node default is `idle`).
fn parse_policy(policy: Option<&str>) -> anyhow::Result<VhcPolicy> {
    let mode = match policy.map(str::trim) {
        None | Some("idle") => VhcPolicyMode::Idle,
        Some("always") => VhcPolicyMode::Always,
        Some("manual") => VhcPolicyMode::Manual,
        Some(other) => anyhow::bail!("unknown --policy {other:?} (expected always|idle|manual)"),
    };
    Ok(VhcPolicy {
        mode,
        vram_cap_mb: 0,
        duty_cycle_pct: 100,
        schedule: None,
    })
}

/// Lowercase hex of a 16-byte digest.
fn hex16(d: &[u8; 16]) -> String {
    d.iter().map(|b| format!("{b:02x}")).collect()
}

/// The last-round digest carried by the snapshot (wire v44), rendered as hex or `-`.
fn digest_str(detail: &VhcRunDetail) -> String {
    detail
        .last_round_digest
        .as_ref()
        .map_or_else(|| "-".to_string(), hex16)
}

/// The peer count for the newest round, read from the windowed recent events (the latest
/// `Progress.peers`, else the latest `RoundOutcome.committed`).
fn peers_for_latest(detail: &VhcRunDetail) -> u32 {
    let mut progress_peers = None;
    let mut committed = None;
    for e in &detail.recent_events {
        match e {
            VhcEvent::Progress { peers, .. } => progress_peers = Some(*peers),
            VhcEvent::RoundOutcome { committed: c, .. } => committed = Some(*c),
            _ => {}
        }
    }
    progress_peers.or(committed).unwrap_or(0)
}

/// Route a `vhc` subcommand.
pub(super) async fn run(client: &ApiClient, cmd: VhcCmd) -> anyhow::Result<()> {
    match cmd {
        VhcCmd::Runs { json } => runs(client, json).await,
        VhcCmd::Detail {
            run_id,
            watch,
            json,
        } => detail(client, run_id, watch, json).await,
        VhcCmd::Join {
            run_id,
            policy,
            json,
        } => {
            let op_id = mint_op_id("join", &run_id);
            let policy = parse_policy(policy.as_deref())?;
            let resp = client
                .call(ApiRequest::VhcJoin {
                    run_id: run_id.clone(),
                    policy,
                    op_id: op_id.clone(),
                })
                .await?;
            ack(&resp, json, "joined", &run_id, &op_id)
        }
        VhcCmd::Leave {
            run_id,
            immediate,
            json,
        } => {
            let op_id = mint_op_id("leave", &run_id);
            let mode = if immediate {
                daemon_api::VhcLeaveMode::Immediate
            } else {
                daemon_api::VhcLeaveMode::Graceful
            };
            let resp = client
                .call(ApiRequest::VhcLeave {
                    run_id: run_id.clone(),
                    mode,
                    op_id: op_id.clone(),
                })
                .await?;
            ack(&resp, json, "left", &run_id, &op_id)
        }
        VhcCmd::Pause { run_id, json } => {
            let op_id = mint_op_id("pause", &run_id);
            let resp = client
                .call(ApiRequest::VhcPause {
                    run_id: run_id.clone(),
                    op_id: op_id.clone(),
                })
                .await?;
            ack(&resp, json, "paused", &run_id, &op_id)
        }
        VhcCmd::Resume { run_id, json } => {
            let op_id = mint_op_id("resume", &run_id);
            let resp = client
                .call(ApiRequest::VhcResume {
                    run_id: run_id.clone(),
                    op_id: op_id.clone(),
                })
                .await?;
            ack(&resp, json, "resumed", &run_id, &op_id)
        }
        VhcCmd::Hardware { json } => hardware(client, json).await,
        VhcCmd::Disk { json } => disk(client, json).await,
        VhcCmd::Wipe {
            run_id,
            evidence,
            json,
        } => wipe(client, run_id, evidence, json).await,
        VhcCmd::Identity { state_dir, json } => identity(state_dir, json),
    }
}

async fn runs(client: &ApiClient, json: bool) -> anyhow::Result<()> {
    match client.call(ApiRequest::VhcRunList).await? {
        ApiResponse::VhcRuns(runs) => {
            if json {
                println!("{}", serde_json::to_string_pretty(&runs)?);
            } else if runs.is_empty() {
                println!("(no runs known to this node)");
            } else {
                for r in &runs {
                    let state = r.run_state.as_deref().unwrap_or("-");
                    println!(
                        "{}  phase={} round={} joined={} state={} eligible={}",
                        r.run_id, r.phase, r.last_round, r.joined, state, r.eligibility.eligible
                    );
                }
            }
            Ok(())
        }
        ApiResponse::Error(e) => anyhow::bail!("vhc runs: {e}"),
        other => anyhow::bail!("unexpected response to VhcRunList: {other:?}"),
    }
}

async fn detail(
    client: &ApiClient,
    run_id: String,
    watch: Option<u64>,
    json: bool,
) -> anyhow::Result<()> {
    match watch {
        Some(secs) if secs > 0 => loop {
            print_detail_snapshot(client, &run_id, json).await?;
            tokio::time::sleep(Duration::from_secs(secs)).await;
        },
        _ => print_detail_snapshot(client, &run_id, json).await,
    }
}

async fn print_detail_snapshot(client: &ApiClient, run_id: &str, json: bool) -> anyhow::Result<()> {
    match client
        .call(ApiRequest::VhcRunDetail {
            run_id: run_id.to_string(),
        })
        .await?
    {
        ApiResponse::VhcRunDetail(Some(detail)) => {
            if json {
                // One stable JSON object per poll: the machine-readable watch line carries the
                // digest hex explicitly so a transcript collector never re-encodes the byte array.
                let obj = serde_json::json!({
                    "run_id": run_id,
                    "phase": detail.summary.phase,
                    "round": detail.summary.last_round,
                    "last_round_digest": detail.last_round_digest.as_ref().map(hex16),
                    "peers": peers_for_latest(&detail),
                    "detail": detail,
                });
                println!("{}", serde_json::to_string(&obj)?);
            } else {
                println!(
                    "{run_id}  phase={} round={} digest={} peers={}",
                    detail.summary.phase,
                    detail.summary.last_round,
                    digest_str(&detail),
                    peers_for_latest(&detail),
                );
            }
            Ok(())
        }
        ApiResponse::VhcRunDetail(None) => {
            if json {
                println!(
                    "{}",
                    serde_json::json!({ "run_id": run_id, "known": false })
                );
                Ok(())
            } else {
                anyhow::bail!("run {run_id} is unknown to this node")
            }
        }
        ApiResponse::Error(e) => anyhow::bail!("vhc detail: {e}"),
        other => anyhow::bail!("unexpected response to VhcRunDetail: {other:?}"),
    }
}

async fn hardware(client: &ApiClient, json: bool) -> anyhow::Result<()> {
    match client.call(ApiRequest::VhcHardwareReport).await? {
        ApiResponse::VhcHardwareReport(hw) => {
            if json {
                println!("{}", serde_json::to_string_pretty(&hw)?);
            } else {
                println!(
                    "gpus={} vram_mb={} shared_mb={} ram_mb={} lanes={:?} class={} disk_free_mb={}",
                    hw.gpus,
                    hw.vram_mb,
                    hw.shared_mb,
                    hw.ram_mb,
                    hw.backend_lanes,
                    hw.throughput_class,
                    hw.disk_free_mb,
                );
            }
            Ok(())
        }
        ApiResponse::Error(e) => anyhow::bail!("vhc hardware: {e}"),
        other => anyhow::bail!("unexpected response to VhcHardwareReport: {other:?}"),
    }
}

async fn disk(client: &ApiClient, json: bool) -> anyhow::Result<()> {
    match client.call(ApiRequest::VhcDiskUsage).await? {
        ApiResponse::VhcDiskUsage(usage) => {
            if json {
                println!("{}", serde_json::to_string_pretty(&usage)?);
            } else {
                println!(
                    "root={} free_mb={} used_mb={} quota_mb={} reserve_mb={} emergency_mb={} pressure={}",
                    usage.root,
                    usage.free_mb,
                    usage.used_mb,
                    usage.quota_mb,
                    usage.reserve_mb,
                    usage.emergency_mb,
                    usage.pressure,
                );
                for s in &usage.scopes {
                    println!(
                        "  {}  recoverable_mb={} evidence_mb={} active={} scope={}",
                        s.run_id.as_deref().unwrap_or("(orphan)"),
                        s.recoverable_mb,
                        s.evidence_mb,
                        s.active,
                        s.scope,
                    );
                }
            }
            Ok(())
        }
        ApiResponse::Error(e) => anyhow::bail!("vhc disk: {e}"),
        other => anyhow::bail!("unexpected response to VhcDiskUsage: {other:?}"),
    }
}

async fn wipe(
    client: &ApiClient,
    run_id: String,
    include_evidence: bool,
    json: bool,
) -> anyhow::Result<()> {
    match client
        .call(ApiRequest::VhcDiskWipe {
            run_id: run_id.clone(),
            include_evidence,
        })
        .await?
    {
        ApiResponse::VhcDiskWipe(outcome) => {
            if json {
                println!("{}", serde_json::to_string_pretty(&outcome)?);
            } else {
                println!(
                    "wiped {}  reclaimed_mb={} evidence_wiped={} preserved={:?}",
                    outcome.run_id, outcome.reclaimed_mb, outcome.wiped_evidence, outcome.preserved,
                );
            }
            Ok(())
        }
        ApiResponse::Error(e) => anyhow::bail!("vhc wipe: {e}"),
        other => anyhow::bail!("unexpected response to VhcDiskWipe: {other:?}"),
    }
}

/// Render an `Ok`/`Error` acknowledgement for the intent verbs (join/leave/pause/resume).
fn ack(
    resp: &ApiResponse,
    json: bool,
    verb: &str,
    run_id: &str,
    op_id: &str,
) -> anyhow::Result<()> {
    match resp {
        ApiResponse::Ok => {
            if json {
                println!(
                    "{}",
                    serde_json::json!({ "run_id": run_id, "op_id": op_id, "result": verb })
                );
            } else {
                println!("{verb} {run_id} (op_id {op_id})");
            }
            Ok(())
        }
        ApiResponse::Error(e) => anyhow::bail!("vhc {verb}: {e}"),
        other => anyhow::bail!("unexpected response: {other:?}"),
    }
}

/// Read the LOCAL base-identity PeerId from the vhc keystore (no wire call). The identity dir is
/// `<state-dir>/vhc/identity` (or `$DAEMON_VHC_IDENTITY_DIR`, else `$DAEMON_DATA_DIR/vhc/identity`).
fn identity(state_dir: Option<PathBuf>, json: bool) -> anyhow::Result<()> {
    let dir = resolve_identity_dir(state_dir)?;
    let store = daemon_vhc_session::keystore::VhcKeystore::open(&dir)
        .map_err(|e| anyhow::anyhow!("open vhc keystore at {}: {e}", dir.display()))?;
    let key = store
        .base_identity()
        .map_err(|e| anyhow::anyhow!("read base identity: {e}"))?;
    let peer = daemon_vhc_proto::peer_id(&key);
    let hex = peer.to_hex();
    if json {
        println!(
            "{}",
            serde_json::json!({ "identity_dir": dir.display().to_string(), "base_identity": hex })
        );
    } else {
        println!("{hex}");
    }
    Ok(())
}

/// Resolve the vhc identity keystore directory (mirrors the node's `<data_dir>/vhc/identity`).
fn resolve_identity_dir(state_dir: Option<PathBuf>) -> anyhow::Result<PathBuf> {
    if let Some(sd) = state_dir {
        return Ok(sd.join("vhc").join("identity"));
    }
    if let Some(dir) = std::env::var_os("DAEMON_VHC_IDENTITY_DIR") {
        return Ok(PathBuf::from(dir));
    }
    if let Some(data) = std::env::var_os("DAEMON_DATA_DIR") {
        return Ok(PathBuf::from(data).join("vhc").join("identity"));
    }
    anyhow::bail!(
        "cannot resolve the vhc identity dir — pass --state-dir <node data dir> (or set \
         DAEMON_VHC_IDENTITY_DIR / DAEMON_DATA_DIR)"
    )
}
