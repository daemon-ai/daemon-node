// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The `pair` subcommand: the LAN pairing admin surface (pairing spec §5.5). `new` arms and
//! prints the one-and-only exposure of the code (grouped for typing) plus the join URI (the QR
//! payload the GUI/TUI arming views render); the rest inspect and manage the state. All five
//! verbs are admin-gated node-side — this module adds no policy of its own.

use daemon_api::{ApiRequest, ApiResponse};
use daemon_host::ApiClient;

use crate::cli::PairCmd;
use crate::render::render;

/// Dispatch a `pair` subcommand over the api mirror.
pub(super) async fn run(client: &ApiClient, cmd: PairCmd) -> anyhow::Result<()> {
    match cmd {
        PairCmd::New => match client.call(ApiRequest::PairingBegin).await? {
            ApiResponse::PairingCode(armed) => {
                println!("code:    {}", daemon_host::pairing::group_code(&armed.code));
                println!("uri:     {}", armed.uri);
                println!("expires: {}", expiry(armed.expires_at));
                if armed.addresses.is_empty() {
                    println!("warning: no non-loopback address — LAN peers cannot dial this node");
                }
            }
            other => render(other),
        },
        PairCmd::Status => match client.call(ApiRequest::PairingStatus).await? {
            ApiResponse::PairingState(s) => {
                if s.locked {
                    println!("locked (attempt budget exhausted — `pair new` re-arms, `pair cancel` clears)");
                } else if s.armed {
                    let attempts = s
                        .attempts_remaining
                        .map(|n| format!(", {n} attempts remaining"))
                        .unwrap_or_default();
                    println!(
                        "armed ({}{attempts})",
                        s.expires_at
                            .map(expiry)
                            .unwrap_or_else(|| "no expiry".into())
                    );
                } else {
                    println!("disarmed");
                }
            }
            other => render(other),
        },
        PairCmd::Cancel => render(client.call(ApiRequest::PairingCancel).await?),
        PairCmd::Devices => match client.call(ApiRequest::PairedDeviceList).await? {
            ApiResponse::PairedDevices(rows) => {
                println!("paired devices: {}", rows.len());
                for d in rows {
                    let name = d.display_name.as_deref().unwrap_or("-");
                    let state = if d.enabled { "enabled" } else { "revoked" };
                    let seen = d
                        .last_seen_at
                        .map(|t| format!("last seen {t}"))
                        .unwrap_or_else(|| "never seen".into());
                    println!(
                        "  {}  {name}  [{state}]  fp={}  enrolled {}  {seen}",
                        d.username, d.fingerprint, d.created_at
                    );
                }
            }
            other => render(other),
        },
        PairCmd::Revoke { fingerprint } => render(
            client
                .call(ApiRequest::PairedDeviceRevoke { fingerprint })
                .await?,
        ),
    }
    Ok(())
}

/// Render an epoch-ms expiry as absolute seconds + relative remaining time.
fn expiry(expires_at_ms: u64) -> String {
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    if expires_at_ms > now_ms {
        format!("in {}s", (expires_at_ms - now_ms).div_ceil(1000))
    } else {
        "expired".to_string()
    }
}
