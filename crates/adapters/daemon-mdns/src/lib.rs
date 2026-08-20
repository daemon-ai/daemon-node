// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The LAN DNS-SD advertiser for the node's TLS api listener (daemon-lan-discovery-spec.md §3).
//!
//! A thin, synchronous-at-the-edges shell over the pure-Rust [`mdns-sd`] responder: [`Advertiser::
//! start`] registers one `_daemon-node._tcp.local.` instance from a fully-composed
//! [`ServiceSpec`], and [`Advertiser::shutdown`] unregisters it so goodbye records (TTL 0) go out
//! before the listener disappears. Interface churn (address add/remove) is handled inside
//! `mdns-sd`'s daemon — no hand-rolled republish logic here.
//!
//! Policy lives with the caller (`bins/daemon`): this crate never reads config, computes
//! fingerprints, or knows version constants. It publishes what it is told, verbatim.

use std::collections::BTreeMap;

use anyhow::Context;
use mdns_sd::{ServiceDaemon, ServiceInfo};

/// The DNS-SD service type for a daemon node's TLS api listener (spec §2.1). One type names the
/// product's node; v1 binds it to the TLS carrier. A future second carrier registers a distinct
/// type — it does not overload this one.
pub const SERVICE_TYPE: &str = "_daemon-node._tcp.local.";

/// One fully-composed DNS-SD registration (spec §2). The caller owns every value: the TXT map is
/// published verbatim (spec §2.2 schema, `txtvers=1`), the port is the *actually bound* TLS port,
/// and the instance name is display-only (identity is the TXT `node` key).
#[derive(Clone, Debug)]
pub struct ServiceSpec {
    /// The user-visible instance name (RFC 6763; spaces allowed). LAN uniqueness is delegated to
    /// standard DNS-SD conflict resolution (auto-rename) — never treated as identity.
    pub instance: String,
    /// The SRV host, as `<hostname>.local.`.
    pub hostname: String,
    /// The SRV port: the bound TLS port from `TcpListener::local_addr()` — never a configured
    /// string (a `:0` bind would otherwise advertise a lie).
    pub port: u16,
    /// The `txtvers=1` TXT record (spec §2.2), fully composed by the caller.
    pub txt: BTreeMap<String, String>,
}

/// An active registration: owns the `mdns-sd` daemon for the process. Dropping it without
/// [`Advertiser::shutdown`] still stops the responder, but a clean shutdown is what guarantees
/// goodbye records precede the listener disappearing.
pub struct Advertiser {
    daemon: ServiceDaemon,
    fullname: String,
}

impl Advertiser {
    /// Register `spec` on the LAN and keep responding until [`Advertiser::shutdown`]. Addresses
    /// are auto-detected (and tracked across interface churn) by the `mdns-sd` daemon.
    pub fn start(spec: ServiceSpec) -> anyhow::Result<Self> {
        let daemon = ServiceDaemon::new().context("starting mdns responder daemon")?;
        let info = service_info(&spec)?;
        let fullname = info.get_fullname().to_string();
        daemon
            .register(info)
            .with_context(|| format!("registering {fullname}"))?;
        tracing::debug!(%fullname, port = spec.port, "mdns: service registered");
        Ok(Self { daemon, fullname })
    }

    /// Unregister (=> goodbye records with TTL 0) and stop the responder. Best-effort by design:
    /// shutdown must never wedge the node's exit path, so failures are logged and swallowed.
    pub fn shutdown(self) {
        match self.daemon.unregister(&self.fullname) {
            Ok(rx) => {
                // Bounded wait so the goodbye actually hits the wire before the process exits.
                if rx.recv_timeout(std::time::Duration::from_secs(2)).is_err() {
                    tracing::debug!(fullname = %self.fullname, "mdns: unregister ack timed out");
                }
            }
            Err(e) => tracing::debug!(fullname = %self.fullname, "mdns: unregister failed: {e}"),
        }
        if let Err(e) = self.daemon.shutdown() {
            tracing::debug!("mdns: daemon shutdown failed: {e}");
        }
    }
}

/// The host's current non-loopback, non-link-local-v6 IP addresses (IPv4 first, then IPv6) —
/// the same interface set the responder advertises on. Consumed by the pairing admin surface
/// (`PairingBegin.addresses`, daemon-pairing-spec.md §5.5) so the join URI names endpoints a
/// LAN peer can actually dial. Reuses the `if-addrs` enumeration `mdns-sd` already vendors.
pub fn non_loopback_addrs() -> Vec<std::net::IpAddr> {
    let mut v4 = Vec::new();
    let mut v6 = Vec::new();
    for iface in if_addrs::get_if_addrs().unwrap_or_default() {
        if iface.is_loopback() {
            continue;
        }
        match iface.ip() {
            std::net::IpAddr::V4(a) => v4.push(std::net::IpAddr::V4(a)),
            std::net::IpAddr::V6(a) => {
                // Link-local v6 needs a zone id to dial; useless in a URI.
                if (a.segments()[0] & 0xffc0) != 0xfe80 {
                    v6.push(std::net::IpAddr::V6(a));
                }
            }
        }
    }
    v4.extend(v6);
    v4
}

/// Map a [`ServiceSpec`] onto an `mdns-sd` [`ServiceInfo`] with address auto-detection enabled.
fn service_info(spec: &ServiceSpec) -> anyhow::Result<ServiceInfo> {
    let txt: Vec<(&str, &str)> = spec
        .txt
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();
    let info = ServiceInfo::new(
        SERVICE_TYPE,
        &spec.instance,
        &spec.hostname,
        (),
        spec.port,
        &txt[..],
    )
    .with_context(|| format!("composing service info for {:?}", spec.instance))?
    .enable_addr_auto();
    Ok(info)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> ServiceSpec {
        ServiceSpec {
            instance: "Office Daemon".into(),
            hostname: "office.local.".into(),
            port: 7443,
            txt: BTreeMap::from([
                ("txtvers".to_string(), "1".to_string()),
                (
                    "node".to_string(),
                    "00112233445566778899aabbccddeeff".to_string(),
                ),
                ("auth".to_string(), "scram".to_string()),
            ]),
        }
    }

    #[test]
    fn service_info_carries_spec_verbatim() {
        let info = service_info(&spec()).expect("valid spec composes");
        assert_eq!(info.get_type(), SERVICE_TYPE);
        assert_eq!(
            info.get_fullname(),
            "Office Daemon._daemon-node._tcp.local."
        );
        assert_eq!(info.get_hostname(), "office.local.");
        assert_eq!(info.get_port(), 7443);
        assert_eq!(info.get_property_val_str("txtvers"), Some("1"));
        assert_eq!(
            info.get_property_val_str("node"),
            Some("00112233445566778899aabbccddeeff")
        );
        assert_eq!(info.get_property_val_str("auth"), Some("scram"));
        // Nothing beyond the caller's keys is invented into the TXT record.
        assert_eq!(info.get_properties().len(), 3);
    }
}
