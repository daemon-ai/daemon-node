// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! URL safety (§9 `url_safety`) — an SSRF/egress guard for the network-facing tools.
//!
//! Web/browser tools fetch arbitrary model-supplied URLs. Without a guard the model could be
//! steered (directly or via prompt injection in a fetched page) into reaching the loopback
//! interface, link-local metadata endpoints (`169.254.169.254`), or RFC-1918 hosts on the operator's
//! network. [`check_url`] enforces an http(s)-only, public-host-only policy before any request is
//! made. It is dependency-free (a small hand-rolled split plus [`std::net::IpAddr`] literal
//! classification) so the core engine crate stays lean.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

/// Why a URL was rejected by [`check_url`].
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum UrlReject {
    /// The URL could not be parsed into a scheme + host.
    #[error("malformed url: {0}")]
    Malformed(String),
    /// The scheme is not `http`/`https` (e.g. `file:`, `data:`, `javascript:`).
    #[error("scheme not allowed: {0} (only http/https)")]
    Scheme(String),
    /// The host is empty.
    #[error("empty host")]
    EmptyHost,
    /// The host resolves to a private, loopback, link-local, or otherwise non-public address, or is
    /// a name (`localhost`) that conventionally does.
    #[error("host not allowed (private/loopback/link-local): {0}")]
    PrivateHost(String),
}

/// A URL that passed [`check_url`]: its (lowercased) scheme and host, plus the original string the
/// caller should hand to its HTTP client.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CheckedUrl {
    /// The lowercased scheme (`http` or `https`).
    pub scheme: String,
    /// The lowercased host (a registered name or IP literal).
    pub host: String,
    /// The original, unmodified URL string.
    pub url: String,
}

/// Validate `raw` against the http(s)-only, public-host-only egress policy. Returns the parsed
/// [`CheckedUrl`] on success, or the [`UrlReject`] reason. Does **not** perform DNS resolution — a
/// hostile name that resolves to a private address is a known limitation; resolution-time guards
/// belong in the HTTP layer.
pub fn check_url(raw: &str) -> Result<CheckedUrl, UrlReject> {
    let trimmed = raw.trim();
    let (scheme, rest) = trimmed
        .split_once("://")
        .ok_or_else(|| UrlReject::Malformed(raw.to_string()))?;
    let scheme = scheme.to_ascii_lowercase();
    if scheme != "http" && scheme != "https" {
        return Err(UrlReject::Scheme(scheme));
    }
    // The authority ends at the first '/', '?', or '#'.
    let authority = rest.split(['/', '?', '#']).next().unwrap_or(rest);
    // Strip any userinfo ("user:pass@host") then the optional ":port".
    let host_port = authority
        .rsplit_once('@')
        .map(|(_, h)| h)
        .unwrap_or(authority);
    let host = strip_port(host_port).to_ascii_lowercase();
    if host.is_empty() {
        return Err(UrlReject::EmptyHost);
    }
    if is_blocked_host(&host) {
        return Err(UrlReject::PrivateHost(host));
    }
    Ok(CheckedUrl {
        scheme,
        host,
        url: trimmed.to_string(),
    })
}

/// Strip a trailing `:port` from a host, leaving bracketed IPv6 literals (`[::1]`) intact.
fn strip_port(host_port: &str) -> &str {
    if let Some(end) = host_port.strip_prefix('[') {
        // `[ipv6]` or `[ipv6]:port` — the host is up to and including the closing bracket.
        if let Some(idx) = end.find(']') {
            return &host_port[..idx + 2];
        }
        return host_port;
    }
    match host_port.rsplit_once(':') {
        // Only treat the suffix as a port when it is all digits (avoids mangling a bare IPv6).
        Some((h, p)) if !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()) => h,
        _ => host_port,
    }
}

/// Whether a host (name or IP literal) is on the blocklist: `localhost`/`*.localhost`, or an IP
/// literal that is loopback, unspecified, private, link-local, or IPv6 unique-local.
fn is_blocked_host(host: &str) -> bool {
    if host == "localhost" || host.ends_with(".localhost") {
        return true;
    }
    // Bracketed IPv6 literal.
    let ip_str = host
        .strip_prefix('[')
        .and_then(|h| h.strip_suffix(']'))
        .unwrap_or(host);
    match ip_str.parse::<IpAddr>() {
        Ok(IpAddr::V4(v4)) => is_blocked_v4(v4),
        Ok(IpAddr::V6(v6)) => is_blocked_v6(v6),
        // A registered name we cannot classify without DNS — allow it (resolution-time guard is a
        // separate layer); the http(s) + literal checks above cover the common SSRF vectors.
        Err(_) => false,
    }
}

/// Loopback / unspecified / private (RFC-1918) / link-local / CGNAT / benchmarking IPv4.
fn is_blocked_v4(v4: Ipv4Addr) -> bool {
    v4.is_loopback()
        || v4.is_unspecified()
        || v4.is_private()
        || v4.is_link_local()
        || v4.is_broadcast()
        || v4.is_documentation()
        // 100.64.0.0/10 CGNAT (not stable in std) — classify by octet.
        || (v4.octets()[0] == 100 && (64..=127).contains(&v4.octets()[1]))
}

/// Loopback / unspecified IPv6, plus unique-local (fc00::/7) and link-local (fe80::/10).
fn is_blocked_v6(v6: Ipv6Addr) -> bool {
    if v6.is_loopback() || v6.is_unspecified() {
        return true;
    }
    let seg0 = v6.segments()[0];
    // fc00::/7 unique-local, fe80::/10 link-local.
    (seg0 & 0xfe00) == 0xfc00 || (seg0 & 0xffc0) == 0xfe80
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allows_public_http_and_https() {
        let ok = check_url("https://example.com/path?q=1").unwrap();
        assert_eq!(ok.scheme, "https");
        assert_eq!(ok.host, "example.com");
        assert!(check_url("http://1.1.1.1/").is_ok());
        assert!(check_url("https://example.com:8443/x").is_ok());
    }

    #[test]
    fn rejects_non_http_schemes() {
        assert!(matches!(
            check_url("file:///etc/passwd"),
            Err(UrlReject::Scheme(_))
        ));
        assert!(matches!(
            check_url("data:text/html,<b>x</b>"),
            Err(UrlReject::Malformed(_)) | Err(UrlReject::Scheme(_))
        ));
        assert!(matches!(
            check_url("javascript://alert(1)"),
            Err(UrlReject::Scheme(_))
        ));
    }

    #[test]
    fn rejects_localhost_and_loopback() {
        assert!(matches!(
            check_url("http://localhost:3000/"),
            Err(UrlReject::PrivateHost(_))
        ));
        assert!(matches!(
            check_url("http://127.0.0.1/"),
            Err(UrlReject::PrivateHost(_))
        ));
        assert!(matches!(
            check_url("http://[::1]/"),
            Err(UrlReject::PrivateHost(_))
        ));
        assert!(matches!(
            check_url("http://api.localhost/"),
            Err(UrlReject::PrivateHost(_))
        ));
    }

    #[test]
    fn rejects_private_and_metadata_ranges() {
        for u in [
            "http://10.0.0.5/",
            "http://192.168.1.1/",
            "http://172.16.0.9/",
            "http://169.254.169.254/latest/meta-data/",
            "http://100.64.1.1/",
            "http://[fe80::1]/",
            "http://[fc00::1]/",
        ] {
            assert!(
                matches!(check_url(u), Err(UrlReject::PrivateHost(_))),
                "expected {u} to be rejected"
            );
        }
    }

    #[test]
    fn rejects_malformed_and_empty_host() {
        assert!(matches!(
            check_url("not a url"),
            Err(UrlReject::Malformed(_))
        ));
        assert!(matches!(
            check_url("http:///path"),
            Err(UrlReject::EmptyHost)
        ));
    }
}
