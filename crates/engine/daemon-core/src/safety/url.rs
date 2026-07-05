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
    /// The host is a registered name that **resolved** to a private/loopback/link-local/metadata
    /// address (a DNS-rebinding attempt). Only produced by [`check_url_resolved`] /
    /// [`check_url_resolved_with`]; plain [`check_url`] never resolves.
    #[error("host resolves to a non-public address (rebinding): {0}")]
    ResolvedPrivate(String),
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
    let host = normalize_host(strip_port(host_port));
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

/// Normalize a host to the canonical form the resolver will actually use, closing two bypass classes
/// **unconditionally** (every caller inherits this):
/// - **trailing FQDN dot** — `localhost.` / `127.0.0.1.` are still localhost / loopback;
/// - **IDNA/punycode** — a unicode/punycode host (e.g. alternate label separators `127。0。0。1`,
///   fullwidth digits) is folded through UTS#46 ToASCII so it cannot smuggle an IP literal or
///   `localhost` past the blocklist that the HTTP stack (reqwest → url → idna) would later resolve.
///
/// Bracketed IPv6 literals are returned lowercased as-is (UTS#46 does not apply). On IDNA failure the
/// trimmed host is kept (no regression; the standard `idna` crate is the same normalizer the HTTP
/// stack uses, so divergence is unlikely).
fn normalize_host(host_port_stripped: &str) -> String {
    let lowered = host_port_stripped.to_ascii_lowercase();
    if lowered.starts_with('[') {
        return lowered;
    }
    let trimmed = lowered.trim_end_matches('.');
    match idna::domain_to_ascii(trimmed) {
        Ok(ascii) => ascii.trim_end_matches('.').to_string(),
        Err(_) => trimmed.to_string(),
    }
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
        Ok(ip) => ip_is_blocked(ip),
        // A registered name we cannot classify without DNS — allow it (the opt-in
        // [`check_url_resolved`] adds the resolution-time guard); the http(s) + literal + IDNA checks
        // above cover the common SSRF vectors.
        Err(_) => false,
    }
}

/// Classify a resolved/literal [`IpAddr`] against the loopback / unspecified / private / link-local /
/// CGNAT / metadata denylist. Shared by the literal-host path ([`check_url`]) and the resolved-IP
/// path ([`check_url_resolved_with`]) so both key on one denylist.
fn ip_is_blocked(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => is_blocked_v4(v4),
        IpAddr::V6(v6) => is_blocked_v6(v6),
    }
}

/// Like [`check_url`], plus a **connect-time resolved-IP check** (DNS-rebinding defense): after the
/// string checks pass, a host that is a registered name (not already an IP literal) is resolved and
/// rejected with [`UrlReject::ResolvedPrivate`] if **any** resolved address is on the
/// private/loopback/link-local/metadata denylist.
///
/// This is the **surfaced opt-in** — plain [`check_url`] never resolves, so callers that legitimately
/// target private hosts (an operator-configured peer) are unaffected. It uses the blocking system
/// resolver; from async code call it inside a blocking context (e.g. `tokio::task::spawn_blocking`).
///
/// Note: this is resolve-then-check, not a pinned connector, so a strict rebind between this check and
/// the HTTP client's own resolution is still theoretically possible; closing that fully needs a
/// connector that pins the validated IP (a follow-on).
pub fn check_url_resolved(raw: &str) -> Result<CheckedUrl, UrlReject> {
    check_url_resolved_with(raw, resolve_system)
}

/// [`check_url_resolved`] with an **injectable resolver** — the seam that lets tests exercise the
/// rebinding path deterministically and offline (no live DNS). `resolve` maps a host name to its
/// addresses; a resolver `Err` is treated as "unresolvable" and passed through (the request would
/// fail at connect time anyway) rather than newly rejecting a name.
pub fn check_url_resolved_with<R>(raw: &str, resolve: R) -> Result<CheckedUrl, UrlReject>
where
    R: FnOnce(&str) -> std::io::Result<Vec<IpAddr>>,
{
    let checked = check_url(raw)?;
    // An IP literal was already classified by `check_url` — no DNS needed.
    let literal = checked
        .host
        .strip_prefix('[')
        .and_then(|h| h.strip_suffix(']'))
        .unwrap_or(&checked.host)
        .parse::<IpAddr>();
    if literal.is_ok() {
        return Ok(checked);
    }
    if let Ok(addrs) = resolve(&checked.host) {
        if addrs.into_iter().any(ip_is_blocked) {
            return Err(UrlReject::ResolvedPrivate(checked.host));
        }
    }
    Ok(checked)
}

/// The default system resolver used by [`check_url_resolved`]: resolve `host:0` and collect the
/// candidate addresses.
fn resolve_system(host: &str) -> std::io::Result<Vec<IpAddr>> {
    use std::net::ToSocketAddrs;
    Ok((host, 0u16).to_socket_addrs()?.map(|sa| sa.ip()).collect())
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
    fn rejects_trailing_dot_hostnames() {
        // A trailing FQDN dot must not bypass the blocklist: `localhost.` is still localhost, and
        // `127.0.0.1.` is still loopback (the resolver treats the trailing dot as the root label).
        for u in [
            "http://localhost./",
            "http://127.0.0.1./",
            "http://169.254.169.254./latest/meta-data/",
            "http://10.0.0.5./",
            "http://api.localhost./",
        ] {
            assert!(
                matches!(check_url(u), Err(UrlReject::PrivateHost(_))),
                "expected {u} to be rejected (trailing-dot bypass)"
            );
        }
    }

    #[test]
    fn rejects_idna_and_punycode_bypass() {
        // Non-ASCII label separators that UTS#46 maps to '.', yielding the loopback IP literal
        // `127.0.0.1` — which the resolver (reqwest -> url -> idna) would connect to. Without IDNA
        // normalization `check_url` sees an unclassifiable name and (pre-fix) allows it.
        for u in [
            "http://127\u{3002}0\u{3002}0\u{3002}1/", // U+3002 ideographic full stop
            "http://127\u{ff0e}0\u{ff0e}0\u{ff0e}1/", // U+FF0E fullwidth full stop
            "http://127\u{ff61}0\u{ff61}0\u{ff61}1/", // U+FF61 halfwidth ideographic full stop
        ] {
            assert!(
                matches!(check_url(u), Err(UrlReject::PrivateHost(_))),
                "expected {u:?} to normalize to loopback and be rejected"
            );
        }
        // A legitimate internationalized public domain must still pass (no over-blocking): it
        // normalizes to its punycode ASCII form, which is not on the denylist.
        assert!(
            check_url("https://m\u{fc}nchen.de/").is_ok(),
            "münchen.de should normalize to xn--mnchen-3ya.de and pass"
        );
    }

    #[test]
    fn resolved_ip_check_rejects_rebinding_when_enabled() {
        // A public-looking name that (via the injected resolver) resolves to a private/loopback/
        // metadata address is rejected — the DNS-rebinding case. Deterministic + offline.
        for blocked in [
            [127, 0, 0, 1],
            [169, 254, 169, 254],
            [10, 0, 0, 5],
            [192, 168, 1, 1],
        ] {
            let resolve = move |_host: &str| Ok(vec![IpAddr::from(blocked)]);
            assert!(
                matches!(
                    check_url_resolved_with("https://rebind.example/x", resolve),
                    Err(UrlReject::ResolvedPrivate(_))
                ),
                "expected rebind to {blocked:?} to be rejected"
            );
        }
        // Even one blocked address among several rejects (a mixed A-record rebind).
        let mixed = |_host: &str| {
            Ok(vec![
                IpAddr::from([93, 184, 216, 34]),
                IpAddr::from([127, 0, 0, 1]),
            ])
        };
        assert!(matches!(
            check_url_resolved_with("https://mixed.example/", mixed),
            Err(UrlReject::ResolvedPrivate(_))
        ));
    }

    #[test]
    fn resolved_ip_check_allows_public_and_defers_to_string_checks() {
        // A name resolving only to public addresses passes.
        let public = |_host: &str| Ok(vec![IpAddr::from([93, 184, 216, 34])]);
        assert!(check_url_resolved_with("https://example.com/", public).is_ok());
        // An unresolvable name is passed through (the request fails at connect time) — not newly
        // rejected by the resolver step.
        let unresolvable = |_host: &str| {
            Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "nxdomain",
            ))
        };
        assert!(check_url_resolved_with("https://example.com/", unresolvable).is_ok());
        // The string checks still run first: a literal loopback is rejected before any resolution,
        // and the resolver must not even be consulted for an IP literal.
        let must_not_call = |_host: &str| -> std::io::Result<Vec<IpAddr>> {
            panic!("resolver must not be called for an IP literal");
        };
        assert!(matches!(
            check_url_resolved_with("http://127.0.0.1/", must_not_call),
            Err(UrlReject::PrivateHost(_))
        ));
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
