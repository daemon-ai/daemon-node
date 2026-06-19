//! `daemon-credentials` — the credential authority.
//!
//! Brokers scoped, short-lived credentials to units the host activates; keeps secret material out of
//! engine and tool crates. Depends only on `daemon-common`.

#![forbid(unsafe_code)]

// TODO: define CredentialAuthority trait + scoped-lease issuance.
