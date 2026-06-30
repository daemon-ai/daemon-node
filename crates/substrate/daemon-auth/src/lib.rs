// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-auth` — the node's identity, credential, and RBAC foundation.
//!
//! This crate is transport- and protocol-agnostic: it owns *who a principal is* and *what they may
//! do*, not *how* the bytes are exchanged. The wire handshake (SASL/SCRAM over TLS) and the
//! per-request authorization gate live in `daemon-api`/`daemon-host` and consume the types here.
//!
//! - [`capability`]: the RBAC vocabulary — [`Capability`], [`Role`], and the resolved [`Principal`].
//! - [`store`]: the SQLite identity store — users, Argon2id passwords, opaque server-side session
//!   tokens, role assignments, plus reserved tables for SCRAM material, API keys, and (future)
//!   per-resource grants.
//!
//! Design notes:
//! - Passwords are Argon2id PHC strings (`password-auth`); session tokens are random and stored only
//!   as a SHA-256 hash (OWASP: opaque, server-side, revocable — never JWTs in the DB).
//! - Authorization is two-step: a coarse per-request [`Capability`] gate, plus a per-resource
//!   *ownership* check in the session layer (with [`Capability::SessionSeeAll`] /
//!   [`Capability::SessionControlAny`] as operator overrides). Finer per-resource sharing is a
//!   reserved extension (`resource_grants`).

#![forbid(unsafe_code)]

pub mod capability;
pub mod error;
pub mod scram;
pub mod store;

pub use capability::{Capability, Principal, Role};
pub use error::{Error, Result};
pub use scram::{ScramMaterial, SCRAM_DEFAULT_ITERATIONS, SCRAM_SHA_256};
pub use store::{AuthStore, UserRecord, DEFAULT_SESSION_TTL_SECS};
