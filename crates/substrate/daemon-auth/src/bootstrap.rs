// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! First-admin bootstrap (#2): the store-side, idempotent "seed exactly one admin when the users
//! table is empty" decision.
//!
//! The *policy* (resolving the operator's environment vs. auto-generating, reading a password file,
//! and the one-time emission of a generated secret) lives in the binary (`bins/daemon`) — that is
//! where process env + filesystem side effects belong. This module owns only the pure,
//! unit-testable core: given a resolved [`AdminSeed`], create the admin **iff** no users exist yet,
//! and hand back what was created so the caller can emit a generated password once.
//!
//! Idempotency: a second boot (users already present) is a no-op. This never weakens the
//! fail-closed capability gate — it only ensures a networked/TLS operator has a real Admin identity
//! to log in as (the default `local_trust=system` operator is already admin without SASL).

use crate::capability::Role;
use crate::error::Result;
use crate::store::{generate_secret_hex, AuthStore};

/// The resolved first-admin seeding policy. Environment resolution + validation happen in the
/// binary; this is the decision the store executes.
pub enum AdminSeed {
    /// Operator-supplied identity (the env path). The caller has already validated that `password`
    /// is non-empty/non-whitespace.
    Explicit {
        /// The admin username to create.
        username: String,
        /// The admin password (validated non-empty by the caller).
        password: String,
    },
    /// Auto-generate a random `admin-<hex>` username and a strong random password. The generated
    /// password is returned once (see [`SeededAdmin::generated_password`]) for the caller to emit.
    Generate,
}

/// What [`AuthStore::seed_first_admin_if_empty`] created (returned only when it actually seeded).
pub struct SeededAdmin {
    /// The created admin's username.
    pub username: String,
    /// The plaintext password — `Some` **only** for the auto-generated path, so the caller can emit
    /// it exactly once (to stderr + a `0600` file). `None` for the operator-supplied path (the
    /// operator already knows their password). Never persisted in plaintext (it is Argon2id/SCRAM
    /// hashed by [`AuthStore::create_user`]).
    pub generated_password: Option<String>,
}

impl AuthStore {
    /// Seed exactly one [`Role::Admin`] user **iff the users table is empty**. Idempotent: returns
    /// `Ok(None)` when any user already exists (a second boot never re-seeds). On an empty store it
    /// creates the admin per `seed` and returns `Ok(Some(_))`.
    ///
    /// For [`AdminSeed::Generate`], the returned [`SeededAdmin::generated_password`] carries the
    /// freshly-minted password so the binary can emit it once; the store keeps only its hash.
    pub fn seed_first_admin_if_empty(&self, seed: AdminSeed) -> Result<Option<SeededAdmin>> {
        if self.user_count()? > 0 {
            return Ok(None);
        }
        let (username, password, generated_password) = match seed {
            AdminSeed::Explicit { username, password } => (username, password, None),
            AdminSeed::Generate => {
                // `admin-<8 hex>` username + a 192-bit hex password (24 random bytes).
                let username = format!("admin-{}", generate_secret_hex(4)?);
                let password = generate_secret_hex(24)?;
                (username, password.clone(), Some(password))
            }
        };
        self.create_user(&username, &password, &[Role::Admin])?;
        Ok(Some(SeededAdmin {
            username,
            generated_password,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::Capability;
    use crate::store::AuthStore;

    fn store() -> AuthStore {
        AuthStore::open_in_memory().expect("open")
    }

    #[test]
    fn user_count_reflects_created_users() {
        let s = store();
        assert_eq!(s.user_count().unwrap(), 0, "fresh store has no users");
        s.create_user("solo", "pw", &[Role::User]).unwrap();
        assert_eq!(s.user_count().unwrap(), 1);
    }

    #[test]
    fn generate_secret_hex_is_hex_of_expected_len_and_unique() {
        let a = generate_secret_hex(24).unwrap();
        let b = generate_secret_hex(24).unwrap();
        assert_eq!(a.len(), 48, "2 hex chars per byte");
        assert!(a.chars().all(|c| c.is_ascii_hexdigit()));
        assert_ne!(a, b, "two draws differ (CSPRNG)");
    }

    #[test]
    fn explicit_seed_creates_exactly_one_admin_on_empty_store() {
        let s = store();
        let created = s
            .seed_first_admin_if_empty(AdminSeed::Explicit {
                username: "root".to_string(),
                password: "rootpw".to_string(),
            })
            .unwrap()
            .expect("seeded on empty store");
        assert_eq!(created.username, "root");
        assert!(
            created.generated_password.is_none(),
            "explicit path emits nothing"
        );
        assert_eq!(s.user_count().unwrap(), 1, "exactly one user");
        // The created principal is a real Admin (holds AccessAdmin).
        let p = s.authenticate_password("root", "rootpw").unwrap();
        assert!(p.has(Capability::AccessAdmin), "seeded user is an admin");
    }

    #[test]
    fn second_boot_does_not_reseed() {
        let s = store();
        s.seed_first_admin_if_empty(AdminSeed::Explicit {
            username: "root".to_string(),
            password: "rootpw".to_string(),
        })
        .unwrap()
        .expect("first seed");
        // A second call (even a different policy) is a no-op once users exist.
        let again = s.seed_first_admin_if_empty(AdminSeed::Generate).unwrap();
        assert!(again.is_none(), "idempotent: no re-seed");
        assert_eq!(s.user_count().unwrap(), 1, "still exactly one user");
    }

    #[test]
    fn generated_seed_mints_usable_admin() {
        let s = store();
        let created = s
            .seed_first_admin_if_empty(AdminSeed::Generate)
            .unwrap()
            .expect("seeded on empty store");
        assert!(
            created.username.starts_with("admin-"),
            "auto username prefix"
        );
        let password = created
            .generated_password
            .expect("generated path returns the password once");
        assert!(!password.is_empty());
        // The generated credentials actually authenticate and resolve an Admin principal.
        let p = s
            .authenticate_password(&created.username, &password)
            .unwrap();
        assert!(p.has(Capability::AccessAdmin));
    }
}
