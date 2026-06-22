//! Resolved Matrix adapter configuration.
//!
//! The *route table* is the config surface of the §5.9.1 routing registry: it carries no account
//! secrets (accounts and their session blobs come from `ProfileSpec.bound_accounts` +
//! `CredentialStore`, enumerated at bring-up via `AccountProvisioning`). A route only narrows *which*
//! rooms the adapter engages and *how it classifies addressing* (mention-gating). Profile selection
//! (account->profile + per-room overrides) and session isolation stay host-owned via
//! `bound_accounts` + the `[routing]` registry, so the gate's derived session id always matches the
//! id the host resolves (`PerThread`) — the invariant the outbound busy-tracking relies on.

use daemon_ingest::IngestPolicy;
use daemon_protocol::IsolationPolicy;

/// The resolved `[matrix]` config the host hands to [`crate::serve`]. Built in `bins/daemon`'s
/// `config.rs` from the TOML table + `DAEMON_MATRIX_*` env (the adapter owns the *shape*, the binary
/// owns the parsing — mirroring `WebConfig`).
#[derive(Clone, Debug)]
pub struct MatrixConfig {
    /// Whether the adapter is spawned at all. `false` (default) leaves Matrix off.
    pub enabled: bool,
    /// The absolute per-account store root (the binary resolves `[matrix].store_root` against the
    /// node `data_dir`). Each account gets its own `<store_root>/<instance>/` subdir holding the
    /// matrix-sdk state + E2EE crypto SQLite store.
    pub store_root: std::path::PathBuf,
    /// The route table — which rooms to engage + how addressing is classified. Empty engages every
    /// room of every account with mention-gating on.
    pub routes: Vec<MatrixRoute>,
}

impl Default for MatrixConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            store_root: std::path::PathBuf::from("matrix"),
            routes: Vec::new(),
        }
    }
}

impl MatrixConfig {
    /// The account-level ingest policy. Isolation is pinned to `PerThread` to match the host's
    /// no-binding routing resolution, so the gate and the outbound stream key busy-state on the same
    /// session id (see module docs).
    pub fn ingest_policy(&self) -> IngestPolicy {
        IngestPolicy {
            isolation: IsolationPolicy::PerThread,
            ..IngestPolicy::default()
        }
    }
}

/// One `[[matrix.route]]` entry. All matchers are optional; an empty route matches every room of
/// every account with mention-gating on.
#[derive(Clone, Debug)]
pub struct MatrixRoute {
    /// Match only this account instance (the bare `@user:hs.org`); `None` matches any account.
    pub account: Option<String>,
    /// A glob over the room (alias or id) this route applies to; `None` matches any room.
    pub room_glob: Option<String>,
    /// Restrict to direct-message rooms only.
    pub dm_only: bool,
    /// Whether the agent only *turns* on an explicit mention / DM / `!command` (ambient chatter is
    /// surfaced as context via the gate's ambient policy). When `false`, every message in a matching
    /// room is treated as addressed.
    pub mention_gating: bool,
}

impl Default for MatrixRoute {
    fn default() -> Self {
        Self {
            account: None,
            room_glob: None,
            dm_only: false,
            mention_gating: true,
        }
    }
}

impl MatrixRoute {
    /// Whether this route matches `account` (the bare user id) and `room` (id or alias).
    pub fn matches(&self, account: &str, room: &str, is_dm: bool) -> bool {
        if let Some(a) = &self.account {
            if a != account {
                return false;
            }
        }
        if self.dm_only && !is_dm {
            return false;
        }
        if let Some(g) = &self.room_glob {
            if !glob_match(g, room) {
                return false;
            }
        }
        true
    }
}

/// Pick the route governing `(account, room, is_dm)`. With no configured routes, every room is
/// engaged with mention-gating on (the default route). With a configured table, only rooms matching
/// some route are engaged; a non-matching room returns `None` (the adapter ignores it).
pub fn route_for<'a>(
    routes: &'a [MatrixRoute],
    account: &str,
    room: &str,
    is_dm: bool,
) -> Option<std::borrow::Cow<'a, MatrixRoute>> {
    if routes.is_empty() {
        return Some(std::borrow::Cow::Owned(MatrixRoute::default()));
    }
    routes
        .iter()
        .find(|r| r.matches(account, room, is_dm))
        .map(std::borrow::Cow::Borrowed)
}

/// A tiny `*`/`?` glob (no character classes) — enough for `#secops*` style room matchers without
/// pulling a glob crate into the adapter. `*` matches any run (including empty), `?` one char.
fn glob_match(pattern: &str, text: &str) -> bool {
    glob_inner(pattern.as_bytes(), text.as_bytes())
}

fn glob_inner(pat: &[u8], text: &[u8]) -> bool {
    match pat.first() {
        None => text.is_empty(),
        Some(b'*') => glob_inner(&pat[1..], text) || (!text.is_empty() && glob_inner(pat, &text[1..])),
        Some(b'?') => !text.is_empty() && glob_inner(&pat[1..], &text[1..]),
        Some(&c) => !text.is_empty() && text[0] == c && glob_inner(&pat[1..], &text[1..]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glob_basics() {
        assert!(glob_match("#secops*", "#secops"));
        assert!(glob_match("#secops*", "#secops-prod"));
        assert!(!glob_match("#secops*", "#general"));
        assert!(glob_match("*", "anything"));
        assert!(glob_match("#a?c", "#abc"));
        assert!(!glob_match("#a?c", "#ac"));
    }

    #[test]
    fn empty_table_engages_all_with_gating() {
        let r = route_for(&[], "@ops:hs.org", "#general", false).unwrap();
        assert!(r.mention_gating);
    }

    #[test]
    fn configured_table_ignores_unmatched_rooms() {
        let routes = vec![MatrixRoute {
            room_glob: Some("#secops*".into()),
            mention_gating: false,
            ..Default::default()
        }];
        let r = route_for(&routes, "@ops:hs.org", "#secops-prod", false).unwrap();
        assert!(!r.mention_gating);
        assert!(route_for(&routes, "@ops:hs.org", "#general", false).is_none());
    }

    #[test]
    fn dm_only_route_matches_dms() {
        let routes = vec![MatrixRoute {
            dm_only: true,
            ..Default::default()
        }];
        assert!(route_for(&routes, "@a:hs", "!room:hs", true).is_some());
        assert!(route_for(&routes, "@a:hs", "!room:hs", false).is_none());
    }
}
