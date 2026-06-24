//! Host-level bidirectional routing (daemon-event-io-spec §5.9).
//!
//! The routing registry maps an inbound [`Origin`] to the `(SessionId, profile, DeliveryTarget)` the
//! host binds a live session to. It sits on top of [`session_id_for`] (which keeps owning *naming*)
//! and adds the new degree of freedom: **agent selection** — which profile runs the derived session.
//!
//! Resolution precedence (event-io spec decision 10):
//! 1. an explicit per-binding `profile` override,
//! 2. else the transport-instance's bound profile (the account→profile baseline — the
//!    [`instance_profiles`](RoutingRegistry) map, derived from each profile's `bound_accounts`
//!    (§5.9.4) via [`RoutingRegistry::bind_instances_from_profiles`], with explicit config bindings
//!    taking precedence),
//! 3. else the node default profile (`None` resolves the node's active default at build time,
//!    preserving the legacy single-profile behavior).
//!
//! Outbound is the symmetric half: the registry also answers *where* a matched session's replies post
//! (its [`DeliveryTarget`]), auto-seeded as the inverse of the opening origin unless a binding pins it.

use daemon_api::ProfileSpec;
use daemon_common::{ProfileRef, SessionId};
use daemon_protocol::{
    session_id_for, DeliveryTarget, IsolationPolicy, Origin, OriginScope, TransportId,
};
use std::collections::HashMap;

/// Which transport(s) a binding matches.
#[derive(Clone, Debug)]
pub enum TransportPattern {
    /// Exactly this instance-qualified transport id (e.g. `matrix/@bot:hs.org`).
    Exact(TransportId),
    /// Any instance of a transport *family*: matches the `family/...` prefix before the first `/`
    /// (so `Family("matrix")` matches `matrix/@a:hs` and `matrix/@b:hs`), or the bare family name.
    /// This is how the instance-qualified-`TransportId` convention expresses "any matrix account."
    Family(String),
    /// Any transport.
    Any,
}

impl TransportPattern {
    fn matches(&self, t: &TransportId) -> bool {
        match self {
            TransportPattern::Exact(want) => want == t,
            TransportPattern::Family(fam) => t.as_str().split('/').next() == Some(fam.as_str()),
            TransportPattern::Any => true,
        }
    }
}

/// Which conversational scope(s) a binding matches.
#[derive(Clone, Debug)]
pub enum ScopePattern {
    /// Any direct/1:1 conversation.
    Dm,
    /// A group/channel whose chat handle matches `chat_glob` (a `*`-wildcard glob; `*` matches any).
    Group {
        /// The chat-handle glob (`*` = any run, including empty).
        chat_glob: String,
    },
    /// Any programmatic API caller.
    Api,
    /// A host-internal origin (schedule / background triggers).
    Internal,
    /// Any scope.
    Any,
}

impl ScopePattern {
    fn matches(&self, scope: &OriginScope) -> bool {
        match (self, scope) {
            (ScopePattern::Any, _) => true,
            (ScopePattern::Dm, OriginScope::Dm { .. }) => true,
            (ScopePattern::Api, OriginScope::Api { .. }) => true,
            (ScopePattern::Internal, OriginScope::Internal) => true,
            (ScopePattern::Group { chat_glob }, OriginScope::Group { chat, .. }) => {
                glob_match(chat_glob, chat)
            }
            _ => false,
        }
    }
}

/// A `*`-wildcard glob match — the only metacharacter is `*` (matching any run, including empty).
fn glob_match(pattern: &str, value: &str) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    let v: Vec<char> = value.chars().collect();
    // Classic linear two-pointer wildcard match with backtracking on the last `*`.
    let (mut pi, mut vi) = (0usize, 0usize);
    let (mut star, mut mark) = (None, 0usize);
    while vi < v.len() {
        if pi < p.len() && (p[pi] == v[vi]) {
            pi += 1;
            vi += 1;
        } else if pi < p.len() && p[pi] == '*' {
            star = Some(pi);
            mark = vi;
            pi += 1;
        } else if let Some(s) = star {
            pi = s + 1;
            mark += 1;
            vi = mark;
        } else {
            return false;
        }
    }
    while pi < p.len() && p[pi] == '*' {
        pi += 1;
    }
    pi == p.len()
}

/// Matches an inbound [`Origin`] (transport-instance + scope).
#[derive(Clone, Debug)]
pub struct OriginMatcher {
    /// The transport-instance pattern.
    pub transport: TransportPattern,
    /// The scope pattern.
    pub scope: ScopePattern,
}

impl OriginMatcher {
    /// Match any origin (the catch-all row).
    pub fn any() -> Self {
        Self {
            transport: TransportPattern::Any,
            scope: ScopePattern::Any,
        }
    }

    fn matches(&self, origin: &Origin) -> bool {
        self.transport.matches(&origin.transport) && self.scope.matches(&origin.scope)
    }
}

/// Where a matched session's outbound replies post.
#[derive(Clone, Debug)]
pub enum DeliveryPolicy {
    /// Seed the `Primary` from the opening origin ([`Origin::primary_target`]) — the common case.
    FromOrigin,
    /// Pin the `Primary` to a fixed target regardless of origin.
    Fixed(DeliveryTarget),
}

/// One ordered routing rule: the origins it matches, how to name their session ([`IsolationPolicy`]),
/// which profile runs it (optional override), and where its replies post.
#[derive(Clone, Debug)]
pub struct SessionBinding {
    /// The origins this rule matches.
    pub matcher: OriginMatcher,
    /// The isolation policy used to derive the session id.
    pub isolation: IsolationPolicy,
    /// An explicit profile override (precedence step 1); `None` falls through to the instance/default.
    pub profile: Option<ProfileRef>,
    /// Where matched sessions' replies post.
    pub delivery: DeliveryPolicy,
}

impl SessionBinding {
    /// A binding that matches `matcher` and derives sessions under `isolation`, with origin-seeded
    /// delivery and no profile override (fall through to instance/default).
    pub fn new(matcher: OriginMatcher, isolation: IsolationPolicy) -> Self {
        Self {
            matcher,
            isolation,
            profile: None,
            delivery: DeliveryPolicy::FromOrigin,
        }
    }

    /// Set the per-binding profile override.
    pub fn with_profile(mut self, profile: ProfileRef) -> Self {
        self.profile = Some(profile);
        self
    }

    /// Pin the delivery target.
    pub fn with_delivery(mut self, delivery: DeliveryPolicy) -> Self {
        self.delivery = delivery;
        self
    }
}

/// A canonical, order-stable key for an inbound [`Origin`] used to index the chat→session pin map
/// (§5.9, I5). Keyed by transport instance + scope identity so a pin matches the same logical chat
/// regardless of ephemeral fields; mirrors the granularity [`session_id_for`] keys on.
pub fn origin_pin_key(origin: &Origin) -> String {
    let t = origin.transport.as_str();
    match &origin.scope {
        OriginScope::Dm { user } => format!("{t}\u{1}dm\u{1}{user}"),
        OriginScope::Group { chat, thread } => {
            format!(
                "{t}\u{1}group\u{1}{chat}\u{1}{}",
                thread.as_deref().unwrap_or("")
            )
        }
        OriginScope::Api { key } => format!("{t}\u{1}api\u{1}{key}"),
        OriginScope::Internal => format!("{t}\u{1}internal"),
        // `OriginScope` is `#[non_exhaustive]`: fall back to a transport-scoped debug key so a future
        // scope still produces a stable (if coarse) pin key rather than failing to compile/route.
        other => format!("{t}\u{1}other\u{1}{other:?}"),
    }
}

/// A resolve-first chat→session pin (§5.9, I5): an operator/GUI binding of a canonical origin to an
/// explicit session, overriding the deterministic [`session_id_for`] naming. An optional `profile`
/// override falls through to the deterministic precedence when `None`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChatPin {
    /// The session this origin is pinned to.
    pub session: SessionId,
    /// An explicit profile override; `None` falls through to instance/default precedence.
    pub profile: Option<ProfileRef>,
}

/// The outcome of routing an origin: the derived session, the profile to run it under (if any), and
/// where its replies post.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Resolved {
    /// The deterministic session id.
    pub session: SessionId,
    /// The profile to build the session's engine from; `None` = node active default.
    pub profile: Option<ProfileRef>,
    /// The `Primary` delivery target for the session.
    pub delivery: DeliveryTarget,
}

/// The host routing registry (§5.9): an ordered binding table, the per-instance default-profile map
/// (the account→profile baseline), and a node default. See the module docs for precedence.
#[derive(Clone, Debug, Default)]
pub struct RoutingRegistry {
    bindings: Vec<SessionBinding>,
    instance_profiles: HashMap<TransportId, ProfileRef>,
    default_profile: Option<ProfileRef>,
    /// Resolve-first chat→session pins (§5.9, I5), keyed by [`origin_pin_key`]. A pin overrides the
    /// deterministic `session_id_for` derivation for its origin. Layered onto a freshly-built
    /// registry by the host's hot-reload hook from the durable `chat_routes` store.
    pins: HashMap<String, ChatPin>,
}

impl RoutingRegistry {
    /// An empty registry (a pure passthrough: `PerThread` naming, no profile selection).
    pub fn new() -> Self {
        Self::default()
    }

    /// Append an ordered binding (first match wins at resolve time).
    pub fn with_binding(mut self, binding: SessionBinding) -> Self {
        self.bindings.push(binding);
        self
    }

    /// Bind a transport instance to a default profile (precedence step 2 — the account→profile
    /// baseline; a chat account's rooms all run this profile unless a binding overrides).
    pub fn bind_instance(mut self, transport: impl Into<TransportId>, profile: ProfileRef) -> Self {
        self.instance_profiles.insert(transport.into(), profile);
        self
    }

    /// Derive the account→profile baseline (precedence step 2) from profile data (§5.9.4): for every
    /// profile, each declared `bound_accounts` entry binds its transport instance to that profile.
    /// Any instance binding already present (e.g. an explicit config `[[routing.instance_profile]]`,
    /// installed via [`bind_instance`](Self::bind_instance)) is **kept** — the operator's config wins
    /// over a profile-declared binding.
    pub fn bind_instances_from_profiles(mut self, profiles: &[ProfileSpec]) -> Self {
        for profile in profiles {
            for account in &profile.bound_accounts {
                self.instance_profiles
                    .entry(TransportId::new(account.transport_instance.clone()))
                    .or_insert_with(|| ProfileRef::new(&profile.id));
            }
        }
        self
    }

    /// Set the node default profile (precedence step 3).
    pub fn with_default_profile(mut self, profile: ProfileRef) -> Self {
        self.default_profile = Some(profile);
        self
    }

    /// Install the resolve-first chat→session pin map (§5.9, I5), replacing any prior pins. Called by
    /// the host's hot-reload rebuild hook with the pins loaded from the durable `chat_routes` store.
    pub fn set_pins(&mut self, pins: HashMap<String, ChatPin>) {
        self.pins = pins;
    }

    /// Add a single chat→session pin (builder form; handy for tests).
    pub fn with_pin(mut self, origin: &Origin, pin: ChatPin) -> Self {
        self.pins.insert(origin_pin_key(origin), pin);
        self
    }

    /// Whether the registry carries no routing information at all.
    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
            && self.instance_profiles.is_empty()
            && self.default_profile.is_none()
            && self.pins.is_empty()
    }

    /// The deterministic profile precedence for an origin (binding override > transport-instance
    /// bound profile > node default), independent of session naming. Shared by the pinned and
    /// deterministic resolve paths.
    fn profile_for(&self, origin: &Origin, binding: Option<&SessionBinding>) -> Option<ProfileRef> {
        binding
            .and_then(|b| b.profile.clone())
            .or_else(|| self.instance_profiles.get(&origin.transport).cloned())
            .or_else(|| self.default_profile.clone())
    }

    /// Resolve an origin to `(session, profile, delivery)`. A chat→session **pin** is consulted first
    /// (§5.9, I5): when present it overrides the deterministic `session_id_for` naming for that
    /// origin (its profile falls through to the deterministic precedence when the pin carries none).
    /// Otherwise the first matching binding wins; with no matching binding, naming defaults to
    /// `PerThread` and delivery is seeded from the origin.
    pub fn resolve(&self, origin: &Origin) -> Resolved {
        let binding = self.bindings.iter().find(|b| b.matcher.matches(origin));
        // Resolve-first pin: an explicit chat→session binding overrides the deterministic id.
        if let Some(pin) = self.pins.get(&origin_pin_key(origin)) {
            let profile = pin
                .profile
                .clone()
                .or_else(|| self.profile_for(origin, binding));
            let delivery = match binding.map(|b| &b.delivery) {
                Some(DeliveryPolicy::Fixed(target)) => target.clone(),
                _ => origin.primary_target(),
            };
            return Resolved {
                session: pin.session.clone(),
                profile,
                delivery,
            };
        }
        let isolation = binding
            .map(|b| b.isolation)
            .unwrap_or(IsolationPolicy::PerThread);
        let session = session_id_for(origin, isolation);
        let profile = self.profile_for(origin, binding);
        let delivery = match binding.map(|b| &b.delivery) {
            Some(DeliveryPolicy::Fixed(target)) => target.clone(),
            _ => origin.primary_target(),
        };
        Resolved {
            session,
            profile,
            delivery,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use daemon_protocol::SinkKind;

    fn matrix(account: &str, room: &str) -> Origin {
        Origin::new(
            TransportId::new(format!("matrix/{account}")),
            OriginScope::Group {
                chat: room.to_string(),
                thread: None,
            },
        )
    }

    #[test]
    fn glob_matches_wildcards() {
        assert!(glob_match("*", "anything"));
        assert!(glob_match("#sec*", "#secops"));
        assert!(glob_match("*ops", "#secops"));
        assert!(glob_match("#a*c", "#abbbc"));
        assert!(!glob_match("#sec*", "#general"));
        assert!(glob_match("#exact", "#exact"));
        assert!(!glob_match("#exact", "#exact2"));
    }

    #[test]
    fn empty_registry_is_passthrough() {
        let reg = RoutingRegistry::new();
        assert!(reg.is_empty());
        let o = matrix("@ops:hs.org", "#general");
        let r = reg.resolve(&o);
        assert_eq!(r.session, session_id_for(&o, IsolationPolicy::PerThread));
        assert_eq!(r.profile, None);
        assert_eq!(r.delivery, o.primary_target());
        assert_eq!(r.delivery.kind, SinkKind::Primary);
    }

    #[test]
    fn precedence_binding_override_beats_instance_and_default() {
        let reg = RoutingRegistry::new()
            .with_default_profile(ProfileRef::new("node-default"))
            .bind_instance(
                TransportId::new("matrix/@ops:hs.org"),
                ProfileRef::new("ops-agent"),
            )
            .with_binding(
                SessionBinding::new(
                    OriginMatcher {
                        transport: TransportPattern::Exact(TransportId::new("matrix/@ops:hs.org")),
                        scope: ScopePattern::Group {
                            chat_glob: "#secops*".to_string(),
                        },
                    },
                    IsolationPolicy::PerChat,
                )
                .with_profile(ProfileRef::new("secops-agent")),
            );

        // Binding override wins for the matched room.
        let secops = reg.resolve(&matrix("@ops:hs.org", "#secops-alerts"));
        assert_eq!(secops.profile, Some(ProfileRef::new("secops-agent")));

        // Other rooms of the same account fall to the instance-bound profile (step 2).
        let general = reg.resolve(&matrix("@ops:hs.org", "#general"));
        assert_eq!(general.profile, Some(ProfileRef::new("ops-agent")));

        // A different account with no instance binding falls to the node default (step 3).
        let other = reg.resolve(&matrix("@help:hs.org", "#general"));
        assert_eq!(other.profile, Some(ProfileRef::new("node-default")));
    }

    #[test]
    fn family_matcher_spans_instances() {
        let reg = RoutingRegistry::new().with_binding(
            SessionBinding::new(
                OriginMatcher {
                    transport: TransportPattern::Family("matrix".to_string()),
                    scope: ScopePattern::Any,
                },
                IsolationPolicy::PerThread,
            )
            .with_profile(ProfileRef::new("any-matrix")),
        );
        assert_eq!(
            reg.resolve(&matrix("@a:hs", "#x")).profile,
            Some(ProfileRef::new("any-matrix"))
        );
        assert_eq!(
            reg.resolve(&matrix("@b:hs", "#y")).profile,
            Some(ProfileRef::new("any-matrix"))
        );
    }

    #[test]
    fn instances_derived_from_profile_bound_accounts() {
        use daemon_api::{BoundAccount, ProfileSpec, ProviderSelector};

        let ops =
            ProfileSpec::new("ops-agent", ProviderSelector::Mock, "m").with_bound_accounts(vec![
                BoundAccount::new("matrix/@ops:hs.org", "matrix/ops-agent/ops"),
            ]);
        let support = ProfileSpec::new("support", ProviderSelector::Mock, "m")
            .with_bound_accounts(vec![BoundAccount::new("matrix/@help:hs.org", "cred-help")]);

        // Profile-declared bindings fill the instance map (precedence step 2 from profile data).
        let reg = RoutingRegistry::new().bind_instances_from_profiles(&[ops.clone(), support]);
        assert_eq!(
            reg.resolve(&matrix("@ops:hs.org", "#general")).profile,
            Some(ProfileRef::new("ops-agent"))
        );
        assert_eq!(
            reg.resolve(&matrix("@help:hs.org", "#general")).profile,
            Some(ProfileRef::new("support"))
        );

        // An explicit config `bind_instance` for the same instance wins over the profile-derived one.
        let reg = RoutingRegistry::new()
            .bind_instance(
                TransportId::new("matrix/@ops:hs.org"),
                ProfileRef::new("config-override"),
            )
            .bind_instances_from_profiles(&[ops]);
        assert_eq!(
            reg.resolve(&matrix("@ops:hs.org", "#general")).profile,
            Some(ProfileRef::new("config-override"))
        );
    }

    #[test]
    fn pin_overrides_deterministic_session_id() {
        let o = matrix("@ops:hs.org", "#secops");
        let deterministic = session_id_for(&o, IsolationPolicy::PerThread);
        let pinned = SessionId::new("existing-conversation-7");

        let reg = RoutingRegistry::new()
            .with_default_profile(ProfileRef::new("node-default"))
            .with_pin(
                &o,
                ChatPin {
                    session: pinned.clone(),
                    profile: Some(ProfileRef::new("pinned-agent")),
                },
            );

        let r = reg.resolve(&o);
        // Resolve-first: the pin wins over `session_id_for` and the deterministic profile.
        assert_eq!(r.session, pinned);
        assert_ne!(r.session, deterministic);
        assert_eq!(r.profile, Some(ProfileRef::new("pinned-agent")));
        // Delivery still seeds from the origin.
        assert_eq!(r.delivery, o.primary_target());

        // An unpinned origin on the same registry falls back to the deterministic path.
        let other = matrix("@ops:hs.org", "#general");
        let r2 = reg.resolve(&other);
        assert_eq!(
            r2.session,
            session_id_for(&other, IsolationPolicy::PerThread)
        );
        assert_eq!(r2.profile, Some(ProfileRef::new("node-default")));
    }

    #[test]
    fn pin_without_profile_falls_through_to_precedence() {
        let o = matrix("@ops:hs.org", "#secops");
        let reg = RoutingRegistry::new()
            .bind_instance(
                TransportId::new("matrix/@ops:hs.org"),
                ProfileRef::new("ops-agent"),
            )
            .with_pin(
                &o,
                ChatPin {
                    session: SessionId::new("pinned"),
                    profile: None,
                },
            );
        let r = reg.resolve(&o);
        assert_eq!(r.session, SessionId::new("pinned"));
        // No pin profile => the instance-bound profile (precedence step 2) applies.
        assert_eq!(r.profile, Some(ProfileRef::new("ops-agent")));
    }

    #[test]
    fn first_matching_binding_wins() {
        let reg = RoutingRegistry::new()
            .with_binding(
                SessionBinding::new(
                    OriginMatcher {
                        transport: TransportPattern::Family("matrix".to_string()),
                        scope: ScopePattern::Group {
                            chat_glob: "#secops*".to_string(),
                        },
                    },
                    IsolationPolicy::PerChat,
                )
                .with_profile(ProfileRef::new("secops-agent")),
            )
            .with_binding(
                SessionBinding::new(OriginMatcher::any(), IsolationPolicy::PerThread)
                    .with_profile(ProfileRef::new("catch-all")),
            );
        assert_eq!(
            reg.resolve(&matrix("@ops:hs", "#secops-1")).profile,
            Some(ProfileRef::new("secops-agent"))
        );
        assert_eq!(
            reg.resolve(&matrix("@ops:hs", "#random")).profile,
            Some(ProfileRef::new("catch-all"))
        );
    }

    #[test]
    fn arc_swap_routing_reads_complete_snapshots_during_swaps() {
        let origin = matrix("@ops:hs", "#secops-1");
        let snapshots = std::sync::Arc::new(arc_swap::ArcSwap::from_pointee(
            RoutingRegistry::new().with_default_profile(ProfileRef::new("old")),
        ));

        let readers: Vec<_> = (0..8)
            .map(|_| {
                let snapshots = snapshots.clone();
                let origin = origin.clone();
                std::thread::spawn(move || {
                    for _ in 0..200 {
                        let profile = snapshots.load().resolve(&origin).profile;
                        assert!(
                            profile == Some(ProfileRef::new("old"))
                                || profile == Some(ProfileRef::new("new"))
                        );
                    }
                })
            })
            .collect();

        for _ in 0..50 {
            snapshots.store(std::sync::Arc::new(
                RoutingRegistry::new().with_default_profile(ProfileRef::new("new")),
            ));
            snapshots.store(std::sync::Arc::new(
                RoutingRegistry::new().with_default_profile(ProfileRef::new("old")),
            ));
        }

        for reader in readers {
            reader.join().unwrap();
        }
    }
}
