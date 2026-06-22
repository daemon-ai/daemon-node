//! Host-level bidirectional routing (daemon-event-io-spec §5.9).
//!
//! The routing registry maps an inbound [`Origin`] to the `(SessionId, profile, DeliveryTarget)` the
//! host binds a live session to. It sits on top of [`session_id_for`] (which keeps owning *naming*)
//! and adds the new degree of freedom: **agent selection** — which profile runs the derived session.
//!
//! Resolution precedence (event-io spec decision 10):
//! 1. an explicit per-binding `profile` override,
//! 2. else the transport-instance's bound profile (the account→profile baseline — the
//!    [`instance_profiles`](RoutingRegistry) map, filled from credential bindings in a later phase),
//! 3. else the node default profile (`None` resolves the node's active default at build time,
//!    preserving the legacy single-profile behavior).
//!
//! Outbound is the symmetric half: the registry also answers *where* a matched session's replies post
//! (its [`DeliveryTarget`]), auto-seeded as the inverse of the opening origin unless a binding pins it.

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

    /// Set the node default profile (precedence step 3).
    pub fn with_default_profile(mut self, profile: ProfileRef) -> Self {
        self.default_profile = Some(profile);
        self
    }

    /// Whether the registry carries no routing information at all.
    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
            && self.instance_profiles.is_empty()
            && self.default_profile.is_none()
    }

    /// Resolve an origin to `(session, profile, delivery)`. The first matching binding wins; with no
    /// matching binding, naming defaults to `PerThread` and delivery is seeded from the origin.
    pub fn resolve(&self, origin: &Origin) -> Resolved {
        let binding = self.bindings.iter().find(|b| b.matcher.matches(origin));
        let isolation = binding
            .map(|b| b.isolation)
            .unwrap_or(IsolationPolicy::PerThread);
        let session = session_id_for(origin, isolation);
        // Precedence: binding override > transport-instance bound profile > node default.
        let profile = binding
            .and_then(|b| b.profile.clone())
            .or_else(|| self.instance_profiles.get(&origin.transport).cloned())
            .or_else(|| self.default_profile.clone());
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
}
