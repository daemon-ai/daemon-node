// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The `Person`/MetaContact model ported from libpurple (work package W3-J) — the one concept the
//! transport-adapter spec explicitly deferred, now un-deferred.
//!
//! A [`Person`] is "the same human across transports": an auto-identified association carrying an
//! optional user alias, an optional [`Image`] avatar, and a set of contact [`endpoints`](Person::endpoints)
//! (each binding a [`TransportId`] to the [`ContactInfo`] the person is reachable at on that
//! transport). Ported from `purpleperson.c` (`PurplePerson` holds a `GPtrArray` of
//! `PurpleContactInfo`; the daemon holds `Vec<PersonEndpoint>`).
//!
//! Like [`crate::notify`] / [`crate::saved_presence`] this **touches the wire**: [`Person`] and
//! [`PersonEndpoint`] are reachable from [`ApiResponse::Persons`](crate::ApiResponse), so they are
//! serde types mirrored in `daemon-api.cddl` and derive feature-gated [`arbitrary::Arbitrary`].
//!
//! The priority-contact algorithm ([`Person::preferred_endpoint`]) is `purple_person`'s exactly:
//! sort the contacts by [`Presence::compare`](crate::Presence::compare) (from `src/details.rs`) and
//! take the best — the endpoint a UI messages by default.

use crate::matching::str_matches;
use crate::{ContactInfo, Image};
use daemon_protocol::TransportId;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Process-unique counter mixed into generated ids so two mints in the same nanosecond differ.
static ID_SALT: AtomicU64 = AtomicU64::new(0);

/// One transport-scoped contact endpoint of a [`Person`] (← an entry in `PurplePerson`'s contacts
/// list, which the daemon keys by the account/transport it lives on). Binds a [`TransportId`] to the
/// [`ContactInfo`] the person is reachable at there (the contact's presence rides on the
/// [`ContactInfo`], so it drives [`Person::preferred_endpoint`]).
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PersonEndpoint {
    /// The transport/account this endpoint lives on.
    pub transport: TransportId,
    /// The contact id/handle + presence the person is reachable at on `transport`.
    pub contact: ContactInfo,
}

impl PersonEndpoint {
    /// A fresh endpoint binding `contact` on `transport`.
    pub fn new(transport: TransportId, contact: ContactInfo) -> Self {
        Self { transport, contact }
    }

    /// Whether this endpoint addresses `(transport, contact_id)` — the identity edges/lookups key on.
    pub fn addresses(&self, transport: &TransportId, contact_id: &str) -> bool {
        self.transport == *transport && self.contact.id == contact_id
    }
}

/// A person / metacontact (← `PurplePerson`): an auto-identified, optionally-aliased,
/// optionally-avatared grouping of contact [`endpoints`](Person::endpoints) across transports.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Person {
    /// The stable identifier (auto-minted by [`Person::new`] / [`Person::ensure_id`] when empty,
    /// mirroring `purple_person_set_id`'s `g_uuid_string_random`).
    pub id: String,
    /// The user-controlled alias, when set (← `PurplePerson:alias`).
    #[serde(default)]
    pub alias: Option<String>,
    /// The user-controlled avatar, when set (← `PurplePerson:avatar`).
    #[serde(default)]
    pub avatar: Option<Image>,
    /// The contact endpoints this person is reachable at (← `PurplePerson`'s contacts list).
    #[serde(default)]
    pub endpoints: Vec<PersonEndpoint>,
}

impl Person {
    /// A new person with a supplied or auto-minted non-empty `id` (← `purple_person_new` +
    /// `constructed`, which mints a random id when none was given).
    pub fn new(id: Option<String>) -> Self {
        Self {
            id: id.filter(|s| !s.is_empty()).unwrap_or_else(Self::gen_id),
            ..Self::default()
        }
    }

    /// Ensure the id is set, minting a fresh one if it is empty (← `purple_person_constructed`,
    /// which mints only when `birb_str_is_empty(id)`).
    pub fn ensure_id(&mut self) {
        if self.id.is_empty() {
            self.id = Self::gen_id();
        }
    }

    /// Mint a fresh unique id (nanosecond clock + process-unique counter, UUID-shaped — no external
    /// `uuid` dep, mirroring [`crate::SavedPresence`]).
    fn gen_id() -> String {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let salt = ID_SALT.fetch_add(1, Ordering::Relaxed);
        let hi = (nanos as u64) ^ (salt << 1);
        let lo = ((nanos >> 64) as u64).wrapping_add(salt);
        format!(
            "{:08x}-{:04x}-{:04x}-{:04x}-{:012x}",
            hi & 0xffff_ffff,
            (hi >> 32) & 0xffff,
            (hi >> 48) & 0xffff,
            lo & 0xffff,
            (lo >> 16) & 0xffff_ffff_ffff,
        )
    }

    /// Add a contact endpoint (← `purple_person_add_contact_info`, which appends).
    pub fn add_endpoint(&mut self, endpoint: PersonEndpoint) {
        self.endpoints.push(endpoint);
    }

    /// Remove the endpoint addressing `(transport, contact_id)`
    /// (← `purple_person_remove_contact_info`); returns whether one was removed (a second remove is
    /// a no-op — `g_ptr_array_find` misses).
    pub fn remove_endpoint(&mut self, transport: &TransportId, contact_id: &str) -> bool {
        let Some(position) = self
            .endpoints
            .iter()
            .position(|e| e.addresses(transport, contact_id))
        else {
            return false;
        };
        self.endpoints.remove(position);
        true
    }

    /// Remove every endpoint (← `purple_person_remove_all_contact_infos`).
    pub fn remove_all_endpoints(&mut self) {
        self.endpoints.clear();
    }

    /// Whether the person has any endpoints (← `purple_person_has_contacts`).
    pub fn has_contacts(&self) -> bool {
        !self.endpoints.is_empty()
    }

    /// The priority (preferred) contact endpoint (← `purple_person_get_priority_contact_info`): the
    /// endpoint whose contact presence is best under [`Presence::compare`](crate::Presence::compare)
    /// (the exact comparator of `purple_person_contact_compare`), ties resolved to the first in
    /// insertion order (the C sort + index-0). `None` when the person has no endpoints.
    pub fn preferred_endpoint(&self) -> Option<&PersonEndpoint> {
        self.endpoints.iter().reduce(|best, candidate| {
            // Replace only on a STRICT win so ties keep the earlier endpoint (stable, index-0).
            if candidate
                .contact
                .presence
                .compare(&best.contact.presence)
                .is_lt()
            {
                candidate
            } else {
                best
            }
        })
    }

    /// The name to display for this person (← `purple_person_get_name_for_display`): the alias when
    /// non-empty, else the preferred endpoint's contact name-for-display, else `None`.
    pub fn name_for_display(&self) -> Option<&str> {
        if let Some(alias) = self.alias.as_deref() {
            if !alias.is_empty() {
                return Some(alias);
            }
        }
        self.preferred_endpoint()
            .map(|endpoint| endpoint.contact.name_for_display())
    }

    /// The avatar to display for this person (← `purple_person_get_avatar_for_display`): the person's
    /// own avatar when set. (The daemon `ContactInfo` has no avatar, so there is no contact-level
    /// fallback — see the ledger.)
    pub fn avatar_for_display(&self) -> Option<&Image> {
        self.avatar.as_ref()
    }

    /// Whether the person matches `needle` (← `purple_person_matches`): an empty/`None` needle
    /// matches; else a caseless subsequence match against the alias
    /// (`birb_str_matches`), then any endpoint's contact ([`ContactInfo::matches`]).
    pub fn matches(&self, needle: Option<&str>) -> bool {
        let needle = match needle {
            None | Some("") => return true,
            Some(needle) => needle,
        };
        if let Some(alias) = self.alias.as_deref() {
            if !alias.is_empty() && str_matches(needle, alias) {
                return true;
            }
        }
        self.endpoints
            .iter()
            .any(|endpoint| endpoint.contact.matches(Some(needle)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ContactPermission, Presence, PresencePrimitive};

    fn transport() -> TransportId {
        TransportId::new("test")
    }

    fn contact(id: &str) -> ContactInfo {
        ContactInfo {
            id: id.to_string(),
            ..Default::default()
        }
    }

    fn endpoint(id: &str) -> PersonEndpoint {
        PersonEndpoint::new(transport(), contact(id))
    }

    fn endpoint_with_presence(id: &str, primitive: PresencePrimitive) -> PersonEndpoint {
        PersonEndpoint::new(
            transport(),
            ContactInfo {
                id: id.to_string(),
                presence: Presence {
                    primitive,
                    ..Default::default()
                },
                permission: ContactPermission::Unset,
                ..Default::default()
            },
        )
    }

    // -- construction (test_person.c /new + derived id mint) ----------------

    #[test]
    fn person_new() {
        // /person/new: a constructed person exists (auto-minted id).
        let p = Person::new(None);
        assert!(!p.id.is_empty());
        assert!(p.endpoints.is_empty());
    }

    #[test]
    fn person_generates_id() {
        // D1: id auto-minted, unique, and ensure_id idempotent once set.
        let p = Person::new(None);
        assert!(!p.id.is_empty(), "id must be auto-generated");
        let q = Person::new(None);
        assert_ne!(p.id, q.id, "each mint is unique");
        // A supplied id is preserved.
        let r = Person::new(Some("supplied".into()));
        assert_eq!(r.id, "supplied");
        // ensure_id mints only when empty.
        let mut s = Person::default();
        assert!(s.id.is_empty());
        s.ensure_id();
        assert!(!s.id.is_empty());
        let minted = s.id.clone();
        s.ensure_id();
        assert_eq!(s.id, minted);
    }

    // -- /person/properties -------------------------------------------------

    #[test]
    fn person_properties() {
        // /person/properties: the modeled subset (id/alias/avatar/avatar-for-display/
        // name-for-display) round-trips. color/color-for-display/tags are not modeled (see ledger).
        let mut p = Person::new(Some("id1".into()));
        p.alias = Some("alias".into());
        assert_eq!(p.id, "id1");
        assert_eq!(p.alias.as_deref(), Some("alias"));
        // name-for-display resolves to the alias.
        assert_eq!(p.name_for_display(), Some("alias"));
    }

    // -- /person/avatar-for-display/person ----------------------------------

    #[test]
    fn person_avatar_for_display_person() {
        // /person/avatar-for-display/person: the person's avatar overrides (there is no contact
        // avatar on the daemon model, so it is simply the person's avatar).
        use crate::BlobRef;
        let mut p = Person::new(None);
        assert!(p.avatar_for_display().is_none());
        let avatar = Image {
            blob: BlobRef::new(daemon_common::ContentHash::new([7u8; 32]), 3),
        };
        p.avatar = Some(avatar.clone());
        p.add_endpoint(endpoint("id"));
        assert_eq!(p.avatar_for_display(), Some(&avatar));
    }

    // -- /person/name-for-display/{person,contact} --------------------------

    #[test]
    fn person_name_for_display_person() {
        // /person/name-for-display/person: alias overrides the contact.
        let mut p = Person::new(None);
        p.alias = Some("person-alias".into());
        p.add_endpoint(endpoint("id"));
        assert_eq!(p.name_for_display(), Some("person-alias"));
    }

    #[test]
    fn person_name_for_display_contact() {
        // /person/name-for-display/contact: no alias -> the preferred contact's name-for-display (id).
        let mut p = Person::new(None);
        p.add_endpoint(endpoint("id"));
        assert_eq!(p.name_for_display(), Some("id"));
    }

    // -- /person/contacts/{single,multiple} ---------------------------------

    #[test]
    fn person_contacts_single() {
        // /person/contacts/single: add one endpoint (n_items 1), remove it (n_items 0).
        let mut p = Person::new(None);
        assert!(!p.has_contacts());
        assert_eq!(p.endpoints.len(), 0);
        p.add_endpoint(endpoint("id"));
        assert!(p.has_contacts());
        assert_eq!(p.endpoints.len(), 1);
        assert!(p.remove_endpoint(&transport(), "id"));
        assert_eq!(p.endpoints.len(), 0);
        assert!(!p.has_contacts());
    }

    #[test]
    fn person_contacts_multiple() {
        // /person/contacts/multiple: add 5, remove 5.
        let mut p = Person::new(None);
        for i in 0..5 {
            assert_eq!(p.endpoints.len(), i);
            p.add_endpoint(endpoint(&format!("username{i}")));
            assert_eq!(p.endpoints.len(), i + 1);
        }
        for i in 0..5 {
            assert_eq!(p.endpoints.len(), 5 - i);
            assert!(p.remove_endpoint(&transport(), &format!("username{i}")));
            assert_eq!(p.endpoints.len(), 5 - (i + 1));
        }
        assert!(!p.has_contacts());
    }

    #[test]
    fn person_endpoint_double_edges() {
        // D3: remove of an addressed endpoint returns true; a second remove is a no-op (false).
        let mut p = Person::new(None);
        p.add_endpoint(endpoint("id"));
        assert!(p.remove_endpoint(&transport(), "id"));
        assert!(!p.remove_endpoint(&transport(), "id"));
        // remove_all clears everything.
        p.add_endpoint(endpoint("a"));
        p.add_endpoint(endpoint("b"));
        assert_eq!(p.endpoints.len(), 2);
        p.remove_all_endpoints();
        assert!(!p.has_contacts());
    }

    // -- /person/priority/{single,multiple-with-change} + empty ------------

    #[test]
    fn person_priority_empty_none() {
        // D2: an empty person has no priority endpoint.
        let p = Person::new(None);
        assert!(p.preferred_endpoint().is_none());
    }

    #[test]
    fn person_priority_single() {
        // /person/priority/single: one contact, set available -> it is the priority.
        let mut p = Person::new(None);
        p.add_endpoint(endpoint_with_presence("id", PresencePrimitive::Available));
        let pref = p.preferred_endpoint().expect("priority");
        assert_eq!(pref.contact.id, "id");
    }

    #[test]
    fn person_priority_multiple_with_change() {
        // /person/priority/multiple-with-change: 5 offline, first added is priority; flip the
        // (n-2)th to available -> it becomes the priority.
        let mut p = Person::new(None);
        for i in 1..=5 {
            p.add_endpoint(endpoint_with_presence(
                &format!("username{i}"),
                PresencePrimitive::Offline,
            ));
        }
        // All offline -> the first added stays priority (stable/index-0).
        assert_eq!(
            p.preferred_endpoint().expect("priority").contact.id,
            "username1"
        );
        // Flip the 4th (index 3, the n-2th) to available.
        p.endpoints[3].contact.presence.primitive = PresencePrimitive::Available;
        assert_eq!(
            p.preferred_endpoint().expect("priority").contact.id,
            "username4"
        );
    }

    // -- /person/matches/* --------------------------------------------------

    #[test]
    fn person_matches_accepts_null() {
        assert!(Person::new(None).matches(None));
    }

    #[test]
    fn person_matches_empty_string() {
        assert!(Person::new(None).matches(Some("")));
    }

    #[test]
    fn person_matches_alias() {
        let mut p = Person::new(None);
        p.alias = Some("this is the alias".into());
        assert!(p.matches(Some("the")));
        assert!(!p.matches(Some("what")));
    }

    #[test]
    fn person_matches_contact_info() {
        let mut p = Person::new(None);
        p.add_endpoint(endpoint("user1"));
        assert!(p.matches(Some("user1")));
        assert!(!p.matches(Some("nobody")));
    }

    // -- wire round-trip (daemon-native) ------------------------------------

    #[test]
    fn person_cbor_round_trips() {
        // D4: a person with an aliased, multi-endpoint shape round-trips through CBOR.
        let mut p = Person::new(Some("p1".into()));
        p.alias = Some("Ada".into());
        p.add_endpoint(endpoint_with_presence(
            "@ada:hs.org",
            PresencePrimitive::Available,
        ));
        p.add_endpoint(endpoint("ada#1234"));
        let bytes = crate::to_cbor(&p);
        let back: Person = crate::from_cbor(&bytes).unwrap();
        assert_eq!(p, back);
    }
}
