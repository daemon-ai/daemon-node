// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `PersonManager` — the node-side person/metacontact registry (work package W3-J), ported from
//! the person half of libpurple's `purplecontactmanager.c` (`manager->people`).
//!
//! It owns the live collection of [`Person`]s behind the person surface
//! ([`ControlApi::person_list`](daemon_api::ControlApi::person_list) / the
//! [`NodeEvent::PersonsChanged`](daemon_api::NodeEvent) pointer): create/remove persons,
//! associate/dissociate contact endpoints (the person ↔ contact edges), and lookup by id or by
//! contact endpoint. Like [`crate::notifications::NotificationManager`] (and unlike the
//! store-backed [`crate::presence::PresenceManager`]) it is a plain in-memory collection whose
//! mutations return typed outcomes — there are no GObject signals, so the manager reports
//! transitions by value and the node emits the change pointer.
//!
//! C semantics mirrored:
//! - `purple_contact_manager_add_person` bails when the person is already known → a double-add is
//!   [`AddOutcome::DuplicateRejected`] (keyed by `Person::id`; C keys by pointer identity).
//! - `purple_contact_manager_remove_person(person, remove_contacts)` is a no-op when unknown, and
//!   with `remove_contacts=TRUE` also removes the person's contacts → `remove_person(id, true)`
//!   strips the person's endpoints so no endpoint lookup resolves to it afterwards.
//! - `purple_contact_manager_add` auto-adds a contact's person when the contact already carries one
//!   → [`PersonManager::associate`] with an unknown person id is rejected, while
//!   [`PersonManager::add_person`] with a pre-endpointed person carries its edges in.

use daemon_api::{Person, PersonEndpoint};
use daemon_protocol::TransportId;

/// The outcome of [`PersonManager::add_person`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AddOutcome {
    /// The person was added.
    Added,
    /// A person with the same id already exists; the add was rejected
    /// (`purple_contact_manager_add_person`'s already-known bail, modeled as a rejected no-op).
    DuplicateRejected,
}

/// The live person registry (← the `people` collection of `PurpleContactManager`).
#[derive(Debug, Default)]
pub struct PersonManager {
    persons: Vec<Person>,
}

impl PersonManager {
    /// An empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// The number of persons.
    pub fn len(&self) -> usize {
        self.persons.len()
    }

    /// Whether the registry holds no persons.
    pub fn is_empty(&self) -> bool {
        self.persons.is_empty()
    }

    /// A snapshot of the persons (insertion order) — what the `person_list` op returns.
    pub fn list(&self) -> Vec<Person> {
        self.persons.clone()
    }

    /// Add a person (← `purple_contact_manager_add_person`): mints an id when empty
    /// ([`Person::ensure_id`]), rejects a double-add by id, else appends. A person arriving with
    /// endpoints carries its edges in (the C "contact already has a person" auto-add path).
    pub fn add_person(&mut self, mut person: Person) -> AddOutcome {
        person.ensure_id();
        if self.persons.iter().any(|p| p.id == person.id) {
            return AddOutcome::DuplicateRejected;
        }
        self.persons.push(person);
        AddOutcome::Added
    }

    /// Remove a person by id (← `purple_contact_manager_remove_person`). With
    /// `remove_endpoints=true` the person's contact endpoints are stripped as part of removal (the
    /// C `remove_contacts=TRUE` path, which also removes the person's contacts from the manager).
    /// Returns whether one was removed (a second remove is a no-op).
    pub fn remove_person(&mut self, id: &str, remove_endpoints: bool) -> bool {
        let Some(position) = self.persons.iter().position(|p| p.id == id) else {
            return false;
        };
        let mut removed = self.persons.remove(position);
        if remove_endpoints {
            removed.remove_all_endpoints();
        }
        true
    }

    /// Associate a contact endpoint with an existing person (the person ↔ contact edge;
    /// ← `purple_person_add_contact_info` under the manager). Rejected (`false`) when the person id
    /// is unknown or the `(transport, contact-id)` edge already exists on it.
    pub fn associate(&mut self, person_id: &str, endpoint: PersonEndpoint) -> bool {
        let Some(person) = self.persons.iter_mut().find(|p| p.id == person_id) else {
            return false;
        };
        if person
            .endpoints
            .iter()
            .any(|e| e.addresses(&endpoint.transport, &endpoint.contact.id))
        {
            return false;
        }
        person.add_endpoint(endpoint);
        true
    }

    /// Dissociate a contact endpoint from a person (← `purple_person_remove_contact_info` under the
    /// manager). Returns whether the edge existed (a second dissociate is a no-op).
    pub fn dissociate(
        &mut self,
        person_id: &str,
        transport: &TransportId,
        contact_id: &str,
    ) -> bool {
        let Some(person) = self.persons.iter_mut().find(|p| p.id == person_id) else {
            return false;
        };
        person.remove_endpoint(transport, contact_id)
    }

    /// Find a person by id.
    pub fn find_person(&self, id: &str) -> Option<&Person> {
        self.persons.iter().find(|p| p.id == id)
    }

    /// Find the person holding the `(transport, contact-id)` endpoint, if any (the reverse edge —
    /// ← `purple_contact_info_get_person` from the contact side).
    pub fn find_by_endpoint(&self, transport: &TransportId, contact_id: &str) -> Option<&Person> {
        self.persons.iter().find(|p| {
            p.endpoints
                .iter()
                .any(|e| e.addresses(transport, contact_id))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use daemon_api::ContactInfo;

    fn transport() -> TransportId {
        TransportId::new("test")
    }

    fn endpoint(id: &str) -> PersonEndpoint {
        PersonEndpoint::new(
            transport(),
            ContactInfo {
                id: id.to_string(),
                ..Default::default()
            },
        )
    }

    // /contact-manager/person/add-remove
    #[test]
    fn manager_person_add_remove() {
        let mut m = PersonManager::new();
        assert_eq!(m.len(), 0);

        let person = Person::new(Some("p1".into()));
        assert_eq!(m.add_person(person), AddOutcome::Added);
        assert_eq!(m.len(), 1);

        assert!(m.remove_person("p1", false));
        assert_eq!(m.len(), 0);
    }

    // D5: double-add rejected by id.
    #[test]
    fn manager_person_double_add() {
        let mut m = PersonManager::new();
        let person = Person::new(Some("p1".into()));
        assert_eq!(m.add_person(person.clone()), AddOutcome::Added);
        assert_eq!(m.add_person(person), AddOutcome::DuplicateRejected);
        assert_eq!(m.len(), 1);
    }

    // D5: double-remove is a no-op.
    #[test]
    fn manager_person_double_remove() {
        let mut m = PersonManager::new();
        m.add_person(Person::new(Some("p1".into())));
        assert!(m.remove_person("p1", false));
        assert!(!m.remove_person("p1", false));
        assert!(m.is_empty());
    }

    // /contact-manager/person/add-via-contact-remove-person-with-contacts
    #[test]
    fn manager_person_remove_with_contacts() {
        let mut m = PersonManager::new();

        // A person arriving WITH an endpoint carries its edge in (the C "contact already has a
        // person -> the person is added too" path).
        let mut person = Person::new(Some("p1".into()));
        person.add_endpoint(endpoint("foo"));
        assert_eq!(m.add_person(person), AddOutcome::Added);
        assert_eq!(m.len(), 1);
        assert!(m.find_by_endpoint(&transport(), "foo").is_some());

        // remove_endpoints=true removes the person AND its endpoints (the C remove_contacts=TRUE):
        // afterwards neither the person nor the endpoint resolves.
        assert!(m.remove_person("p1", true));
        assert_eq!(m.len(), 0);
        assert!(m.find_person("p1").is_none());
        assert!(m.find_by_endpoint(&transport(), "foo").is_none());
    }

    // D6: associate/dissociate double edges.
    #[test]
    fn manager_associate_dissociate_edges() {
        let mut m = PersonManager::new();
        m.add_person(Person::new(Some("p1".into())));

        // Associate onto an unknown person is rejected.
        assert!(!m.associate("nope", endpoint("foo")));

        // First associate succeeds; the same edge again is rejected.
        assert!(m.associate("p1", endpoint("foo")));
        assert!(!m.associate("p1", endpoint("foo")));
        assert_eq!(m.find_person("p1").expect("person").endpoints.len(), 1);

        // Dissociate removes the edge; a second dissociate is a no-op.
        assert!(m.dissociate("p1", &transport(), "foo"));
        assert!(!m.dissociate("p1", &transport(), "foo"));
        assert_eq!(m.find_person("p1").expect("person").endpoints.len(), 0);
    }

    // D7: lookup by id and by endpoint.
    #[test]
    fn manager_lookup_by_id_and_endpoint() {
        let mut m = PersonManager::new();
        let mut ada = Person::new(Some("ada".into()));
        ada.alias = Some("Ada".into());
        m.add_person(ada);
        m.add_person(Person::new(Some("bob".into())));
        m.associate("ada", endpoint("@ada:hs.org"));
        m.associate(
            "ada",
            PersonEndpoint::new(
                TransportId::new("discord"),
                ContactInfo {
                    id: "ada#1234".into(),
                    ..Default::default()
                },
            ),
        );

        // By id.
        assert_eq!(
            m.find_person("ada").expect("ada").alias.as_deref(),
            Some("Ada")
        );
        assert!(m.find_person("carol").is_none());

        // By endpoint — either transport edge resolves to the same person.
        assert_eq!(
            m.find_by_endpoint(&transport(), "@ada:hs.org")
                .expect("by matrix endpoint")
                .id,
            "ada"
        );
        assert_eq!(
            m.find_by_endpoint(&TransportId::new("discord"), "ada#1234")
                .expect("by discord endpoint")
                .id,
            "ada"
        );
        // A known contact id on the WRONG transport does not resolve.
        assert!(m
            .find_by_endpoint(&TransportId::new("discord"), "@ada:hs.org")
            .is_none());
        assert!(m.find_by_endpoint(&transport(), "nobody").is_none());
    }

    // An id-less person gets one minted on add (mirrors SavedPresence/ensure_id discipline).
    #[test]
    fn manager_add_mints_missing_id() {
        let mut m = PersonManager::new();
        assert_eq!(m.add_person(Person::default()), AddOutcome::Added);
        let list = m.list();
        assert_eq!(list.len(), 1);
        assert!(!list[0].id.is_empty(), "an empty id is minted on add");
    }
}
