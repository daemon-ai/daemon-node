//! Display-name, matching, ordering, and member-collection logic.
//!
//! Ported from libpurple (`purplecontactinfo.c`, `purpleconversationmember.c`,
//! `purpleconversationmembers.c`, `purpleconversationmanager.c`). This is host-side derived logic
//! over the existing DTOs — it adds no wire-contract surface. Documented divergences: person
//! precedence lands as the `*_with_person` layer below; badges map to [`MemberRole`]; UTF-8
//! collation is approximated by casefolded codepoint order.

use crate::{
    ContactInfo, ConversationInfo, ConversationMember, ConversationType, MemberRole, Person,
    TypingState,
};
use daemon_protocol::TransportId;
use std::cmp::Ordering;

// ===========================================================================
// String primitives (← birb + purple_utf8_strcasecmp)
// ===========================================================================

/// Port of `birb_str_matches(pattern, str)`: TRUE when `pattern` occurs in sequential order within
/// `haystack`, caseless, ignoring characters in between (a caseless *subsequence* match — e.g.
/// `Br` matches `biRb`). Casefolding is Unicode lowercase (approximates `g_utf8_casefold`).
///
/// `pub(crate)` so the sibling saved-presence port ([`crate::saved_presence`]) reuses the exact
/// birb matcher rather than duplicating it (W2-F).
pub(crate) fn str_matches(pattern: &str, haystack: &str) -> bool {
    let mut pat = pattern.chars().flat_map(char::to_lowercase).peekable();
    // An empty pattern is a subsequence of anything.
    if pat.peek().is_none() {
        return true;
    }
    for hc in haystack.chars().flat_map(char::to_lowercase) {
        if pat.peek() == Some(&hc) {
            pat.next();
            if pat.peek().is_none() {
                return true;
            }
        }
    }
    pat.peek().is_none()
}

/// Port of `purple_utf8_strcasecmp` for non-NULL operands: casefold (Unicode lowercase) then
/// codepoint order. **Divergence:** approximates `g_utf8_collate` — identical for ASCII inputs.
fn utf8_strcasecmp(a: &str, b: &str) -> Ordering {
    let a_fold: String = a.chars().flat_map(char::to_lowercase).collect();
    let b_fold: String = b.chars().flat_map(char::to_lowercase).collect();
    a_fold.cmp(&b_fold)
}

/// Sort weight for a role: a higher weight means "more standing", which sorts *first* (the daemon
/// analog of libpurple's "more/higher badges sorts first" in `purple_badges_compare`).
fn role_rank(role: MemberRole) -> u8 {
    match role {
        MemberRole::None => 0,
        MemberRole::Voice => 1,
        MemberRole::HalfOp => 2,
        MemberRole::Op => 3,
        MemberRole::Founder => 4,
    }
}

// ===========================================================================
// ContactInfo
// ===========================================================================

impl ContactInfo {
    /// Port of `purple_contact_info_get_name_for_display`. The libpurple chain is
    /// `alias → person-alias → display_name → id`; the daemon `ContactInfo` has neither an alias
    /// nor a person field (those live on [`ConversationMember`] / [`Person`]), so the chain reduces
    /// to `display_name → id`. The person-aware layer is
    /// [`name_for_display_with_person`](ContactInfo::name_for_display_with_person) (W3-J).
    pub fn name_for_display(&self) -> &str {
        if let Some(display_name) = self.display_name.as_deref() {
            if !display_name.is_empty() {
                return display_name;
            }
        }
        &self.id
    }

    /// Port of `purple_contact_info_matches`. `None`/empty needle matches; otherwise a caseless
    /// subsequence match against `id` then `display_name`.
    pub fn matches(&self, needle: Option<&str>) -> bool {
        let needle = match needle {
            None | Some("") => return true,
            Some(needle) => needle,
        };
        if !self.id.is_empty() && str_matches(needle, &self.id) {
            return true;
        }
        if let Some(display_name) = self.display_name.as_deref() {
            if !display_name.is_empty() && str_matches(needle, display_name) {
                return true;
            }
        }
        false
    }

    /// Non-NULL ordering by name-for-display (person-blind; the person-aware variant is
    /// [`contact_info_compare_with_person`]). Suitable for `slice::sort_by`. See
    /// [`contact_info_compare`] for the NULL-safe variant.
    pub fn cmp_for_display(&self, other: &ContactInfo) -> Ordering {
        utf8_strcasecmp(self.name_for_display(), other.name_for_display())
    }
}

impl ContactInfo {
    /// The person-aware name-for-display (W3-J): the full libpurple chain
    /// `contact-alias → person-alias → display_name → id`. The daemon `ContactInfo` has no
    /// contact-alias field, so the effective chain is `person-alias → display_name → id` — the
    /// person's alias (when the contact is associated with a [`Person`]) is inserted ahead of the
    /// contact's own `display_name → id`. `None` (no person) reduces to [`ContactInfo::name_for_display`].
    pub fn name_for_display_with_person<'a>(&'a self, person: Option<&'a Person>) -> &'a str {
        if let Some(alias) = person.and_then(|p| p.alias.as_deref()) {
            if !alias.is_empty() {
                return alias;
            }
        }
        self.name_for_display()
    }
}

/// Person-aware port of `purple_contact_info_compare` (W3-J): the NULL rules, then the
/// person tier (a contact WITH an associated [`Person`] sorts before one without —
/// `purplecontactinfo.c` `person_a != NULL && person_b == NULL → -1`), then person-aware
/// name-for-display caseless. Each side is `(contact, its optional person)`.
pub fn contact_info_compare_with_person(
    a: Option<(&ContactInfo, Option<&Person>)>,
    b: Option<(&ContactInfo, Option<&Person>)>,
) -> Ordering {
    match (a, b) {
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
        (Some((a, person_a)), Some((b, person_b))) => {
            match (person_a.is_some(), person_b.is_some()) {
                (true, false) => Ordering::Less,
                (false, true) => Ordering::Greater,
                _ => utf8_strcasecmp(
                    a.name_for_display_with_person(person_a),
                    b.name_for_display_with_person(person_b),
                ),
            }
        }
    }
}

/// Port of `purple_contact_info_compare` including the NULL rules
/// (`Some,None → Less`, `None,Some → Greater`, `None,None → Equal`).
pub fn contact_info_compare(a: Option<&ContactInfo>, b: Option<&ContactInfo>) -> Ordering {
    match (a, b) {
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
        (Some(a), Some(b)) => a.cmp_for_display(b),
    }
}

/// Port of `purple_contact_info_equal` (`compare == 0`, NULL-safe).
pub fn contact_info_equal(a: Option<&ContactInfo>, b: Option<&ContactInfo>) -> bool {
    contact_info_compare(a, b) == Ordering::Equal
}

// ===========================================================================
// ConversationMember
// ===========================================================================

impl ConversationMember {
    /// Port of `purple_conversation_member_get_name_for_display`: `alias → nickname →
    /// contact.name_for_display`.
    pub fn name_for_display(&self) -> &str {
        if let Some(alias) = self.alias.as_deref() {
            if !alias.is_empty() {
                return alias;
            }
        }
        if let Some(nickname) = self.nickname.as_deref() {
            if !nickname.is_empty() {
                return nickname;
            }
        }
        self.contact.name_for_display()
    }

    /// Port of `purple_conversation_member_matches`: `None`/empty needle matches; otherwise
    /// `alias`, then `nickname`, then the contact-info chain.
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
        if let Some(nickname) = self.nickname.as_deref() {
            if !nickname.is_empty() && str_matches(needle, nickname) {
                return true;
            }
        }
        self.contact.matches(Some(needle))
    }

    /// Port of `purple_conversation_member_compare` less the NULL handling: role first (a higher
    /// role sorts first — the daemon analog of libpurple's badge ordering), then name-for-display.
    /// Suitable for `slice::sort_by`.
    pub fn cmp_in_conversation(&self, other: &ConversationMember) -> Ordering {
        // A higher role sorts first, so the higher rank must yield `Less`.
        let by_role = role_rank(other.role).cmp(&role_rank(self.role));
        if by_role != Ordering::Equal {
            return by_role;
        }
        utf8_strcasecmp(self.name_for_display(), other.name_for_display())
    }
}

/// Port of `purple_conversation_member_compare` NULL rules + delegation to
/// [`ConversationMember::cmp_in_conversation`].
pub fn member_compare(a: Option<&ConversationMember>, b: Option<&ConversationMember>) -> Ordering {
    match (a, b) {
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
        (Some(a), Some(b)) => a.cmp_in_conversation(b),
    }
}

/// Port of `purple_conversation_member_equal` (`compare == 0`, NULL-safe).
pub fn member_equal(a: Option<&ConversationMember>, b: Option<&ConversationMember>) -> bool {
    member_compare(a, b) == Ordering::Equal
}

// ===========================================================================
// Member collections (← PurpleConversationMembers)
// ===========================================================================

/// Port of `purple_conversation_members_find_member`: the first member whose contact info compares
/// equal to `info` (`purple_contact_info_compare == 0`).
pub fn find_member<'a>(
    members: &'a [ConversationMember],
    info: &ContactInfo,
) -> Option<&'a ConversationMember> {
    position_of_member(members, info).map(|position| &members[position])
}

/// Port of `purple_conversation_members_has_member`.
pub fn has_member(members: &[ConversationMember], info: &ContactInfo) -> bool {
    position_of_member(members, info).is_some()
}

/// The index of the first member whose contact compares equal to `info`
/// (`purple_contact_info_compare == 0`, mirroring `check_member_equal`).
fn position_of_member(members: &[ConversationMember], info: &ContactInfo) -> Option<usize> {
    members
        .iter()
        .position(|member| contact_info_equal(Some(&member.contact), Some(info)))
}

/// Port of `purple_conversation_members_find_or_add_member`: returns the existing member (and
/// `false`), or appends a fresh member for `info` (and `true`).
pub fn find_or_add_member<'a>(
    members: &'a mut Vec<ConversationMember>,
    info: &ContactInfo,
) -> (&'a mut ConversationMember, bool) {
    match position_of_member(members, info) {
        Some(position) => (&mut members[position], false),
        None => {
            members.push(new_member(info));
            let position = members.len() - 1;
            (&mut members[position], true)
        }
    }
}

/// Port of `purple_conversation_members_remove_member`: removes the member whose contact compares
/// equal to `info`; returns whether a member was removed.
pub fn remove_member(members: &mut Vec<ConversationMember>, info: &ContactInfo) -> bool {
    match position_of_member(members, info) {
        Some(position) => {
            members.remove(position);
            true
        }
        None => false,
    }
}

/// Port of `purple_conversation_members_remove_all_members`.
pub fn remove_all_members(members: &mut Vec<ConversationMember>) {
    members.clear();
}

/// Port of `purple_conversation_members_get_active_typers`: members whose typing state is
/// [`TypingState::Typing`], in order.
pub fn active_typers(members: &[ConversationMember]) -> Vec<&ConversationMember> {
    members
        .iter()
        .filter(|member| member.typing == TypingState::Typing)
        .collect()
}

/// Port of `purple_conversation_members_find_first_other`: the first member whose contact does not
/// compare equal to `info`.
pub fn find_first_other<'a>(
    members: &'a [ConversationMember],
    info: &ContactInfo,
) -> Option<&'a ConversationMember> {
    members
        .iter()
        .find(|member| !contact_info_equal(Some(&member.contact), Some(info)))
}

/// Port of `purple_conversation_members_extend`: appends every member of `source` onto `existing`
/// (raw append, no dedup — matching `g_ptr_array_extend_and_steal`) and empties `source`.
pub fn extend_members(
    existing: &mut Vec<ConversationMember>,
    source: &mut Vec<ConversationMember>,
) {
    existing.append(source);
}

/// A fresh member for `info` with default per-conversation state
/// (← `purple_conversation_member_new`).
fn new_member(info: &ContactInfo) -> ConversationMember {
    ConversationMember {
        contact: info.clone(),
        alias: None,
        nickname: None,
        typing: TypingState::None,
        role: MemberRole::None,
        session: None,
    }
}

// ===========================================================================
// Conversation manager: find-DM
// ===========================================================================

/// Port of `purple_conversation_manager_find_dm`: the first conversation on `transport` (← account)
/// that is a DM and contains `contact`.
pub fn find_dm<'a>(
    conversations: &'a [ConversationInfo],
    transport: &TransportId,
    contact: &ContactInfo,
) -> Option<&'a ConversationInfo> {
    conversations.iter().find(|conversation| {
        conversation.transport == *transport
            && conversation.kind == ConversationType::Dm
            && has_member(&conversation.members, contact)
    })
}

/// Order-independent member-set equality (scope item 5 — "same set of participants regardless of
/// order"): true when both collections contain the same set of contacts (by
/// `purple_contact_info_compare == 0`). Derived helper; no direct libpurple g_test.
pub fn same_member_set(a: &[ConversationMember], b: &[ConversationMember]) -> bool {
    a.len() == b.len()
        && a.iter().all(|member| has_member(b, &member.contact))
        && b.iter().all(|member| has_member(a, &member.contact))
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn info(id: &str) -> ContactInfo {
        ContactInfo {
            id: id.into(),
            ..Default::default()
        }
    }

    fn info_dn(id: &str, display_name: &str) -> ContactInfo {
        ContactInfo {
            id: id.into(),
            display_name: Some(display_name.into()),
            ..Default::default()
        }
    }

    fn member(contact: ContactInfo) -> ConversationMember {
        new_member(&contact)
    }

    // -- ContactInfo::name_for_display -------------------------------------

    #[test]
    fn contact_info_name_for_display_display_name() {
        // ← contact_with_display_name: display_name set, no id-only fallback.
        let c = info_dn("id", "display name");
        assert_eq!(c.name_for_display(), "display name");
    }

    #[test]
    fn contact_info_name_for_display_id_fallback() {
        // ← id_fallback: nothing but the id.
        let c = info("id");
        assert_eq!(c.name_for_display(), "id");
    }

    // -- ContactInfo::name_for_display (person-aware, W3-J re-activation) ---

    #[test]
    fn contact_info_name_for_display_person_alias() {
        // ← /contact-info/get_name_for_display/person_with_alias: a contact whose person has an
        // alias resolves to the person's alias (which outranks the contact's display_name).
        let person = Person {
            alias: Some("person alias".into()),
            ..Default::default()
        };
        let c = info_dn("id", "display name");
        assert_eq!(
            c.name_for_display_with_person(Some(&person)),
            "person alias"
        );
        // No person -> the plain display_name -> id chain.
        assert_eq!(c.name_for_display_with_person(None), "display name");
        // A person WITHOUT an alias -> also falls through to the contact's display_name.
        let no_alias = Person::default();
        assert_eq!(
            c.name_for_display_with_person(Some(&no_alias)),
            "display name"
        );
    }

    // -- ContactInfo::compare (person-aware, W3-J re-activation) -----------

    #[test]
    fn contact_info_compare_person_no_person() {
        // ← /contact-info/compare/person__no_person: a contact WITH a person sorts before one
        // WITHOUT (both have empty ids, so only the person tier decides).
        let person = Person::default();
        let a = info("");
        let b = info("");
        assert_eq!(
            contact_info_compare_with_person(Some((&a, Some(&person))), Some((&b, None))),
            Ordering::Less
        );
    }

    #[test]
    fn contact_info_compare_no_person_person() {
        // ← /contact-info/compare/no_person__person: the mirror — no-person sorts after person.
        let person = Person::default();
        let a = info("");
        let b = info("");
        assert_eq!(
            contact_info_compare_with_person(Some((&a, None)), Some((&b, Some(&person)))),
            Ordering::Greater
        );
    }

    // -- ContactInfo::compare ----------------------------------------------

    #[test]
    fn contact_info_compare_not_null_null() {
        let c = info("");
        assert_eq!(contact_info_compare(Some(&c), None), Ordering::Less);
    }

    #[test]
    fn contact_info_compare_null_not_null() {
        let c = info("");
        assert_eq!(contact_info_compare(None, Some(&c)), Ordering::Greater);
    }

    #[test]
    fn contact_info_compare_null_null() {
        assert_eq!(contact_info_compare(None, None), Ordering::Equal);
    }

    #[test]
    fn contact_info_compare_name_name() {
        let a = info("aaa");
        let mut b = info("zzz");
        assert_eq!(contact_info_compare(Some(&a), Some(&b)), Ordering::Less);
        assert_eq!(contact_info_compare(Some(&b), Some(&a)), Ordering::Greater);
        b.id = "aaa".into();
        assert_eq!(contact_info_compare(Some(&b), Some(&a)), Ordering::Equal);
    }

    // -- ContactInfo::equal ------------------------------------------------

    #[test]
    fn contact_info_equal_not_null_not_null() {
        let mut a = info("");
        let mut b = info("");
        assert!(contact_info_equal(Some(&a), Some(&b)));
        a.id = "foo".into();
        assert!(!contact_info_equal(Some(&a), Some(&b)));
        b.id = "foo".into();
        assert!(contact_info_equal(Some(&a), Some(&b)));
    }

    #[test]
    fn contact_info_equal_not_null_null() {
        let a = info("");
        assert!(!contact_info_equal(Some(&a), None));
    }

    #[test]
    fn contact_info_equal_null_not_null() {
        let a = info("");
        assert!(!contact_info_equal(None, Some(&a)));
    }

    #[test]
    fn contact_info_equal_null_null() {
        assert!(contact_info_equal(None, None));
    }

    // -- ContactInfo::matches ----------------------------------------------

    #[test]
    fn contact_info_matches_accepts_null() {
        assert!(info("").matches(None));
    }

    #[test]
    fn contact_info_matches_empty_string() {
        assert!(info("").matches(Some("")));
    }

    #[test]
    fn contact_info_matches_display_name() {
        let c = info_dn("", "display name");
        assert!(c.matches(Some("play")));
    }

    #[test]
    fn contact_info_matches_none() {
        // ← matches/none: id + display_name set, needle matches neither.
        let c = info_dn("id", "display name");
        assert!(!c.matches(Some("nothing")));
    }

    // -- ConversationMember::name_for_display ------------------------------

    #[test]
    fn member_name_for_display_precedence() {
        let mut m = member(info("tron"));
        // default falls back to the contact info's id.
        assert_eq!(m.name_for_display(), "tron");
        // nickname takes precedence over the contact chain.
        m.nickname = Some("rinzler".into());
        assert_eq!(m.name_for_display(), "rinzler");
        // alias takes precedence over the nickname.
        m.alias = Some("Alan".into());
        assert_eq!(m.name_for_display(), "Alan");
        // remove the alias -> back to the nickname.
        m.alias = None;
        assert_eq!(m.name_for_display(), "rinzler");
        // remove the nickname -> back to the contact chain.
        m.nickname = None;
        assert_eq!(m.name_for_display(), "tron");
    }

    // -- ConversationMember::matches ---------------------------------------

    #[test]
    fn member_matches_accepts_null() {
        assert!(member(info("")).matches(None));
    }

    #[test]
    fn member_matches_empty_string() {
        assert!(member(info("")).matches(Some("")));
    }

    #[test]
    fn member_matches_alias() {
        let mut m = member(info(""));
        m.alias = Some("this is the alias".into());
        assert!(m.matches(Some("the")));
        assert!(!m.matches(Some("what")));
    }

    #[test]
    fn member_matches_nickname() {
        // Faithful to the C test, which sets the *alias* (its name is a misnomer).
        let mut m = member(info(""));
        m.alias = Some("nickosaurus".into());
        assert!(m.matches(Some("nick")));
        assert!(!m.matches(Some("dinosaur")));
    }

    #[test]
    fn member_matches_contact_info() {
        // C sets an alias on the contact info; the daemon models that as display_name.
        let m = member(info_dn("", "something"));
        assert!(m.matches(Some("some")));
        assert!(!m.matches(Some("any")));
    }

    // -- ConversationMember::compare ---------------------------------------

    #[test]
    fn member_compare_not_null_null() {
        let m = member(info(""));
        assert_eq!(member_compare(Some(&m), None), Ordering::Less);
    }

    #[test]
    fn member_compare_null_not_null() {
        let m = member(info(""));
        assert_eq!(member_compare(None, Some(&m)), Ordering::Greater);
    }

    #[test]
    fn member_compare_null_null() {
        assert_eq!(member_compare(None, None), Ordering::Equal);
    }

    #[test]
    fn member_compare_same() {
        let m = member(info(""));
        assert_eq!(member_compare(Some(&m), Some(&m)), Ordering::Equal);
    }

    #[test]
    fn member_compare_nickname_nickname() {
        let mut m1 = member(info(""));
        m1.nickname = Some("aaa".into());
        let mut m2 = member(info(""));
        m2.nickname = Some("zzz".into());
        assert_eq!(member_compare(Some(&m1), Some(&m2)), Ordering::Less);
        assert_eq!(member_compare(Some(&m2), Some(&m1)), Ordering::Greater);
        m2.nickname = Some("aaa".into());
        assert_eq!(member_compare(Some(&m1), Some(&m2)), Ordering::Equal);
    }

    #[test]
    fn member_compare_role_nickname() {
        // ← badges__nickname, badges mapped to role. A higher role sorts first; once roles are
        // equal, the nickname breaks the tie.
        let mut m1 = member(info(""));
        m1.nickname = Some("zzz".into());
        m1.role = MemberRole::Op;
        let mut m2 = member(info(""));
        m2.nickname = Some("aaa".into());
        assert_eq!(member_compare(Some(&m1), Some(&m2)), Ordering::Less);
        assert_eq!(member_compare(Some(&m2), Some(&m1)), Ordering::Greater);
        // Clear the role; now the nickname decides and m2 ("aaa") sorts first.
        m1.role = MemberRole::None;
        assert_eq!(member_compare(Some(&m1), Some(&m2)), Ordering::Greater);
    }

    // -- Member collections ------------------------------------------------

    #[test]
    fn members_add_remove() {
        let mut members: Vec<ConversationMember> = Vec::new();
        let info = info("745c50ba-1189-48d9-827c-051783026c96");

        // Add the member.
        let (_, added) = find_or_add_member(&mut members, &info);
        assert!(added);
        assert_eq!(members.len(), 1);
        assert!(find_member(&members, &info).is_some());

        // Adding again returns the existing member, no growth.
        let (_, added) = find_or_add_member(&mut members, &info);
        assert!(!added);
        assert_eq!(members.len(), 1);

        // Remove it.
        assert!(remove_member(&mut members, &info));
        assert!(find_member(&members, &info).is_none());

        // Double remove does nothing.
        assert!(!remove_member(&mut members, &info));
    }

    #[test]
    fn members_remove_all() {
        let mut members = vec![
            member(info("8af5f81d-dee3-4d4a-b2fb-1cfa07f96337")),
            member(info("06f0efc1-357c-45e6-af2a-e58edcd5af22")),
        ];
        assert_eq!(members.len(), 2);
        remove_all_members(&mut members);
        assert_eq!(members.len(), 0);
    }

    #[test]
    fn members_find_or_add_member() {
        let mut members: Vec<ConversationMember> = Vec::new();
        let info = info("uuid");

        let idx1 = {
            let (m, added) = find_or_add_member(&mut members, &info);
            assert!(added);
            m.contact.clone()
        };
        assert_eq!(members.len(), 1);

        let idx2 = {
            let (m, added) = find_or_add_member(&mut members, &info);
            assert!(!added);
            m.contact.clone()
        };
        assert_eq!(members.len(), 1);
        assert_eq!(idx1, idx2);
    }

    #[test]
    fn members_active_typers() {
        let mut members = vec![member(info(""))];
        assert_eq!(active_typers(&members).len(), 0);

        members[0].typing = TypingState::Typing;
        assert_eq!(active_typers(&members).len(), 1);

        // A typing refresh is idempotent.
        members[0].typing = TypingState::Typing;
        assert_eq!(active_typers(&members).len(), 1);

        members[0].typing = TypingState::None;
        assert_eq!(active_typers(&members).len(), 0);
    }

    #[test]
    fn members_extend() {
        let mut existing = vec![member(info("existing"))];
        let mut source = vec![member(info("a")), member(info("b")), member(info("c"))];

        extend_members(&mut existing, &mut source);

        assert_eq!(existing.len(), 4);
        assert!(source.is_empty());
        assert_eq!(existing[0].contact.id, "existing");
        assert_eq!(existing[3].contact.id, "c");
    }

    #[test]
    fn members_find_first_other() {
        let info1 = info("one");
        let info2 = info("two");
        let mut members: Vec<ConversationMember> = Vec::new();

        // Empty -> None.
        assert!(find_first_other(&members, &info1).is_none());

        // Only self -> None.
        members.push(member(info1.clone()));
        assert!(find_first_other(&members, &info1).is_none());

        // A second, distinct member is returned.
        members.push(member(info2.clone()));
        let found = find_first_other(&members, &info1).expect("first other");
        assert!(contact_info_equal(Some(&found.contact), Some(&info2)));
    }

    // -- find_dm -----------------------------------------------------------

    fn conv(
        transport: &str,
        kind: ConversationType,
        members: Vec<ConversationMember>,
    ) -> ConversationInfo {
        ConversationInfo {
            transport: TransportId::new(transport),
            id: "conv".into(),
            kind,
            title: None,
            topic: None,
            description: None,
            members,
            parent: None,
        }
    }

    #[test]
    fn find_dm_empty() {
        let convs: Vec<ConversationInfo> = Vec::new();
        assert!(find_dm(&convs, &TransportId::new("test"), &info("c")).is_none());
    }

    #[test]
    fn find_dm_exists() {
        let contact = info("a9780f2a-eeb5-4d6b-89cb-52e5dad3973f");
        let convs = vec![conv(
            "test",
            ConversationType::Dm,
            vec![member(contact.clone())],
        )];
        let found = find_dm(&convs, &TransportId::new("test"), &contact);
        assert!(found.is_some());
    }

    #[test]
    fn find_dm_does_not_exist() {
        let contact = info("contact");
        let convs = vec![
            // Same transport, but a channel (not a DM).
            conv(
                "test1",
                ConversationType::Channel,
                vec![member(contact.clone())],
            ),
            // A DM on the same transport, but without the contact.
            conv("test1", ConversationType::Dm, vec![member(info("other"))]),
            // A channel on a different transport.
            conv(
                "test2",
                ConversationType::Channel,
                vec![member(contact.clone())],
            ),
        ];
        assert!(find_dm(&convs, &TransportId::new("test1"), &contact).is_none());
    }

    // -- same_member_set (derived) -----------------------------------------

    #[test]
    fn same_member_set_order_independent() {
        let a = vec![member(info("x")), member(info("y")), member(info("z"))];
        let b = vec![member(info("z")), member(info("x")), member(info("y"))];
        let c = vec![member(info("x")), member(info("y"))];
        let d = vec![member(info("x")), member(info("y")), member(info("w"))];

        assert!(same_member_set(&a, &b));
        assert!(!same_member_set(&a, &c));
        assert!(!same_member_set(&a, &d));
    }
}
