//! Display-name, matching, ordering, and member-collection logic.
//!
//! Ported from libpurple (`purplecontactinfo.c`, `purpleconversationmember.c`,
//! `purpleconversationmembers.c`, `purpleconversationmanager.c`). This is host-side derived logic
//! over the existing DTOs — it adds no wire-contract surface. See `docs/port-ledger/matching.md`
//! for the case-by-case provenance and the documented divergences (person precedence is Wave-3;
//! badges map to [`MemberRole`]; UTF-8 collation is approximated by casefolded codepoint order).

use crate::{
    ContactInfo, ConversationInfo, ConversationMember, ConversationType, MemberRole, TypingState,
};
use daemon_protocol::TransportId;
use std::cmp::Ordering;

// ===========================================================================
// String primitives (← birb + purple_utf8_strcasecmp)
// ===========================================================================

/// Port of `birb_str_matches(pattern, str)`: TRUE when `pattern` occurs in sequential order within
/// `haystack`, caseless, ignoring characters in between (a caseless *subsequence* match — e.g.
/// `Br` matches `biRb`). Casefolding is Unicode lowercase (approximates `g_utf8_casefold`).
fn str_matches(pattern: &str, haystack: &str) -> bool {
    let _ = (pattern, haystack);
    false
}

/// Port of `purple_utf8_strcasecmp` for non-NULL operands: casefold (Unicode lowercase) then
/// codepoint order. **Divergence:** approximates `g_utf8_collate` — identical for ASCII inputs.
fn utf8_strcasecmp(a: &str, b: &str) -> Ordering {
    let _ = (a, b);
    Ordering::Equal
}

fn role_rank(_role: MemberRole) -> u8 {
    0
}

// ===========================================================================
// ContactInfo
// ===========================================================================

impl ContactInfo {
    /// Port of `purple_contact_info_get_name_for_display`. The libpurple chain is
    /// `alias → person-alias → display_name → id`; the daemon `ContactInfo` has neither an alias
    /// nor a person field (those live on [`ConversationMember`]; person precedence is Wave-3), so
    /// the chain reduces to `display_name → id`.
    pub fn name_for_display(&self) -> &str {
        ""
    }

    /// Port of `purple_contact_info_matches`. `None`/empty needle matches; otherwise a caseless
    /// subsequence match against `id` then `display_name`.
    pub fn matches(&self, needle: Option<&str>) -> bool {
        let _ = needle;
        false
    }

    /// Non-NULL ordering by name-for-display (person precedence is Wave-3). Suitable for
    /// `slice::sort_by`. See [`contact_info_compare`] for the NULL-safe variant.
    pub fn cmp_for_display(&self, other: &ContactInfo) -> Ordering {
        let _ = other;
        Ordering::Equal
    }
}

/// Port of `purple_contact_info_compare` including the NULL rules
/// (`Some,None → Less`, `None,Some → Greater`, `None,None → Equal`).
pub fn contact_info_compare(a: Option<&ContactInfo>, b: Option<&ContactInfo>) -> Ordering {
    let _ = (a, b);
    Ordering::Equal
}

/// Port of `purple_contact_info_equal` (`compare == 0`, NULL-safe).
pub fn contact_info_equal(a: Option<&ContactInfo>, b: Option<&ContactInfo>) -> bool {
    let _ = (a, b);
    false
}

// ===========================================================================
// ConversationMember
// ===========================================================================

impl ConversationMember {
    /// Port of `purple_conversation_member_get_name_for_display`: `alias → nickname →
    /// contact.name_for_display`.
    pub fn name_for_display(&self) -> &str {
        ""
    }

    /// Port of `purple_conversation_member_matches`: `None`/empty needle matches; otherwise
    /// `alias`, then `nickname`, then the contact-info chain.
    pub fn matches(&self, needle: Option<&str>) -> bool {
        let _ = needle;
        false
    }

    /// Port of `purple_conversation_member_compare` less the NULL handling: role first (a higher
    /// role sorts first — the daemon analog of libpurple's badge ordering), then name-for-display.
    /// Suitable for `slice::sort_by`.
    pub fn cmp_in_conversation(&self, other: &ConversationMember) -> Ordering {
        let _ = other;
        Ordering::Equal
    }
}

/// Port of `purple_conversation_member_compare` NULL rules + delegation to
/// [`ConversationMember::cmp_in_conversation`].
pub fn member_compare(a: Option<&ConversationMember>, b: Option<&ConversationMember>) -> Ordering {
    let _ = (a, b);
    Ordering::Equal
}

/// Port of `purple_conversation_member_equal` (`compare == 0`, NULL-safe).
pub fn member_equal(a: Option<&ConversationMember>, b: Option<&ConversationMember>) -> bool {
    let _ = (a, b);
    false
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
    let _ = (members, info);
    None
}

/// Port of `purple_conversation_members_has_member`.
pub fn has_member(members: &[ConversationMember], info: &ContactInfo) -> bool {
    let _ = (members, info);
    false
}

/// Port of `purple_conversation_members_find_or_add_member`: returns the existing member (and
/// `false`), or appends a fresh member for `info` (and `true`).
pub fn find_or_add_member<'a>(
    members: &'a mut Vec<ConversationMember>,
    info: &ContactInfo,
) -> (&'a mut ConversationMember, bool) {
    // Stub: unconditionally append (wrong — must dedup) so the RED assertions fail.
    members.push(new_member(info));
    let last = members.len() - 1;
    (&mut members[last], true)
}

/// Port of `purple_conversation_members_remove_member`: removes the member whose contact compares
/// equal to `info`; returns whether a member was removed.
pub fn remove_member(members: &mut Vec<ConversationMember>, info: &ContactInfo) -> bool {
    let _ = (members, info);
    false
}

/// Port of `purple_conversation_members_remove_all_members`.
pub fn remove_all_members(members: &mut Vec<ConversationMember>) {
    let _ = members;
}

/// Port of `purple_conversation_members_get_active_typers`: members whose typing state is
/// [`TypingState::Typing`], in order.
pub fn active_typers(members: &[ConversationMember]) -> Vec<&ConversationMember> {
    let _ = members;
    Vec::new()
}

/// Port of `purple_conversation_members_find_first_other`: the first member whose contact does not
/// compare equal to `info`.
pub fn find_first_other<'a>(
    members: &'a [ConversationMember],
    info: &ContactInfo,
) -> Option<&'a ConversationMember> {
    let _ = (members, info);
    None
}

/// Port of `purple_conversation_members_extend`: appends every member of `source` onto `existing`
/// (raw append, no dedup — matching `g_ptr_array_extend_and_steal`) and empties `source`.
pub fn extend_members(
    existing: &mut Vec<ConversationMember>,
    source: &mut Vec<ConversationMember>,
) {
    let _ = (existing, source);
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
    let _ = (conversations, transport, contact);
    None
}

/// Order-independent member-set equality (scope item 5 — "same set of participants regardless of
/// order"): true when both collections contain the same set of contacts (by
/// `purple_contact_info_compare == 0`). Derived helper; no direct libpurple g_test.
pub fn same_member_set(a: &[ConversationMember], b: &[ConversationMember]) -> bool {
    let _ = (a, b);
    false
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

    // -- ContactInfo::compare ----------------------------------------------

    #[test]
    fn contact_info_compare_not_null__null() {
        let c = info("");
        assert_eq!(contact_info_compare(Some(&c), None), Ordering::Less);
    }

    #[test]
    fn contact_info_compare_null__not_null() {
        let c = info("");
        assert_eq!(contact_info_compare(None, Some(&c)), Ordering::Greater);
    }

    #[test]
    fn contact_info_compare_null__null() {
        assert_eq!(contact_info_compare(None, None), Ordering::Equal);
    }

    #[test]
    fn contact_info_compare_name__name() {
        let a = info("aaa");
        let mut b = info("zzz");
        assert_eq!(contact_info_compare(Some(&a), Some(&b)), Ordering::Less);
        assert_eq!(contact_info_compare(Some(&b), Some(&a)), Ordering::Greater);
        b.id = "aaa".into();
        assert_eq!(contact_info_compare(Some(&b), Some(&a)), Ordering::Equal);
    }

    // -- ContactInfo::equal ------------------------------------------------

    #[test]
    fn contact_info_equal_not_null__not_null() {
        let mut a = info("");
        let mut b = info("");
        assert!(contact_info_equal(Some(&a), Some(&b)));
        a.id = "foo".into();
        assert!(!contact_info_equal(Some(&a), Some(&b)));
        b.id = "foo".into();
        assert!(contact_info_equal(Some(&a), Some(&b)));
    }

    #[test]
    fn contact_info_equal_not_null__null() {
        let a = info("");
        assert!(!contact_info_equal(Some(&a), None));
    }

    #[test]
    fn contact_info_equal_null__not_null() {
        let a = info("");
        assert!(!contact_info_equal(None, Some(&a)));
    }

    #[test]
    fn contact_info_equal_null__null() {
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
    fn member_compare_not_null__null() {
        let m = member(info(""));
        assert_eq!(member_compare(Some(&m), None), Ordering::Less);
    }

    #[test]
    fn member_compare_null__not_null() {
        let m = member(info(""));
        assert_eq!(member_compare(None, Some(&m)), Ordering::Greater);
    }

    #[test]
    fn member_compare_null__null() {
        assert_eq!(member_compare(None, None), Ordering::Equal);
    }

    #[test]
    fn member_compare_same() {
        let m = member(info(""));
        assert_eq!(member_compare(Some(&m), Some(&m)), Ordering::Equal);
    }

    #[test]
    fn member_compare_nickname__nickname() {
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
    fn member_compare_role__nickname() {
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
