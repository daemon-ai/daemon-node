// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The `Tags` container ported from libpurple `PurpleTags` (`purpletags.c`, work package W2-E).
//!
//! An ordered collection of string tags, each either bare (`"name"`) or valued (`"name:value"`,
//! split on the **first** `:`). This is a **non-wire** domain type (like the typed models in
//! [`crate::details`]): a pure in-memory container with no serde/CDDL surface.
//!
//! It deliberately reuses [`ConversationType::tag_value`] (W1-B, [`crate::details`]) for the
//! conversation-`"type"` tag rather than re-deriving that mapping.

use crate::ConversationType;

/// Split a tag into `(name, value)` on the first `:` (`purple_tag_split`). No colon → `value` is
/// `None`. A leading/sole `:` yields `("", Some(rest))`.
pub fn tag_split(tag: &str) -> (String, Option<String>) {
    match tag.find(':') {
        None => (tag.to_string(), None),
        Some(idx) => (tag[..idx].to_string(), Some(tag[idx + 1..].to_string())),
    }
}

/// An ordered set of string tags (← `PurpleTags`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Tags {
    tags: Vec<String>,
}

impl Tags {
    /// An empty container (`purple_tags_new`).
    pub fn new() -> Self {
        Self::default()
    }

    /// The number of tags.
    pub fn len(&self) -> usize {
        self.tags.len()
    }

    /// Whether the container is empty.
    pub fn is_empty(&self) -> bool {
        self.tags.is_empty()
    }

    /// The tags in insertion order (`GListModel` iteration).
    pub fn all(&self) -> &[String] {
        &self.tags
    }

    /// Add a full tag (`purple_tags_add` → `purple_tags_real_add`): remove any exactly-equal existing
    /// tag, then append. A duplicate add therefore moves the tag to the end and keeps the length.
    pub fn add(&mut self, tag: &str) {
        self.remove(tag);
        self.tags.push(tag.to_string());
    }

    /// Add a tag from `(name, value)` (`purple_tags_add_with_value`): builds `"name:value"` (or
    /// `"name"` when `value` is `None`) then adds it.
    pub fn add_with_value(&mut self, name: &str, value: Option<&str>) {
        let tag = match value {
            Some(v) => format!("{name}:{v}"),
            None => name.to_string(),
        };
        self.add(&tag);
    }

    /// Remove an exactly-equal tag (`purple_tags_remove`). Returns whether one was removed.
    pub fn remove(&mut self, tag: &str) -> bool {
        if let Some(idx) = self.tags.iter().position(|t| t == tag) {
            self.tags.remove(idx);
            true
        } else {
            false
        }
    }

    /// Remove the tag built from `(name, value)` (`purple_tags_remove_with_value`).
    pub fn remove_with_value(&mut self, name: &str, value: Option<&str>) -> bool {
        match value {
            None => self.remove(name),
            Some(v) => self.remove(&format!("{name}:{v}")),
        }
    }

    /// Remove every tag (`purple_tags_remove_all`).
    pub fn remove_all(&mut self) {
        self.tags.clear();
    }

    /// Whether an exactly-equal tag exists (`purple_tags_exists`). An empty tag is never present.
    pub fn exists(&self, tag: &str) -> bool {
        if tag.is_empty() {
            return false;
        }
        self.tags.iter().any(|t| t == tag)
    }

    /// Look up a tag by name (`purple_tags_lookup`): returns `(value, found)`. For a tag that has
    /// `name` as a prefix, the char after the prefix decides — end-of-string → bare tag
    /// (`(None, true)`); `:` → `(Some(value), true)`. A partial name match (`"pur"` vs `"purple"`)
    /// does not match.
    pub fn lookup(&self, name: &str) -> (Option<&str>, bool) {
        for tag in &self.tags {
            if let Some(rest) = tag.strip_prefix(name) {
                if rest.is_empty() {
                    return (None, true);
                } else if let Some(value) = rest.strip_prefix(':') {
                    // A ':' right after the name means a valued tag; its value is the remainder.
                    return (Some(value), true);
                }
            }
        }
        (None, false)
    }

    /// The value for `name`, ignoring the found-flag (`purple_tags_get`).
    pub fn get(&self, name: &str) -> Option<&str> {
        self.lookup(name).0
    }

    /// The sub-collection of tags whose name is exactly `name` (`purple_tags_get_all_with_name`): a
    /// prefix match followed by end-of-string or `:`. An empty `name` yields an empty collection.
    pub fn get_all_with_name(&self, name: &str) -> Tags {
        let mut filtered = Tags::new();
        if name.is_empty() {
            return filtered;
        }
        for tag in &self.tags {
            if let Some(rest) = tag.strip_prefix(name) {
                if rest.is_empty() || rest.starts_with(':') {
                    filtered.tags.push(tag.clone());
                }
            }
        }
        filtered
    }

    /// Join the tags into a string (`purple_tags_to_string`); `separator = None` concatenates.
    pub fn to_string_with(&self, separator: Option<&str>) -> String {
        match separator {
            Some(sep) => self.tags.join(sep),
            None => self.tags.concat(),
        }
    }

    /// Whether every tag in `needle` exists in `self` (`purple_tags_contains`).
    pub fn contains(&self, needle: &Tags) -> bool {
        needle.tags.iter().all(|tag| self.exists(tag))
    }

    /// Set the conversation `"type"` tag from a [`ConversationType`], reusing
    /// [`ConversationType::tag_value`] (W1-B, `details.rs`) rather than re-deriving the mapping.
    /// `Unset` removes any existing `"type"` tag; otherwise the previous `"type"` tag is replaced.
    pub fn set_conversation_type(&mut self, ty: ConversationType) {
        // Drop any existing `type` / `type:*` tag first.
        self.tags.retain(|t| t != "type" && !t.starts_with("type:"));
        if let Some(value) = ty.tag_value() {
            self.add_with_value("type", Some(value));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- exists / lookup ----------------------------------------------------

    #[test]
    fn tags_exists() {
        let mut tags = Tags::new();
        tags.add("foo");
        tags.add("bar:1");
        tags.add("baz");
        assert!(tags.exists("foo"));
        assert!(tags.exists("bar:1"));
        assert!(!tags.exists("baz:"));
        assert!(!tags.exists("qux"));
    }

    #[test]
    fn tags_lookup_exists() {
        let mut tags = Tags::new();
        tags.add("foo");
        assert_eq!(tags.len(), 1);
        let (value, found) = tags.lookup("foo");
        assert_eq!(value, None);
        assert!(found);

        tags.add("bar:baz");
        assert_eq!(tags.len(), 2);
        let (value, found) = tags.lookup("bar");
        assert_eq!(value, Some("baz"));
        assert!(found);

        // A name of "pur" must not match a tag of "purple".
        tags.add("purple");
        assert_eq!(tags.len(), 3);
        let (value, found) = tags.lookup("pur");
        assert_eq!(value, None);
        assert!(!found);
    }

    #[test]
    fn tags_lookup_non_existent() {
        let tags = Tags::new();
        let (value, found) = tags.lookup("foo");
        assert_eq!(value, None);
        assert!(!found);
    }

    // -- add / remove (bare) ------------------------------------------------

    #[test]
    fn tags_add_remove_bare() {
        let mut tags = Tags::new();
        tags.add("tag1");
        assert_eq!(tags.len(), 1);
        assert!(tags.remove("tag1"));
        assert_eq!(tags.len(), 0);
    }

    #[test]
    fn tags_add_duplicate_bare() {
        let mut tags = Tags::new();
        tags.add("tag1");
        assert_eq!(tags.len(), 1);
        // Re-adding removes then appends: length stays 1.
        tags.add("tag1");
        assert_eq!(tags.len(), 1);
    }

    #[test]
    fn tags_remove_non_existent_bare() {
        let mut tags = Tags::new();
        assert!(!tags.remove("tag1"));
        assert_eq!(tags.len(), 0);
    }

    // -- add-with-value -----------------------------------------------------

    #[test]
    fn tags_add_with_value() {
        let mut tags = Tags::new();
        tags.add_with_value("tag1", Some("purple"));
        assert_eq!(tags.len(), 1);
        let (value, found) = tags.lookup("tag1");
        assert_eq!(value, Some("purple"));
        assert!(found);
    }

    #[test]
    fn tags_add_with_value_null() {
        let mut tags = Tags::new();
        tags.add_with_value("tag1", None);
        assert_eq!(tags.len(), 1);
        let (value, found) = tags.lookup("tag1");
        assert_eq!(value, None);
        assert!(found);
    }

    #[test]
    fn tags_add_remove() {
        let mut tags = Tags::new();
        tags.add("tag1:purple");
        assert_eq!(tags.len(), 1);
        assert!(tags.remove("tag1:purple"));
        assert_eq!(tags.len(), 0);
    }

    #[test]
    fn tags_add_remove_with_null_value() {
        let mut tags = Tags::new();
        tags.add_with_value("tag1", None);
        assert_eq!(tags.len(), 1);
        assert!(tags.remove_with_value("tag1", None));
        assert_eq!(tags.len(), 0);
    }

    #[test]
    fn tags_add_remove_with_value() {
        let mut tags = Tags::new();
        tags.add_with_value("tag1", Some("purple"));
        assert_eq!(tags.len(), 1);
        assert!(tags.remove_with_value("tag1", Some("purple")));
        assert_eq!(tags.len(), 0);
    }

    #[test]
    fn tags_add_duplicate_with_value() {
        let mut tags = Tags::new();
        tags.add("tag1:purple");
        assert_eq!(tags.len(), 1);
        tags.add("tag1:purple");
        assert_eq!(tags.len(), 1);
    }

    #[test]
    fn tags_remove_non_existent_with_value() {
        let mut tags = Tags::new();
        tags.remove("tag1:purple");
        assert_eq!(tags.len(), 0);
    }

    // -- remove-all ---------------------------------------------------------

    #[test]
    fn tags_remove_all_empty() {
        let mut tags = Tags::new();
        tags.remove_all();
        assert_eq!(tags.len(), 0);
    }

    #[test]
    fn tags_remove_all_single() {
        let mut tags = Tags::new();
        tags.add("foo");
        assert_eq!(tags.len(), 1);
        tags.remove_all();
        assert_eq!(tags.len(), 0);
    }

    #[test]
    fn tags_remove_all_multiple() {
        let mut tags = Tags::new();
        tags.add("foo");
        tags.add("bar");
        tags.add("baz");
        assert_eq!(tags.len(), 3);
        tags.remove_all();
        assert_eq!(tags.len(), 0);
    }

    // -- get ----------------------------------------------------------------

    #[test]
    fn tags_get_single() {
        let mut tags = Tags::new();
        tags.add("tag1:purple");
        assert_eq!(tags.get("tag1"), Some("purple"));
    }

    #[test]
    fn tags_get_multiple() {
        let mut tags = Tags::new();
        tags.add("tag1:purple");
        tags.add("tag1:pink");
        // The first match wins.
        assert_eq!(tags.get("tag1"), Some("purple"));
    }

    #[test]
    fn tags_get_all() {
        let mut tags = Tags::new();
        let values = ["foo", "bar", "baz", "qux", "quux"];
        for v in values {
            tags.add(v);
        }
        assert_eq!(tags.all(), &values);
    }

    #[test]
    fn tags_get_all_with_name() {
        let mut tags = Tags::new();
        // No matches -> empty.
        assert_eq!(tags.get_all_with_name("group").len(), 0);

        tags.add("group");
        tags.add("groups");
        tags.add("group:");
        tags.add("grouping");
        tags.add("group:a");
        tags.add("grouped");

        let filtered = tags.get_all_with_name("group");
        assert_eq!(filtered.len(), 3);

        let mut expected = Tags::new();
        expected.add("group");
        expected.add("group:");
        expected.add("group:a");
        assert!(filtered.contains(&expected));
    }

    // -- to-string ----------------------------------------------------------

    #[test]
    fn tags_to_string_single() {
        let mut tags = Tags::new();
        tags.add("foo");
        assert_eq!(tags.to_string_with(None), "foo");
    }

    #[test]
    fn tags_to_string_multiple_with_separator() {
        let mut tags = Tags::new();
        tags.add("foo");
        tags.add("bar");
        tags.add("baz");
        assert_eq!(tags.to_string_with(Some(", ")), "foo, bar, baz");
    }

    #[test]
    fn tags_to_string_multiple_with_null_separator() {
        let mut tags = Tags::new();
        tags.add("foo");
        tags.add("bar");
        tags.add("baz");
        assert_eq!(tags.to_string_with(None), "foobarbaz");
    }

    // -- tag_split ----------------------------------------------------------

    #[test]
    fn tag_split_table() {
        let cases: &[(&str, &str, Option<&str>)] = &[
            ("", "", None),
            ("foo", "foo", None),
            ("🐦", "🐦", None),
            (":", "", Some("")),
            ("foo:bar", "foo", Some("bar")),
            ("🐦:", "🐦", Some("")),
            (":🐦", "", Some("🐦")),
        ];
        for (tag, name, value) in cases {
            let (n, v) = tag_split(tag);
            assert_eq!(&n, name, "tag: {tag:?}");
            assert_eq!(v.as_deref(), *value, "tag: {tag:?}");
        }
    }

    // -- contains -----------------------------------------------------------

    #[test]
    fn tags_contains_full() {
        let mut tags = Tags::new();
        tags.add("foo");
        tags.add("bar:");
        tags.add("baz:1");

        let mut needle = Tags::new();
        needle.add("foo");
        needle.add("bar:");
        needle.add("baz:1");
        assert!(tags.contains(&needle));
    }

    #[test]
    fn tags_contains_partial() {
        let mut tags = Tags::new();
        tags.add("foo");
        tags.add("bar:");
        tags.add("baz:1");

        let mut needle = Tags::new();
        needle.add("foo");
        needle.add("baz:1");
        assert!(tags.contains(&needle));
    }

    #[test]
    fn tags_contains_none() {
        let mut tags = Tags::new();
        tags.add("foo");
        tags.add("bar:1");

        let mut needle = Tags::new();
        needle.add("baz:qux");
        assert!(!tags.contains(&needle));
    }

    // -- extra: conversation-type alignment (reuses W1-B tag_value) --------

    #[test]
    fn tags_set_conversation_type() {
        let mut tags = Tags::new();
        tags.set_conversation_type(ConversationType::Channel);
        assert_eq!(tags.get("type"), Some("channel"));
        // Switching type replaces the tag (not duplicate).
        tags.set_conversation_type(ConversationType::Dm);
        assert_eq!(tags.get("type"), Some("dm"));
        assert_eq!(tags.get_all_with_name("type").len(), 1);
        // Unset removes it.
        tags.set_conversation_type(ConversationType::Unset);
        assert!(!tags.exists("type:dm"));
        assert_eq!(tags.get_all_with_name("type").len(), 0);
    }
}
