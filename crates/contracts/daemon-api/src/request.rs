// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! libpurple's **request-field model** ported node-internally (work package W2-I).
//!
//! This is the node-authoritative model behind an interactive request form: typed
//! [`RequestField`]s grouped into [`RequestGroup`]s aggregated onto a [`RequestPage`], with the
//! exact filled/valid semantics libpurple's `purplerequestfield*.c` / `purplerequestgroup.c` /
//! `purplerequestpage.c` test. It adds **no wire-contract surface** — like `details.rs`/`matching.rs`
//! it is a pure, in-memory model the node owns; validators never leave the node.
//!
//! # Relationship to [`crate::AuthParamField`]
//!
//! [`crate::AuthParamField`] (`{ key, label, required }`) is the *wire* discovery shape for the
//! interactive-auth `params` form and STAYS as-is. [`RequestField`] is its fuller, node-internal
//! generalisation: it carries per-variant typed data plus the filled/valid decisions
//! `AuthParamField` deliberately omits. `AuthParamField` is, in effect, the on-the-wire projection
//! of a [`RequestField::string`]-shaped field; this module is the authority that computes validity.
//!
//! # Wire exposure — node-internal (candidate for a future wire version)
//!
//! These types are **not** on the wire (no serde, no `Arbitrary`, no CDDL, no ops, no `WireVersion`
//! bump). The wire carries data, not validators, and there is no consuming surface today that should
//! carry a full [`RequestPage`]. When a client eventually needs to render an interactive request form
//! (e.g. an `AuthChallenge::Form` upgrade or a protocol-driven prompt), lift the *data* projection of
//! these types onto the wire then — append-only, feature-gated `Arbitrary`, CDDL in lockstep — while
//! the validators stay node-side per the "node decides, apps render" invariant. **Tagged: a
//! candidate for a future wire version (beyond the v37 libpurple-parity bump).**

use crate::LocalizedString;
use daemon_protocol::TransportId;
use std::sync::Arc;

/// A custom field validator closure (← `PurpleRequestFieldValidator`): `Ok(())` when valid, else
/// `Err(message)`.
pub type FieldValidator = Arc<dyn Fn(&RequestField) -> Result<(), String> + Send + Sync>;

// ===========================================================================
// ImageRef — opaque handle for the image field
// ===========================================================================

/// An opaque reference to an image (← `PurpleImage`). The request model only stores and round-trips
/// it; rendering/decoding is out of scope. Placeholder for the future image surface.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageRef(pub String);

// ===========================================================================
// Per-variant field bodies
// ===========================================================================

/// A single/multiline (optionally masked) string field (← `PurpleRequestFieldString`).
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct StringField {
    multiline: bool,
    masked: bool,
    default_value: Option<String>,
    value: Option<String>,
}

impl StringField {
    /// Whether multiline input is allowed.
    pub fn is_multiline(&self) -> bool {
        self.multiline
    }

    /// Whether the field should be masked (e.g. a password).
    pub fn is_masked(&self) -> bool {
        self.masked
    }

    /// Set whether the field is masked (`purple_request_field_string_set_masked`).
    pub fn set_masked(&mut self, masked: bool) {
        self.masked = masked;
    }

    /// The default value, if any.
    pub fn default_value(&self) -> Option<&str> {
        self.default_value.as_deref()
    }

    /// The current value, if any (inner `None` = the C `NULL` value).
    pub fn value(&self) -> Option<&str> {
        self.value.as_deref()
    }

    /// Whether the value counts as filled (`!birb_str_is_empty`): `Some(s)` with `s != ""`.
    /// `None` and `Some("")` are both empty; whitespace is NOT empty.
    fn is_filled(&self) -> bool {
        self.value.as_deref().is_some_and(|s| !s.is_empty())
    }

    /// Set (or clear, with `None`) the value (`purple_request_field_string_set_value`). Returns
    /// whether the *filled* state changed (the `notify::filled` the C test counts): true iff the
    /// value actually changed AND its emptiness toggled.
    pub fn set_value(&mut self, value: Option<&str>) -> bool {
        let before = self.is_filled();
        // g_set_str semantics: NULL and "" are distinct, so NULL→"" counts as a change.
        let changed = match (self.value.as_deref(), value) {
            (Some(a), Some(b)) => a != b,
            (None, None) => false,
            _ => true,
        };
        if !changed {
            return false;
        }
        self.value = value.map(str::to_string);
        let after = self.is_filled();
        before != after
    }
}

/// A bounded integer field (← `PurpleRequestFieldInt`).
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct IntField {
    default_value: i32,
    value: i32,
    lower_bound: i32,
    upper_bound: i32,
}

impl IntField {
    /// The default value.
    pub fn default_value(&self) -> i32 {
        self.default_value
    }

    /// The current value.
    pub fn value(&self) -> i32 {
        self.value
    }

    /// The inclusive lower bound.
    pub fn lower_bound(&self) -> i32 {
        self.lower_bound
    }

    /// The inclusive upper bound.
    pub fn upper_bound(&self) -> i32 {
        self.upper_bound
    }

    /// Set the value.
    pub fn set_value(&mut self, value: i32) {
        self.value = value;
    }

    /// Set the inclusive lower bound.
    pub fn set_lower_bound(&mut self, lower_bound: i32) {
        self.lower_bound = lower_bound;
    }

    /// Set the inclusive upper bound.
    pub fn set_upper_bound(&mut self, upper_bound: i32) {
        self.upper_bound = upper_bound;
    }

    /// Bounds check (`purple_request_field_int_is_valid`): the subclass validator run first by
    /// [`RequestField::is_valid`].
    fn is_valid(&self) -> Result<(), String> {
        if self.value < self.lower_bound {
            return Err(format!(
                "Int value {} exceeds lower bound {}",
                self.value, self.lower_bound
            ));
        }
        if self.value > self.upper_bound {
            return Err(format!(
                "Int value {} exceeds upper bound {}",
                self.value, self.upper_bound
            ));
        }
        Ok(())
    }
}

/// A boolean field (← `PurpleRequestFieldBool`).
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct BoolField {
    default_value: bool,
    value: bool,
}

impl BoolField {
    /// The default value.
    pub fn default_value(&self) -> bool {
        self.default_value
    }

    /// The current value.
    pub fn value(&self) -> bool {
        self.value
    }

    /// Set the value.
    pub fn set_value(&mut self, value: bool) {
        self.value = value;
    }
}

/// A single-select choice field with typed options (← `PurpleRequestFieldChoice`).
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct ChoiceField {
    items: Vec<LocalizedString>,
    selected: usize,
}

impl ChoiceField {
    /// The options, in order.
    pub fn items(&self) -> &[LocalizedString] {
        &self.items
    }

    /// The number of options.
    pub fn n_items(&self) -> usize {
        self.items.len()
    }

    /// Add an option by id/label (`purple_request_field_choice_add`). Empty id or label is ignored.
    pub fn add(&mut self, id: &str, label: &str) {
        if id.is_empty() || label.is_empty() {
            return;
        }
        self.add_item(LocalizedString {
            id: id.to_string(),
            label: label.to_string(),
        });
    }

    /// Add a prebuilt option (`purple_request_field_choice_add_item`).
    pub fn add_item(&mut self, item: LocalizedString) {
        self.items.push(item);
    }

    /// Remove every option and reset the selection (`purple_request_field_choice_clear`).
    pub fn clear(&mut self) {
        self.selected = 0;
        self.items.clear();
    }

    /// Remove the option at `position` (`purple_request_field_choice_remove`); returns whether one
    /// was removed. Removing the selected position resets the selection to 0.
    pub fn remove(&mut self, position: usize) -> bool {
        if position >= self.items.len() {
            return false;
        }
        self.items.remove(position);
        if position == self.selected {
            self.selected = 0;
        }
        true
    }

    /// Remove the first option whose id matches (`purple_request_field_choice_remove_by_id`).
    pub fn remove_by_id(&mut self, id: &str) -> bool {
        match self.items.iter().position(|item| item.id == id) {
            Some(index) => self.remove(index),
            None => false,
        }
    }

    /// Remove the first option matching `item`'s id (`purple_request_field_choice_remove_item`).
    pub fn remove_item(&mut self, item: &LocalizedString) -> bool {
        self.remove_by_id(&item.id)
    }

    /// The selected index, or `None` when empty (C `G_MAXUINT`)
    /// (`purple_request_field_choice_get_selected`).
    pub fn get_selected(&self) -> Option<usize> {
        if self.items.is_empty() {
            None
        } else {
            Some(self.selected)
        }
    }

    /// Select the option at `selected` (`purple_request_field_choice_set_selected`); ignored when
    /// out of bounds or already selected.
    pub fn set_selected(&mut self, selected: usize) {
        if selected != self.selected && selected < self.items.len() {
            self.selected = selected;
        }
    }

    /// The selected option (`purple_request_field_choice_get_selected_item`), or `None`.
    pub fn selected_item(&self) -> Option<&LocalizedString> {
        if !self.items.is_empty() && self.selected < self.items.len() {
            Some(&self.items[self.selected])
        } else {
            None
        }
    }
}

/// A multi-select list field (← `PurpleRequestFieldList`).
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct ListField {
    items: Vec<LocalizedString>,
    multi_select: bool,
    selected: Vec<LocalizedString>,
}

impl ListField {
    /// The items, in order.
    pub fn items(&self) -> &[LocalizedString] {
        &self.items
    }

    /// The number of items.
    pub fn n_items(&self) -> usize {
        self.items.len()
    }

    /// Whether multiple items may be selected.
    pub fn multi_select(&self) -> bool {
        self.multi_select
    }

    /// Set whether multiple items may be selected
    /// (`purple_request_field_list_set_multi_select`).
    pub fn set_multi_select(&mut self, multi_select: bool) {
        self.multi_select = multi_select;
    }

    /// The selected items, in selection order.
    pub fn selected(&self) -> &[LocalizedString] {
        &self.selected
    }

    /// Add an item by id/label (`purple_request_field_list_add`). Duplicates are allowed.
    pub fn add(&mut self, id: &str, label: &str) {
        self.add_item(LocalizedString {
            id: id.to_string(),
            label: label.to_string(),
        });
    }

    /// Add a prebuilt item (`purple_request_field_list_add_item`).
    pub fn add_item(&mut self, item: LocalizedString) {
        self.items.push(item);
    }

    /// Remove every item (clearing the selection first) (`purple_request_field_list_clear`).
    pub fn clear(&mut self) {
        self.clear_selected();
        self.items.clear();
    }

    /// Clear the selection (`purple_request_field_list_clear_selected`).
    pub fn clear_selected(&mut self) {
        self.selected.clear();
    }

    /// Remove the first item whose id matches (`purple_request_field_list_remove_by_id`).
    pub fn remove_by_id(&mut self, id: &str) -> bool {
        match self.items.iter().position(|item| item.id == id) {
            Some(index) => {
                self.items.remove(index);
                true
            }
            None => false,
        }
    }

    /// Select the item with `id` (`purple_request_field_list_select_item`): already-selected →
    /// `false`; in single-select mode a prior selection is cleared first; returns whether a new
    /// selection was made.
    pub fn select_item(&mut self, id: &str) -> bool {
        // Already selected → nothing to do.
        if self.selected.iter().any(|item| item.id == id) {
            return false;
        }
        // Single-select replaces any prior selection.
        if !self.multi_select && !self.selected.is_empty() {
            self.selected.clear();
        }
        // Select the matching item if it exists.
        match self.items.iter().find(|item| item.id == id) {
            Some(item) => {
                self.selected.push(item.clone());
                true
            }
            None => false,
        }
    }
}

/// An image field (← `PurpleRequestFieldImage`).
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct ImageField {
    image: Option<ImageRef>,
}

impl ImageField {
    /// The image, if any.
    pub fn image(&self) -> Option<&ImageRef> {
        self.image.as_ref()
    }

    /// Set (or clear) the image (`purple_request_field_image_set_image`).
    pub fn set_image(&mut self, image: Option<ImageRef>) {
        self.image = image;
    }
}

/// An account-picker field (← `PurpleRequestFieldAccount`). Accounts are instance-qualified
/// transport ids in this daemon (see [`crate::profile`]'s `BoundAccount::transport_instance`).
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct AccountField {
    account: Option<TransportId>,
    model: Vec<TransportId>,
}

impl AccountField {
    /// The selected account, if any.
    pub fn account(&self) -> Option<&TransportId> {
        self.account.as_ref()
    }

    /// The candidate accounts to choose from.
    pub fn model(&self) -> &[TransportId] {
        &self.model
    }

    /// Set (or clear) the selected account (`purple_request_field_account_set_account`).
    pub fn set_account(&mut self, account: Option<TransportId>) {
        self.account = account;
    }
}

// ===========================================================================
// RequestFieldKind + RequestField
// ===========================================================================

/// The typed body of a [`RequestField`] (← the `PurpleRequestField` subclasses). `Label` (and
/// libpurple's info-label) carry no extra data beyond the base id/label.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RequestFieldKind {
    /// A string field.
    String(StringField),
    /// A bounded integer field.
    Int(IntField),
    /// A boolean field.
    Boolean(BoolField),
    /// A single-select choice.
    Choice(ChoiceField),
    /// A multi-select list.
    List(ListField),
    /// An image.
    Image(ImageField),
    /// An account picker.
    Account(AccountField),
    /// A non-interactive label / info line.
    Label,
}

/// One request field (← the `PurpleRequestField` base + subclass). Carries the common id/label/
/// visibility/required/validator state plus the typed [`RequestFieldKind`] body.
#[derive(Clone)]
pub struct RequestField {
    id: String,
    label: String,
    visible: bool,
    sensitive: bool,
    type_hint: Option<String>,
    tooltip: Option<String>,
    required: bool,
    validator: Option<FieldValidator>,
    kind: RequestFieldKind,
}

impl std::fmt::Debug for RequestField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RequestField")
            .field("id", &self.id)
            .field("label", &self.label)
            .field("required", &self.required)
            .field("kind", &self.kind)
            .field("has_validator", &self.validator.is_some())
            .finish()
    }
}

impl RequestField {
    fn base(id: &str, label: &str, kind: RequestFieldKind) -> Self {
        Self {
            id: id.to_string(),
            label: label.to_string(),
            visible: true,
            sensitive: true,
            type_hint: None,
            tooltip: None,
            required: false,
            validator: None,
            kind,
        }
    }

    /// A string field (`purple_request_field_string_new`): `value` is seeded from `default_value`.
    pub fn string(id: &str, label: &str, default_value: Option<&str>, multiline: bool) -> Self {
        Self::base(
            id,
            label,
            RequestFieldKind::String(StringField {
                multiline,
                masked: false,
                default_value: default_value.map(str::to_string),
                value: default_value.map(str::to_string),
            }),
        )
    }

    /// An integer field (`purple_request_field_int_new`): `value` is seeded from `default_value`.
    pub fn int(
        id: &str,
        label: &str,
        default_value: i32,
        lower_bound: i32,
        upper_bound: i32,
    ) -> Self {
        Self::base(
            id,
            label,
            RequestFieldKind::Int(IntField {
                default_value,
                value: default_value,
                lower_bound,
                upper_bound,
            }),
        )
    }

    /// A boolean field (`purple_request_field_bool_new`).
    pub fn boolean(id: &str, label: &str, default_value: bool) -> Self {
        Self::base(
            id,
            label,
            RequestFieldKind::Boolean(BoolField {
                default_value,
                value: default_value,
            }),
        )
    }

    /// A single-select choice field (`purple_request_field_choice_new`).
    pub fn choice(id: &str, label: &str) -> Self {
        Self::base(id, label, RequestFieldKind::Choice(ChoiceField::default()))
    }

    /// A multi-select list field (`purple_request_field_list_new`).
    pub fn list(id: &str, label: &str) -> Self {
        Self::base(id, label, RequestFieldKind::List(ListField::default()))
    }

    /// An image field (`purple_request_field_image_new`).
    pub fn image(id: &str, label: &str, image: Option<ImageRef>) -> Self {
        Self::base(id, label, RequestFieldKind::Image(ImageField { image }))
    }

    /// An account-picker field (`purple_request_field_account_new`).
    pub fn account(id: &str, label: &str, model: Vec<TransportId>) -> Self {
        Self::base(
            id,
            label,
            RequestFieldKind::Account(AccountField {
                account: None,
                model,
            }),
        )
    }

    /// A label / info field (`purple_request_field_label_new`).
    pub fn label_field(id: &str, label: &str) -> Self {
        Self::base(id, label, RequestFieldKind::Label)
    }

    /// The field id.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// The field label.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Set the field label.
    pub fn set_label(&mut self, label: &str) {
        self.label = label.to_string();
    }

    /// Whether the field should be visible (default true).
    pub fn is_visible(&self) -> bool {
        self.visible
    }

    /// Set visibility.
    pub fn set_visible(&mut self, visible: bool) {
        self.visible = visible;
    }

    /// Whether the field should be sensitive/enabled (default true).
    pub fn is_sensitive(&self) -> bool {
        self.sensitive
    }

    /// Set sensitivity.
    pub fn set_sensitive(&mut self, sensitive: bool) {
        self.sensitive = sensitive;
    }

    /// The type hint, if any.
    pub fn type_hint(&self) -> Option<&str> {
        self.type_hint.as_deref()
    }

    /// Set (or clear) the type hint.
    pub fn set_type_hint(&mut self, type_hint: Option<&str>) {
        self.type_hint = type_hint.map(str::to_string);
    }

    /// The tooltip, if any.
    pub fn tooltip(&self) -> Option<&str> {
        self.tooltip.as_deref()
    }

    /// Set (or clear) the tooltip.
    pub fn set_tooltip(&mut self, tooltip: Option<&str>) {
        self.tooltip = tooltip.map(str::to_string);
    }

    /// Whether the field is required (default false).
    pub fn is_required(&self) -> bool {
        self.required
    }

    /// Set whether the field is required (`purple_request_field_set_required`).
    pub fn set_required(&mut self, required: bool) {
        self.required = required;
    }

    /// Install a custom validator (`purple_request_field_set_validator`).
    pub fn set_validator(&mut self, validator: FieldValidator) {
        self.validator = Some(validator);
    }

    /// Remove any custom validator.
    pub fn clear_validator(&mut self) {
        self.validator = None;
    }

    /// The typed body.
    pub fn kind(&self) -> &RequestFieldKind {
        &self.kind
    }

    /// The string body, if this is a string field.
    pub fn as_string(&self) -> Option<&StringField> {
        match &self.kind {
            RequestFieldKind::String(f) => Some(f),
            _ => None,
        }
    }

    /// The mutable string body, if this is a string field.
    pub fn as_string_mut(&mut self) -> Option<&mut StringField> {
        match &mut self.kind {
            RequestFieldKind::String(f) => Some(f),
            _ => None,
        }
    }

    /// The int body, if this is an int field.
    pub fn as_int(&self) -> Option<&IntField> {
        match &self.kind {
            RequestFieldKind::Int(f) => Some(f),
            _ => None,
        }
    }

    /// The mutable int body, if this is an int field.
    pub fn as_int_mut(&mut self) -> Option<&mut IntField> {
        match &mut self.kind {
            RequestFieldKind::Int(f) => Some(f),
            _ => None,
        }
    }

    /// The bool body, if this is a bool field.
    pub fn as_bool(&self) -> Option<&BoolField> {
        match &self.kind {
            RequestFieldKind::Boolean(f) => Some(f),
            _ => None,
        }
    }

    /// The mutable bool body, if this is a bool field.
    pub fn as_bool_mut(&mut self) -> Option<&mut BoolField> {
        match &mut self.kind {
            RequestFieldKind::Boolean(f) => Some(f),
            _ => None,
        }
    }

    /// The choice body, if this is a choice field.
    pub fn as_choice(&self) -> Option<&ChoiceField> {
        match &self.kind {
            RequestFieldKind::Choice(f) => Some(f),
            _ => None,
        }
    }

    /// The mutable choice body, if this is a choice field.
    pub fn as_choice_mut(&mut self) -> Option<&mut ChoiceField> {
        match &mut self.kind {
            RequestFieldKind::Choice(f) => Some(f),
            _ => None,
        }
    }

    /// The list body, if this is a list field.
    pub fn as_list(&self) -> Option<&ListField> {
        match &self.kind {
            RequestFieldKind::List(f) => Some(f),
            _ => None,
        }
    }

    /// The mutable list body, if this is a list field.
    pub fn as_list_mut(&mut self) -> Option<&mut ListField> {
        match &mut self.kind {
            RequestFieldKind::List(f) => Some(f),
            _ => None,
        }
    }

    /// The image body, if this is an image field.
    pub fn as_image(&self) -> Option<&ImageField> {
        match &self.kind {
            RequestFieldKind::Image(f) => Some(f),
            _ => None,
        }
    }

    /// The mutable image body, if this is an image field.
    pub fn as_image_mut(&mut self) -> Option<&mut ImageField> {
        match &mut self.kind {
            RequestFieldKind::Image(f) => Some(f),
            _ => None,
        }
    }

    /// The account body, if this is an account field.
    pub fn as_account(&self) -> Option<&AccountField> {
        match &self.kind {
            RequestFieldKind::Account(f) => Some(f),
            _ => None,
        }
    }

    /// The mutable account body, if this is an account field.
    pub fn as_account_mut(&mut self) -> Option<&mut AccountField> {
        match &mut self.kind {
            RequestFieldKind::Account(f) => Some(f),
            _ => None,
        }
    }

    /// Whether the field is filled (`purple_request_field_is_filled`): only strings compute this;
    /// every other variant is always filled.
    pub fn is_filled(&self) -> bool {
        match &self.kind {
            RequestFieldKind::String(f) => f.is_filled(),
            _ => true,
        }
    }

    /// Validate the field (`purple_request_field_is_valid`): subclass validator (int bounds), then
    /// the custom validator, then the required-and-filled check — in that order.
    pub fn is_valid(&self) -> Result<(), String> {
        // 1. The subclass validator (only int has one — the bounds check).
        if let RequestFieldKind::Int(f) = &self.kind {
            f.is_valid()?;
        }
        // 2. The custom validator, iff the subclass validator passed.
        if let Some(validator) = &self.validator {
            validator(self)?;
        }
        // 3. A required field must be filled.
        if self.required && !self.is_filled() {
            return Err("Required field is not filled.".to_string());
        }
        Ok(())
    }
}

// ===========================================================================
// RequestGroup
// ===========================================================================

/// A named group of fields (← `PurpleRequestGroup`). Validity aggregates its fields; the cached
/// `valid` mirrors the C `invalid_fields`-empty state so validity *flips* can be detected.
#[derive(Clone, Debug)]
pub struct RequestGroup {
    title: Option<String>,
    fields: Vec<RequestField>,
    valid: bool,
}

impl RequestGroup {
    /// An empty group (`purple_request_group_new`). Empty groups are valid.
    pub fn new(title: Option<&str>) -> Self {
        Self {
            title: title.map(str::to_string),
            fields: Vec::new(),
            valid: true,
        }
    }

    /// The group title, if any.
    pub fn title(&self) -> Option<&str> {
        self.title.as_deref()
    }

    /// The number of fields.
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    /// Whether the group has no fields.
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    /// The fields, in order.
    pub fn fields(&self) -> &[RequestField] {
        &self.fields
    }

    /// The field at `index`.
    pub fn field(&self, index: usize) -> Option<&RequestField> {
        self.fields.get(index)
    }

    /// The mutable field at `index`.
    pub fn field_mut(&mut self, index: usize) -> Option<&mut RequestField> {
        self.fields.get_mut(index)
    }

    /// The first field with `id`.
    pub fn find_field(&self, id: &str) -> Option<&RequestField> {
        self.fields.iter().find(|f| f.id() == id)
    }

    /// The first mutable field with `id`.
    pub fn find_field_mut(&mut self, id: &str) -> Option<&mut RequestField> {
        self.fields.iter_mut().find(|f| f.id() == id)
    }

    /// Whether every field is valid (`purple_request_group_is_valid`). Empty ⇒ valid.
    pub fn is_valid(&self) -> bool {
        self.fields.iter().all(|field| field.is_valid().is_ok())
    }

    /// Add a field (`purple_request_group_add_field`). Returns whether the group's validity flipped
    /// (the `notify::valid` the C test counts). `before` is the cached validity of the group
    /// *without* the new field, matching how the C `invalid_fields` set is consulted before the new
    /// field is inserted into it.
    pub fn add_field(&mut self, field: RequestField) -> bool {
        let before = self.valid;
        self.fields.push(field);
        self.valid = self.is_valid();
        before != self.valid
    }

    /// Recompute validity after a member field changed; returns whether it flipped.
    pub fn revalidate(&mut self) -> bool {
        let before = self.valid;
        self.valid = self.is_valid();
        before != self.valid
    }
}

// ===========================================================================
// RequestPage
// ===========================================================================

/// A page of grouped request fields (← `PurpleRequestPage`). Validity aggregates its groups; fields
/// are looked up by id across all groups.
#[derive(Clone, Debug)]
pub struct RequestPage {
    title: Option<String>,
    subtitle: Option<String>,
    groups: Vec<RequestGroup>,
    valid: bool,
    close_emissions: u32,
}

impl Default for RequestPage {
    fn default() -> Self {
        Self::new()
    }
}

impl RequestPage {
    /// An empty page (`purple_request_page_new`). Empty pages are valid.
    pub fn new() -> Self {
        Self {
            title: None,
            subtitle: None,
            groups: Vec::new(),
            valid: true,
            close_emissions: 0,
        }
    }

    /// The page title, if any.
    pub fn title(&self) -> Option<&str> {
        self.title.as_deref()
    }

    /// Set (or clear) the page title.
    pub fn set_title(&mut self, title: Option<&str>) {
        self.title = title.map(str::to_string);
    }

    /// The page subtitle, if any.
    pub fn subtitle(&self) -> Option<&str> {
        self.subtitle.as_deref()
    }

    /// Set (or clear) the page subtitle.
    pub fn set_subtitle(&mut self, subtitle: Option<&str>) {
        self.subtitle = subtitle.map(str::to_string);
    }

    /// The number of groups.
    pub fn len(&self) -> usize {
        self.groups.len()
    }

    /// Whether the page has no groups.
    pub fn is_empty(&self) -> bool {
        self.groups.is_empty()
    }

    /// The groups, in order.
    pub fn groups(&self) -> &[RequestGroup] {
        &self.groups
    }

    /// The group at `index`.
    pub fn group(&self, index: usize) -> Option<&RequestGroup> {
        self.groups.get(index)
    }

    /// The mutable group at `index`.
    pub fn group_mut(&mut self, index: usize) -> Option<&mut RequestGroup> {
        self.groups.get_mut(index)
    }

    /// Whether every group is valid (`purple_request_page_is_valid`). Empty ⇒ valid.
    pub fn is_valid(&self) -> bool {
        self.groups.iter().all(RequestGroup::is_valid)
    }

    /// Add a group (`purple_request_page_add_group`). Returns whether the page's validity flipped.
    /// `before` is the cached validity *without* the new group (see [`RequestGroup::add_field`]).
    pub fn add_group(&mut self, group: RequestGroup) -> bool {
        let before = self.valid;
        self.groups.push(group);
        self.valid = self.is_valid();
        before != self.valid
    }

    /// Recompute validity after a member group/field changed; returns whether it flipped.
    pub fn revalidate(&mut self) -> bool {
        let before = self.valid;
        self.valid = self.is_valid();
        before != self.valid
    }

    /// The first field with `id` across all groups (`purple_request_page_get_field`).
    pub fn get_field(&self, id: &str) -> Option<&RequestField> {
        self.groups.iter().find_map(|g| g.find_field(id))
    }

    /// The first mutable field with `id` across all groups.
    pub fn get_field_mut(&mut self, id: &str) -> Option<&mut RequestField> {
        self.groups.iter_mut().find_map(|g| g.find_field_mut(id))
    }

    /// Whether a field with `id` exists (`purple_request_page_exists`).
    pub fn exists(&self, id: &str) -> bool {
        self.get_field(id).is_some()
    }

    /// Whether the field with `id` is required (`purple_request_page_is_field_required`); false if
    /// no such field.
    pub fn is_field_required(&self, id: &str) -> bool {
        self.get_field(id).is_some_and(RequestField::is_required)
    }

    /// The string value of the field with `id` (`purple_request_page_get_string`); `None` on type
    /// mismatch / null / missing.
    pub fn get_string(&self, id: &str) -> Option<&str> {
        self.get_field(id)
            .and_then(RequestField::as_string)
            .and_then(StringField::value)
    }

    /// The integer value of the field with `id` (`purple_request_page_get_integer`); `0` fallback.
    pub fn get_integer(&self, id: &str) -> i32 {
        self.get_field(id)
            .and_then(RequestField::as_int)
            .map_or(0, IntField::value)
    }

    /// The boolean value of the field with `id` (`purple_request_page_get_bool`); `false` fallback.
    pub fn get_bool(&self, id: &str) -> bool {
        self.get_field(id)
            .and_then(RequestField::as_bool)
            .is_some_and(BoolField::value)
    }

    /// The selected choice item of the field with `id` (`purple_request_page_get_choice`).
    pub fn get_choice(&self, id: &str) -> Option<&LocalizedString> {
        self.get_field(id)
            .and_then(RequestField::as_choice)
            .and_then(ChoiceField::selected_item)
    }

    /// The selected account of the field with `id` (`purple_request_page_get_account`).
    pub fn get_account(&self, id: &str) -> Option<&TransportId> {
        self.get_field(id)
            .and_then(RequestField::as_account)
            .and_then(AccountField::account)
    }

    /// Emit the one-shot close (`purple_request_page_close` → the `::close` signal). Modelled as an
    /// emission counter; every call emits.
    pub fn close(&mut self) {
        self.close_emissions += 1;
    }

    /// How many times [`RequestPage::close`] has emitted.
    pub fn close_emissions(&self) -> u32 {
        self.close_emissions
    }

    /// Whether the page has been closed at least once.
    pub fn is_closed(&self) -> bool {
        self.close_emissions > 0
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    fn ls(id: &str, label: &str) -> LocalizedString {
        LocalizedString {
            id: id.to_string(),
            label: label.to_string(),
        }
    }

    // -- test_request_field.c ------------------------------------------------

    #[test]
    fn field_filled_string() {
        // ← /request-field/filled-string. NULL and "" are the same (empty); the filled transition
        // is what notify::filled counts.
        let mut field = RequestField::string("test-string", "Test string", None, false);
        assert!(!field.is_filled());

        // NULL → NULL: no change, no filled transition.
        assert!(!field.as_string_mut().unwrap().set_value(None));
        assert!(!field.is_filled());

        // NULL → "": value changes but both are empty, so no filled transition.
        assert!(!field.as_string_mut().unwrap().set_value(Some("")));
        assert!(!field.is_filled());

        // "" → "text": empty→filled transition.
        assert!(field.as_string_mut().unwrap().set_value(Some("text")));
        assert!(field.is_filled());

        // "text" → "text": no change.
        assert!(!field.as_string_mut().unwrap().set_value(Some("text")));
        assert!(field.is_filled());

        // "text" → "": filled→empty transition.
        assert!(field.as_string_mut().unwrap().set_value(Some("")));
        assert!(!field.is_filled());
    }

    #[test]
    fn field_filled_nonstring() {
        // ← /request-field/filled-nonstring. Non-strings are always filled.
        let mut field = RequestField::int("test-int", "Test int", 50, 0, 100);
        assert!(field.is_filled());

        for v in [50, 0, 100] {
            field.as_int_mut().unwrap().set_value(v);
            assert!(field.is_filled());
        }
    }

    #[test]
    fn field_valid_int() {
        // ← /request-field/valid-int. Bounds + exact error strings.
        let mut field = RequestField::int("test-int", "Test int", 50, 0, 100);
        assert!(field.is_valid().is_ok());

        field.as_int_mut().unwrap().set_value(-42);
        assert_eq!(
            field.is_valid(),
            Err("Int value -42 exceeds lower bound 0".to_string())
        );

        field.as_int_mut().unwrap().set_value(1337);
        assert_eq!(
            field.is_valid(),
            Err("Int value 1337 exceeds upper bound 100".to_string())
        );
    }

    #[test]
    fn field_valid_custom() {
        // ← /request-field/valid-custom. The subclass (bounds) validator runs BEFORE the custom
        // one, so an out-of-bounds value never reaches the custom validator.
        let called = Arc::new(AtomicBool::new(false));
        let called_v = Arc::clone(&called);

        let mut field = RequestField::int("test-int", "Test int", 50, 0, 100);
        field.set_validator(Arc::new(move |f: &RequestField| {
            called_v.store(true, Ordering::SeqCst);
            let value = f.as_int().expect("int field").value();
            if value % 2 != 0 {
                Err(format!("Value {value} is not even."))
            } else {
                Ok(())
            }
        }));

        // 50 passes bounds and is even.
        assert!(field.is_valid().is_ok());

        // -42 fails bounds; the custom validator is NOT called.
        called.store(false, Ordering::SeqCst);
        field.as_int_mut().unwrap().set_value(-42);
        assert_eq!(
            field.is_valid(),
            Err("Int value -42 exceeds lower bound 0".to_string())
        );
        assert!(!called.load(Ordering::SeqCst));

        // 23 passes bounds, so the custom validator runs and rejects the odd value.
        called.store(false, Ordering::SeqCst);
        field.as_int_mut().unwrap().set_value(23);
        assert_eq!(field.is_valid(), Err("Value 23 is not even.".to_string()));
        assert!(called.load(Ordering::SeqCst));
    }

    #[test]
    fn field_required_validity() {
        // ← /request-field/required-validity.
        let mut field = RequestField::string("test-string", "Test string", None, false);
        assert!(field.is_valid().is_ok());

        field.set_required(true);
        assert_eq!(
            field.is_valid(),
            Err("Required field is not filled.".to_string())
        );

        field.as_string_mut().unwrap().set_value(Some("valid"));
        assert!(field.is_valid().is_ok());
    }

    // -- test_request_field_choice.c ----------------------------------------

    #[test]
    fn choice_new() {
        // ← /request/field/choice/new (GObject type identity / list-model skipped-with-reason).
        let field = RequestField::choice("id", "label");
        assert_eq!(field.id(), "id");
        assert_eq!(field.label(), "label");
        assert_eq!(field.as_choice().unwrap().n_items(), 0);
    }

    #[test]
    fn choice_properties() {
        // ← /request/field/choice/properties.
        let field = RequestField::choice("choice", "Pepsi Challenge");
        assert_eq!(field.id(), "choice");
        assert_eq!(field.label(), "Pepsi Challenge");
        let choice = field.as_choice().unwrap();
        assert_eq!(choice.n_items(), 0);
        assert_eq!(choice.get_selected(), None);
        assert_eq!(choice.selected_item(), None);
    }

    #[test]
    fn choice_add_remove() {
        // ← /request/field/choice/add-remove (items-changed emission counts skipped-with-reason).
        let mut field = RequestField::choice("id", "Pepsi Challenge");
        let choice = field.as_choice_mut().unwrap();
        assert_eq!(choice.n_items(), 0);

        choice.add("pepsi", "Pepsi");
        assert_eq!(choice.n_items(), 1);
        choice.add("coca-cola", "Coca Cola");
        assert_eq!(choice.n_items(), 2);
        choice.add("rc", "Royal Crown");
        assert_eq!(choice.n_items(), 3);
        choice.add_item(ls("shasta", "Shasta"));
        assert_eq!(choice.n_items(), 4);
        choice.add("sprecher", "Sprecher");
        assert_eq!(choice.n_items(), 5);
        choice.add("jolt", "Jolt");
        assert_eq!(choice.n_items(), 6);

        // Removing non-existent items does nothing.
        assert!(!choice.remove(10000));
        assert_eq!(choice.n_items(), 6);
        assert!(!choice.remove_item(&ls("slurm", "Slurm")));
        assert_eq!(choice.n_items(), 6);
        assert!(!choice.remove_by_id("slurm"));
        assert_eq!(choice.n_items(), 6);

        // Remove the last item (index 5).
        assert!(choice.remove(5));
        assert_eq!(choice.n_items(), 5);
        // Remove the item added via add_item.
        assert!(choice.remove_item(&ls("shasta", "Shasta")));
        assert_eq!(choice.n_items(), 4);
        // Remove by id.
        assert!(choice.remove_by_id("sprecher"));
        assert_eq!(choice.n_items(), 3);
        // Remove the first item.
        assert!(choice.remove(0));
        assert_eq!(choice.n_items(), 2);
        // Clear everything.
        choice.clear();
        assert_eq!(choice.n_items(), 0);
    }

    #[test]
    fn choice_selected() {
        // ← /request/field/choice/selected.
        let mut field = RequestField::choice("id", "Pepsi Challenge");
        let choice = field.as_choice_mut().unwrap();
        assert_eq!(choice.get_selected(), None);

        for (id, label) in [
            ("pepsi", "Pepsi"),
            ("coca-cola", "Coca Cola"),
            ("rc", "Royal Crown"),
            ("shasta", "Shasta"),
            ("sprecher", "Sprecher"),
            ("jolt", "Jolt"),
        ] {
            choice.add(id, label);
        }

        choice.set_selected(2);
        assert_eq!(choice.get_selected(), Some(2));
        // Out of bounds is ignored.
        choice.set_selected(1337);
        assert_eq!(choice.get_selected(), Some(2));
        choice.set_selected(4);
        assert_eq!(choice.get_selected(), Some(4));
    }

    #[test]
    fn choice_selected_item() {
        // ← /request/field/choice/selected-item.
        let mut field = RequestField::choice("id", "Pepsi Challenge");
        let choice = field.as_choice_mut().unwrap();
        assert_eq!(choice.selected_item(), None);

        for (id, label) in [
            ("pepsi", "Pepsi"),
            ("coca-cola", "Coca Cola"),
            ("rc", "Royal Crown"),
            ("shasta", "Shasta"),
            ("sprecher", "Sprecher"),
            ("jolt", "Jolt"),
        ] {
            choice.add(id, label);
        }

        choice.set_selected(2);
        assert_eq!(choice.selected_item().unwrap().id, "rc");
        choice.set_selected(1337);
        assert_eq!(choice.selected_item().unwrap().id, "rc");
        choice.set_selected(4);
        assert_eq!(choice.selected_item().unwrap().id, "sprecher");
    }

    #[test]
    fn choice_remove_resets_selected() {
        // Derived: removing the selected position resets the selection to 0.
        let mut field = RequestField::choice("id", "Choices");
        let choice = field.as_choice_mut().unwrap();
        choice.add("a", "A");
        choice.add("b", "B");
        choice.add("c", "C");
        choice.set_selected(2);
        assert_eq!(choice.get_selected(), Some(2));
        assert!(choice.remove(2));
        assert_eq!(choice.get_selected(), Some(0));
    }

    // -- test_request_field_account.c ---------------------------------------

    #[test]
    fn account_new_without_model() {
        // ← /request/field/account/new/without-model.
        let field = RequestField::account("id", "Account", Vec::new());
        let account = field.as_account().unwrap();
        assert!(account.model().is_empty());
        assert_eq!(account.account(), None);
    }

    #[test]
    fn account_new_with_model() {
        // ← /request/field/account/new/with-model.
        let model = vec![TransportId::new("test/test")];
        let field = RequestField::account("id", "Account", model);
        assert_eq!(field.as_account().unwrap().model().len(), 1);
    }

    #[test]
    fn account_properties() {
        // ← /request/field/account/properties.
        let acct = TransportId::new("test/test");
        let model = vec![acct.clone()];
        let mut field = RequestField::account("id", "Account", model.clone());
        field
            .as_account_mut()
            .unwrap()
            .set_account(Some(acct.clone()));

        let account = field.as_account().unwrap();
        assert_eq!(account.account(), Some(&acct));
        assert_eq!(account.model(), model.as_slice());
    }

    #[test]
    fn account_supports_null() {
        // ← /request/field/account/account-supports-null.
        let acct = TransportId::new("test/test");
        let mut field = RequestField::account("id", "account", vec![acct.clone()]);
        assert_eq!(field.as_account().unwrap().account(), None);

        field
            .as_account_mut()
            .unwrap()
            .set_account(Some(acct.clone()));
        assert_eq!(field.as_account().unwrap().account(), Some(&acct));

        field.as_account_mut().unwrap().set_account(None);
        assert_eq!(field.as_account().unwrap().account(), None);
    }

    // -- test_request_field_image.c -----------------------------------------

    #[test]
    fn image_new_normal() {
        // ← /request/field/image/new/normal.
        let image = ImageRef("img".to_string());
        let field = RequestField::image("id", "label", Some(image.clone()));
        assert_eq!(field.as_image().unwrap().image(), Some(&image));
    }

    #[test]
    fn image_new_null() {
        // ← /request/field/image/new/null.
        let field = RequestField::image("id", "label", None);
        assert_eq!(field.as_image().unwrap().image(), None);
    }

    #[test]
    fn image_properties() {
        // ← /request/field/image/properties.
        let image = ImageRef("img".to_string());
        let field = RequestField::image("42", "The Answer", Some(image.clone()));
        assert_eq!(field.id(), "42");
        assert_eq!(field.label(), "The Answer");
        assert_eq!(field.as_image().unwrap().image(), Some(&image));
    }

    #[test]
    fn image_supports_null() {
        // ← /request/field/image/image-supports-null.
        let image = ImageRef("img".to_string());
        let mut field = RequestField::image("id", "label", Some(image.clone()));
        assert_eq!(field.as_image().unwrap().image(), Some(&image));

        field.as_image_mut().unwrap().set_image(None);
        assert_eq!(field.as_image().unwrap().image(), None);
    }

    // -- test_request_field_list.c ------------------------------------------

    #[test]
    fn list_new_normal() {
        // ← /request/field/list/new/normal (list-model identity skipped-with-reason).
        let field = RequestField::list("id", "label");
        assert_eq!(field.as_list().unwrap().n_items(), 0);
    }

    #[test]
    fn list_properties() {
        // ← /request/field/list/properties.
        let mut field = RequestField::list("naughty", "Santa's Naughty List");
        field.as_list_mut().unwrap().set_multi_select(true);
        assert_eq!(field.id(), "naughty");
        assert_eq!(field.label(), "Santa's Naughty List");
        let list = field.as_list().unwrap();
        assert!(list.multi_select());
        assert_eq!(list.n_items(), 0);
    }

    #[test]
    fn list_add_remove() {
        // ← /request/field/list/add-remove. Duplicates are allowed; remove-missing → false.
        let mut field = RequestField::list("naughty", "Santa's Naughty List");
        let list = field.as_list_mut().unwrap();
        assert_eq!(list.n_items(), 0);

        list.add_item(ls("grim", "Gary"));
        assert_eq!(list.n_items(), 1);
        list.add_item(ls("pidgy", "Pidgy"));
        assert_eq!(list.n_items(), 2);
        // Duplicate id is allowed.
        list.add("grim", "Gary");
        assert_eq!(list.n_items(), 3);
        list.add_item(ls("robotichead", "robotichead"));
        assert_eq!(list.n_items(), 4);

        // Remove the first "grim".
        assert!(list.remove_by_id("grim"));
        assert_eq!(list.n_items(), 3);
        assert!(list.remove_by_id("pidgy"));
        assert_eq!(list.n_items(), 2);
        // Remove a non-existent id (the C NULL case) → false.
        assert!(!list.remove_by_id("does-not-exist"));
        assert_eq!(list.n_items(), 2);

        list.clear();
        assert_eq!(list.n_items(), 0);
    }

    #[test]
    fn list_single_selection() {
        // ← /request/field/list/single-selection. Selecting a new item replaces the old one.
        let mut field = RequestField::list("testing", "Testing");
        let list = field.as_list_mut().unwrap();
        list.add("grim", "Grim");
        list.add("pidgy", "Pidgy");
        list.add("csb6", "csb6");

        assert_eq!(list.selected().len(), 0);
        list.clear_selected();
        assert_eq!(list.selected().len(), 0);

        assert!(list.select_item("grim"));
        assert_eq!(list.selected().len(), 1);
        // Re-selecting the same item is a no-op.
        assert!(!list.select_item("grim"));
        assert_eq!(list.selected().len(), 1);
        // Single-select replaces the prior selection.
        assert!(list.select_item("csb6"));
        assert_eq!(list.selected().len(), 1);
        assert_eq!(list.selected()[0].id, "csb6");

        list.clear_selected();
        assert_eq!(list.selected().len(), 0);
    }

    #[test]
    fn list_multi_selection() {
        // ← /request/field/list/multi-selection. Selecting adds without replacing.
        let mut field = RequestField::list("testing", "Testing");
        let list = field.as_list_mut().unwrap();
        list.set_multi_select(true);
        list.add("grim", "Grim");
        list.add("pidgy", "Pidgy");
        list.add("csb6", "csb6");

        assert_eq!(list.selected().len(), 0);
        list.clear_selected();
        assert_eq!(list.selected().len(), 0);

        assert!(list.select_item("grim"));
        assert_eq!(list.selected().len(), 1);
        assert!(!list.select_item("grim"));
        assert_eq!(list.selected().len(), 1);
        // Multi-select adds a second selection.
        assert!(list.select_item("csb6"));
        assert_eq!(list.selected().len(), 2);

        list.clear_selected();
        assert_eq!(list.selected().len(), 0);
    }

    // -- test_request_group.c -----------------------------------------------

    #[test]
    fn group_valid() {
        // ← /request-group/valid. `flip` mirrors the notify::valid count the C test asserts.
        let mut group = RequestGroup::new(Some("test-group"));

        // Empty groups are always valid.
        assert!(group.is_valid());

        // An added valid field keeps the group valid, no flip.
        let flip = group.add_field(RequestField::int("test-int", "Test int", 50, 0, 100));
        assert!(group.is_valid());
        assert!(!flip);

        // Making the field invalid makes the group invalid — a flip.
        group
            .find_field_mut("test-int")
            .unwrap()
            .as_int_mut()
            .unwrap()
            .set_value(-42);
        let flip = group.revalidate();
        assert!(!group.is_valid());
        assert!(flip);

        // Adding an invalid field keeps the group invalid, no flip.
        let flip = group.add_field(RequestField::int("invalid", "Invalid", -42, 0, 100));
        assert!(!group.is_valid());
        assert!(!flip);

        // Adding a valid field to an already invalid group does not flip it.
        let flip = group.add_field(RequestField::int("valid", "Valid", 42, 0, 100));
        assert!(!group.is_valid());
        assert!(!flip);

        // Making one field valid while another stays invalid keeps the group invalid.
        group
            .find_field_mut("test-int")
            .unwrap()
            .as_int_mut()
            .unwrap()
            .set_value(42);
        let flip = group.revalidate();
        assert!(!group.is_valid());
        assert!(!flip);

        // Making the last invalid field valid makes the group valid again — a flip.
        group
            .find_field_mut("invalid")
            .unwrap()
            .as_int_mut()
            .unwrap()
            .set_value(42);
        let flip = group.revalidate();
        assert!(group.is_valid());
        assert!(flip);
    }

    // -- test_request_page.c ------------------------------------------------

    fn page_string_validator(field: &RequestField) -> Result<(), String> {
        let value = field
            .as_string()
            .expect("string field")
            .value()
            .unwrap_or("");
        if value == "valid" {
            Ok(())
        } else {
            Err(format!("String value is not valid: {value}"))
        }
    }

    fn valid_group(name: &str) -> RequestGroup {
        let mut group = RequestGroup::new(Some(name));
        let field_name = format!("{name}-string");
        let mut field = RequestField::string(&field_name, &field_name, Some("valid"), false);
        field.set_validator(Arc::new(page_string_validator));
        group.add_field(field);
        group
    }

    fn invalid_group(name: &str) -> RequestGroup {
        let mut group = RequestGroup::new(Some(name));
        let field_name = format!("{name}-string");
        let mut field = RequestField::string(&field_name, &field_name, Some("invalid"), false);
        field.set_validator(Arc::new(page_string_validator));
        group.add_field(field);
        group
    }

    fn make_group_valid(group: &mut RequestGroup) {
        for index in 0..group.len() {
            if let Some(field) = group.field_mut(index).unwrap().as_string_mut() {
                field.set_value(Some("valid"));
            }
        }
    }

    fn make_group_invalid(group: &mut RequestGroup) {
        for index in 0..group.len() {
            if let Some(field) = group.field_mut(index).unwrap().as_string_mut() {
                field.set_value(Some("invalid"));
            }
        }
    }

    #[test]
    fn page_new() {
        // ← /request/page/new.
        let page = RequestPage::new();
        assert!(page.is_empty());
        assert_eq!(page.title(), None);
        assert_eq!(page.subtitle(), None);
    }

    #[test]
    fn page_properties() {
        // ← /request/page/properties.
        let mut page = RequestPage::new();
        page.set_title(Some("titled"));
        page.set_subtitle(Some("subtitled"));
        assert_eq!(page.title(), Some("titled"));
        assert_eq!(page.subtitle(), Some("subtitled"));
    }

    #[test]
    fn page_valid() {
        // ← /request/page/valid. `flip` mirrors the notify::valid count the C test asserts.
        let mut page = RequestPage::new();

        // Empty pages are always valid.
        assert!(page.is_valid());

        // An added valid group keeps the page valid, no flip.
        let flip = page.add_group(valid_group("group1"));
        assert!(page.is_valid());
        assert!(!flip);

        // Making the group invalid makes the page invalid — a flip.
        make_group_invalid(page.group_mut(0).unwrap());
        let flip = page.revalidate();
        assert!(!page.is_valid());
        assert!(flip);

        // Adding an invalid group keeps the page invalid, no flip.
        let flip = page.add_group(invalid_group("group2"));
        assert!(!page.is_valid());
        assert!(!flip);

        // Adding a valid group to an already invalid page does not flip it.
        let flip = page.add_group(valid_group("group3"));
        assert!(!page.is_valid());
        assert!(!flip);

        // Making one group valid while another stays invalid keeps the page invalid.
        make_group_valid(page.group_mut(0).unwrap());
        let flip = page.revalidate();
        assert!(!page.is_valid());
        assert!(!flip);

        // Making the last invalid group valid makes the page valid again — a flip.
        make_group_valid(page.group_mut(1).unwrap());
        let flip = page.revalidate();
        assert!(page.is_valid());
        assert!(flip);
    }

    #[test]
    fn page_close() {
        // ← /request/page/close. The ::close signal is modelled as an emission counter.
        let mut page = RequestPage::new();
        assert_eq!(page.close_emissions(), 0);
        assert!(!page.is_closed());

        page.close();
        assert_eq!(page.close_emissions(), 1);
        assert!(page.is_closed());
    }

    #[test]
    fn page_field_lookup_and_typed_getters() {
        // Derived: cross-group field lookup + the typed getters with type-mismatch fallbacks.
        let mut page = RequestPage::new();
        let mut group = RequestGroup::new(Some("group"));
        group.add_field(RequestField::string("s", "S", Some("hello"), false));
        group.add_field(RequestField::int("i", "I", 7, 0, 100));
        group.add_field(RequestField::boolean("b", "B", true));

        let mut choice = RequestField::choice("c", "C");
        choice.as_choice_mut().unwrap().add("x", "X");
        choice.as_choice_mut().unwrap().set_selected(0);
        group.add_field(choice);

        let acct = TransportId::new("t/acct");
        let mut account = RequestField::account("a", "A", vec![acct.clone()]);
        account
            .as_account_mut()
            .unwrap()
            .set_account(Some(acct.clone()));
        group.add_field(account);

        page.add_group(group);

        assert!(page.exists("s"));
        assert!(!page.exists("nope"));
        assert!(page.get_field("i").is_some());
        assert!(page.get_field_mut("b").is_some());

        assert_eq!(page.get_string("s"), Some("hello"));
        assert_eq!(page.get_integer("i"), 7);
        assert!(page.get_bool("b"));
        assert_eq!(page.get_choice("c").unwrap().id, "x");
        assert_eq!(page.get_account("a"), Some(&acct));

        // Type-mismatch fallbacks.
        assert_eq!(page.get_string("i"), None);
        assert_eq!(page.get_integer("s"), 0);
        assert!(!page.get_bool("s"));
        assert_eq!(page.get_choice("s"), None);
        assert_eq!(page.get_account("s"), None);

        assert!(!page.is_field_required("s"));
    }
}
