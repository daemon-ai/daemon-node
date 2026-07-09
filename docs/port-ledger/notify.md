# Port ledger — W2-G `port-notify`

Authorization requests, add-contact requests, typed notifications, and the host
`NotificationManager`, ported from libpurple.

DTOs land in a new module `crates/contracts/daemon-api/src/notify.rs` (`lib.rs`
gains only `mod notify;` + a re-export). The host `NotificationManager` lands in
`crates/substrate/daemon-host/src/notifications.rs`. This package **touches the
wire**: `NotificationInfo`/`NotificationKind`/`AuthorizationRequest`/
`AddContactRequest` cross the wire (reachable from `ApiResponse::Notifications`),
so they are mirrored in `daemon-api.cddl`, derive `Arbitrary`, and gain CBOR
fixtures; a `NotificationList` op and a payload-free `NodeEvent::NotificationsChanged`
pointer are appended (wire v37; the integration branch bumps `WireVersion` to 37).

"Done" = every row below is green or explicitly skipped-with-reason.

Legend: **P** = ported (green), **S** = skipped (reason given), **+** = extra
Rust case with no direct libpurple `g_test` (derived from the `.c` impl /
daemon-native wire surface). **D** = derived row (no libpurple `g_test` file —
derived from the implementation `.c`, per the work package).

---

## 1. `purpleauthorizationrequest.c` — `test_authorization_request.c` (8 g_test)

Module `daemon_api::notify` (`AuthorizationRequest`).

| # | libpurple case | Rust test (`notify::tests`) | status |
|---|---|---|---|
| 1 | `/request-authorization/new` | `authz_new` | P |
| 2 | `/request-authorization/properties` | `authz_properties` | P |
| 3 | `/request-authorization/accept` | `authz_accept_idempotent` | P |
| 4 | `/request-authorization/accept-deny` | `authz_accept_then_deny_rejected` | P |
| 5 | `/request-authorization/deny` | `authz_deny_idempotent` | P |
| 6 | `/request-authorization/deny-accept` | `authz_deny_then_accept_rejected` | P |
| 7 | `/request-authorization/deny-message/null` | `authz_deny_message_null` | P |
| 8 | `/request-authorization/deny-message/non-null` | `authz_deny_message_non_null` | P |

Truth table (`purple_authorization_request_{accept,deny}`): a single `handled`
flag couples the two. The first `accept()` OR `deny(msg)` succeeds
(`Ok`), records the decision, and sets `handled`; every subsequent `accept()` or
`deny()` is rejected (`Err(RequestError::AlreadyHandled)`) — the C
`g_return_if_fail(handled == FALSE)` no-op / CRITICAL, modeled as a `Result`.
`deny(message)` carries the **argument** message through (independent of the
stored `message` field), matching the `denied` signal's `message` param
(`None`/`Some` both exercised).

## 2. `purpleaddcontactrequest.c` — no test file (rows derived from the impl `.c`)

Module `daemon_api::notify` (`AddContactRequest`). No `test_add_contact_request.c`
exists; rows are **derived** from `purpleaddcontactrequest.c`.

| # | derived case | Rust test | status |
|---|---|---|---|
| 9 | `purple_add_contact_request_new` (contact stored) | `add_contact_new` | D/P |
| 10 | contact + message accessors | `add_contact_properties` | D/P |
| 11 | `set_message` roundtrip | `add_contact_set_message` | D/P |
| 12 | `purple_add_contact_request_add` idempotency (double-add rejected) | `add_contact_add_idempotent` | D/P |

`add()` mirrors accept/deny: first call `Ok`, sets `handled`; a second is
`Err(AlreadyHandled)`.

## 3. `purplenotification.c` — `test_notification.c` (2 g_test)

Module `daemon_api::notify` (`NotificationInfo`, generic kind).

| # | libpurple case | Rust test | status |
|---|---|---|---|
| 13 | `/notification/new` | `notification_new_generates_id` | P |
| 14 | `/notification/properties` | `notification_properties` | P |

`NotificationInfo::new(id, title)` auto-generates a non-empty `id` when `None`
(`g_uuid_string_random`) and stamps `created_ms` to now
(`g_date_time_new_now_local`). The properties test round-trips the modeled
subset (`created_ms`, `icon_name`, `interactive`, `persistent`, `read`,
`subtitle`, `title`). `purple_notification_compare` (created-timestamp order) is
ported as `+` (`notification_compare_by_created`).

| — | `purple_notification_compare` (impl) | `notification_compare_by_created` | + |
| — | `purple_notification_delete` idempotency (impl) | `notification_delete_idempotent` | + |

## 4. `purplenotificationaddcontact.c` — `test_notification_add_contact.c` (3 g_test)

`NotificationKind::AddContact` + `add_contact_title`.

| # | libpurple case | Rust test | status |
|---|---|---|---|
| 15 | `/notification/add-contact/new` | `notification_add_contact_new` | P |
| 16 | `/notification/add-contact/properties` | `notification_add_contact_properties` | P |
| 17 | `/notification/add-contact/updates-title` | `notification_add_contact_title` | P (title derivation) / S (signal-driven live update) |

The C `updates-title` test asserts the title (a) contains the remote contact's
name after construction and (b) updates when the contact alias changes, tracked
via a `notify::title` signal counter. We port the **title derivation**
(`add_contact_title(request)` puts the contact's `name_for_display()` into the
title; recomputing after the contact name changes yields the new title). The
signal-counter half is a GObject artifact (no signals in the DTO) — **S**.
`account` is left `None` (the daemon `ContactInfo` has no account field; the C
title's account-name half is not reconstructable here).

## 5. `purplenotificationauthorizationrequest.c` — `test_notification_authorization_request.c` (3 g_test)

`NotificationKind::Authorization` + `authorization_title`.

| # | libpurple case | Rust test | status |
|---|---|---|---|
| 18 | `/notification-request-authorization/new` | `notification_authz_new` | P |
| 19 | `/notification-request-authorization/properties` | `notification_authz_properties` | P |
| 20 | `/notification-request-authorization/updates-title` | `notification_authz_title` | P (title derivation) / S (signal-driven live update) |

Same split as #17: title-derivation ported, signal counter skipped.

## 6. `purplenotificationlink.c` — `test_notification_link.c` (3 g_test)

`NotificationKind::Link` + `link_text` null handling.

| # | libpurple case | Rust test | status |
|---|---|---|---|
| 21 | `/notification/link/new` | `notification_link_new` | P |
| 22 | `/notification/link/properties` | `notification_link_properties` | P |
| 23 | `/notification/link/null-link-text` | `notification_link_null_link_text` | P |

`link_text()` returns `link_text` when non-empty, else falls back to `link_uri`
(`purple_notification_link_get_link_text`).

## 7. `purplenotificationconnectionerror.c` — no dedicated test (exercised via the manager)

`NotificationKind::ConnectionError` (account-bound). No
`test_notification_connection_error.c`; the type is exercised by the manager's
`remove-with-account/all` (a connection-error notification is *transient* — only
removed when `all == true`). Constructor row is **derived**.

| # | derived case | Rust test | status |
|---|---|---|---|
| 24 | `purple_notification_connection_error_new` (account-bound) | `notification_connection_error_new` | D/P |

## 8. `purplenotificationmanager.c` — `test_notification_manager.c` (8 g_test)

Module `daemon_host::notifications` (`NotificationManager`).

| # | libpurple case | Rust test (`notifications::tests`) | status |
|---|---|---|---|
| 25 | `/notification-manager/add-remove` | `manager_add_remove` | P |
| 26 | `/notification-manager/double-add` | `manager_double_add_rejected` | P |
| 27 | `/notification-manager/double-remove` | `manager_double_remove` | P |
| 28 | `/notification-manager/remove-with-account/simple` | `manager_remove_with_account_simple` | P |
| 29 | `/notification-manager/remove-with-account/mixed` | `manager_remove_with_account_mixed` | P |
| 30 | `/notification-manager/remove-with-account/all` | `manager_remove_with_account_all` | P |
| 31 | `/notification-manager/read-propagation` | `manager_read_propagation` | P |
| 32 | `/notification-manager/remove-on-delete` | `manager_remove_on_delete` | P |

Manager semantics (`purplenotificationmanager.c`):
- `add` prepends (index 0 = newest), rejects a double-add by `id`
  (C `g_ptr_array_find` by pointer → C emits a `g_warning`; modeled as
  `AddOutcome::DuplicateRejected`), and increments `unread_count` when the
  notification is unread. **Divergence:** the C double-add aborts a subprocess
  via `g_warning`; the Rust port returns a rejected-outcome enum (a
  `g_return`/warn precondition is not portable), so `manager_double_add_rejected`
  asserts the outcome + that only one copy is stored.
- `remove` removes by `id`, decrements `unread_count` when the removed item was
  unread, and is a no-op on a second remove (double-remove).
- `remove_with_account(account, all)` removes every notification whose `account`
  matches; a `ConnectionError` notification is transient — removed only when
  `all == true` (C `can_remove = all` for connection-error). Returns the removed
  count.
- `set_read(id, read)` adjusts `unread_count` only on an actual state change and
  reports the transition (`ReadChange::{MarkedRead, MarkedUnread, Unchanged}`) —
  the daemon analog of the C `read`/`unread` signals + the `unread-count`
  property notify. **Divergence:** we assert `unread_count` values + the returned
  transition rather than GObject `notify::unread-count` emission counts.
- `delete(id)` removes the notification (the C `deleted` signal → manager
  `remove`), the `remove-on-delete` path.
- `clear` empties the manager.

## `test_notification_link.c` etc. — GObject property-bag rows

Every `*/new` and `*/properties` case that only asserts GObject construction +
property round-trip is ported as a Rust struct construction + field assertion (P
above). Pure GObject signal-emission assertions (the `notify::title` counters,
`added`/`removed`/`read`/`unread` signal counters) are represented in the Rust
port by the manager's return-value transitions and by direct state assertions;
the raw signal-emission-count mechanics are **S** (GObject artifacts).

---

## Totals

- libpurple `g_test` cases enumerated (files with a test): **24**
  (authorization 8, notification 2, notification-add-contact 3,
  notification-authorization 3, notification-link 3, notification-manager 8) —
  = 27 with the 3 title cases counted once each. Enumerated distinct `g_test`
  funcs: **27**.
- Ported (P): **27** (all enumerated cases; #17/#20 split — title derivation
  ported, signal-counter half noted S).
- Derived rows (D, no libpurple test file): **5** (`AddContactRequest` ×4,
  `ConnectionError` constructor ×1).
- Extra derived/daemon-native (+): **2** (`purple_notification_compare`,
  `purple_notification_delete` idempotency).
- Skipped (S): the signal-emission-count mechanics folded into #17/#20 and the
  manager transitions (GObject artifacts; no signals in the DTO/manager).

## Wire additions (wire v37 — integration bumps `WireVersion` to 37)

- Types: `NotificationInfo`, `NotificationKind` (`Generic`/`AddContact`/
  `Authorization`/`Link`/`ConnectionError`), `AuthorizationRequest`,
  `AddContactRequest` — all `Serialize`/`Deserialize` + feature-gated `Arbitrary`.
- Op: `ApiRequest::NotificationList` → `ApiResponse::Notifications(Vec<NotificationInfo>)`
  (`ControlApi::notification_list`, default empty; classified `ControlRead` in
  `authz.rs`, `NotSessionTouching` in the ownership matrix).
- Event: `NodeEvent::NotificationsChanged` (payload-free pointer; clients re-list),
  emitted by `NodeApiImpl::emit_notifications_changed` (mirrors `emit_contacts_changed`).
- CDDL: `notification-info`, `notification-kind` (+ arm rules), `authorization-request`,
  `add-contact-request`, `request-notification-list`, `response-notifications`,
  `node-event-notifications-changed` (appended to the respective unions).
- Fixtures: `request-notification-list.cbor`, `response-notifications.cbor`; the
  `NotificationsChanged` event added to `response-events-page.cbor`.

## Helper APIs added

- `AuthorizationRequest::{new, is_handled, accept, deny}` + `RequestError`.
- `AddContactRequest::{new, is_handled, add}`.
- `NotificationInfo::{new, new_add_contact, new_authorization, new_link,
  new_connection_error, link_text, compare, delete, is_deleted}`.
- `notify::{add_contact_title, authorization_title}`.
- `NotificationManager::{new, len, is_empty, unread_count, list, add, remove,
  remove_with_account, clear, set_read, delete}` + `AddOutcome` + `ReadChange`.
