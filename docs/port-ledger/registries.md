# Port ledger — W1-D "port-registries"

Host-side registry / manager test coverage: **AdapterRegistry**, **cron scheduler**,
**command registry**, **credential manager**. This file is the executable spec: every
libpurple `g_test` case in scope maps to a planned Rust test (or is explicitly
skipped-with-reason), plus daemon-native rows for behaviors with no libpurple analogue.

Reference C sources (read-only):
`/home/j/experiments/multiprotocol-instant-messengers/pidgin-496de266ac6c/libpurple/tests/`.

Status legend: **green-parity** = test passes against current code on first run (porting
existing correct behavior); **red→green** = test was red, an implementation fix made it
green; **skipped** = no daemon analogue, reason given. "divergence" rows note an intentional
behavioral difference between the daemon and libpurple.

---

## Cluster 1 — AdapterRegistry (`crates/substrate/daemon-host/src/adapters.rs`)

Daemon-native. The declarative transport-adapter registry is the daemon analogue of
libpurple's `PurpleProtocolManager` (a `GListModel` of `PurpleProtocol`). It had **zero**
direct tests before this package. Mirrors the spirit of `test_protocol_manager.c` plus the
daemon-specific `messaging()` capability probe and `TransportId` lookup.

| libpurple case (`test_protocol_manager.c`) | Rust test | File | Status |
|---|---|---|---|
| `/protocol-manager/new` (empty, is a list model) | `empty_registry_is_inert` | adapters.rs | green-parity |
| `/protocol-manager/properties` (item-type, n-items==0) | `empty_registry_is_inert` (n_items==0 → `infos()`/`instances()` empty) | adapters.rs | green-parity |
| — daemon-native: register + enumerate `infos()`, ordered first-registered-first | `register_orders_and_enumerates_infos` | adapters.rs | green-parity |
| — daemon-native: `instances()` aggregates across adapters | `instances_aggregate_across_adapters` | adapters.rs | green-parity |
| — daemon-native: `messaging()` probe (adapter that is / isn't a `MessagingProtocol`) | `messaging_probe_distinguishes_messaging_from_generic` | adapters.rs | green-parity |
| — daemon-native: `MessagingProtocol` default feature probes (`conversations()`/… → None, `validate_account` → Ok) — analogue of `test_credential_provider_empty.c` default-impl behavior | `messaging_feature_probes_default_to_none` | adapters.rs | green-parity |
| — daemon-native: lookup by `family` and by `TransportId` (exact, `family/…`, `family:…`, miss) | `lookup_by_family_and_transport` | adapters.rs | green-parity |
| — daemon-native (divergence): duplicate `family` — both retained in `infos()`; `adapter_for_family` returns the first. libpurple's manager *rejects* a duplicate id; the daemon registry is an ordered `Vec` with no dedup (registration is a trusted host-assembly step, not a runtime plugin add). | `duplicate_family_both_retained_first_wins` | adapters.rs | green-parity (divergence noted) |

## Cluster 2 — Cron scheduler

Ported from `test_scheduled_task.c`. libpurple's `PurpleScheduledTask` is a single
GObject with an in-process GLib timeout; the daemon splits the same semantics across
`CronOps` (durable CRUD + first-fire, `daemon-host/src/cron.rs`) and the resident
`CronWorker` scheduler tick (`daemon-node/src/cron/{schedule,worker}.rs`). Schedule
arithmetic (`should_fire`/`advanced`) is pure and covered directly; firing is covered
end-to-end via `tick_once` against an `InMemoryStore` + `MockProvider` profile.

| libpurple case (`test_scheduled_task.c`) | Rust test | File | Status |
|---|---|---|---|
| `/scheduled-task/new`, `/properties` (default state UNSCHEDULED, no execute-at) | `create_disabled_has_no_next_fire` (a disabled create is the "unscheduled" analogue: paused, `next_fire_unix == None`) | cron.rs | green-parity |
| — daemon-native: an enabled create computes a first fire | `create_enabled_sets_next_fire` | cron.rs | green-parity |
| `/scheduled-task/schedule/cancelled` (cancel → CANCELLED, does not fire) | `pause_clears_next_fire_resume_recomputes` (pause → `next_fire None`, not due; resume recomputes) | cron.rs | green-parity |
| `/scheduled-task/schedule/cancelled` — end-to-end "cancelled doesn't fire" | `tick_skips_paused_job` | worker.rs | green-parity |
| `/scheduled-task/schedule/reschedule` (reschedule replaces previous execute-at) | `update_replaces_schedule_single_job` (update recomputes next fire; still one job) | cron.rs | green-parity |
| `/scheduled-task/schedule/past` (past execute-at is refused) | `advanced_one_shot_past_exhausts` — **divergence**: the daemon does not error on a past one-shot; it yields `next_fire None` (job never fires / is auto-deleted by the tick). A recurring schedule in the past instead fast-forwards. | schedule.rs | green-parity (divergence noted) |
| — daemon-native: a recently-past due fire still runs (catch-up grace) | `should_fire_recent_past_due_within_grace` | schedule.rs | green-parity |
| — daemon-native: a stale miss beyond tolerance is skipped (Skip policy) | `should_fire_skips_stale_beyond_tolerance` | schedule.rs | green-parity |
| `/scheduled-task/schedule/normal` (schedule → SCHEDULED → fires once → EXECUTED) | `tick_fires_due_recurring_and_rearms` | worker.rs | green-parity |
| `/scheduled-task/schedule/reuse` (re-schedule after execution fires again) | `advanced_recurring_rearms_future` (post-fire the recurring job re-arms with a future fire) + `tick_fires_due_recurring_and_rearms` (asserts re-arm) | schedule.rs / worker.rs | green-parity |
| — daemon-native: multi-period downtime collapses to a single next fire (no thundering herd) | `advanced_fast_forwards_stale_downtime` | schedule.rs | green-parity |
| — daemon-native: `update`/`pause` on an unknown job id error cleanly | `mutate_unknown_job_errors` | cron.rs | green-parity |
| `/scheduled-task/properties` GObject property plumbing (cancellable/tags/subtitle/…) | — | — | skipped: GObject-property introspection has no daemon analogue (the wire `CronSpec` is a plain serde struct; field round-trip is owned by daemon-api conformance, out of this package's scope). |

## Cluster 3 — Command registry (`crates/substrate/daemon-host/src/commands.rs`)

Ported from `test_command_manager.c`. The daemon's `CommandRegistry` is a build-once,
alias-aware, first-wins catalog (not a mutable, priority-stacked, per-conversation-filtered
`PurpleCommandManager`). Existing tests already cover resolve-by-name/alias/case, provider
fold-in, built-in collision rejection, and the access gate; this package adds the missing
find-and-execute dispatch and duplicate/enumeration edges without rewriting them.

| libpurple case (`test_command_manager.c`) | Rust test | File | Status |
|---|---|---|---|
| `/command-manager/new` (empty manager, 0 items) | `empty_registry_reports_empty` | commands.rs | green-parity (new) |
| `/command-manager/find` (find by name; unknown → none) | `builtins_resolve_by_name_alias_and_case` | commands.rs | **existing** — do not rewrite |
| `/command-manager/add-remove` (add; duplicate add ignored) | `provider_commands_fold_in_with_source_and_aliases` + `duplicate_provider_registration_first_wins` | commands.rs | existing + new |
| `/command-manager/add-remove` (remove; name+source must match) | — | — | skipped: the registry is build-once (no `remove`/`remove_all_with_source`); there is no runtime unregister path to cover. Divergence from libpurple's mutable manager. |
| `/command-manager/remove-all-with-source` | — | — | skipped: same reason (no removal API). |
| `/command-manager/find-and-execute` (dispatch to handler) | `resolve_provider_command_executes_via_owner` (resolve → `Owner::Provider` → `run_command` executes) | commands.rs | green-parity (new) |
| `/command-manager/find-all` (all registrations for a name, priority-ordered) | — | — | skipped: the daemon rejects a duplicate name/alias (first-wins) rather than stacking multiple registrations with priority; there is no "find all for a name" surface. Divergence noted. `duplicate_provider_registration_first_wins` documents the first-wins choice. |
| `/command-manager/get-commands-for-conversation` (per-conversation tag filtering) | — | — | skipped: command visibility in the daemon is by `CommandScope` (Node/Session) + `min_access` tier (already covered by `access_gate_*`), not per-conversation tag filtering. Divergence noted. |
| — daemon-native: `specs()` enumerates the catalog; `len`/`is_empty` | `empty_registry_reports_empty` + `builtins_catalog_is_enumerable` | commands.rs | green-parity (new) |
| — daemon-native: a provider alias colliding with an existing entry is rejected whole | `provider_alias_collision_rejects_whole_entry` | commands.rs | green-parity (new) |

## Cluster 4 — Credential manager (`daemon-host/src/{credstore,credentials}.rs`)

Ported from `test_credential_manager.c` / `test_credential_provider_normal.c` /
`test_credential_provider_empty.c`. The daemon has **no** single "credential manager with
one active provider": secrets live in a per-profile `CredentialStore`, a `CredentialSource`
provisions them, and a `CredentialAuthority`/`MultiProfileStoreBroker` mints leases. So the
manager cases map to *store/source/broker* edges. Existing tests already cover set/get/remove,
redaction, multi-key pool + rotation, mode-awareness, multi-profile serving, and per-profile
revoke; this package only adds the **missing** clean-handling-of-non-existent / no-op edges.

| libpurple case | Rust test | File | Status |
|---|---|---|---|
| `/credential-manager/new`, `add-remove` (register/double-register) | `mem_store_set_get_remove_redacts`, `mem_store_multi_key_pool`, `file_store_set_is_create_or_update_no_duplicate` | credstore.rs | **existing** — set is idempotent create-or-update (the daemon's "double registration is a no-op replace"); do not rewrite |
| `/credential-manager/set-active/non-existent` (operate on unknown provider → error/clean) | `revoke_profile_on_unacquired_profile_is_noop` (broker op on a never-acquired profile is a clean no-op; a later acquire+use for it still works) | credentials.rs | green-parity (new; daemon has no active-provider — divergence: unknown profile is a clean no-op / fallback, not an error) |
| `/credential-manager/no-provider/{read,write,clear}-password-async` (op with no active provider fails cleanly) | `mem_store_ops_on_unset_profile_are_clean`, `file_store_ops_on_unset_profile_are_clean` (get → None, remove → Ok no-op, keys → empty) + `pooled_source_rotate_revoke_unknown_cap_are_noops` | credstore.rs | green-parity (new) |
| `/credential-manager/set-active/{null,normal}`; provider `{read,write,clear}` | `source_provisions_stored_then_fallback`, `store_source_is_mode_aware`, `pooled_source_*`, `multi_profile_broker_*`, `revoke_profile_invalidates_outstanding_lease` | credstore.rs / credentials.rs | **existing** — do not rewrite |
| `test_credential_provider_normal.c` read/write/clear reach the impl | `source_provisions_stored_then_fallback` (read), `mem_store_set_get_remove_redacts` (write/clear via store) + new `store_source_revoke_and_store_remove_clear_secret` (clear-password analogue) | credstore.rs | existing + new |
| `test_credential_provider_empty.c` (default methods return NOT_IMPLEMENTED) | — | — | skipped: the daemon `CredentialSource` trait has no defaulted async methods that error; `Native`-mode refusal on a store-backed source (`store_source_is_mode_aware`, existing) is the closest "unsupported operation" analogue. The default-impl-returns-default pattern is instead exercised on `MessagingProtocol` (`messaging_feature_probes_default_to_none`, Cluster 1). |

---

## Summary

- **libpurple cases in scope:** 32 across 6 files (protocol-manager 2, scheduled-task 7,
  command-manager 7, credential-manager 11, credential-provider-normal 5 — the
  credential-provider-empty 5 fold into the same "default-impl" row).
- **Ported (green-parity or covered by existing):** see per-row status.
- **Daemon-native rows (no libpurple analogue):** AdapterRegistry ordering/instances/lookup/
  dup, cron catch-up/fast-forward/unknown-job, command enumeration/dispatch/alias-collision,
  credential no-op edges.
- **Skipped (with reason):** GObject property introspection; command remove/remove-all/
  find-all/per-conversation filtering (build-once first-wins registry — divergence);
  credential-provider-empty NOT_IMPLEMENTED defaults (no daemon analogue on the credential
  trait; covered on the messaging trait instead).
- **Divergences noted:** duplicate adapter family retained (Vec, not rejected); past one-shot
  yields no fire instead of erroring; command registry is first-wins build-once (no removal,
  no priority stacking, no per-conversation filter); credentials have no active-provider
  concept (unknown profile is a clean no-op/fallback, not an error).
