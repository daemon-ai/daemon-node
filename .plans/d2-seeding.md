# D2 — First-boot Daemon Cloud credential + default-profile seeding

**Branch/worktree:** `feat/cloud-credential-seed` @ `/home/j/experiments/daemon-worktrees/daemon-node-cloud-seed`
**Status:** PLAN (no source touched yet — awaiting approval)
**Scope:** the only remaining daemon-side blocker for the hosted-node attach flow
(daemon-api `docs/hosted-nodes.md` §18 row **D2**; boot step 2 of §8.3).

---

## 0. TL;DR

On boot, when a Daemon Cloud attach key is present in the environment, idempotently
seed (a) the **credential-store entry** for the node's default profile and (b) ensure
the **default profile selects the `daemon_api` ("Daemon Cloud") provider** at the
gateway base — so a freshly-provisioned hosted node routes inference through the
metered gateway with no GUI setup. This mirrors the already-shipped D1 first-admin
seeding pattern (`DAEMON_ADMIN_USERNAME` / `DAEMON_ADMIN_PASSWORD[_FILE]`), reusing the
existing `CredentialStore` (create-or-update `set`) and `ProfileStore::seed`
(first-boot-only) machinery already wired into `run_as_host`.

The code change is small and localized to `bins/daemon/src/main.rs`; the store/profile
primitives it needs already exist and are already invoked at boot.

---

## 1. Proposed env var name(s) + rationale

### Decision: `DAEMON_CLOUD_API_KEY` and `DAEMON_CLOUD_API_KEY_FILE`

| Var | Meaning |
|---|---|
| `DAEMON_CLOUD_API_KEY` | The Daemon Cloud attach key (secret). Read directly at startup. |
| `DAEMON_CLOUD_API_KEY_FILE` | Path to a file whose (trimmed) contents are the attach key. `_FILE` wins only when the direct var is unset — exactly D1's precedence. |

Companion vars are **reused, not re-invented** (they already exist as `NodeConfig`
fields and are injected per §8.2):

- `DAEMON_PROFILE` → `cfg.profile` — the profile id the credential + default profile
  are keyed under (default `"default"`; hosted image sets `"hosted"`).
- `DAEMON_BASE_URL` → `cfg.base_url` — the gateway base; resolves through the existing
  `NodeConfig::daemon_api_base()` (default `https://api.daemon.ai/api/v1/`, trailing
  slash load-bearing) when unset.

### Why not the spec's provisional `DAEMON_BOOTSTRAP__DAEMON_API_KEY`

1. **`__` is figment's config-nesting delimiter.** `NodeConfig` is loaded via
   `Env::prefixed("DAEMON_").split("__")` (`config.rs` docstring + `base_figment`).
   `DAEMON_BOOTSTRAP__DAEMON_API_KEY` reads as the config path `bootstrap.daemon_api_key`
   — a struct that does not exist on `NodeConfig`. Figment silently ignores unknown keys
   (deliberately — see the module doc), so the name *looks* like config but is invisible
   to config. That is a trap for anyone grepping `config.rs`.
2. **D1 set the house convention: flat, direct-read secret vars.**
   `DAEMON_ADMIN_USERNAME` / `DAEMON_ADMIN_PASSWORD[_FILE]` are read *directly* in
   `main.rs` (`ADMIN_USERNAME_ENV` etc., lines ~1614–1650), **not** through `NodeConfig`,
   precisely because process-env + filesystem side effects belong in the binary. A flat
   `DAEMON_CLOUD_API_KEY[_FILE]` mirrors that 1:1 (same shape, same `_FILE` sibling, same
   read path), so the two seeding paths are visibly consistent.
3. **Self-documenting product noun.** "Daemon Cloud" is the product name for the
   `daemon_api` provider selector (spec §1 glossary, §2.1.5). `DAEMON_CLOUD_API_KEY` says
   what it is; `DAEMON_BOOTSTRAP__DAEMON_API_KEY` conflates the bootstrap mechanism with
   the provider id.

**Coordinator action:** the daemon-api spec (§8.2 table row, §18 D2, Appendix B) should
be updated to `DAEMON_CLOUD_API_KEY[_FILE]`. This repo owns the name (spec explicitly
defers it); the image launcher passes env through unchanged either way
(`daemon/docs/hosted-node-image.md` Q1).

### Why a dedicated var and not the existing `DAEMON_CREDENTIAL_KEY`

`cfg.credential_key` (`DAEMON_CREDENTIAL_KEY`) already seeds `credential_store.set(&cfg.profile, …)`
at boot (`main.rs` ~1797), **but** it doubles as the *global broker fallback*
(`build_multi_profile_broker(&cfg.credential_key, …)`, ~1807) — i.e. it is handed to
*every* profile's credential source as the last-resort key. Injecting the attach key
there would leak it as a fallback to BYOK profiles. A dedicated D2 var seeds **only the
specific profile's store entry**, keeping the attach key isolated to the Daemon Cloud
profile. (It also lacks a `_FILE` variant and Daemon-Cloud semantics.) Recommendation:
hosted provisioning sets `DAEMON_CLOUD_API_KEY` and leaves `DAEMON_CREDENTIAL_KEY` unset.

---

## 2. Exact hook locations

All in **`bins/daemon/src/main.rs`**, `async fn run_as_host(cfg: NodeConfig)`.

### 2.1 New env-resolution helper (mirrors `resolve_admin_seed`, ~line 1620)

```rust
const CLOUD_API_KEY_ENV: &str = "DAEMON_CLOUD_API_KEY";
const CLOUD_API_KEY_FILE_ENV: &str = "DAEMON_CLOUD_API_KEY_FILE";

/// Env-first: `DAEMON_CLOUD_API_KEY`, else `DAEMON_CLOUD_API_KEY_FILE` (file contents),
/// trimmed. `Ok(None)` when neither source is set (no seeding). A source that is set but
/// blank/whitespace is a misconfiguration -> `Err` (never seed an empty attach key).
fn resolve_cloud_api_key() -> anyhow::Result<Option<String>> { … }
```

Semantics identical to `resolve_admin_seed` (env wins over `_FILE`; explicit-but-blank
is refused), except that *absence* yields `None` (no admin analogue because admin has a
generate fallback; D2 has none — a hosted node with no attach key just boots without a
cloud credential).

### 2.2 Credential seeding — fold into the existing credential-store block (~lines 1795–1801)

Today:
```rust
if !cfg.credential_key.is_empty() {
    credential_store.set(&cfg.profile, &cfg.credential_key)?;   // existing
}
```
Add, right after (D2):
```rust
let cloud_key = resolve_cloud_api_key()?;                       // NEW
if let Some(key) = &cloud_key {
    // create-or-update: rotation + restart re-seeds; never duplicates.
    credential_store.set(&cfg.profile, key)
        .map_err(|e| anyhow::anyhow!("seeding Daemon Cloud credential: {e}"))?;
    tracing::info!(profile = %cfg.profile, "seeded Daemon Cloud credential from environment");
    // NB: never log `key`.
}
```
`CredentialStore::set` (`crates/substrate/daemon-host/src/credstore.rs`) is already an
idempotent create-or-update for both backends (`FileCredentialStore` = 0600 JSON map
insert/replace; `MemCredentialStore` = replace-pool). No new store method is required.

### 2.3 Default-profile provider selection (~line 2006 + `default_profile_spec`, line 650)

The default profile is already seeded first-boot-only:
```rust
profile_store.seed(default_profile_spec(&cfg, provider_kind))?;   // existing, ~2007
```
`default_profile_spec` (line 650) already maps `provider_kind == None` **and**
`Some(DaemonApi)` to `ProviderSelector::DaemonApi` (the "Daemon Cloud" selector), with
`base_url: cfg.base_url.clone()` (resolves to the gateway default when unset). Because
the hosted image injects `DAEMON_PROFILE`/`DAEMON_BASE_URL` + the secret but **not**
`DAEMON_MODEL_PROVIDER` (§8.2), a hosted first boot already lands on a DaemonApi default
profile.

To make D2 self-contained (spec: "seed … a default profile that selects that
provider") and independent of whether some operator set a conflicting
`DAEMON_MODEL_PROVIDER`, thread the cloud-key presence into the seed:

- Change the call to `default_profile_spec(&cfg, provider_kind, cloud_key.is_some())`.
- In `default_profile_spec`, when the `cloud_seed` flag is `true`, force
  `provider = ProviderSelector::DaemonApi` (overriding only the provider selector; model
  and everything else unchanged). This is a 1-line guard at the top of the existing
  `match`.

Idempotency is preserved because `ProfileStore::seed` is first-boot-only
(`crates/substrate/daemon-host/src/profiles.rs` — inserts only when the store is empty,
never resurrects a deleted placeholder). Rotation reboots never rewrite the profile;
they only re-`set` the credential (§2.2).

### 2.4 (Optional, symmetry) transport-server path (~line 2427)

`run_as_transport_server` has the same `credential_key` seed block. A hosted node runs
the **host** role, not the transport role, so D2 does **not** need to touch it. Leave it
out to keep blast radius minimal (note only).

---

## 3. Idempotency semantics (the load-bearing decisions)

| Boot scenario | Behavior | Rationale |
|---|---|---|
| **Same secret, restart** | `set` rewrites the identical value (effective no-op); profile untouched (first-boot-only seed). Stable across reboots. | Matches §8.3 "a second boot reuses the volume (no re-seed)" for the profile; the credential converges to the same value. |
| **Rotated secret, restart** | `set` overwrites the stored credential with the new key. Profile untouched. | This *is* the attach-key rotation path (§9.4: control plane updates the Fly secret then restarts). Create-or-update, never duplicate — exactly the spec's D2 wording. |
| **Secret removed (env unset), restart** | **No-op**: `resolve_cloud_api_key()` returns `None`, the stored credential is left in place (not scrubbed). | Mirrors D1 (unset admin env ⇒ no re-seed, never a destructive sync). Provisioning per §8.2 always sets the secret, so "unset" is not a normal hosted state. Deleting a credential is an explicit GUI/API `CredentialRemove` or a volume wipe — never an implicit consequence of an env change. |
| **First boot, no secret** | Boots with a DaemonApi default profile but no cloud credential; a turn fails clearly at turn time (never a silent success) — identical to today's `daemon_cloud_turn_without_credential_errors` conformance behavior. | Keyless boot is already a supported, tested state. |

**Env-authoritative on the managed attach credential:** while `DAEMON_CLOUD_API_KEY`
stays set, every boot re-asserts it onto `cfg.profile`. A customer's GUI-set key on the
*same* profile is therefore overwritten on the next restart. This is intended — the
attach key is provisioner-managed (rotated through the control plane, not the node GUI);
BYOK belongs on a *different* profile. Flagged as a coordinator confirmation (§7).

**Never leaks the secret:** unlike D1's generated-password path (which prints once to
stderr + a `0600` file), the attach key is always supplied by the provisioner, so D2
generates nothing and prints nothing — it logs only the profile id + "seeded", never the
value. No new secret-to-stderr surface (avoids the §8.2 D1 caveat about provider logs).

---

## 4. How the default profile ends up selecting `daemon_api`

- `ProviderSelector::DaemonApi` is the wire selector for "Daemon Cloud" (confirmed:
  `daemon_cloud_e2e.rs` maps `ProviderKindWire::DaemonCloud` ⇒ `ProviderSelector::DaemonApi`;
  the binary's `provider_builder_for` pins genai's OpenAI adapter at `daemon_api_base()`).
- The credential store is keyed by **profile id** with `credential_ref: None`
  (`default_profile_spec` sets `credential_ref: None`; `daemon_cloud_e2e.rs` proves
  `CredentialSet{ profile:"gateway" }` reaches a profile whose id is `"gateway"`). So
  `credential_store.set(&cfg.profile, key)` + a default profile whose id is `cfg.profile`
  = the credential resolves for that profile's sessions.
- Base URL: `default_profile_spec` stores `cfg.base_url` (often `None`); the DaemonApi
  builder resolves the gateway default via `daemon_api_base()` (trailing-slash
  normalized). If `DAEMON_BASE_URL` is injected it is honored. No extra work.

Net: after first boot with `DAEMON_CLOUD_API_KEY` (+ optional `DAEMON_BASE_URL`,
`DAEMON_PROFILE`), the node has: a DaemonApi default profile at the gateway base + a
stored attach credential for it. The one remaining input for a *turn* is a model name
(see §7 Q1).

---

## 5. Test plan

### 5.1 Unit — `bins/daemon/src/main.rs` `#[cfg(test)]` (mirror the `resolve_admin_seed` tests ~2660+)

- `resolve_cloud_api_key` → `None` when neither var set.
- reads `DAEMON_CLOUD_API_KEY`.
- reads `DAEMON_CLOUD_API_KEY_FILE` (temp file; trims trailing newline).
- env var wins over `_FILE`.
- blank/whitespace key when a source is set → `Err`.
- Follow the exact env-mutation idiom the existing admin tests use (same
  set_var/remove_var discipline / serialization) to avoid cross-test env races.

### 5.2 Unit — credential create-or-update (`crates/substrate/daemon-host/src/credstore.rs`)

`credstore.rs` already proves `set` replaces (`mem_store_set_get_remove_redacts`,
`mem_store_multi_key_pool`). Add one focused test: `set(p, "A")` then `set(p, "B")` ⇒
`get(p) == "B"` and `list_redacted().len() == 1` (no duplicate) — the D2 idempotency
invariant at the store layer.

### 5.3 Binary integration — `bins/daemon/tests/host_launch.rs` (mirror the persistence gates)

Using the existing `spawn_host_launch` + post-boot file inspection idiom, with
`DAEMON_STORE=sqlite` + a throwaway `DAEMON_DATA_DIR` (so `credentials.json` /
`profiles/` persist):

1. **seed happy path**: boot with `DAEMON_CLOUD_API_KEY=sk-test`, `DAEMON_PROFILE=hosted`
   → after boot assert `credentials.json` maps `"hosted" → "sk-test"` and the seeded
   profile's provider is `daemon_api`.
2. **rotation**: boot with key A, shut down, boot with key B over the same data dir →
   `credentials.json` now has B (create-or-update, single entry).
3. **`_FILE` variant**: boot with `DAEMON_CLOUD_API_KEY_FILE=<tmp>` → key seeded.
4. **unset-after-set**: boot with key, shut down, boot again with the var removed →
   credential still present (no scrub).
5. **keyless boot** already covered by `unconfigured_provider_host_launch_boots_and_serves`.

### 5.4 Conformance / integration — `tests/daemon-conformance/src/node/`

Primary end-to-end gate reuses the existing Daemon Cloud harness
(`daemon_cloud_e2e.rs`: mock gateway serving keyless `/models` + bearer-authed
`/chat/completions`, `drain_until_terminal`). Add a **zero-GUI** variant
(`daemon_cloud_seed.rs`, or a case in `daemon_cloud_e2e.rs`): construct a node whose
credential store + default profile are **pre-seeded exactly as D2 seeds them** (same
`credential_store.set` + DaemonApi default profile, no `CredentialSet`/`ProfileCreate`
API calls), drive a turn, and assert the injected bearer reaches the upstream — proving a
provisioned node runs inference with no GUI interaction. The negative
(`daemon_cloud_turn_without_credential_errors`) already covers the keyless-fails-clearly
case.

> Harness note: the conformance `assemble_*` helpers build the node in-process (they do
> not exercise the process-env → seed mapping), so the *env* mapping is proven by §5.3
> (binary) and the *turn* behavior by §5.4 (harness). If cheap, expose the §2.2/§2.3
> seeding as a small pure helper the harness can call, so both layers exercise the same
> code path.

### 5.5 Gates to run before finishing (per AGENTS.md, via `nix develop --command`)

- `cargo fmt` (leave `--check` clean)
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test -p daemon` (host_launch + main unit tests) and
  `cargo test -p daemon-host` (credstore) — targeted first per the no-full-build guidance
- `cargo test -p daemon-conformance` for the new e2e (targeted)
- `cargo deny check`
- **No CDDL impact expected** (no new wire type: `ProfileSpec`/`ProviderSelector`/
  `CredentialSet` are unchanged). If the seeding helper is exposed but no wire type
  changes, the CDDL gates stay green — will still run `cargo test -p daemon-api --features arbitrary`
  to confirm.
- **No config-reference impact**: the new vars are read directly (like `DAEMON_ADMIN_*`),
  not `NodeConfig` fields, so `docs/config-reference.md` and its `check-config-docs` gate
  are unaffected.

---

## 6. Risks

1. **Overwriting a GUI-set key each boot** (env-authoritative). Intended for the managed
   attach key, but surprising if a customer expects to BYOK on the hosted profile.
   Mitigation: BYOK on a separate profile; documented. → Coordinator Q4.
2. **Model still required for a turn.** D2 wires provider + credential + base, but a turn
   needs a model. With only §8.2's injected vars the default profile has an empty model
   (⇒ `UnconfiguredProvider`, clean "pick a model" failure, never silent). "Zero-GUI
   *inference*" needs a default model too. → Coordinator Q1 (should §8.2 add `DAEMON_MODEL`?).
3. **Plaintext credential at rest** in `credentials.json` (0600) on the volume — same
   posture as the existing credential store (§13.1 notes Fly volumes are encrypted at
   rest). No regression; note only.
4. **Double-seed with `DAEMON_CREDENTIAL_KEY`.** If both are set they both `set` the same
   profile key (last wins). Recommend hosted nodes use only `DAEMON_CLOUD_API_KEY`.
5. **Deleted-placeholder edge**: if the GUI replaced/deleted the default profile,
   `ProfileStore::seed` won't resurrect it, but the credential `set` still writes
   `cfg.profile`'s entry (harmless orphan). Low risk; documented.
6. **`_FILE` trailing newline / whitespace** → trim (mirror admin), else the bearer
   carries a stray `\n`.
7. **base_url trailing slash** is load-bearing for genai's `Url::join` — already handled
   by `daemon_api_base()`/`ensure_trailing_slash`; don't bypass it.

---

## 7. Questions for the coordinator

1. **Default model.** Should hosted provisioning also inject `DAEMON_MODEL` (and/or
   `DAEMON_MODEL_PROVIDER=daemon_api`) so a node runs inference with *truly* zero GUI
   setup, or is "provider + credential + base wired, model pending selection" the intended
   first-boot state? (Determines whether §8.2 gains a `DAEMON_MODEL` row.)
2. **Env var name sign-off.** Confirm `DAEMON_CLOUD_API_KEY[_FILE]` and update the
   daemon-api spec (§8.2, §18 D2, Appendix B) away from `DAEMON_BOOTSTRAP__DAEMON_API_KEY`.
3. **Delete-on-unset.** Confirm "seed-if-present, never remove" (mirrors D1) is the
   desired semantics — unsetting the env does not scrub a stored credential.
4. **Env-authoritative vs seed-once** for the attach credential when the env stays set
   across reboots (proposed: env-authoritative; the managed attach key wins over a
   same-profile GUI key). Confirm.

---

## 8. Stretch — D3 `/healthz` readiness endpoint (separate, non-blocking)

**Assessment: cheap and low-risk — recommend doing it only after D2 lands clean.**

The single-origin web front is a hand-rolled HTTP/1.1 server in
`crates/substrate/daemon-host/src/web.rs` (`serve_web` → `handle_conn`; `/ws` is
special-cased for the mux upgrade, everything else falls to `respond_static`, GET/HEAD
only, unknown → 404). A `/healthz` route is a small special-case in `handle_conn`
*before* `respond_static`:

- **Liveness today**: providers health-check `GET /` (returns the GUI HTML, 200) — §8.4,
  verified. D3 adds *deep readiness*.
- **Readiness signal**: report store + journal readiness. The node already holds the
  durable `store` (`SessionStore`) and the journal `signer`; a `/healthz` handler would
  do a cheap store liveness probe (e.g. a bounded read) and return `200
  {"status":"ready"}` vs `503 {"status":"degraded", …}`. Needs a small readiness accessor
  threaded from the assembled node into `serve_web` (today `serve_web` takes the static
  `site` + the mux handover; it would need a `readiness: Arc<dyn Fn -> Health>` or a
  handle to the node).
- **Auth**: keep `/healthz` unauthenticated (like `/`) — it exposes only ready/degraded,
  no secrets — so the fly-proxy check can use it without credentials.
- **Tests**: unit test in `web.rs` (route parses; `/healthz` returns 200/503 by injected
  readiness); the image check in `daemon/docs/hosted-node-image.md` can later upgrade its
  probe from `/` to `/healthz` (no image change needed — same listener, §8.4/D3).
- **Skip condition**: if threading a readiness handle into `serve_web` turns out to be
  invasive (the web front is deliberately decoupled from `NodeApiImpl`), defer — `GET /`
  liveness suffices for v1 (§18 D3 is explicitly non-blocking) and the Fly machine check
  already uses it (§7.3).

D3 is intentionally out of the D2 change set and will be its own commit/plan section.
