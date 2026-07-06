# HARDENING-PLAN — DEFERRED WIRE BUNDLE (codec-wire-bundle track)

Worktree: `/home/j/experiments/daemon-worktrees/codec-wire-bundle` (branch `hardening/codec-wire-bundle`,
off `hardening/integration`). Repo: **daemon-node only**. This is **PLAN ONLY** — no source touched
yet; STOP after this for review.

## Objective

Batch the four wire-contract additions that Phase 2/3 tracks deliberately deferred so they land as
**ONE `WireVersion` bump + ONE `daemon-api.cddl` update + ONE fixture regen**:

1. `Origin.sender: Option<SenderId>` — carry the immutable sender ID onward for downstream attribution.
2. `ApprovalInfo.fingerprint: Option<String>` — promote the exec-approval fingerprint from the prompt
   string to a structured operator-facing field.
3. `InstalledModel.sha256: Option<String>` — display-only exposure of the node-local pinned sha256.
4. `SkillBundle.signature: Option<String>` + an opt-in, default-off verify-at-import gate.

The coordinator runs the superproject-level `just update-codec` / `just codec-drift` (vendored C codec
regen in daemon-app) **after** this merges — **NOT my job**. My scope stops at the Rust types, the
CDDL, the ciborium fixtures, and the daemon-api conformance proptest.

---

## Anchors located (current state)

| Thing | Location |
|---|---|
| `WireVersion(pub u16)` + `CURRENT` | `crates/contracts/daemon-common/src/lib.rs:381`, `CURRENT = Self(27)` at `:560` |
| Version-history docstring | `crates/contracts/daemon-common/src/lib.rs:383–559` (append a `v28` block) |
| `API_WIRE_VERSION` / `WIRE_VERSION` re-export | `crates/contracts/daemon-api/src/lib.rs:84` (`WireVersion::CURRENT` — no literal to edit) |
| CDDL | `crates/contracts/daemon-api/daemon-api.cddl` — `wire_version` comment `current = 27` at `:22` |
| `Origin { transport, scope }` | `crates/contracts/daemon-protocol/src/lib.rs:897`; ctors `new` `:906`, `internal` `:914` |
| `SenderId(pub String)` | `crates/contracts/daemon-protocol/src/lib.rs:828` (already exists, has `new`/`as_str`/`local_loopback`) |
| `session_id_for` | `crates/contracts/daemon-protocol/src/lib.rs:1060` — **reads only `transport` + `scope`** |
| `Reception { origin, sender, input, addressed }` | `crates/adapters/daemon-ingest/src/lib.rs:139` (already carries required `sender: SenderId`) |
| `Ingestor::receive` | `crates/adapters/daemon-ingest/src/lib.rs:220` (submit at `:239` via `submit_routed(r.origin.clone(), …)`) |
| `ApprovalInfo` | `crates/contracts/daemon-api/src/lib.rs:1944` |
| `ApprovalInfo` construction | `crates/substrate/daemon-host/src/node_api/control.rs:302` (`approvals_pending`) |
| `PendingApproval.fingerprint: Option<CommandFingerprint>` | `crates/engine/daemon-core/src/snapshot.rs:108` (already exists; enforcement at `engine.rs:1517–1529`) |
| `CommandFingerprint(String)` + `.as_str()`/`.short()` | `crates/engine/daemon-core/src/exec/mod.rs:202`, `:249`, `:254` |
| `InstalledModel` | `crates/contracts/daemon-common/src/lib.rs:1262` |
| `build_record` (wire `InstalledModel` construction) | `crates/providers/daemon-models/src/manager.rs:578` |
| `read_pin(<artifact>.sha256)` (node-local pin sidecar) | `crates/providers/daemon-models/src/manager.rs:642` |
| catalog listing (`ModelCatalog` source) | `crates/providers/daemon-models/src/registry.rs:76` `list()` (stored records) |
| `SkillBundle` | `crates/contracts/daemon-common/src/lib.rs:1414` (derives `Default`) |
| `SkillStore` + builders | `crates/skills/daemon-skills/src/lib.rs:237` (`new` `:255`, `with_revisions` `:266`, `with_usage` `:273`) |
| `SkillStore::import_bundle` (choke point) | `crates/skills/daemon-skills/src/lib.rs:549` |
| `SkillsProvider` / `for_profile` | `crates/skills/daemon-skills/src/lib.rs:837`, `:894` |
| `SkillsProvider` node construction | `bins/daemon/src/main.rs:2259–2289` |
| ed25519 pattern to reuse | `crates/substrate/daemon-credentials/src/capability.rs:14–120` (`bc_components` + `bc_envelope::prelude`) |
| `bc-components` / `bc-envelope` workspace deps (vetted) | `Cargo.toml:226,244` |
| CDDL rules to edit | `origin` `:456`, `approval-info` `:339`, `installed-model` `:609`, `skill-bundle` `:793` |
| Newtype→CDDL precedent | `transport-id = tstr` `:446` (so `sender-id = tstr`) |
| Fixture regen | `cargo run -p xtask -- api-fixtures` → `crates/contracts/daemon-api/fixtures/cbor` (`xtask/src/main.rs:247`) |
| CDDL conformance | `crates/contracts/daemon-api/tests/conformance_proptest.rs` (cddl-cat over arbitrary `ApiRequest`/`ApiResponse`) + `conformance.rs` |

---

## The single WireVersion bump

- `crates/contracts/daemon-common/src/lib.rs:560`: `pub const CURRENT: Self = Self(27);` → **`Self(28)`**.
- Append a `v28 (deferred-wire bundle)` block to the docstring (`:383–559`) documenting all four
  additive fields together. Note: `is_compatible` is strict-equal, so even purely-additive fields
  require the bump (matches the v15–v23 additive precedent).
- `crates/contracts/daemon-api/daemon-api.cddl:22`: update the inline comment
  `; daemon-common WireVersion (current = 27)` → `current = 28`.
- No literal edit in `crates/contracts/daemon-api/src/lib.rs` — it reads `WireVersion::CURRENT`.

All four wire fields are **optional + `#[serde(default)]`** and serialize as CBOR `null` when absent
(the same style as existing `mmproj_path`/`context_length` — CDDL marks them `?` with `/ null`). No
`skip_serializing_if`, to stay consistent with the sibling optional fields already in these structs.

---

## Addition 1 — `Origin.sender: Option<SenderId>`

**Rust** (`crates/contracts/daemon-protocol/src/lib.rs`)
- Add to `Origin` (`:897`): `#[serde(default)] pub sender: Option<SenderId>` (doc: *immutable
  platform sender carried for downstream attribution; NEVER a routing/derivation input*).
- `Origin::new` (`:906`) and `Origin::internal` (`:914`): set `sender: None`.
- Every direct `Origin { … }` struct literal must gain `sender: None`:
  - `crates/substrate/daemon-host/src/node_api/internals.rs:10`, `:20`
  - `crates/engine/daemon-core/src/events.rs:44`
  - `crates/engine/daemon-core/src/actor.rs:37`, `:396`
  - `bindings/daemon-core-ffi/src/lib.rs:56`, `:64`
  - Test/helper literals: `tests/daemon-conformance/src/node/ownership_matrix.rs:409`,
    `crates/substrate/daemon-host/src/routing.rs:358`, `crates/adapters/daemon-rooms/src/inbound.rs:109`,
    `crates/adapters/daemon-ingest/tests/ingest.rs:86` — inspect each; add `sender: None` if a literal,
    unaffected if it goes through `Origin::new`.

**Consumer wiring** (`crates/adapters/daemon-ingest/src/lib.rs`, `Ingestor::receive` `:220`)
- After the sender gate passes and before submit (`:239`), stamp the immutable sender onto the origin
  handed to the host:
  ```rust
  let mut origin = r.origin.clone();
  origin.sender = Some(r.sender.clone());
  // … submit_routed(origin.clone(), command) …
  ```
  This carries `SenderId` (already enforced at ingest) ONWARD onto `Origin` for downstream attribution
  (log/journal/routing all embed the same `origin` rule and inherit the field automatically).

**Invariant to verify (group sessions stay shared)**
- `session_id_for` (`:1060`) reads only `origin.transport` + `origin.scope` — a new top-level
  `Origin.sender` cannot perturb derivation. Add a test asserting two receptions with the **same group
  origin but different senders** derive the **same** `SessionId` under `IsolationPolicy::Shared`.

**CDDL** (`daemon-api.cddl`)
- Add rule near `transport-id`: `sender-id = tstr` (newtype `SenderId(String)` → transparent `tstr`).
- `origin` rule (`:456`): add `? "sender": (sender-id / null),`.

**Tests FIRST**
- `origin_roundtrips_sender_through_cddl` (daemon-api): encode an `Origin`/`ApiRequest` with a populated
  `sender`, CBOR round-trip + cddl-cat validate (proptest already exercises arbitrary values; add a
  targeted positive fixture-style case).
- `receive_stamps_immutable_sender_onto_origin` (daemon-ingest): mock `NodeApi` capturing the `origin`
  passed to `submit_routed`; assert `origin.sender == Some(r.sender)`. Pre-change: `sender == None`.
- `group_session_shared_across_senders` (daemon-protocol or ingest): differing sender ⇒ identical
  session id (the deferral’s stated safety invariant).

---

## Addition 2 — `ApprovalInfo.fingerprint: Option<String>`

**Rust** (`crates/contracts/daemon-api/src/lib.rs`)
- Add to `ApprovalInfo` (`:1944`): `#[serde(default)] pub fingerprint: Option<String>` (doc:
  *lowercase-hex sha256 of the resolved command tuple the operator approved; display/correlation only —
  enforcement stays snapshot-side*).

**Consumer wiring** (`crates/substrate/daemon-host/src/node_api/control.rs:302`, `approvals_pending`)
- Populate from the already-existing `PendingApproval.fingerprint`:
  ```rust
  fingerprint: p.fingerprint.as_ref().map(|f| f.as_str().to_string()),
  ```
  `CommandFingerprint` lives in daemon-core; map to `String` here (daemon-host depends on both), so
  daemon-api stays free of a daemon-core dep.
- **Enforcement unchanged**: `engine.rs:1517–1529` still gates the re-run on
  `PendingApproval.fingerprint`. The `honest_prompt` string (shell tool) is left as-is (harmless
  redundant display; minimal hunks).

**CDDL**: `approval-info` rule (`:339`): add `? "fingerprint": (tstr / null),`.

**Tests FIRST**
- `approval_info_roundtrips_fingerprint_through_cddl` (daemon-api): proptest + targeted case.
- `approvals_pending_surfaces_fingerprint_structurally` (daemon-host): park an approval whose
  `PendingApproval.fingerprint = Some(..)`; assert the returned `ApprovalInfo.fingerprint == Some(..)`
  (structural, not buried in `prompt`). Pre-change: field absent.

---

## Addition 3 — `InstalledModel.sha256: Option<String>`

**Rust** (`crates/contracts/daemon-common/src/lib.rs`)
- Add to `InstalledModel` (`:1262`): `#[serde(default)] pub sha256: Option<String>` (doc:
  *node-local pinned sha256 for provenance display; node-side pin/verify in daemon-models stays
  authoritative — this is display-only exposure*).

**Consumer wiring** (`crates/providers/daemon-models/src/manager.rs`, `build_record` `:578`)
- Read the node-local pin sidecar before the struct literal (`local_path` is moved into the struct):
  ```rust
  let sha256 = read_pin(&local_path);   // read_pin is module-local at :642
  let mut record = InstalledModel { …, sha256, model };
  ```
  `build_record` is called by `catalog_artifact` after `write_pin`, so a fresh catalog persists the
  sha256; `Registry::list()` (the `ModelCatalog` source) then surfaces it.
- **Legacy note (accepted, display-only/best-effort):** records cataloged before this field show
  `sha256: null` in `catalog.json` until re-cataloged. Sidecar-backfill-on-list is an explicit
  **non-goal** (would touch the registry crate + add per-list fs reads). Flag for review; recommend
  keeping it out of scope.
- Other `InstalledModel { … }` literals gain `sha256: None`:
  `crates/providers/daemon-models/src/registry.rs:160` (test),
  `crates/providers/daemon-models/src/quantize.rs:196`,
  `crates/providers/daemon-models/src/manager.rs:781` (test), `:861` (test),
  `crates/providers/daemon-models/tests/acquire_mock.rs:352`,
  `tests/daemon-conformance/src/node/provider_discovery.rs:222`,
  `xtask/src/main.rs:824` (fixture — set `None`).

**CDDL**: `installed-model` rule (`:609`): add `? "sha256": (tstr / null),`.

**Tests FIRST**
- `installed_model_roundtrips_sha256_through_cddl` (daemon-api): proptest + targeted case.
- `catalog_record_exposes_pin_sha256` (daemon-models `manager.rs`): seed a cataloged artifact WITH a
  `<path>.sha256` sidecar (the existing `seed_pinned` helper) ⇒ built record’s `sha256 == pin`; a record
  with no sidecar ⇒ `None`.

---

## Addition 4 — `SkillBundle.signature: Option<String>` + verify-at-import gate

**Rust — wire field** (`crates/contracts/daemon-common/src/lib.rs`)
- Add to `SkillBundle` (`:1414`): `#[serde(default)] pub signature: Option<String>` — **hex-encoded**
  ed25519 detached signature over the canonical bundle digest. String (not raw bytes / bc types) keeps
  daemon-common dependency-free and CDDL simple. `SkillBundle` already derives `Default` ⇒ new field
  defaults `None`. Export sites (`export_bundle` `:526`, the `SkillBundle { … }` literals at `:689`, and
  the daemon-host distribution literals) set `signature: None` (unsigned by default).

**Rust — signing/verify primitive + gate** (`crates/skills/daemon-skills/`)
- `Cargo.toml`: add `bc-components = { workspace = true }` and `bc-envelope = { workspace = true }`
  (both already in-tree via daemon-credentials/daemon-telemetry ⇒ **no new crate in the graph ⇒ no
  `cargo deny` impact**). Reuse the exact stack in `daemon-credentials/src/capability.rs`.
- New `skill_bundle_digest(bundle: &SkillBundle) -> [u8; 32]`: SHA-256 over a length-prefixed,
  domain-separated feed of `(name, category, sorted files as (path,content))`, **excluding** the
  `signature` field. Mirror `capability_digest` / `CommandFingerprint::feed` (unambiguous field
  boundaries so no two bundles collide).
- New `SkillBundleVerifier(SigningPublicKey)` with
  `verify(&self, bundle) -> Result<(), SkillError>`: require `bundle.signature`, hex-decode → CBOR →
  `bc_components::Signature`, verify against `skill_bundle_digest`. Distinct error variant
  `SkillError::Signature(String)`.
- `SkillStore`: add `verifier: Option<Arc<SkillBundleVerifier>>` (default `None`) + builder
  `with_import_verification(verifier)`.
- `import_bundle` (`:549`) gate — **top of the fn, before any fs write** (the choke point):
  ```rust
  if let Some(v) = &self.verifier {
      v.verify(bundle)?;   // require+valid signature; else Err, nothing written
  }
  ```
  When `verifier` is `None` (default) behavior is unchanged (unsigned imports pass, as today).
- `SkillsProvider` (`:837`): carry `Option<Arc<SkillBundleVerifier>>`, propagate to each `for_profile`
  store via `with_import_verification`. Builder `with_import_verification` on the provider; default none.
- `bins/daemon/src/main.rs:2259–2289`: read an **optional** configured verify key (default absent →
  off) and attach it to the provider. **Open decision for review:** wire a minimal config read now
  (feature actually usable, still default-off) vs. keep key injection test-only this pass. Recommend
  the minimal config read so the gate is reachable.

**CDDL**: `skill-bundle` rule (`:793`): add `? "signature": (tstr / null)` →
`skill-bundle = { "name": tstr, "category": (tstr / null), "files": { * tstr => tstr }, ? "signature": (tstr / null) }`.
Covers every use (`response-skill-bundle` `:1386`, profile distribution payloads).

**Tests FIRST**
- `skill_bundle_roundtrips_signature_through_cddl` (daemon-api): proptest + targeted case.
- `import_refuses_signature_mismatch` (daemon-skills): store `.with_import_verification(v)`; bundle
  signed then a file mutated (or a forged signature) ⇒ `Err(SkillError::Signature)`, and the target dir
  is **not** written.
- `import_refuses_absent_signature_when_required` (daemon-skills): verifier set, `signature: None` ⇒
  refused.
- `import_accepts_valid_signature` (daemon-skills): sign with the matching key ⇒ import succeeds; on-disk
  bundle present.
- `import_unsigned_default_off_regression` (daemon-skills): no verifier ⇒ unsigned bundle imports as
  today (no regression).

---

## `cargo deny` impact

**None.** `bc-components`/`bc-envelope` are already in the dependency graph (daemon-credentials,
daemon-telemetry) and in `Cargo.lock`; adding them to `daemon-skills` introduces no new crate,
advisory, license, or source. All four wire fields are plain types (`Option<String>`,
`Option<SenderId>`) — no new deps in daemon-common/daemon-api/daemon-protocol.

## Clippy disallow-lints (Phase 4 active)

- `read_pin` (#3) uses `std::fs` inside `daemon-models` — that crate/module already carries
  `#![allow(clippy::disallowed_methods)]` (`registry.rs:6`; `read_pin` at `manager.rs:642` already
  compiles on this branch), so reusing it triggers nothing new.
- `daemon-skills/src/lib.rs:26` already has `#![allow(clippy::disallowed_methods)]`; the signing code
  adds **no** new `fs`/`reqwest`/`Command`. Origin stamping and the fingerprint map are pure. No new
  disallowed-method anchors expected; if clippy flags anything, route through the sanctioned API or add
  a commented anchor.

---

## Cross-track deconfliction flags (for coordinator)

Sibling tracks run concurrently: **authz-f3f4** (node_api fleet/unit + EventsSince/DeliverySessions
ownership gates) and **env-ban-migration** (spawn-site EnvPolicy + clippy env lint). Neither touches
wire types or CDDL. My node_api handler touches:

- **`crates/substrate/daemon-host/src/node_api/control.rs`** — I edit `approvals_pending` (`:302`) to
  populate `ApprovalInfo.fingerprint` (#2). `control.rs` is **not** in authz-f3f4’s stated scope
  (fleet/unit + EventsSince/DeliverySessions), but flagging it explicitly for rebase awareness.
- `Origin.sender` stamping (#1) is in **daemon-ingest** (`Ingestor::receive`), **not** a node_api
  handler. `submit_routed`/`SubmitRouted` merely carry the `origin` and inherit the field — no handler
  edit.
- The skill verify gate (#4) lives inside `import_bundle` (**daemon-skills**), not the
  `node_api/profile.rs` handlers that call it — no handler-body edit needed there; the key attaches at
  `SkillsProvider` construction (`bins/daemon/src/main.rs`).

No overlap with env-ban-migration.

---

## The exact gate (from worktree root; paste tails)

```bash
nix develop --command cargo fmt --all -- --check
nix develop --command cargo clippy --workspace --all-targets -- -D warnings
nix develop --command cargo test --workspace --no-fail-fast
nix develop --command cargo test -p daemon-api --features arbitrary      # CDDL <-> Rust MUST agree
nix develop --command cargo run -p xtask -- api-fixtures                 # regen ciborium fixtures
```

- The `-p daemon-api --features arbitrary` run (conformance_proptest.rs, cddl-cat) is the lockstep gate:
  any Origin/ApprovalInfo/InstalledModel/SkillBundle field vs. CDDL drift fails here.
- Fixture regen is deterministic from the Rust types; new optional fields emit as `null`. Commit the
  regenerated `crates/contracts/daemon-api/fixtures/cbor/*` alongside the CDDL edit.

**Machine-load note:** if `bins/daemon/tests/host_launch.rs` fails under the concurrent Opus builds,
re-run isolated: `nix develop --command cargo test -p daemon --test host_launch -- --test-threads=2`.
KNOWN FLAKES (do not chase): `detached_delegation` ×2, `process_notify` store-seam.

---

## COORDINATOR post-merge (NOT this track)

After this branch merges to `hardening/integration`, the coordinator runs at the **superproject** level:

- `just update-codec` — regenerate the vendored C codec in daemon-app from the bumped CDDL.
- `just codec-drift` — gate the vendored copy against the pinned contract.

**I do NOT** run `just update-codec`/`just codec-drift`, do NOT touch daemon-app’s vendored codec, do
NOT merge, and do NOT remove the worktree.

---

## Residual coverage / caveats

- **#3 legacy records**: pre-existing catalog entries show `sha256: null` until re-cataloged (sidecar
  backfill-on-list deliberately out of scope). Flagged above.
- **#4 default-off**: with no verifier configured, unsigned bundles import exactly as today — the gate
  is inert until an operator supplies a verify key. Signing key *distribution/management* is out of
  scope (only the verify primitive + gate ship here).
- **#2**: the fingerprint also remains embedded in the `prompt` string (unchanged) — structural field
  is additive, not a replacement, so no display regression for older clients.
- All four fields are additive/optional; the only breaking aspect is the strict-equal `WireVersion`
  bump, which the single v28 handles.
