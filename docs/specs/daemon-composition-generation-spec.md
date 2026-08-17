# Composition-generation inheritance — design note (spec only, no code)

Status: DESIGN NOTE (engine-determinism track, Item 5, review addition). No implementation on
this branch. Naming deliberately avoids "fingerprint" (taken by command approvals) and reuses
"digest" only where it means the B1a request digest.

## Problem

A session's behaviour is determined by a *resolved configuration* — prompt composition, tool set,
model routing, policies — that today is re-resolved from mutable profile state at composition
boundaries. Two sessions "on the same profile" can therefore silently diverge when the profile is
edited between their compositions, and a resumed or forked session can wake into a different
configuration than the one its history was produced under. The fix is to make the resolved
configuration a first-class immutable value — a **composition generation** — that sessions hold by
value and inherit.

## The immutable resolved bundle

A generation is created at a composition boundary and never mutated. It contains, resolved and
frozen:

- the composed prompt (the `ComposedPrompt` slots, byte-exact) and the model identity it was
  composed under (`composed_model`);
- the ordered tool set with each tool's schema (order is provider-visible; it is part of the
  bundle, not a set);
- provider/model selection: primary profile, fallback profile, and the routing parameters the
  engine config contributes (retry budget, backoff bounds, watchdog, cache TTL);
- the context strategy (context-engine selection + its budget parameters);
- approval and sandbox policies (approval mode, sensitive-path set, containment roots).

**Credentials are EXCLUDED.** A generation carries credential *references/selectors* (profile
refs, scope templates) only — never lease material, never secrets. Children inherit the pointer,
and each acquisition still goes through the live credential provider with its own leasing,
rotation, and expiry. A generation is therefore safe to persist, log, and transfer.

Each generation has an id (content-addressed over a canonical encoding of the bundle — the B1a
encoding idiom: label-tagged, length-prefixed, order-preserving) plus lineage metadata: the
generation it was derived from and why.

## Inheritance rules

- **Children inherit by value.** A delegated child session receives its parent's generation id and
  resolves nothing afresh.
- **Resumes, retries, and forks retain the generation by value.** Waking a snapshot, retrying a
  model call, and forking at a quiescent boundary (sibling note) all keep the exact generation the
  history was produced under. A mutable profile edit affects only sessions composed *after* it —
  running and suspended sessions are immune.
- **Migration is explicit.** Moving a live session to a new generation is an operator-directed
  operation: it creates (or selects) the new generation, records an audit event
  `(session, from_generation, to_generation, actor, reason, at)`, and recomposes at the next
  boundary. The existing model-switch recompose becomes a special case of migration.
- **Persistence.** The generation id (and the lineage edge, on migration) persists in the session
  snapshot and session metadata, so provenance survives restart and transfer and is queryable
  store-side.

## Tie-in to B1a

The engine-side request-context tuple that B1a holds per call in memory/tracing —
`(profile, model, injection digest)` — gains the generation id as a fourth element once
generations exist. B1b's durable request records then carry it, closing the chain: every recorded
request names the exact resolved configuration that assembled it.

## Known limitations / deferred

- Generation GC (unreferenced generations after migration) is a store concern, deferred.
- Cross-node generation transfer rides the sync protocol; content-addressing makes it idempotent.
- Diffing two generations for operator display ("what changed between these sessions") is UI work
  on top of the canonical encoding, deferred.
