# Provider auth — compliance posture and storage threat model (credential plan Phase 4)

Status: decided (Aug 2026). This is the ledger for the credential plan's Phase 4 decisions: what
is deliberately NOT implemented and why, plus the explicit storage threat-model boundary the OAuth
refresh work rides on.

## Anthropic consumer OAuth: NOT implemented (ToS-prohibited)

Validated against Anthropic's official Claude Code
[legal-and-compliance page](https://code.claude.com/docs/en/legal-and-compliance) (Feb 2026
update): consumer OAuth tokens (Free/Pro/Max) are exclusively for Claude.ai and Claude Code —
"Anthropic does not permit third-party developers to offer Claude.ai login or to route requests
through Free, Pro, or Max plan credentials on behalf of their users", enforced without prior
notice. Secondary reporting (not independently verified) attributes the Jan 2026 token blocks on
OpenCode / Roo Code / OpenClaw to this policy. Reusing Claude Code's public client id in daemon
would violate the Consumer ToS.

Compliant Anthropic access in daemon:

- **API key via Claude Console** — works today through the existing key field; no work.
- **Bedrock / Vertex** (Anthropic models under commercial cloud terms) — the Phase-3
  cloud-credentials descriptors (`bedrock_sigv4` SigV4 chain; `vertex` operator-supplied token).

**Parked, compliance-gated:** some partner apps carry an Anthropic-sanctioned "Sign in with
Claude" (extra-usage billing). If Anthropic opens third-party client registration, a properly
registered descriptor is a straightforward `daemon-oauth` addition on the Phase-4 machinery
(PKCE descriptor + token-set envelope + lease-time refresh all exist). Do NOT implement with the
Claude Code client id in the interim.

## GitHub Copilot device flow: shipped, diligence noted

The `github_copilot()` descriptor ships the first-party editor plugins' well-known public
device-flow client id. Same diligence standard as Anthropic: whether GitHub's terms permit
third-party use of that client id — and whether a `read:user`-scoped device token is accepted by
GitHub Models as a bearer — remains research-gated pending the live evidence leg (see the
descriptor's doc comment). If the terms turn out not to permit it, the descriptor becomes
config-gated exactly like Hugging Face (operator-supplied client id, unregistered by default).

## Refresh-token reality check (why the fixture, not a live vendor, is the gate)

No currently supported provider exercises refresh live:

- **OpenRouter** mints a static API key (`JsonPost` key-mint) — nothing to refresh.
- **Hugging Face** was the Phase-4 descriptor fix: its `FormPost` + `ProviderKey` shape could
  never complete (completion demands a JSON key-mint for provider keys). It now mints a
  versioned `OAuthTokenSet` envelope (HF token responses carry `expires_in`, 8h default, and a
  refresh token; `refresh_token` is in HF's advertised `grant_types_supported`) — but the family
  only registers when an operator supplies `oauth.huggingface_client_id`.
- **generic `oauth2`** stores the RAW token-response JSON under `oauth2/<label>` — deliberately
  unchanged: transport adapters (WhatsApp/LINE account provisioning) re-parse that exact shape.
  When a dynamic flow mints token sets, the envelope's persisted refresh context
  (`token_endpoint` + `client_id`, validated at completion) is the specified path — the refresh
  engine already honors it (fixture-tested); absent context, the set is explicitly
  non-refreshable across restart and expires into `reauth_required`.

Refresh mechanics (expiry-skew, single-flight, rotation, atomic rewrite, lease invalidation,
`refresh_failed` vs `reauth_required` classification) are therefore proven against the hermetic
wiremock fixture (`daemon-oauth/tests/refresh.rs`); a live HF sign-in with an operator client id
is the manual evidence leg.

## Storage threat model (decided before landing refresh tokens)

The credential store is a plaintext `0600` JSON file. Long-lived refresh tokens raise the value
of that file. **Decision: acceptable as a DOCUMENTED single-user MVP only.**

- Atomic writes are REQUIRED and implemented (`FileCredentialStore::write_map`: sibling temp
  file, `0600` applied while invisible, rename) — a crash mid-refresh can never truncate the
  store or tear a token set.
- The boundary, stated plainly: the `0600` mode is the whole inter-user boundary; any same-user
  process compromise reads everything in the store. This is not the long-term storage design.
- The envelope carries token MATERIAL plus trusted descriptor identity, never request policy —
  a tampered store cannot redirect a secret to an attacker header or endpoint (the projector's
  curated table owns formatting/headers, keyed on `method_id`).
- **Filed follow-up:** OS-keyring backing for the credential store.
