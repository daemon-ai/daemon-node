#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 Jarrad Hope
#
# THE CLAUDE ApiKeyEnv VERIFICATION GATE (wire v47, curated-table policy).
#
# The curated `claude` row ships `OAuthFamily`, NOT `ApiKeyEnv{ANTHROPIC_API_KEY}`, because
# whether Claude Code honors ANTHROPIC_API_KEY in `--input-format stream-json` mode has not been
# experimentally verified. A descriptor is a tested claim, not a guess: run this experiment with
# a real key, and ONLY if it passes may the curated row change to
# `ApiKeyEnv{var: "ANTHROPIC_API_KEY", label: "Anthropic API key"}`.
#
# Usage:   ANTHROPIC_API_KEY=sk-ant-... scripts/verify-claude-api-key.sh
#
# Method: a pristine HOME (no ~/.claude, no OAuth blob, no keychain) so the ONLY credential the
# binary can possibly use is the env var, then one single-turn stream-json prompt. PASS = a
# result frame with `"is_error":false`; anything else (auth error frame, nonzero exit, timeout)
# = FAIL and the descriptor stays OAuthFamily.
set -euo pipefail

[ -n "${ANTHROPIC_API_KEY:-}" ] || { echo "FAIL: set ANTHROPIC_API_KEY" >&2; exit 2; }
command -v claude >/dev/null || { echo "FAIL: no claude binary on PATH" >&2; exit 2; }

sandbox=$(mktemp -d /tmp/claude-apikey-verify.XXXXXX)
trap 'rm -rf "$sandbox"' EXIT
mkdir -p "$sandbox/home"

echo '{"type":"user","message":{"role":"user","content":[{"type":"text","text":"Reply with the single word: ok"}]}}' |
    env -i \
        HOME="$sandbox/home" \
        PATH="$PATH" \
        TERM=dumb \
        ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
        timeout 120 claude \
        --input-format stream-json \
        --output-format stream-json \
        --verbose \
        --max-turns 1 \
        >"$sandbox/out.ndjson" 2>"$sandbox/err.log" || {
    echo "FAIL: claude exited nonzero (see below)" >&2
    tail -5 "$sandbox/err.log" >&2 || true
    exit 1
}

if grep -q '"type":"result"' "$sandbox/out.ndjson" &&
    grep -q '"is_error":false' "$sandbox/out.ndjson"; then
    echo "PASS: stream-json turn completed on ANTHROPIC_API_KEY alone."
    echo "The curated claude row MAY move to ApiKeyEnv{ANTHROPIC_API_KEY}."
else
    echo "FAIL: no successful result frame — the descriptor stays OAuthFamily." >&2
    tail -5 "$sandbox/out.ndjson" >&2 || true
    exit 1
fi
