// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Real foreign-agent discovery e2e (env-gated, NON-gating): probe the curated direct-binary recipe
//! table against the actual coding-agent CLIs on `$PATH`. This is the counterpart to the mock-based
//! `foreign_protocols` conformance test — it runs the genuine `daemon_acp::AcpDiscoverer` against the
//! real binaries the `nix develop .#e2e` shell puts on `PATH` (from `llm-agents.nix`).
//!
//! It is gated behind `DAEMON_E2E_AGENTS=1` and skips otherwise, so the offline `cargo test
//! --workspace` gate stays mock-only and hermetic. Real ACP `initialize` handshakes may need API
//! keys / network, so the strict "installed ACP agents must verify" assertion is a further opt-in
//! (`DAEMON_E2E_AGENTS_STRICT=1`); by default we assert only the deterministic invariants.

use daemon_api::{AgentProtocol, AgentVerification};
use daemon_host::AgentDiscovery;

/// Every name the curated table is expected to surface (installed or not — `discover` lists the whole
/// table). Keep in lockstep with `CURATED` in `daemon-acp/src/lib.rs`.
const EXPECTED_NAMES: &[&str] = &[
    // ACP
    "gemini",
    "qwen",
    "goose",
    "opencode",
    "codex",
    "kimi",
    "crow-cli",
    "cursor-agent",
    "copilot",
    "droid",
    "iflow",
    "qoder",
    "kilocode",
    "mistral-vibe",
    "junie",
    "eca",
    // stream-json
    "claude",
    "amp",
];

/// The stream-json members of the curated table (no `initialize` handshake → always unverified).
const STREAM_JSON_NAMES: &[&str] = &["claude", "amp"];

fn enabled() -> bool {
    std::env::var("DAEMON_E2E_AGENTS").as_deref() == Ok("1")
}

fn strict() -> bool {
    std::env::var("DAEMON_E2E_AGENTS_STRICT").as_deref() == Ok("1")
}

#[tokio::test]
async fn curated_agents_discover_on_path() {
    if !enabled() {
        eprintln!(
            "skipping foreign-agent discovery e2e (set DAEMON_E2E_AGENTS=1, run in `nix develop .#e2e`)"
        );
        return;
    }

    let entries = daemon_acp::AcpDiscoverer::new().discover().await;

    // 1. The catalog surfaces the entire curated table regardless of install state.
    for name in EXPECTED_NAMES {
        assert!(
            entries.iter().any(|e| e.name == *name),
            "curated agent `{name}` missing from the discovery catalog"
        );
    }

    // 2. Report + tally, projecting the NODE-derived `verification` verdict (not a client re-derive).
    //    Also assert the node's derivation is internally consistent with the raw fields it computed
    //    from: NotInstalled ⇔ !installed; Verified ⇒ installed ACP with a version; Unverified ⇒
    //    installed without a confirmed ACP handshake version.
    let mut installed = 0usize;
    let mut verified = 0usize;
    for e in &entries {
        match e.verification {
            AgentVerification::NotInstalled => assert!(
                !e.installed,
                "`{}` reports NotInstalled but installed=true",
                e.name
            ),
            AgentVerification::Verified => {
                verified += 1;
                assert!(
                    e.installed && matches!(e.protocol, AgentProtocol::Acp) && e.version.is_some(),
                    "`{}` reports Verified but is not an installed ACP agent with a version",
                    e.name
                );
            }
            AgentVerification::Unverified => assert!(
                e.installed && !(matches!(e.protocol, AgentProtocol::Acp) && e.version.is_some()),
                "`{}` reports Unverified but looks installed+verified or not installed",
                e.name
            ),
        }
        if e.installed {
            installed += 1;
        }
        eprintln!(
            "  {:<14} [{:?}] {:<13?} version={:?}",
            e.name, e.protocol, e.verification, e.version
        );
    }
    eprintln!(
        "discovered {} entries, {installed} installed, {verified} verified",
        entries.len()
    );

    assert!(
        installed > 0,
        "no curated agent resolved on PATH — is this running in `nix develop .#e2e`?"
    );

    // 3. Deterministic invariant: an installed stream-json agent has NO handshake, so the node must
    //    surface it as `Unverified` (version stays None) — the "surface their unverified nature".
    for name in STREAM_JSON_NAMES {
        if let Some(e) = entries.iter().find(|e| e.name == *name) {
            assert_eq!(
                e.protocol,
                AgentProtocol::StreamJson,
                "`{name}` must be catalogued as a stream-json agent"
            );
            if e.installed {
                assert_eq!(
                    e.verification,
                    AgentVerification::Unverified,
                    "stream-json agent `{name}` must report Unverified (no initialize handshake)"
                );
                assert!(
                    e.version.is_none(),
                    "stream-json agent `{name}` must have no handshake version"
                );
            }
        }
    }

    // 4. Opt-in strict mode: every installed ACP agent must be `Verified` (reported a version via
    //    `initialize`). Off by default because a real handshake can require credentials/network the
    //    shell doesn't inject.
    if strict() {
        for e in &entries {
            if e.installed && matches!(e.protocol, AgentProtocol::Acp) {
                assert_eq!(
                    e.verification,
                    AgentVerification::Verified,
                    "installed ACP agent `{}` did not verify via initialize (strict mode)",
                    e.name
                );
            }
        }
    }
}
