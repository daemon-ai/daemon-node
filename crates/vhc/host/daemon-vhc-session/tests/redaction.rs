// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Redaction by construction (D-P8; ABI §12.3 [CI-9]): NO secret ever crosses the node↔worker
//! command wire or lands in a durable journal.
//!
//! Two byte-scan negatives over distinctive canaries:
//! 1. the WS **auth token** — authored into the keystore CREDENTIALS RECORD, referenced on the
//!    wire only by `secret_ref`; and
//! 2. the **per-run key seed** — minted into the keystore, resolved by the worker read-only.
//!
//! Both canaries must be ABSENT from (a) the encoded `Command::JoinRun` bytes the node sends and
//! (b) every durable journal segment the role session writes. A journal by construction records
//! opaque frames + hashes, never credentials; the command wire carries a reference, never the
//! secret.

// Test-only scan of segment files inside the test's own tempdir — not a production fs path.
#![allow(clippy::disallowed_methods)]

use daemon_vhc_host::run::{JournalSink, RunIdentity};
use daemon_vhc_session::journal_home::{journal_dir, DurableSink};
use daemon_vhc_session::keystore::VhcKeystore;
use daemon_vhc_session::protocol::{
    self, Command, CredentialsRecord, JoinPolicy, PolicyMode, SessionCredentials, WsAuthSpec,
};
use daemon_vhc_session::provisioning::{provision_run_identity, ProvisionScope};

/// A token no legitimate encoding path would ever produce — its bytes are the scan canary.
const TOKEN: &str = "sk-REDACTION-CANARY-DEADBEEF-DO-NOT-LEAK";
const RUN_LABEL: &str = "redaction-run";
const ROLE: &str = "trainer";
const INCARNATION: u64 = 7;
const GENESIS: [u8; 32] = [0x9E; 32];
const MODULE: [u8; 32] = [0x2A; 32];

/// True when `needle` occurs anywhere in `haystack` (the leak check).
fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    !needle.is_empty() && haystack.windows(needle.len()).any(|w| w == needle)
}

#[test]
fn no_secret_crosses_the_command_wire_or_lands_in_a_journal() {
    let dir = tempfile::tempdir().expect("tempdir");
    let keystore = VhcKeystore::open(dir.path()).expect("open keystore");

    // Provision the per-run identity (mint key + issue cert) and read back the key SEED — the
    // second canary. Nothing legitimate should ever serialize the raw seed.
    provision_run_identity(
        &keystore,
        &ProvisionScope {
            run_label: RUN_LABEL,
            genesis_hash: GENESIS,
            epoch: 0,
            role: ROLE,
            incarnation: INCARNATION,
            module_hash: MODULE,
        },
    )
    .expect("provision");
    let seed = keystore
        .existing_run_signing_key(RUN_LABEL, ROLE, INCARNATION)
        .expect("read key")
        .expect("key provisioned")
        .to_bytes();

    // Author the credentials: the token lives ONLY in the keystore record; the wire body carries
    // a bare reference name.
    let secret_ref = keystore
        .store_run_credentials(
            RUN_LABEL,
            ROLE,
            INCARNATION,
            &CredentialsRecord {
                ws_auth: WsAuthSpec::Bearer(TOKEN.to_string()),
                expires_at_ms: 0,
            },
        )
        .expect("store credentials record");
    let credentials = SessionCredentials {
        genesis_hash: GENESIS,
        ws_base: Some("http://127.0.0.1:8795/api/v1/vhc".into()),
        ws_auth: WsAuthSpec::None, // secret is by-reference, never inline
        iroh: None,
        presign_base: Some("http://127.0.0.1:8795/api/v1/vhc".into()),
        peer_certs: Vec::new(),
        seat_grant: None,
        secret_ref: Some(secret_ref),
        expires_at_ms: 0,
        restore: None,
        reconstruct: None,
        catch_up: None,
    };

    // (a) the command wire: encode the exact `JoinRun` the node sends.
    let cmd = Command::JoinRun {
        run_id: RUN_LABEL.into(),
        coordinator: "http://127.0.0.1:8795/api/v1/vhc".into(),
        credentials: credentials.to_bytes().expect("encode credentials"),
        policy: JoinPolicy {
            mode: PolicyMode::Always,
            vram_cap_mb: 0,
            duty_cycle_pct: 100,
            schedule: None,
        },
        admitted_tuple: None,
    };
    let wire = protocol::encode(&cmd).expect("encode command");
    assert!(
        !contains(&wire, TOKEN.as_bytes()),
        "the WS auth token must never ride the command wire (it lives in the keystore record)"
    );
    assert!(
        !contains(&wire, &seed),
        "the per-run key seed must never ride the command wire"
    );

    // (b) the durable journal: write a header + a publish + an event, then scan every segment.
    let identity = RunIdentity {
        run_id: GENESIS,
        epoch: 0,
        role: ROLE.into(),
        instance: INCARNATION,
        module: MODULE,
    };
    let jdir = journal_dir(dir.path(), RUN_LABEL, ROLE, INCARNATION);
    {
        let mut sink = DurableSink::open(&jdir, &identity, [0x5C; 32]).expect("open journal");
        sink.run_header(
            2 << 16,
            &[("vhc".into(), 2)],
            false,
            b"m",
            b"c",
            b"g",
            daemon_vhc_host::run::RunHeaderResources::Declared(b"cl"),
            b"ch",
            b"d",
        )
        .unwrap();
        sink.event(1, b"an-opaque-inbound-frame").unwrap();
        sink.publish(0, 0, b"an-opaque-payload", b"an-opaque-signed-frame")
            .unwrap();
        sink.terminal(0, Some(0), None).unwrap();
    }
    let mut scanned = 0usize;
    for entry in std::fs::read_dir(&jdir).expect("journal dir") {
        let path = entry.expect("entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("dvhcjrn") {
            continue;
        }
        let bytes = std::fs::read(&path).expect("read segment");
        assert!(
            !contains(&bytes, TOKEN.as_bytes()),
            "no auth token in journal segment {}",
            path.display()
        );
        assert!(
            !contains(&bytes, &seed),
            "no per-run key seed in journal segment {}",
            path.display()
        );
        scanned += 1;
    }
    assert!(scanned > 0, "at least one journal segment was scanned");
}
