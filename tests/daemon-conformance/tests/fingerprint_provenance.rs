// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Fingerprint provenance (wire v30, item 8): `RememberedFingerprint` gains `remembered_at_ms` and
//! a populated `label`, with pre-v30 back-compat (both are serde-default). The durable
//! snapshot-provenance round-trip is a `daemon-core` unit test; the live `fingerprint_list`
//! surfacing is exercised by `fingerprint_manage.rs`.

use daemon_api::{from_cbor, to_cbor, RememberedFingerprint};
use serde::Serialize;

#[test]
fn remembered_fingerprint_with_provenance_round_trips() {
    let fp = RememberedFingerprint {
        fingerprint: "ab12cd34".into(),
        label: Some("git status".into()),
        remembered_at_ms: 1_700_000_000_000,
    };
    assert_eq!(
        fp,
        from_cbor::<RememberedFingerprint>(&to_cbor(&fp)).unwrap()
    );
}

#[test]
fn pre_v30_fingerprint_decodes_with_defaults() {
    #[derive(Serialize)]
    struct OldFingerprint {
        fingerprint: String,
    }
    let old = OldFingerprint {
        fingerprint: "ab12cd34".into(),
    };
    let decoded = from_cbor::<RememberedFingerprint>(&to_cbor(&old)).unwrap();
    assert_eq!(decoded.fingerprint, "ab12cd34");
    assert_eq!(decoded.remembered_at_ms, 0);
    assert!(decoded.label.is_none());
}
