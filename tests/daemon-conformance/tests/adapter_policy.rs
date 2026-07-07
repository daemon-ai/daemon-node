// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Adapter policy query (wire v30, item 4): `PolicyEntry` on `AdapterInfo`, with pre-v30
//! back-compat (`policies` is serde-default). The live matrix `auto_accept_invites` assertion is a
//! `daemon-matrix` unit test (it owns the config).

use daemon_api::{
    from_cbor, to_cbor, AccountSettingsSchema, AdapterCapabilities, AdapterInfo, PolicyEntry,
};
use serde::Serialize;

#[test]
fn adapter_info_with_policies_round_trips() {
    let info = AdapterInfo {
        family: "matrix".into(),
        display_name: "Matrix".into(),
        capabilities: AdapterCapabilities::default(),
        account_schema: AccountSettingsSchema::default(),
        policies: vec![PolicyEntry {
            key: "auto_accept_invites".into(),
            label: "Automatically accept room invites".into(),
            value: "true".into(),
        }],
    };
    assert_eq!(info, from_cbor::<AdapterInfo>(&to_cbor(&info)).unwrap());
}

#[test]
fn pre_v30_adapter_info_decodes_with_empty_policies() {
    #[derive(Serialize)]
    struct OldAdapterInfo {
        family: String,
        display_name: String,
        capabilities: AdapterCapabilities,
        account_schema: AccountSettingsSchema,
    }
    let old = OldAdapterInfo {
        family: "room".into(),
        display_name: "Rooms (internal)".into(),
        capabilities: AdapterCapabilities::default(),
        account_schema: AccountSettingsSchema::default(),
    };
    let decoded = from_cbor::<AdapterInfo>(&to_cbor(&old)).unwrap();
    assert!(decoded.policies.is_empty());
}
