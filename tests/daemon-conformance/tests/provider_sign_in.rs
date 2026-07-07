// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Provider sign-in advertisement (wire v30, item 5, CON-15 wire half): `ProviderSignIn` on
//! `ProviderDescriptor`, with pre-v30 back-compat (`sign_in` is serde-default). The OpenRouter
//! catalog row assertion is a `bins/daemon` binary-level test.

use daemon_api::{
    from_cbor, to_cbor, ProviderDescriptor, ProviderKindWire, ProviderSelector, ProviderSignIn,
};
use serde::Serialize;

#[test]
fn provider_descriptor_with_sign_in_round_trips() {
    let with = ProviderDescriptor {
        id: "open_router".into(),
        display_name: "OpenRouter".into(),
        kind: ProviderKindWire::Cloud,
        wire_selector: ProviderSelector::GenAi,
        requires_key: true,
        supports_model_discovery: true,
        default_base_url: None,
        sign_in: Some(ProviderSignIn {
            family: "provider/openrouter".into(),
            label: "Sign in with OpenRouter".into(),
        }),
    };
    assert_eq!(
        with,
        from_cbor::<ProviderDescriptor>(&to_cbor(&with)).unwrap()
    );

    let without = ProviderDescriptor {
        sign_in: None,
        ..with
    };
    assert_eq!(
        without,
        from_cbor::<ProviderDescriptor>(&to_cbor(&without)).unwrap()
    );
}

#[test]
fn pre_v30_provider_descriptor_decodes_with_none_sign_in() {
    #[derive(Serialize)]
    struct OldDescriptor {
        id: String,
        display_name: String,
        kind: ProviderKindWire,
        wire_selector: ProviderSelector,
        requires_key: bool,
        supports_model_discovery: bool,
        default_base_url: Option<String>,
    }
    let old = OldDescriptor {
        id: "anthropic".into(),
        display_name: "Anthropic".into(),
        kind: ProviderKindWire::Cloud,
        wire_selector: ProviderSelector::GenAi,
        requires_key: true,
        supports_model_discovery: true,
        default_base_url: None,
    };
    let decoded = from_cbor::<ProviderDescriptor>(&to_cbor(&old)).unwrap();
    assert!(decoded.sign_in.is_none());
}
