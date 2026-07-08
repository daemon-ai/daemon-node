// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Disconnect reason (wire v30, item 2): `DisconnectReason` + the transient
//! `ConnectionState::Disconnecting` + `reason`/`message`/`fatal` on both `TransportChanged` and
//! `TransportInstanceInfo`, plus pre-v30 back-compat (the new fields are serde-default).

use daemon_api::{
    from_cbor, to_cbor, ConnectionState, DisconnectReason, NodeEvent, PresenceState,
    TransportInstanceInfo,
};
use daemon_protocol::TransportId;
use serde::Serialize;

#[test]
fn disconnect_reason_and_states_round_trip() {
    for r in [
        DisconnectReason::UserRequested,
        DisconnectReason::NetworkError,
        DisconnectReason::AuthenticationFailed,
        DisconnectReason::ReplacedByOtherClient,
        DisconnectReason::InvalidSettings,
        DisconnectReason::CertificateError,
        DisconnectReason::Other,
    ] {
        assert_eq!(r, from_cbor::<DisconnectReason>(&to_cbor(&r)).unwrap());
    }
    let ds = ConnectionState::Disconnecting;
    assert_eq!(ds, from_cbor::<ConnectionState>(&to_cbor(&ds)).unwrap());
}

#[test]
fn transport_changed_and_instance_carry_reason_fatal() {
    let ev = NodeEvent::TransportChanged {
        transport: TransportId::new("matrix/@bot:hs.org"),
        connection: ConnectionState::Error,
        presence: PresenceState::Offline,
        reason: Some(DisconnectReason::AuthenticationFailed),
        message: Some("invalid token".into()),
        fatal: true,
    };
    assert_eq!(ev, from_cbor::<NodeEvent>(&to_cbor(&ev)).unwrap());

    let info = TransportInstanceInfo {
        transport: TransportId::new("matrix/@bot:hs.org"),
        family: "matrix".into(),
        display_name: "@bot:hs.org".into(),
        connection: ConnectionState::Disconnecting,
        presence: PresenceState::Offline,
        bound_profile: None,
        reason: Some(DisconnectReason::NetworkError),
        message: None,
        fatal: false,
        enabled: true,
        label: None,
    };
    assert_eq!(
        info,
        from_cbor::<TransportInstanceInfo>(&to_cbor(&info)).unwrap()
    );
}

#[test]
fn pre_v30_transport_changed_decodes_with_defaults() {
    // The pre-v30 shape: no reason/message/fatal (they are serde-default on decode).
    #[derive(Serialize)]
    enum OldEvent {
        TransportChanged {
            transport: TransportId,
            connection: ConnectionState,
            presence: PresenceState,
        },
    }
    let old = OldEvent::TransportChanged {
        transport: TransportId::new("matrix/@bot:hs.org"),
        connection: ConnectionState::Connected,
        presence: PresenceState::Unknown,
    };
    let decoded = from_cbor::<NodeEvent>(&to_cbor(&old)).unwrap();
    match decoded {
        NodeEvent::TransportChanged {
            reason,
            message,
            fatal,
            connection,
            ..
        } => {
            assert_eq!(connection, ConnectionState::Connected);
            assert!(reason.is_none() && message.is_none() && !fatal);
        }
        other => panic!("expected TransportChanged, got {other:?}"),
    }

    #[derive(Serialize)]
    struct OldInstance {
        transport: TransportId,
        family: String,
        display_name: String,
        connection: ConnectionState,
        presence: PresenceState,
        bound_profile: Option<String>,
    }
    let old = OldInstance {
        transport: TransportId::new("matrix/@bot:hs.org"),
        family: "matrix".into(),
        display_name: "@bot:hs.org".into(),
        connection: ConnectionState::Connected,
        presence: PresenceState::Available,
        bound_profile: None,
    };
    let decoded = from_cbor::<TransportInstanceInfo>(&to_cbor(&old)).unwrap();
    assert!(decoded.reason.is_none() && decoded.message.is_none() && !decoded.fatal);
}
