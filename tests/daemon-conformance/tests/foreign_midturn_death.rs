// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Structured foreign failure (wire v30, item 9, C6): `ForeignFailure`/`ForeignStage` +
//! `TurnSummary.failure`, with pre-v30 back-compat (`failure` is serde-default). The live
//! exit-watch (a foreign child dying mid-turn synthesizes a `TurnFinished{Failed}`) is a
//! `daemon-host` unit test in `foreign.rs`.

use daemon_api::{from_cbor, to_cbor};
use daemon_common::UsageDelta;
use daemon_protocol::{EndReason, ForeignFailure, ForeignStage, TurnSummary};
use serde::Serialize;

#[test]
fn turn_summary_with_foreign_failure_round_trips() {
    let summary = TurnSummary::foreign_failed(ForeignFailure {
        stage: ForeignStage::Turn,
        agent: Some("gemini".into()),
    });
    assert_eq!(
        summary,
        from_cbor::<TurnSummary>(&to_cbor(&summary)).unwrap()
    );
    assert_eq!(summary.end_reason, EndReason::Failed);

    for stage in [
        ForeignStage::Spawn,
        ForeignStage::Handshake,
        ForeignStage::Turn,
        ForeignStage::Unknown,
    ] {
        let f = ForeignFailure { stage, agent: None };
        assert_eq!(f, from_cbor::<ForeignFailure>(&to_cbor(&f)).unwrap());
    }
}

#[test]
fn pre_v30_turn_summary_decodes_with_none_failure() {
    #[derive(Serialize)]
    struct OldTurnSummary {
        end_reason: EndReason,
        final_text: Option<String>,
        usage: UsageDelta,
    }
    let old = OldTurnSummary {
        end_reason: EndReason::Completed,
        final_text: Some("done".into()),
        usage: UsageDelta::default(),
    };
    let decoded = from_cbor::<TurnSummary>(&to_cbor(&old)).unwrap();
    assert!(decoded.failure.is_none());
    assert_eq!(decoded.end_reason, EndReason::Completed);
}
