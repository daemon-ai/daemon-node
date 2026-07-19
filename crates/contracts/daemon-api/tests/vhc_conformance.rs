// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
// Phase 4: integration test crate; raw ciborium is expected in tests.
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

//! WIRE-1 — the `VhcApi` wire surface (spec §10.4) validates against `daemon-api.cddl`.
//!
//! Mirrors `tests/conformance.rs` but scoped to the vhc additions, and constructs the values
//! in-test (no committed-fixture dependency): every `Vhc*` request/response variant, the
//! `NodeEvent::VhcChanged` feed pointer, and representative DTO edge cases (eligibility headroom,
//! optional policy schedule, every `VhcEvent` arm) must validate against the authoritative CDDL
//! under `api-request` / `api-response`; and clearly-invalid vhc payloads must be rejected (proving
//! the schema discriminates). `WIRE-2` (`conformance_proptest.rs`, `--features arbitrary`) covers the
//! whole variant space; this is the readable, deterministic golden set.

use std::collections::BTreeMap;

use daemon_api::{
    ApiRequest, ApiResponse, EventsPage, NodeEvent, VhcCapabilities, VhcContribution,
    VhcEligibility, VhcEvent, VhcHardwareReport, VhcLeaveMode, VhcPolicy, VhcPolicyMode,
    VhcRunDetail, VhcRunSummary,
};

const CDDL: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/daemon-api.cddl"));

fn enc<T: serde::Serialize>(v: &T) -> Vec<u8> {
    let mut b = Vec::new();
    ciborium::ser::into_writer(v, &mut b).expect("encode");
    b
}

#[track_caller]
fn valid(root: &str, bytes: &[u8], label: &str) {
    cddl_cat::validate_cbor_bytes(root, CDDL, bytes)
        .unwrap_or_else(|e| panic!("`{label}` failed to validate against `{root}`: {e:?}"));
}

fn policy(mode: VhcPolicyMode, schedule: Option<&str>) -> VhcPolicy {
    VhcPolicy {
        mode,
        vram_cap_mb: 12_000,
        duty_cycle_pct: 80,
        schedule: schedule.map(str::to_string),
    }
}

fn eligibility() -> VhcEligibility {
    let mut headroom = BTreeMap::new();
    headroom.insert("vram_mb".to_string(), 4096);
    headroom.insert("ram_mb".to_string(), -512);
    VhcEligibility {
        eligible: false,
        reasons: vec!["insufficient host RAM".into()],
        headroom,
    }
}

fn hardware() -> VhcHardwareReport {
    VhcHardwareReport {
        gpus: 1,
        vram_mb: 24_000,
        shared_mb: 120_000,
        ram_mb: 64_000,
        backend_lanes: vec!["cpu".into(), "vulkan".into()],
        capabilities: VhcCapabilities {
            abi_version: 1,
            ops: vec!["matmul@1".into(), "adamw_step@1".into()],
            payload_stores: vec!["r2".into()],
        },
        up_kbps: 10_000,
        down_kbps: 50_000,
        disk_free_mb: 200_000,
        throughput_class: "c2".into(),
    }
}

fn contribution() -> VhcContribution {
    VhcContribution {
        rounds: 42,
        tokens: 1_000_000,
        bytes_up: 2_048,
        bytes_down: 8_192,
        witness_count: 7,
        checkpoint_credits: 2,
    }
}

fn all_events() -> Vec<VhcEvent> {
    vec![
        VhcEvent::Phase {
            run_id: "run-1".into(),
            phase: "RoundTrain".into(),
            epoch: 3,
            round: 17,
        },
        VhcEvent::Progress {
            run_id: "run-1".into(),
            inner_step: 4,
            loss_micros: 3_907_700,
            tokens_per_s_milli: 12_500,
            peers: 3,
        },
        VhcEvent::RoundOutcome {
            run_id: "run-1".into(),
            round: 17,
            committed: 3,
            ingested: 3,
            stalled: false,
        },
        VhcEvent::Contribution {
            run_id: "run-1".into(),
            contribution: contribution(),
        },
        VhcEvent::Warning {
            run_id: "run-1".into(),
            class: "stall".into(),
            detail: "peer slow".into(),
        },
        VhcEvent::Error {
            run_id: "run-1".into(),
            class: "desync".into(),
            detail: "digest mismatch".into(),
        },
    ]
}

fn summary(joined: bool, policy: Option<VhcPolicy>) -> VhcRunSummary {
    VhcRunSummary {
        run_id: "run-1".into(),
        phase: "RoundTrain".into(),
        joined,
        eligibility: eligibility(),
        policy,
        last_round: 17,
        // The D0 additive fields absent — the pre-D0 / v1-only-row encoding (skip-if-none).
        ..VhcRunSummary::default()
    }
}

/// A v2-identified run row carrying every D0 additive field (envelope v2: the hex RunId,
/// execution-identity trio, and the D5 sunset-observability fields) plus the additive
/// run-instance lifecycle fields (effective state, retry budget, terminal reason).
fn summary_v2() -> VhcRunSummary {
    VhcRunSummary {
        run_id_hash: Some("ab".repeat(32)),
        epoch: Some(3),
        role: Some("worker".into()),
        instance: Some(42),
        envelope_schema_major: Some(2),
        module_abi_major: Some(2),
        selected_driver: Some("v2".into()),
        module_hash: Some("22".repeat(32)),
        run_state: Some("failed_retryable".into()),
        retry_count: Some(2),
        terminal_reason: Some("transport loss".into()),
        ..summary(true, Some(policy(VhcPolicyMode::Idle, None)))
    }
}

#[test]
fn vhc_requests_validate() {
    let cases: Vec<(&str, ApiRequest)> = vec![
        ("VhcRunList", ApiRequest::VhcRunList),
        (
            "VhcRunDetail",
            ApiRequest::VhcRunDetail {
                run_id: "run-1".into(),
            },
        ),
        (
            "VhcJoin(scheduled+schedule)",
            ApiRequest::VhcJoin {
                run_id: "run-1".into(),
                policy: policy(VhcPolicyMode::Scheduled, Some("0 2 * * *")),
                op_id: "op-1".into(),
            },
        ),
        (
            "VhcJoin(idle,no schedule)",
            ApiRequest::VhcJoin {
                run_id: "run-1".into(),
                policy: policy(VhcPolicyMode::Idle, None),
                op_id: "op-2".into(),
            },
        ),
        (
            "VhcLeave(graceful)",
            ApiRequest::VhcLeave {
                run_id: "run-1".into(),
                mode: VhcLeaveMode::Graceful,
                op_id: "op-3".into(),
            },
        ),
        (
            "VhcLeave(immediate)",
            ApiRequest::VhcLeave {
                run_id: "run-1".into(),
                mode: VhcLeaveMode::Immediate,
                op_id: "op-4".into(),
            },
        ),
        (
            "VhcPause",
            ApiRequest::VhcPause {
                run_id: "run-1".into(),
                op_id: "op-5".into(),
            },
        ),
        (
            "VhcResume",
            ApiRequest::VhcResume {
                run_id: "run-1".into(),
                op_id: "op-6".into(),
            },
        ),
        (
            "VhcSetPolicy",
            ApiRequest::VhcSetPolicy {
                policy: policy(VhcPolicyMode::Always, None),
            },
        ),
        ("VhcHardwareReport", ApiRequest::VhcHardwareReport),
    ];
    for (label, req) in cases {
        valid("api-request", &enc(&req), label);
    }
}

#[test]
fn vhc_responses_validate() {
    let detail = VhcRunDetail {
        summary: summary(true, Some(policy(VhcPolicyMode::Idle, None))),
        coordinator: "https://api.daemon.ai/api/v1/vhc".into(),
        contribution: contribution(),
        recent_events: all_events(),
    };
    let cases: Vec<(&str, ApiResponse)> = vec![
        (
            "VhcRuns",
            ApiResponse::VhcRuns(vec![
                summary(false, None),
                summary(true, Some(policy(VhcPolicyMode::Manual, None))),
                // D0 additive: a v2-identified row with the full run-identity + observability set.
                summary_v2(),
            ]),
        ),
        (
            "VhcRunDetail(Some)",
            ApiResponse::VhcRunDetail(Some(detail)),
        ),
        ("VhcRunDetail(None)", ApiResponse::VhcRunDetail(None)),
        (
            "VhcHardwareReport",
            ApiResponse::VhcHardwareReport(hardware()),
        ),
    ];
    for (label, resp) in cases {
        valid("api-response", &enc(&resp), label);
    }
}

#[test]
fn vhc_changed_feed_pointer_validates() {
    // The live `vhc_subscribe` rides the existing events feed as a `VhcChanged` pointer.
    let page = EventsPage {
        events: vec![
            NodeEvent::VhcChanged {
                run_id: Some("run-1".into()),
                rev: 9,
            },
            NodeEvent::VhcChanged {
                run_id: None,
                rev: 10,
            },
        ],
        next_cursor: 10,
        head_cursor: 10,
        epoch: Some(1),
    };
    valid(
        "api-response",
        &enc(&ApiResponse::EventsPage(page)),
        "EventsPage[VhcChanged]",
    );
}

#[test]
fn invalid_vhc_payloads_are_rejected() {
    use ciborium::value::{Integer, Value};
    let int = |n: i64| Value::Integer(Integer::from(n));
    let enc_v = |v: &Value| {
        let mut b = Vec::new();
        ciborium::ser::into_writer(v, &mut b).unwrap();
        b
    };

    // VhcJoin missing the required `op_id`.
    let missing_op = enc_v(&Value::Map(vec![(
        Value::Text("VhcJoin".into()),
        Value::Map(vec![
            (Value::Text("run_id".into()), Value::Text("r".into())),
            (
                Value::Text("policy".into()),
                Value::Map(vec![
                    (Value::Text("mode".into()), Value::Text("idle".into())),
                    (Value::Text("vram_cap_mb".into()), int(0)),
                    (Value::Text("duty_cycle_pct".into()), int(100)),
                ]),
            ),
        ]),
    )]));
    // VhcSetPolicy with an out-of-vocabulary policy mode.
    let bad_mode = enc_v(&Value::Map(vec![(
        Value::Text("VhcSetPolicy".into()),
        Value::Map(vec![(
            Value::Text("policy".into()),
            Value::Map(vec![
                (Value::Text("mode".into()), Value::Text("turbo".into())),
                (Value::Text("vram_cap_mb".into()), int(0)),
                (Value::Text("duty_cycle_pct".into()), int(100)),
            ]),
        )]),
    )]));
    // VhcRunDetail with a wrong-typed `run_id` (must be tstr).
    let bad_run_id = enc_v(&Value::Map(vec![(
        Value::Text("VhcRunDetail".into()),
        Value::Map(vec![(Value::Text("run_id".into()), int(1))]),
    )]));

    for (label, bytes) in [
        ("VhcJoin missing op_id", missing_op),
        ("VhcSetPolicy bad mode", bad_mode),
        ("VhcRunDetail wrong run_id type", bad_run_id),
    ] {
        assert!(
            cddl_cat::validate_cbor_bytes("api-request", CDDL, &bytes).is_err(),
            "expected `{label}` to be rejected by the CDDL, but it validated"
        );
    }
}
