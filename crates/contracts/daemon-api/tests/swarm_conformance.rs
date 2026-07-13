// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
// Phase 4: integration test crate; raw ciborium is expected in tests.
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

//! WIRE-1 — the `SwarmApi` wire surface (spec §10.4) validates against `daemon-api.cddl`.
//!
//! Mirrors `tests/conformance.rs` but scoped to the swarm additions, and constructs the values
//! in-test (no committed-fixture dependency): every `Swarm*` request/response variant, the
//! `NodeEvent::SwarmChanged` feed pointer, and representative DTO edge cases (eligibility headroom,
//! optional policy schedule, every `SwarmEvent` arm) must validate against the authoritative CDDL
//! under `api-request` / `api-response`; and clearly-invalid swarm payloads must be rejected (proving
//! the schema discriminates). `WIRE-2` (`conformance_proptest.rs`, `--features arbitrary`) covers the
//! whole variant space; this is the readable, deterministic golden set.

use std::collections::BTreeMap;

use daemon_api::{
    ApiRequest, ApiResponse, EventsPage, NodeEvent, SwarmCapabilities, SwarmContribution,
    SwarmEligibility, SwarmEvent, SwarmHardwareReport, SwarmLeaveMode, SwarmPolicy,
    SwarmPolicyMode, SwarmRunDetail, SwarmRunSummary,
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

fn policy(mode: SwarmPolicyMode, schedule: Option<&str>) -> SwarmPolicy {
    SwarmPolicy {
        mode,
        vram_cap_mb: 12_000,
        duty_cycle_pct: 80,
        schedule: schedule.map(str::to_string),
    }
}

fn eligibility() -> SwarmEligibility {
    let mut headroom = BTreeMap::new();
    headroom.insert("vram_mb".to_string(), 4096);
    headroom.insert("ram_mb".to_string(), -512);
    SwarmEligibility {
        eligible: false,
        reasons: vec!["insufficient host RAM".into()],
        headroom,
    }
}

fn hardware() -> SwarmHardwareReport {
    SwarmHardwareReport {
        gpus: 1,
        vram_mb: 24_000,
        shared_mb: 120_000,
        ram_mb: 64_000,
        backend_lanes: vec!["cpu".into(), "vulkan".into()],
        capabilities: SwarmCapabilities {
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

fn contribution() -> SwarmContribution {
    SwarmContribution {
        rounds: 42,
        tokens: 1_000_000,
        bytes_up: 2_048,
        bytes_down: 8_192,
        witness_count: 7,
        checkpoint_credits: 2,
    }
}

fn all_events() -> Vec<SwarmEvent> {
    vec![
        SwarmEvent::Phase {
            run_id: "run-1".into(),
            phase: "RoundTrain".into(),
            epoch: 3,
            round: 17,
        },
        SwarmEvent::Progress {
            run_id: "run-1".into(),
            inner_step: 4,
            loss_micros: 3_907_700,
            tokens_per_s_milli: 12_500,
            peers: 3,
        },
        SwarmEvent::RoundOutcome {
            run_id: "run-1".into(),
            round: 17,
            committed: 3,
            ingested: 3,
            stalled: false,
        },
        SwarmEvent::Contribution {
            run_id: "run-1".into(),
            contribution: contribution(),
        },
        SwarmEvent::Warning {
            run_id: "run-1".into(),
            class: "stall".into(),
            detail: "peer slow".into(),
        },
        SwarmEvent::Error {
            run_id: "run-1".into(),
            class: "desync".into(),
            detail: "digest mismatch".into(),
        },
    ]
}

fn summary(joined: bool, policy: Option<SwarmPolicy>) -> SwarmRunSummary {
    SwarmRunSummary {
        run_id: "run-1".into(),
        phase: "RoundTrain".into(),
        joined,
        eligibility: eligibility(),
        policy,
        last_round: 17,
    }
}

#[test]
fn swarm_requests_validate() {
    let cases: Vec<(&str, ApiRequest)> = vec![
        ("SwarmRunList", ApiRequest::SwarmRunList),
        (
            "SwarmRunDetail",
            ApiRequest::SwarmRunDetail {
                run_id: "run-1".into(),
            },
        ),
        (
            "SwarmJoin(scheduled+schedule)",
            ApiRequest::SwarmJoin {
                run_id: "run-1".into(),
                policy: policy(SwarmPolicyMode::Scheduled, Some("0 2 * * *")),
                op_id: "op-1".into(),
            },
        ),
        (
            "SwarmJoin(idle,no schedule)",
            ApiRequest::SwarmJoin {
                run_id: "run-1".into(),
                policy: policy(SwarmPolicyMode::Idle, None),
                op_id: "op-2".into(),
            },
        ),
        (
            "SwarmLeave(graceful)",
            ApiRequest::SwarmLeave {
                run_id: "run-1".into(),
                mode: SwarmLeaveMode::Graceful,
                op_id: "op-3".into(),
            },
        ),
        (
            "SwarmLeave(immediate)",
            ApiRequest::SwarmLeave {
                run_id: "run-1".into(),
                mode: SwarmLeaveMode::Immediate,
                op_id: "op-4".into(),
            },
        ),
        (
            "SwarmSetPolicy",
            ApiRequest::SwarmSetPolicy {
                policy: policy(SwarmPolicyMode::Always, None),
            },
        ),
        ("SwarmHardwareReport", ApiRequest::SwarmHardwareReport),
    ];
    for (label, req) in cases {
        valid("api-request", &enc(&req), label);
    }
}

#[test]
fn swarm_responses_validate() {
    let detail = SwarmRunDetail {
        summary: summary(true, Some(policy(SwarmPolicyMode::Idle, None))),
        coordinator: "https://api.daemon.ai/api/v1/swarm".into(),
        contribution: contribution(),
        recent_events: all_events(),
    };
    let cases: Vec<(&str, ApiResponse)> = vec![
        (
            "SwarmRuns",
            ApiResponse::SwarmRuns(vec![
                summary(false, None),
                summary(true, Some(policy(SwarmPolicyMode::Manual, None))),
            ]),
        ),
        (
            "SwarmRunDetail(Some)",
            ApiResponse::SwarmRunDetail(Some(detail)),
        ),
        ("SwarmRunDetail(None)", ApiResponse::SwarmRunDetail(None)),
        (
            "SwarmHardwareReport",
            ApiResponse::SwarmHardwareReport(hardware()),
        ),
    ];
    for (label, resp) in cases {
        valid("api-response", &enc(&resp), label);
    }
}

#[test]
fn swarm_changed_feed_pointer_validates() {
    // The live `swarm_subscribe` rides the existing events feed as a `SwarmChanged` pointer.
    let page = EventsPage {
        events: vec![
            NodeEvent::SwarmChanged {
                run_id: Some("run-1".into()),
                rev: 9,
            },
            NodeEvent::SwarmChanged {
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
        "EventsPage[SwarmChanged]",
    );
}

#[test]
fn invalid_swarm_payloads_are_rejected() {
    use ciborium::value::{Integer, Value};
    let int = |n: i64| Value::Integer(Integer::from(n));
    let enc_v = |v: &Value| {
        let mut b = Vec::new();
        ciborium::ser::into_writer(v, &mut b).unwrap();
        b
    };

    // SwarmJoin missing the required `op_id`.
    let missing_op = enc_v(&Value::Map(vec![(
        Value::Text("SwarmJoin".into()),
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
    // SwarmSetPolicy with an out-of-vocabulary policy mode.
    let bad_mode = enc_v(&Value::Map(vec![(
        Value::Text("SwarmSetPolicy".into()),
        Value::Map(vec![(
            Value::Text("policy".into()),
            Value::Map(vec![
                (Value::Text("mode".into()), Value::Text("turbo".into())),
                (Value::Text("vram_cap_mb".into()), int(0)),
                (Value::Text("duty_cycle_pct".into()), int(100)),
            ]),
        )]),
    )]));
    // SwarmRunDetail with a wrong-typed `run_id` (must be tstr).
    let bad_run_id = enc_v(&Value::Map(vec![(
        Value::Text("SwarmRunDetail".into()),
        Value::Map(vec![(Value::Text("run_id".into()), int(1))]),
    )]));

    for (label, bytes) in [
        ("SwarmJoin missing op_id", missing_op),
        ("SwarmSetPolicy bad mode", bad_mode),
        ("SwarmRunDetail wrong run_id type", bad_run_id),
    ] {
        assert!(
            cddl_cat::validate_cbor_bytes("api-request", CDDL, &bytes).is_err(),
            "expected `{label}` to be rejected by the CDDL, but it validated"
        );
    }
}
