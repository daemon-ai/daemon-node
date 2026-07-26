// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// The trainer's Logical Resource Plan, checked at the geometry it will actually be admitted for.
//
// The plan is the module's whole statement about resources under the three-object model, and it is
// the input every downstream authority composes from. Two things therefore have to be true of it
// before anything else can be, and neither is provable by reading it:
//
//   1. it **validates** — canonical, sorted, resolvable, maximal live sets, no physical content —
//      because a plan that does not validate is not a refusal at admission, it is a module that
//      cannot be admitted at all; and
//   2. its **footprint is the arithmetic the contract specifies** — persistent floor plus the
//      maximum concurrently-live transient set plus a declared fragmentation allowance, per
//      resource domain — because that is the number the host prices, and an arithmetic slip here
//      is a wrong claim rather than a wrong estimate.
//
// The plan module is dual-compiled here (the same `#[path]` include the model uses) so the source
// under test is the exact source the guest compiles, not a restatement of it.
//
// This lane deliberately asserts SHAPE and ARITHMETIC, never absolute byte figures for the fleet
// geometry. What a real machine can host is the composed claim's business; if the honest figures
// refuse a box, that is a finding for the owner and it belongs in the admission evidence, not in a
// floor quietly widened here to admit the box.

#[path = "../../../guests/tiny-llama/src/model.rs"]
#[allow(dead_code)]
// `slice_assign([off..off + len], patch)` is the compute framework's rank-1 slice API — one range
// per tensor dimension, and this tensor has one dimension. The single-range-in-vec-init lint reads
// it as a collection literal someone got wrong; it is not one. Scoped to this include so the lint
// stays live for real collection literals in the test itself.
#[allow(clippy::single_range_in_vec_init)]
mod model;

#[path = "../../../guests/tiny-llama/src/plan.rs"]
#[allow(dead_code)]
mod plan;

use daemon_vhc_proto::resource_plan::{
    Binding, DimensionValue, Domain, LinearLifetime, SelectionScope,
};

use model::ModelCfg;

/// The frozen ceremony geometry (`daemon-vhc-testkit`'s `ceremony` module owns these values; they
/// are restated here because this lane must not depend on the testkit's harness machinery).
fn ceremony_model() -> ModelCfg {
    ModelCfg {
        d_model: 1536,
        n_layers: 24,
        n_heads: 24,
        head_dim: 64,
        vocab: 32_768,
        seq_len: 2_048,
        ffn_mult: 3,
        rope_theta: 10_000.0,
        rmsnorm_eps: 1.0e-5,
        lr: 3.0e-4,
        beta1: 0.9,
        beta2: 0.95,
        adam_eps: 1.0e-8,
        wd: 0.1,
    }
}

/// A small geometry, so the lane proves the derivation is a function of the configuration rather
/// than a set of constants that happen to suit one model.
fn small_model() -> ModelCfg {
    ModelCfg {
        d_model: 64,
        n_layers: 2,
        n_heads: 2,
        head_dim: 32,
        vocab: 128,
        seq_len: 16,
        ffn_mult: 2,
        ..ceremony_model()
    }
}

fn derive(
    cfg: &ModelCfg,
    micro_batch_max: u32,
) -> daemon_vhc_proto::resource_plan::LogicalResourcePlan {
    plan::derive(cfg, micro_batch_max, 30, 4_190_208, 8_192, 12 << 20)
}

fn binding(micro_batch: u64) -> Binding {
    let mut b = Binding::new();
    b.insert(
        plan::DIM_MICRO_BATCH.to_string(),
        DimensionValue::Uint(micro_batch),
    );
    b
}

#[test]
fn the_plan_validates_at_every_geometry_it_is_derived_for() {
    for (label, cfg) in [("ceremony", ceremony_model()), ("small", small_model())] {
        let p = derive(&cfg, 4);
        p.validate()
            .unwrap_or_else(|e| panic!("[{label}] the derived plan must validate: {e}"));
        // Canonical encoding round-trips, and the plan is inside the schema's byte ceiling — the
        // host rejects an over-ceiling length WITHOUT reading it, so a plan that grew past it
        // would be unadmittable with no diagnosis.
        let bytes = p
            .to_canonical_bytes()
            .unwrap_or_else(|e| panic!("[{label}] canonical encoding: {e}"));
        assert!(
            bytes.len() <= daemon_vhc_proto::resource_plan::LOGICAL_RESOURCE_PLAN_BYTES_MAX,
            "[{label}] the plan is {} B, past the schema ceiling",
            bytes.len()
        );
        assert!(
            p.node_count() <= daemon_vhc_proto::resource_plan::LOGICAL_RESOURCE_PLAN_NODES_MAX,
            "[{label}] the plan has {} nodes, past the schema ceiling",
            p.node_count()
        );
    }
}

#[test]
fn the_plan_is_deterministic_and_carries_no_physical_content() {
    // Determinism from identical inputs is what lets a participant reproduce the envelope's
    // embedded plan byte-for-byte at join; without it the reproduction check is unsatisfiable.
    let a = derive(&ceremony_model(), 4).to_canonical_bytes().unwrap();
    let b = derive(&ceremony_model(), 4).to_canonical_bytes().unwrap();
    assert_eq!(a, b, "identical configuration, identical plan bytes");

    // The scope pair is checked rather than interpreted: the reference trainer is uniform, and a
    // uniform plan must carry no equivalence contract.
    let p = derive(&ceremony_model(), 4);
    assert_eq!(p.selection_scope, SelectionScope::UniformRun);
    assert_eq!(p.equivalence_contract_hash, None);

    // No identifier names a backend, allocator, driver or measurement. `validate` enforces this;
    // asserting it here states the property in the lane that would otherwise only see it as a
    // generic validation failure.
    for name in p
        .tensors
        .iter()
        .map(|t| t.name.as_str())
        .chain(p.operations.iter().map(|o| o.name.as_str()))
        .chain(p.transfers.iter().map(|t| t.name.as_str()))
        .chain(p.linear_memory.iter().map(|t| t.name.as_str()))
        .chain(p.dimensions.iter().map(|d| d.name.as_str()))
    {
        let lowered = name.to_ascii_lowercase();
        for fragment in [
            "vulkan",
            "metal",
            "cuda",
            "dx12",
            "driver",
            "vram",
            "allocator",
        ] {
            assert!(
                !lowered.contains(fragment),
                "identifier `{name}` names physical content (`{fragment}`)"
            );
        }
    }
}

#[test]
fn the_micro_batch_is_the_one_free_dimension_and_its_range_is_offered_whole() {
    let p = derive(&ceremony_model(), 4);
    assert_eq!(p.dimensions.len(), 1, "exactly one dimension is left free");
    assert_eq!(p.dimensions[0].name, plan::DIM_MICRO_BATCH);
    assert_eq!(
        p.dimensions[0].domain,
        Domain::UintRange { lo: 1, hi: 4 },
        "the module offers every value it is willing to run at, and lets the host choose"
    );
    // Every value in the offered range binds and prices. A range with an admissible-looking value
    // the plan cannot actually evaluate would make the host's selection a lottery.
    for mb in 1..=4u64 {
        p.footprint(&binding(mb))
            .unwrap_or_else(|e| panic!("micro-batch {mb} is offered but does not price: {e}"));
    }
    assert!(
        p.check_binding(&binding(5)).is_err(),
        "a value outside the offered range is refused, not clamped"
    );
}

#[test]
fn the_footprint_is_the_peak_arithmetic_and_scales_with_the_selection() {
    let cfg = ceremony_model();
    let p = derive(&cfg, 4);
    let one = p.footprint(&binding(1)).expect("micro-batch 1 prices");
    let four = p.footprint(&binding(4)).expect("micro-batch 4 prices");

    // The persistent device floor is the model plus its two optimizer moments — three f32 copies
    // of the parameter layout — and it does not move with the micro-batch, which is exactly the
    // distinction persistent/transient exists to draw.
    let params: u64 = cfg.param_numels().iter().map(|&n| n as u64).sum();
    assert_eq!(
        one.device_persistent_bytes,
        params * 4 * 3,
        "persistent device floor is the parameters and both moments"
    );
    assert_eq!(
        four.device_persistent_bytes, one.device_persistent_bytes,
        "the floor is independent of the selected micro-batch"
    );

    // The transient peak DOES move with it — a larger micro-batch buys throughput with activation
    // memory, which is the whole reason the choice is the host's and not the module's.
    assert!(
        four.device_transient_peak_bytes > one.device_transient_peak_bytes,
        "a larger micro-batch costs more transient device memory"
    );
    assert_eq!(
        one.device_peak_bytes,
        one.device_persistent_bytes + one.device_transient_peak_bytes,
        "device peak is floor plus transient peak"
    );

    // Linear memory: floor plus the maximum concurrently-live transient set plus the declared
    // fragmentation allowance — never the sum over phases.
    assert_eq!(
        one.linear_peak_bytes,
        one.linear_persistent_bytes
            + one.linear_transient_peak_bytes
            + one.linear_fragmentation_headroom_bytes,
        "linear peak is floor + maximum concurrent transient set + declared allowance"
    );
    let transient_sum: u64 = p
        .linear_memory
        .iter()
        .filter(|t| matches!(t.lifetime, LinearLifetime::Transient(_)))
        .map(|t| t.bytes.evaluate(&binding(1)).expect("term prices"))
        .sum();
    assert!(
        one.linear_transient_peak_bytes < transient_sum,
        "the peak is a maximum over overlap groups ({} B), not their sum ({transient_sum} B) — \
         the phases do not overlap and the allowance, not a sum, covers the size-class mismatch",
        one.linear_transient_peak_bytes
    );
    assert!(
        one.linear_fragmentation_headroom_bytes > 0,
        "the allowance is declared, not implied"
    );

    eprintln!(
        "trainer plan @ ceremony geometry, micro-batch 1: device floor {} B + transient {} B = \
         {} B; linear floor {} B + transient {} B + allowance {} B = {} B; largest logical tensor \
         {} B; largest transfer window {} B",
        one.device_persistent_bytes,
        one.device_transient_peak_bytes,
        one.device_peak_bytes,
        one.linear_persistent_bytes,
        one.linear_transient_peak_bytes,
        one.linear_fragmentation_headroom_bytes,
        one.linear_peak_bytes,
        one.largest_logical_tensor_bytes,
        one.largest_transfer_window_bytes,
    );
}
