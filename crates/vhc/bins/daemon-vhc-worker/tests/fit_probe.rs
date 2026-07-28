// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// The fit-probe mode's end-to-end gate (`[RC-15]`): the REAL worker binary — the same executable
// whose sealed revision identity the verdict names — is spawned in `DAEMON_TRAIN_FIT_PROBE` mode
// over a probe directory authored through the testkit seam, at the tiny t2 parity geometry on
// the CPU lane, and must produce a GREEN content-addressed FitVerdict whose key members are the
// artifacts this test can independently derive.
//
// What this pins, end to end:
//  1. The DIRECTORY CONTRACT holds across the process boundary: the orchestrator's authored
//     inputs (config-derived open frame, staged batches, drive spec as pure data) drive the
//     worker's opaque consumer to the module's committed voice.
//  2. The ADMITTED PATH is the probe path: the worker composes its estimate through the real
//     funnel (provisioned dev-authority profile, measured CPU selection), sizes the enforced
//     budget from it, and the run completes UNDER that budget — the doctrine's "estimate sizes,
//     probe decides" in one artifact.
//  3. The verdict is CONTENT-ADDRESSED and honest: the key's module hash is the module's blake3,
//     the measured peak is nonzero and inside the budget, and the file name carries the key
//     digest the verdict re-derives.

// Dev/test harness: spawns the worker binary and touches the filesystem (the env/spawn bans
// target the shipped node paths).
#![allow(clippy::disallowed_methods)]

use daemon_vhc_proto::PeerId;
use daemon_vhc_resource::probe as contract;
use daemon_vhc_resource::{FitOutcome, FitVerdict};

/// The suite's development authority (`[PC-12]`): pure data — nothing signs with it — acceptable
/// only because BOTH the provisioned owner policy and the probe's requirements name it.
fn development_authority() -> PeerId {
    PeerId(*blake3::hash(b"fit-probe-suite/development-authority").as_bytes())
}

/// The provisioned profile home (`DAEMON_VHC_PROFILE_DIR`): the development-authority CPU profile
/// set, built against the revision record the REAL worker binary exports about itself.
fn provisioned_profile_dir() -> tempfile::TempDir {
    let export = tempfile::tempdir().expect("revision export dir");
    let status = std::process::Command::new(env!("CARGO_BIN_EXE_daemon-vhc-worker"))
        .env("DAEMON_TRAIN_REVISION_OUT", export.path())
        .stdout(std::process::Stdio::null())
        .status()
        .expect("run the worker's revision export");
    assert!(status.success(), "worker revision export failed");
    let bytes = std::fs::read(export.path().join("revision-cpu.cbor"))
        .expect("the worker exported its CPU-lane revision record");
    let record: daemon_vhc_resource::BackendImplementationRevision =
        daemon_vhc_proto::from_canonical_slice(&bytes).expect("the exported revision decodes");
    let set = daemon_vhc_resource::test_support::development_provisioned_profiles(
        &record,
        development_authority(),
    );
    let dir = tempfile::tempdir().expect("profile dir");
    daemon_vhc_resource::provision::write(dir.path(), &set).expect("write provisioned set");
    dir
}

/// The trainer role's execution requirements, derived from the module's own assessment over the
/// SAME config the probe runs (the plan is a function of the configuration), naming the suite's
/// development authority.
fn trainer_requirements(wasm: &[u8], config: &[u8]) -> daemon_vhc_proto::RoleExecutionRequirements {
    use daemon_vhc_host::run::{author_execution, GrantPolicy, RoleAuthoringInput};
    let engine = daemon_vhc_host::Worker::new(daemon_vhc_host::EngineConfig::default())
        .expect("authoring engine");
    let authored = author_execution(
        &engine,
        vec![RoleAuthoringInput {
            role: "trainer",
            wasm,
            config,
            grants: &[],
            allowed_backend_classes: vec!["cpu".to_string()],
            profile_certification: daemon_vhc_proto::ProfileCertificationRequirements {
                accepted_development_authorities: vec![development_authority()],
                ..Default::default()
            },
            minima: daemon_vhc_proto::HardwareIndependentMinima::default(),
            grant: GrantPolicy::DomainMinimum,
        }],
    )
    .expect("derive the trainer execution requirements");
    authored
        .for_role("trainer")
        .expect("the authored set carries the trainer role")
}

#[test]
fn the_probe_mode_produces_a_green_content_addressed_verdict_on_the_cpu_lane() {
    let wasm = std::fs::read(daemon_vhc_guest_build::built_module_path("tiny_llama"))
        .expect("the tiny_llama guest builds");
    let peer = daemon_vhc_testkit::fit_probe::probe_peer();
    // The tiny t2 parity geometry: two real training steps, micro-batch 1 — minutes on a CPU
    // lane, with every seam of the real drive (seed init to the pinned root, train, commit).
    let config = daemon_vhc_testkit::genesis_run::trainer_config(&peer, &[peer], 2, 1, 4);
    let requirements = trainer_requirements(&wasm, &config);

    let probe_dir = tempfile::tempdir().expect("probe dir");
    daemon_vhc_testkit::fit_probe::write_trainer_probe_dir(
        probe_dir.path(),
        &wasm,
        &config,
        &requirements,
        600,
    )
    .expect("author the probe directory");

    let profiles = provisioned_profile_dir();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_daemon-vhc-worker"))
        .env("DAEMON_TRAIN_FIT_PROBE", probe_dir.path())
        .env(daemon_vhc_resource::PROFILE_DIR_ENV, profiles.path())
        // The directed CPU lane: deterministic on every host, and the direction the join engine
        // reads to build the roomy real-model budgets (`engine_for_join`'s own rule).
        .env("DAEMON_TRAIN_BACKEND", "cpu")
        // The owner's CPU-admitting lane (the acceptance nodes' switch): the launch trainer lane
        // requires a GPU, and this gate's subject is the probe seam, not the lane floor.
        .env("DAEMON_VHC_LANE_GPU_OPTIONAL", "1")
        .output()
        .expect("run the worker's fit-probe mode");
    assert!(
        output.status.success(),
        "the probe mode failed:\n--- stdout ---\n{}\n--- stderr ---\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    // Exactly one verdict, named by its key digest, decoding to a GREEN outcome under the budget.
    let verdicts: Vec<std::path::PathBuf> = std::fs::read_dir(probe_dir.path())
        .expect("read the probe dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with(contract::VERDICT_FILE_PREFIX))
        })
        .collect();
    assert_eq!(
        verdicts.len(),
        1,
        "one probe records one verdict: {verdicts:?}"
    );
    let verdict: FitVerdict = daemon_vhc_proto::from_canonical_slice(
        &std::fs::read(&verdicts[0]).expect("read the verdict"),
    )
    .expect("the verdict decodes canonically");

    assert!(
        verdict.is_green(),
        "the tiny geometry fits its own admitted budget: {:?}",
        verdict.outcome
    );
    assert_eq!(
        verdict.key.module_hash.0,
        *blake3::hash(&wasm).as_bytes(),
        "the key names the module that ran"
    );
    assert!(
        verdict.key.budget_bytes > 0,
        "the enforced budget is the admitted figure, never zero"
    );
    let FitOutcome::Fits {
        measured_peak_bytes,
    } = &verdict.outcome
    else {
        unreachable!("asserted green above")
    };
    assert!(
        *measured_peak_bytes > 0 && *measured_peak_bytes <= verdict.key.budget_bytes,
        "the measured peak ({measured_peak_bytes} B) is real and inside the enforced budget \
         ({} B)",
        verdict.key.budget_bytes
    );
    // The file name is the key's memo address — the content addressing a store looks up by.
    let key_digest = verdict.key.digest().expect("address the key");
    assert_eq!(
        verdicts[0].file_name().and_then(|n| n.to_str()),
        Some(
            format!(
                "{}{}.cbor",
                contract::VERDICT_FILE_PREFIX,
                key_digest.to_hex()
            )
            .as_str()
        ),
        "the verdict file is named by the probe key's digest"
    );
}
