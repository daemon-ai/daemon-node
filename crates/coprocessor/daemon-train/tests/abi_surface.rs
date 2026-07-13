// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// Frozen-surface sync (ABI §9): the host `Linker` + phase-legality table (`daemon-train`) must
// agree name-for-name with the guest SDK's `tabi@1` import list (`daemon_train_sdk::TABI_IMPORTS`).
// Extended from Wave 1's phase-table coverage: the vocabulary grows additively (Merge-1 50 + the
// Wave-2 additions), and this test is the tripwire that a new op landed on BOTH sides or neither.

use std::collections::BTreeSet;

use daemon_train::phase::PHASE_TABLE;
use daemon_train_sdk::TABI_IMPORTS;

#[test]
fn phase_table_matches_sdk_import_list_name_for_name() {
    let host: BTreeSet<&str> = PHASE_TABLE.iter().map(|(n, _)| *n).collect();
    let guest: BTreeSet<&str> = TABI_IMPORTS.iter().copied().collect();

    let host_only: Vec<&&str> = host.difference(&guest).collect();
    let guest_only: Vec<&&str> = guest.difference(&host).collect();
    assert!(
        host_only.is_empty() && guest_only.is_empty(),
        "tabi@1 surface drift — host-only: {host_only:?}, guest-only: {guest_only:?}"
    );
}

#[test]
fn frozen_vocabulary_is_the_expected_size() {
    // The frozen v1 vocabulary: Merge-1's 50 + 16 Wave-2 additions. Bumping this is a deliberate,
    // reviewed act (a new op must be added to the SDK extern block, the host Linker, AND the phase
    // table together) — never an accident.
    assert_eq!(TABI_IMPORTS.len(), 66, "frozen tabi@1 vocabulary size");
    assert_eq!(PHASE_TABLE.len(), 66, "host phase table size");
}
