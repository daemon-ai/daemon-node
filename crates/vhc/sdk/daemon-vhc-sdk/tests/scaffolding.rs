// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// The Phase-A migration/claim scaffolding acceptance (refactor §5 A2 item 4; ABI §10.1/§10.2):
// state round-trips in sim through the typed manifest protocol, the descriptor codec is
// canonical-stable, and the SDK-derived claim/manifest match the wire schema the admission
// funnel decodes (§9.1/§6.2).

use daemon_vhc_proto::{blake3_hash, Hash};
use daemon_vhc_sdk::{
    build_manifest, derive_claim, manifest_bytes, migrate, MigrateState, MigrationDescriptor,
    ModuleDecl, OwnedSection, SectionReader,
};

/// A toy migratable module state: a consensus counter + a replica-local cursor.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct Toy {
    counter: u64,
    cursor: Vec<u8>,
}

impl MigrateState for Toy {
    fn snapshot(&self) -> Vec<OwnedSection> {
        vec![
            OwnedSection {
                name: "consensus".into(),
                schema: 1,
                class: 0,
                bytes: self.counter.to_le_bytes().to_vec(),
            },
            OwnedSection {
                name: "data-cursor".into(),
                schema: 1,
                class: 1,
                bytes: self.cursor.clone(),
            },
        ]
    }

    fn restore(&mut self, descriptor: &MigrationDescriptor, reader: &mut dyn SectionReader) -> u32 {
        if descriptor.manifest.schema != 1 {
            return 16; // module-defined incompatibility detail (§10.2)
        }
        for (decl, binding) in descriptor
            .manifest
            .sections
            .iter()
            .zip(&descriptor.sections)
        {
            assert_eq!(decl.name, binding.name, "descriptor order (§10.2)");
            let bytes = reader.read(binding.staging_id);
            // The host verified hashes before staging; the module may re-verify for free.
            assert_eq!(blake3_hash(&bytes), decl.hash);
            match decl.name.as_str() {
                "consensus" => {
                    let Ok(arr) = <[u8; 8]>::try_from(bytes.as_slice()) else {
                        return 17;
                    };
                    self.counter = u64::from_le_bytes(arr);
                }
                "data-cursor" => self.cursor = bytes,
                _ => return 18,
            }
        }
        0
    }
}

#[test]
fn state_round_trips_through_the_manifest_protocol_in_sim() {
    let old = Toy {
        counter: 0xDEAD_BEEF_0042,
        cursor: b"shard-0000@byte-77".to_vec(),
    };
    let mut new = Toy::default();
    let status = migrate::roundtrip(&old, &mut new, Hash([7; 32]), 1);
    assert_eq!(status, 0, "da_migrate contract: 0 = Ready (§10.2)");
    assert_eq!(new, old, "restored state is bit-identical");
}

#[test]
fn incompatible_schema_is_a_typed_module_detail_not_a_panic() {
    let old = Toy::default();
    let mut new = Toy::default();
    // schema 2: the module answers a module-defined detail (≥ 16), the host rolls back (§10.3).
    let status = migrate::roundtrip(&old, &mut new, Hash([7; 32]), 2);
    assert_eq!(status, 16);
}

#[test]
fn manifest_hashes_and_sizes_are_computed_from_the_section_bytes() {
    let sections = Toy {
        counter: 3,
        cursor: vec![9; 100],
    }
    .snapshot();
    let m = build_manifest(Hash([1; 32]), 1, &sections);
    assert_eq!(m.sections.len(), 2);
    assert_eq!(m.sections[0].hash, blake3_hash(&3u64.to_le_bytes()));
    assert_eq!(m.sections[0].size, 8);
    assert_eq!(m.sections[1].size, 100);
    assert_eq!((m.sections[0].class, m.sections[1].class), (0, 1));
}

fn decl() -> ModuleDecl {
    ModuleDecl {
        name: "toy",
        version: "0.0.1",
        abi_minor: 0,
        channels: vec![0],
        host_state_bytes: 5000,  // → 8192 page-rounded
        host_scratch_bytes: 100, // → 4096
        device_state_bytes: 0,
        device_scratch_bytes: 0,
    }
}

#[test]
fn derived_claim_matches_the_section_9_1_wire_schema_and_page_rounds() {
    let bytes = derive_claim(&decl());
    let v: ciborium::value::Value = ciborium::from_reader(bytes.as_slice()).expect("claim cbor");
    let ciborium::value::Value::Map(m) = v else {
        panic!("claim is a map")
    };
    let tier = |name: &str| -> (u64, u64) {
        let ciborium::value::Value::Map(t) = m
            .iter()
            .find(|(k, _)| matches!(k, ciborium::value::Value::Text(s) if s == name))
            .map(|(_, v)| v.clone())
            .expect(name)
        else {
            panic!("{name} is a map")
        };
        let get = |f: &str| -> u64 {
            t.iter()
                .find(|(k, _)| matches!(k, ciborium::value::Value::Text(s) if s == f))
                .and_then(|(_, v)| v.as_integer())
                .map(|n| u64::try_from(i128::from(n)).unwrap())
                .expect(f)
        };
        (get("device"), get("host"))
    };
    assert_eq!(tier("hard_accountable"), (0, 8192));
    assert_eq!(tier("workspace"), (0, 4096));
    // peak = state + scratch, rounded ONCE over the sum (5100 → 8192).
    assert_eq!(tier("declared_peak"), (0, 8192));
    let pressure = m
        .iter()
        .find(|(k, _)| matches!(k, ciborium::value::Value::Text(s) if s == "under_pressure"))
        .map(|(_, v)| v.clone())
        .expect("under_pressure");
    let ciborium::value::Value::Array(steps) = pressure else {
        panic!("under_pressure is an array")
    };
    assert_eq!(steps.len(), 2, "deny buffers, then trap (§9.1)");
}

#[test]
fn derived_manifest_echoes_abi_and_declares_migratable() {
    let bytes = manifest_bytes(&decl());
    let v: ciborium::value::Value = ciborium::from_reader(bytes.as_slice()).expect("cbor");
    let ciborium::value::Value::Map(m) = v else {
        panic!("manifest is a map")
    };
    let get = |name: &str| -> ciborium::value::Value {
        m.iter()
            .find(|(k, _)| matches!(k, ciborium::value::Value::Text(s) if s == name))
            .map(|(_, v)| v.clone())
            .expect(name)
    };
    assert_eq!(
        get("abi").as_integer().map(i128::from),
        Some(i128::from(2u32 << 16))
    );
    assert_eq!(get("migratable"), ciborium::value::Value::Bool(true));
    assert_eq!(
        get("sdk"),
        ciborium::value::Value::Text("daemon-vhc-sdk".into())
    );
}
