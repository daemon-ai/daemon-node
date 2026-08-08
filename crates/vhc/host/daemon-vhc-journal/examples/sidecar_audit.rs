//! Gate A measurement: the sidecar data-classification audit over real production journals.
//!
//! For each journal directory given, scans every local segment and reports, per read-back
//! `kind` (ABI §6.4 / §8.3 tag 2): record counts, inline vs sidecar-referenced, sidecar byte
//! volume, and whether each referenced sidecar file is present locally. This is the evidence
//! for which kinds are reconstruction-required vs replay-skipped (the plan's Gate A audit).
//!
//! Usage: `cargo run -p daemon-vhc-journal --example sidecar_audit -- <journal-dir>...`

// Trusted-straggler anchor (clippy.toml convention): this is an operator diagnostic over
// operator-supplied journal paths, not an attacker-influenced path — ContainedRoot does
// not apply.
#![allow(clippy::disallowed_methods)]

use std::collections::BTreeMap;
use std::path::Path;

use daemon_vhc_journal::{scan_file, Body};

#[derive(Default)]
struct KindStat {
    records: u64,
    inline: u64,
    sidecar: u64,
    sidecar_bytes: u64,
    sidecar_present: u64,
    sidecar_missing: u64,
}

fn main() {
    let dirs: Vec<String> = std::env::args().skip(1).collect();
    assert!(!dirs.is_empty(), "usage: sidecar_audit <journal-dir>...");

    for dir in &dirs {
        let dir = Path::new(dir);
        let mut segments: Vec<_> = std::fs::read_dir(dir)
            .expect("read journal dir")
            .filter_map(|e| {
                let p = e.expect("dir entry").path();
                (p.extension().is_some_and(|x| x == "dvhcjrn")).then_some(p)
            })
            .collect();
        segments.sort();

        let mut per_kind: BTreeMap<u64, KindStat> = BTreeMap::new();
        let mut tags: BTreeMap<u8, u64> = BTreeMap::new();
        let mut instantiations = 0u64;

        for seg in &segments {
            let scan = match scan_file(seg) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("  !! {} unreadable: {e}", seg.display());
                    continue;
                }
            };
            for rec in &scan.records {
                *tags.entry(rec.body.tag()).or_default() += 1;
                match &rec.body {
                    Body::ReadBack(rb) => {
                        let st = per_kind.entry(rb.kind).or_default();
                        st.records += 1;
                        if let Some(sc) = &rb.sidecar {
                            st.sidecar += 1;
                            st.sidecar_bytes += sc.size;
                            let f = dir
                                .join("sidecars")
                                .join(format!("{}.dvhcsc", hex(&sc.hash.0)));
                            if f.exists() {
                                st.sidecar_present += 1;
                            } else {
                                st.sidecar_missing += 1;
                            }
                        } else {
                            st.inline += 1;
                        }
                    }
                    Body::Instantiation(_) => instantiations += 1,
                    _ => {}
                }
            }
        }

        let on_disk: Vec<_> = std::fs::read_dir(dir.join("sidecars"))
            .map(|rd| rd.filter_map(|e| e.ok().map(|e| e.path())).collect())
            .unwrap_or_default();
        let on_disk_bytes: u64 = on_disk
            .iter()
            .filter_map(|p| p.metadata().ok().map(|m| m.len()))
            .sum();

        println!("\n== {}", dir.display());
        println!(
            "   {} local segments; record tags: {:?}; tag-13 instantiations: {instantiations}",
            segments.len(),
            tags
        );
        println!(
            "   sidecar files on disk: {} ({} bytes)",
            on_disk.len(),
            on_disk_bytes
        );
        for (kind, st) in &per_kind {
            println!(
                "   readback kind {kind}: {} records ({} inline, {} sidecar-ref; {} bytes \
                 referenced; {} present / {} missing locally)",
                st.records,
                st.inline,
                st.sidecar,
                st.sidecar_bytes,
                st.sidecar_present,
                st.sidecar_missing
            );
        }
        if per_kind.is_empty() {
            println!("   no read-back records in the local window");
        }
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
