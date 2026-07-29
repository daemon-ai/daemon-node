//! Checkpoint-document dump for live-run diagnosis: manifest + section names/kinds/sizes.
//!
//! Usage: `ckptdump <doc.bin>` — prints each section a checkpoint document carries, so a
//! restore/migrate refusal can be compared against the consuming guest's expectations.

fn main() {
    let path = std::env::args().nth(1).expect("usage: ckptdump <doc.bin>");
    let bytes = std::fs::read(&path).expect("read doc");
    let (manifest, sections) =
        daemon_vhc_proto::det_state::decode_checkpoint_doc(&bytes).expect("decode checkpoint doc");
    println!("manifest: {manifest:?}");
    for section in &sections {
        match section {
            daemon_vhc_proto::det_state::CkptDocSection::Inline(name, data) => {
                println!("inline  {name:<12} {} bytes", data.len());
            }
            daemon_vhc_proto::det_state::CkptDocSection::ByRef(name, fref) => {
                println!(
                    "by-ref  {name:<12} fold={} len={}",
                    fref.fold.to_hex(),
                    fref.byte_len
                );
            }
        }
    }
}
