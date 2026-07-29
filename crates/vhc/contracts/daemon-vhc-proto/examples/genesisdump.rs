//! Frozen-genesis dump for run diagnosis: opens a `SignedEnvelope` wire file and prints the
//! envelope's admission-relevant fields (roster, trust, profile-certification requirements,
//! coordinator timers) so an authored genesis is auditable without re-deriving it.
//!
//! Usage: `genesisdump <envelope.cbor>`

use daemon_vhc_proto::from_canonical_slice;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: genesisdump <envelope.cbor>");
    // An operator diagnostic over an operator-named wire file: no containment boundary applies
    // (the disallowed-methods rule guards the engine's contained-exec roots, not dump tools).
    #[allow(clippy::disallowed_methods)]
    let bytes = std::fs::read(&path).expect("read envelope");
    let signed: daemon_vhc_proto::SignedEnvelope =
        from_canonical_slice(&bytes).expect("decode SignedEnvelope");
    let frozen =
        daemon_vhc_proto::FrozenGenesis::open(signed.bytes, signed.signature, signed.signer)
            .expect("open frozen genesis");
    println!("run id: {}", frozen.run_id().to_hex());
    let env = frozen.decode().expect("decode envelope");
    println!("{env:#?}");
}
