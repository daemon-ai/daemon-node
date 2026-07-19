// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Chunk-addressed corpus helpers: decoding the `register_chunks` descriptor and verifying a
//! chunk-aligned covering span against the registered (fold-committed) chunk map — the
//! sub-resource verification rule shared by the `data@2` imports and the completion pump.

/// Decode the `register_chunks` descriptor — canonical CBOR
/// `[chunk_size, token_count, byte_len, [c_0, …]]` (each `c_i` a 32-byte chunk blake3) — into a
/// well-formed [`daemon_vhc_proto::ChunkMap`]. Malformed shape/geometry is described (the
/// import traps it typed — a bad descriptor is a module authoring fault, not a store fault).
pub(crate) fn decode_chunk_descriptor(desc: &[u8]) -> Result<daemon_vhc_proto::ChunkMap, String> {
    let v: ciborium::value::Value =
        ciborium::de::from_reader(desc).map_err(|e| format!("descriptor is not CBOR: {e}"))?;
    let ciborium::value::Value::Array(parts) = v else {
        return Err("descriptor is not a CBOR array".into());
    };
    let uint = |i: usize, name: &str| -> Result<u64, String> {
        parts
            .get(i)
            .and_then(ciborium::value::Value::as_integer)
            .and_then(|n| u64::try_from(i128::from(n)).ok())
            .ok_or_else(|| format!("descriptor `{name}` is not a uint"))
    };
    let chunk_size = uint(0, "chunk_size")?;
    let token_count = uint(1, "token_count")?;
    let byte_len = uint(2, "byte_len")?;
    let Some(ciborium::value::Value::Array(hashes)) = parts.get(3) else {
        return Err("descriptor chunk-hash list is not an array".into());
    };
    let mut chunk_hashes = Vec::with_capacity(hashes.len());
    for (i, h) in hashes.iter().enumerate() {
        let ciborium::value::Value::Bytes(b) = h else {
            return Err(format!("chunk hash {i} is not a byte string"));
        };
        let arr: [u8; 32] = b
            .as_slice()
            .try_into()
            .map_err(|_| format!("chunk hash {i} is not 32 bytes"))?;
        chunk_hashes.push(daemon_vhc_proto::Hash(arr));
    }
    let map = daemon_vhc_proto::ChunkMap {
        chunk_size,
        token_count,
        byte_len,
        chunk_hashes,
    };
    if !map.is_well_formed() {
        return Err(format!(
            "degenerate chunk geometry (chunk_size {chunk_size}, byte_len {byte_len}, {} \
             chunk hashes)",
            map.chunk_hashes.len()
        ));
    }
    Ok(map)
}

/// Verify a chunk-aligned covering span against the registered chunk map: split `bytes` at
/// `chunk_size`, and every chunk's blake3 must equal the registered hash at its absolute index
/// (`span_off / chunk_size + i`). Returns the verified bytes, or the first mismatch described.
pub(crate) fn verify_covering_span(
    map: &daemon_vhc_proto::ChunkMap,
    span_off: u64,
    bytes: Vec<u8>,
) -> Result<Vec<u8>, String> {
    let base = span_off / map.chunk_size;
    let mut cursor = 0usize;
    let mut index = base;
    while cursor < bytes.len() {
        let expected_len = map.chunk_len(index) as usize;
        let Some(expected) = map.chunk_hashes.get(index as usize) else {
            return Err(format!("span reaches past the chunk list (chunk {index})"));
        };
        let end = cursor + expected_len;
        if end > bytes.len() {
            return Err(format!(
                "span truncates chunk {index} ({} of {expected_len} bytes)",
                bytes.len() - cursor
            ));
        }
        if blake3::hash(&bytes[cursor..end]).as_bytes() != &expected.0 {
            return Err(format!(
                "chunk {index} does not hash to the registered chunk hash"
            ));
        }
        cursor = end;
        index += 1;
    }
    Ok(bytes)
}
