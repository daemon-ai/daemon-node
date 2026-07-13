// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Canonical (deterministic) CBOR — RFC 8949 §4.2.
//!
//! This is the consensus-critical codec: envelope hashes, signatures, merkle leaves, and the
//! `[experiment.config]` byte slice fed to `da_build` are all defined over *these* bytes, so every
//! peer must produce byte-identical output for equal values (spec §5.6, §6.1; ABI §6.1). We
//! therefore encode to the RFC 8949 §4.2 deterministic profile:
//!
//! * **Definite lengths** for every string, array, and map.
//! * **Shortest-form** unsigned/negative integer arguments (§4.2.1).
//! * **Shortest-form floats** (§4.2.2): an `f64` is emitted as half (f16) → single (f32) → double,
//!   choosing the smallest width whose value round-trips bit-exactly; every NaN normalizes to the
//!   canonical half `0xf9 7e00`.
//! * **Map keys sorted** by the bytewise lexicographic order of their own deterministic encodings
//!   (§4.2.1).
//!
//! ## Approach (documented per the lane brief)
//!
//! `ciborium` is not itself a canonical encoder (it preserves struct field order and always emits
//! `f64` as eight bytes). We use it only to turn a `serde::Serialize` value into an in-memory
//! [`ciborium::value::Value`] tree, then emit the final bytes with our own writer below — a
//! decode→re-encode canonicalization. The tree walk sorts every map and shortens every scalar, so
//! the output is independent of the authoring representation (TOML field order, whitespace, integer
//! vs. float spelling of a whole number is *not* normalized — CBOR keeps the major type).

use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::error::SwarmProtoError;

/// Serialize `value` to canonical (RFC 8949 §4.2 deterministic) CBOR bytes.
pub fn to_canonical_vec<T: Serialize + ?Sized>(value: &T) -> Result<Vec<u8>, SwarmProtoError> {
    // ciborium gives us the value tree; our writer imposes the deterministic profile.
    let mut scratch = Vec::new();
    ciborium::ser::into_writer(value, &mut scratch)
        .map_err(|e| SwarmProtoError::Codec(format!("serialize: {e}")))?;
    let tree: ciborium::value::Value = ciborium::de::from_reader(scratch.as_slice())
        .map_err(|e| SwarmProtoError::Codec(format!("re-read: {e}")))?;
    let mut out = Vec::with_capacity(scratch.len());
    write_value(&tree, &mut out)?;
    Ok(out)
}

/// Decode canonical CBOR bytes produced by [`to_canonical_vec`] (or any well-formed CBOR).
pub fn from_canonical_slice<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, SwarmProtoError> {
    ciborium::de::from_reader(bytes)
        .map_err(|e| SwarmProtoError::Codec(format!("deserialize: {e}")))
}

fn write_value(value: &ciborium::value::Value, out: &mut Vec<u8>) -> Result<(), SwarmProtoError> {
    use ciborium::value::Value;
    match value {
        Value::Integer(i) => {
            let n: i128 = (*i).into();
            if n >= 0 {
                let arg = u64::try_from(n)
                    .map_err(|_| SwarmProtoError::Codec("uint out of 64-bit range".into()))?;
                write_head(0, arg, out);
            } else {
                let arg = u64::try_from(-1 - n)
                    .map_err(|_| SwarmProtoError::Codec("nint out of 64-bit range".into()))?;
                write_head(1, arg, out);
            }
        }
        Value::Bytes(b) => {
            write_head(2, b.len() as u64, out);
            out.extend_from_slice(b);
        }
        Value::Text(s) => {
            write_head(3, s.len() as u64, out);
            out.extend_from_slice(s.as_bytes());
        }
        Value::Array(items) => {
            write_head(4, items.len() as u64, out);
            for item in items {
                write_value(item, out)?;
            }
        }
        Value::Map(entries) => {
            // Encode each (key, value) pair independently, then sort pairs by the *encoded key*
            // bytes (RFC 8949 §4.2.1 bytewise lexicographic order).
            let mut encoded: Vec<(Vec<u8>, Vec<u8>)> = Vec::with_capacity(entries.len());
            for (k, v) in entries {
                let mut kb = Vec::new();
                write_value(k, &mut kb)?;
                let mut vb = Vec::new();
                write_value(v, &mut vb)?;
                encoded.push((kb, vb));
            }
            encoded.sort_by(|a, b| a.0.cmp(&b.0));
            write_head(5, encoded.len() as u64, out);
            for (kb, vb) in encoded {
                out.extend_from_slice(&kb);
                out.extend_from_slice(&vb);
            }
        }
        Value::Bool(false) => out.push(0xf4),
        Value::Bool(true) => out.push(0xf5),
        Value::Null => out.push(0xf6),
        Value::Float(f) => write_float(*f, out),
        Value::Tag(tag, inner) => {
            write_head(6, *tag, out);
            write_value(inner, out)?;
        }
        other => {
            return Err(SwarmProtoError::Codec(format!(
                "unsupported CBOR value in canonical encoding: {other:?}"
            )));
        }
    }
    Ok(())
}

/// Emit a CBOR head: `major` type (0..=7) with `arg` in the shortest additional-information form.
fn write_head(major: u8, arg: u64, out: &mut Vec<u8>) {
    let mt = major << 5;
    if arg < 24 {
        out.push(mt | (arg as u8));
    } else if arg <= u64::from(u8::MAX) {
        out.push(mt | 24);
        out.push(arg as u8);
    } else if arg <= u64::from(u16::MAX) {
        out.push(mt | 25);
        out.extend_from_slice(&(arg as u16).to_be_bytes());
    } else if arg <= u64::from(u32::MAX) {
        out.push(mt | 26);
        out.extend_from_slice(&(arg as u32).to_be_bytes());
    } else {
        out.push(mt | 27);
        out.extend_from_slice(&arg.to_be_bytes());
    }
}

/// Emit an f64 in the shortest floating-point form that round-trips it (RFC 8949 §4.2.2).
fn write_float(f: f64, out: &mut Vec<u8>) {
    if f.is_nan() {
        // Canonical quiet NaN as a half-float.
        out.extend_from_slice(&[0xf9, 0x7e, 0x00]);
        return;
    }
    let as_f32 = f as f32;
    // Compare bit patterns so ±0.0 and ±∞ are handled without float-equality pitfalls.
    if f64::from(as_f32).to_bits() == f.to_bits() {
        if let Some(half) = f32_to_half_exact(as_f32) {
            out.push(0xf9);
            out.extend_from_slice(&half.to_be_bytes());
            return;
        }
        out.push(0xfa);
        out.extend_from_slice(&as_f32.to_be_bytes());
        return;
    }
    out.push(0xfb);
    out.extend_from_slice(&f.to_be_bytes());
}

/// Convert `value` to IEEE-754 half precision, returning `Some(bits)` only when the conversion is
/// bit-exact (no precision lost). Uses truncation then verifies the round-trip, which is sufficient
/// because an f16-representable value has all discarded low bits already zero.
fn f32_to_half_exact(value: f32) -> Option<u16> {
    let half = f32_to_half_trunc(value);
    if half_to_f32(half).to_bits() == value.to_bits() {
        Some(half)
    } else {
        None
    }
}

fn f32_to_half_trunc(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007f_ffff;
    if exp == 0xff {
        // inf (mant == 0) or NaN.
        return if mant == 0 {
            sign | 0x7c00
        } else {
            sign | 0x7e00
        };
    }
    let unbiased = exp - 127;
    if unbiased > 15 {
        // Overflow: not representable — return inf so the round-trip check rejects it.
        return sign | 0x7c00;
    }
    if unbiased < -24 {
        return sign; // Underflow to (signed) zero.
    }
    if unbiased < -14 {
        // Subnormal half: value = half_mant · 2⁻²⁴, and the f32 significand (with its implicit 1)
        // is `mant_with_implicit · 2^(unbiased-23)`, so half_mant = significand >> (-unbiased - 1).
        // For unbiased ∈ [-24, -15] the shift is in [14, 23] — always < 32 (truncating).
        let mant_with_implicit = mant | 0x0080_0000;
        let shift = (-unbiased - 1) as u32;
        let half_mant = (mant_with_implicit >> shift) as u16;
        return sign | half_mant;
    }
    let half_exp = ((unbiased + 15) as u16) << 10;
    let half_mant = (mant >> 13) as u16;
    sign | half_exp | half_mant
}

fn half_to_f32(half: u16) -> f32 {
    let sign = (u32::from(half) & 0x8000) << 16;
    let exp = (u32::from(half) >> 10) & 0x1f;
    let mant = u32::from(half) & 0x03ff;
    let bits = if exp == 0 {
        if mant == 0 {
            sign // ±0
        } else {
            // Subnormal half → normalized single.
            let mut e = -14i32;
            let mut m = mant;
            while m & 0x0400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x03ff;
            let f32_exp = (e + 127) as u32;
            sign | (f32_exp << 23) | (m << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (mant << 13) // inf / NaN
    } else {
        let f32_exp = exp + 112; // (exp - 15) + 127
        sign | (f32_exp << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}
