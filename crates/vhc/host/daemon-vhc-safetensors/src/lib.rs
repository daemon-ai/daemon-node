// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-train-safetensors` — the swarm-checkpoint ↔ safetensors converter (spec §9).
//!
//! A small, additive library that maps a canonical **state dict** — the module's parameter list in
//! registration order (ABI §6.3), which is the checkpoint tensor order and digest coverage — to and
//! from the `safetensors` on-disk format, **bit-exactly** in both directions. It is the substrate
//! for spec §9's `checkpoints/round-<r>.safetensors` (B3) and for M2's llama-burn reference model
//! (init matched to the guest via a safetensors round-trip).
//!
//! ## Where it lands (single-writer / frozen-surface hygiene)
//!
//! `TrainerBackend::checkpoint_save`/`checkpoint_load` are **frozen** (they carry the blake3-tagged
//! CBOR `CheckpointWire`, `daemon-train/runtime.rs`). This crate does **not** touch them — it is an
//! *export/import alongside*. A caller assembles a [`StateDict`] from the live instance
//! (`Instance::params()` for the ordered `(name, shape)` list + `Instance::param_master(name)` for
//! each fp32 master) and calls [`StateDict::to_safetensors`]; the reverse rebuilds a `StateDict` a
//! caller can feed back into its own registration.
//!
//! ## Order preservation (why we embed `__metadata__["order"]`)
//!
//! `safetensors::serialize` **sorts tensors by `(dtype desc, name)`** on write (safetensors 0.4
//! `prepare`), so the file's physical order is *not* the canonical registration order. Consensus
//! and checkpoint semantics require the exact registration order, so [`StateDict::to_safetensors`]
//! records it in the reserved `__metadata__` map (`order` = names joined by `\n`) and
//! [`StateDict::from_safetensors`] reconstructs from it. A foreign safetensors file with no `order`
//! key falls back to name-sorted order (deterministic).
//!
//! ## Integrity
//!
//! safetensors carries no content hash of its own; the swarm addresses artifacts by blake3 (spec
//! §8/§9). [`blake3_hex`] hashes the serialized bytes — the value that goes in the run manifest /
//! checkpoint record. A safetensors round-trip is bit-exact, so the blake3 of the re-serialized
//! bytes equals the original's.
//!
//! ## Scope (P1)
//!
//! fp32 tensors only (the canonical masters + fp32-exact `replicated` persistents, spec §9); other
//! dtypes are a documented follow-on ([`ConvertError::UnsupportedDtype`]).

#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::error::Error;
use std::fmt;

use safetensors::tensor::{Dtype, SafeTensors, TensorView};

/// The `__metadata__` key under which the canonical registration order is stored (see module docs).
///
/// **Exactly one** metadata key is written: safetensors serializes `__metadata__` as a
/// `serde_json` `HashMap`, whose multi-entry iteration order is non-deterministic — which would make
/// the file bytes (and thus its blake3) unstable and break spec §9's "register only when both
/// checkpointer uploads hash-match". A single key serializes deterministically. Provenance
/// (tokenizer/dataset) lives in the swarm run manifest (`data.rs`), not here.
const ORDER_KEY: &str = "order";
/// The order separator (parameter names are UTF-8 ≤128 bytes and never contain `\n`, ABI §6.3).
const ORDER_SEP: char = '\n';

/// One named fp32 tensor: `(name, shape, row-major data)`.
pub type NamedTensor = (String, Vec<usize>, Vec<f32>);

/// A canonical state dict — the module's parameters in **registration order** (ABI §6.3).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StateDict {
    /// The tensors, in canonical order.
    pub tensors: Vec<NamedTensor>,
}

impl StateDict {
    /// An empty state dict.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a tensor (registration order is the push order).
    pub fn push(&mut self, name: impl Into<String>, shape: Vec<usize>, data: Vec<f32>) {
        self.tensors.push((name.into(), shape, data));
    }

    /// Build from an ordered list of `(name, shape, data)` (a convenience for callers assembling
    /// from `Instance::params()` + `Instance::param_master`).
    #[must_use]
    pub fn from_named(tensors: Vec<NamedTensor>) -> Self {
        Self { tensors }
    }

    /// The canonical name order.
    #[must_use]
    pub fn names(&self) -> Vec<&str> {
        self.tensors.iter().map(|(n, _, _)| n.as_str()).collect()
    }

    /// Serialize to safetensors bytes, preserving canonical order via `__metadata__` (module docs).
    ///
    /// # Errors
    ///
    /// [`ConvertError::ShapeMismatch`] if a tensor's declared shape's element count ≠ its data len;
    /// [`ConvertError::Safetensors`] on a serializer error.
    pub fn to_safetensors(&self) -> Result<Vec<u8>, ConvertError> {
        // Validate + materialize LE byte buffers (owned; the TensorViews borrow them).
        let mut byte_bufs: Vec<Vec<u8>> = Vec::with_capacity(self.tensors.len());
        for (name, shape, data) in &self.tensors {
            let numel: usize = shape.iter().product();
            if numel != data.len() {
                return Err(ConvertError::ShapeMismatch {
                    name: name.clone(),
                    numel,
                    data_len: data.len(),
                });
            }
            let mut buf = Vec::with_capacity(data.len() * 4);
            for &v in data {
                buf.extend_from_slice(&v.to_le_bytes());
            }
            byte_bufs.push(buf);
        }

        let mut views: Vec<(String, TensorView<'_>)> = Vec::with_capacity(self.tensors.len());
        for ((name, shape, _), buf) in self.tensors.iter().zip(byte_bufs.iter()) {
            let tv = TensorView::new(Dtype::F32, shape.clone(), buf)
                .map_err(|e| ConvertError::Safetensors(e.to_string()))?;
            views.push((name.clone(), tv));
        }

        // Exactly one key ⇒ deterministic header bytes ⇒ stable blake3 (see ORDER_KEY docs).
        let mut meta = HashMap::new();
        meta.insert(
            ORDER_KEY.to_string(),
            self.names().join(&ORDER_SEP.to_string()),
        );

        safetensors::serialize(views, &Some(meta))
            .map_err(|e| ConvertError::Safetensors(e.to_string()))
    }

    /// Parse safetensors bytes back into a canonical [`StateDict`] (fp32 only, spec §9).
    ///
    /// Order is taken from `__metadata__["order"]` if present (our writer always sets it), else the
    /// file's names sorted (deterministic) for foreign files.
    ///
    /// # Errors
    ///
    /// [`ConvertError::Safetensors`] on a malformed buffer or a missing named tensor;
    /// [`ConvertError::UnsupportedDtype`] if any listed tensor is not fp32.
    pub fn from_safetensors(bytes: &[u8]) -> Result<Self, ConvertError> {
        let (_, meta) = SafeTensors::read_metadata(bytes)
            .map_err(|e| ConvertError::Safetensors(e.to_string()))?;
        let st = SafeTensors::deserialize(bytes)
            .map_err(|e| ConvertError::Safetensors(e.to_string()))?;

        let order: Vec<String> = match meta.metadata().as_ref().and_then(|m| m.get(ORDER_KEY)) {
            Some(joined) => joined
                .split(ORDER_SEP)
                .filter(|s| !s.is_empty())
                .map(String::from)
                .collect(),
            None => {
                let mut names: Vec<String> =
                    st.names().into_iter().map(|n| n.to_string()).collect();
                names.sort();
                names
            }
        };

        let mut tensors = Vec::with_capacity(order.len());
        for name in order {
            let tv = st
                .tensor(&name)
                .map_err(|e| ConvertError::Safetensors(format!("{name}: {e}")))?;
            if tv.dtype() != Dtype::F32 {
                return Err(ConvertError::UnsupportedDtype {
                    name,
                    dtype: format!("{:?}", tv.dtype()),
                });
            }
            let shape = tv.shape().to_vec();
            let data: Vec<f32> = tv
                .data()
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            tensors.push((name, shape, data));
        }
        Ok(Self { tensors })
    }
}

/// The lowercase-hex blake3 of the serialized bytes — the integrity value the run manifest /
/// checkpoint record carries (spec §8/§9). A bit-exact round-trip reproduces it exactly.
#[must_use]
pub fn blake3_hex(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

/// Conversion errors.
#[derive(Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ConvertError {
    /// A tensor's declared shape element count did not match its data length.
    ShapeMismatch {
        /// The tensor name.
        name: String,
        /// The element count implied by the shape.
        numel: usize,
        /// The actual data length.
        data_len: usize,
    },
    /// A listed tensor was not fp32 (P1 stores fp32 masters only; other dtypes are a follow-on).
    UnsupportedDtype {
        /// The tensor name.
        name: String,
        /// The offending dtype.
        dtype: String,
    },
    /// The underlying `safetensors` (de)serializer failed.
    Safetensors(String),
}

impl fmt::Display for ConvertError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeMismatch {
                name,
                numel,
                data_len,
            } => write!(
                f,
                "tensor {name}: shape implies {numel} elements but data has {data_len}"
            ),
            Self::UnsupportedDtype { name, dtype } => {
                write!(
                    f,
                    "tensor {name}: unsupported dtype {dtype} (fp32 only in P1)"
                )
            }
            Self::Safetensors(detail) => write!(f, "safetensors error: {detail}"),
        }
    }
}

impl Error for ConvertError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> StateDict {
        // Deliberately NOT alphabetical, and mixed sizes, so the round-trip proves order is
        // preserved despite safetensors' internal (dtype, name) sort.
        let mut sd = StateDict::new();
        sd.push("tok.weight", vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        sd.push(
            "l0.wq",
            vec![2, 2],
            vec![-1.5, 0.25, f32::MIN_POSITIVE, 7.0],
        );
        sd.push("norm.weight", vec![2], vec![1.0, 1.0]);
        sd
    }

    #[test]
    fn round_trip_is_bit_exact_and_order_preserving() {
        let sd = sample();
        let bytes = sd.to_safetensors().unwrap();
        let back = StateDict::from_safetensors(&bytes).unwrap();
        assert_eq!(
            back, sd,
            "names, shapes, order, and fp32 data are bit-exact"
        );
        // The recovered order is the canonical (push) order, not safetensors' sorted order.
        assert_eq!(back.names(), vec!["tok.weight", "l0.wq", "norm.weight"]);
        // Re-serializing reproduces identical bytes ⇒ blake3 integrity is stable.
        assert_eq!(back.to_safetensors().unwrap(), bytes);
        assert_eq!(
            blake3_hex(&bytes),
            blake3_hex(&sd.to_safetensors().unwrap())
        );
    }

    #[test]
    fn serialization_is_deterministic() {
        // Spec §9: two elected checkpointers must upload byte-identical files (hash-match gate). The
        // same state dict must serialize to identical bytes across independent calls.
        let sd = sample();
        assert_eq!(sd.to_safetensors().unwrap(), sd.to_safetensors().unwrap());
    }

    #[test]
    fn negative_and_special_floats_survive() {
        let mut sd = StateDict::new();
        sd.push(
            "x",
            vec![4],
            vec![-0.0, f32::MIN, f32::MAX, std::f32::consts::PI],
        );
        let back = StateDict::from_safetensors(&sd.to_safetensors().unwrap()).unwrap();
        // Compare raw bits so -0.0 vs 0.0 is caught.
        for ((_, _, a), (_, _, b)) in sd.tensors.iter().zip(back.tensors.iter()) {
            for (x, y) in a.iter().zip(b.iter()) {
                assert_eq!(x.to_bits(), y.to_bits());
            }
        }
    }

    #[test]
    fn shape_mismatch_rejected() {
        let mut sd = StateDict::new();
        sd.push("bad", vec![2, 2], vec![1.0, 2.0]); // 4 != 2
        assert!(matches!(
            sd.to_safetensors(),
            Err(ConvertError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn garbage_bytes_rejected() {
        assert!(StateDict::from_safetensors(b"not safetensors").is_err());
    }

    /// The **real 160M canonical param layout** (names/shapes/order) survives a round-trip. Uses the
    /// tiny default config (small tensors) but the identical layout machinery the 160M preset uses.
    #[test]
    fn canonical_llama_layout_round_trips() {
        use daemon_train_sdk::models::TinyLlamaCfg;
        let cfg = TinyLlamaCfg::default();
        let layout = cfg.canonical_param_layout();
        let mut sd = StateDict::new();
        for (i, (name, shape)) in layout.iter().enumerate() {
            let numel: usize = shape.iter().map(|&d| d as usize).product();
            // Deterministic per-tensor values so the round-trip check is meaningful.
            let data: Vec<f32> = (0..numel)
                .map(|j| (i * 131 + j) as f32 * 0.5 - 3.0)
                .collect();
            sd.push(
                name.clone(),
                shape.iter().map(|&d| d as usize).collect(),
                data,
            );
        }
        let back = StateDict::from_safetensors(&sd.to_safetensors().unwrap()).unwrap();
        assert_eq!(back, sd);
        assert_eq!(back.names().first(), Some(&"tok.weight"));
        assert_eq!(back.names().last(), Some(&"norm.weight"));
    }
}
