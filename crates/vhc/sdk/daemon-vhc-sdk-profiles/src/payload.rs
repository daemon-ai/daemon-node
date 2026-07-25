// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The **range-addressable committed-update layout** — the profile's payload bytes, laid out so a
//! consumer can read exactly the rows one fold window needs out of a host buffer instead of
//! materializing the container.
//!
//! # What this is NOT
//!
//! It is not a contract change. A committed update is still ONE opaque blob with `blake3(whole
//! bytes)` as its identity; it still rides `payload_put`/`payload_get` unchanged; record entries,
//! [`daemon_vhc_sdk_consensus::Committed`] minting, the coordinator's availability check and the
//! replay oracle all see exactly what they saw before. Payload *format* is module policy
//! (architecture §8 — the vhc never parses payloads), and this module is that policy.
//!
//! # Why the layout changed
//!
//! [SF-R3] (streaming det fold §5.4) requires the fold to read per-(peer, parameter) payload
//! section rows through ranged `read_into` while the payload stays in a host buffer. That needs
//! byte offsets that are *arithmetic*, not the result of walking a CBOR value tree: the retired
//! layout was a CBOR `[Section…]` list whose every element is length-prefixed, so locating a row
//! meant decoding the container — the very thing [SF-R3] exists to avoid.
//!
//! So the bytes are a fixed header followed by the per-window section pairs, in the fold walk's
//! window order:
//!
//! ```text
//! [ 40-byte header ][ values_0 ‖ indices_0 ][ values_1 ‖ indices_1 ] …
//! ```
//!
//! Every span is computable from `(parameter numels, state chunk size, chunk, topk, bits)` — all of
//! which both peers hold from the run's pinned config — so a consumer computes the byte range of
//! any window's rows with no decode at all, and the header is a CROSS-CHECK (a producer whose
//! geometry disagrees with the run is a typed refusal, not a mis-decode).
//!
//! Two consequences fall out of laying the bytes out at all:
//!
//! - **Chunk-local indices ride their own width.** An index is bounded by the profile chunk
//!   (11 bits at `chunk = 1536`), so it is packed at [`daemon_vhc_det::index_bits`] bits instead of
//!   riding as one f32 — the 3–4× the retired layout spent per index, which at the ceremony profile
//!   was the bulk of a ~210 MB container.
//! - **The values rows are byte-identical** to what the retired container carried for the same rows
//!   (`absmax_pack` output verbatim), so the decoded values, the decoded indices, and therefore the
//!   folded masters and the det digests are unchanged. Only the container's bytes — and hence its
//!   blake3, the record entry, and the journal — move.

use daemon_vhc_sdk_consensus::fold_walk::{windows, Window};

use crate::{bytes_of, indices_of, tensor_data, Section, SparseLocoCfg};

/// The committed-update layout format major. `1` was the retired CBOR `[Section…]` container; `2`
/// is this range-addressable layout. A consumer that reads a different major refuses typed.
pub const PAYLOAD_FORMAT: u32 = 2;

/// The layout's fixed header magic.
const MAGIC: [u8; 4] = *b"VHCP";

/// The header's byte length (fixed — the first section starts here).
pub const HEADER_BYTES: u64 = 40;

/// The byte width of one state element (state families and θ are f32-le by contract).
const ELEM_BYTES: u64 = 4;

/// A byte range inside a committed payload: `(offset, len)`.
pub type PayloadSpan = (u64, u64);

/// The absmax-packed byte stride of one `[k]` value row at `bits` — the frozen §6.6 record layout,
/// from the kernel that writes it ([`daemon_vhc_det::absmax_row_bytes`]).
#[must_use]
pub fn packed_row_stride(k: usize, bits: u32) -> usize {
    daemon_vhc_det::absmax_row_bytes(k, bits)
}

/// The geometry of one run's committed-update layout: the fold-window schedule the sections are
/// ordered by, plus the arithmetic that turns a window into byte spans.
///
/// Built from the run's own pinned inputs on both sides — the producer to append its rows, the
/// consumer to range-read them — so the two agree by construction rather than by negotiation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PayloadLayout {
    chunk: usize,
    topk: usize,
    bits: u32,
    window_size: u64,
    params: usize,
    /// The fold-window schedule (window ordinal ≡ section-pair ordinal).
    schedule: Vec<Window>,
    /// `(values offset, values len, indices offset, indices len)` per window, in window order.
    spans: Vec<(u64, u64, u64, u64)>,
    /// The container's total byte length (header + every section).
    total_len: u64,
}

impl PayloadLayout {
    /// Derive the layout for a parameter registration order at the run's state chunk size.
    ///
    /// # Errors
    /// A `String` when the profile geometry is degenerate (`chunk` does not divide a numel, the
    /// window is not a multiple of the chunk's byte width, `topk > chunk`) — the same geometry
    /// rules the fold walks refuse at construction.
    pub fn new(cfg: &SparseLocoCfg, numels: &[usize], window_size: u64) -> Result<Self, String> {
        let chunk = cfg.chunk as usize;
        let topk = cfg.topk as usize;
        if chunk == 0 || numels.is_empty() {
            return Err("a committed-update layout needs a non-empty layout and chunk > 0".into());
        }
        if topk == 0 || topk > chunk {
            return Err(format!("topk {topk} must be in 1..={chunk}"));
        }
        for (i, &numel) in numels.iter().enumerate() {
            if numel == 0 || !numel.is_multiple_of(chunk) {
                return Err(format!(
                    "profile chunk {chunk} does not divide parameter {i}'s numel {numel}"
                ));
            }
        }
        let width = chunk as u64 * ELEM_BYTES;
        if window_size == 0 || !window_size.is_multiple_of(width) {
            return Err(format!(
                "window size {window_size} must be a non-zero multiple of the profile chunk byte \
                 width {width}"
            ));
        }
        let numels_u64: Vec<u64> = numels.iter().map(|&n| n as u64).collect();
        let schedule = windows(&numels_u64, window_size);
        let stride = packed_row_stride(topk, cfg.bits) as u64;
        let mut spans = Vec::with_capacity(schedule.len());
        let mut cursor = HEADER_BYTES;
        for w in &schedule {
            let rows = w.len / width;
            let vlen = rows * stride;
            let ilen = daemon_vhc_det::packed_index_len((rows * topk as u64) as usize, chunk)
                .map_err(|e| format!("index geometry: {e:?}"))? as u64;
            spans.push((cursor, vlen, cursor + vlen, ilen));
            cursor += vlen + ilen;
        }
        Ok(Self {
            chunk,
            topk,
            bits: cfg.bits,
            window_size,
            params: numels.len(),
            schedule,
            spans,
            total_len: cursor,
        })
    }

    /// The fold-window schedule the sections are ordered by.
    #[must_use]
    pub fn schedule(&self) -> &[Window] {
        &self.schedule
    }

    /// The container's total byte length.
    #[must_use]
    pub fn total_len(&self) -> u64 {
        self.total_len
    }

    /// The compression rows one window covers.
    #[must_use]
    pub fn window_rows(&self, ordinal: u64) -> usize {
        let w = self.schedule[usize::try_from(ordinal).expect("ordinal fits usize")];
        usize::try_from(w.len / (self.chunk as u64 * ELEM_BYTES)).expect("rows fit usize")
    }

    /// The byte span of one window's packed VALUE rows.
    #[must_use]
    pub fn values_span(&self, ordinal: u64) -> PayloadSpan {
        let (off, len, _, _) = self.spans[usize::try_from(ordinal).expect("ordinal fits usize")];
        (off, len)
    }

    /// The byte span of one window's packed chunk-local INDEX rows.
    #[must_use]
    pub fn indices_span(&self, ordinal: u64) -> PayloadSpan {
        let (_, _, off, len) = self.spans[usize::try_from(ordinal).expect("ordinal fits usize")];
        (off, len)
    }

    /// The container's fixed header: the geometry a consumer cross-checks against the run's own
    /// pinned config before it computes a single offset.
    #[must_use]
    pub fn header(&self) -> Vec<u8> {
        let mut h = Vec::with_capacity(HEADER_BYTES as usize);
        h.extend_from_slice(&MAGIC);
        h.extend_from_slice(&PAYLOAD_FORMAT.to_le_bytes());
        h.extend_from_slice(&(self.chunk as u32).to_le_bytes());
        h.extend_from_slice(&(self.topk as u32).to_le_bytes());
        h.extend_from_slice(&self.bits.to_le_bytes());
        h.extend_from_slice(&(self.params as u32).to_le_bytes());
        h.extend_from_slice(&(self.schedule.len() as u32).to_le_bytes());
        h.extend_from_slice(&0u32.to_le_bytes()); // reserved
        h.extend_from_slice(&self.window_size.to_le_bytes());
        debug_assert_eq!(h.len() as u64, HEADER_BYTES);
        h
    }

    /// Cross-check a fetched container's header against this layout: the magic, the format major,
    /// and every geometry field. A producer that compressed under a different density, value width,
    /// window size or layout is a typed refusal here — never a silent mis-decode of its rows.
    ///
    /// # Errors
    /// A `String` naming the first disagreement.
    pub fn check_header(&self, header: &[u8]) -> Result<(), String> {
        if header.len() < HEADER_BYTES as usize {
            return Err(format!(
                "committed payload header is {} bytes, the layout needs {HEADER_BYTES}",
                header.len()
            ));
        }
        let u32_at = |off: usize| {
            u32::from_le_bytes([
                header[off],
                header[off + 1],
                header[off + 2],
                header[off + 3],
            ])
        };
        if header[0..4] != MAGIC {
            return Err("committed payload does not carry the layout magic".into());
        }
        let got = (
            u32_at(4),
            u32_at(8),
            u32_at(12),
            u32_at(16),
            u32_at(20),
            u32_at(24),
            u64::from_le_bytes([
                header[32], header[33], header[34], header[35], header[36], header[37], header[38],
                header[39],
            ]),
        );
        let want = (
            PAYLOAD_FORMAT,
            self.chunk as u32,
            self.topk as u32,
            self.bits,
            self.params as u32,
            self.schedule.len() as u32,
            self.window_size,
        );
        if got != want {
            return Err(format!(
                "committed payload geometry (format {}, chunk {}, topk {}, bits {}, params {}, \
                 windows {}, window {}) disagrees with the run's contract (format {}, chunk {}, \
                 topk {}, bits {}, params {}, windows {}, window {})",
                got.0,
                got.1,
                got.2,
                got.3,
                got.4,
                got.5,
                got.6,
                want.0,
                want.1,
                want.2,
                want.3,
                want.4,
                want.5,
                want.6
            ));
        }
        Ok(())
    }

    /// Re-express a **resident** payload ([`Section`] list — the reference profile's output) in this
    /// layout, whole, in one allocation.
    ///
    /// The definition bridge between the two forms: the parity suites use it to prove the streamed
    /// producer emits exactly these bytes, the golden capture uses it to reconstruct a guest's own
    /// committed container natively, and a harness uses it to synthesize a committed set. It is the
    /// only place that materializes a whole container by design — nothing on the guest's path does.
    ///
    /// # Errors
    /// A `String` when the section list does not match the layout.
    pub fn encode_sections(&self, sections: &[Section]) -> Result<Vec<u8>, String> {
        let stride = packed_row_stride(self.topk, self.bits);
        let mut packed: Vec<Vec<u8>> = Vec::with_capacity(self.params);
        let mut indices: Vec<Vec<u32>> = Vec::with_capacity(self.params);
        for i in 0..self.params {
            let p = bytes_of(tensor_data(sections, 2 * i)?);
            let idx = indices_of(tensor_data(sections, 2 * i + 1)?);
            packed.push(p);
            indices.push(idx);
        }
        let mut out = self.header();
        out.reserve(usize::try_from(self.total_len).unwrap_or(0));
        for w in &self.schedule {
            let rows = self.window_rows(w.ordinal);
            let row0 = usize::try_from(w.param_off / (self.chunk as u64 * ELEM_BYTES))
                .expect("row offset fits usize");
            let param = w.param as usize;
            let vals = packed[param]
                .get(row0 * stride..(row0 + rows) * stride)
                .ok_or_else(|| format!("param {param}: packed section is short of the layout"))?;
            let idx = indices[param]
                .get(row0 * self.topk..(row0 + rows) * self.topk)
                .ok_or_else(|| format!("param {param}: index section is short of the layout"))?;
            out.extend_from_slice(vals);
            out.extend_from_slice(
                &daemon_vhc_det::pack_chunk_indices(idx, self.chunk)
                    .map_err(|e| format!("pack indices: {e:?}"))?,
            );
        }
        if out.len() as u64 != self.total_len {
            return Err(format!(
                "encoded container is {} bytes, the layout says {}",
                out.len(),
                self.total_len
            ));
        }
        Ok(out)
    }

    /// Decode one window's packed VALUE rows (length-checked against the window's row count, so a
    /// short or over-long read is typed rather than silently mis-framed).
    ///
    /// # Errors
    /// A `String` on a mis-sized span or a det-kernel refusal.
    pub fn decode_values(&self, ordinal: u64, bytes: &[u8]) -> Result<Vec<f32>, String> {
        let (_, len) = self.values_span(ordinal);
        if bytes.len() as u64 != len {
            return Err(format!(
                "window {ordinal}: value rows are {} bytes, the layout needs {len}",
                bytes.len()
            ));
        }
        daemon_vhc_det::det_absmax_unpack(bytes, self.topk, self.bits)
            .map_err(|e| format!("window {ordinal} values: {e:?}"))
    }

    /// Decode one window's packed chunk-local INDEX rows.
    ///
    /// # Errors
    /// A `String` on a mis-sized span or an index outside the chunk domain.
    pub fn decode_indices(&self, ordinal: u64, bytes: &[u8]) -> Result<Vec<u32>, String> {
        let count = self.window_rows(ordinal) * self.topk;
        daemon_vhc_det::unpack_chunk_indices(bytes, self.chunk, count)
            .map_err(|e| format!("window {ordinal} indices: {e:?}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> SparseLocoCfg {
        SparseLocoCfg {
            h: 1,
            ef_decay: 0.95,
            chunk: 16,
            topk: 4,
            bits: 2,
            outer_alpha: 1.0,
            clip: true,
        }
    }

    /// The spans tile the container exactly: header, then every window's value + index rows, in
    /// fold order, with no gap and no overlap.
    #[test]
    fn spans_tile_the_container_in_fold_order() {
        let numels = [64usize, 32, 16];
        let layout = PayloadLayout::new(&cfg(), &numels, 64).unwrap();
        let mut cursor = HEADER_BYTES;
        for w in layout.schedule() {
            let (voff, vlen) = layout.values_span(w.ordinal);
            let (ioff, ilen) = layout.indices_span(w.ordinal);
            assert_eq!(
                voff, cursor,
                "values start where the previous section ended"
            );
            assert_eq!(ioff, voff + vlen, "indices follow their values");
            cursor = ioff + ilen;
            let rows = layout.window_rows(w.ordinal);
            assert_eq!(vlen, (rows * packed_row_stride(4, 2)) as u64);
        }
        assert_eq!(cursor, layout.total_len());
        assert_eq!(layout.header().len() as u64, HEADER_BYTES);
    }

    /// The header is a cross-check: a container authored under a different density, value width,
    /// window size or layout refuses instead of mis-decoding.
    #[test]
    fn the_header_refuses_a_container_from_a_different_geometry() {
        let numels = [64usize, 32];
        let layout = PayloadLayout::new(&cfg(), &numels, 64).unwrap();
        layout.check_header(&layout.header()).unwrap();

        let mut other = cfg();
        other.topk = 2;
        let alien = PayloadLayout::new(&other, &numels, 64).unwrap();
        assert!(layout.check_header(&alien.header()).is_err());

        let alien = PayloadLayout::new(&cfg(), &numels, 128).unwrap();
        assert!(layout.check_header(&alien.header()).is_err());

        let alien = PayloadLayout::new(&cfg(), &[64usize], 64).unwrap();
        assert!(layout.check_header(&alien.header()).is_err());

        let mut torn = layout.header();
        torn[0] ^= 0xff;
        assert!(layout.check_header(&torn).is_err());
        assert!(layout.check_header(&layout.header()[..8]).is_err());
    }

    /// Degenerate geometry is a typed construction refusal (the fold walks' rules, applied where
    /// the bytes are laid out).
    #[test]
    fn degenerate_geometry_is_refused() {
        assert!(
            PayloadLayout::new(&cfg(), &[100], 64).is_err(),
            "chunk ∤ numel"
        );
        assert!(
            PayloadLayout::new(&cfg(), &[64], 60).is_err(),
            "window ∤ chunk width"
        );
        assert!(PayloadLayout::new(&cfg(), &[], 64).is_err(), "empty layout");
        let mut wide = cfg();
        wide.topk = 32;
        assert!(
            PayloadLayout::new(&wide, &[64], 64).is_err(),
            "topk > chunk"
        );
    }
}
