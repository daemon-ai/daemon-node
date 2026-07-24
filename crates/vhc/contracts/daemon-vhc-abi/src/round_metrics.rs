// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The reserved **round-outcome metric contract** — how a trainer guest surfaces a peer's
//! per-round outcome (the post-ingest det digest plus the barrier's committed / ingested /
//! stalled bookkeeping) through the host METRIC ABI (`sys@2::emit_metric`, name + `f64` pairs)
//! instead of through any module-authored control frame.
//!
//! ## Why this lives here
//!
//! `emit_metric` is a host ABI capability: the guest emits opaque `(name, f64)` telemetry, the
//! host relays it. That makes the metric plane an OPACITY-SAFE seam for the digest — the host
//! never decodes a module payload frame to obtain it, it only recognizes a reserved metric-name
//! namespace and reads the numbers back. This crate is the shared, dependency-free contract
//! ground both sides of the wasm boundary pin, so the guest's emission and the host's recognition
//! are defined once, in lock-step.
//!
//! This is a metric-NAMING contract, not round message vocabulary: it carries no coordinator
//! schema and decodes no frame. The `[tag, round, bytes]` publish vocabulary (the tag-4 digest
//! voice) stays entirely in the guest SDK; the host still never touches it.
//!
//! ## The wire (reserved names + `f64` encoding)
//!
//! Every metric in the group is named `vhc.round.<round>.<field>` (the [`RESERVED_PREFIX`]
//! namespace), where `<round>` is the decimal round id and `<field>` is one of:
//!
//! - `committed` — the number of payloads listed in the round's record (`f64` of a `u32`);
//! - `ingested`  — the number of payloads this peer folded at the barrier (`f64` of a `u32`);
//! - `stalled`   — `1.0` if this peer stalled at the barrier (caught up late), else `0.0`;
//! - `digest0`..`digest3` — the four little-endian `u32` words of the 16-byte det digest, each an
//!   `f64` of a `u32`.
//!
//! An `f64` carries 53 mantissa bits, so every `u32` word rides losslessly. The host emits a round
//! outcome once ALL SEVEN fields of a round's group have arrived (the three counts + the four
//! digest words). Gating on the full group — rather than on the digest words alone — makes the
//! accumulator fully order-independent: it never emits a partial or defaulted outcome, and stays
//! robust to reordering, interleaving with ordinary telemetry, and partially-delivered groups.

use std::collections::BTreeMap;

/// The reserved metric-name namespace. A metric whose name begins with this prefix is part of the
/// round-outcome carrier and MUST NOT be authored as ordinary telemetry.
pub const RESERVED_PREFIX: &str = "vhc.round.";

/// One field of the round-outcome group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoundMetricField {
    /// The number of payloads listed in the round's record.
    Committed,
    /// The number of payloads this peer folded at the barrier.
    Ingested,
    /// Whether this peer stalled at the barrier (caught up late).
    Stalled,
    /// One of the four little-endian `u32` words of the 16-byte digest (`0..=3`).
    DigestWord(u8),
}

impl RoundMetricField {
    /// The field's stable name suffix (no `.`, so the round id is the only dotted split).
    #[must_use]
    pub fn suffix(self) -> String {
        match self {
            RoundMetricField::Committed => "committed".to_string(),
            RoundMetricField::Ingested => "ingested".to_string(),
            RoundMetricField::Stalled => "stalled".to_string(),
            RoundMetricField::DigestWord(w) => format!("digest{w}"),
        }
    }

    /// Parse a field suffix back. `None` for anything outside the reserved vocabulary.
    #[must_use]
    pub fn from_suffix(s: &str) -> Option<Self> {
        match s {
            "committed" => Some(RoundMetricField::Committed),
            "ingested" => Some(RoundMetricField::Ingested),
            "stalled" => Some(RoundMetricField::Stalled),
            _ => {
                let w = s.strip_prefix("digest")?;
                let w: u8 = w.parse().ok()?;
                (w < 4).then_some(RoundMetricField::DigestWord(w))
            }
        }
    }
}

/// The number of little-endian `u32` words the 16-byte digest is carried in.
pub const DIGEST_WORDS: usize = 4;

/// Parse a metric name into `(round, field)`. Returns `None` for any name outside the reserved
/// [`RESERVED_PREFIX`] namespace — the host relays those as ordinary telemetry, untouched.
#[must_use]
pub fn parse_metric_name(name: &str) -> Option<(u64, RoundMetricField)> {
    let rest = name.strip_prefix(RESERVED_PREFIX)?;
    // `<round>.<field>`: the round is pure digits, the field suffix carries no `.`, so the first
    // separator splits them unambiguously.
    let (round, field) = rest.split_once('.')?;
    let round: u64 = round.parse().ok()?;
    let field = RoundMetricField::from_suffix(field)?;
    Some((round, field))
}

/// Encode a `u32` as the exact `f64` the metric ABI carries (lossless; 32 < 53 mantissa bits).
#[must_use]
pub fn encode_u32(v: u32) -> f64 {
    f64::from(v)
}

/// Decode a metric `f64` back to a `u32` word (round-to-nearest, clamped to the `u32` range and to
/// non-negative). A non-finite carrier reads as `0` — the metric ABI already drops non-finite
/// values, so this is defensive only.
#[must_use]
pub fn decode_u32(v: f64) -> u32 {
    if v.is_finite() && v >= 0.0 {
        let r = v.round();
        if r >= f64::from(u32::MAX) {
            u32::MAX
        } else {
            // Safe: bounded to `[0, u32::MAX]` above.
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            {
                r as u32
            }
        }
    } else {
        0
    }
}

/// A peer's per-round outcome as carried over the metric plane: the barrier bookkeeping plus the
/// post-ingest det digest. Mirrors the fields of the worker-protocol `Event::RoundOutcome` and the
/// app-wire `VhcEvent::RoundOutcome`, but this type is transport-agnostic ABI ground.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RoundOutcome {
    /// The round this outcome reports.
    pub round: u64,
    /// The number of payloads committed to the round's record.
    pub committed: u32,
    /// The number of payloads this peer ingested at the barrier.
    pub ingested: u32,
    /// Whether this peer stalled at the barrier (caught up late).
    pub stalled: bool,
    /// The post-ingest det-state digest (§5.6).
    pub digest: [u8; 16],
}

impl RoundOutcome {
    /// The `(name, value)` metric pairs the guest emits for this outcome, in the canonical emission
    /// order: `committed`, `ingested`, `stalled`, then the four digest words. The host gates on the
    /// FULL group (all seven), so it is order-independent regardless; this order is merely stable.
    #[must_use]
    pub fn metric_pairs(&self) -> Vec<(String, f64)> {
        let name =
            |field: RoundMetricField| format!("{RESERVED_PREFIX}{}.{}", self.round, field.suffix());
        let mut pairs = Vec::with_capacity(3 + DIGEST_WORDS);
        pairs.push((
            name(RoundMetricField::Committed),
            encode_u32(self.committed),
        ));
        pairs.push((name(RoundMetricField::Ingested), encode_u32(self.ingested)));
        pairs.push((
            name(RoundMetricField::Stalled),
            if self.stalled { 1.0 } else { 0.0 },
        ));
        for w in 0..DIGEST_WORDS {
            let word = u32::from_le_bytes([
                self.digest[w * 4],
                self.digest[w * 4 + 1],
                self.digest[w * 4 + 2],
                self.digest[w * 4 + 3],
            ]);
            #[allow(clippy::cast_possible_truncation)]
            pairs.push((
                name(RoundMetricField::DigestWord(w as u8)),
                encode_u32(word),
            ));
        }
        pairs
    }
}

/// A partial round-outcome group being assembled from arriving metrics.
#[derive(Debug, Default, Clone)]
struct PartialGroup {
    committed: Option<u32>,
    ingested: Option<u32>,
    stalled: Option<bool>,
    digest: [Option<u32>; DIGEST_WORDS],
}

impl PartialGroup {
    /// The completed [`RoundOutcome`] once ALL seven fields (the three counts + the four digest
    /// words) have arrived; `None` while any field is still outstanding.
    fn complete(&self, round: u64) -> Option<RoundOutcome> {
        let mut digest = [0u8; 16];
        for w in 0..DIGEST_WORDS {
            let word = self.digest[w]?;
            digest[w * 4..w * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        Some(RoundOutcome {
            round,
            committed: self.committed?,
            ingested: self.ingested?,
            stalled: self.stalled?,
            digest,
        })
    }
}

/// The host-side accumulator that folds arriving `(name, f64)` metrics into complete
/// [`RoundOutcome`]s. Robust to partial groups, reordering, and interleaving with ordinary
/// telemetry: it keys partial groups by round and fires exactly once per round, the moment all
/// four digest words have arrived.
#[derive(Debug, Default)]
pub struct RoundOutcomeAccumulator {
    groups: BTreeMap<u64, PartialGroup>,
}

impl RoundOutcomeAccumulator {
    /// A fresh accumulator (one per role incarnation — a module switch mints a new session
    /// instance and hence a new accumulator).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Fold one metric. Returns `Some(outcome)` exactly when this metric completes a round's full
    /// group (all three counts + all four digest words); the group is then retired. Returns `None`
    /// for a metric outside the reserved namespace (the caller relays those as ordinary telemetry)
    /// or for a still-incomplete group.
    pub fn observe(&mut self, name: &str, value: f64) -> Option<RoundOutcome> {
        let (round, field) = parse_metric_name(name)?;
        let group = self.groups.entry(round).or_default();
        match field {
            RoundMetricField::Committed => group.committed = Some(decode_u32(value)),
            RoundMetricField::Ingested => group.ingested = Some(decode_u32(value)),
            RoundMetricField::Stalled => group.stalled = Some(value != 0.0),
            RoundMetricField::DigestWord(w) => group.digest[w as usize] = Some(decode_u32(value)),
        }
        let outcome = group.complete(round)?;
        // Complete: retire the group and surface the outcome (gating on the full group means it is
        // never partial or defaulted, and the fold is order-independent).
        self.groups.remove(&round);
        Some(outcome)
    }

    /// Whether the metric name belongs to the reserved round-outcome namespace (a convenience for
    /// callers deciding whether to relay a metric as ordinary telemetry).
    #[must_use]
    pub fn is_reserved(name: &str) -> bool {
        name.starts_with(RESERVED_PREFIX)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_round_trip_through_parse() {
        let outcome = RoundOutcome {
            round: 128,
            committed: 3,
            ingested: 3,
            stalled: false,
            digest: [0xAB; 16],
        };
        for (name, _v) in outcome.metric_pairs() {
            assert!(RoundOutcomeAccumulator::is_reserved(&name));
            let (round, _field) = parse_metric_name(&name).expect("reserved name parses");
            assert_eq!(round, 128);
        }
        // A large round id (the name must stay well under the 128-byte metric-name ceiling).
        let big = RoundOutcome {
            round: u64::MAX,
            ..outcome
        };
        for (name, _v) in big.metric_pairs() {
            assert!(
                name.len() <= 128,
                "reserved metric name fits the ABI ceiling"
            );
            assert!(parse_metric_name(&name).is_some());
        }
    }

    #[test]
    fn ordinary_telemetry_is_not_reserved() {
        assert!(!RoundOutcomeAccumulator::is_reserved("grad_norm"));
        assert!(parse_metric_name("grad_norm").is_none());
        assert!(parse_metric_name("replay_decisions").is_none());
        // A superficially-similar but malformed name never parses.
        assert!(parse_metric_name("vhc.round.notanumber.digest0").is_none());
        assert!(parse_metric_name("vhc.round.3.unknownfield").is_none());
        assert!(parse_metric_name("vhc.round.3.digest9").is_none());
    }

    #[test]
    fn accumulator_fires_once_on_digest_completeness() {
        let mut acc = RoundOutcomeAccumulator::new();
        let digest = {
            let mut d = [0u8; 16];
            for (i, b) in d.iter_mut().enumerate() {
                *b = (i as u8).wrapping_mul(7).wrapping_add(1);
            }
            d
        };
        let outcome = RoundOutcome {
            round: 5,
            committed: 4,
            ingested: 4,
            stalled: true,
            digest,
        };
        let pairs = outcome.metric_pairs();
        let mut fired = Vec::new();
        for (name, value) in &pairs {
            if let Some(o) = acc.observe(name, *value) {
                fired.push(o);
            }
        }
        assert_eq!(
            fired,
            vec![outcome],
            "fires exactly once with the exact outcome"
        );
        // The group was retired: replaying the last (digest) metric alone never re-fires.
        let (last_name, last_v) = pairs.last().unwrap();
        assert!(acc.observe(last_name, *last_v).is_none());
    }

    #[test]
    fn accumulator_is_robust_to_reordering_and_interleaving() {
        let mut acc = RoundOutcomeAccumulator::new();
        let a = RoundOutcome {
            round: 1,
            committed: 2,
            ingested: 2,
            stalled: false,
            digest: [1u8; 16],
        };
        let b = RoundOutcome {
            round: 2,
            committed: 2,
            ingested: 1,
            stalled: true,
            digest: [2u8; 16],
        };
        // Interleave both rounds' metrics AND an ordinary metric, digest words in reverse order.
        let mut stream: Vec<(String, f64)> = Vec::new();
        let mut ap = a.metric_pairs();
        let mut bp = b.metric_pairs();
        ap.reverse();
        bp.reverse();
        stream.push(("grad_norm".to_string(), 0.5));
        for i in 0..ap.len().max(bp.len()) {
            if let Some(p) = ap.get(i) {
                stream.push(p.clone());
            }
            if let Some(p) = bp.get(i) {
                stream.push(p.clone());
            }
        }
        let mut fired = Vec::new();
        for (name, value) in &stream {
            if let Some(o) = acc.observe(name, *value) {
                fired.push(o);
            }
        }
        fired.sort_by_key(|o| o.round);
        assert_eq!(fired, vec![a, b]);
    }

    #[test]
    fn u32_words_ride_losslessly() {
        for v in [0u32, 1, 255, 65_535, u32::MAX / 2, u32::MAX - 1, u32::MAX] {
            assert_eq!(decode_u32(encode_u32(v)), v);
        }
    }
}
