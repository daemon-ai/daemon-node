// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `Authority` — trust topology as policy (architecture §4.2; refactor §8/D1).
//!
//! The question "is this record authoritative?" is answered here, in guest code, by an SDK
//! abstraction both roles link: the coordinator module uses its `Authority` to *produce* records;
//! worker modules use the same `Authority` to *accept* them and mint [`crate::committed::Committed`]
//! (the wrapper type consensus transitions require). This module is the **vocabulary D2 consumes**:
//! the [`Authority`] trait, its declared safety/liveness [`AuthorityContract`], the launch
//! implementations [`SingleKey`] and [`ThresholdKeys`], and the typed [`AuthorityConfig`] the
//! envelope's opaque `authority` section (D0 landed it as raw CBOR the host never interprets)
//! decodes into.
//!
//! ## What D1 owns here (stable surface for D2)
//!
//! - [`Authority`] — the trait every trust topology implements. `authorize()` turns a record's
//!   signed preimage + the presented signatures into an [`Authorized`] token; that token is the
//!   *only* thing [`crate::committed::Committed::mint`] accepts, so "consensus state is a pure
//!   function of authority-verified, hash-verified inputs" is a compile-time property of cooperative
//!   module code (architecture §4.2 — the mint's soundness is a wasm/SDK property, distinct from the
//!   protocol-level agreement-on-which-records-exist property `Authority` also governs).
//! - [`AuthorityContract`] — every impl ships its declared safety/liveness assumptions (quorum
//!   intersection, fault threshold, synchrony, finality, and the signer-transfer / reconfiguration
//!   rule of architecture §4.4). The conformance surface for an `Authority` is adversarial, not
//!   merely functional (architecture §4.2); the contract is what those adversarial suites assert
//!   against.
//! - [`SingleKey`] / [`ThresholdKeys`] — the two launch topologies (architecture §4.2 table). D2's
//!   `ElectedLeader` / `ChainAnchored` land later "with zero host/ABI change", which is the test that
//!   this abstraction is placed correctly — so this trait/`AuthorityConfig` pair is deliberately the
//!   whole extension seam.
//! - [`AuthorityConfig`] — the typed config the variants decode from. **The signing oracle uses
//!   channel declarations, not guest-selected message classes** (the ratified decision): the config
//!   names the *authoritative records channel* a record must arrive on, never a message class the
//!   guest promotes. [`AuthorityConfig::records_channel`] surfaces that channel to the mint.
//!
//! ## Note for the D2 track (contract shapes you build against)
//!
//! D2 stubs against [`SingleKey`] until this crate merges (this track merges first). Every
//! contract-shape decision D2 needs is fixed here:
//! - construct an authority from the envelope: [`AuthorityConfig::decode`] over
//!   `GenesisEnvelope::authority`, then [`AuthorityConfig::authority`] → `Box<dyn Authority>` (or
//!   the borrowing [`AuthorityConfig::topology`] for a zero-alloc match);
//! - the coordinator produces records by signing the record preimage with its `Authority` key(s)
//!   and presenting [`RecordSig`]s; the worker calls [`Authority::authorize`] on `(preimage, sigs)`
//!   to obtain the [`Authorized`] token the `Committed` mint requires;
//! - the authoritative records channel is [`AuthorityConfig::records_channel`] (default
//!   [`DEFAULT_RECORDS_CHANNEL`] = the control channel, ABI §6.2 channel 0);
//! - equivocation / conflicting-history detection is out of the mint's scope (it is the protocol
//!   property, architecture §4.2) — the mint only decides "authoritative under this topology".

use daemon_vhc_proto::{verify_sig, PeerId, Signature, VerifyOutcome};

/// The authoritative records channel a record must arrive on when the config does not name one
/// (ABI §6.2 channel 0 = `control`, authoritative). The signing oracle derives the delivery class
/// from this **channel declaration**, never from a guest-selected message class.
pub const DEFAULT_RECORDS_CHANNEL: u32 = 0;

// -- the declared contract (architecture §4.2 "every Authority ships a declared contract") ---------

/// How many faulty (equivocating / unavailable / compromised) signers a topology tolerates before
/// its safety or liveness guarantee is void.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaultThreshold {
    /// No fault tolerance: a single signer is the whole authority. A duplicated or leaked key can
    /// produce valid conflicting histories — **detectable** (architecture §4.3 evidence), not
    /// **preventable** ([`SingleKey`]).
    None,
    /// Byzantine-fault-tolerant up to `f` faulty signers out of `n`, requiring an `m`-of-`n`
    /// quorum with `m = n - f` and quorum intersection `2m > n` for safety ([`ThresholdKeys`]).
    Threshold {
        /// Total signers in the set (`n`).
        n: u32,
        /// Required signatures (`m`); safety needs `2m > n`.
        m: u32,
    },
}

/// The finality a topology's records carry once authorized.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Finality {
    /// A record is final the instant it authorizes (no probabilistic reversion); reversing a
    /// *committed* transition is protocol-level recovery under the reconfiguration rule, never a
    /// local operation (architecture §5.4).
    Immediate,
}

/// How signing authority transfers between epochs — the signer-transfer protocol of architecture
/// §4.4 (split-brain prevention, fencing the old signer, sequence continuity).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reconfiguration {
    /// Authority is a single key; a standby is a journal-replicated warm spare and transfer is a
    /// key-custody problem (fence the old signer via a signed epoch-fence record or lease expiry).
    /// A leaked key breaks safety silently — the reason [`ThresholdKeys`] exists.
    SingleSigner,
    /// Authority is an `m`-of-`n` set; signer transfer and compromise tolerance are **protocol**
    /// properties (quorum reconfiguration), not key-custody properties (architecture §4.4).
    QuorumReconfigurable,
}

/// The declared safety/liveness contract an [`Authority`] ships (architecture §4.2). This is the
/// standing acceptance surface for the adversarial `Authority` suite (partitions, equivocation,
/// withheld records, conflicting valid-looking histories).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AuthorityContract {
    /// A stable identifier for the topology (`"single-key"` / `"threshold-keys"`).
    pub name: &'static str,
    /// The fault model the safety guarantee holds under.
    pub fault_threshold: FaultThreshold,
    /// Finality of an authorized record.
    pub finality: Finality,
    /// The signer-transfer / reconfiguration rule (architecture §4.4).
    pub reconfiguration: Reconfiguration,
    /// A one-line prose statement of the safety assumption (quorum intersection / identity /
    /// custody), for logs and the conformance report.
    pub safety: &'static str,
    /// A one-line prose statement of the liveness assumption (synchrony / availability).
    pub liveness: &'static str,
}

// -- the trait + the authorization token -----------------------------------------------------------

/// One presented `(signer, signature)` over a record's signed preimage. A coordinator producing an
/// `m`-of-`n` record presents `m` of these; a `SingleKey` coordinator presents one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecordSig {
    /// The purported signer (ed25519 public key).
    pub signer: PeerId,
    /// The ed25519 signature over the record preimage bytes.
    pub sig: Signature,
}

/// Why an authorization failed. Deliberately distinguishes "structurally garbage" from
/// "well-formed but insufficient" so an adversarial suite (and a running module) can branch on the
/// difference (matching the tri-state [`VerifyOutcome`] intent).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum AuthError {
    /// No signature was presented at all.
    NotSigned,
    /// A `SingleKey` record was signed by an identity other than the envelope-named coordinator.
    WrongSigner {
        /// The identity that actually signed.
        got: PeerId,
    },
    /// A presented signature does not verify over the record preimage (a well-formed-but-invalid
    /// signature — the classic downgrade/forgery attempt).
    BadSignature {
        /// The signer whose signature failed to verify.
        signer: PeerId,
    },
    /// A presented signer is not a member of the envelope-named keyset ([`ThresholdKeys`]).
    UnknownSigner {
        /// The non-member identity.
        signer: PeerId,
    },
    /// The same member's signature was presented more than once (cannot count toward the quorum
    /// twice — an inflation attempt).
    DuplicateSigner {
        /// The doubly-presented member.
        signer: PeerId,
    },
    /// Not enough distinct valid member signatures to meet the `m`-of-`n` threshold.
    InsufficientSignatures {
        /// Distinct valid member signatures presented.
        have: u32,
        /// The threshold `m` required.
        need: u32,
    },
    /// A public key or signature was structurally malformed (wrong length / not on-curve).
    Malformed,
    /// The typed [`AuthorityConfig`] could not be decoded from the opaque envelope value.
    Config {
        /// A human-readable decode reason.
        reason: String,
    },
}

impl core::fmt::Display for AuthError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotSigned => write!(f, "record carried no signature"),
            Self::WrongSigner { got } => {
                write!(
                    f,
                    "record signed by {} (not the coordinator identity)",
                    got.to_hex()
                )
            }
            Self::BadSignature { signer } => {
                write!(
                    f,
                    "signature by {} does not verify over the record",
                    signer.to_hex()
                )
            }
            Self::UnknownSigner { signer } => {
                write!(
                    f,
                    "signer {} is not in the authority keyset",
                    signer.to_hex()
                )
            }
            Self::DuplicateSigner { signer } => {
                write!(f, "signer {} presented more than once", signer.to_hex())
            }
            Self::InsufficientSignatures { have, need } => {
                write!(f, "only {have} of {need} required signatures verified")
            }
            Self::Malformed => write!(f, "malformed public key or signature"),
            Self::Config { reason } => write!(f, "authority config: {reason}"),
        }
    }
}

impl std::error::Error for AuthError {}

/// The proof that a record carries sufficient authority under some [`Authority`] — the **only**
/// input [`crate::committed::Committed::mint`] accepts (architecture §4.2). Minted by
/// [`Authority::authorize`] on the signature path, or by [`Authorized::from_authoritative_channel`]
/// on the host-delivery path (the frame was signature-verified above the pump on a declared
/// authoritative channel — the bridge / mixed-fleet cell-6 topology, where the host verifies the
/// coordinator's frame before the guest ever sees it).
///
/// A cooperative module can only obtain this token by going through an authority check; the type is
/// otherwise opaque. (A *malicious* module can ignore the SDK entirely — what bounds it is the
/// sandbox, grants, budgets, and attribution-by-replay, exactly as architecture §4.2 states; the
/// token makes the committed-input discipline a compile-time property for cooperative authors, not a
/// runtime enforcement against adversaries.)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Authorized {
    channel: u32,
}

impl Authorized {
    /// The channel this authorization is scoped to (the authoritative records channel — a channel
    /// declaration, never a message class).
    #[must_use]
    pub fn channel(&self) -> u32 {
        self.channel
    }

    /// The host-delivery path: the frame arrived on a **declared authoritative channel** and was
    /// signature-verified above the pump (ABI §12.1, `daemon-vhc-session::v2_attach`) before
    /// delivery, so the in-guest re-verification is delegated to that host mechanism. This is the
    /// bridge / mixed-fleet cell-6 topology (native coordinator, frames verified host-side). D2's
    /// wasm coordinator mints records in-guest and threads a real [`Authority::authorize`] token
    /// instead.
    #[must_use]
    pub fn from_authoritative_channel(channel: u32) -> Self {
        Self { channel }
    }
}

/// A run's trust topology as policy (architecture §4.2). Both roles link one; the coordinator
/// produces records under it, the worker accepts them under it.
pub trait Authority {
    /// The declared safety/liveness contract (architecture §4.2 — "every Authority implementation
    /// ships a declared contract"). Its `name` also tags journal/log lines.
    fn contract(&self) -> AuthorityContract;

    /// Decide whether a record carries sufficient authority: verify the presented [`RecordSig`]s
    /// over the record's signed `preimage` bytes against this topology's rule. On success returns
    /// the [`Authorized`] token scoped to `records_channel` (the authoritative channel the record
    /// is expected on). On failure returns the typed [`AuthError`].
    ///
    /// # Errors
    /// The applicable [`AuthError`] variant (wrong/unknown/duplicate signer, bad signature,
    /// insufficient signatures, malformed key/sig, or no signature at all).
    fn authorize(
        &self,
        records_channel: u32,
        preimage: &[u8],
        sigs: &[RecordSig],
    ) -> Result<Authorized, AuthError>;
}

// -- SingleKey (launch) ----------------------------------------------------------------------------

/// `SingleKey` — records are valid when signed by the envelope-named coordinator identity
/// (architecture §4.2). This **formalizes today's implicit trust**: the launch topology is a
/// publisher-designated coordinator identity with journal-replicated warm standbys (architecture
/// §4.4). Detectability over preventability: a duplicated or leaked key can produce valid
/// conflicting histories, exposed via equivocation evidence (architecture §4.3/§10), not prevented.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SingleKey {
    /// The envelope-named coordinator identity.
    pub coordinator: PeerId,
}

impl SingleKey {
    /// The launch-topology coordinator identity.
    #[must_use]
    pub fn new(coordinator: PeerId) -> Self {
        Self { coordinator }
    }
}

impl Authority for SingleKey {
    fn contract(&self) -> AuthorityContract {
        AuthorityContract {
            name: "single-key",
            fault_threshold: FaultThreshold::None,
            finality: Finality::Immediate,
            reconfiguration: Reconfiguration::SingleSigner,
            safety: "records authoritative iff signed by the one envelope-named coordinator \
                     identity; a leaked/duplicated key yields valid conflicting histories \
                     (detectable via equivocation evidence, not preventable)",
            liveness: "the designated signer is available; a journal-replicated standby resumes \
                       state and takes over via a signed epoch-fence (architecture §4.4)",
        }
    }

    fn authorize(
        &self,
        records_channel: u32,
        preimage: &[u8],
        sigs: &[RecordSig],
    ) -> Result<Authorized, AuthError> {
        let first = sigs.first().ok_or(AuthError::NotSigned)?;
        if first.signer != self.coordinator {
            return Err(AuthError::WrongSigner { got: first.signer });
        }
        match verify_sig(&first.signer.0, &first.sig.0, preimage) {
            VerifyOutcome::Valid => Ok(Authorized::from_authoritative_channel(records_channel)),
            VerifyOutcome::Invalid => Err(AuthError::BadSignature {
                signer: first.signer,
            }),
            VerifyOutcome::Malformed => Err(AuthError::Malformed),
        }
    }
}

// -- ThresholdKeys (launch) ------------------------------------------------------------------------

/// `ThresholdKeys(m, n)` — records are valid when carrying `m` valid signatures from the
/// envelope-named set of `n` identities (architecture §4.2). This makes signer transfer and
/// compromise tolerance **protocol** properties instead of key-custody properties (architecture
/// §4.4): safety survives up to `n - m` faulty signers, and quorum intersection (`2m > n`) prevents
/// two disjoint quorums from authorizing conflicting records.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThresholdKeys {
    /// The envelope-named signer set (`n` identities), order-insensitive.
    members: Vec<PeerId>,
    /// The required number of valid distinct member signatures (`m`).
    threshold: u32,
}

impl ThresholdKeys {
    /// Build an `m`-of-`n` authority from the member set and threshold.
    ///
    /// # Errors
    /// [`AuthError::Config`] if `m == 0`, `m > n`, or the set has duplicate members.
    pub fn new(members: Vec<PeerId>, threshold: u32) -> Result<Self, AuthError> {
        let n = u32::try_from(members.len()).unwrap_or(u32::MAX);
        if threshold == 0 {
            return Err(AuthError::Config {
                reason: "threshold m must be >= 1".into(),
            });
        }
        if threshold > n {
            return Err(AuthError::Config {
                reason: format!("threshold m={threshold} exceeds keyset size n={n}"),
            });
        }
        let mut sorted = members.clone();
        sorted.sort_unstable();
        sorted.dedup();
        if sorted.len() != members.len() {
            return Err(AuthError::Config {
                reason: "keyset contains duplicate members".into(),
            });
        }
        Ok(Self { members, threshold })
    }

    /// The keyset size `n`.
    #[must_use]
    pub fn n(&self) -> u32 {
        u32::try_from(self.members.len()).unwrap_or(u32::MAX)
    }

    /// The required signature count `m`.
    #[must_use]
    pub fn m(&self) -> u32 {
        self.threshold
    }

    /// The envelope-named signer set.
    #[must_use]
    pub fn members(&self) -> &[PeerId] {
        &self.members
    }
}

impl Authority for ThresholdKeys {
    fn contract(&self) -> AuthorityContract {
        AuthorityContract {
            name: "threshold-keys",
            fault_threshold: FaultThreshold::Threshold {
                n: self.n(),
                m: self.threshold,
            },
            finality: Finality::Immediate,
            reconfiguration: Reconfiguration::QuorumReconfigurable,
            safety: "records authoritative iff m distinct member signatures verify; quorum \
                     intersection 2m>n prevents two disjoint quorums authorizing conflicting \
                     records (safe while faulty signers < 2m-n)",
            liveness: "at least m of the n members are honest and available to sign; signer \
                       transfer is quorum reconfiguration, not key custody (architecture §4.4)",
        }
    }

    fn authorize(
        &self,
        records_channel: u32,
        preimage: &[u8],
        sigs: &[RecordSig],
    ) -> Result<Authorized, AuthError> {
        if sigs.is_empty() {
            return Err(AuthError::NotSigned);
        }
        let mut seen: Vec<PeerId> = Vec::with_capacity(sigs.len());
        let mut valid = 0u32;
        for rs in sigs {
            if !self.members.contains(&rs.signer) {
                return Err(AuthError::UnknownSigner { signer: rs.signer });
            }
            if seen.contains(&rs.signer) {
                return Err(AuthError::DuplicateSigner { signer: rs.signer });
            }
            seen.push(rs.signer);
            match verify_sig(&rs.signer.0, &rs.sig.0, preimage) {
                VerifyOutcome::Valid => valid += 1,
                VerifyOutcome::Invalid => {
                    return Err(AuthError::BadSignature { signer: rs.signer })
                }
                VerifyOutcome::Malformed => return Err(AuthError::Malformed),
            }
        }
        if valid >= self.threshold {
            Ok(Authorized::from_authoritative_channel(records_channel))
        } else {
            Err(AuthError::InsufficientSignatures {
                have: valid,
                need: self.threshold,
            })
        }
    }
}

// -- the typed AuthorityConfig (decodes from the envelope's opaque `authority` section) ------------

/// The trust topology a run declares, decoded from [`GenesisEnvelope::authority`] (the opaque CBOR
/// the host never interprets, D0). This is the guest's vocabulary (architecture §5.1: "opaque to the
/// host; interpreted by modules").
///
/// [`Topology`] borrows from the config for a zero-alloc match; [`AuthorityConfig::authority`] boxes
/// a `dyn Authority` for the dynamic path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Topology {
    /// The launch [`SingleKey`] topology.
    SingleKey(SingleKey),
    /// The launch [`ThresholdKeys`] topology.
    ThresholdKeys(ThresholdKeys),
}

/// The typed authority configuration a run's modules link. Decodes from the opaque CBOR value the
/// envelope carries; the [`AuthorityConfig::records_channel`] is the **channel declaration** the
/// signing oracle uses to classify authoritative records (never a guest-selected message class).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthorityConfig {
    /// The trust topology.
    pub topology: Topology,
    /// The authoritative records channel (ABI §6.2). Defaults to [`DEFAULT_RECORDS_CHANNEL`].
    pub records_channel: u32,
}

impl AuthorityConfig {
    /// Decode the typed config from the envelope's opaque `authority` CBOR value (architecture
    /// §5.1). Wire shape (canonical CBOR map):
    ///
    /// ```text
    /// { "topology": "single-key",     "coordinator": bstr32, "records_channel"?: uint }
    /// { "topology": "threshold-keys", "members": [bstr32…], "threshold": uint, "records_channel"?: uint }
    /// ```
    ///
    /// # Errors
    /// [`AuthError::Config`] for an unknown/missing topology tag, missing/mis-typed fields, or an
    /// invalid `m`-of-`n` (`m == 0`, `m > n`, duplicate members).
    pub fn decode(value: &ciborium::value::Value) -> Result<Self, AuthError> {
        let ciborium::value::Value::Map(entries) = value else {
            return Err(AuthError::Config {
                reason: "authority section is not a CBOR map".into(),
            });
        };
        let get = |name: &str| -> Option<&ciborium::value::Value> {
            entries.iter().find_map(|(k, v)| match k {
                ciborium::value::Value::Text(t) if t == name => Some(v),
                _ => None,
            })
        };
        let topology_tag = match get("topology") {
            Some(ciborium::value::Value::Text(t)) => t.as_str(),
            _ => {
                return Err(AuthError::Config {
                    reason: "missing string `topology` tag".into(),
                })
            }
        };
        let records_channel = match get("records_channel") {
            None => DEFAULT_RECORDS_CHANNEL,
            Some(v) => v
                .as_integer()
                .and_then(|n| u32::try_from(i128::from(n)).ok())
                .ok_or_else(|| AuthError::Config {
                    reason: "`records_channel` is not a u32".into(),
                })?,
        };
        let topology = match topology_tag {
            "single-key" => {
                let coordinator =
                    decode_peer(get("coordinator")).ok_or_else(|| AuthError::Config {
                        reason: "single-key requires a 32-byte `coordinator`".into(),
                    })?;
                Topology::SingleKey(SingleKey::new(coordinator))
            }
            "threshold-keys" => {
                let members = match get("members") {
                    Some(ciborium::value::Value::Array(items)) => {
                        let mut out = Vec::with_capacity(items.len());
                        for it in items {
                            out.push(decode_peer(Some(it)).ok_or_else(|| AuthError::Config {
                                reason: "`members` entry is not a 32-byte key".into(),
                            })?);
                        }
                        out
                    }
                    _ => {
                        return Err(AuthError::Config {
                            reason: "threshold-keys requires an array `members`".into(),
                        })
                    }
                };
                let threshold = get("threshold")
                    .and_then(ciborium::value::Value::as_integer)
                    .and_then(|n| u32::try_from(i128::from(n)).ok())
                    .ok_or_else(|| AuthError::Config {
                        reason: "threshold-keys requires a uint `threshold`".into(),
                    })?;
                Topology::ThresholdKeys(ThresholdKeys::new(members, threshold)?)
            }
            other => {
                return Err(AuthError::Config {
                    reason: format!("unknown authority topology `{other}`"),
                })
            }
        };
        Ok(Self {
            topology,
            records_channel,
        })
    }

    /// Encode back to the opaque CBOR value (the authoring side — the envelope author writes this
    /// into `GenesisEnvelope::authority`). Round-trips [`AuthorityConfig::decode`].
    #[must_use]
    pub fn encode(&self) -> ciborium::value::Value {
        use ciborium::value::Value;
        let mut map = vec![(Value::from("topology"), Value::from(self.topology_tag()))];
        match &self.topology {
            Topology::SingleKey(sk) => {
                map.push((
                    Value::from("coordinator"),
                    Value::Bytes(sk.coordinator.0.to_vec()),
                ));
            }
            Topology::ThresholdKeys(tk) => {
                let members = tk
                    .members()
                    .iter()
                    .map(|p| Value::Bytes(p.0.to_vec()))
                    .collect();
                map.push((Value::from("members"), Value::Array(members)));
                map.push((Value::from("threshold"), Value::from(u64::from(tk.m()))));
            }
        }
        map.push((
            Value::from("records_channel"),
            Value::from(u64::from(self.records_channel)),
        ));
        Value::Map(map)
    }

    /// The topology tag string used on the wire.
    #[must_use]
    pub fn topology_tag(&self) -> &'static str {
        match &self.topology {
            Topology::SingleKey(_) => "single-key",
            Topology::ThresholdKeys(_) => "threshold-keys",
        }
    }

    /// The declared contract of the configured topology.
    #[must_use]
    pub fn contract(&self) -> AuthorityContract {
        match &self.topology {
            Topology::SingleKey(a) => a.contract(),
            Topology::ThresholdKeys(a) => a.contract(),
        }
    }

    /// Box a `dyn Authority` for the configured topology (the dynamic construction path D2 uses to
    /// hold one authority regardless of variant).
    #[must_use]
    pub fn authority(&self) -> Box<dyn Authority> {
        match &self.topology {
            Topology::SingleKey(a) => Box::new(*a),
            Topology::ThresholdKeys(a) => Box::new(a.clone()),
        }
    }

    /// Authorize a record directly against the configured topology and the config's authoritative
    /// records channel — the convenience the worker driver calls (equivalent to
    /// `self.authority().authorize(self.records_channel, preimage, sigs)` without the box).
    ///
    /// # Errors
    /// The topology's [`AuthError`].
    pub fn authorize(&self, preimage: &[u8], sigs: &[RecordSig]) -> Result<Authorized, AuthError> {
        match &self.topology {
            Topology::SingleKey(a) => a.authorize(self.records_channel, preimage, sigs),
            Topology::ThresholdKeys(a) => a.authorize(self.records_channel, preimage, sigs),
        }
    }
}

/// Decode a 32-byte peer id from a CBOR byte string (or a tolerant uint-array, matching the proto
/// byte-newtype decoder).
fn decode_peer(value: Option<&ciborium::value::Value>) -> Option<PeerId> {
    match value? {
        ciborium::value::Value::Bytes(b) => b.as_slice().try_into().ok().map(PeerId),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use daemon_vhc_proto::{peer_id, sign_canonical, to_canonical_vec, SigningKey};

    fn key(seed: u8) -> SigningKey {
        SigningKey::from_bytes(&[seed; 32])
    }

    /// A record preimage + a real signature over its canonical bytes by `sk`.
    fn sign_record(sk: &SigningKey, preimage_value: &u64) -> (Vec<u8>, RecordSig) {
        let preimage = to_canonical_vec(preimage_value).unwrap();
        let sig = sign_canonical(sk, preimage_value).unwrap();
        (
            preimage,
            RecordSig {
                signer: peer_id(sk),
                sig,
            },
        )
    }

    // -- SingleKey ---------------------------------------------------------------------------------

    #[test]
    fn single_key_accepts_the_coordinator_signature() {
        let sk = key(1);
        let auth = SingleKey::new(peer_id(&sk));
        let (preimage, rs) = sign_record(&sk, &42);
        let ok = auth.authorize(0, &preimage, &[rs]).expect("authorized");
        assert_eq!(ok.channel(), 0);
        assert_eq!(auth.contract().name, "single-key");
        assert_eq!(auth.contract().fault_threshold, FaultThreshold::None);
    }

    #[test]
    fn single_key_refuses_a_different_signer() {
        let coord = key(1);
        let impostor = key(2);
        let auth = SingleKey::new(peer_id(&coord));
        let (preimage, rs) = sign_record(&impostor, &42);
        assert_eq!(
            auth.authorize(0, &preimage, &[rs]),
            Err(AuthError::WrongSigner {
                got: peer_id(&impostor)
            })
        );
    }

    #[test]
    fn single_key_refuses_no_signature_and_bad_signature() {
        let sk = key(1);
        let auth = SingleKey::new(peer_id(&sk));
        let (preimage, mut rs) = sign_record(&sk, &42);
        assert_eq!(auth.authorize(0, &preimage, &[]), Err(AuthError::NotSigned));
        // Flip the signature: right signer, invalid signature.
        rs.sig.0[0] ^= 0xff;
        assert_eq!(
            auth.authorize(0, &preimage, &[rs]),
            Err(AuthError::BadSignature { signer: rs.signer })
        );
    }

    // -- ThresholdKeys (m-of-n) --------------------------------------------------------------------

    #[test]
    fn threshold_construction_rejects_degenerate_sets() {
        let a = peer_id(&key(1));
        assert!(ThresholdKeys::new(vec![a], 0).is_err(), "m=0");
        assert!(ThresholdKeys::new(vec![a], 2).is_err(), "m>n");
        assert!(
            ThresholdKeys::new(vec![a, a], 1).is_err(),
            "duplicate members"
        );
    }

    #[test]
    fn threshold_accepts_exactly_m_distinct_valid_signatures() {
        let (k1, k2, k3) = (key(1), key(2), key(3));
        let members = vec![peer_id(&k1), peer_id(&k2), peer_id(&k3)];
        let auth = ThresholdKeys::new(members, 2).unwrap();
        let (p1, s1) = sign_record(&k1, &7);
        let (_p2, s2) = sign_record(&k2, &7);
        // 2-of-3 with two distinct valid members → authorized.
        assert!(auth.authorize(0, &p1, &[s1, s2]).is_ok());
        assert_eq!(
            auth.contract().fault_threshold,
            FaultThreshold::Threshold { n: 3, m: 2 }
        );
    }

    #[test]
    fn threshold_refuses_below_quorum_unknown_and_duplicate_signers() {
        let (k1, k2, outsider) = (key(1), key(2), key(9));
        let members = vec![peer_id(&k1), peer_id(&k2)];
        let auth = ThresholdKeys::new(members, 2).unwrap();
        let (p1, s1) = sign_record(&k1, &7);
        // Only one valid member signature: below the 2-of-2 quorum.
        assert_eq!(
            auth.authorize(0, &p1, &[s1]),
            Err(AuthError::InsufficientSignatures { have: 1, need: 2 })
        );
        // A non-member signer is refused outright.
        let (_po, so) = sign_record(&outsider, &7);
        assert_eq!(
            auth.authorize(0, &p1, &[so]),
            Err(AuthError::UnknownSigner {
                signer: peer_id(&outsider)
            })
        );
        // The same member twice cannot inflate the count.
        assert_eq!(
            auth.authorize(0, &p1, &[s1, s1]),
            Err(AuthError::DuplicateSigner {
                signer: peer_id(&k1)
            })
        );
    }

    // -- AuthorityConfig round-trip + decode -------------------------------------------------------

    #[test]
    fn config_round_trips_single_and_threshold() {
        let sk = key(1);
        let cfg = AuthorityConfig {
            topology: Topology::SingleKey(SingleKey::new(peer_id(&sk))),
            records_channel: 0,
        };
        assert_eq!(AuthorityConfig::decode(&cfg.encode()).unwrap(), cfg);

        let tk = AuthorityConfig {
            topology: Topology::ThresholdKeys(
                ThresholdKeys::new(
                    vec![peer_id(&key(1)), peer_id(&key(2)), peer_id(&key(3))],
                    2,
                )
                .unwrap(),
            ),
            records_channel: 5,
        };
        assert_eq!(AuthorityConfig::decode(&tk.encode()).unwrap(), tk);
    }

    #[test]
    fn config_defaults_records_channel_and_rejects_unknown_topology() {
        use ciborium::value::Value;
        let sk = key(1);
        // No records_channel → defaults to the control channel.
        let v = Value::Map(vec![
            (Value::from("topology"), Value::from("single-key")),
            (
                Value::from("coordinator"),
                Value::Bytes(peer_id(&sk).0.to_vec()),
            ),
        ]);
        let cfg = AuthorityConfig::decode(&v).unwrap();
        assert_eq!(cfg.records_channel, DEFAULT_RECORDS_CHANNEL);

        let bad = Value::Map(vec![(
            Value::from("topology"),
            Value::from("elected-leader"),
        )]);
        assert!(matches!(
            AuthorityConfig::decode(&bad),
            Err(AuthError::Config { .. })
        ));
    }

    #[test]
    fn config_authorize_matches_the_boxed_authority() {
        let sk = key(1);
        let cfg = AuthorityConfig {
            topology: Topology::SingleKey(SingleKey::new(peer_id(&sk))),
            records_channel: 3,
        };
        let (preimage, rs) = sign_record(&sk, &99);
        let via_cfg = cfg.authorize(&preimage, &[rs]).unwrap();
        assert_eq!(via_cfg.channel(), 3);
        assert_eq!(cfg.contract().name, "single-key");
    }
}
