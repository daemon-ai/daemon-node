// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The genesis + transition chain — a run's identity through time (architecture §5.1, §5.4;
//! refactor §9; ABI §8.1/§10.3).
//!
//! A run is *founded* by an immutable genesis envelope whose hash is the run's stable `RunId`
//! ([`crate::genesis::FrozenGenesis::run_id`]), and *defined at any moment* by that genesis plus an
//! **append-only transition chain**: the authorized upgrade records that have amended module hashes
//! (and their grants/config anchors) since genesis. The head of the chain is the current
//! [`EpochDescriptor`]; the execution identity `(run_id, epoch, role, instance, module_hash)`
//! (ABI §8.1, frozen) is keyed by the `(run_id, epoch, role)` → `module_hash` mapping this chain
//! defines — the named invariant D1-EPOCH: **`module_hash` is a pure function of
//! `(run_id, epoch, role)` via this chain**, because every module transition advances the epoch.
//!
//! This is algorithm-free wire mechanism (hashes, canonical CBOR, ed25519 signatures over
//! host-readable [`crate::genesis::Identities::upgrade_authority`] keys), so it lives in
//! `daemon-vhc-proto` and is usable by the host session **and** by modules — neither re-derives it.
//!
//! ## The two-key model (architecture §5.4)
//!
//! An upgrade record is **run-internal authorization**: `(role, old_hash → new_hash, epoch fence)`,
//! signed per the envelope's **upgrade [`UpgradeAuthority`]**, committed to this chain **once,
//! globally, before any host acts**. That global commit is the run-level event — it advances the
//! epoch. The separate **machine-owner authorization** (owner-law re-check, grant-expanding fails
//! closed) is host law, applied at each local switch by the upgrade transaction (ABI §10.3), and is
//! deliberately NOT part of this chain: a failed *local* migration never rolls back the chain.
//!
//! ## Launch authority rule
//!
//! The genesis stores the upgrade authority as a bare key list ([`Identities::upgrade_authority`]);
//! it carries no threshold field (schema-frozen at D0). [`UpgradeAuthority::from_genesis`] therefore
//! adopts the **fail-closed launch rule: unanimous over the listed keys** — which degenerates to
//! `SingleKey` for the one-key launch topology and requires every listed key otherwise. An empty
//! list authorizes no upgrade at all (a run with no upgrade authority is immutable — fail closed).
//! An explicit `m`-of-`n` threshold is available through [`UpgradeAuthority::new`] for a node that
//! sources the threshold out-of-band; wiring it into the opaque `authority` section is a D1
//! `Authority`-contract extension, deferred.

use serde::{Deserialize, Serialize};

use crate::bytes::{Hash, PeerId, Signature};
use crate::canonical::to_canonical_vec;
use crate::error::VhcProtoError;
use crate::genesis::{GenesisEnvelope, Identities};
use crate::hash::blake3_hash;
use crate::sign::{peer_id, sign_canonical, verify_canonical, SigningKey};

/// The domain-separation tag every upgrade-record body carries at ABI major 2. Distinct from the
/// frame-envelope (`daemon-vhc/frame/2`) and certificate (`daemon-vhc/cert/2`) domains so an
/// upgrade-record signature can never be replayed as a frame or certificate signature or vice
/// versa.
pub const UPGRADE_RECORD_DOMAIN_V2: &str = "daemon-vhc/upgrade/2";

/// Why a transition-chain operation was refused. Every variant is fail-closed: a chain rejects a
/// record it cannot fully authorize and validate, and never advances the epoch on a bad record.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum TransitionError {
    /// The record's `domain` tag is not [`UPGRADE_RECORD_DOMAIN_V2`].
    WrongDomain {
        /// The tag actually carried.
        got: String,
    },
    /// The record's `run_id` does not match the chain's run (a record from another run).
    RunMismatch,
    /// The record's `epoch` is not exactly `head epoch + 1` (the chain is strictly monotone and
    /// gap-free — it is committed once, globally, in order).
    NonMonotoneEpoch {
        /// The epoch the record establishes.
        got: u64,
        /// The epoch the chain expected (`head + 1`).
        expected: u64,
    },
    /// The record's `prev` link does not equal the current chain head hash (a fork / reorder — the
    /// chain is append-only and hash-linked).
    BrokenChain,
    /// The record names a role the chain does not carry (roles are fixed by genesis, §5.1).
    UnknownRole {
        /// The offending role label.
        role: String,
    },
    /// The record's `old_module` does not equal the role's current module hash (a stale upgrade
    /// authored against a superseded epoch).
    StaleOldModule {
        /// The role the record targets.
        role: String,
    },
    /// The upgrade authority did not authorize the record: fewer than `threshold` valid, distinct
    /// signatures from the authority key set.
    NotAuthorized {
        /// How many valid distinct authority signatures were found.
        have: usize,
        /// How many were required.
        need: usize,
    },
    /// The authority key set is empty (a run with no upgrade authority is immutable, fail closed).
    NoAuthority,
    /// A threshold outside `1..=keys.len()` was requested.
    BadThreshold {
        /// The requested threshold.
        threshold: usize,
        /// The key-set size.
        keys: usize,
    },
    /// A role's module artifact is missing from the genesis artifact map (a malformed genesis).
    MissingModuleArtifact {
        /// The role whose module could not be resolved.
        role: String,
    },
    /// A canonical-CBOR encode failure while hashing/signing a record body.
    Codec(String),
}

impl core::fmt::Display for TransitionError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::WrongDomain { got } => {
                write!(
                    f,
                    "upgrade-record domain `{got}` is not `{UPGRADE_RECORD_DOMAIN_V2}`"
                )
            }
            Self::RunMismatch => write!(f, "upgrade record is for a different run"),
            Self::NonMonotoneEpoch { got, expected } => write!(
                f,
                "upgrade record establishes epoch {got}, chain expected {expected} (strictly \
                 monotone, gap-free)"
            ),
            Self::BrokenChain => write!(f, "upgrade record `prev` does not link the chain head"),
            Self::UnknownRole { role } => write!(f, "upgrade record names unknown role `{role}`"),
            Self::StaleOldModule { role } => write!(
                f,
                "upgrade record `old_module` for role `{role}` is not the current module hash"
            ),
            Self::NotAuthorized { have, need } => write!(
                f,
                "upgrade record not authorized: {have} valid authority signatures, need {need}"
            ),
            Self::NoAuthority => write!(f, "run has no upgrade authority (immutable, fail closed)"),
            Self::BadThreshold { threshold, keys } => {
                write!(f, "threshold {threshold} outside 1..={keys}")
            }
            Self::MissingModuleArtifact { role } => {
                write!(
                    f,
                    "role `{role}` module artifact missing from the genesis artifact map"
                )
            }
            Self::Codec(e) => write!(f, "upgrade-record codec error: {e}"),
        }
    }
}

impl std::error::Error for TransitionError {}

impl From<TransitionError> for VhcProtoError {
    fn from(e: TransitionError) -> Self {
        VhcProtoError::Validation(e.to_string())
    }
}

/// One authority signature over an upgrade-record body: `(signer, sig)`. The same shape the
/// consensus `RecordSig` uses, kept host-local here so the chain needs no SDK dependency.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct UpgradeSig {
    /// The signing authority key (one of [`UpgradeAuthority::keys`]).
    pub signer: PeerId,
    /// ed25519 signature over the canonical CBOR of the [`UpgradeRecordBody`].
    pub sig: Signature,
}

/// The run's upgrade authority: whose signatures make an upgrade record authoritative
/// (architecture §5.4, §4.2 `Authority`). Realized here as an `m`-of-`n` threshold over the
/// host-readable [`Identities::upgrade_authority`] keys.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UpgradeAuthority {
    keys: Vec<PeerId>,
    threshold: usize,
}

impl UpgradeAuthority {
    /// A single-key upgrade authority (the launch `SingleKey` topology).
    #[must_use]
    pub fn single(key: PeerId) -> Self {
        Self {
            keys: vec![key],
            threshold: 1,
        }
    }

    /// An explicit `m`-of-`n` threshold authority.
    ///
    /// # Errors
    /// [`TransitionError::NoAuthority`] on an empty key set; [`TransitionError::BadThreshold`] if
    /// `threshold` is not in `1..=keys.len()`.
    pub fn new(keys: Vec<PeerId>, threshold: usize) -> Result<Self, TransitionError> {
        if keys.is_empty() {
            return Err(TransitionError::NoAuthority);
        }
        if threshold == 0 || threshold > keys.len() {
            return Err(TransitionError::BadThreshold {
                threshold,
                keys: keys.len(),
            });
        }
        Ok(Self { keys, threshold })
    }

    /// Derive the upgrade authority from a genesis envelope's [`Identities::upgrade_authority`],
    /// under the **fail-closed launch rule: unanimous over the listed keys** (§module docs). An
    /// empty list is [`TransitionError::NoAuthority`] — such a run admits no upgrades.
    ///
    /// # Errors
    /// [`TransitionError::NoAuthority`] if `identities.upgrade_authority` is empty.
    pub fn from_genesis(identities: &Identities) -> Result<Self, TransitionError> {
        let keys = identities.upgrade_authority.clone();
        let n = keys.len();
        Self::new(keys, n)
    }

    /// The authority key set.
    #[must_use]
    pub fn keys(&self) -> &[PeerId] {
        &self.keys
    }

    /// The signature threshold.
    #[must_use]
    pub fn threshold(&self) -> usize {
        self.threshold
    }

    /// Authorize a record body: at least `threshold` valid signatures from **distinct** authority
    /// keys over the body's canonical CBOR. Signatures from non-authority keys and duplicate
    /// signers are ignored (never counted twice).
    ///
    /// # Errors
    /// [`TransitionError::NotAuthorized`] when fewer than `threshold` distinct valid authority
    /// signatures are present.
    pub fn authorize(
        &self,
        body: &UpgradeRecordBody,
        sigs: &[UpgradeSig],
    ) -> Result<(), TransitionError> {
        let mut credited: Vec<&PeerId> = Vec::new();
        for s in sigs {
            if !self.keys.contains(&s.signer) {
                continue; // not an authority key
            }
            if credited.iter().any(|k| **k == s.signer) {
                continue; // already credited this signer
            }
            if verify_canonical(&s.signer, &s.sig, body).is_ok() {
                credited.push(&s.signer);
            }
        }
        if credited.len() >= self.threshold {
            Ok(())
        } else {
            Err(TransitionError::NotAuthorized {
                have: credited.len(),
                need: self.threshold,
            })
        }
    }
}

/// The signed body of one upgrade record (architecture §5.4; ABI §10.3). Every field is part of the
/// signed preimage and the chain hash.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct UpgradeRecordBody {
    /// Domain-separation tag — MUST be [`UPGRADE_RECORD_DOMAIN_V2`].
    pub domain: String,
    /// The run's cryptographic identity: the genesis-envelope hash (ABI §8.1 `run_id`).
    pub run_id: Hash,
    /// The epoch this record **establishes** — exactly `previous epoch + 1` (strictly monotone,
    /// gap-free). It is the `epoch` field of the execution identity for `role` from here on.
    pub epoch: u64,
    /// The hash of the previous chain link: the previous [`UpgradeRecordBody`]'s hash, or the
    /// `run_id` (genesis hash) for the first record (epoch 1). This makes the chain append-only and
    /// tamper-evident.
    pub prev: Hash,
    /// The role whose module is replaced (roles are fixed by genesis, §5.1).
    pub role: String,
    /// The module hash being replaced — MUST equal the role's current module at the head epoch.
    pub old_module: Hash,
    /// The module hash to switch to at `epoch` (the new `module_hash` of the execution identity).
    pub new_module: Hash,
    /// The epoch fence (the SDK/coordinator-module-selected quiesce point, architecture §5.4). The
    /// host treats it as opaque ordering metadata; the module places and interprets it.
    pub fence: u64,
    /// blake3 of the new role grants document — the owner-law re-check anchor (ABI §2.6/§10.3).
    pub grants_hash: Hash,
    /// blake3 of the new role config — the migration-determinism anchor (ABI §10.2).
    pub config_hash: Hash,
}

impl UpgradeRecordBody {
    /// The chain link hash: blake3 of the body's canonical CBOR. This is the `prev` the *next*
    /// record must carry.
    ///
    /// # Errors
    /// [`TransitionError::Codec`] on a canonical-CBOR encode failure (structurally unreachable).
    pub fn hash(&self) -> Result<Hash, TransitionError> {
        let bytes = to_canonical_vec(self).map_err(|e| TransitionError::Codec(e.to_string()))?;
        Ok(blake3_hash(&bytes))
    }
}

/// One committed upgrade record: the [`UpgradeRecordBody`] plus its upgrade-authority signatures.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct UpgradeRecord {
    /// The signed binding.
    pub body: UpgradeRecordBody,
    /// The upgrade-authority signatures over the body's canonical CBOR (`m`-of-`n`, §two-key model).
    pub sigs: Vec<UpgradeSig>,
}

impl UpgradeRecord {
    /// Author and sign an upgrade record with the given authority signing keys. A convenience for
    /// authoring/tests; a production authority may sign the body out-of-band and assemble
    /// [`UpgradeSig`]s directly.
    ///
    /// # Errors
    /// [`TransitionError::Codec`] on a signing/encode failure.
    #[allow(clippy::too_many_arguments)]
    pub fn author(
        run_id: Hash,
        epoch: u64,
        prev: Hash,
        role: impl Into<String>,
        old_module: Hash,
        new_module: Hash,
        fence: u64,
        grants_hash: Hash,
        config_hash: Hash,
        signers: &[&SigningKey],
    ) -> Result<Self, TransitionError> {
        let body = UpgradeRecordBody {
            domain: UPGRADE_RECORD_DOMAIN_V2.to_string(),
            run_id,
            epoch,
            prev,
            role: role.into(),
            old_module,
            new_module,
            fence,
            grants_hash,
            config_hash,
        };
        let mut sigs = Vec::with_capacity(signers.len());
        for key in signers {
            let sig =
                sign_canonical(key, &body).map_err(|e| TransitionError::Codec(e.to_string()))?;
            sigs.push(UpgradeSig {
                signer: peer_id(key),
                sig,
            });
        }
        Ok(Self { body, sigs })
    }

    /// The chain link hash of this record's body.
    ///
    /// # Errors
    /// [`TransitionError::Codec`] on a canonical-CBOR encode failure.
    pub fn hash(&self) -> Result<Hash, TransitionError> {
        self.body.hash()
    }
}

/// The head of the transition chain at a point in time (architecture §5.1 "epoch descriptor"): the
/// `(run_id, epoch)` position and the resolved `role → module_hash` map that keys every live
/// role-instance's execution identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EpochDescriptor {
    /// The run's cryptographic identity (genesis hash).
    pub run_id: Hash,
    /// The current transition-chain position (`0` at genesis).
    pub epoch: u64,
    /// `role → module_hash` at this epoch (D1-EPOCH: a pure function of `(run_id, epoch, role)`).
    pub modules: std::collections::BTreeMap<String, Hash>,
    /// The chain head hash — the `prev` link the next upgrade record must carry (the `run_id`
    /// itself at epoch 0).
    pub head: Hash,
}

impl EpochDescriptor {
    /// The module hash a `role` runs at this epoch, if the role exists.
    #[must_use]
    pub fn module_for(&self, role: &str) -> Option<Hash> {
        self.modules.get(role).copied()
    }
}

/// The append-only transition chain: genesis anchor + the ordered upgrade records committed since.
/// Advancing it is the run-level upgrade event (committed once, globally); the host never advances
/// it locally (ABI §10.3 step 6).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransitionChain {
    run_id: Hash,
    epoch: u64,
    modules: std::collections::BTreeMap<String, Hash>,
    head: Hash,
    records: Vec<UpgradeRecord>,
}

impl TransitionChain {
    /// Anchor a chain at genesis (epoch 0): the `role → module_hash` map is resolved from the
    /// genesis role set through its artifact map, and the head is the `run_id` itself.
    ///
    /// # Errors
    /// [`TransitionError::MissingModuleArtifact`] if a role's module artifact is absent.
    pub fn genesis(genesis: &GenesisEnvelope, run_id: Hash) -> Result<Self, TransitionError> {
        let mut modules = std::collections::BTreeMap::new();
        for (role, entry) in &genesis.roles {
            let artifact = genesis
                .artifacts
                .get(&entry.module)
                .ok_or_else(|| TransitionError::MissingModuleArtifact { role: role.clone() })?;
            modules.insert(role.clone(), artifact.blake3);
        }
        Ok(Self {
            run_id,
            epoch: 0,
            modules,
            head: run_id,
            records: Vec::new(),
        })
    }

    /// The current epoch (`0` at genesis).
    #[must_use]
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// The run's cryptographic identity.
    #[must_use]
    pub fn run_id(&self) -> Hash {
        self.run_id
    }

    /// The current chain head hash (the `prev` link the next record must carry).
    #[must_use]
    pub fn head(&self) -> Hash {
        self.head
    }

    /// The module hash a `role` runs at the head epoch.
    #[must_use]
    pub fn module_for(&self, role: &str) -> Option<Hash> {
        self.modules.get(role).copied()
    }

    /// The current epoch descriptor (the chain head).
    #[must_use]
    pub fn descriptor(&self) -> EpochDescriptor {
        EpochDescriptor {
            run_id: self.run_id,
            epoch: self.epoch,
            modules: self.modules.clone(),
            head: self.head,
        }
    }

    /// The committed upgrade records, in order.
    #[must_use]
    pub fn records(&self) -> &[UpgradeRecord] {
        &self.records
    }

    /// Validate and append one authorized upgrade record, advancing the epoch by one. This is the
    /// **global commit** (architecture §5.4): it happens exactly once, in order, and the epoch is
    /// advanced by it — no host advances the chain locally.
    ///
    /// Validation is total and fail-closed: domain, run binding, strictly-monotone gap-free epoch,
    /// the `prev` hash-link, a known role, the `old_module` match against the head, and the upgrade
    /// authority's threshold over distinct valid signatures. On any failure the chain is unchanged.
    ///
    /// # Errors
    /// The applicable [`TransitionError`]; the chain is left untouched on error.
    pub fn append(
        &mut self,
        record: UpgradeRecord,
        authority: &UpgradeAuthority,
    ) -> Result<EpochDescriptor, TransitionError> {
        let body = &record.body;
        if body.domain != UPGRADE_RECORD_DOMAIN_V2 {
            return Err(TransitionError::WrongDomain {
                got: body.domain.clone(),
            });
        }
        if body.run_id != self.run_id {
            return Err(TransitionError::RunMismatch);
        }
        let expected_epoch = self.epoch + 1;
        if body.epoch != expected_epoch {
            return Err(TransitionError::NonMonotoneEpoch {
                got: body.epoch,
                expected: expected_epoch,
            });
        }
        if body.prev != self.head {
            return Err(TransitionError::BrokenChain);
        }
        let current = self
            .modules
            .get(&body.role)
            .ok_or_else(|| TransitionError::UnknownRole {
                role: body.role.clone(),
            })?;
        if *current != body.old_module {
            return Err(TransitionError::StaleOldModule {
                role: body.role.clone(),
            });
        }
        // Run-internal authorization: the two-key model's first key (§5.4).
        authority.authorize(body, &record.sigs)?;

        // Commit: advance the epoch, swap the role's module, extend the hash chain.
        let link = record.hash()?;
        self.modules.insert(body.role.clone(), body.new_module);
        self.epoch = body.epoch;
        self.head = link;
        self.records.push(record);
        // The just-pushed record defines the new head descriptor.
        Ok(self.descriptor())
    }

    /// Rebuild a chain from genesis by applying an ordered slice of records under `authority` —
    /// the append-only replay a late joiner or a standby performs from the record archive
    /// (architecture §5.3). Equivalent to [`TransitionChain::genesis`] followed by [`append`] per
    /// record, so the same total validation applies.
    ///
    /// # Errors
    /// The first [`TransitionError`] encountered; a partially-built chain is not returned.
    ///
    /// [`append`]: TransitionChain::append
    pub fn replay(
        genesis: &GenesisEnvelope,
        run_id: Hash,
        records: &[UpgradeRecord],
        authority: &UpgradeAuthority,
    ) -> Result<Self, TransitionError> {
        let mut chain = Self::genesis(genesis, run_id)?;
        for record in records {
            chain.append(record.clone(), authority)?;
        }
        Ok(chain)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::envelope::{Access, DeviceMinimums};
    use crate::genesis::{
        GenesisEnvelope, Identities, RoleEntry, RoleGrants, RunSection, SnapshotArtifact,
        TransportSelection, GENESIS_SCHEMA_MAJOR,
    };
    use std::collections::BTreeMap;

    fn key(seed: u8) -> SigningKey {
        SigningKey::from_bytes(&[seed; 32])
    }

    fn hash(n: u8) -> Hash {
        Hash([n; 32])
    }

    // A minimal two-role genesis: worker (module hash 1) + coordinator (module hash 2), with an
    // upgrade authority of the given keys.
    fn sample_genesis(upgrade_authority: Vec<PeerId>) -> GenesisEnvelope {
        let mut artifacts = BTreeMap::new();
        artifacts.insert(
            "worker-mod".to_string(),
            SnapshotArtifact {
                url: "r2://mods/worker.wasm".into(),
                blake3: hash(1),
                size: Some(4096),
            },
        );
        artifacts.insert(
            "coord-mod".to_string(),
            SnapshotArtifact {
                url: "r2://mods/coord.wasm".into(),
                blake3: hash(2),
                size: Some(2048),
            },
        );

        let mut roles = BTreeMap::new();
        roles.insert(
            "worker".to_string(),
            RoleEntry {
                lane: "trainer".into(),
                module: "worker-mod".into(),
                abi: "vhc@2".into(),
                config: ciborium::value::Value::Map(vec![]),
                grants: RoleGrants::default(),
                device_min: DeviceMinimums::default(),
            },
        );
        roles.insert(
            "coordinator".to_string(),
            RoleEntry {
                lane: "coordinator".into(),
                module: "coord-mod".into(),
                abi: "vhc@2".into(),
                config: ciborium::value::Value::Map(vec![]),
                grants: RoleGrants::default(),
                device_min: DeviceMinimums::default(),
            },
        );

        GenesisEnvelope {
            run: RunSection {
                schema: GENESIS_SCHEMA_MAJOR,
                run_label: "demo-run".into(),
                min_peers: 1,
                max_peers: 64,
                access: Access::Org,
            },
            roles,
            artifacts,
            authority: ciborium::value::Value::Map(vec![]),
            transport: TransportSelection::default(),
            identities: Identities {
                upgrade_authority,
                ..Default::default()
            },
        }
    }

    // Freeze `genesis` and anchor a chain at its run_id, alongside the derived unanimous authority.
    fn anchor(genesis: &GenesisEnvelope) -> (TransitionChain, Hash, UpgradeAuthority) {
        let frozen = genesis.freeze(&key(200)).unwrap();
        let run_id = *frozen.run_id();
        let chain = TransitionChain::genesis(genesis, run_id).unwrap();
        let authority = UpgradeAuthority::from_genesis(&genesis.identities).unwrap();
        (chain, run_id, authority)
    }

    #[test]
    fn genesis_anchor_resolves_modules_and_head() {
        let g = sample_genesis(vec![peer_id(&key(1))]);
        let (chain, run_id, _auth) = anchor(&g);
        assert_eq!(chain.epoch(), 0);
        assert_eq!(chain.head(), run_id, "head is the run_id at genesis");
        assert_eq!(chain.module_for("worker"), Some(hash(1)));
        assert_eq!(chain.module_for("coordinator"), Some(hash(2)));
        let d = chain.descriptor();
        assert_eq!(d.epoch, 0);
        assert_eq!(d.module_for("worker"), Some(hash(1)));
    }

    #[test]
    fn append_advances_epoch_swaps_module_and_extends_chain() {
        let g = sample_genesis(vec![peer_id(&key(1))]);
        let (mut chain, run_id, auth) = anchor(&g);

        let rec = UpgradeRecord::author(
            run_id,
            1,
            run_id, // prev = genesis head
            "worker",
            hash(1),  // old
            hash(42), // new
            7,
            hash(50),
            hash(51),
            &[&key(1)],
        )
        .unwrap();
        let expected_head = rec.hash().unwrap();
        let desc = chain.append(rec, &auth).unwrap();

        assert_eq!(desc.epoch, 1);
        assert_eq!(desc.module_for("worker"), Some(hash(42)));
        // An untouched role keeps its module — module_hash is a pure function of (run_id, epoch,
        // role) via the chain (invariant D1-EPOCH).
        assert_eq!(desc.module_for("coordinator"), Some(hash(2)));
        assert_eq!(chain.epoch(), 1);
        assert_eq!(chain.head(), expected_head);
        assert_eq!(chain.records().len(), 1);
    }

    #[test]
    fn two_records_chain_and_replay_is_identical() {
        let g = sample_genesis(vec![peer_id(&key(1))]);
        let (mut chain, run_id, auth) = anchor(&g);

        let r1 = UpgradeRecord::author(
            run_id,
            1,
            run_id,
            "worker",
            hash(1),
            hash(42),
            7,
            hash(50),
            hash(51),
            &[&key(1)],
        )
        .unwrap();
        chain.append(r1.clone(), &auth).unwrap();
        let head1 = chain.head();
        let r2 = UpgradeRecord::author(
            run_id,
            2,
            head1,
            "coordinator",
            hash(2),
            hash(99),
            9,
            hash(60),
            hash(61),
            &[&key(1)],
        )
        .unwrap();
        chain.append(r2.clone(), &auth).unwrap();

        assert_eq!(chain.epoch(), 2);
        assert_eq!(chain.module_for("worker"), Some(hash(42)));
        assert_eq!(chain.module_for("coordinator"), Some(hash(99)));

        // Replaying genesis + the same records reconstructs the identical chain (the late-join /
        // standby path, architecture §5.3).
        let replayed = TransitionChain::replay(&g, run_id, &[r1, r2], &auth).unwrap();
        assert_eq!(replayed, chain);
    }

    #[test]
    fn rejects_wrong_domain() {
        let g = sample_genesis(vec![peer_id(&key(1))]);
        let (mut chain, run_id, auth) = anchor(&g);
        let mut rec = UpgradeRecord::author(
            run_id,
            1,
            run_id,
            "worker",
            hash(1),
            hash(42),
            7,
            hash(50),
            hash(51),
            &[&key(1)],
        )
        .unwrap();
        rec.body.domain = "daemon-vhc/frame/2".into();
        // The signature no longer matches the mutated body either, but the domain guard fires first.
        assert!(matches!(
            chain.append(rec, &auth),
            Err(TransitionError::WrongDomain { .. })
        ));
        assert_eq!(chain.epoch(), 0, "chain untouched on refusal");
    }

    #[test]
    fn rejects_run_mismatch() {
        let g = sample_genesis(vec![peer_id(&key(1))]);
        let (mut chain, run_id, auth) = anchor(&g);
        let other_run = hash(123);
        let rec = UpgradeRecord::author(
            other_run,
            1,
            run_id,
            "worker",
            hash(1),
            hash(42),
            7,
            hash(50),
            hash(51),
            &[&key(1)],
        )
        .unwrap();
        assert_eq!(chain.append(rec, &auth), Err(TransitionError::RunMismatch));
    }

    #[test]
    fn rejects_non_monotone_epoch() {
        let g = sample_genesis(vec![peer_id(&key(1))]);
        let (mut chain, run_id, auth) = anchor(&g);
        // epoch 2 with no epoch 1 — a gap.
        let rec = UpgradeRecord::author(
            run_id,
            2,
            run_id,
            "worker",
            hash(1),
            hash(42),
            7,
            hash(50),
            hash(51),
            &[&key(1)],
        )
        .unwrap();
        assert_eq!(
            chain.append(rec, &auth),
            Err(TransitionError::NonMonotoneEpoch {
                got: 2,
                expected: 1
            })
        );
    }

    #[test]
    fn rejects_broken_chain_link() {
        let g = sample_genesis(vec![peer_id(&key(1))]);
        let (mut chain, run_id, auth) = anchor(&g);
        // prev != head (should be run_id at epoch 1).
        let rec = UpgradeRecord::author(
            run_id,
            1,
            hash(88),
            "worker",
            hash(1),
            hash(42),
            7,
            hash(50),
            hash(51),
            &[&key(1)],
        )
        .unwrap();
        assert_eq!(chain.append(rec, &auth), Err(TransitionError::BrokenChain));
    }

    #[test]
    fn rejects_unknown_role_and_stale_old_module() {
        let g = sample_genesis(vec![peer_id(&key(1))]);
        let (mut chain, run_id, auth) = anchor(&g);

        let unknown = UpgradeRecord::author(
            run_id,
            1,
            run_id,
            "verifier",
            hash(1),
            hash(42),
            7,
            hash(50),
            hash(51),
            &[&key(1)],
        )
        .unwrap();
        assert!(matches!(
            chain.append(unknown, &auth),
            Err(TransitionError::UnknownRole { .. })
        ));

        // old_module does not match the head module (hash(1)) for worker.
        let stale = UpgradeRecord::author(
            run_id,
            1,
            run_id,
            "worker",
            hash(9),
            hash(42),
            7,
            hash(50),
            hash(51),
            &[&key(1)],
        )
        .unwrap();
        assert!(matches!(
            chain.append(stale, &auth),
            Err(TransitionError::StaleOldModule { .. })
        ));
    }

    #[test]
    fn rejects_unauthorized_record() {
        let g = sample_genesis(vec![peer_id(&key(1))]);
        let (mut chain, run_id, auth) = anchor(&g);
        // Signed by a non-authority key.
        let rec = UpgradeRecord::author(
            run_id,
            1,
            run_id,
            "worker",
            hash(1),
            hash(42),
            7,
            hash(50),
            hash(51),
            &[&key(2)],
        )
        .unwrap();
        assert_eq!(
            chain.append(rec, &auth),
            Err(TransitionError::NotAuthorized { have: 0, need: 1 })
        );

        // No signatures at all.
        let unsigned = UpgradeRecord::author(
            run_id,
            1,
            run_id,
            "worker",
            hash(1),
            hash(42),
            7,
            hash(50),
            hash(51),
            &[],
        )
        .unwrap();
        assert_eq!(
            chain.append(unsigned, &auth),
            Err(TransitionError::NotAuthorized { have: 0, need: 1 })
        );
    }

    #[test]
    fn threshold_m_of_n_and_no_double_counting() {
        // A 2-of-3 upgrade authority.
        let keys = [key(1), key(2), key(3)];
        let peers: Vec<PeerId> = keys.iter().map(peer_id).collect();
        let g = sample_genesis(peers.clone());
        let frozen = g.freeze(&key(200)).unwrap();
        let run_id = *frozen.run_id();
        let auth = UpgradeAuthority::new(peers, 2).unwrap();

        // One valid signature: below threshold.
        let mut chain = TransitionChain::genesis(&g, run_id).unwrap();
        let one = UpgradeRecord::author(
            run_id,
            1,
            run_id,
            "worker",
            hash(1),
            hash(42),
            7,
            hash(50),
            hash(51),
            &[&key(1)],
        )
        .unwrap();
        assert_eq!(
            chain.append(one, &auth),
            Err(TransitionError::NotAuthorized { have: 1, need: 2 })
        );

        // The same signer twice does not count twice (dedup by signer).
        let dup = UpgradeRecord::author(
            run_id,
            1,
            run_id,
            "worker",
            hash(1),
            hash(42),
            7,
            hash(50),
            hash(51),
            &[&key(1), &key(1)],
        )
        .unwrap();
        assert_eq!(
            chain.append(dup, &auth),
            Err(TransitionError::NotAuthorized { have: 1, need: 2 })
        );

        // Two distinct authority signers meet the threshold; an extra non-authority sig is ignored.
        let ok = UpgradeRecord::author(
            run_id,
            1,
            run_id,
            "worker",
            hash(1),
            hash(42),
            7,
            hash(50),
            hash(51),
            &[&key(1), &key(2), &key(99)],
        )
        .unwrap();
        assert!(chain.append(ok, &auth).is_ok());
        assert_eq!(chain.epoch(), 1);
    }

    #[test]
    fn from_genesis_is_unanimous_over_listed_keys() {
        // Two upgrade-authority keys → unanimous (2-of-2) launch rule.
        let peers = vec![peer_id(&key(1)), peer_id(&key(2))];
        let g = sample_genesis(peers);
        let auth = UpgradeAuthority::from_genesis(&g.identities).unwrap();
        assert_eq!(auth.threshold(), 2);

        let frozen = g.freeze(&key(200)).unwrap();
        let run_id = *frozen.run_id();
        let mut chain = TransitionChain::genesis(&g, run_id).unwrap();

        // Only one of the two signs — refused.
        let partial = UpgradeRecord::author(
            run_id,
            1,
            run_id,
            "worker",
            hash(1),
            hash(42),
            7,
            hash(50),
            hash(51),
            &[&key(1)],
        )
        .unwrap();
        assert_eq!(
            chain.append(partial, &auth),
            Err(TransitionError::NotAuthorized { have: 1, need: 2 })
        );

        // Both sign — accepted.
        let full = UpgradeRecord::author(
            run_id,
            1,
            run_id,
            "worker",
            hash(1),
            hash(42),
            7,
            hash(50),
            hash(51),
            &[&key(1), &key(2)],
        )
        .unwrap();
        assert!(chain.append(full, &auth).is_ok());
    }

    #[test]
    fn empty_upgrade_authority_is_immutable() {
        let g = sample_genesis(vec![]);
        assert_eq!(
            UpgradeAuthority::from_genesis(&g.identities),
            Err(TransitionError::NoAuthority)
        );
    }

    #[test]
    fn bad_threshold_is_rejected() {
        let peers = vec![peer_id(&key(1)), peer_id(&key(2))];
        assert!(matches!(
            UpgradeAuthority::new(peers.clone(), 0),
            Err(TransitionError::BadThreshold { .. })
        ));
        assert!(matches!(
            UpgradeAuthority::new(peers, 3),
            Err(TransitionError::BadThreshold { .. })
        ));
    }

    #[test]
    fn record_round_trips_through_canonical_cbor() {
        let g = sample_genesis(vec![peer_id(&key(1))]);
        let (_chain, run_id, _auth) = anchor(&g);
        let rec = UpgradeRecord::author(
            run_id,
            1,
            run_id,
            "worker",
            hash(1),
            hash(42),
            7,
            hash(50),
            hash(51),
            &[&key(1)],
        )
        .unwrap();
        let bytes = to_canonical_vec(&rec).unwrap();
        let back: UpgradeRecord = crate::from_canonical_slice(&bytes).unwrap();
        assert_eq!(rec, back);
        assert_eq!(rec.hash().unwrap(), back.hash().unwrap());
    }
}
