// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The session's v2 network attach — the inbound §12.1 seam **above the pump** (ABI §12.1/§12.2;
//! refactor §5 A2): every authoritative frame is signature-verified, scope-checked, deduped on
//! the full scope tuple, and gap-checked BEFORE `PumpHandle::deliver_frame` sees it. The pump's
//! delivery contract ("pre-verified") is discharged exactly here; the original signed frame
//! rides along as tag-12 evidence (already journaled by the driver).
//!
//! **Gap semantics (Phase A):** a detected gap in a signed stream's `(run_id, epoch, role,
//! instance, channel, seq, sender)` scope is SURFACED typed and the frame is HELD — never
//! silently skipped (§12.2). The backfill machinery (record archive) is Phase B; until then a
//! gap is the caller's decision point (`SequenceGapUnrecoverable` → `Stop{Fault}` in the ABI's
//! run-condition table).
//!
//! Duplicates (an already-accepted scope tuple) are idempotently dropped — re-delivery of a
//! signed frame is normal gossip behavior, not an error.
//!
//! **Certified senders are mandatory** (architecture §4.3): every production attach carries a
//! [`CertCheck`] — the frame's `sender` per-run key must chain to a trusted base identity for the
//! frame's full execution-identity scope and must not be revoked/superseded. The only
//! certificate-less constructor is the `harness`-gated [`InboundFrames::without_certs`].

use std::collections::HashMap;

use ciborium::value::Value;
use daemon_vhc_proto::sign::verify_bytes;
use daemon_vhc_proto::{
    to_canonical_vec, verify_certified_sender, CertError, CertScope, Hash, PeerId, RevocationError,
    RevocationLedger, RunKeyCertificate, RunKeyRevocation, Signature,
};

/// The verdict on one inbound §12.1 frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InboundVerdict {
    /// Verified, in-sequence, first sighting: deliver `(channel, seq, sender, payload)`.
    Deliver {
        /// The channel the frame targets.
        channel: u64,
        /// The channel-scoped durable seq.
        seq: u64,
        /// The sender identity (the verified signer).
        sender: [u8; 32],
        /// The module-authored payload bytes.
        payload: Vec<u8>,
    },
    /// The envelope signature does not verify for the claimed sender — refused, never delivered.
    BadSignature(String),
    /// The payload does not hash to the signed `payload_hash` — refused, never delivered.
    TamperedPayload,
    /// The frame's scope names a different run/epoch than this attach — refused.
    ScopeMismatch(String),
    /// The frame is malformed (not the §12.1 `[envelope, payload, sig]` shape) — refused.
    Malformed(String),
    /// An already-accepted scope tuple — idempotently dropped.
    Duplicate {
        /// The sending peer.
        sender: [u8; 32],
        /// The channel.
        channel: u64,
        /// The duplicated seq.
        seq: u64,
    },
    /// A sequence gap in the sender's stream: `seq > expected`. The frame is HELD (§12.2 — no
    /// silent skip); backfill is Phase B's archive fetch.
    Gap {
        /// The sending peer.
        sender: [u8; 32],
        /// The channel.
        channel: u64,
        /// The next seq this attach expected from the sender on the channel.
        expected: u64,
        /// The seq the frame carries.
        got: u64,
    },
    /// The frame's signature verified over its `sender`, but that per-run key is **not certified**
    /// to a trusted base identity for this frame's full execution-identity scope (certified
    /// per-run keys, architecture §4.3). This is the signature-downgrade refusal — mandatory on
    /// every production attach.
    UncertifiedSender {
        /// The sending per-run key that lacks a valid certificate chain.
        sender: [u8; 32],
        /// The refusal reason (the underlying `CertError`).
        reason: String,
    },
    /// The sender's per-run key IS certified but is dead: explicitly revoked by a signed record,
    /// or its incarnation is superseded by a higher one for the same role slot.
    CertRevoked {
        /// The revoked per-run key.
        sender: [u8; 32],
    },
    /// Verified and in-sequence, but the pump's bounded spool (or this sender's quota) is full
    /// (§4.7): the frame was NOT delivered and the cursor was rewound — hold + retry the same
    /// frame; the reliable class never drops.
    Backpressure {
        /// The sending peer.
        sender: [u8; 32],
        /// The channel.
        channel: u64,
        /// The held frame's seq (the retry re-presents it).
        seq: u64,
    },
}

/// The certified-per-run-key layer every production attach carries (architecture §4.3): which
/// base identities the receiver trusts to certify per-run keys (from the run's genesis/Authority
/// configuration — never ambient config), the certificates it holds (distributed beside frames /
/// via join records), and the revocation state (explicit signed records + the incarnation
/// supersession floor).
pub struct CertCheck {
    /// The base machine identities whose certificates this attach trusts.
    trusted_bases: Vec<PeerId>,
    /// The certificates that authenticate per-run keys to a trusted base.
    certs: Vec<RunKeyCertificate>,
    /// Revocation state: explicit signed records + the supersession floor derived from the
    /// certificates observed above.
    revocations: RevocationLedger,
    /// The leadership-term floors for seat-governed roles ([SEAT-1] v2): fed only by verified
    /// seat grants ([`CertCheck::ingest_seat_grant`]) — a frame under a governed role from
    /// anyone but the claimant bound at the highest verified term is refused
    /// ([`CertError::SeatSuperseded`]), regardless of its certificate.
    seat_terms: daemon_vhc_proto::SeatTermLedger,
}

impl CertCheck {
    /// A certificate check trusting `trusted_bases` with the given starting certificate store.
    /// Every certificate's incarnation is observed into the supersession floor (a higher
    /// incarnation for a role slot implicitly revokes lower ones).
    #[must_use]
    pub fn new(trusted_bases: Vec<PeerId>, certs: Vec<RunKeyCertificate>) -> Self {
        let mut revocations = RevocationLedger::new();
        revocations.observe_certificates(&certs);
        Self {
            trusted_bases,
            certs,
            revocations,
            seat_terms: daemon_vhc_proto::SeatTermLedger::new(),
        }
    }

    /// Ingest a later-arriving certificate (control-plane distribution): stored for sender
    /// authentication and observed into the supersession floor. Idempotent — a re-delivered
    /// record (plane redundancy, resubscribe replay) changes nothing.
    ///
    /// TRUST GATE (caller-enforced by construction): the distribution handler verifies the
    /// record's chain AND that its base identity is genesis-trusted BEFORE calling this — an
    /// unverified record must never advance the supersession floor (a forged high incarnation
    /// would fence out the legitimate holder).
    pub fn ingest_certificate(&mut self, cert: RunKeyCertificate) {
        if self.certs.contains(&cert) {
            return;
        }
        self.revocations
            .observe_certificates(std::slice::from_ref(&cert));
        self.certs.push(cert);
    }

    /// Whether `base` is one of this attach's genesis-trusted certificate issuers (the
    /// distribution handler's trust gate input).
    #[must_use]
    pub fn trusts_base(&self, base: &PeerId) -> bool {
        self.trusted_bases.contains(base)
    }

    /// Ingest a distributed **seat grant** ([SEAT-1] v2 grant distribution): the
    /// observation-grade acceptance (structure, self-signature, genesis-trusted certificate
    /// chain over exactly the grant's scope and claimant — expiry deliberately excluded: the
    /// term floor is monotonic ownership HISTORY, expiry gates takeover liveness) plus the
    /// per-base revocation judgment, then the term floor advances and the embedded certificate
    /// joins the store (a grant is also how a late subscriber first learns the incumbent's
    /// cert). Idempotent; a stale (lower-term) grant is verified but changes no floor.
    ///
    /// # Errors
    /// A human-readable refusal — a refused grant changes no state.
    pub fn ingest_seat_grant(&mut self, grant: daemon_vhc_proto::SeatLease) -> Result<(), String> {
        grant
            .verify_grant(&self.trusted_bases)
            .map_err(|e| format!("seat grant: {e}"))?;
        self.revocations
            .judge(
                &grant.body.cert_scope(),
                &grant.body.claimant,
                &grant.certificate.base_identity,
            )
            .map_err(|e| format!("seat grant claimant: {e}"))?;
        self.ingest_certificate(grant.certificate.clone());
        self.seat_terms.observe_verified_grant(&grant);
        Ok(())
    }

    /// The highest verified leadership term for `(run, role)` (observability / tests).
    #[must_use]
    pub fn seat_term_floor(&self, run_id: &daemon_vhc_proto::Hash, role: &str) -> Option<u64> {
        self.seat_terms.floor(run_id, role)
    }

    /// Ingest a signed revocation record: accepted only from a trusted base and only with a
    /// strictly-monotonic per-slot sequence (replay protection).
    ///
    /// # Errors
    /// The applicable [`RevocationError`]; a refused record changes no state.
    pub fn ingest_revocation(&mut self, record: &RunKeyRevocation) -> Result<(), RevocationError> {
        let mut last_err = RevocationError::UntrustedBase;
        for base in &self.trusted_bases {
            match self.revocations.ingest(record, base) {
                Ok(()) => return Ok(()),
                Err(e) => last_err = e,
            }
        }
        Err(last_err)
    }

    /// Judge one verified frame's sender: certified to a trusted base for the full scope, and not
    /// revoked/superseded.
    fn judge(&self, scope: &CertScope, sender: &PeerId) -> Result<(), CertError> {
        let mut last = CertError::NoCertifiedChain;
        // The certifying base is retained for the revocation judgement: supersession ladders are
        // per base identity, so the sender is judged against the ladder of the identity that
        // certified it — never a roster sibling's.
        let mut certified: Option<PeerId> = None;
        for base in &self.trusted_bases {
            match verify_certified_sender(scope, sender, base, &self.certs) {
                Ok(()) => {
                    certified = Some(*base);
                    break;
                }
                Err(e) => last = e,
            }
        }
        let Some(certifying_base) = certified else {
            // Diagnostic: which certs are present and how their scopes compare to the frame's.
            // A live session that refuses a peer frame without saying why is undebuggable.
            if tracing::enabled!(tracing::Level::DEBUG) {
                let want = format!(
                    "run={} epoch={} role={} inst={} module={}",
                    hex8(&scope.run_id.0),
                    scope.epoch,
                    scope.role,
                    scope.instance,
                    hex8(&scope.module_hash.0),
                );
                let have: Vec<String> = self
                    .certs
                    .iter()
                    .map(|c| {
                        format!(
                            "[base={} run_key={} role={} inst={} epoch={} module={} trusted={}]",
                            hex8(&c.base_identity.0),
                            hex8(&c.body.run_key.0),
                            c.body.scope.role,
                            c.body.scope.instance,
                            c.body.scope.epoch,
                            hex8(&c.body.scope.module_hash.0),
                            self.trusted_bases.contains(&c.base_identity),
                        )
                    })
                    .collect();
                tracing::debug!(
                    sender = %hex8(&sender.0),
                    want = %want,
                    certs = ?have,
                    "cert judge: no certificate authenticates the frame sender"
                );
            }
            return Err(last);
        };
        self.revocations.judge(scope, sender, &certifying_base)?;
        // The cross-base leadership judgment ([SEAT-1] v2): when the frame's role is governed by
        // an observed verified seat grant, only the claimant bound at the highest verified term
        // may speak under it — a certified-but-fenced predecessor is refused here. Ungoverned
        // roles (no grant observed — every trainer role, and the coordinator before the first
        // grant arrives) pass: certification + the per-base ladders above remain their gate.
        if self.seat_terms.binds(&scope.run_id, &scope.role, sender) == Some(false) {
            return Err(CertError::SeatSuperseded);
        }
        Ok(())
    }
}

/// A short hex prefix for diagnostics (never a security surface).
fn hex8(bytes: &[u8; 32]) -> String {
    bytes[..4].iter().map(|b| format!("{b:02x}")).collect()
}

/// Inbound §12.1 verification state for one run attach: the expected run scope + per
/// `(sender, channel)` sequence cursors (the dedup/gap substrate, §12.2), plus the mandatory
/// certified-key layer.
pub struct InboundFrames {
    run_id: [u8; 32],
    epoch: u64,
    /// Next expected seq per `(sender, channel)` — everything below is a duplicate, everything
    /// above is a gap.
    cursors: HashMap<([u8; 32], u64), u64>,
    /// The certified-per-run-key layer. `Some` on every production construction path; `None`
    /// exists only behind the harness gate ([`InboundFrames::without_certs`]).
    cert_check: Option<CertCheck>,
}

impl InboundFrames {
    /// A verifier for one run scope. The certificate check is **mandatory**: every accepted
    /// frame's `sender` per-run key must be certified to a trusted base identity for the frame's
    /// full execution-identity scope, and must not be revoked or superseded. There is no
    /// production constructor that accepts an uncertified key.
    #[must_use]
    pub fn new(run_id: [u8; 32], epoch: u64, cert_check: CertCheck) -> Self {
        Self {
            run_id,
            epoch,
            cursors: HashMap::new(),
            cert_check: Some(cert_check),
        }
    }

    /// The certificate-less verifier — frame signature over `sender`, scope, dedup/gap, but NO
    /// certified-key requirement. Harness/test seat only: production paths must construct via
    /// [`InboundFrames::new`].
    #[cfg(any(test, feature = "harness"))]
    #[must_use]
    pub fn without_certs(run_id: [u8; 32], epoch: u64) -> Self {
        Self {
            run_id,
            epoch,
            cursors: HashMap::new(),
            cert_check: None,
        }
    }

    /// The mutable certificate layer (ingesting later-arriving certificates / revocations from
    /// the control plane). `None` only on the harness path.
    pub fn certs_mut(&mut self) -> Option<&mut CertCheck> {
        self.cert_check.as_mut()
    }

    /// Route one inbound §12.3 distribution record into the certificate layer (the records
    /// travel on the control plane beside frames). A certificate ingests only after its chain
    /// verifies AND its base identity is genesis-trusted — an unverified record must never
    /// advance the supersession floor (a forged high incarnation would fence out the legitimate
    /// holder). A revocation goes through the replay-protected ledger.
    ///
    /// # Errors
    /// A human-readable refusal — a refused record changes no state.
    pub fn ingest_distribution(
        &mut self,
        record: crate::distribution::DistributionRecord,
    ) -> Result<(), String> {
        let Some(certs) = self.cert_check.as_mut() else {
            return Err("no certificate layer on this attach".into());
        };
        match record {
            crate::distribution::DistributionRecord::Cert(cert) => {
                cert.verify_chain()
                    .map_err(|e| format!("certificate chain: {e}"))?;
                if !certs.trusts_base(&cert.base_identity) {
                    return Err("certificate base identity is not genesis-trusted".into());
                }
                certs.ingest_certificate(cert);
                Ok(())
            }
            crate::distribution::DistributionRecord::Revocation(record) => certs
                .ingest_revocation(&record)
                .map_err(|e| format!("revocation record: {e}")),
            crate::distribution::DistributionRecord::SeatGrant(grant) => {
                certs.ingest_seat_grant(*grant)
            }
        }
    }

    /// Verify one wire frame (`[envelope, payload, sig]`, §12.1) and judge its sequence position.
    /// Only [`InboundVerdict::Deliver`] advances the sender's cursor.
    pub fn accept(&mut self, frame: &[u8]) -> InboundVerdict {
        let v: Value = match ciborium::de::from_reader(frame) {
            Ok(v) => v,
            Err(e) => return InboundVerdict::Malformed(format!("frame cbor: {e}")),
        };
        let Value::Array(parts) = v else {
            return InboundVerdict::Malformed("frame is not [envelope, payload, sig]".into());
        };
        if parts.len() != 3 {
            return InboundVerdict::Malformed(format!("frame arity {}", parts.len()));
        }
        let Value::Map(env) = &parts[0] else {
            return InboundVerdict::Malformed("envelope is not a map".into());
        };
        let Value::Bytes(payload) = &parts[1] else {
            return InboundVerdict::Malformed("payload is not bytes".into());
        };
        let Value::Bytes(sig) = &parts[2] else {
            return InboundVerdict::Malformed("sig is not bytes".into());
        };
        let field = |name: &str| -> Option<&Value> {
            env.iter().find_map(|(k, v)| match k {
                Value::Text(t) if t == name => Some(v),
                _ => None,
            })
        };
        let bytes32 = |name: &str| -> Option<[u8; 32]> {
            match field(name) {
                Some(Value::Bytes(b)) => b.as_slice().try_into().ok(),
                _ => None,
            }
        };
        let uint = |name: &str| -> Option<u64> {
            field(name)
                .and_then(Value::as_integer)
                .and_then(|n| u64::try_from(i128::from(n)).ok())
        };
        let text = |name: &str| -> Option<String> {
            match field(name) {
                Some(Value::Text(t)) => Some(t.clone()),
                _ => None,
            }
        };
        let (Some(run_id), Some(sender), Some(payload_hash)) = (
            bytes32("run_id"),
            bytes32("sender"),
            bytes32("payload_hash"),
        ) else {
            return InboundVerdict::Malformed("missing run_id/sender/payload_hash".into());
        };
        let (Some(channel), Some(seq), Some(epoch)) = (uint("channel"), uint("seq"), uint("epoch"))
        else {
            return InboundVerdict::Malformed("missing channel/seq/epoch".into());
        };

        // 1. Signature over the canonical envelope, by the claimed sender (§12.1).
        let env_bytes = match to_canonical_vec(&parts[0]) {
            Ok(b) => b,
            Err(e) => return InboundVerdict::Malformed(format!("envelope encode: {e}")),
        };
        let Ok(sig) = <&[u8] as TryInto<[u8; 64]>>::try_into(sig.as_slice()) else {
            return InboundVerdict::Malformed("sig is not 64 bytes".into());
        };
        if let Err(e) = verify_bytes(&PeerId(sender), &Signature(sig), &env_bytes) {
            return InboundVerdict::BadSignature(e.to_string());
        }

        // 2. The payload is the one that was signed (its hash is inside the envelope).
        if blake3::hash(payload).as_bytes() != &payload_hash {
            return InboundVerdict::TamperedPayload;
        }

        // 3. Scope: this attach's run + epoch (§12.2 scope tuple).
        if run_id != self.run_id || epoch != self.epoch {
            return InboundVerdict::ScopeMismatch(format!(
                "frame scope (run {}, epoch {epoch}) is not this attach's (run {}, epoch {})",
                hex_prefix(&run_id),
                hex_prefix(&self.run_id),
                self.epoch,
            ));
        }

        // 3b. Certified per-run key (architecture §4.3) — mandatory on every production attach:
        // the per-run `sender` key (whose frame signature just verified — the retained check
        // above) MUST be certified to a trusted base identity for this frame's full
        // execution-identity scope (run, epoch, role, instance, module) AND must not be revoked
        // or superseded. An uncertified sender is the signature-downgrade refusal; a dead key is
        // the CertRevoked refusal. The scope fields are read from the frozen §12.1 envelope
        // (read-only — the envelope shape is untouched).
        if let Some(check) = &self.cert_check {
            let (Some(role), Some(instance), Some(module)) =
                (text("role"), uint("instance"), bytes32("module"))
            else {
                return InboundVerdict::Malformed(
                    "missing role/instance/module for the certified-key check".into(),
                );
            };
            let scope = CertScope {
                run_id: Hash(run_id),
                epoch,
                role,
                instance,
                module_hash: Hash(module),
            };
            match check.judge(&scope, &PeerId(sender)) {
                Ok(()) => {}
                Err(CertError::Revoked) => {
                    return InboundVerdict::CertRevoked { sender };
                }
                Err(e) => {
                    return InboundVerdict::UncertifiedSender {
                        sender,
                        reason: e.to_string(),
                    };
                }
            }
        }

        // 4. Sequence position per (sender, channel): dense from 0 (§12.2).
        self.judge(sender, channel, seq, payload)
    }

    /// Rewind a sender's channel cursor by one — the back-pressure path: an accepted-but-undelivered
    /// frame will be re-presented, and must be in-sequence again rather than a duplicate (§4.7).
    pub fn rewind(&mut self, sender: [u8; 32], channel: u64) {
        if let Some(cursor) = self.cursors.get_mut(&(sender, channel)) {
            *cursor = cursor.saturating_sub(1);
        }
    }

    fn judge(
        &mut self,
        sender: [u8; 32],
        channel: u64,
        seq: u64,
        payload: &[u8],
    ) -> InboundVerdict {
        // MID-STREAM ADOPTION: the first observed frame from an unseen `(sender, channel)` seats
        // the cursor at ITS seq. A late joiner (fresh incarnation restoring mid-run) cannot — and
        // need not — see a peer's stream from seq 0: its state comes from the checkpoint, the
        // relay replays no history, and insisting on density-from-zero turned every late attach
        // into an unrecoverable-gap terminal (a rejoin loop). Density (dedup/gap/hold, §12.2)
        // applies from the adopted point on; a replayed pre-adoption frame is at worst an
        // authentic duplicate the module's own round machinery already treats idempotently.
        let cursor = self.cursors.entry((sender, channel)).or_insert(seq);
        match seq.cmp(cursor) {
            std::cmp::Ordering::Less => InboundVerdict::Duplicate {
                sender,
                channel,
                seq,
            },
            std::cmp::Ordering::Greater => InboundVerdict::Gap {
                sender,
                channel,
                expected: *cursor,
                got: seq,
            },
            std::cmp::Ordering::Equal => {
                *cursor += 1;
                InboundVerdict::Deliver {
                    channel,
                    seq,
                    sender,
                    payload: payload.to_vec(),
                }
            }
        }
    }
}

fn hex_prefix(b: &[u8; 32]) -> String {
    b[..4].iter().map(|x| format!("{x:02x}")).collect()
}

/// The attach itself: verification above the pump, delivery below it. `deliver` returns the
/// verdict either way — only `Deliver` reaches the pump, with the original signed frame as the
/// tag-12 evidence the driver journals.
pub struct Attach {
    frames: InboundFrames,
    pump: daemon_vhc_host::run::PumpHandle,
}

impl Attach {
    /// Attach the verifier in front of a running v2 pump. The certificate check is mandatory —
    /// there is no production attach without one.
    #[must_use]
    pub fn new(
        run_id: [u8; 32],
        epoch: u64,
        cert_check: CertCheck,
        pump: daemon_vhc_host::run::PumpHandle,
    ) -> Self {
        Self {
            frames: InboundFrames::new(run_id, epoch, cert_check),
            pump,
        }
    }

    /// Route one inbound §12.3 distribution record into the certificate layer (see
    /// [`InboundFrames::ingest_distribution`]).
    ///
    /// # Errors
    /// A human-readable refusal for the caller's advisory surface — a refused record is a typed
    /// per-record event, never a session fault.
    pub fn ingest_distribution(
        &mut self,
        record: crate::distribution::DistributionRecord,
    ) -> Result<(), String> {
        self.frames.ingest_distribution(record)
    }

    /// Verify + (iff verified, first-sighted, in-sequence) deliver one inbound wire frame.
    ///
    /// The pump's reliable class is bounded and back-pressures rather than drops (§4.7): on a
    /// [`daemon_vhc_host::run::DeliverVerdict::SpoolFull`]/`SenderQuota` verdict this seam
    /// **rewinds the sender's sequence cursor** and returns [`InboundVerdict::Backpressure`], so
    /// the caller's retry of the very same frame is in-sequence again — never a duplicate, never
    /// a silent skip.
    ///
    /// # Errors
    /// A pump/journal error while delivering an already-verified frame.
    pub fn deliver(
        &mut self,
        frame: &[u8],
    ) -> Result<InboundVerdict, daemon_vhc_host::run::SinkError> {
        let verdict = self.frames.accept(frame);
        if let InboundVerdict::Deliver {
            channel,
            seq,
            sender,
            payload,
        } = &verdict
        {
            let pump_verdict = self.pump.deliver_frame(
                u32::try_from(*channel).unwrap_or(u32::MAX),
                *seq,
                *sender,
                payload.clone(),
                frame.to_vec(),
            )?;
            match pump_verdict {
                daemon_vhc_host::run::DeliverVerdict::Accepted => {}
                daemon_vhc_host::run::DeliverVerdict::SpoolFull
                | daemon_vhc_host::run::DeliverVerdict::SenderQuota => {
                    // Held, not delivered: rewind so the retry is in-sequence (§4.7).
                    self.frames.rewind(*sender, *channel);
                    return Ok(InboundVerdict::Backpressure {
                        sender: *sender,
                        channel: *channel,
                        seq: *seq,
                    });
                }
                daemon_vhc_host::run::DeliverVerdict::FrameTooLarge => {
                    return Ok(InboundVerdict::Malformed(
                        "frame exceeds the channel's max_frame_bytes".into(),
                    ));
                }
            }
        }
        Ok(verdict)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use daemon_vhc_proto::sign::{peer_id, sign_canonical};
    use daemon_vhc_proto::SigningKey;

    fn frame(key: &SigningKey, run: [u8; 32]) -> Vec<u8> {
        let sender = peer_id(key).0;
        let payload = b"hello".to_vec();
        let envelope = Value::Map(vec![
            (Value::from("domain"), Value::from("daemon-vhc/frame/2")),
            (Value::from("run_id"), Value::Bytes(run.to_vec())),
            (Value::from("epoch"), Value::from(0u64)),
            (Value::from("role"), Value::from("trainer")),
            (Value::from("instance"), Value::from(1u64)),
            (Value::from("module"), Value::Bytes(vec![0; 32])),
            (Value::from("sender"), Value::Bytes(sender.to_vec())),
            (Value::from("channel"), Value::from(0u64)),
            (Value::from("seq"), Value::from(0u64)),
            (
                Value::from("payload_hash"),
                Value::Bytes(blake3::hash(&payload).as_bytes().to_vec()),
            ),
        ]);
        let sig = sign_canonical(key, &envelope).expect("sign");
        let wire = Value::Array(vec![
            envelope,
            Value::Bytes(payload),
            Value::Bytes(sig.0.to_vec()),
        ]);
        let mut out = Vec::new();
        ciborium::into_writer(&wire, &mut out).expect("frame cbor");
        out
    }

    /// The certificate-less constructor is the HARNESS seat only (cfg-gated off production
    /// builds): it retains the frame-signature verifier but requires no certification.
    #[test]
    fn the_harness_path_without_certs_delivers_an_uncertified_sender() {
        let uncertified = SigningKey::from_bytes(&[42; 32]);
        let run = [0xA1; 32];
        let mut v = InboundFrames::without_certs(run, 0);
        assert!(matches!(
            v.accept(&frame(&uncertified, run)),
            InboundVerdict::Deliver { .. }
        ));
    }
}
