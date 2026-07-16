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

use std::collections::HashMap;

use ciborium::value::Value;
use daemon_vhc_proto::sign::verify_bytes;
use daemon_vhc_proto::{
    to_canonical_vec, verify_certified_sender, Hash, PeerId, RunKeyCertificate, Signature,
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
    /// to the trusted base identity for this frame's `(run, role, instance, epoch)` scope (D1
    /// certified per-run keys, architecture §4.3). Only surfaced when the attach was configured
    /// with a certificate store ([`InboundFrames::with_certs`]); the certificate-less transition
    /// path (the retained A2 verifier) never produces it. This is the signature-downgrade refusal.
    UncertifiedSender {
        /// The sending per-run key that lacks a valid certificate chain.
        sender: [u8; 32],
        /// The refusal reason (the underlying `CertError`).
        reason: String,
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

/// The D1 certified-per-run-key check layered **around** the A2 frame verifier (architecture §4.3):
/// which base identity a receiver trusts to certify per-run keys, and the certificates it holds
/// (distributed beside frames / via join records). Configured only on the cert-aware path; absent on
/// the retained transition path.
pub struct CertCheck {
    /// The base machine identity whose certificates this attach trusts (e.g. the coordinator's).
    pub trusted_base: PeerId,
    /// The certificates that authenticate per-run keys to `trusted_base`.
    pub certs: Vec<RunKeyCertificate>,
}

/// Inbound §12.1 verification state for one run attach: the expected run scope + per
/// `(sender, channel)` sequence cursors (the dedup/gap substrate, §12.2), plus the optional D1
/// certified-key layer.
pub struct InboundFrames {
    run_id: [u8; 32],
    epoch: u64,
    /// Next expected seq per `(sender, channel)` — everything below is a duplicate, everything
    /// above is a gap.
    cursors: HashMap<([u8; 32], u64), u64>,
    /// The optional D1 certified-per-run-key layer (`None` = the retained A2 verifier, the
    /// transition path — the frame signature over `sender` is checked, but `sender` is not required
    /// to be certified).
    cert_check: Option<CertCheck>,
}

impl InboundFrames {
    /// A verifier for one run scope — the **retained A2 verifier** (frame signature over `sender`,
    /// scope, dedup/gap). No certified-key requirement: the transition path, behaviorally identical
    /// to before D1.
    #[must_use]
    pub fn new(run_id: [u8; 32], epoch: u64) -> Self {
        Self {
            run_id,
            epoch,
            cursors: HashMap::new(),
            cert_check: None,
        }
    }

    /// A verifier that additionally requires every accepted frame's `sender` per-run key to be
    /// **certified** to `trusted_base` for the frame's `(run, role, instance, epoch)` scope (D1
    /// certified per-run keys, architecture §4.3). Everything the A2 verifier checks still runs
    /// first; the certified-key check is an additional guard that turns an uncertified sender into
    /// [`InboundVerdict::UncertifiedSender`] (the signature-downgrade refusal) rather than a
    /// delivery.
    #[must_use]
    pub fn with_certs(
        run_id: [u8; 32],
        epoch: u64,
        trusted_base: PeerId,
        certs: Vec<RunKeyCertificate>,
    ) -> Self {
        Self {
            run_id,
            epoch,
            cursors: HashMap::new(),
            cert_check: Some(CertCheck {
                trusted_base,
                certs,
            }),
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

        // 3b. Certified per-run key (D1, architecture §4.3): when a cert store is configured, the
        // per-run `sender` key (whose frame signature just verified — the retained A2 check above)
        // MUST be certified to the trusted base identity for this frame's (run, role, instance,
        // epoch) scope. An uncertified sender is the signature-downgrade refusal. The `role` and
        // `instance` are read from the frozen §12.1 envelope fields (read-only — the envelope shape
        // is untouched).
        if let Some(check) = &self.cert_check {
            let (Some(role), Some(instance)) = (text("role"), uint("instance")) else {
                return InboundVerdict::Malformed(
                    "missing role/instance for the certified-key check".into(),
                );
            };
            if let Err(e) = verify_certified_sender(
                &Hash(run_id),
                &role,
                instance,
                epoch,
                &PeerId(sender),
                &check.trusted_base,
                &check.certs,
            ) {
                return InboundVerdict::UncertifiedSender {
                    sender,
                    reason: e.to_string(),
                };
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
        let cursor = self.cursors.entry((sender, channel)).or_insert(0);
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
pub struct V2Attach {
    frames: InboundFrames,
    pump: daemon_vhc_host::v2::PumpHandle,
}

impl V2Attach {
    /// Attach the verifier in front of a running v2 pump.
    #[must_use]
    pub fn new(run_id: [u8; 32], epoch: u64, pump: daemon_vhc_host::v2::PumpHandle) -> Self {
        Self {
            frames: InboundFrames::new(run_id, epoch),
            pump,
        }
    }

    /// Verify + (iff verified, first-sighted, in-sequence) deliver one inbound wire frame.
    ///
    /// The pump's reliable class is bounded and back-pressures rather than drops (§4.7): on a
    /// [`daemon_vhc_host::v2::DeliverVerdict::SpoolFull`]/`SenderQuota` verdict this seam
    /// **rewinds the sender's sequence cursor** and returns [`InboundVerdict::Backpressure`], so
    /// the caller's retry of the very same frame is in-sequence again — never a duplicate, never
    /// a silent skip.
    ///
    /// # Errors
    /// A pump/journal error while delivering an already-verified frame.
    pub fn deliver(
        &mut self,
        frame: &[u8],
    ) -> Result<InboundVerdict, daemon_vhc_host::v2::SinkError> {
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
                daemon_vhc_host::v2::DeliverVerdict::Accepted => {}
                daemon_vhc_host::v2::DeliverVerdict::SpoolFull
                | daemon_vhc_host::v2::DeliverVerdict::SenderQuota => {
                    // Held, not delivered: rewind so the retry is in-sequence (§4.7).
                    self.frames.rewind(*sender, *channel);
                    return Ok(InboundVerdict::Backpressure {
                        sender: *sender,
                        channel: *channel,
                        seq: *seq,
                    });
                }
                daemon_vhc_host::v2::DeliverVerdict::FrameTooLarge => {
                    return Ok(InboundVerdict::Malformed(
                        "frame exceeds the channel's max_frame_bytes".into(),
                    ));
                }
            }
        }
        Ok(verdict)
    }
}
