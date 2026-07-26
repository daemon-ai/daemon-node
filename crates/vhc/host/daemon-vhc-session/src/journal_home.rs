// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The node-delivered **durable journal home** — the per-incarnation on-disk journal a role
//! session binds instead of the in-memory sink (ABI §8; architecture §6.1 state-dir conventions).
//!
//! Layout, under the node state dir (beside `vhc.db` and `vhc/identity/`):
//!
//! ```text
//! <data_dir>/vhc/runs/                      the run-state root (DAEMON_VHC_RUN_DIR)
//!   <blake3(run label)>/                    one run's state home (label hashing per the keystore)
//!     <role>-<incarnation>/journal/         the §8 segmented journal for ONE incarnation
//!     payload/                              the run's filesystem payload plane (content-addressed)
//! ```
//!
//! Delivery follows the identity-store pattern ([`crate::keystore::IDENTITY_DIR_ENV`]): the node
//! exports the run-state ROOT as an inherited path reference ([`RUN_DIR_ENV`]); the worker
//! composes the per-incarnation path from the `(run label, role, incarnation)` it already holds.
//! Per-incarnation directories are what the node lifecycle needs later: a retained incarnation
//! (node restart) re-opens and APPENDS (the substrate's crash recovery), a new incarnation gets a
//! fresh home, and a reaped incarnation can be deleted atomically. Journals are the digest/replay
//! oracle's product input — they are NOT deleted on terminal completion (unlike key material).
//!
//! The sidecar encryption key (§8.5) is a node-local secret from the identity keystore
//! ([`crate::keystore::VhcKeystore::journal_sidecar_key`]) — a construction input here, never
//! minted here, never on the command wire.

use std::path::{Path, PathBuf};

use daemon_vhc_host::run::{Dropped, JournalSink, RunIdentity, SinkError};
use daemon_vhc_journal::record::{
    Body, ClockRec, CompletionRec, ConditionRec, DeviceProfileRec, DropId, DropRec, EventRec,
    ExecIdentity, ExecutionGrantRec, InitRec, InstantiationRec, RunHeader, SignedFrameRec,
    SnapshotRec, TerminalRec, TimerArmRec, TimerCancelRec, TrapInfo,
};
use daemon_vhc_journal::{format_version, Journal, JournalError, RotatePolicy, StaticKey};
use daemon_vhc_proto::Hash;

/// The environment variable through which the node hands a worker subprocess the run-state root
/// (a path reference — never secret material), mirroring the identity-store delivery.
pub const RUN_DIR_ENV: &str = "DAEMON_VHC_RUN_DIR";

/// The environment variable through which the node hands a worker subprocess an OVERRIDE root
/// for the filesystem payload plane (`[vhc] payload_dir`; a path reference like [`RUN_DIR_ENV`]).
/// Absent ⇒ the plane roots under the run's own state dir. A multi-node single-host deployment
/// shares one root so peers can serve each other's content-addressed objects — the filesystem
/// plane is the local stand-in for a shared object store (journals stay per-node).
pub const PAYLOAD_DIR_ENV: &str = "DAEMON_VHC_PAYLOAD_DIR";

/// The run-state root a worker was handed by reference, if any.
#[must_use]
pub fn run_dir_from_env() -> Option<PathBuf> {
    std::env::var_os(RUN_DIR_ENV).map(PathBuf::from)
}

/// The payload-plane override root a worker was handed by reference, if any.
#[must_use]
pub fn payload_dir_from_env() -> Option<PathBuf> {
    std::env::var_os(PAYLOAD_DIR_ENV).map(PathBuf::from)
}

/// One run's state home under the run-state root: `<root>/<blake3(run label)>/`. The label is
/// hashed exactly like the keystore's run directories (labels are free-form strings; hashing
/// keeps the path safe and collision-free — the journal's run header binds the cryptographic
/// run id, the label only namespaces storage).
#[must_use]
pub fn run_state_dir(root: &Path, run_label: &str) -> PathBuf {
    root.join(blake3::hash(run_label.as_bytes()).to_hex().as_str())
}

/// The per-incarnation journal directory: `<root>/<blake3(label)>/<role>-<incarnation>/journal`.
#[must_use]
pub fn journal_dir(root: &Path, run_label: &str, role: &str, incarnation: u64) -> PathBuf {
    run_state_dir(root, run_label)
        .join(format!("{role}-{incarnation}"))
        .join("journal")
}

/// The run's filesystem payload-plane directory: `<root>/<blake3(label)>/payload` — shared
/// across the run's incarnations (payloads are content-addressed; an incarnation change does not
/// invalidate content).
#[must_use]
pub fn payload_dir(root: &Path, run_label: &str) -> PathBuf {
    run_state_dir(root, run_label).join("payload")
}

/// The per-incarnation **state-store spill directory** (design §8.1):
/// `<root>/<blake3(label)>/<role>-<incarnation>/state`. The host state store spills canonical
/// det-lane chunk bytes here (content-addressed) so the retained roots live on disk rather than
/// the memory-floor peer's unified RAM. Per-incarnation like the journal: a fresh incarnation
/// gets a fresh spill; a reaped incarnation's spill can be deleted atomically. (Unlike the
/// content-addressed payload plane, the spill is instance-scoped — a torn/unsealed fold is never
/// durable, [SF-4].)
#[must_use]
pub fn state_dir(root: &Path, run_label: &str, role: &str, incarnation: u64) -> PathBuf {
    run_state_dir(root, run_label)
        .join(format!("{role}-{incarnation}"))
        .join("state")
}

/// The durable [`JournalSink`]: the §8 crash-safe segmented [`Journal`] adapted onto the
/// driver's dependency-inverted sink seam. Open-or-recover semantics: a fresh incarnation
/// creates segment 0; a resumed incarnation (node restart, retained incarnation) verifies the
/// chain, truncates a torn tail, and appends.
pub struct DurableSink {
    journal: Journal<StaticKey>,
    id: ExecIdentity,
}

impl DurableSink {
    /// Open (or crash-recover) the journal at `dir` for `identity`, with the node-local sidecar
    /// key (§8.5, from the identity keystore).
    ///
    /// # Errors
    /// [`JournalError`] on a broken chain, unreadable header, or filesystem failure. A run that
    /// cannot journal must not run (§8.4) — callers refuse the join typed on error.
    pub fn open(
        dir: &Path,
        identity: &RunIdentity,
        sidecar_key: [u8; 32],
    ) -> Result<Self, JournalError> {
        let id = ExecIdentity {
            run_id: Hash(identity.run_id),
            epoch: identity.epoch,
            role: identity.role.clone(),
            instance: identity.instance,
            module: Hash(identity.module),
        };
        let journal = Journal::open(
            dir,
            id.clone(),
            StaticKey::new(sidecar_key),
            RotatePolicy::default(),
        )?;
        Ok(Self { journal, id })
    }

    /// Open the journal at the **live-upgrade seam** (§8.1/§10.3): the retired incarnation's
    /// records remain as the prefix of one continued file series; the seam seals + rolls a
    /// segment so appends land under the incoming incarnation's identity header; the record
    /// ordinal stays globally monotone; the per-channel publish counters reset (the new signed
    /// stream opens at seq 0, §12.2). The incoming instance's own tag-0 run-header is written by
    /// the driver as the new span's first record.
    ///
    /// The retired incarnation's sink MUST have been dropped first (one writer per file series);
    /// the role session sequences that drop before invoking this.
    ///
    /// # Errors
    /// [`JournalError`] on a broken chain, an empty directory (nothing to continue), or
    /// filesystem failure.
    pub fn open_continuation(
        dir: &Path,
        identity: &RunIdentity,
        sidecar_key: [u8; 32],
    ) -> Result<Self, JournalError> {
        let id = ExecIdentity {
            run_id: Hash(identity.run_id),
            epoch: identity.epoch,
            role: identity.role.clone(),
            instance: identity.instance,
            module: Hash(identity.module),
        };
        let journal = Journal::open_continuation(
            dir,
            id.clone(),
            StaticKey::new(sidecar_key),
            RotatePolicy::default(),
        )?;
        Ok(Self { journal, id })
    }

    /// The journal root (observability / tests).
    #[must_use]
    pub fn root(&self) -> &Path {
        self.journal.paths().root()
    }
}

/// Map a substrate error onto the driver's sink error.
fn sink_err(e: JournalError) -> SinkError {
    SinkError(e.to_string())
}

impl JournalSink for DurableSink {
    fn run_header(
        &mut self,
        abi: u64,
        worlds: &[(String, u64)],
        bridge: bool,
        manifest: &[u8],
        config: &[u8],
        grants: &[u8],
        claim: &[u8],
        channels: &[u8],
        device: &[u8],
    ) -> Result<(), SinkError> {
        self.journal
            .append(Body::RunHeader(RunHeader {
                run_id: self.id.run_id,
                epoch: self.id.epoch,
                role: self.id.role.clone(),
                instance: self.id.instance,
                module: self.id.module,
                abi,
                worlds: worlds.iter().cloned().collect(),
                bridge,
                manifest: manifest.to_vec(),
                config: config.to_vec(),
                grants: grants.to_vec(),
                claim: claim.to_vec(),
                channels: channels.to_vec(),
                device: device.to_vec(),
                format: u64::from(format_version()),
            }))
            .map(|_| ())
            .map_err(sink_err)
    }

    fn instantiation(&mut self, counter: u64, reason: u64, at: u64) -> Result<(), SinkError> {
        // The counter is also the sidecar nonce input (§8.5) — thread it into the substrate.
        self.journal.set_instantiation_counter(counter);
        self.journal
            .append(Body::Instantiation(InstantiationRec {
                counter,
                reason,
                at,
            }))
            .map(|_| ())
            .map_err(sink_err)
    }

    fn init(
        &mut self,
        config_hash: [u8; 32],
        grants_hash: [u8; 32],
        status: u64,
    ) -> Result<(), SinkError> {
        self.journal
            .append(Body::Init(InitRec {
                config_hash: Hash(config_hash),
                grants_hash: Hash(grants_hash),
                status,
            }))
            .map(|_| ())
            .map_err(sink_err)
    }

    fn execution_grant(
        &mut self,
        execution_grant_hash: [u8; 32],
        status: u64,
    ) -> Result<(), SinkError> {
        self.journal
            .append(Body::ExecutionGrant(ExecutionGrantRec {
                execution_grant_hash: Hash(execution_grant_hash),
                status,
            }))
            .map(|_| ())
            .map_err(sink_err)
    }

    fn event(&mut self, at: u64, frame: &[u8]) -> Result<(), SinkError> {
        self.journal
            .append(Body::Event(EventRec {
                at,
                frame: frame.to_vec(),
            }))
            .map(|_| ())
            .map_err(sink_err)
    }

    fn signed_frame(
        &mut self,
        channel: u64,
        seq: u64,
        sender: [u8; 32],
        frame: &[u8],
    ) -> Result<(), SinkError> {
        self.journal
            .append(Body::SignedFrame(SignedFrameRec {
                channel,
                seq,
                sender: Hash(sender),
                frame: Some(frame.to_vec()),
                evidence: None,
            }))
            .map(|_| ())
            .map_err(sink_err)
    }

    fn next_seq(&mut self, channel: u64) -> u64 {
        self.journal.next_seq(channel)
    }

    fn publish(
        &mut self,
        channel: u64,
        seq: u64,
        payload: &[u8],
        frame: &[u8],
    ) -> Result<(), SinkError> {
        // The substrate allocates the durable seq itself under the same §8.4 barrier; the driver
        // derived its seq from `next_seq`, so the two must agree. A disagreement is a session
        // wiring bug — surfaced as a typed sink error (aborting the publish), never a panic in
        // the production path.
        let (_, journal_seq) = self
            .journal
            .publish(channel, payload, frame.to_vec())
            .map_err(sink_err)?;
        if journal_seq != seq {
            return Err(SinkError(format!(
                "driver/journal publish seq divergence on channel {channel}: driver {seq}, \
                 journal {journal_seq} (§8.4 rule 2)"
            )));
        }
        Ok(())
    }

    fn clock(&mut self, now: u64) -> Result<(), SinkError> {
        self.journal
            .append(Body::Clock(ClockRec { now }))
            .map(|_| ())
            .map_err(sink_err)
    }

    fn timer_arm(&mut self, id: u64, delay: u64, armed_at: u64) -> Result<(), SinkError> {
        self.journal
            .append(Body::TimerArm(TimerArmRec {
                id,
                delay,
                armed_at,
            }))
            .map(|_| ())
            .map_err(sink_err)
    }

    fn timer_cancel(&mut self, id: u64, status: u64) -> Result<(), SinkError> {
        self.journal
            .append(Body::TimerCancel(TimerCancelRec { id, status }))
            .map(|_| ())
            .map_err(sink_err)
    }

    fn read_back(
        &mut self,
        src: u64,
        kind: u64,
        status: u64,
        value: &[u8],
    ) -> Result<(), SinkError> {
        // The substrate routes oversize values to encrypted sidecars itself (§8.5).
        self.journal
            .read_back(src, kind, status, value)
            .map(|_| ())
            .map_err(sink_err)
    }

    fn device_profile(&mut self, profile: &[u8]) -> Result<(), SinkError> {
        self.journal
            .append(Body::DeviceProfile(DeviceProfileRec {
                profile: profile.to_vec(),
            }))
            .map(|_| ())
            .map_err(sink_err)
    }

    fn drop_coalesced(&mut self, class: u64, rule: u64, dropped: Dropped) -> Result<(), SinkError> {
        self.journal
            .append(Body::Drop(DropRec {
                class,
                rule,
                dropped: DropId {
                    hash: dropped.hash.map(Hash),
                    timer_id: dropped.timer_id,
                    channel: dropped.channel,
                    sender: dropped.sender.map(Hash),
                    seq: dropped.seq,
                },
            }))
            .map(|_| ())
            .map_err(sink_err)
    }

    fn condition(&mut self, code: &str, detail: &str) -> Result<(), SinkError> {
        self.journal
            .append(Body::Condition(ConditionRec {
                code: code.to_string(),
                detail: detail.to_string(),
            }))
            .map(|_| ())
            .map_err(sink_err)
    }

    fn completion(&mut self, op: u64, result: &[u8]) -> Result<(), SinkError> {
        self.journal
            .append(Body::Completion(CompletionRec {
                op,
                result: result.to_vec(),
            }))
            .map(|_| ())
            .map_err(sink_err)
    }

    fn snapshot(&mut self, manifest: &[u8]) -> Result<(), SinkError> {
        // tag 10, committed: §8.4 rule 2 — the barrier crosses before `snapshot_state` returns
        // `Accepted` to the guest (the upgrade transaction's durability point).
        self.journal
            .append_committed(Body::Snapshot(SnapshotRec {
                manifest: manifest.to_vec(),
            }))
            .map(|_| ())
            .map_err(sink_err)
    }

    fn terminal(
        &mut self,
        kind: u64,
        outcome: Option<u64>,
        trap: Option<(String, String, String, String)>,
    ) -> Result<(), SinkError> {
        self.journal
            .append_committed(Body::Terminal(TerminalRec {
                kind,
                outcome,
                trap: trap.map(|(code, import, context, detail)| TrapInfo {
                    code,
                    import,
                    context,
                    detail,
                }),
            }))
            .map(|_| ())
            .map_err(sink_err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use daemon_vhc_journal::record::Record;
    use daemon_vhc_journal::scan_file;

    fn identity() -> RunIdentity {
        RunIdentity {
            run_id: [0x1D; 32],
            epoch: 0,
            role: "trainer".into(),
            instance: 3,
            module: [0x2A; 32],
        }
    }

    #[test]
    fn layout_paths_hash_the_label_and_key_the_incarnation() {
        let root = Path::new("/state/vhc/runs");
        let hashed = blake3::hash(b"run-x").to_hex().to_string();
        assert_eq!(run_state_dir(root, "run-x"), root.join(&hashed));
        assert_eq!(
            journal_dir(root, "run-x", "trainer", 7),
            root.join(&hashed).join("trainer-7").join("journal")
        );
        assert_eq!(
            payload_dir(root, "run-x"),
            root.join(&hashed).join("payload")
        );
        assert_eq!(
            state_dir(root, "run-x", "trainer", 7),
            root.join(&hashed).join("trainer-7").join("state")
        );
    }

    #[test]
    fn durable_sink_writes_recover_across_reopen() {
        // The crash-resume half of the incarnation policy: the SAME incarnation re-opens its
        // journal and appends; the records written before the "crash" are still there.
        let dir = tempfile::tempdir().unwrap();
        let jdir = journal_dir(dir.path(), "run-x", "trainer", 3);
        let key = [0x5C; 32];

        {
            let mut sink = DurableSink::open(&jdir, &identity(), key).expect("fresh journal");
            sink.run_header(
                2 << 16,
                &[("vhc".into(), 2)],
                false,
                b"m",
                b"c",
                b"g",
                b"cl",
                b"ch",
                b"d",
            )
            .unwrap();
            sink.event(1, b"frame-1").unwrap();
            sink.publish(0, 0, b"payload", b"signed-frame").unwrap();
        }

        let mut sink = DurableSink::open(&jdir, &identity(), key).expect("recovered journal");
        // The durable seq counter recovered: channel 0's next publish seq is 1.
        assert_eq!(sink.next_seq(0), 1);
        sink.event(2, b"frame-2").unwrap();

        // The on-disk records are scannable and in order.
        let scan = scan_file(jdir.join("segment-00000000.dvhcjrn")).expect("scan");
        let tags: Vec<u8> = scan.records.iter().map(Record::tag).collect();
        assert!(
            tags.len() >= 4,
            "header + event + publish + post-recovery event, got {tags:?}"
        );
    }

    #[test]
    fn continuation_rolls_a_segment_and_opens_a_fresh_stream_under_the_new_identity() {
        // The live-upgrade seam (§8.1/§10.3): one continued file series — the retired
        // incarnation's records stay as the prefix, the seam forces a segment roll, the new
        // segment header carries the incoming identity, the record ordinal stays monotone
        // across the seam, and the per-channel publish seq restarts at 0 (a fresh §12.2 stream).
        let dir = tempfile::tempdir().unwrap();
        let jdir = journal_dir(dir.path(), "run-z", "trainer", 3);
        let key = [0x7E; 32];
        let old = identity();

        {
            let mut sink = DurableSink::open(&jdir, &old, key).expect("fresh journal");
            sink.run_header(
                2 << 16,
                &[("vhc".into(), 2)],
                false,
                b"m",
                b"c",
                b"g",
                b"cl",
                b"ch",
                b"d",
            )
            .unwrap();
            sink.publish(0, 0, b"p0", b"f0").unwrap();
            sink.publish(0, 1, b"p1", b"f1").unwrap();
            sink.snapshot(b"drain-manifest").unwrap();
            sink.terminal(0, Some(2), None).unwrap();
        } // the retired incarnation's sink drops before the seam opens (one writer)

        let new = RunIdentity {
            epoch: 1,
            instance: 4,
            module: [0x3B; 32],
            ..identity()
        };
        let mut sink =
            DurableSink::open_continuation(&jdir, &new, key).expect("continuation at the seam");
        // The new signed stream opens at seq 0 — never inheriting the retired counter.
        assert_eq!(sink.next_seq(0), 0, "publish seq restarts at the seam");
        sink.run_header(
            2 << 16,
            &[("vhc".into(), 2)],
            false,
            b"m2",
            b"c2",
            b"g2",
            b"cl2",
            b"ch2",
            b"d2",
        )
        .unwrap();
        sink.publish(0, 0, b"q0", b"g0").unwrap();

        // On disk: the seam sealed the old segment and the new head segment carries the NEW
        // identity; ordinals stay monotone across the whole series.
        let scans: Vec<_> = (0..=1)
            .map(|n| scan_file(jdir.join(format!("segment-{n:08}.dvhcjrn"))).expect("scan"))
            .collect();
        assert!(scans[0].sealed, "the seam seals the retired span's segment");
        assert_eq!(scans[0].header.id.instance, 3);
        assert_eq!(scans[1].header.id.instance, 4);
        assert_eq!(scans[1].header.id.epoch, 1);
        let mut ords = Vec::new();
        for scan in &scans {
            for r in &scan.records {
                if !matches!(r.body, daemon_vhc_journal::Body::Seal(_)) {
                    ords.push(r.ord);
                }
            }
        }
        let mut sorted = ords.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(ords, sorted, "record ordinal monotone across the seam");

        // Crash recovery AFTER the seam re-keys at the last run-header: channel 0's next seq is
        // 1 (the new span's publish), not 2 (the retired span's high-water mark).
        drop(sink);
        let mut sink = DurableSink::open(&jdir, &new, key).expect("recovered continuation");
        assert_eq!(
            sink.next_seq(0),
            1,
            "recovery re-keys at the seam run-header"
        );
    }

    #[test]
    fn publish_seq_divergence_is_a_typed_sink_error() {
        let dir = tempfile::tempdir().unwrap();
        let jdir = journal_dir(dir.path(), "run-y", "trainer", 1);
        let mut sink = DurableSink::open(&jdir, &identity(), [0; 32]).unwrap();
        // The journal will allocate seq 0; a driver claiming 9 is a wiring bug, surfaced typed.
        let err = sink.publish(0, 9, b"p", b"f").unwrap_err();
        assert!(err.to_string().contains("seq divergence"));
    }
}
