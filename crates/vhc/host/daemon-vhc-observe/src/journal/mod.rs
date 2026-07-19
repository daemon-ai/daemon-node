// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The journal surface: the crash-safe segmented SUBSTRATE (re-exported from
//! [`daemon_vhc_journal`]) plus the oracle/archive tooling that decodes it.
//!
//! The substrate — the §8.3 record grammar ([`record`]), §8.2 segments ([`segment`]), §8.5
//! encrypted sidecars ([`sidecar`]), and the commit-barrier [`Journal`] store — was extracted to
//! the schema-free `daemon-vhc-journal` crate so production crates (the role session's durable
//! sink) can link the store without this crate's oracle tooling (observe decodes SDK round
//! schemas, which no production host graph may link — ABI §12.5 [OWN-3]). Everything is
//! re-exported here under the original paths, so existing consumers and suites are unchanged.
//!
//! What REMAINS here (oracle tooling over the substrate):
//! * [`archive`] — the coordinator sealed record archive: attested chain heads, fork detection,
//!   replication/retention policy.
//! * [`consensus`] — consensus replay from an archive through the sandboxed coordinator module.
//! * [`oracle`] — the coordinator-oracle migration: the replay oracle over journal-backed capture.
//! * [`verifier`] — the worker input-replay verifier skeleton (types + a sim-fed harness shape).

pub mod archive;
pub mod consensus;
pub mod oracle;
pub mod verifier;

// The substrate, at its original paths (`journal::record`, `journal::segment`, …).
pub use daemon_vhc_journal::{record, segment, sidecar, store};

pub use archive::{
    detect_fork, ArchiveError, AttestedHead, ChainHead, ForkEvidence, RecordArchive,
    ReplicationPolicy, RetentionPolicy,
};
pub use consensus::{
    extract_consensus_capture, recover_chain_from_archive, replay_consensus_from_archive,
    ConsensusCapture, ConsensusReplayError, ConsensusReplayReport, RecoveredChain,
};
pub use daemon_vhc_journal::{
    format_version, scan_bytes, scan_file, Body, ExecIdentity, Journal, JournalError, JournalPaths,
    KeyProvider, Record, RotatePolicy, ScanResult, SegmentHeader, SegmentWriter, SidecarError,
    SidecarStore, StaticKey,
};
