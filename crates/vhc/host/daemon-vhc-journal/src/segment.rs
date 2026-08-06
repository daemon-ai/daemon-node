// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Segment layout, framing, the BLAKE3 chain + CRC32C, the seal record, and crash recovery
//! (ABI companion §8.2).
//!
//! ```text
//! segment file = header || record* || seal-record(optional, on clean roll)
//!
//! header  = magic "DVHCJRN2" (8)
//!         || u32-LE format_version (= 1)
//!         || prev_segment_blake3 (32; all-zero for the first segment)
//!         || u32-LE len || header-body-CBOR || u32-LE CRC32C(header-body-CBOR)
//!
//! record framing = u32-LE len || record-CBOR || u32-LE CRC32C(record-CBOR)
//! ```
//!
//! * **Chaining** (§8.2): each header carries the blake3 of the *complete previous segment file*; the
//!   final `seal` record (§8.3 tag 17) of a cleanly-rolled segment carries the blake3 of its own
//!   segment's bytes **from the start of the file up to but excluding the seal record's own framing**
//!   — the seal hash never covers itself. This chain is the substrate the Phase D archive signs.
//! * **Crash recovery** (§8.2): on open, [`scan_segment`] validates length + CRC32C of each frame;
//!   the first torn/corrupt frame truncates the segment there (the tail is discarded), yielding a
//!   clean recovery point with no silent corruption past a CRC/chain break.

// Sanctioned raw-fs home (see journal/mod.rs): host-owned journal path + low-level fdatasync /
// set_len durability the ContainedRoot API does not model. No spawn / env mutation here.
#![allow(clippy::disallowed_methods)]

use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use ciborium::value::Value;

use daemon_vhc_abi::{JOURNAL_FORMAT_VERSION, JOURNAL_SEGMENT_MAGIC};
use daemon_vhc_proto::to_canonical_vec;

use super::record::{Body, ExecIdentity, Record, SealRec};
use super::{fsync_dir, JournalError};

/// The 32-byte all-zero `prev_segment_blake3` of the first segment (§8.2).
pub const GENESIS_PREV: [u8; 32] = [0u8; 32];

/// A segment header: the execution identity (§8.1), the 0-based segment ordinal, and the chain link
/// to the previous segment (§8.2).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SegmentHeader {
    /// The frozen execution identity keying this journal.
    pub id: ExecIdentity,
    /// The 0-based segment ordinal.
    pub segment: u64,
    /// blake3 of the complete previous segment file (all-zero for the first segment).
    pub prev_blake3: [u8; 32],
}

impl SegmentHeader {
    /// The canonical-CBOR header body (`segment-header-body`, ABI §8.2 / journal.cddl).
    fn body_cbor(&self) -> Result<Vec<u8>, JournalError> {
        // Explicit string keys, matching the `segment-header-body` grammar exactly.
        let map = Value::Map(vec![
            (
                Value::Text("run_id".into()),
                Value::Bytes(self.id.run_id.as_bytes().to_vec()),
            ),
            (
                Value::Text("epoch".into()),
                Value::Integer(self.id.epoch.into()),
            ),
            (
                Value::Text("role".into()),
                Value::Text(self.id.role.clone()),
            ),
            (
                Value::Text("instance".into()),
                Value::Integer(self.id.instance.into()),
            ),
            (
                Value::Text("module".into()),
                Value::Bytes(self.id.module.as_bytes().to_vec()),
            ),
            (
                Value::Text("segment".into()),
                Value::Integer(self.segment.into()),
            ),
        ]);
        to_canonical_vec(&map).map_err(|e| JournalError::Codec(format!("encode header: {e}")))
    }

    /// The complete on-disk header bytes (magic || version || prev || len || body || CRC).
    pub fn encode(&self) -> Result<Vec<u8>, JournalError> {
        let body = self.body_cbor()?;
        let mut out = Vec::with_capacity(8 + 4 + 32 + 4 + body.len() + 4);
        out.extend_from_slice(JOURNAL_SEGMENT_MAGIC);
        out.extend_from_slice(&JOURNAL_FORMAT_VERSION.to_le_bytes());
        out.extend_from_slice(&self.prev_blake3);
        let len = u32::try_from(body.len())
            .map_err(|_| JournalError::Codec("header body exceeds u32".into()))?;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&body);
        out.extend_from_slice(&crc32c::crc32c(&body).to_le_bytes());
        Ok(out)
    }

    /// Parse a header from the start of `bytes`, returning the header + the number of bytes consumed.
    pub fn decode(bytes: &[u8]) -> Result<(Self, usize), JournalError> {
        let mut off = 0usize;
        let need = |off: usize, n: usize| -> Result<(), JournalError> {
            if off + n > bytes.len() {
                Err(JournalError::BadHeader("truncated header".into()))
            } else {
                Ok(())
            }
        };
        need(off, 8)?;
        if &bytes[off..off + 8] != JOURNAL_SEGMENT_MAGIC.as_slice() {
            return Err(JournalError::BadHeader("bad magic".into()));
        }
        off += 8;
        need(off, 4)?;
        let version = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
        off += 4;
        if version != JOURNAL_FORMAT_VERSION {
            return Err(JournalError::BadHeader(format!(
                "unsupported format version {version}"
            )));
        }
        need(off, 32)?;
        let mut prev = [0u8; 32];
        prev.copy_from_slice(&bytes[off..off + 32]);
        off += 32;
        need(off, 4)?;
        let len = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) as usize;
        off += 4;
        need(off, len)?;
        let body = &bytes[off..off + len];
        off += len;
        need(off, 4)?;
        let crc = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
        off += 4;
        if crc32c::crc32c(body) != crc {
            return Err(JournalError::BadHeader("header CRC mismatch".into()));
        }
        let value: Value = ciborium::de::from_reader(body)
            .map_err(|e| JournalError::BadHeader(format!("header body decode: {e}")))?;
        let id = decode_header_body(&value)?;
        Ok((
            SegmentHeader {
                id: id.0,
                segment: id.1,
                prev_blake3: prev,
            },
            off,
        ))
    }
}

fn decode_header_body(v: &Value) -> Result<(ExecIdentity, u64), JournalError> {
    #[derive(serde::Deserialize)]
    struct Raw {
        run_id: daemon_vhc_proto::Hash,
        epoch: u64,
        role: String,
        instance: u64,
        module: daemon_vhc_proto::Hash,
        segment: u64,
    }
    let raw: Raw = v
        .deserialized()
        .map_err(|e| JournalError::BadHeader(format!("header body shape: {e}")))?;
    Ok((
        ExecIdentity {
            run_id: raw.run_id,
            epoch: raw.epoch,
            role: raw.role,
            instance: raw.instance,
            module: raw.module,
        },
        raw.segment,
    ))
}

/// Encode a single record frame: `u32-LE len || record-CBOR || u32-LE CRC32C` (§8.2).
fn frame(record_cbor: &[u8]) -> Result<Vec<u8>, JournalError> {
    let len = u32::try_from(record_cbor.len())
        .map_err(|_| JournalError::Codec("record exceeds u32 frame length".into()))?;
    let mut out = Vec::with_capacity(4 + record_cbor.len() + 4);
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(record_cbor);
    out.extend_from_slice(&crc32c::crc32c(record_cbor).to_le_bytes());
    Ok(out)
}

/// An append-only writer for one segment file.
///
/// Records are framed (len + CRC32C) and appended; [`SegmentWriter::commit`] is the §8.4 fdatasync
/// commit barrier. The writer keeps a running BLAKE3 of the entire file so it can (a) emit a
/// self-excluding `seal` record and (b) hand the next segment its `prev_blake3` chain link.
pub struct SegmentWriter {
    file: File,
    path: PathBuf,
    /// blake3 of every byte written so far (header + all frames).
    hasher: blake3::Hasher,
    records: u64,
    sealed: bool,
}

impl SegmentWriter {
    /// Create + open a new segment file, writing its header and durably linking its directory entry
    /// (§8.4: on segment creation, also the directory entry).
    ///
    /// # Errors
    /// [`JournalError::Io`] on any filesystem failure.
    pub fn create(path: impl AsRef<Path>, header: &SegmentHeader) -> Result<Self, JournalError> {
        let path = path.as_ref().to_path_buf();
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .read(true)
            .open(&path)?;
        let header_bytes = header.encode()?;
        file.write_all(&header_bytes)?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(&header_bytes);
        // Durability of the fresh segment file's directory entry (§8.4).
        file.sync_all()?;
        if let Some(dir) = path.parent() {
            fsync_dir(dir)?;
        }
        Ok(Self {
            file,
            path,
            hasher,
            records: 0,
            sealed: false,
        })
    }

    /// Re-open an existing, **unsealed** segment for continued appends after crash recovery. The file
    /// is truncated to `intact_prefix.len()` (discarding any torn tail — §8.2) and the running hash is
    /// re-seeded from the intact prefix, so the chain + a future seal stay correct.
    ///
    /// # Errors
    /// [`JournalError::Io`] on any filesystem failure.
    pub fn reopen(
        path: impl AsRef<Path>,
        intact_prefix: &[u8],
        records: u64,
    ) -> Result<Self, JournalError> {
        let path = path.as_ref().to_path_buf();
        let file = OpenOptions::new().write(true).read(true).open(&path)?;
        // Truncate any torn/corrupt tail to the last durable frame boundary.
        file.set_len(intact_prefix.len() as u64)?;
        file.sync_all()?;
        if let Some(dir) = path.parent() {
            fsync_dir(dir)?;
        }
        // Seek to the (new) end so appends land after the intact prefix.
        let mut file = file;
        io::Seek::seek(&mut file, io::SeekFrom::End(0))?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(intact_prefix);
        Ok(Self {
            file,
            path,
            hasher,
            records,
            sealed: false,
        })
    }

    /// The segment file path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The number of records appended (excluding a seal).
    #[must_use]
    pub fn records(&self) -> u64 {
        self.records
    }

    /// Append a record (write, not necessarily commit — §8.4 rule 3/4).
    ///
    /// # Errors
    /// [`JournalError::Codec`] if the record cannot be encoded; [`JournalError::Io`] on write failure.
    pub fn append(&mut self, record: &Record) -> Result<(), JournalError> {
        let framed = Self::encode(record)?;
        self.append_framed(&framed)
    }

    /// Encode one record to its on-disk framed bytes — the exact byte count an append will
    /// write, so a capacity custodian can reserve it before the write (the [`super::Journal`]'s
    /// reservation seam) without encoding twice.
    ///
    /// # Errors
    /// [`JournalError::Codec`] if the record cannot be encoded.
    pub fn encode(record: &Record) -> Result<Vec<u8>, JournalError> {
        frame(&record.to_canonical()?)
    }

    /// Append pre-encoded framed bytes (the [`Self::encode`] product).
    ///
    /// # Errors
    /// [`JournalError::Io`] on write failure or a sealed segment.
    pub fn append_framed(&mut self, framed: &[u8]) -> Result<(), JournalError> {
        if self.sealed {
            return Err(JournalError::Io(io::Error::other(
                "append to a sealed segment",
            )));
        }
        self.file.write_all(framed)?;
        self.hasher.update(framed);
        self.records += 1;
        Ok(())
    }

    /// The §8.4 commit barrier: fdatasync the segment file so every record written before it is
    /// durable.
    ///
    /// # Errors
    /// [`JournalError::Io`] if the sync fails.
    pub fn commit(&mut self) -> Result<(), JournalError> {
        // `sync_data` is fdatasync on Linux: flush data (+ the size metadata needed to read it back)
        // without the full inode sync of `sync_all`.
        self.file.sync_data()?;
        Ok(())
    }

    /// blake3 of the complete segment file as written so far (the chain link the next segment's
    /// `prev_blake3` carries — §8.2).
    #[must_use]
    pub fn file_blake3(&self) -> [u8; 32] {
        *self.hasher.finalize().as_bytes()
    }

    /// Seal the segment cleanly (§8.2, §8.3 tag 17): append a `seal` record whose `segment_blake3` is
    /// the hash of this segment's bytes **excluding the seal record's own framing**, then commit. A
    /// sealed segment is immutable. Returns the complete-file blake3 (post-seal) for the chain.
    ///
    /// # Errors
    /// [`JournalError::Codec`]/[`JournalError::Io`] on encode/write/sync failure.
    pub fn seal(&mut self) -> Result<[u8; 32], JournalError> {
        if self.sealed {
            return Err(JournalError::Io(io::Error::other("segment already sealed")));
        }
        // The seal hash covers everything written BEFORE the seal frame (no self-reference, §8.2).
        let pre_seal = *self.hasher.clone().finalize().as_bytes();
        let seal = Record::new(
            self.records,
            Body::Seal(SealRec {
                segment_blake3: daemon_vhc_proto::Hash(pre_seal),
                records: self.records,
            }),
        );
        let cbor = seal.to_canonical()?;
        let framed = frame(&cbor)?;
        self.file.write_all(&framed)?;
        self.hasher.update(&framed);
        self.sealed = true;
        self.commit()?;
        Ok(self.file_blake3())
    }
}

/// The result of scanning a segment file for crash recovery (§8.2).
#[derive(Clone, Debug)]
pub struct ScanResult {
    /// The parsed segment header.
    pub header: SegmentHeader,
    /// The records recovered up to the first torn/corrupt frame (or the seal).
    pub records: Vec<Record>,
    /// Whether the segment ended with a valid `seal` record (a clean roll).
    pub sealed: bool,
    /// Whether a torn/corrupt tail was found and discarded (the segment was truncated on recovery).
    pub truncated: bool,
    /// The byte length of the intact prefix (header + every good frame). A crash-recovery reopen
    /// truncates the file to this length; nothing past it is trusted.
    pub durable_len: u64,
    /// blake3 of the intact prefix `bytes[0..durable_len]` — the chain link for the next segment.
    pub complete_file_blake3: [u8; 32],
}

/// Scan a segment's raw bytes, validating framing + CRC32C and stopping at the first torn/corrupt
/// frame (§8.2 crash recovery). No silent corruption is admitted past a CRC/chain break: the tail is
/// discarded and reported via [`ScanResult::truncated`].
///
/// # Errors
/// [`JournalError::BadHeader`] if the header itself is unreadable (a segment with no clean recovery
/// point at all).
pub fn scan_bytes(bytes: &[u8]) -> Result<ScanResult, JournalError> {
    let (header, mut off) = SegmentHeader::decode(bytes)?;
    let mut records = Vec::new();
    let mut sealed = false;
    let mut truncated = false;
    let mut hasher = blake3::Hasher::new();
    hasher.update(&bytes[..off]); // the header is always part of the intact prefix.

    loop {
        if off == bytes.len() {
            break; // clean end at a frame boundary.
        }
        // Need at least the 4-byte length prefix.
        if off + 4 > bytes.len() {
            truncated = true;
            break;
        }
        let len = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) as usize;
        let frame_end = off + 4 + len + 4;
        if frame_end > bytes.len() {
            truncated = true; // a torn frame: length promises more bytes than exist.
            break;
        }
        let cbor = &bytes[off + 4..off + 4 + len];
        let crc = u32::from_le_bytes(bytes[off + 4 + len..frame_end].try_into().unwrap());
        if crc32c::crc32c(cbor) != crc {
            truncated = true; // a corrupt frame: CRC mismatch.
            break;
        }
        let record = match Record::from_canonical(cbor) {
            Ok(r) => r,
            Err(_) => {
                // A well-framed, CRC-valid frame that is not a decodable record is corruption too.
                truncated = true;
                break;
            }
        };
        // A seal record (tag 17) must be last and its hash must exclude its own frame.
        if let Body::Seal(seal) = &record.body {
            let pre_seal = *hasher.clone().finalize().as_bytes();
            if seal.segment_blake3.as_bytes() != &pre_seal {
                return Err(JournalError::ChainBroken(
                    "seal segment_blake3 does not match segment bytes".into(),
                ));
            }
            hasher.update(&bytes[off..frame_end]);
            off = frame_end;
            records.push(record);
            sealed = true;
            break;
        }
        hasher.update(&bytes[off..frame_end]);
        off = frame_end;
        records.push(record);
    }

    Ok(ScanResult {
        header,
        records,
        sealed,
        truncated,
        durable_len: off as u64,
        complete_file_blake3: *hasher.finalize().as_bytes(),
    })
}

/// Read a segment file from disk and scan it (§8.2 crash recovery).
///
/// # Errors
/// [`JournalError::Io`] on read failure; [`JournalError::BadHeader`]/[`JournalError::ChainBroken`]
/// per [`scan_bytes`].
pub fn scan_file(path: impl AsRef<Path>) -> Result<ScanResult, JournalError> {
    let mut f = File::open(path)?;
    let mut bytes = Vec::new();
    f.read_to_end(&mut bytes)?;
    scan_bytes(&bytes)
}
